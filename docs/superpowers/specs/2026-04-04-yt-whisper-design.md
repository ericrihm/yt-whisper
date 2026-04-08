# yt-whisper Design Spec

**Date**: 2026-04-04
**Status**: Reviewed
**Purpose**: Python CLI tool that transcribes YouTube videos lacking usable subtitles by downloading audio via yt-dlp and transcribing with faster-whisper on local CUDA GPU.

---

## 1. Project Structure

```
yt-whisper/
├── yt_whisper/
│   ├── __init__.py          # version string only
│   ├── __main__.py          # python -m entrypoint → cli.main()
│   ├── cli.py               # argparse, orchestration, exit codes
│   ├── downloader.py        # yt-dlp: subtitle check + audio extraction
│   ├── transcriber.py       # faster-whisper CUDA transcription
│   ├── formatter.py         # segments → markdown + JSON output
│   ├── prompts.py           # named domain vocabulary profiles
│   └── cuda_preload.py      # Windows DLL preloading
├── transcripts/             # default output directory (gitignored)
├── requirements.txt
├── README.md
└── .gitignore
```

## 2. Data Flow

```
cli.py receives URL + args
  → downloader.check_subtitles(url, language)
    → always returns (text_or_None, metadata)  ← metadata includes url
    → if text found and not --force-whisper: use subtitle text
    → else: downloader.download_audio(url, temp_dir, metadata) returns audio_path
      → transcriber.transcribe(audio_path, model, prompt, language, verbose) returns segments[]
  → formatter.format_output(text_or_segments, metadata, output_format, output_dir, ...) writes files
  → cli validates word count (skipped if duration < 30s), prints summary, exits with code
```

All imports of `faster_whisper` happen lazily inside `transcriber.py` function bodies, never at module top level. `cuda_preload.ensure_dlls()` is called before the lazy import.

## 3. CLI Interface

```bash
python -m yt_whisper <url> \
  --prompt grc|general|infosec|<custom string> \
  --force-whisper \
  --output-dir ./transcripts \
  --model large-v3|medium \
  --format md|json|both \
  --language en \
  --verbose
```

### Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `url` | required | YouTube video URL (positional) |
| `--prompt` | `general` | Named prompt profile or custom string in quotes |
| `--force-whisper` | `False` | Skip YouTube subtitle check, always use Whisper |
| `--output-dir` | `./transcripts` | Output directory (created if missing) |
| `--model` | `large-v3` | Whisper model size. Accepts any valid faster-whisper model name (e.g., `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`). No argparse `choices` restriction — model validation happens at load time. |
| `--format` | `both` | Output format: `md`, `json`, or `both`. Argparse `dest="output_format"` to avoid shadowing Python builtin. |
| `--language` | `en` | Language code for subtitle search and transcription |
| `--verbose` | `False` | Show progress details and segment timing |

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Input/network error (bad URL, private video, no network) |
| `2` | Transcription failure (CUDA error, empty output, model load failure) |

## 4. Downloader (`downloader.py`)

### 4.1 `check_subtitles(url, language="en") -> tuple[str | None, dict]`

Returns `(subtitle_text, metadata)` or `(None, metadata)`.

1. Use `yt_dlp.YoutubeDL` (default options, no `extract_flat`) with `download=False` to fetch video info.
2. Extract metadata: `{video_id, title, channel, upload_date, duration, url}`.
3. Build language priority list from language arg (e.g., `"en"` → `["en", "en-US", "en-GB"]`).
4. Check `info_dict["subtitles"]` first (manual/human captions, higher quality), then `info_dict["automatic_captions"]`.
5. If match found:
   - Get the subtitle URL from `info_dict["subtitles"]` or `info_dict["automatic_captions"]`.
   - Prefer `json3` format entry. Fallback chain: `json3` → `vtt` → `srv1`.
   - Fetch subtitle content via `yt_dlp.YoutubeDL.urlopen()` on the subtitle URL (in-memory, no temp file).
   - Parse `json3`: iterate `events[]`, concatenate `segs[].utf8` values. Strip HTML tags via regex (`<[^>]+>`).
   - Join fragments, collapse whitespace into continuous text.
   - Return `(cleaned_text, metadata)`.
6. If no match: return `(None, metadata)`.

### 4.2 `download_audio(url, temp_dir, metadata, verbose=False) -> str`

Accepts metadata dict from `check_subtitles()` (avoids redundant `extract_info` call). Returns `audio_path`.

```python
ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "wav",
    }],
    "postprocessor_args": {"ffmpeg": ["-ar", "16000", "-ac", "1"]},
    "quiet": not verbose,
    "no_warnings": not verbose,
}
```

Key points:
- Audio only, no video download.
- `postprocessor_args` uses the yt-dlp dict format (key=executable, value=args list), NOT the legacy list format.
- 16kHz mono WAV produced directly by yt-dlp postprocessor, no post-conversion.
- Uses yt-dlp Python API, not subprocess.

### 4.3 Error Handling

Custom exceptions are defined in `downloader.py`:
- `VideoUnavailableError(Exception)` — raised when yt-dlp cannot access the video.
- Imported by `cli.py` for exit code mapping (exit code `1`).

yt-dlp raises `DownloadError` for network/availability issues. Catch and re-raise as `VideoUnavailableError` with a clear message indicating cause (private, age-restricted, geo-blocked, deleted, or generic network error).

## 5. Transcriber (`transcriber.py`)

### 5.1 `transcribe(audio_path, model_size, prompt_text, language, verbose) -> list[dict]`

Returns list of `{"start": float, "end": float, "text": str}`.

Custom exception defined in `transcriber.py`:
- `TranscriptionError(Exception)` — raised on empty output or model failure.
- Imported by `cli.py` for exit code mapping (exit code `2`).

```
1. Call cuda_preload.ensure_dlls()
2. Import faster_whisper (local import inside function)
3. Detect CUDA:
   - Try device="cuda", compute_type="float16"
   - On failure: warn, fallback to device="cpu", compute_type="int8"
4. Check model cache (best-effort heuristic):
   - Look for models--Systran--faster-whisper-{model_size} in ~/.cache/huggingface/hub/
   - Note: HF_HUB_CACHE env var can override this path; check is not exhaustive
   - If not found: print "Downloading Whisper model '{model}' (~3GB for large-v3). One-time download."
5. Instantiate WhisperModel(model_size, device, compute_type)
6. segments, info = model.transcribe(...)  ← MUST unpack the 2-tuple
   - model.transcribe() returns (generator, TranscriptionInfo), NOT just segments
   - Arguments: audio_path, language=language, beam_size=5, vad_filter=True, initial_prompt=prompt_text
   - prompt_text may be None (for "general" profile) — this is valid, faster-whisper skips initial_prompt when None
7. Iterate segments generator, collect into list of {"start", "end", "text"} dicts
8. If verbose: print each segment with [start → end] timestamp
9. If zero segments: raise TranscriptionError("No speech detected")
10. Return segments list
```

### 5.2 CUDA Fallback

Catches `RuntimeError` and `ValueError` from CTranslate2's CUDA initialization. Warning message:
> "CUDA unavailable — falling back to CPU. This will be significantly slower. Check NVIDIA drivers and CUDA toolkit."

Never crashes. Falls back to CPU with `int8` compute type.

### 5.3 Model Download Detection

`faster_whisper` uses huggingface_hub's default cache at `~/.cache/huggingface/hub/`. Check for directory pattern `models--Systran--faster-whisper-{model_size}` to decide whether to print the download notice.

## 6. Formatter (`formatter.py`)

### 6.1 `format_output(text_or_segments, metadata, output_format, output_dir, model, prompt_profile, method, language) -> list[str]`

`metadata` dict includes `url` (the original input URL, stored during `check_subtitles`).

Returns list of output file paths written.

### 6.2 Text Assembly

- If input is segments list: join all `segment["text"]` with spaces.
- If input is raw text string (YouTube subs): use as-is.
- Strip extra whitespace, normalize. Produces `full_text`.

### 6.3 Paragraph Formatting

1. Split on sentence boundaries: `re.split(r'(?<=[.!?])\s+', text)`
2. Group into paragraphs of ~5 sentences each.
3. Join paragraphs with double newline.

Same logic for both Whisper output and YouTube subs (YouTube caption lines are first joined into continuous text, then paragraphed).

### 6.4 Duration Formatting

`seconds → "H:MM:SS"` if ≥ 3600, else `"M:SS"`. No leading zero on the largest unit.

Examples: `"1:05:30"`, `"4:22"`, `"45:07"`.

### 6.5 Markdown Output (`{video_id}.md`)

```markdown
# {title}

- **Channel**: {channel}
- **Date**: {upload_date}
- **Duration**: {duration_formatted}
- **URL**: {url}
- **Language**: {language}
- **Word Count**: {word_count}
- **Transcription Method**: {method} ({model} / {prompt_profile})

---

{paragraphed text}
```

Transcription method line examples:
- Whisper with named prompt: `whisper (large-v3 / grc)`
- Whisper with general (no prompt): `whisper (large-v3)`
- Whisper with custom prompt: `whisper (large-v3 / custom)`
- YouTube subs: `youtube_subs` (no model/prompt shown)

### 6.6 JSON Output (`{video_id}.json`)

```json
{
  "video_id": "...",
  "title": "...",
  "channel": "...",
  "url": "...",
  "upload_date": "...",
  "language": "en",
  "duration_seconds": 900,
  "duration_formatted": "15:00",
  "word_count": 2340,
  "transcription_method": "whisper",
  "model": "large-v3",
  "prompt_profile": "grc",
  "segments": [
    {"start": 0.0, "end": 4.5, "text": "..."}
  ],
  "full_text": "..."
}
```

When method is `youtube_subs`: `segments`, `model`, and `prompt_profile` are `null`.

### 6.7 File Collision

Overwrites silently. Re-running with different options is the expected use case.

## 7. Prompt Profiles (`prompts.py`)

```python
PROMPTS = {
    "general": None,
    "grc": (
        "NIST, RMF, Risk Management Framework, CMMC, FedRAMP, SOC2, SOC 2, GRC, "
        "cybersecurity, compliance, audit, control, framework, assessment, authorization, "
        "ATO, FISMA, FIPS 199, SP 800-53, SP 800-37, risk register, risk assessment, "
        "threat modeling, likelihood, impact, inherent risk, residual risk, Gerald Auger"
    ),
    "infosec": (
        "CVE, CVSS, vulnerability, exploit, zero-day, malware, ransomware, phishing, "
        "SOC, SIEM, EDR, XDR, MITRE ATT&CK, threat intelligence, incident response, "
        "penetration testing, red team, blue team, OSINT, IOC, indicators of compromise"
    ),
}

def resolve_prompt(name_or_string):
    """Return prompt text. Known key → stored value. Unknown key → treat as custom string."""
    return PROMPTS.get(name_or_string, name_or_string)
```

Returns `None` for `"general"` — faster-whisper skips `initial_prompt` when `None`.

## 8. CUDA Preload (`cuda_preload.py`)

Windows-specific DLL preloading for faster-whisper CUDA support.

**Problem**: Microsoft Store Python's sandbox prevents normal DLL discovery.
**Solution**: Explicitly load DLLs via `ctypes.WinDLL()` before importing `faster_whisper`.

```python
def ensure_dlls():
    """Pre-load CUDA DLLs on Windows. No-op on Linux/Mac."""
    if sys.platform != "win32":
        return

    nvidia_spec = importlib.util.find_spec("nvidia")
    if nvidia_spec is None or nvidia_spec.submodule_search_locations is None:
        return

    nvidia_base = list(nvidia_spec.submodule_search_locations)[0]

    dll_paths = [
        os.path.join(nvidia_base, "cublas", "bin", "cublasLt64_12.dll"),
        os.path.join(nvidia_base, "cublas", "bin", "cublas64_12.dll"),
        os.path.join(nvidia_base, "cudnn", "bin", "cudnn_ops64_9.dll"),
    ]

    for dll_path in dll_paths:
        if os.path.exists(dll_path):
            try:
                ctypes.WinDLL(dll_path)
            except OSError as e:
                print(f"Warning: Failed to preload {os.path.basename(dll_path)}: {e}")
```

Must be called BEFORE any import of `faster_whisper` or `ctranslate2`.

## 9. Word Count Validation

Performed in `cli.py` after transcription:

```
expected_wpm = 150
duration_minutes = duration_seconds / 60

# Guard: skip validation for zero/very short durations (live streams, premieres)
if duration_minutes < 0.5:
    skip validation, print note: "Video too short for word count validation."
else:
    actual_wpm = word_count / duration_minutes

    if actual_wpm < 100:
        warn: "Low word count ({wpm} wpm). Expected ~150. Transcript may be incomplete."
    if actual_wpm > 200:
        warn: "High word count ({wpm} wpm). May indicate repeated or hallucinated text."
```

The high-wpm check catches Whisper's known hallucination pattern of repeating phrases.

## 10. CLI Summary Output

Always printed on success:

```
✓ {title}
  Duration:    {duration_formatted}
  Words:       {word_count} ({wpm:.0f} words/min)
  Method:      {method}
  Output:      {path1}
               {path2}
```

## 11. Anti-Patterns (Do NOT)

1. Do NOT import `faster_whisper` at module top level in any file except inside `transcriber.py` function bodies.
2. Do NOT use subprocess to call yt-dlp. Use the `yt_dlp` Python API.
3. Do NOT download video+audio. Audio only.
4. Do NOT convert audio format after download. Use yt-dlp postprocessor args.
5. Do NOT hardcode paths for nvidia DLLs. Use `importlib.util.find_spec`.
6. Do NOT batch process. Single URL only.
7. Do NOT suppress errors silently. Every failure prints a clear message.

## 12. Dependencies

**requirements.txt:**
```
yt-dlp>=2024.01.01
faster-whisper>=1.0.0
nvidia-cublas-cu12>=12.0,<13
nvidia-cudnn-cu12>=9.0,<10
```

Note: `nvidia-cudnn-cu12` is pinned to 9.x because `cuda_preload.py` hardcodes `cudnn_ops64_9.dll`. A cuDNN major version change would require updating the DLL filename.

**System requirements:**
- Python 3.10+
- ffmpeg on PATH
- NVIDIA GPU with CUDA drivers (CPU fallback available)

## 13. Performance Expectations

| Video Length | Transcription Time (RTX 3080 Ti) | Expected Words |
|-------------|----------------------------------|----------------|
| 15 min | ~30s | ~2,300 |
| 30 min | ~60s | ~4,500 |
| 60 min | ~2 min | ~9,000 |

## 14. Testing Sequence

1. `python -m yt_whisper https://www.youtube.com/watch?v=dQw4w9WgXcQ --format both` — short video with known subs, should use YouTube subs.
2. `python -m yt_whisper https://www.youtube.com/watch?v=dQw4w9WgXcQ --force-whisper` — same video, forces Whisper transcription.
3. Verify word count validation prints for both.
4. Verify both `.md` and `.json` outputs created with correct metadata.
