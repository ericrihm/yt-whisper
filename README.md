# yt-whisper

Transcribe YouTube videos by downloading audio via yt-dlp and transcribing with faster-whisper on local CUDA GPU. Automatically checks for existing YouTube subtitles first.

## Prerequisites

- **Python 3.10+** (system install from python.org -- NOT Microsoft Store Python)
- **ffmpeg** installed and on PATH
- **NVIDIA GPU** with CUDA drivers (CPU fallback available but significantly slower)

> **Windows note:** Microsoft Store Python's sandbox blocks native DLLs required by faster-whisper. Use system Python from [python.org](https://www.python.org/downloads/) instead.

## Install

```bash
# Create a virtual environment with system Python
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### ffmpeg

ffmpeg must be installed separately and available on PATH.

- **Windows:** `winget install ffmpeg`
- **Mac:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

## Usage

```bash
# Basic -- grabs YouTube subs if available, else transcribes
python -m yt_whisper https://www.youtube.com/watch?v=VIDEO_ID

# Force Whisper transcription (skip subtitle check)
python -m yt_whisper https://www.youtube.com/watch?v=VIDEO_ID --force-whisper

# Use domain-specific vocabulary prompt
python -m yt_whisper https://www.youtube.com/watch?v=VIDEO_ID --prompt grc

# Custom prompt string
python -m yt_whisper https://www.youtube.com/watch?v=VIDEO_ID --prompt "kubernetes, Helm, pod, ingress"

# Output only markdown
python -m yt_whisper https://www.youtube.com/watch?v=VIDEO_ID --format md

# Verbose mode with segment timing
python -m yt_whisper https://www.youtube.com/watch?v=VIDEO_ID --force-whisper --verbose
```

## Options

| Argument | Default | Description |
|----------|---------|-------------|
| `url` | required | YouTube video URL |
| `--prompt` | `general` | Prompt profile or custom string |
| `--force-whisper` | off | Always use Whisper, skip subtitle check |
| `--output-dir` | `./transcripts` | Output directory |
| `--model` | `large-v3` | Whisper model size (tiny, base, small, medium, large-v2, large-v3) |
| `--format` | `both` | Output: `md`, `json`, or `both` |
| `--language` | `en` | Language code |
| `--verbose` | off | Show detailed progress and segment timing |

## Prompt Profiles

| Profile | Use Case |
|---------|----------|
| `general` | No domain vocabulary (default) |
| `grc` | GRC, compliance, NIST, FedRAMP, risk management |
| `infosec` | Vulnerability, SOC, MITRE ATT&CK, incident response |

Any value not matching a profile name is used as a custom prompt string.

## Performance

| Video Length | Time (RTX 3080 Ti, CUDA) | Time (CPU fallback) | Expected Words |
|-------------|--------------------------|---------------------|----------------|
| 15 min | ~30s | ~5 min | ~2,300 |
| 30 min | ~60s | ~10 min | ~4,500 |
| 60 min | ~2 min | ~20 min | ~9,000 |

First run downloads the model (~3GB for large-v3, ~1GB for small/medium). Subsequent runs use the cached model.

## Output

Transcripts are saved to `./transcripts/` by default (configurable with `--output-dir`).

**Markdown** (`{video_id}.md`) -- formatted with metadata header and paragraphed text.

**JSON** (`{video_id}.json`) -- structured data with segments, timestamps, and full text.

## Troubleshooting

### Python / DLL errors

- **"Application Control policy has blocked this file"**: You're using Microsoft Store Python. Switch to system Python from python.org.
- **"DLL load failed"**: Same issue -- use system Python, not MS Store Python.

### CUDA errors

- **"CUDA driver version is insufficient"**: Your NVIDIA driver is too old for the CUDA runtime. Update drivers from [nvidia.com/drivers](https://www.nvidia.com/drivers/).
- **"Could not locate cudnn_ops64_9.dll"**: The CUDA DLL preloader should handle this. If it persists: `pip install "nvidia-cudnn-cu12>=9.0,<10"`
- **"CUDA unavailable" warning**: The tool falls back to CPU automatically. To restore CUDA: update NVIDIA drivers, verify with `nvidia-smi`.
- **Slow transcription**: You're likely on CPU. Check for the CUDA warning in output.

### yt-dlp errors

- **"Video unavailable"**: Video may be private, age-restricted, or geo-blocked.
- **"ffprobe and ffmpeg not found"**: Install ffmpeg and ensure it's on your PATH (see Install section).
- **"No supported JavaScript runtime"**: yt-dlp warning -- does not affect functionality for most videos.

### Transcription issues

- **"No speech detected"**: VAD filter found no speech. Common with music-only content. Try a different video or a larger model.
- **"Low word count" warning**: Transcript may be incomplete. Try `--force-whisper` if using YouTube subs, or a larger `--model`.
- **"High word count" warning**: May indicate Whisper hallucination (repeated phrases). Check output for repetition.

## Optional: Speaker Diarization

Diarization labels distinct speakers in the transcript. It is off by default and requires extra setup.

**Install the extra dependencies:**

```bash
.venv/Scripts/python.exe -m pip install -r requirements-diarize.txt
```

**Accept the model license and get a token:**

1. Create a free account at https://huggingface.co/
2. Visit https://huggingface.co/pyannote/speaker-diarization-3.1 and accept the user agreement
3. Generate a token at https://huggingface.co/settings/tokens (read access is sufficient)
4. Set the environment variable `HF_TOKEN`:
   - Powershell: `$env:HF_TOKEN="hf_..."`
   - Or put it in a `.env` file next to the project (python-dotenv will load it)

**Use it:**

```bash
yt-whisper <url> --diarize
yt-whisper <url> --diarize --speakers 3   # if you know the count
```

In the TUI, toggle the "Diarize" checkbox on the home screen. If dependencies or the token are missing, the runner emits a clean error with install instructions.

## Interactive TUI

Running `yt-whisper` with no arguments launches a full-screen interactive UI with three screens:

- **Home** -- paste a URL, pick model/language/profile/diarize/format, see past runs
- **Run** -- live progress bars, streaming transcript, and log panel while the pipeline executes
- **Preview** -- rendered markdown of a completed transcript

Keyboard-first. Press `Q` to quit, `R` to re-run a past entry, `P` to preview, `D` to delete.

## Prompt Profile Auto-Detection

When you don't pass `--prompt`, yt-whisper inspects the video's title, channel, description, and tags and picks the best matching profile (general, grc, infosec) using keyword matching. Pass `--prompt <name>` explicitly to override.
