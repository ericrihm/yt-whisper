# yt-whisper

> Fast, local YouTube transcription with speaker diarization and a keyboard-first TUI.

Transcribe any YouTube video on your own GPU. yt-whisper checks for existing YouTube subtitles first (fast path), falls back to [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on CUDA, can label distinct speakers via [pyannote.audio](https://github.com/pyannote/pyannote-audio), and ships with an interactive Textual TUI for a full dashboard experience.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![CUDA](https://img.shields.io/badge/CUDA-optional-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Tests](https://img.shields.io/badge/tests-98%20passing-brightgreen)

---

## Features

- **YouTube subtitle fast path** — uses existing subs when available, skips transcription entirely
- **Local GPU transcription** — faster-whisper on CUDA (float16), automatic CPU fallback
- **Speaker diarization** — opt-in pyannote pipeline, outputs `**Speaker 1:**` / `**Speaker 2:**` blocks
- **Prompt profile auto-detection** — picks `general` / `grc` / `infosec` vocabulary from video metadata
- **Full-screen TUI** — Home / Run / Preview screens, live streaming transcript, history browser
- **Flag mode** — scriptable CLI identical to pre-TUI behavior
- **Structured output** — markdown + JSON with segments, timestamps, speakers, config for re-run

---

## Install

### Option A: global install with pipx (recommended)

Makes `yt-whisper` available from any shell, in its own isolated environment.

```bash
# one-time: install pipx
py -3.14 -m pip install --user pipx
py -3.14 -m pipx ensurepath

# install yt-whisper globally (editable, so git pull updates it)
pipx install --editable .
```

### Option B: venv (for development)

```bash
py -3.14 -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -e ".[dev]"
```

### Prerequisites

- **Python 3.10+** — use system Python from [python.org](https://www.python.org/downloads/), **not Microsoft Store Python** (its sandbox blocks native DLLs that faster-whisper needs)
- **ffmpeg** on PATH:
  - Windows: `winget install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
- **NVIDIA GPU + CUDA drivers** (optional — falls back to CPU, much slower)

---

## Quick start

```bash
# Launch the interactive TUI
yt-whisper

# Flag mode
yt-whisper https://www.youtube.com/watch?v=VIDEO_ID

# Force whisper (skip subtitle check)
yt-whisper <url> --force-whisper --verbose

# With speaker diarization (requires optional setup — see below)
yt-whisper <url> --diarize --speakers 3

# Use a domain vocabulary profile
yt-whisper <url> --prompt infosec

# Custom prompt string
yt-whisper <url> --prompt "kubernetes, Helm, pod, ingress"
```

Running `yt-whisper` with no arguments launches the TUI.

---

## Interactive TUI

```
+------------------------+--------------------------+
| History                | New Transcription        |
|                        |                          |
|   Keynote Talk (1:02)  | URL:    [__________]     |
| * Panel Discussion     | Model:  [large-v3  v]    |
|   Lightning Talk       | Lang:   [en]             |
|                        | Profile:[general   v]    |
|                        | [ ] Diarize              |
|                        | Format: (o) both         |
|                        |         ( ) md           |
|                        |         ( ) json         |
|                        |                          |
|                        |         [   Run   ]      |
+------------------------+--------------------------+
 q quit  r re-run  d delete  p preview
```

Three screens:
- **Home** — form + history of past runs (scanned from `./transcripts/*.json`)
- **Run** — live progress bars per phase, streaming transcript, log panel, `Esc` to cancel
- **Preview** — rendered markdown of a completed run, `o` to open in system editor

Keyboard-first. `Q` quit, `R` re-run selected, `P` preview, `D` delete.

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `url` | — | YouTube URL (omit to launch TUI) |
| `--prompt` | `general` | Profile (`general`, `grc`, `infosec`) or custom vocabulary string |
| `--force-whisper` | off | Skip subtitle check, always transcribe |
| `--output-dir` | `./transcripts` | Where to write `.md` / `.json` |
| `--model` | `large-v3` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` |
| `--format` | `both` | `md`, `json`, or `both` |
| `--language` | `en` | Language code passed to whisper |
| `--diarize` | off | Enable speaker diarization (requires optional setup) |
| `--speakers N` | — | Exact number of speakers (diarize only) |
| `--min-speakers N` | — | Lower bound on speakers |
| `--max-speakers N` | — | Upper bound on speakers |
| `--verbose` | off | Per-segment timing, phase logs, auto-detection info |

---

## Prompt profiles

| Profile | Vocabulary hints |
|---------|------------------|
| `general` | No domain bias (default) |
| `grc` | NIST, RMF, CMMC, FedRAMP, SOC 2, ISO 27001, compliance |
| `infosec` | CVE, CVSS, MITRE ATT&CK, SOC, incident response, exploit |

**Auto-detection:** when you don't pass `--prompt`, yt-whisper scans the video's title, channel, description, and tags and picks the best matching profile by keyword scoring. Verbose mode prints what was matched:

```
[auto] profile: grc (matched: NIST, 800-53, SOC 2, FedRAMP)
```

Pass `--prompt <name>` explicitly to override.

---

## Optional: speaker diarization

Diarization labels distinct speakers in the output. It is off by default because it requires accepting a model license and setting an API token.

### 1. Install extra dependencies

```bash
# inside venv
pip install -r requirements-diarize.txt

# or if installed via pipx:
pipx inject yt-whisper pyannote.audio torchaudio
```

### 2. Accept the pyannote license and get a token

1. Create a free account at <https://huggingface.co/>
2. Visit <https://huggingface.co/pyannote/speaker-diarization-3.1> and accept the user agreement
3. Generate a token at <https://huggingface.co/settings/tokens> (read access is enough)
4. Set `HF_TOKEN`:
   - PowerShell: `$env:HF_TOKEN="hf_..."`
   - Or put `HF_TOKEN=hf_...` in a `.env` file next to the project (python-dotenv loads it)

### 3. Run it

```bash
yt-whisper <url> --diarize                    # auto-detect speaker count
yt-whisper <url> --diarize --speakers 3       # if you know the count
yt-whisper <url> --diarize --min-speakers 2 --max-speakers 4
```

If the dependencies or token are missing, yt-whisper emits a clean error with install hints instead of crashing.

Diarized markdown output groups consecutive segments by speaker:

```markdown
**Speaker 1:** Welcome everyone to today's session. I'm excited to...

**Speaker 2:** Thanks for having me. Let me start by introducing...
```

JSON output includes a top-level `speakers` list and a `speaker` field on every segment.

---

## Output

Transcripts land in `./transcripts/` by default.

- **`{video_id}.md`** — metadata header + paragraphed text (or speaker blocks if diarized)
- **`{video_id}.json`** — structured: `segments`, `speakers`, `full_text`, timestamps, model, prompt profile, and a `config` block for easy re-run from the TUI history

---

## Performance

Benchmarked on RTX 3080 Ti, CUDA, `large-v3` + VAD filter. Diarization adds ~0.2x the transcription time.

| Video length | GPU (CUDA) | CPU fallback | ~Word count |
|--------------|------------|--------------|-------------|
| 15 min | ~30 s | ~5 min | ~2,300 |
| 30 min | ~60 s | ~10 min | ~4,500 |
| 60 min | ~2 min | ~20 min | ~9,000 |

First run downloads the whisper model (~3 GB for `large-v3`, ~1 GB for `small`/`medium`). Cached thereafter at `~/.cache/huggingface/hub`.

---

## Architecture

```
cli.py         (thin shim: argparse -> RunConfig -> runner.run)
   |
   v
runner.py      (RunConfig, Listener, ConsoleListener, run())
   |
   +-- downloader.py    (yt-dlp: check_subtitles, download_audio)
   +-- transcriber.py   (faster-whisper, streams segments)
   +-- diarizer.py      (pyannote, optional, local import)
   +-- profile_detect.py (keyword scoring over metadata)
   +-- formatter.py     (markdown + JSON output, speaker-aware)
   |
   v
tui/app.py     (Textual: Home / Run / Preview, worker thread)
tui/listener.py (bridges runner events via call_from_thread)
tui/history.py  (scans transcripts/*.json for past runs)
```

Both flag mode and the TUI go through `runner.run()` — a single pipeline source. The runner raises no exceptions; all failures route through `listener.on_error`, making it safe to drive from a UI worker thread.

---

## Troubleshooting

### Python / DLLs

- **"Application Control policy has blocked this file"** / **"DLL load failed"** — You're on Microsoft Store Python. Install system Python from [python.org](https://www.python.org/downloads/) and recreate the venv.

### CUDA

- **"CUDA driver version is insufficient"** — update NVIDIA drivers at [nvidia.com/drivers](https://www.nvidia.com/drivers/), verify with `nvidia-smi`.
- **"Could not locate cudnn_ops64_9.dll"** — the DLL preloader should handle this; if not, `pip install "nvidia-cudnn-cu12>=9.0,<10"`.
- **"CUDA unavailable" warning** — falls back to CPU automatically. Check `nvidia-smi`.
- **Slow transcription** — likely on CPU. Look for the CUDA warning in verbose output.

### yt-dlp

- **"Video unavailable"** — private, age-restricted, or geo-blocked video.
- **"ffprobe and ffmpeg not found"** — install ffmpeg and make sure it's on PATH.

### Transcription quality

- **"No speech detected"** — VAD found nothing (common with music). Try a different video or larger model.
- **"Low word count" warning** — incomplete transcript; try `--force-whisper` or a larger `--model`.
- **"High word count" warning** — likely whisper hallucination (repeated phrases). Switch to a better model or shorter clip.

### Diarization

- **"HF_TOKEN environment variable not set"** — see the diarization setup steps above.
- **"pyannote.audio is not installed"** — `pip install -r requirements-diarize.txt` (or `pipx inject yt-whisper pyannote.audio torchaudio`).
- **"User agreement not accepted"** — visit <https://huggingface.co/pyannote/speaker-diarization-3.1> and click accept.

---

## Development

```bash
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# Run the 98 unit tests (no GPU or network required)
pytest tests/ -v
```

Tests mock yt-dlp, faster-whisper, and pyannote, so they run anywhere in under 3 seconds.

See `docs/superpowers/` for the design spec and implementation plan.

---

## License

MIT
