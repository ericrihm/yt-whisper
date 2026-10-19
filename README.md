# yt-whisper

> Fast, local YouTube transcription with speaker diarization and a keyboard-first TUI.

Transcribe YouTube videos locally. yt-whisper checks for existing subtitles first, then uses faster-whisper on CPU/CUDA or optional [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) on Apple-silicon Metal. Speaker diarization remains opt-in, and the Textual TUI exposes the same backend choice as the CLI.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![CUDA](https://img.shields.io/badge/CUDA-optional-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Tests](https://img.shields.io/badge/tests-98%20passing-brightgreen)

---

## Features

- **YouTube subtitle fast path** — uses existing subs when available, skips transcription entirely
- **Native ARM / GPU transcription** — Apple-silicon CPU/int8 or optional Metal/MLX; CUDA on supported Linux/Windows architectures. Automatic CUDA fallback applies only to `auto`; explicit backend failures propagate.
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

### Apple silicon: native CPU and optional Metal

Use a native arm64 Python interpreter, not Rosetta:

```bash
brew install ffmpeg
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -e '.[metal]'
.venv/bin/yt-whisper --help
```

Omit `[metal]` for CPU-only installation. NVIDIA dependencies are excluded on macOS; PyAV16.x is selected because newer releases removed an API used by faster-whisper. The optional Metal extra installs MLX/MLX-Whisper, including its upstream dependencies; it needs Apple-silicon macOS with an available Metal device. It does not install pyannote or start recording.

`--backend auto` prefers Metal on Apple silicon when the extra and device are available, otherwise CPU/int8. Elsewhere it retains CUDA with CPU fallback. `--backend cpu`, `cuda` or `metal` requires that backend; incompatible hardware, a missing Metal extra, and runtime errors fail rather than silently changing an explicit selection. The TUI Backend field and stored re-run config preserve that choice. `config.backend` records the requested policy; runtime logs identify the selected backend.

Metal maps standard Whisper model names to preconverted `mlx-community/whisper-<name>-mlx` repositories, or accepts an MLX model directory/repository. Its weights are separate from CTranslate2's CPU/CUDA cache. Use an explicit small model for initial qualification: the unchanged default `large-v3` downloads much larger weights.

For an already authorized local recording, use the existing upstream file command:

```bash
.venv/bin/mlx_whisper recording.mp4 --model mlx-community/whisper-tiny.en-mlx \
    --language en --output-format all --output-dir transcripts
```

This produces TXT, JSON, SRT, VTT and TSV without a YouTube URL or account. MLX returns completed segments as a batch, unlike faster-whisper's incremental generator; TUI segment progress and cancellation checks therefore occur after its decode returns. Metal calls are serialized within the process to protect the upstream shared model cache when TUI workers overlap. A short bootstrap sample establishes functionality, not long-recording accuracy, alignment or performance.


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
| `--backend` | `auto` | `auto`, `cpu`, `cuda`, `metal`; explicit choices never silently fall back |
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

## Changelog

### Unreleased

- **TUI**: live transcribe progress bar now updates per segment (previously stuck at 10%)
- **TUI**: pressing Enter in any form input kicks off the run (not just the Run button)
- **TUI**: toggling Diarize on without `HF_TOKEN` or `pyannote.audio` pops a setup modal with clickable HuggingFace links and copyable install commands
- **TUI**: worker thread now redirects stdout/stderr so library noise (tqdm, CUDA init, faster-whisper, yt-dlp) can't corrupt Textual's alternate screen buffer
- **Runner**: logs the active prompt profile on every run (`Prompt profile: grc`)
- **Runner**: logs GPU vs CPU device selection (`whisper: using cuda (float16)` or `whisper: using cpu (int8) -- <err>`)
- **Tests**: 107 passing (up from 98)

---

## License

MIT
