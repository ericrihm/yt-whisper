# yt-whisper

YouTube video transcription CLI using yt-dlp + faster-whisper on local CUDA GPU.

## Architecture

7-module pipeline: `cli.py` orchestrates `downloader.py` → `transcriber.py` → `formatter.py`. Supporting: `prompts.py`, `cuda_preload.py`. Entry via `__main__.py`.

- **downloader.py**: Checks YouTube subs first (manual > auto, json3 > vtt > srv1), downloads audio as 16kHz mono WAV via yt-dlp Python API
- **transcriber.py**: faster-whisper with CUDA/float16, CPU/int8 fallback. `faster_whisper` is imported locally inside `transcribe()`, never at module level
- **formatter.py**: Markdown + JSON output with paragraph formatting
- **prompts.py**: Domain vocabulary profiles (general, grc, infosec) + custom string passthrough
- **cuda_preload.py**: Windows DLL preloading for MS Store Python (cudnn_ops64_9.dll, cublas)

## Key Constraints

- **Anti-Pattern #1**: Never import `faster_whisper` at module top level. Must be local import inside `transcribe()` after `cuda_preload.ensure_dlls()`
- **yt-dlp**: Always use Python API (`yt_dlp.YoutubeDL`), never subprocess. `postprocessor_args` uses dict format: `{"ffmpeg": ["-ar", "16000", "-ac", "1"]}`
- **model.transcribe()**: Returns a 2-tuple `(segments_generator, info)` -- must unpack
- **check_subtitles()**: Uses single `YoutubeDL` context for both `extract_info` and `urlopen`
- **Windows console**: No unicode symbols in print output. Use ASCII only (`[OK]` not checkmark, `--` not em dash, `->` not arrow). Windows cp1252 encoding crashes on unicode.
- **MS Store Python**: Do NOT use. Native DLLs (PyAV, ctranslate2) fail with "Application Control policy blocked" errors. Use system Python from python.org instead.

## Development Setup

```bash
# Use system Python (NOT MS Store Python)
py -3.14 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt pytest

# ffmpeg must be on PATH (installed via winget)
# WinGet links dir: %LOCALAPPDATA%/Microsoft/WinGet/Links

# Run tests
.venv/Scripts/python.exe -m pytest tests/ -v

# Run tool
.venv/Scripts/python.exe -m yt_whisper <url>
```

39 unit tests across 5 test files. Tests mock yt-dlp and faster-whisper -- no network or GPU needed for unit tests.

## Dependencies

- System Python 3.10+ (NOT MS Store Python), ffmpeg on PATH, NVIDIA GPU + CUDA drivers (CPU fallback available)
- `pip install -r requirements.txt`
- CUDA driver must match CUDA runtime version -- if "CUDA driver version is insufficient" appears, update NVIDIA drivers

## Specs & Plans

- Design spec: `docs/superpowers/specs/2026-04-04-yt-whisper-design.md`
- Implementation plan: `docs/superpowers/plans/2026-04-04-yt-whisper.md`
