# yt-whisper

Transcribe YouTube videos by downloading audio via yt-dlp and transcribing with faster-whisper on local CUDA GPU. Automatically checks for existing YouTube subtitles first.

## Prerequisites

- **Python 3.10+**
- **ffmpeg** installed and on PATH
- **NVIDIA GPU** with CUDA drivers (CPU fallback available but significantly slower)

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Basic — grabs YouTube subs if available, else transcribes
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
| `--model` | `large-v3` | Whisper model size |
| `--format` | `both` | Output: `md`, `json`, or `both` |
| `--language` | `en` | Language code |
| `--verbose` | off | Show detailed progress |

## Prompt Profiles

| Profile | Use Case |
|---------|----------|
| `general` | No domain vocabulary (default) |
| `grc` | GRC, compliance, NIST, FedRAMP, risk management |
| `infosec` | Vulnerability, SOC, MITRE ATT&CK, incident response |

## Performance

| Video Length | Time (RTX 3080 Ti) | Expected Words |
|-------------|---------------------|----------------|
| 15 min | ~30s | ~2,300 |
| 30 min | ~60s | ~4,500 |
| 60 min | ~2 min | ~9,000 |

First run downloads the model (~3GB for large-v3). Subsequent runs use the cached model.

## Troubleshooting

### CUDA errors

- **"Could not locate cudnn_ops64_9.dll"**: The CUDA DLL preloader should handle this automatically. If it persists, ensure `nvidia-cudnn-cu12` is installed: `pip install "nvidia-cudnn-cu12>=9.0,<10"`
- **"CUDA unavailable" warning**: The tool falls back to CPU automatically. To fix: update NVIDIA drivers, ensure CUDA toolkit is installed, verify with `nvidia-smi`
- **Slow transcription**: You're likely running on CPU. Check the CUDA warning above.

### yt-dlp errors

- **"Video unavailable"**: Video may be private, age-restricted, or geo-blocked
- **ffmpeg not found**: Install ffmpeg and ensure it's on your PATH

### Word count warnings

- **"Low word count"**: Transcript may be incomplete. Try `--force-whisper` if using YouTube subs, or try a different `--model`
- **"High word count"**: May indicate Whisper hallucination (repeated phrases). Check the output for repetition.
