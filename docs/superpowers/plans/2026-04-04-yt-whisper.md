# yt-whisper Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI tool that transcribes YouTube videos by downloading audio via yt-dlp and transcribing with faster-whisper on local CUDA GPU.

**Architecture:** A 7-module pipeline: `cli.py` orchestrates `downloader.py` (subtitle check + audio extraction via yt-dlp Python API) → `transcriber.py` (faster-whisper with CUDA/CPU fallback) → `formatter.py` (markdown + JSON output). Supporting modules: `prompts.py` (domain vocabulary), `cuda_preload.py` (Windows DLL preloading). Entry via `__main__.py`.

**Tech Stack:** Python 3.10+, yt-dlp (Python API), faster-whisper, CUDA 12 / cuDNN 9, ffmpeg

**Spec:** `docs/superpowers/specs/2026-04-04-yt-whisper-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `yt_whisper/__init__.py` | Create | Version string |
| `yt_whisper/__main__.py` | Create | `python -m` entrypoint |
| `yt_whisper/prompts.py` | Create | Named prompt profiles + resolve function |
| `yt_whisper/cuda_preload.py` | Create | Windows CUDA DLL preloading |
| `yt_whisper/downloader.py` | Create | yt-dlp subtitle check + audio download |
| `yt_whisper/transcriber.py` | Create | faster-whisper transcription with CUDA fallback |
| `yt_whisper/formatter.py` | Create | Markdown + JSON output generation |
| `yt_whisper/cli.py` | Create | Argparse, orchestration, validation, exit codes |
| `tests/test_prompts.py` | Create | Tests for prompt resolution |
| `tests/test_formatter.py` | Create | Tests for formatting, paragraphing, duration |
| `tests/test_downloader.py` | Create | Tests for subtitle parsing, metadata extraction |
| `tests/test_transcriber.py` | Create | Tests for CUDA fallback, segment collection |
| `tests/test_cli.py` | Create | Tests for arg parsing, validation, orchestration |
| `requirements.txt` | Create | Dependencies |
| `README.md` | Create | Documentation |
| `.gitignore` | Create | Ignore patterns |

---

## Chunk 1: Project Scaffolding and Pure-Logic Modules

### Task 1: Project scaffolding

**Files:**
- Create: `yt_whisper/__init__.py`
- Create: `yt_whisper/__main__.py`
- Create: `requirements.txt`
- Create: `.gitignore`

- [ ] **Step 1: Initialize git repo**

```bash
cd C:/Users/ericr/OneDrive/Documents/yt-whisper
git init
```

- [ ] **Step 2: Create package init**

Create `yt_whisper/__init__.py`:
```python
"""yt-whisper: YouTube video transcription via faster-whisper."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Create __main__.py entrypoint**

Create `yt_whisper/__main__.py`:
```python
from yt_whisper.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create requirements.txt**

Create `requirements.txt`:
```
yt-dlp>=2024.01.01
faster-whisper>=1.0.0
nvidia-cublas-cu12>=12.0,<13
nvidia-cudnn-cu12>=9.0,<10
```

- [ ] **Step 5: Create .gitignore**

Create `.gitignore`:
```
transcripts/
*.wav
__pycache__/
*.pyc
*.pyo
.env
*.egg-info/
dist/
build/
.pytest_cache/
```

- [ ] **Step 6: Create transcripts/.gitkeep and tests/__init__.py**

```bash
mkdir -p transcripts tests
touch transcripts/.gitkeep tests/__init__.py
```

- [ ] **Step 7: Commit**

```bash
git add yt_whisper/__init__.py yt_whisper/__main__.py requirements.txt .gitignore transcripts/.gitkeep tests/__init__.py
git commit -m "feat: project scaffolding with package init, entrypoint, deps"
```

---

### Task 2: Prompt profiles module

**Files:**
- Create: `yt_whisper/prompts.py`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write failing tests for prompt resolution**

Create `tests/test_prompts.py`:
```python
from yt_whisper.prompts import resolve_prompt, PROMPTS


def test_general_returns_none():
    assert resolve_prompt("general") is None


def test_grc_returns_prompt_string():
    result = resolve_prompt("grc")
    assert isinstance(result, str)
    assert "NIST" in result
    assert "FedRAMP" in result


def test_infosec_returns_prompt_string():
    result = resolve_prompt("infosec")
    assert isinstance(result, str)
    assert "CVE" in result
    assert "MITRE ATT&CK" in result


def test_unknown_key_treated_as_custom_string():
    custom = "my custom vocabulary terms"
    assert resolve_prompt(custom) == custom


def test_prompts_dict_has_expected_keys():
    assert set(PROMPTS.keys()) == {"general", "grc", "infosec"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_prompts.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'yt_whisper.prompts'`

- [ ] **Step 3: Implement prompts.py**

Create `yt_whisper/prompts.py`:
```python
"""Named domain vocabulary prompt profiles for Whisper transcription."""

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
    """Return prompt text. Known key -> stored value. Unknown key -> treat as custom string."""
    return PROMPTS.get(name_or_string, name_or_string)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_prompts.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add yt_whisper/prompts.py tests/test_prompts.py
git commit -m "feat: add prompt profiles module with general, grc, infosec profiles"
```

---

### Task 3: CUDA preload module

**Files:**
- Create: `yt_whisper/cuda_preload.py`

- [ ] **Step 1: Implement cuda_preload.py**

Create `yt_whisper/cuda_preload.py`:
```python
"""
Windows-specific DLL preloading for faster-whisper CUDA support.

Problem: Microsoft Store Python's sandbox prevents normal DLL discovery.
Solution: Explicitly load DLLs via ctypes.WinDLL() before importing faster_whisper.

This module must be called BEFORE any import of faster_whisper or ctranslate2.
"""

import os
import sys
import ctypes
import importlib.util


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

- [ ] **Step 2: Verify module imports cleanly**

```bash
python -c "from yt_whisper.cuda_preload import ensure_dlls; ensure_dlls(); print('OK')"
```

Expected: `OK` (no errors — it either loads DLLs or silently skips if nvidia package not found)

- [ ] **Step 3: Commit**

```bash
git add yt_whisper/cuda_preload.py
git commit -m "feat: add CUDA DLL preloading for Windows MS Store Python"
```

---

### Task 4: Formatter module — duration and paragraph helpers

**Files:**
- Create: `yt_whisper/formatter.py`
- Create: `tests/test_formatter.py`

- [ ] **Step 1: Write failing tests for duration formatting**

Create `tests/test_formatter.py`:
```python
from yt_whisper.formatter import format_duration, format_paragraphs


def test_duration_seconds_only():
    assert format_duration(42) == "0:42"


def test_duration_minutes():
    assert format_duration(262) == "4:22"


def test_duration_minutes_leading_zero_seconds():
    assert format_duration(2707) == "45:07"


def test_duration_hours():
    assert format_duration(3930) == "1:05:30"


def test_duration_zero():
    assert format_duration(0) == "0:00"


def test_paragraphs_groups_sentences():
    text = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten."
    result = format_paragraphs(text)
    paragraphs = result.split("\n\n")
    assert len(paragraphs) == 2
    assert "One." in paragraphs[0]
    assert "Five." in paragraphs[0]
    assert "Six." in paragraphs[1]


def test_paragraphs_short_text_single_paragraph():
    text = "Just one sentence."
    result = format_paragraphs(text)
    assert "\n\n" not in result
    assert result == "Just one sentence."


def test_paragraphs_handles_question_marks():
    text = "What is this? It is a test. Does it work? Yes it does. Great!"
    result = format_paragraphs(text)
    assert "What is this?" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_formatter.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement duration and paragraph helpers**

Create `yt_whisper/formatter.py`:
```python
"""Format transcription output as markdown and JSON."""

import json
import os
import re


def format_duration(seconds):
    """Format seconds as H:MM:SS (if >= 1 hour) or M:SS."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_paragraphs(text):
    """Split text into paragraphs of ~5 sentences each."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]

    if len(sentences) <= 5:
        return " ".join(sentences)

    paragraphs = []
    for i in range(0, len(sentences), 5):
        paragraph = " ".join(sentences[i:i + 5])
        paragraphs.append(paragraph)

    return "\n\n".join(paragraphs)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_formatter.py -v
```

Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add yt_whisper/formatter.py tests/test_formatter.py
git commit -m "feat: add duration formatting and paragraph helpers"
```

---

### Task 5: Formatter module — markdown and JSON output

**Files:**
- Modify: `yt_whisper/formatter.py`
- Modify: `tests/test_formatter.py`

- [ ] **Step 1: Write failing tests for format_output**

Append to `tests/test_formatter.py`:
```python
import json
import os
import tempfile


def _sample_metadata():
    return {
        "video_id": "test123",
        "title": "Test Video Title",
        "channel": "Test Channel",
        "upload_date": "20260101",
        "duration": 600,
        "url": "https://www.youtube.com/watch?v=test123",
    }


def _sample_segments():
    return [
        {"start": 0.0, "end": 3.0, "text": "Hello world."},
        {"start": 3.0, "end": 6.0, "text": "This is a test."},
    ]


def test_format_output_md_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = format_output(
            _sample_segments(), _sample_metadata(), "md", tmpdir,
            model="large-v3", prompt_profile="grc", method="whisper", language="en",
        )
        assert len(paths) == 1
        assert paths[0].endswith(".md")
        assert os.path.exists(paths[0])
        content = open(paths[0]).read()
        assert "# Test Video Title" in content
        assert "**Channel**: Test Channel" in content
        assert "whisper (large-v3 / grc)" in content


def test_format_output_json_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = format_output(
            _sample_segments(), _sample_metadata(), "json", tmpdir,
            model="large-v3", prompt_profile="grc", method="whisper", language="en",
        )
        assert len(paths) == 1
        assert paths[0].endswith(".json")
        data = json.loads(open(paths[0]).read())
        assert data["video_id"] == "test123"
        assert data["transcription_method"] == "whisper"
        assert data["model"] == "large-v3"
        assert len(data["segments"]) == 2
        assert data["word_count"] == 6


def test_format_output_both_creates_two_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = format_output(
            _sample_segments(), _sample_metadata(), "both", tmpdir,
            model="large-v3", prompt_profile="general", method="whisper", language="en",
        )
        assert len(paths) == 2
        extensions = {os.path.splitext(p)[1] for p in paths}
        assert extensions == {".md", ".json"}


def test_format_output_youtube_subs_string_input():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = format_output(
            "Hello world. This is a test.", _sample_metadata(), "json", tmpdir,
            model=None, prompt_profile=None, method="youtube_subs", language="en",
        )
        data = json.loads(open(paths[0]).read())
        assert data["transcription_method"] == "youtube_subs"
        assert data["segments"] is None
        assert data["model"] is None


def test_format_output_md_general_prompt_no_slash():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = format_output(
            _sample_segments(), _sample_metadata(), "md", tmpdir,
            model="large-v3", prompt_profile="general", method="whisper", language="en",
        )
        content = open(paths[0]).read()
        assert "whisper (large-v3)" in content
        assert "/ general" not in content
```

- [ ] **Step 2: Run tests to verify new tests fail**

```bash
python -m pytest tests/test_formatter.py -v
```

Expected: New tests FAIL — `format_output` not defined

- [ ] **Step 3: Implement format_output**

Add to `yt_whisper/formatter.py`:
```python
def format_output(text_or_segments, metadata, output_format, output_dir,
                  model=None, prompt_profile=None, method="whisper", language="en"):
    """Write transcript to markdown and/or JSON files. Returns list of paths written."""
    os.makedirs(output_dir, exist_ok=True)
    video_id = metadata["video_id"]

    # Assemble full text
    if isinstance(text_or_segments, list):
        full_text = " ".join(seg["text"].strip() for seg in text_or_segments)
        segments = text_or_segments
    else:
        full_text = text_or_segments
        segments = None

    full_text = re.sub(r'\s+', ' ', full_text).strip()
    word_count = len(full_text.split())
    duration_formatted = format_duration(metadata["duration"])

    # Build method display string
    if method == "youtube_subs":
        method_display = "youtube_subs"
    elif prompt_profile and prompt_profile != "general":
        method_display = f"whisper ({model} / {prompt_profile})"
    else:
        method_display = f"whisper ({model})"

    paths = []

    if output_format in ("md", "both"):
        md_path = os.path.join(output_dir, f"{video_id}.md")
        paragraphed = format_paragraphs(full_text)
        md_content = (
            f"# {metadata['title']}\n\n"
            f"- **Channel**: {metadata['channel']}\n"
            f"- **Date**: {metadata['upload_date']}\n"
            f"- **Duration**: {duration_formatted}\n"
            f"- **URL**: {metadata['url']}\n"
            f"- **Language**: {language}\n"
            f"- **Word Count**: {word_count}\n"
            f"- **Transcription Method**: {method_display}\n\n"
            f"---\n\n"
            f"{paragraphed}\n"
        )
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        paths.append(md_path)

    if output_format in ("json", "both"):
        json_path = os.path.join(output_dir, f"{video_id}.json")
        json_data = {
            "video_id": video_id,
            "title": metadata["title"],
            "channel": metadata["channel"],
            "url": metadata["url"],
            "upload_date": metadata["upload_date"],
            "language": language,
            "duration_seconds": metadata["duration"],
            "duration_formatted": duration_formatted,
            "word_count": word_count,
            "transcription_method": method,
            "model": model,
            "prompt_profile": prompt_profile if prompt_profile != "general" else None,
            "segments": segments,
            "full_text": full_text,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        paths.append(json_path)

    return paths
```

- [ ] **Step 4: Run all formatter tests**

```bash
python -m pytest tests/test_formatter.py -v
```

Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add yt_whisper/formatter.py tests/test_formatter.py
git commit -m "feat: add markdown and JSON output formatting"
```

---

## Chunk 2: Downloader Module

### Task 6: Downloader — subtitle parsing helpers

**Files:**
- Create: `yt_whisper/downloader.py`
- Create: `tests/test_downloader.py`

- [ ] **Step 1: Write failing tests for subtitle text parsing**

Create `tests/test_downloader.py`:
```python
from yt_whisper.downloader import parse_json3_subtitles, VideoUnavailableError


def test_parse_json3_basic():
    json3_data = {
        "events": [
            {"segs": [{"utf8": "Hello "}, {"utf8": "world."}]},
            {"segs": [{"utf8": " This is a test."}]},
        ]
    }
    result = parse_json3_subtitles(json3_data)
    assert result == "Hello world. This is a test."


def test_parse_json3_strips_html_tags():
    json3_data = {
        "events": [
            {"segs": [{"utf8": "<c>Hello</c> world."}]},
        ]
    }
    result = parse_json3_subtitles(json3_data)
    assert "<c>" not in result
    assert "Hello world." in result


def test_parse_json3_handles_newlines():
    json3_data = {
        "events": [
            {"segs": [{"utf8": "Line one.\n"}]},
            {"segs": [{"utf8": "Line two."}]},
        ]
    }
    result = parse_json3_subtitles(json3_data)
    assert "\n" not in result
    assert "Line one. Line two." in result


def test_parse_json3_skips_events_without_segs():
    json3_data = {
        "events": [
            {"tStartMs": 0},
            {"segs": [{"utf8": "Hello."}]},
        ]
    }
    result = parse_json3_subtitles(json3_data)
    assert result == "Hello."


def test_parse_json3_empty_events():
    json3_data = {"events": []}
    result = parse_json3_subtitles(json3_data)
    assert result == ""


def test_video_unavailable_error_is_exception():
    err = VideoUnavailableError("Video is private")
    assert isinstance(err, Exception)
    assert str(err) == "Video is private"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_downloader.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement subtitle parsing and exception**

Create `yt_whisper/downloader.py`:
```python
"""YouTube subtitle checking and audio downloading via yt-dlp Python API."""

import os
import re


class VideoUnavailableError(Exception):
    """Raised when yt-dlp cannot access the video."""
    pass


def parse_json3_subtitles(json3_data):
    """Extract plain text from json3 subtitle format.

    json3 contains events[] with segs[] arrays. Each seg has a utf8 field.
    Strips HTML tags, collapses whitespace.
    """
    fragments = []
    for event in json3_data.get("events", []):
        segs = event.get("segs")
        if not segs:
            continue
        for seg in segs:
            text = seg.get("utf8", "")
            if text:
                fragments.append(text)

    raw = "".join(fragments)
    # Strip HTML tags (e.g., <c>, </c>)
    raw = re.sub(r'<[^>]+>', '', raw)
    # Replace newlines with spaces
    raw = raw.replace("\n", " ")
    # Collapse whitespace
    raw = re.sub(r'\s+', ' ', raw).strip()
    return raw
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_downloader.py -v
```

Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add yt_whisper/downloader.py tests/test_downloader.py
git commit -m "feat: add subtitle parsing and VideoUnavailableError"
```

---

### Task 7: Downloader — check_subtitles and download_audio

**Files:**
- Modify: `yt_whisper/downloader.py`
- Modify: `tests/test_downloader.py`

Note: These functions call the yt-dlp API which requires network access. Unit tests will mock the yt-dlp calls. Integration tests happen in the manual testing phase (Task 11).

- [ ] **Step 1: Write failing tests for check_subtitles**

Append to `tests/test_downloader.py`:
```python
import json
from unittest.mock import patch, MagicMock
from yt_whisper.downloader import check_subtitles, download_audio


def _make_info_dict(subtitles=None, automatic_captions=None):
    """Build a minimal yt-dlp info_dict for testing."""
    return {
        "id": "test123",
        "title": "Test Video",
        "channel": "Test Channel",
        "upload_date": "20260101",
        "duration": 600,
        "webpage_url": "https://www.youtube.com/watch?v=test123",
        "subtitles": subtitles or {},
        "automatic_captions": automatic_captions or {},
    }


@patch("yt_whisper.downloader.yt_dlp.YoutubeDL")
def test_check_subtitles_returns_metadata(mock_ydl_class):
    mock_ydl = MagicMock()
    mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
    mock_ydl.extract_info.return_value = _make_info_dict()

    text, metadata = check_subtitles("https://youtube.com/watch?v=test123")
    assert metadata["video_id"] == "test123"
    assert metadata["title"] == "Test Video"
    assert metadata["duration"] == 600
    assert metadata["url"] == "https://www.youtube.com/watch?v=test123"


@patch("yt_whisper.downloader.yt_dlp.YoutubeDL")
def test_check_subtitles_no_subs_returns_none(mock_ydl_class):
    mock_ydl = MagicMock()
    mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
    mock_ydl.extract_info.return_value = _make_info_dict()

    text, metadata = check_subtitles("https://youtube.com/watch?v=test123")
    assert text is None


@patch("yt_whisper.downloader.yt_dlp.YoutubeDL")
def test_check_subtitles_prefers_manual_over_auto(mock_ydl_class):
    mock_ydl = MagicMock()
    mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)

    json3_content = json.dumps({"events": [{"segs": [{"utf8": "Manual sub."}]}]}).encode()
    mock_response = MagicMock()
    mock_response.read.return_value = json3_content
    mock_ydl.urlopen.return_value = mock_response

    mock_ydl.extract_info.return_value = _make_info_dict(
        subtitles={"en": [{"ext": "json3", "url": "http://example.com/sub.json3"}]},
        automatic_captions={"en": [{"ext": "json3", "url": "http://example.com/auto.json3"}]},
    )

    text, metadata = check_subtitles("https://youtube.com/watch?v=test123")
    assert text == "Manual sub."
    # Should have fetched the manual sub URL, not auto
    mock_ydl.urlopen.assert_called_once_with("http://example.com/sub.json3")
```

- [ ] **Step 2: Run tests to verify new tests fail**

```bash
python -m pytest tests/test_downloader.py -v -k "check_subtitles"
```

Expected: FAIL — `check_subtitles` not defined

- [ ] **Step 3: Implement check_subtitles**

Add to top of `yt_whisper/downloader.py` (after existing imports):
```python
import json
import yt_dlp
```

Add function to `yt_whisper/downloader.py`:
```python
def _build_language_priority(language):
    """Build language code priority list. e.g., 'en' -> ['en', 'en-US', 'en-GB']."""
    variants = [language]
    if "-" not in language and "_" not in language:
        variants.extend([f"{language}-US", f"{language}-GB"])
    return variants


def _find_subtitle_entry(caption_dict, lang_priority):
    """Find best subtitle entry from a caption dict. Returns (url, ext) or None."""
    format_priority = ["json3", "vtt", "srv1"]
    for lang in lang_priority:
        if lang not in caption_dict:
            continue
        entries = caption_dict[lang]
        for fmt in format_priority:
            for entry in entries:
                if entry.get("ext") == fmt:
                    return entry["url"], fmt
    return None


def check_subtitles(url, language="en"):
    """Check YouTube for existing subtitles. Returns (text_or_None, metadata)."""
    ydl_opts = {"quiet": True, "no_warnings": True}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            metadata = {
                "video_id": info.get("id", ""),
                "title": info.get("title", "Unknown"),
                "channel": info.get("channel", info.get("uploader", "Unknown")),
                "upload_date": info.get("upload_date", ""),
                "duration": info.get("duration", 0),
                "url": info.get("webpage_url", url),
            }

            lang_priority = _build_language_priority(language)
            subtitles = info.get("subtitles") or {}
            auto_captions = info.get("automatic_captions") or {}

            # Prefer manual subs over auto-generated
            result = _find_subtitle_entry(subtitles, lang_priority)
            if result is None:
                result = _find_subtitle_entry(auto_captions, lang_priority)

            if result is None:
                return None, metadata

            sub_url, fmt = result

            # Fetch subtitle content using same YoutubeDL instance
            try:
                response = ydl.urlopen(sub_url)
                raw_data = response.read()
            except Exception as e:
                print(f"Warning: Failed to download subtitles: {e}")
                return None, metadata

    except yt_dlp.utils.DownloadError as e:
        raise VideoUnavailableError(str(e)) from e

    if fmt == "json3":
        json3_data = json.loads(raw_data)
        text = parse_json3_subtitles(json3_data)
    else:
        # VTT/SRV1 fallback: strip tags and timestamps, extract text
        text = raw_data.decode("utf-8", errors="replace")
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}', '', text)
        text = re.sub(r'WEBVTT.*?\n\n', '', text, flags=re.DOTALL)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

    return text if text else None, metadata
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_downloader.py -v
```

Expected: All passed

- [ ] **Step 5: Implement download_audio**

Add to `yt_whisper/downloader.py`:
```python
def download_audio(url, temp_dir, metadata, verbose=False):
    """Download audio from YouTube as 16kHz mono WAV. Returns audio file path."""
    video_id = metadata["video_id"]

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

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        raise VideoUnavailableError(str(e)) from e

    audio_path = os.path.join(temp_dir, f"{video_id}.wav")
    if not os.path.exists(audio_path):
        raise VideoUnavailableError(
            f"Audio file not found after download. Expected: {audio_path}"
        )

    return audio_path
```

- [ ] **Step 6: Commit**

```bash
git add yt_whisper/downloader.py tests/test_downloader.py
git commit -m "feat: add subtitle checking and audio download via yt-dlp API"
```

---

## Chunk 3: Transcriber and CLI Orchestration

### Task 8: Transcriber module

**Files:**
- Create: `yt_whisper/transcriber.py`
- Create: `tests/test_transcriber.py`

- [ ] **Step 1: Write failing tests for transcriber**

Create `tests/test_transcriber.py`:
```python
import pytest
from unittest.mock import patch, MagicMock, call
from yt_whisper.transcriber import TranscriptionError


def _mock_segment(start, end, text):
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = text
    return seg


def test_transcription_error_is_exception():
    err = TranscriptionError("No speech detected")
    assert isinstance(err, Exception)
    assert str(err) == "No speech detected"


@patch("yt_whisper.transcriber.cuda_preload")
@patch("yt_whisper.transcriber.faster_whisper")
def test_transcribe_calls_cuda_preload(mock_fw, mock_preload):
    """Verify cuda_preload.ensure_dlls() is called before model creation."""
    mock_model = MagicMock()
    mock_fw.WhisperModel.return_value = mock_model
    mock_model.transcribe.return_value = (
        iter([_mock_segment(0.0, 3.0, " Hello world.")]),
        MagicMock(),
    )

    # Patch the local import inside transcribe()
    with patch.dict("sys.modules", {"faster_whisper": mock_fw}):
        from yt_whisper.transcriber import transcribe
        result = transcribe("test.wav", "tiny", None, "en", False)

    mock_preload.ensure_dlls.assert_called_once()
    assert len(result) == 1
    assert result[0]["text"] == "Hello world."
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 3.0


@patch("yt_whisper.transcriber.cuda_preload")
def test_transcribe_empty_segments_raises(mock_preload):
    mock_fw = MagicMock()
    mock_model = MagicMock()
    mock_fw.WhisperModel.return_value = mock_model
    mock_model.transcribe.return_value = (iter([]), MagicMock())

    with patch.dict("sys.modules", {"faster_whisper": mock_fw}):
        from yt_whisper.transcriber import transcribe
        with pytest.raises(TranscriptionError, match="No speech detected"):
            transcribe("test.wav", "tiny", None, "en", False)


@patch("yt_whisper.transcriber.cuda_preload")
def test_transcribe_cuda_fallback_to_cpu(mock_preload, capsys):
    """Verify CUDA failure falls back to CPU with warning."""
    mock_fw = MagicMock()

    # First WhisperModel() call raises RuntimeError (CUDA fail)
    # Second call succeeds (CPU fallback)
    mock_cpu_model = MagicMock()
    mock_cpu_model.transcribe.return_value = (
        iter([_mock_segment(0.0, 3.0, " Fallback text.")]),
        MagicMock(),
    )
    mock_fw.WhisperModel.side_effect = [RuntimeError("CUDA error"), mock_cpu_model]

    with patch.dict("sys.modules", {"faster_whisper": mock_fw}):
        from yt_whisper.transcriber import transcribe
        result = transcribe("test.wav", "tiny", None, "en", False)

    assert len(result) == 1
    assert result[0]["text"] == "Fallback text."

    captured = capsys.readouterr()
    assert "CUDA unavailable" in captured.out
    assert "falling back to CPU" in captured.out

    # Verify second call used cpu/int8
    calls = mock_fw.WhisperModel.call_args_list
    assert calls[1] == call("tiny", device="cpu", compute_type="int8")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_transcriber.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement transcriber.py**

Create `yt_whisper/transcriber.py`:
```python
"""Transcribe audio using faster-whisper with CUDA GPU support."""

import os
import sys

from yt_whisper import cuda_preload


class TranscriptionError(Exception):
    """Raised on transcription failure (empty output, model load error)."""
    pass


def _check_model_cached(model_size):
    """Best-effort check if model is already downloaded."""
    cache_dir = os.environ.get(
        "HF_HUB_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    )
    model_dir = f"models--Systran--faster-whisper-{model_size}"
    return os.path.isdir(os.path.join(cache_dir, model_dir))


def transcribe(audio_path, model_size, prompt_text, language, verbose):
    """Transcribe audio file. Returns list of {"start", "end", "text"} dicts."""
    cuda_preload.ensure_dlls()

    # Local import — faster_whisper must not be imported at module level (Anti-Pattern #1)
    from faster_whisper import WhisperModel

    # Detect CUDA availability
    device = "cuda"
    compute_type = "float16"
    try:
        if not _check_model_cached(model_size):
            size_hint = "~3GB" if "large" in model_size else "~1GB"
            print(f"Downloading Whisper model '{model_size}' ({size_hint}). One-time download.")

        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except (RuntimeError, ValueError) as e:
        print(
            f"Warning: CUDA unavailable — falling back to CPU. "
            f"This will be significantly slower. "
            f"Check NVIDIA drivers and CUDA toolkit. ({e})"
        )
        device = "cpu"
        compute_type = "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments_gen, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,
        initial_prompt=prompt_text,
    )

    segments = []
    for seg in segments_gen:
        text = seg.text.strip()
        if verbose:
            print(f"  [{seg.start:.1f}s → {seg.end:.1f}s] {text}")
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": text,
        })

    if not segments:
        raise TranscriptionError("No speech detected in audio")

    return segments
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_transcriber.py -v
```

Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add yt_whisper/transcriber.py tests/test_transcriber.py
git commit -m "feat: add faster-whisper transcriber with CUDA fallback"
```

---

### Task 9: CLI orchestration module

**Files:**
- Create: `yt_whisper/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for argument parsing**

Create `tests/test_cli.py`:
```python
from yt_whisper.cli import build_parser, validate_word_count


def test_parser_required_url():
    parser = build_parser()
    args = parser.parse_args(["https://youtube.com/watch?v=test"])
    assert args.url == "https://youtube.com/watch?v=test"


def test_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["https://youtube.com/watch?v=test"])
    assert args.prompt == "general"
    assert args.force_whisper is False
    assert args.output_dir == "./transcripts"
    assert args.model == "large-v3"
    assert args.output_format == "both"
    assert args.language == "en"
    assert args.verbose is False


def test_parser_all_args():
    parser = build_parser()
    args = parser.parse_args([
        "https://youtube.com/watch?v=test",
        "--prompt", "grc",
        "--force-whisper",
        "--output-dir", "/tmp/out",
        "--model", "medium",
        "--format", "json",
        "--language", "es",
        "--verbose",
    ])
    assert args.prompt == "grc"
    assert args.force_whisper is True
    assert args.output_dir == "/tmp/out"
    assert args.model == "medium"
    assert args.output_format == "json"
    assert args.language == "es"
    assert args.verbose is True


def test_validate_word_count_normal(capsys):
    wpm = validate_word_count(1500, 600)  # 150 wpm, 10 min
    captured = capsys.readouterr()
    assert "Warning" not in captured.out
    assert wpm == 150.0


def test_validate_word_count_low(capsys):
    wpm = validate_word_count(200, 600)  # 20 wpm
    captured = capsys.readouterr()
    assert "Low word count" in captured.out
    assert wpm is not None


def test_validate_word_count_high(capsys):
    wpm = validate_word_count(5000, 600)  # 500 wpm
    captured = capsys.readouterr()
    assert "High word count" in captured.out
    assert wpm is not None


def test_validate_word_count_short_video(capsys):
    wpm = validate_word_count(10, 20)  # 20 seconds
    captured = capsys.readouterr()
    assert "too short" in captured.out
    assert wpm is None


def test_validate_word_count_boundary_30s(capsys):
    wpm = validate_word_count(75, 30)  # exactly 30 seconds, 150 wpm
    captured = capsys.readouterr()
    assert "too short" not in captured.out
    assert wpm is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_cli.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement cli.py**

Create `yt_whisper/cli.py`:
```python
"""CLI entrypoint and orchestration for yt-whisper."""

import argparse
import os
import sys
import tempfile

from yt_whisper.downloader import check_subtitles, download_audio, VideoUnavailableError
from yt_whisper.formatter import format_output, format_duration
from yt_whisper.prompts import resolve_prompt
from yt_whisper.transcriber import transcribe, TranscriptionError

MIN_VALIDATION_SECONDS = 30


def build_parser():
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        prog="yt-whisper",
        description="Transcribe YouTube videos using faster-whisper or YouTube subtitles.",
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--prompt", default="general",
                        help="Named prompt profile (general, grc, infosec) or custom string")
    parser.add_argument("--force-whisper", action="store_true",
                        help="Skip YouTube subtitle check, always use Whisper")
    parser.add_argument("--output-dir", default="./transcripts",
                        help="Output directory (default: ./transcripts)")
    parser.add_argument("--model", default="large-v3",
                        help="Whisper model size (default: large-v3)")
    parser.add_argument("--format", dest="output_format", default="both",
                        choices=["md", "json", "both"],
                        help="Output format (default: both)")
    parser.add_argument("--language", default="en",
                        help="Language code (default: en)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show progress details and segment timing")
    return parser


def validate_word_count(word_count, duration_seconds):
    """Print warnings if word count seems abnormal for the video duration.
    Returns wpm as float or None if too short to validate."""
    if duration_seconds < MIN_VALIDATION_SECONDS:
        print("  Note: Video too short for word count validation.")
        return None

    wpm = word_count / (duration_seconds / 60)

    if wpm < 100:
        print(f"  Warning: Low word count ({wpm:.0f} words/min). "
              f"Expected ~150. Transcript may be incomplete.")
    if wpm > 200:
        print(f"  Warning: High word count ({wpm:.0f} words/min). "
              f"May indicate repeated or hallucinated text.")
    return wpm


def main():
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    prompt_text = resolve_prompt(args.prompt)

    try:
        # Step 1: Check for existing subtitles
        subtitle_text = None
        metadata = None

        if args.verbose:
            print(f"Fetching video info...")

        subtitle_text, metadata = check_subtitles(args.url, args.language)

        if subtitle_text and not args.force_whisper:
            method = "youtube_subs"
            text_or_segments = subtitle_text
            if args.verbose:
                print(f"Found YouTube subtitles ({len(subtitle_text.split())} words)")
        else:
            if args.verbose:
                if args.force_whisper:
                    print("Forced Whisper transcription (--force-whisper)")
                else:
                    print("No usable subtitles found. Transcribing with Whisper...")

            # Step 2: Download audio and transcribe
            method = "whisper"
            with tempfile.TemporaryDirectory() as temp_dir:
                audio_path = download_audio(
                    args.url, temp_dir, metadata, verbose=args.verbose
                )
                if args.verbose:
                    print(f"Audio downloaded: {audio_path}")

                # Step 3: Transcribe
                text_or_segments = transcribe(
                    audio_path, args.model, prompt_text, args.language, args.verbose
                )

        # Step 4: Format output
        output_paths = format_output(
            text_or_segments, metadata, args.output_format, args.output_dir,
            model=args.model if method == "whisper" else None,
            prompt_profile=args.prompt if method == "whisper" else None,
            method=method,
            language=args.language,
        )

        # Step 5: Summary
        if isinstance(text_or_segments, list):
            full_text = " ".join(seg["text"] for seg in text_or_segments)
        else:
            full_text = text_or_segments

        word_count = len(full_text.split())
        duration_formatted = format_duration(metadata["duration"])

        # Validate and get wpm
        wpm = validate_word_count(word_count, metadata["duration"])

        print(f"\n\u2713 {metadata['title']}")
        print(f"  Duration:    {duration_formatted}")
        if wpm is not None:
            print(f"  Words:       {word_count} ({wpm:.0f} words/min)")
        else:
            print(f"  Words:       {word_count}")
        print(f"  Method:      {method}")
        for i, path in enumerate(output_paths):
            prefix = "  Output:      " if i == 0 else "               "
            print(f"{prefix}{path}")

    except VideoUnavailableError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except TranscriptionError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_cli.py -v
```

Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add yt_whisper/cli.py tests/test_cli.py
git commit -m "feat: add CLI orchestration with arg parsing, validation, error handling"
```

---

## Chunk 4: Documentation and Integration Testing

### Task 10: README and documentation

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create README.md**

Create `README.md`:
```markdown
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

- **"Could not locate cudnn_ops64_9.dll"**: The CUDA DLL preloader should handle this automatically. If it persists, ensure `nvidia-cudnn-cu12` is installed: `pip install nvidia-cudnn-cu12>=9.0,<10`
- **"CUDA unavailable" warning**: The tool falls back to CPU automatically. To fix: update NVIDIA drivers, ensure CUDA toolkit is installed, verify with `nvidia-smi`
- **Slow transcription**: You're likely running on CPU. Check the CUDA warning above.

### yt-dlp errors

- **"Video unavailable"**: Video may be private, age-restricted, or geo-blocked
- **ffmpeg not found**: Install ffmpeg and ensure it's on your PATH

### Word count warnings

- **"Low word count"**: Transcript may be incomplete. Try `--force-whisper` if using YouTube subs, or try a different `--model`
- **"High word count"**: May indicate Whisper hallucination (repeated phrases). Check the output for repetition.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with usage, options, troubleshooting"
```

---

### Task 11: Integration test — manual verification

**Files:** None (manual testing)

- [ ] **Step 1: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 2: Run all unit tests**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass

- [ ] **Step 3: Test with YouTube subs (short video)**

```bash
python -m yt_whisper https://www.youtube.com/watch?v=dQw4w9WgXcQ --format both --verbose
```

Expected:
- Should find YouTube subtitles and use them
- Creates `transcripts/dQw4w9WgXcQ.md` and `transcripts/dQw4w9WgXcQ.json`
- Summary shows method: `youtube_subs`
- Word count validation prints

- [ ] **Step 4: Test force-whisper mode**

```bash
python -m yt_whisper https://www.youtube.com/watch?v=dQw4w9WgXcQ --force-whisper --verbose
```

Expected:
- Downloads audio, transcribes with Whisper
- Summary shows method: `whisper`
- Output files overwritten with Whisper transcript

- [ ] **Step 5: Verify output files**

Check `transcripts/dQw4w9WgXcQ.md`:
- Has title, channel, date, duration, URL metadata
- Has paragraphed transcript text

Check `transcripts/dQw4w9WgXcQ.json`:
- Has all metadata fields
- Has segments array (for whisper) or null (for youtube_subs)
- Has full_text field

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "chore: integration testing complete"
```
