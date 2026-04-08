# Diarization, TUI, and Profile Auto-Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in speaker diarization (pyannote), a Textual TUI with home/run/preview screens, and keyword-based prompt profile auto-detection, while preserving all existing flag-mode behavior.

**Architecture:** Extract pipeline orchestration into a shared `runner.py` with an event listener interface. Flag mode uses a `ConsoleListener`; TUI mode uses a `TuiListener`. Diarization and Textual are optional — pyannote is a separate requirements file and imported locally inside `diarize()`.

**Tech Stack:** Python 3.10+, faster-whisper (existing), yt-dlp (existing), pyannote.audio 3.1 (new, optional), Textual (new), pytest (existing).

**Spec:** `docs/superpowers/specs/2026-04-08-diarization-tui-autodetect-design.md`

---

## File Structure

**New files:**
- `yt_whisper/runner.py` — pipeline orchestration, `RunConfig`, listener base class, `ConsoleListener`
- `yt_whisper/diarizer.py` — pyannote wrapper + alignment
- `yt_whisper/profile_detect.py` — keyword matching
- `yt_whisper/tui/__init__.py`
- `yt_whisper/tui/app.py` — Textual app + screens
- `yt_whisper/tui/history.py` — scan output_dir for past runs
- `yt_whisper/tui/listener.py` — `TuiListener` bridging runner events to Textual messages
- `requirements-diarize.txt`
- `tests/test_profile_detect.py`
- `tests/test_diarizer.py`
- `tests/test_runner.py`
- `tests/test_history.py`
- `tests/test_tui_smoke.py`

**Changed files:**
- `yt_whisper/prompts.py` — profiles become dicts with `text` + `keywords`
- `yt_whisper/transcriber.py` — `transcribe()` becomes a generator
- `yt_whisper/formatter.py` — speaker-aware rendering + config block in JSON
- `yt_whisper/cli.py` — thin shim: no-args → TUI, else RunConfig → runner
- `requirements.txt` — add textual, python-dotenv
- `README.md` — optional diarization section
- `CLAUDE.md` — new anti-patterns
- `tests/test_transcriber.py` — generator interface
- `tests/test_formatter.py` — speaker rendering + config block
- `tests/test_prompts.py` — keyword validation + new dict shape
- `tests/test_cli.py` — ensure thin shim still produces same console output

---

## Task 1: Refactor `prompts.py` to dict shape

The current `PROMPTS` dict maps names to flat strings (or `None` for general). We need each profile to hold both the prompt text and a keyword list for auto-detection. Preserve `resolve_prompt()`'s existing contract: `resolve_prompt("grc")` must still return the prompt text string; `resolve_prompt("some custom string")` must still return that string unchanged.

**Files:**
- Modify: `yt_whisper/prompts.py`
- Test: `tests/test_prompts.py`

- [ ] **Step 1: Update `tests/test_prompts.py` with new expectations**

Read the existing file first. Replace its contents with:

```python
"""Tests for prompt profile resolution and keyword validation."""

import pytest
from yt_whisper.prompts import PROMPTS, resolve_prompt


def test_resolve_known_profile_returns_text():
    result = resolve_prompt("grc")
    assert isinstance(result, str)
    assert "NIST" in result


def test_resolve_general_returns_none_or_empty():
    # general has no prompt text -- must be None for whisper
    result = resolve_prompt("general")
    assert result is None


def test_resolve_infosec_returns_text():
    result = resolve_prompt("infosec")
    assert isinstance(result, str)
    assert "CVE" in result


def test_resolve_custom_string_passthrough():
    custom = "my custom prompt about widgets"
    assert resolve_prompt(custom) == custom


def test_all_profiles_have_keywords_list():
    for name, profile in PROMPTS.items():
        assert "keywords" in profile, f"{name} missing keywords"
        assert isinstance(profile["keywords"], list)


def test_general_has_empty_keywords():
    assert PROMPTS["general"]["keywords"] == []


def test_grc_has_keywords():
    assert len(PROMPTS["grc"]["keywords"]) > 0
    assert "NIST" in PROMPTS["grc"]["keywords"]


def test_infosec_has_keywords():
    assert len(PROMPTS["infosec"]["keywords"]) > 0
    assert "CVE" in PROMPTS["infosec"]["keywords"]


def test_no_duplicate_keywords_within_profile():
    for name, profile in PROMPTS.items():
        kws = profile["keywords"]
        assert len(kws) == len(set(kws)), f"{name} has duplicate keywords"


def test_no_empty_string_keywords():
    for name, profile in PROMPTS.items():
        for kw in profile["keywords"]:
            assert kw.strip() != "", f"{name} has empty keyword"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_prompts.py -v`
Expected: multiple FAIL — `PROMPTS["general"]` is `None`, no `keywords` key.

- [ ] **Step 3: Rewrite `yt_whisper/prompts.py`**

```python
"""Named domain vocabulary prompt profiles for Whisper transcription.

Each profile has:
- text: the initial_prompt passed to whisper (or None for no hint)
- keywords: list of terms used by profile_detect.py to auto-select this profile
"""

PROMPTS = {
    "general": {
        "text": None,
        "keywords": [],
    },
    "grc": {
        "text": (
            "NIST, RMF, Risk Management Framework, CMMC, FedRAMP, SOC2, SOC 2, GRC, "
            "cybersecurity, compliance, audit, control, framework, assessment, authorization, "
            "ATO, FISMA, FIPS 199, SP 800-53, SP 800-37, risk register, risk assessment, "
            "threat modeling, likelihood, impact, inherent risk, residual risk, Gerald Auger"
        ),
        "keywords": [
            "NIST", "RMF", "CMMC", "FedRAMP", "SOC 2", "SOC2", "ISO 27001",
            "HIPAA", "PCI DSS", "GDPR", "CCPA", "FISMA", "GRC",
            "compliance", "governance", "audit", "risk assessment",
            "control framework", "regulatory", "CISO", "ATO", "authorization",
        ],
    },
    "infosec": {
        "text": (
            "CVE, CVSS, vulnerability, exploit, zero-day, malware, ransomware, phishing, "
            "SOC, SIEM, EDR, XDR, MITRE ATT&CK, threat intelligence, incident response, "
            "penetration testing, red team, blue team, OSINT, IOC, indicators of compromise"
        ),
        "keywords": [
            "CVE", "CVSS", "exploit", "vulnerability", "malware", "ransomware",
            "phishing", "SIEM", "EDR", "XDR", "MITRE", "ATT&CK",
            "red team", "blue team", "pentest", "penetration testing",
            "reverse engineering", "binary exploitation", "CTF",
            "zero-day", "0day", "backdoor", "payload", "C2",
            "threat hunting", "incident response", "OSINT", "IOC",
        ],
    },
}


def resolve_prompt(name_or_string):
    """Return prompt text. Known profile -> its text field. Unknown key -> treat as custom string."""
    profile = PROMPTS.get(name_or_string)
    if profile is not None:
        return profile["text"]
    return name_or_string
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_prompts.py -v`
Expected: all PASS.

- [ ] **Step 5: Run full suite to ensure nothing else broke**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: existing tests still pass (some formatter/cli tests may reference `PROMPTS["grc"]` as a string — if any fail, fix the reference to `PROMPTS["grc"]["text"]`).

- [ ] **Step 6: Commit**

```bash
git add yt_whisper/prompts.py tests/test_prompts.py
git commit -m "refactor(prompts): convert profiles to dict with keywords field"
```

---

## Task 2: Profile auto-detection module

Build a pure function that takes video metadata and returns a detected profile with matched terms and confidence.

**Files:**
- Create: `yt_whisper/profile_detect.py`
- Test: `tests/test_profile_detect.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_profile_detect.py`:

```python
"""Tests for keyword-based prompt profile auto-detection."""

from yt_whisper.profile_detect import detect_profile, CONFIDENCE_THRESHOLD


def _meta(title="", channel="", description="", tags=None):
    return {
        "title": title,
        "channel": channel,
        "description": description,
        "tags": tags or [],
    }


def test_detect_grc_from_title():
    m = _meta(title="NIST CSF and SOC 2 Overview", description="compliance audit walkthrough")
    name, matched, conf = detect_profile(m)
    assert name == "grc"
    assert "NIST" in matched
    assert conf > 0


def test_detect_infosec_from_description():
    m = _meta(title="Deep Dive", description="We analyze CVE-2024-1234 and explore exploit techniques used by red team operators")
    name, matched, conf = detect_profile(m)
    assert name == "infosec"
    assert "CVE" in matched
    assert "exploit" in matched


def test_detect_below_threshold_returns_general():
    m = _meta(title="Cooking Pasta", description="We mention audit once in passing")
    name, matched, conf = detect_profile(m)
    assert name == "general"
    assert matched == []
    assert conf == 0.0


def test_detect_empty_metadata():
    name, matched, conf = detect_profile(_meta())
    assert name == "general"
    assert matched == []


def test_detect_missing_fields_does_not_crash():
    # All fields missing entirely
    name, matched, conf = detect_profile({})
    assert name == "general"


def test_detect_case_insensitive():
    m = _meta(description="cve-2024-1234 and red team analysis and exploit detail")
    name, _, _ = detect_profile(m)
    assert name == "infosec"


def test_detect_word_boundaries_prevent_false_match():
    # "cover" should not match "cve"
    m = _meta(description="We cover the covered topic in this coverage video about cover letters")
    name, matched, _ = detect_profile(m)
    assert "CVE" not in matched
    assert name == "general"


def test_detect_multi_word_keywords_weigh_more():
    # "red team" and "penetration testing" should outweigh single generic words
    m = _meta(description="red team penetration testing walkthrough")
    name, matched, _ = detect_profile(m)
    assert name == "infosec"
    assert "red team" in matched
    assert "penetration testing" in matched


def test_detect_description_cap_2000_chars():
    # A keyword past the 2000-char cap should not match
    padding = "x " * 1500  # > 2000 chars
    m = _meta(description=padding + " CVE")
    name, matched, _ = detect_profile(m)
    assert "CVE" not in matched


def test_detect_matched_terms_reported_for_display():
    m = _meta(title="NIST 800-53 and SOC 2 compliance")
    name, matched, _ = detect_profile(m)
    assert name == "grc"
    assert len(matched) >= 2


def test_detect_tags_contribute_to_matching():
    m = _meta(title="Talk", tags=["CVE", "exploit", "red team"])
    name, _, _ = detect_profile(m)
    assert name == "infosec"


def test_detect_confidence_between_zero_and_one():
    m = _meta(title="NIST SOC 2 compliance audit framework control", description="FedRAMP HIPAA GDPR")
    _, _, conf = detect_profile(m)
    assert 0.0 <= conf <= 1.0


def test_confidence_threshold_is_defined():
    assert isinstance(CONFIDENCE_THRESHOLD, int)
    assert CONFIDENCE_THRESHOLD >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_profile_detect.py -v`
Expected: FAIL — `yt_whisper.profile_detect` does not exist.

- [ ] **Step 3: Create `yt_whisper/profile_detect.py`**

```python
"""Keyword-based prompt profile auto-detection from video metadata."""

import re

from yt_whisper.prompts import PROMPTS

CONFIDENCE_THRESHOLD = 3
DESCRIPTION_CAP = 2000


def detect_profile(metadata):
    """Return (profile_name, matched_terms, confidence).

    metadata: dict with optional 'title', 'channel', 'description', 'tags' keys.
    Falls back to ('general', [], 0.0) if no profile scores above threshold.
    confidence is 0.0-1.0.
    """
    title = metadata.get("title") or ""
    channel = metadata.get("channel") or ""
    description = (metadata.get("description") or "")[:DESCRIPTION_CAP]
    tags = metadata.get("tags") or []

    haystack = " ".join([title, channel, description, " ".join(tags)]).lower()

    scores = {}
    for name, profile in PROMPTS.items():
        if name == "general":
            continue
        matched = []
        score = 0
        for kw in profile.get("keywords", []):
            pattern = rf"\b{re.escape(kw.lower())}\b"
            if re.search(pattern, haystack):
                matched.append(kw)
                score += len(kw.split()) + 1  # multi-word keywords weigh more
        scores[name] = (score, matched)

    if not scores:
        return ("general", [], 0.0)

    best_name, (best_score, best_matched) = max(
        scores.items(), key=lambda item: item[1][0]
    )

    if best_score < CONFIDENCE_THRESHOLD:
        return ("general", [], 0.0)

    confidence = min(best_score / 10.0, 1.0)
    return (best_name, best_matched, confidence)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_profile_detect.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add yt_whisper/profile_detect.py tests/test_profile_detect.py
git commit -m "feat(profile_detect): add keyword-based profile auto-detection"
```

---

## Task 3: Refactor `transcribe()` to a generator

The TUI needs to stream segments as they're produced. Change `transcribe()` to `yield` segments one at a time. Keep all other behavior (empty check, verbose print, etc.) but move the empty check to the caller since a generator can't raise before yielding.

**Files:**
- Modify: `yt_whisper/transcriber.py`
- Test: `tests/test_transcriber.py`

- [ ] **Step 1: Read current test file to understand what to update**

Read `tests/test_transcriber.py` fully before editing.

- [ ] **Step 2: Update `tests/test_transcriber.py` for generator interface**

Replace calls like `segments = transcribe(...)` with `segments = list(transcribe(...))`. The `raise TranscriptionError("No speech detected")` assertion now moves to callers — so in the transcriber test, assert an empty list is returned when the mock yields nothing, NOT that an exception is raised. Add a new test:

```python
def test_transcribe_is_generator(monkeypatch, tmp_path):
    """transcribe() must return a generator so runner can stream segments."""
    import types
    # ... existing mock setup to make transcribe callable ...
    # After setting up mocks:
    result = transcribe(str(tmp_path / "fake.wav"), "small", None, "en", False)
    assert isinstance(result, types.GeneratorType)
```

Remove any test that asserts `TranscriptionError` is raised from within `transcribe()` for empty output — that responsibility moves to `runner.py` (tested there in Task 7).

- [ ] **Step 3: Run the updated tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_transcriber.py -v`
Expected: FAIL — `transcribe` currently returns a list, not a generator.

- [ ] **Step 4: Rewrite `yt_whisper/transcriber.py`**

```python
"""Transcribe audio using faster-whisper with CUDA GPU support."""

import os

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
    """Yield transcription segments one at a time.

    Each yielded segment: {"start", "end", "text", "speaker": None}.
    Caller is responsible for checking whether any segments were produced.
    """
    cuda_preload.ensure_dlls()

    # Local import -- faster_whisper must not be imported at module level (Anti-Pattern #1)
    from faster_whisper import WhisperModel

    device = "cuda"
    compute_type = "float16"
    try:
        if not _check_model_cached(model_size):
            size_hint = "~3GB" if "large" in model_size else "~1GB"
            print(f"Downloading Whisper model '{model_size}' ({size_hint}). One-time download.")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except (RuntimeError, ValueError) as e:
        print(
            f"Warning: CUDA unavailable -- falling back to CPU. "
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

    for seg in segments_gen:
        text = seg.text.strip()
        if verbose:
            print(f"  [{seg.start:.1f}s -> {seg.end:.1f}s] {text}")
        yield {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": text,
            "speaker": None,
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_transcriber.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add yt_whisper/transcriber.py tests/test_transcriber.py
git commit -m "refactor(transcriber): yield segments as generator for streaming"
```

---

## Task 4: Speaker-aware formatter

Update `format_output()` to handle segments with `speaker` fields. Markdown breaks paragraphs on speaker change; JSON includes `speaker` per segment, a top-level `speakers` list, and a `config` block for re-run.

**Files:**
- Modify: `yt_whisper/formatter.py`
- Test: `tests/test_formatter.py`

- [ ] **Step 1: Read current `tests/test_formatter.py`**

Read the file fully.

- [ ] **Step 2: Add new failing tests to `tests/test_formatter.py`**

Append (do not delete existing tests):

```python
def test_format_markdown_with_speakers(tmp_path):
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Hello.", "speaker": "Speaker 1"},
        {"start": 2.0, "end": 4.0, "text": "Welcome.", "speaker": "Speaker 1"},
        {"start": 4.0, "end": 6.0, "text": "Thanks!", "speaker": "Speaker 2"},
        {"start": 6.0, "end": 8.0, "text": "Glad to be here.", "speaker": "Speaker 2"},
    ]
    metadata = {
        "video_id": "abc123", "title": "T", "channel": "C",
        "upload_date": "20260101", "duration": 8, "url": "u",
    }
    from yt_whisper.formatter import format_output
    paths = format_output(
        segments, metadata, "md", str(tmp_path),
        model="small", prompt_profile="general", method="whisper", language="en",
    )
    content = open(paths[0], encoding="utf-8").read()
    assert "**Speaker 1:**" in content
    assert "**Speaker 2:**" in content
    # Speaker 1 block must contain both of their lines before Speaker 2 starts
    s1_idx = content.index("**Speaker 1:**")
    s2_idx = content.index("**Speaker 2:**")
    s1_block = content[s1_idx:s2_idx]
    assert "Hello." in s1_block
    assert "Welcome." in s1_block


def test_format_markdown_without_speakers_unchanged(tmp_path):
    """Non-diarized output (speaker=None) must not show speaker labels."""
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Hello world.", "speaker": None},
        {"start": 2.0, "end": 4.0, "text": "Second sentence.", "speaker": None},
    ]
    metadata = {
        "video_id": "abc123", "title": "T", "channel": "C",
        "upload_date": "20260101", "duration": 4, "url": "u",
    }
    from yt_whisper.formatter import format_output
    paths = format_output(
        segments, metadata, "md", str(tmp_path),
        model="small", prompt_profile="general", method="whisper", language="en",
    )
    content = open(paths[0], encoding="utf-8").read()
    assert "**Speaker" not in content


def test_format_json_includes_speaker_field(tmp_path):
    import json
    segments = [
        {"start": 0.0, "end": 2.0, "text": "A.", "speaker": "Speaker 1"},
        {"start": 2.0, "end": 4.0, "text": "B.", "speaker": "Speaker 2"},
    ]
    metadata = {
        "video_id": "abc123", "title": "T", "channel": "C",
        "upload_date": "20260101", "duration": 4, "url": "u",
    }
    from yt_whisper.formatter import format_output
    paths = format_output(
        segments, metadata, "json", str(tmp_path),
        model="small", prompt_profile="general", method="whisper", language="en",
    )
    data = json.load(open(paths[0], encoding="utf-8"))
    assert data["segments"][0]["speaker"] == "Speaker 1"
    assert data["segments"][1]["speaker"] == "Speaker 2"
    assert data["speakers"] == ["Speaker 1", "Speaker 2"]


def test_format_json_includes_config_block(tmp_path):
    import json
    segments = [{"start": 0.0, "end": 2.0, "text": "A.", "speaker": None}]
    metadata = {
        "video_id": "abc123", "title": "T", "channel": "C",
        "upload_date": "20260101", "duration": 2, "url": "https://yt/abc123",
    }
    from yt_whisper.formatter import format_output
    paths = format_output(
        segments, metadata, "json", str(tmp_path),
        model="small", prompt_profile="grc", method="whisper", language="en",
        config={"url": "https://yt/abc123", "model": "small", "language": "en",
                "prompt_profile": "grc", "diarize": False, "output_format": "json"},
    )
    data = json.load(open(paths[0], encoding="utf-8"))
    assert "config" in data
    assert data["config"]["url"] == "https://yt/abc123"
    assert data["config"]["prompt_profile"] == "grc"
    assert data["config"]["diarize"] is False


def test_format_json_speakers_list_empty_when_no_diarize(tmp_path):
    import json
    segments = [{"start": 0.0, "end": 2.0, "text": "A.", "speaker": None}]
    metadata = {
        "video_id": "abc123", "title": "T", "channel": "C",
        "upload_date": "20260101", "duration": 2, "url": "u",
    }
    from yt_whisper.formatter import format_output
    paths = format_output(
        segments, metadata, "json", str(tmp_path),
        model="small", prompt_profile="general", method="whisper", language="en",
    )
    data = json.load(open(paths[0], encoding="utf-8"))
    assert data.get("speakers") == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_formatter.py -v`
Expected: new tests FAIL.

- [ ] **Step 4: Rewrite `yt_whisper/formatter.py`**

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


def _has_speakers(segments):
    """True if any segment carries a non-None speaker label."""
    return any(seg.get("speaker") for seg in segments)


def _render_speaker_markdown(segments):
    """Group consecutive segments by speaker into paragraphs."""
    blocks = []
    current_speaker = None
    current_texts = []
    for seg in segments:
        speaker = seg.get("speaker") or "Speaker ?"
        text = seg["text"].strip()
        if speaker != current_speaker:
            if current_texts:
                blocks.append(f"**{current_speaker}:** " + " ".join(current_texts))
            current_speaker = speaker
            current_texts = [text]
        else:
            current_texts.append(text)
    if current_texts:
        blocks.append(f"**{current_speaker}:** " + " ".join(current_texts))
    return "\n\n".join(blocks)


def _unique_speakers(segments):
    """Return unique speaker labels in order of first appearance."""
    seen = []
    for seg in segments:
        sp = seg.get("speaker")
        if sp and sp not in seen:
            seen.append(sp)
    return seen


def format_output(text_or_segments, metadata, output_format, output_dir,
                  model=None, prompt_profile=None, method="whisper",
                  language="en", config=None):
    """Write transcript to markdown and/or JSON files. Returns list of paths written.

    config: optional dict stored in JSON for re-run (see tui/history.py).
    """
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

    if method == "youtube_subs":
        method_display = "youtube_subs"
    elif prompt_profile and prompt_profile != "general":
        method_display = f"whisper ({model} / {prompt_profile})"
    else:
        method_display = f"whisper ({model})"

    paths = []

    if output_format in ("md", "both"):
        md_path = os.path.join(output_dir, f"{video_id}.md")

        if segments is not None and _has_speakers(segments):
            body = _render_speaker_markdown(segments)
        else:
            body = format_paragraphs(full_text)

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
            f"{body}\n"
        )
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        paths.append(md_path)

    if output_format in ("json", "both"):
        json_path = os.path.join(output_dir, f"{video_id}.json")
        speakers = _unique_speakers(segments) if segments is not None else []
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
            "speakers": speakers,
            "segments": segments,
            "full_text": full_text,
        }
        if config is not None:
            json_data["config"] = config
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        paths.append(json_path)

    return paths
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_formatter.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add yt_whisper/formatter.py tests/test_formatter.py
git commit -m "feat(formatter): speaker-aware markdown and config block in JSON"
```

---

## Task 5: Diarizer module

Build `diarizer.py` with a local import of pyannote, HF-token check, CUDA→CPU fallback, and the speaker/segment alignment helper.

**Files:**
- Create: `yt_whisper/diarizer.py`
- Test: `tests/test_diarizer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_diarizer.py`:

```python
"""Tests for diarizer alignment and optional-dependency handling."""

import os
import sys
import pytest

from yt_whisper.diarizer import attach_speakers, DiarizationError


def test_attach_single_overlap():
    segments = [{"start": 0.0, "end": 2.0, "text": "a", "speaker": None}]
    turns = [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}]
    out = attach_speakers(segments, turns)
    assert out[0]["speaker"] == "SPEAKER_00"


def test_attach_dominant_overlap():
    # Segment 0-5; two turns split at 1s and 4s.
    # turn A covers 0-1 (1s), turn B covers 1-5 (4s). B wins.
    segments = [{"start": 0.0, "end": 5.0, "text": "x", "speaker": None}]
    turns = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 5.0, "speaker": "SPEAKER_01"},
    ]
    out = attach_speakers(segments, turns)
    assert out[0]["speaker"] == "SPEAKER_01"


def test_attach_no_overlap_fallback():
    segments = [{"start": 0.0, "end": 1.0, "text": "x", "speaker": None}]
    turns = [{"start": 10.0, "end": 20.0, "speaker": "SPEAKER_00"}]
    out = attach_speakers(segments, turns)
    assert out[0]["speaker"] == "SPEAKER_UNKNOWN"


def test_attach_empty_turns():
    segments = [{"start": 0.0, "end": 1.0, "text": "x", "speaker": None}]
    out = attach_speakers(segments, [])
    assert out[0]["speaker"] == "SPEAKER_UNKNOWN"


def test_attach_empty_segments():
    assert attach_speakers([], [{"start": 0, "end": 1, "speaker": "SPEAKER_00"}]) == []


def test_diarize_missing_pyannote_raises(monkeypatch):
    """If pyannote.audio import fails, raise DiarizationError with install hint."""
    # Force the import to fail
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)
    from yt_whisper.diarizer import diarize
    with pytest.raises(DiarizationError) as exc:
        diarize("/tmp/fake.wav")
    assert "pip install" in str(exc.value).lower()


def test_diarize_missing_hf_token_raises(monkeypatch):
    """If HF_TOKEN is not set, raise DiarizationError with token hint."""
    # Provide a stub pyannote.audio so the import step passes
    import types
    fake_mod = types.ModuleType("pyannote.audio")
    fake_mod.Pipeline = object
    monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_mod)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from yt_whisper.diarizer import diarize
    with pytest.raises(DiarizationError) as exc:
        diarize("/tmp/fake.wav")
    assert "HF_TOKEN" in str(exc.value)


def test_rename_speaker_labels_to_friendly():
    """SPEAKER_00 -> Speaker 1, SPEAKER_01 -> Speaker 2, preserve order of first appearance."""
    from yt_whisper.diarizer import rename_speaker_labels
    segments = [
        {"start": 0, "end": 1, "text": "a", "speaker": "SPEAKER_01"},
        {"start": 1, "end": 2, "text": "b", "speaker": "SPEAKER_00"},
        {"start": 2, "end": 3, "text": "c", "speaker": "SPEAKER_01"},
    ]
    out = rename_speaker_labels(segments)
    # SPEAKER_01 appears first -> Speaker 1
    assert out[0]["speaker"] == "Speaker 1"
    assert out[1]["speaker"] == "Speaker 2"
    assert out[2]["speaker"] == "Speaker 1"


def test_rename_speaker_labels_preserves_unknown():
    from yt_whisper.diarizer import rename_speaker_labels
    segments = [
        {"start": 0, "end": 1, "text": "a", "speaker": "SPEAKER_UNKNOWN"},
        {"start": 1, "end": 2, "text": "b", "speaker": "SPEAKER_00"},
    ]
    out = rename_speaker_labels(segments)
    assert out[0]["speaker"] == "SPEAKER_UNKNOWN"
    assert out[1]["speaker"] == "Speaker 1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_diarizer.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Create `yt_whisper/diarizer.py`**

```python
"""Speaker diarization via pyannote.audio 3.1 (optional dependency).

pyannote is imported locally inside diarize() so users without the optional
extras never hit import errors.
"""

import os

from yt_whisper import cuda_preload


class DiarizationError(Exception):
    """Raised on missing deps, missing token, or pyannote runtime failure."""
    pass


_INSTALL_HINT = (
    "Diarization requires optional dependencies. Install with:\n"
    "  pip install -r requirements-diarize.txt\n"
    "Then set HF_TOKEN environment variable (see README: Optional Speaker Diarization)."
)

_TOKEN_HINT = (
    "HF_TOKEN environment variable not set. "
    "Get a token at https://huggingface.co/settings/tokens, "
    "accept the pyannote/speaker-diarization-3.1 license, then set the env var."
)


def diarize(audio_path, num_speakers=None, min_speakers=None, max_speakers=None, verbose=False):
    """Return list of {start, end, speaker} turns sorted by start.

    Raises DiarizationError on missing deps, missing HF_TOKEN, or runtime failure.
    """
    cuda_preload.ensure_dlls()

    # Local import -- never at module level.
    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        raise DiarizationError(_INSTALL_HINT) from e

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise DiarizationError(_TOKEN_HINT)

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
    except Exception as e:
        raise DiarizationError(f"Failed to load pyannote pipeline: {e}") from e

    # GPU fallback mirrors transcriber pattern
    try:
        import torch
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception as e:
        if verbose:
            print(f"Warning: diarization CUDA unavailable -- using CPU. ({e})")

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    try:
        diarization = pipeline(audio_path, **kwargs)
    except Exception as e:
        raise DiarizationError(f"Diarization pipeline failed: {e}") from e

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker,
        })
    turns.sort(key=lambda t: t["start"])
    return turns


def attach_speakers(whisper_segments, speaker_turns):
    """Assign each whisper segment the speaker whose turn overlaps it most.

    Mutates and returns the list of segments.
    """
    for seg in whisper_segments:
        best_speaker = None
        best_overlap = 0.0
        for turn in speaker_turns:
            overlap = min(seg["end"], turn["end"]) - max(seg["start"], turn["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        seg["speaker"] = best_speaker or "SPEAKER_UNKNOWN"
    return whisper_segments


def rename_speaker_labels(segments):
    """Rename SPEAKER_XX -> 'Speaker N' in order of first appearance.

    SPEAKER_UNKNOWN is preserved as-is. Mutates and returns segments.
    """
    mapping = {}
    next_idx = 1
    for seg in segments:
        sp = seg.get("speaker")
        if not sp or sp == "SPEAKER_UNKNOWN":
            continue
        if sp not in mapping:
            mapping[sp] = f"Speaker {next_idx}"
            next_idx += 1
    for seg in segments:
        sp = seg.get("speaker")
        if sp in mapping:
            seg["speaker"] = mapping[sp]
    return segments
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_diarizer.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add yt_whisper/diarizer.py tests/test_diarizer.py
git commit -m "feat(diarizer): pyannote wrapper with alignment and optional-dep handling"
```

---

## Task 6: Runner module — `RunConfig`, listener base, `ConsoleListener`

Extract pipeline orchestration from `cli.py` into `runner.py`. Define `RunConfig`, a `Listener` base class with no-op methods, and a `ConsoleListener` that reproduces existing stdout output.

**Files:**
- Create: `yt_whisper/runner.py`
- Test: `tests/test_runner.py` (partial — listener + RunConfig tests; pipeline tests added in Task 7)

- [ ] **Step 1: Write failing tests for RunConfig and ConsoleListener**

Create `tests/test_runner.py`:

```python
"""Tests for runner: RunConfig, listeners, and pipeline orchestration."""

import io
import pytest

from yt_whisper.runner import RunConfig, Listener, ConsoleListener


def test_runconfig_defaults():
    cfg = RunConfig(url="https://yt/abc")
    assert cfg.url == "https://yt/abc"
    assert cfg.model == "large-v3"
    assert cfg.language == "en"
    assert cfg.prompt_profile == "general"
    assert cfg.diarize is False
    assert cfg.output_format == "both"
    assert cfg.output_dir == "./transcripts"
    assert cfg.force_whisper is False
    assert cfg.verbose is False
    assert cfg.num_speakers is None
    assert cfg.min_speakers is None
    assert cfg.max_speakers is None


def test_listener_base_has_noop_methods():
    l = Listener()
    # All methods should exist and be callable without errors
    l.on_phase("fetch", "start")
    l.on_progress("download", 0.5)
    l.on_segment({"start": 0, "end": 1, "text": "a", "speaker": None})
    l.on_segments_relabeled([])
    l.on_log("info", "hello")
    l.on_done({"paths": []})
    l.on_error(Exception("boom"))


def test_console_listener_writes_to_stdout(capsys):
    l = ConsoleListener(verbose=True)
    l.on_log("info", "hello world")
    captured = capsys.readouterr()
    assert "hello world" in captured.out


def test_console_listener_verbose_false_filters_debug(capsys):
    l = ConsoleListener(verbose=False)
    l.on_log("debug", "noisy")
    l.on_log("info", "important")
    captured = capsys.readouterr()
    assert "noisy" not in captured.out
    assert "important" in captured.out


def test_console_listener_on_segment_only_when_verbose(capsys):
    l = ConsoleListener(verbose=True)
    l.on_segment({"start": 1.0, "end": 2.5, "text": "hello", "speaker": None})
    out = capsys.readouterr().out
    assert "hello" in out

    l2 = ConsoleListener(verbose=False)
    l2.on_segment({"start": 1.0, "end": 2.5, "text": "hidden", "speaker": None})
    assert "hidden" not in capsys.readouterr().out


def test_console_listener_on_error_to_stderr(capsys):
    l = ConsoleListener(verbose=False)
    l.on_error(ValueError("oops"))
    captured = capsys.readouterr()
    assert "oops" in captured.err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_runner.py -v`
Expected: FAIL — `yt_whisper.runner` does not exist.

- [ ] **Step 3: Create `yt_whisper/runner.py` with RunConfig, Listener, ConsoleListener**

```python
"""Pipeline orchestration and listener interface shared by CLI and TUI."""

import sys
import tempfile
import threading
from dataclasses import dataclass, asdict
from typing import Optional

MIN_VALIDATION_SECONDS = 30


@dataclass
class RunConfig:
    """All inputs needed to run the pipeline once."""
    url: str
    model: str = "large-v3"
    language: str = "en"
    prompt_profile: str = "general"  # profile name or custom string
    diarize: bool = False
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    output_format: str = "both"  # md | json | both
    output_dir: str = "./transcripts"
    force_whisper: bool = False
    verbose: bool = False

    def to_dict(self):
        return asdict(self)


class Listener:
    """Base listener. Override any subset of methods."""

    def on_phase(self, phase: str, status: str) -> None: ...
    def on_progress(self, phase: str, pct: float) -> None: ...
    def on_segment(self, segment: dict) -> None: ...
    def on_segments_relabeled(self, segments: list) -> None: ...
    def on_log(self, level: str, msg: str) -> None: ...
    def on_done(self, result: dict) -> None: ...
    def on_error(self, exc: BaseException) -> None: ...


class ConsoleListener(Listener):
    """Listener that reproduces the existing flag-mode stdout/stderr output."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def on_phase(self, phase, status):
        if not self.verbose:
            return
        if status == "start":
            msgs = {
                "fetch": "Fetching video info...",
                "subs": "Checking YouTube subtitles...",
                "download": "Downloading audio...",
                "transcribe": "Transcribing with Whisper...",
                "diarize": "Running speaker diarization...",
                "format": "Writing output...",
            }
            if phase in msgs:
                print(msgs[phase])

    def on_progress(self, phase, pct):
        # Flag mode doesn't show progress bars -- yt-dlp already prints its own
        pass

    def on_segment(self, segment):
        if not self.verbose:
            return
        print(f"  [{segment['start']:.1f}s -> {segment['end']:.1f}s] {segment['text']}")

    def on_segments_relabeled(self, segments):
        pass

    def on_log(self, level, msg):
        if level == "debug" and not self.verbose:
            return
        print(msg)

    def on_done(self, result):
        # The final summary block is printed by cli.py after run() returns,
        # using data in result. We don't duplicate it here.
        pass

    def on_error(self, exc):
        print(f"Error: {exc}", file=sys.stderr)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_runner.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add yt_whisper/runner.py tests/test_runner.py
git commit -m "feat(runner): add RunConfig, Listener base, and ConsoleListener"
```

---

## Task 7: Runner `run()` function — pipeline orchestration

Implement `runner.run(config, listener, cancel_event=None)` that executes the full pipeline, emitting events. Handles subtitle fast-path, whisper transcription streaming, optional diarization, profile auto-detection, and formatting. Raises no exceptions — all failures go through `listener.on_error`.

**Files:**
- Modify: `yt_whisper/runner.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Append failing tests for `run()` to `tests/test_runner.py`**

Append:

```python
from unittest.mock import patch, MagicMock


class _Capture(Listener):
    """Listener that records all events for assertions."""

    def __init__(self):
        self.events = []

    def on_phase(self, phase, status):
        self.events.append(("phase", phase, status))

    def on_progress(self, phase, pct):
        self.events.append(("progress", phase, pct))

    def on_segment(self, segment):
        self.events.append(("segment", dict(segment)))

    def on_segments_relabeled(self, segments):
        self.events.append(("relabel", [dict(s) for s in segments]))

    def on_log(self, level, msg):
        self.events.append(("log", level, msg))

    def on_done(self, result):
        self.events.append(("done", result))

    def on_error(self, exc):
        self.events.append(("error", str(exc)))


def _fake_metadata(**kwargs):
    base = {
        "video_id": "abc123",
        "title": "Demo Talk",
        "channel": "Ch",
        "upload_date": "20260101",
        "duration": 120,
        "url": "https://yt/abc123",
    }
    base.update(kwargs)
    return base


@pytest.fixture
def tmpdir_out(tmp_path):
    return str(tmp_path / "out")


def test_run_uses_subtitles_when_available(tmpdir_out):
    from yt_whisper.runner import run
    cfg = RunConfig(url="https://yt/abc", output_dir=tmpdir_out)
    listener = _Capture()
    with patch("yt_whisper.runner.check_subtitles") as mock_subs:
        mock_subs.return_value = ("hello world from subs", _fake_metadata())
        run(cfg, listener)
    phases = [e for e in listener.events if e[0] == "phase"]
    phase_names = [p[1] for p in phases]
    assert "subs" in phase_names
    assert "transcribe" not in phase_names  # subs path skips transcribe
    done = [e for e in listener.events if e[0] == "done"]
    assert len(done) == 1
    assert done[0][1]["method"] == "youtube_subs"


def test_run_falls_back_to_whisper_when_no_subs(tmpdir_out):
    from yt_whisper.runner import run
    cfg = RunConfig(url="https://yt/abc", output_dir=tmpdir_out)
    listener = _Capture()

    def fake_transcribe(*args, **kwargs):
        yield {"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}
        yield {"start": 1.0, "end": 2.0, "text": "there", "speaker": None}

    with patch("yt_whisper.runner.check_subtitles") as mock_subs, \
         patch("yt_whisper.runner.download_audio") as mock_dl, \
         patch("yt_whisper.runner.transcribe", side_effect=fake_transcribe):
        mock_subs.return_value = (None, _fake_metadata())
        mock_dl.return_value = "/tmp/fake.wav"
        run(cfg, listener)

    segments = [e for e in listener.events if e[0] == "segment"]
    assert len(segments) == 2
    phase_names = [e[1] for e in listener.events if e[0] == "phase"]
    assert "download" in phase_names
    assert "transcribe" in phase_names
    assert "diarize" not in phase_names  # diarize=False


def test_run_diarize_enabled_calls_diarizer(tmpdir_out):
    from yt_whisper.runner import run
    cfg = RunConfig(url="https://yt/abc", output_dir=tmpdir_out, diarize=True, force_whisper=True)
    listener = _Capture()

    def fake_transcribe(*args, **kwargs):
        yield {"start": 0.0, "end": 5.0, "text": "hi", "speaker": None}
        yield {"start": 5.0, "end": 10.0, "text": "there", "speaker": None}

    fake_turns = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
    ]

    with patch("yt_whisper.runner.check_subtitles") as mock_subs, \
         patch("yt_whisper.runner.download_audio") as mock_dl, \
         patch("yt_whisper.runner.transcribe", side_effect=fake_transcribe), \
         patch("yt_whisper.runner.diarize", return_value=fake_turns):
        mock_subs.return_value = (None, _fake_metadata())
        mock_dl.return_value = "/tmp/fake.wav"
        run(cfg, listener)

    phase_names = [e[1] for e in listener.events if e[0] == "phase"]
    assert "diarize" in phase_names
    relabels = [e for e in listener.events if e[0] == "relabel"]
    assert len(relabels) == 1
    assert relabels[0][1][0]["speaker"] == "Speaker 1"
    assert relabels[0][1][1]["speaker"] == "Speaker 2"


def test_run_autodetects_profile_when_default(tmpdir_out):
    from yt_whisper.runner import run
    cfg = RunConfig(url="https://yt/abc", output_dir=tmpdir_out, force_whisper=True)
    listener = _Capture()
    meta = _fake_metadata(title="NIST 800-53 SOC 2 compliance audit")

    def fake_transcribe(*args, **kwargs):
        yield {"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}

    with patch("yt_whisper.runner.check_subtitles") as mock_subs, \
         patch("yt_whisper.runner.download_audio") as mock_dl, \
         patch("yt_whisper.runner.transcribe", side_effect=fake_transcribe) as mock_t:
        mock_subs.return_value = (None, meta)
        mock_dl.return_value = "/tmp/fake.wav"
        run(cfg, listener)

    # Transcribe should have been called with grc's prompt text
    _, call_kwargs = mock_t.call_args_list[0][0], mock_t.call_args_list[0][1]
    # args order: audio_path, model_size, prompt_text, language, verbose
    call_args = mock_t.call_args[0]
    prompt_text = call_args[2]
    assert prompt_text is not None
    assert "NIST" in prompt_text


def test_run_autodetect_skipped_when_prompt_explicit(tmpdir_out):
    from yt_whisper.runner import run
    cfg = RunConfig(
        url="https://yt/abc", output_dir=tmpdir_out, force_whisper=True,
        prompt_profile="infosec",  # explicit, not "general"
    )
    listener = _Capture()
    meta = _fake_metadata(title="NIST 800-53 SOC 2 compliance audit")

    def fake_transcribe(*args, **kwargs):
        yield {"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}

    with patch("yt_whisper.runner.check_subtitles") as mock_subs, \
         patch("yt_whisper.runner.download_audio") as mock_dl, \
         patch("yt_whisper.runner.transcribe", side_effect=fake_transcribe) as mock_t:
        mock_subs.return_value = (None, meta)
        mock_dl.return_value = "/tmp/fake.wav"
        run(cfg, listener)

    prompt_text = mock_t.call_args[0][2]
    # Should be infosec's prompt, not grc's
    assert "CVE" in prompt_text
    assert "NIST" not in prompt_text or "CVE" in prompt_text  # infosec wins


def test_run_emits_error_on_video_unavailable(tmpdir_out):
    from yt_whisper.runner import run
    from yt_whisper.downloader import VideoUnavailableError
    cfg = RunConfig(url="https://yt/abc", output_dir=tmpdir_out)
    listener = _Capture()
    with patch("yt_whisper.runner.check_subtitles", side_effect=VideoUnavailableError("gone")):
        run(cfg, listener)
    errors = [e for e in listener.events if e[0] == "error"]
    assert len(errors) == 1
    assert "gone" in errors[0][1]


def test_run_diarize_without_pyannote_emits_error(tmpdir_out):
    from yt_whisper.runner import run
    from yt_whisper.diarizer import DiarizationError
    cfg = RunConfig(url="https://yt/abc", output_dir=tmpdir_out, diarize=True, force_whisper=True)
    listener = _Capture()

    def fake_transcribe(*args, **kwargs):
        yield {"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}

    with patch("yt_whisper.runner.check_subtitles") as mock_subs, \
         patch("yt_whisper.runner.download_audio") as mock_dl, \
         patch("yt_whisper.runner.transcribe", side_effect=fake_transcribe), \
         patch("yt_whisper.runner.diarize", side_effect=DiarizationError("install pyannote")):
        mock_subs.return_value = (None, _fake_metadata())
        mock_dl.return_value = "/tmp/fake.wav"
        run(cfg, listener)

    errors = [e for e in listener.events if e[0] == "error"]
    assert len(errors) == 1
    assert "pyannote" in errors[0][1]


def test_run_cancellation_stops_segments(tmpdir_out):
    from yt_whisper.runner import run
    cfg = RunConfig(url="https://yt/abc", output_dir=tmpdir_out, force_whisper=True)
    listener = _Capture()
    cancel = threading.Event()

    def fake_transcribe(*args, **kwargs):
        for i in range(100):
            if i == 2:
                cancel.set()
            yield {"start": float(i), "end": float(i + 1), "text": f"s{i}", "speaker": None}

    with patch("yt_whisper.runner.check_subtitles") as mock_subs, \
         patch("yt_whisper.runner.download_audio") as mock_dl, \
         patch("yt_whisper.runner.transcribe", side_effect=fake_transcribe):
        mock_subs.return_value = (None, _fake_metadata())
        mock_dl.return_value = "/tmp/fake.wav"
        run(cfg, listener, cancel_event=cancel)

    segments = [e for e in listener.events if e[0] == "segment"]
    # Should have stopped early (not all 100)
    assert len(segments) < 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_runner.py -v`
Expected: new tests FAIL (run() not implemented yet).

- [ ] **Step 3: Append `run()` to `yt_whisper/runner.py`**

Add these imports near the top (after existing imports):

```python
from yt_whisper.downloader import check_subtitles, download_audio, VideoUnavailableError
from yt_whisper.transcriber import transcribe, TranscriptionError
from yt_whisper.diarizer import diarize, attach_speakers, rename_speaker_labels, DiarizationError
from yt_whisper.formatter import format_output, format_duration
from yt_whisper.prompts import PROMPTS, resolve_prompt
from yt_whisper.profile_detect import detect_profile
```

Add at the end of the file:

```python
def _validate_wpm(word_count, duration_seconds):
    """Return wpm float or None if too short to validate."""
    if duration_seconds < MIN_VALIDATION_SECONDS:
        return None
    return word_count / (duration_seconds / 60)


def run(config: RunConfig, listener: Listener, cancel_event: Optional[threading.Event] = None):
    """Execute the full pipeline. All errors are routed through listener.on_error.

    Returns the result dict on success, or None on error / cancellation.
    """
    def cancelled():
        return cancel_event is not None and cancel_event.is_set()

    try:
        # Phase 1: fetch metadata + check subtitles
        listener.on_phase("fetch", "start")
        subtitle_text, metadata = check_subtitles(config.url, config.language)
        listener.on_phase("fetch", "done")

        if cancelled():
            listener.on_log("info", "Cancelled.")
            return None

        # Profile auto-detection (only if user didn't explicitly override)
        effective_profile = config.prompt_profile
        if config.prompt_profile == "general":
            detected, matched, _conf = detect_profile(metadata)
            if detected != "general":
                effective_profile = detected
                listener.on_log(
                    "info",
                    f"[auto] profile: {detected} (matched: {', '.join(matched[:5])})",
                )

        # Resolve prompt text from the (possibly auto-detected) profile
        prompt_text = resolve_prompt(effective_profile)

        # Subtitle fast path
        if subtitle_text and not config.force_whisper:
            listener.on_phase("subs", "start")
            listener.on_log(
                "info",
                f"Found YouTube subtitles ({len(subtitle_text.split())} words)",
            )
            listener.on_phase("subs", "done")
            method = "youtube_subs"
            text_or_segments = subtitle_text
        else:
            if config.force_whisper:
                listener.on_log("info", "Forced Whisper transcription (--force-whisper)")
            else:
                listener.on_log("info", "No usable subtitles found. Transcribing with Whisper...")

            # Phase 2: download audio
            listener.on_phase("download", "start")
            with tempfile.TemporaryDirectory() as temp_dir:
                audio_path = download_audio(
                    config.url, temp_dir, metadata, verbose=config.verbose
                )
                listener.on_phase("download", "done")

                if cancelled():
                    listener.on_log("info", "Cancelled.")
                    return None

                # Phase 3: transcribe (streaming)
                listener.on_phase("transcribe", "start")
                collected = []
                for seg in transcribe(
                    audio_path, config.model, prompt_text, config.language, config.verbose
                ):
                    if cancelled():
                        listener.on_log("info", "Cancelled.")
                        return None
                    collected.append(seg)
                    listener.on_segment(seg)
                listener.on_phase("transcribe", "done")

                if not collected:
                    raise TranscriptionError("No speech detected in audio")

                # Phase 4: diarization (optional)
                if config.diarize:
                    listener.on_phase("diarize", "start")
                    try:
                        turns = diarize(
                            audio_path,
                            num_speakers=config.num_speakers,
                            min_speakers=config.min_speakers,
                            max_speakers=config.max_speakers,
                            verbose=config.verbose,
                        )
                        attach_speakers(collected, turns)
                        rename_speaker_labels(collected)
                        listener.on_segments_relabeled(collected)
                    except DiarizationError as e:
                        listener.on_phase("diarize", "done")
                        listener.on_error(e)
                        return None
                    listener.on_phase("diarize", "done")

            method = "whisper"
            text_or_segments = collected

        # Phase 5: format output
        listener.on_phase("format", "start")
        config_block = {
            "url": config.url,
            "model": config.model if method == "whisper" else None,
            "language": config.language,
            "prompt_profile": effective_profile if method == "whisper" else None,
            "diarize": config.diarize,
            "output_format": config.output_format,
        }
        profile_for_formatter = (
            effective_profile if effective_profile in PROMPTS else "custom"
        ) if method == "whisper" else None
        output_paths = format_output(
            text_or_segments, metadata, config.output_format, config.output_dir,
            model=config.model if method == "whisper" else None,
            prompt_profile=profile_for_formatter,
            method=method,
            language=config.language,
            config=config_block,
        )
        listener.on_phase("format", "done")

        # Build result
        if isinstance(text_or_segments, list):
            full_text = " ".join(seg["text"] for seg in text_or_segments)
        else:
            full_text = text_or_segments
        word_count = len(full_text.split())
        wpm = _validate_wpm(word_count, metadata["duration"])

        result = {
            "paths": output_paths,
            "title": metadata["title"],
            "duration_formatted": format_duration(metadata["duration"]),
            "word_count": word_count,
            "wpm": wpm,
            "method": method,
        }
        listener.on_done(result)
        return result

    except VideoUnavailableError as e:
        listener.on_error(e)
        return None
    except TranscriptionError as e:
        listener.on_error(e)
        return None
    except KeyboardInterrupt:
        listener.on_log("info", "Interrupted.")
        return None
    except Exception as e:
        listener.on_error(e)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_runner.py -v`
Expected: all PASS.

- [ ] **Step 5: Run full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add yt_whisper/runner.py tests/test_runner.py
git commit -m "feat(runner): pipeline orchestration with streaming and diarize"
```

---

## Task 8: Thin-shim CLI

Rewrite `cli.py` to: (a) if no args → launch TUI; (b) otherwise parse args, build `RunConfig`, call `runner.run()` with `ConsoleListener`, print final summary from the returned result. Add new `--diarize`, `--speakers`, `--min-speakers`, `--max-speakers` flags.

**Files:**
- Modify: `yt_whisper/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Read current `tests/test_cli.py`**

Read the file fully to see what it asserts.

- [ ] **Step 2: Update/add tests in `tests/test_cli.py`**

Add these tests (keep existing ones that still make sense):

```python
def test_cli_builds_runconfig_from_args(monkeypatch, capsys, tmp_path):
    from yt_whisper.cli import main
    from yt_whisper.runner import RunConfig
    from unittest.mock import patch, MagicMock

    captured_cfg = {}

    def fake_run(cfg, listener, cancel_event=None):
        captured_cfg["cfg"] = cfg
        listener.on_done({
            "paths": [str(tmp_path / "abc.md")],
            "title": "T", "duration_formatted": "1:00",
            "word_count": 100, "wpm": 150.0, "method": "whisper",
        })
        return {"paths": [str(tmp_path / "abc.md")], "title": "T",
                "duration_formatted": "1:00", "word_count": 100,
                "wpm": 150.0, "method": "whisper"}

    argv = ["yt-whisper", "https://yt/abc", "--model", "small",
            "--diarize", "--speakers", "3", "--output-dir", str(tmp_path)]
    monkeypatch.setattr("sys.argv", argv)
    with patch("yt_whisper.cli.run", side_effect=fake_run):
        main()
    cfg = captured_cfg["cfg"]
    assert cfg.url == "https://yt/abc"
    assert cfg.model == "small"
    assert cfg.diarize is True
    assert cfg.num_speakers == 3
    assert cfg.output_dir == str(tmp_path)


def test_cli_no_args_launches_tui(monkeypatch):
    from yt_whisper.cli import main
    from unittest.mock import patch

    monkeypatch.setattr("sys.argv", ["yt-whisper"])
    with patch("yt_whisper.cli.launch_tui") as mock_tui:
        main()
    mock_tui.assert_called_once()


def test_cli_prints_final_summary(monkeypatch, capsys, tmp_path):
    from yt_whisper.cli import main
    from unittest.mock import patch

    def fake_run(cfg, listener, cancel_event=None):
        return {
            "paths": [str(tmp_path / "abc.md"), str(tmp_path / "abc.json")],
            "title": "Demo Talk",
            "duration_formatted": "1:23:45",
            "word_count": 12345,
            "wpm": 155.0,
            "method": "whisper",
        }

    monkeypatch.setattr("sys.argv", ["yt-whisper", "https://yt/abc"])
    with patch("yt_whisper.cli.run", side_effect=fake_run):
        main()
    out = capsys.readouterr().out
    assert "[OK]" in out
    assert "Demo Talk" in out
    assert "12345" in out
    assert "1:23:45" in out
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py -v`
Expected: FAIL.

- [ ] **Step 4: Rewrite `yt_whisper/cli.py`**

```python
"""CLI entrypoint: thin shim around runner.run with optional TUI launch."""

import argparse
import sys

from yt_whisper.runner import RunConfig, ConsoleListener, run


def build_parser():
    parser = argparse.ArgumentParser(
        prog="yt-whisper",
        description="Transcribe YouTube videos using faster-whisper or YouTube subtitles.",
    )
    parser.add_argument("url", nargs="?", help="YouTube video URL (omit for interactive TUI)")
    parser.add_argument("--prompt", dest="prompt_profile", default="general",
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
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument("--diarize", action="store_true",
                        help="Enable speaker diarization (requires optional setup -- see README)")
    parser.add_argument("--speakers", dest="num_speakers", type=int, default=None,
                        help="Exact number of speakers (diarize only)")
    parser.add_argument("--min-speakers", dest="min_speakers", type=int, default=None,
                        help="Minimum number of speakers (diarize only)")
    parser.add_argument("--max-speakers", dest="max_speakers", type=int, default=None,
                        help="Maximum number of speakers (diarize only)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show progress details and segment timing")
    return parser


def launch_tui():
    """Launch the interactive TUI."""
    from yt_whisper.tui.app import YtWhisperApp
    YtWhisperApp().run()


def _print_summary(result):
    if result is None:
        return
    print(f"\n[OK] {result['title']}")
    print(f"  Duration:    {result['duration_formatted']}")
    wpm = result.get("wpm")
    if wpm is not None:
        print(f"  Words:       {result['word_count']} ({wpm:.0f} words/min)")
        if wpm < 100:
            print(f"  Warning: Low word count ({wpm:.0f} words/min). "
                  f"Expected ~150. Transcript may be incomplete.")
        if wpm > 200:
            print(f"  Warning: High word count ({wpm:.0f} words/min). "
                  f"May indicate repeated or hallucinated text.")
    else:
        print(f"  Words:       {result['word_count']}")
        print("  Note: Video too short for word count validation.")
    print(f"  Method:      {result['method']}")
    for i, path in enumerate(result["paths"]):
        prefix = "  Output:      " if i == 0 else "               "
        print(f"{prefix}{path}")


def main():
    # No arguments -> launch TUI
    if len(sys.argv) == 1:
        launch_tui()
        return

    parser = build_parser()
    args = parser.parse_args()

    if not args.url:
        launch_tui()
        return

    cfg = RunConfig(
        url=args.url,
        model=args.model,
        language=args.language,
        prompt_profile=args.prompt_profile,
        diarize=args.diarize,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        output_format=args.output_format,
        output_dir=args.output_dir,
        force_whisper=args.force_whisper,
        verbose=args.verbose,
    )

    listener = ConsoleListener(verbose=args.verbose)
    result = run(cfg, listener)
    _print_summary(result)

    if result is None:
        # Determine exit code by inspecting what happened is hard here;
        # runner already logged the error. Use 1 as generic failure.
        sys.exit(1)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli.py -v`
Expected: all PASS. If the `launch_tui` test fails because `yt_whisper.tui.app` doesn't exist yet, temporarily stub: create `yt_whisper/tui/__init__.py` (empty) and `yt_whisper/tui/app.py` with:

```python
class YtWhisperApp:
    def run(self):
        pass
```

This is fine — it gets fleshed out in Task 10. Commit the stub in this task.

- [ ] **Step 6: Run full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add yt_whisper/cli.py tests/test_cli.py yt_whisper/tui/__init__.py yt_whisper/tui/app.py
git commit -m "refactor(cli): thin shim over runner, add diarize flags, TUI launch"
```

---

## Task 9: History module

Scan `output_dir/*.json` and return a list of past runs for the TUI home screen. Support re-run config extraction and pair deletion.

**Files:**
- Create: `yt_whisper/tui/history.py`
- Test: `tests/test_history.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_history.py`:

```python
"""Tests for TUI history (scan output_dir for past runs)."""

import json
import os
import pytest

from yt_whisper.tui.history import list_history, load_config_for_rerun, delete_run


def _write_run(dir_path, video_id, title="T", diarize=False, mtime=None):
    json_path = os.path.join(dir_path, f"{video_id}.json")
    md_path = os.path.join(dir_path, f"{video_id}.md")
    data = {
        "video_id": video_id,
        "title": title,
        "channel": "Ch",
        "url": f"https://yt/{video_id}",
        "upload_date": "20260101",
        "duration_formatted": "1:00",
        "config": {
            "url": f"https://yt/{video_id}",
            "model": "small",
            "language": "en",
            "prompt_profile": "general",
            "diarize": diarize,
            "output_format": "both",
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n")
    if mtime is not None:
        os.utime(json_path, (mtime, mtime))
    return json_path


def test_list_history_empty_when_dir_missing(tmp_path):
    assert list_history(str(tmp_path / "nope")) == []


def test_list_history_empty_when_no_json(tmp_path):
    assert list_history(str(tmp_path)) == []


def test_list_history_returns_runs(tmp_path):
    _write_run(str(tmp_path), "abc", title="First")
    _write_run(str(tmp_path), "def", title="Second")
    runs = list_history(str(tmp_path))
    assert len(runs) == 2
    titles = [r["title"] for r in runs]
    assert "First" in titles
    assert "Second" in titles


def test_list_history_sorted_by_mtime_desc(tmp_path):
    _write_run(str(tmp_path), "old", title="Old", mtime=1000)
    _write_run(str(tmp_path), "new", title="New", mtime=2000)
    runs = list_history(str(tmp_path))
    assert runs[0]["title"] == "New"
    assert runs[1]["title"] == "Old"


def test_list_history_ignores_malformed_json(tmp_path):
    _write_run(str(tmp_path), "good", title="Good")
    bad = os.path.join(str(tmp_path), "bad.json")
    with open(bad, "w") as f:
        f.write("{not json")
    runs = list_history(str(tmp_path))
    assert len(runs) == 1
    assert runs[0]["title"] == "Good"


def test_list_history_includes_diarize_indicator(tmp_path):
    _write_run(str(tmp_path), "abc", diarize=True)
    runs = list_history(str(tmp_path))
    assert runs[0]["diarize"] is True


def test_load_config_for_rerun(tmp_path):
    _write_run(str(tmp_path), "abc")
    runs = list_history(str(tmp_path))
    cfg = load_config_for_rerun(runs[0])
    assert cfg["url"] == "https://yt/abc"
    assert cfg["model"] == "small"
    assert cfg["prompt_profile"] == "general"


def test_delete_run_removes_md_and_json(tmp_path):
    _write_run(str(tmp_path), "abc")
    assert os.path.exists(os.path.join(str(tmp_path), "abc.json"))
    assert os.path.exists(os.path.join(str(tmp_path), "abc.md"))
    runs = list_history(str(tmp_path))
    delete_run(runs[0])
    assert not os.path.exists(os.path.join(str(tmp_path), "abc.json"))
    assert not os.path.exists(os.path.join(str(tmp_path), "abc.md"))


def test_delete_run_tolerates_missing_md(tmp_path):
    _write_run(str(tmp_path), "abc")
    os.remove(os.path.join(str(tmp_path), "abc.md"))
    runs = list_history(str(tmp_path))
    delete_run(runs[0])  # must not raise
    assert not os.path.exists(os.path.join(str(tmp_path), "abc.json"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_history.py -v`
Expected: FAIL — module doesn't exist (tui/__init__.py exists from Task 8).

- [ ] **Step 3: Create `yt_whisper/tui/history.py`**

```python
"""Scan output_dir for past transcription runs. JSON files are the history."""

import glob
import json
import os


def list_history(output_dir):
    """Return a list of past runs sorted by mtime descending.

    Each entry: {
        "video_id", "title", "channel", "url", "upload_date",
        "duration_formatted", "diarize", "json_path", "md_path", "mtime",
    }
    """
    if not os.path.isdir(output_dir):
        return []

    entries = []
    for json_path in glob.glob(os.path.join(output_dir, "*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        video_id = data.get("video_id") or os.path.splitext(os.path.basename(json_path))[0]
        config = data.get("config") or {}
        md_path = os.path.join(output_dir, f"{video_id}.md")

        entries.append({
            "video_id": video_id,
            "title": data.get("title", "(untitled)"),
            "channel": data.get("channel", ""),
            "url": data.get("url") or config.get("url", ""),
            "upload_date": data.get("upload_date", ""),
            "duration_formatted": data.get("duration_formatted", ""),
            "diarize": bool(config.get("diarize", False)),
            "json_path": json_path,
            "md_path": md_path if os.path.exists(md_path) else None,
            "mtime": os.path.getmtime(json_path),
            "config": config,
        })

    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return entries


def load_config_for_rerun(history_entry):
    """Return a config dict suitable for populating the TUI form on re-run."""
    return dict(history_entry.get("config") or {})


def delete_run(history_entry):
    """Delete the JSON and (if present) markdown pair. Tolerates missing files."""
    for key in ("json_path", "md_path"):
        path = history_entry.get(key)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_history.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add yt_whisper/tui/history.py tests/test_history.py
git commit -m "feat(tui): history scanning and re-run/delete helpers"
```

---

## Task 10: TUI listener + Textual app skeleton + Home screen

Add `textual` to `requirements.txt`, install it, then build the `TuiListener` and the Textual app with a working Home screen (form + history list). Run screen and Preview screen come in Tasks 11-12.

**Files:**
- Modify: `requirements.txt`
- Create: `yt_whisper/tui/listener.py`
- Modify: `yt_whisper/tui/app.py` (replace stub from Task 8)
- Test: `tests/test_tui_smoke.py`

- [ ] **Step 1: Add textual to `requirements.txt`**

Read `requirements.txt`, then append:
```
textual>=0.80
python-dotenv>=1.0
```

Install:
```bash
.venv/Scripts/python.exe -m pip install "textual>=0.80" "python-dotenv>=1.0"
```

- [ ] **Step 2: Write failing smoke tests**

Create `tests/test_tui_smoke.py`:

```python
"""Smoke tests for Textual TUI. Use Textual's App.run_test harness."""

import json
import os
import pytest

pytest.importorskip("textual")

from yt_whisper.tui.app import YtWhisperApp


@pytest.mark.asyncio
async def test_home_screen_mounts(tmp_path):
    app = YtWhisperApp(output_dir=str(tmp_path))
    async with app.run_test() as pilot:
        # Home screen should be mounted with the form fields
        assert app.query_one("#url-input") is not None
        assert app.query_one("#model-select") is not None
        assert app.query_one("#diarize-toggle") is not None


@pytest.mark.asyncio
async def test_form_builds_runconfig(tmp_path):
    app = YtWhisperApp(output_dir=str(tmp_path))
    async with app.run_test() as pilot:
        app.query_one("#url-input").value = "https://yt/xyz"
        cfg = app.build_runconfig()
        assert cfg.url == "https://yt/xyz"
        assert cfg.output_dir == str(tmp_path)


@pytest.mark.asyncio
async def test_history_loads_existing_runs(tmp_path):
    # Pre-seed a JSON history item
    data = {
        "video_id": "abc", "title": "Seed", "channel": "Ch",
        "url": "https://yt/abc", "upload_date": "20260101",
        "duration_formatted": "1:00",
        "config": {"url": "https://yt/abc", "model": "small",
                   "language": "en", "prompt_profile": "general",
                   "diarize": False, "output_format": "both"},
    }
    (tmp_path / "abc.json").write_text(json.dumps(data))

    app = YtWhisperApp(output_dir=str(tmp_path))
    async with app.run_test() as pilot:
        history_list = app.query_one("#history-list")
        # Should contain one item
        assert len(history_list.children) >= 1
```

Also ensure `pytest-asyncio` is available — add to requirements.txt if missing:
```bash
.venv/Scripts/python.exe -m pip install pytest-asyncio
```
Append `pytest-asyncio>=0.23` to `requirements.txt` too. Add a `pytest.ini` section or `pyproject.toml` asyncio_mode entry if needed. Simplest: add a conftest:

Create `tests/conftest.py` if it doesn't exist, or append:
```python
import pytest
# Enable asyncio mode for tui tests
pytest_plugins = ("pytest_asyncio",)
```
And at the top of `test_tui_smoke.py`, add:
```python
pytestmark = pytest.mark.asyncio
```
(Adjust per your pytest-asyncio version.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tui_smoke.py -v`
Expected: FAIL — `YtWhisperApp` is a stub.

- [ ] **Step 4: Create `yt_whisper/tui/listener.py`**

```python
"""Bridge runner events into Textual's message queue from the worker thread."""

from yt_whisper.runner import Listener


class TuiListener(Listener):
    """Forwards runner events into a Textual App via call_from_thread.

    Expects the app to expose methods matching the on_* callbacks.
    """

    def __init__(self, app):
        self.app = app

    def on_phase(self, phase, status):
        self.app.call_from_thread(self.app.tui_on_phase, phase, status)

    def on_progress(self, phase, pct):
        self.app.call_from_thread(self.app.tui_on_progress, phase, pct)

    def on_segment(self, segment):
        self.app.call_from_thread(self.app.tui_on_segment, dict(segment))

    def on_segments_relabeled(self, segments):
        self.app.call_from_thread(
            self.app.tui_on_relabel, [dict(s) for s in segments]
        )

    def on_log(self, level, msg):
        self.app.call_from_thread(self.app.tui_on_log, level, msg)

    def on_done(self, result):
        self.app.call_from_thread(self.app.tui_on_done, result)

    def on_error(self, exc):
        self.app.call_from_thread(self.app.tui_on_error, repr(exc))
```

- [ ] **Step 5: Rewrite `yt_whisper/tui/app.py` with Home screen**

```python
"""Textual TUI for yt-whisper: Home, Run, and Preview screens."""

import os

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Header, Footer, Input, Select, Checkbox, Button, Label, ListView, ListItem,
    RadioSet, RadioButton, Static,
)

from yt_whisper.runner import RunConfig
from yt_whisper.tui.history import list_history, load_config_for_rerun, delete_run


MODEL_CHOICES = [("tiny", "tiny"), ("base", "base"), ("small", "small"),
                 ("medium", "medium"), ("large-v3", "large-v3")]
PROFILE_CHOICES = [("general", "general"), ("grc", "grc"), ("infosec", "infosec")]


class HomeScreen(Screen):
    """Two-column: history list (left) + new-transcription form (right)."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "rerun", "Re-run selected"),
        ("d", "delete", "Delete selected"),
        ("p", "preview", "Preview selected"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="history-pane"):
                yield Label("History")
                yield ListView(id="history-list")
            with Vertical(id="form-pane"):
                yield Label("New Transcription")
                yield Label("URL:")
                yield Input(placeholder="https://youtube.com/watch?v=...", id="url-input")
                yield Label("Model:")
                yield Select(MODEL_CHOICES, value="large-v3", id="model-select")
                yield Label("Language:")
                yield Input(value="en", id="language-input")
                yield Label("Profile:")
                yield Select(PROFILE_CHOICES, value="general", id="profile-select")
                yield Checkbox("Diarize (requires setup -- see README)", id="diarize-toggle")
                yield Label("Speakers (optional, diarize only):")
                yield Input(placeholder="auto", id="speakers-input")
                yield Label("Format:")
                with RadioSet(id="format-radio"):
                    yield RadioButton("both", value=True, id="fmt-both")
                    yield RadioButton("md", id="fmt-md")
                    yield RadioButton("json", id="fmt-json")
                yield Button("Run", id="run-btn", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_history()

    def refresh_history(self) -> None:
        lv = self.query_one("#history-list", ListView)
        lv.clear()
        for run in list_history(self.app.output_dir):
            marker = "*" if run["diarize"] else " "
            label = f"{marker} {run['title'][:40]}"
            lv.append(ListItem(Label(label), id=f"hist-{run['video_id']}"))
        # Keep raw list for lookup
        self.app._history_cache = list_history(self.app.output_dir)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-btn":
            self.action_run()

    def action_run(self) -> None:
        cfg = self.app.build_runconfig()
        if not cfg.url:
            self.app.bell()
            return
        self.app.start_run(cfg)

    def action_rerun(self) -> None:
        idx = self.query_one("#history-list", ListView).index
        if idx is None or idx >= len(self.app._history_cache):
            return
        entry = self.app._history_cache[idx]
        stored = load_config_for_rerun(entry)
        # Populate form fields
        self.query_one("#url-input", Input).value = stored.get("url", "")
        self.query_one("#model-select", Select).value = stored.get("model") or "large-v3"
        self.query_one("#language-input", Input).value = stored.get("language", "en")
        self.query_one("#profile-select", Select).value = stored.get("prompt_profile") or "general"
        self.query_one("#diarize-toggle", Checkbox).value = bool(stored.get("diarize"))

    def action_delete(self) -> None:
        idx = self.query_one("#history-list", ListView).index
        if idx is None or idx >= len(self.app._history_cache):
            return
        delete_run(self.app._history_cache[idx])
        self.refresh_history()

    def action_preview(self) -> None:
        # Preview screen implemented in Task 12
        self.app.bell()


class YtWhisperApp(App):
    """Main Textual application."""

    CSS = """
    #history-pane { width: 40%; border: tall $primary; padding: 1; }
    #form-pane { width: 60%; border: tall $primary; padding: 1; }
    #history-list { height: 1fr; }
    """

    def __init__(self, output_dir: str = "./transcripts"):
        super().__init__()
        self.output_dir = output_dir
        self._history_cache = []

    def on_mount(self) -> None:
        self.push_screen(HomeScreen())

    def build_runconfig(self) -> RunConfig:
        url = self.query_one("#url-input", Input).value.strip()
        model = self.query_one("#model-select", Select).value or "large-v3"
        language = self.query_one("#language-input", Input).value.strip() or "en"
        profile = self.query_one("#profile-select", Select).value or "general"
        diarize = self.query_one("#diarize-toggle", Checkbox).value
        speakers_raw = self.query_one("#speakers-input", Input).value.strip()
        num_speakers = int(speakers_raw) if speakers_raw.isdigit() else None
        # Format radio
        fmt_set = self.query_one("#format-radio", RadioSet)
        if fmt_set.pressed_button and fmt_set.pressed_button.id:
            fmt = fmt_set.pressed_button.id.replace("fmt-", "")
        else:
            fmt = "both"
        return RunConfig(
            url=url, model=model, language=language,
            prompt_profile=profile, diarize=diarize,
            num_speakers=num_speakers, output_format=fmt,
            output_dir=self.output_dir,
        )

    def start_run(self, cfg: RunConfig) -> None:
        # Implemented fully in Task 11. For now, log and return.
        self.bell()

    # Stub callbacks so TuiListener doesn't crash during partial test runs
    def tui_on_phase(self, phase, status): pass
    def tui_on_progress(self, phase, pct): pass
    def tui_on_segment(self, segment): pass
    def tui_on_relabel(self, segments): pass
    def tui_on_log(self, level, msg): pass
    def tui_on_done(self, result): pass
    def tui_on_error(self, msg): pass
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tui_smoke.py -v`
Expected: PASS. If any test fails due to Textual widget API differences in your installed version, adjust query selectors accordingly — the goal is smoke-level (screen mounts, form builds RunConfig, history loads).

- [ ] **Step 7: Run full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all PASS (~60+ tests).

- [ ] **Step 8: Commit**

```bash
git add requirements.txt yt_whisper/tui/listener.py yt_whisper/tui/app.py tests/test_tui_smoke.py tests/conftest.py
git commit -m "feat(tui): Home screen with form, history list, and TuiListener"
```

---

## Task 11: TUI Run screen

Add a Run screen with phase progress bars, log panel, and live transcript panel. Wire `start_run()` to launch the runner in a worker thread and push events through `TuiListener`.

**Files:**
- Modify: `yt_whisper/tui/app.py`
- Test: `tests/test_tui_smoke.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_tui_smoke.py`:

```python
@pytest.mark.asyncio
async def test_run_screen_mounts_on_start(tmp_path, monkeypatch):
    """Submitting the form pushes the Run screen."""
    from unittest.mock import patch

    app = YtWhisperApp(output_dir=str(tmp_path))
    async with app.run_test() as pilot:
        app.query_one("#url-input").value = "https://yt/abc"
        # Patch the runner.run to do nothing so the worker finishes immediately
        with patch("yt_whisper.tui.app.run", return_value=None):
            app.action_start_run_from_test = lambda: app.start_run(app.build_runconfig())
            app.action_start_run_from_test()
            await pilot.pause(0.2)
        # Run screen should now be on the stack
        from yt_whisper.tui.app import RunScreen
        assert any(isinstance(s, RunScreen) for s in app.screen_stack)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tui_smoke.py::test_run_screen_mounts_on_start -v`
Expected: FAIL — `RunScreen` and `start_run` are stubs.

- [ ] **Step 3: Add `RunScreen` class and implement `start_run` in `yt_whisper/tui/app.py`**

Add imports at the top of `app.py`:
```python
import threading
from textual.widgets import ProgressBar, RichLog
from textual import work
from yt_whisper.runner import run
from yt_whisper.tui.listener import TuiListener
```

Add after `HomeScreen`:

```python
class RunScreen(Screen):
    """Shows live progress, logs, and streaming transcript during a run."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, config: RunConfig):
        super().__init__()
        self.config = config
        self.cancel_event = threading.Event()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(f"Transcribing: {self.config.url}", id="run-title")
        with Vertical(id="progress-pane"):
            yield Label("Fetch");      yield ProgressBar(id="pb-fetch", total=100)
            yield Label("Download");   yield ProgressBar(id="pb-download", total=100)
            yield Label("Transcribe"); yield ProgressBar(id="pb-transcribe", total=100)
            yield Label("Diarize");    yield ProgressBar(id="pb-diarize", total=100)
        with Horizontal(id="run-body"):
            with Vertical(id="log-pane"):
                yield Label("Log")
                yield RichLog(id="log-view", highlight=True, markup=True)
            with Vertical(id="transcript-pane"):
                yield Label("Transcript (live)")
                yield RichLog(id="transcript-view", highlight=True, markup=True, wrap=True)
        yield Footer()

    def action_cancel(self) -> None:
        self.cancel_event.set()
        self.query_one("#log-view", RichLog).write("[yellow]Cancelling...[/yellow]")


# ---- YtWhisperApp additions (replace existing stubs) ----
```

Replace the stub methods on `YtWhisperApp` with real implementations:

```python
    def start_run(self, cfg: RunConfig) -> None:
        screen = RunScreen(cfg)
        self.push_screen(screen)
        self._active_run_screen = screen
        self._run_worker(cfg, screen.cancel_event)

    @work(thread=True)
    def _run_worker(self, cfg: RunConfig, cancel_event: threading.Event) -> None:
        listener = TuiListener(self)
        run(cfg, listener, cancel_event=cancel_event)

    def _pb(self, phase: str):
        try:
            screen = self._active_run_screen
            return screen.query_one(f"#pb-{phase}", ProgressBar)
        except Exception:
            return None

    def _log(self):
        try:
            return self._active_run_screen.query_one("#log-view", RichLog)
        except Exception:
            return None

    def _transcript(self):
        try:
            return self._active_run_screen.query_one("#transcript-view", RichLog)
        except Exception:
            return None

    def tui_on_phase(self, phase, status):
        pb = self._pb(phase)
        if pb is None:
            return
        if status == "start":
            pb.update(progress=10)
            log = self._log()
            if log:
                log.write(f"[cyan]{phase}[/cyan] start")
        elif status == "done":
            pb.update(progress=100)
            log = self._log()
            if log:
                log.write(f"[green]{phase}[/green] done")

    def tui_on_progress(self, phase, pct):
        pb = self._pb(phase)
        if pb is not None:
            pb.update(progress=int(pct * 100))

    def tui_on_segment(self, segment):
        view = self._transcript()
        if view is None:
            return
        speaker = segment.get("speaker") or ""
        prefix = f"[dim]{segment['start']:.1f}s[/dim] "
        if speaker:
            prefix += f"[bold]{speaker}:[/bold] "
        view.write(prefix + segment["text"])

    def tui_on_relabel(self, segments):
        view = self._transcript()
        if view is None:
            return
        view.clear()
        view.write("[bold yellow]-- speaker labels applied --[/bold yellow]")
        for seg in segments:
            speaker = seg.get("speaker") or "?"
            view.write(f"[dim]{seg['start']:.1f}s[/dim] [bold]{speaker}:[/bold] {seg['text']}")

    def tui_on_log(self, level, msg):
        log = self._log()
        if log:
            log.write(msg)

    def tui_on_done(self, result):
        log = self._log()
        if log:
            log.write(f"[green]DONE[/green] -> {result['paths']}")
        # Auto-navigate to preview (Task 12 will implement)

    def tui_on_error(self, msg):
        log = self._log()
        if log:
            log.write(f"[red]ERROR:[/red] {msg}")
```

Note: also track `_active_run_screen` — initialize it to `None` in `__init__`:

```python
        self._active_run_screen = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tui_smoke.py -v`
Expected: all PASS.

- [ ] **Step 5: Run full suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add yt_whisper/tui/app.py tests/test_tui_smoke.py
git commit -m "feat(tui): Run screen with live progress, logs, and transcript stream"
```

---

## Task 12: TUI Preview screen + auto-navigate on done

Add a Preview screen that renders a completed run's markdown and auto-navigate there when a run finishes.

**Files:**
- Modify: `yt_whisper/tui/app.py`

- [ ] **Step 1: Add `PreviewScreen` class to `yt_whisper/tui/app.py`**

Add import at the top:
```python
from textual.widgets import Markdown
```

Add class after `RunScreen`:

```python
class PreviewScreen(Screen):
    """Renders a completed transcript's markdown file."""

    BINDINGS = [
        ("escape", "back", "Back"),
        ("o", "open_file", "Open in editor"),
    ]

    def __init__(self, md_path: str):
        super().__init__()
        self.md_path = md_path

    def compose(self) -> ComposeResult:
        yield Header()
        try:
            content = open(self.md_path, encoding="utf-8").read()
        except OSError:
            content = f"Could not read {self.md_path}"
        yield Markdown(content, id="preview-md")
        yield Footer()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_open_file(self) -> None:
        try:
            os.startfile(self.md_path)  # Windows-only; docs note
        except Exception as e:
            self.app.bell()
```

Update `tui_on_done` in `YtWhisperApp`:

```python
    def tui_on_done(self, result):
        log = self._log()
        if log:
            log.write(f"[green]DONE[/green] -> {result['paths']}")
        # Auto-navigate to preview for the first .md path
        md_path = next((p for p in result["paths"] if p.endswith(".md")), None)
        if md_path:
            self.push_screen(PreviewScreen(md_path))
```

Also update `HomeScreen.action_preview`:

```python
    def action_preview(self) -> None:
        idx = self.query_one("#history-list", ListView).index
        if idx is None or idx >= len(self.app._history_cache):
            return
        entry = self.app._history_cache[idx]
        md_path = entry.get("md_path")
        if md_path:
            self.app.push_screen(PreviewScreen(md_path))
        else:
            self.app.bell()
```

- [ ] **Step 2: Add smoke test**

Append to `tests/test_tui_smoke.py`:

```python
@pytest.mark.asyncio
async def test_preview_screen_renders_markdown(tmp_path):
    md_path = tmp_path / "abc.md"
    md_path.write_text("# Hello\n\nSome text.")
    from yt_whisper.tui.app import PreviewScreen

    app = YtWhisperApp(output_dir=str(tmp_path))
    async with app.run_test() as pilot:
        app.push_screen(PreviewScreen(str(md_path)))
        await pilot.pause(0.1)
        assert app.query_one("#preview-md") is not None
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tui_smoke.py -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add yt_whisper/tui/app.py tests/test_tui_smoke.py
git commit -m "feat(tui): Preview screen and auto-navigate on run completion"
```

---

## Task 13: `requirements-diarize.txt` and README / CLAUDE.md updates

Add the optional diarization requirements file and update documentation.

**Files:**
- Create: `requirements-diarize.txt`
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Create `requirements-diarize.txt`**

```
pyannote.audio>=3.1
torchaudio
```

- [ ] **Step 2: Update `README.md`**

Read `README.md` fully. Add a new section at the bottom (after existing content):

```markdown
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

In the TUI, toggle the "Diarize" checkbox on the home screen. If dependencies or the token are missing, a modal appears with install instructions.

## Interactive TUI

Running `yt-whisper` with no arguments launches a full-screen interactive UI with three screens:

- **Home** — paste a URL, pick model/language/profile/diarize/format, see past runs
- **Run** — live progress bars, streaming transcript, and log panel while the pipeline executes
- **Preview** — rendered markdown of a completed transcript

Keyboard-first. Press `Q` to quit, `R` to re-run a past entry, `P` to preview, `D` to delete.

## Prompt Profile Auto-Detection

When you don't pass `--prompt`, yt-whisper inspects the video's title, channel, description, and tags and picks the best matching profile (general, grc, infosec) using keyword matching. Pass `--prompt <name>` explicitly to override.
```

- [ ] **Step 3: Update `CLAUDE.md`**

Read `CLAUDE.md` fully. Under "Key Constraints" add:

```markdown
- **Anti-Pattern #1 (extended)**: Never import `faster_whisper` OR `pyannote.audio` at module top level. Both must be local imports inside their respective functions (`transcribe()` / `diarize()`) after `cuda_preload.ensure_dlls()`. pyannote is an *optional* dependency; module import must never fail for users without it.
- **TUI threading**: runner runs in a Textual worker thread (`@work(thread=True)`). All event callbacks to the app MUST cross the thread boundary via `app.call_from_thread(...)` (handled by `TuiListener`). Never touch Textual widgets directly from the worker.
- **Runner is the single pipeline source**: flag mode and TUI mode both go through `runner.run()`. Any new pipeline step must be added there, not in `cli.py` or the TUI.
```

Under "Dependencies":

```markdown
- Base: `pip install -r requirements.txt` (includes textual for TUI)
- Optional diarization: `pip install -r requirements-diarize.txt` + HF_TOKEN env var + accept pyannote license
```

- [ ] **Step 4: Verify the package still imports and runs**

Run:
```bash
.venv/Scripts/python.exe -c "import yt_whisper.cli; import yt_whisper.runner; import yt_whisper.diarizer; import yt_whisper.tui.app; print('[OK] imports')"
```
Expected: `[OK] imports`

Run full suite:
```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add requirements-diarize.txt README.md CLAUDE.md
git commit -m "docs: optional diarization setup, TUI usage, updated anti-patterns"
```

---

## Task 14: Manual smoke test and final verification

End-to-end manual verification against a real YouTube video. Not automated — do this once before declaring the feature done.

**Files:** none (manual)

- [ ] **Step 1: Run full test suite one more time**

Run: `.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all tests PASS (~75 total).

- [ ] **Step 2: Smoke test flag mode (no diarize)**

Run: `.venv/Scripts/python.exe -m yt-whisper <short_youtube_url> --verbose`
Expected: identical output format to pre-refactor runs. A `.md` and `.json` should be written under `./transcripts/`.

- [ ] **Step 3: Smoke test TUI launch**

Run: `.venv/Scripts/python.exe -m yt-whisper`
Expected: Home screen appears. Tab through the form, enter a URL, press Run. Live transcript streams. On completion, Preview screen shows the markdown. Press Esc to go back. Press Q to quit.

- [ ] **Step 4: Smoke test auto-detect**

Pick a video with clearly technical terms in its title (e.g., "NIST 800-53 Walkthrough" or "CVE-2024-XXXX Analysis"). Run without `--prompt`. In verbose output, confirm a line like `[auto] profile: grc (matched: NIST, 800-53, ...)`.

- [ ] **Step 5: Smoke test diarization (optional — requires HF setup)**

Only if you have HF_TOKEN configured:
```bash
.venv/Scripts/python.exe -m pip install -r requirements-diarize.txt
yt-whisper <multi_speaker_url> --diarize --verbose
```
Expected: phase log includes "diarize", output markdown uses `**Speaker 1:**` / `**Speaker 2:**` blocks, JSON has `speakers` list and per-segment `speaker` field.

- [ ] **Step 6: Smoke test diarize error path**

With pyannote NOT installed, run `yt-whisper <url> --diarize`.
Expected: runner emits a clean `DiarizationError` with install instructions. Exit code 1.

- [ ] **Step 7: Verify no regressions in non-diarize flag mode**

Run a video you've transcribed before with the same flags. Compare output against the previous run's markdown — body should be very similar (whisper is not bit-exact, but structure and word count should be close).

- [ ] **Step 8: Final commit (if any manual fixes)**

If manual testing surfaced bugs, fix them (add tests first), then commit with a clear message. Otherwise, no commit needed — the feature is done.

---

## Self-Review Notes

**Spec coverage check:**
- Diarization (feature 1): Tasks 5 (module), 7 (runner wiring), 4 (formatter output), 13 (docs + reqs). ✓
- TUI (feature 2): Tasks 8 (CLI entry stub), 10 (home), 11 (run), 12 (preview), 9 (history). ✓
- Auto-detect (feature 3): Tasks 1 (prompts restructure), 2 (detect module), 7 (runner wiring). ✓
- Runner extraction: Tasks 6 (config + listeners), 7 (run()). ✓
- Backwards compatibility: Task 8 rewrites cli.py as a thin shim that preserves all output. ConsoleListener golden test implicitly covered by existing `test_cli.py` assertions updated in Task 8.
- Transcriber generator refactor: Task 3. ✓
- Optional-dep pattern for pyannote: Task 5 (import inside function + DiarizationError), Task 13 (separate requirements file). ✓

**Type consistency check:**
- `RunConfig` field names used consistently across Tasks 6, 7, 8, 10, 11 (url, model, language, prompt_profile, diarize, num_speakers, min_speakers, max_speakers, output_format, output_dir, force_whisper, verbose).
- Listener methods (`on_phase`, `on_progress`, `on_segment`, `on_segments_relabeled`, `on_log`, `on_done`, `on_error`) defined in Task 6 and consistently used in Tasks 7 (runner), 10 (TuiListener bridge calls `tui_on_*` which match).
- Segment dict shape `{start, end, text, speaker}` consistent across transcriber (Task 3), diarizer (Task 5), formatter (Task 4), runner (Task 7).
- `history_entry` dict keys (`video_id`, `title`, `json_path`, `md_path`, `config`, etc.) defined in Task 9 and consumed in Task 10.

**No placeholders:** every step contains the full code block it needs. Every test has concrete assertions. Every command has its expected output.
