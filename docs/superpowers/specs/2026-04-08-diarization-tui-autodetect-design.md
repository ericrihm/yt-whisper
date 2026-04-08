# yt-whisper: Diarization, TUI, and Profile Auto-Detection

**Date:** 2026-04-08
**Status:** Design approved, ready for implementation planning

## Summary

Add three capabilities to yt-whisper:

1. **Speaker diarization** (opt-in, optional dependency) — identify distinct speakers in transcripts using pyannote.audio 3.1, attach speaker labels to each segment in both markdown and JSON output. Users who don't need diarization never install pyannote or set up an HF token.

2. **Interactive TUI** (Textual-based, full dashboard) — launching `yt-whisper` with no arguments opens a multi-screen TUI with a home screen (history + form), a run screen (live progress and streaming transcript), and a preview screen (rendered markdown of past runs). Passing any flag bypasses the TUI for scripting.

3. **Prompt profile auto-detection** — infer the best prompt profile (general/grc/infosec/…) from video metadata using keyword matching. Used as the default in both TUI and flag mode unless the user explicitly overrides.

All three features preserve the existing flag-mode behavior exactly for backwards compatibility with scripts.

## Motivation

- Diarization is the single most-requested transcription feature for multi-speaker content (interviews, panels, podcasts). Whisper alone produces a wall of unattributed text.
- The current flag-driven CLI is hostile to new users who don't know which model size, prompt profile, or format to pick. A TUI with sensible defaults and discoverable options lowers the barrier.
- Prompt profiles meaningfully improve transcription accuracy for technical domains, but only if users know which one to pick. Auto-detection removes that burden.

## Constraints

- **Backwards compatibility:** all existing flags and their behavior must work identically. Scripts must not break.
- **Optional diarization:** pyannote must NOT be a hard dependency. Users doing plain transcription must not need an HF account, token, or any extra install.
- **Windows-first:** no unicode in console output (cp1252), no MS Store Python, must work with system Python from python.org.
- **Local-only compute:** all models run locally. No API calls. CUDA-preferred with CPU fallback.
- **Lean deps:** prefer additions that are pure Python and widely used. Textual is the only non-trivial new base dependency.
- **Existing architecture discipline:** `faster_whisper` must remain a local import inside its function. Same rule applies to `pyannote.audio`.

## Architecture

The pipeline grows from 3 stages to 4. A TUI layer and a shared runner wrap the existing flag-mode CLI.

```
Entry (__main__.py)
    |
    v
cli.py  ──(no args)──> tui/app.py  ──> runner.py
    |                       |              ^
    |                       v              |
    |                  tui/history.py      |
    +──(flags)───────────────────────────-─+
                                           |
                                           v
              downloader -> transcriber -> [diarizer] -> formatter
                                 (yields)    (aligns)
```

### New modules

- **`yt_whisper/runner.py`** — pipeline orchestration extracted from current `cli.main()`. Takes a `RunConfig` dataclass plus an optional listener object with event callbacks. Shared by flag mode and TUI mode to keep their behavior in lockstep.
- **`yt_whisper/diarizer.py`** — wraps pyannote, aligns speaker turns with whisper segments, handles HF token validation and model caching. pyannote is imported locally inside `diarize()`; module import never fails for users without the extra.
- **`yt_whisper/profile_detect.py`** — keyword matching over video metadata; returns `(profile_name, matched_terms, confidence)`.
- **`yt_whisper/tui/__init__.py`** — package marker.
- **`yt_whisper/tui/app.py`** — Textual app with three screens (home, run, preview).
- **`yt_whisper/tui/history.py`** — scans `output_dir/*.json` and returns a list of past runs. Uses existing JSON files as the source of truth; no database.

### Changed modules

- **`yt_whisper/cli.py`** — becomes thin: parse args → if no args, launch TUI; else build `RunConfig` → call `runner.run()` with a `ConsoleListener` that reproduces current stdout output exactly.
- **`yt_whisper/transcriber.py`** — `transcribe()` becomes a generator yielding segments one at a time. Internal-only breaking change; all call sites updated.
- **`yt_whisper/formatter.py`** — gains speaker-aware rendering (`**Speaker N:**` prefix in markdown, `speaker` field in JSON segments, top-level `speakers` list). Also stores a `config` block in the JSON so re-run can reconstruct form state.
- **`yt_whisper/prompts.py`** — each profile gains a `keywords` list used by `profile_detect.py`.

### Data contracts

**`RunConfig` (runner input):**

```python
@dataclass
class RunConfig:
    url: str
    model: str = "large-v3"
    language: str = "en"
    prompt_profile: str = "general"  # profile name or custom string
    diarize: bool = False
    num_speakers: int | None = None      # optional hint for pyannote
    min_speakers: int | None = None
    max_speakers: int | None = None
    output_format: str = "both"          # md | json | both
    output_dir: str = "./transcripts"
    force_whisper: bool = False
    verbose: bool = False
```

**Runner events (listener methods):**

| Event | Payload | When |
|---|---|---|
| `on_phase(phase, status)` | `phase ∈ {fetch, subs, download, transcribe, diarize, format}`, `status ∈ {start, done}` | Start/end of each stage |
| `on_progress(phase, pct)` | phase name, 0.0–1.0 | Download progress (yt-dlp hook); transcribe progress (segments elapsed / audio duration) |
| `on_segment(segment)` | `{start, end, text, speaker}` (speaker may be None) | Each segment as transcriber yields it |
| `on_segments_relabeled(segments)` | full list | After diarization attaches speakers |
| `on_log(level, msg)` | stdout-style lines | Replaces `print()` calls |
| `on_done(result)` | `{paths, word_count, wpm, method, title, duration_formatted}` | Final summary |
| `on_error(exc)` | exception | Any failure |

**Two listener implementations:**

- `ConsoleListener` — maps events to `print()` calls matching current CLI output exactly. A golden-output test guards against regression.
- `TuiListener` — forwards events into Textual's message queue via `app.call_from_thread(...)`.

## Feature 1: Speaker Diarization

### Library choice

**pyannote.audio 3.1** with the `pyannote/speaker-diarization-3.1` pipeline. Rationale: de facto standard for local diarization, actively maintained, works on CUDA/Windows, integrates cleanly with faster-whisper segment timestamps.

### Optional dependency pattern

- pyannote moved to `requirements-diarize.txt` (separate file). Base `requirements.txt` stays lean.
- README adds a clearly labeled "Optional: Speaker Diarization" section documenting the 3-step setup (install extras → HF account + accept license → set `HF_TOKEN`).
- Base "Quick Start" in README is unchanged. Users who never diarize never hit HF.

### Module interface

```python
# yt_whisper/diarizer.py
class DiarizationError(Exception): pass

def diarize(audio_path: str,
            num_speakers: int | None = None,
            min_speakers: int | None = None,
            max_speakers: int | None = None,
            verbose: bool = False) -> list[dict]:
    """Returns list of {start, end, speaker} turns sorted by start time.
    speaker is a stable string like 'SPEAKER_00', 'SPEAKER_01', ..."""
```

### Import guarding and HF token check

Import pyannote locally inside `diarize()`. On `ImportError`, raise `DiarizationError` with the install instructions. On missing `HF_TOKEN` env var, raise `DiarizationError` with the token setup instructions. Both messages include copy-pasteable commands.

### Speaker/segment alignment

Pyannote returns speaker *turns* (time ranges per speaker). Whisper returns transcription *segments*. These don't map one-to-one. Assign each whisper segment the speaker whose turn has maximum temporal overlap with it:

```python
def attach_speakers(whisper_segments, speaker_turns):
    for seg in whisper_segments:
        best_speaker, best_overlap = None, 0.0
        for turn in speaker_turns:
            overlap = min(seg["end"], turn["end"]) - max(seg["start"], turn["start"])
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, turn["speaker"]
        seg["speaker"] = best_speaker or "SPEAKER_UNKNOWN"
    return whisper_segments
```

Edge cases: segments spanning multiple speakers get the dominant one; segments with no overlap (VAD mismatch) fall back to `SPEAKER_UNKNOWN`.

### GPU handling

Mirror the existing whisper CUDA→CPU fallback pattern. Try CUDA, catch, fall back to CPU with a warning log event. Uses the same `cuda_preload.ensure_dlls()` path already needed for faster-whisper.

### Output labels

pyannote emits `SPEAKER_00`, `SPEAKER_01`, … . The formatter renames them to user-friendly `Speaker 1`, `Speaker 2`, … in both outputs. Rename UI (editable names) is deliberately out of scope for v1; it's a natural follow-up feature attached to the Preview screen.

### Markdown output shape (diarized)

```markdown
**Speaker 1:** Welcome everyone to today's episode. We're going to be talking about...

**Speaker 2:** Thanks for having me. I'm excited to dig into this.

**Speaker 1:** So let's start with the basics...
```

Paragraph breaks occur on speaker change instead of every 5 sentences. Non-diarized markdown output is unchanged.

### JSON output shape (diarized)

Each segment gains a `speaker` field. A top-level `speakers` list holds the unique speaker set for quick access. A top-level `config` block holds the fields needed for re-run (see Feature 2 / history).

## Feature 2: Interactive TUI (Textual)

### Framework

**Textual** — actively maintained, works in Windows Terminal, pure Python, ~5MB.

### Threading model

Textual runs in the main thread. Runner runs in a worker thread (`@work(thread=True)`). Events cross into the UI thread via `app.call_from_thread(...)`. This is the standard Textual worker pattern.

### Home screen

Two-column layout. Left: history list scanned from `output_dir/*.json`, sorted by mtime, showing title / channel / date / diarize-indicator. Right: new-transcription form with URL, model, language, profile, diarize toggle, optional speaker count, and output format fields.

Keybindings: `[Enter]` Run, `[R]` Re-run selected history item (loads config into form), `[P]` Preview selected, `[D]` Delete selected (with confirmation — removes md+json pair), `[Q]` Quit.

On URL blur, the TUI triggers metadata fetch + `detect_profile()` in a worker and updates the Profile dropdown label to `auto (grc — matched: NIST, SOC 2, compliance)`. User can always override.

### Run screen

Phase progress bars (fetch, download, transcribe, diarize), a log panel showing runner log events with bounded scrollback (~500 lines), and a live transcript panel streaming segments as they arrive. Speaker labels "snap in" when `on_segments_relabeled` fires at the end of diarization — this is acknowledged as acceptable because transcripts are consumed after the run, not live.

Keybinding: `[Esc]` cancels. Cancellation sets a `threading.Event`; runner checks it between segments and at phase boundaries. Pyannote's diarize call is a single blocking operation and cannot be interrupted mid-stream; this is documented, not worked around.

On successful completion, the TUI auto-navigates to the Preview screen for the just-finished run.

### Preview screen

Renders the saved `.md` file using Textual's `Markdown` widget. Header shows metadata and speaker count. Keybindings: `[O]` open file with OS default handler (`os.startfile` on Windows), `[C]` copy path to clipboard, `[Esc]` back to home.

### Diarization gating in the TUI

- The Diarize checkbox label reads: `[ ] Diarize (requires setup — see README)`
- Toggling it on pre-validates: if pyannote is not installed OR `HF_TOKEN` is missing, a modal appears with install/setup instructions and the checkbox auto-unticks. The run then proceeds as a normal transcription rather than erroring — no dead-end.

### History as files

`tui/history.py` scans `output_dir/*.json` and parses each for display + re-run data. Deleting a history item deletes the `.md` and `.json` pair. No separate state file, no SQLite. Moving `output_dir` moves the history with it.

## Feature 3: Prompt Profile Auto-Detection

### Approach

Rule-based keyword matching. Explainable, deterministic, no new dependencies.

### Keyword storage

Each profile in `prompts.py` gains a `keywords` list. Example:

```python
PROMPTS = {
    "general": {"text": "...", "keywords": []},
    "grc": {
        "text": "...",
        "keywords": [
            "NIST", "SOC 2", "ISO 27001", "HIPAA", "PCI DSS", "FedRAMP",
            "GDPR", "CCPA", "compliance", "governance", "audit",
            "risk assessment", "control", "framework", "policy",
            "regulatory", "CISO", "GRC",
        ],
    },
    "infosec": {
        "text": "...",
        "keywords": [
            "CVE", "exploit", "vulnerability", "malware", "ransomware",
            "red team", "blue team", "pentest", "penetration test",
            "reverse engineering", "binary exploitation", "CTF",
            "zero-day", "0day", "backdoor", "payload", "C2",
            "threat hunting", "incident response",
        ],
    },
}
```

### Detection function

```python
# yt_whisper/profile_detect.py
CONFIDENCE_THRESHOLD = 3

def detect_profile(metadata: dict) -> tuple[str, list[str], float]:
    """Returns (profile_name, matched_terms, confidence 0.0-1.0).
    Falls back to 'general' if no profile scores above threshold."""
    haystack = " ".join([
        metadata.get("title", ""),
        metadata.get("channel", ""),
        metadata.get("description", "")[:2000],  # cap runaway
        " ".join(metadata.get("tags", [])),
    ]).lower()

    scores = {}
    for name, profile in PROMPTS.items():
        if name == "general":
            continue
        matched = []
        score = 0
        for kw in profile.get("keywords", []):
            if re.search(rf"\b{re.escape(kw.lower())}\b", haystack):
                matched.append(kw)
                score += len(kw.split()) + 1  # multi-word keywords weigh more
        scores[name] = (score, matched)

    best = max(scores.items(), key=lambda x: x[1][0], default=("general", (0, [])))
    name, (score, matched) = best
    if score < CONFIDENCE_THRESHOLD:
        return ("general", [], 0.0)
    return (name, matched, min(score / 10.0, 1.0))
```

### Wire-up

- **TUI:** on URL blur, worker calls `check_subtitles()` (already fetches metadata), then `detect_profile()`, then updates the Profile dropdown label.
- **Flag mode:** if `--prompt` was not explicitly passed, runner calls `detect_profile()` after metadata fetch and uses the result. Verbose log: `[auto] profile: infosec (matched: CVE, exploit, red team)`.
- Passing `--prompt X` explicitly disables detection.
- Custom prompt strings (not profile names) also disable detection.

### Edge cases

- Metadata fetch failed or fields missing → returns `general`, empty matched list
- Tied scores → deterministic tiebreak (Python dict insertion order + max stability)
- Cooking video mentioning "audit" once → score below threshold → `general`
- Word-boundary match prevents "cover" matching "cve"

## Error Handling

All new error types inherit from `Exception` and are caught at the CLI boundary, same pattern as existing `VideoUnavailableError` and `TranscriptionError`.

- `DiarizationError` — new. Raised on missing pyannote, missing HF token, or pyannote runtime failure. CLI exit code 3.
- Runner catches all pipeline exceptions and emits `on_error(exc)`; both listeners surface them appropriately (ConsoleListener → stderr + exit; TuiListener → modal dialog + return to home screen).
- TUI never crashes the terminal on pipeline failure — all errors become modals.

## Testing Strategy

Existing suite: 39 tests across 5 files, all mocked. New tests follow the same pattern.

### New test files

- **`tests/test_profile_detect.py`** (~12 tests, pure unit): positive matches per profile, weighted scoring, threshold behavior, word boundaries, case insensitivity, empty/missing metadata, description cap, tied scores, matched-terms return value.
- **`tests/test_diarizer.py`** (~8 tests, pyannote mocked): alignment single/dominant/none, empty turns, missing-pyannote error, missing-HF-token error, CUDA→CPU fallback, speaker label rename.
- **`tests/test_runner.py`** (~10 tests, downloader/transcriber/diarizer mocked): phase event ordering, segment streaming, diarize-disabled skip, cancellation, ConsoleListener golden output, error event propagation, auto-detect runs when default, auto-detect skipped when explicit, diarize-without-pyannote flag-mode error, re-run from stored JSON config.
- **`tests/test_history.py`** (~6 tests, filesystem fixtures): scan dir, sort by mtime, malformed JSON tolerance, missing dir, config extraction for re-run, delete md+json pair.
- **`tests/test_tui_smoke.py`** (~4 tests, Textual `App.run_test()`): home screen mounts, form builds RunConfig, diarize-toggle-without-pyannote modal, re-run loads selected config.

### Updated test files

- `tests/test_transcriber.py` — update for generator interface.
- `tests/test_formatter.py` — add speaker-aware rendering tests + backwards-compat golden test for non-diarized output + config-block presence in JSON.
- `tests/test_prompts.py` — keyword list validation (no duplicates, no empty strings).

### Not tested (and why)

- Real pyannote pipeline — no GPU in CI, gated model.
- Real yt-dlp network calls — already mocked in existing suite.
- Visual correctness of TUI widgets — snapshot fragility outweighs value.
- HF token validity — it's just an env var presence check.

### Target

~75 tests total (39 existing + ~36 new). All mocked. Complete run under 10 seconds. `pytest tests/ -v` remains the single command.

## Dependencies

**Base `requirements.txt` additions:**
- `textual` (TUI framework)
- `python-dotenv` (load `HF_TOKEN` from `.env` if present — optional convenience)

**New `requirements-diarize.txt`:**
- `pyannote.audio>=3.1`
- `torchaudio` (pyannote transitive)

**Install paths:**
- Normal users: `pip install -r requirements.txt` — no HF, no extra setup
- Diarization users: `pip install -r requirements.txt -r requirements-diarize.txt` + HF token setup

## Rollout and Scope Boundaries

### In scope (v1)

Everything above.

### Explicitly out of scope (v1 — natural follow-ups)

- Editable speaker names in the Preview screen (rename `Speaker 1` → `Alice`)
- Embedding-based profile detection as a fallback when keywords miss
- New prompt profiles beyond the existing three
- Exporting diarized transcripts to SRT/VTT with speaker tags
- Search across history
- Multi-video batch runs from the TUI

### Fallback if dashboard scope creeps

If the Preview screen proves costly during implementation, it can ship as a follow-up — home + run screens are the minimum viable TUI and the preview screen is read-only with no pipeline coupling.

## Files Changed Summary

**New:**
- `yt_whisper/runner.py`
- `yt_whisper/diarizer.py`
- `yt_whisper/profile_detect.py`
- `yt_whisper/tui/__init__.py`
- `yt_whisper/tui/app.py`
- `yt_whisper/tui/history.py`
- `requirements-diarize.txt`
- `tests/test_profile_detect.py`
- `tests/test_diarizer.py`
- `tests/test_runner.py`
- `tests/test_history.py`
- `tests/test_tui_smoke.py`

**Changed:**
- `yt_whisper/cli.py` — thin shim
- `yt_whisper/transcriber.py` — generator
- `yt_whisper/formatter.py` — speaker-aware rendering + config block
- `yt_whisper/prompts.py` — keyword lists
- `requirements.txt` — add textual, python-dotenv
- `README.md` — optional diarization section
- `CLAUDE.md` — new anti-patterns (pyannote local import, TUI threading)
- `tests/test_transcriber.py` — generator interface
- `tests/test_formatter.py` — speaker rendering
- `tests/test_prompts.py` — keyword validation
