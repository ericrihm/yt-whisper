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
    # Segment 0-5; turn A covers 0-1 (1s), turn B covers 1-5 (4s). B wins.
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
    # Force the import to fail by setting the module to None
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)
    from yt_whisper.diarizer import diarize
    with pytest.raises(DiarizationError) as exc:
        diarize("/tmp/fake.wav")
    assert "pip install" in str(exc.value).lower()


def test_diarize_missing_hf_token_raises(monkeypatch):
    """If HF_TOKEN is not set, raise DiarizationError with token hint."""
    import types
    # Provide stub pyannote.audio so the import step passes
    fake_pyannote = types.ModuleType("pyannote")
    fake_audio = types.ModuleType("pyannote.audio")
    fake_audio.Pipeline = object
    monkeypatch.setitem(sys.modules, "pyannote", fake_pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from yt_whisper.diarizer import diarize
    with pytest.raises(DiarizationError) as exc:
        diarize("/tmp/fake.wav")
    assert "HF_TOKEN" in str(exc.value)


def test_rename_speaker_labels_to_friendly():
    """SPEAKER_00 -> Speaker N, in order of first appearance."""
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
