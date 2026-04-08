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
