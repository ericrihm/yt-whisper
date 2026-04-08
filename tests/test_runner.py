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


import threading
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


# --- Fix 1: transcribe progress bar ---

def test_run_emits_transcribe_progress(tmpdir_out):
    """on_progress('transcribe', pct) should be called during the transcribe loop."""
    from yt_whisper.runner import run
    cfg = RunConfig(url="https://yt/abc", output_dir=tmpdir_out, force_whisper=True)
    listener = _Capture()

    # duration=120s; segments end at 60s and 90s -> pct 0.5 and 0.75
    def fake_transcribe(*args, **kwargs):
        yield {"start": 0.0, "end": 60.0, "text": "first", "speaker": None}
        yield {"start": 60.0, "end": 90.0, "text": "second", "speaker": None}

    with patch("yt_whisper.runner.check_subtitles") as mock_subs, \
         patch("yt_whisper.runner.download_audio") as mock_dl, \
         patch("yt_whisper.runner.transcribe", side_effect=fake_transcribe):
        mock_subs.return_value = (None, _fake_metadata(duration=120))
        mock_dl.return_value = "/tmp/fake.wav"
        run(cfg, listener)

    progress_events = [
        e for e in listener.events
        if e[0] == "progress" and e[1] == "transcribe"
    ]
    assert len(progress_events) >= 1, "Expected at least one transcribe progress event"
    # All progress values should be > 0.1 (well past the initial 10% shown by TUI)
    for evt in progress_events:
        assert evt[2] > 0.1, f"Expected pct > 0.1, got {evt[2]}"


def test_run_transcribe_progress_zero_duration(tmpdir_out):
    """No on_progress('transcribe') events when duration is 0 (avoid division by zero)."""
    from yt_whisper.runner import run
    cfg = RunConfig(url="https://yt/abc", output_dir=tmpdir_out, force_whisper=True)
    listener = _Capture()

    def fake_transcribe(*args, **kwargs):
        yield {"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}

    with patch("yt_whisper.runner.check_subtitles") as mock_subs, \
         patch("yt_whisper.runner.download_audio") as mock_dl, \
         patch("yt_whisper.runner.transcribe", side_effect=fake_transcribe):
        mock_subs.return_value = (None, _fake_metadata(duration=0))
        mock_dl.return_value = "/tmp/fake.wav"
        run(cfg, listener)

    progress_events = [
        e for e in listener.events
        if e[0] == "progress" and e[1] == "transcribe"
    ]
    assert len(progress_events) == 0, "Should emit no transcribe progress when duration is 0"


# --- Fix 2: log prompt profile ---

def test_run_logs_prompt_profile_manual(tmpdir_out):
    """on_log should be called with a message containing 'Prompt profile: infosec'."""
    from yt_whisper.runner import run
    cfg = RunConfig(
        url="https://yt/abc", output_dir=tmpdir_out,
        force_whisper=True, prompt_profile="infosec",
    )
    listener = _Capture()

    def fake_transcribe(*args, **kwargs):
        yield {"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}

    with patch("yt_whisper.runner.check_subtitles") as mock_subs, \
         patch("yt_whisper.runner.download_audio") as mock_dl, \
         patch("yt_whisper.runner.transcribe", side_effect=fake_transcribe):
        mock_subs.return_value = (None, _fake_metadata())
        mock_dl.return_value = "/tmp/fake.wav"
        run(cfg, listener)

    log_events = [e for e in listener.events if e[0] == "log"]
    matching = [e for e in log_events if "Prompt profile:" in e[2] and "infosec" in e[2]]
    assert len(matching) >= 1, (
        f"Expected a log containing 'Prompt profile: infosec', got logs: {log_events}"
    )
