"""Smoke tests for Textual TUI. Use Textual's App.run_test harness."""

import json
import os
import pytest

pytest.importorskip("textual")

from yt_whisper.tui.app import YtWhisperApp


async def test_home_screen_mounts(tmp_path):
    app = YtWhisperApp(output_dir=str(tmp_path))
    async with app.run_test() as pilot:
        assert app.query_one("#url-input") is not None
        assert app.query_one("#model-select") is not None
        assert app.query_one("#diarize-toggle") is not None


async def test_form_builds_runconfig(tmp_path):
    app = YtWhisperApp(output_dir=str(tmp_path))
    async with app.run_test() as pilot:
        app.query_one("#url-input").value = "https://yt/xyz"
        cfg = app.build_runconfig()
        assert cfg.url == "https://yt/xyz"
        assert cfg.output_dir == str(tmp_path)


async def test_history_loads_existing_runs(tmp_path):
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
        assert len(history_list.children) >= 1


async def test_run_screen_mounts_on_start(tmp_path):
    """Submitting the form pushes the Run screen."""
    from unittest.mock import patch

    app = YtWhisperApp(output_dir=str(tmp_path))
    async with app.run_test() as pilot:
        app.query_one("#url-input").value = "https://yt/abc"
        # Patch runner.run so the worker finishes immediately without side effects
        with patch("yt_whisper.tui.app.run", return_value=None):
            app.start_run(app.build_runconfig())
            await pilot.pause(0.3)
        from yt_whisper.tui.app import RunScreen
        assert any(isinstance(s, RunScreen) for s in app.screen_stack)


async def test_enter_in_url_input_starts_run(tmp_path):
    """Pressing Enter in the URL input should trigger action_run."""
    from unittest.mock import patch

    app = YtWhisperApp(output_dir=str(tmp_path))
    async with app.run_test() as pilot:
        app.query_one("#url-input").value = "https://yt/abc"
        with patch("yt_whisper.tui.app.run", return_value=None):
            await pilot.press("tab")  # ensure focus somewhere
            app.query_one("#url-input").focus()
            await pilot.press("enter")
            await pilot.pause(0.3)
        from yt_whisper.tui.app import RunScreen
        assert any(isinstance(s, RunScreen) for s in app.screen_stack)


async def test_diarize_toggle_shows_modal_when_missing(tmp_path, monkeypatch):
    """Toggling Diarize on without HF_TOKEN should push the setup modal."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    app = YtWhisperApp(output_dir=str(tmp_path))
    async with app.run_test() as pilot:
        app.query_one("#diarize-toggle").value = True
        await pilot.pause(0.1)
        from yt_whisper.tui.app import DiarizeSetupModal
        assert any(isinstance(s, DiarizeSetupModal) for s in app.screen_stack)


async def test_diarize_toggle_no_modal_when_ready(tmp_path, monkeypatch):
    """If HF_TOKEN is set and pyannote is importable, no modal should appear."""
    monkeypatch.setenv("HF_TOKEN", "hf_fake")
    import sys
    import types
    # Stub pyannote.audio so the import check passes
    if "pyannote" not in sys.modules:
        pyannote_mod = types.ModuleType("pyannote")
        pyannote_audio_mod = types.ModuleType("pyannote.audio")
        pyannote_mod.audio = pyannote_audio_mod
        sys.modules["pyannote"] = pyannote_mod
        sys.modules["pyannote.audio"] = pyannote_audio_mod

    app = YtWhisperApp(output_dir=str(tmp_path))
    async with app.run_test() as pilot:
        app.query_one("#diarize-toggle").value = True
        await pilot.pause(0.1)
        from yt_whisper.tui.app import DiarizeSetupModal
        assert not any(isinstance(s, DiarizeSetupModal) for s in app.screen_stack)


async def test_run_worker_suppresses_stdout(tmp_path):
    """Worker thread must not leak library noise to real stdout/stderr."""
    import contextlib
    import sys
    from unittest.mock import patch, MagicMock

    stdout_redirected = []
    stderr_redirected = []
    original_redirect_stdout = contextlib.redirect_stdout
    original_redirect_stderr = contextlib.redirect_stderr

    def tracking_redirect_stdout(target):
        stdout_redirected.append(target)
        return original_redirect_stdout(target)

    def tracking_redirect_stderr(target):
        stderr_redirected.append(target)
        return original_redirect_stderr(target)

    def noisy_run(cfg, listener, cancel_event=None):
        # Simulate library writing directly to stdout/stderr
        print("NOISE_STDOUT", flush=True)
        print("NOISE_STDERR", file=sys.stderr, flush=True)

    app = YtWhisperApp(output_dir=str(tmp_path))
    async with app.run_test() as pilot:
        app.query_one("#url-input").value = "https://yt/abc"
        with patch("yt_whisper.tui.app.run", side_effect=noisy_run), \
             patch("yt_whisper.tui.app.contextlib.redirect_stdout", side_effect=tracking_redirect_stdout), \
             patch("yt_whisper.tui.app.contextlib.redirect_stderr", side_effect=tracking_redirect_stderr):
            app.start_run(app.build_runconfig())
            await pilot.pause(0.5)

    # Both redirects must have been called with StringIO targets (not real stdout/stderr)
    import io
    assert len(stdout_redirected) >= 1, "redirect_stdout was never called in worker"
    assert len(stderr_redirected) >= 1, "redirect_stderr was never called in worker"
    assert all(isinstance(t, io.StringIO) for t in stdout_redirected), \
        "redirect_stdout target must be StringIO, not devnull or real stdout"
    assert all(isinstance(t, io.StringIO) for t in stderr_redirected), \
        "redirect_stderr target must be StringIO, not devnull or real stderr"


async def test_preview_screen_renders_markdown(tmp_path):
    md_path = tmp_path / "abc.md"
    md_path.write_text("# Hello\n\nSome text.")
    from yt_whisper.tui.app import PreviewScreen

    app = YtWhisperApp(output_dir=str(tmp_path))
    async with app.run_test() as pilot:
        app.push_screen(PreviewScreen(str(md_path)))
        await pilot.pause(0.1)
        assert app.query_one("#preview-md") is not None
