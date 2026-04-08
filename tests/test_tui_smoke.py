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
