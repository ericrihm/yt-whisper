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
