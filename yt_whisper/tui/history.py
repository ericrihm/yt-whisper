"""Scan output_dir for past transcription runs. JSON files are the history."""

import glob
import json
import os


def list_history(output_dir):
    """Return a list of past runs sorted by mtime descending.

    Each entry: {
        "video_id", "title", "channel", "url", "upload_date",
        "duration_formatted", "diarize", "json_path", "md_path", "mtime", "config",
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
