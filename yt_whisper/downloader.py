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
