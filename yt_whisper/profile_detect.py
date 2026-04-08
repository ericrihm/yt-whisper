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
