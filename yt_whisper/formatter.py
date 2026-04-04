"""Format transcription output as markdown and JSON."""

import json
import os
import re


def format_duration(seconds):
    """Format seconds as H:MM:SS (if >= 1 hour) or M:SS."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_paragraphs(text):
    """Split text into paragraphs of ~5 sentences each."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]

    if len(sentences) <= 5:
        return " ".join(sentences)

    paragraphs = []
    for i in range(0, len(sentences), 5):
        paragraph = " ".join(sentences[i:i + 5])
        paragraphs.append(paragraph)

    return "\n\n".join(paragraphs)
