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


def format_output(text_or_segments, metadata, output_format, output_dir,
                  model=None, prompt_profile=None, method="whisper", language="en"):
    """Write transcript to markdown and/or JSON files. Returns list of paths written."""
    os.makedirs(output_dir, exist_ok=True)
    video_id = metadata["video_id"]

    # Assemble full text
    if isinstance(text_or_segments, list):
        full_text = " ".join(seg["text"].strip() for seg in text_or_segments)
        segments = text_or_segments
    else:
        full_text = text_or_segments
        segments = None

    full_text = re.sub(r'\s+', ' ', full_text).strip()
    word_count = len(full_text.split())
    duration_formatted = format_duration(metadata["duration"])

    # Build method display string
    if method == "youtube_subs":
        method_display = "youtube_subs"
    elif prompt_profile and prompt_profile != "general":
        method_display = f"whisper ({model} / {prompt_profile})"
    else:
        method_display = f"whisper ({model})"

    paths = []

    if output_format in ("md", "both"):
        md_path = os.path.join(output_dir, f"{video_id}.md")
        paragraphed = format_paragraphs(full_text)
        md_content = (
            f"# {metadata['title']}\n\n"
            f"- **Channel**: {metadata['channel']}\n"
            f"- **Date**: {metadata['upload_date']}\n"
            f"- **Duration**: {duration_formatted}\n"
            f"- **URL**: {metadata['url']}\n"
            f"- **Language**: {language}\n"
            f"- **Word Count**: {word_count}\n"
            f"- **Transcription Method**: {method_display}\n\n"
            f"---\n\n"
            f"{paragraphed}\n"
        )
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        paths.append(md_path)

    if output_format in ("json", "both"):
        json_path = os.path.join(output_dir, f"{video_id}.json")
        json_data = {
            "video_id": video_id,
            "title": metadata["title"],
            "channel": metadata["channel"],
            "url": metadata["url"],
            "upload_date": metadata["upload_date"],
            "language": language,
            "duration_seconds": metadata["duration"],
            "duration_formatted": duration_formatted,
            "word_count": word_count,
            "transcription_method": method,
            "model": model,
            "prompt_profile": prompt_profile if prompt_profile != "general" else None,
            "segments": segments,
            "full_text": full_text,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        paths.append(json_path)

    return paths
