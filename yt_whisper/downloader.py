"""YouTube subtitle checking and audio downloading via yt-dlp Python API."""

import json
import os
import re

import yt_dlp


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


def _build_language_priority(language):
    """Build language code priority list. e.g., 'en' -> ['en', 'en-US', 'en-GB']."""
    variants = [language]
    if "-" not in language and "_" not in language:
        variants.extend([f"{language}-US", f"{language}-GB"])
    return variants


def _find_subtitle_entry(caption_dict, lang_priority):
    """Find best subtitle entry from a caption dict. Returns (url, ext) or None."""
    format_priority = ["json3", "vtt", "srv1"]
    for lang in lang_priority:
        if lang not in caption_dict:
            continue
        entries = caption_dict[lang]
        for fmt in format_priority:
            for entry in entries:
                if entry.get("ext") == fmt:
                    return entry["url"], fmt
    return None


def check_subtitles(url, language="en"):
    """Check YouTube for existing subtitles. Returns (text_or_None, metadata)."""
    ydl_opts = {"quiet": True, "no_warnings": True}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            metadata = {
                "video_id": info.get("id", ""),
                "title": info.get("title", "Unknown"),
                "channel": info.get("channel", info.get("uploader", "Unknown")),
                "upload_date": info.get("upload_date", ""),
                "duration": info.get("duration", 0),
                "url": info.get("webpage_url", url),
            }

            lang_priority = _build_language_priority(language)
            subtitles = info.get("subtitles") or {}
            auto_captions = info.get("automatic_captions") or {}

            # Prefer manual subs over auto-generated
            result = _find_subtitle_entry(subtitles, lang_priority)
            if result is None:
                result = _find_subtitle_entry(auto_captions, lang_priority)

            if result is None:
                return None, metadata

            sub_url, fmt = result

            # Fetch subtitle content using same YoutubeDL instance
            try:
                response = ydl.urlopen(sub_url)
                raw_data = response.read()
            except Exception as e:
                print(f"Warning: Failed to download subtitles: {e}")
                return None, metadata

    except yt_dlp.utils.DownloadError as e:
        raise VideoUnavailableError(str(e)) from e

    if fmt == "json3":
        json3_data = json.loads(raw_data)
        text = parse_json3_subtitles(json3_data)
    else:
        # VTT/SRV1 fallback: strip tags and timestamps, extract text
        text = raw_data.decode("utf-8", errors="replace")
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}', '', text)
        text = re.sub(r'WEBVTT.*?\n\n', '', text, flags=re.DOTALL)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

    if not text:
        return None, metadata
    return text, metadata


def download_audio(url, temp_dir, metadata, verbose=False):
    """Download audio from YouTube as 16kHz mono WAV. Returns audio file path."""
    video_id = metadata["video_id"]

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "postprocessor_args": {"ffmpeg": ["-ar", "16000", "-ac", "1"]},
        "quiet": not verbose,
        "no_warnings": not verbose,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        raise VideoUnavailableError(str(e)) from e

    audio_path = os.path.join(temp_dir, f"{video_id}.wav")
    if not os.path.exists(audio_path):
        raise VideoUnavailableError(
            f"Audio file not found after download. Expected: {audio_path}"
        )

    return audio_path
