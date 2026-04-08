"""Pipeline orchestration and listener interface shared by CLI and TUI."""

import sys
import tempfile
import threading
from dataclasses import dataclass, asdict
from typing import Optional

from yt_whisper.downloader import check_subtitles, download_audio, VideoUnavailableError
from yt_whisper.transcriber import transcribe, TranscriptionError
from yt_whisper.diarizer import diarize, attach_speakers, rename_speaker_labels, DiarizationError
from yt_whisper.formatter import format_output, format_duration
from yt_whisper.prompts import PROMPTS, resolve_prompt
from yt_whisper.profile_detect import detect_profile

MIN_VALIDATION_SECONDS = 30


@dataclass
class RunConfig:
    """All inputs needed to run the pipeline once."""
    url: str
    model: str = "large-v3"
    language: str = "en"
    prompt_profile: str = "general"  # profile name or custom string
    diarize: bool = False
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    output_format: str = "both"  # md | json | both
    output_dir: str = "./transcripts"
    force_whisper: bool = False
    verbose: bool = False

    def to_dict(self):
        return asdict(self)


class Listener:
    """Base listener. Override any subset of methods."""

    def on_phase(self, phase: str, status: str) -> None: ...
    def on_progress(self, phase: str, pct: float) -> None: ...
    def on_segment(self, segment: dict) -> None: ...
    def on_segments_relabeled(self, segments: list) -> None: ...
    def on_log(self, level: str, msg: str) -> None: ...
    def on_done(self, result: dict) -> None: ...
    def on_error(self, exc: BaseException) -> None: ...


class ConsoleListener(Listener):
    """Listener that reproduces the existing flag-mode stdout/stderr output."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def on_phase(self, phase, status):
        if not self.verbose:
            return
        if status == "start":
            msgs = {
                "fetch": "Fetching video info...",
                "subs": "Checking YouTube subtitles...",
                "download": "Downloading audio...",
                "transcribe": "Transcribing with Whisper...",
                "diarize": "Running speaker diarization...",
                "format": "Writing output...",
            }
            if phase in msgs:
                print(msgs[phase])

    def on_progress(self, phase, pct):
        # Flag mode doesn't show progress bars -- yt-dlp already prints its own
        pass

    def on_segment(self, segment):
        if not self.verbose:
            return
        print(f"  [{segment['start']:.1f}s -> {segment['end']:.1f}s] {segment['text']}")

    def on_segments_relabeled(self, segments):
        pass

    def on_log(self, level, msg):
        if level == "debug" and not self.verbose:
            return
        print(msg)

    def on_done(self, result):
        # The final summary block is printed by cli.py after run() returns,
        # using data in result. We don't duplicate it here.
        pass

    def on_error(self, exc):
        print(f"Error: {exc}", file=sys.stderr)


def _validate_wpm(word_count, duration_seconds):
    """Return wpm float or None if too short to validate."""
    if duration_seconds < MIN_VALIDATION_SECONDS:
        return None
    return word_count / (duration_seconds / 60)


def run(config: RunConfig, listener: Listener, cancel_event: Optional[threading.Event] = None):
    """Execute the full pipeline. All errors are routed through listener.on_error.

    Returns the result dict on success, or None on error / cancellation.
    """
    def cancelled():
        return cancel_event is not None and cancel_event.is_set()

    try:
        # Phase 1: fetch metadata + check subtitles
        listener.on_phase("fetch", "start")
        subtitle_text, metadata = check_subtitles(config.url, config.language)
        listener.on_phase("fetch", "done")

        if cancelled():
            listener.on_log("info", "Cancelled.")
            return None

        # Profile auto-detection (only if user didn't explicitly override)
        effective_profile = config.prompt_profile
        if config.prompt_profile == "general":
            detected, matched, _conf = detect_profile(metadata)
            if detected != "general":
                effective_profile = detected
                listener.on_log(
                    "info",
                    f"[auto] profile: {detected} (matched: {', '.join(matched[:5])})",
                )

        # Resolve prompt text from the (possibly auto-detected) profile
        prompt_text = resolve_prompt(effective_profile)
        listener.on_log("info", f"Prompt profile: {effective_profile}")

        # Subtitle fast path
        if subtitle_text and not config.force_whisper:
            listener.on_phase("subs", "start")
            listener.on_log(
                "info",
                f"Found YouTube subtitles ({len(subtitle_text.split())} words)",
            )
            listener.on_phase("subs", "done")
            method = "youtube_subs"
            text_or_segments = subtitle_text
        else:
            if config.force_whisper:
                listener.on_log("info", "Forced Whisper transcription (--force-whisper)")
            else:
                listener.on_log("info", "No usable subtitles found. Transcribing with Whisper...")

            # Phase 2: download audio
            listener.on_phase("download", "start")
            with tempfile.TemporaryDirectory() as temp_dir:
                audio_path = download_audio(
                    config.url, temp_dir, metadata, verbose=config.verbose
                )
                listener.on_phase("download", "done")

                if cancelled():
                    listener.on_log("info", "Cancelled.")
                    return None

                # Phase 3: transcribe (streaming)
                listener.on_phase("transcribe", "start")
                collected = []
                duration = metadata.get("duration") or 0
                for seg in transcribe(
                    audio_path, config.model, prompt_text, config.language, config.verbose,
                    listener=listener,
                ):
                    if cancelled():
                        listener.on_log("info", "Cancelled.")
                        return None
                    collected.append(seg)
                    listener.on_segment(seg)
                    if duration > 0:
                        pct = min(seg["end"] / duration, 1.0)
                        listener.on_progress("transcribe", pct)
                listener.on_phase("transcribe", "done")

                if not collected:
                    raise TranscriptionError("No speech detected in audio")

                # Phase 4: diarization (optional)
                if config.diarize:
                    listener.on_phase("diarize", "start")
                    try:
                        turns = diarize(
                            audio_path,
                            num_speakers=config.num_speakers,
                            min_speakers=config.min_speakers,
                            max_speakers=config.max_speakers,
                            verbose=config.verbose,
                        )
                        attach_speakers(collected, turns)
                        rename_speaker_labels(collected)
                        listener.on_segments_relabeled(collected)
                    except DiarizationError as e:
                        listener.on_phase("diarize", "done")
                        listener.on_error(e)
                        return None
                    listener.on_phase("diarize", "done")

            method = "whisper"
            text_or_segments = collected

        # Phase 5: format output
        listener.on_phase("format", "start")
        config_block = {
            "url": config.url,
            "model": config.model if method == "whisper" else None,
            "language": config.language,
            "prompt_profile": effective_profile if method == "whisper" else None,
            "diarize": config.diarize,
            "output_format": config.output_format,
        }
        profile_for_formatter = (
            effective_profile if effective_profile in PROMPTS else "custom"
        ) if method == "whisper" else None
        output_paths = format_output(
            text_or_segments, metadata, config.output_format, config.output_dir,
            model=config.model if method == "whisper" else None,
            prompt_profile=profile_for_formatter,
            method=method,
            language=config.language,
            config=config_block,
        )
        listener.on_phase("format", "done")

        # Build result
        if isinstance(text_or_segments, list):
            full_text = " ".join(seg["text"] for seg in text_or_segments)
        else:
            full_text = text_or_segments
        word_count = len(full_text.split())
        wpm = _validate_wpm(word_count, metadata["duration"])

        result = {
            "paths": output_paths,
            "title": metadata["title"],
            "duration_formatted": format_duration(metadata["duration"]),
            "word_count": word_count,
            "wpm": wpm,
            "method": method,
        }
        listener.on_done(result)
        return result

    except VideoUnavailableError as e:
        listener.on_error(e)
        return None
    except TranscriptionError as e:
        listener.on_error(e)
        return None
    except KeyboardInterrupt:
        listener.on_log("info", "Interrupted.")
        return None
    except Exception as e:
        listener.on_error(e)
        return None
