"""Pipeline orchestration and listener interface shared by CLI and TUI."""

import sys
import tempfile
import threading
from dataclasses import dataclass, asdict
from typing import Optional

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
