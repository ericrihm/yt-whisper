"""CLI entrypoint: thin shim around runner.run with optional TUI launch."""

import argparse
import sys

from yt_whisper.runner import RunConfig, ConsoleListener, run


def build_parser():
    parser = argparse.ArgumentParser(
        prog="yt-whisper",
        description="Transcribe YouTube videos using faster-whisper or YouTube subtitles.",
    )
    parser.add_argument("url", nargs="?", help="YouTube video URL (omit for interactive TUI)")
    parser.add_argument("--prompt", dest="prompt_profile", default="general",
                        help="Named prompt profile (general, grc, infosec) or custom string")
    parser.add_argument("--force-whisper", action="store_true",
                        help="Skip YouTube subtitle check, always use Whisper")
    parser.add_argument("--output-dir", default="./transcripts",
                        help="Output directory (default: ./transcripts)")
    parser.add_argument("--model", default="large-v3",
                        help="Whisper model size (default: large-v3)")
    parser.add_argument("--format", dest="output_format", default="both",
                        choices=["md", "json", "both"],
                        help="Output format (default: both)")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument("--diarize", action="store_true",
                        help="Enable speaker diarization (requires optional setup -- see README)")
    parser.add_argument("--speakers", dest="num_speakers", type=int, default=None,
                        help="Exact number of speakers (diarize only)")
    parser.add_argument("--min-speakers", dest="min_speakers", type=int, default=None,
                        help="Minimum number of speakers (diarize only)")
    parser.add_argument("--max-speakers", dest="max_speakers", type=int, default=None,
                        help="Maximum number of speakers (diarize only)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show progress details and segment timing")
    return parser


def launch_tui():
    """Launch the interactive TUI."""
    from yt_whisper.tui.app import YtWhisperApp
    YtWhisperApp().run()


def _print_summary(result):
    if result is None:
        return
    print(f"\n[OK] {result['title']}")
    print(f"  Duration:    {result['duration_formatted']}")
    wpm = result.get("wpm")
    if wpm is not None:
        print(f"  Words:       {result['word_count']} ({wpm:.0f} words/min)")
        if wpm < 100:
            print(f"  Warning: Low word count ({wpm:.0f} words/min). "
                  f"Expected ~150. Transcript may be incomplete.")
        if wpm > 200:
            print(f"  Warning: High word count ({wpm:.0f} words/min). "
                  f"May indicate repeated or hallucinated text.")
    else:
        print(f"  Words:       {result['word_count']}")
        print("  Note: Video too short for word count validation.")
    print(f"  Method:      {result['method']}")
    for i, path in enumerate(result["paths"]):
        prefix = "  Output:      " if i == 0 else "               "
        print(f"{prefix}{path}")


def main():
    # No arguments -> launch TUI
    if len(sys.argv) == 1:
        launch_tui()
        return

    parser = build_parser()
    args = parser.parse_args()

    if not args.url:
        launch_tui()
        return

    cfg = RunConfig(
        url=args.url,
        model=args.model,
        language=args.language,
        prompt_profile=args.prompt_profile,
        diarize=args.diarize,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        output_format=args.output_format,
        output_dir=args.output_dir,
        force_whisper=args.force_whisper,
        verbose=args.verbose,
    )

    listener = ConsoleListener(verbose=args.verbose)
    result = run(cfg, listener)
    _print_summary(result)

    if result is None:
        sys.exit(1)
