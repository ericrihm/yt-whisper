"""CLI entrypoint and orchestration for yt-whisper."""

import argparse
import os
import sys
import tempfile

from yt_whisper.downloader import check_subtitles, download_audio, VideoUnavailableError
from yt_whisper.formatter import format_output, format_duration
from yt_whisper.prompts import resolve_prompt, PROMPTS
from yt_whisper.transcriber import transcribe, TranscriptionError

MIN_VALIDATION_SECONDS = 30


def build_parser():
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        prog="yt-whisper",
        description="Transcribe YouTube videos using faster-whisper or YouTube subtitles.",
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--prompt", default="general",
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
    parser.add_argument("--language", default="en",
                        help="Language code (default: en)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show progress details and segment timing")
    return parser


def validate_word_count(word_count, duration_seconds):
    """Print warnings if word count seems abnormal for the video duration.
    Returns wpm as float or None if too short to validate."""
    if duration_seconds < MIN_VALIDATION_SECONDS:
        print("  Note: Video too short for word count validation.")
        return None

    wpm = word_count / (duration_seconds / 60)

    if wpm < 100:
        print(f"  Warning: Low word count ({wpm:.0f} words/min). "
              f"Expected ~150. Transcript may be incomplete.")
    if wpm > 200:
        print(f"  Warning: High word count ({wpm:.0f} words/min). "
              f"May indicate repeated or hallucinated text.")
    return wpm


def main():
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    prompt_text = resolve_prompt(args.prompt)

    try:
        # Step 1: Check for existing subtitles
        subtitle_text = None
        metadata = None

        if args.verbose:
            print(f"Fetching video info...")

        subtitle_text, metadata = check_subtitles(args.url, args.language)

        if subtitle_text and not args.force_whisper:
            method = "youtube_subs"
            text_or_segments = subtitle_text
            if args.verbose:
                print(f"Found YouTube subtitles ({len(subtitle_text.split())} words)")
        else:
            if args.verbose:
                if args.force_whisper:
                    print("Forced Whisper transcription (--force-whisper)")
                else:
                    print("No usable subtitles found. Transcribing with Whisper...")

            # Step 2: Download audio and transcribe
            method = "whisper"
            with tempfile.TemporaryDirectory() as temp_dir:
                audio_path = download_audio(
                    args.url, temp_dir, metadata, verbose=args.verbose
                )
                if args.verbose:
                    print(f"Audio downloaded: {audio_path}")

                # Step 3: Transcribe
                text_or_segments = transcribe(
                    audio_path, args.model, prompt_text, args.language, args.verbose
                )

        # Step 4: Format output
        output_paths = format_output(
            text_or_segments, metadata, args.output_format, args.output_dir,
            model=args.model if method == "whisper" else None,
            prompt_profile=(args.prompt if args.prompt in PROMPTS else "custom") if method == "whisper" else None,
            method=method,
            language=args.language,
        )

        # Step 5: Summary
        if isinstance(text_or_segments, list):
            full_text = " ".join(seg["text"] for seg in text_or_segments)
        else:
            full_text = text_or_segments

        word_count = len(full_text.split())
        duration_formatted = format_duration(metadata["duration"])

        # Validate and get wpm
        wpm = validate_word_count(word_count, metadata["duration"])

        print(f"\n\u2713 {metadata['title']}")
        print(f"  Duration:    {duration_formatted}")
        if wpm is not None:
            print(f"  Words:       {word_count} ({wpm:.0f} words/min)")
        else:
            print(f"  Words:       {word_count}")
        print(f"  Method:      {method}")
        for i, path in enumerate(output_paths):
            prefix = "  Output:      " if i == 0 else "               "
            print(f"{prefix}{path}")

    except VideoUnavailableError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except TranscriptionError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
