"""Speaker diarization via pyannote.audio 3.1 (optional dependency).

pyannote is imported locally inside diarize() so users without the optional
extras never hit import errors.
"""

import os

from yt_whisper import cuda_preload


class DiarizationError(Exception):
    """Raised on missing deps, missing token, or pyannote runtime failure."""
    pass


_INSTALL_HINT = (
    "Diarization requires optional dependencies. Install with:\n"
    "  pip install -r requirements-diarize.txt\n"
    "Then set HF_TOKEN environment variable (see README: Optional Speaker Diarization)."
)

_TOKEN_HINT = (
    "HF_TOKEN environment variable not set. "
    "Get a token at https://huggingface.co/settings/tokens, "
    "accept the pyannote/speaker-diarization-3.1 license, then set the env var."
)


def diarize(audio_path, num_speakers=None, min_speakers=None, max_speakers=None, verbose=False):
    """Return list of {start, end, speaker} turns sorted by start.

    Raises DiarizationError on missing deps, missing HF_TOKEN, or runtime failure.
    """
    cuda_preload.ensure_dlls()

    # Local import -- never at module level.
    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        raise DiarizationError(_INSTALL_HINT) from e

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise DiarizationError(_TOKEN_HINT)

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
    except Exception as e:
        raise DiarizationError(f"Failed to load pyannote pipeline: {e}") from e

    # GPU fallback mirrors transcriber pattern
    try:
        import torch
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception as e:
        if verbose:
            print(f"Warning: diarization CUDA unavailable -- using CPU. ({e})")

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    try:
        diarization = pipeline(audio_path, **kwargs)
    except Exception as e:
        raise DiarizationError(f"Diarization pipeline failed: {e}") from e

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker,
        })
    turns.sort(key=lambda t: t["start"])
    return turns


def attach_speakers(whisper_segments, speaker_turns):
    """Assign each whisper segment the speaker whose turn overlaps it most.

    Mutates and returns the list of segments.
    """
    for seg in whisper_segments:
        best_speaker = None
        best_overlap = 0.0
        for turn in speaker_turns:
            overlap = min(seg["end"], turn["end"]) - max(seg["start"], turn["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        seg["speaker"] = best_speaker or "SPEAKER_UNKNOWN"
    return whisper_segments


def rename_speaker_labels(segments):
    """Rename SPEAKER_XX -> 'Speaker N' in order of first appearance.

    SPEAKER_UNKNOWN is preserved as-is. Mutates and returns segments.
    """
    mapping = {}
    next_idx = 1
    for seg in segments:
        sp = seg.get("speaker")
        if not sp or sp == "SPEAKER_UNKNOWN":
            continue
        if sp not in mapping:
            mapping[sp] = f"Speaker {next_idx}"
            next_idx += 1
    for seg in segments:
        sp = seg.get("speaker")
        if sp in mapping:
            seg["speaker"] = mapping[sp]
    return segments
