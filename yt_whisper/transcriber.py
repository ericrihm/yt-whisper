"""Transcribe audio using faster-whisper with CUDA GPU support."""

import os

from yt_whisper import cuda_preload


class TranscriptionError(Exception):
    """Raised on transcription failure (empty output, model load error)."""
    pass


def _check_model_cached(model_size):
    """Best-effort check if model is already downloaded."""
    cache_dir = os.environ.get(
        "HF_HUB_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    )
    model_dir = f"models--Systran--faster-whisper-{model_size}"
    return os.path.isdir(os.path.join(cache_dir, model_dir))


def transcribe(audio_path, model_size, prompt_text, language, verbose):
    """Yield transcription segments one at a time.

    Each yielded segment: {"start", "end", "text", "speaker": None}.
    Caller is responsible for checking whether any segments were produced.
    """
    cuda_preload.ensure_dlls()

    # Local import -- faster_whisper must not be imported at module level (Anti-Pattern #1)
    from faster_whisper import WhisperModel

    device = "cuda"
    compute_type = "float16"
    try:
        if not _check_model_cached(model_size):
            size_hint = "~3GB" if "large" in model_size else "~1GB"
            print(f"Downloading Whisper model '{model_size}' ({size_hint}). One-time download.")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except (RuntimeError, ValueError) as e:
        print(
            f"Warning: CUDA unavailable -- falling back to CPU. "
            f"This will be significantly slower. "
            f"Check NVIDIA drivers and CUDA toolkit. ({e})"
        )
        device = "cpu"
        compute_type = "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments_gen, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,
        initial_prompt=prompt_text,
    )

    for seg in segments_gen:
        text = seg.text.strip()
        if verbose:
            print(f"  [{seg.start:.1f}s -> {seg.end:.1f}s] {text}")
        yield {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": text,
            "speaker": None,
        }
