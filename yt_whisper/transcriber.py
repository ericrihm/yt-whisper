"""Transcribe audio with faster-whisper CPU/CUDA or optional Apple Metal."""

import importlib.util
import os
import platform
import threading

from yt_whisper import cuda_preload

# MLX Whisper holds a process-wide mutable model cache; threaded TUI runs share it.
_METAL_TRANSCRIBE_LOCK = threading.Lock()


class TranscriptionError(Exception):
    """Raised on transcription failure (empty output, model load error)."""
    pass


def _select_backend(backend):
    if backend not in ("auto", "cpu", "cuda", "metal"):
        raise TranscriptionError(f"Unknown transcription backend: {backend}")
    darwin = platform.system() == "Darwin"
    apple_silicon = darwin and platform.machine().lower() in ("arm64", "aarch64")
    if backend == "cpu":
        return "cpu"
    if backend == "cuda":
        if darwin:
            raise TranscriptionError("CUDA is unavailable on macOS; use cpu or metal")
        return "cuda"
    if backend == "metal" and not apple_silicon:
        raise TranscriptionError("Metal transcription requires Apple-silicon macOS")
    if not apple_silicon:
        return "cpu" if darwin else "cuda"
    if importlib.util.find_spec("mlx_whisper") is None:
        if backend == "metal":
            raise TranscriptionError("Metal backend requires: pip install 'yt-whisper[metal]'")
        return "cpu"
    import mlx.core as mx
    if not mx.metal.is_available():
        if backend == "metal":
            raise TranscriptionError("Metal is unavailable on this Apple-silicon runtime")
        return "cpu"
    return "metal"


def _transcribe_metal(audio_path, model_size, prompt_text, language, verbose, listener):
    import mlx.core as mx
    import mlx_whisper

    aliases = {
        "tiny", "tiny.en", "base", "base.en", "small", "small.en",
        "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3",
    }
    if model_size in aliases:
        model_path = f"mlx-community/whisper-{model_size}-mlx"
    elif os.path.isdir(model_size) or "/" in model_size:
        model_path = model_size
    else:
        raise TranscriptionError("Use a standard Whisper model name or an MLX model directory/repository")
    with _METAL_TRANSCRIBE_LOCK:
        mx.set_default_device(mx.gpu)
        if listener is not None:
            listener.on_log("info", f"whisper: using metal (MLX), model={model_path}")
        # A selected Metal failure propagates; it must not silently rerun on CPU.
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_path,
            language=language,
            initial_prompt=prompt_text,
            verbose=verbose,
        )
    for segment in result["segments"]:
        yield {
            "start": round(segment["start"], 2),
            "end": round(segment["end"], 2),
            "text": segment["text"].strip(),
            "speaker": None,
        }


def _check_model_cached(model_size):
    """Best-effort check if model is already downloaded."""
    cache_dir = os.environ.get(
        "HF_HUB_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    )
    model_dir = f"models--Systran--faster-whisper-{model_size}"
    return os.path.isdir(os.path.join(cache_dir, model_dir))


def transcribe(audio_path, model_size, prompt_text, language, verbose, listener=None, *, backend="auto"):
    """Yield transcription segments one at a time.

    Each yielded segment: {"start", "end", "text", "speaker": None}.
    Caller is responsible for checking whether any segments were produced.
    """
    selected = _select_backend(backend)
    if selected == "metal":
        yield from _transcribe_metal(audio_path, model_size, prompt_text, language, verbose, listener)
        return
    cuda_preload.ensure_dlls()

    # Local import -- faster_whisper must not be imported at module level (Anti-Pattern #1)
    from faster_whisper import WhisperModel

    device = selected
    compute_type = "int8" if device == "cpu" else "float16"
    try:
        if not _check_model_cached(model_size):
            size_hint = "~3GB" if "large" in model_size else "~1GB"
            print(f"Downloading Whisper model '{model_size}' ({size_hint}). One-time download.")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        if listener is not None:
            listener.on_log("info", f"whisper: using {device} ({compute_type})")
    except (RuntimeError, ValueError) as e:
        if device == "cpu" or backend != "auto":
            raise
        print(
            f"Warning: CUDA unavailable -- falling back to CPU. "
            f"This will be significantly slower. "
            f"Check NVIDIA drivers and CUDA toolkit. ({e})"
        )
        device = "cpu"
        compute_type = "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        if listener is not None:
            listener.on_log("warning", f"whisper: CUDA unavailable, using {device} ({compute_type}) -- {e}")

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
