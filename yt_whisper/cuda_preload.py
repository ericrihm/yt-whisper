"""
Windows-specific DLL preloading for faster-whisper CUDA support.

Problem: Microsoft Store Python's sandbox prevents normal DLL discovery.
Solution: Explicitly load DLLs via ctypes.WinDLL() before importing faster_whisper.

This module must be called BEFORE any import of faster_whisper or ctranslate2.
"""

import os
import sys
import ctypes
import importlib.util


def ensure_dlls():
    """Pre-load CUDA DLLs on Windows. No-op on Linux/Mac."""
    if sys.platform != "win32":
        return

    nvidia_spec = importlib.util.find_spec("nvidia")
    if nvidia_spec is None or nvidia_spec.submodule_search_locations is None:
        return

    nvidia_base = list(nvidia_spec.submodule_search_locations)[0]

    dll_paths = [
        os.path.join(nvidia_base, "cublas", "bin", "cublasLt64_12.dll"),
        os.path.join(nvidia_base, "cublas", "bin", "cublas64_12.dll"),
        os.path.join(nvidia_base, "cudnn", "bin", "cudnn_ops64_9.dll"),
    ]

    for dll_path in dll_paths:
        if os.path.exists(dll_path):
            try:
                ctypes.WinDLL(dll_path)
            except OSError as e:
                print(f"Warning: Failed to preload {os.path.basename(dll_path)}: {e}")
