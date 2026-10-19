import platform
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from yt_whisper import transcriber
from yt_whisper.transcriber import TranscriptionError, transcribe


@pytest.fixture
def model_factory(monkeypatch):
    model = SimpleNamespace(transcribe=lambda *args, **kwargs: (iter(()), None))
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=factory))
    monkeypatch.setattr(transcriber.importlib.util, "find_spec", lambda name: None)
    return factory


def test_apple_silicon_uses_cpu_without_attempting_cuda(monkeypatch, model_factory):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    list(transcribe("sample.wav", "tiny.en", None, "en", False))
    model_factory.assert_called_once_with("tiny.en", device="cpu", compute_type="int8")


def test_native_cpu_model_failure_is_not_retried_as_cuda_fallback(monkeypatch, model_factory):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    error = RuntimeError("invalid local model")
    model_factory.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        list(transcribe("sample.wav", "tiny.en", None, "en", False))
    assert caught.value is error
    assert model_factory.call_count == 1


def test_linux_preserves_cuda_selection(monkeypatch, model_factory):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    list(transcribe("sample.wav", "tiny.en", None, "en", False))
    model_factory.assert_called_once_with("tiny.en", device="cuda", compute_type="float16")


def test_cuda_initialization_failure_falls_back_to_cpu(monkeypatch, model_factory):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    model_factory.side_effect = [RuntimeError("CUDA unavailable"), model_factory.return_value]
    list(transcribe("sample.wav", "tiny.en", None, "en", False))
    assert model_factory.call_args_list == [
        call("tiny.en", device="cuda", compute_type="float16"),
        call("tiny.en", device="cpu", compute_type="int8"),
    ]


def test_failed_cpu_fallback_propagates_its_error(monkeypatch, model_factory):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    cpu_error = ValueError("model cannot be loaded")
    model_factory.side_effect = [RuntimeError("CUDA unavailable"), cpu_error]
    with pytest.raises(ValueError) as caught:
        list(transcribe("sample.wav", "tiny.en", None, "en", False))
    assert caught.value is cpu_error
    assert model_factory.call_count == 2


def test_explicit_cpu_ignores_installed_metal(monkeypatch, model_factory):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(transcriber.importlib.util, "find_spec", lambda name: object())
    list(transcribe("sample.wav", "tiny.en", None, "en", False, backend="cpu"))
    model_factory.assert_called_once_with("tiny.en", device="cpu", compute_type="int8")


def test_explicit_cuda_failure_does_not_change_backend(monkeypatch, model_factory):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    error = RuntimeError("CUDA initialization failed")
    model_factory.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        list(transcribe("sample.wav", "tiny.en", None, "en", False, backend="cuda"))
    assert caught.value is error
    assert model_factory.call_count == 1


@pytest.mark.parametrize("backend", ["metal", "cuda"])
def test_incompatible_hardware_refuses_before_model_load(monkeypatch, model_factory, backend):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    with pytest.raises(TranscriptionError):
        list(transcribe("sample.wav", "tiny.en", None, "en", False, backend=backend))
    model_factory.assert_not_called()


def test_missing_metal_extra_refuses_without_cpu_fallback(monkeypatch, model_factory):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    with pytest.raises(TranscriptionError):
        list(transcribe("sample.wav", "tiny.en", None, "en", False, backend="metal"))
    model_factory.assert_not_called()


def test_selected_metal_failure_propagates_without_cpu_fallback(monkeypatch, model_factory):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(transcriber.importlib.util, "find_spec", lambda name: object())
    mx = SimpleNamespace(
        metal=SimpleNamespace(is_available=lambda: True),
        gpu=object(),
        set_default_device=lambda device: None,
    )
    error = RuntimeError("Metal allocation failed")
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=Mock(side_effect=error)))
    with pytest.raises(RuntimeError) as caught:
        list(transcribe("sample.wav", "tiny.en", None, "en", False, backend="metal"))
    assert caught.value is error
    model_factory.assert_not_called()
