import types
import pytest
from unittest.mock import patch, MagicMock, call
from yt_whisper.transcriber import TranscriptionError


def _mock_segment(start, end, text):
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = text
    return seg


def test_transcription_error_is_exception():
    err = TranscriptionError("No speech detected")
    assert isinstance(err, Exception)
    assert str(err) == "No speech detected"


@patch("yt_whisper.transcriber.cuda_preload")
def test_transcribe_calls_cuda_preload(mock_preload):
    """Verify cuda_preload.ensure_dlls() is called before model creation."""
    mock_fw = MagicMock()
    mock_model = MagicMock()
    mock_fw.WhisperModel.return_value = mock_model
    mock_model.transcribe.return_value = (
        iter([_mock_segment(0.0, 3.0, " Hello world.")]),
        MagicMock(),
    )

    # Patch sys.modules to intercept the local 'from faster_whisper import WhisperModel'
    with patch.dict("sys.modules", {"faster_whisper": mock_fw}):
        from yt_whisper.transcriber import transcribe
        result = list(transcribe("test.wav", "tiny", None, "en", False))

    mock_preload.ensure_dlls.assert_called_once()
    assert len(result) == 1
    assert result[0]["text"] == "Hello world."
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 3.0


@patch("yt_whisper.transcriber.cuda_preload")
def test_transcribe_cuda_fallback_to_cpu(mock_preload, capsys):
    """Verify CUDA failure falls back to CPU with warning."""
    mock_fw = MagicMock()

    # First WhisperModel() call raises RuntimeError (CUDA fail)
    # Second call succeeds (CPU fallback)
    mock_cpu_model = MagicMock()
    mock_cpu_model.transcribe.return_value = (
        iter([_mock_segment(0.0, 3.0, " Fallback text.")]),
        MagicMock(),
    )
    mock_fw.WhisperModel.side_effect = [RuntimeError("CUDA error"), mock_cpu_model]

    with patch.dict("sys.modules", {"faster_whisper": mock_fw}):
        from yt_whisper.transcriber import transcribe
        result = list(transcribe("test.wav", "tiny", None, "en", False))

    assert len(result) == 1
    assert result[0]["text"] == "Fallback text."

    captured = capsys.readouterr()
    assert "CUDA unavailable" in captured.out
    assert "falling back to CPU" in captured.out
    assert "--" in captured.out  # ASCII em dash

    # Verify second call used cpu/int8
    calls = mock_fw.WhisperModel.call_args_list
    assert calls[1] == call("tiny", device="cpu", compute_type="int8")


@patch("yt_whisper.transcriber.cuda_preload")
def test_transcribe_is_generator(mock_preload):
    """transcribe() must return a generator so runner can stream segments."""
    mock_fw = MagicMock()
    mock_model = MagicMock()
    mock_fw.WhisperModel.return_value = mock_model
    mock_model.transcribe.return_value = (
        iter([_mock_segment(0.0, 1.0, " hi")]),
        MagicMock(),
    )
    with patch.dict("sys.modules", {"faster_whisper": mock_fw}):
        from yt_whisper.transcriber import transcribe
        result = transcribe("test.wav", "tiny", None, "en", False)
    assert isinstance(result, types.GeneratorType)


@patch("yt_whisper.transcriber.cuda_preload")
def test_transcribe_yields_speaker_none(mock_preload):
    """Each yielded segment must have speaker=None (diarizer fills it later)."""
    mock_fw = MagicMock()
    mock_model = MagicMock()
    mock_fw.WhisperModel.return_value = mock_model
    mock_model.transcribe.return_value = (
        iter([_mock_segment(0.0, 1.0, " hi")]),
        MagicMock(),
    )
    with patch.dict("sys.modules", {"faster_whisper": mock_fw}):
        from yt_whisper.transcriber import transcribe
        segs = list(transcribe("test.wav", "tiny", None, "en", False))
    assert segs[0]["speaker"] is None


# --- Fix 3: log GPU vs CPU device selection ---

@patch("yt_whisper.transcriber.cuda_preload")
def test_transcribe_listener_logs_cuda(mock_preload):
    """When CUDA succeeds, listener.on_log is called with a message containing 'cuda'."""
    mock_fw = MagicMock()
    mock_model = MagicMock()
    mock_fw.WhisperModel.return_value = mock_model
    mock_model.transcribe.return_value = (
        iter([_mock_segment(0.0, 1.0, " hi")]),
        MagicMock(),
    )

    listener = MagicMock()

    with patch.dict("sys.modules", {"faster_whisper": mock_fw}):
        from yt_whisper.transcriber import transcribe
        list(transcribe("test.wav", "tiny", None, "en", False, listener=listener))

    # listener.on_log should have been called with level "info" and message containing "cuda"
    calls = listener.on_log.call_args_list
    info_cuda_calls = [
        c for c in calls
        if c[0][0] == "info" and "cuda" in c[0][1].lower()
    ]
    assert len(info_cuda_calls) >= 1, f"Expected on_log('info', '...cuda...'), got calls: {calls}"


@patch("yt_whisper.transcriber.cuda_preload")
def test_transcribe_listener_logs_cpu_fallback(mock_preload, capsys):
    """When CUDA fails, listener.on_log is called with level 'warning' and message containing 'cpu'."""
    mock_fw = MagicMock()
    mock_cpu_model = MagicMock()
    mock_cpu_model.transcribe.return_value = (
        iter([_mock_segment(0.0, 1.0, " fallback")]),
        MagicMock(),
    )
    mock_fw.WhisperModel.side_effect = [RuntimeError("CUDA error"), mock_cpu_model]

    listener = MagicMock()

    with patch.dict("sys.modules", {"faster_whisper": mock_fw}):
        from yt_whisper.transcriber import transcribe
        list(transcribe("test.wav", "tiny", None, "en", False, listener=listener))

    # listener.on_log should have been called with level "warning" and message containing "cpu"
    calls = listener.on_log.call_args_list
    warning_cpu_calls = [
        c for c in calls
        if c[0][0] == "warning" and "cpu" in c[0][1].lower()
    ]
    assert len(warning_cpu_calls) >= 1, (
        f"Expected on_log('warning', '...cpu...'), got calls: {calls}"
    )

    # Existing print-based test still works -- stdout should still have the warning
    captured = capsys.readouterr()
    assert "CUDA unavailable" in captured.out
