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
        result = transcribe("test.wav", "tiny", None, "en", False)

    mock_preload.ensure_dlls.assert_called_once()
    assert len(result) == 1
    assert result[0]["text"] == "Hello world."
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 3.0


@patch("yt_whisper.transcriber.cuda_preload")
def test_transcribe_empty_segments_raises(mock_preload):
    mock_fw = MagicMock()
    mock_model = MagicMock()
    mock_fw.WhisperModel.return_value = mock_model
    mock_model.transcribe.return_value = (iter([]), MagicMock())

    with patch.dict("sys.modules", {"faster_whisper": mock_fw}):
        from yt_whisper.transcriber import transcribe
        with pytest.raises(TranscriptionError, match="No speech detected"):
            transcribe("test.wav", "tiny", None, "en", False)


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
        result = transcribe("test.wav", "tiny", None, "en", False)

    assert len(result) == 1
    assert result[0]["text"] == "Fallback text."

    captured = capsys.readouterr()
    assert "CUDA unavailable" in captured.out
    assert "falling back to CPU" in captured.out

    # Verify second call used cpu/int8
    calls = mock_fw.WhisperModel.call_args_list
    assert calls[1] == call("tiny", device="cpu", compute_type="int8")
