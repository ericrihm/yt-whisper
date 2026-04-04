from yt_whisper.downloader import parse_json3_subtitles, VideoUnavailableError


def test_parse_json3_basic():
    json3_data = {
        "events": [
            {"segs": [{"utf8": "Hello "}, {"utf8": "world."}]},
            {"segs": [{"utf8": " This is a test."}]},
        ]
    }
    result = parse_json3_subtitles(json3_data)
    assert result == "Hello world. This is a test."


def test_parse_json3_strips_html_tags():
    json3_data = {
        "events": [
            {"segs": [{"utf8": "<c>Hello</c> world."}]},
        ]
    }
    result = parse_json3_subtitles(json3_data)
    assert "<c>" not in result
    assert "Hello world." in result


def test_parse_json3_handles_newlines():
    json3_data = {
        "events": [
            {"segs": [{"utf8": "Line one.\n"}]},
            {"segs": [{"utf8": "Line two."}]},
        ]
    }
    result = parse_json3_subtitles(json3_data)
    assert "\n" not in result
    assert "Line one. Line two." in result


def test_parse_json3_skips_events_without_segs():
    json3_data = {
        "events": [
            {"tStartMs": 0},
            {"segs": [{"utf8": "Hello."}]},
        ]
    }
    result = parse_json3_subtitles(json3_data)
    assert result == "Hello."


def test_parse_json3_empty_events():
    json3_data = {"events": []}
    result = parse_json3_subtitles(json3_data)
    assert result == ""


def test_video_unavailable_error_is_exception():
    err = VideoUnavailableError("Video is private")
    assert isinstance(err, Exception)
    assert str(err) == "Video is private"


import json
from unittest.mock import patch, MagicMock
from yt_whisper.downloader import check_subtitles, download_audio


def _make_info_dict(subtitles=None, automatic_captions=None):
    """Build a minimal yt-dlp info_dict for testing."""
    return {
        "id": "test123",
        "title": "Test Video",
        "channel": "Test Channel",
        "upload_date": "20260101",
        "duration": 600,
        "webpage_url": "https://www.youtube.com/watch?v=test123",
        "subtitles": subtitles or {},
        "automatic_captions": automatic_captions or {},
    }


@patch("yt_whisper.downloader.yt_dlp.YoutubeDL")
def test_check_subtitles_returns_metadata(mock_ydl_class):
    mock_ydl = MagicMock()
    mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
    mock_ydl.extract_info.return_value = _make_info_dict()

    text, metadata = check_subtitles("https://youtube.com/watch?v=test123")
    assert metadata["video_id"] == "test123"
    assert metadata["title"] == "Test Video"
    assert metadata["duration"] == 600
    assert metadata["url"] == "https://www.youtube.com/watch?v=test123"


@patch("yt_whisper.downloader.yt_dlp.YoutubeDL")
def test_check_subtitles_no_subs_returns_none(mock_ydl_class):
    mock_ydl = MagicMock()
    mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
    mock_ydl.extract_info.return_value = _make_info_dict()

    text, metadata = check_subtitles("https://youtube.com/watch?v=test123")
    assert text is None


@patch("yt_whisper.downloader.yt_dlp.YoutubeDL")
def test_check_subtitles_prefers_manual_over_auto(mock_ydl_class):
    mock_ydl = MagicMock()
    mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)

    json3_content = json.dumps({"events": [{"segs": [{"utf8": "Manual sub."}]}]}).encode()
    mock_response = MagicMock()
    mock_response.read.return_value = json3_content
    mock_ydl.urlopen.return_value = mock_response

    mock_ydl.extract_info.return_value = _make_info_dict(
        subtitles={"en": [{"ext": "json3", "url": "http://example.com/sub.json3"}]},
        automatic_captions={"en": [{"ext": "json3", "url": "http://example.com/auto.json3"}]},
    )

    text, metadata = check_subtitles("https://youtube.com/watch?v=test123")
    assert text == "Manual sub."
    # Should have fetched the manual sub URL, not auto
    mock_ydl.urlopen.assert_called_once_with("http://example.com/sub.json3")
