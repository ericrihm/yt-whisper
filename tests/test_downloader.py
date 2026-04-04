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
