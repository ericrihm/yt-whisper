import json
import os
import tempfile

from yt_whisper.formatter import format_duration, format_output, format_paragraphs


def test_duration_seconds_only():
    assert format_duration(42) == "0:42"


def test_duration_minutes():
    assert format_duration(262) == "4:22"


def test_duration_minutes_leading_zero_seconds():
    assert format_duration(2707) == "45:07"


def test_duration_hours():
    assert format_duration(3930) == "1:05:30"


def test_duration_zero():
    assert format_duration(0) == "0:00"


def test_paragraphs_groups_sentences():
    text = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten."
    result = format_paragraphs(text)
    paragraphs = result.split("\n\n")
    assert len(paragraphs) == 2
    assert "One." in paragraphs[0]
    assert "Five." in paragraphs[0]
    assert "Six." in paragraphs[1]


def test_paragraphs_short_text_single_paragraph():
    text = "Just one sentence."
    result = format_paragraphs(text)
    assert "\n\n" not in result
    assert result == "Just one sentence."


def test_paragraphs_handles_question_marks():
    text = "What is this? It is a test. Does it work? Yes it does. Great!"
    result = format_paragraphs(text)
    assert "What is this?" in result


def _sample_metadata():
    return {
        "video_id": "test123",
        "title": "Test Video Title",
        "channel": "Test Channel",
        "upload_date": "20260101",
        "duration": 600,
        "url": "https://www.youtube.com/watch?v=test123",
    }


def _sample_segments():
    return [
        {"start": 0.0, "end": 3.0, "text": "Hello world."},
        {"start": 3.0, "end": 6.0, "text": "This is a test."},
    ]


def test_format_output_md_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = format_output(
            _sample_segments(), _sample_metadata(), "md", tmpdir,
            model="large-v3", prompt_profile="grc", method="whisper", language="en",
        )
        assert len(paths) == 1
        assert paths[0].endswith(".md")
        assert os.path.exists(paths[0])
        content = open(paths[0]).read()
        assert "# Test Video Title" in content
        assert "**Channel**: Test Channel" in content
        assert "whisper (large-v3 / grc)" in content


def test_format_output_json_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = format_output(
            _sample_segments(), _sample_metadata(), "json", tmpdir,
            model="large-v3", prompt_profile="grc", method="whisper", language="en",
        )
        assert len(paths) == 1
        assert paths[0].endswith(".json")
        data = json.loads(open(paths[0]).read())
        assert data["video_id"] == "test123"
        assert data["transcription_method"] == "whisper"
        assert data["model"] == "large-v3"
        assert len(data["segments"]) == 2
        assert data["word_count"] == 6


def test_format_output_both_creates_two_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = format_output(
            _sample_segments(), _sample_metadata(), "both", tmpdir,
            model="large-v3", prompt_profile="general", method="whisper", language="en",
        )
        assert len(paths) == 2
        extensions = {os.path.splitext(p)[1] for p in paths}
        assert extensions == {".md", ".json"}


def test_format_output_youtube_subs_string_input():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = format_output(
            "Hello world. This is a test.", _sample_metadata(), "json", tmpdir,
            model=None, prompt_profile=None, method="youtube_subs", language="en",
        )
        data = json.loads(open(paths[0]).read())
        assert data["transcription_method"] == "youtube_subs"
        assert data["segments"] is None
        assert data["model"] is None


def test_format_output_md_general_prompt_no_slash():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = format_output(
            _sample_segments(), _sample_metadata(), "md", tmpdir,
            model="large-v3", prompt_profile="general", method="whisper", language="en",
        )
        content = open(paths[0]).read()
        assert "whisper (large-v3)" in content
        assert "/ general" not in content


def test_format_markdown_with_speakers(tmp_path):
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Hello.", "speaker": "Speaker 1"},
        {"start": 2.0, "end": 4.0, "text": "Welcome.", "speaker": "Speaker 1"},
        {"start": 4.0, "end": 6.0, "text": "Thanks!", "speaker": "Speaker 2"},
        {"start": 6.0, "end": 8.0, "text": "Glad to be here.", "speaker": "Speaker 2"},
    ]
    metadata = {
        "video_id": "abc123", "title": "T", "channel": "C",
        "upload_date": "20260101", "duration": 8, "url": "u",
    }
    paths = format_output(
        segments, metadata, "md", str(tmp_path),
        model="small", prompt_profile="general", method="whisper", language="en",
    )
    content = open(paths[0], encoding="utf-8").read()
    assert "**Speaker 1:**" in content
    assert "**Speaker 2:**" in content
    s1_idx = content.index("**Speaker 1:**")
    s2_idx = content.index("**Speaker 2:**")
    s1_block = content[s1_idx:s2_idx]
    assert "Hello." in s1_block
    assert "Welcome." in s1_block


def test_format_markdown_without_speakers_unchanged(tmp_path):
    """Non-diarized output (speaker=None) must not show speaker labels."""
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Hello world.", "speaker": None},
        {"start": 2.0, "end": 4.0, "text": "Second sentence.", "speaker": None},
    ]
    metadata = {
        "video_id": "abc123", "title": "T", "channel": "C",
        "upload_date": "20260101", "duration": 4, "url": "u",
    }
    paths = format_output(
        segments, metadata, "md", str(tmp_path),
        model="small", prompt_profile="general", method="whisper", language="en",
    )
    content = open(paths[0], encoding="utf-8").read()
    assert "**Speaker" not in content


def test_format_json_includes_speaker_field(tmp_path):
    segments = [
        {"start": 0.0, "end": 2.0, "text": "A.", "speaker": "Speaker 1"},
        {"start": 2.0, "end": 4.0, "text": "B.", "speaker": "Speaker 2"},
    ]
    metadata = {
        "video_id": "abc123", "title": "T", "channel": "C",
        "upload_date": "20260101", "duration": 4, "url": "u",
    }
    paths = format_output(
        segments, metadata, "json", str(tmp_path),
        model="small", prompt_profile="general", method="whisper", language="en",
    )
    data = json.load(open(paths[0], encoding="utf-8"))
    assert data["segments"][0]["speaker"] == "Speaker 1"
    assert data["segments"][1]["speaker"] == "Speaker 2"
    assert data["speakers"] == ["Speaker 1", "Speaker 2"]


def test_format_json_includes_config_block(tmp_path):
    segments = [{"start": 0.0, "end": 2.0, "text": "A.", "speaker": None}]
    metadata = {
        "video_id": "abc123", "title": "T", "channel": "C",
        "upload_date": "20260101", "duration": 2, "url": "https://yt/abc123",
    }
    paths = format_output(
        segments, metadata, "json", str(tmp_path),
        model="small", prompt_profile="grc", method="whisper", language="en",
        config={"url": "https://yt/abc123", "model": "small", "language": "en",
                "prompt_profile": "grc", "diarize": False, "output_format": "json"},
    )
    data = json.load(open(paths[0], encoding="utf-8"))
    assert "config" in data
    assert data["config"]["url"] == "https://yt/abc123"
    assert data["config"]["prompt_profile"] == "grc"
    assert data["config"]["diarize"] is False


def test_format_json_speakers_list_empty_when_no_diarize(tmp_path):
    segments = [{"start": 0.0, "end": 2.0, "text": "A.", "speaker": None}]
    metadata = {
        "video_id": "abc123", "title": "T", "channel": "C",
        "upload_date": "20260101", "duration": 2, "url": "u",
    }
    paths = format_output(
        segments, metadata, "json", str(tmp_path),
        model="small", prompt_profile="general", method="whisper", language="en",
    )
    data = json.load(open(paths[0], encoding="utf-8"))
    assert data.get("speakers") == []
