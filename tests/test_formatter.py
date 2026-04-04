from yt_whisper.formatter import format_duration, format_paragraphs


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
