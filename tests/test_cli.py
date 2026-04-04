from yt_whisper.cli import build_parser, validate_word_count


def test_parser_required_url():
    parser = build_parser()
    args = parser.parse_args(["https://youtube.com/watch?v=test"])
    assert args.url == "https://youtube.com/watch?v=test"


def test_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["https://youtube.com/watch?v=test"])
    assert args.prompt == "general"
    assert args.force_whisper is False
    assert args.output_dir == "./transcripts"
    assert args.model == "large-v3"
    assert args.output_format == "both"
    assert args.language == "en"
    assert args.verbose is False


def test_parser_all_args():
    parser = build_parser()
    args = parser.parse_args([
        "https://youtube.com/watch?v=test",
        "--prompt", "grc",
        "--force-whisper",
        "--output-dir", "/tmp/out",
        "--model", "medium",
        "--format", "json",
        "--language", "es",
        "--verbose",
    ])
    assert args.prompt == "grc"
    assert args.force_whisper is True
    assert args.output_dir == "/tmp/out"
    assert args.model == "medium"
    assert args.output_format == "json"
    assert args.language == "es"
    assert args.verbose is True


def test_validate_word_count_normal(capsys):
    wpm = validate_word_count(1500, 600)  # 150 wpm, 10 min
    captured = capsys.readouterr()
    assert "Warning" not in captured.out
    assert wpm == 150.0


def test_validate_word_count_low(capsys):
    wpm = validate_word_count(200, 600)  # 20 wpm
    captured = capsys.readouterr()
    assert "Low word count" in captured.out
    assert wpm is not None


def test_validate_word_count_high(capsys):
    wpm = validate_word_count(5000, 600)  # 500 wpm
    captured = capsys.readouterr()
    assert "High word count" in captured.out
    assert wpm is not None


def test_validate_word_count_short_video(capsys):
    wpm = validate_word_count(10, 20)  # 20 seconds
    captured = capsys.readouterr()
    assert "too short" in captured.out
    assert wpm is None


def test_validate_word_count_boundary_30s(capsys):
    wpm = validate_word_count(75, 30)  # exactly 30 seconds, 150 wpm
    captured = capsys.readouterr()
    assert "too short" not in captured.out
    assert wpm is not None
