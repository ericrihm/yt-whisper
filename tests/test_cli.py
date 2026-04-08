from yt_whisper.cli import build_parser


def test_parser_required_url():
    parser = build_parser()
    args = parser.parse_args(["https://youtube.com/watch?v=test"])
    assert args.url == "https://youtube.com/watch?v=test"


def test_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["https://youtube.com/watch?v=test"])
    assert args.prompt_profile == "general"
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
    assert args.prompt_profile == "grc"
    assert args.force_whisper is True
    assert args.output_dir == "/tmp/out"
    assert args.model == "medium"
    assert args.output_format == "json"
    assert args.language == "es"
    assert args.verbose is True


def test_cli_builds_runconfig_from_args(monkeypatch, capsys, tmp_path):
    from yt_whisper.cli import main
    from yt_whisper.runner import RunConfig
    from unittest.mock import patch, MagicMock

    captured_cfg = {}

    def fake_run(cfg, listener, cancel_event=None):
        captured_cfg["cfg"] = cfg
        listener.on_done({
            "paths": [str(tmp_path / "abc.md")],
            "title": "T", "duration_formatted": "1:00",
            "word_count": 100, "wpm": 150.0, "method": "whisper",
        })
        return {"paths": [str(tmp_path / "abc.md")], "title": "T",
                "duration_formatted": "1:00", "word_count": 100,
                "wpm": 150.0, "method": "whisper"}

    argv = ["yt-whisper", "https://yt/abc", "--model", "small",
            "--diarize", "--speakers", "3", "--output-dir", str(tmp_path)]
    monkeypatch.setattr("sys.argv", argv)
    with patch("yt_whisper.cli.run", side_effect=fake_run):
        main()
    cfg = captured_cfg["cfg"]
    assert cfg.url == "https://yt/abc"
    assert cfg.model == "small"
    assert cfg.diarize is True
    assert cfg.num_speakers == 3
    assert cfg.output_dir == str(tmp_path)


def test_cli_no_args_launches_tui(monkeypatch):
    from yt_whisper.cli import main
    from unittest.mock import patch

    monkeypatch.setattr("sys.argv", ["yt-whisper"])
    with patch("yt_whisper.cli.launch_tui") as mock_tui:
        main()
    mock_tui.assert_called_once()


def test_cli_prints_final_summary(monkeypatch, capsys, tmp_path):
    from yt_whisper.cli import main
    from unittest.mock import patch

    def fake_run(cfg, listener, cancel_event=None):
        return {
            "paths": [str(tmp_path / "abc.md"), str(tmp_path / "abc.json")],
            "title": "Demo Talk",
            "duration_formatted": "1:23:45",
            "word_count": 12345,
            "wpm": 155.0,
            "method": "whisper",
        }

    monkeypatch.setattr("sys.argv", ["yt-whisper", "https://yt/abc"])
    with patch("yt_whisper.cli.run", side_effect=fake_run):
        main()
    out = capsys.readouterr().out
    assert "[OK]" in out
    assert "Demo Talk" in out
    assert "12345" in out
    assert "1:23:45" in out
