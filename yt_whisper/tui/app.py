"""Textual TUI for yt-whisper: Home, Run, and Preview screens."""

import os

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Header, Footer, Input, Select, Checkbox, Button, Label, ListView, ListItem,
    RadioSet, RadioButton, Static,
)

from yt_whisper.runner import RunConfig
from yt_whisper.tui.history import list_history, load_config_for_rerun, delete_run


MODEL_CHOICES = [("tiny", "tiny"), ("base", "base"), ("small", "small"),
                 ("medium", "medium"), ("large-v3", "large-v3")]
PROFILE_CHOICES = [("general", "general"), ("grc", "grc"), ("infosec", "infosec")]


class HomeScreen(Screen):
    """Two-column: history list (left) + new-transcription form (right)."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "rerun", "Re-run selected"),
        ("d", "delete", "Delete selected"),
        ("p", "preview", "Preview selected"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="history-pane"):
                yield Label("History")
                yield ListView(id="history-list")
            with Vertical(id="form-pane"):
                yield Label("New Transcription")
                yield Label("URL:")
                yield Input(placeholder="https://youtube.com/watch?v=...", id="url-input")
                yield Label("Model:")
                yield Select(MODEL_CHOICES, value="large-v3", id="model-select")
                yield Label("Language:")
                yield Input(value="en", id="language-input")
                yield Label("Profile:")
                yield Select(PROFILE_CHOICES, value="general", id="profile-select")
                yield Checkbox("Diarize (requires setup -- see README)", id="diarize-toggle")
                yield Label("Speakers (optional, diarize only):")
                yield Input(placeholder="auto", id="speakers-input")
                yield Label("Format:")
                with RadioSet(id="format-radio"):
                    yield RadioButton("both", value=True, id="fmt-both")
                    yield RadioButton("md", id="fmt-md")
                    yield RadioButton("json", id="fmt-json")
                yield Button("Run", id="run-btn", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_history()

    def refresh_history(self) -> None:
        lv = self.query_one("#history-list", ListView)
        lv.clear()
        runs = list_history(self.app.output_dir)
        for run in runs:
            marker = "*" if run["diarize"] else " "
            label = f"{marker} {run['title'][:40]}"
            lv.append(ListItem(Label(label), id=f"hist-{run['video_id']}"))
        self.app._history_cache = runs

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-btn":
            self.action_run()

    def action_run(self) -> None:
        cfg = self.app.build_runconfig()
        if not cfg.url:
            self.app.bell()
            return
        self.app.start_run(cfg)

    def action_rerun(self) -> None:
        idx = self.query_one("#history-list", ListView).index
        if idx is None or idx >= len(self.app._history_cache):
            return
        entry = self.app._history_cache[idx]
        stored = load_config_for_rerun(entry)
        self.query_one("#url-input", Input).value = stored.get("url", "")
        self.query_one("#model-select", Select).value = stored.get("model") or "large-v3"
        self.query_one("#language-input", Input).value = stored.get("language", "en")
        self.query_one("#profile-select", Select).value = stored.get("prompt_profile") or "general"
        self.query_one("#diarize-toggle", Checkbox).value = bool(stored.get("diarize"))

    def action_delete(self) -> None:
        idx = self.query_one("#history-list", ListView).index
        if idx is None or idx >= len(self.app._history_cache):
            return
        delete_run(self.app._history_cache[idx])
        self.refresh_history()

    def action_preview(self) -> None:
        # Preview screen implemented in Task 12
        self.app.bell()


class YtWhisperApp(App):
    """Main Textual application."""

    CSS = """
    #history-pane { width: 40%; border: tall $primary; padding: 1; }
    #form-pane { width: 60%; border: tall $primary; padding: 1; }
    #history-list { height: 1fr; }
    """

    def __init__(self, output_dir: str = "./transcripts"):
        super().__init__()
        self.output_dir = output_dir
        self._history_cache = []
        self._active_run_screen = None

    def _get_dom_base(self):
        """Override so app.query_one() searches the active screen, not the default."""
        return self.screen

    def on_mount(self) -> None:
        self.push_screen(HomeScreen())

    def build_runconfig(self) -> RunConfig:
        url = self.query_one("#url-input", Input).value.strip()
        model = self.query_one("#model-select", Select).value or "large-v3"
        language = self.query_one("#language-input", Input).value.strip() or "en"
        profile = self.query_one("#profile-select", Select).value or "general"
        diarize = self.query_one("#diarize-toggle", Checkbox).value
        speakers_raw = self.query_one("#speakers-input", Input).value.strip()
        num_speakers = int(speakers_raw) if speakers_raw.isdigit() else None
        fmt_set = self.query_one("#format-radio", RadioSet)
        if fmt_set.pressed_button and fmt_set.pressed_button.id:
            fmt = fmt_set.pressed_button.id.replace("fmt-", "")
        else:
            fmt = "both"
        return RunConfig(
            url=url, model=model, language=language,
            prompt_profile=profile, diarize=diarize,
            num_speakers=num_speakers, output_format=fmt,
            output_dir=self.output_dir,
        )

    def start_run(self, cfg: RunConfig) -> None:
        # Implemented fully in Task 11. For now, just bell.
        self.bell()

    # Stub callbacks so TuiListener doesn't crash during partial test runs
    def tui_on_phase(self, phase, status): pass
    def tui_on_progress(self, phase, pct): pass
    def tui_on_segment(self, segment): pass
    def tui_on_relabel(self, segments): pass
    def tui_on_log(self, level, msg): pass
    def tui_on_done(self, result): pass
    def tui_on_error(self, msg): pass
