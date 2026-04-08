"""Bridge runner events into Textual's message queue from the worker thread."""

from yt_whisper.runner import Listener


class TuiListener(Listener):
    """Forwards runner events into a Textual App via call_from_thread.

    Expects the app to expose methods matching the on_* callbacks.
    """

    def __init__(self, app):
        self.app = app

    def on_phase(self, phase, status):
        self.app.call_from_thread(self.app.tui_on_phase, phase, status)

    def on_progress(self, phase, pct):
        self.app.call_from_thread(self.app.tui_on_progress, phase, pct)

    def on_segment(self, segment):
        self.app.call_from_thread(self.app.tui_on_segment, dict(segment))

    def on_segments_relabeled(self, segments):
        self.app.call_from_thread(
            self.app.tui_on_relabel, [dict(s) for s in segments]
        )

    def on_log(self, level, msg):
        self.app.call_from_thread(self.app.tui_on_log, level, msg)

    def on_done(self, result):
        self.app.call_from_thread(self.app.tui_on_done, result)

    def on_error(self, exc):
        self.app.call_from_thread(self.app.tui_on_error, repr(exc))
