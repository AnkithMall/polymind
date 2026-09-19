from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Static


class HomeScreen(Widget):
    """Dashboard home screen showing overview of system status."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("[b]Polymind Dashboard[/b]", classes="title")
            yield Static("")
            yield Static("Welcome to Polymind - Local LLM Management")
            yield Static("")
            yield Static("[b]Quick Actions[/b]")
            yield Static("  [1] Models    - Search, download, and manage GGUF models")
            yield Static("  [2] Config    - View and edit configuration")
            yield Static("  [3] Hardware  - Scan and view system hardware")
            yield Static("  [4] Runtime   - Run and optimize model inference")
            yield Static("")
            yield Static("[b]System Status[/b]")
            yield Static(self._get_status_summary(), id="status-summary")

    def _get_status_summary(self) -> str:
        lines: list[str] = []
        try:
            from polymind.core.model.registry import ModelRegistry

            models = ModelRegistry().load()
            lines.append(f"  Installed models: {len(models)}")
        except Exception:
            lines.append("  Installed models: unavailable")

        try:
            from polymind.core.paths import hardware_path

            status = "available" if hardware_path().exists() else "not created"
            lines.append(f"  Hardware profile: {status}")
        except Exception:
            lines.append("  Hardware profile: unavailable")

        return "\n".join(lines)
