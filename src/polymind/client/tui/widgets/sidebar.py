from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static


class Sidebar(Vertical):
    """Navigation sidebar with buttons for each screen."""

    def compose(self) -> ComposeResult:
        yield Static("[b]Polymind[/b]", classes="nav-button")
        yield Static("─" * 20)
        yield Button("Home [1]", id="nav-home", variant="default", classes="nav-button")
        yield Button("Models [2]", id="nav-model", variant="default", classes="nav-button")
        yield Button("Config [3]", id="nav-config", variant="default", classes="nav-button")
        yield Button("Hardware [4]", id="nav-hardware", variant="default", classes="nav-button")
        yield Button("Runtime [5]", id="nav-runtime", variant="default", classes="nav-button")
        yield Static("")
        yield Static("[dim]Press q to quit[/dim]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        app = self.app
        match event.button.id:
            case "nav-home":
                app.switch_screen("home")
            case "nav-model":
                app.switch_screen("model")
            case "nav-config":
                app.switch_screen("config")
            case "nav-hardware":
                app.switch_screen("hardware")
            case "nav-runtime":
                app.switch_screen("runtime")
