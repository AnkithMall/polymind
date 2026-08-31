from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical

from polymind.client.tui.screens.config import ConfigScreen
from polymind.client.tui.screens.hardware import HardwareScreen
from polymind.client.tui.screens.home import HomeScreen
from polymind.client.tui.screens.model import ModelScreen
from polymind.client.tui.screens.runtime import RuntimeScreen
from polymind.client.tui.widgets.sidebar import Sidebar


class PolymindApp(App):
    """Main Polymind TUI application."""

    TITLE = "Polymind"
    SUB_TITLE = "Local LLM Management"

    CSS = """
    Screen {
        layout: horizontal;
    }

    #sidebar {
        width: 24;
        min-width: 24;
        max-width: 24;
        height: 100%;
        background: $surface;
        border-right: tall $primary;
        padding: 1;
    }

    #main-content {
        width: 1fr;
        height: 100%;
    }

    .screen-page {
        width: 100%;
        height: 100%;
        padding: 1 2;
        display: none;
    }

    .screen-page-active {
        display: block;
    }

    .hidden {
        display: none;
    }

    .button-row {
        height: auto;
    }

    .button-row Button {
        min-width: 14;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("1", "nav_home", "Home"),
        Binding("2", "nav_model", "Models"),
        Binding("3", "nav_config", "Config"),
        Binding("4", "nav_hardware", "Hardware"),
        Binding("5", "nav_runtime", "Runtime"),
    ]

    def compose(self) -> ComposeResult:
        yield Sidebar(id="sidebar")
        with Vertical(id="main-content"):
            yield HomeScreen(classes="screen-page screen-page-active", id="page-home")
            yield ModelScreen(classes="screen-page", id="page-model")
            yield ConfigScreen(classes="screen-page", id="page-config")
            yield HardwareScreen(classes="screen-page", id="page-hardware")
            yield RuntimeScreen(classes="screen-page", id="page-runtime")

    def switch_screen(self, screen_id: str) -> None:
        for page in self.query(".screen-page"):
            page.remove_class("screen-page-active")
        target = self.query_one(f"#page-{screen_id}")
        target.add_class("screen-page-active")

    def action_nav_home(self) -> None:
        self.switch_screen("home")

    def action_nav_model(self) -> None:
        self.switch_screen("model")

    def action_nav_config(self) -> None:
        self.switch_screen("config")

    def action_nav_hardware(self) -> None:
        self.switch_screen("hardware")

    def action_nav_runtime(self) -> None:
        self.switch_screen("runtime")


def main() -> None:
    app = PolymindApp()
    app.run()


if __name__ == "__main__":
    main()
