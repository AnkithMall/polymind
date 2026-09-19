"""TUI command — placeholder for future text user interface."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True)
def tui() -> None:
    """Launch the Polymind TUI (Text User Interface).

    The TUI is not yet available in this release. Use the CLI
    commands directly for all functionality.
    """
    console.print("[yellow]The TUI is not yet available in this release.[/]")
    console.print()
    console.print("Use these CLI commands instead:")
    console.print()
    console.print("  polymind model list          — List installed models")
    console.print("  polymind hardware scan       — Scan system hardware")
    console.print("  polymind runtime optimize    — Optimize runtime settings")
    console.print("  polymind confidence compute  — Evaluate model capabilities")
    console.print("  polymind pipeline run        — Run multi-model pipeline")
    console.print("  polymind category            — Manage scoring categories")
    console.print("  polymind capability show     — View model capability profiles")
    console.print("  polymind doctor              — System diagnostics")
    console.print()
    console.print("[dim]Run 'polymind <command> --help' for details on any command.[/]")
