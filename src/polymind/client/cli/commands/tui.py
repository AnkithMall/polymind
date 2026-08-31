import typer

app = typer.Typer()


@app.callback(invoke_without_command=True)
def tui() -> None:
    """Launch the Polymind TUI (Text User Interface)."""
    from polymind.client.tui.app import PolymindApp

    PolymindApp().run()
