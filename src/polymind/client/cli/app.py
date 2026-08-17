import typer

from polymind.client.cli.commands import (
    capability,
    category,
    doctor,
    hardware,
    model,
    run,
    runtime,
    suite,
)

app = typer.Typer()

app.add_typer(capability.app, name="capability")
app.add_typer(category.app, name="category")
app.add_typer(doctor.app, name="doctor")
app.add_typer(hardware.app, name="hardware")
app.add_typer(model.app, name="model")
app.add_typer(run.app, name="run")
app.add_typer(runtime.app, name="runtime")
app.add_typer(suite.app, name="suite")


def main() -> None:
    app()
