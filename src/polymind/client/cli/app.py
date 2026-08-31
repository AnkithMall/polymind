import typer

from polymind.client.cli.commands import (
    capability,
    category,
    confidence,
    config,
    doctor,
    domain,
    hardware,
    model,
    pipeline,
    run,
    runtime,
    suite,
    tui,
)

app = typer.Typer(
    help="Polymind — hardware-aware local LLM toolkit.\n\n"
    "Manage models, optimize runtime, score confidence,\n"
    "and run multi-model pipelines on your local hardware.",
    add_completion=False,
)

app.add_typer(model.app, name="model", help="Search, download, and manage GGUF models.")
app.add_typer(hardware.app, name="hardware", help="Scan and display system hardware.")
app.add_typer(runtime.app, name="runtime", help="Benchmark and optimize model runtime settings.")
app.add_typer(pipeline.app, name="pipeline", help="Decompose prompts and run multi-model pipelines.")
app.add_typer(confidence.app, name="confidence", help="Compute and view model confidence scores.")
app.add_typer(domain.app, name="domain", help="Manage scoring domains (predefined + custom).")
app.add_typer(suite.app, name="suite", help="Manage test suites within domains.")
app.add_typer(config.app, name="config", help="View and edit polymind configuration.")
app.add_typer(run.app, name="run", help="Run commands (alias).")
app.add_typer(capability.app, name="capability", help="Hardware capability detection.")
app.add_typer(category.app, name="category", help="Model categories.")
app.add_typer(doctor.app, name="doctor", help="Diagnostics and health checks.")
app.add_typer(tui.app, name="tui", help="Launch the text user interface.")


def main() -> None:
    app()
