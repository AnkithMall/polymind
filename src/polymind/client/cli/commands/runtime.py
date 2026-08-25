import typer

from polymind.core.model.registry import ModelRegistry
from polymind.core.runtime.config import default_runtime_config
from polymind.core.runtime.runner import RuntimeRunner

from pathlib import Path

from polymind.core.runtime.artifact import write_runtime_config

app = typer.Typer()


@app.command("run")
def run(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Model ID or model name to run.",
    ),
) -> None:
    """Run a local model for interactive testing."""

    registry = ModelRegistry()
    models = registry.load()

    if not models:
        typer.echo(
            "No models installed.",
            err=True,
        )
        raise typer.Exit(code=1)

    # First try model ID.
    selected = next(
        (
            item
            for item in models
            if str(item.id) == model
        ),
        None,
    )

    # Then try filename.
    if selected is None:
        selected = next(
            (
                item
                for item in models
                if item.filename == model
            ),
            None,
        )

    # Finally try repository ID.
    if selected is None:
        selected = next(
            (
                item
                for item in models
                if item.repo_id == model
            ),
            None,
        )

    if selected is None:
        typer.echo(
            f"Model not found: {model}",
            err=True,
        )
        raise typer.Exit(code=1)

    model_path = Path(selected.local_path)

    if not model_path.exists():
        typer.echo(
            f"Model file does not exist: {model_path}",
            err=True,
        )
        raise typer.Exit(code=1)

    config = default_runtime_config(
        str(selected.id)
    )

    typer.echo(
        f"Loading model: {selected.filename}"
    )
    typer.echo(
        f"GPU layers: {config.gpu_layers}"
    )
    typer.echo(
        f"Context: {config.context_size}"
    )
    typer.echo()

    runner = RuntimeRunner(
        config=config,
        model_path=model_path,
    )

    runner.chat()


@app.command("optimize")
def optimize(
    model: int | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Installed model ID to optimize.",
    ),
    all_models: bool = typer.Option(
        False,
        "--all",
        help="Optimize all installed models.",
    ),
) -> None:
    """Create optimized runtime artifacts."""

    if model is None and not all_models:
        typer.echo(
            "Error: specify --model/-m or --all.",
            err=True,
        )
        raise typer.Exit(code=1)

    if model is not None and all_models:
        typer.echo(
            "Error: use either --model/-m or --all, not both.",
            err=True,
        )
        raise typer.Exit(code=1)

    registry = ModelRegistry()
    models = registry.load()

    if not models:
        typer.echo("No models installed.")
        raise typer.Exit(code=1)

    if all_models:
        selected_models = models
    else:
        selected_models = [
            item
            for item in models
            if item.id == model
        ]

        if not selected_models:
            typer.echo(
                f"Model not found: {model}",
                err=True,
            )
            raise typer.Exit(code=1)

    for installed_model in selected_models:
        config = default_runtime_config(
            model_id=str(installed_model.id),
        )

        artifact = write_runtime_config(config)

        typer.echo(
            f"Optimized runtime for "
            f"{installed_model.repo_id}"
        )
        typer.echo(
            f"Artifact: {artifact}"
        )
        typer.echo(
            f"GPU layers: {config.gpu_layers}"
        )
        typer.echo(
            f"Threads: {config.threads}"
        )
        typer.echo(
            f"Context: {config.context_size}"
        )
        typer.echo(
            f"Batch size: {config.batch_size}"
        )
