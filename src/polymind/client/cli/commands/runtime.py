from pathlib import Path

import typer

from polymind.core.hardware.loader import load_hardware_profile
from polymind.core.model.registry import ModelRegistry
from polymind.core.model.utils import format_size
from polymind.core.paths import runtime_path
from polymind.core.runtime.artifact import load_runtime_config, write_runtime_config
from polymind.core.runtime.config import default_runtime_config
from polymind.core.runtime.optimizer import optimize_config
from polymind.core.runtime.runner import RuntimeRunner

app = typer.Typer()


@app.command("run")
def run(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Model ID (number), filename, or repository ID of the installed model to run.",
    ),
) -> None:
    """Run a local model for interactive testing.

    Loads a downloaded GGUF model using llama.cpp and starts an
    interactive chat session. Uses optimized runtime settings
    from runtime.yaml if available.

    Examples:

        polymind runtime run -m 1

        polymind runtime run -m my-model-filename.gguf
    """

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
        (item for item in models if str(item.id) == model),
        None,
    )

    # Then try filename.
    if selected is None:
        selected = next(
            (item for item in models if item.filename == model),
            None,
        )

    # Finally try repository ID.
    if selected is None:
        selected = next(
            (item for item in models if item.repo_id == model),
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

    # Load optimized config from runtime.yaml, fall back to defaults
    config = load_runtime_config(str(selected.id))

    if config is None:
        config = default_runtime_config(
            str(selected.id),
            model_size_bytes=selected.size_bytes,
        )
        typer.echo(
            f"Warning: No optimized config found. "
            f"Run 'polymind runtime optimize -m {selected.id}' "
            f"for best performance.",
        )
        typer.echo()

    typer.echo(f"Loading model: {selected.filename}")
    typer.echo(f"GPU layers: {config.gpu_layers}")
    typer.echo(f"Context: {config.context_size}")
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
        help="Model ID (number) of the installed model to benchmark and optimize.",
    ),
    all_models: bool = typer.Option(
        False,
        "--all",
        help="Benchmark and optimize all installed models at once.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed per-configuration benchmark results during optimization.",
    ),
) -> None:
    """Benchmark and find optimal runtime settings.

    Runs actual inference benchmarks against installed models to
    determine the best GPU layer offloading, thread count, context
    size, and batch size for your hardware. Results are saved to
    .polymind/runtime.yaml for use by the run command.

    Examples:

        polymind runtime optimize -m 1

        polymind runtime optimize --all --verbose

        polymind runtime optimize -m 2 -v
    """

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

    # Load hardware profile
    try:
        hardware = load_hardware_profile()
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(
            f"Error: {exc}",
            err=True,
        )
        raise typer.Exit(code=1)

    registry = ModelRegistry()
    models = registry.load()

    if not models:
        typer.echo("No models installed.")
        typer.echo()
        typer.echo("Use 'polymind model search <query>' to find models.")
        raise typer.Exit(code=1)

    if all_models:
        selected_models = models
    else:
        selected_models = [item for item in models if item.id == model]

        if not selected_models:
            typer.echo(
                f"Model not found: {model}",
                err=True,
            )
            raise typer.Exit(code=1)

    typer.echo("Polymind Runtime Optimizer")
    typer.echo("=" * 60)
    typer.echo()

    # Show hardware summary
    typer.echo("Hardware:")
    typer.echo(f"  CPU:     {hardware.cpu.model}")
    typer.echo(
        f"  Cores:   {hardware.cpu.physical_cores} physical / {hardware.cpu.logical_cores} logical"
    )
    typer.echo(f"  RAM:     {format_size(hardware.memory.total_bytes)}")

    selected_gpus = [
        gpu for gpu in hardware.gpus if gpu.selection.enabled and gpu.compute.llama_cpp_usable
    ]

    if selected_gpus:
        for gpu in selected_gpus:
            vram = gpu.memory.available_bytes or gpu.memory.total_bytes
            typer.echo(f"  GPU:     {gpu.model} ({format_size(vram)})")
    else:
        typer.echo("  GPU:     None (CPU only)")

    typer.echo()

    # Optimize each model
    configs_written = 0
    artifact = runtime_path()  # Default path

    for installed_model in selected_models:
        model_path = Path(installed_model.local_path)

        if not model_path.exists():
            typer.echo(f"Skipping {installed_model.filename}: file not found")
            continue

        typer.echo(f"Optimizing: {installed_model.filename}")
        typer.echo(f"Size: {format_size(installed_model.size_bytes)}")
        typer.echo()

        typer.echo("Running adaptive benchmark (progressive search)...")
        typer.echo()

        def on_progress(msg: str, current: int, total: int) -> None:
            typer.echo(f"  [{current}/{total}] {msg}")

        config, bench_summary = optimize_config(
            model_id=str(installed_model.id),
            model_size_bytes=installed_model.size_bytes,
            model_path=model_path,
            hardware=hardware,
            on_progress=on_progress,
        )

        # Add benchmark metadata to config
        if bench_summary:
            config.benchmark = {
                "generation_tps": round(bench_summary.eval_tokens_per_sec_median, 1),
                "prompt_tps": round(bench_summary.prompt_tokens_per_sec_median, 1),
                "runs": bench_summary.run_count,
            }

        artifact = write_runtime_config(config)
        configs_written += 1

        typer.echo()
        typer.echo("Best configuration:")
        typer.echo(f"  GPU layers:  {config.gpu_layers}")
        typer.echo(f"  Threads:     {config.threads}")
        typer.echo(f"  Context:     {config.context_size}")
        typer.echo(f"  Batch:       {config.batch_size}")
        if bench_summary and bench_summary.success:
            typer.echo(f"  Gen speed:   {bench_summary.eval_tokens_per_sec_median:.1f} tok/s")
            typer.echo(f"  Prompt speed:{bench_summary.prompt_tokens_per_sec_median:.1f} tok/s")
        elif bench_summary is None:
            typer.echo()
            typer.echo("  Note: All benchmarks failed. Config is a conservative fallback.")
            typer.echo(f"  Try: polymind config set runtime.{installed_model.id}.gpu_layers 0")
        typer.echo()

    typer.echo("=" * 60)
    typer.echo(f"Wrote {configs_written} config(s) to: {artifact}")
