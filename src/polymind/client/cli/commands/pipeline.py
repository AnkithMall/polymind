"""Pipeline CLI — prompt decomposition and execution with verbose mode and logging."""

import json
import time

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from polymind.core.config import load_config
from polymind.core.model.registry import ModelRegistry
from polymind.core.pipeline.log import PipelineLogWriter, get_pipeline_logger
from polymind.core.pipeline.orchestrator import Pipeline
from polymind.core.pipeline.selector import suggest_models
from polymind.core.pipeline.types import PipelineConfig, TaskStatus

app = typer.Typer()
console = Console()


def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.0f} KB"
    if size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    return f"{size_bytes / 1024**3:.1f} GB"


def _verbose_progress(event: str, task, start_time: float = 0.0) -> None:
    """Rich verbose progress handler showing all pipeline details."""

    if event == "artifact_check":
        artifacts = task.metadata
        console.print()
        console.print("[bold magenta]═══ Artifact Detection ═══[/]")
        for name, info in artifacts.items():
            icon = "✓" if info != "missing" else "✗"
            color = "green" if info != "missing" else "red"
            console.print(f"  [{color}]{icon}[/] {name}: {info}")

    elif event == "analyze":
        is_simple = task.metadata.get("is_simple", "false")
        word_count = task.metadata.get("word_count", "?")
        console.print()
        console.print("[bold magenta]═══ Complexity Analysis ═══[/]")
        console.print(f"  Words: {word_count}")
        console.print(f"  Simple: {is_simple}")
        if is_simple == "true":
            console.print("  [dim]→ Skipping decomposition, single task mode[/]")
        else:
            console.print("  [dim]→ Decomposing into subtasks[/]")

    elif event == "decompose_start":
        console.print()
        console.print("[bold cyan]═══ Decomposition ═══[/]")
        console.print("  Loading decomposer model...")

    elif event == "decompose_done":
        tasks = task.metadata.get("tasks", [])
        if tasks:
            console.print(f"  [green]✓ Decomposed into {len(tasks)} task(s)[/]")
            console.print()
            table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
            table.add_column("Task", style="cyan", width=8)
            table.add_column("Domain", style="magenta", width=14)
            table.add_column("Prompt", width=50)
            for t in tasks:
                table.add_row(
                    t.get("id", "?"),
                    t.get("domain", "?"),
                    t.get("prompt", "?")[:50],
                )
            console.print(table)

    elif event == "assign_start":
        console.print()
        console.print("[bold cyan]═══ Model Assignment ═══[/]")

    elif event == "assign_done":
        assignments = task.metadata.get("assignments", {})
        if assignments:
            table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
            table.add_column("Task", style="cyan", width=8)
            table.add_column("Domain", style="magenta", width=14)
            table.add_column("Model", style="yellow", width=40)
            table.add_column("Confidence", justify="right", width=10)
            table.add_column("Reason", width=30)
            for tid, a in assignments.items():
                table.add_row(
                    tid,
                    a.get("domain", "?"),
                    a.get("model_id", "?"),
                    a.get("confidence", "?"),
                    a.get("reason", "?"),
                )
            console.print(table)

    elif event == "schedule":
        groups = task.metadata.get("groups", [])
        loads = task.metadata.get("model_loads", 0)
        console.print()
        console.print("[bold cyan]═══ Execution Plan ═══[/]")
        console.print(f"  Model loads needed: {loads}")
        console.print(f"  Task groups: {len(groups)}")
        for i, g in enumerate(groups, 1):
            model = g.get("model_id", "?")
            tids = g.get("task_ids", [])
            console.print(f"  [yellow]Group {i}[/]: {model} → {', '.join(tids)}")

    elif event == "model_load":
        model_id = task.metadata.get("model_id", "?")
        model_file = task.metadata.get("model_file", "?")
        gpu = task.metadata.get("gpu_layers", "?")
        threads = task.metadata.get("threads", "?")
        ctx = task.metadata.get("context_size", "?")
        size = task.metadata.get("size_bytes", 0)
        console.print()
        console.print(f"  [yellow]▶ Loading model:[/] {model_id}")
        console.print(f"    File: {model_file}")
        console.print(f"    Size: {_format_size(int(size))}")
        console.print(f"    Config: gpu={gpu}, threads={threads}, ctx={ctx}")

    elif event == "model_loaded":
        model_id = task.metadata.get("model_id", "?")
        console.print(f"    [green]✓ Model loaded: {model_id}[/]")

    elif event == "model_load_failed":
        model_id = task.metadata.get("model_id", "?")
        reason = task.metadata.get("reason", "unknown")
        console.print(f"    [red]✗ Failed to load {model_id}: {reason}[/]")

    elif event == "task_start":
        console.print()
        console.print(
            f"  [bold yellow]▶ {task.id}[/] "
            f"[dim]{task.domain}[/] "
            f"[cyan]{task.prompt[:70]}[/]"
        )

    elif event == "task_done":
        elapsed = task.metadata.get("elapsed_s", "")
        tokens = task.metadata.get("tokens", "")
        speed = task.metadata.get("tok_per_s", "")

        if task.status == TaskStatus.COMPLETED:
            timing_str = f" ({elapsed}s, {tokens}tok, {speed} tok/s)" if elapsed else ""
            console.print(
                f"    [green]✓ {task.id}[/] "
                f"[dim]score={task.score:.0%}[/]{timing_str}"
            )
        elif task.status == TaskStatus.FAILED:
            console.print(f"    [red]✗ {task.id}[/] {task.error}")

    elif event == "regenerate_start":
        console.print()
        console.print("[bold cyan]═══ Regeneration ═══[/]")
        console.print("  Combining task results...")

    elif event == "regenerate_done":
        console.print("  [green]✓ Final response generated[/]")


@app.command("run")
def run_pipeline(
    prompt: str = typer.Argument(
        ...,
        help="User prompt to process through the full pipeline.",
    ),
    decomposer: str = typer.Option(
        "",
        "--decomposer",
        help="Model ID for decomposition step (default: auto-select best).",
    ),
    generator: str = typer.Option(
        "",
        "--generator",
        help="Model ID for generation step (default: auto-select best).",
    ),
    regenerator: str = typer.Option(
        "",
        "--regenerator",
        help="Model ID for regeneration/synthesis step (default: auto-select).",
    ),
    judge: str = typer.Option(
        "",
        "--judge",
        help="Model ID for judging/evaluation step (default: auto-select).",
    ),
    no_regenerate: bool = typer.Option(
        False,
        "--no-regenerate",
        help="Skip regeneration — return raw task results.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed pipeline execution (tasks, models, scores).",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        help="Output full result as JSON.",
    ),
) -> None:
    """Run the full pipeline: decompose → assign → execute → regenerate.

    Analyzes prompt complexity, breaks it into subtasks, assigns the best
    model per task, executes, and synthesizes a final response.

    Examples:

        polymind pipeline run "Build a REST API with auth and tests"

        polymind pipeline run "What is quicksort?" -v

        polymind pipeline run "Write a sorting function" --generator 2

        polymind pipeline run "Debug this code" --json
    """
    # Load pipeline settings from polymind.yaml
    polymind_cfg = load_config().get("pipeline", {})

    config = PipelineConfig(
        decomposer_model=decomposer,
        generator_model=generator,
        regenerator_model=regenerator if not no_regenerate else "",
        judge_model=judge,
        auto_select_models=True,
        max_concurrent=polymind_cfg.get("max_concurrent", 1),
        timeout_seconds=polymind_cfg.get("timeout_seconds", 120),
        max_retries=polymind_cfg.get("max_retries", 1),
        min_task_score=polymind_cfg.get("min_task_score", 0.5),
    )

    pipeline = Pipeline(config)
    start_time = time.time()

    if verbose:
        # Verbose: show everything on screen
        pipeline.on_progress(lambda e, t: _verbose_progress(e, t, start_time))
    else:
        # Non-verbose: write logs to file
        log_writer = PipelineLogWriter()
        pipeline.on_progress(log_writer.log_event)

    with console.status("[bold green]Running pipeline..."):
        result = pipeline.run(prompt)

    if output_json:
        typer.echo(json.dumps(result.to_dict(), indent=2))
        return

    # Display warnings
    if pipeline.warnings:
        console.print()
        for warn in pipeline.warnings:
            console.print(f"  [yellow]⚠ {warn}[/]")

    # Display response as rendered Markdown
    console.print()
    if result.response:
        try:
            md = Markdown(result.response)
            console.print(Panel(md, title="Response", border_style="green", expand=True))
        except Exception:
            console.print(Panel(result.response, title="Response", border_style="green"))
    else:
        console.print(Panel("[dim]No response generated.[/]", title="Response", border_style="red"))
    console.print()

    # Task summary
    table = Table(title="Task Summary")
    table.add_column("Task", style="cyan")
    table.add_column("Domain", style="magenta")
    table.add_column("Model", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Score", justify="right")

    for task in result.tasks:
        status_style = {
            TaskStatus.COMPLETED: "[green]done[/]",
            TaskStatus.FAILED: "[red]failed[/]",
            TaskStatus.RUNNING: "[yellow]running[/]",
            TaskStatus.PENDING: "[dim]pending[/]",
        }.get(task.status, task.status.value)

        table.add_row(
            task.id,
            task.domain,
            task.model_id or "-",
            status_style,
            f"{task.score:.0%}" if task.score > 0 else "-",
        )

    console.print(table)

    # Stats
    console.print()
    console.print(f"[dim]Models loaded: {result.model_loads}[/]")
    console.print(f"[dim]Total time: {result.total_time_ms:.0f}ms[/]")
    console.print(f"[dim]Overall score: {result.overall_score:.0%}[/]")

    if not verbose:
        logger = get_pipeline_logger()
        log_file = logger.handlers[0].baseFilename if logger.handlers else "?"
        console.print(f"[dim]Logs written to: {log_file}[/]")


@app.command("suggest")
def suggest_command() -> None:
    """Suggest the best models for each pipeline role.

    Shows top-3 model recommendations for decomposer, generator,
    regenerator, and judge roles based on confidence scores and
    model capabilities.

    Examples:

        polymind pipeline suggest
    """
    suggestions = suggest_models()

    for role, models in suggestions.items():
        if not models:
            continue

        console.print(f"\n[bold]{role.upper()} MODEL[/]")
        console.print("-" * 60)

        for i, m in enumerate(models[:3], 1):
            score = m["score"]
            reason = m["reason"]
            console.print(
                f"  {i}. [cyan]ID:{m['model_id']}[/] "
                f"{m['filename'][:40]:<40} "
                f"score={score:.1f} ({reason})"
            )


@app.command("status")
def pipeline_status() -> None:
    """Show installed models and pipeline readiness.

    Displays all registered models, their sizes, and quantization
    levels. Also shows confidence scores if computed.

    Examples:

        polymind pipeline status
    """
    registry = ModelRegistry()
    models = registry.load()

    console.print("[bold]Pipeline Status[/]")
    console.print()

    if not models:
        console.print("[red]No models installed.[/]")
        console.print("Run: polymind model download <repo>")
        return

    table = Table(title="Installed Models")
    table.add_column("ID", style="cyan")
    table.add_column("Filename", style="yellow")
    table.add_column("Size", justify="right")
    table.add_column("Quant", style="magenta")

    for m in models:
        from polymind.core.model.utils import format_size

        table.add_row(
            str(m.id),
            m.filename[:50],
            format_size(m.size_bytes),
            m.quantization or "-",
        )

    console.print(table)

    # Show confidence data
    from polymind.core.confidence.artifact import load_confidence

    confidence = load_confidence()

    if confidence:
        console.print()
        console.print("[bold]Confidence Scores[/]")
        for mid, conf in confidence.items():
            console.print(
                f"  Model {mid}: overall={conf.overall_score:.1f}% best={conf.best_domain}"
            )
    else:
        console.print()
        console.print("[dim]No confidence scores computed yet.[/]")
        console.print("[dim]Run: polymind confidence compute[/]")


@app.command("quick")
def quick_run(
    prompt: str = typer.Argument(
        ...,
        help="Prompt to send directly to a single model (no decomposition).",
    ),
    model_id: str = typer.Option(
        "",
        "--model",
        "-m",
        help="Model ID to use (default: largest installed model).",
    ),
) -> None:
    """Send a prompt directly to a single model (no pipeline).

    Skips decomposition and regeneration. Best for simple prompts
    or when you want raw model output without orchestration.

    Examples:

        polymind pipeline quick "What is 2+2?"

        polymind pipeline quick "Hello, how are you?" -m 2
    """
    registry = ModelRegistry()
    models = registry.load()

    if not models:
        console.print("[red]No models installed.[/]")
        raise typer.Exit(code=1)

    if model_id:
        model = None
        for m in models:
            if str(m.id) == model_id:
                model = m
                break
        if model is None:
            console.print(f"[red]Model not found: {model_id}[/]")
            raise typer.Exit(code=1)
    else:
        model = max(models, key=lambda m: m.size_bytes)

    from pathlib import Path

    from polymind.core.runtime.config import default_runtime_config

    model_path = Path(model.local_path)
    if not model_path.exists():
        console.print(f"[red]Model file not found: {model.local_path}[/]")
        raise typer.Exit(code=1)

    runtime = default_runtime_config(str(model.id), model.size_bytes)

    console.print(f"[dim]Using: {model.filename}[/]")

    from llama_cpp import Llama

    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=runtime.gpu_layers,
        n_threads=runtime.threads,
        n_ctx=runtime.context_size,
        n_batch=runtime.batch_size,
        verbose=False,
    )

    try:
        output = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )
        content = output["choices"][0]["message"]["content"] or ""
        try:
            md = Markdown(content.strip())
            console.print(Panel(md, title="Response", border_style="green", expand=True))
        except Exception:
            console.print(Panel(content.strip(), title="Response", border_style="green"))
    finally:
        del llm
