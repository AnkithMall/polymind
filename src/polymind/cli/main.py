from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from polymind.core import (
    ALL_DOMAINS,
    AnalyzerPlan,
    DomainType,
    ExecutionSchedule,
    ExecutionStrategy,
    PipelineResult,
    SubtaskResult,
    analyze_prompt,
    build_schedule,
    execute_subtask,
    get_tasks_for_domain,
    load_ranks,
    resolve_model_string,
    run_benchmark,
    save_ranks,
    synthesize,
)
from polymind.core.config import Config

app = typer.Typer(
    name="polymind",
    help="Multi-Specialist LLM Orchestrator — CLI · TUI · Local-First · Hardware-Optimised",
    no_args_is_help=True,
)
console = Console()

CONFIG_PATH = Path("~/.polymind/config.yaml").expanduser()
RANKS_PATH = Path("~/.polymind/ranks.yaml").expanduser()


def _load_config() -> Config:
    return Config.from_yaml(CONFIG_PATH)


def _ensure_config() -> Config:
    if not CONFIG_PATH.exists():
        console.print(
            "[yellow]No config found. Run [bold]polymind config init[/] first.[/]"
        )
        raise typer.Exit(1)
    return _load_config()


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="The prompt to send"),
    review: bool = typer.Option(
        False, "--review", "-r", help="Review subtask plan before execution"
    ),
    profile: Optional[str] = typer.Option(
        None, "--profile", "-p", help="Routing profile name"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override model for all subtasks"
    ),
):
    """Run a prompt through the full PolyMind pipeline."""
    asyncio.run(_async_ask(prompt, review=review, profile=profile, model=model))


async def _async_ask(
    prompt: str,
    review: bool = False,
    profile: str | None = None,
    model: str | None = None,
):
    config = _ensure_config()

    router = config.router_model
    synthesizer_m = config.synthesizer_model or router

    with console.status("[bold green]Analyzing prompt...") as _:
        plan: AnalyzerPlan = await analyze_prompt(prompt, router_model=router)

    if review:
        console.print("\n[bold]Subtask Plan:[/]")
        for s in plan.subtasks:
            deps = f" (depends on: {', '.join(s.depends_on)})" if s.depends_on else ""
            console.print(
                f"  [cyan]{s.id}[/] [yellow]{s.domain.value}[/]: {s.prompt}{deps}"
            )
        if not Confirm.ask("\nProceed with execution?"):
            raise typer.Exit(0)

    rank_store = load_ranks(RANKS_PATH)
    schedule: ExecutionSchedule = build_schedule(
        plan,
        rank_store=rank_store,
        strategy=config.scheduler.strategy,
    )

    if model:
        for batch in schedule.batches:
            batch.model = model

    console.print(f"\n[bold]Schedule:[/] [dim]{schedule.strategy.value}[/]")
    for batch in schedule.batches:
        console.print(f"  [cyan]{batch.model}[/]: {', '.join(batch.subtask_ids)}")

    subtask_results: list[SubtaskResult] = []
    prior_outputs: dict[str, SubtaskResult] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task(
            "[green]Executing subtasks...", total=len(schedule.batches)
        )
        for batch in schedule.batches:
            for sid in batch.subtask_ids:
                subtask = next(s for s in plan.subtasks if s.id == sid)
                info = resolve_model_string(batch.model)
                result = await execute_subtask(
                    subtask,
                    model_ref=batch.model,
                    provider_info=info,
                )
                subtask_results.append(result)
                prior_outputs[sid] = result
            progress.update(task, advance=1)

    result = PipelineResult(
        original_prompt=prompt,
        subtask_results=subtask_results,
        schedule=schedule,
    )

    with console.status("[bold green]Synthesizing response...") as _:
        result = await synthesize(result, synthesizer_model=synthesizer_m)

    console.print()
    if result.synthesis:
        console.print(Panel(result.synthesis, title="Response", border_style="green"))
    else:
        console.print("[red]Synthesis failed.[/]")

    if config.verbose:
        _print_breakdown(result)


def _print_breakdown(result: PipelineResult):
    table = Table(title="Subtask Breakdown")
    table.add_column("Subtask", style="cyan")
    table.add_column("Model", style="yellow")
    table.add_column("Latency")
    table.add_column("Status")

    for r in result.subtask_results:
        lat = f"{r.latency_ms:.0f}ms" if r.latency_ms else "-"
        status = "[red]ERROR[/]" if r.error else "[green]OK[/]"
        table.add_row(r.subtask_id, r.model, lat, status)

    console.print(table)


@app.command()
def benchmark(
    models: list[str] = typer.Argument(
        ..., help="Models to benchmark (e.g. ollama/llama3.2:1b)"
    ),
    domains: Optional[list[str]] = typer.Option(
        None, "--domain", "-d", help="Domains to benchmark (default: all)"
    ),
):
    """Run benchmark tasks against models to build rankings."""
    config = _load_config()
    domain_list: list[DomainType] = (
        [DomainType(d) for d in domains] if domains else ALL_DOMAINS
    )
    judge = config.judge_model

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
    ) as progress:
        total_tasks = sum(len(get_tasks_for_domain(d)) for d in domain_list) * len(
            models
        )

        prog_task = progress.add_task("[green]Benchmarking...", total=total_tasks)

        def progress_callback(msg: str):
            progress.update(prog_task, description=f"[green]{msg}")

        store = asyncio.run(
            run_benchmark(
                models=models,
                domains=domain_list,
                judge_model=judge,
                progress_callback=progress_callback,
            )
        )

    save_ranks(store, RANKS_PATH)
    console.print(f"\n[green]Benchmark complete![/] Saved to [bold]{RANKS_PATH}[/]")

    _print_ranks(store)


@app.command()
def ranks():
    """Display current model rankings."""
    store = load_ranks(RANKS_PATH)
    if not store.entries:
        console.print(
            "[yellow]No rankings found. Run [bold]polymind benchmark[/] first.[/]"
        )
        raise typer.Exit(0)

    if store.is_stale():
        console.print("[yellow]Warning: Rankings are older than 30 days.[/]\n")

    _print_ranks(store)


def _print_ranks(store):
    from datetime import datetime

    table = Table(title="Model Rankings")
    table.add_column("Domain", style="cyan")
    table.add_column("Model", style="yellow")
    table.add_column("Score", justify="right")
    table.add_column("Latency")
    table.add_column("Age")

    now = datetime.now()
    for entry in sorted(store.entries, key=lambda e: (e.domain.value, -e.score)):
        age_days = (now - entry.timestamp).days if entry.timestamp else 0
        age_str = f"{age_days}d" if age_days > 0 else "today"
        lat = f"{entry.latency_ms:.0f}ms" if entry.latency_ms else "-"
        score_str = f"{entry.score:.2f}"
        score_style = (
            "green" if entry.score >= 0.8 else "yellow" if entry.score >= 0.5 else "red"
        )
        table.add_row(
            entry.domain.value,
            entry.model,
            Text(score_str, style=score_style),
            lat,
            age_str,
        )

    console.print(table)


@app.command()
def config_init():
    """Create an initial config interactively."""
    if CONFIG_PATH.exists():
        if not Confirm.ask(f"Config already exists at {CONFIG_PATH}. Overwrite?"):
            raise typer.Exit(0)

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    console.print("[bold]PolyMind Config Initialization[/]\n")

    model_name = Prompt.ask("Default Ollama model", default="llama3.2:1b")
    router = Prompt.ask(
        "Router model (lightweight for task decomposition)",
        default=f"ollama/{model_name}",
    )
    strategy = Prompt.ask(
        "Scheduler strategy",
        choices=["model_aware", "sequential", "parallel"],
        default="model_aware",
    )

    config = Config(
        models=[{"name": model_name, "provider": "ollama"}],
        router_model=router,
        synthesizer_model=router,
        judge_model=router,
        scheduler={"strategy": strategy, "pass_context": True},
    )
    config.to_yaml(CONFIG_PATH)
    console.print(f"\n[green]Config written to {CONFIG_PATH}[/]")


@app.command()
def status():
    """Show PolyMind status, config health, and rankings age."""
    if CONFIG_PATH.exists():
        config = _load_config()
        console.print("[green]Config:[/] Found")
        console.print(f"  Router: [cyan]{config.router_model}[/]")
        console.print(f"  Strategy: [cyan]{config.scheduler.strategy.value}[/]")
        console.print(f"  Models configured: {len(config.models)}")
    else:
        console.print("[yellow]Config: Not found[/]")

    if RANKS_PATH.exists():
        store = load_ranks(RANKS_PATH)
        console.print(f"[green]Rankings:[/] {len(store.entries)} entries")
        if store.is_stale():
            console.print("  [yellow]Rankings are stale (>30 days old)[/]")
        top_entries = {}
        for e in store.entries:
            if e.domain not in top_entries or e.score > top_entries[e.domain].score:
                top_entries[e.domain] = e
        if top_entries:
            console.print("  Top models per domain:")
            for domain, entry in sorted(top_entries.items(), key=lambda x: x[0].value):
                console.print(
                    f"    [cyan]{domain.value}[/]: {entry.model} ({entry.score:.2f})"
                )
    else:
        console.print("[yellow]Rankings: Not found (run polymind benchmark)[/]")


@app.command()
def diff(
    prompt: str = typer.Argument(..., help="Prompt to test"),
    models: list[str] = typer.Argument(
        ..., help="Models to compare (e.g. ollama/model1 ollama/model2)"
    ),
):
    """Compare outputs from multiple models side by side."""
    asyncio.run(_async_diff(prompt, models))


async def _async_diff(prompt: str, models: list[str]):
    outputs: list[tuple[str, str]] = []

    for model in models:
        with console.status(f"[green]Querying {model}...") as _:
            info = resolve_model_string(model)
            subtask = __import__("polymind.core.types", fromlist=["Subtask"]).Subtask(
                id="t1", domain=DomainType.general, prompt=prompt
            )
            result = await execute_subtask(subtask, model_ref=model, provider_info=info)
            outputs.append((model, result.output))

    table = Table(title="Model Comparison")
    for model, _ in outputs:
        table.add_column(model, style="cyan", no_wrap=False)

    max_lines = max(len(o[1].split("\n")) for o in outputs)
    for i in range(max_lines):
        row: list[str] = []
        for _, output in outputs:
            lines = output.split("\n")
            row.append(lines[i] if i < len(lines) else "")
        table.add_row(*row)

    console.print(table)
