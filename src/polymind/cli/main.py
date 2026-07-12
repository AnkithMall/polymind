from __future__ import annotations

import asyncio
import logging
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
    detect_local_providers,
    execute_subtask,
    get_all_ollama_models,
    get_tasks_for_domain,
    load_ranks,
    resolve_model_string,
    run_benchmark,
    save_ranks,
    setup_logging,
    synthesize,
)
from polymind.core.config import Config, ModelConfig
from polymind.core.providers import ProviderType
from polymind.core.types import _rank_key, RankingMode

app = typer.Typer(
    name="polymind",
    help="Multi-Specialist LLM Orchestrator — CLI · TUI · Local-First · Hardware-Optimised",
    no_args_is_help=True,
)
config_app = typer.Typer(
    name="config",
    help="Manage configuration and providers",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")
console = Console()

CONFIG_PATH = Path("~/.polymind/config.yaml").expanduser()
RANKS_PATH = Path("~/.polymind/ranks.yaml").expanduser()


@app.callback()
def main_callback(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed logs of what is happening",
    ),
):
    if verbose:
        setup_logging(logging.DEBUG)


def _load_config() -> Config:
    config = Config.from_yaml(CONFIG_PATH)
    if config.verbose:
        setup_logging(logging.DEBUG)
    return config


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
        ranking_mode=config.ranking_mode,
        model_source=config.model_source,
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
    models: Optional[list[str]] = typer.Argument(
        None, help="Models to benchmark (e.g. ollama/llama3.2:1b)"
    ),
    domains: Optional[list[str]] = typer.Option(
        None, "--domain", "-d", help="Domains to benchmark (default: all)"
    ),
    auto_detect: bool = typer.Option(
        False,
        "--auto-detect",
        "-a",
        help="Auto-detect models from all configured providers",
    ),
):
    """Run benchmark tasks against models to build rankings."""
    config = _load_config()

    resolved_models = models
    if auto_detect or not models:
        detected = detect_local_providers()
        if not detected:
            console.print("[red]No local models detected. Install models or specify them explicitly.[/]")
            raise typer.Exit(1)
        if models:
            detected = [m for m in detected if m["name"] in models or f"{m['provider']}/{m['name']}" in models]
        resolved_models = [f"{m['provider']}/{m['name']}" for m in detected]
        console.print(f"[cyan]Auto-detected models:[/] {', '.join(resolved_models)}")

    domain_list: list[DomainType] = (
        [DomainType(d) for d in domains] if domains else ALL_DOMAINS
    )
    judge = config.judge_model

    total_models = len(resolved_models)
    total_domains = len(domain_list)
    tasks_per_domain = {d: len(get_tasks_for_domain(d)) for d in domain_list}
    global_total = sum(tasks_per_domain.values()) * total_models

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        prog_task = progress.add_task("[green]Benchmarking...", total=global_total)

        model_idx = [0]
        domain_idx = [0]

        def progress_callback(pct: float, msg: str):
            if " / " in msg:
                parts = msg.split(" / ")
                model_name = parts[0]
                domain_name = parts[1]
                task_info = parts[2] if len(parts) > 2 else ""
            else:
                model_name = ""
                domain_name = ""
                task_info = ""

            domain_tasks = tasks_per_domain.get(
                next((d for d in domain_list if d.value == domain_name), None), 0
            )
            desc = f"[green]{model_name} | {domain_name} | {task_info}"
            progress.update(
                prog_task,
                completed=pct * global_total,
                description=desc,
            )

        store = asyncio.run(
            run_benchmark(
                models=resolved_models,
                domains=domain_list,
                judge_model=judge,
                progress_callback=progress_callback,
            )
        )

    save_ranks(store, RANKS_PATH)
    console.print(f"\n[green]Benchmark complete![/] Saved to [bold]{RANKS_PATH}[/]")

    _print_ranks(store)


@app.command()
def ranks(
    best: bool = typer.Option(
        False,
        "--best",
        "-b",
        help="Show only the best model per domain",
    ),
    mode: str = typer.Option(
        "accuracy",
        "--mode",
        "-m",
        help="Ranking mode: accuracy, cost, or cost_effective",
    ),
):
    """Display current model rankings."""
    try:
        ranking_mode = RankingMode(mode)
    except ValueError:
        console.print(f"[red]Invalid mode: {mode}. Choose from: accuracy, cost, cost_effective[/]")
        raise typer.Exit(1)

    store = load_ranks(RANKS_PATH)
    if not store.entries:
        console.print(
            "[yellow]No rankings found. Run [bold]polymind benchmark[/] first.[/]"
        )
        raise typer.Exit(0)

    if store.is_stale():
        console.print("[yellow]Warning: Rankings are older than 30 days.[/]\n")

    _print_ranks(store, best=best, mode=ranking_mode)


def _print_ranks(store, best: bool = False, mode: RankingMode = RankingMode.accuracy):
    from datetime import datetime

    now = datetime.now()

    if best:
        rows: list[tuple[str, str, str, str, str, str]] = []
        for domain in ALL_DOMAINS:
            top = store.top_for_domain(domain, mode)
            if top is None:
                continue
            age_days = (now - top.timestamp).days if top.timestamp else 0
            age_str = f"{age_days}d" if age_days > 0 else "today"
            lat = f"{top.latency_ms:.0f}ms" if top.latency_ms else "-"
            cost_str = f"${top.cost:.6f}" if top.cost is not None else "-"
            score_str = f"{top.score:.2f}"
            score_style = (
                "green" if top.score >= 0.8 else "yellow" if top.score >= 0.5 else "red"
            )
            rows.append((top.domain.value, top.model, score_str, lat, cost_str,
                         age_str, score_style))

        if not rows:
            console.print("[yellow]No rankings found for any domain.[/]")
            return

        title = f"Best per Domain (by {mode.value})"
        table = Table(title=title)
        table.add_column("Domain", style="cyan")
        table.add_column("Model", style="yellow")
        table.add_column("Score", justify="right")
        table.add_column("Latency")
        table.add_column("Cost", justify="right")
        table.add_column("Age")
        for domain, model, score_str, lat, cost_str, age_str, style in rows:
            table.add_row(domain, model, Text(score_str, style=style), lat, cost_str, age_str)
        console.print(table)
        console.print(f"[dim]{len(rows)} domains shown[/]")
        return

    table = Table(title=f"Model Rankings (by {mode.value})")
    table.add_column("Domain", style="cyan")
    table.add_column("Model", style="yellow")
    table.add_column("Score", justify="right")
    table.add_column("Latency")
    table.add_column("Cost", justify="right")
    table.add_column("Age")

    key_fn = lambda e: (e.domain.value, -_rank_key(e, mode))
    for entry in sorted(store.entries, key=key_fn):
        age_days = (now - entry.timestamp).days if entry.timestamp else 0
        age_str = f"{age_days}d" if age_days > 0 else "today"
        lat = f"{entry.latency_ms:.0f}ms" if entry.latency_ms else "-"
        cost_str = f"${entry.cost:.6f}" if entry.cost is not None else "-"
        score_str = f"{entry.score:.2f}"
        score_style = (
            "green" if entry.score >= 0.8 else "yellow" if entry.score >= 0.5 else "red"
        )
        table.add_row(
            entry.domain.value,
            entry.model,
            Text(score_str, style=score_style),
            lat,
            cost_str,
            age_str,
        )

    console.print(table)


@config_app.command("init")
def config_init():
    """Create an initial config interactively with provider setup."""
    if CONFIG_PATH.exists():
        if not Confirm.ask(f"Config already exists at {CONFIG_PATH}. Overwrite?"):
            raise typer.Exit(0)

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    console.print("[bold]PolyMind Config Initialization[/]\n")

    models: list[ModelConfig] = []

    if Confirm.ask("Auto-detect local providers (Ollama, LM Studio)?", default=True):
        detected = detect_local_providers()
        if detected:
            console.print(f"[green]Detected {len(detected)} local model(s):[/]")
            for m in detected:
                console.print(f"  - {m['provider']}/{m['name']}")
            if Confirm.ask("Add all detected models?", default=True):
                models.extend(ModelConfig(**m) for m in detected)
        else:
            console.print("[yellow]No local providers detected.[/]")

    if not models:
        while True:
            console.print("\n[bold]Add a provider:[/]")
            provider = Prompt.ask(
                "Provider type",
                choices=["ollama", "lm_studio", "openai", "anthropic", "openrouter", "(done)"],
                default="ollama",
            )
            if provider == "(done)":
                break
            model_name = Prompt.ask("Model name")
            mc = ModelConfig(name=model_name, provider=provider)
            if provider in ("lm_studio", "openai", "anthropic", "openrouter"):
                base_url = Prompt.ask("Base URL (optional)", default="")
                if base_url:
                    mc.base_url = base_url
                if provider in ("openai", "anthropic"):
                    api_key = Prompt.ask("API Key (optional)", default="")
                    if api_key:
                        mc.api_key = api_key
            models.append(mc)
            if not Confirm.ask("Add another model?", default=True):
                break

    if not models:
        models.append(ModelConfig(name="llama3.2:1b", provider="ollama"))

    router = Prompt.ask(
        "Router model (lightweight for task decomposition)",
        default=f"{models[0].provider}/{models[0].name}",
    )
    strategy = Prompt.ask(
        "Scheduler strategy",
        choices=["model_aware", "sequential", "parallel"],
        default="model_aware",
    )
    ranking_mode = Prompt.ask(
        "Ranking mode",
        choices=["accuracy", "cost", "cost_effective"],
        default="accuracy",
    )
    model_source = Prompt.ask(
        "Model source filter",
        choices=["all", "local", "online"],
        default="all",
    )
    proxy = Prompt.ask("LiteLLM proxy base URL (optional)", default="")
    litellm_proxy = proxy.strip() or None

    config = Config(
        models=[m.model_dump() for m in models],
        router_model=router,
        synthesizer_model=router,
        judge_model=router,
        scheduler={"strategy": strategy, "pass_context": True},
        ranking_mode=ranking_mode,
        model_source=model_source,
        litellm_proxy=litellm_proxy,
    )
    config.to_yaml(CONFIG_PATH)
    console.print(f"\n[green]Config written to {CONFIG_PATH}[/]")


@config_app.command("add-provider")
def add_provider():
    """Add a new provider/model to the config interactively."""
    config = _ensure_config()

    console.print("[bold]Add a Provider/Model[/]\n")

    provider = Prompt.ask(
        "Provider type",
        choices=["ollama", "lm_studio", "openai", "anthropic", "openrouter"],
    )
    model_name = Prompt.ask("Model name")
    mc = ModelConfig(name=model_name, provider=provider)

    if provider in ("lm_studio", "openai", "anthropic", "openrouter"):
        base_url = Prompt.ask("Base URL (optional)", default="")
        if base_url:
            mc.base_url = base_url
        if provider in ("openai", "anthropic"):
            api_key = Prompt.ask("API Key (optional)", default="")
            if api_key:
                mc.api_key = api_key

    config.models.append(mc)
    config.to_yaml(CONFIG_PATH)
    console.print(f"[green]Added {provider}/{model_name} to config.[/]")


@config_app.command("auto-detect")
def auto_detect_providers():
    """Auto-detect local providers and update config."""
    config = _ensure_config()

    console.print("[bold]Auto-detecting local providers...[/]\n")

    detected = detect_local_providers()
    if not detected:
        console.print("[yellow]No local providers detected. Install Ollama or start LM Studio first.[/]")
        raise typer.Exit(0)

    console.print(f"[green]Detected {len(detected)} model(s):[/]")
    for m in detected:
        console.print(f"  - {m['provider']}/{m['name']}")

    if Confirm.ask("Add all to config?", default=True):
        for m in detected:
            mc = ModelConfig(**m)
            if mc not in config.models:
                config.models.append(mc)
        config.to_yaml(CONFIG_PATH)
        console.print(f"\n[green]Config updated at {CONFIG_PATH}[/]")

    # Set router/judge to first model if none set
    if not config.router_model or config.router_model == "ollama/llama3.2:1b":
        first = detected[0]
        config.router_model = f"{first['provider']}/{first['name']}"
        config.judge_model = config.router_model
        config.synthesizer_model = config.router_model
        config.to_yaml(CONFIG_PATH)
        console.print(f"[green]Router set to {config.router_model}[/]")


@app.command()
def status():
    """Show PolyMind status, config health, and rankings age."""
    if CONFIG_PATH.exists():
        config = _load_config()
        console.print("[green]Config:[/] Found")
        console.print(f"  Router: [cyan]{config.router_model}[/]")
        console.print(f"  Strategy: [cyan]{config.scheduler.strategy.value}[/]")
        console.print(f"  Ranking mode: [cyan]{config.ranking_mode.value}[/]")
        console.print(f"  Model source: [cyan]{config.model_source.value}[/]")
        if config.litellm_proxy:
            console.print(f"  Proxy: [cyan]{config.litellm_proxy}[/]")
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
