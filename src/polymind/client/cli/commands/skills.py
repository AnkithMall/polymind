"""Skills CLI — manage and use skills with local LLMs."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(help="Manage and use skills (tools) with local LLMs.")
console = Console()


@app.command("list")
def list_skills() -> None:
    """List all available skills."""
    from polymind.core.skills.registry import SkillRegistry

    registry = SkillRegistry()
    registry.discover()

    manifests = registry.list_skills()
    if not manifests:
        console.print("[yellow]No skills available.[/]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan", width=20)
    table.add_column("Description", width=50)
    table.add_column("Permissions", width=20)
    table.add_column("Approval", width=10)

    for m in manifests:
        perms = ", ".join(p.value for p in m.permissions) or "—"
        approval = "⚠️ Yes" if m.requires_approval else "No"
        table.add_row(
            m.name,
            m.description[:50] + "…" if len(m.description) > 50 else m.description,
            perms,
            approval,
        )

    console.print(table)
    console.print(f"\n[dim]{len(manifests)} skill(s) available[/]")


@app.command("show")
def show_skill(name: str = typer.Argument(help="Skill name")) -> None:
    """Show detailed information about a skill."""
    from polymind.core.skills.registry import SkillRegistry

    registry = SkillRegistry()
    registry.discover()

    manifest = registry.get_manifest(name)
    if manifest is None:
        console.print(f"[red]Skill '{name}' not found.[/]")
        raise typer.Exit(code=1)

    lines = [
        f"[bold]Name:[/] {manifest.name}",
        f"[bold]Version:[/] {manifest.version}",
        f"[bold]Author:[/] {manifest.author}",
        f"[bold]Description:[/] {manifest.description}",
        "",
        "[bold]Permissions:[/]",
    ]
    for p in manifest.permissions:
        lines.append(f"  • {p.value}")

    lines.append("")
    lines.append("[bold]Parameters:[/]")
    for p in manifest.params:
        req = "(required)" if p.required else f"(optional, default={p.default})"
        lines.append(f"  • {p.name} ({p.type}) {req}: {p.description}")
        if p.enum:
            lines.append(f"    Allowed: {', '.join(p.enum)}")

    if manifest.examples:
        lines.append("")
        lines.append("[bold]Examples:[/]")
        for ex in manifest.examples:
            lines.append(f"  {ex}")

    lines.append("")
    lines.append(f"[bold]Timeout:[/] {manifest.timeout_seconds}s")
    lines.append(f"[bold]Requires approval:[/] {'Yes' if manifest.requires_approval else 'No'}")

    console.print(Panel("\n".join(lines), title=f"Skill: {manifest.name}", border_style="cyan"))


@app.command("test")
def test_skill(
    name: str = typer.Argument(help="Skill name to test"),
    args_json: str = typer.Option("{}", "--args", "-a", help="JSON arguments for the skill"),
) -> None:
    """Test a skill directly (bypasses the agent loop)."""
    from polymind.core.skills.registry import SkillRegistry
    from polymind.core.skills.sandbox import SandboxConfig, SandboxEnforcer

    registry = SkillRegistry()
    registry.discover()

    skill = registry.get(name)
    if skill is None:
        console.print(f"[red]Skill '{name}' not found.[/]")
        raise typer.Exit(code=1)

    try:
        args = json.loads(args_json)
    except json.JSONDecodeError:
        console.print("[red]Invalid JSON in --args[/]")
        raise typer.Exit(code=1)

    enforcer = SandboxEnforcer(SandboxConfig())
    ctx = enforcer.create_context(skill_name=name)

    console.print(f"[dim]Testing skill '{name}' with args: {args}[/]")
    result = skill.execute(ctx, **args)

    if result.success:
        console.print(Panel(result.output, title="Result", border_style="green"))
    else:
        console.print(Panel(result.error, title="Error", border_style="red"))


@app.command("run")
def run_agent(
    prompt: str = typer.Argument(help="Prompt to process with the agent"),
    model_id: str = typer.Option("", "--model", "-m", help="Model ID to use"),
    approve: bool = typer.Option(False, "--approve", help="Auto-approve all tool calls"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show agent reasoning steps"),
) -> None:
    """Run the agent loop with skills enabled.

    The agent will reason about your prompt and use tools as needed.
    """
    from polymind.core.model.registry import ModelRegistry
    from polymind.core.runtime.artifact import load_runtime_config
    from polymind.core.skills.agent import AgentConfig, AgentLoop, LlamaProvider
    from polymind.core.skills.registry import SkillRegistry
    from polymind.core.skills.sandbox import SandboxConfig

    # Discover skills
    skill_registry = SkillRegistry()
    skill_registry.discover()

    # Find the model
    model_reg = ModelRegistry()
    models = model_reg.load()

    if not models:
        console.print("[red]No models installed. Run: polymind model download[/]")
        raise typer.Exit(code=1)

    # Select model
    selected = None
    if model_id:
        for m in models:
            if str(m.id) == model_id or m.filename == model_id:
                selected = m
                break
        if not selected:
            console.print(f"[red]Model '{model_id}' not found.[/]")
            raise typer.Exit(code=1)
    else:
        # Use the first/largest model
        selected = models[0]

    console.print(f"[dim]Using model: {selected.filename}[/]")
    console.print(f"[dim]Skills available: {', '.join(skill_registry.list_names())}[/]")
    console.print()

    # Load the model
    from llama_cpp import Llama

    config = load_runtime_config(str(selected.id))
    gpu_layers = config.gpu_layers if config else 0
    threads = config.threads if config else 4
    ctx_size = config.context_size if config else 2048
    batch_size = config.batch_size if config else 256

    console.print("[dim]Loading model...[/]")
    try:
        llm = Llama(
            model_path=selected.local_path,
            n_gpu_layers=gpu_layers,
            n_threads=threads,
            n_ctx=ctx_size,
            n_batch=batch_size,
            verbose=False,
        )
    except Exception:
        console.print("[yellow]GPU load failed, falling back to CPU...[/]")
        llm = Llama(
            model_path=selected.local_path,
            n_gpu_layers=0,
            n_threads=threads,
            n_ctx=ctx_size,
            n_batch=batch_size,
            verbose=False,
        )

    provider = LlamaProvider(llm)

    # Sandbox config
    sandbox_cfg = SandboxConfig()
    if approve:
        sandbox_cfg.auto_approved_skills = list(skill_registry.list_names())

    # Agent config
    agent_cfg = AgentConfig(
        verbose=verbose,
        approval_callback=lambda name, perm, args: (
            True if approve else _ask_approval(name, perm, args)
        ),
    )

    # Run the agent
    agent = AgentLoop(provider, skill_registry, sandbox_cfg, agent_cfg)

    console.print("[bold cyan]Polymind Agent[/] (skills-enabled)")
    console.print(f"[dim]Prompt: {prompt}[/]")
    console.print()

    result = agent.run(prompt, context={"working_dir": str(Path.cwd())})

    # Show steps if verbose
    if verbose and result.steps:
        for step in result.steps:
            if step.tool_name:
                status_icon = "✓" if step.status.value == "complete" else "✗"
                console.print(
                    f"  [dim]Step {step.step_number}:[/] {status_icon} "
                    f"{step.tool_name}({json.dumps(step.tool_args)[:80]})"
                )

    # Show final response
    console.print()
    console.print(Panel(result.response, title="Response", border_style="green"))

    if result.tools_used:
        console.print(f"[dim]Tools used: {', '.join(result.tools_used)}[/]")


@app.command("install")
def install_skill(
    path: str = typer.Argument(help="Path to skill .py file or directory"),
) -> None:
    """Install a user skill to ~/.polymind/skills/."""
    from polymind.core.paths import artifact_dir

    source = Path(path)
    if not source.exists():
        console.print(f"[red]Path not found: {path}[/]")
        raise typer.Exit(code=1)

    skills_dir = artifact_dir() / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    if source.is_file():
        dest = skills_dir / source.name
        dest.write_text(source.read_text())
    elif source.is_dir():
        import shutil

        dest = skills_dir / source.name
        shutil.copytree(source, dest, dirs_exist_ok=True)

    console.print(f"[green]Skill installed to {dest}[/]")


def _ask_approval(skill_name: str, permission: str, args: dict) -> bool:
    """Ask user for approval."""
    console.print(f"[yellow]⚠️  Skill '{skill_name}' requests {permission} permission.[/]")
    console.print(f"[dim]Arguments: {json.dumps(args, indent=2)[:200]}[/]")
    response = typer.confirm("Allow execution?")
    return response
