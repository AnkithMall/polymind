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
def list_skills(
    source: str = typer.Option("", "--source", "-s", help="Filter by source: builtin, user, mcp"),
    all: bool = typer.Option(False, "--all", "-a", help="Show disabled skills too"),
) -> None:
    """List all available skills."""
    from polymind.core.skills.registry import SkillRegistry
    from polymind.core.skills.types import SkillSource

    registry = SkillRegistry()
    registry.discover()

    if all:
        manifests = registry.list_skills()
    else:
        manifests = registry.list_enabled()

    if source:
        try:
            src = SkillSource(source)
            manifests = [m for m in manifests if m.source == src]
        except ValueError:
            console.print(f"[red]Unknown source '{source}'. Use: builtin, user, mcp[/]")
            raise typer.Exit(code=1)

    if not manifests:
        console.print("[yellow]No skills available.[/]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan", width=20)
    table.add_column("Status", width=10)
    table.add_column("Source", width=10)
    table.add_column("Description", width=45)
    table.add_column("Permissions", width=18)
    table.add_column("Approval", width=10)

    for m in manifests:
        status_icon = "✓" if m.status.value == "enabled" else "✗"
        status_style = "green" if m.status.value == "enabled" else "dim"
        perms = ", ".join(p.value for p in m.permissions) or "—"
        approval = "⚠️ Yes" if m.requires_approval else "No"
        desc = m.description[:45] + "…" if len(m.description) > 45 else m.description

        table.add_row(
            m.name,
            f"[{status_style}]{status_icon} {m.status.value}[/]",
            m.source.value,
            desc,
            perms,
            approval,
        )

    console.print(table)
    total = len(registry.list_skills())
    enabled = len(registry.list_enabled())
    console.print(f"\n[dim]{enabled}/{total} skill(s) enabled[/]")


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

    status_color = "green" if manifest.status.value == "enabled" else "red"

    lines = [
        f"[bold]Name:[/] {manifest.name}",
        f"[bold]Status:[/] [{status_color}]{manifest.status.value}[/]",
        f"[bold]Source:[/] {manifest.source.value}" + (
            f" ({manifest.source_detail})" if manifest.source_detail else ""
        ),
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


@app.command("status")
def skill_status() -> None:
    """Show status of all skills — installed, enabled, source, and health."""
    from polymind.core.skills.registry import SkillRegistry
    from polymind.core.skills.types import SkillSource

    registry = SkillRegistry()
    registry.discover()

    all_manifests = registry.list_skills()
    enabled = registry.list_enabled()
    disabled = registry.list_disabled()

    # Summary panel
    builtin = registry.list_by_source(SkillSource.BUILTIN)
    user = registry.list_by_source(SkillSource.USER)
    mcp = registry.list_by_source(SkillSource.MCP)

    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("Total skills", str(len(all_manifests)))
    summary.add_row("[green]Enabled[/]", str(len(enabled)))
    summary.add_row("[dim]Disabled[/]", str(len(disabled)))
    summary.add_row("Built-in", str(len(builtin)))
    summary.add_row("User-installed", str(len(user)))
    summary.add_row("MCP servers", str(len(mcp)))

    console.print(Panel(summary, title="Skills Status", border_style="cyan"))

    # Per-source breakdown
    for source_label, source_val in [
        ("Built-in Skills", SkillSource.BUILTIN),
        ("User Skills", SkillSource.USER),
        ("MCP Skills", SkillSource.MCP),
    ]:
        skills = registry.list_by_source(source_val)
        if not skills:
            continue

        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("Name", style="cyan", width=20)
        table.add_column("Status", width=10)
        table.add_column("Description", width=50)

        for m in skills:
            status_icon = "✓" if m.status.value == "enabled" else "✗"
            status_style = "green" if m.status.value == "enabled" else "dim"
            desc = m.description[:50] + "…" if len(m.description) > 50 else m.description
            table.add_row(
                m.name,
                f"[{status_style}]{status_icon} {m.status.value}[/]",
                desc,
            )

        console.print(Panel(table, title=source_label, border_style="dim"))

    # Check for errors
    error_skills = [m for m in all_manifests if m.status.value == "error"]
    if error_skills:
        console.print("\n[bold red]Skills with errors:[/]")
        for m in error_skills:
            console.print(f"  ✗ {m.name}: {m.source_detail or 'unknown error'}")


@app.command("enable")
def enable_skill(name: str = typer.Argument(help="Skill name to enable")) -> None:
    """Enable a skill."""
    from polymind.core.skills.registry import SkillRegistry

    registry = SkillRegistry()
    if registry.enable(name):
        console.print(f"[green]✓ Enabled skill '{name}'[/]")
    else:
        console.print(f"[red]Skill '{name}' not found.[/]")
        raise typer.Exit(code=1)


@app.command("disable")
def disable_skill(name: str = typer.Argument(help="Skill name to disable")) -> None:
    """Disable a skill."""
    from polymind.core.skills.registry import SkillRegistry

    registry = SkillRegistry()
    if registry.disable(name):
        console.print(f"[yellow]✗ Disabled skill '{name}'[/]")
    else:
        console.print(f"[red]Skill '{name}' not found.[/]")
        raise typer.Exit(code=1)


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
    """Run the agent loop with skills enabled."""
    from polymind.core.model.registry import ModelRegistry
    from polymind.core.runtime.artifact import load_runtime_config
    from polymind.core.skills.agent import AgentConfig, AgentLoop, LlamaProvider
    from polymind.core.skills.registry import SkillRegistry
    from polymind.core.skills.sandbox import SandboxConfig

    skill_registry = SkillRegistry()
    skill_registry.discover()

    model_reg = ModelRegistry()
    models = model_reg.load()

    if not models:
        console.print("[red]No models installed. Run: polymind model download[/]")
        raise typer.Exit(code=1)

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
        selected = models[0]

    enabled_names = skill_registry.list_enabled_names()
    console.print(f"[dim]Using model: {selected.filename}[/]")
    console.print(f"[dim]Skills enabled: {', '.join(enabled_names)}[/]")
    console.print()

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

    sandbox_cfg = SandboxConfig()
    if approve:
        sandbox_cfg.auto_approved_skills = enabled_names

    agent_cfg = AgentConfig(
        verbose=verbose,
        approval_callback=lambda name, perm, args: (
            True if approve else _ask_approval(name, perm, args)
        ),
    )

    agent = AgentLoop(provider, skill_registry, sandbox_cfg, agent_cfg)

    console.print("[bold cyan]Polymind Agent[/] (skills-enabled)")
    console.print(f"[dim]Prompt: {prompt}[/]")
    console.print()

    result = agent.run(prompt, context={"working_dir": str(Path.cwd())})

    if verbose and result.steps:
        for step in result.steps:
            if step.tool_name:
                status_icon = "✓" if step.status.value == "complete" else "✗"
                console.print(
                    f"  [dim]Step {step.step_number}:[/] {status_icon} "
                    f"{step.tool_name}({json.dumps(step.tool_args)[:80]})"
                )

    console.print()
    console.print(Panel(result.response, title="Response", border_style="green"))

    if result.tools_used:
        console.print(f"[dim]Tools used: {', '.join(result.tools_used)}[/]")


@app.command("install")
def install_skill(
    path: str = typer.Argument(help="Path to skill .py file or directory"),
) -> None:
    """Install a user skill to .polymind/skills/."""
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


# ── MCP sub-commands ────────────────────────────────────

mcp_app = typer.Typer(help="Manage MCP (Model Context Protocol) servers.")
app.add_typer(mcp_app, name="mcp")


@mcp_app.command("add")
def mcp_add(
    name: str = typer.Argument(help="Server name (e.g., 'filesystem')"),
    command: str = typer.Option(..., "--command", "-c", help="Command to run (e.g., npx, python)"),
    args: str = typer.Option("[]", "--args", help="JSON array of arguments"),
    env: str = typer.Option("{}", "--env", help="JSON object of environment variables"),
    cwd: str = typer.Option("", "--cwd", help="Working directory for the server"),
) -> None:
    """Add an MCP server.

    Examples:
        polymind skills mcp add filesystem --command npx --args '["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]'
        polymind skills mcp add github --command npx --args '["-y", "@modelcontextprotocol/server-github"]'
    """
    from polymind.core.skills.registry import _load_mcp_config, _save_mcp_config

    try:
        args_list = json.loads(args)
        env_dict = json.loads(env)
    except json.JSONDecodeError:
        console.print("[red]Invalid JSON in --args or --env[/]")
        raise typer.Exit(code=1)

    servers = _load_mcp_config()
    servers[name] = {
        "command": command,
        "args": args_list,
        "env": env_dict if env_dict else None,
        "cwd": cwd or None,
        "enabled": True,
    }
    _save_mcp_config(servers)
    console.print(f"[green]✓ Added MCP server '{name}'[/]")
    console.print(f"[dim]  Command: {command} {' '.join(args_list)}[/]")
    console.print("[dim]  Run 'polymind skills mcp list' to see status[/]")


@mcp_app.command("list")
def mcp_list() -> None:
    """List configured MCP servers and their tools."""
    from polymind.core.skills.registry import SkillRegistry, _load_mcp_config
    from polymind.core.skills.types import SkillSource

    servers = _load_mcp_config()
    if not servers:
        console.print("[yellow]No MCP servers configured.[/]")
        console.print("[dim]Add one with: polymind skills mcp add <name> --command <cmd> --args '[...]''[/]")
        return

    registry = SkillRegistry()
    registry.discover()

    for server_name, cfg in servers.items():
        status = "✓ enabled" if cfg.get("enabled", True) else "✗ disabled"
        status_style = "green" if cfg.get("enabled", True) else "dim"

        console.print(f"\n[bold]MCP Server: {server_name}[/] [{status_style}]{status}[/]")
        console.print(f"  Command: {cfg.get('command', '')} {' '.join(cfg.get('args', []))}")

        # Show tools from this server
        mcp_skills = [
            m for m in registry.list_skills()
            if m.source == SkillSource.MCP and m.source_detail == f"mcp:{server_name}"
        ]
        if mcp_skills:
            console.print(f"  Tools ({len(mcp_skills)}):")
            for m in mcp_skills:
                status_icon = "✓" if m.status.value == "enabled" else "✗"
                console.print(f"    {status_icon} {m.name}: {m.description[:60]}")
        else:
            console.print("  [dim]No tools discovered (server may not be running)[/]")


@mcp_app.command("remove")
def mcp_remove(name: str = typer.Argument(help="Server name to remove")) -> None:
    """Remove an MCP server."""
    from polymind.core.skills.registry import _load_mcp_config, _save_mcp_config

    servers = _load_mcp_config()
    if name not in servers:
        console.print(f"[red]MCP server '{name}' not found.[/]")
        raise typer.Exit(code=1)

    del servers[name]
    _save_mcp_config(servers)
    console.print(f"[yellow]Removed MCP server '{name}'[/]")


@mcp_app.command("test")
def mcp_test(
    name: str = typer.Argument(help="Server name to test"),
) -> None:
    """Test connection to an MCP server and list its tools."""
    from polymind.core.skills.mcp_client import list_mcp_tools
    from polymind.core.skills.registry import _load_mcp_config

    servers = _load_mcp_config()
    if name not in servers:
        console.print(f"[red]MCP server '{name}' not found.[/]")
        raise typer.Exit(code=1)

    cfg = servers[name]
    console.print(f"[dim]Connecting to MCP server '{name}'...[/]")

    tools = list_mcp_tools(cfg)
    if tools:
        console.print(f"[green]✓ Connected! Found {len(tools)} tool(s):[/]")
        for t in tools:
            console.print(f"  • {t['name']}: {t.get('description', '')[:60]}")
    else:
        console.print("[red]✗ Failed to connect or no tools found.[/]")
        console.print("[dim]  Make sure the server command is correct and dependencies are installed.[/]")


def _ask_approval(skill_name: str, permission: str, args: dict) -> bool:
    """Ask user for approval."""
    console.print(f"[yellow]⚠️  Skill '{skill_name}' requests {permission} permission.[/]")
    console.print(f"[dim]Arguments: {json.dumps(args, indent=2)[:200]}[/]")
    response = typer.confirm("Allow execution?")
    return response
