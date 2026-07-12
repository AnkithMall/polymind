from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from polymind.core.hardware import detect_local_providers

logger = logging.getLogger(__name__)


SETUP_STEPS = """
[bold yellow]PolyMind Setup Required[/]

PolyMind needs at least one working model to operate. Here's how to get started:

[bold]Step 1: Install a local LLM engine[/]
  • [cyan]Ollama[/]       — https://ollama.com  (recommended, free)
  • [cyan]LM Studio[/]    — https://lmstudio.ai  (free, GUI-based)

  After installing, pull a model:
    [dim]ollama pull llama3.2:1b[/]

[bold]Step 2: Configure PolyMind[/]
  Run the interactive setup wizard:
    [cyan]polymind config init[/]

  Or auto-detect your local models:
    [cyan]polymind config auto-detect[/]

[bold]Step 3: Run a benchmark[/]
  Test which models perform best on each domain:
    [cyan]polymind benchmark --auto-detect[/]

[bold]Step 4: Ask a question[/]
    [cyan]polymind ask "What is quantum computing?"[/]

[bold]Other useful commands[/]
  • [cyan]polymind status[/]       — Check your setup health
  • [cyan]polymind ranks[/]        — View model rankings
  • [cyan]polymind --help[/]       — See all commands
"""


def check_provider_health(config: Any) -> list[str]:
    issues: list[str] = []
    config_path = Path("~/.polymind/config.yaml").expanduser()

    if not config_path.exists():
        issues.append("Config file not found")
        return issues

    models = getattr(config, "models", [])
    if not models:
        issues.append("No models configured")

    router = getattr(config, "router_model", None)
    if not router:
        issues.append("No router model configured")

    return issues


def print_setup_guide() -> str:
    detected = detect_local_providers()
    parts = [SETUP_STEPS.strip()]
    if detected:
        models_str = "\n  ".join(
            f"• {m['provider']}/{m['name']}" for m in detected[:5]
        )
        if len(detected) > 5:
            models_str += f"\n  • … and {len(detected) - 5} more"
        parts.append(f"\n[green]Detected {len(detected)} local model(s):[/]\n  {models_str}")
    return "\n".join(parts)


def health_report(config: Any) -> str:
    issues = check_provider_health(config)
    config_path = Path("~/.polymind/config.yaml").expanduser()
    ranks_path = Path("~/.polymind/ranks.yaml").expanduser()

    lines: list[str] = ["[bold cyan]PolyMind Health Report[/]\n"]

    lines.append(f"[bold]Config file:[/] {'[green]OK[/]' if config_path.exists() else '[red]MISSING[/]'}")
    lines.append(f"  Path: {config_path}")

    models = getattr(config, "models", [])
    lines.append(f"[bold]Models configured:[/] {len(models)}")
    for m in models:
        lines.append(f"  • {m.provider}/{m.name}")

    router = getattr(config, "router_model", None)
    lines.append(f"[bold]Router model:[/] {router or '[red]not set[/]'}")

    synth = getattr(config, "synthesizer_model", None)
    lines.append(f"[bold]Synthesizer model:[/] {synth or '(same as router)'}")

    lines.append(f"[bold]Rankings file:[/] {'[green]OK[/]' if ranks_path.exists() else '[yellow]not found (run benchmark)[/]'}")
    if ranks_path.exists():
        try:
            import yaml
            with open(ranks_path) as f:
                data = yaml.safe_load(f)
            entry_count = len(data.get("entries", [])) if data else 0
            lines.append(f"  {entry_count} rank entries")
        except Exception:
            lines.append("  [red]unable to read[/]")

    lines.append(f"\n[bold]Execution strategy:[/] {getattr(config.scheduler, 'strategy', 'model_aware').value}")

    if issues:
        lines.append(f"\n[red]{len(issues)} issue(s) found:[/]")
        for issue in issues:
            lines.append(f"  • {issue}")
        lines.append(f"\n{print_setup_guide()}")

    return "\n".join(lines)
