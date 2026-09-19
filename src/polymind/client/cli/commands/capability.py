"""Capability commands — display model capability profiles.

Shows how well each model performs across all evaluated domains,
using confidence/evidence data from the scoring system.
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from polymind.core.confidence.artifact import load_all_domains, load_confidence
from polymind.core.model.registry import ModelRegistry

app = typer.Typer()
console = Console()


@app.command("show")
def show_capability(
    model_id: str = typer.Option("", "--model", "-m", help="Model ID to show capabilities for."),
    explain: bool = typer.Option(
        False, "--explain", help="Show explanation of capability scoring."
    ),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Show model capability profiles across all evaluated domains.

    Displays a matrix of models × domains with their capability scores.
    Higher scores indicate better performance in that domain.

    Capability vs Confidence:
      Capability Score — how well the model performed (the score itself)
      Confidence/Evidence — how much data supports the score
        (more questions = higher confidence in the score)

      A model scoring 90% on 2 questions is less reliable than
      scoring 85% on 20 questions.

    The pipeline router uses these capability scores to assign
    the best model for each domain in a multi-domain prompt.

    Examples:

        polymind capability show

        polymind capability show -m 1

        polymind capability show --explain

        polymind capability show --json
    """
    if explain:
        console.print("[bold]Capability Profiles Explained[/]")
        console.print()
        console.print("  A capability profile shows how well a model performs across")
        console.print("  different domains. The pipeline uses these scores to route")
        console.print("  each subtask to the best-suited model.")
        console.print()
        console.print("  [cyan]Domain Score[/]   How well the model performs in a domain")
        console.print("                   (e.g., 'frontend: 91%'). Computed from")
        console.print("                   task scores → suite scores → domain scores.")
        console.print()
        console.print("  [cyan]Best Domain[/]   The domain where this model excels most.")
        console.print("                   The pipeline prioritizes this model for")
        console.print("                   tasks in its best domain.")
        console.print()
        console.print("  [cyan]Overall[/]       Average across all evaluated domains.")
        console.print("                   Used as a fallback when no domain-specific")
        console.print("                   score exists.")
        console.print()
        console.print("  [dim]Colors: green ≥ 80%  yellow ≥ 60%  red < 60%[/]")
        console.print()
        return

    confidence = load_confidence()

    if not confidence:
        console.print("[yellow]No capability data available.[/]")
        console.print("Run: polymind confidence compute")
        return

    # Filter by model if specified
    if model_id:
        if model_id not in confidence:
            console.print(f"[red]No data for model {model_id}[/]")
            raise typer.Exit(code=1)
        scores = {model_id: confidence[model_id]}
    else:
        scores = confidence

    # Get all domain IDs across all models
    all_domain_ids: set[str] = set()
    for conf in scores.values():
        all_domain_ids.update(conf.domains.keys())

    if not all_domain_ids:
        console.print("[yellow]No domain scores found.[/]")
        return

    sorted_domains = sorted(all_domain_ids)

    if output_json:
        data = {}
        for mid, conf in scores.items():
            data[mid] = {
                "overall": conf.overall_score,
                "domains": {
                    did: ds.overall for did, ds in conf.domains.items()
                },
            }
        typer.echo(json.dumps(data, indent=2))
        return

    # Display capability matrix
    console.print("[bold]Model Capability Profiles[/]")
    console.print()

    table = Table(show_lines=True)
    table.add_column("Domain", style="cyan", width=20)
    for mid, conf in sorted(scores.items(), key=lambda x: float(x[0])):
        table.add_column(f"Model {mid}", justify="right", width=12)

    for domain_id in sorted_domains:
        row: list[str] = [domain_id]
        for mid, conf in sorted(scores.items(), key=lambda x: float(x[0])):
            ds = conf.domains.get(domain_id)
            if ds is not None:
                score = ds.overall
                # Color-code the score
                if score >= 80:
                    cell = f"[green]{score:>6.1f}%[/]"
                elif score >= 60:
                    cell = f"[yellow]{score:>6.1f}%[/]"
                else:
                    cell = f"[red]{score:>6.1f}%[/]"
                row.append(cell)
            else:
                row.append("[dim]  —  [/]")
        table.add_row(*row)

    # Overall row
    row = ["[bold]Overall[/bold]"]
    for mid, conf in sorted(scores.items(), key=lambda x: float(x[0])):
        row.append(f"[bold]{conf.overall_score:>6.1f}%[/bold]")
    table.add_row(*row)

    console.print(table)

    # Show best domain for each model
    console.print()
    for mid, conf in sorted(scores.items(), key=lambda x: float(x[0])):
        if conf.best_domain:
            console.print(
                f"  Model {mid}: best domain = [cyan]{conf.best_domain}[/]"
                f" ({conf.overall_score:.1f}% overall)"
            )


@app.command("domains")
def list_scored_domains() -> None:
    """List all domains that have been scored.

    Shows which domains have capability data and how many
    questions/suites contribute to each score.

    Examples:

        polymind capability domains
    """
    confidence = load_confidence()
    if not confidence:
        console.print("[yellow]No capability data available.[/]")
        return

    # Collect all domains with evidence
    domain_info: dict[str, dict] = {}
    for conf in confidence.values():
        for domain_id, ds in conf.domains.items():
            if domain_id not in domain_info:
                domain_info[domain_id] = {
                    "models": 0,
                    "total_tasks": 0,
                    "scores": [],
                }
            domain_info[domain_id]["models"] += 1
            domain_info[domain_id]["scores"].append(ds.overall)
            for ss in ds.suites.values():
                domain_info[domain_id]["total_tasks"] += ss.total

    table = Table(title="Scored Domains")
    table.add_column("Domain", style="cyan")
    table.add_column("Models", justify="right")
    table.add_column("Avg Score", justify="right")
    table.add_column("Questions", justify="right")

    for domain_id, info in sorted(domain_info.items()):
        avg_score = sum(info["scores"]) / len(info["scores"]) if info["scores"] else 0
        table.add_row(
            domain_id,
            str(info["models"]),
            f"{avg_score:.1f}%",
            str(info["total_tasks"]),
        )

    console.print(table)
