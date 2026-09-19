"""Domain management commands."""

import typer

from polymind.core.confidence.artifact import (
    delete_custom_domain,
    load_all_domains,
    load_domain_by_id,
    save_custom_domain,
)
from polymind.core.confidence.suites import get_all_domains
from polymind.core.confidence.types import Domain

app = typer.Typer()


@app.command("list")
def list_domains() -> None:
    """List all available domains (predefined + custom).

    Shows all scoring domains with their IDs, names, suite counts,
    and question counts. Predefined and custom domains are listed
    separately.

    Examples:

        polymind domain list
    """
    domains = load_all_domains()

    if not domains:
        typer.echo("No domains found.")
        return

    predefined = [d for d in domains if not d.custom]
    custom = [d for d in domains if d.custom]

    if predefined:
        typer.echo("Predefined Domains")
        typer.echo("=" * 60)
        for d in predefined:
            suite_count = len(d.suites)
            q_count = sum(len(s.questions) for s in d.suites)
            typer.echo(f"  {d.id:<20} {d.name:<30} {suite_count} suites, {q_count} questions")
        typer.echo()

    if custom:
        typer.echo("Custom Domains")
        typer.echo("=" * 60)
        for d in custom:
            suite_count = len(d.suites)
            q_count = sum(len(s.questions) for s in d.suites)
            typer.echo(f"  {d.id:<20} {d.name:<30} {suite_count} suites, {q_count} questions")
        typer.echo()

    typer.echo(f"Total: {len(predefined)} predefined, {len(custom)} custom")


@app.command("show")
def show_domain(
    domain_id: str = typer.Argument(
        ..., help="Domain ID to display, e.g. math, coding, or a custom ID."
    ),
) -> None:
    """Show detailed information about a domain.

    Displays the domain name, ID, type (predefined or custom),
    description, and lists all test suites with their details.

    Examples:

        polymind domain show math

        polymind domain show my_custom_domain
    """
    domain = load_domain_by_id(domain_id)

    if domain is None:
        typer.echo(f"Domain not found: {domain_id}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Domain: {domain.name}")
    typer.echo(f"ID:     {domain.id}")
    typer.echo(f"Type:   {'Custom' if domain.custom else 'Predefined'}")
    typer.echo(f"Description: {domain.description}")
    typer.echo()

    if not domain.suites:
        typer.echo("No test suites in this domain.")
        return

    typer.echo(f"Test Suites ({len(domain.suites)})")
    typer.echo("-" * 60)

    for suite in domain.suites:
        typer.echo(f"  {suite.id}")
        typer.echo(f"    Name:        {suite.name}")
        typer.echo(f"    Description: {suite.description}")
        typer.echo(f"    Difficulty:  {suite.difficulty}")
        typer.echo(f"    Questions:   {len(suite.questions)}")
        typer.echo()


@app.command("create")
def create_domain(
    domain_id: str = typer.Argument(
        ..., help="Unique ID for the new domain (letters, numbers, hyphens, underscores)."
    ),
    name: str = typer.Option(..., prompt=True, help="Human-readable display name for the domain."),
    description: str = typer.Option("", prompt=True, help="Description of what this domain tests."),
) -> None:
    """Create a new custom domain.

    Creates a new empty custom domain with a unique ID, name, and
    description. After creation, add suites using polymind suite create.

    Examples:

        polymind domain create my_domain --name "My Domain" --description "Custom tests"

        polymind domain create coding_eval --name "Coding Eval" --description "Programming tests"
    """
    # Validate ID
    if not domain_id.replace("_", "").replace("-", "").isalnum():
        typer.echo(
            "Domain ID must contain only letters, numbers, hyphens, and underscores.", err=True
        )
        raise typer.Exit(code=1)

    # Check if already exists
    existing = load_domain_by_id(domain_id)
    if existing is not None:
        typer.echo(f"Domain already exists: {domain_id}", err=True)
        raise typer.Exit(code=1)

    domain = Domain(
        id=domain_id,
        name=name,
        description=description,
        custom=True,
        suites=[],
    )

    path = save_custom_domain(domain)
    typer.echo(f"Created domain: {domain_id}")
    typer.echo(f"Saved to: {path}")
    typer.echo()
    typer.echo("Next steps:")
    typer.echo(f"  polymind suite create {domain_id}")


@app.command("delete")
def delete_domain(
    domain_id: str = typer.Argument(
        ..., help="Custom domain ID to delete (cannot delete predefined domains)."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Skip confirmation prompt and delete immediately."
    ),
) -> None:
    """Delete a custom domain. Cannot delete predefined domains.

    Removes a custom domain and all its suites and questions.
    Predefined domains cannot be deleted.

    Examples:

        polymind domain delete my_custom_domain

        polymind domain delete test_domain --force
    """
    # Check if predefined
    for d in get_all_domains():
        if d.id == domain_id:
            typer.echo(f"Cannot delete predefined domain: {domain_id}", err=True)
            raise typer.Exit(code=1)

    # Check if exists
    existing = load_domain_by_id(domain_id)
    if existing is None:
        typer.echo(f"Domain not found: {domain_id}", err=True)
        raise typer.Exit(code=1)

    if not force:
        typer.echo(f"About to delete domain: {existing.name}")
        if not typer.confirm("Are you sure?"):
            typer.echo("Cancelled.")
            return

    delete_custom_domain(domain_id)
    typer.echo(f"Deleted domain: {domain_id}")


@app.command("export")
def export_domain(
    domain_id: str = typer.Argument(..., help="Domain ID to export to a YAML file."),
    output: str = typer.Option(
        "", "--output", "-o", help="Output file path. Defaults to <domain_id>.yaml."
    ),
) -> None:
    """Export a domain to a YAML file.

    Exports a domain's full definition (suites, questions, settings)
    to a YAML file for sharing, backup, or manual editing.

    Examples:

        polymind domain export math

        polymind domain export my_domain -o backup.yaml
    """
    import yaml

    domain = load_domain_by_id(domain_id)
    if domain is None:
        typer.echo(f"Domain not found: {domain_id}", err=True)
        raise typer.Exit(code=1)

    output_path = output or f"{domain_id}.yaml"
    data = domain.to_dict()

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    typer.echo(f"Exported to: {output_path}")


@app.command("import")
def import_domain(
    file_path: str = typer.Argument(..., help="Path to the YAML file to import the domain from."),
) -> None:
    """Import a domain from a YAML file.

    Imports a domain definition from a YAML file, creating or
    overwriting the domain with all its suites and questions.

    Examples:

        polymind domain import my_domain.yaml

        polymind domain import backup.yaml
    """
    from pathlib import Path

    import yaml

    path = Path(file_path)
    if not path.exists():
        typer.echo(f"File not found: {file_path}", err=True)
        raise typer.Exit(code=1)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    from polymind.core.confidence.types import TestQuestion, TestSuite

    suites = []
    for suite_data in data.get("suites", []):
        questions = []
        for q_data in suite_data.get("questions", []):
            questions.append(
                TestQuestion(
                    id=q_data.get("id", ""),
                    prompt=q_data.get("prompt", ""),
                    expected=q_data.get("expected", ""),
                    evaluation=q_data.get("evaluation", "hybrid"),
                    keywords=q_data.get("keywords", []),
                    code=q_data.get("code"),
                    weight=q_data.get("weight", 1.0),
                    explanation=q_data.get("explanation", ""),
                )
            )
        suites.append(
            TestSuite(
                id=suite_data.get("id", ""),
                name=suite_data.get("name", ""),
                description=suite_data.get("description", ""),
                difficulty=suite_data.get("difficulty", "medium"),
                questions=questions,
            )
        )

    domain = Domain(
        id=data.get("id", path.stem),
        name=data.get("name", path.stem),
        description=data.get("description", ""),
        custom=True,
        suites=suites,
    )

    # Check if exists
    existing = load_domain_by_id(domain.id)
    if existing is not None:
        typer.echo(f"Domain already exists: {domain.id}. Overwriting.")

    save_custom_domain(domain)
    typer.echo(f"Imported domain: {domain.id}")
    typer.echo(
        f"  {len(domain.suites)} suites, {sum(len(s.questions) for s in domain.suites)} questions"
    )
