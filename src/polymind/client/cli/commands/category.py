"""Category management commands.

Categories are the user-friendly layer over the domain system.
They reuse domain persistence but provide a more intuitive UX.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from polymind.core.confidence.artifact import (
    delete_custom_domain,
    load_all_domains,
    load_domain_by_id,
    save_custom_domain,
)
from polymind.core.confidence.suites import get_all_domains
from polymind.core.confidence.types import Domain

app = typer.Typer()
console = Console()


def _print_category(domain: Domain) -> None:
    """Print a single category's details."""
    suite_count = len(domain.suites)
    q_count = sum(len(s.questions) for s in domain.suites)
    marker = " [custom]" if domain.custom else " [built-in]"
    aliases_str = ", ".join(domain.aliases) if domain.aliases else "(none)"

    console.print(f"  [bold cyan]{domain.id}[/]{marker}")
    console.print(f"    Name:        {domain.name}")
    console.print(f"    Description: {domain.description}")
    console.print(f"    Aliases:     {aliases_str}")
    console.print(f"    Suites:      {suite_count} ({q_count} questions)")
    if domain.suites:
        for s in domain.suites:
            console.print(f"      - {s.id}: {s.name} ({s.difficulty}, {len(s.questions)} questions)")
    console.print()


@app.callback(invoke_without_command=True)
def category_callback(
    ctx: typer.Context,
) -> None:
    """Category manager — manage scoring domains interactively.

    Without a subcommand, opens an interactive menu for browsing,
    creating, editing, and deleting categories.
    """
    if ctx.invoked_subcommand is not None:
        return

    while True:
        console.print()
        console.print("[bold]PolyMind Category Manager[/]")
        console.print()
        console.print("  1. List categories")
        console.print("  2. Show category")
        console.print("  3. Add category")
        console.print("  4. Edit category")
        console.print("  5. Delete category")
        console.print("  6. Exit")
        console.print()

        try:
            choice = input("  Choose [1-6]: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  Goodbye.")
            break

        if choice == "1":
            _interactive_list()
        elif choice == "2":
            _interactive_show()
        elif choice == "3":
            _interactive_add()
        elif choice == "4":
            _interactive_edit()
        elif choice == "5":
            _interactive_delete()
        elif choice == "6":
            console.print("  Goodbye.")
            break
        else:
            console.print("  [yellow]Invalid choice.[/]")


def _interactive_list() -> None:
    """Interactive: list all categories."""
    domains = load_all_domains()
    _display_categories(domains)


def _interactive_show() -> None:
    """Interactive: show a specific category."""
    domain_id = input("  Category ID: ").strip()
    domain = load_domain_by_id(domain_id)
    if domain is None:
        console.print(f"  [red]Category not found: {domain_id}[/]")
        return
    _print_category(domain)


def _interactive_add() -> None:
    """Interactive: add a new custom category."""
    console.print("  [bold]Add New Category[/]")
    console.print()

    cat_id = input("  ID (lowercase, no spaces): ").strip()
    if not cat_id:
        console.print("  [red]ID is required.[/]")
        return

    # Check if already exists
    existing = load_domain_by_id(cat_id)
    if existing is not None:
        console.print(f"  [red]Category '{cat_id}' already exists.[/]")
        return

    name = input("  Display name: ").strip() or cat_id
    description = input("  Description: ").strip()
    aliases_raw = input("  Aliases (comma-separated, or empty): ").strip()
    aliases = [a.strip() for a in aliases_raw.split(",") if a.strip()] if aliases_raw else []

    domain = Domain(
        id=cat_id,
        name=name,
        description=description,
        custom=True,
        aliases=aliases,
    )
    save_custom_domain(domain)
    console.print(f"  [green]✓ Created category '{cat_id}'[/]")


def _interactive_edit() -> None:
    """Interactive: edit an existing custom category."""
    cat_id = input("  Category ID to edit: ").strip()
    domain = load_domain_by_id(cat_id)
    if domain is None:
        console.print(f"  [red]Category not found: {cat_id}[/]")
        return

    if not domain.custom:
        console.print(f"  [yellow]Category '{cat_id}' is built-in. Only aliases can be edited.[/]")

    console.print(f"  Current: {domain.name}")
    console.print(f"  Description: {domain.description}")
    console.print(f"  Aliases: {', '.join(domain.aliases) if domain.aliases else '(none)'}")
    console.print()
    console.print("  (Press Enter to keep current value)")

    name = input(f"  Name [{domain.name}]: ").strip()
    if name:
        domain.name = name

    desc = input(f"  Description [{domain.description}]: ").strip()
    if desc:
        domain.description = desc

    aliases_raw = input(
        f"  Aliases [{', '.join(domain.aliases) if domain.aliases else '(none)'}]: "
    ).strip()
    if aliases_raw:
        domain.aliases = [a.strip() for a in aliases_raw.split(",") if a.strip()]

    save_custom_domain(domain)
    console.print(f"  [green]✓ Updated category '{cat_id}'[/]")


def _interactive_delete() -> None:
    """Interactive: delete a custom category."""
    cat_id = input("  Category ID to delete: ").strip()
    domain = load_domain_by_id(cat_id)
    if domain is None:
        console.print(f"  [red]Category not found: {cat_id}[/]")
        return

    if not domain.custom:
        console.print(f"  [red]Cannot delete built-in category '{cat_id}'. Built-in categories are protected.[/]")
        return

    confirm = input(f"  Delete '{cat_id}'? This cannot be undone. [y/N]: ").strip().lower()
    if confirm == "y":
        delete_custom_domain(cat_id)
        console.print(f"  [green]✓ Deleted category '{cat_id}'[/]")
    else:
        console.print("  Cancelled.")


def _display_categories(domains: list[Domain]) -> None:
    """Display categories in a table."""
    if not domains:
        console.print("  No categories found.")
        return

    predefined = [d for d in domains if not d.custom]
    custom = [d for d in domains if d.custom]

    if predefined:
        table = Table(title="Built-in Categories", show_lines=False)
        table.add_column("ID", style="cyan")
        table.add_column("Name")
        table.add_column("Aliases", style="dim")
        table.add_column("Suites", justify="right")
        table.add_column("Questions", justify="right")

        for d in predefined:
            aliases_str = ", ".join(d.aliases[:3]) if d.aliases else "-"
            if len(d.aliases) > 3:
                aliases_str += f" +{len(d.aliases) - 3}"
            suite_count = len(d.suites)
            q_count = sum(len(s.questions) for s in d.suites)
            table.add_row(d.id, d.name, aliases_str, str(suite_count), str(q_count))

        console.print(table)

    if custom:
        table = Table(title="Custom Categories", show_lines=False)
        table.add_column("ID", style="cyan")
        table.add_column("Name")
        table.add_column("Aliases", style="dim")
        table.add_column("Suites", justify="right")
        table.add_column("Questions", justify="right")

        for d in custom:
            aliases_str = ", ".join(d.aliases[:3]) if d.aliases else "-"
            if len(d.aliases) > 3:
                aliases_str += f" +{len(d.aliases) - 3}"
            suite_count = len(d.suites)
            q_count = sum(len(s.questions) for s in d.suites)
            table.add_row(d.id, d.name, aliases_str, str(suite_count), str(q_count))

        console.print(table)
    elif not predefined:
        console.print("  No custom categories yet. Use 'polymind category add' to create one.")

    console.print(f"\n  Total: {len(predefined)} built-in, {len(custom)} custom")


@app.command("list")
def list_categories() -> None:
    """List all categories (built-in and custom).

    Shows category IDs, names, aliases, suite counts, and question counts.

    Examples:

        polymind category list
    """
    domains = load_all_domains()
    _display_categories(domains)


@app.command("show")
def show_category(
    category_id: str = typer.Argument(..., help="Category ID to display."),
) -> None:
    """Show details for a specific category.

    Displays the category's name, description, aliases, and all
    associated suites with their questions.

    Examples:

        polymind category show coding

        polymind category show frontend
    """
    domain = load_domain_by_id(category_id)
    if domain is None:
        console.print(f"[red]Category not found: {category_id}[/]")
        raise typer.Exit(code=1)

    _print_category(domain)


@app.command("add")
def add_category(
    category_id: str = typer.Argument(..., help="ID for the new category (lowercase, no spaces)."),
    name: str = typer.Option("", "--name", "-n", help="Display name (defaults to ID)."),
    description: str = typer.Option("", "--description", "-d", help="Description of the category."),
    aliases: str = typer.Option(
        "", "--aliases", "-a", help="Comma-separated routing keywords/aliases."
    ),
) -> None:
    """Add a new custom category.

    Creates a custom scoring domain that can be used for confidence
    evaluation and pipeline routing.

    Examples:

        polymind category add frontend --name "Frontend Development" \\
            --description "UI engineering, React, TypeScript" \\
            --aliases "ui,react,typescript,browser"

        polymind category add backend
    """
    existing = load_domain_by_id(category_id)
    if existing is not None:
        console.print(f"[red]Category '{category_id}' already exists.[/]")
        raise typer.Exit(code=1)

    display_name = name or category_id
    alias_list = [a.strip() for a in aliases.split(",") if a.strip()] if aliases else []

    domain = Domain(
        id=category_id,
        name=display_name,
        description=description,
        custom=True,
        aliases=alias_list,
    )
    save_custom_domain(domain)
    console.print(f"[green]✓ Created category '{category_id}'[/]")


@app.command("edit")
def edit_category(
    category_id: str = typer.Argument(..., help="Category ID to edit."),
    name: str = typer.Option("", "--name", "-n", help="New display name."),
    description: str = typer.Option("", "--description", "-d", help="New description."),
    aliases: str = typer.Option(
        "", "--aliases", "-a", help="New comma-separated aliases (replaces existing)."
    ),
) -> None:
    """Edit an existing category.

    Built-in categories can have their name, description, and aliases
    updated. Custom categories can be fully edited.

    Examples:

        polymind category edit frontend --name "Frontend Engineering"

        polymind category edit backend --aliases "api,rest,graphql,node"
    """
    domain = load_domain_by_id(category_id)
    if domain is None:
        console.print(f"[red]Category not found: {category_id}[/]")
        raise typer.Exit(code=1)

    if name:
        domain.name = name
    if description:
        domain.description = description
    if aliases:
        domain.aliases = [a.strip() for a in aliases.split(",") if a.strip()]

    save_custom_domain(domain)
    console.print(f"[green]✓ Updated category '{category_id}'[/]")


@app.command("delete")
def delete_category(
    category_id: str = typer.Argument(..., help="Category ID to delete."),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt."),
) -> None:
    """Delete a custom category.

    Built-in categories cannot be deleted. Custom categories can be
    removed with confirmation (or use --force to skip).

    Examples:

        polymind category delete frontend

        polymind category delete backend --force
    """
    domain = load_domain_by_id(category_id)
    if domain is None:
        console.print(f"[red]Category not found: {category_id}[/]")
        raise typer.Exit(code=1)

    if not domain.custom:
        console.print(
            f"[red]Cannot delete predefined category '{category_id}'. "
            "Built-in categories are protected.[/]"
        )
        raise typer.Exit(code=1)

    if not force:
        if not typer.confirm(f"Delete category '{category_id}'? This cannot be undone."):
            console.print("Cancelled.")
            return

    delete_custom_domain(category_id)
    console.print(f"[green]✓ Deleted category '{category_id}'[/]")


@app.command("clone")
def clone_category(
    category_id: str = typer.Argument(..., help="Category ID to clone."),
    new_id: str = typer.Argument(..., help="ID for the cloned category."),
) -> None:
    """Clone an existing category as a new custom category.

    Creates a copy of the specified category with a new ID. Useful
    for creating variations of existing categories.

    Examples:

        polymind category clone coding python-coding
    """
    source = load_domain_by_id(category_id)
    if source is None:
        console.print(f"[red]Category not found: {category_id}[/]")
        raise typer.Exit(code=1)

    existing = load_domain_by_id(new_id)
    if existing is not None:
        console.print(f"[red]Category '{new_id}' already exists.[/]")
        raise typer.Exit(code=1)

    clone = Domain(
        id=new_id,
        name=f"{source.name} (copy)",
        description=source.description,
        custom=True,
        aliases=list(source.aliases),
        suites=source.suites,
    )
    save_custom_domain(clone)
    console.print(f"[green]✓ Cloned '{category_id}' → '{new_id}'[/]")
