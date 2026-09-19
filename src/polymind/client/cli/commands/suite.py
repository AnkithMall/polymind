"""Suite management commands."""

import typer
from rich.console import Console

from polymind.core.confidence.artifact import (
    load_all_domains,
    load_domain_by_id,
    save_custom_domain,
)
from polymind.core.confidence.types import Domain, TestQuestion, TestSuite

app = typer.Typer()
console = Console()


@app.command("list")
def list_suites(
    domain_id: str = typer.Argument(
        "", help="Domain ID to filter suites by. Lists all domains if empty."
    ),
) -> None:
    """List test suites, optionally filtered by domain.

    Shows all test suites across all domains, or within a specific
    domain. Displays suite IDs, names, difficulty levels, and
    question counts.

    Examples:

        polymind suite list

        polymind suite list math
    """
    if domain_id:
        domain = load_domain_by_id(domain_id)
        if domain is None:
            typer.echo(f"Domain not found: {domain_id}", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"Suites in {domain.name}")
        typer.echo("=" * 60)
        _print_suites(domain.suites)
    else:
        domains = load_all_domains()
        for domain in domains:
            if domain.suites:
                typer.echo(f"\n{domain.name} ({domain.id})")
                typer.echo("-" * 40)
                _print_suites(domain.suites)


def _print_suites(suites: list[TestSuite]) -> None:
    for suite in suites:
        q_count = len(suite.questions)
        set(q.evaluation for q in suite.questions)
        typer.echo(f"  {suite.id:<30} {suite.name:<30} {suite.difficulty:<8} {q_count} questions")


@app.command("show")
def show_suite(
    suite_id: str = typer.Argument(..., help="Suite ID to display, e.g. math_arithmetic."),
    domain_id: str = typer.Option(
        "",
        "--domain",
        "-d",
        help="Domain ID to search in. Auto-detected from suite ID if not provided.",
    ),
) -> None:
    """Show detailed information about a test suite.

    Displays the suite's name, ID, domain, difficulty level, and
    lists all questions with their prompts, expected answers, and
    evaluation methods.

    Examples:

        polymind suite show math_arithmetic

        polymind suite show coding_basics -d coding
    """
    suite, domain = _find_suite(suite_id, domain_id)

    if suite is None:
        typer.echo(f"Suite not found: {suite_id}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Suite: {suite.name}")
    typer.echo(f"ID:         {suite.id}")
    typer.echo(f"Domain:     {domain.name if domain else 'unknown'}")
    typer.echo(f"Difficulty: {suite.difficulty}")
    typer.echo(f"Questions:  {len(suite.questions)}")
    typer.echo()

    for i, q in enumerate(suite.questions, 1):
        typer.echo(f"  {i}. [{q.evaluation}] {q.prompt}")
        typer.echo(f"     Expected: {q.expected[:80]}")
        if q.keywords:
            typer.echo(f"     Keywords: {', '.join(q.keywords[:5])}")
        typer.echo()


@app.command("create")
def create_suite(
    domain_id: str = typer.Argument(..., help="Custom domain ID to add the suite to."),
    suite_id: str = typer.Option(
        ..., prompt=True, help="Unique suite ID within the domain (letters, numbers, underscores)."
    ),
    name: str = typer.Option(..., prompt=True, help="Human-readable display name for the suite."),
    description: str = typer.Option("", prompt=True, help="Description of what this suite tests."),
    difficulty: str = typer.Option(
        "medium", prompt=True, help="Difficulty level: easy, medium, hard, or expert."
    ),
) -> None:
    """Create a new test suite in a domain.

    Creates a new empty test suite within a custom domain.
    After creation, add questions using polymind suite add-question.

    Examples:

        polymind suite create math --suite-id math_basics --name "Basic Math" --description "Simple arithmetic"

        polymind suite create coding --suite-id functions --name "Functions" --difficulty hard
    """
    domain = load_domain_by_id(domain_id)
    if domain is None:
        typer.echo(f"Domain not found: {domain_id}", err=True)
        raise typer.Exit(code=1)

    if not domain.custom:
        typer.echo(
            "Cannot add suites to predefined domains. Create a custom domain first.", err=True
        )
        raise typer.Exit(code=1)

    # Check suite ID uniqueness
    for s in domain.suites:
        if s.id == suite_id:
            typer.echo(f"Suite ID already exists in domain: {suite_id}", err=True)
            raise typer.Exit(code=1)

    if difficulty not in ("easy", "medium", "hard", "expert"):
        typer.echo("Difficulty must be one of: easy, medium, hard, expert", err=True)
        raise typer.Exit(code=1)

    suite = TestSuite(
        id=suite_id,
        name=name,
        description=description,
        difficulty=difficulty,
        questions=[],
    )

    domain.suites.append(suite)
    save_custom_domain(domain)

    typer.echo(f"Created suite: {suite_id}")
    typer.echo()
    typer.echo("Next steps:")
    typer.echo(f"  polymind suite add-question {domain_id} {suite_id}")


@app.command("add-question")
def add_question(
    domain_id: str = typer.Argument(..., help="Domain ID containing the suite."),
    suite_id: str = typer.Argument(..., help="Suite ID to add the question to."),
    question_id: str = typer.Option(..., prompt=True, help="Unique question ID within the suite."),
    prompt: str = typer.Option(..., prompt=True, help="The question text to ask the model."),
    expected: str = typer.Option(
        ..., prompt=True, help="Expected answer or reference answer for scoring."
    ),
    evaluation: str = typer.Option(
        "hybrid",
        prompt=True,
        help="Evaluation method: exact_match, keyword_match, code_execution, llm_judge, or hybrid.",
    ),
    keywords: str = typer.Option(
        "", prompt=False, help="Comma-separated keywords for keyword_match evaluation."
    ),
) -> None:
    """Add a question to a test suite.

    Adds a new test question with a prompt, expected answer, and
    evaluation method to an existing suite in a custom domain.

    Examples:

        polymind suite add-question math math_basics --question-id q1 --prompt "What is 2+2?" --expected "4"

        polymind suite add-question coding functions --question-id q2 --prompt "Write a function" --expected "def" --evaluation code_execution
    """
    domain = load_domain_by_id(domain_id)
    if domain is None:
        typer.echo(f"Domain not found: {domain_id}", err=True)
        raise typer.Exit(code=1)

    suite = None
    for s in domain.suites:
        if s.id == suite_id:
            suite = s
            break

    if suite is None:
        typer.echo(f"Suite not found: {suite_id}", err=True)
        raise typer.Exit(code=1)

    if evaluation not in ("exact_match", "keyword_match", "code_execution", "llm_judge", "hybrid"):
        typer.echo("Invalid evaluation method.", err=True)
        raise typer.Exit(code=1)

    # Check question ID uniqueness
    for q in suite.questions:
        if q.id == question_id:
            typer.echo(f"Question ID already exists: {question_id}", err=True)
            raise typer.Exit(code=1)

    keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]

    question = TestQuestion(
        id=question_id,
        prompt=prompt,
        expected=expected,
        evaluation=evaluation,
        keywords=keyword_list,
    )

    suite.questions.append(question)
    save_custom_domain(domain)

    typer.echo(f"Added question: {question_id}")
    typer.echo(f"Suite now has {len(suite.questions)} questions.")


@app.command("remove-question")
def remove_question(
    domain_id: str = typer.Argument(..., help="Domain ID containing the suite."),
    suite_id: str = typer.Argument(..., help="Suite ID containing the question."),
    question_id: str = typer.Argument(..., help="Question ID to remove from the suite."),
    force: bool = typer.Option(
        False, "--force", "-f", help="Skip confirmation prompt and remove immediately."
    ),
) -> None:
    """Remove a question from a test suite.

    Removes a specific question by ID from a test suite in a
    custom domain.

    Examples:

        polymind suite remove-question math math_basics q1

        polymind suite remove-question coding functions q2 --force
    """
    domain = load_domain_by_id(domain_id)
    if domain is None:
        typer.echo(f"Domain not found: {domain_id}", err=True)
        raise typer.Exit(code=1)

    suite = None
    for s in domain.suites:
        if s.id == suite_id:
            suite = s
            break

    if suite is None:
        typer.echo(f"Suite not found: {suite_id}", err=True)
        raise typer.Exit(code=1)

    found = False
    for q in suite.questions:
        if q.id == question_id:
            found = True
            break

    if not found:
        typer.echo(f"Question not found: {question_id}", err=True)
        raise typer.Exit(code=1)

    if not force:
        if not typer.confirm(f"Remove question {question_id}?"):
            typer.echo("Cancelled.")
            return

    suite.questions = [q for q in suite.questions if q.id != question_id]
    save_custom_domain(domain)

    typer.echo(f"Removed question: {question_id}")
    typer.echo(f"Suite now has {len(suite.questions)} questions.")


@app.command("delete")
def delete_suite(
    domain_id: str = typer.Argument(..., help="Custom domain ID containing the suite."),
    suite_id: str = typer.Argument(..., help="Suite ID to delete (with all its questions)."),
    force: bool = typer.Option(
        False, "--force", "-f", help="Skip confirmation prompt and delete immediately."
    ),
) -> None:
    """Delete a test suite from a domain.

    Removes a test suite and all its questions from a custom domain.
    Cannot delete suites from predefined domains.

    Examples:

        polymind suite delete math math_basics

        polymind suite delete coding functions --force
    """
    domain = load_domain_by_id(domain_id)
    if domain is None:
        typer.echo(f"Domain not found: {domain_id}", err=True)
        raise typer.Exit(code=1)

    if not domain.custom:
        typer.echo("Cannot delete suites from predefined domains.", err=True)
        raise typer.Exit(code=1)

    found = False
    for s in domain.suites:
        if s.id == suite_id:
            found = True
            break

    if not found:
        typer.echo(f"Suite not found: {suite_id}", err=True)
        raise typer.Exit(code=1)

    if not force:
        if not typer.confirm(f"Delete suite {suite_id}?"):
            typer.echo("Cancelled.")
            return

    domain.suites = [s for s in domain.suites if s.id != suite_id]
    save_custom_domain(domain)

    typer.echo(f"Deleted suite: {suite_id}")


def _find_suite(
    suite_id: str,
    domain_id: str = "",
) -> tuple[TestSuite | None, Domain | None]:
    """Find a suite by ID, optionally within a specific domain."""
    if domain_id:
        domain = load_domain_by_id(domain_id)
        if domain is not None:
            for s in domain.suites:
                if s.id == suite_id:
                    return s, domain
        return None, domain

    # Search all domains
    for domain in load_all_domains():
        for s in domain.suites:
            if s.id == suite_id:
                return s, domain

    return None, None


@app.command("edit")
def edit_suite(
    suite_id: str = typer.Argument(..., help="Suite ID to edit."),
    name: str = typer.Option("", "--name", "-n", help="New display name."),
    description: str = typer.Option("", "--description", "-d", help="New description."),
    difficulty: str = typer.Option(
        "", "--difficulty", help="New difficulty: easy, medium, hard, expert."
    ),
    domain_id: str = typer.Option("", "--domain", help="Domain ID (auto-detected if not set)."),
) -> None:
    """Edit a test suite's metadata.

    Updates the name, description, or difficulty of an existing suite.
    Cannot edit suites in predefined domains.

    Examples:

        polymind suite edit frontend-core --name "Frontend Core Skills"

        polymind suite edit backend-api --difficulty hard
    """
    suite, domain = _find_suite(suite_id, domain_id)

    if suite is None:
        typer.echo(f"Suite not found: {suite_id}", err=True)
        raise typer.Exit(code=1)

    if domain is not None and not domain.custom:
        typer.echo("Cannot edit suites in predefined domains.", err=True)
        raise typer.Exit(code=1)

    if name:
        suite.name = name
    if description:
        suite.description = description
    if difficulty:
        if difficulty not in ("easy", "medium", "hard", "expert"):
            typer.echo("Difficulty must be one of: easy, medium, hard, expert", err=True)
            raise typer.Exit(code=1)
        suite.difficulty = difficulty

    if domain is not None:
        save_custom_domain(domain)

    typer.echo(f"Updated suite: {suite_id}")


@app.command("edit-question")
def edit_question(
    domain_id: str = typer.Argument(..., help="Domain ID containing the suite."),
    suite_id: str = typer.Argument(..., help="Suite ID containing the question."),
    question_id: str = typer.Argument(..., help="Question ID to edit."),
    prompt: str = typer.Option("", "--prompt", "-p", help="New question prompt."),
    expected: str = typer.Option("", "--expected", "-e", help="New expected answer."),
    evaluation: str = typer.Option(
        "", "--evaluation", help="New evaluation method."
    ),
    keywords: str = typer.Option(
        "", "--keywords", help="New comma-separated keywords."
    ),
) -> None:
    """Edit a question in a test suite.

    Updates the prompt, expected answer, evaluation method, or
    keywords of an existing question.

    Examples:

        polymind suite edit-question frontend frontend-core q1 --prompt "What is React?"

        polymind suite edit-question backend backend-api q2 --expected "REST API" --evaluation keyword_match
    """
    domain = load_domain_by_id(domain_id)
    if domain is None:
        typer.echo(f"Domain not found: {domain_id}", err=True)
        raise typer.Exit(code=1)

    suite = None
    for s in domain.suites:
        if s.id == suite_id:
            suite = s
            break

    if suite is None:
        typer.echo(f"Suite not found: {suite_id}", err=True)
        raise typer.Exit(code=1)

    question = None
    for q in suite.questions:
        if q.id == question_id:
            question = q
            break

    if question is None:
        typer.echo(f"Question not found: {question_id}", err=True)
        raise typer.Exit(code=1)

    if prompt:
        question.prompt = prompt
    if expected:
        question.expected = expected
    if evaluation:
        if evaluation not in ("exact_match", "keyword_match", "code_execution", "llm_judge", "hybrid"):
            typer.echo("Invalid evaluation method.", err=True)
            raise typer.Exit(code=1)
        question.evaluation = evaluation
    if keywords:
        question.keywords = [k.strip() for k in keywords.split(",") if k.strip()]

    save_custom_domain(domain)
    typer.echo(f"Updated question: {question_id}")


@app.callback(invoke_without_command=True)
def suite_callback(
    ctx: typer.Context,
) -> None:
    """Suite manager — manage test suites interactively.

    Without a subcommand, opens an interactive menu.
    """
    if ctx.invoked_subcommand is not None:
        return

    while True:
        console.print()
        console.print("[bold]PolyMind Suite Manager[/]")
        console.print()
        console.print("  1. List suites")
        console.print("  2. Show suite")
        console.print("  3. Create suite")
        console.print("  4. Edit suite")
        console.print("  5. Delete suite")
        console.print("  6. Add question")
        console.print("  7. Edit question")
        console.print("  8. Remove question")
        console.print("  9. Exit")
        console.print()

        try:
            choice = input("  Choose [1-9]: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  Goodbye.")
            break

        if choice == "1":
            _interactive_list()
        elif choice == "2":
            _interactive_show()
        elif choice == "3":
            _interactive_create()
        elif choice == "4":
            _interactive_edit()
        elif choice == "5":
            _interactive_delete()
        elif choice == "6":
            _interactive_add_question()
        elif choice == "7":
            _interactive_edit_question()
        elif choice == "8":
            _interactive_remove_question()
        elif choice == "9":
            console.print("  Goodbye.")
            break
        else:
            console.print("  [yellow]Invalid choice.[/]")


def _interactive_list() -> None:
    domains = load_all_domains()
    for domain in domains:
        if domain.suites:
            console.print(f"\n  {domain.name} ({domain.id})")
            console.print("  " + "-" * 40)
            _print_suites(domain.suites)


def _interactive_show() -> None:
    suite_id = input("  Suite ID: ").strip()
    suite, domain = _find_suite(suite_id)
    if suite is None:
        console.print(f"  [red]Suite not found: {suite_id}[/]")
        return
    console.print(f"  Suite: {suite.name}")
    console.print(f"  ID:         {suite.id}")
    console.print(f"  Domain:     {domain.name if domain else 'unknown'}")
    console.print(f"  Difficulty: {suite.difficulty}")
    console.print(f"  Questions:  {len(suite.questions)}")
    console.print()
    for i, q in enumerate(suite.questions, 1):
        console.print(f"    {i}. [{q.evaluation}] {q.prompt}")
        console.print(f"       Expected: {q.expected[:80]}")


def _interactive_create() -> None:
    console.print("  [bold]Create New Suite[/]")
    domain_id = input("  Domain ID: ").strip()
    domain = load_domain_by_id(domain_id)
    if domain is None:
        console.print(f"  [red]Domain not found: {domain_id}[/]")
        return
    if not domain.custom:
        console.print("  [red]Cannot add suites to predefined domains.[/]")
        return

    suite_id = input("  Suite ID: ").strip()
    name = input("  Name: ").strip() or suite_id
    description = input("  Description: ").strip()
    difficulty = input("  Difficulty [medium]: ").strip() or "medium"

    if difficulty not in ("easy", "medium", "hard", "expert"):
        console.print("  [red]Invalid difficulty.[/]")
        return

    for s in domain.suites:
        if s.id == suite_id:
            console.print(f"  [red]Suite ID already exists: {suite_id}[/]")
            return

    suite = TestSuite(id=suite_id, name=name, description=description, difficulty=difficulty)
    domain.suites.append(suite)
    save_custom_domain(domain)
    console.print(f"  [green]✓ Created suite '{suite_id}'[/]")


def _interactive_edit() -> None:
    suite_id = input("  Suite ID to edit: ").strip()
    suite, domain = _find_suite(suite_id)
    if suite is None:
        console.print(f"  [red]Suite not found: {suite_id}[/]")
        return
    if domain and not domain.custom:
        console.print("  [red]Cannot edit predefined suites.[/]")
        return

    console.print(f"  Current: {suite.name} ({suite.difficulty})")
    console.print("  (Press Enter to keep current value)")

    name = input(f"  Name [{suite.name}]: ").strip()
    if name:
        suite.name = name

    desc = input(f"  Description [{suite.description}]: ").strip()
    if desc:
        suite.description = desc

    diff = input(f"  Difficulty [{suite.difficulty}]: ").strip()
    if diff and diff in ("easy", "medium", "hard", "expert"):
        suite.difficulty = diff

    if domain:
        save_custom_domain(domain)
    console.print(f"  [green]✓ Updated suite '{suite_id}'[/]")


def _interactive_delete() -> None:
    suite_id = input("  Suite ID to delete: ").strip()
    domain_id = input("  Domain ID: ").strip()
    domain = load_domain_by_id(domain_id)
    if domain is None:
        console.print(f"  [red]Domain not found: {domain_id}[/]")
        return
    if not domain.custom:
        console.print("  [red]Cannot delete suites from predefined domains.[/]")
        return

    found = any(s.id == suite_id for s in domain.suites)
    if not found:
        console.print(f"  [red]Suite not found: {suite_id}[/]")
        return

    confirm = input(f"  Delete '{suite_id}'? [y/N]: ").strip().lower()
    if confirm == "y":
        domain.suites = [s for s in domain.suites if s.id != suite_id]
        save_custom_domain(domain)
        console.print(f"  [green]✓ Deleted suite '{suite_id}'[/]")


def _interactive_add_question() -> None:
    console.print("  [bold]Add Question[/]")
    domain_id = input("  Domain ID: ").strip()
    suite_id = input("  Suite ID: ").strip()
    domain = load_domain_by_id(domain_id)
    if domain is None:
        console.print(f"  [red]Domain not found: {domain_id}[/]")
        return

    suite = None
    for s in domain.suites:
        if s.id == suite_id:
            suite = s
            break
    if suite is None:
        console.print(f"  [red]Suite not found: {suite_id}[/]")
        return

    q_id = input("  Question ID: ").strip()
    prompt_text = input("  Prompt: ").strip()
    expected = input("  Expected answer: ").strip()
    evaluation = input("  Evaluation [hybrid]: ").strip() or "hybrid"
    keywords_raw = input("  Keywords (comma-separated): ").strip()
    keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()] if keywords_raw else []

    question = TestQuestion(id=q_id, prompt=prompt_text, expected=expected, evaluation=evaluation, keywords=keywords)
    suite.questions.append(question)
    save_custom_domain(domain)
    console.print(f"  [green]✓ Added question '{q_id}'[/]")


def _interactive_edit_question() -> None:
    domain_id = input("  Domain ID: ").strip()
    suite_id = input("  Suite ID: ").strip()
    question_id = input("  Question ID: ").strip()

    domain = load_domain_by_id(domain_id)
    if domain is None:
        console.print(f"  [red]Domain not found: {domain_id}[/]")
        return

    suite = None
    for s in domain.suites:
        if s.id == suite_id:
            suite = s
            break
    if suite is None:
        console.print(f"  [red]Suite not found: {suite_id}[/]")
        return

    question = None
    for q in suite.questions:
        if q.id == question_id:
            question = q
            break
    if question is None:
        console.print(f"  [red]Question not found: {question_id}[/]")
        return

    console.print(f"  Current prompt: {question.prompt}")
    console.print("  (Press Enter to keep current value)")

    p = input(f"  Prompt: ").strip()
    if p:
        question.prompt = p

    e = input(f"  Expected: ").strip()
    if e:
        question.expected = e

    ev = input(f"  Evaluation [{question.evaluation}]: ").strip()
    if ev:
        question.evaluation = ev

    kw = input(f"  Keywords [{', '.join(question.keywords)}]: ").strip()
    if kw:
        question.keywords = [k.strip() for k in kw.split(",") if k.strip()]

    save_custom_domain(domain)
    console.print(f"  [green]✓ Updated question '{question_id}'[/]")


def _interactive_remove_question() -> None:
    domain_id = input("  Domain ID: ").strip()
    suite_id = input("  Suite ID: ").strip()
    question_id = input("  Question ID: ").strip()

    domain = load_domain_by_id(domain_id)
    if domain is None:
        console.print(f"  [red]Domain not found: {domain_id}[/]")
        return

    suite = None
    for s in domain.suites:
        if s.id == suite_id:
            suite = s
            break
    if suite is None:
        console.print(f"  [red]Suite not found: {suite_id}[/]")
        return

    found = any(q.id == question_id for q in suite.questions)
    if not found:
        console.print(f"  [red]Question not found: {question_id}[/]")
        return

    confirm = input(f"  Remove question '{question_id}'? [y/N]: ").strip().lower()
    if confirm == "y":
        suite.questions = [q for q in suite.questions if q.id != question_id]
        save_custom_domain(domain)
        console.print(f"  [green]✓ Removed question '{question_id}'[/]")
