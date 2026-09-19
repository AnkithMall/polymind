"""Confidence scoring commands."""

from pathlib import Path

import typer
from rich.console import Console

from polymind.core.confidence.artifact import (
    compute_domain_hash,
    load_all_domains,
    load_confidence,
    load_domain_by_id,
    load_domain_hashes,
    save_confidence,
)
from polymind.core.confidence.scorer import (
    generate_test_prompt,
    score_question,
)
from polymind.core.confidence.types import (
    DomainScore,
    ModelConfidence,
    SuiteScore,
)
from polymind.core.model.registry import ModelRegistry

app = typer.Typer()
console = Console()


@app.command("compute")
def compute_confidence(
    model_id: str = typer.Option(
        "",
        "--model",
        "-m",
        help="Model ID (number) to score. Scores all installed models if empty.",
    ),
    domain_id: str = typer.Option(
        "", "--domain", "-d", help="Domain ID to score against. Uses all domains if empty."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force recomputation even if domain suites haven't changed."
    ),
    output_json: bool = typer.Option(
        False, "--json", help="Output full result as JSON instead of formatted text."
    ),
) -> None:
    """Compute capability scores for models across domains.

    Runs model inference against test suites and computes accuracy
    scores. The command is idempotent — if a domain's suites and
    questions haven't changed since the last evaluation, the scores
    are reused (skipping expensive inference). Use --force to
    recompute everything.

    Scoring terminology:
      Task Score    — accuracy for a single question (0.0–1.0)
      Suite Score   — weighted average of task scores in a suite
      Domain Score  — weighted average of suite scores in a domain
      Overall Score — average of all domain scores for a model
      Confidence    — how much evidence supports a score
                      (more questions = higher confidence)

    Results are saved to .polymind/confidence.yaml with content
    hashes that detect when suites change.

    Examples:

        polymind confidence compute -m 1

        polymind confidence compute --domain math

        polymind confidence compute --model 2 --domain coding

        polymind confidence compute --force
    """
    registry = ModelRegistry()
    models = registry.load()

    if not models:
        typer.echo("No models installed. Use 'polymind model download' first.", err=True)
        raise typer.Exit(code=1)

    # Filter models
    if model_id:
        selected = [m for m in models if str(m.id) == model_id]
        if not selected:
            typer.echo(f"Model not found: {model_id}", err=True)
            raise typer.Exit(code=1)
    else:
        selected = models

    # Load domains
    if domain_id:
        domain = load_domain_by_id(domain_id)
        if domain is None:
            typer.echo(f"Domain not found: {domain_id}", err=True)
            raise typer.Exit(code=1)
        domains = [domain]
    else:
        domains = load_all_domains()

    if not domains:
        typer.echo("No domains found.", err=True)
        raise typer.Exit(code=1)

    # Load existing scores and hashes
    all_scores = load_confidence()
    stored_hashes = load_domain_hashes()

    # Determine which domains need recomputation
    skipped_domains: list[str] = []
    domains_to_compute: list[tuple[Domain, str]] = []  # (domain, current_hash)

    for domain in domains:
        current_hash = compute_domain_hash(domain)
        stored = stored_hashes.get(domain.id)

        if not force and stored == current_hash and domain.id in {
            did
            for conf in all_scores.values()
            for did in conf.domains
        }:
            skipped_domains.append(domain.id)
        else:
            domains_to_compute.append((domain, current_hash))

    if skipped_domains:
        typer.echo("Skipped (unchanged):")
        for did in skipped_domains:
            typer.echo(f"  {did}")
        typer.echo()

    if not domains_to_compute:
        typer.echo("All domain scores are up to date. Use --force to recompute.")
        return

    typer.echo(f"Computing scores for {len(domains_to_compute)} domain(s)...")
    typer.echo()

    # Track new hashes
    new_hashes = dict(stored_hashes)

    # Compute for each model
    for model in selected:
        typer.echo()
        typer.echo(f"Model: {model.filename} (ID: {model.id})")
        typer.echo("=" * 60)

        model_path = Path(model.local_path)
        if not model_path.exists():
            typer.echo(f"  File not found, skipping: {model.local_path}")
            continue

        try:
            from llama_cpp import Llama

            typer.echo("  Loading model...")
            llm = Llama(
                model_path=str(model_path),
                n_gpu_layers=0,
                n_threads=4,
                n_ctx=2048,
                verbose=False,
            )
        except Exception as e:
            typer.echo(f"  Failed to load model: {e}")
            continue

        # Preserve existing scores for skipped domains
        existing_conf = all_scores.get(str(model.id))
        if existing_conf:
            model_conf = ModelConfidence(
                model_id=str(model.id),
                domains=dict(existing_conf.domains),
                best_domain=existing_conf.best_domain,
                overall_score=existing_conf.overall_score,
            )
        else:
            model_conf = ModelConfidence(model_id=str(model.id))

        for domain, domain_hash in domains_to_compute:
            typer.echo(f"\n  Domain: {domain.name}")
            typer.echo(f"  {'-' * 50}")

            domain_scores: dict[str, SuiteScore] = {}

            for suite in domain.suites:
                typer.echo(f"    Suite: {suite.name} ({suite.difficulty})")

                responses: dict[str, str] = {}

                for question in suite.questions:
                    prompt = generate_test_prompt(question)

                    try:
                        output = llm(
                            prompt,
                            max_tokens=256,
                            temperature=0.0,
                            stop=["\n\n", "```"],
                        )
                        response_text = output["choices"][0]["text"].strip()
                    except Exception as e:
                        response_text = ""
                        typer.echo(f"      Error on {question.id}: {e}")

                    responses[question.id] = response_text

                suite_score = score_suite_from_responses(responses, suite)
                domain_scores[suite.id] = suite_score

                typer.echo(
                    f"      Score: {suite_score.score:.1f}% "
                    f"({suite_score.passed}/{suite_score.total} passed)"
                )

            # Compute domain overall
            if domain_scores:
                avg = sum(s.score for s in domain_scores.values()) / len(domain_scores)
            else:
                avg = 0.0

            domain_score = DomainScore(
                domain_id=domain.id,
                overall=round(avg, 2),
                suites=domain_scores,
            )
            model_conf.domains[domain.id] = domain_score
            new_hashes[domain.id] = domain_hash

            typer.echo(f"    Domain Overall: {avg:.1f}%")

        # Compute overall and best domain
        if model_conf.domains:
            overall = sum(d.overall for d in model_conf.domains.values()) / len(model_conf.domains)
            best = max(model_conf.domains.items(), key=lambda x: x[1].overall)
            model_conf.overall_score = round(overall, 2)
            model_conf.best_domain = best[0]
        else:
            model_conf.overall_score = 0.0
            model_conf.best_domain = ""

        all_scores[str(model.id)] = model_conf

        typer.echo()
        typer.echo(f"  Overall Score: {model_conf.overall_score:.1f}%")
        typer.echo(f"  Best Domain:   {model_conf.best_domain}")

        # Free memory
        del llm

    # Save results with hashes
    save_confidence(all_scores, domain_hashes=new_hashes)
    typer.echo()
    typer.echo("Confidence scores saved to .polymind/confidence.yaml")


@app.command("show")
def show_confidence(
    model_id: str = typer.Option("", "--model", "-m", help="Model ID (number) to show scores for."),
    domain_id: str = typer.Option(
        "", "--domain", "-d", help="Domain ID to show detailed breakdown for."
    ),
    explain: bool = typer.Option(
        False, "--explain", help="Show explanation of scoring terms."
    ),
    output_json: bool = typer.Option(
        False, "--json", help="Output full result as JSON instead of formatted text."
    ),
) -> None:
    """Show confidence and capability scores.

    Displays previously computed confidence scores for models,
    with optional filtering by model or domain. Shows overall
    scores, best domain, and per-domain breakdowns.

    Scoring terminology:

      Task Score:  Accuracy for a single question (0.0–1.0 or 0–100%).
                   Higher means the model's answer matched the expected
                   answer more closely.

      Suite Score: Weighted average of task scores within a test suite.
                   A suite groups related questions (e.g., "Basic Arithmetic").
                   Score = sum(task_score × task_weight) / sum(task_weight).

      Domain Score: Weighted average of suite scores for a domain.
                    A domain groups related suites (e.g., "Mathematics").
                    Score = sum(suite_score × suite_weight) / sum(suite_weight).

      Overall Score: Average of all domain scores for a model.
                     Represents the model's general capability across
                     all evaluated domains.

      Confidence: How much evidence supports the score.
                  More questions and suites = higher confidence.
                  A score from 5 questions is less reliable than
                  one from 50 questions.

    Examples:

        polymind confidence show

        polymind confidence show -m 1

        polymind confidence show -d math --json

        polymind confidence show --explain
    """
    if explain:
        console.print("[bold]Scoring Concepts[/]")
        console.print()
        console.print("  [cyan]Task Score[/]     0.0–1.0 accuracy on a single question.")
        console.print("                   Computed by evaluation methods:")
        console.print("                     • exact_match  — text comparison with fuzzy matching")
        console.print("                     • keyword_match — checks required keywords are present")
        console.print("                     • code_execution — runs generated code in sandbox")
        console.print("                     • llm_judge — heuristic response quality grading")
        console.print("                     • hybrid — runs ALL methods, takes the best score")
        console.print()
        console.print("  [cyan]Suite Score[/]    Weighted average of task scores within a suite.")
        console.print("                   Each suite groups related questions at a")
        console.print("                   consistent difficulty level.")
        console.print()
        console.print("  [cyan]Domain Score[/]   Weighted average of suite scores for a domain.")
        console.print("                   A domain (e.g., 'coding') contains multiple")
        console.print("                   suites testing different aspects.")
        console.print()
        console.print("  [cyan]Overall Score[/]  Average of all domain scores for a model.")
        console.print("                   Represents general capability across all")
        console.print("                   evaluated domains.")
        console.print()
        console.print("  [cyan]Best Domain[/]    The domain where a model scored highest.")
        console.print("                   Used by the pipeline to route tasks to the")
        console.print("                   best-suited model for each domain.")
        console.print()
        return

    all_scores = load_confidence()

    if not all_scores:
        typer.echo("No confidence scores computed yet.")
        typer.echo("Run: polymind confidence compute")
        return

    if output_json:
        import json

        data = {k: v.to_dict() for k, v in all_scores.items()}
        typer.echo(json.dumps(data, indent=2))
        return

    # Filter by model
    if model_id:
        if model_id not in all_scores:
            typer.echo(f"No scores found for model: {model_id}", err=True)
            raise typer.Exit(code=1)
        display_scores = {model_id: all_scores[model_id]}
    else:
        display_scores = all_scores

    # Summary table
    typer.echo("Confidence Scores")
    typer.echo("=" * 70)
    typer.echo(f"{'Model':<15} {'Overall':>10} {'Best Domain':<25} {'Domains':>8}")
    typer.echo("-" * 70)

    for model_id_key, conf in sorted(
        display_scores.items(), key=lambda x: x[1].overall_score, reverse=True
    ):
        domain_count = len(conf.domains)
        typer.echo(
            f"  {model_id_key:<13} {conf.overall_score:>8.1f}% "
            f"{conf.best_domain:<25} {domain_count:>6}"
        )

    typer.echo()

    # Detailed per-domain view
    if domain_id:
        for model_id_key, conf in display_scores.items():
            if domain_id in conf.domains:
                ds = conf.domains[domain_id]
                typer.echo(f"Model {model_id_key} - {domain_id}")
                typer.echo(f"  Overall: {ds.overall:.1f}%")
                typer.echo()
                for suite_id, ss in ds.suites.items():
                    typer.echo(f"  {suite_id}: {ss.score:.1f}% ({ss.passed}/{ss.total})")
                typer.echo()
    else:
        # Show domain breakdown for each model
        for model_id_key, conf in display_scores.items():
            typer.echo(f"Model {model_id_key}")
            typer.echo("-" * 40)
            for domain_id_key, ds in sorted(
                conf.domains.items(), key=lambda x: x[1].overall, reverse=True
            ):
                typer.echo(f"  {domain_id_key:<25} {ds.overall:>6.1f}%")
            typer.echo()


@app.command("domains")
def list_domains() -> None:
    """List all available scoring domains.

    Shows all domains that can be used for confidence scoring,
    with their IDs, names, suite counts, and question counts.

    Examples:

        polymind confidence domains
    """
    domains = load_all_domains()

    typer.echo("Available Domains")
    typer.echo("=" * 60)

    for d in domains:
        suite_count = len(d.suites)
        q_count = sum(len(s.questions) for s in d.suites)
        marker = " [custom]" if d.custom else ""
        typer.echo(f"  {d.id:<25} {d.name:<30} {suite_count} suites, {q_count} questions{marker}")

    typer.echo()
    typer.echo(f"Total: {len(domains)} domains")


@app.command("reset")
def reset_scores(
    model_id: str = typer.Option(
        "", "--model", "-m", help="Model ID (number) to reset scores for. Resets all if empty."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Skip confirmation prompt and reset immediately."
    ),
) -> None:
    """Reset confidence scores.

    Deletes computed confidence scores for a specific model or
    all models. Use this to clear stale scores before recomputing.

    Examples:

        polymind confidence reset -m 1

        polymind confidence reset --all

        polymind confidence reset --force
    """
    all_scores = load_confidence()

    if not all_scores:
        typer.echo("No scores to reset.")
        return

    if not force:
        if model_id:
            typer.echo(f"About to reset scores for model {model_id}.")
        else:
            typer.echo("About to reset ALL confidence scores.")
        if not typer.confirm("Are you sure?"):
            typer.echo("Cancelled.")
            return

    if model_id:
        if model_id in all_scores:
            del all_scores[model_id]
            typer.echo(f"Reset scores for model {model_id}")
        else:
            typer.echo(f"No scores found for model {model_id}")
    else:
        all_scores = {}
        typer.echo("Reset all confidence scores.")

    save_confidence(all_scores)


def score_suite_from_responses(
    responses: dict[str, str],
    suite,
) -> SuiteScore:
    """Score a suite from a dict of responses."""
    scores = []
    details = {}
    total_weight = sum(q.weight for q in suite.questions)

    for question in suite.questions:
        response = responses.get(question.id, "")
        qs = score_question(response, question)
        weighted = qs.score * question.weight
        scores.append(qs)
        details[question.id] = round(weighted, 3)

    if total_weight > 0:
        overall = (
            sum(s.score * q.weight for s, q in zip(scores, suite.questions, strict=False))
            / total_weight
        )
    else:
        overall = 0.0

    passed = sum(1 for s in scores if s.score >= 0.7)

    return SuiteScore(
        suite_id=suite.id,
        score=round(overall * 100, 2),
        passed=passed,
        total=len(suite.questions),
        details=details,
    )
