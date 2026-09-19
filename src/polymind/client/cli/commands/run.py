"""Run command — alias for pipeline execution."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True)
def run_callback(
    ctx: typer.Context,
    prompt: str = typer.Argument(
        default="",
        help="Prompt to process through the pipeline.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed pipeline execution."
    ),
    model_id: str = typer.Option(
        "", "--model", "-m", help="Model ID for single-model quick run."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output result as JSON."
    ),
) -> None:
    """Run a prompt through the pipeline (alias for 'pipeline run').

    Processes the prompt through: decompose → assign → execute → regenerate.
    Use --model for direct single-model execution (no decomposition).

    Examples:

        polymind run "Build a REST API with tests"

        polymind run "What is quicksort?" -v

        polymind run "Hello" -m 1
    """
    if ctx.invoked_subcommand is not None:
        return

    if not prompt:
        console.print("[red]Please provide a prompt.[/]")
        console.print("Usage: polymind run \"your prompt here\"")
        raise typer.Exit(code=1)

    if model_id:
        _quick_run(prompt, model_id)
    else:
        from polymind.client.cli.commands.pipeline import run_pipeline

        ctx.invoke(
            run_pipeline,
            prompt=prompt,
            verbose=verbose,
            output_json=json_output,
        )


def _quick_run(prompt: str, model_id: str) -> None:
    """Quick single-model run."""
    from pathlib import Path

    from rich.markdown import Markdown
    from rich.panel import Panel

    from polymind.core.model.registry import ModelRegistry
    from polymind.core.runtime.artifact import load_runtime_config
    from polymind.core.runtime.config import default_runtime_config

    registry = ModelRegistry()
    models = registry.load()

    if not models:
        console.print("[red]No models installed.[/]")
        raise typer.Exit(code=1)

    # Find model
    model = None
    for m in models:
        if str(m.id) == model_id:
            model = m
            break
    if model is None:
        console.print(f"[red]Model not found: {model_id}[/]")
        raise typer.Exit(code=1)

    model_path = Path(model.local_path)
    if not model_path.exists():
        console.print(f"[red]Model file not found: {model.local_path}[/]")
        raise typer.Exit(code=1)

    runtime = load_runtime_config(str(model.id))
    if runtime is None:
        runtime = default_runtime_config(str(model.id), model.size_bytes)

    console.print(f"[dim]Using: {model.filename}[/]")

    from llama_cpp import Llama

    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=runtime.gpu_layers,
        n_threads=runtime.threads,
        n_ctx=runtime.context_size,
        n_batch=runtime.batch_size,
        verbose=False,
    )

    try:
        output = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )
        content = output["choices"][0]["message"]["content"] or ""
        try:
            md = Markdown(content.strip())
            console.print(Panel(md, title="Response", border_style="green", expand=True))
        except Exception:
            console.print(Panel(content.strip(), title="Response", border_style="green"))
    finally:
        del llm
