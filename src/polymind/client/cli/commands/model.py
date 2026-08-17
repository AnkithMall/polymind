import typer
from pathlib import Path

from polymind.core.hardware.loader import load_hardware_profile
from polymind.core.model.hf import (
    HuggingFaceClient,
    detect_quantization,
)
from polymind.core.model.ranking import rank_models

from polymind.core.model.registry import (
    DEFAULT_MODEL_DIR,
    ModelRegistry,
)
from polymind.core.model.search import ModelSearchOptions
from polymind.core.model.utils import (
    file_size_bytes,
    format_size,
    parse_size
)
from typing import Literal
from polymind.core.model.grouping import group_models

app = typer.Typer()


@app.command("search")
def search(
    query: str = typer.Argument(
        ...,
        help="Text to search for on Hugging Face.",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Maximum number of Hugging Face repositories to inspect.",
    ),
    author: str | None = typer.Option(
        None,
        "--author",
        help="Restrict search to a Hugging Face author.",
    ),
    quantization: str | None = typer.Option(
        None,
        "--quant",
        help="Only show a quantization such as Q4_K_M.",
    ),
    max_size: str | None = typer.Option(
        None,
        "--max-size",
        help="Maximum GGUF size, e.g. 3GiB.",
    ),
    min_size: str | None = typer.Option(
        None,
        "--min-size",
        help="Minimum GGUF size, e.g. 1GiB.",
    ),
    sort: Literal[
        "created_at",
        "downloads",
        "last_modified",
        "likes",
        "trending_score",
    ] = typer.Option(
        "downloads",
        "--sort",
        help="Hugging Face sort order.",
    )
) -> None:
    """Search Hugging Face for hardware-compatible GGUF models."""

    # ---------------------------------------------------------
    # Load hardware profile
    # ---------------------------------------------------------

    try:
        hardware = load_hardware_profile()
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(
            f"Error: {exc}",
            err=True,
        )
        raise typer.Exit(code=1)

    # ---------------------------------------------------------
    # Parse size filters
    # ---------------------------------------------------------

    try:
        max_size_bytes = (
            parse_size(max_size)
            if max_size
            else None
        )

        min_size_bytes = (
            parse_size(min_size)
            if min_size
            else None
        )
    except ValueError as exc:
        typer.echo(
            f"Error: {exc}",
            err=True,
        )
        raise typer.Exit(code=1)

    # ---------------------------------------------------------
    # Build search options
    # ---------------------------------------------------------

    options = ModelSearchOptions(
        query=query,
        limit=limit,
        author=author,
        quantization=quantization,
        max_size_bytes=max_size_bytes,
        min_size_bytes=min_size_bytes,
        sort=sort,
    )

    typer.echo(
        f"Searching Hugging Face for GGUF models: {query}"
    )
    typer.echo()

    # ---------------------------------------------------------
    # Search Hugging Face
    # ---------------------------------------------------------

    client = HuggingFaceClient()

    try:
        models = client.search_gguf(options)
    except Exception as exc:
        typer.echo(
            f"Hugging Face search failed: {exc}",
            err=True,
        )
        raise typer.Exit(code=1)

    if not models:
        typer.echo("No GGUF models found.")
        return

    # ---------------------------------------------------------
    # Hardware-aware ranking
    # ---------------------------------------------------------

    ranked = rank_models(
        models,
        hardware,
    )

    if not ranked:
        typer.echo(
            "GGUF repositories were found, but no "
            "matching GGUF files were found."
        )
        return

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    groups = group_models(ranked)

    for index, group in enumerate(
        groups[:limit],
        start=1,
    ):
        first = group[0]

        typer.echo(
            f"{index}. {first.model_name}"
        )
        typer.echo(
            f"   Repository: {first.repo_id}"
        )

        typer.echo()
        typer.echo(
            "   Quantization   Size       VRAM              RAM"
        )
        typer.echo(
            "   ------------------------------------------"
        )

        for model in group:
            size = (
                format_size(model.size_bytes)
                if model.size_bytes is not None
                else "unknown"
            )
            shards = (
                f" ({model.shard_count} shards)"
                if model.shard_count > 1
                else ""
            )
            vram = model.vram_status
            ram = model.ram_status
            recommended = (
                " ★"
                if model.recommended
                else ""
            )

            typer.echo(
                f"   "
                f"{(model.quantization or 'unidentified'):<15}"
                f"{size:<11}"
                f"{vram:<18}"
                f"{ram:<12}"
                f"{shards}"
                f"{recommended}"
            )

        typer.echo()

@app.command("download")
def download(
    repo_id: str = typer.Argument(
        ...,
        help=(
            "Hugging Face repository, "
            "e.g. bartowski/Llama-3.2-3B-Instruct-GGUF"
        ),
    ),
    filename: str = typer.Argument(
        ...,
        help="GGUF filename to download.",
    ),
    output: Path = typer.Option(
        DEFAULT_MODEL_DIR,
        "--output",
        "-o",
        help="Model download directory.",
    ),
) -> None:
    """Download a GGUF model from Hugging Face."""

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    typer.echo()
    typer.echo(f"Repository: {repo_id}")
    typer.echo(f"File:       {filename}")
    typer.echo(f"Destination: {output}")
    typer.echo()

    client = HuggingFaceClient()

    try:
        path = client.download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(output),
        )
    except Exception as exc:
        typer.echo(
            f"Download failed: {exc}",
            err=True,
        )
        raise typer.Exit(code=1)

    local_path = Path(path)

    if not local_path.exists():
        typer.echo(
            "Download reported success, "
            "but the file was not found.",
            err=True,
        )
        raise typer.Exit(code=1)

    size_bytes = file_size_bytes(local_path)

    quantization = detect_quantization(filename)

    registry = ModelRegistry()

    model = registry.add(
        repo_id=repo_id,
        filename=filename,
        local_path=local_path,
        size_bytes=size_bytes,
        quantization=quantization,
    )

    typer.echo()
    typer.echo("Download complete.")
    typer.echo(f"Model ID:   {model.id}")
    typer.echo(f"Path:       {model.local_path}")
    typer.echo(f"Size:       {format_size(model.size_bytes)}")
    typer.echo(
        f"Quantization: "
        f"{model.quantization or 'unknown'}"
    )



@app.command("list")
def list_models() -> None:
    """List locally installed models."""

    registry = ModelRegistry()

    models = registry.load()

    if not models:
        typer.echo("No models installed.")
        typer.echo()
        typer.echo(
            "Use 'polymind model search <query>' "
            "to find models."
        )
        return

    typer.echo()
    typer.echo("Installed models")
    typer.echo("----------------")

    for model in models:
        path = Path(model.local_path)

        exists = path.exists()

        typer.echo()
        typer.echo(
            f"[{model.id}] "
            f"{model.repo_id}"
        )
        typer.echo(
            f"    File: {model.filename}"
        )
        typer.echo(
            f"    Size: "
            f"{format_size(model.size_bytes)}"
        )
        typer.echo(
            f"    Quantization: "
            f"{model.quantization or 'unknown'}"
        )
        typer.echo(
            f"    Path: {model.local_path}"
        )
        typer.echo(
            f"    Status: "
            f"{'available' if exists else 'missing'}"
        )
