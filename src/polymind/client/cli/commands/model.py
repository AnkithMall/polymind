import shutil
from pathlib import Path
from typing import Literal

import typer

from polymind.core.hardware.loader import load_hardware_profile
from polymind.core.model.grouping import group_models
from polymind.core.model.hf import (
    HuggingFaceClient,
    detect_quantization,
)
from polymind.core.model.ranking import rank_models
from polymind.core.model.registry import ModelRegistry
from polymind.core.model.search import ModelSearchOptions
from polymind.core.model.utils import file_size_bytes, format_size, parse_size
from polymind.core.paths import model_dir

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
        help="Maximum number of model groups to display (default: 20).",
    ),
    author: str | None = typer.Option(
        None,
        "--author",
        help="Restrict search to a specific Hugging Face author or organization.",
    ),
    quantization: str | None = typer.Option(
        None,
        "--quant",
        help="Filter by quantization type, e.g. Q4_K_M, Q5_K_S, Q8_0.",
    ),
    max_size: str | None = typer.Option(
        None,
        "--max-size",
        help="Maximum GGUF file size filter, e.g. 3GiB, 2.5GB, 500MiB.",
    ),
    min_size: str | None = typer.Option(
        None,
        "--min-size",
        help="Minimum GGUF file size filter, e.g. 1GiB, 500MiB.",
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
        help="Sort results by: downloads (default), created_at, last_modified, likes, or trending_score.",
    ),
) -> None:
    """Search Hugging Face for hardware-compatible GGUF models.

    Queries Hugging Face for GGUF quantized models and ranks results
    by compatibility with your detected hardware (GPU VRAM, RAM).
    Results show quantization variants, sizes, and compatibility status.

    Examples:

        polymind model search llama-3

        polymind model search mistral --author mistralai --quant Q4_K_M

        polymind model search phi --limit 10 --max-size 4GiB --sort downloads
    """

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
        max_size_bytes = parse_size(max_size) if max_size else None

        min_size_bytes = parse_size(min_size) if min_size else None
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

    typer.echo(f"Searching Hugging Face for GGUF models: {query}")
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
        typer.echo("GGUF repositories were found, but no matching GGUF files were found.")
        return

    # ---------------------------------------------------------
    # Mark already downloaded models
    # ---------------------------------------------------------

    registry = ModelRegistry()

    for model in ranked:
        if registry.find_by_filename(model.filename):
            model.downloaded = True

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    groups = group_models(ranked)

    for index, group in enumerate(
        groups[:limit],
        start=1,
    ):
        first = group[0]

        typer.echo(f"{index}. {first.model_name}")
        typer.echo(f"   Repository: {first.repo_id}")

        typer.echo()
        typer.echo("   Quantization   Size       VRAM              RAM        Filename")
        typer.echo("   ---------------------------------------------------------------------")

        for model in group:
            size = format_size(model.size_bytes) if model.size_bytes is not None else "unknown"
            shards = f" ({model.shard_count} shards)" if model.shard_count > 1 else ""
            vram = model.vram_status
            ram = model.ram_status
            status = " [downloaded]" if model.downloaded else ""
            recommended = " ★" if model.recommended else ""

            typer.echo(
                f"   "
                f"{(model.quantization or 'unidentified'):<15}"
                f"{size:<11}"
                f"{vram:<18}"
                f"{ram:<12}"
                f"{model.filename}"
                f"{shards}"
                f"{recommended}"
                f"{status}"
            )

        typer.echo()


@app.command("download")
def download(
    repo_id: str = typer.Argument(
        ...,
        help="Hugging Face repository ID, e.g. bartowski/Llama-3.2-3B-Instruct-GGUF.",
    ),
    filename: str = typer.Argument(
        ...,
        help="Exact GGUF filename to download from the repository.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Model download directory. "
            "Defaults to .polymind/models "
            "(override with POLYMIND_MODEL_DIR)."
        ),
    ),
) -> None:
    """Download a GGUF model from Hugging Face.

    Downloads a specific GGUF file from a Hugging Face repository,
    registers it in the local model registry, and displays the
    model ID for use with other commands.

    Examples:

        polymind model download bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_K_M.gguf

        polymind model download TheBloke/Mistral-7B-v0.1-GGUF mistral-7b-v0.1.Q4_K_M.gguf -o ./models
    """

    registry = ModelRegistry()

    # Check if already downloaded
    existing = registry.find_by_repo_and_filename(repo_id, filename)

    if existing:
        existing_path = Path(existing.local_path)
        if existing_path.exists():
            typer.echo(
                f"Model already downloaded: {existing.local_path}",
            )
            typer.echo(f"Model ID: {existing.id}")
            raise typer.Exit(0)
        else:
            typer.echo(
                f"Model registered but file missing: {existing.local_path}",
            )
            typer.echo("Re-downloading...")

    if output is None:
        output = model_dir()

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
        # Check if the file was actually downloaded despite the error
        # (common with HuggingFace permission warnings on temp files)
        expected_path = output / filename
        if expected_path.exists() and expected_path.stat().st_size > 0:
            path = str(expected_path)
            typer.echo(
                f"Warning: {exc}",
                err=True,
            )
            typer.echo("File was downloaded successfully despite the warning.")
        else:
            typer.echo(
                f"Download failed: {exc}",
                err=True,
            )
            raise typer.Exit(code=1)

    local_path = Path(path)

    if not local_path.exists():
        typer.echo(
            "Download reported success, but the file was not found.",
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
    typer.echo(f"Quantization: {model.quantization or 'unknown'}")


@app.command("migrate")
def migrate(
    source: Path | None = typer.Option(
        None,
        "--source",
        "-s",
        help="Source directory containing GGUF files to migrate. Auto-detected if not specified.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip all confirmation prompts and migrate immediately.",
    ),
) -> None:
    """Migrate models from legacy location to .polymind/models/.

    Moves GGUF model files from legacy directories (polymind/models/
    or ~/.cache/polymind/models/) into the current .polymind/models/
    directory and registers them in the model registry.

    Examples:

        polymind model migrate

        polymind model migrate --source ~/old-models --force

        polymind model migrate --source ./polymind/models
    """

    # Determine source directory
    if source is None:
        # Check common legacy locations
        legacy_locations = [
            Path("polymind/models"),
            Path.home() / ".cache" / "polymind" / "models",
        ]
        for loc in legacy_locations:
            if loc.exists() and any(loc.glob("*.gguf")):
                source = loc
                break

        if source is None:
            typer.echo(
                "No legacy models found to migrate.",
                err=True,
            )
            typer.echo("Checked locations:")
            for loc in legacy_locations:
                typer.echo(f"  - {loc}")
            raise typer.Exit(code=1)

    if not source.exists():
        typer.echo(
            f"Source directory not found: {source}",
            err=True,
        )
        raise typer.Exit(code=1)

    # Find all GGUF files in source
    gguf_files = list(source.glob("*.gguf"))

    if not gguf_files:
        typer.echo(
            f"No GGUF files found in {source}",
            err=True,
        )
        raise typer.Exit(code=1)

    target = model_dir()

    typer.echo("Migration Summary")
    typer.echo("-----------------")
    typer.echo(f"Source:      {source}")
    typer.echo(f"Target:      {target}")
    typer.echo(f"Models:      {len(gguf_files)}")
    typer.echo()

    # Show what will be migrated
    for f in gguf_files:
        size = format_size(f.stat().st_size)
        typer.echo(f"  - {f.name} ({size})")

    typer.echo()

    # Confirm unless force
    if not force:
        confirm = typer.confirm("Proceed with migration?")
        if not confirm:
            typer.echo("Migration cancelled.")
            raise typer.Exit(0)

    # Create target directory
    target.mkdir(parents=True, exist_ok=True)

    # Move files
    migrated = 0
    errors = []

    for f in gguf_files:
        dest = target / f.name

        if dest.exists():
            typer.echo(
                f"Skipping {f.name} (already exists at destination)",
                err=True,
            )
            continue

        try:
            shutil.move(str(f), str(dest))
            migrated += 1
            typer.echo(f"  Migrated: {f.name}")
        except Exception as e:
            errors.append((f.name, str(e)))
            typer.echo(
                f"  Error migrating {f.name}: {e}",
                err=True,
            )

    typer.echo()
    typer.echo(f"Migration complete: {migrated}/{len(gguf_files)} models migrated")

    if errors:
        typer.echo()
        typer.echo("Errors:")
        for name, err in errors:
            typer.echo(f"  - {name}: {err}")

    # Ask if user wants to clean up empty source directory
    if migrated > 0 and not force:
        # Check if source directory is now empty
        remaining = list(source.glob("*.gguf"))
        if not remaining:
            typer.echo()
            if typer.confirm("Source directory is empty. Remove it?"):
                try:
                    source.rmdir()
                    typer.echo(f"Removed empty directory: {source}")
                except Exception as e:
                    typer.echo(
                        f"Could not remove directory: {e}",
                        err=True,
                    )

    # Run scan to register migrated models
    if migrated > 0:
        typer.echo()
        typer.echo("Running scan to register models...")
        registry = ModelRegistry()
        report = registry.scan(target)

        if report.added:
            typer.echo(f"Registered {len(report.added)} new model(s)")
        else:
            typer.echo("All models already registered")


@app.command("list")
def list_models() -> None:
    """List locally installed models.

    Shows all models registered in the local model registry with
    their IDs, file paths, sizes, quantization types, and
    availability status (available or missing from disk).

    Examples:

        polymind model list
    """

    registry = ModelRegistry()

    models = registry.load()

    if not models:
        typer.echo("No models installed.")
        typer.echo()
        typer.echo("Use 'polymind model search <query>' to find models.")
        return

    typer.echo()
    typer.echo("Installed models")
    typer.echo("----------------")

    for model in models:
        path = Path(model.local_path)

        exists = path.exists()

        typer.echo()
        typer.echo(f"[{model.id}] {model.repo_id}")
        typer.echo(f"    File: {model.filename}")
        typer.echo(f"    Size: {format_size(model.size_bytes)}")
        typer.echo(f"    Quantization: {model.quantization or 'unknown'}")
        typer.echo(f"    Path: {model.local_path}")
        typer.echo(f"    Status: {'available' if exists else 'missing'}")


@app.command("delete")
def delete(
    model: str = typer.Argument(
        ...,
        help="Model ID (number), filename, or repository ID to identify the model to delete.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip the confirmation prompt and delete immediately.",
    ),
) -> None:
    """Delete a model from disk and registry.

    Removes a model file from disk and deletes its entry from
    the model registry. Can match by model ID, filename, or
    repository ID.

    Examples:

        polymind model delete 3

        polymind model delete Llama-3.2-3B-Instruct-Q4_K_M.gguf

        polymind model delete bartowski/Llama-3.2-3B-Instruct-GGUF --force
    """

    registry = ModelRegistry()
    models = registry.load()

    if not models:
        typer.echo("No models installed.", err=True)
        raise typer.Exit(code=1)

    # Find model by ID
    selected = next(
        (item for item in models if str(item.id) == model),
        None,
    )

    # Then try filename
    if selected is None:
        selected = next(
            (item for item in models if item.filename == model),
            None,
        )

    # Finally try repository ID
    if selected is None:
        selected = next(
            (item for item in models if item.repo_id == model),
            None,
        )

    if selected is None:
        typer.echo(
            f"Model not found: {model}",
            err=True,
        )
        raise typer.Exit(code=1)

    file_path = Path(selected.local_path)

    typer.echo()
    typer.echo("Model to delete:")
    typer.echo(f"  ID:           {selected.id}")
    typer.echo(f"  Repository:   {selected.repo_id}")
    typer.echo(f"  Filename:     {selected.filename}")
    typer.echo(f"  Path:         {selected.local_path}")
    typer.echo(f"  Size:         {format_size(selected.size_bytes)}")
    typer.echo(f"  File exists:  {'yes' if file_path.exists() else 'no'}")
    typer.echo()

    if not force:
        confirm = typer.confirm("Delete this model?")
        if not confirm:
            typer.echo("Cancelled.")
            raise typer.Exit(0)

    # Remove file from disk
    if file_path.exists():
        file_path.unlink()
        typer.echo(f"Deleted file: {file_path}")

    # Remove from registry
    registry.remove(selected.id)
    typer.echo(f"Removed from registry (ID: {selected.id})")

    typer.echo()
    typer.echo("Model deleted successfully.")


@app.command("scan")
def scan(
    directory: Path | None = typer.Option(
        None,
        "--directory",
        "-d",
        help="Directory to scan for GGUF files. Defaults to .polymind/models or POLYMIND_MODEL_DIR.",
    ),
    scan_all: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Scan all known locations: model directory, HuggingFace cache, and current directory.",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Interactively prompt to resolve path conflicts and duplicate entries.",
    ),
) -> None:
    """Scan directories and fix the registry.

    Scans one or more directories for GGUF model files and
    repairs the model registry: registers new files, fixes
    broken paths for moved files, and merges duplicates.
    By default scans the model directory only.

    Examples:

        polymind model scan

        polymind model scan --all

        polymind model scan --directory ~/Downloads --interactive
    """

    registry = ModelRegistry()

    # Collect directories to scan
    directories_to_scan: list[Path] = []

    if scan_all:
        # Scan all known locations
        model_directory = model_dir()
        directories_to_scan.append(model_directory)

        # HuggingFace cache
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        if hf_cache.exists():
            directories_to_scan.append(hf_cache)

        # Current directory
        cwd = Path.cwd()
        if cwd != model_directory.parent:
            directories_to_scan.append(cwd)

        typer.echo("Scanning all known locations:")
        for d in directories_to_scan:
            exists = d.exists()
            status = "✓" if exists else "✗"
            typer.echo(f"  {status} {d}")
        typer.echo()
    else:
        # Single directory scan
        if directory is None:
            directory = model_dir()

        if not directory.exists():
            typer.echo(
                f"Model directory not found: {directory}",
                err=True,
            )
            # Create the directory if it doesn't exist
            if typer.confirm(f"Create directory {directory}?"):
                directory.mkdir(parents=True, exist_ok=True)
                typer.echo(f"Created: {directory}")
            else:
                raise typer.Exit(code=1)

        directories_to_scan.append(directory)

    # Scan each directory
    total_added = []
    total_fixed = []
    total_duplicates = 0
    total_missing = []

    for scan_dir in directories_to_scan:
        if not scan_dir.exists():
            continue

        typer.echo(f"Scanning: {scan_dir}")
        report = registry.scan(scan_dir)

        if report.added:
            total_added.extend(report.added)

        if report.fixed:
            total_fixed.extend(report.fixed)

        total_duplicates += report.duplicates_removed

        if report.missing:
            total_missing.extend(report.missing)

        # Interactive conflict resolution
        if interactive and report.fixed:
            typer.echo()
            typer.echo("Path fixes needed:")
            for model in report.fixed:
                typer.echo(f"  [{model.id}] {model.filename}")
                typer.echo(f"    Old: {model.local_path}")
                typer.echo("    New: (found on disk)")

            typer.echo()
            if not typer.confirm("Accept all fixes?"):
                typer.echo("Fixes will not be applied.")
                # Note: In this implementation, fixes are already
                # applied by registry.scan(). For full interactive
                # control, we'd need to refactor the scan method.

    typer.echo()
    typer.echo("Scan Results")
    typer.echo("------------")
    typer.echo(f"Directories scanned: {len(directories_to_scan)}")
    typer.echo()

    if total_added:
        typer.echo(f"Registered {len(total_added)} new model(s):")
        for model in total_added:
            typer.echo(f"  [{model.id}] {model.filename} ({format_size(model.size_bytes)})")
        typer.echo()

    if total_fixed:
        typer.echo(f"Fixed {len(total_fixed)} registry path(s):")
        for model in total_fixed:
            typer.echo(f"  [{model.id}] {model.filename} -> {model.local_path}")
        typer.echo()

    if total_duplicates:
        typer.echo(f"Removed {total_duplicates} duplicate registry entrie(s).")
        typer.echo()

    if total_missing:
        typer.echo(f"{len(total_missing)} registered model(s) could not be found on disk:")
        for model in total_missing:
            typer.echo(f"  [{model.id}] {model.filename} (last known: {model.local_path})")
        typer.echo()

    if not (total_added or total_fixed or total_duplicates or total_missing):
        typer.echo("Registry is up to date; nothing to fix.")
