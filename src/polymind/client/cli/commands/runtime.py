"""Runtime commands — benchmark, optimize, show, validate, and run models."""

from pathlib import Path

import typer

from polymind.core.hardware.loader import load_hardware_profile
from polymind.core.model.registry import ModelRegistry
from polymind.core.model.utils import format_size
from polymind.core.paths import runtime_path
from polymind.core.runtime.artifact import (
    load_runtime_config,
    load_runtime_profile,
    validate_profile_for_current_hardware,
    validate_profile_for_current_model,
    write_runtime_profile,
)
from polymind.core.runtime.benchmark import get_available_gpu_memory_mb, get_available_ram_mb
from polymind.core.runtime.config import default_runtime_config
from polymind.core.runtime.optimizer import RuntimeOptimizer
from polymind.core.runtime.runner import RuntimeRunner
from polymind.core.runtime.types import (
    DEFAULT_WORKLOADS,
    HardwareFingerprint,
    ModelFingerprint,
    ValidationStatus,
)

app = typer.Typer()


@app.command("run")
def run(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Model ID (number), filename, or repository ID.",
    ),
) -> None:
    """Run a local model for interactive testing."""
    registry = ModelRegistry()
    models = registry.load()

    if not models:
        typer.echo("No models installed.", err=True)
        raise typer.Exit(code=1)

    selected = next((item for item in models if str(item.id) == model), None)
    if selected is None:
        selected = next((item for item in models if item.filename == model), None)
    if selected is None:
        selected = next((item for item in models if item.repo_id == model), None)

    if selected is None:
        typer.echo(f"Model not found: {model}", err=True)
        raise typer.Exit(code=1)

    model_path = Path(selected.local_path)
    if not model_path.exists():
        typer.echo(f"Model file does not exist: {model_path}", err=True)
        raise typer.Exit(code=1)

    config = load_runtime_config(str(selected.id))
    if config is None:
        config = default_runtime_config(str(selected.id), model_size_bytes=selected.size_bytes)
        typer.echo(
            f"Warning: No optimized config found. "
            f"Run 'polymind runtime optimize -m {selected.id}' for best performance."
        )
        typer.echo()

    typer.echo(f"Loading model: {selected.filename}")
    typer.echo(f"GPU layers: {config.gpu_layers}")
    typer.echo(f"Context: {config.context_size}")
    typer.echo()

    runner = RuntimeRunner(config=config, model_path=model_path)
    runner.chat()


@app.command("optimize")
def optimize(
    model: int | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model ID (number) to benchmark and optimize.",
    ),
    all_models: bool = typer.Option(
        False,
        "--all",
        help="Benchmark and optimize all installed models.",
    ),
    workload: str = typer.Option(
        "default",
        "--workload",
        "-w",
        help="Workload profile: default, decomposer, generator, interactive, long_context, all.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-optimization even if a valid profile exists.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed optimization output.",
    ),
) -> None:
    """Benchmark and find optimal runtime settings.

    Runs isolated inference benchmarks to determine the best GPU layer
    offloading, thread count, context size, and batch size. Uses a
    safety-first approach: only configurations that successfully execute
    real inference are considered.

    The optimizer:
      1. Generates placement candidates (CPU, single GPU, multi-GPU)
      2. Filters by memory feasibility
      3. Searches GPU layers, context, threads, and batch sizes
      4. Benchmarks each candidate in an isolated subprocess
      5. Selects the best SAFE configuration

    Examples:
        polymind runtime optimize -m 1
        polymind runtime optimize -m 2 --workload decomposer
        polymind runtime optimize --all --verbose
    """
    if model is None and not all_models:
        typer.echo("Error: specify --model/-m or --all.", err=True)
        raise typer.Exit(code=1)

    if model is not None and all_models:
        typer.echo("Error: use either --model/-m or --all, not both.", err=True)
        raise typer.Exit(code=1)

    try:
        hardware = load_hardware_profile()
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    registry = ModelRegistry()
    models = registry.load()

    if not models:
        typer.echo("No models installed.")
        typer.echo("Use 'polymind model search <query>' to find models.")
        raise typer.Exit(code=1)

    if all_models:
        selected_models = models
    else:
        selected_models = [item for item in models if item.id == model]
        if not selected_models:
            typer.echo(f"Model not found: {model}", err=True)
            raise typer.Exit(code=1)

    # Determine workloads to optimize
    if workload == "all":
        workload_names = list(DEFAULT_WORKLOADS.keys())
    elif workload in DEFAULT_WORKLOADS:
        workload_names = [workload]
    else:
        typer.echo(
            f"Unknown workload: {workload}. Available: {', '.join(DEFAULT_WORKLOADS.keys())}, all",
            err=True,
        )
        raise typer.Exit(code=1)

    # Show hardware summary
    typer.echo()
    typer.echo("═══ Hardware ═══")
    typer.echo(f"  CPU:     {hardware.cpu.model}")
    typer.echo(
        f"  Cores:   {hardware.cpu.physical_cores} physical / {hardware.cpu.logical_cores} logical"
    )
    typer.echo(f"  RAM:     {format_size(hardware.memory.total_bytes)}")

    selected_gpus = [
        gpu for gpu in hardware.gpus if gpu.selection.enabled and gpu.compute.llama_cpp_usable
    ]
    if selected_gpus:
        for gpu in selected_gpus:
            vram = gpu.memory.available_bytes or gpu.memory.total_bytes
            typer.echo(f"  GPU:     {gpu.model} ({format_size(vram)})")
    else:
        typer.echo("  GPU:     None (CPU only)")

    free_vram = get_available_gpu_memory_mb()
    free_ram = get_available_ram_mb()
    if free_vram > 0:
        typer.echo(f"  Free VRAM: {free_vram:.0f} MB")
    typer.echo(f"  Free RAM:  {free_ram:.0f} MB")
    typer.echo()

    # Hardware fingerprint
    hw_fp = HardwareFingerprint.from_hardware_profile(hardware)
    typer.echo(f"  Hardware FP: {hw_fp.compute_hash()}")
    typer.echo()

    # Optimize each model
    configs_written = 0

    for installed_model in selected_models:
        model_path = Path(installed_model.local_path)
        if not model_path.exists():
            typer.echo(f"Skipping {installed_model.filename}: file not found")
            continue

        # Model fingerprint
        model_fp = ModelFingerprint.from_model_file(
            model_path, installed_model.size_bytes, installed_model.quantization or ""
        )

        typer.echo(f"═══ Optimizing: {installed_model.filename} ═══")
        typer.echo(f"  Size: {format_size(installed_model.size_bytes)}")
        typer.echo(f"  Layers: {model_fp.num_layers or '~unknown~'}")
        typer.echo(f"  Model FP: {model_fp.compute_hash()}")
        typer.echo()

        for wl_name in workload_names:
            wl = DEFAULT_WORKLOADS[wl_name]
            typer.echo(f"  Workload: {wl.name} — {wl.description}")

            # ── Idempotency check ──────────────────────────────
            if not force:
                existing = load_runtime_profile(str(installed_model.id))
                if existing is not None:
                    hw_ok, hw_reason = validate_profile_for_current_hardware(
                        existing, hw_fp.compute_hash()
                    )
                    model_ok, model_reason = validate_profile_for_current_model(
                        existing, model_fp.compute_hash()
                    )
                    from polymind.core.runtime.types import ValidationStatus

                    status_ok = existing.validation.status == ValidationStatus.READY

                    if hw_ok and model_ok and status_ok:
                        typer.echo(
                            f"    ✓ Already optimized (gen={existing.benchmark.generation_tokens_per_sec:.1f} tok/s, "
                            f"gpu_layers={existing.gpu_layers}). Use --force to re-benchmark."
                        )
                        configs_written += 1
                        continue
                    else:
                        reasons = []
                        if not hw_ok:
                            reasons.append(f"hardware changed ({hw_reason})")
                        if not model_ok:
                            reasons.append(f"model changed ({model_reason})")
                        if not status_ok:
                            reasons.append(f"status={existing.validation.status.value}")
                        typer.echo(f"    Re-optimizing: {', '.join(reasons)}")

            def on_progress(msg: str, current: int, total: int) -> None:
                if verbose:
                    typer.echo(f"    [{current}/{total}] {msg}")

            optimizer = RuntimeOptimizer(
                model_path=model_path,
                model_id=str(installed_model.id),
                model_size_bytes=installed_model.size_bytes,
                hardware=hardware,
                model_fingerprint=model_fp,
                workload=wl,
                on_progress=on_progress,
                benchmark_runs=3,
                benchmark_timeout=120,
            )

            profile = optimizer.optimize()

            # Update profile with fingerprints
            profile.hardware_fingerprint = hw_fp.compute_hash()
            profile.model_fingerprint = model_fp.compute_hash()
            profile.workload = wl_name

            # Persist profile
            write_runtime_profile(profile)

            # Show results
            typer.echo()
            if profile.benchmark.runs > 0:
                typer.echo("    Result: PASS")
                typer.echo(f"    Backend:    {profile.placement.backend}")
                typer.echo(f"    GPU layers: {profile.gpu_layers}")
                typer.echo(f"    Threads:    {profile.threads}")
                typer.echo(f"    Context:    {profile.context_size}")
                typer.echo(f"    Batch:      {profile.batch_size}")
                typer.echo(
                    f"    Gen speed:  {profile.benchmark.generation_tokens_per_sec:.1f} tok/s"
                )
                typer.echo(f"    Prompt:     {profile.benchmark.prompt_tokens_per_sec:.1f} tok/s")
                typer.echo(f"    Stability:  {profile.benchmark.stability:.0%}")
                typer.echo(f"    Peak VRAM:  {profile.benchmark.peak_vram_mb:.0f} MB")
                typer.echo(f"    Safety:     {profile.safety_margin_mb:.0f} MB headroom")
            else:
                typer.echo("    Result: FALLBACK (all benchmarks failed)")
                typer.echo("    Using conservative CPU config")
                typer.echo("    GPU layers: 0")
                typer.echo(f"    Threads:    {profile.threads}")
                typer.echo(f"    Context:    {profile.context_size}")

            typer.echo()

            configs_written += 1

            # Show summary if verbose
            if verbose:
                typer.echo(optimizer.get_summary())
                typer.echo()

    typer.echo("═══ Done ═══")
    typer.echo(f"Wrote {configs_written} config(s) to: {runtime_path()}")


@app.command("show")
def show(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Model ID (number) to show runtime config for.",
    ),
) -> None:
    """Show runtime configuration and profile details for a model.

    Displays the current runtime configuration, validation status,
    hardware/model fingerprints, and benchmark metrics.
    """
    profile = load_runtime_profile(model)

    if profile is None:
        # Try legacy config
        config = load_runtime_config(model)
        if config is None:
            typer.echo(f"No runtime config for model {model}.", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"Runtime Config for model {model} (legacy):")
        typer.echo(f"  GPU layers:  {config.gpu_layers}")
        typer.echo(f"  Threads:     {config.threads}")
        typer.echo(f"  Context:     {config.context_size}")
        typer.echo(f"  Batch:       {config.batch_size}")
        if config.benchmark:
            typer.echo(f"  Benchmark:   {config.benchmark}")
        typer.echo()
        typer.echo("  Note: This is a legacy profile. Run 'polymind runtime optimize' to upgrade.")
        return

    # Show full profile
    typer.echo(f"═══ Runtime Profile: Model {model} ═══")
    typer.echo()

    # Validation status
    status = profile.validation.status.value
    if status == "ready":
        typer.echo("  Status:      ✓ READY")
    elif status == "stale":
        typer.echo("  Status:      ⚠ STALE (revalidation needed)")
    elif status == "failed":
        typer.echo("  Status:      ✗ FAILED")
    else:
        typer.echo("  Status:      ? UNKNOWN")

    typer.echo(f"  Workload:    {profile.workload}")
    typer.echo(f"  Validated:   {profile.validation.last_validated or 'never'}")
    typer.echo(
        f"  Runs:        {profile.validation.successful_runs} ok / {profile.validation.failed_runs} failed"
    )
    typer.echo()

    # Fingerprints
    typer.echo("  Fingerprints:")
    typer.echo(f"    Hardware:  {profile.hardware_fingerprint or '(none)'}")
    typer.echo(f"    Model:     {profile.model_fingerprint or '(none)'}")
    typer.echo()

    # Placement
    typer.echo("  Placement:")
    typer.echo(f"    Backend:       {profile.placement.backend}")
    typer.echo(f"    Devices:       {profile.placement.devices or '[]'}")
    typer.echo(f"    Split mode:    {profile.placement.split_mode.value}")
    typer.echo(f"    Main GPU:      {profile.placement.main_gpu}")
    typer.echo()

    # Execution
    typer.echo("  Execution:")
    typer.echo(f"    GPU layers:    {profile.gpu_layers}")
    typer.echo(f"    Threads:       {profile.threads}")
    typer.echo(f"    Threads batch: {profile.threads_batch}")
    typer.echo(f"    Context:       {profile.context_size}")
    typer.echo(f"    Batch:         {profile.batch_size}")
    typer.echo()

    # Benchmarks
    if profile.benchmark.runs > 0:
        typer.echo("  Benchmark:")
        typer.echo(f"    Gen speed:     {profile.benchmark.generation_tokens_per_sec:.1f} tok/s")
        typer.echo(f"    Prompt speed:  {profile.benchmark.prompt_tokens_per_sec:.1f} tok/s")
        typer.echo(f"    TTFT:          {profile.benchmark.time_to_first_token_ms:.0f} ms")
        typer.echo(f"    Load time:     {profile.benchmark.load_time_ms:.0f} ms")
        typer.echo(f"    Peak VRAM:     {profile.benchmark.peak_vram_mb:.0f} MB")
        typer.echo(f"    Peak RAM:      {profile.benchmark.peak_ram_mb:.0f} MB")
        typer.echo(f"    Stability:     {profile.benchmark.stability:.0%}")
        typer.echo(f"    Runs:          {profile.benchmark.runs}")
    typer.echo()

    # Memory
    typer.echo("  Memory:")
    typer.echo(f"    Estimated:     {profile.estimated_memory_mb:.0f} MB")
    typer.echo(f"    Peak actual:   {profile.actual_peak_memory_mb:.0f} MB")
    typer.echo(f"    Safety margin: {profile.safety_margin_mb:.0f} MB")
    typer.echo()

    # Failure reasons
    if profile.validation.failure_reasons:
        typer.echo("  Failure reasons:")
        for reason in profile.validation.failure_reasons:
            typer.echo(f"    - {reason}")
        typer.echo()


@app.command("validate")
def validate(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Model ID (number) to validate.",
    ),
) -> None:
    """Validate a model's runtime profile against current hardware.

    Checks if the profile was built for the current hardware and model,
    and reports whether it needs revalidation.
    """
    profile = load_runtime_profile(model)

    if profile is None:
        typer.echo(f"No runtime config for model {model}.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"═══ Validation: Model {model} ═══")
    typer.echo()

    # Check hardware
    try:
        hardware = load_hardware_profile()
        hw_fp = HardwareFingerprint.from_hardware_profile(hardware)
        hw_fp_hash = hw_fp.compute_hash()

        hw_valid, hw_reason = validate_profile_for_current_hardware(profile, hw_fp_hash)
        if hw_valid:
            typer.echo(f"  ✓ Hardware: {hw_reason}")
        else:
            typer.echo(f"  ✗ Hardware: {hw_reason}")
    except Exception as e:
        typer.echo(f"  ? Hardware: Cannot check ({e})")

    # Check model fingerprint
    try:
        registry = ModelRegistry()
        models = registry.load()
        selected = [m for m in models if str(m.id) == str(model)]
        if selected:
            m = selected[0]
            model_fp = ModelFingerprint.from_model_file(
                Path(m.local_path), m.size_bytes, m.quantization or ""
            )
            model_fp_valid, model_reason = validate_profile_for_current_model(
                profile, model_fp.compute_hash()
            )
        else:
            model_fp_valid, model_reason = True, "Model not found in registry"
    except Exception as e:
        model_fp_valid, model_reason = True, f"Cannot check ({e})"

    if model_fp_valid:
        typer.echo(f"  ✓ Model:    {model_reason}")
    else:
        typer.echo(f"  ✗ Model:    {model_reason}")

    # Check validation status
    status = profile.validation.status
    if status == ValidationStatus.READY:
        typer.echo(f"  ✓ Status:   READY ({profile.validation.successful_runs} successful runs)")
    elif status == ValidationStatus.STALE:
        typer.echo("  ⚠ Status:   STALE — revalidation recommended")
    elif status == ValidationStatus.FAILED:
        typer.echo("  ✗ Status:   FAILED — reoptimization required")
        for reason in profile.validation.failure_reasons:
            typer.echo(f"    Reason: {reason}")
    else:
        typer.echo("  ? Status:   UNKNOWN")

    typer.echo()

    # Recommendation
    if not hw_valid or status in (ValidationStatus.FAILED, ValidationStatus.STALE):
        typer.echo("  Recommendation: Run 'polymind runtime optimize' to revalidate.")
    else:
        typer.echo("  Profile is valid and ready for use.")
