"""Doctor command — system diagnostics and health checks."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True)
def doctor_callback() -> None:
    """Run diagnostics and health checks.

    Checks the Polymind installation for common issues:
    - Artifact directory
    - Hardware profile
    - Model registry
    - Installed models
    - Runtime configuration
    - Evaluation data
    - Pipeline readiness

    Examples:

        polymind doctor
    """
    checks: list[tuple[str, bool, str]] = []

    # 1. Artifact directory
    from polymind.core.paths import artifact_dir

    art_dir = artifact_dir()
    if art_dir.exists():
        checks.append(("Artifact directory", True, str(art_dir)))
    else:
        checks.append(("Artifact directory", False, f"Not found: {art_dir}"))

    # 2. Hardware profile
    from polymind.core.paths import hardware_path

    hw_path = hardware_path()
    if hw_path.exists():
        try:
            from polymind.core.hardware.loader import load_hardware_profile

            hw = load_hardware_profile()
            gpu_count = len(hw.gpus)
            if gpu_count > 0:
                gpu_names = ", ".join(g.model for g in hw.gpus)
                checks.append(("Hardware profile", True, f"{gpu_names} ({gpu_count} GPU(s))"))
            else:
                checks.append(("Hardware profile", True, "No GPUs detected (CPU only)"))
        except Exception as e:
            checks.append(("Hardware profile", False, f"Error loading: {e}"))
    else:
        checks.append(("Hardware profile", False, "Not found. Run: polymind hardware scan"))

    # 3. Model registry
    from polymind.core.paths import registry_path

    reg_path = registry_path()
    if reg_path.exists():
        from polymind.core.model.registry import ModelRegistry

        registry = ModelRegistry()
        models = registry.load()
        checks.append(("Model registry", True, f"{len(models)} model(s) registered"))
    else:
        checks.append(("Model registry", False, "Not found. Run: polymind model scan"))

    # 4. Installed model files
    from polymind.core.paths import model_dir

    mdl_dir = model_dir()
    if mdl_dir.exists():
        gguf_files = list(mdl_dir.glob("*.gguf"))
        if gguf_files:
            total_size = sum(f.stat().st_size for f in gguf_files)
            size_gb = total_size / (1024**3)
            checks.append(("Model files", True, f"{len(gguf_files)} GGUF file(s), {size_gb:.1f} GB"))
        else:
            checks.append(("Model files", False, "No GGUF files found"))
    else:
        checks.append(("Model files", False, f"Model directory not found: {mdl_dir}"))

    # 5. Runtime configuration
    from polymind.core.paths import runtime_path

    rt_path = runtime_path()
    if rt_path.exists():
        import yaml

        with rt_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        rt_models = data.get("models", {})
        checks.append(("Runtime config", True, f"{len(rt_models)} model(s) configured"))
    else:
        checks.append(("Runtime config", False, "Not found. Run: polymind runtime optimize"))

    # 6. Confidence/evaluation data
    from polymind.core.confidence.artifact import load_confidence

    confidence = load_confidence()
    if confidence:
        domain_count = set()
        for conf in confidence.values():
            domain_count.update(conf.domains.keys())
        checks.append(
            ("Confidence scores", True, f"{len(confidence)} model(s), {len(domain_count)} domain(s)")
        )
    else:
        checks.append(("Confidence scores", False, "Not computed. Run: polymind confidence compute"))

    # 7. Custom domains
    from polymind.core.paths import custom_domains_dir

    custom_dir = custom_domains_dir()
    if custom_dir.exists():
        custom_files = list(custom_dir.glob("*.yaml"))
        if custom_files:
            checks.append(("Custom domains", True, f"{len(custom_files)} custom domain(s)"))
        else:
            checks.append(("Custom domains", True, "None (using built-in only)"))
    else:
        checks.append(("Custom domains", True, "None (using built-in only)"))

    # 8. Pipeline readiness
    models_ok = reg_path.exists() and any(mdl_dir.glob("*.gguf")) if mdl_dir.exists() else False
    runtime_ok = rt_path.exists()
    hw_ok = hw_path.exists()
    pipeline_ready = models_ok and runtime_ok
    if pipeline_ready:
        checks.append(("Pipeline readiness", True, "Ready"))
    else:
        missing = []
        if not models_ok:
            missing.append("models")
        if not runtime_ok:
            missing.append("runtime config")
        if not hw_ok:
            missing.append("hardware profile")
        checks.append(("Pipeline readiness", False, f"Missing: {', '.join(missing)}"))

    # Display results
    console.print("[bold]Polymind Doctor[/]")
    console.print()

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Check", width=25)
    table.add_column("Status", width=10)
    table.add_column("Details")

    for name, ok, detail in checks:
        status = "[green]✓ OK[/]" if ok else "[red]✗ FAIL[/]"
        table.add_row(name, status, detail)

    console.print(table)

    # Summary
    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    console.print()
    if passed == total:
        console.print(f"[green]All {total} checks passed. System is healthy.[/]")
    else:
        console.print(f"[yellow]{passed}/{total} checks passed.[/]")
        console.print("[dim]Fix failing checks above before running the pipeline.[/]")


@app.command("check")
def doctor_check() -> None:
    """Run diagnostics (alias for 'doctor')."""
    doctor_callback()
