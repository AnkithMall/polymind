import os
import subprocess

import typer
import yaml

from polymind.core.config import load_config, update_logging_config
from polymind.core.paths import (
    ARTIFACT_DIR_ENV,
    MODEL_DIR_ENV,
    artifact_dir,
    config_path,
    hardware_path,
    logs_dir,
    model_dir,
    registry_path,
    runtime_path,
)
from polymind.core.runtime.artifact import load_runtime_config, write_runtime_config
from polymind.core.runtime.types import RuntimeConfig

app = typer.Typer()


@app.command("show")
def show(
    raw: bool = typer.Option(
        False,
        "--raw",
        "-r",
        help="Show raw YAML file contents instead of human-readable summaries.",
    ),
) -> None:
    """Show all current Polymind configurations.

    Displays a comprehensive summary of all Polymind settings:
    environment variables, resolved paths, per-model runtime
    configs, artifact files, and model directory contents.

    Examples:

        polymind config show

        polymind config show --raw
    """

    typer.echo("Polymind Configuration")
    typer.echo("=" * 50)
    typer.echo()

    # ---------------------------------------------------------
    # Environment Variables
    # ---------------------------------------------------------

    typer.echo("Environment Variables")
    typer.echo("-" * 50)

    env_vars = {
        ARTIFACT_DIR_ENV: os.environ.get(ARTIFACT_DIR_ENV),
        MODEL_DIR_ENV: os.environ.get(MODEL_DIR_ENV),
        "HF_TOKEN": os.environ.get("HF_TOKEN"),
    }

    for var, value in env_vars.items():
        if value:
            typer.echo(f"  {var} = {value}")
        else:
            typer.echo(f"  {var} = (not set)")

    typer.echo()

    # ---------------------------------------------------------
    # Resolved Paths
    # ---------------------------------------------------------

    typer.echo("Resolved Paths")
    typer.echo("-" * 50)

    paths = {
        "Artifact Directory": artifact_dir(),
        "Model Directory": model_dir(),
        "Hardware Profile": hardware_path(),
        "Runtime Config": runtime_path(),
        "Model Registry": registry_path(),
    }

    for name, path in paths.items():
        exists = path.exists()
        status = "✓" if exists else "✗"

        if path.is_dir():
            type_str = "dir"
        elif path.is_file():
            type_str = "file"
        else:
            type_str = "missing"

        typer.echo(f"  {status} {name:<20} {path} ({type_str})")

    typer.echo()

    # ---------------------------------------------------------
    # Runtime Configs (per-model)
    # ---------------------------------------------------------

    typer.echo("Runtime Configs")
    typer.echo("-" * 50)

    if runtime_path().exists():
        with runtime_path().open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        models = data.get("models", {})
        if models:
            for model_id, config in models.items():
                typer.echo(f"  Model {model_id}:")
                typer.echo(f"    gpu_layers:  {config.get('gpu_layers', -1)}")
                typer.echo(f"    threads:     {config.get('threads', 4)}")
                typer.echo(f"    context:     {config.get('context_size', 4096)}")
                typer.echo(f"    batch:       {config.get('batch_size', 512)}")

                # Show benchmark data if available
                benchmark = config.get("benchmark", {})
                if benchmark:
                    gen_tps = benchmark.get("generation_tps")
                    prompt_tps = benchmark.get("prompt_tps")
                    runs = benchmark.get("runs")
                    if gen_tps:
                        typer.echo("    benchmark:")
                        typer.echo(f"      gen speed:    {gen_tps} tok/s")
                        if prompt_tps:
                            typer.echo(f"      prompt speed: {prompt_tps} tok/s")
                        if runs:
                            typer.echo(f"      runs:         {runs}")
        else:
            typer.echo("  No model configs")
    else:
        typer.echo("  No runtime config file")

    typer.echo()

    # ---------------------------------------------------------
    # Artifact Files
    # ---------------------------------------------------------

    typer.echo("Artifact Files")
    typer.echo("-" * 50)

    artifact_files = [
        ("hardware.yaml", hardware_path()),
        ("runtime.yaml", runtime_path()),
        ("registry.yaml", registry_path()),
    ]

    for name, path in artifact_files:
        if path.exists():
            size = path.stat().st_size
            typer.echo(f"  ✓ {name:<20} {size:,} bytes")

            if raw:
                typer.echo("    Content:")
                content = path.read_text(encoding="utf-8")
                for line in content.splitlines():
                    typer.echo(f"      {line}")
                typer.echo()
        else:
            typer.echo(f"  ✗ {name:<20} (not created)")

    typer.echo()

    # ---------------------------------------------------------
    # Models Directory
    # ---------------------------------------------------------

    typer.echo("Models Directory")
    typer.echo("-" * 50)

    model_directory = model_dir()

    if model_directory.exists():
        gguf_files = list(model_directory.glob("*.gguf"))

        if gguf_files:
            typer.echo(f"  Found {len(gguf_files)} model(s):")
            for f in gguf_files:
                size = f.stat().st_size
                typer.echo(f"    - {f.name} ({size:,} bytes)")
        else:
            typer.echo("  No models found")
    else:
        typer.echo(f"  Directory does not exist: {model_directory}")

    typer.echo()

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    typer.echo("Summary")
    typer.echo("-" * 50)

    issues = []

    if not hardware_path().exists():
        issues.append("Hardware profile not created. Run: polymind hardware scan")

    if not registry_path().exists():
        issues.append("Model registry not created. Run: polymind model scan")

    if not model_dir().exists():
        issues.append("Models directory does not exist")

    if issues:
        for issue in issues:
            typer.echo(f"  ⚠ {issue}")
    else:
        typer.echo("  ✓ All configurations look good")


@app.command("get")
def get(
    key: str = typer.Argument(
        ...,
        help="Config key to retrieve, e.g. POLYMIND_ARTIFACT_DIR, HF_TOKEN, or runtime.<id>.gpu_layers.",
    ),
) -> None:
    """Get a configuration value.

    Retrieves a single configuration value by key. Supports
    environment variables (POLYMIND_*, HF_TOKEN) and runtime
    config fields (runtime.<model_id>.<field>).

    Examples:

        polymind config get POLYMIND_ARTIFACT_DIR

        polymind config get runtime.1.gpu_layers

        polymind config get HF_TOKEN
    """

    # Environment variables
    if key.startswith("POLYMIND_") or key == "HF_TOKEN":
        value = os.environ.get(key)
        if value:
            typer.echo(value)
        else:
            typer.echo("(not set)")
        return

    # Runtime config: runtime.<model_id>.<field>
    if key.startswith("runtime."):
        parts = key.split(".")
        if len(parts) == 3:
            _, model_id, field = parts
            config = load_runtime_config(model_id)
            if config is None:
                typer.echo(f"No config for model {model_id}", err=True)
                raise typer.Exit(code=1)

            value = getattr(config, field, None)
            if value is None:
                typer.echo(f"Unknown field: {field}", err=True)
                raise typer.Exit(code=1)

            typer.echo(value)
            return

    typer.echo(f"Unknown config key: {key}", err=True)
    typer.echo()
    typer.echo("Available keys:")
    typer.echo("  Environment: POLYMIND_ARTIFACT_DIR, POLYMIND_MODEL_DIR, HF_TOKEN")
    typer.echo("  Runtime:     runtime.<model_id>.gpu_layers")
    typer.echo("               runtime.<model_id>.threads")
    typer.echo("               runtime.<model_id>.context_size")
    typer.echo("               runtime.<model_id>.batch_size")
    raise typer.Exit(code=1)


@app.command("set")
def set_config(
    key: str = typer.Argument(
        ...,
        help="Config key to set, e.g. runtime.<id>.gpu_layers or POLYMIND_ARTIFACT_DIR.",
    ),
    value: str = typer.Argument(
        ...,
        help="New value to assign to the configuration key.",
    ),
) -> None:
    """Set a configuration value.

    Updates a configuration value by key. For environment variables,
    prints shell instructions. For runtime config fields, writes
    the value directly to the runtime config file.

    Examples:

        polymind config set runtime.1.gpu_layers 35

        polymind config set runtime.2.context_size 8192

        polymind config set POLYMIND_ARTIFACT_DIR /data/polymind
    """

    # Environment variables (print instruction)
    if key.startswith("POLYMIND_") or key == "HF_TOKEN":
        shell = os.environ.get("SHELL", "/bin/bash")
        if "zsh" in shell:
            rc = "~/.zshrc"
        elif "fish" in shell:
            rc = "~/.config/fish/config.fish"
        else:
            rc = "~/.bashrc"

        typer.echo(f"Add to {rc}:")
        typer.echo(f"  export {key}={value}")
        typer.echo()
        typer.echo("Or run:")
        typer.echo(f"  {key}={value} polymind <command>")
        return

    # Runtime config: runtime.<model_id>.<field>
    if key.startswith("runtime."):
        parts = key.split(".")
        if len(parts) != 3:
            typer.echo("Invalid runtime key format. Use: runtime.<model_id>.<field>", err=True)
            raise typer.Exit(code=1)

        _, model_id, field = parts

        valid_fields = {"gpu_layers", "threads", "context_size", "batch_size"}
        if field not in valid_fields:
            typer.echo(f"Unknown field: {field}", err=True)
            typer.echo(f"Valid fields: {', '.join(sorted(valid_fields))}")
            raise typer.Exit(code=1)

        # Parse value
        try:
            int_value = int(value)
        except ValueError:
            typer.echo(f"Value must be an integer, got: {value}", err=True)
            raise typer.Exit(code=1)

        # Load or create config
        config = load_runtime_config(model_id)
        if config is None:
            config = RuntimeConfig(model_id=model_id)

        # Set the field
        setattr(config, field, int_value)

        # Write back
        write_runtime_config(config)

        typer.echo(f"Set runtime.{model_id}.{field} = {int_value}")
        return

    typer.echo(f"Unknown config key: {key}", err=True)
    raise typer.Exit(code=1)


@app.command("edit")
def edit_config(
    file: str = typer.Option(
        "runtime",
        "--file",
        "-f",
        help="Config file to open: runtime (default), hardware, or registry.",
    ),
) -> None:
    """Open config file in editor.

    Opens a Polymind YAML config file in your default editor
    ($EDITOR or $VISUAL). Use this for manual edits to runtime,
    hardware, or registry configuration files.

    Examples:

        polymind config edit

        polymind config edit --file hardware

        polymind config edit -f registry
    """

    file_map = {
        "runtime": runtime_path(),
        "hardware": hardware_path(),
        "registry": registry_path(),
    }

    path = file_map.get(file)
    if path is None:
        typer.echo(f"Unknown file: {file}", err=True)
        typer.echo(f"Available: {', '.join(file_map.keys())}")
        raise typer.Exit(code=1)

    if not path.exists():
        typer.echo(f"File does not exist: {path}", err=True)
        raise typer.Exit(code=1)

    # Try common editors
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vi"))

    typer.echo(f"Opening {path} in {editor}...")

    try:
        subprocess.run([editor, str(path)], check=True)
    except FileNotFoundError:
        typer.echo(f"Editor not found: {editor}", err=True)
        typer.echo("Install an editor or set EDITOR environment variable")
        raise typer.Exit(code=1)
    except subprocess.CalledProcessError:
        typer.echo("Editor exited with error", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Done. Changes saved to {path}")


@app.command("logging")
def logging_show(
    raw: bool = typer.Option(
        False, "--raw", "-r", help="Show raw YAML of the logging configuration."
    ),
) -> None:
    """Show and manage pipeline logging configuration.

    Displays the current logging settings including enabled state,
    log directory, file rotation limits, and retention policies.
    Shows existing log files with their sizes.

    Examples:

        polymind config logging

        polymind config logging --raw
    """
    cfg = load_config()
    log_cfg = cfg.get("logging", {})

    typer.echo("Pipeline Logging Configuration")
    typer.echo("=" * 50)
    typer.echo()

    if raw:
        typer.echo(yaml.safe_dump({"logging": log_cfg}, default_flow_style=False))
        return

    typer.echo(f"  Enabled:          {log_cfg.get('enabled', True)}")
    typer.echo(f"  Log directory:    {logs_dir()}")
    typer.echo(f"  Max file size:    {log_cfg.get('max_bytes', 5_242_880) / 1024 / 1024:.1f} MB")
    typer.echo(f"  Backup count:     {log_cfg.get('backup_count', 3)}")
    typer.echo(f"  Max age (days):   {log_cfg.get('max_age_days', 7)}")
    typer.echo(f"  Max total files:  {log_cfg.get('max_total_files', 10)}")
    typer.echo(f"  Max total size:   {log_cfg.get('max_total_size_mb', 50)} MB")
    typer.echo(f"  Config file:      {config_path()}")
    typer.echo()

    # Show existing log files
    log_dir = logs_dir()
    log_files = sorted(log_dir.glob("*.log"))
    if log_files:
        typer.echo("Log Files")
        typer.echo("-" * 50)
        total_size = 0
        for f in log_files:
            size = f.stat().st_size
            total_size += size
            typer.echo(f"  {f.name:<30} {size / 1024:.1f} KB")
        typer.echo(f"  {'TOTAL':<30} {total_size / 1024:.1f} KB")
    else:
        typer.echo("[dim]No log files yet.[/]")


@app.command("logging-set")
def logging_set(
    key: str = typer.Argument(
        ...,
        help="Logging setting key: enabled, max_bytes, backup_count, max_age_days, max_total_files, or max_total_size_mb.",
    ),
    value: str = typer.Argument(
        ...,
        help="New value for the logging setting (type auto-detected from key).",
    ),
) -> None:
    """Update a logging configuration value.

    Modifies a single logging setting in the Polymind config file.
    Valid keys control log rotation, retention, and directory behavior.

    Examples:

        polymind config logging-set enabled false

        polymind config logging-set max_bytes 10485760

        polymind config logging-set max_age_days 14
    """
    valid_keys = {
        "enabled": bool,
        "max_bytes": int,
        "backup_count": int,
        "max_age_days": int,
        "max_total_files": int,
        "max_total_size_mb": int,
    }

    if key not in valid_keys:
        typer.echo(f"Unknown key: {key}", err=True)
        typer.echo(f"Valid keys: {', '.join(sorted(valid_keys))}")
        raise typer.Exit(code=1)

    target_type = valid_keys[key]

    if target_type is bool:
        parsed = value.lower() in ("true", "1", "yes", "on")
    else:
        try:
            parsed = target_type(value)
        except ValueError:
            typer.echo(f"Invalid {target_type.__name__} value: {value}", err=True)
            raise typer.Exit(code=1)

    update_logging_config(**{key: parsed})
    typer.echo(f"logging.{key} = {parsed}")


@app.command("pipeline")
def pipeline_show(
    raw: bool = typer.Option(
        False, "--raw", "-r", help="Show raw YAML of the pipeline configuration."
    ),
) -> None:
    """Show pipeline configuration from polymind.yaml.

    Displays the pipeline execution settings including concurrency
    limits, timeout, retry count, and minimum task score threshold.

    Examples:

        polymind config pipeline

        polymind config pipeline --raw
    """
    cfg = load_config()
    pipe_cfg = cfg.get("pipeline", {})

    typer.echo("Pipeline Configuration")
    typer.echo("=" * 50)
    typer.echo()

    if raw:
        typer.echo(yaml.safe_dump({"pipeline": pipe_cfg}, default_flow_style=False))
        return

    typer.echo(f"  max_concurrent:    {pipe_cfg.get('max_concurrent', 1)}")
    typer.echo(f"  timeout_seconds:   {pipe_cfg.get('timeout_seconds', 120)}")
    typer.echo(f"  max_retries:       {pipe_cfg.get('max_retries', 1)}")
    typer.echo(f"  min_task_score:    {pipe_cfg.get('min_task_score', 0.5)}")
    typer.echo(f"  Config file:       {config_path()}")


@app.command("pipeline-set")
def pipeline_set(
    key: str = typer.Argument(
        ...,
        help="Pipeline setting key: max_concurrent, timeout_seconds, max_retries, or min_task_score.",
    ),
    value: str = typer.Argument(
        ...,
        help="New value for the pipeline setting (type auto-detected from key).",
    ),
) -> None:
    """Update a pipeline configuration value.

    Modifies a single pipeline setting in the Polymind config file.
    Controls concurrency, timeouts, retries, and task scoring.

    Examples:

        polymind config pipeline-set max_concurrent 4

        polymind config pipeline-set timeout_seconds 300

        polymind config pipeline-set min_task_score 0.7
    """
    valid_keys = {
        "max_concurrent": int,
        "timeout_seconds": int,
        "max_retries": int,
        "min_task_score": float,
    }

    if key not in valid_keys:
        typer.echo(f"Unknown key: {key}", err=True)
        typer.echo(f"Valid keys: {', '.join(sorted(valid_keys))}")
        raise typer.Exit(code=1)

    target_type = valid_keys[key]

    try:
        parsed = target_type(value)
    except ValueError:
        typer.echo(f"Invalid {target_type.__name__} value: {value}", err=True)
        raise typer.Exit(code=1)

    cfg = load_config()
    pipe_cfg = cfg.get("pipeline", {})
    pipe_cfg[key] = parsed
    cfg["pipeline"] = pipe_cfg
    from polymind.core.config import save_config

    save_config(cfg)
    typer.echo(f"pipeline.{key} = {parsed}")
