import typer

from polymind.core.hardware.artifact import write_hardware_profile
from polymind.core.hardware.loader import load_hardware_profile
from polymind.core.hardware.scanner import scan_hardware
from polymind.core.hardware.validator import validate_hardware_file
from polymind.core.paths import hardware_path

app = typer.Typer()


@app.command("scan")
def scan() -> None:
    """Scan system hardware and create a hardware profile.

    Detects CPU, memory, GPUs, and llama.cpp availability,
    then writes the profile to .polymind/hardware.yaml.
    Run this after hardware changes or first-time setup.

    Examples:

        polymind hardware scan
    """

    typer.echo("Scanning hardware...")

    profile = scan_hardware()
    path = write_hardware_profile(profile)

    typer.echo(f"Hardware profile written to {path}")


@app.command("show")
def show() -> None:
    """Display the current hardware profile.

    Shows detailed information about the detected system including
    CPU model and cores, memory, GPU details (VRAM, backend,
    llama.cpp support), and GPU selection status.

    Examples:

        polymind hardware show
    """

    try:
        profile = load_hardware_profile()
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo()
    typer.echo("System")
    typer.echo("------")
    typer.echo(f"OS:           {profile.system.operating_system}")
    typer.echo(f"Architecture: {profile.system.architecture}")
    typer.echo(f"Kernel:       {profile.system.kernel}")

    typer.echo()
    typer.echo("CPU")
    typer.echo("---")
    typer.echo(f"Model:          {profile.cpu.model}")
    typer.echo(f"Physical cores: {profile.cpu.physical_cores}")
    typer.echo(f"Logical cores:  {profile.cpu.logical_cores}")

    typer.echo()
    typer.echo("Memory")
    typer.echo("------")
    typer.echo(f"Total: {profile.memory.total_bytes / (1024**3):.2f} GiB")

    typer.echo()
    typer.echo("GPUs")
    typer.echo("----")

    if not profile.gpus:
        typer.echo("No GPUs detected.")
    else:
        for gpu in profile.gpus:
            total_gib = gpu.memory.total_bytes / (1024**3)

            if gpu.memory.available_bytes is not None:
                available_gib = gpu.memory.available_bytes / (1024**3)
            else:
                available_gib = None

            typer.echo(f"[{gpu.id}] {gpu.vendor} {gpu.model}")
            typer.echo(f"    PCI:       {gpu.pci_address}")
            typer.echo(f"    VRAM:      {total_gib:.2f} GiB")

            if available_gib is not None:
                typer.echo(f"    Available: {available_gib:.2f} GiB")

            typer.echo(f"    Backend:   {gpu.compute.backend or 'none'}")
            typer.echo(f"    llama.cpp: {'yes' if gpu.compute.llama_cpp_usable else 'no'}")
            typer.echo(f"    Selected:  {'yes' if gpu.selection.enabled else 'no'}")

    typer.echo()
    typer.echo("llama.cpp")
    typer.echo("---------")
    typer.echo(f"Available:     {'yes' if profile.llama_cpp.available else 'no'}")
    typer.echo(f"Backends:      {', '.join(profile.llama_cpp.backends) or 'none'}")
    typer.echo(f"Usable GPUs:   {profile.llama_cpp.usable_gpus}")
    typer.echo(f"Selected GPUs: {profile.llama_cpp.selected_gpus}")
    typer.echo(f"Multi-GPU:     {'yes' if profile.llama_cpp.multi_gpu_available else 'no'}")


@app.command("validate")
def validate() -> None:
    """Validate the current hardware profile.

    Checks the hardware profile file for consistency and reports
    any errors or warnings. Use this to diagnose issues with
    model compatibility or GPU selection.

    Examples:

        polymind hardware validate
    """

    path = hardware_path()

    result = validate_hardware_file(path)

    if result.errors:
        typer.echo("Hardware profile is invalid.")
        typer.echo()

        for error in result.errors:
            typer.echo(f"ERROR: {error}")

    if result.warnings:
        typer.echo()

        for warning in result.warnings:
            typer.echo(f"WARNING: {warning}")

    if result.valid:
        typer.echo()
        typer.echo("Hardware profile is valid.")

        try:
            profile = load_hardware_profile(path)
        except (FileNotFoundError, ValueError):
            raise typer.Exit(code=1)

        selected = profile.llama_cpp.selected_gpus

        if selected:
            typer.echo("Selected GPUs: " + ", ".join(str(gpu_id) for gpu_id in selected))
        else:
            typer.echo("Selected GPUs: none")

        return

    raise typer.Exit(code=1)
