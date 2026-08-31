from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Button, Static


class HardwareScreen(Widget):
    """Hardware management screen - scan, show, validate."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._profile = None

    def _get_profile(self):
        if self._profile is None:
            from polymind.core.hardware.loader import load_hardware_profile

            self._profile = load_hardware_profile()
        return self._profile

    def invalidate_cache(self) -> None:
        self._profile = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("[b]Hardware Management[/b]", classes="title")
            yield Static("")

            with Horizontal(classes="button-row"):
                yield Button("Scan", id="btn-scan", variant="primary")
                yield Button("Show", id="btn-show", variant="default")
                yield Button("Validate", id="btn-validate", variant="warning")

            yield Static("")
            yield Static("", id="hardware-output")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        output = self.query_one("#hardware-output", Static)

        match event.button.id:
            case "btn-scan":
                self._action_scan(output)
            case "btn-show":
                self._action_show(output)
            case "btn-validate":
                self._action_validate(output)

    def _action_scan(self, output: Static) -> None:
        output.update("Scanning hardware...")
        self.run_worker(lambda: self._do_scan(output), thread=True)

    def _do_scan(self, output: Static) -> None:
        try:
            from polymind.core.hardware.artifact import write_hardware_profile
            from polymind.core.hardware.scanner import scan_hardware

            profile = scan_hardware()
            path = write_hardware_profile(profile)
            self._profile = profile
            self.app.call_from_thread(
                output.update, f"[green]Hardware profile written to {path}[/green]"
            )
        except Exception as exc:
            self.app.call_from_thread(output.update, f"[red]Scan failed: {exc}[/red]")

    def _action_show(self, output: Static) -> None:
        try:
            profile = self._get_profile()
        except (FileNotFoundError, ValueError) as exc:
            output.update(f"[red]Error: {exc}[/red]")
            return

        lines: list[str] = []
        lines.append("[b]System[/b]")
        lines.append(f"  OS:           {profile.system.operating_system}")
        lines.append(f"  Architecture: {profile.system.architecture}")
        lines.append(f"  Kernel:       {profile.system.kernel}")
        lines.append("")

        lines.append("[b]CPU[/b]")
        lines.append(f"  Model:          {profile.cpu.model}")
        lines.append(f"  Physical cores: {profile.cpu.physical_cores}")
        lines.append(f"  Logical cores:  {profile.cpu.logical_cores}")
        lines.append("")

        lines.append("[b]Memory[/b]")
        lines.append(f"  Total: {profile.memory.total_bytes / (1024**3):.2f} GiB")
        lines.append("")

        lines.append("[b]GPUs[/b]")
        if not profile.gpus:
            lines.append("  No GPUs detected.")
        else:
            for gpu in profile.gpus:
                total_gib = gpu.memory.total_bytes / (1024**3)
                avail_gib = (
                    gpu.memory.available_bytes / (1024**3) if gpu.memory.available_bytes else None
                )
                lines.append(f"  [{gpu.id}] {gpu.vendor} {gpu.model}")
                lines.append(f"      VRAM:      {total_gib:.2f} GiB")
                if avail_gib is not None:
                    lines.append(f"      Available: {avail_gib:.2f} GiB")
                lines.append(f"      llama.cpp: {'yes' if gpu.compute.llama_cpp_usable else 'no'}")
                lines.append(f"      Selected:  {'yes' if gpu.selection.enabled else 'no'}")
        lines.append("")

        lines.append("[b]llama.cpp[/b]")
        lines.append(f"  Available:     {'yes' if profile.llama_cpp.available else 'no'}")
        lines.append(f"  Backends:      {', '.join(profile.llama_cpp.backends) or 'none'}")
        lines.append(f"  Selected GPUs: {profile.llama_cpp.selected_gpus}")

        output.update("\n".join(lines))

    def _action_validate(self, output: Static) -> None:
        try:
            from polymind.core.hardware.validator import validate_hardware_file
            from polymind.core.paths import hardware_path

            result = validate_hardware_file(hardware_path())

            lines: list[str] = []
            if result.errors:
                lines.append("[red]Invalid.[/red]")
                for e in result.errors:
                    lines.append(f"  [red]{e}[/red]")
            if result.warnings:
                for w in result.warnings:
                    lines.append(f"  [yellow]{w}[/yellow]")
            if result.valid:
                lines.append("[green]Hardware profile is valid.[/green]")
                try:
                    profile = self._get_profile()
                    sel = profile.llama_cpp.selected_gpus
                    lines.append(f"  GPUs: {', '.join(str(g) for g in sel) if sel else 'none'}")
                except Exception:
                    pass

            output.update("\n".join(lines))
        except Exception as exc:
            output.update(f"[red]Validate failed: {exc}[/red]")
