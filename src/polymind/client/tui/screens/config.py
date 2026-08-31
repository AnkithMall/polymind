from __future__ import annotations

import os

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Select, Static

from polymind.core.paths import (
    ARTIFACT_DIR_ENV,
    MODEL_DIR_ENV,
    artifact_dir,
    hardware_path,
    model_dir,
    registry_path,
    runtime_path,
)


class ConfigScreen(Widget):
    """Configuration management screen - show, get, set, edit."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("[b]Configuration[/b]", classes="title")
            yield Static("")

            with Horizontal(classes="button-row"):
                yield Button("Show All", id="btn-show", variant="primary")
                yield Button("Get Value", id="btn-get", variant="default")
                yield Button("Set Value", id="btn-set", variant="default")
                yield Button("Edit File", id="btn-edit", variant="warning")

            yield Static("")

            # Action-specific panels
            with Vertical(id="panel-get"):
                yield Label("Key:")
                yield Input(
                    placeholder="e.g. runtime.2.gpu_layers or POLYMIND_ARTIFACT_DIR",
                    id="config-key",
                )
                yield Button("Get", id="btn-get-run", variant="primary")

            with Vertical(id="panel-set", classes="hidden"):
                yield Label("Key:")
                yield Input(
                    placeholder="e.g. runtime.2.gpu_layers",
                    id="config-set-key",
                )
                yield Label("Value:")
                yield Input(placeholder="Value to set...", id="config-set-value")
                yield Button("Set", id="btn-set-run", variant="warning")

            with Vertical(id="panel-edit", classes="hidden"):
                yield Label("Config file to edit:")
                yield Select(
                    [
                        ("runtime", "runtime"),
                        ("hardware", "hardware"),
                        ("registry", "registry"),
                    ],
                    id="edit-file-select",
                    value="runtime",
                )
                yield Static(
                    "[dim]Opens the file in $EDITOR. Press any key to continue after editing.[/dim]"
                )
                yield Button("Open in Editor", id="btn-edit-run", variant="warning")

            yield Static("")
            yield Static("", id="config-output")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        output = self.query_one("#config-output", Static)

        match event.button.id:
            case "btn-show":
                self._show_panel(None)
                self._action_show(output)
            case "btn-get":
                self._show_panel("panel-get")
            case "btn-set":
                self._show_panel("panel-set")
            case "btn-edit":
                self._show_panel("panel-edit")
            case "btn-get-run":
                self._action_get(output)
            case "btn-set-run":
                self._action_set(output)
            case "btn-edit-run":
                self._action_edit(output)

    def _show_panel(self, panel_id: str | None) -> None:
        for pid in ("panel-get", "panel-set", "panel-edit"):
            panel = self.query_one(f"#{pid}")
            if pid == panel_id:
                panel.remove_class("hidden")
            else:
                panel.add_class("hidden")

    def _action_show(self, output: Static) -> None:
        lines: list[str] = []
        lines.append("[b]Polymind Configuration[/b]")
        lines.append("=" * 50)
        lines.append("")

        # Environment Variables
        lines.append("[b]Environment Variables[/b]")
        env_vars = {
            ARTIFACT_DIR_ENV: os.environ.get(ARTIFACT_DIR_ENV),
            MODEL_DIR_ENV: os.environ.get(MODEL_DIR_ENV),
            "HF_TOKEN": os.environ.get("HF_TOKEN"),
        }
        for var, value in env_vars.items():
            lines.append(f"  {var} = {value or '(not set)'}")
        lines.append("")

        # Resolved Paths
        lines.append("[b]Resolved Paths[/b]")
        paths = {
            "Artifact Directory": artifact_dir(),
            "Model Directory": model_dir(),
            "Hardware Profile": hardware_path(),
            "Runtime Config": runtime_path(),
            "Model Registry": registry_path(),
        }
        for name, path in paths.items():
            exists = path.exists()
            status = "[green]OK[/green]" if exists else "[red]MISSING[/red]"
            ptype = "dir" if path.is_dir() else ("file" if path.is_file() else "missing")
            lines.append(f"  {status} {name}: {path} ({ptype})")
        lines.append("")

        # Runtime Configs
        lines.append("[b]Runtime Configs[/b]")
        if runtime_path().exists():
            import yaml

            with runtime_path().open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            models = data.get("models", {})
            if models:
                for model_id, cfg in models.items():
                    lines.append(f"  Model {model_id}:")
                    lines.append(f"    gpu_layers:  {cfg.get('gpu_layers', -1)}")
                    lines.append(f"    threads:     {cfg.get('threads', 4)}")
                    lines.append(f"    context:     {cfg.get('context_size', 4096)}")
                    lines.append(f"    batch:       {cfg.get('batch_size', 512)}")
                    benchmark = cfg.get("benchmark", {})
                    if benchmark:
                        gen_tps = benchmark.get("generation_tps")
                        if gen_tps:
                            lines.append(f"    benchmark:   {gen_tps} tok/s")
            else:
                lines.append("  No model configs")
        else:
            lines.append("  No runtime config file")
        lines.append("")

        # Artifact Files
        lines.append("[b]Artifact Files[/b]")
        for name, path in [
            ("hardware.yaml", hardware_path()),
            ("runtime.yaml", runtime_path()),
            ("registry.yaml", registry_path()),
        ]:
            if path.exists():
                size = path.stat().st_size
                lines.append(f"  [green]OK[/green] {name}: {size:,} bytes")
            else:
                lines.append(f"  [red]MISSING[/red] {name}")
        lines.append("")

        # Summary
        lines.append("[b]Summary[/b]")
        issues = []
        if not hardware_path().exists():
            issues.append("Hardware profile not created (run: polymind hardware scan)")
        if not registry_path().exists():
            issues.append("Model registry not created (run: polymind model scan)")
        if not model_dir().exists():
            issues.append("Models directory does not exist")

        if issues:
            for issue in issues:
                lines.append(f"  [yellow]![/yellow] {issue}")
        else:
            lines.append("  [green]All configurations look good[/green]")

        output.update("\n".join(lines))

    def _action_get(self, output: Static) -> None:
        key = self.query_one("#config-key", Input).value.strip()
        if not key:
            output.update("[red]Please enter a key.[/red]")
            return

        # Environment variables
        if key.startswith("POLYMIND_") or key == "HF_TOKEN":
            value = os.environ.get(key)
            output.update(f"{key} = {value or '(not set)'}")
            return

        # Runtime config
        if key.startswith("runtime."):
            parts = key.split(".")
            if len(parts) == 3:
                _, model_id, field = parts
                try:
                    from polymind.core.runtime.artifact import load_runtime_config

                    config = load_runtime_config(model_id)
                    if config is None:
                        output.update(f"[red]No config for model {model_id}[/red]")
                        return
                    value = getattr(config, field, None)
                    if value is None:
                        output.update(f"[red]Unknown field: {field}[/red]")
                        return
                    output.update(f"{key} = {value}")
                except Exception as exc:
                    output.update(f"[red]Error: {exc}[/red]")
                return

        output.update(
            f"[red]Unknown config key: {key}[/red]\n\n"
            "Available keys:\n"
            "  Environment: POLYMIND_ARTIFACT_DIR, POLYMIND_MODEL_DIR, HF_TOKEN\n"
            "  Runtime: runtime.<model_id>.gpu_layers|threads|context_size|batch_size"
        )

    def _action_set(self, output: Static) -> None:
        key = self.query_one("#config-set-key", Input).value.strip()
        value = self.query_one("#config-set-value", Input).value.strip()

        if not key or not value:
            output.update("[red]Please enter both key and value.[/red]")
            return

        # Environment variables - print instruction
        if key.startswith("POLYMIND_") or key == "HF_TOKEN":
            shell = os.environ.get("SHELL", "/bin/bash")
            rc = (
                "~/.zshrc"
                if "zsh" in shell
                else ("~/.config/fish/config.fish" if "fish" in shell else "~/.bashrc")
            )
            output.update(
                f"Add to {rc}:\n  export {key}={value}\n\nOr run:\n  {key}={value} polymind <command>"
            )
            return

        # Runtime config
        if key.startswith("runtime."):
            parts = key.split(".")
            if len(parts) != 3:
                output.update("[red]Invalid format. Use: runtime.<model_id>.<field>[/red]")
                return

            _, model_id, field = parts
            valid_fields = {"gpu_layers", "threads", "context_size", "batch_size"}
            if field not in valid_fields:
                output.update(
                    f"[red]Unknown field: {field}. Valid: {', '.join(sorted(valid_fields))}[/red]"
                )
                return

            try:
                int_value = int(value)
            except ValueError:
                output.update(f"[red]Value must be an integer, got: {value}[/red]")
                return

            try:
                from polymind.core.runtime.artifact import load_runtime_config, write_runtime_config
                from polymind.core.runtime.types import RuntimeConfig

                config = load_runtime_config(model_id)
                if config is None:
                    config = RuntimeConfig(model_id=model_id)
                setattr(config, field, int_value)
                write_runtime_config(config)
                output.update(f"[green]Set runtime.{model_id}.{field} = {int_value}[/green]")
            except Exception as exc:
                output.update(f"[red]Error: {exc}[/red]")
            return

        output.update(f"[red]Unknown config key: {key}[/red]")

    def _action_edit(self, output: Static) -> None:
        file_choice = self.query_one("#edit-file-select", Select).value

        file_map = {
            "runtime": runtime_path(),
            "hardware": hardware_path(),
            "registry": registry_path(),
        }

        path = file_map.get(file_choice)
        if path is None:
            output.update(f"[red]Unknown file: {file_choice}[/red]")
            return

        if not path.exists():
            output.update(f"[red]File does not exist: {path}[/red]")
            return

        editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vi"))
        output.update(
            f"[yellow]Cannot open editor from within TUI.[/yellow]\n\n"
            f"Run this command in a separate terminal:\n\n"
            f"  {editor} {path}\n\n"
            f"Or use the CLI:\n  polymind config edit -f {file_choice}"
        )
