from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, Label, Static


class RuntimeScreen(Widget):
    """Runtime management screen - run models and optimize settings."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._models = None

    def _get_models(self):
        if self._models is None:
            from polymind.core.model.registry import ModelRegistry

            self._models = ModelRegistry().load()
        return self._models

    def invalidate_cache(self) -> None:
        self._models = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("[b]Runtime Management[/b]", classes="title")
            yield Static("")

            with Horizontal(classes="button-row"):
                yield Button("Refresh", id="btn-refresh", variant="default")
                yield Button("Run Model", id="btn-run", variant="primary")
                yield Button("Optimize", id="btn-optimize", variant="warning")
                yield Button("Optimize All", id="btn-optimize-all", variant="default")

            yield Static("")
            yield Label("Installed models:")
            yield DataTable(id="runtime-model-table")

            yield Static("")

            with Vertical(id="panel-run", classes="hidden"):
                yield Static("Chat requires an interactive terminal.\nUse the CLI command below:")
                yield Static(
                    "  [cyan]uv run polymind runtime run -m <model_id>[/cyan]",
                    id="run-command",
                )

            with Vertical(id="panel-optimize", classes="hidden"):
                yield Label("Model ID to optimize:")
                yield Input(placeholder="Enter model ID...", id="optimize-model-input")
                yield Button("Run Optimization", id="btn-optimize-run", variant="warning")

            yield Static("")
            yield Static("", id="runtime-status")

    def on_mount(self) -> None:
        self._load_models()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        status = self.query_one("#runtime-status", Static)

        match event.button.id:
            case "btn-refresh":
                self.invalidate_cache()
                self._load_models()
                status.update("[green]Refreshed.[/green]")
            case "btn-run":
                self._show_panel("panel-run")
            case "btn-optimize":
                self._show_panel("panel-optimize")
            case "btn-optimize-all":
                self._show_panel(None)
                self._action_optimize(status, all_models=True)
            case "btn-optimize-run":
                self._action_optimize(status, all_models=False)

    def _show_panel(self, panel_id: str | None) -> None:
        for pid in ("panel-run", "panel-optimize"):
            panel = self.query_one(f"#{pid}")
            if pid == panel_id:
                panel.remove_class("hidden")
            else:
                panel.add_class("hidden")

    def _load_models(self) -> None:
        table = self.query_one("#runtime-model-table", DataTable)
        try:
            from pathlib import Path

            from polymind.core.model.utils import format_size

            models = self._get_models()
            table.clear()

            if not models:
                table.add_columns("Info")
                table.add_row("No models installed.")
                return

            table.add_columns("ID", "File", "Size", "Quant", "Status")
            for m in models:
                exists = Path(m.local_path).exists()
                table.add_row(
                    str(m.id),
                    m.filename,
                    format_size(m.size_bytes),
                    m.quantization or "?",
                    "ok" if exists else "missing",
                )
        except Exception as exc:
            table.clear()
            table.add_columns("Error")
            table.add_row(str(exc))

    def _action_optimize(self, status: Static, all_models: bool) -> None:
        model_ref = self.query_one("#optimize-model-input", Input).value.strip()

        if not all_models and not model_ref:
            status.update("[red]Enter a model ID or use 'Optimize All'.[/red]")
            return

        status.update("Loading hardware...")
        self.run_worker(lambda: self._do_optimize(status, model_ref, all_models), thread=True)

    def _do_optimize(self, status: Static, model_ref: str, all_models: bool) -> None:
        try:
            from pathlib import Path

            from polymind.core.hardware.loader import load_hardware_profile
            from polymind.core.paths import runtime_path
            from polymind.core.runtime.artifact import write_runtime_config
            from polymind.core.runtime.optimizer import optimize_config

            hardware = load_hardware_profile()
            models = self._get_models()

            if not models:
                self.app.call_from_thread(status.update, "[red]No models installed.[/red]")
                return

            if all_models:
                selected = [m for m in models if Path(m.local_path).exists()]
            else:
                selected = [m for m in models if str(m.id) == model_ref]
                if not selected:
                    self.app.call_from_thread(status.update, f"[red]Not found: {model_ref}[/red]")
                    return
                selected = [m for m in selected if Path(m.local_path).exists()]

            if not selected:
                self.app.call_from_thread(status.update, "[red]No model files found.[/red]")
                return

            total = len(selected)
            configs_written = 0

            for i, model in enumerate(selected, 1):
                model_path = Path(model.local_path)
                idx = i

                def on_progress(
                    msg: str,
                    current: int,
                    t: int,
                    name: str = model.filename,
                    n: int = idx,
                ) -> None:
                    self.app.call_from_thread(
                        status.update, f"[{n}/{total}] {name}: [{current}/{t}] {msg}"
                    )

                config, bench = optimize_config(
                    model_id=str(model.id),
                    model_size_bytes=model.size_bytes,
                    model_path=model_path,
                    hardware=hardware,
                    on_progress=on_progress,
                )

                if bench:
                    config.benchmark = {
                        "generation_tps": round(bench.eval_tokens_per_sec_median, 1),
                        "prompt_tps": round(bench.prompt_tokens_per_sec_median, 1),
                        "runs": bench.run_count,
                    }

                write_runtime_config(config)
                configs_written += 1

            self.invalidate_cache()
            self.app.call_from_thread(
                status.update,
                f"[green]Optimized {configs_written} model(s) -> {runtime_path()}[/green]",
            )
        except Exception as exc:
            self.app.call_from_thread(status.update, f"[red]Optimize failed: {exc}[/red]")
