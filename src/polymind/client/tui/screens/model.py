from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, Label, Select, Static


class ModelScreen(Widget):
    """Model management screen with search, list, download, delete, scan, migrate."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._hardware = None
        self._registry = None

    def _get_hardware(self):
        if self._hardware is None:
            from polymind.core.hardware.loader import load_hardware_profile

            self._hardware = load_hardware_profile()
        return self._hardware

    def _get_registry(self):
        if self._registry is None:
            from polymind.core.model.registry import ModelRegistry

            self._registry = ModelRegistry()
        return self._registry

    def invalidate_cache(self) -> None:
        self._hardware = None
        self._registry = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("[b]Model Management[/b]", classes="title")
            yield Static("")

            with Horizontal(classes="button-row"):
                yield Button("Search", id="btn-search", variant="primary")
                yield Button("List", id="btn-list", variant="default")
                yield Button("Download", id="btn-download", variant="default")
                yield Button("Delete", id="btn-delete", variant="error")
                yield Button("Scan", id="btn-scan", variant="default")
                yield Button("Migrate", id="btn-migrate", variant="warning")

            yield Static("")

            with Vertical(id="panel-search"):
                yield Label("Search query:")
                yield Input(placeholder="Enter search query...", id="search-input")

            with Vertical(id="panel-download", classes="hidden"):
                yield Label("Repository ID:")
                yield Input(
                    placeholder="e.g. bartowski/Llama-3.2-3B-Instruct-GGUF",
                    id="download-repo",
                )
                yield Label("Filename:")
                yield Input(placeholder="e.g. gguf-q4_k_m.gguf", id="download-filename")
                yield Label("Output directory (optional):")
                yield Input(placeholder="Defaults to .polymind/models/", id="download-output")
                yield Button("Start Download", id="btn-download-start", variant="primary")

            with Vertical(id="panel-delete", classes="hidden"):
                yield Label("Model to delete (ID, filename, or repo_id):")
                yield Input(placeholder="Enter model reference...", id="delete-ref")
                yield Label("Force delete without confirmation:")
                yield Select(
                    [("No", "no"), ("Yes", "yes")],
                    id="delete-force",
                    value="no",
                )
                yield Button("Delete Model", id="btn-delete-start", variant="error")

            with Vertical(id="panel-migrate", classes="hidden"):
                yield Label("Source directory (leave empty for auto-detect):")
                yield Input(
                    placeholder="e.g. polymind/models/ or ~/.cache/polymind/models/",
                    id="migrate-source",
                )
                yield Label("Skip confirmation:")
                yield Select(
                    [("No", "no"), ("Yes", "yes")],
                    id="migrate-force",
                    value="no",
                )
                yield Button("Start Migration", id="btn-migrate-start", variant="warning")

            yield Static("")
            yield DataTable(id="model-table")
            yield Static("")
            yield Static("[i]Select an action above.[/i]", id="model-status")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        status = self.query_one("#model-status", Static)
        table = self.query_one("#model-table", DataTable)

        match event.button.id:
            case "btn-search":
                self._show_panel("panel-search")
                self._action_search(table, status)
            case "btn-list":
                self._show_panel(None)
                self._action_list(table, status)
            case "btn-download":
                self._show_panel("panel-download")
            case "btn-delete":
                self._show_panel("panel-delete")
            case "btn-scan":
                self._show_panel(None)
                self._action_scan(table, status)
            case "btn-migrate":
                self._show_panel("panel-migrate")
            case "btn-download-start":
                self._action_download(status)
            case "btn-delete-start":
                self._action_delete(status)
            case "btn-migrate-start":
                self._action_migrate(status)

    def _show_panel(self, panel_id: str | None) -> None:
        for pid in ("panel-search", "panel-download", "panel-delete", "panel-migrate"):
            panel = self.query_one(f"#{pid}")
            if pid == panel_id:
                panel.remove_class("hidden")
            else:
                panel.add_class("hidden")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _action_search(self, table: DataTable, status: Static) -> None:
        query = self.query_one("#search-input", Input).value.strip()
        if not query:
            status.update("[red]Please enter a search query.[/red]")
            return
        status.update(f"Searching for '{query}'...")
        self.run_worker(lambda: self._do_search(table, status, query), thread=True)

    def _do_search(self, table: DataTable, status: Static, query: str) -> None:
        try:
            from polymind.core.model.hf import HuggingFaceClient
            from polymind.core.model.ranking import rank_models
            from polymind.core.model.search import ModelSearchOptions

            hardware = self._get_hardware()
            models = HuggingFaceClient().search_gguf(ModelSearchOptions(query=query, limit=20))

            if not models:
                self.app.call_from_thread(status.update, "No GGUF models found.")
                return

            ranked = rank_models(models, hardware)
            registry = self._get_registry()
            for m in ranked:
                if registry.find_by_filename(m.filename):
                    m.downloaded = True

            def _update() -> None:
                table.clear()
                table.add_columns("Name", "Repo", "Quant", "Size", "VRAM", "DL")
                for m in ranked[:20]:
                    size = f"{m.size_bytes / (1024**3):.2f}G" if m.size_bytes else "?"
                    table.add_row(
                        m.model_name,
                        m.repo_id,
                        m.quantization or "?",
                        size,
                        m.vram_status,
                        "*" if m.downloaded else "",
                    )
                status.update(f"Found {len(ranked)} model(s).")

            self.app.call_from_thread(_update)
        except Exception as exc:
            self.app.call_from_thread(status.update, f"[red]Search failed: {exc}[/red]")

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def _action_list(self, table: DataTable, status: Static) -> None:
        status.update("Loading...")
        self.run_worker(lambda: self._do_list(table, status), thread=True)

    def _do_list(self, table: DataTable, status: Static) -> None:
        try:
            from pathlib import Path

            from polymind.core.model.utils import format_size

            models = self._get_registry().load()

            if not models:
                self.app.call_from_thread(status.update, "No models installed.")
                return

            def _update() -> None:
                table.clear()
                table.add_columns("ID", "Repo", "File", "Size", "Quant", "Status")
                for m in models:
                    exists = Path(m.local_path).exists()
                    table.add_row(
                        str(m.id),
                        m.repo_id,
                        m.filename,
                        format_size(m.size_bytes),
                        m.quantization or "?",
                        "ok" if exists else "missing",
                    )
                status.update(f"{len(models)} installed model(s).")

            self.app.call_from_thread(_update)
        except Exception as exc:
            self.app.call_from_thread(status.update, f"[red]Failed: {exc}[/red]")

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _action_download(self, status: Static) -> None:
        repo_id = self.query_one("#download-repo", Input).value.strip()
        filename = self.query_one("#download-filename", Input).value.strip()
        output_raw = self.query_one("#download-output", Input).value.strip()

        if not repo_id or not filename:
            status.update("[red]Enter both repository ID and filename.[/red]")
            return

        status.update(f"Downloading {filename}...")
        self.run_worker(
            lambda: self._do_download(status, repo_id, filename, output_raw),
            thread=True,
        )

    def _do_download(self, status: Static, repo_id: str, filename: str, output_raw: str) -> None:
        try:
            from pathlib import Path

            from polymind.core.model.hf import HuggingFaceClient, detect_quantization
            from polymind.core.model.utils import file_size_bytes, format_size
            from polymind.core.paths import model_dir

            registry = self._get_registry()
            existing = registry.find_by_repo_and_filename(repo_id, filename)
            if existing and Path(existing.local_path).exists():
                self.app.call_from_thread(
                    status.update,
                    f"[yellow]Already downloaded: {existing.local_path}[/yellow]",
                )
                return

            output = Path(output_raw) if output_raw else model_dir()
            output.mkdir(parents=True, exist_ok=True)

            client = HuggingFaceClient()
            try:
                path = client.download(repo_id=repo_id, filename=filename, local_dir=str(output))
            except Exception as exc:
                expected = output / filename
                if expected.exists() and expected.stat().st_size > 0:
                    path = str(expected)
                else:
                    self.app.call_from_thread(status.update, f"[red]Download failed: {exc}[/red]")
                    return

            local_path = Path(path)
            if not local_path.exists():
                self.app.call_from_thread(status.update, "[red]File not found.[/red]")
                return

            model = registry.add(
                repo_id=repo_id,
                filename=filename,
                local_path=local_path,
                size_bytes=file_size_bytes(local_path),
                quantization=detect_quantization(filename),
            )
            self.invalidate_cache()

            self.app.call_from_thread(
                status.update,
                f"[green]Downloaded![/green] ID={model.id} "
                f"Size={format_size(model.size_bytes)} Path={model.local_path}",
            )
        except Exception as exc:
            self.app.call_from_thread(status.update, f"[red]Download failed: {exc}[/red]")

    # ------------------------------------------------------------------
    # Delete (sync — fast)
    # ------------------------------------------------------------------

    def _action_delete(self, status: Static) -> None:
        model_ref = self.query_one("#delete-ref", Input).value.strip()
        force = self.query_one("#delete-force", Select).value == "yes"

        if not model_ref:
            status.update("[red]Enter a model ID, filename, or repo_id.[/red]")
            return

        try:
            from pathlib import Path

            from polymind.core.model.utils import format_size

            registry = self._get_registry()
            models = registry.load()
            if not models:
                status.update("[red]No models installed.[/red]")
                return

            selected = next((m for m in models if str(m.id) == model_ref), None)
            if selected is None:
                selected = next((m for m in models if m.filename == model_ref), None)
            if selected is None:
                selected = next((m for m in models if m.repo_id == model_ref), None)
            if selected is None:
                status.update(f"[red]Not found: {model_ref}[/red]")
                return

            if not force:
                status.update(
                    f"Delete: {selected.filename} (ID {selected.id})?\n"
                    f"Size: {format_size(selected.size_bytes)}\n"
                    f"[yellow]Set Force=Yes and click Delete again.[/yellow]"
                )
                return

            file_path = Path(selected.local_path)
            if file_path.exists():
                file_path.unlink()
            registry.remove(selected.id)
            self.invalidate_cache()
            status.update(f"[green]Deleted {selected.filename} (ID {selected.id}).[/green]")
        except Exception as exc:
            status.update(f"[red]Delete failed: {exc}[/red]")

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    def _action_scan(self, table: DataTable, status: Static) -> None:
        status.update("Scanning...")
        self.run_worker(lambda: self._do_scan(table, status), thread=True)

    def _do_scan(self, table: DataTable, status: Static) -> None:
        try:
            from polymind.core.model.utils import format_size
            from polymind.core.paths import model_dir

            registry = self._get_registry()
            target = model_dir()
            if not target.exists():
                self.app.call_from_thread(status.update, f"Model dir not found: {target}")
                return

            report = registry.scan(target)
            self.invalidate_cache()

            def _update() -> None:
                table.clear()
                table.add_columns("Action", "Model", "Details")
                for m in report.added or []:
                    table.add_row("Added", m.filename, format_size(m.size_bytes))
                for m in report.fixed or []:
                    table.add_row("Fixed", m.filename, m.local_path)
                for m in report.missing or []:
                    table.add_row("Missing", m.filename, m.local_path)
                status.update(
                    f"Scan: +{len(report.added or [])} "
                    f"fixed={len(report.fixed or [])} "
                    f"dup={report.duplicates_removed} "
                    f"missing={len(report.missing or [])}"
                )

            self.app.call_from_thread(_update)
        except Exception as exc:
            self.app.call_from_thread(status.update, f"[red]Scan failed: {exc}[/red]")

    # ------------------------------------------------------------------
    # Migrate
    # ------------------------------------------------------------------

    def _action_migrate(self, status: Static) -> None:
        source_raw = self.query_one("#migrate-source", Input).value.strip()
        force = self.query_one("#migrate-force", Select).value == "yes"
        self.run_worker(lambda: self._do_migrate(status, source_raw, force), thread=True)

    def _do_migrate(self, status: Static, source_raw: str, force: bool) -> None:
        try:
            import shutil
            from pathlib import Path

            from polymind.core.model.utils import format_size
            from polymind.core.paths import model_dir

            source = None
            if source_raw:
                source = Path(source_raw)
            else:
                for loc in [
                    Path("polymind/models"),
                    Path.home() / ".cache" / "polymind" / "models",
                ]:
                    if loc.exists() and any(loc.glob("*.gguf")):
                        source = loc
                        break

            if source is None:
                self.app.call_from_thread(
                    status.update, "[red]No legacy models found.[/red] Specify source dir."
                )
                return

            if not source.exists():
                self.app.call_from_thread(status.update, f"[red]Not found: {source}[/red]")
                return

            gguf_files = list(source.glob("*.gguf"))
            if not gguf_files:
                self.app.call_from_thread(status.update, f"[red]No GGUF in {source}[/red]")
                return

            target = model_dir()
            if not force:
                lines = [f"Migrate {len(gguf_files)} files from {source} to {target}?"]
                for f in gguf_files:
                    lines.append(f"  - {f.name} ({format_size(f.stat().st_size)})")
                lines.append("[yellow]Set Skip=Yes and click again.[/yellow]")
                self.app.call_from_thread(status.update, "\n".join(lines))
                return

            target.mkdir(parents=True, exist_ok=True)
            migrated = 0
            for f in gguf_files:
                dest = target / f.name
                if not dest.exists():
                    shutil.move(str(f), str(dest))
                    migrated += 1

            if migrated > 0 and not list(source.glob("*.gguf")):
                try:
                    source.rmdir()
                except Exception:
                    pass

            if migrated > 0:
                from polymind.core.model.registry import ModelRegistry

                ModelRegistry().scan(target)
                self.invalidate_cache()

            self.app.call_from_thread(
                status.update, f"Migrated {migrated}/{len(gguf_files)} models."
            )
        except Exception as exc:
            self.app.call_from_thread(status.update, f"[red]Migration failed: {exc}[/red]")
