from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rich.text import Text as RichText
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Static,
)

from polymind.core import (
    ALL_DOMAINS,
    AnalyzerPlan,
    Config,
    DomainType,
    ExecutionSchedule,
    ExecutionStrategy,
    PipelineResult,
    SubtaskResult,
    analyze_prompt,
    build_schedule,
    execute_subtask,
    load_ranks,
    resolve_model_string,
    run_benchmark,
    save_ranks,
    synthesize,
)


class ShortcutsScreen(ModalScreen):
    def compose(self) -> ComposeResult:
        yield Static(
            "\n".join(
                [
                    "[bold]Keyboard Shortcuts[/]",
                    "",
                    "Ctrl+R    Review / override subtask plan",
                    "Ctrl+M    Cycle execution strategy",
                    "Ctrl+P    Open profile picker",
                    "Ctrl+B    Run benchmark in background",
                    "Ctrl+S    Save current session",
                    "Ctrl+O    Load a session",
                    "?         Toggle this help",
                    "Q         Quit",
                    "",
                    "Press any key to close.",
                ]
            ),
            id="shortcuts-content",
        )

    def on_key(self, event) -> None:
        self.dismiss()


class SubtaskOverrideScreen(ModalScreen):
    def __init__(self, plan: AnalyzerPlan) -> None:
        super().__init__()
        self.plan = plan

    def compose(self) -> ComposeResult:
        yield Static(f"[bold]Subtask Plan Review[/]\n", id="override-header")
        for s in self.plan.subtasks:
            deps = f" depends_on=[{', '.join(s.depends_on)}]" if s.depends_on else ""
            yield Label(
                f"[cyan]{s.id}[/] ([yellow]{s.domain.value}[/]): {s.prompt}{deps}"
            )
        yield Label("")
        yield Label("Press [bold]Y[/] to proceed, [bold]N[/] to cancel.")

    def on_key(self, event) -> None:
        if event.key == "y":
            self.dismiss(True)
        elif event.key == "n":
            self.dismiss(False)


class ProfilePicker(ModalScreen):
    def compose(self) -> ComposeResult:
        yield Static("[bold]Select Routing Profile[/]\n")
        yield ListView(
            ListItem(Label("quality")),
            ListItem(Label("fast")),
            ListItem(Label("private")),
            ListItem(Label("custom")),
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.children[0].renderable)


class ChatMessage(Static):
    pass


class PipelinePanel(Vertical):
    def compose(self) -> ComposeResult:
        yield Static("[bold]Pipeline Inspector[/]", id="pipeline-title")
        yield Static("Strategy: model_aware", id="strategy-label")
        yield Static("Ready", id="status-label")
        yield RichLog(highlight=True, markup=True, id="pipeline-log")

    def _log(self) -> RichLog:
        return self.query_one("#pipeline-log", RichLog)

    def _strategy_label(self) -> Static:
        return self.query_one("#strategy-label", Static)

    def _status_label(self) -> Static:
        return self.query_one("#status-label", Static)

    def log_line(self, text: str) -> None:
        self._log().write(text)

    def set_strategy(self, strategy: ExecutionStrategy) -> None:
        self._strategy_label().update(f"Strategy: [cyan]{strategy.value}[/]")

    def set_status(self, text: str) -> None:
        self._status_label().update(f"[green]{text}[/]")


class PolyMindApp(App):
    CSS = """
    Screen {
        background: #1a1b26;
    }

    #main-layout {
        layout: grid;
        grid-size: 2 2;
        grid-rows: 1fr auto;
    }

    #chat-panel {
        background: #1f2335;
        border-right: solid #3b4261;
        padding: 1;
    }

    #pipeline-panel {
        background: #1f2335;
        padding: 1;
    }

    #input-bar {
        column-span: 2;
        background: #24283b;
        padding: 0 1;
        border-top: solid #3b4261;
    }

    #pipeline-title {
        text-style: bold;
        color: #7aa2f7;
        margin-bottom: 1;
    }

    #strategy-label, #status-label {
        margin-bottom: 1;
    }

    ChatMessage {
        margin: 1 0;
        padding: 1;
        background: #2f3347;
        border-left: solid #7aa2f7;
    }

    ChatMessage.user {
        border-left: solid #bb9af7;
    }

    ChatMessage.assistant {
        border-left: solid #7aa2f7;
    }

    #shortcuts-content {
        background: #1f2335;
        border: solid #3b4261;
        padding: 2;
        margin: 4;
    }

    #override-header {
        padding: 1;
        text-style: bold;
    }

    Input {
        background: #1f2335;
        border: none;
        color: #c0caf5;
    }

    Input:focus {
        border: none;
    }

    ListView {
        background: #1f2335;
        border: solid #3b4261;
    }

    ListItem {
        padding: 1;
    }

    ListItem:hover {
        background: #2f3347;
    }
    """

    BINDINGS = [
        Binding("ctrl+r", "review_plan", "Review Plan"),
        Binding("ctrl+m", "cycle_strategy", "Cycle Strategy"),
        Binding("ctrl+p", "pick_profile", "Profile"),
        Binding("ctrl+b", "run_benchmark", "Benchmark"),
        Binding("ctrl+s", "save_session", "Save Session"),
        Binding("ctrl+o", "load_session", "Load Session"),
        Binding("?", "show_shortcuts", "Shortcuts"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.config = Config()
        self.strategy = ExecutionStrategy.model_aware
        self.current_plan: AnalyzerPlan | None = None
        self.current_profile: str = "quality"
        self.chat_history: list[dict[str, str]] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-layout"):
            with Vertical(id="chat-panel"):
                yield Static("[bold]Chat History[/]", id="chat-title")
                yield RichLog(highlight=True, markup=True, id="chat-log")
            yield PipelinePanel()
        yield Input(placeholder="Type a message and press Enter...", id="input-bar")

    def on_mount(self) -> None:
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write(
            "[dim]PolyMind TUI ready. Type a message or press ? for help.[/]"
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return

        input_widget = self.query_one(Input)
        input_widget.clear()

        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write(RichText(f"You: {prompt}", style="bold #bb9af7"))
        self.chat_history.append({"role": "user", "content": prompt})

        pipeline = self.query_one(PipelinePanel)
        pipeline.set_status("Analyzing...")
        self.run_pipeline(prompt)

    @work(exclusive=True)
    async def run_pipeline(self, prompt: str) -> None:
        pipeline = self.query_one(PipelinePanel)
        chat_log = self.query_one("#chat-log", RichLog)

        try:
            pipeline.set_status("Analyzing prompt...")
            plan = await analyze_prompt(prompt, router_model=self.config.router_model)
            self.current_plan = plan

            pipeline.log_line(f"[cyan]Plan:[/] {len(plan.subtasks)} subtask(s)")
            for s in plan.subtasks:
                deps = f" (depends: {', '.join(s.depends_on)})" if s.depends_on else ""
                pipeline.log_line(
                    f"  {s.id} [yellow]{s.domain.value}[/]: {s.prompt}{deps}"
                )

            ranks_path = Path("~/.polymind/ranks.yaml").expanduser()
            rank_store = load_ranks(ranks_path)
            schedule = build_schedule(
                plan, rank_store=rank_store, strategy=self.strategy
            )
            pipeline.log_line(f"[cyan]Schedule:[/] {self.strategy.value}")
            for batch in schedule.batches:
                pipeline.log_line(f"  {batch.model}: {', '.join(batch.subtask_ids)}")

            results: list[SubtaskResult] = []
            prior_outputs: dict[str, SubtaskResult] = {}

            pipeline.set_status("Executing subtasks...")
            for batch_idx, batch in enumerate(schedule.batches):
                pipeline.log_line(
                    f"[dim]Batch {batch_idx + 1}/{len(schedule.batches)}[/]"
                )
                for sid in batch.subtask_ids:
                    subtask = next(s for s in plan.subtasks if s.id == sid)
                    info = resolve_model_string(batch.model)
                    result = await execute_subtask(
                        subtask, model_ref=batch.model, provider_info=info
                    )
                    results.append(result)
                    prior_outputs[sid] = result
                    status = "[red]ERR[/]" if result.error else "[green]OK[/]"
                    pipeline.log_line(
                        f"  {result.subtask_id} on {result.model}: {status}"
                    )

            pipeline.set_status("Synthesizing...")
            pipeline_log = PipelineResult(
                original_prompt=prompt,
                subtask_results=results,
                schedule=schedule,
            )
            final = await synthesize(
                pipeline_log, synthesizer_model=self.config.synthesizer_model
            )

            output = final.synthesis or "[Synthesis failed]"
            chat_log.write(RichText(f"PolyMind: {output}", style="#7aa2f7"))
            self.chat_history.append({"role": "assistant", "content": output})
            pipeline.set_status("Done")

        except Exception as e:
            pipeline.log_line(f"[red]Pipeline error:[/] {e}")
            pipeline.set_status("Error")
            chat_log.write(RichText(f"[Error: {e}]", style="red"))

    def action_review_plan(self) -> None:
        if self.current_plan is None:
            return
        self.push_screen(SubtaskOverrideScreen(self.current_plan), self._on_review)

    def _on_review(self, proceed: bool | None) -> None:
        if proceed is False:
            pipeline = self.query_one(PipelinePanel)
            pipeline.log_line("[yellow]Execution cancelled by user[/]")

    def action_cycle_strategy(self) -> None:
        strategies = list(ExecutionStrategy)
        idx = strategies.index(self.strategy)
        self.strategy = strategies[(idx + 1) % len(strategies)]
        pipeline = self.query_one(PipelinePanel)
        pipeline.set_strategy(self.strategy)
        pipeline.log_line(f"[cyan]Strategy changed to:[/] {self.strategy.value}")

    def action_pick_profile(self) -> None:
        self.push_screen(ProfilePicker(), self._on_profile_picked)

    def _on_profile_picked(self, profile: str | None) -> None:
        if profile:
            self.current_profile = profile
            pipeline = self.query_one(PipelinePanel)
            pipeline.log_line(f"[cyan]Profile set to:[/] {profile}")

    @work(exclusive=True)
    async def action_run_benchmark(self) -> None:
        pipeline = self.query_one(PipelinePanel)
        pipeline.set_status("Benchmarking...")
        pipeline.log_line("[cyan]Starting benchmark...[/]")

        models = [f"ollama/{m.name}" for m in self.config.models]
        if not models:
            pipeline.log_line("[red]No models configured[/]")
            pipeline.set_status("Error")
            return

        store = await run_benchmark(models=models, judge_model=self.config.judge_model)
        ranks_path = Path("~/.polymind/ranks.yaml").expanduser()
        save_ranks(store, ranks_path)
        pipeline.log_line(f"[green]Benchmark done: {len(store.entries)} entries[/]")
        pipeline.set_status("Done")

    def action_save_session(self) -> None:
        data = {
            "timestamp": datetime.now().isoformat(),
            "history": self.chat_history,
            "strategy": self.strategy.value,
            "profile": self.current_profile,
        }
        sess_dir = Path("~/.polymind").expanduser()
        sess_dir.mkdir(parents=True, exist_ok=True)
        sess_path = (
            sess_dir / f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        )
        sess_path.write_text(json.dumps(data, indent=2))
        pipeline = self.query_one(PipelinePanel)
        pipeline.log_line(f"[green]Session saved:[/] {sess_path}")

    def action_load_session(self) -> None:
        sessions = sorted(Path("~/.polymind").expanduser().glob("session-*.json"))
        if not sessions:
            pipeline = self.query_one(PipelinePanel)
            pipeline.log_line("[yellow]No saved sessions found[/]")
            return
        latest = sessions[-1]
        data = json.loads(latest.read_text())
        self.chat_history = data.get("history", [])
        self.strategy = ExecutionStrategy(data.get("strategy", "model_aware"))
        self.current_profile = data.get("profile", "quality")

        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.clear()
        for msg in self.chat_history:
            style = "bold #bb9af7" if msg["role"] == "user" else "#7aa2f7"
            chat_log.write(
                RichText(f"{msg['role'].title()}: {msg['content']}", style=style)
            )

        pipeline = self.query_one(PipelinePanel)
        pipeline.set_strategy(self.strategy)
        pipeline.log_line(f"[green]Session loaded:[/] {latest}")

    def action_show_shortcuts(self) -> None:
        self.push_screen(ShortcutsScreen())


def main() -> None:
    app = PolyMindApp()
    app.run()


if __name__ == "__main__":
    main()
