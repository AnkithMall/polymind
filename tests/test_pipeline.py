"""Tests for the pipeline system."""

from typer.testing import CliRunner

from polymind.client.cli.app import app

runner = CliRunner()


class TestPipelineTypes:
    """Tests for pipeline type definitions."""

    def test_task_creation(self):
        """Task should be creatable with required fields."""
        from polymind.core.pipeline.types import ModelRole, Task, TaskStatus, TaskType

        task = Task(
            id="task_1",
            prompt="What is 2+2?",
            domain="mathematics",
        )
        assert task.id == "task_1"
        assert task.status == TaskStatus.PENDING
        assert task.task_type == TaskType.GENERATE
        assert task.model_role == ModelRole.GENERATOR

    def test_task_serialization(self):
        """Task.to_dict should produce valid dict."""
        from polymind.core.pipeline.types import Task

        task = Task(id="task_1", prompt="test", domain="coding")
        d = task.to_dict()
        assert d["id"] == "task_1"
        assert d["domain"] == "coding"
        assert d["status"] == "pending"

    def test_pipeline_config_defaults(self):
        """PipelineConfig should have sensible defaults."""
        from polymind.core.pipeline.types import PipelineConfig

        config = PipelineConfig()
        assert config.auto_select_models is True
        assert config.max_concurrent == 1
        assert config.max_retries == 1


class TestScheduler:
    """Tests for the task scheduler."""

    def test_schedule_linear_tasks(self):
        """Scheduler should group consecutive same-model tasks."""
        from polymind.core.pipeline.scheduler import schedule_tasks
        from polymind.core.pipeline.types import Task

        tasks = [
            Task(id="t1", prompt="a", domain="math"),
            Task(id="t2", prompt="b", domain="math"),
            Task(id="t3", prompt="c", domain="coding"),
        ]
        assignments = {"t1": "model_a", "t2": "model_a", "t3": "model_b"}

        plan = schedule_tasks(tasks, assignments)

        assert plan.total_tasks == 3
        assert plan.estimated_model_loads == 2  # model_a once, model_b once
        assert len(plan.groups) == 2
        assert len(plan.groups[0].tasks) == 2  # t1, t2 share model_a
        assert len(plan.groups[1].tasks) == 1  # t3 uses model_b

    def test_schedule_with_dependencies(self):
        """Scheduler should respect dependency ordering."""
        from polymind.core.pipeline.scheduler import schedule_tasks
        from polymind.core.pipeline.types import Task

        tasks = [
            Task(id="t1", prompt="a", domain="math", depends_on=[]),
            Task(id="t2", prompt="b", domain="math", depends_on=["t1"]),
        ]
        assignments = {"t1": "model_a", "t2": "model_a"}

        plan = schedule_tasks(tasks, assignments)

        assert len(plan.execution_order) >= 2
        # t1 must come before t2
        t1_wave = next(i for i, w in enumerate(plan.execution_order) if "t1" in w)
        t2_wave = next(i for i, w in enumerate(plan.execution_order) if "t2" in w)
        assert t1_wave < t2_wave

    def test_get_ready_tasks(self):
        """get_ready_tasks should return tasks with all deps met."""
        from polymind.core.pipeline.scheduler import get_ready_tasks
        from polymind.core.pipeline.types import Task

        tasks = [
            Task(id="t1", prompt="a", domain="math", depends_on=[]),
            Task(id="t2", prompt="b", domain="math", depends_on=["t1"]),
            Task(id="t3", prompt="c", domain="math", depends_on=["t1", "t2"]),
        ]

        # Initially t1 and t2? No — t2 depends on t1 which is not completed
        # But t1 has no deps so it's always ready
        ready = get_ready_tasks(tasks, set())
        ready_ids = {t.id for t in ready}
        assert "t1" in ready_ids
        assert "t2" not in ready_ids  # t1 not completed yet
        assert "t3" not in ready_ids  # t1, t2 not completed yet

        # After t1 completes, t2 becomes ready (t1 in completed_ids)
        ready = get_ready_tasks(tasks, {"t1"})
        ready_ids = {t.id for t in ready}
        assert "t2" in ready_ids
        assert "t3" not in ready_ids  # t2 not completed yet

        # After t1 and t2 complete, t3 is ready
        ready = get_ready_tasks(tasks, {"t1", "t2"})
        ready_ids = {t.id for t in ready}
        assert "t3" in ready_ids


class TestDecomposer:
    """Tests for the prompt decomposer."""

    def test_detect_domain_coding(self):
        """_detect_domain should detect coding prompts."""
        from polymind.core.pipeline.decomposer import _detect_domain

        assert _detect_domain("Write a Python function to sort a list") == "coding"
        assert _detect_domain("Debug this code") == "coding"

    def test_detect_domain_math(self):
        """_detect_domain should detect math prompts."""
        from polymind.core.pipeline.decomposer import _detect_domain

        assert _detect_domain("Calculate the square root of 144") == "mathematics"
        assert _detect_domain("Solve this equation: 2x + 3 = 7") == "mathematics"

    def test_detect_domain_general(self):
        """_detect_domain should default to general."""
        from polymind.core.pipeline.decomposer import _detect_domain

        assert _detect_domain("The quick brown fox jumps over the lazy dog") == "general"

    def test_parse_tasks_valid(self):
        """_parse_tasks should parse decomposer output correctly."""
        from polymind.core.pipeline.decomposer import _parse_tasks

        content = """TASK: Write a sorting function
DOMAIN: coding
DEPENDS: NONE
TASK: Test the function
DOMAIN: coding
DEPENDS: 1"""

        tasks = _parse_tasks(content, "Sort and test a list")
        assert len(tasks) == 2
        assert tasks[0].domain == "coding"
        assert tasks[1].domain == "coding"
        assert len(tasks[1].depends_on) == 1


class TestRegenerator:
    """Tests for the response regenerator."""

    def test_quick_synthesize_single(self):
        """quick_synthesize with one task returns its result."""
        from polymind.core.pipeline.regenerator import quick_synthesize
        from polymind.core.pipeline.types import Task, TaskStatus

        task = Task(id="t1", prompt="test", domain="math")
        task.status = TaskStatus.COMPLETED
        task.result = "The answer is 42."

        result = quick_synthesize([task])
        assert result == "The answer is 42."

    def test_quick_synthesize_empty(self):
        """quick_synthesize with no results returns error message."""
        from polymind.core.pipeline.regenerator import quick_synthesize

        result = quick_synthesize([])
        assert "No results" in result

    def test_quick_synthesize_multiple(self):
        """quick_synthesize with multiple tasks combines results."""
        from polymind.core.pipeline.regenerator import quick_synthesize
        from polymind.core.pipeline.types import Task, TaskStatus

        tasks = []
        for i, domain in enumerate(["math", "coding"]):
            t = Task(id=f"t{i}", prompt="test", domain=domain)
            t.status = TaskStatus.COMPLETED
            t.result = f"Result from {domain}"
            tasks.append(t)

        result = quick_synthesize(tasks)
        assert "math" in result
        assert "coding" in result


class TestSelector:
    """Tests for the model selector."""

    def test_select_model_no_models(self):
        """select_model should handle no models gracefully."""
        from pathlib import Path

        from polymind.core.model.registry import ModelRegistry
        from polymind.core.pipeline.selector import select_model_for_task
        from polymind.core.pipeline.types import PipelineConfig, Task

        # Create a registry pointing to a nonexistent path
        registry = ModelRegistry(path=Path("/nonexistent/registry.yaml"))

        task = Task(id="t1", prompt="test", domain="math")
        config = PipelineConfig()
        assignment = select_model_for_task(task, config, registry)

        assert assignment.model is None
        assert "no models" in assignment.reason

    def test_suggest_models_empty(self):
        """suggest_models should handle empty registry."""
        from pathlib import Path

        from polymind.core.model.registry import ModelRegistry
        from polymind.core.pipeline.selector import suggest_models

        registry = ModelRegistry(path=Path("/nonexistent/registry.yaml"))

        suggestions = suggest_models(registry)
        assert "decomposer" in suggestions
        assert "generator" in suggestions
        assert suggestions["decomposer"] == []


class TestPipelineCLI:
    """Tests for pipeline CLI commands."""

    def test_pipeline_help(self):
        """pipeline --help should work."""
        result = runner.invoke(app, ["pipeline", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output.lower() or "pipeline" in result.output.lower()

    def test_pipeline_suggest(self):
        """pipeline suggest should show model suggestions."""
        result = runner.invoke(app, ["pipeline", "suggest"])
        # Should not crash even with no models
        assert result.exit_code == 0

    def test_pipeline_status(self):
        """pipeline status should show installed models."""
        result = runner.invoke(app, ["pipeline", "status"])
        assert result.exit_code == 0


class TestComplexityAnalysis:
    """Tests for the prompt complexity analyzer."""

    def test_simple_math(self):
        """Simple math should be detected as simple."""
        from polymind.core.pipeline.orchestrator import Pipeline

        p = Pipeline()
        assert p._is_simple("2+2") is True
        assert p._is_simple("What is 15 * 3?") is True
        assert p._is_simple("100 / 4") is True

    def test_simple_greeting(self):
        """Greetings should be simple."""
        from polymind.core.pipeline.orchestrator import Pipeline

        p = Pipeline()
        assert p._is_simple("hi") is True
        assert p._is_simple("hello") is True
        assert p._is_simple("thanks") is True

    def test_simple_question(self):
        """Short questions should be simple."""
        from polymind.core.pipeline.orchestrator import Pipeline

        p = Pipeline()
        assert p._is_simple("What is Python?") is True
        assert p._is_simple("Define recursion") is True

    def test_complex_prompt(self):
        """Multi-step prompts should NOT be simple."""
        from polymind.core.pipeline.orchestrator import Pipeline

        p = Pipeline()
        assert p._is_simple("Build a REST API with auth and tests") is False
        assert p._is_simple("Step 1: Write the code. Step 2: Test it.") is False
        assert p._is_simple("First, analyze the data, then create a visualization") is False

    def test_long_prompt(self):
        """Very long prompts should NOT be simple."""
        from polymind.core.pipeline.orchestrator import Pipeline

        p = Pipeline()
        long = " ".join(["word"] * 150)
        assert p._is_simple(long) is False

    def test_medium_prompt_with_connectors(self):
        """Medium prompts with connectors should NOT be simple."""
        from polymind.core.pipeline.orchestrator import Pipeline

        p = Pipeline()
        assert p._is_simple("Write a function and test it") is False


class TestPipelineLogging:
    """Tests for pipeline logging system."""

    def test_get_logger(self, tmp_polymind):
        """get_pipeline_logger should return a working logger."""
        from polymind.core.pipeline.log import get_pipeline_logger

        logger = get_pipeline_logger("test_log")
        assert logger is not None
        logger.info("Test message")

    def test_log_writer(self, tmp_polymind):
        """PipelineLogWriter should write events without crashing."""
        from polymind.core.pipeline.log import PipelineLogWriter
        from polymind.core.pipeline.types import Task, TaskType

        writer = PipelineLogWriter()

        # Simulate events
        writer.log_event(
            "artifact_check",
            Task(id="a", prompt="", domain="", task_type=TaskType.CUSTOM,
                 metadata={"registry.yaml": "1 model", "hardware.yaml": "missing"}),
        )
        writer.log_event(
            "analyze",
            Task(id="a", prompt="hi", domain="", task_type=TaskType.CUSTOM,
                 metadata={"is_simple": "true", "word_count": "1"}),
        )
        writer.log_event(
            "task_start",
            Task(id="t1", prompt="test prompt", domain="math", task_type=TaskType.GENERATE),
        )

    def test_cleanup_respects_config(self, tmp_polymind):
        """_cleanup_old_logs should respect max_age_days."""
        from polymind.core.pipeline.log import _cleanup_old_logs, _logs_dir

        logs_dir = _logs_dir()
        old_log = logs_dir / "pipeline_2020.log"
        old_log.write_text("old log")

        # Make it appear old
        import os
        import time

        os.utime(str(old_log), (time.time() - 86400 * 30, time.time() - 86400 * 30))

        _cleanup_old_logs({"max_age_days": 7, "max_total_files": 100, "max_total_size_mb": 100})
        assert not old_log.exists()


class TestPolymindConfig:
    """Tests for polymind.yaml configuration."""

    def test_load_config_defaults(self, tmp_polymind):
        """load_config should return defaults when no file exists."""
        from polymind.core.config import load_config

        cfg = load_config()
        assert "logging" in cfg
        assert cfg["logging"]["enabled"] is True
        assert cfg["logging"]["max_age_days"] == 7

    def test_save_and_load_config(self, tmp_polymind):
        """save_config should write and load_config should read."""
        from polymind.core.config import load_config, save_config

        cfg = load_config()
        cfg["logging"]["max_age_days"] = 30
        save_config(cfg)

        loaded = load_config()
        assert loaded["logging"]["max_age_days"] == 30

    def test_update_logging_config(self, tmp_polymind):
        """update_logging_config should update specific keys."""
        from polymind.core.config import load_config, update_logging_config

        update_logging_config(max_age_days=14, enabled=False)

        cfg = load_config()
        assert cfg["logging"]["max_age_days"] == 14
        assert cfg["logging"]["enabled"] is False
        # Other defaults should persist
        assert cfg["logging"]["backup_count"] == 3

    def test_logging_cli(self, tmp_polymind):
        """config logging command should work."""
        result = runner.invoke(app, ["config", "logging"])
        assert result.exit_code == 0
        assert "Pipeline Logging Configuration" in result.output

    def test_logging_set_cli(self, tmp_polymind):
        """config logging-set command should update values."""
        result = runner.invoke(app, ["config", "logging-set", "max_age_days", "14"])
        assert result.exit_code == 0
        assert "logging.max_age_days = 14" in result.output
