"""Comprehensive tests for demo milestone implementation.

Covers:
  - BUG-003 regression
  - Domain type extensions (aliases)
  - Category command
  - Suite command improvements
  - Dynamic domain resolution
  - Capability profiles
  - Doctor diagnostics
  - Demo seed script
  - Pipeline selector domain resolution
  - Decomposer dynamic domains
"""

from pathlib import Path

import yaml
from typer.testing import CliRunner

from polymind.client.cli.app import app

runner = CliRunner()


# ═══════════════════════════════════════════════════════════════
# BUG-003 REGRESSION TESTS
# ═══════════════════════════════════════════════════════════════


class TestBug003Regression:
    """BUG-003: config set runtime.2.gpu_layers causes other models to disappear."""

    def test_write_preserves_all_models(self, tmp_polymind, env_override):
        """Writing config for model 2 must not remove models 1, 3, 4."""
        from polymind.core.runtime.artifact import write_runtime_config
        from polymind.core.runtime.types import RuntimeConfig

        runtime_yaml = tmp_polymind / ".polymind" / "runtime.yaml"

        # Create configs for models 1, 2, 3, 4
        for mid in ["1", "2", "3", "4"]:
            config = RuntimeConfig(
                model_id=mid,
                gpu_layers=16 if mid != "1" else 24,
                threads=4,
                context_size=2048,
                batch_size=128,
            )
            write_runtime_config(config, runtime_yaml)

        # Verify all 4 models exist
        with runtime_yaml.open() as f:
            data = yaml.safe_load(f)
        assert len(data["models"]) == 4
        for mid in ["1", "2", "3", "4"]:
            assert mid in data["models"]

        # Now update model 2
        config2 = RuntimeConfig(model_id="2", gpu_layers=32, threads=8, context_size=4096, batch_size=256)
        write_runtime_config(config2, runtime_yaml)

        # Verify all 4 models still exist
        with runtime_yaml.open() as f:
            data = yaml.safe_load(f)
        assert len(data["models"]) == 4
        for mid in ["1", "2", "3", "4"]:
            assert mid in data["models"]

        # Verify model 2 was updated
        assert data["models"]["2"]["gpu_layers"] == 32
        assert data["models"]["2"]["threads"] == 8

        # Verify other models unchanged
        assert data["models"]["1"]["gpu_layers"] == 24
        assert data["models"]["3"]["gpu_layers"] == 16
        assert data["models"]["4"]["gpu_layers"] == 16

    def test_write_with_integer_keys(self, tmp_polymind, env_override):
        """YAML may deserialize numeric keys as integers; write must handle this."""
        from polymind.core.runtime.artifact import write_runtime_config
        from polymind.core.runtime.types import RuntimeConfig

        runtime_yaml = tmp_polymind / ".polymind" / "runtime.yaml"

        # Write a config that will produce integer keys after YAML round-trip
        config1 = RuntimeConfig(model_id="1", gpu_layers=24)
        write_runtime_config(config1, runtime_yaml)

        # Simulate YAML round-trip with integer keys
        with runtime_yaml.open() as f:
            data = yaml.safe_load(f)

        # Manually set integer keys to simulate the bug scenario
        int_data = {1: data["models"]["1"]}
        data["models"] = int_data
        with runtime_yaml.open("w") as f:
            yaml.safe_dump(data, f)

        # Now write a new model - should not lose model 1
        config2 = RuntimeConfig(model_id="2", gpu_layers=16)
        write_runtime_config(config2, runtime_yaml)

        with runtime_yaml.open() as f:
            data = yaml.safe_load(f)
        assert len(data["models"]) == 2
        assert "1" in data["models"]
        assert "2" in data["models"]

    def test_load_with_integer_keys(self, tmp_polymind, env_override):
        """load_runtime_config should handle integer YAML keys."""
        from polymind.core.runtime.artifact import load_runtime_config

        runtime_yaml = tmp_polymind / ".polymind" / "runtime.yaml"

        # Write with integer keys
        data = {
            "version": 1,
            "models": {
                1: {"model_id": "1", "gpu_layers": 24, "threads": 4, "context_size": 2048, "batch_size": 128},
            },
        }
        with runtime_yaml.open("w") as f:
            yaml.safe_load
            yaml.safe_dump(data, f)

        config = load_runtime_config("1", runtime_yaml)
        assert config is not None
        assert config.gpu_layers == 24

    def test_config_set_preserves_other_models(self, tmp_polymind, env_override):
        """CLI config set should preserve all other models."""
        # Write initial configs for 3 models
        runtime_yaml = tmp_polymind / ".polymind" / "runtime.yaml"
        data = {
            "version": 1,
            "models": {
                "1": {"model_id": "1", "gpu_layers": 24, "threads": 4, "context_size": 2048, "batch_size": 128},
                "2": {"model_id": "2", "gpu_layers": 16, "threads": 4, "context_size": 2048, "batch_size": 256},
                "3": {"model_id": "3", "gpu_layers": 8, "threads": 2, "context_size": 1024, "batch_size": 128},
            },
        }
        with runtime_yaml.open("w") as f:
            yaml.safe_dump(data, f)

        # Update model 2 via CLI
        result = runner.invoke(app, ["config", "set", "runtime.2.gpu_layers", "32"])
        assert result.exit_code == 0

        # Verify all 3 models still exist
        with runtime_yaml.open() as f:
            data = yaml.safe_load(f)
        assert len(data["models"]) == 3
        assert data["models"]["1"]["gpu_layers"] == 24
        assert data["models"]["2"]["gpu_layers"] == 32
        assert data["models"]["3"]["gpu_layers"] == 8


# ═══════════════════════════════════════════════════════════════
# DOMAIN TYPE EXTENSIONS
# ═══════════════════════════════════════════════════════════════


class TestDomainAliases:
    """Tests for domain aliases field."""

    def test_predefined_domains_have_aliases(self):
        """All predefined domains should have aliases."""
        from polymind.core.confidence.suites import get_all_domains

        domains = get_all_domains()
        for d in domains:
            assert isinstance(d.aliases, list)
            assert len(d.aliases) > 0, f"Domain {d.id} has no aliases"

    def test_domain_to_dict_includes_aliases(self):
        """Domain.to_dict should include aliases when present."""
        from polymind.core.confidence.types import Domain

        d = Domain(id="test", name="Test", description="Test", aliases=["a", "b"])
        data = d.to_dict()
        assert "aliases" in data
        assert data["aliases"] == ["a", "b"]

    def test_domain_to_dict_excludes_empty_aliases(self):
        """Domain.to_dict should not include empty aliases."""
        from polymind.core.confidence.types import Domain

        d = Domain(id="test", name="Test", description="Test", aliases=[])
        data = d.to_dict()
        assert "aliases" not in data

    def test_custom_domain_with_aliases_persists(self):
        """Custom domain with aliases should persist correctly."""
        from polymind.core.confidence.artifact import save_custom_domain, load_domain_by_id, delete_custom_domain
        from polymind.core.confidence.types import Domain

        domain = Domain(
            id="test_aliases",
            name="Test Aliases",
            description="Testing aliases",
            custom=True,
            aliases=["ta", "testing", "test domain"],
        )
        save_custom_domain(domain)

        loaded = load_domain_by_id("test_aliases")
        assert loaded is not None
        assert loaded.aliases == ["ta", "testing", "test domain"]

        # Cleanup
        delete_custom_domain("test_aliases")


# ═══════════════════════════════════════════════════════════════
# DYNAMIC DOMAIN RESOLUTION
# ═══════════════════════════════════════════════════════════════


class TestDomainResolution:
    """Tests for resolve_domain_id and dynamic domain loading."""

    def test_resolve_by_id(self):
        """Should resolve by exact domain ID."""
        from polymind.core.confidence.suites import resolve_domain_id, get_all_domains

        result = resolve_domain_id("mathematics", get_all_domains())
        assert result == "mathematics"

    def test_resolve_by_name(self):
        """Should resolve by domain name."""
        from polymind.core.confidence.suites import resolve_domain_id, get_all_domains

        result = resolve_domain_id("Mathematics", get_all_domains())
        assert result == "mathematics"

    def test_resolve_by_alias(self):
        """Should resolve by alias."""
        from polymind.core.confidence.suites import resolve_domain_id, get_all_domains

        result = resolve_domain_id("math", get_all_domains())
        assert result == "mathematics"

    def test_resolve_case_insensitive(self):
        """Should be case insensitive."""
        from polymind.core.confidence.suites import resolve_domain_id, get_all_domains

        assert resolve_domain_id("MATH", get_all_domains()) == "mathematics"
        assert resolve_domain_id("Code", get_all_domains()) == "coding"

    def test_resolve_unknown_returns_none(self):
        """Unknown domain should return None."""
        from polymind.core.confidence.suites import resolve_domain_id, get_all_domains

        result = resolve_domain_id("nonexistent_domain_xyz", get_all_domains())
        assert result is None

    def test_resolve_custom_alias(self, tmp_polymind, env_override):
        """Should resolve custom domain aliases."""
        from polymind.core.confidence.artifact import save_custom_domain, delete_custom_domain
        from polymind.core.confidence.suites import resolve_domain_id
        from polymind.core.confidence.types import Domain

        domain = Domain(
            id="frontend",
            name="Frontend",
            description="Frontend dev",
            custom=True,
            aliases=["ui", "react", "typescript"],
        )
        save_custom_domain(domain)

        from polymind.core.confidence.artifact import load_all_domains

        result = resolve_domain_id("react", load_all_domains())
        assert result == "frontend"

        delete_custom_domain("frontend")


# ═══════════════════════════════════════════════════════════════
# CATEGORY COMMAND TESTS
# ═══════════════════════════════════════════════════════════════


class TestCategoryCommand:
    """Tests for the category CLI commands."""

    def test_category_list(self, tmp_polymind, env_override):
        """category list should show all categories."""
        result = runner.invoke(app, ["category", "list"])
        assert result.exit_code == 0
        assert "mathematics" in result.output.lower()

    def test_category_show_predefined(self, tmp_polymind, env_override):
        """category show should display built-in category details."""
        result = runner.invoke(app, ["category", "show", "coding"])
        assert result.exit_code == 0
        assert "coding" in result.output.lower()

    def test_category_add_and_delete(self, tmp_polymind, env_override):
        """category add and delete should work for custom categories."""
        result = runner.invoke(
            app,
            ["category", "add", "test_cat", "--name", "Test Category",
             "--description", "A test", "--aliases", "test,testing"],
        )
        assert result.exit_code == 0
        assert "Created" in result.output

        # Verify it shows up
        result = runner.invoke(app, ["category", "show", "test_cat"])
        assert result.exit_code == 0

        # Delete it
        result = runner.invoke(app, ["category", "delete", "test_cat", "--force"])
        assert result.exit_code == 0
        assert "Deleted" in result.output

    def test_category_cannot_delete_predefined(self, tmp_polymind, env_override):
        """category delete should not work on predefined categories."""
        result = runner.invoke(app, ["category", "delete", "mathematics", "--force"])
        assert result.exit_code != 0
        assert "cannot delete predefined" in result.output.lower() or "protected" in result.output.lower()

    def test_category_edit(self, tmp_polymind, env_override):
        """category edit should update custom category."""
        # Create
        runner.invoke(
            app,
            ["category", "add", "edit_test", "--name", "Original"],
        )

        # Edit
        result = runner.invoke(
            app,
            ["category", "edit", "edit_test", "--name", "Updated", "--aliases", "new1,new2"],
        )
        assert result.exit_code == 0
        assert "Updated" in result.output

        # Verify
        result = runner.invoke(app, ["category", "show", "edit_test"])
        assert "Updated" in result.output

        # Cleanup
        runner.invoke(app, ["category", "delete", "edit_test", "--force"])

    def test_category_clone(self, tmp_polymind, env_override):
        """category clone should copy a category."""
        # Create source
        runner.invoke(
            app,
            ["category", "add", "clone_src", "--name", "Source"],
        )

        result = runner.invoke(app, ["category", "clone", "clone_src", "clone_dst"])
        assert result.exit_code == 0
        assert "Cloned" in result.output

        # Cleanup
        runner.invoke(app, ["category", "delete", "clone_src", "--force"])
        runner.invoke(app, ["category", "delete", "clone_dst", "--force"])

    def test_category_show_not_found(self, tmp_polymind, env_override):
        """category show should fail for non-existent category."""
        result = runner.invoke(app, ["category", "show", "nonexistent_xyz"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_category_help(self, tmp_polymind, env_override):
        """category --help should show usage."""
        result = runner.invoke(app, ["category", "--help"])
        assert result.exit_code == 0
        assert "category" in result.output.lower() or "manage" in result.output.lower()


# ═══════════════════════════════════════════════════════════════
# SUITE COMMAND TESTS
# ═══════════════════════════════════════════════════════════════


class TestSuiteEditCommands:
    """Tests for suite edit and question edit commands."""

    def test_edit_suite(self, tmp_polymind, env_override):
        """suite edit should update suite metadata."""
        # Create a custom domain with a suite
        from polymind.core.confidence.artifact import save_custom_domain, delete_custom_domain
        from polymind.core.confidence.types import Domain, TestSuite

        domain = Domain(
            id="edit_test_domain",
            name="Edit Test",
            description="Test",
            custom=True,
            suites=[
                TestSuite(
                    id="edit_suite_test",
                    name="Original Name",
                    description="Original desc",
                    difficulty="easy",
                    questions=[],
                )
            ],
        )
        save_custom_domain(domain)

        result = runner.invoke(
            app,
            ["suite", "edit", "edit_suite_test", "--name", "New Name", "--difficulty", "hard",
             "--domain", "edit_test_domain"],
        )
        assert result.exit_code == 0
        assert "Updated" in result.output

        # Cleanup
        delete_custom_domain("edit_test_domain")

    def test_edit_question(self, tmp_polymind, env_override):
        """suite edit-question should update question fields."""
        from polymind.core.confidence.artifact import save_custom_domain, delete_custom_domain
        from polymind.core.confidence.types import Domain, TestQuestion, TestSuite

        domain = Domain(
            id="eq_test_domain",
            name="EQ Test",
            description="Test",
            custom=True,
            suites=[
                TestSuite(
                    id="eq_suite",
                    name="EQ Suite",
                    description="Test",
                    difficulty="easy",
                    questions=[
                        TestQuestion(id="q1", prompt="Original prompt", expected="Original answer"),
                    ],
                )
            ],
        )
        save_custom_domain(domain)

        result = runner.invoke(
            app,
            ["suite", "edit-question", "eq_test_domain", "eq_suite", "q1",
             "--prompt", "New prompt", "--expected", "New answer"],
        )
        assert result.exit_code == 0
        assert "Updated" in result.output

        # Cleanup
        delete_custom_domain("eq_test_domain")


# ═══════════════════════════════════════════════════════════════
# CAPABILITY COMMAND TESTS
# ═══════════════════════════════════════════════════════════════


class TestCapabilityCommand:
    """Tests for capability show and domains commands."""

    def test_capability_show_no_data(self, tmp_polymind, env_override):
        """capability show should handle no data gracefully."""
        result = runner.invoke(app, ["capability", "show"])
        assert result.exit_code == 0
        assert "no capability data" in result.output.lower() or "no data" in result.output.lower()

    def test_capability_show_with_data(self, tmp_polymind, env_override):
        """capability show should display a matrix with scores."""
        from polymind.core.confidence.artifact import save_confidence
        from polymind.core.confidence.types import (
            DomainScore, ModelConfidence, SuiteScore,
        )

        scores = {
            "1": ModelConfidence(
                model_id="1",
                domains={
                    "math": DomainScore(
                        domain_id="math",
                        overall=85.0,
                        suites={
                            "basic": SuiteScore(suite_id="basic", score=90.0, passed=9, total=10)
                        },
                    ),
                    "coding": DomainScore(
                        domain_id="coding",
                        overall=72.0,
                        suites={
                            "basics": SuiteScore(suite_id="basics", score=72.0, passed=7, total=10)
                        },
                    ),
                },
                best_domain="math",
                overall_score=78.5,
            ),
        }
        save_confidence(scores)

        result = runner.invoke(app, ["capability", "show"])
        assert result.exit_code == 0
        assert "math" in result.output
        assert "coding" in result.output

    def test_capability_json_output(self, tmp_polymind, env_override):
        """capability show --json should output valid JSON."""
        import json
        from polymind.core.confidence.artifact import save_confidence
        from polymind.core.confidence.types import (
            DomainScore, ModelConfidence, SuiteScore,
        )

        scores = {
            "1": ModelConfidence(
                model_id="1",
                domains={
                    "math": DomainScore(domain_id="math", overall=85.0, suites={}),
                },
                overall_score=85.0,
            ),
        }
        save_confidence(scores)

        result = runner.invoke(app, ["capability", "show", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "1" in data
        assert data["1"]["overall"] == 85.0

    def test_capability_domains(self, tmp_polymind, env_override):
        """capability domains should list scored domains."""
        result = runner.invoke(app, ["capability", "domains"])
        assert result.exit_code == 0


# ═══════════════════════════════════════════════════════════════
# DOCTOR COMMAND TESTS
# ═══════════════════════════════════════════════════════════════


class TestDoctorCommand:
    """Tests for the doctor diagnostics command."""

    def test_doctor_runs(self, tmp_polymind, env_override):
        """doctor should run without crashing."""
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "Polymind Doctor" in result.output

    def test_doctor_shows_checks(self, tmp_polymind, env_override):
        """doctor should show individual check results."""
        result = runner.invoke(app, ["doctor"])
        assert "Artifact directory" in result.output
        assert "Model registry" in result.output
        assert "Runtime config" in result.output

    def test_doctor_check_alias(self, tmp_polymind, env_override):
        """doctor check should be an alias."""
        result = runner.invoke(app, ["doctor", "check"])
        assert result.exit_code == 0
        assert "Polymind Doctor" in result.output


# ═══════════════════════════════════════════════════════════════
# DEMO SEED TESTS
# ═══════════════════════════════════════════════════════════════


class TestDemoSeed:
    """Tests for the demo seed command."""

    def test_demo_seed_creates_domains(self, tmp_polymind, env_override):
        """demo seed should create all 5 demo domains."""
        result = runner.invoke(app, ["demo", "seed"])
        assert result.exit_code == 0
        assert "frontend" in result.output
        assert "backend" in result.output
        assert "system-design" in result.output
        assert "architecture" in result.output
        assert "cybersecurity" in result.output
        assert "5" in result.output  # 5 demo domains

    def test_demo_seed_persists(self, tmp_polymind, env_override):
        """demo seed domains should be loadable after creation."""
        runner.invoke(app, ["demo", "seed"])

        from polymind.core.confidence.artifact import load_all_domains

        domains = load_all_domains()
        custom_ids = {d.id for d in domains if d.custom}
        assert "frontend" in custom_ids
        assert "backend" in custom_ids
        assert "system-design" in custom_ids
        assert "architecture" in custom_ids
        assert "cybersecurity" in custom_ids

    def test_demo_seed_has_suites(self, tmp_polymind, env_override):
        """demo seed domains should have suites with questions."""
        runner.invoke(app, ["demo", "seed"])

        from polymind.core.confidence.artifact import load_domain_by_id

        for domain_id in ["frontend", "backend", "system-design", "architecture", "cybersecurity"]:
            domain = load_domain_by_id(domain_id)
            assert domain is not None
            assert len(domain.suites) == 3
            total_q = sum(len(s.questions) for s in domain.suites)
            assert total_q >= 8

    def test_demo_seed_reset(self, tmp_polymind, env_override):
        """demo seed --reset should replace existing custom domains."""
        runner.invoke(app, ["demo", "seed"])
        runner.invoke(app, ["demo", "seed", "--reset"])

        from polymind.core.confidence.artifact import load_all_domains

        domains = load_all_domains()
        custom = [d for d in domains if d.custom]
        assert len(custom) == 5  # Only the 5 demo domains

    def test_demo_status(self, tmp_polymind, env_override):
        """demo status should show domain status."""
        runner.invoke(app, ["demo", "seed"])
        result = runner.invoke(app, ["demo", "status"])
        assert result.exit_code == 0
        assert "frontend" in result.output

    def test_demo_help(self, tmp_polymind, env_override):
        """demo --help should work."""
        result = runner.invoke(app, ["demo", "--help"])
        assert result.exit_code == 0
        assert "seed" in result.output.lower()


# ═══════════════════════════════════════════════════════════════
# PIPELINE SELECTOR DOMAIN RESOLUTION
# ═══════════════════════════════════════════════════════════════


class TestSelectorDomainResolution:
    """Tests for model selector with domain resolution."""

    def test_selector_resolves_custom_domain(self, tmp_polymind, env_override):
        """Selector should resolve custom domain IDs for scoring."""
        from polymind.core.pipeline.selector import _auto_select
        from polymind.core.pipeline.types import ModelRole
        from polymind.core.model.registry import InstalledModel

        # Create a mock model
        model = InstalledModel(
            id=1, repo_id="test/repo", filename="test.gguf",
            local_path="/tmp/test.gguf", size_bytes=1024**3, quantization="Q4_K_M",
        )

        # This should not crash even with no confidence data
        result = _auto_select("frontend", ModelRole.GENERATOR, [model])
        # Without confidence data, it falls back to size-based
        assert result is not None
        assert result.model_id == "1"

    def test_selector_handles_unknown_domain(self, tmp_polymind, env_override):
        """Selector should handle unknown domains gracefully."""
        from polymind.core.pipeline.selector import _auto_select
        from polymind.core.pipeline.types import ModelRole
        from polymind.core.model.registry import InstalledModel

        model = InstalledModel(
            id=1, repo_id="test/repo", filename="test.gguf",
            local_path="/tmp/test.gguf", size_bytes=1024**3, quantization="Q4_K_M",
        )

        result = _auto_select("totally_unknown_domain_xyz", ModelRole.GENERATOR, [model])
        assert result is not None


# ═══════════════════════════════════════════════════════════════
# DECOMPOSER DYNAMIC DOMAINS
# ═══════════════════════════════════════════════════════════════


class TestDecomposerDynamicDomains:
    """Tests for decomposer dynamic domain loading."""

    def test_build_domain_list_includes_predefined(self):
        """_build_domain_list should include all predefined domains."""
        from polymind.core.pipeline.decomposer import _build_domain_list

        domain_list = _build_domain_list()
        assert "mathematics" in domain_list
        assert "coding" in domain_list
        assert "writing" in domain_list

    def test_build_domain_list_includes_custom(self, tmp_polymind, env_override):
        """_build_domain_list should include custom domains."""
        from polymind.core.confidence.artifact import save_custom_domain, delete_custom_domain
        from polymind.core.confidence.types import Domain

        domain = Domain(
            id="test_domain_list",
            name="Test",
            description="Test",
            custom=True,
            aliases=["td"],
        )
        save_custom_domain(domain)

        from polymind.core.pipeline.decomposer import _build_domain_list

        domain_list = _build_domain_list()
        assert "test_domain_list" in domain_list

        delete_custom_domain("test_domain_list")

    def test_detect_domain_uses_registry(self, tmp_polymind, env_override):
        """_detect_domain should match against registered domain aliases."""
        from polymind.core.confidence.artifact import save_custom_domain, delete_custom_domain
        from polymind.core.confidence.types import Domain
        from polymind.core.pipeline.decomposer import _detect_domain

        domain = Domain(
            id="cybersecurity",
            name="Cybersecurity",
            description="Security",
            custom=True,
            aliases=["security", "threat modeling"],
        )
        save_custom_domain(domain)

        # Should detect via alias
        assert _detect_domain("How do I improve security?") == "cybersecurity"
        assert _detect_domain("Perform threat modeling on this system") == "cybersecurity"

        delete_custom_domain("cybersecurity")

    def test_detect_domain_fallback_to_keywords(self):
        """_detect_domain should fall back to keyword heuristics."""
        from polymind.core.pipeline.decomposer import _detect_domain

        # "What is" matches knowledge keywords first
        assert _detect_domain("What is 2+2?") == "knowledge"
        # Direct math terms should match mathematics
        assert _detect_domain("Calculate the sum of 5 and 7") == "mathematics"
        assert _detect_domain("Write a Python function") == "coding"


# ═══════════════════════════════════════════════════════════════
# TUI COMMAND TESTS
# ═══════════════════════════════════════════════════════════════


class TestTuiCommand:
    """Tests for TUI stub command."""

    def test_tui_shows_unavailable(self, tmp_polymind, env_override):
        """tui should show unavailable message."""
        result = runner.invoke(app, ["tui"])
        assert result.exit_code == 0
        assert "not yet available" in result.output.lower()

    def test_tui_shows_alternatives(self, tmp_polymind, env_override):
        """tui should suggest CLI alternatives."""
        result = runner.invoke(app, ["tui"])
        assert "polymind" in result.output.lower()


# ═══════════════════════════════════════════════════════════════
# RUN COMMAND TESTS
# ═══════════════════════════════════════════════════════════════


class TestRunCommand:
    """Tests for run alias command."""

    def test_run_no_prompt_shows_usage(self, tmp_polymind, env_override):
        """run with no prompt should show usage."""
        result = runner.invoke(app, ["run"])
        # Should either show usage or fail gracefully
        assert result.exit_code in (0, 1)

    def test_run_help(self, tmp_polymind, env_override):
        """run --help should work."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "pipeline" in result.output.lower() or "prompt" in result.output.lower()


# ═══════════════════════════════════════════════════════════════
# EVIDENCE METADATA
# ═══════════════════════════════════════════════════════════════


class TestEvidenceMetadata:
    """Tests for evaluation evidence and metadata."""

    def test_domain_score_includes_suites(self):
        """DomainScore should contain suite-level data."""
        from polymind.core.confidence.types import DomainScore, SuiteScore

        ds = DomainScore(
            domain_id="test",
            overall=85.0,
            suites={
                "s1": SuiteScore(suite_id="s1", score=90.0, passed=9, total=10),
                "s2": SuiteScore(suite_id="s2", score=80.0, passed=8, total=10),
            },
        )
        assert len(ds.suites) == 2
        assert ds.suites["s1"].passed == 9

    def test_model_confidence_tracks_best_domain(self):
        """ModelConfidence should identify best domain."""
        from polymind.core.confidence.types import DomainScore, ModelConfidence

        mc = ModelConfidence(
            model_id="1",
            domains={
                "math": DomainScore(domain_id="math", overall=90.0),
                "coding": DomainScore(domain_id="coding", overall=70.0),
            },
            best_domain="math",
            overall_score=80.0,
        )
        assert mc.best_domain == "math"
        assert mc.overall_score == 80.0

    def test_confidence_persistence_roundtrip(self, tmp_polymind, env_override):
        """Confidence data should survive save/load."""
        from polymind.core.confidence.artifact import load_confidence, save_confidence
        from polymind.core.confidence.types import (
            DomainScore, ModelConfidence, SuiteScore,
        )

        scores = {
            "1": ModelConfidence(
                model_id="1",
                domains={
                    "frontend": DomainScore(
                        domain_id="frontend",
                        overall=88.5,
                        suites={
                            "fe-core": SuiteScore(suite_id="fe-core", score=92.0, passed=9, total=10),
                            "fe-arch": SuiteScore(suite_id="fe-arch", score=85.0, passed=8, total=10),
                        },
                    ),
                    "backend": DomainScore(
                        domain_id="backend",
                        overall=75.0,
                        suites={},
                    ),
                },
                best_domain="frontend",
                overall_score=81.75,
            ),
        }
        save_confidence(scores)
        loaded = load_confidence()

        assert "1" in loaded
        assert loaded["1"].overall_score == 81.75
        assert loaded["1"].best_domain == "frontend"
        assert "frontend" in loaded["1"].domains
        assert loaded["1"].domains["frontend"].overall == 88.5
        assert len(loaded["1"].domains["frontend"].suites) == 2


# ═══════════════════════════════════════════════════════════════
# CLI HELP VERIFICATION
# ═══════════════════════════════════════════════════════════════


class TestCLIHelp:
    """Verify all commands appear in help and don't crash."""

    def test_main_help(self):
        """Main help should show all commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "model" in result.output
        assert "pipeline" in result.output
        assert "confidence" in result.output
        assert "domain" in result.output
        assert "category" in result.output
        assert "capability" in result.output
        assert "doctor" in result.output
        assert "demo" in result.output

    def test_category_help(self):
        """category --help should show subcommands."""
        result = runner.invoke(app, ["category", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "show" in result.output
        assert "add" in result.output

    def test_suite_help(self):
        """suite --help should show subcommands."""
        result = runner.invoke(app, ["suite", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "show" in result.output
        assert "create" in result.output
        assert "edit" in result.output
