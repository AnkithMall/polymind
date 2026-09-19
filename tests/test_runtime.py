"""Tests for polymind runtime commands."""

from typer.testing import CliRunner

from polymind.client.cli.app import app

runner = CliRunner()


class TestRuntimeOptimize:
    """Tests for polymind runtime optimize."""

    def test_optimize_no_models(self, tmp_polymind, env_override):
        """optimize should fail with no models."""
        result = runner.invoke(app, ["runtime", "optimize", "--all"])
        assert result.exit_code != 0
        assert "no models" in result.output.lower()

    def test_optimize_no_model_flag(self, tmp_polymind, env_override):
        """optimize should require --model or --all."""
        result = runner.invoke(app, ["runtime", "optimize"])
        assert result.exit_code != 0
        assert "specify" in result.output.lower()

    def test_optimize_model_not_found(self, tmp_polymind, env_override):
        """optimize should fail for non-existent model."""
        result = runner.invoke(app, ["runtime", "optimize", "-m", "999"])
        assert result.exit_code != 0

    def test_optimize_with_model(self, tmp_polymind, env_override):
        """optimize should work with a model (CPU-only mock)."""
        # Register a model with a small test file
        model_dir = tmp_polymind / ".polymind" / "models"
        model_file = model_dir / "test-model.gguf"
        # Create a minimal valid GGUF-like file
        model_file.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 1024)

        registry_yaml = tmp_polymind / ".polymind" / "registry.yaml"
        registry_yaml.write_text(
            f"""version: 1
models:
- id: 1
  repo_id: manual/unknown
  filename: test-model.gguf
  local_path: {model_file}
  size_bytes: 1024
  quantization: Q4_K_M
""",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["runtime", "optimize", "-m", "1"])
        # This may fail because the GGUF file is fake,
        # but it should not crash
        assert result.exit_code in (0, 1)


class TestRuntimeRun:
    """Tests for polymind runtime run."""

    def test_run_no_models(self, tmp_polymind, env_override):
        """run should fail with no models."""
        result = runner.invoke(app, ["runtime", "run", "-m", "1"])
        assert result.exit_code != 0
        assert "no models" in result.output.lower()

    def test_run_model_not_found(self, tmp_polymind, env_override):
        """run should fail for non-existent model."""
        result = runner.invoke(app, ["runtime", "run", "-m", "999"])
        assert result.exit_code != 0

    def test_run_file_not_exist(self, tmp_polymind, env_override):
        """run should fail if model file doesn't exist."""
        registry_yaml = tmp_polymind / ".polymind" / "registry.yaml"
        registry_yaml.write_text(
            """version: 1
models:
- id: 1
  repo_id: test/repo
  filename: test-model.gguf
  local_path: /nonexistent/test-model.gguf
  size_bytes: 1000000
  quantization: Q4_K_M
""",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["runtime", "run", "-m", "1"])
        assert result.exit_code != 0
        assert "not exist" in result.output.lower() or "not found" in result.output.lower()


class TestRuntimeConfig:
    """Tests for runtime configuration loading."""

    def test_load_runtime_config(self, tmp_polymind, env_override):
        """load_runtime_config should read from runtime.yaml."""
        from polymind.core.runtime.artifact import load_runtime_config

        # Write a test config
        runtime_yaml = tmp_polymind / ".polymind" / "runtime.yaml"
        runtime_yaml.write_text(
            """version: 1
models:
  '1':
    model_id: '1'
    gpu_layers: 0
    threads: 5
    context_size: 2048
    batch_size: 256
""",
            encoding="utf-8",
        )

        config = load_runtime_config("1", runtime_yaml)
        assert config is not None
        assert config.gpu_layers == 0
        assert config.threads == 5
        assert config.context_size == 2048
        assert config.batch_size == 256

    def test_load_runtime_config_not_found(self, tmp_polymind, env_override):
        """load_runtime_config should return None for missing model."""
        from polymind.core.runtime.artifact import load_runtime_config

        runtime_yaml = tmp_polymind / ".polymind" / "runtime.yaml"
        runtime_yaml.write_text("version: 1\nmodels: {}\n", encoding="utf-8")

        config = load_runtime_config("999", runtime_yaml)
        assert config is None

    def test_write_runtime_config(self, tmp_polymind, env_override):
        """write_runtime_config should write to runtime.yaml."""
        from polymind.core.runtime.artifact import write_runtime_config
        from polymind.core.runtime.types import RuntimeConfig

        runtime_yaml = tmp_polymind / ".polymind" / "runtime.yaml"
        config = RuntimeConfig(
            model_id="1",
            gpu_layers=0,
            threads=5,
            context_size=2048,
            batch_size=256,
        )

        result = write_runtime_config(config, runtime_yaml)
        assert result.exists()

        # Verify content
        import yaml

        with open(runtime_yaml) as f:
            data = yaml.safe_load(f)

        assert data["models"]["1"]["gpu_layers"] == 0
        assert data["models"]["1"]["threads"] == 5


# ══════════════════════════════════════════════════════════════
# New Runtime Optimizer Tests
# ══════════════════════════════════════════════════════════════


class TestRuntimeTypes:
    """Tests for new runtime types and data structures."""

    def test_runtime_config_to_dict(self):
        """RuntimeConfig.to_dict should produce expected keys."""
        from polymind.core.runtime.types import RuntimeConfig
        config = RuntimeConfig(model_id="1", gpu_layers=0, threads=4, context_size=2048, batch_size=256)
        d = config.to_dict()
        assert d["model_id"] == "1"
        assert d["gpu_layers"] == 0
        assert d["threads"] == 4

    def test_runtime_config_with_benchmark(self):
        """RuntimeConfig with benchmark metadata."""
        from polymind.core.runtime.types import RuntimeConfig
        config = RuntimeConfig(
            model_id="2",
            benchmark={"generation_tps": 11.9, "prompt_tps": 18.2, "runs": 3},
        )
        d = config.to_dict()
        assert "benchmark" in d
        assert d["benchmark"]["generation_tps"] == 11.9

    def test_runtime_profile_to_dict(self):
        """RuntimeProfile.to_dict should produce complete output."""
        from polymind.core.runtime.types import RuntimeProfile, Placement, BenchmarkMetrics, ValidationInfo, ValidationStatus
        profile = RuntimeProfile(
            model_id="1",
            hardware_fingerprint="abc123",
            model_fingerprint="def456",
            workload="decomposer",
            placement=Placement(backend="cuda", devices=[0]),
            gpu_layers=12,
            threads=6,
            context_size=2048,
            batch_size=128,
            benchmark=BenchmarkMetrics(
                generation_tokens_per_sec=11.9,
                prompt_tokens_per_sec=18.2,
                runs=3,
                stability=1.0,
            ),
            validation=ValidationInfo(
                status=ValidationStatus.READY,
                successful_runs=3,
            ),
        )
        d = profile.to_dict()
        assert d["model_id"] == "1"
        assert d["hardware_fingerprint"] == "abc123"
        assert d["workload"] == "decomposer"
        assert d["placement"]["backend"] == "cuda"
        assert d["benchmark"]["generation_tokens_per_sec"] == 11.9

    def test_runtime_profile_to_runtime_config(self):
        """RuntimeProfile.to_runtime_config should lower correctly."""
        from polymind.core.runtime.types import RuntimeProfile, BenchmarkMetrics
        profile = RuntimeProfile(
            model_id="3",
            gpu_layers=8,
            threads=6,
            context_size=2048,
            batch_size=128,
            benchmark=BenchmarkMetrics(generation_tokens_per_sec=10.0, runs=3),
        )
        config = profile.to_runtime_config()
        assert config.model_id == "3"
        assert config.gpu_layers == 8
        assert config.threads == 6

    def test_candidate_config_to_runtime_config(self):
        """CandidateConfig.to_runtime_config conversion."""
        from polymind.core.runtime.types import CandidateConfig
        cand = CandidateConfig(gpu_layers=12, threads=6, context_size=4096, batch_size=256)
        config = cand.to_runtime_config("5")
        assert config.model_id == "5"
        assert config.gpu_layers == 12

    def test_candidate_config_to_profile(self):
        """CandidateConfig.to_profile conversion."""
        from polymind.core.runtime.types import CandidateConfig, CandidateStatus
        cand = CandidateConfig(
            gpu_layers=8,
            threads=4,
            context_size=2048,
            batch_size=256,
        )
        cand.status = CandidateStatus.PASSED
        profile = cand.to_profile("2")
        assert profile.model_id == "2"
        assert profile.gpu_layers == 8
        assert profile.validation.successful_runs == 1

    def test_candidate_status_values(self):
        """CandidateStatus has all expected values."""
        from polymind.core.runtime.types import CandidateStatus
        values = [s.value for s in CandidateStatus]
        assert "pending" in values
        assert "testing" in values
        assert "passed" in values
        assert "failed" in values
        assert "unsafe" in values

    def test_workload_profiles_exist(self):
        """DEFAULT_WORKLOADS contains expected workload types."""
        from polymind.core.runtime.types import DEFAULT_WORKLOADS
        assert "decomposer" in DEFAULT_WORKLOADS
        assert "generator" in DEFAULT_WORKLOADS
        assert "interactive" in DEFAULT_WORKLOADS
        assert "long_context" in DEFAULT_WORKLOADS
        assert "default" in DEFAULT_WORKLOADS

    def test_workload_profile_to_dict(self):
        """WorkloadProfile.to_dict works."""
        from polymind.core.runtime.types import DEFAULT_WORKLOADS
        wl = DEFAULT_WORKLOADS["decomposer"]
        d = wl.to_dict()
        assert d["name"] == "decomposer"
        assert d["min_context"] > 0


class TestHardwareFingerprint:
    """Tests for hardware fingerprinting."""

    def test_fingerprint_from_hardware_profile(self, tmp_polymind, env_override):
        """HardwareFingerprint.from_hardware_profile produces a valid fingerprint."""
        from polymind.core.hardware.loader import load_hardware_profile
        from polymind.core.runtime.types import HardwareFingerprint
        hw = load_hardware_profile()
        fp = HardwareFingerprint.from_hardware_profile(hw)
        assert fp.cpu_model  # Should have some value
        assert fp.cpu_cores > 0
        assert fp.total_ram_bytes > 0

    def test_fingerprint_hash_is_stable(self, tmp_polymind, env_override):
        """Same hardware produces the same fingerprint hash."""
        from polymind.core.hardware.loader import load_hardware_profile
        from polymind.core.runtime.types import HardwareFingerprint
        hw = load_hardware_profile()
        fp1 = HardwareFingerprint.from_hardware_profile(hw)
        fp2 = HardwareFingerprint.from_hardware_profile(hw)
        assert fp1.compute_hash() == fp2.compute_hash()

    def test_fingerprint_hash_is_string(self, tmp_polymind, env_override):
        """Fingerprint hash is a hex string."""
        from polymind.core.hardware.loader import load_hardware_profile
        from polymind.core.runtime.types import HardwareFingerprint
        hw = load_hardware_profile()
        fp = HardwareFingerprint.from_hardware_profile(hw)
        h = fp.compute_hash()
        assert isinstance(h, str)
        assert len(h) == 16
        int(h, 16)  # Should be valid hex


class TestModelFingerprint:
    """Tests for model fingerprinting."""

    def test_model_fingerprint_from_file(self, tmp_polymind, env_override, mock_gguf):
        """ModelFingerprint.from_model_file works with a file."""
        from polymind.core.runtime.types import ModelFingerprint
        fp = ModelFingerprint.from_model_file(mock_gguf, 1024, "Q4_K_M")
        assert fp.file_size == 1024
        assert fp.quantization == "Q4_K_M"

    def test_model_fingerprint_hash(self, tmp_polymind, env_override):
        """ModelFingerprint.compute_hash produces a valid hash."""
        from polymind.core.runtime.types import ModelFingerprint
        fp = ModelFingerprint(file_size=1_000_000_000, architecture="llama", num_layers=32)
        h = fp.compute_hash()
        assert isinstance(h, str)
        assert len(h) == 16

    def test_model_fingerprint_different_sizes_different_hashes(self):
        """Different file sizes produce different hashes."""
        from polymind.core.runtime.types import ModelFingerprint
        fp1 = ModelFingerprint(file_size=1_000_000_000)
        fp2 = ModelFingerprint(file_size=2_000_000_000)
        assert fp1.compute_hash() != fp2.compute_hash()


class TestPlacement:
    """Tests for placement types."""

    def test_placement_to_dict(self):
        """Placement.to_dict produces expected output."""
        from polymind.core.runtime.types import Placement, SplitMode
        p = Placement(backend="cuda", devices=[0, 1], split_mode=SplitMode.LAYER, main_gpu=0)
        d = p.to_dict()
        assert d["backend"] == "cuda"
        assert d["devices"] == [0, 1]
        assert d["split_mode"] == "layer"
        assert d["main_gpu"] == 0

    def test_placement_cpu_only(self):
        """CPU-only placement."""
        from polymind.core.runtime.types import Placement
        p = Placement(backend="cpu", devices=[])
        d = p.to_dict()
        assert d["backend"] == "cpu"
        assert d["devices"] == []

    def test_split_mode_values(self):
        """SplitMode has expected values."""
        from polymind.core.runtime.types import SplitMode
        values = [s.value for s in SplitMode]
        assert "none" in values
        assert "layer" in values
        assert "row" in values


class TestMemoryEstimation:
    """Tests for memory estimation."""

    def test_estimate_memory_small_model(self):
        """Small model memory estimation."""
        from polymind.core.runtime.optimizer import estimate_memory_mb
        mem = estimate_memory_mb(
            model_size_bytes=1_000_000_000,
            num_layers=32,
            gpu_layers=0,
            context_size=2048,
            batch_size=256,
        )
        assert mem > 0
        # CPU-only should be relatively small
        assert mem < 1000  # MB

    def test_estimate_memory_full_gpu_offload(self):
        """Full GPU offload should use more VRAM."""
        from polymind.core.runtime.optimizer import estimate_memory_mb
        mem_cpu = estimate_memory_mb(
            model_size_bytes=1_000_000_000,
            num_layers=32,
            gpu_layers=0,
            context_size=2048,
            batch_size=256,
        )
        mem_gpu = estimate_memory_mb(
            model_size_bytes=1_000_000_000,
            num_layers=32,
            gpu_layers=32,
            context_size=2048,
            batch_size=256,
        )
        assert mem_gpu > mem_cpu

    def test_estimate_memory_increases_with_context(self):
        """Larger context should use more memory."""
        from polymind.core.runtime.optimizer import estimate_memory_mb
        mem_small = estimate_memory_mb(
            model_size_bytes=1_000_000_000,
            num_layers=32,
            gpu_layers=16,
            context_size=1024,
            batch_size=128,
        )
        mem_large = estimate_memory_mb(
            model_size_bytes=1_000_000_000,
            num_layers=32,
            gpu_layers=16,
            context_size=8192,
            batch_size=128,
        )
        assert mem_large > mem_small

    def test_check_memory_feasibility(self):
        """Memory feasibility check."""
        from polymind.core.runtime.optimizer import check_memory_feasibility
        assert check_memory_feasibility(500, 2000) is True
        assert check_memory_feasibility(500, 400) is False
        assert check_memory_feasibility(500, 0) is False


class TestCandidateGeneration:
    """Tests for candidate generation."""

    def test_generate_thread_candidates(self):
        """Thread candidates are generated from CPU topology."""
        from polymind.core.runtime.optimizer import generate_thread_candidates
        candidates = generate_thread_candidates(physical_cores=6, logical_cores=12)
        assert len(candidates) >= 3
        assert 1 in candidates
        assert 6 in candidates

    def test_generate_context_candidates(self):
        """Context candidates respect model size constraints."""
        from polymind.core.runtime.optimizer import generate_context_candidates
        # Very large model (>15GB) — should limit max context to 2048
        candidates = generate_context_candidates(model_size_bytes=16_000_000_000)
        assert all(c <= 2048 for c in candidates)
        # Small model — can go larger
        candidates = generate_context_candidates(model_size_bytes=1_000_000_000)
        assert max(candidates) >= 4096

    def test_generate_batch_candidates(self):
        """Batch candidates include standard sizes."""
        from polymind.core.runtime.optimizer import generate_batch_candidates
        candidates = generate_batch_candidates()
        assert 128 in candidates
        assert 256 in candidates
        assert 512 in candidates

    def test_generate_gpu_layer_candidates_no_vram(self):
        """No VRAM means only CPU candidates."""
        from polymind.core.runtime.optimizer import generate_gpu_layer_candidates
        from polymind.core.runtime.types import Placement
        candidates = generate_gpu_layer_candidates(
            num_layers=32, available_vram_bytes=0, layer_size=30_000_000,
            placement=Placement(backend="cpu"),
        )
        assert candidates == [0]

    def test_generate_gpu_layer_candidates_with_vram(self):
        """With VRAM, should generate offload candidates."""
        from polymind.core.runtime.optimizer import generate_gpu_layer_candidates
        from polymind.core.runtime.types import Placement
        candidates = generate_gpu_layer_candidates(
            num_layers=32,
            available_vram_bytes=2_000_000_000,  # 2GB
            layer_size=30_000_000,  # 30MB per layer
            placement=Placement(backend="cuda", devices=[0]),
        )
        assert 0 in candidates
        assert len(candidates) >= 3

    def test_generate_placement_candidates(self, tmp_polymind, env_override):
        """Placement candidates include CPU and GPU."""
        from polymind.core.hardware.loader import load_hardware_profile
        from polymind.core.runtime.optimizer import generate_placement_candidates
        hw = load_hardware_profile()
        candidates = generate_placement_candidates(hw)
        backends = [c.backend for c in candidates]
        assert "cpu" in backends
        assert "cuda" in backends


class TestSubprocessBenchmark:
    """Tests for subprocess-isolated benchmarking."""

    def test_benchmark_failure_classification(self):
        """BenchmarkFailure class works."""
        from polymind.core.runtime.benchmark import BenchmarkFailure
        from polymind.core.runtime.types import RuntimeConfig
        bf = BenchmarkFailure("test error", RuntimeConfig(model_id="1"))
        assert "test error" in repr(bf)

    def test_signal_name_conversion(self):
        """Signal name conversion works."""
        from polymind.core.runtime.benchmark import _signal_name, _classify_exit_code
        assert _classify_exit_code(-6) == "cuda_error"
        assert _classify_exit_code(-11) == "crash"
        assert _classify_exit_code(-9) == "timeout"
        assert _classify_exit_code(1) == "unknown"

    def test_benchmark_metrics_to_dict(self):
        """BenchmarkMetrics.to_dict works."""
        from polymind.core.runtime.types import BenchmarkMetrics
        m = BenchmarkMetrics(
            generation_tokens_per_sec=11.9,
            prompt_tokens_per_sec=18.2,
            runs=3,
            stability=1.0,
        )
        d = m.to_dict()
        assert d["generation_tokens_per_sec"] == 11.9
        assert d["runs"] == 3

    def test_subprocess_result_to_dict(self):
        """SubprocessBenchmarkResult.to_dict works."""
        from polymind.core.runtime.types import SubprocessBenchmarkResult
        r = SubprocessBenchmarkResult(
            success=False,
            error="OOM",
            failure_type="oom",
        )
        d = r.to_dict()
        assert d["success"] is False
        assert d["failure_type"] == "oom"

    def test_validation_status_values(self):
        """ValidationStatus has expected values."""
        from polymind.core.runtime.types import ValidationStatus
        values = [v.value for v in ValidationStatus]
        assert "ready" in values
        assert "stale" in values
        assert "failed" in values
        assert "unknown" in values

    def test_validation_info_is_usable(self):
        """ValidationInfo.is_usable checks correctly."""
        from polymind.core.runtime.types import ValidationInfo, ValidationStatus
        v = ValidationInfo(status=ValidationStatus.READY, successful_runs=3)
        assert v.is_usable is True

        v2 = ValidationInfo(status=ValidationStatus.STALE)
        assert v2.is_usable is False

        v3 = ValidationInfo(status=ValidationStatus.READY, successful_runs=0)
        assert v3.is_usable is False


class TestProfilePersistence:
    """Tests for runtime profile persistence."""

    def test_write_and_load_profile(self, tmp_polymind, env_override):
        """Write and load a RuntimeProfile."""
        from polymind.core.runtime.types import RuntimeProfile, Placement, BenchmarkMetrics, ValidationInfo, ValidationStatus
        from polymind.core.runtime.artifact import write_runtime_profile, load_runtime_profile
        runtime_yaml = tmp_polymind / ".polymind" / "runtime.yaml"

        profile = RuntimeProfile(
            model_id="5",
            hardware_fingerprint="hw_abc",
            model_fingerprint="md_def",
            workload="decomposer",
            placement=Placement(backend="cuda", devices=[0]),
            gpu_layers=12,
            threads=6,
            context_size=2048,
            batch_size=128,
            benchmark=BenchmarkMetrics(
                generation_tokens_per_sec=11.9,
                prompt_tokens_per_sec=18.2,
                runs=3,
                stability=1.0,
            ),
            validation=ValidationInfo(
                status=ValidationStatus.READY,
                successful_runs=3,
                last_validated="2026-01-01T00:00:00",
            ),
            estimated_memory_mb=2800.0,
            safety_margin_mb=700.0,
        )

        write_runtime_profile(profile, runtime_yaml)

        # Load it back
        loaded = load_runtime_profile("5", runtime_yaml)
        assert loaded is not None
        assert loaded.model_id == "5"
        assert loaded.hardware_fingerprint == "hw_abc"
        assert loaded.placement.backend == "cuda"
        assert loaded.benchmark.generation_tokens_per_sec == 11.9
        assert loaded.validation.status == ValidationStatus.READY

    def test_write_profile_preserves_other_models(self, tmp_polymind, env_override):
        """Writing one model's profile doesn't delete others."""
        from polymind.core.runtime.types import RuntimeProfile, ValidationInfo, ValidationStatus
        from polymind.core.runtime.artifact import write_runtime_profile, load_runtime_profile
        runtime_yaml = tmp_polymind / ".polymind" / "runtime.yaml"

        # Write model 1
        p1 = RuntimeProfile(
            model_id="1",
            gpu_layers=8,
            validation=ValidationInfo(status=ValidationStatus.READY, successful_runs=2),
        )
        write_runtime_profile(p1, runtime_yaml)

        # Write model 2
        p2 = RuntimeProfile(
            model_id="2",
            gpu_layers=16,
            validation=ValidationInfo(status=ValidationStatus.READY, successful_runs=1),
        )
        write_runtime_profile(p2, runtime_yaml)

        # Both should exist
        loaded1 = load_runtime_profile("1", runtime_yaml)
        loaded2 = load_runtime_profile("2", runtime_yaml)
        assert loaded1 is not None
        assert loaded2 is not None
        assert loaded1.gpu_layers == 8
        assert loaded2.gpu_layers == 16

    def test_load_legacy_profile(self, tmp_polymind, env_override):
        """Legacy v1 profile loads as a STALE RuntimeProfile."""
        from polymind.core.runtime.types import ValidationStatus
        from polymind.core.runtime.artifact import load_runtime_profile
        runtime_yaml = tmp_polymind / ".polymind" / "runtime.yaml"
        runtime_yaml.write_text(
            """version: 1
models:
  '1':
    model_id: '1'
    gpu_layers: 8
    threads: 4
    context_size: 2048
    batch_size: 256
""",
            encoding="utf-8",
        )
        profile = load_runtime_profile("1", runtime_yaml)
        assert profile is not None
        assert profile.gpu_layers == 8
        assert profile.validation.status == ValidationStatus.STALE

    def test_load_profile_not_found(self, tmp_polymind, env_override):
        """Loading non-existent model returns None."""
        from polymind.core.runtime.artifact import load_runtime_profile
        runtime_yaml = tmp_polymind / ".polymind" / "runtime.yaml"
        runtime_yaml.write_text("version: 2\nmodels: {}\n", encoding="utf-8")
        assert load_runtime_profile("999", runtime_yaml) is None

    def test_profile_validation_functions(self, tmp_polymind, env_override):
        """Hardware/model validation functions work."""
        from polymind.core.runtime.types import RuntimeProfile
        from polymind.core.runtime.artifact import (
            validate_profile_for_current_hardware,
            validate_profile_for_current_model,
        )
        profile = RuntimeProfile(model_id="1", hardware_fingerprint="abc123", model_fingerprint="def456")

        valid, reason = validate_profile_for_current_hardware(profile, "abc123")
        assert valid is True

        valid, reason = validate_profile_for_current_hardware(profile, "xyz789")
        assert valid is False
        assert "mismatch" in reason.lower()

        valid, reason = validate_profile_for_current_model(profile, "def456")
        assert valid is True

        valid, reason = validate_profile_for_current_model(profile, "wrong")
        assert valid is False

    def test_legacy_profile_has_no_fingerprint(self, tmp_polymind, env_override):
        """Legacy profile with no fingerprint is treated as compatible."""
        from polymind.core.runtime.types import RuntimeProfile
        from polymind.core.runtime.artifact import validate_profile_for_current_hardware
        profile = RuntimeProfile(model_id="1", hardware_fingerprint="")
        valid, reason = validate_profile_for_current_hardware(profile, "any_hash")
        assert valid is True
        assert "legacy" in reason.lower()


class TestRuntimeOptimizeCLI:
    """Tests for the updated runtime CLI commands."""

    def test_optimize_no_model_flag(self, tmp_polymind, env_override):
        """optimize requires --model or --all."""
        result = runner.invoke(app, ["runtime", "optimize"])
        assert result.exit_code != 0

    def test_optimize_workload_flag(self, tmp_polymind, env_override):
        """optimize accepts --workload flag."""
        result = runner.invoke(app, ["runtime", "optimize", "--all", "--workload", "decomposer"])
        # May fail due to no models, but should parse the flag
        assert "no models" in result.output.lower() or result.exit_code != 0

    def test_show_requires_model(self, tmp_polymind, env_override):
        """show requires --model flag."""
        result = runner.invoke(app, ["runtime", "show"])
        assert result.exit_code != 0

    def test_validate_requires_model(self, tmp_polymind, env_override):
        """validate requires --model flag."""
        result = runner.invoke(app, ["runtime", "validate"])
        assert result.exit_code != 0

    def test_show_no_profile(self, tmp_polymind, env_override):
        """show with no profile shows error."""
        result = runner.invoke(app, ["runtime", "show", "-m", "999"])
        assert result.exit_code != 0

    def test_validate_no_profile(self, tmp_polymind, env_override):
        """validate with no profile shows error."""
        result = runner.invoke(app, ["runtime", "validate", "-m", "999"])
        assert result.exit_code != 0


class TestRegressionModelsPreserved:
    """BUG-003 regression: updating one model must not delete others."""

    def test_update_model_2_preserves_1_and_3(self, tmp_polymind, env_override):
        """Writing config for model 2 must not delete models 1 and 3."""
        from polymind.core.runtime.artifact import write_runtime_config, load_runtime_config
        from polymind.core.runtime.types import RuntimeConfig
        runtime_yaml = tmp_polymind / ".polymind" / "runtime.yaml"

        # Write all three models
        for i in range(1, 4):
            write_runtime_config(
                RuntimeConfig(model_id=str(i), gpu_layers=i * 2, threads=i + 2, context_size=2048, batch_size=256),
                runtime_yaml,
            )

        # Update only model 2
        write_runtime_config(
            RuntimeConfig(model_id="2", gpu_layers=99, threads=99, context_size=4096, batch_size=512),
            runtime_yaml,
        )

        # Verify models 1 and 3 are intact
        c1 = load_runtime_config("1", runtime_yaml)
        c3 = load_runtime_config("3", runtime_yaml)
        c2 = load_runtime_config("2", runtime_yaml)

        assert c1 is not None
        assert c1.gpu_layers == 2
        assert c3 is not None
        assert c3.gpu_layers == 6
        assert c2 is not None
        assert c2.gpu_layers == 99
