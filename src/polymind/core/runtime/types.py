"""Runtime types — configuration, profiles, fingerprints, and workload definitions.

Supports backward compatibility with legacy runtime.yaml files that contain
only: model_id, gpu_layers, threads, context_size, batch_size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


# ── Workload profiles ──────────────────────────────────────────

class WorkloadType(StrEnum):
    """Recognized workload types for profile-specific optimization."""
    DECOMPOSER = "decomposer"
    GENERATOR = "generator"
    REGENERATOR = "regenerator"
    JUDGE = "judge"
    LONG_CONTEXT = "long_context"
    INTERACTIVE = "interactive"
    DEFAULT = "default"


@dataclass
class WorkloadProfile:
    """Describes a workload for targeted benchmarking.

    Each workload defines a representative prompt, token ranges,
    and performance priorities that guide optimization decisions.
    """
    name: str
    description: str
    representative_prompt: str
    min_context: int = 2048
    target_output_tokens: int = 256
    latency_priority: float = 0.5   # 0.0 = throughput, 1.0 = latency
    throughput_priority: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "representative_prompt": self.representative_prompt[:200],
            "min_context": self.min_context,
            "target_output_tokens": self.target_output_tokens,
            "latency_priority": self.latency_priority,
            "throughput_priority": self.throughput_priority,
        }


# ── Placement ──────────────────────────────────────────────────

class SplitMode(StrEnum):
    NONE = "none"
    LAYER = "layer"
    ROW = "row"


@dataclass
class Placement:
    """Describes where and how a model executes."""
    backend: str = "cpu"
    devices: list[int] = field(default_factory=list)
    split_mode: SplitMode = SplitMode.NONE
    main_gpu: int | None = None
    tensor_split: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "backend": self.backend,
            "devices": self.devices,
            "split_mode": self.split_mode.value,
            "main_gpu": self.main_gpu,
        }
        if self.tensor_split is not None:
            d["tensor_split"] = self.tensor_split
        return d


# ── Benchmark metrics ──────────────────────────────────────────

@dataclass
class BenchmarkMetrics:
    """Measured performance metrics from benchmark runs."""
    load_time_ms: float = 0.0
    prompt_tokens_per_sec: float = 0.0
    generation_tokens_per_sec: float = 0.0
    time_to_first_token_ms: float = 0.0
    total_latency_ms: float = 0.0
    peak_vram_mb: float = 0.0
    peak_ram_mb: float = 0.0
    warmup_time_ms: float = 0.0
    stability: float = 0.0       # successful_runs / total_runs
    runs: int = 0
    benchmark_timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "load_time_ms": round(self.load_time_ms, 1),
            "prompt_tokens_per_sec": round(self.prompt_tokens_per_sec, 1),
            "generation_tokens_per_sec": round(self.generation_tokens_per_sec, 1),
            "time_to_first_token_ms": round(self.time_to_first_token_ms, 1),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "peak_ram_mb": round(self.peak_ram_mb, 1),
            "warmup_time_ms": round(self.warmup_time_ms, 1),
            "stability": round(self.stability, 2),
            "runs": self.runs,
            "benchmark_timestamp": self.benchmark_timestamp,
        }


# ── Validation status ─────────────────────────────────────────

class ValidationStatus(StrEnum):
    READY = "ready"
    STALE = "stale"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class ValidationInfo:
    """Tracks whether a profile has been validated and is still safe."""
    status: ValidationStatus = ValidationStatus.UNKNOWN
    successful_runs: int = 0
    failed_runs: int = 0
    last_validated: str = ""
    failure_reasons: list[str] = field(default_factory=list)
    runtime_version: str = ""

    @property
    def is_usable(self) -> bool:
        return self.status == ValidationStatus.READY and self.successful_runs > 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "status": self.status.value,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "last_validated": self.last_validated,
            "runtime_version": self.runtime_version,
        }
        if self.failure_reasons:
            d["failure_reasons"] = self.failure_reasons
        return d


# ── Runtime config (backward compatible) ───────────────────────

@dataclass
class RuntimeConfig:
    """Minimal runtime config — backward compatible with legacy runtime.yaml.

    This is the interface the orchestrator and runner consume. The optimizer
    produces a full RuntimeProfile which is then lowered to a RuntimeConfig
    for execution.
    """
    model_id: str

    gpu_layers: int = -1
    threads: int = 4
    context_size: int = 4096
    batch_size: int = 512

    # Benchmark metadata (populated by optimizer)
    benchmark: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "model_id": self.model_id,
            "gpu_layers": self.gpu_layers,
            "threads": self.threads,
            "context_size": self.context_size,
            "batch_size": self.batch_size,
        }
        if self.benchmark:
            data["benchmark"] = self.benchmark
        return data


# ── Full runtime profile ───────────────────────────────────────

@dataclass
class RuntimeProfile:
    """Complete runtime profile with fingerprints, placement, and validation.

    Stored in runtime.yaml. Backward compatible — legacy profiles are
    loaded as RuntimeConfig and upgraded when optimize runs.
    """
    model_id: str

    # Fingerprints (for staleness detection)
    hardware_fingerprint: str = ""
    model_fingerprint: str = ""
    workload: str = "default"

    # Placement
    placement: Placement = field(default_factory=Placement)

    # Execution parameters
    gpu_layers: int = -1
    threads: int = 4
    threads_batch: int = 4
    context_size: int = 4096
    batch_size: int = 512
    ubatch_size: int = 64
    flash_attention: bool | None = None
    kv_cache_type: str | None = None

    # Measured metrics
    benchmark: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)

    # Validation
    validation: ValidationInfo = field(default_factory=ValidationInfo)

    # Memory estimates
    estimated_memory_mb: float = 0.0
    actual_peak_memory_mb: float = 0.0
    safety_margin_mb: float = 0.0

    @property
    def is_usable(self) -> bool:
        """Check if this profile can be safely used right now."""
        return self.validation.is_usable

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "hardware_fingerprint": self.hardware_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "workload": self.workload,
            "placement": self.placement.to_dict(),
            "gpu_layers": self.gpu_layers,
            "threads": self.threads,
            "threads_batch": self.threads_batch,
            "context_size": self.context_size,
            "batch_size": self.batch_size,
            "ubatch_size": self.ubatch_size,
            "flash_attention": self.flash_attention,
            "kv_cache_type": self.kv_cache_type,
            "benchmark": self.benchmark.to_dict(),
            "validation": self.validation.to_dict(),
            "estimated_memory_mb": round(self.estimated_memory_mb, 1),
            "actual_peak_memory_mb": round(self.actual_peak_memory_mb, 1),
            "safety_margin_mb": round(self.safety_margin_mb, 1),
        }

    def to_runtime_config(self) -> RuntimeConfig:
        """Lower to a RuntimeConfig for execution."""
        return RuntimeConfig(
            model_id=self.model_id,
            gpu_layers=self.gpu_layers,
            threads=self.threads,
            context_size=self.context_size,
            batch_size=self.batch_size,
            benchmark=self.benchmark.to_dict() if self.benchmark.runs > 0 else {},
        )


# ── Candidate configuration for optimization ───────────────────

class CandidateStatus(StrEnum):
    PENDING = "pending"
    TESTING = "testing"
    PASSED = "passed"
    FAILED = "failed"
    UNSAFE = "unsafe"


@dataclass
class CandidateConfig:
    """A candidate configuration to benchmark during optimization."""
    # Execution
    gpu_layers: int = 0
    threads: int = 4
    threads_batch: int = 4
    context_size: int = 2048
    batch_size: int = 256
    ubatch_size: int = 64

    # Placement
    placement: Placement = field(default_factory=Placement)

    # Status tracking
    status: CandidateStatus = CandidateStatus.PENDING
    failure_reason: str = ""
    failure_type: str = ""  # oom, cuda_error, crash, timeout, unknown

    # Measured results
    metrics: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)

    # Memory
    estimated_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0

    def to_runtime_config(self, model_id: str) -> RuntimeConfig:
        """Convert to RuntimeConfig for execution."""
        return RuntimeConfig(
            model_id=model_id,
            gpu_layers=self.gpu_layers,
            threads=self.threads,
            context_size=self.context_size,
            batch_size=self.batch_size,
        )

    def to_profile(self, model_id: str) -> RuntimeProfile:
        """Convert to full RuntimeProfile."""
        return RuntimeProfile(
            model_id=model_id,
            placement=self.placement,
            gpu_layers=self.gpu_layers,
            threads=self.threads,
            threads_batch=self.threads_batch,
            context_size=self.context_size,
            batch_size=self.batch_size,
            ubatch_size=self.ubatch_size,
            benchmark=self.metrics,
            validation=ValidationInfo(
                status=ValidationStatus.READY if self.status == CandidateStatus.PASSED else ValidationStatus.UNKNOWN,
                successful_runs=1 if self.status == CandidateStatus.PASSED else 0,
                failed_runs=1 if self.status in (CandidateStatus.FAILED, CandidateStatus.UNSAFE) else 0,
            ),
            estimated_memory_mb=self.estimated_memory_mb,
            actual_peak_memory_mb=self.peak_memory_mb,
        )


# ── Benchmark subprocess result ────────────────────────────────

@dataclass
class SubprocessBenchmarkResult:
    """Result returned from an isolated benchmark subprocess."""
    success: bool
    load_time_ms: float = 0.0
    prompt_tokens_per_sec: float = 0.0
    generation_tokens_per_sec: float = 0.0
    time_to_first_token_ms: float = 0.0
    total_latency_ms: float = 0.0
    peak_vram_mb: float = 0.0
    peak_ram_mb: float = 0.0
    generated_tokens: int = 0
    prompt_tokens: int = 0
    error: str = ""
    failure_type: str = ""  # oom, cuda_error, crash, timeout, load_error, unknown

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "success": self.success,
            "load_time_ms": round(self.load_time_ms, 1),
            "prompt_tokens_per_sec": round(self.prompt_tokens_per_sec, 1),
            "generation_tokens_per_sec": round(self.generation_tokens_per_sec, 1),
            "time_to_first_token_ms": round(self.time_to_first_token_ms, 1),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "peak_ram_mb": round(self.peak_ram_mb, 1),
        }
        if not self.success:
            d["error"] = self.error
            d["failure_type"] = self.failure_type
        return d


# ── Hardware fingerprint ───────────────────────────────────────

@dataclass
class HardwareFingerprint:
    """Compact representation of hardware state for profile matching."""
    cpu_model: str = ""
    cpu_cores: int = 0
    total_ram_bytes: int = 0
    gpu_ids: list[int] = field(default_factory=list)
    gpu_models: list[str] = field(default_factory=list)
    gpu_total_vram: list[int] = field(default_factory=list)
    driver_version: str = ""
    llama_cpp_version: str = ""

    def compute_hash(self) -> str:
        """Compute a stable fingerprint hash."""
        import hashlib
        parts = [
            self.cpu_model,
            str(self.cpu_cores),
            str(self.total_ram_bytes),
            ",".join(str(i) for i in self.gpu_ids),
            ",".join(self.gpu_models),
            ",".join(str(v) for v in self.gpu_total_vram),
            self.driver_version,
            self.llama_cpp_version,
        ]
        canonical = "|".join(parts)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @classmethod
    def from_hardware_profile(cls, hw: Any) -> HardwareFingerprint:
        """Build fingerprint from a HardwareProfile."""
        gpu_ids = [g.id for g in hw.gpus]
        gpu_models = [g.model for g in hw.gpus]
        gpu_vram = [g.memory.total_bytes for g in hw.gpus]
        driver = ""
        for g in hw.gpus:
            if g.driver_version:
                driver = g.driver_version
                break

        fp = cls(
            cpu_model=hw.cpu.model,
            cpu_cores=hw.cpu.physical_cores,
            total_ram_bytes=hw.memory.total_bytes,
            gpu_ids=gpu_ids,
            gpu_models=gpu_models,
            gpu_total_vram=gpu_vram,
            driver_version=driver,
        )
        return fp


def _read_gguf_header_metadata(model_path: Any) -> dict[str, Any]:
    """Read GGUF metadata from file header without loading the model.

    This is much faster and lighter than instantiating a Llama object.
    Only reads the KV metadata section at the start of the file.
    """
    import struct

    def _skip_value(f: Any, val_type: int, count: int = 1) -> None:
        """Skip a GGUF value in the file."""
        for _ in range(count):
            if val_type == 0:  # UINT8
                f.read(1)
            elif val_type == 1:  # INT8
                f.read(1)
            elif val_type == 2:  # UINT16
                f.read(2)
            elif val_type == 3:  # INT16
                f.read(2)
            elif val_type == 4:  # UINT32
                f.read(4)
            elif val_type == 5:  # INT32
                f.read(4)
            elif val_type == 6:  # FLOAT32
                f.read(4)
            elif val_type == 7:  # BOOL
                f.read(1)
            elif val_type == 8:  # STRING
                str_len = struct.unpack("<Q", f.read(8))[0]
                f.read(str_len)
            elif val_type == 9:  # ARRAY
                arr_type = struct.unpack("<I", f.read(4))[0]
                arr_len = struct.unpack("<Q", f.read(8))[0]
                _skip_value(f, arr_type, arr_len)

    with open(str(model_path), "rb") as f:
        # Magic number
        magic = f.read(4)
        if magic != b"GGUF":
            return {}

        # Version
        version = struct.unpack("<I", f.read(4))[0]

        # Tensor count and metadata kv count
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        metadata: dict[str, Any] = {}
        for _ in range(n_kv):
            # Key
            key_len = struct.unpack("<Q", f.read(8))[0]
            key = f.read(key_len).decode("utf-8")

            # Value type
            val_type = struct.unpack("<I", f.read(4))[0]

            if val_type == 0:  # UINT8
                val = struct.unpack("<B", f.read(1))[0]
            elif val_type == 1:  # INT8
                val = struct.unpack("<b", f.read(1))[0]
            elif val_type == 2:  # UINT16
                val = struct.unpack("<H", f.read(2))[0]
            elif val_type == 3:  # INT16
                val = struct.unpack("<h", f.read(2))[0]
            elif val_type == 4:  # UINT32
                val = struct.unpack("<I", f.read(4))[0]
            elif val_type == 5:  # INT32
                val = struct.unpack("<i", f.read(4))[0]
            elif val_type == 6:  # FLOAT32
                val = struct.unpack("<f", f.read(4))[0]
            elif val_type == 7:  # BOOL
                val = struct.unpack("<B", f.read(1))[0] != 0
            elif val_type == 8:  # STRING
                str_len = struct.unpack("<Q", f.read(8))[0]
                val = f.read(str_len).decode("utf-8")
            elif val_type == 9:  # ARRAY
                # Skip array contents — read element type and count, then skip
                arr_type = struct.unpack("<I", f.read(4))[0]
                arr_len = struct.unpack("<Q", f.read(8))[0]
                _skip_value(f, arr_type, arr_len)
                continue  # Don't store, just move to next KV
            else:
                break

            metadata[key] = val

        return metadata


# ── Model fingerprint ──────────────────────────────────────────

@dataclass
class ModelFingerprint:
    """Compact representation of a model's characteristics."""
    file_size: int = 0
    architecture: str = ""
    num_layers: int = 0
    embedding_dim: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    quantization: str = ""
    is_moe: bool = False
    expert_count: int = 0
    active_experts: int = 0
    file_hash: str = ""  # partial hash for change detection

    def compute_hash(self) -> str:
        import hashlib
        parts = [
            str(self.file_size),
            self.architecture,
            str(self.num_layers),
            self.quantization,
            str(self.is_moe),
            str(self.expert_count),
            self.file_hash,
        ]
        canonical = "|".join(parts)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @classmethod
    def from_model_file(cls, model_path: Any, file_size: int, quantization: str = "") -> ModelFingerprint:
        """Build fingerprint by reading GGUF metadata if possible."""
        fp = cls(
            file_size=file_size,
            quantization=quantization,
        )

        # Try to read GGUF metadata directly from file header (no model loading needed)
        try:
            meta = _read_gguf_header_metadata(model_path)
            if meta:
                fp.architecture = str(meta.get("general.architecture", ""))
                arch = fp.architecture
                if arch:
                    fp.num_layers = int(meta.get(f"{arch}.block_count", 0) or 0)
                    fp.embedding_dim = int(meta.get(f"{arch}.embedding_length", 0) or 0)
                    fp.num_heads = int(meta.get(f"{arch}.attention.head_count", 0) or 0)
                    fp.num_kv_heads = int(meta.get(f"{arch}.attention.head_count_kv", 0) or 0)

                # MoE detection
                expert_count = int(meta.get(f"{arch}.expert_count", 0) or 0)
                if expert_count > 0:
                    fp.is_moe = True
                    fp.expert_count = expert_count
                    fp.active_experts = int(meta.get(f"{arch}.expert_used_count", 0) or 0)
        except Exception:
            pass  # Fallback: use file size only

        return fp


# ── Default workload profiles ──────────────────────────────────

DEFAULT_WORKLOADS: dict[str, WorkloadProfile] = {
    "decomposer": WorkloadProfile(
        name="decomposer",
        description="Realistic PolyMind decomposer workload",
        representative_prompt=(
            "You are a task decomposer. Given a user request, break it into "
            "subtasks with domain assignments.\n\n"
            "## Available Domains\n- coding: Software development tasks\n"
            "- writing: Content creation tasks\n- reasoning: Analytical tasks\n\n"
            "## User Request\nBuild a REST API with authentication and tests.\n\n"
            "## Output Format\nReturn JSON array of tasks."
        ),
        min_context=2048,
        target_output_tokens=256,
        latency_priority=0.7,
        throughput_priority=0.3,
    ),
    "generator": WorkloadProfile(
        name="generator",
        description="Code/content generation workload",
        representative_prompt=(
            "Write a Python function that implements a binary search algorithm. "
            "Include type hints and a docstring."
        ),
        min_context=2048,
        target_output_tokens=512,
        latency_priority=0.4,
        throughput_priority=0.6,
    ),
    "interactive": WorkloadProfile(
        name="interactive",
        description="Quick interactive response",
        representative_prompt="What is the time complexity of quicksort?",
        min_context=1024,
        target_output_tokens=128,
        latency_priority=0.9,
        throughput_priority=0.1,
    ),
    "long_context": WorkloadProfile(
        name="long_context",
        description="Long context analysis workload",
        representative_prompt="Analyze the following code and identify all bugs:\n\n" + ("def f(x): return x * 2\n" * 200),
        min_context=4096,
        target_output_tokens=512,
        latency_priority=0.3,
        throughput_priority=0.7,
    ),
    "default": WorkloadProfile(
        name="default",
        description="General purpose workload",
        representative_prompt="Explain what an operating system does in exactly 3 sentences.",
        min_context=2048,
        target_output_tokens=256,
        latency_priority=0.5,
        throughput_priority=0.5,
    ),
}
