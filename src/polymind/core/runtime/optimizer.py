"""Adaptive, hardware-aware runtime configuration optimizer.

Design:
    MODEL + HARDWARE + CURRENT STATE + WORKLOAD + REQUIREMENTS
    → CANDIDATE GENERATION
    → FEASIBILITY FILTER
    → ISOLATED BENCHMARK
    → MEASURED PERFORMANCE
    → SAFETY / MEMORY VALIDATION
    → PROFILE SELECTION
    → PERSISTED RUNTIME PROFILE

Safety rules:
    1. Never mark a config optimized before successful real inference.
    2. Never assume free VRAM = usable model memory.
    3. Never trust stale hardware memory measurements.
    4. Never let a failed candidate kill the parent process.
    5. Never choose a config only because it has higher tok/s.
    6. Always provide a deterministic CPU fallback.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from polymind.core.hardware.loader import load_hardware_profile
from polymind.core.hardware.models import HardwareProfile
from polymind.core.runtime.benchmark import (
    benchmark_candidate,
    get_available_gpu_memory_mb,
    get_available_ram_mb,
)
from polymind.core.runtime.types import (
    BenchmarkMetrics,
    CandidateConfig,
    CandidateStatus,
    HardwareFingerprint,
    ModelFingerprint,
    Placement,
    RuntimeConfig,
    RuntimeProfile,
    SplitMode,
    ValidationInfo,
    ValidationStatus,
    WorkloadProfile,
    DEFAULT_WORKLOADS,
)


# ── Layer estimation ───────────────────────────────────────────

def _estimate_num_layers(model: ModelFingerprint, model_size_bytes: int) -> int:
    """Estimate total layer count from model metadata or heuristics."""
    try:
        if int(model.num_layers) > 0:
            return int(model.num_layers)
    except (TypeError, ValueError):
        pass

    # Fallback heuristic based on file size
    if model_size_bytes < 1_000_000_000:
        return 22
    elif model_size_bytes < 3_000_000_000:
        return 32
    elif model_size_bytes < 8_000_000_000:
        return 40
    else:
        return 80


def _estimate_layer_size(model_size_bytes: int, num_layers: int) -> int:
    """Estimate per-layer memory in bytes.

    For MoE models, file size includes all experts but only a few
    are active per token. We still need to load all weights to VRAM
    though, so use full file size / layer count.
    """
    return model_size_bytes // max(num_layers, 1)


# ── Candidate Generation ──────────────────────────────────────

def generate_placement_candidates(
    hardware: HardwareProfile,
) -> list[Placement]:
    """Generate eligible device placements based on hardware.

    Does not blindly test every combination. Uses hardware info
    to generate only feasible candidates.
    """
    candidates: list[Placement] = []

    # Always include CPU
    candidates.append(Placement(backend="cpu", devices=[]))

    # Get usable GPUs
    usable_gpus = [
        gpu for gpu in hardware.gpus
        if gpu.compute.llama_cpp_usable and gpu.selection.enabled
    ]

    if not usable_gpus:
        return candidates

    # Single GPU candidates
    for gpu in usable_gpus:
        backend = gpu.compute.backend or "cuda"
        candidates.append(
            Placement(
                backend=backend,
                devices=[gpu.id],
                split_mode=SplitMode.NONE,
                main_gpu=gpu.id,
            )
        )

    # Multi-GPU candidates (if more than one usable GPU)
    if len(usable_gpus) > 1:
        all_ids = [gpu.id for gpu in usable_gpus]

        # Layer split multi-GPU
        candidates.append(
            Placement(
                backend=usable_gpus[0].compute.backend or "cuda",
                devices=all_ids,
                split_mode=SplitMode.LAYER,
                main_gpu=all_ids[0],
            )
        )

    return candidates


def generate_gpu_layer_candidates(
    num_layers: int,
    available_vram_bytes: int,
    layer_size: int,
    placement: Placement,
) -> list[int]:
    """Generate GPU layer offload candidates.

    Uses adaptive search: coarse grid → binary refinement.
    Never blindly tries full offload.
    """
    if placement.backend == "cpu" or available_vram_bytes <= 0:
        return [0]

    if layer_size <= 0:
        return [0]

    # Maximum layers that could possibly fit (conservative)
    max_by_vram = int(available_vram_bytes * 0.85 / layer_size)
    max_by_vram = min(max_by_vram, num_layers)

    if max_by_vram <= 0:
        return [0]

    # Coarse candidates
    candidates = set()
    candidates.add(0)

    # Try quarter points
    for fraction in [0.25, 0.5, 0.75, 1.0]:
        layer_count = max(1, int(max_by_vram * fraction))
        candidates.add(min(layer_count, num_layers))

    # Try specific common layer counts
    for target in [4, 8, 12, 16, 20, 24, 32, 40, 48]:
        if target <= max_by_vram:
            candidates.add(target)

    # Always try full offload if it looks feasible
    if max_by_vram >= num_layers:
        candidates.add(num_layers)

    return sorted(candidates)


def generate_thread_candidates(
    physical_cores: int,
    logical_cores: int,
) -> list[int]:
    """Generate thread count candidates based on CPU topology."""
    candidates = set()
    candidates.add(1)
    candidates.add(max(1, physical_cores // 2))
    candidates.add(physical_cores)
    candidates.add(min(physical_cores, logical_cores))

    if physical_cores > 2:
        candidates.add(physical_cores - 1)

    return sorted(c for c in candidates if c > 0)


def generate_context_candidates(
    model_size_bytes: int,
    min_context: int = 2048,
) -> list[int]:
    """Generate context size candidates based on model size and workload."""
    all_contexts = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    # Filter by model size constraints
    if model_size_bytes > 15_000_000_000:
        max_ctx = 2048
    elif model_size_bytes > 8_000_000_000:
        max_ctx = 4096
    elif model_size_bytes > 4_000_000_000:
        max_ctx = 8192
    else:
        max_ctx = 16384

    return [c for c in all_contexts if min_context <= c <= max_ctx]


def generate_batch_candidates() -> list[int]:
    """Generate batch size candidates."""
    return [64, 128, 256, 512]


# ── Memory estimation ──────────────────────────────────────────

def estimate_memory_mb(
    model_size_bytes: int,
    num_layers: int,
    gpu_layers: int,
    context_size: int,
    batch_size: int,
    is_moe: bool = False,
    num_kv_heads: int = 0,
    embedding_dim: int = 0,
) -> float:
    """Estimate total memory requirement in MB.

    Considers model weights, KV cache, compute buffers, and overhead.
    """
    # Model weights (GPU portion)
    layer_size = model_size_bytes // max(num_layers, 1)
    gpu_weight_mb = (gpu_layers * layer_size) / (1024 * 1024)

    # KV cache estimation
    # Rough: 2 bytes per token per layer per head * context * kv_heads
    if num_kv_heads > 0 and embedding_dim > 0:
        head_dim = embedding_dim // max(num_kv_heads, 1)
        kv_bytes = 2 * context_size * num_layers * num_kv_heads * head_dim
        kv_cache_mb = kv_bytes / (1024 * 1024)
    else:
        # Rough fallback: ~0.5MB per 1024 tokens per layer
        kv_cache_mb = (context_size / 1024) * num_layers * 0.5

    # Compute/workspace buffers (rough: batch_size * 0.1MB)
    compute_mb = batch_size * 0.001

    # Runtime overhead (~200MB for CUDA context + llama.cpp overhead)
    overhead_mb = 200 if gpu_layers > 0 else 50

    # Safety margin (15% of weight + KV)
    safety_mb = (gpu_weight_mb + kv_cache_mb) * 0.15

    total = gpu_weight_mb + kv_cache_mb + compute_mb + overhead_mb + safety_mb

    return total


def check_memory_feasibility(
    estimated_mb: float,
    available_vram_mb: float,
    safety_margin_mb: float = 200,
) -> bool:
    """Check if estimated memory fits within available VRAM with headroom."""
    if available_vram_mb <= 0:
        return False
    return estimated_mb < (available_vram_mb - safety_margin_mb)


# ── Hierarchical Search Optimizer ──────────────────────────────

class RuntimeOptimizer:
    """Adaptive hierarchical runtime optimizer.

    Search stages:
    1. Placement (CPU, single GPU, multi-GPU)
    2. Memory feasibility filter
    3. GPU layer search (adaptive)
    4. Context size search
    5. Batch/ubatch tuning
    6. Thread optimization

    Safety: All benchmarks run in isolated subprocesses.
    Performance: Only among feasible candidates.
    """

    def __init__(
        self,
        model_path: Path,
        model_id: str,
        model_size_bytes: int,
        hardware: HardwareProfile | None = None,
        model_fingerprint: ModelFingerprint | None = None,
        workload: WorkloadProfile | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
        benchmark_runs: int = 3,
        benchmark_timeout: int = 120,
    ):
        self.model_path = model_path
        self.model_id = model_id
        self.model_size_bytes = model_size_bytes
        self.hardware = hardware or load_hardware_profile()
        self.model_fp = model_fingerprint or ModelFingerprint(file_size=model_size_bytes)
        self.workload = workload or DEFAULT_WORKLOADS["default"]
        self.on_progress = on_progress
        self.benchmark_runs = benchmark_runs
        self.benchmark_timeout = benchmark_timeout

        # Hardware fingerprint
        self.hw_fingerprint = HardwareFingerprint.from_hardware_profile(self.hardware)
        self.hw_fp_hash = self.hw_fingerprint.compute_hash()
        self.model_fp_hash = self.model_fp.compute_hash()

        # Results
        self._all_candidates: list[CandidateConfig] = []
        self._tested_candidates: list[CandidateConfig] = []
        self._best_candidate: CandidateConfig | None = None

        # Hardware specs
        self.physical_cores = self.hardware.cpu.physical_cores
        self.logical_cores = self.hardware.cpu.logical_cores
        self.total_ram_mb = self.hardware.memory.total_bytes / (1024 * 1024)

        self.num_layers = _estimate_num_layers(self.model_fp, model_size_bytes)
        self.layer_size = _estimate_layer_size(model_size_bytes, self.num_layers)

        # Best GPU info
        self.usable_gpus = [
            gpu for gpu in self.hardware.gpus
            if gpu.compute.llama_cpp_usable and gpu.selection.enabled
        ]
        self.best_gpu_total_vram_mb = 0.0
        if self.usable_gpus:
            best = max(self.usable_gpus, key=lambda g: g.memory.total_bytes)
            self.best_gpu_total_vram_mb = best.memory.total_bytes / (1024 * 1024)

    def _emit(self, msg: str, step: int = 0, total: int = 0) -> None:
        if self.on_progress:
            self.on_progress(msg, step, total)

    def optimize(self) -> RuntimeProfile:
        """Run the full optimization pipeline.

        Returns the best RuntimeProfile found.
        """
        self._emit("Starting runtime optimization", 0, 0)

        # ── Stage 1: Generate placement candidates ────────────
        placements = generate_placement_candidates(self.hardware)
        self._emit(f"Generated {len(placements)} placement candidates", 1, 6)

        # ── Stage 2: Filter by memory feasibility ─────────────
        feasible_placements = self._filter_placements(placements)
        self._emit(f"{len(feasible_placements)} placements feasible", 2, 6)

        # ── Stage 3-5: For each feasible placement, search params
        for placement in feasible_placements:
            self._search_for_placement(placement)

        # ── Stage 6: Select best among all tested candidates ──
        self._select_best()

        if self._best_candidate is None:
            # Nothing worked — return conservative CPU config
            self._emit("All candidates failed — using conservative CPU fallback", 6, 6)
            return self._build_fallback_profile()

        profile = self._build_profile(self._best_candidate)
        self._emit(f"Selected best configuration", 6, 6)
        return profile

    def _filter_placements(self, placements: list[Placement]) -> list[Placement]:
        """Filter placements by rough memory feasibility."""
        feasible = []
        for p in placements:
            if p.backend == "cpu":
                # CPU: check if model fits in RAM
                needed_mb = self.model_size_bytes / (1024 * 1024) * 1.5  # 50% overhead
                if needed_mb < self.total_ram_mb * 0.85:
                    feasible.append(p)
            else:
                # GPU: rough check
                available_vram = get_available_gpu_memory_mb()
                if available_vram <= 0:
                    # Can't query — try anyway
                    feasible.append(p)
                elif self.model_size_bytes < available_vram * 0.9:
                    # Model file alone fits
                    feasible.append(p)
                else:
                    # Might fit with partial offload
                    feasible.append(p)
        return feasible if feasible else [Placement(backend="cpu", devices=[])]

    def _search_for_placement(self, placement: Placement) -> None:
        """Hierarchical search for best params under a placement.

        Uses a two-phase approach for efficiency:
        Phase 1: Quick 1-run scans to find the best parameter region
        Phase 2: Multi-run validation of the top candidate only
        """
        available_vram = get_available_gpu_memory_mb() if placement.backend != "cpu" else 0
        is_cpu = placement.backend == "cpu"

        # ── Phase 1: Quick scan (1 run each) ──────────────────
        quick_run = 1  # Use 1 run for speed

        # ── GPU layer search ──────────────────────────────────
        if is_cpu:
            gpu_layer_candidates = [0]
        else:
            gpu_layer_candidates = generate_gpu_layer_candidates(
                self.num_layers,
                int(available_vram * 1024 * 1024),
                self.layer_size,
                placement,
            )

        working_gpu_layers = 0
        best_gen_tps = 0.0

        gpu_layer_total = len(gpu_layer_candidates)
        for idx, gpu_layers in enumerate(gpu_layer_candidates):
            candidate = CandidateConfig(
                gpu_layers=gpu_layers,
                threads=self.physical_cores,
                context_size=2048,
                batch_size=256,
                placement=placement,
            )

            # Memory pre-check
            est_memory = estimate_memory_mb(
                self.model_size_bytes, self.num_layers, gpu_layers,
                2048, 256,
                self.model_fp.is_moe, self.model_fp.num_kv_heads, self.model_fp.embedding_dim,
            )
            candidate.estimated_memory_mb = est_memory

            if not is_cpu and available_vram > 0:
                if not check_memory_feasibility(est_memory, available_vram):
                    candidate.status = CandidateStatus.UNSAFE
                    candidate.failure_reason = f"Estimated {est_memory:.0f}MB > {available_vram:.0f}MB VRAM"
                    candidate.failure_type = "oom"
                    self._tested_candidates.append(candidate)
                    self._emit(f"  [gpu={gpu_layers}] SKIP (memory)", idx + 1, gpu_layer_total)
                    continue

            self._emit(f"  [gpu={gpu_layers}] scanning...", idx + 1, gpu_layer_total)

            candidate = benchmark_candidate(
                self.model_path, candidate, self.workload,
                repeat=quick_run, timeout=self.benchmark_timeout,
            )
            self._tested_candidates.append(candidate)

            if candidate.status == CandidateStatus.PASSED:
                if candidate.metrics.generation_tokens_per_sec > best_gen_tps:
                    best_gen_tps = candidate.metrics.generation_tokens_per_sec
                    working_gpu_layers = gpu_layers
            elif candidate.failure_type in ("oom", "cuda_error"):
                break

        # ── Context scan (with best GPU layers, 1 run each) ──
        context_candidates = generate_context_candidates(
            self.model_size_bytes, self.workload.min_context
        )

        for ctx in context_candidates:
            candidate = CandidateConfig(
                gpu_layers=working_gpu_layers,
                threads=self.physical_cores,
                context_size=ctx,
                batch_size=256,
                placement=placement,
            )
            est_memory = estimate_memory_mb(
                self.model_size_bytes, self.num_layers, working_gpu_layers,
                ctx, 256,
                self.model_fp.is_moe, self.model_fp.num_kv_heads, self.model_fp.embedding_dim,
            )
            candidate.estimated_memory_mb = est_memory

            if not is_cpu and available_vram > 0:
                if not check_memory_feasibility(est_memory, available_vram):
                    candidate.status = CandidateStatus.UNSAFE
                    self._tested_candidates.append(candidate)
                    continue

            candidate = benchmark_candidate(
                self.model_path, candidate, self.workload,
                repeat=quick_run, timeout=self.benchmark_timeout,
            )
            self._tested_candidates.append(candidate)

        # Find best context from tested candidates
        best_ctx = 2048
        best_tps_for_ctx = 0.0
        for c in self._tested_candidates:
            if c.status == CandidateStatus.PASSED and c.placement == placement:
                if c.metrics.generation_tokens_per_sec > best_tps_for_ctx:
                    best_tps_for_ctx = c.metrics.generation_tokens_per_sec
                    best_ctx = c.context_size

        # ── Thread scan (with best GPU layers and context) ────
        thread_candidates = generate_thread_candidates(self.physical_cores, self.logical_cores)

        for threads in thread_candidates:
            candidate = CandidateConfig(
                gpu_layers=working_gpu_layers,
                threads=threads,
                context_size=best_ctx,
                batch_size=256,
                placement=placement,
            )
            candidate = benchmark_candidate(
                self.model_path, candidate, self.workload,
                repeat=quick_run, timeout=self.benchmark_timeout,
            )
            self._tested_candidates.append(candidate)

        # Find best thread count
        best_threads = self.physical_cores
        best_tps_for_threads = 0.0
        for c in self._tested_candidates:
            if c.status == CandidateStatus.PASSED and c.placement == placement:
                if c.metrics.generation_tokens_per_sec > best_tps_for_threads:
                    best_tps_for_threads = c.metrics.generation_tokens_per_sec
                    best_threads = c.threads

        # ── Batch scan ────────────────────────────────────────
        batch_candidates = generate_batch_candidates()

        for batch in batch_candidates:
            candidate = CandidateConfig(
                gpu_layers=working_gpu_layers,
                threads=best_threads,
                context_size=best_ctx,
                batch_size=batch,
                placement=placement,
            )
            candidate = benchmark_candidate(
                self.model_path, candidate, self.workload,
                repeat=quick_run, timeout=self.benchmark_timeout,
            )
            self._tested_candidates.append(candidate)

        # ── Phase 2: Validate the best candidate with multi-run
        best_overall = None
        best_overall_tps = 0.0
        for c in self._tested_candidates:
            if c.status == CandidateStatus.PASSED and c.placement == placement:
                if c.metrics.generation_tokens_per_sec > best_overall_tps:
                    best_overall_tps = c.metrics.generation_tokens_per_sec
                    best_overall = c

        if best_overall is not None and self.benchmark_runs > 1:
            self._emit(f"  Validating best candidate with {self.benchmark_runs} runs...", 0, 0)
            validated = CandidateConfig(
                gpu_layers=best_overall.gpu_layers,
                threads=best_overall.threads,
                context_size=best_overall.context_size,
                batch_size=best_overall.batch_size,
                placement=best_overall.placement,
            )
            validated = benchmark_candidate(
                self.model_path, validated, self.workload,
                repeat=self.benchmark_runs, timeout=self.benchmark_timeout,
            )
            if validated.status == CandidateStatus.PASSED:
                # Replace the quick-scan result with the validated one
                for i, c in enumerate(self._tested_candidates):
                    if (c.gpu_layers == validated.gpu_layers and
                        c.threads == validated.threads and
                        c.context_size == validated.context_size and
                        c.batch_size == validated.batch_size and
                        c.placement == placement):
                        self._tested_candidates[i] = validated
                        break

    def _select_best(self) -> None:
        """Select the best candidate among all tested.

        Safety first: only PASSED candidates are considered.
        Score = weighted combination of generation TPS, prompt TPS,
        and memory headroom.
        """
        passed = [c for c in self._tested_candidates if c.status == CandidateStatus.PASSED]

        if not passed:
            return

        def _score(c: CandidateConfig) -> float:
            """Score a candidate. Higher is better."""
            gen_tps = c.metrics.generation_tokens_per_sec
            prompt_tps = c.metrics.prompt_tokens_per_sec
            stability = c.metrics.stability

            # Base score: generation speed (primary)
            score = gen_tps * 10.0

            # Prompt throughput bonus
            score += prompt_tps * 2.0

            # Stability bonus
            score += stability * 5.0

            # Memory headroom bonus (prefer candidates with more headroom)
            if c.estimated_memory_mb > 0 and self.best_gpu_total_vram_mb > 0:
                headroom = self.best_gpu_total_vram_mb - c.peak_memory_mb
                if headroom > 0:
                    score += min(headroom / 100, 5.0)  # Up to 5 points for headroom

            # Prefer GPU over CPU (if GPU is available)
            if c.placement.backend != "cpu":
                score += 3.0

            return score

        passed.sort(key=_score, reverse=True)
        self._best_candidate = passed[0]

    def _build_profile(self, candidate: CandidateConfig) -> RuntimeProfile:
        """Build a full RuntimeProfile from the best candidate."""
        profile = candidate.to_profile(self.model_id)
        profile.hardware_fingerprint = self.hw_fp_hash
        profile.model_fingerprint = self.model_fp_hash
        profile.workload = self.workload.name
        profile.validation = ValidationInfo(
            status=ValidationStatus.READY,
            successful_runs=candidate.metrics.runs,
            failed_runs=0,
            last_validated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        profile.safety_margin_mb = max(
            0, self.best_gpu_total_vram_mb - candidate.peak_memory_mb
        ) if self.best_gpu_total_vram_mb > 0 else 0
        return profile

    def _build_fallback_profile(self) -> RuntimeProfile:
        """Build a conservative CPU-only fallback profile."""
        profile = RuntimeProfile(
            model_id=self.model_id,
            hardware_fingerprint=self.hw_fp_hash,
            model_fingerprint=self.model_fp_hash,
            workload=self.workload.name,
            placement=Placement(backend="cpu", devices=[]),
            gpu_layers=0,
            threads=self.physical_cores,
            threads_batch=self.physical_cores,
            context_size=self.workload.min_context,
            batch_size=128,
            validation=ValidationInfo(
                status=ValidationStatus.UNKNOWN,
                successful_runs=0,
            ),
        )
        return profile

    @property
    def all_candidates(self) -> list[CandidateConfig]:
        return self._tested_candidates

    def get_summary(self) -> str:
        """Human-readable optimization summary."""
        lines: list[str] = []
        lines.append("Optimization Summary")
        lines.append("=" * 60)
        lines.append(f"Model: {self.model_path.name}")
        lines.append(f"Layers: {self.num_layers}")
        lines.append(f"Workload: {self.workload.name}")
        lines.append(f"Hardware FP: {self.hw_fp_hash}")
        lines.append(f"Model FP: {self.model_fp_hash}")
        lines.append("")
        lines.append(f"Candidates tested: {len(self._tested_candidates)}")

        passed = [c for c in self._tested_candidates if c.status == CandidateStatus.PASSED]
        failed = [c for c in self._tested_candidates if c.status == CandidateStatus.FAILED]
        unsafe = [c for c in self._tested_candidates if c.status == CandidateStatus.UNSAFE]

        lines.append(f"  Passed:  {len(passed)}")
        lines.append(f"  Failed:  {len(failed)}")
        lines.append(f"  Unsafe:  {len(unsafe)}")
        lines.append("")

        if self._best_candidate:
            c = self._best_candidate
            lines.append("Selected:")
            lines.append(f"  Backend:     {c.placement.backend}")
            lines.append(f"  GPU layers:  {c.gpu_layers}")
            lines.append(f"  Threads:     {c.threads}")
            lines.append(f"  Context:     {c.context_size}")
            lines.append(f"  Batch:       {c.batch_size}")
            if c.metrics.runs > 0:
                lines.append(f"  Gen speed:   {c.metrics.generation_tokens_per_sec:.1f} tok/s")
                lines.append(f"  Prompt speed:{c.metrics.prompt_tokens_per_sec:.1f} tok/s")
                lines.append(f"  Stability:   {c.metrics.stability:.0%}")
            lines.append(f"  Peak VRAM:   {c.peak_memory_mb:.0f} MB")
            lines.append(f"  Est. memory: {c.estimated_memory_mb:.0f} MB")
        else:
            lines.append("No viable candidate found.")

        return "\n".join(lines)


# ── Public API ─────────────────────────────────────────────────

def optimize_config(
    model_id: str,
    model_size_bytes: int,
    model_path: Path,
    hardware: HardwareProfile | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
    workload_name: str = "default",
) -> tuple[RuntimeConfig, BenchmarkMetrics | None]:
    """Find the best runtime configuration using adaptive hierarchical search.

    Returns the best RuntimeConfig and its benchmark metrics.
    Backward compatible with the old API.
    """
    workload = DEFAULT_WORKLOADS.get(workload_name, DEFAULT_WORKLOADS["default"])

    optimizer = RuntimeOptimizer(
        model_path=model_path,
        model_id=model_id,
        model_size_bytes=model_size_bytes,
        hardware=hardware,
        workload=workload,
        on_progress=on_progress,
        benchmark_runs=3,
        benchmark_timeout=120,
    )

    profile = optimizer.optimize()
    config = profile.to_runtime_config()
    metrics = profile.benchmark if profile.benchmark.runs > 0 else None

    return config, metrics


def get_benchmark_summary(results: list[Any]) -> str:
    """Format a summary of benchmark results (backward compatible)."""
    lines: list[str] = []
    successful = [r for r in results if hasattr(r, "success") and r.success and hasattr(r, "run_count") and r.run_count > 0]
    failed = [r for r in results if hasattr(r, "success") and not r.success]

    lines.append(f"Benchmark Results: {len(successful)} passed, {len(failed)} failed")

    if successful:
        successful.sort(key=lambda r: r.eval_tokens_per_sec_median, reverse=True)
        lines.append("")
        lines.append("Top Configurations:")
        for i, r in enumerate(successful[:5], 1):
            c = r.config
            lines.append(
                f"  [{i}] gpu={c.gpu_layers} threads={c.threads} "
                f"ctx={c.context_size} batch={c.batch_size}: "
                f"{r.eval_tokens_per_sec_median:.1f} tok/s"
            )

    return "\n".join(lines)
