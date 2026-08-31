"""
Runtime benchmark for testing model performance
with different configurations.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from llama_cpp import Llama

from polymind.core.runtime.types import RuntimeConfig

# Standard test prompt for benchmarking
BENCHMARK_PROMPT = "Explain what an operating system does in exactly 3 sentences."

# Benchmark parameters
WARMUP_TOKENS = 10
BENCHMARK_TOKENS = 50
REPEAT_COUNT = 3  # Run each config N times, take median


class BenchmarkFailure:
    """Represents a failed benchmark with a reason."""

    def __init__(self, reason: str, config: RuntimeConfig) -> None:
        self.reason = reason
        self.config = config

    def __repr__(self) -> str:
        return f"BenchmarkFailure({self.reason})"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    config: RuntimeConfig

    # Timing
    load_time_ms: float
    prompt_eval_time_ms: float
    eval_time_ms: float

    # Throughput
    prompt_tokens_per_sec: float
    eval_tokens_per_sec: float

    # Success
    success: bool
    error: str | None = None

    @property
    def total_score(self) -> float:
        """Generation speed is the primary metric."""
        if not self.success:
            return 0.0
        return self.eval_tokens_per_sec


@dataclass
class BenchmarkSummary:
    """Aggregated results from multiple runs of the same config."""

    config: RuntimeConfig
    run_count: int
    eval_tokens_per_sec_median: float
    prompt_tokens_per_sec_median: float
    load_time_ms_median: float
    all_results: list[BenchmarkResult] = field(default_factory=list)
    success: bool = True

    @property
    def total_score(self) -> float:
        return self.eval_tokens_per_sec_median if self.success else 0.0


def benchmark_config(
    model_path: Path,
    config: RuntimeConfig,
    repeat: int = REPEAT_COUNT,
) -> BenchmarkSummary:
    """
    Run a benchmark with the given configuration.

    Runs multiple times and returns median metrics for stability.
    """
    results: list[BenchmarkResult] = []

    for _ in range(repeat):
        result = _single_benchmark(model_path, config)
        results.append(result)

        # If first run fails hard (OOM, load error), don't retry
        if not results[0].success:
            return BenchmarkSummary(
                config=config,
                run_count=0,
                eval_tokens_per_sec_median=0,
                prompt_tokens_per_sec_median=0,
                load_time_ms_median=0,
                all_results=results,
                success=False,
            )

    # All runs succeeded — compute medians
    successful = [r for r in results if r.success]
    if not successful:
        return BenchmarkSummary(
            config=config,
            run_count=0,
            eval_tokens_per_sec_median=0,
            prompt_tokens_per_sec_median=0,
            load_time_ms_median=0,
            all_results=results,
            success=False,
        )

    return BenchmarkSummary(
        config=config,
        run_count=len(successful),
        eval_tokens_per_sec_median=median([r.eval_tokens_per_sec for r in successful]),
        prompt_tokens_per_sec_median=median([r.prompt_tokens_per_sec for r in successful]),
        load_time_ms_median=median([r.load_time_ms for r in successful]),
        all_results=results,
        success=True,
    )


def _single_benchmark(
    model_path: Path,
    config: RuntimeConfig,
) -> BenchmarkResult:
    """Run a single benchmark iteration."""
    llm = None

    try:
        # Load model
        start = time.perf_counter()
        llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=config.gpu_layers,
            n_threads=config.threads,
            n_ctx=config.context_size,
            n_batch=config.batch_size,
            verbose=False,
        )
        load_time = (time.perf_counter() - start) * 1000

        # Prompt processing benchmark
        prompt_tokens = llm.tokenize(BENCHMARK_PROMPT.encode("utf-8"))

        start = time.perf_counter()
        llm.eval(prompt_tokens)
        prompt_eval_time = (time.perf_counter() - start) * 1000

        prompt_tokens_per_sec = (
            len(prompt_tokens) / (prompt_eval_time / 1000) if prompt_eval_time > 0 else 0
        )

        # Generation benchmark
        start = time.perf_counter()
        output = llm.create_completion(
            prompt=BENCHMARK_PROMPT,
            max_tokens=BENCHMARK_TOKENS,
            temperature=0.0,
        )
        eval_time = (time.perf_counter() - start) * 1000

        generated_tokens = len(llm.tokenize(output["choices"][0]["text"].encode("utf-8")))

        eval_tokens_per_sec = generated_tokens / (eval_time / 1000) if eval_time > 0 else 0

        return BenchmarkResult(
            config=config,
            load_time_ms=load_time,
            prompt_eval_time_ms=prompt_eval_time,
            eval_time_ms=eval_time,
            prompt_tokens_per_sec=prompt_tokens_per_sec,
            eval_tokens_per_sec=eval_tokens_per_sec,
            success=True,
        )

    except Exception as e:
        error_msg = str(e)
        # Classify the error
        if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
            error_msg = f"OOM: {error_msg[:80]}"
        elif "invalid" in error_msg.lower():
            error_msg = f"Invalid config: {error_msg[:80]}"

        return BenchmarkResult(
            config=config,
            load_time_ms=0,
            prompt_eval_time_ms=0,
            eval_time_ms=0,
            prompt_tokens_per_sec=0,
            eval_tokens_per_sec=0,
            success=False,
            error=error_msg,
        )

    finally:
        if llm is not None:
            del llm


def generate_test_configs(
    model_size_bytes: int,
    total_ram: int,
    available_vram: int,
    physical_cores: int,
    num_layers: int | None = None,
) -> list[RuntimeConfig]:
    """Generate configs for the adaptive progressive search."""
    # This is now a fallback — the real logic is in adaptive_benchmark()
    return []


def adaptive_benchmark(
    model_path: Path,
    model_size_bytes: int,
    total_ram: int,
    available_vram: int,
    physical_cores: int,
    num_layers: int | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> tuple[RuntimeConfig, BenchmarkSummary | None]:
    """
    Progressive adaptive benchmark.

    Instead of testing a fixed grid, it searches intelligently:

    Phase 1 — Baseline: Test CPU-only to establish a floor.
    Phase 2 — GPU layers: Binary search to find max layers that fit in VRAM.
    Phase 3 — Threads: Test a few thread counts around the best GPU config.
    Phase 4 — Context/batch: Fine-tune with the best GPU+thread combo.

    Returns the best config found and its benchmark summary.
    """
    # Estimate total layers
    if num_layers is None:
        if model_size_bytes < 1_000_000_000:
            num_layers = 22
        elif model_size_bytes < 3_000_000_000:
            num_layers = 32
        elif model_size_bytes < 8_000_000_000:
            num_layers = 40
        else:
            num_layers = 80

    # Layer size estimate
    layer_size = model_size_bytes // max(num_layers, 1)
    max_layers_by_vram = available_vram // layer_size if layer_size > 0 else 0

    # Thread candidates
    thread_candidates = sorted(set([
        2,
        max(1, physical_cores // 2),
        max(1, physical_cores - 1),
        physical_cores,
    ]))

    # Context candidates based on model size
    if model_size_bytes > 8_000_000_000:
        context_candidates = [1024, 2048]
    elif model_size_bytes > 4_000_000_000:
        context_candidates = [2048, 4096]
    else:
        context_candidates = [2048, 4096, 8192]

    best_config = RuntimeConfig(model_id="bench", gpu_layers=0, threads=physical_cores, context_size=2048, batch_size=256)
    best_summary: BenchmarkSummary | None = None
    best_speed = 0.0
    total_tests = 0

    def _test(config: RuntimeConfig, phase: str, step: int, total: int) -> BenchmarkSummary:
        nonlocal total_tests
        total_tests += 1
        if on_progress:
            g = f"gpu={config.gpu_layers}"
            t = f"threads={config.threads}"
            c = f"ctx={config.context_size}"
            b = f"batch={config.batch_size}"
            on_progress(f"[{phase}] {g} {t} {c} {b}", step, total)
        return benchmark_config(model_path, config)

    def _update_best(summary: BenchmarkSummary) -> None:
        nonlocal best_config, best_summary, best_speed
        if summary.success and summary.eval_tokens_per_sec_median > best_speed:
            best_speed = summary.eval_tokens_per_sec_median
            best_config = RuntimeConfig(
                model_id="bench",
                gpu_layers=summary.config.gpu_layers,
                threads=summary.config.threads,
                context_size=summary.config.context_size,
                batch_size=summary.config.batch_size,
            )
            best_summary = summary

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Baseline — CPU only
    # ═══════════════════════════════════════════════════════════
    baseline_config = RuntimeConfig(
        model_id="bench", gpu_layers=0,
        threads=physical_cores, context_size=2048, batch_size=256,
    )
    baseline = _test(baseline_config, "baseline", 1, 1)
    _update_best(baseline)

    if not baseline.success:
        # Can't even load on CPU — try smaller context
        for ctx in [1024, 512]:
            fallback = RuntimeConfig(
                model_id="bench", gpu_layers=0,
                threads=physical_cores, context_size=ctx, batch_size=128,
            )
            result = _test(fallback, "baseline", 1, 1)
            _update_best(result)
            if result.success:
                break

    if best_speed == 0:
        # Nothing works at all — return conservative config
        return best_config, best_summary

    # ═══════════════════════════════════════════════════════════
    # Phase 2: GPU layers — binary search for max that fits
    # ═══════════════════════════════════════════════════════════
    if available_vram > 0 and max_layers_by_vram > 0:
        # Binary search: find highest gpu_layers that works
        low = 0
        high = min(max_layers_by_vram, num_layers)
        working_gpu = 0  # highest that worked

        # Also test a few points to bracket the range
        test_points = sorted(set([
            0,
            max_layers_by_vram // 4,
            max_layers_by_vram // 2,
            max_layers_by_vram,
            num_layers,  # full offload attempt
        ]))

        phase2_total = len(test_points) + 6  # points + binary search steps
        phase2_step = 0

        for point in test_points:
            phase2_step += 1
            config = RuntimeConfig(
                model_id="bench", gpu_layers=point,
                threads=physical_cores, context_size=2048, batch_size=256,
            )
            result = _test(config, "gpu-search", phase2_step, phase2_total)
            if result.success:
                working_gpu = max(working_gpu, point)
                _update_best(result)
                low = max(low, point)
            else:
                high = min(high, point)

        # Binary search between low and high
        while low <= high and phase2_step < phase2_total:
            mid = (low + high) // 2
            if mid == low:
                break  # No progress

            phase2_step += 1
            config = RuntimeConfig(
                model_id="bench", gpu_layers=mid,
                threads=physical_cores, context_size=2048, batch_size=256,
            )
            result = _test(config, "gpu-search", phase2_step, phase2_total)

            if result.success:
                working_gpu = mid
                _update_best(result)
                low = mid + 1
            else:
                high = mid - 1

        # Test working_gpu + a few above to confirm
        for offset in [1, 2]:
            candidate = working_gpu + offset
            if candidate <= max_layers_by_vram:
                phase2_step += 1
                config = RuntimeConfig(
                    model_id="bench", gpu_layers=candidate,
                    threads=physical_cores, context_size=2048, batch_size=256,
                )
                result = _test(config, "gpu-search", phase2_step, phase2_total)
                if result.success:
                    working_gpu = candidate
                    _update_best(result)

        best_gpu_layers = working_gpu
    else:
        best_gpu_layers = 0

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Thread optimization
    # ═══════════════════════════════════════════════════════════
    phase3_total = len(thread_candidates)
    for i, threads in enumerate(thread_candidates, 1):
        config = RuntimeConfig(
            model_id="bench", gpu_layers=best_gpu_layers,
            threads=threads, context_size=2048, batch_size=256,
        )
        result = _test(config, "threads", i, phase3_total)
        _update_best(result)

    # Use best thread count going forward
    best_threads = best_config.threads

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Context and batch size tuning
    # ═══════════════════════════════════════════════════════════
    batch_candidates = [128, 256, 512]
    phase4_combos = [(ctx, batch) for ctx in context_candidates for batch in batch_candidates]
    phase4_total = len(phase4_combos)

    for i, (ctx, batch) in enumerate(phase4_combos, 1):
        config = RuntimeConfig(
            model_id="bench", gpu_layers=best_gpu_layers,
            threads=best_threads, context_size=ctx, batch_size=batch,
        )
        result = _test(config, "tuning", i, phase4_total)
        _update_best(result)

    return best_config, best_summary


def find_best_config(
    results: list[BenchmarkSummary],
) -> BenchmarkSummary | None:
    """
    Find the best performing configuration.

    Primary metric: generation tokens/sec (median).
    """
    successful = [r for r in results if r.success and r.run_count > 0]

    if not successful:
        return None

    # Sort by generation speed (primary metric)
    successful.sort(
        key=lambda r: r.eval_tokens_per_sec_median,
        reverse=True,
    )

    return successful[0]
