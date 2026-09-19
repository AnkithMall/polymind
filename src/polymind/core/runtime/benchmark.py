"""Isolated benchmark system for runtime configuration testing.

Every benchmark candidate runs in a separate subprocess so that CUDA
crashes, SIGABRT, or native segfaults in llama.cpp never kill the
main PolyMind process.

Architecture:
    Optimizer parent process
    ├── benchmark candidate subprocess
    ├── benchmark candidate subprocess
    └── benchmark candidate subprocess

Each subprocess:
    1. Loads the model
    2. Performs warmup inference
    3. Measures prompt processing and generation speed
    4. Measures peak VRAM/RAM
    5. Exits cleanly with structured JSON results
"""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path
from statistics import median

from polymind.core.runtime.types import (
    BenchmarkMetrics,
    CandidateConfig,
    CandidateStatus,
    SubprocessBenchmarkResult,
    WorkloadProfile,
    DEFAULT_WORKLOADS,
)

# Subprocess timeout (seconds)
BENCHMARK_TIMEOUT = 120

# Number of repeated runs per candidate
REPEAT_COUNT = 3


# ── Benchmark worker (runs in subprocess) ──────────────────────

def _benchmark_worker(
    model_path: str,
    config_dict: dict,
    prompt: str,
    max_tokens: int,
    result_queue: multiprocessing.Queue,
) -> None:
    """Worker function that runs inside a subprocess.

    This function loads the model, runs inference, measures metrics,
    and puts the result in a queue. If anything crashes (CUDA, OOM,
    segfault), the subprocess dies but the parent survives.
    """
    try:
        from llama_cpp import Llama

        t_start = time.perf_counter()

        # Load model
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=config_dict.get("gpu_layers", 0),
            n_threads=config_dict.get("threads", 4),
            n_ctx=config_dict.get("context_size", 2048),
            n_batch=config_dict.get("batch_size", 256),
            verbose=False,
        )

        load_time = (time.perf_counter() - t_start) * 1000

        # Measure peak memory after load
        peak_ram_mb = _get_peak_ram_mb()

        # Tokenize prompt
        prompt_tokens = llm.tokenize(prompt.encode("utf-8"))
        prompt_token_count = len(prompt_tokens)

        # Warmup (short)
        try:
            llm.eval(prompt_tokens[:min(8, len(prompt_tokens))])
        except Exception:
            pass

        # Measure prompt processing (prefill)
        t_prefill_start = time.perf_counter()
        llm.eval(prompt_tokens)
        t_prefill_end = time.perf_counter()
        prefill_ms = (t_prefill_end - t_prefill_start) * 1000

        ttft_ms = prefill_ms  # Time to first token ≈ prefill time for single prompt

        prompt_tps = prompt_token_count / (prefill_ms / 1000) if prefill_ms > 0 else 0

        # Measure generation (decode)
        t_gen_start = time.perf_counter()
        output = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        t_gen_end = time.perf_counter()
        gen_ms = (t_gen_end - t_gen_start) * 1000

        generated_text = output["choices"][0]["text"]
        generated_tokens = len(llm.tokenize(generated_text.encode("utf-8")))

        gen_tps = generated_tokens / (gen_ms / 1000) if gen_ms > 0 else 0

        # Measure peak VRAM
        peak_vram_mb = _get_peak_vram_mb()

        # Total RAM after inference
        final_ram_mb = _get_peak_ram_mb()

        del llm

        result = SubprocessBenchmarkResult(
            success=True,
            load_time_ms=load_time,
            prompt_tokens_per_sec=prompt_tps,
            generation_tokens_per_sec=gen_tps,
            time_to_first_token_ms=ttft_ms,
            total_latency_ms=gen_ms,
            peak_vram_mb=peak_vram_mb,
            peak_ram_mb=final_ram_mb,
            generated_tokens=generated_tokens,
            prompt_tokens=prompt_token_count,
        )

    except MemoryError:
        result = SubprocessBenchmarkResult(
            success=False,
            error="Out of memory",
            failure_type="oom",
        )
    except Exception as e:
        error_str = str(e).lower()
        if "cuda" in error_str or "out of memory" in error_str or "oom" in error_str:
            failure_type = "oom" if "memory" in error_str else "cuda_error"
        elif "timeout" in error_str:
            failure_type = "timeout"
        else:
            failure_type = "load_error"
        result = SubprocessBenchmarkResult(
            success=False,
            error=str(e)[:200],
            failure_type=failure_type,
        )

    try:
        result_queue.put_nowait(result)
    except Exception:
        pass


def _get_peak_vram_mb() -> float:
    """Query current GPU VRAM usage via nvidia-smi."""
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            values = [float(v.strip()) for v in r.stdout.strip().split("\n") if v.strip()]
            return max(values) if values else 0.0
    except Exception:
        pass
    return 0.0


def _get_peak_ram_mb() -> float:
    """Query current process RAM usage."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def _get_free_vram_mb() -> float:
    """Query current free GPU VRAM."""
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            values = [float(v.strip()) for v in r.stdout.strip().split("\n") if v.strip()]
            return min(values) if values else 0.0
    except Exception:
        pass
    return 0.0


# ── Subprocess benchmark runner ────────────────────────────────

def run_benchmark_subprocess(
    model_path: Path,
    config: CandidateConfig,
    prompt: str,
    max_tokens: int = 256,
    timeout: int = BENCHMARK_TIMEOUT,
) -> SubprocessBenchmarkResult:
    """Run a single benchmark in an isolated subprocess.

    This is the core safety mechanism: if llama.cpp crashes with
    SIGABRT or SIGSEGV, only the subprocess dies, not the main process.
    Uses 'spawn' start method to avoid forking CUDA state.
    """
    config_dict = {
        "gpu_layers": config.gpu_layers,
        "threads": config.threads,
        "context_size": config.context_size,
        "batch_size": config.batch_size,
    }

    try:
        ctx = multiprocessing.get_context("spawn")
    except ValueError:
        ctx = multiprocessing.get_context()

    result_queue: multiprocessing.Queue[SubprocessBenchmarkResult] = ctx.Queue()

    proc = ctx.Process(
        target=_benchmark_worker,
        args=(str(model_path), config_dict, prompt, max_tokens, result_queue),
        daemon=True,
    )

    proc.start()

    # Wait for completion with timeout
    proc.join(timeout=timeout)

    if proc.is_alive():
        # Timeout — kill the subprocess
        try:
            proc.kill()
        except Exception:
            pass
        proc.join(timeout=5)
        return SubprocessBenchmarkResult(
            success=False,
            error=f"Benchmark timed out after {timeout}s",
            failure_type="timeout",
        )

    exit_code = proc.exitcode

    if exit_code != 0:
        # Process crashed (SIGABRT, SIGSEGV, CUDA abort, etc.)
        signal_name = _signal_name(exit_code) if exit_code and exit_code < 0 else ""
        return SubprocessBenchmarkResult(
            success=False,
            error=f"Subprocess crashed with exit code {exit_code} {signal_name}",
            failure_type=_classify_exit_code(exit_code),
        )

    # Check if result was placed in queue
    if not result_queue.empty():
        try:
            return result_queue.get_nowait()
        except Exception:
            pass

    return SubprocessBenchmarkResult(
        success=False,
        error="No result from benchmark subprocess",
        failure_type="unknown",
    )


def _signal_name(code: int) -> str:
    """Convert negative exit code to signal name."""
    try:
        sig = signal.Signals(-code)
        return f"({sig.name})"
    except (ValueError, OSError):
        return ""


def _classify_exit_code(code: int) -> str:
    """Classify a non-zero exit code into a failure type."""
    if code == -6:
        return "cuda_error"  # SIGABRT
    elif code == -11:
        return "crash"  # SIGSEGV
    elif code == -9:
        return "timeout"  # SIGKILL
    else:
        return "unknown"


# ── Multi-run benchmark with median ────────────────────────────

def benchmark_candidate(
    model_path: Path,
    candidate: CandidateConfig,
    workload: WorkloadProfile | None = None,
    repeat: int = REPEAT_COUNT,
    timeout: int = BENCHMARK_TIMEOUT,
    on_progress: callable | None = None,
) -> CandidateConfig:
    """Benchmark a candidate configuration with repeated runs.

    Takes median metrics for stability. Records all results for
    stability analysis.

    Returns the candidate with updated status and metrics.
    """
    if workload is None:
        workload = DEFAULT_WORKLOADS["default"]

    prompt = workload.representative_prompt
    max_tokens = workload.target_output_tokens

    results: list[SubprocessBenchmarkResult] = []

    for run_idx in range(repeat):
        if on_progress:
            on_progress(
                f"  run {run_idx + 1}/{repeat}",
                run_idx + 1,
                repeat,
            )

        result = run_benchmark_subprocess(
            model_path=model_path,
            config=candidate,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        results.append(result)

        # If first run fails hard (OOM, load error), don't retry
        if not result.success and result.failure_type in ("oom", "cuda_error", "load_error"):
            candidate.status = CandidateStatus.FAILED
            candidate.failure_reason = result.error
            candidate.failure_type = result.failure_type
            return candidate

    # Analyze results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not successful:
        candidate.status = CandidateStatus.FAILED
        if failed:
            candidate.failure_reason = failed[0].error
            candidate.failure_type = failed[0].failure_type
        else:
            candidate.failure_reason = "All runs failed"
        return candidate

    # Compute medians
    candidate.metrics = BenchmarkMetrics(
        load_time_ms=median([r.load_time_ms for r in successful]),
        prompt_tokens_per_sec=median([r.prompt_tokens_per_sec for r in successful]),
        generation_tokens_per_sec=median([r.generation_tokens_per_sec for r in successful]),
        time_to_first_token_ms=median([r.time_to_first_token_ms for r in successful]),
        total_latency_ms=median([r.total_latency_ms for r in successful]),
        peak_vram_mb=max(r.peak_vram_mb for r in successful),
        peak_ram_mb=max(r.peak_ram_mb for r in successful),
        stability=len(successful) / len(results),
        runs=len(successful),
    )

    candidate.peak_memory_mb = candidate.metrics.peak_vram_mb

    # Stability check: need at least 2/3 runs to succeed
    if len(successful) < 2 and len(results) > 1:
        candidate.status = CandidateStatus.FAILED
        candidate.failure_reason = f"Unstable: only {len(successful)}/{len(results)} runs succeeded"
        candidate.failure_type = "instability"
        return candidate

    candidate.status = CandidateStatus.PASSED
    return candidate


# ── Free memory check ──────────────────────────────────────────

def get_available_gpu_memory_mb() -> float:
    """Get current free GPU VRAM in MB. Returns 0 if no GPU."""
    return _get_free_vram_mb()


def get_available_ram_mb() -> float:
    """Get current free system RAM in MB."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except Exception:
        return 0.0


# ── Legacy compatibility ───────────────────────────────────────

BENCHMARK_PROMPT = "Explain what an operating system does in exactly 3 sentences."
WARMUP_TOKENS = 10
BENCHMARK_TOKENS = 50
REPEAT_COUNT_DEFAULT = 3


class BenchmarkFailure:
    """Represents a failed benchmark with a reason."""

    def __init__(self, reason: str, config: object) -> None:
        self.reason = reason
        self.config = config

    def __repr__(self) -> str:
        return f"BenchmarkFailure({self.reason})"
