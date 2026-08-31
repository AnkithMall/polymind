"""
Hardware-aware runtime configuration optimizer.

Runs adaptive benchmarks to determine the best llama.cpp
parameters for the detected hardware and model.
"""

from collections.abc import Callable
from pathlib import Path

from polymind.core.hardware.loader import load_hardware_profile
from polymind.core.hardware.models import HardwareProfile
from polymind.core.runtime.benchmark import (
    BenchmarkSummary,
    adaptive_benchmark,
)
from polymind.core.runtime.types import RuntimeConfig


def optimize_config(
    model_id: str,
    model_size_bytes: int,
    model_path: Path,
    hardware: HardwareProfile | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> tuple[RuntimeConfig, BenchmarkSummary | None]:
    """
    Find the best runtime configuration by running adaptive benchmarks.

    Progressively searches for optimal GPU layers, threads, context,
    and batch size based on what works on the hardware.

    Returns the best config and its benchmark summary.
    """
    if hardware is None:
        hardware = load_hardware_profile()

    # Get hardware specs
    available_vram = 0
    selected_gpus = [
        gpu for gpu in hardware.gpus if gpu.selection.enabled and gpu.compute.llama_cpp_usable
    ]

    if selected_gpus:
        best_gpu = max(
            selected_gpus,
            key=lambda g: g.memory.total_bytes,
        )
        available_vram = best_gpu.memory.available_bytes or best_gpu.memory.total_bytes or 0

    total_ram = hardware.memory.total_bytes
    physical_cores = hardware.cpu.physical_cores

    # Run adaptive benchmark
    best_config, best_summary = adaptive_benchmark(
        model_path=model_path,
        model_size_bytes=model_size_bytes,
        total_ram=total_ram,
        available_vram=available_vram,
        physical_cores=physical_cores,
        on_progress=on_progress,
    )

    # Return with actual model_id
    return RuntimeConfig(
        model_id=model_id,
        gpu_layers=best_config.gpu_layers,
        threads=best_config.threads,
        context_size=best_config.context_size,
        batch_size=best_config.batch_size,
    ), best_summary


def get_benchmark_summary(
    results: list[BenchmarkSummary],
) -> str:
    """Format a summary of benchmark results."""
    lines: list[str] = []

    successful = [r for r in results if r.success and r.run_count > 0]
    failed = [r for r in results if not r.success]

    lines.append(f"Benchmark Results: {len(successful)} passed, {len(failed)} failed")
    lines.append("")

    if successful:
        successful.sort(key=lambda r: r.eval_tokens_per_sec_median, reverse=True)

        lines.append("Top Configurations (by generation speed):")
        lines.append("-" * 75)
        lines.append(
            f"{'Rank':<5} {'GPU':<6} {'Threads':<8} {'Context':<9} {'Batch':<7} "
            f"{'Gen tok/s':<12} {'Prompt tok/s':<14} {'Score':<8}"
        )
        lines.append("-" * 75)

        for i, result in enumerate(successful[:5], 1):
            c = result.config
            lines.append(
                f"{i:<5} {c.gpu_layers:<6} {c.threads:<8} {c.context_size:<9} "
                f"{c.batch_size:<7} {result.eval_tokens_per_sec_median:<12.1f} "
                f"{result.prompt_tokens_per_sec_median:<14.1f} {result.total_score:<8.1f}"
            )

        lines.append("-" * 75)

    if failed:
        lines.append("")
        lines.append(f"Failed Configurations ({len(failed)}):")
        for r in failed[:3]:
            error_msg = "unknown"
            for result in r.all_results:
                if not result.success and result.error:
                    error_msg = result.error
                    break
            lines.append(
                f"  gpu={r.config.gpu_layers} threads={r.config.threads} "
                f"ctx={r.config.context_size} batch={r.config.batch_size}: {error_msg[:50]}"
            )

    return "\n".join(lines)
