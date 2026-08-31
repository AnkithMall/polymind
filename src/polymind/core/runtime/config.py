from polymind.core.runtime.types import RuntimeConfig


def default_runtime_config(
    model_id: str,
    model_size_bytes: int | None = None,
) -> RuntimeConfig:
    """
    Return a smart default runtime configuration.

    Estimates good defaults based on model size if provided.
    The optimizer can replace these values after benchmarking.
    """

    # Estimate threads based on CPU cores
    try:
        import psutil

        physical_cores = psutil.cpu_count(logical=False) or 4
    except Exception:
        physical_cores = 4

    threads = max(1, physical_cores - 1)

    # Estimate GPU layers based on available VRAM
    gpu_layers = _estimate_gpu_layers(model_size_bytes)

    # Estimate context based on model size
    context_size = _estimate_context(model_size_bytes)

    # Estimate batch based on context
    batch_size = min(512, context_size // 4)

    return RuntimeConfig(
        model_id=model_id,
        gpu_layers=gpu_layers,
        threads=threads,
        context_size=context_size,
        batch_size=batch_size,
    )


def _estimate_gpu_layers(model_size_bytes: int | None) -> int:
    """
    Estimate how many GPU layers to use.

    - Small models (< 2GB): try full offload
    - Medium models (2-6GB): partial offload
    - Large models (> 6GB): CPU only or minimal offload
    """
    if model_size_bytes is None:
        return -1  # Let optimizer decide

    # Get available VRAM
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            free_vram_mb = int(result.stdout.strip().split("\n")[0])
            free_vram_bytes = free_vram_mb * 1024 * 1024
        else:
            return 0  # No GPU or error
    except Exception:
        return 0  # Can't detect GPU

    if free_vram_bytes <= 0:
        return 0

    # Use 85% of VRAM as safe limit
    usable_vram = int(free_vram_bytes * 0.85)

    if model_size_bytes <= usable_vram:
        return -1  # Full offload fits

    # Partial offload - estimate layers
    # Roughly 50% of model is weights that go to GPU
    weight_size = model_size_bytes // 2
    layers_that_fit = int(usable_vram / (weight_size / 32))  # Assume ~32 layers

    return max(0, min(layers_that_fit, 32))


def _estimate_context(model_size_bytes: int | None) -> int:
    """
    Estimate context size based on available RAM and model size.
    """
    try:
        import psutil

        total_ram = psutil.virtual_memory().total
    except Exception:
        total_ram = 16 * 1024 * 1024 * 1024  # Assume 16GB

    if model_size_bytes is None:
        return 4096

    # Reserve RAM for model weights + overhead
    model_overhead = int(model_size_bytes * 1.5)
    available = total_ram - model_overhead

    if available < 0:
        return 2048  # Very tight, use minimal context

    # Each token uses roughly 2-8KB depending on model
    bytes_per_token = max(2048, model_size_bytes // 500_000_000 * 1024)
    max_tokens = available // bytes_per_token

    if max_tokens >= 16384:
        return 16384
    elif max_tokens >= 8192:
        return 8192
    elif max_tokens >= 4096:
        return 4096
    else:
        return 2048
