from pathlib import Path

from llama_cpp import Llama

from polymind.core.runtime.config import default_runtime_config
from polymind.core.runtime.types import RuntimeConfig


def optimize_model(
    model_id: str,
    model_path: Path,
) -> RuntimeConfig:
    """
    Determine an initial working runtime configuration.

    The first implementation only validates the default
    configuration. Actual benchmarking will be added later.
    """

    config = default_runtime_config(model_id)

    print(f"Testing runtime configuration for model {model_id}...")
    print(f"  GPU layers: {config.gpu_layers}")
    print(f"  Threads:    {config.threads}")
    print(f"  Context:    {config.context_size}")
    print(f"  Batch:      {config.batch_size}")

    try:
        llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=config.gpu_layers,
            n_threads=config.threads,
            n_ctx=config.context_size,
            n_batch=config.batch_size,
            verbose=False,
        )

        llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": "Say OK.",
                }
            ],
            max_tokens=8,
        )

    except Exception as exc:
        raise RuntimeError(
            f"Runtime configuration failed: {exc}"
        ) from exc

    print("Runtime configuration works.")

    return config
