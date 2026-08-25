from polymind.core.runtime.types import RuntimeConfig


def default_runtime_config(model_id: str) -> RuntimeConfig:
    """
    Return the initial runtime configuration.

    This is intentionally conservative. The optimizer can
    replace these values after hardware/model testing.
    """
    return RuntimeConfig(
        model_id=model_id,
        gpu_layers=-1,
        threads=4,
        context_size=4096,
        batch_size=512,
    )
