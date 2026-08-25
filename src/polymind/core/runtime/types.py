from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    model_id: str

    gpu_layers: int = -1
    threads: int = 4
    context_size: int = 4096
    batch_size: int = 512

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "gpu_layers": self.gpu_layers,
            "threads": self.threads,
            "context_size": self.context_size,
            "batch_size": self.batch_size,
        }
