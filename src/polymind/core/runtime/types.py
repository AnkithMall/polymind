from dataclasses import dataclass, field


@dataclass
class RuntimeConfig:
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
