# polymind

Hardware-aware local LLM toolkit that auto-optimizes runtime configs, ranks models by domain-specific benchmarks, and runs multi-model pipelines -- built for hardware-constrained machines.

[![CI](https://github.com/AnkithMall/polymind/actions/workflows/release.yml/badge.svg)](https://github.com/AnkithMall/polymind/actions)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Release](https://img.shields.io/badge/version-0.1.0-orange.svg)](https://github.com/AnkithMall/polymind/releases)

---

## The problem

Running LLMs locally is hard. You need to:

- Figure out what your hardware can actually handle (VRAM, RAM, GPU backends)
- Pick the right quantization from hundreds of GGUF variants on Hugging Face
- Manually tune GPU layers, thread count, context size, and batch size
- Know which model is best for coding vs. math vs. writing vs. reasoning
- Orchestrate multiple models for complex tasks

**polymind solves all of this automatically.** It scans your hardware, finds compatible models, benchmarks optimal settings, scores models across 8 domains, and orchestrates multi-model pipelines -- all from one CLI.

---

## How it works

```
Hardware Scan --> Model Search --> Auto-Rank --> Download
       |                                           |
       v                                           v
 Hardware Profile                           Model Registry
       |                                           |
       v                                           v
 Adaptive Benchmark --> Optimal Config --> Interactive Chat
       |                                           |
       v                                           v
 Domain Scores --> Multi-Model Pipeline --> Final Response
```

1. **Scan hardware** -- detects CPU, RAM, GPUs (NVIDIA/Intel), and llama.cpp with CUDA/Metal/Vulkan
2. **Search & rank** -- queries Hugging Face for GGUF models, scores each against your actual VRAM/RAM
3. **Benchmark & optimize** -- runs adaptive 4-phase benchmarks (GPU layer binary search, thread tuning, context/batch sweep) to find the fastest config per model
4. **Domain scoring** -- evaluates models across 8 domains with 160+ test questions and 5 evaluation methods
5. **Pipeline orchestration** -- decomposes complex prompts, assigns the best model per sub-task, schedules execution, synthesizes responses

---

## Prerequisites

| Requirement | Required | Notes |
|-------------|----------|-------|
| Python 3.12+ | Yes | `python3 --version` to check |
| [uv](https://docs.astral.sh/uv/) | Yes | Package manager: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| llama-cpp-python | Yes | Installed automatically with dependencies |
| NVIDIA GPU + CUDA | Optional | For GPU offloading |
| Hugging Face token | Optional | `export HF_TOKEN=hf_xxx` for gated models |

---

## Installation

```bash
# Clone
git clone git@github.com:AnkithMall/polymind.git
cd polymind

# Install (uses uv)
uv sync

# Or install with dev tools
uv sync --all-extras
```

### Quick verify

```bash
uv run polymind --help
uv run polymind hardware scan
```

---

## Usage

### CLI

```bash
# Scan your hardware first
uv run polymind hardware scan

# Search for models
uv run polymind model search "code generation"

# Download a model
uv run polymind model download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf

# List installed models
uv run polymind model list

# Optimize runtime settings (auto-benchmarks)
uv run polymind runtime optimize -m <model_id>

# Run interactive chat
uv run polymind runtime run -m <model_id>

# Launch TUI
uv run polymind tui
```

### Make targets

```bash
make tui              # Launch TUI
make cli ARGS="..."   # Run CLI with arguments
make cli-help         # Show CLI help
make test             # Run tests
make lint             # Lint code
make format           # Format code
make build            # Build distribution
```

---

## CLI reference

### Model management

| Command | Description |
|---------|-------------|
| `polymind model search <query>` | Search Hugging Face for GGUF models |
| `polymind model download <repo> <file>` | Download a GGUF model file |
| `polymind model list` | List all installed models |
| `polymind model delete <model_id>` | Remove a model from disk and registry |
| `polymind model scan` | Scan directories for GGUF files |
| `polymind model migrate` | Migrate models from legacy locations |

### Hardware

| Command | Description |
|---------|-------------|
| `polymind hardware scan` | Detect CPU, RAM, GPUs, llama.cpp backends |
| `polymind hardware show` | Display saved hardware profile |
| `polymind hardware validate` | Validate hardware profile consistency |

### Runtime

| Command | Description |
|---------|-------------|
| `polymind runtime run -m <model>` | Launch interactive chat session |
| `polymind runtime optimize -m <model>` | Adaptive benchmark for optimal settings |
| `polymind runtime optimize --all` | Optimize all installed models |

### Configuration

| Command | Description |
|---------|-------------|
| `polymind config show` | Display all configuration |
| `polymind config get <key>` | Get a config value |
| `polymind config set <key> <value>` | Set a config value |
| `polymind config edit` | Open config in $EDITOR |

### TUI

| Command | Description |
|---------|-------------|
| `polymind tui` | Launch text user interface |

---

## Architecture

```
polymind/
  src/polymind/
    core/
      hardware/       # System detection (CPU, GPU, memory, llama.cpp)
      model/          # Search, download, rank, registry, grouping
      runtime/        # Benchmark, optimize, interactive chat
      paths.py        # Artifact and model path resolution
    client/
      cli/            # Typer CLI (13 subcommands)
      tui/            # Textual TUI (5 screens, async workers)
```

### Artifact storage

All persistent data lives in `.polymind/`:

| File | Contents |
|------|----------|
| `hardware.yaml` | System hardware profile |
| `runtime.yaml` | Per-model optimized runtime configs |
| `registry.yaml` | Index of downloaded models |
| `models/` | Downloaded GGUF model files |

---

## Configuration

Override defaults with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `POLYMIND_ARTIFACT_DIR` | `./.polymind/` | Artifact storage directory |
| `POLYMIND_MODEL_DIR` | `./.polymind/models/` | Model download directory |
| `HF_TOKEN` | -- | Hugging Face auth token |

---

## Development

```bash
# Setup
git clone git@github.com:AnkithMall/polymind.git
cd polymind
uv sync

# Test
uv run pytest tests/ -v

# Lint + format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run pyright src/
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**Ankith Mall** -- [ankithmall1729@gmail.com](mailto:ankithmall1729@gmail.com)

Repository: [github.com/AnkithMall/polymind](https://github.com/AnkithMall/polymind)
