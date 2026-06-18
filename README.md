
# PolyMind

**Multi-Specialist LLM Orchestrator** — CLI · TUI · Web — Local-First · Hardware-Optimised

[![CI](https://github.com/AnkithMall/polymind/actions/workflows/ci.yml/badge.svg)](https://github.com/AnkithMall/polymind/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Table of Contents

1. [What is PolyMind?](#what-is-polymind)
2. [Problem Statement](#problem-statement)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Core Pipeline](#core-pipeline)
6. [Features](#features)
7. [Installation](#installation)
8. [Quick Start](#quick-start)
9. [Usage](#usage)
10. [Configuration](#configuration)
11. [Frontends](#frontends)
12. [Comparison with Existing Tools](#comparison-with-existing-tools)
13. [Development](#development)
14. [License](#license)

---

## What is PolyMind?

PolyMind is an open-source, local-first application that orchestrates **multiple specialist LLMs** to answer a single prompt better than any one model could alone.

Instead of relying on one general-purpose model, PolyMind:

1. **Decomposes** your prompt into typed subtasks (code, math, reasoning, creative, etc.)
2. **Routes** each subtask to the model that is measurably best at that domain
3. **Schedules** execution to minimise VRAM load/unload cycles on your hardware
4. **Synthesizes** all outputs into one coherent final response

It works fully offline with Ollama or LM Studio, requires no cloud dependency, and provides three frontends — CLI, TUI, and Web UI.

---

## Problem Statement

Running a single large general-purpose model on a laptop is slow and produces mediocre results on specialist tasks. Running multiple models naively is even slower — loading and unloading models from VRAM is expensive.

**No existing tool helps users:**

- Discover which of their local models is actually best at each task type
- Automatically route subtasks to the right model based on measured accuracy
- Schedule execution to batch tasks by model, minimising VRAM swaps
- Do all of this from a terminal (or browser) with no cloud dependency

PolyMind solves all of these problems with a data-driven, hardware-aware approach.

---
>>>>>>> dcf73f6 (Add comprehensive README with architecture, features, usage, and competitive analysis)

## Architecture

```
<<<<<<< HEAD
User prompt
    │
    ▼
Analyzer LLM (small, fast)
  → decomposes into subtasks with domain tags
    │
    ▼
Execution engine
  → sequential (laptop/single-GPU safe) or parallel
  → routes each subtask to its specialist model
    │
    ▼
Synthesizer LLM
  → merges all outputs into one coherent response
    │
    ▼
Final response + transparency panel
```

## Configuration

Edit `config.yaml` to map domains to your models:

```yaml
execution:
  mode: sequential      # or parallel

specialists:
  code:
    model: deepseek-coder:6.7b
    provider: ollama
  math:
    model: qwen2.5-math:7b
    provider: ollama
  creative:
    model: mistralai/mistral-7b-instruct
    provider: openrouter
    api_key: ${OPENROUTER_API_KEY}
```

## Supported Providers

| Provider   | Config value  | Notes                          |
|------------|---------------|--------------------------------|
| Ollama     | `ollama`      | Local, free, GPU optional      |
| LM Studio  | `lmstudio`    | Local, free, GUI               |
| OpenRouter | `openrouter`  | 200+ models, free tier exists  |
| OpenAI     | `openai`      | Paid API                       |
| Anthropic  | `anthropic`   | Paid API                       |

## Execution Modes

- **Sequential** — subtasks run one at a time. Safe for single-GPU laptops. Use `pass_context: true` to chain model outputs.
- **Parallel** — independent subtasks run concurrently. Faster for cloud providers.

Switch modes without restarting: use the toggle in the UI header, or set `execution_mode` in API requests.

## API

PolyMind is OpenAI-compatible. Point any OpenAI client at `http://localhost:8000`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
response = client.chat.completions.create(
    model="polymind",
    messages=[{"role": "user", "content": "Explain recursion and write a Python example"}]
)
```

## License

MIT
# polymind
=======
┌──────────────────────────────────────────────────────────────────────────┐
│                             USER PROMPT                                  │
└──────────────────────────┬───────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Analyzer    │───►│  Scheduler   │───►│  Executor    │               │
│  │  (Router     │    │  (DAG +      │    │  (Subtask    │               │
│  │   LLM)       │    │   Batching)  │    │   Runner)    │               │
│  └──────────────┘    └──────────────┘    └──────┬───────┘               │
│         │                      │                 │                       │
│         │              ┌───────┴───────┐         │                       │
│         │              │  Rank Store   │         │                       │
│         │              │  (ranks.yaml) │         │                       │
│         │              └───────────────┘         │                       │
│         ▼                                        ▼                       │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                     Synthesizer                                  │    │
│  │           (Merges all subtask outputs)                           │    │
│  └──────────────────────────┬───────────────────────────────────────┘    │
│                             │                                            │
│                             ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                      FINAL RESPONSE                               │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │   CLI        │  │   TUI        │  │   Web UI     │                   │
│  │   (Typer)    │  │   (Textual)  │  │   (FastAPI)   │                   │
│  └──────────────┘  └──────────────┘  └──────────────┘                   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

```
Prompt ──► Analyze ──► Rank Lookup ──► Schedule ──► Execute ──► Synthesize ──► Response
              │              │               │            │              │
              ▼              ▼               ▼            ▼              ▼
         Subtask Plan   Best Model per   Batch Tasks   Run Subtasks   Merge Outputs
         (JSON +        Domain from      by Model to   with Retry +   via Synthesizer
         Domains)       ranks.yaml       Minimize       Fallback       LLM
                                          VRAM Swaps
```

### Smart Scheduler Algorithm

The scheduler is PolyMind's most distinctive feature. Loading a model into VRAM typically takes 5–30 seconds. Naive execution causes repeated load/unload cycles.

```
Input: 6 tasks, 3 models
T1(code), T2(math), T3(reasoning), T4(code), T5(code), T6(math)
Dependencies: T2 depends on T1, T5 depends on T3

Naive (dependency order):  8 model loads
  Load A → T1 → Unload A → Load B → T2 → Unload B → Load C → T3
  → Unload C → Load A → T4 → Load A → T5 → Unload A → Load B → T6

Smart (model-aware):       4 model loads
  Load A → T1, T4, T5 → Unload A → Load B → T2, T6 → Unload B → Load C → T3
```

The smart scheduler:
1. Builds a dependency DAG from the analyzer's plan
2. Groups tasks by their assigned model
3. Walks the DAG topologically with **lookahead** — when loading a model, scans ahead for all ready tasks that use the same model
4. Executes the full batch before unloading

---

## Project Structure

```
polymind/
├── pyproject.toml                  # Project config, deps, entry points
├── .github/workflows/ci.yml        # CI: pytest on push (3.11, 3.12)
│
├── src/polymind/
│   ├── __init__.py
│   │
│   ├── core/                       # ★ Core library (provider-agnostic)
│   │   ├── __init__.py             #   Public API exports
│   │   ├── types.py                #   Pydantic models (Subtask, AnalyzerPlan,
│   │   │                           #     RankEntry, ExecutionSchedule, etc.)
│   │   ├── config.py               #   YAML config loader with ${ENV_VAR}
│   │   ├── providers.py            #   LiteLLM model string builder
│   │   ├── fallback.py             #   Retry with backoff + fallback chain
│   │   ├── analyzer.py             #   Router LLM: prompt → subtask plan
│   │   ├── executor.py             #   Subtask execution with retry/context
│   │   ├── synthesizer.py          #   Merge subtask outputs (streaming + non)
│   │   ├── scheduler.py            #   DAG builder, topological sort,
│   │   │                           #     model-aware batching
│   │   ├── benchmark.py            #   9 domain × 5 tasks, scoring,
│   │   │                           #     ranks.yaml I/O
│   │   ├── hardware.py             #   RAM/VRAM/CPU scanner
│   │   └── context.py              #   Token estimation, budget manager
│   │
│   ├── cli/                        # CLI frontend (Typer + Rich)
│   │   ├── __init__.py
│   │   └── main.py                 #   6 commands: ask, benchmark, ranks,
│   │                               #     config-init, status, diff
│   │
│   ├── tui/                        # TUI frontend (Textual)
│   │   ├── __init__.py
│   │   ├── __main__.py             #   Entry: python3 -m polymind.tui
│   │   └── app.py                  #   2-panel chat + pipeline inspector
│   │
│   └── web/                        # Web frontend (FastAPI + SSE)
│       ├── __init__.py
│       ├── app.py                  #   API: /api/ask, /api/benchmark, etc.
│       └── static/
│           └── index.html          #   SPA with chat + pipeline panel
│
└── tests/                          # 114+ tests across all modules
    ├── test_types.py
    ├── test_config.py
    ├── test_providers.py
    ├── test_fallback.py
    ├── test_analyzer.py
    ├── test_executor.py
    ├── test_synthesizer.py
    ├── test_scheduler.py
    ├── test_benchmark.py
    ├── test_hardware.py
    ├── test_context.py
    ├── test_profiles.py
    └── test_cli.py
```

---

## Core Pipeline

### 1. Analyze (`core/analyzer.py`)
A lightweight router LLM reads the prompt and emits a structured JSON plan: a list of subtasks with domain tags and dependency edges. If the router is unavailable or returns invalid JSON, PolyMind falls back to a single-task plan with domain `general`.

### 2. Schedule (`core/scheduler.py`)
- **Model assignment**: each subtask's domain is looked up in `ranks.yaml`; the best model is assigned based on the configured `ranking_mode` (accuracy, cost, or cost_effective)
- **Model source filter**: models can be restricted to `local`, `online`, or `all` via the `model_source` config option
- **DAG construction**: dependency edges form a directed acyclic graph
- **Topological sort**: Kahn's algorithm produces a valid execution order
- **Lookahead batching**: when loading a model, all ready tasks using that model are batched together, minimising VRAM swaps

Three strategies available:
| Strategy | Description | Use Case |
|----------|-------------|---------|
| `model_aware` | Batch tasks by model with lookahead | Default, best for < 16GB VRAM |
| `sequential` | One task per batch, dependency order | Debugging, minimal VRAM |
| `parallel` | All ready tasks execute per round | High-RAM servers |

### 3. Execute (`core/executor.py`)
Each subtask is passed to its assigned model via LiteLLM. Execution includes:
- **Retry with exponential backoff** (configurable, default 2 retries)
- **Fallback chain** (primary → fallback → error)
- **Context injection** — prior outputs from dependencies are prepended
- **Keep-alive** — Ollama's `keep_alive` parameter keeps models warm

### 4. Synthesize (`core/synthesizer.py`)
A configurable synthesizer model receives the original prompt and all subtask outputs, producing a single coherent response. Supports both non-streaming and async streaming modes.

---

## Features

### Core (MVP)

| Feature | File | Description |
|---------|------|-------------|
| **Prompt Decomposition** | `core/analyzer.py` | Router LLM breaks prompts into typed subtasks with dependency graphs |
| **Model Benchmarking** | `core/benchmark.py` | 9 domains × 5 benchmark tasks; exact-match + LLM-as-judge scoring |
| **Cost Tracking** | `core/benchmark.py` | Tracks token usage and computes cost per model during benchmarks |
| **Cost-Based Ranking** | `core/types.py` | Ranking modes: `accuracy`, `cost`, `cost_effective` (accuracy ÷ cost) |
| **Model Source Filter** | `core/scheduler.py` | Filter models by source: `local`, `online`, or `all` |
| **LiteLLM Proxy** | `core/config.py` | Route all calls through a LiteLLM proxy server |
| **Smart Scheduling** | `core/scheduler.py` | DAG-based model-aware batching reduces VRAM load events by 40-60% |
| **Multi-Provider** | `core/providers.py` | Ollama, LM Studio, OpenRouter, OpenAI, Anthropic via LiteLLM |
| **Fallback Chain** | `core/fallback.py` | Retry with backoff → fallback model → error result |
| **Config Management** | `core/config.py` | YAML config with `${ENV_VAR}` resolution |

### Polish Features

| Feature | File | Description |
|---------|------|-------------|
| **Hardware Profiler** | `core/hardware.py` | Scans RAM/VRAM/CPU, recommends optimal strategy |
| **Context Budget** | `core/context.py` | Token estimation, truncation, 17 model family limits |
| **Routing Profiles** | `core/config.py` | `quality`/`fast`/`private` presets |
| **Keep-Alive** | `core/executor.py` | Ollama model warm-up between batches |

### Frontends

| Frontend | Framework | Entry Point |
|----------|-----------|-------------|
| **CLI** | Typer + Rich | `polymind` |
| **TUI** | Textual | `polymind-tui` |
| **Web UI** | FastAPI + SSE | `polymind-web` (→ `http://127.0.0.1:8765`) |

### CLI Commands

```
polymind ask <prompt>          Run a prompt through the full pipeline
polymind benchmark <models>    Run benchmark tasks against models
polymind ranks                 Display current model rankings
polymind status                Show config health and rankings age
polymind config-init           Interactive config wizard
polymind diff <prompt> <models> Compare model outputs side by side
```

---

## Installation

### From Source

```bash
git clone git@github.com:AnkithMall/polymind.git
cd polymind
git checkout polymind-v2
pip install -e .                    # Core only
pip install -e ".[cli]"             # Core + CLI (recommended)
pip install -e ".[tui]"             # Core + TUI
pip install -e ".[web]"             # Core + Web UI
pip install -e ".[all]"             # Everything
```

### Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| `pydantic>=2.0` | Core | Data models and validation |
| `pyyaml>=6.0` | Core | Config/session file I/O |
| `litellm>=1.40` | Core | Universal LLM API |
| `typer>=0.12` | CLI | CLI framework |
| `rich>=13.0` | CLI | Terminal formatting |
| `textual>=1.0` | TUI | Terminal UI framework |
| `fastapi>=0.100` | Web | Web server |
| `uvicorn>=0.20` | Web | ASGI server |

---

## Quick Start

### 1. Configure

```bash
polymind config-init
```

This creates `~/.polymind/config.yaml` interactively.

### 2. Run a Benchmark

```bash
polymind benchmark ollama/llama3.2:1b ollama/mistral
```

Measures each model across 9 domains and writes results to `~/.polymind/ranks.yaml`.

### 3. Ask a Question

```bash
polymind ask "Write a Python script that fetches stock prices and analyzes trends"
```

### 4. Launch TUI

```bash
polymind-tui
```

![TUI Screenshot](docs/tui-screenshot.png)

### 5. Launch Web UI

```bash
polymind-web
# Open http://127.0.0.1:8765
```

---

## Configuration

### `~/.polymind/config.yaml`

```yaml
models:
  - name: llama3.2:1b
    provider: ollama
router_model: ollama/llama3.2:1b
synthesizer_model: null
judge_model: ollama/llama3.2:1b
scheduler:
  strategy: model_aware
  pass_context: true
data_dir: ~/.polymind
verbose: false
profile: null
keep_alive: null
litellm_proxy: null              # LiteLLM proxy base URL (e.g. http://localhost:4000)
ranking_mode: accuracy           # accuracy | cost | cost_effective
model_source: all                # local | online | all
```

### `~/.polymind/ranks.yaml`

Generated by `polymind benchmark`. Example:

```yaml
entries:
  - model: ollama/llama3.2:1b
    domain: code
    score: 0.85
    latency_ms: 2340.5
    cost: 0.0                   # $0 for local models
    timestamp: "2026-06-18T09:30:00"
  - model: ollama/mistral
    domain: code
    score: 0.92
    latency_ms: 1890.2
    cost: 0.0
    timestamp: "2026-06-18T09:35:00"
  - model: openai/gpt-4o-mini
    domain: code
    score: 0.97
    latency_ms: 870.2
    cost: 0.000423               # $0.0004 per task
    timestamp: "2026-06-18T09:40:00"
```

### Routing Profiles

| Profile | Strategy | Best For |
|---------|----------|----------|
| `quality` | model_aware | Maximum accuracy |
| `fast` | sequential | Quick responses |
| `private` | sequential | Fully offline |

Set via config: `profile: quality`

### Ranking Modes

| Mode | Behaviour | Formula |
|------|-----------|---------|
| `accuracy` | Highest benchmark score wins (default) | `score` |
| `cost` | Lowest cost per task wins | `-(cost)` |
| `cost_effective` | Best accuracy per dollar | `score / cost` |

When `ranking_mode` is set to `cost` or `cost_effective`, benchmark cost data is required for all models. Local models (Ollama, LM Studio) cost \$0. Online model pricing is built-in for common models (GPT-4o, Claude, etc.).

### Model Source Filter

| Value | Behaviour |
|-------|-----------|
| `all` | Use any provider's models (default) |
| `local` | Only use local models (Ollama, LM Studio) |
| `online` | Only use online models (OpenAI, Anthropic, OpenRouter) |

### LiteLLM Proxy

Set `litellm_proxy` to a base URL to route all LLM calls through a [LiteLLM proxy](https://litellm.vercel.app/docs/proxy/proxy_server):

```yaml
litellm_proxy: "http://localhost:4000"
```

This is useful for centralized API key management, rate limiting, and cost tracking across your team.

### Environment Variables

Variables in config are resolved from the environment:

```yaml
api_key: ${OPENAI_API_KEY}
base_url: ${OLLAMA_HOST}
```

---

## Usage Examples

### CLI

```bash
# Ask with custom model
polymind ask "Explain quantum computing" --model ollama/mistral

# Review subtask plan before execution
polymind ask "Build a REST API" --review

# Compare two models
polymind diff "What is the capital of France?" ollama/llama3.2:1b ollama/mistral

# Benchmark specific domains
polymind benchmark ollama/llama3.2:1b --domain code --domain math

# Check system status
polymind status
```

### TUI Keybindings

| Key | Action |
|-----|--------|
| `Ctrl+R` | Review / override subtask plan |
| `Ctrl+M` | Cycle execution strategy |
| `Ctrl+P` | Open profile picker |
| `Ctrl+B` | Run benchmark in background |
| `Ctrl+S` | Save current session |
| `Ctrl+O` | Load a session |
| `?` | Shortcut reference |
| `Q` | Quit |

---

## Comparison with Existing Tools

### How does PolyMind compare?

| Feature | PolyMind | ChatGPT | Ollama CLI | LangChain | OpenRouter |
|---------|----------|---------|------------|-----------|------------|
| **Multi-model orchestration** | ✅ Automatic | ❌ Single model | ❌ Manual | ✅ Requires code | ❌ Single model |
| **Local-first** | ✅ Full offline | ❌ Cloud-only | ✅ | ✅ | ❌ |
| **Model benchmarking** | ✅ Built-in | ❌ | ❌ | ❌ | ❌ |
| **Cost-based ranking** | ✅ accuracy / cost / cost_effective | ❌ | ❌ | ❌ | ❌ |
| **Task decomposition** | ✅ Auto | ❌ Manual | ❌ | ❌ | ❌ |
| **VRAM-aware scheduling** | ✅ DAG + lookahead | ❌ | ❌ | ❌ | ❌ |
| **CLI / TUI / Web** | ✅ All three | ❌ Web-only | ❌ CLI only | ❌ Library only | ❌ API only |
| **Hardware profiling** | ✅ Auto-recommend | ❌ | ❌ | ❌ | ❌ |
| **Provider agnostic** | ✅ Ollama/LM Studio/OpenAI/Anthropic | ❌ | ❌ Ollama only | ✅ | ❌ |
| **LiteLLM proxy support** | ✅ | ❌ | ❌ | ✅ | ❌ |
| **Context budget mgmt** | ✅ | ❌ | ❌ | ❌ | ❌ |

### Are there tools that achieve the same goal with better performance?

**Current gaps vs. production systems:**

| Limitation | PolyMind (v2) | Better alternative |
|------------|--------------|-------------------|
| **Concurrent execution** | Sequential batches within a model | Parallel task execution with async I/O |
| **Distributed execution** | Single machine | Ray-based distributed scheduling |
| **Streaming during execution** | After synthesis | Real-time per-subtask streaming |
| **Caching** | None | Semantic cache for repeated subtasks |
| **Fine-tuning integration** | None | Use rankings to build fine-tuning datasets |
| **Plugin ecosystem** | Skeleton only | Full plugin SDK with pip packages |

**Roadmap items that address these gaps:**
- Parallel execution mode (sprint item)
- Semantic caching for repeated prompts
- Plugin SDK for custom domains
- Pipeline export/import for sharing

### What makes PolyMind unique?

1. **Model-aware scheduling** — No other tool batches LLM subtasks to minimise VRAM swaps. This is critical for local execution where model loading dominates latency.

2. **Data-driven routing** — Rankings are based on actual benchmark scores, not heuristics. The system improves over time as benchmarks are re-run.

3. **Three frontends from one core** — The same library powers CLI, TUI, and Web UIs, all importing from `polymind.core`.

4. **Hardware-first design** — The scheduler adapts to your hardware (RAM, VRAM, CPU cores) rather than assuming a server environment.

---

## Development

### Running Tests

```bash
pytest tests/ -v          # 114 tests
pytest tests/ -v --cov    # With coverage
```

### Test Structure

```
tests/
├── test_analyzer.py       # 9 tests — router prompts, JSON parsing, fallback
├── test_benchmark.py      # 15 tests — task suites, scoring, ranks I/O
├── test_cli.py            # 9 tests — help, status, error handling
├── test_config.py         # 8 tests — YAML load, env vars, serialization
├── test_context.py        # 11 tests — token estimation, budget, truncation
├── test_executor.py       # 3 tests — subtask execution, context injection
├── test_fallback.py       # 6 tests — retry, backoff, fallback chain
├── test_hardware.py       # 6 tests — hardware info, strategy recommendation
├── test_profiles.py       # 7 tests — routing profiles, keep-alive
├── test_providers.py      # 7 tests — model string resolution, kwargs
├── test_scheduler.py      # 14 tests — DAG, topological sort, batching
├── test_synthesizer.py    # 6 tests — message building, streaming
└── test_types.py          # 7 tests — all Pydantic models
```

### Adding a New Provider

Add to `core/providers.py`:

```python
class ProviderType(str, Enum):
    # ... existing providers ...
    my_provider = "my_provider"
```

Update `ProviderInfo.litellm_string` property to handle the new type.

### Adding a Custom Domain

```python
from polymind.core.types import DomainType
from polymind.core.benchmark import BenchmarkTask, BUILTIN_TASKS

# Register tasks
BUILTIN_TASKS[DomainType("medical")] = [
    BenchmarkTask(DomainType("medical"), "Diagnose symptoms", "diagnosis", "llm_judge"),
    # ...
]
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Repository

- **Branch**: `polymind-v2` (separate from `main`)
- **Remote**: `git@github.com:AnkithMall/polymind.git`
- **Clone**: `git clone -b polymind-v2 git@github.com:AnkithMall/polymind.git`

