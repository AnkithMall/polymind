# PolyMind Demo Guide

Two demos showcasing hardware-aware local LLM management and multi-model pipelines.

**Tested on:** Intel i7-9850H, Quadro T1000 (3.5 GiB VRAM), 31 GiB RAM, Linux
**Date:** 2026-09-19

---

## Demo 1: Multi-Domain Pipeline (Built-in Domains)

> End-to-end: scan hardware → search models → download → optimize → score → run pipeline.

### Step 1: Scan Hardware

```bash
polymind hardware scan
polymind hardware show
```

```
System
------
OS:           Linux
Architecture: x86_64

CPU
---
Model:          Intel(R) Core(TM) i7-9850H CPU @ 2.60GHz
Physical cores: 6
Logical cores:  12

Memory
------
Total: 31.08 GiB

GPUs
----
[1] NVIDIA Quadro T1000
    VRAM:      4.00 GiB
    Available: 3.53 GiB
    Backend:   cuda
    llama.cpp: yes
    Selected:  yes
```

### Step 2: Search for Models

```bash
polymind model search "qwen3 30b" --limit 3
polymind model search "gemma" --limit 3
```

PolyMind queries HuggingFace, lists GGUF variants, and shows VRAM/RAM fit per quant:

```
1. Qwen3-Coder-30B-A3B-Instruct-GGUF
   Repository: unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF

   Quantization   Size       VRAM              RAM        Filename
   ---------------------------------------------------------------------
   IQ1_M          8.97 GiB   fits (offload)    fits        Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_M.gguf
   IQ1_S          8.30 GiB   fits (offload)    fits        Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_S.gguf
   IQ2_XXS        6.69 GiB   fits (offload)    fits        Qwen3-Coder-30B-A3B-Instruct-UD-IQ2_XXS.gguf
```

### Step 3: Download Models

```bash
polymind model download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF \
  Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_M.gguf

polymind model download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF \
  Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_S.gguf

polymind model download HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive \
  Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf
```

### Step 4: Verify Installed Models

```bash
polymind model list
```

```
Installed models
----------------

[2] unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
    File: Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_M.gguf
    Size: 8.97 GiB
    Quantization: IQ1_M
    Status: available

[3] HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive
    File: Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf
    Size: 5.82 GiB
    Quantization: Q6_K
    Status: available

[4] unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
    File: Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_S.gguf
    Size: 8.30 GiB
    Quantization: IQ1_S
    Status: available
```

### Step 5: Optimize Runtime Configs

```bash
polymind runtime optimize --all -v
```

Runs adaptive benchmarks (GPU layer search → context tuning → thread optimization)
for each model. Outputs optimal `gpu_layers`, `threads`, `context_size`, `batch_size`.

```
═══ Optimizing: Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_M.gguf ═══
  Size: 8.97 GiB
  Layers: 48
  Model FP: cd21617a987fa1ba

  Workload: default — General purpose workload
    [1/8]   [gpu=0] scanning...
    [2/8]   [gpu=3] scanning...
    [3/8]   [gpu=4] scanning...
    ...
    [6/6] Selected best configuration

    Result: PASS
    Backend:    cuda
    GPU layers: 11
    Threads:    6
    Context:    2048
    Batch:      256
    Gen speed:  12.8 tok/s
    Peak VRAM:  3220 MB

═══ Optimizing: Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf ═══
  ...
    Result: PASS
    GPU layers: 16
    Gen speed:  9.0 tok/s
    Peak VRAM:  2112 MB
```

**Idempotent:** re-running skips already-optimized models:

```bash
polymind runtime optimize --all
# → ✓ Already optimized (gen=12.8 tok/s, gpu_layers=11). Use --force to re-benchmark.
```

Verify optimized configs:

```bash
polymind runtime show -m 2
polymind runtime show -m 3
polymind runtime show -m 4
```

### Step 6: Compute Confidence Scores

```bash
polymind confidence compute
```

Runs all models against 8 built-in domains (260+ questions, 5 evaluation methods).

```bash
polymind confidence show
```

```
Confidence Scores
======================================================================
Model              Overall Best Domain                Domains
----------------------------------------------------------------------
  4                 33.0% knowledge                     13
  2                 31.9% mathematics                   13
  3                  1.4% coding                        13

Model 2
----------------------------------------
  mathematics                 79.5%
  knowledge                   64.0%
  reasoning                   54.1%
  coding                      43.7%
  conversation                42.6%
  instruction                 40.3%
  writing                     35.1%
  safety                      25.8%

Model 4
----------------------------------------
  knowledge                   73.9%
  coding                      55.8%
  mathematics                 52.9%
  reasoning                   43.9%
  conversation                43.3%
  instruction                 35.2%
  safety                      29.4%
  writing                     25.0%
```

### Step 7: Check Pipeline Readiness

```bash
polymind pipeline suggest
```

Shows the best model per pipeline role, ranked by confidence + runtime feasibility:

```
DECOMPOSER MODEL
------------------------------------------------------------
  1. ID:2 Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_M.gg score=47.2 (runtime=ready)
  2. ID:4 Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_S.gg score=39.5 (runtime=ready)
  3. ID:3 Gemma-4-E4B-Uncensored-HauhauCS-Aggressi score=0.6 (runtime=ready)

GENERATOR MODEL
------------------------------------------------------------
  1. ID:4 Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_S.gg score=33.0 (runtime=ready)
  2. ID:2 Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_M.gg score=31.9 (runtime=ready)
  3. ID:3 Gemma-4-E4B-Uncensored-HauhauCS-Aggressi score=1.4 (runtime=ready)

REGENERATOR MODEL
------------------------------------------------------------
  1. ID:2 Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_M.gg score=35.1 (runtime=ready)
  2. ID:4 Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_S.gg score=25.0 (runtime=ready)
```

```bash
polymind pipeline status
```

```
Pipeline Status

                               Installed Models
┏━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┓
┃ ID ┃ Filename                                           ┃     Size ┃ Quant ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━┩
│ 2  │ Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_M.gguf         │ 8.97 GiB │ IQ1_M │
│ 3  │ Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q6_K_P. │ 5.82 GiB │ Q6_K  │
│ 4  │ Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_S.gguf         │ 8.30 GiB │ IQ1_S │
└────┴────────────────────────────────────────────────────┴──────────┴───────┘

Confidence Scores
  Model 2: overall=31.9% best=mathematics
  Model 3: overall=1.4% best=coding
  Model 4: overall=33.0% best=knowledge
```

### Step 8: Run the Pipeline

**Full pipeline** (decompose → assign → execute → regenerate):

```bash
polymind pipeline run "Explain the difference between quicksort and mergesort, including time complexity and when to use each" -v
```

**Quick mode** (single model, no decomposition):

```bash
polymind pipeline quick "What is 2+2?" -m 3
```

**JSON output:**

```bash
polymind pipeline run "Write a Python function to validate email addresses" --json
```

---

## Demo 2: Custom Domain Pipeline (5 User-Created Domains)

> Create custom test domains → score models on your domain → run domain-aware pipelines.

### Step 1: Seed Demo Domains

```bash
polymind demo seed
```

Creates 5 custom domains with test suites:

```
  ✓ Created domain 'frontend' (3 suites, 11 questions)
  ✓ Created domain 'backend' (3 suites, 10 questions)
  ✓ Created domain 'system-design' (3 suites, 9 questions)
  ✓ Created domain 'architecture' (3 suites, 9 questions)
  ✓ Created domain 'cybersecurity' (3 suites, 9 questions)

  Seeded 5 demo domains.
```

### Step 2: Verify Domains

```bash
polymind domain list
```

```
Predefined Domains
============================================================
  mathematics          Mathematics                    4 suites, 40 questions
  coding               Coding                         4 suites, 40 questions
  reasoning            Reasoning                      3 suites, 30 questions
  knowledge            Knowledge                      3 suites, 30 questions
  writing              Writing                        3 suites, 30 questions
  instruction          Instruction Following          3 suites, 30 questions
  safety               Safety                         3 suites, 30 questions
  conversation         Conversation                   2 suites, 20 questions

Custom Domains
============================================================
  architecture         Architecture                   3 suites, 9 questions
  backend              Backend Development            3 suites, 10 questions
  cybersecurity        Cybersecurity                  3 suites, 9 questions
  frontend             Frontend Development           3 suites, 11 questions
  system-design        System Design                  3 suites, 9 questions

Total: 8 predefined, 5 custom
```

```bash
polymind demo status
```

```
Demo Domain Status

  ✓ architecture: 3 suites, 9 questions, scored 4.4%
  ✓ backend: 3 suites, 10 questions, scored 0.0%
  ✓ cybersecurity: 3 suites, 9 questions, scored 4.4%
  ✓ frontend: 3 suites, 11 questions, scored 8.9%
  ✓ system-design: 3 suites, 9 questions, scored 11.1%
```

### Step 3: Explore a Domain

```bash
polymind domain show frontend
```

```
Domain: frontend
  Name: Frontend Development
  Type: custom
  Description: Frontend web development knowledge and practices

  Suites:
    frontend_basics      Frontend Basics         easy     4 questions
    frontend_frameworks  Frontend Frameworks     medium   4 questions
    frontend_performance Frontend Performance    hard     3 questions
```

```bash
polymind suite show frontend_basics
```

```
Suite: frontend_basics
  Domain:   frontend
  Difficulty: easy

  Questions:
    [1] What does the DOM stand for in web development?
        Expected: Document Object Model
        Keywords: document, object, model
        Eval: keyword_match

    [2] What is the purpose of CSS media queries?
        Expected: To apply different styles based on device characteristics
        Keywords: responsive, design, screen, breakpoint
        Eval: keyword_match
    ...
```

### Step 4: Compute Confidence on Custom Domains

Score models against the custom domains:

```bash
polymind confidence compute -d frontend
polymind confidence compute -d backend
polymind confidence compute -d system-design
polymind confidence compute -d architecture
polymind confidence compute -d cybersecurity
```

Or compute all at once:

```bash
polymind confidence compute
```

View updated scores:

```bash
polymind confidence show
```

```
Confidence Scores
======================================================================
Model              Overall Best Domain                Domains
----------------------------------------------------------------------
  4                 33.0% knowledge                     13
  2                 31.9% mathematics                   13
  3                  1.4% coding                        13

Model 2
----------------------------------------
  mathematics                 79.5%
  knowledge                   64.0%
  reasoning                   54.1%
  coding                      43.7%
  ...
  system-design               11.1%     ← custom domain
  frontend                     8.9%     ← custom domain
  architecture                 4.4%     ← custom domain
  cybersecurity                4.4%     ← custom domain
  backend                      0.0%     ← custom domain

Model 4
----------------------------------------
  knowledge                   73.9%
  coding                      55.8%
  ...
  cybersecurity               21.7%     ← custom domain
  architecture                17.2%     ← custom domain
  frontend                    15.6%     ← custom domain
  system-design               15.6%     ← custom domain
  backend                      0.0%     ← custom domain
```

### Step 5: Check Pipeline with Custom Domains

```bash
polymind pipeline suggest
```

The pipeline selector now includes custom domain scores in its role assignments.
Models that score higher on your custom domains get preferred for relevant tasks.

### Step 6: Run Domain-Aware Pipeline

The decomposer automatically detects custom domain terms in prompts:

```bash
polymind pipeline run "Design a REST API with authentication, rate limiting, and proper error handling" -v
```

Verbose output shows domain detection:

```
  Domain detection: backend (confidence: 0.85)
  Domain detection: system-design (confidence: 0.72)

  Task assignments:
    Task 1: backend        → Model 4 (score=33.0, runtime=ready)
    Task 2: system-design  → Model 2 (score=47.2, runtime=ready)
    Task 3: coding         → Model 4 (score=55.8, runtime=ready)
```

More examples:

```bash
# Frontend domain
polymind pipeline run "Build a React component with lazy loading and error boundaries" -v

# Cybersecurity domain
polymind pipeline run "Review this nginx config for security vulnerabilities" -v

# Architecture domain
polymind pipeline run "Design a microservices architecture for an e-commerce platform" -v
```

### Step 7: Interactive Chat

```bash
polymind runtime run -m 2
```

```
Loading model: Qwen3-Coder-30B-A3B-Instruct-UD-IQ1_M.gguf
GPU layers: 11
Context: 2048

You: What is the difference between TCP and UDP?
Assistant: TCP (Transmission Control Protocol) and UDP (User Datagram Protocol)...
```

---

## Quick Reference

### Model Management

| Command | Description |
|---------|-------------|
| `polymind model search <query>` | Search HuggingFace for GGUF models |
| `polymind model download <repo> <file>` | Download a specific GGUF file |
| `polymind model list` | List installed models |
| `polymind model delete <id>` | Remove a model |

### Hardware & Runtime

| Command | Description |
|---------|-------------|
| `polymind hardware scan` | Detect CPU, RAM, GPUs |
| `polymind hardware show` | Display hardware profile |
| `polymind runtime optimize -m <id>` | Benchmark optimal settings |
| `polymind runtime optimize --all` | Optimize all models |
| `polymind runtime show -m <id>` | Show optimized config |
| `polymind runtime validate -m <id>` | Check profile validity |
| `polymind runtime run -m <id>` | Interactive chat |

### Domains & Confidence

| Command | Description |
|---------|-------------|
| `polymind domain list` | List all domains |
| `polymind domain show <id>` | Show domain details |
| `polymind domain create <id>` | Create custom domain |
| `polymind confidence compute` | Score all models |
| `polymind confidence compute -d <id>` | Score specific domain |
| `polymind confidence show` | Display scores |

### Pipeline

| Command | Description |
|---------|-------------|
| `polymind pipeline suggest` | Best model per role |
| `polymind pipeline status` | Pipeline readiness |
| `polymind pipeline run "<prompt>"` | Full multi-model pipeline |
| `polymind pipeline quick "<prompt>"` | Single-model quick mode |

### Demo

| Command | Description |
|---------|-------------|
| `polymind demo seed` | Create 5 demo custom domains |
| `polymind demo seed --reset` | Reset and re-seed |
| `polymind demo status` | Show demo domain status |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Layer                             │
│  model · hardware · runtime · pipeline · confidence · demo   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     Core Engine                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │ Hardware │ │  Model   │ │ Runtime  │ │  Confidence  │   │
│  │ Scanner  │ │ Registry │ │ Optimizer│ │   Scorer     │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Pipeline Orchestrator                    │   │
│  │  Decomposer → Selector → Executor → Regenerator      │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   llama.cpp Backend                          │
│         GPU offloading · CUDA · Metal · Vulkan               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
runtime.yaml          ← optimizer writes, pipeline reads
confidence/           ← scorer writes, selector reads
domains/              ← demo seed / user creates
models/               ← download writes, registry reads
hardware.yaml         ← scanner writes, optimizer reads
```
