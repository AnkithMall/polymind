# QA Backlog — polymind v0.1.0

**Tested on:** Intel i7-9850H, Quadro T1000 (3.5GB VRAM), 31GB RAM, Linux  
**Date:** 2026-09-12  
**102 unit tests passing**

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Tested — works correctly |
| ⚠️ | Partial — works but has issues |
| 🐛 | Bug — known defect, needs fix |
| 💤 | Stub — registered but empty, no implementation |
| ⏳ | Untested — not manually verified |

---

## Full Command Table

| # | Command | Subcommand | Implemented | Tested | Bug | Test File | Explanation | Usage Example | Remarks |
|---|---------|------------|-------------|--------|-----|-----------|-------------|---------------|---------|
| 1 | `polymind` | `--help` | ✅ | ✅ | — | `app.py:29` | Top-level help, lists all 13 subcommands | `polymind --help` | Shows all available commands |
| 2 | `model` | `search <query>` | ✅ | ✅ | — | `commands/model.py` | Queries HuggingFace for GGUF models, ranks by VRAM/RAM compatibility, shows quant variants | `polymind model search "tiny llama" --limit 5` | Flags: `--limit`, `--author`, `--quant`, `--max-size`, `--min-size`, `--sort` |
| 3 | `model` | `download <repo> <file>` | ✅ | ⏳ | — | `commands/model.py` | Downloads specific GGUF file from HuggingFace repo, registers in local model registry | `polymind model download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | Flags: `--output -o <path>` for custom download dir |
| 4 | `model` | `list` | ✅ | ✅ | — | `commands/model.py` | Lists installed models with ID, filename, size, quantization, path, status | `polymind model list` | No flags. Shows model IDs for use with `-m` in other commands |
| 5 | `model` | `delete <model>` | ✅ | ✅ | — | `commands/model.py` | Removes model file from disk and deletes registry entry. Matches by ID, filename, or repo ID | `polymind model delete 3` | Flags: `--force -f` skip confirmation. Error: "Model not found: 999" |
| 6 | `model` | `scan` | ✅ | ✅ | — | `commands/model.py` | Scans model directories for GGUF files, fixes registry inconsistencies | `polymind model scan` | Flags: `--all` scan all known locations, `--interactive` prompt for each found file |
| 7 | `model` | `migrate` | ✅ | ✅ | — | `commands/model.py` | Moves GGUF files from legacy directories (polymind/models/, ~/.cache/polymind/models/) to .polymind/models/ | `polymind model migrate` | Flags: `--source -s <path>` custom source dir, `--force -f` skip prompts |
| 8 | `hardware` | `scan` | ✅ | ✅ | — | `commands/hardware.py` | Detects CPU, RAM, GPUs (NVIDIA/Intel), llama.cpp backends, writes hardware.yaml | `polymind hardware scan` | No flags. Writes .polymind/hardware.yaml |
| 9 | `hardware` | `show` | ✅ | ✅ | — | `commands/hardware.py` | Displays formatted hardware profile: CPU model/cores, RAM, GPUs with VRAM, backend info | `polymind hardware show` | No flags. Shows selected GPUs and llama.cpp availability |
| 10 | `hardware` | `validate` | ✅ | ✅ | — | `commands/hardware.py` | Validates hardware profile consistency, checks selected GPUs are usable | `polymind hardware validate` | No flags. Reports "profile is valid" or lists issues |
| 11 | `runtime` | `run -m <model>` | ✅ | ⚠️ | — | `commands/runtime.py` | Loads model using llama.cpp with runtime.yaml settings, starts interactive chat REPL | `polymind runtime run -m 3` | Required: `--model -m <id/filename>`. Loads optimized config from runtime.yaml |
| 12 | `runtime` | `optimize -m <model>` | ✅ | ✅ | BUG-001 fixed | `commands/runtime.py` | 4-phase adaptive benchmark: baseline CPU → GPU layer binary search → thread tuning → context/batch sweep. Writes results to runtime.yaml | `polymind runtime optimize -m 3` | Flags: `--model -m <id>`, `--all` benchmark all models, `--verbose -v` show per-config results |
| 13 | `runtime` | `optimize --all` | ✅ | ✅ | — | `commands/runtime.py` | Benchmarks all installed models sequentially, writes all results to runtime.yaml | `polymind runtime optimize --all -v` | Flags: `--verbose -v` show detailed per-configuration results during optimization |
| 14 | `pipeline` | `run <prompt>` | ✅ | ✅ | — | `commands/pipeline.py` | Full pipeline: complexity analysis → decompose into subtasks → assign best model per task → execute → regenerate final response | `polymind pipeline run "Build a REST API with auth and tests" -v` | Flags: `--decomposer <id>`, `--generator <id>`, `--regenerator <id>`, `--judge <id>`, `--no-regenerate`, `--verbose -v`, `--json` |
| 15 | `pipeline` | `suggest` | ✅ | ✅ | — | `commands/pipeline.py` | Shows best model per pipeline role (decomposer, generator, regenerator, judge) ranked by confidence scores | `polymind pipeline suggest` | No flags. Uses confidence scores for automatic role assignment |
| 16 | `pipeline` | `status` | ✅ | ✅ | — | `commands/pipeline.py` | Shows installed models, confidence scores, pipeline readiness status | `polymind pipeline status` | No flags. Shows model table + per-model confidence scores |
| 17 | `pipeline` | `quick <prompt>` | ✅ | ✅ | BUG-002 fixed | `commands/pipeline.py` | Direct single-model inference — skips decomposition and regeneration, best for simple prompts | `polymind pipeline quick "What is 2+2?" -m 3` | Flags: `--model -m <id>` (default: largest installed model). Uses runtime.yaml config |
| 18 | `pipeline` | `run --json` | ✅ | ⏳ | — | `commands/pipeline.py` | Outputs full pipeline result as JSON (tasks, assignments, scores, timing) | `polymind pipeline run "What is quicksort?" --json` | Flag: `--json` outputs structured JSON instead of formatted text |
| 19 | `pipeline` | `run --no-regenerate` | ✅ | ⏳ | — | `commands/pipeline.py` | Skips regeneration step — returns raw task results without synthesis | `polymind pipeline run "Write a sorting function" --no-regenerate` | Flag: `--no-regenerate` useful for debugging individual task outputs |
| 20 | `confidence` | `compute -m <model>` | ✅ | ⏳ | — | `commands/confidence.py` | Runs model against test suites (260 questions, 5 eval methods: exact_match, keyword_match, code_execution, llm_judge, hybrid), computes accuracy per domain | `polymind confidence compute -m 2 --domain coding` | Flags: `--model -m <id>` (all models if empty), `--domain -d <id>` (all domains if empty), `--json` |
| 21 | `confidence` | `show` | ✅ | ✅ | — | `commands/confidence.py` | Shows computed confidence scores per model per domain with overall best domain | `polymind confidence show` | No flags. Shows overall %, best domain, per-domain breakdown |
| 22 | `confidence` | `domains` | ✅ | ✅ | — | `commands/confidence.py` | Lists all 8 scoring domains with suite and question counts | `polymind confidence domains` | No flags. 8 domains: mathematics, coding, reasoning, knowledge, writing, instruction, safety, conversation |
| 23 | `confidence` | `reset -m <model>` | ✅ | ⏳ | — | `commands/confidence.py` | Deletes computed confidence scores for a specific model or all models | `polymind confidence reset -m 2 --force` | Flags: `--model -m <id>` (all if empty), `--force -f` skip confirmation |
| 24 | `domain` | `list` | ✅ | ✅ | — | `commands/domain.py` | Lists predefined (8) + custom domains with suite/question counts | `polymind domain list` | No flags. Shows type (predefined/custom) and counts |
| 25 | `domain` | `show <id>` | ✅ | ✅ | — | `commands/domain.py` | Shows domain details: description, type, all suites with difficulty and question counts | `polymind domain show mathematics` | Positional: `<domain_id>`. Shows suites: math_arithmetic, math_algebra, math_word_problems, math_statistics |
| 26 | `domain` | `create <id>` | ✅ | ✅ | — | `commands/domain.py` | Creates empty custom domain with unique ID, name, and description | `polymind domain create my_domain --name "My Domain" --description "Custom tests"` | Required: `--name`. Optional: `--description`. Saves to .polymind/domains/<id>.yaml |
| 27 | `domain` | `delete <id>` | ✅ | ✅ | — | `commands/domain.py` | Deletes custom domain and its files. Refuses to delete predefined domains | `polymind domain delete my_domain --force` | Flags: `--force -f` skip confirmation. Error: "Cannot delete predefined domain: mathematics" |
| 28 | `domain` | `export <id>` | ✅ | ✅ | — | `commands/domain.py` | Exports domain to YAML file in current working directory | `polymind domain export my_domain` | Positional: `<domain_id>`. Writes <domain_id>.yaml to CWD |
| 29 | `domain` | `import <file>` | ✅ | ✅ | — | `commands/domain.py` | Imports domain from YAML file, creates domain + suites + questions | `polymind domain import my_domain.yaml` | Positional: `<file_path>`. Error: "File not found: nonexistent.yaml" |
| 30 | `suite` | `list [domain]` | ✅ | ✅ | — | `commands/suite.py` | Lists all test suites across all domains, or within a specific domain | `polymind suite list mathematics` | Optional positional: `<domain_id>` (filters by domain). Shows suite ID, name, difficulty, question count |
| 31 | `suite` | `show <id>` | ✅ | ✅ | — | `commands/suite.py` | Shows suite details: name, domain, difficulty, all questions with prompt, expected answer, keywords, eval method | `polymind suite show math_arithmetic` | Positional: `<suite_id>`. Shows 10 questions with [hybrid] eval, expected answers, keywords |
| 32 | `suite` | `create` | ✅ | ✅ | — | `commands/suite.py` | Creates empty test suite in a custom domain | `polymind suite create my_domain --suite-id my_suite --name "My Suite" --description "Test" --difficulty easy` | Required: `--suite-id`, `--name`. Optional: `--description`, `--difficulty` (easy/medium/hard/expert) |
| 33 | `suite` | `add-question` | ✅ | ✅ | — | `commands/suite.py` | Adds question with prompt, expected answer, evaluation method, optional keywords | `polymind suite add-question my_domain my_suite --question-id q1 --prompt "What is 2+2?" --expected "4" --evaluation exact_match` | Required: `--question-id`, `--prompt`, `--expected`. Optional: `--evaluation` (exact_match/keyword_match/code_execution/llm_judge/hybrid), `--keywords` |
| 34 | `suite` | `remove-question` | ✅ | ✅ | — | `commands/suite.py` | Removes specific question by ID from a suite | `polymind suite remove-question my_domain my_suite q1 --force` | Positional: `<domain_id> <suite_id> <question_id>`. Flags: `--force -f` |
| 35 | `suite` | `delete` | ✅ | ✅ | — | `commands/suite.py` | Deletes suite and all its questions. Refuses predefined domain suites | `polymind suite delete my_domain my_suite --force` | Positional: `<domain_id> <suite_id>`. Flags: `--force -f`. Error: "Cannot delete suites from predefined domains" |
| 36 | `config` | `show` | ✅ | ✅ | — | `commands/config.py` | Shows all config: env vars, resolved paths, runtime configs, artifact files, model count | `polymind config show` | No flags. Shows ✓/✗ status for each path, model sizes, file sizes |
| 37 | `config` | `get <key>` | ✅ | ✅ | — | `commands/config.py` | Gets config value by key (env vars or runtime fields) | `polymind config get runtime.2.gpu_layers` | Positional: `<key>`. Keys: `POLYMIND_ARTIFACT_DIR`, `POLYMIND_MODEL_DIR`, `HF_TOKEN`, `runtime.<id>.gpu_layers/threads/context_size/batch_size` |
| 38 | `config` | `set <key> <value>` | ✅ | ⚠️ | BUG-003 | `commands/config.py` | Sets config value — env vars print shell instructions, runtime fields write to runtime.yaml | `polymind config set runtime.1.gpu_layers 35` | Positional: `<key> <value>`. **Warning:** BUG-003 may lose other models' configs |
| 39 | `config` | `edit` | ✅ | ⏳ | — | `commands/config.py` | Opens config file in $EDITOR for manual editing | `polymind config edit` | Flags: `--file -f <name>` (runtime/hardware/registry, default: runtime). Cannot test headless |
| 40 | `config` | `logging` | ✅ | ✅ | — | `commands/config.py` | Shows logging config: enabled state, log directory, rotation limits, retention policies, existing log files with sizes | `polymind config logging` | Flags: `--raw -r` show raw YAML |
| 41 | `config` | `logging-set <key> <val>` | ✅ | ✅ | — | `commands/config.py` | Updates single logging setting in polymind.yaml | `polymind config logging-set max_bytes 10485760` | Positional: `<key> <value>`. Keys: `enabled`, `max_bytes`, `backup_count`, `max_age_days`, `max_total_files`, `max_total_size_mb` |
| 42 | `config` | `pipeline` | ✅ | ✅ | — | `commands/config.py` | Shows pipeline config: concurrency, timeout, retries, min task score | `polymind config pipeline` | Flags: `--raw -r` show raw YAML |
| 43 | `config` | `pipeline-set <key> <val>` | ✅ | ✅ | — | `commands/config.py` | Updates single pipeline setting in polymind.yaml | `polymind config pipeline-set timeout_seconds 300` | Positional: `<key> <value>`. Keys: `max_concurrent`, `timeout_seconds`, `max_retries`, `min_task_score` |
| 44 | `run` | *(no subcommands)* | 💤 | 💤 | — | `app.py` | Stub — registered as "Run commands (alias)" but has no subcommands | `polymind run` | **Action:** Remove or implement. Currently errors "Missing command" |
| 45 | `capability` | *(no subcommands)* | 💤 | 💤 | — | `app.py` | Stub — registered as "Hardware capability detection" but has no subcommands | `polymind capability` | **Action:** Remove or implement. Currently errors "Missing command" |
| 46 | `category` | *(no subcommands)* | 💤 | 💤 | — | `app.py` | Stub — registered as "Model categories" but has no subcommands | `polymind category` | **Action:** Remove or implement. Currently errors "Missing command" |
| 47 | `doctor` | *(no subcommands)* | 💤 | 💤 | — | `app.py` | Stub — registered as "Diagnostics and health checks" but has no subcommands | `polymind doctor` | **Action:** Remove or implement. Currently errors "Missing command" |
| 48 | `tui` | *(no subcommands)* | 💤 | 💤 | — | `app.py` | Stub — registered as "Launch the text user interface" but has no subcommands. TUI source exists at `client/tui/` | `polymind tui` | **Action:** Remove or implement. TUI module exists but command is empty shell |

---

## Bugs

| ID | Command | File:Line | Severity | Description | Fix | Status |
|----|---------|-----------|----------|-------------|-----|--------|
| BUG-001 | `runtime optimize` | `benchmark.py:167` | Critical | `create_completion()` called without `prompt` arg. llama-cpp-python 0.3.35 requires it. **All benchmarks failed silently.** | Added `prompt=BENCHMARK_PROMPT` | ✅ Fixed |
| BUG-002 | `pipeline quick` | `pipeline.py:449` | Critical | Used `default_runtime_config()` (gpu=32, ctx=16384) ignoring runtime.yaml optimized values. **Models crashed with OOM.** | Changed to `load_runtime_config()` first, fallback to defaults | ✅ Fixed |
| BUG-003 | `config set` | `config.py:346` | Medium | After `config set runtime.2.gpu_layers 16`, models 3/4 disappeared from runtime.yaml. `write_runtime_config()` may not preserve all models correctly. | Needs investigation — suspected YAML serialization or key type issue | ⚠️ Open |

---

## Coverage Summary

| Category | Total | ✅ Tested | ⚠️ Partial | ⏳ Untested | 💤 Stub | Coverage |
|----------|-------|-----------|------------|------------|---------|----------|
| `model` | 6 | 5 | 0 | 1 | 0 | 83% |
| `hardware` | 3 | 3 | 0 | 0 | 0 | 100% |
| `runtime` | 3 | 2 | 1 | 0 | 0 | 83% |
| `pipeline` | 6 | 4 | 0 | 2 | 0 | 67% |
| `confidence` | 4 | 2 | 0 | 2 | 0 | 50% |
| `domain` | 6 | 6 | 0 | 0 | 0 | 100% |
| `suite` | 6 | 6 | 0 | 0 | 0 | 100% |
| `config` | 8 | 6 | 1 | 1 | 0 | 81% |
| `stubs` | 5 | 0 | 0 | 0 | 5 | 0% |
| **Total** | **47** | **34** | **2** | **6** | **5** | **76%** |
