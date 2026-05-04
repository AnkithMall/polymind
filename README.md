# 🧠 PolyMind

> Multi-specialist LLM orchestrator — one prompt, many experts, one answer.

PolyMind decomposes your prompt into typed subtasks, routes each to the best model for that domain (coding, math, creative, research…), then synthesizes a single coherent response.

## Quick Start (Linux, Docker Compose)

```bash
# 1. Clone
git clone https://github.com/yourusername/polymind.git
cd polymind

# 2. Configure environment
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# 3. Start everything
docker compose up -d

# 4. Pull the Ollama models you configured
docker exec -it polymind-ollama-1 ollama pull llama3.2:3b
docker exec -it polymind-ollama-1 ollama pull deepseek-coder:6.7b
docker exec -it polymind-ollama-1 ollama pull qwen2.5-math:7b

# 5. Open the UI
# → http://localhost:3000
```

## Architecture

```
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
