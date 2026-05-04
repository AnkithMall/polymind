import json
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.config import load_config, PolyMindConfig
from core.analyzer import analyze_prompt
from core.executor import execute_plan
from core.synthesizer import synthesize_stream

# ── App state ─────────────────────────────────────────────────────────────────

_config: Optional[PolyMindConfig] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _config
    try:
        _config = load_config("config.yaml")
        print("✓ PolyMind config loaded")
        print(f"  Execution mode : {_config.execution.mode}")
        print(f"  Analyzer       : {_config.analyzer.model} ({_config.analyzer.provider})")
        print(f"  Synthesizer    : {_config.synthesizer.model} ({_config.synthesizer.provider})")
        print(f"  Specialists    : {list(_config.specialists.keys())}")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
    yield


app = FastAPI(title="PolyMind API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = True
    execution_mode: Optional[str] = None   # "parallel" | "sequential" — overrides config


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "config_loaded": _config is not None,
        "execution_mode": _config.execution.mode if _config else None,
    }


@app.get("/api/config")
async def get_config():
    if not _config:
        raise HTTPException(503, "Config not loaded")
    return _config.model_dump()


@app.get("/api/specialists")
async def get_specialists():
    if not _config:
        raise HTTPException(503, "Config not loaded")
    return {
        "specialists": list(_config.specialists.keys()),
        "execution_mode": _config.execution.mode,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    OpenAI-compatible streaming endpoint.
    Events emitted over SSE:
      { "event": "plan",              "subtasks": [...] }
      { "event": "subtask_results",   "results": [...] }
      { "event": "synthesis_start" }
      { "event": "token",             "content": "..." }
      { "event": "done" }
    """
    if not _config:
        raise HTTPException(503, "Config not loaded. Check your config.yaml.")

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(400, "No user message in request")

    prompt = user_messages[-1].content

    # Allow per-request execution mode override
    cfg = _config
    if request.execution_mode and request.execution_mode in ("parallel", "sequential"):
        from core.config import ExecutionConfig
        import copy
        cfg = cfg.model_copy(deep=True)
        cfg.execution.mode = request.execution_mode

    async def event_stream():
        try:
            # ── Step 1: Analyze ──────────────────────────────────────────────
            print(f"\n[PolyMind] Prompt: {prompt[:80]}...")
            print(f"[PolyMind] Analyzing...")
            plan = await analyze_prompt(prompt, cfg)

            yield f"data: {json.dumps({'event': 'plan', 'subtasks': [s.model_dump() for s in plan.subtasks]})}\n\n"
            print(f"[PolyMind] Plan: {[s.domain for s in plan.subtasks]}")

            # ── Step 2: Execute ──────────────────────────────────────────────
            print(f"[PolyMind] Executing ({cfg.execution.mode} mode)...")
            results = await execute_plan(plan.subtasks, cfg)

            yield f"data: {json.dumps({'event': 'subtask_results', 'results': [r.model_dump() for r in results]})}\n\n"

            # ── Step 3: Synthesize ───────────────────────────────────────────
            print(f"[PolyMind] Synthesizing...")
            yield f"data: {json.dumps({'event': 'synthesis_start'})}\n\n"

            async for token in synthesize_stream(prompt, results, cfg):
                yield f"data: {json.dumps({'event': 'token', 'content': token})}\n\n"

            yield f"data: {json.dumps({'event': 'done'})}\n\n"
            yield "data: [DONE]\n\n"
            print(f"[PolyMind] Done.\n")

        except Exception as e:
            print(f"[PolyMind] Pipeline error: {e}")
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
