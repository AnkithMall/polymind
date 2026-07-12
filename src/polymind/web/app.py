from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from polymind.core import (
    Config,
    ModelConfig,
    detect_local_providers,
    analyze_prompt,
    build_schedule,
    describe_schedule,
    execute_subtask,
    load_ranks,
    resolve_model_string,
    save_ranks,
    run_benchmark,
    setup_logging,
    synthesize,
    SubtaskResult,
    PipelineResult,
    DomainType,
    ALL_DOMAINS,
    ExecutionStrategy,
    RankingMode,
    ModelSource,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("~/.polymind/config.yaml").expanduser()
RANKS_PATH = Path("~/.polymind/ranks.yaml").expanduser()

app = FastAPI(title="PolyMind", version="0.1.0")


@app.on_event("startup")
async def startup():
    config = Config.from_yaml(CONFIG_PATH)
    if config.verbose:
        setup_logging(logging.DEBUG)


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


@app.get("/api/config")
async def get_config():
    config = Config.from_yaml(CONFIG_PATH)
    return {
        "models": [m.model_dump() for m in config.models],
        "router_model": config.router_model,
        "synthesizer_model": config.synthesizer_model,
        "judge_model": config.judge_model,
        "strategy": config.scheduler.strategy.value,
        "verbose": config.verbose,
        "profile": config.profile,
        "keep_alive": config.keep_alive,
        "litellm_proxy": config.litellm_proxy,
        "ranking_mode": config.ranking_mode.value,
        "model_source": config.model_source.value,
    }


@app.post("/api/config")
async def save_config(request: Request):
    body = await request.json()
    config = Config.from_yaml(CONFIG_PATH)
    if "models" in body:
        config.models = [ModelConfig(**m) for m in body["models"]]
    if "router_model" in body:
        config.router_model = body["router_model"]
    if "synthesizer_model" in body:
        config.synthesizer_model = body["synthesizer_model"]
    if "judge_model" in body:
        config.judge_model = body["judge_model"]
    if "strategy" in body:
        config.scheduler.strategy = ExecutionStrategy(body["strategy"])
    if "ranking_mode" in body:
        config.ranking_mode = RankingMode(body["ranking_mode"])
    if "model_source" in body:
        config.model_source = ModelSource(body["model_source"])
    if "verbose" in body:
        config.verbose = bool(body["verbose"])
    config.to_yaml(CONFIG_PATH)
    if config.verbose:
        setup_logging(logging.DEBUG)
    return {"status": "ok"}


@app.get("/api/providers/detect")
async def detect_providers():
    detected = detect_local_providers()
    return {"providers": detected}


@app.post("/api/config/providers")
async def add_provider(request: Request):
    body = await request.json()
    config = Config.from_yaml(CONFIG_PATH)
    mc = ModelConfig(**body)
    config.models.append(mc)
    config.to_yaml(CONFIG_PATH)
    return {"status": "ok", "model": mc.model_dump()}


@app.post("/api/ask")
async def ask_endpoint(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    if not prompt:
        return JSONResponse({"error": "No prompt provided"}, status_code=400)

    config = Config.from_yaml(CONFIG_PATH)
    router = config.router_model
    synthesizer_m = config.synthesizer_model or router

    async def event_generator():
        try:
            yield _sse("status", "Analyzing prompt...")
            plan = await analyze_prompt(prompt, router_model=router)

            plan_data = [
                {
                    "id": s.id,
                    "domain": s.domain.value,
                    "prompt": s.prompt,
                    "depends_on": s.depends_on,
                }
                for s in plan.subtasks
            ]
            yield _sse("plan", json.dumps(plan_data))

            rank_store = load_ranks(RANKS_PATH)
            schedule = build_schedule(
                plan,
                rank_store=rank_store,
                strategy=config.scheduler.strategy,
                ranking_mode=config.ranking_mode,
                model_source=config.model_source,
            )

            schedule_data = [
                {"model": b.model, "subtask_ids": b.subtask_ids}
                for b in schedule.batches
            ]
            schedule_visual = describe_schedule(plan, schedule)
            yield _sse("schedule", json.dumps({"batches": schedule_data, "visual": schedule_visual}))

            subtask_results: list[SubtaskResult] = []
            prior_outputs: dict[str, SubtaskResult] = {}
            total = sum(len(b.subtask_ids) for b in schedule.batches)
            completed = 0

            for batch in schedule.batches:
                yield _sse("status", f"Loading model: {batch.model}")
                for sid in batch.subtask_ids:
                    subtask = next(s for s in plan.subtasks if s.id == sid)
                    info = resolve_model_string(batch.model)
                    result = await execute_subtask(
                        subtask,
                        model_ref=batch.model,
                        provider_info=info,
                        keep_alive=config.keep_alive,
                    )
                    subtask_results.append(result)
                    prior_outputs[sid] = result
                    completed += 1
                    yield _sse(
                        "subtask_complete",
                        json.dumps(
                            {
                                "id": result.subtask_id,
                                "model": result.model,
                                "status": "error" if result.error else "ok",
                                "completed": completed,
                                "total": total,
                            }
                        ),
                    )

            yield _sse("status", "Synthesizing response...")
            pipeline_result = PipelineResult(
                original_prompt=prompt,
                subtask_results=subtask_results,
                schedule=schedule,
            )
            final = await synthesize(pipeline_result, synthesizer_model=synthesizer_m)
            yield _sse(
                "result",
                json.dumps(
                    {
                        "synthesis": final.synthesis or "[No synthesis produced]",
                        "subtasks": [
                            {
                                "id": r.subtask_id,
                                "model": r.model,
                                "output": r.output[:500],
                                "error": r.error,
                            }
                            for r in subtask_results
                        ],
                    }
                ),
            )
        except Exception as e:
            logger.exception("Pipeline error")
            yield _sse("error", str(e))

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/benchmark")
async def benchmark_endpoint(request: Request):
    body = await request.json()
    models = body.get("models", [])
    domains_str = body.get("domains", [])
    domains = [DomainType(d) for d in domains_str] if domains_str else ALL_DOMAINS

    config = Config.from_yaml(CONFIG_PATH)
    judge = config.judge_model

    async def event_generator():
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        try:
            def _progress(pct: float, msg: str):
                queue.put_nowait(
                    _sse("progress", json.dumps({"pct": round(pct * 100, 1), "msg": msg}))
                )

            async def _run():
                try:
                    store, details = await run_benchmark(
                        models=models,
                        domains=domains,
                        judge_model=judge,
                        progress_callback=_progress,
                    )
                    return store, details
                finally:
                    queue.put_nowait(None)

            task = asyncio.create_task(_run())
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item

            store, _details = task.result()
            save_ranks(store, RANKS_PATH)
            entries = [
                {
                    "model": e.model,
                    "domain": e.domain.value,
                    "score": e.score,
                    "latency_ms": e.latency_ms,
                }
                for e in store.entries
            ]
            yield _sse("result", json.dumps(entries))
        except Exception as e:
            yield _sse("error", str(e))

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/ranks")
async def get_ranks():
    store = load_ranks(RANKS_PATH)
    entries = [
        {
            "model": e.model,
            "domain": e.domain.value,
            "score": e.score,
            "latency_ms": e.latency_ms,
        }
        for e in store.entries
    ]
    stale = store.is_stale()
    return {"entries": entries, "stale": stale}


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (Path(__file__).parent / "static" / "index.html").read_text()
    return HTMLResponse(html)


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
