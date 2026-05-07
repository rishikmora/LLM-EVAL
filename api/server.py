"""
api/server.py — FastAPI Production Backend
Run: uvicorn api.server:app --host 0.0.0.0 --port 8080 --reload
"""
from __future__ import annotations
import asyncio, json, os, sys, time
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="LLM Eval Platform", version="2.0.0", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

_ws_clients: list[WebSocket] = []

async def _broadcast_ws(message: dict):
    dead = []
    for ws in _ws_clients:
        try: await ws.send_json(message)
        except Exception: dead.append(ws)
    for ws in dead: _ws_clients.remove(ws)

# ── Models ──────────────────────────────────────────────────────────────────
class JobRequest(BaseModel):
    suite: str = "adversarial"; n_per_category: int = 5; provider: str = "gemini"; n_shards: int = 1

class BenchmarkRequest(BaseModel):
    benchmarks: list[str] = ["mmlu","truthfulqa","harmbench"]; n_questions: int = 20

class RAGRequest(BaseModel):
    backend: str = "faiss"; n_samples: int = 3

class PromptRequest(BaseModel):
    prompt: str; provider: str = "gemini"

# ── Auth ────────────────────────────────────────────────────────────────────
def get_user(x_api_key: str = Header(default="")):
    if os.environ.get("API_AUTH_REQUIRED","false").lower() != "true": return None
    from security.layer import rbac
    user = rbac.authenticate(x_api_key)
    if not user: raise HTTPException(status_code=403, detail="Invalid API key")
    return user

# ── Health ──────────────────────────────────────────────────────────────────
@app.get("/")
def root(): return {"name":"LLM Eval Platform","version":"2.0.0","status":"ok","docs":"/docs","timestamp":datetime.utcnow().isoformat()}

@app.get("/health")
def health():
    from workers.distributed import JobStore
    store = JobStore()
    return {"status":"ok","redis":store.is_redis_available(),"timestamp":datetime.utcnow().isoformat()}

# ── Metrics (Prometheus) ────────────────────────────────────────────────────
@app.get("/metrics", response_class=PlainTextResponse)
def prom_metrics():
    from observability.telemetry import telemetry
    return telemetry.prometheus_exposition()

@app.get("/metrics/snapshot")
def metrics_snap():
    from observability.telemetry import telemetry
    return telemetry.metrics_snapshot()

@app.get("/metrics/drift")
def drift():
    from observability.telemetry import telemetry
    return telemetry.drift_report()

# ── Jobs ────────────────────────────────────────────────────────────────────
@app.post("/jobs")
async def submit_job(req: JobRequest, bg: BackgroundTasks, user=Depends(get_user)):
    from workers.distributed import EvalScheduler
    from observability.telemetry import telemetry, metrics
    metrics.inc("api_job_submissions")
    scheduler = EvalScheduler(distributed=False)
    job_id = scheduler.submit(req.suite, req.n_per_category, req.n_shards, req.provider)
    if user:
        from security.layer import audit
        audit.log(user.user_id,"job_submitted",resource=job_id,details={"suite":req.suite})
    bg.add_task(_broadcast_ws,{"event":"job_submitted","job_id":job_id,"suite":req.suite})
    return {"job_id":job_id,"status":"submitted","suite":req.suite}

@app.get("/jobs")
def list_jobs(user=Depends(get_user)):
    from workers.distributed import JobStore
    return {"jobs":JobStore().list_jobs(50)}

@app.get("/jobs/{job_id}")
def get_job(job_id:str, user=Depends(get_user)):
    from workers.distributed import EvalScheduler
    s = EvalScheduler(distributed=False).status(job_id)
    if not s: raise HTTPException(404,"Job not found")
    return s

# ── Results ─────────────────────────────────────────────────────────────────
@app.get("/results")
def list_results(user=Depends(get_user)):
    import yaml; cfg=yaml.safe_load(open(ROOT/"config/eval_config.yaml"))
    db=Path(cfg["database"]["path"])
    if not db.exists(): return {"runs":[]}
    from modules.tracker import list_runs
    return {"runs":list_runs(db)}

@app.get("/results/{run_id}")
def get_stats(run_id:str, user=Depends(get_user)):
    import yaml; cfg=yaml.safe_load(open(ROOT/"config/eval_config.yaml"))
    db=Path(cfg["database"]["path"])
    from modules.tracker import load_run_results,compute_run_stats,check_thresholds
    results=load_run_results(run_id,db)
    if not results: raise HTTPException(404,"Run not found")
    stats=compute_run_stats(results); violations=check_thresholds(stats,cfg)
    return {"run_id":run_id,"n":len(results),"stats":stats,"violations":violations}

# ── Benchmarks ──────────────────────────────────────────────────────────────
@app.post("/benchmarks/run")
async def run_bench(req: BenchmarkRequest, bg: BackgroundTasks, user=Depends(get_user)):
    async def _run():
        import yaml; cfg=yaml.safe_load(open(ROOT/"config/eval_config.yaml"))
        from modules.evaluator import GeminiClient
        from benchmarks.suite import HELMOrchestrator
        client=GeminiClient(os.environ.get("GOOGLE_API_KEY",""),cfg)
        r=await HELMOrchestrator().run_all(client,req.n_questions,cfg["model"]["target"],req.benchmarks)
        await _broadcast_ws({"event":"benchmark_complete","helm_score":r.get("helm_score")})
    bg.add_task(_run)
    return {"status":"started","suites":req.benchmarks}

@app.get("/benchmarks/history/{benchmark}")
def bench_history(benchmark:str, user=Depends(get_user)):
    from benchmarks.suite import get_benchmark_history,compare_models_on_benchmark
    return {"benchmark":benchmark,"history":get_benchmark_history(benchmark),"comparison":compare_models_on_benchmark(benchmark)}

# ── RAG ─────────────────────────────────────────────────────────────────────
@app.post("/rag/evaluate")
async def rag_eval(req: RAGRequest, user=Depends(get_user)):
    from rag.evaluator import RAGEvaluator,DEMO_KNOWLEDGE_BASE,DEMO_SAMPLES
    ev=RAGEvaluator(req.backend); ev.add_knowledge_base(DEMO_KNOWLEDGE_BASE)
    return await ev.evaluate(DEMO_SAMPLES[:req.n_samples])

# ── Plugins ─────────────────────────────────────────────────────────────────
@app.get("/plugins")
def plugins(user=Depends(get_user)):
    from plugins.registry import plugin_registry
    return plugin_registry.list_plugins()

@app.post("/plugins/attacks/{name}")
def attack_prompts(name:str, n:int=10, user=Depends(get_user)):
    from plugins.registry import plugin_registry
    try: return {"plugin":name,"prompts":plugin_registry.generate_attacks(name,n)}
    except ValueError as e: raise HTTPException(404,str(e))

# ── Single prompt ────────────────────────────────────────────────────────────
@app.post("/prompt")
async def single_prompt(req: PromptRequest, user=Depends(get_user)):
    from security.layer import sanitize_prompt, rate_limiter
    uid = user.user_id if user else "anon"
    ok, retry = rate_limiter.is_allowed(uid)
    if not ok: raise HTTPException(429,f"Rate limited. Retry in {retry}s")
    sanitized, flags = sanitize_prompt(req.prompt)
    import yaml; cfg=yaml.safe_load(open(ROOT/"config/eval_config.yaml"))
    from modules.evaluator import GeminiClient
    from observability.telemetry import telemetry
    client=GeminiClient(os.environ.get("GOOGLE_API_KEY",""),cfg)
    span=telemetry.start_span("single_prompt")
    try:
        text,in_t,out_t,lat=await client.generate(sanitized)
        telemetry.record_llm_call("gemini",cfg["model"]["target"],in_t,out_t,lat)
        telemetry.end_span(span,latency_ms=lat)
        return {"response":text,"input_tokens":in_t,"output_tokens":out_t,"latency_ms":lat,"flags":flags}
    except Exception as e:
        span.set_error(str(e)); telemetry.end_span(span)
        raise HTTPException(500,str(e))

# ── Observability ────────────────────────────────────────────────────────────
@app.get("/traces")
def traces(n:int=50, user=Depends(get_user)):
    from observability.telemetry import telemetry
    return {"traces":telemetry.get_recent_spans(n)}

# ── Security ─────────────────────────────────────────────────────────────────
@app.get("/audit")
def audit_log(limit:int=50, user=Depends(get_user)):
    from security.layer import audit
    return {"logs":audit.get_logs(limit=limit)}

@app.get("/audit/integrity")
def audit_integrity(user=Depends(get_user)):
    from security.layer import audit
    return audit.verify_integrity()

# ── WebSocket ────────────────────────────────────────────────────────────────
@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept(); _ws_clients.append(ws)
    try:
        await ws.send_json({"event":"connected","timestamp":datetime.utcnow().isoformat()})
        while True:
            try: await asyncio.wait_for(ws.receive_text(),timeout=30.0)
            except asyncio.TimeoutError: await ws.send_json({"event":"ping"})
    except WebSocketDisconnect:
        if ws in _ws_clients: _ws_clients.remove(ws)

@app.on_event("startup")
async def startup():
    Path("data").mkdir(exist_ok=True)
    print("[API] Platform running → http://localhost:8080/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app",host="0.0.0.0",port=8080,reload=True,log_level="info")
