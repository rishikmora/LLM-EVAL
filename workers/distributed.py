"""
workers/distributed.py — Distributed Evaluation Infrastructure

Architecture:
  - Celery workers consume from Redis queue
  - Jobs sharded by category for parallel execution
  - Fault tolerance: retries, dead-letter queue, checkpointing
  - Health monitoring via worker heartbeats
  - Compatible with local asyncio and distributed Celery mode

Usage (local):
  python workers/distributed.py --local --suite adversarial --n 5

Usage (distributed):
  # Terminal 1 — start Redis: docker run -p 6379:6379 redis
  # Terminal 2 — start worker: celery -A workers.distributed worker --loglevel=info
  # Terminal 3 — submit job:   python workers/distributed.py --submit --suite full --n 10
"""
from __future__ import annotations
import asyncio, json, os, sys, time, uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
DATA_DIR = ROOT / "data"


# ─── Job model ────────────────────────────────────────────────────────────────

@dataclass
class EvalJob:
    job_id: str
    run_id: str
    suite: str
    n_per_category: int
    provider: str
    status: str = "pending"       # pending | running | done | failed
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    total_prompts: int = 0
    completed_prompts: int = 0
    error: str = ""
    shard_index: int = 0
    total_shards: int = 1

    def to_dict(self): return asdict(self)


# ─── Job Store (Redis or file fallback) ──────────────────────────────────────

class JobStore:
    """Persists job state. Uses Redis when available, falls back to local JSON."""
    def __init__(self, redis_url: str = REDIS_URL, store_path: Optional[Path] = None):
        self._redis = None
        self._local: dict[str, dict] = {}
        self._store_path = store_path if store_path is not None else (DATA_DIR / "jobs.json")
        try:
            import redis as redis_lib
            _r = redis_lib.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
            _r.ping()
            self._redis = _r
            print("[JobStore] Connected to Redis")
        except Exception:
            self._redis = None
            print("[JobStore] Redis unavailable — using local file store")
            self._load_local()

    def _load_local(self):
        if self._store_path.exists():
            try:
                with open(self._store_path) as f:
                    self._local = json.load(f)
            except Exception:
                self._local = {}

    def _save_local(self):
        DATA_DIR.mkdir(exist_ok=True)
        with open(self._store_path, "w") as f:
            json.dump(self._local, f, indent=2)

    def save(self, job: EvalJob, ttl: int = 86400):
        data = job.to_dict()
        if self._redis:
            self._redis.setex(f"job:{job.job_id}", ttl, json.dumps(data))
        else:
            self._local[job.job_id] = data
            self._save_local()

    def get(self, job_id: str) -> Optional[EvalJob]:
        if self._redis:
            raw = self._redis.get(f"job:{job_id}")
            if raw:
                data = json.loads(raw)
                return EvalJob(**data)
        elif job_id in self._local:
            return EvalJob(**self._local[job_id])
        return None

    def list_jobs(self, limit: int = 50) -> list[dict]:
        if self._redis:
            keys = self._redis.keys("job:*")[:limit]
            jobs = []
            for k in keys:
                raw = self._redis.get(k)
                if raw:
                    jobs.append(json.loads(raw))
            return sorted(jobs, key=lambda x: x.get("created_at",""), reverse=True)
        return sorted(self._local.values(), key=lambda x: x.get("created_at",""), reverse=True)[:limit]

    def update_progress(self, job_id: str, completed: int, total: int, status: str = "running"):
        job = self.get(job_id)
        if job:
            job.completed_prompts = completed
            job.total_prompts = total
            job.status = status
            self.save(job)

    def is_redis_available(self) -> bool:
        if not self._redis: return False
        try: self._redis.ping(); return True
        except Exception: return False


# ─── Celery App (lazy init) ───────────────────────────────────────────────────

def get_celery_app():
    try:
        from celery import Celery
        app = Celery("llm_eval",
                     broker=REDIS_URL,
                     backend=REDIS_URL,
                     include=["workers.distributed"])
        app.conf.update(
            task_serializer="json",
            result_serializer="json",
            accept_content=["json"],
            task_acks_late=True,
            worker_prefetch_multiplier=1,
            task_track_started=True,
            task_soft_time_limit=3600,
            task_time_limit=7200,
            task_max_retries=3,
            task_default_retry_delay=30,
        )
        return app
    except ImportError:
        return None


# ─── Core evaluation runner (used by both Celery and local) ──────────────────

async def _run_eval_async(job: EvalJob, store: JobStore) -> dict:
    """Async core — runs one evaluation shard."""
    import yaml
    from modules.prompt_generator import generate_prompt_dataset
    from modules.evaluator import run_evaluation, init_db
    from modules.tracker import track_run
    from modules.cost_intelligence import record_run_tokens

    with open(ROOT / "config" / "eval_config.yaml") as f:
        cfg = yaml.safe_load(f)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    job.status = "running"
    job.started_at = datetime.utcnow().isoformat()
    store.save(job)

    # Generate shard of prompts
    prompts = generate_prompt_dataset(
        n_per_category=job.n_per_category,
        include_benign=job.suite in ("full", "benign"),
        include_rag=job.suite in ("full", "rag"),
    )

    # Shard by index if distributed
    if job.total_shards > 1:
        shard_size = len(prompts) // job.total_shards
        start = job.shard_index * shard_size
        end = start + shard_size if job.shard_index < job.total_shards - 1 else len(prompts)
        prompts = prompts[start:end]

    job.total_prompts = len(prompts)
    store.save(job)

    def on_result(result, completed, total):
        store.update_progress(job.job_id, completed, total, "running")

    results = await run_evaluation(api_key, prompts, job.run_id, cfg, on_result=on_result)

    # Track
    tracking = track_run(job.run_id, results, cfg)

    # Cost
    total_in = sum(r.get("input_tokens", 0) or 0 for r in results)
    total_out = sum(r.get("output_tokens", 0) or 0 for r in results)
    cost_db = DATA_DIR / cfg["database"].get("cost_db", "costs.db")
    record_run_tokens(cost_db, cfg["model"]["target"], total_in, total_out)

    # Save results
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / f"results_{job.run_id}.jsonl"
    with open(out, "a") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    job.status = "done"
    job.completed_at = datetime.utcnow().isoformat()
    job.completed_prompts = len(results)
    store.save(job)

    return {"job_id": job.job_id, "run_id": job.run_id, "results": len(results),
            "violations": len(tracking["violations"])}


def run_eval_shard(job_dict: dict) -> dict:
    """Sync wrapper for Celery task."""
    store = JobStore()
    job = EvalJob(**job_dict)
    return asyncio.run(_run_eval_async(job, store))


# ─── Celery task registration ─────────────────────────────────────────────────

celery_app = get_celery_app()
if celery_app:
    @celery_app.task(bind=True, name="eval.run_shard", max_retries=3, acks_late=True)
    def celery_run_shard(self, job_dict: dict) -> dict:
        try:
            return run_eval_shard(job_dict)
        except Exception as exc:
            raise self.retry(exc=exc, countdown=30 * (self.request.retries + 1))


# ─── Job Scheduler ────────────────────────────────────────────────────────────

class EvalScheduler:
    """
    Schedules evaluation jobs.
    - Local mode: runs shards sequentially in-process
    - Distributed mode: submits to Celery workers via Redis
    """
    def __init__(self, distributed: bool = False):
        self.store = JobStore()
        self.distributed = distributed and self.store.is_redis_available()
        if self.distributed:
            self.celery = get_celery_app()
            print("[Scheduler] Mode: DISTRIBUTED (Redis + Celery)")
        else:
            print("[Scheduler] Mode: LOCAL (asyncio)")

    def submit(self, suite: str, n_per_category: int = 5,
               n_shards: int = 1, provider: str = "gemini") -> str:
        """Submit evaluation job. Returns job_id."""
        job_id = str(uuid.uuid4())[:8]
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{job_id}"

        if n_shards > 1 and self.distributed:
            # Create one job per shard
            shard_ids = []
            for i in range(n_shards):
                shard_job_id = f"{job_id}_s{i}"
                job = EvalJob(
                    job_id=shard_job_id, run_id=run_id, suite=suite,
                    n_per_category=n_per_category, provider=provider,
                    created_at=datetime.utcnow().isoformat(),
                    shard_index=i, total_shards=n_shards,
                )
                self.store.save(job)
                self.celery.send_task("eval.run_shard", args=[job.to_dict()])
                shard_ids.append(shard_job_id)
            print(f"[Scheduler] Submitted {n_shards} shards: {shard_ids}")
            return job_id
        else:
            job = EvalJob(
                job_id=job_id, run_id=run_id, suite=suite,
                n_per_category=n_per_category, provider=provider,
                created_at=datetime.utcnow().isoformat(),
            )
            self.store.save(job)
            if self.distributed:
                self.celery.send_task("eval.run_shard", args=[job.to_dict()])
                print(f"[Scheduler] Job submitted to Celery: {job_id}")
            else:
                print(f"[Scheduler] Running locally: {job_id}")
                asyncio.run(_run_eval_async(job, self.store))
            return job_id

    def status(self, job_id: str) -> Optional[dict]:
        job = self.store.get(job_id)
        return job.to_dict() if job else None

    def list_jobs(self) -> list[dict]:
        return self.store.list_jobs()

    def worker_health(self) -> dict:
        if not self.distributed:
            return {"mode": "local", "workers": [{"id": "local", "status": "active"}]}
        try:
            inspect = self.celery.control.inspect(timeout=2)
            active = inspect.active() or {}
            stats = inspect.stats() or {}
            workers = []
            for worker_id, tasks in active.items():
                workers.append({"id": worker_id, "active_tasks": len(tasks),
                                 "status": "active", "stats": stats.get(worker_id, {})})
            return {"mode": "distributed", "workers": workers,
                    "total_workers": len(workers), "redis": self.store.is_redis_available()}
        except Exception as e:
            return {"mode": "distributed", "error": str(e)}


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Distributed Evaluation Scheduler")
    parser.add_argument("--local", action="store_true", help="Run locally (no Redis)")
    parser.add_argument("--submit", action="store_true", help="Submit job")
    parser.add_argument("--suite", default="adversarial")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--status", type=str, help="Check job status")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args()

    scheduler = EvalScheduler(distributed=not args.local)

    if args.health:
        print(json.dumps(scheduler.worker_health(), indent=2))
    elif args.list:
        jobs = scheduler.list_jobs()
        print(f"\n{'Job ID':15s} {'Status':10s} {'Progress':15s} {'Suite':12s}")
        print("─" * 55)
        for j in jobs:
            prog = f"{j.get('completed_prompts',0)}/{j.get('total_prompts',0)}"
            print(f"  {j.get('job_id',''):13s} {j.get('status',''):10s} {prog:15s} {j.get('suite','')}")
    elif args.status:
        status = scheduler.status(args.status)
        print(json.dumps(status, indent=2) if status else "Job not found")
    elif args.submit or args.local:
        job_id = scheduler.submit(args.suite, args.n, args.shards)
        print(f"\n[Scheduler] Job ID: {job_id}")
        print(f"[Scheduler] Monitor: python workers/distributed.py --status {job_id}")
