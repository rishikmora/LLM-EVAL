"""
Module 10 — Cost Intelligence System (Tier 2 Enhancement)

Tracks and analyzes operational costs per evaluation run:
  - Per-category token/cost breakdown
  - Model ROI (quality per dollar)
  - Latency-cost trade-off analysis
  - Daily budget tracking and enforcement
  - Token burn heatmaps (data for dashboard)
"""

import json
import sqlite3
import statistics
from datetime import datetime, date
from pathlib import Path
from typing import Optional
import yaml

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config" / "eval_config.yaml"
DATA_DIR = ROOT / "data"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─── Pricing (Claude Haiku 4.5, as of 2026) ──────────────────────────────────

PRICING = {
    "claude-haiku-4-5-20251001": {
        "input_per_1m": 1.00,    # USD per 1M input tokens
        "output_per_1m": 5.00,   # USD per 1M output tokens
        "free_tier_input_daily": 0,
        "free_tier_rpm": 50,
        "currency": "USD",
    },
    "claude-sonnet-4-6": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
        "free_tier_input_daily": 0,
        "free_tier_rpm": 50,
        "currency": "USD",
    },
}

INR_RATE = 83.0  # approximate USD → INR


def token_cost_usd(input_tokens: int, output_tokens: int, model: str) -> float:
    pricing = PRICING.get(model, PRICING["claude-haiku-4-5-20251001"])
    return (input_tokens / 1_000_000 * pricing["input_per_1m"] +
            output_tokens / 1_000_000 * pricing["output_per_1m"])


# ─── Cost DB ──────────────────────────────────────────────────────────────────

def init_cost_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS token_budget (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            model TEXT,
            total_input_tokens INTEGER DEFAULT 0,
            total_output_tokens INTEGER DEFAULT 0,
            total_cost_usd REAL DEFAULT 0.0,
            run_count INTEGER DEFAULT 0
        )
    """)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_date_model ON token_budget(date, model)")
    conn.commit()
    conn.close()


def record_run_tokens(
    db_path: Path,
    model: str,
    input_tokens: int,
    output_tokens: int,
):
    init_cost_db(db_path)
    cost = token_cost_usd(input_tokens, output_tokens, model)
    today = date.today().isoformat()
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT INTO token_budget (date, model, total_input_tokens, total_output_tokens, total_cost_usd, run_count)
        VALUES (?, ?, ?, ?, ?, 1)
        ON CONFLICT(date, model) DO UPDATE SET
            total_input_tokens = total_input_tokens + excluded.total_input_tokens,
            total_output_tokens = total_output_tokens + excluded.total_output_tokens,
            total_cost_usd = total_cost_usd + excluded.total_cost_usd,
            run_count = run_count + 1
    """, (today, model, input_tokens, output_tokens, cost))
    conn.commit()
    conn.close()


def get_daily_usage(db_path: Path, model: str, target_date: Optional[str] = None) -> dict:
    if not db_path.exists():
        return {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    today = target_date or date.today().isoformat()
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT total_input_tokens, total_output_tokens, total_cost_usd, run_count FROM token_budget WHERE date=? AND model=?",
        (today, model)
    ).fetchone()
    conn.close()
    if not row:
        return {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "runs": 0}
    return {
        "input_tokens": row[0],
        "output_tokens": row[1],
        "cost_usd": round(row[2], 6),
        "runs": row[3],
    }


def check_budget(db_path: Path, model: str, config: dict) -> dict:
    """Check if daily token budget would be exceeded."""
    budget = config.get("api", {}).get("daily_token_budget", 900_000)
    usage = get_daily_usage(db_path, model)
    remaining = max(0, budget - usage["input_tokens"])
    pct_used = usage["input_tokens"] / budget if budget > 0 else 0.0

    return {
        "budget": budget,
        "used": usage["input_tokens"],
        "remaining": remaining,
        "pct_used": round(pct_used, 3),
        "budget_exceeded": usage["input_tokens"] >= budget,
        "warning": pct_used > 0.80,
    }


# ─── Cost analysis ────────────────────────────────────────────────────────────

def compute_cost_analytics(results: list[dict], model: str) -> dict:
    """Compute comprehensive cost analytics for a run."""
    if not results:
        return {}

    total_in = sum(r.get("input_tokens", 0) or 0 for r in results)
    total_out = sum(r.get("output_tokens", 0) or 0 for r in results)
    total_cost = token_cost_usd(total_in, total_out, model)

    # Per-category breakdown
    by_cat: dict[str, dict] = {}
    for r in results:
        cat = r.get("category", "unknown")
        in_t = r.get("input_tokens", 0) or 0
        out_t = r.get("output_tokens", 0) or 0
        by_cat.setdefault(cat, {"input": 0, "output": 0, "count": 0})
        by_cat[cat]["input"] += in_t
        by_cat[cat]["output"] += out_t
        by_cat[cat]["count"] += 1

    category_costs = {}
    for cat, data in by_cat.items():
        cost = token_cost_usd(data["input"], data["output"], model)
        category_costs[cat] = {
            "input_tokens": data["input"],
            "output_tokens": data["output"],
            "cost_usd": round(cost, 6),
            "cost_inr": round(cost * INR_RATE, 4),
            "cost_per_prompt": round(cost / max(data["count"], 1), 8),
            "prompt_count": data["count"],
        }

    # ROI: quality score per dollar
    # Quality = mean relevance (higher is better); safety = 1 - ASR (lower ASR is safer)
    relevance_vals = [r.get("relevance", 0) or 0 for r in results if r.get("relevance") is not None]
    asr_vals = [r.get("asr", 0) or 0 for r in results if r.get("asr") is not None]
    mean_quality = statistics.mean(relevance_vals) if relevance_vals else 0.5
    mean_safety = 1 - statistics.mean(asr_vals) if asr_vals else 1.0
    roi_quality = mean_quality / max(total_cost, 1e-9)
    roi_safety = mean_safety / max(total_cost, 1e-9)

    # Latency-cost frontier
    lats = [r.get("latency_ms", 0) or 0 for r in results]
    mean_lat = statistics.mean(lats) if lats else 0

    # Free tier analysis
    pricing = PRICING.get(model, PRICING["claude-haiku-4-5-20251001"])
    free_tier_remaining = max(0, pricing["free_tier_input_daily"] - total_in)
    within_free_tier = total_in <= pricing["free_tier_input_daily"]

    return {
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_cost_usd": round(total_cost, 6),
        "total_cost_inr": round(total_cost * INR_RATE, 4),
        "cost_per_prompt_usd": round(total_cost / max(len(results), 1), 8),
        "category_breakdown": category_costs,
        "roi_quality_per_dollar": round(roi_quality, 2),
        "roi_safety_per_dollar": round(roi_safety, 2),
        "mean_latency_ms": round(mean_lat, 1),
        "free_tier_remaining_tokens": free_tier_remaining,
        "within_free_tier": within_free_tier,
        "estimated_daily_capacity": int(pricing["free_tier_input_daily"] / max(total_in / len(results), 1)),
        "token_burn_by_category": {
            cat: data["input"] + data["output"]
            for cat, data in by_cat.items()
        },
    }


def print_cost_report(analytics: dict):
    print("\n=== Cost Intelligence Report ===")
    print(f"  Total cost: ${analytics['total_cost_usd']:.6f} USD (₹{analytics['total_cost_inr']:.4f})")
    print(f"  Tokens: {analytics['total_input_tokens']:,} in / {analytics['total_output_tokens']:,} out")
    print(f"  Free tier: {'✓ Within limit' if analytics['within_free_tier'] else '✗ EXCEEDED'}")
    print(f"  Remaining free tokens today: {analytics['free_tier_remaining_tokens']:,}")
    print(f"\n  Category costs:")
    for cat, data in sorted(analytics["category_breakdown"].items(), key=lambda x: -x[1]["cost_usd"]):
        print(f"    {cat:30s}: ${data['cost_usd']:.6f}  ({data['prompt_count']} prompts)")
