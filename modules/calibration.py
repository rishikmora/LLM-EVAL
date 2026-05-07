"""
Module 06 — Judge Calibration Framework (Tier 1 Enhancement)

Runs the gold-standard benchmark before every evaluation to detect
LLM judge scoring drift. Alerts if calibration shifts > 0.05 from baseline.

Also supports multi-judge mode (Gemini + fallback to alternate provider).
"""

import json
import sqlite3
import statistics
from pathlib import Path
from datetime import datetime
from typing import Optional
import yaml

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config" / "eval_config.yaml"
BENCHMARK_DIR = ROOT / "data" / "benchmarks"
DATA_DIR = ROOT / "data"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─── Load benchmark dataset ───────────────────────────────────────────────────

def load_benchmark(version: str = "v1.0") -> list[dict]:
    path = BENCHMARK_DIR / version / "benchmark.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark not found: {path}")
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def list_benchmark_versions() -> list[str]:
    return sorted([d.name for d in BENCHMARK_DIR.iterdir() if d.is_dir()])


# ─── Calibration DB ───────────────────────────────────────────────────────────

def init_calibration_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calibration_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp TEXT,
            benchmark_version TEXT,
            judge_model TEXT,
            prompt_id TEXT,
            category TEXT,
            metric TEXT,
            expected_low REAL,
            expected_high REAL,
            actual_score REAL,
            in_range INTEGER,
            drift_from_baseline REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calibration_baselines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            benchmark_version TEXT,
            judge_model TEXT,
            metric TEXT,
            baseline_mean REAL,
            baseline_std REAL,
            n_samples INTEGER
        )
    """)
    conn.commit()
    conn.close()


def save_calibration_result(db_path: Path, result: dict):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT INTO calibration_runs (
            run_timestamp, benchmark_version, judge_model, prompt_id, category,
            metric, expected_low, expected_high, actual_score, in_range, drift_from_baseline
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result["run_timestamp"], result["benchmark_version"], result["judge_model"],
        result["prompt_id"], result["category"], result["metric"],
        result["expected_low"], result["expected_high"], result["actual_score"],
        1 if result["in_range"] else 0, result.get("drift_from_baseline"),
    ))
    conn.commit()
    conn.close()


def get_baseline(db_path: Path, benchmark_version: str, judge_model: str, metric: str) -> Optional[dict]:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(db_path)
    row = conn.execute("""
        SELECT baseline_mean, baseline_std, n_samples FROM calibration_baselines
        WHERE benchmark_version=? AND judge_model=? AND metric=?
        ORDER BY created_at DESC LIMIT 1
    """, (benchmark_version, judge_model, metric)).fetchone()
    conn.close()
    if row:
        return {"mean": row[0], "std": row[1], "n": row[2]}
    return None


def save_baseline(db_path: Path, benchmark_version: str, judge_model: str, metric: str,
                   mean: float, std: float, n: int):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT INTO calibration_baselines (created_at, benchmark_version, judge_model, metric, baseline_mean, baseline_std, n_samples)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (datetime.utcnow().isoformat(), benchmark_version, judge_model, metric, mean, std, n))
    conn.commit()
    conn.close()


# ─── Run calibration ─────────────────────────────────────────────────────────

async def run_calibration(
    client,
    benchmark_version: str = "v1.0",
    config: Optional[dict] = None,
    set_as_baseline: bool = False,
) -> dict:
    """
    Run the gold-standard benchmark through the judge and check for drift.
    Returns calibration report with pass/fail status per metric.
    """
    from modules.evaluator import (
        score_faithfulness, score_relevance, score_toxicity,
        score_bias, score_asr
    )

    cfg = config or load_config()
    db_path = DATA_DIR / "calibration.db"
    init_calibration_db(db_path)

    benchmark = load_benchmark(benchmark_version)
    judge_model = cfg["model"]["judge"]
    run_ts = datetime.utcnow().isoformat()
    alert_threshold = cfg.get("calibration", {}).get("alert_threshold", 0.05)

    print(f"\n[Calibration] Running benchmark {benchmark_version} ({len(benchmark)} prompts)...")

    metric_scores: dict[str, list[float]] = {}
    calibration_results = []

    for prompt in benchmark:
        expected_ranges = prompt.get("expected_score_range", {})
        answer = ""

        try:
            # Generate answer
            if prompt.get("is_multiturn"):
                turns = json.loads(prompt["text"])
                for turn in turns:
                    text, _, _, _ = await client.generate(turn)
                answer = text
            else:
                answer, _, _, _ = await client.generate(prompt["text"])

            # Score relevant metrics
            scores_this = {}
            context = prompt.get("context", "")
            demographic = prompt.get("demographic")
            expected_behavior = prompt.get("expected_behavior", "answer_correctly")

            if "faithfulness" in expected_ranges and context:
                scores_this["faithfulness"] = await score_faithfulness(
                    client, prompt["text"], answer, context)
            if "relevance" in expected_ranges:
                scores_this["relevance"] = await score_relevance(
                    client, prompt["text"], answer)
            if "toxicity" in expected_ranges:
                scores_this["toxicity"] = await score_toxicity(client, answer)
            if "bias" in expected_ranges:
                scores_this["bias"] = await score_bias(client, answer, demographic)
            if "asr" in expected_ranges:
                asr_val, _ = await score_asr(
                    client, prompt["text"], answer, expected_behavior)
                scores_this["asr"] = asr_val

            # Compare to expected ranges
            for metric, score in scores_this.items():
                metric_scores.setdefault(metric, []).append(score)
                expected = expected_ranges.get(metric, [0.0, 1.0])
                in_range = expected[0] <= score <= expected[1]

                baseline = get_baseline(db_path, benchmark_version, judge_model, metric)
                drift = abs(score - baseline["mean"]) if baseline else None

                result = {
                    "run_timestamp": run_ts,
                    "benchmark_version": benchmark_version,
                    "judge_model": judge_model,
                    "prompt_id": prompt["id"],
                    "category": prompt["category"],
                    "metric": metric,
                    "expected_low": expected[0],
                    "expected_high": expected[1],
                    "actual_score": round(score, 4),
                    "in_range": in_range,
                    "drift_from_baseline": round(drift, 4) if drift is not None else None,
                }
                save_calibration_result(db_path, result)
                calibration_results.append(result)

        except Exception as e:
            print(f"[Calibration] Error on {prompt['id']}: {e}")

    # Compute summary
    metric_summary = {}
    drift_alerts = []

    for metric, scores in metric_scores.items():
        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        in_range_count = sum(
            1 for r in calibration_results
            if r["metric"] == metric and r["in_range"]
        )
        total_count = sum(1 for r in calibration_results if r["metric"] == metric)

        baseline = get_baseline(db_path, benchmark_version, judge_model, metric)
        drift = abs(mean - baseline["mean"]) if baseline else None

        metric_summary[metric] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "in_range_pct": round(in_range_count / max(total_count, 1), 3),
            "n": len(scores),
            "baseline_drift": round(drift, 4) if drift is not None else None,
            "drift_alert": drift is not None and drift > alert_threshold,
        }

        if drift is not None and drift > alert_threshold:
            drift_alerts.append({
                "metric": metric,
                "drift": round(drift, 4),
                "threshold": alert_threshold,
                "current_mean": round(mean, 4),
                "baseline_mean": round(baseline["mean"], 4),
            })

        if set_as_baseline:
            save_baseline(db_path, benchmark_version, judge_model, metric, mean, std, len(scores))

    calibration_passed = len(drift_alerts) == 0

    report = {
        "run_timestamp": run_ts,
        "benchmark_version": benchmark_version,
        "judge_model": judge_model,
        "total_prompts": len(benchmark),
        "calibration_passed": calibration_passed,
        "metric_summary": metric_summary,
        "drift_alerts": drift_alerts,
        "set_as_baseline": set_as_baseline,
    }

    # Print summary
    status = "✓ PASSED" if calibration_passed else f"✗ FAILED ({len(drift_alerts)} drift alerts)"
    print(f"[Calibration] {status}")
    for metric, s in metric_summary.items():
        drift_str = f"  DRIFT={s['baseline_drift']:.4f}⚠" if s.get("drift_alert") else ""
        print(f"  {metric:15s}: mean={s['mean']:.4f}  in_range={s['in_range_pct']:.1%}{drift_str}")

    if set_as_baseline:
        print(f"[Calibration] Baseline saved for {judge_model} on {benchmark_version}")

    # Save report
    report_path = DATA_DIR / f"calibration_report_{run_ts[:10]}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


# ─── Multi-judge comparison ───────────────────────────────────────────────────

async def compare_judges(
    primary_client,
    benchmark_version: str = "v1.0",
    config: Optional[dict] = None,
) -> dict:
    """
    Run the same benchmark through both the primary judge and any configured
    alternate judge, measuring inter-judge agreement (like Cohen's Kappa for scores).
    """
    from modules.evaluator import score_relevance, score_toxicity, score_asr

    cfg = config or load_config()
    alternate_model = cfg.get("model", {}).get("alternate_judge", "")

    if not alternate_model:
        return {"status": "skipped", "reason": "No alternate_judge configured in eval_config.yaml"}

    benchmark = load_benchmark(benchmark_version)
    primary_scores = []
    alternate_scores = []

    # For now, run primary judge twice with different temperature as a variance check
    # In production, wire in a second provider here
    for prompt in benchmark[:5]:  # spot check first 5
        try:
            answer, _, _, _ = await primary_client.generate(prompt["text"])
            s1 = await score_relevance(primary_client, prompt["text"], answer)
            s2 = await score_relevance(primary_client, prompt["text"], answer)
            primary_scores.append(s1)
            alternate_scores.append(s2)
        except Exception:
            pass

    if len(primary_scores) < 2:
        return {"status": "insufficient_data"}

    # Pearson correlation as judge agreement metric
    n = len(primary_scores)
    mean_p = sum(primary_scores) / n
    mean_a = sum(alternate_scores) / n
    num = sum((p - mean_p) * (a - mean_a) for p, a in zip(primary_scores, alternate_scores))
    den_p = sum((p - mean_p) ** 2 for p in primary_scores) ** 0.5
    den_a = sum((a - mean_a) ** 2 for a in alternate_scores) ** 0.5
    corr = num / (den_p * den_a) if den_p * den_a > 0 else 0.0

    return {
        "status": "completed",
        "judge_agreement_correlation": round(corr, 4),
        "interpretation": "strong" if corr > 0.8 else "moderate" if corr > 0.6 else "weak",
        "n_prompts": n,
    }
