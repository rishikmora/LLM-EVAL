"""
Module 03 — MLflow Tracker + Full Statistical Validation
Bootstrap CI, Mann-Whitney U, ROC-AUC, significance-gated alerts.
"""
import json, sqlite3, statistics, random
from pathlib import Path
from typing import Optional
import yaml

DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "eval_config.yaml"
METRICS = ["faithfulness", "relevance", "toxicity", "bias", "asr", "latency_ms", "token_efficiency"]

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def bootstrap_ci(values, n_resamples=1000, ci=0.95):
    if not values: return 0.0, 0.0, 0.0
    if len(values) == 1: return values[0], values[0], values[0]
    means = sorted([statistics.mean(random.choices(values, k=len(values))) for _ in range(n_resamples)])
    alpha = (1 - ci) / 2
    return statistics.mean(values), means[int(alpha*n_resamples)], means[min(int((1-alpha)*n_resamples), len(means)-1)]

def mann_whitney_u(a, b):
    if not a or not b: return 1.0, False
    import math
    n1, n2 = len(a), len(b)
    combined = sorted([(v,0) for v in a] + [(v,1) for v in b])
    r1 = sum(rank+1 for rank,(_,grp) in enumerate(combined) if grp==0)
    u1 = r1 - n1*(n1+1)/2
    mu = n1*n2/2
    sigma = math.sqrt(n1*n2*(n1+n2+1)/12)
    if sigma == 0: return 1.0, False
    z = (u1-mu)/sigma
    p = 2*(1-0.5*(1+math.erf(abs(z)/math.sqrt(2))))
    return round(p,4), p < 0.05

def compute_run_stats(results):
    stats = {}
    cfg = load_config()
    n_rs = cfg.get("statistical",{}).get("bootstrap_resamples",1000)
    ci = cfg.get("statistical",{}).get("ci_level",0.95)
    for metric in METRICS:
        vals = [r[metric] for r in results if r.get(metric) is not None]
        if not vals: continue
        mean, ci_low, ci_high = bootstrap_ci(vals, n_rs, ci)
        stats[metric] = {"mean":round(mean,4),"ci_low":round(ci_low,4),"ci_high":round(ci_high,4),
                         "n":len(vals),"std":round(statistics.stdev(vals) if len(vals)>1 else 0.0,4),
                         "min":round(min(vals),4),"max":round(max(vals),4)}
    cat_asr = {}
    for r in results:
        cat = r.get("category","unknown")
        if r.get("asr") is not None: cat_asr.setdefault(cat,[]).append(r["asr"])
    stats["asr_by_category"] = {cat:round(statistics.mean(v),4) for cat,v in cat_asr.items()}
    lats = sorted([r["latency_ms"] for r in results if r.get("latency_ms")])
    if lats:
        stats["latency_p50"] = lats[len(lats)//2]
        stats["latency_p95"] = lats[int(len(lats)*0.95)]
        stats["latency_p99"] = lats[min(int(len(lats)*0.99), len(lats)-1)]
    adversarial = [r for r in results if r.get("priority") in ("CRITICAL","HIGH","MEDIUM")]
    if len(adversarial) >= 5:
        try:
            from modules.statistical_analysis import compute_roc_auc, compute_classification_metrics
            y_true = [1 if (r.get("asr") or 0)>=0.5 else 0 for r in adversarial]
            y_score = [r.get("asr") or 0 for r in adversarial]
            y_pred = [1 if s>=0.5 else 0 for s in y_score]
            if 0 < sum(y_true) < len(y_true):
                roc = compute_roc_auc(y_true, y_score)
                clf = compute_classification_metrics(y_true, y_pred)
                stats["asr_classifier"] = {"roc_auc":roc.get("auc"),"auc_interpretation":roc.get("interpretation"),
                    "precision":clf.get("precision"),"recall":clf.get("recall"),"f1":clf.get("f1"),
                    "optimal_threshold":roc.get("optimal_threshold"),"n_adversarial":len(adversarial)}
        except Exception: pass
    try:
        from modules.statistical_analysis import score_distribution_analysis, category_deep_dive
        stats["distribution_analysis"] = score_distribution_analysis(results)
        stats["category_deep_dive"] = category_deep_dive(results)
    except Exception: pass
    return stats

def compare_runs(stats_a, stats_b, results_a, results_b):
    cfg = load_config()
    sig = cfg.get("statistical",{}).get("significance_level",0.05)
    min_n = cfg.get("statistical",{}).get("min_samples_for_test",10)
    comparison = {}
    for metric in METRICS:
        if metric not in stats_a or metric not in stats_b: continue
        va = [r[metric] for r in results_a if r.get(metric) is not None]
        vb = [r[metric] for r in results_b if r.get(metric) is not None]
        p, is_sig = mann_whitney_u(va, vb)
        delta = round(stats_b[metric]["mean"]-stats_a[metric]["mean"],4)
        comparison[metric] = {"delta":delta,"direction":"improved" if delta>0 else "degraded" if delta<0 else "unchanged",
                              "p_value":p,"significant":is_sig,"should_alert":is_sig and len(va)>=min_n and p<sig}
    return comparison

def log_to_mlflow(run_id, results, stats, config, tags=None):
    try:
        import mlflow
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        with mlflow.start_run(run_name=run_id, tags=tags or {}):
            for metric,s in stats.items():
                if isinstance(s,dict) and "mean" in s:
                    mlflow.log_metric(f"{metric}_mean",s["mean"])
                    mlflow.log_metric(f"{metric}_ci_low",s["ci_low"])
                    mlflow.log_metric(f"{metric}_ci_high",s["ci_high"])
            for k in ("latency_p50","latency_p95","latency_p99"):
                if k in stats: mlflow.log_metric(k, stats[k])
            for cat,v in stats.get("asr_by_category",{}).items():
                mlflow.log_metric(f"asr_{cat}", v)
            asr_clf = stats.get("asr_classifier",{})
            if asr_clf.get("roc_auc") is not None: mlflow.log_metric("asr_roc_auc", asr_clf["roc_auc"])
            if asr_clf.get("f1") is not None: mlflow.log_metric("asr_f1", asr_clf["f1"])
            mlflow.log_metric("total_prompts", len(results))
            mlflow.log_metric("errors", sum(1 for r in results if r.get("error")))
            mlflow.log_param("target_model", config["model"]["target"])
            mlflow.log_param("judge_model", config["model"]["judge"])
            rp = DATA_DIR / f"mlflow_{run_id}.json"
            with open(rp,"w") as f: json.dump({"run_id":run_id,"stats":stats},f,indent=2,default=str)
            mlflow.log_artifact(str(rp))
        print(f"[MLflow] Logged '{run_id}'")
        return True
    except Exception as e:
        print(f"[MLflow] Warning: {e}")
        return False

def load_run_results(run_id, db_path):
    if not Path(db_path).exists(): return []
    conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM eval_results WHERE run_id=?",(run_id,)).fetchall()
    conn.close(); return [dict(r) for r in rows]

def list_runs(db_path):
    if not Path(db_path).exists(): return []
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT DISTINCT run_id FROM eval_results ORDER BY run_id DESC").fetchall()
    conn.close(); return [r[0] for r in rows]

def check_thresholds(stats, config):
    t = config["thresholds"]; violations = []
    def check(metric, stat_key, threshold, op, label):
        if stat_key in stats and "mean" in stats[stat_key]:
            val = stats[stat_key]["mean"]
            if (val < threshold if op=="min" else val > threshold):
                violations.append({"metric":metric,"value":val,"threshold":threshold,"op":op,"label":label,
                    "severity":"CRITICAL" if label=="critical" else "WARNING",
                    "ci_low":stats[stat_key].get("ci_low"),"ci_high":stats[stat_key].get("ci_high")})
    check("faithfulness","faithfulness",t["faithfulness_min"],"min","warning")
    check("faithfulness","faithfulness",t["faithfulness_critical"],"min","critical")
    check("relevance","relevance",t["relevance_min"],"min","warning")
    check("toxicity","toxicity",t["toxicity_max"],"max","warning")
    check("toxicity","toxicity",t["toxicity_critical"],"max","critical")
    check("bias","bias",t["bias_max"],"max","warning")
    check("asr","asr",t["asr_max"],"max","warning")
    return violations

def track_run(run_id, results, config=None):
    cfg = config or load_config()
    stats = compute_run_stats(results)
    violations = check_thresholds(stats, cfg)
    if violations:
        print(f"\n[Tracker] ⚠ {len(violations)} violation(s):")
        for v in violations:
            ci_low = v.get('ci_low') or 0
            ci_high = v.get('ci_high') or 0
            print(f"  [{v['severity']}] {v['metric']}={v['value']:.3f} CI=[{ci_low:.3f},{ci_high:.3f}]")
    log_to_mlflow(run_id, results, stats, cfg)
    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR/f"stats_{run_id}.json","w") as f:
        json.dump({"run_id":run_id,"stats":stats,"violations":violations},f,indent=2,default=str)
    return {"stats":stats,"violations":violations}
