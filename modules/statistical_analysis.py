"""
Module 08 — Advanced Statistical Analysis

Adds:
  - ROC-AUC curve computation for ASR binary classifier
  - Precision / Recall / F1 for judge outputs
  - Score distribution analysis
  - Evaluator consistency (same prompt, multiple runs)
  - Calibration curves for judge reliability
"""

import math
import statistics
from pathlib import Path
from typing import Optional


# ─── ROC-AUC for ASR Classifier ──────────────────────────────────────────────

def compute_roc_auc(
    y_true: list[int],
    y_score: list[float],
) -> dict:
    """
    Compute ROC curve and AUC for the ASR binary classifier.
    y_true: ground truth (1=attack_succeeded, 0=refused)
    y_score: continuous score or probability from judge (0-1)

    Returns dict with auc, tpr_list, fpr_list, thresholds for plotting.
    """
    if len(y_true) != len(y_score) or not y_true:
        return {"auc": 0.0, "tpr": [], "fpr": [], "thresholds": []}

    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return {"auc": None, "note": "Cannot compute ROC: only one class present"}

    # Sort by decreasing score
    pairs = sorted(zip(y_score, y_true), reverse=True)
    thresholds, tpr_list, fpr_list = [], [], []

    tp, fp = 0, 0
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
        thresholds.append(round(score, 4))

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

    # Find optimal threshold (Youden's J)
    best_j, best_thresh, best_tpr, best_fpr = -1, 0.5, 0, 0
    for t, tpr, fpr in zip(thresholds, tpr_list, fpr_list):
        j = tpr - fpr
        if j > best_j:
            best_j, best_thresh, best_tpr, best_fpr = j, t, tpr, fpr

    return {
        "auc": round(abs(auc), 4),
        "tpr": tpr_list,
        "fpr": fpr_list,
        "thresholds": thresholds,
        "optimal_threshold": best_thresh,
        "optimal_tpr": round(best_tpr, 4),
        "optimal_fpr": round(best_fpr, 4),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "interpretation": (
            "excellent" if abs(auc) >= 0.9 else
            "good" if abs(auc) >= 0.8 else
            "fair" if abs(auc) >= 0.7 else
            "poor"
        ),
    }


# ─── Precision / Recall / F1 ─────────────────────────────────────────────────

def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict:
    """
    Compute precision, recall, F1 for binary classification.
    Applied to ASR classifier (1=attack_success, 0=refusal).
    """
    if not y_true or len(y_true) != len(y_pred):
        return {}

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "fpr": round(fpr, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


# ─── Evaluator Consistency ────────────────────────────────────────────────────

def compute_evaluator_consistency(
    run_results_list: list[list[dict]],
) -> dict:
    """
    Measure how consistently the LLM judge scores the same prompts across runs.
    Takes a list of result sets (one per run) and finds prompts that appear in multiple runs.
    """
    if len(run_results_list) < 2:
        return {"status": "need_min_2_runs"}

    # Build prompt_id -> [scores] across runs
    prompt_scores: dict[str, dict[str, list[float]]] = {}
    for results in run_results_list:
        for r in results:
            pid = r.get("prompt_id", "")
            if not pid:
                continue
            prompt_scores.setdefault(pid, {})
            for metric in ["relevance", "toxicity", "bias", "asr"]:
                val = r.get(metric)
                if val is not None:
                    prompt_scores[pid].setdefault(metric, []).append(val)

    # Find prompts scored in >= 2 runs
    multi_run = {pid: ms for pid, ms in prompt_scores.items()
                 if any(len(v) >= 2 for v in ms.values())}

    if not multi_run:
        return {"status": "no_repeated_prompts"}

    metric_consistency = {}
    for metric in ["relevance", "toxicity", "bias", "asr"]:
        variances = []
        for pid, scores_by_metric in multi_run.items():
            vals = scores_by_metric.get(metric, [])
            if len(vals) >= 2:
                variances.append(statistics.variance(vals))
        if variances:
            metric_consistency[metric] = {
                "mean_variance": round(statistics.mean(variances), 6),
                "max_variance": round(max(variances), 6),
                "n_prompts": len(variances),
                "consistency_rating": (
                    "high" if statistics.mean(variances) < 0.01 else
                    "medium" if statistics.mean(variances) < 0.05 else
                    "low"
                ),
            }

    return {
        "status": "completed",
        "n_repeated_prompts": len(multi_run),
        "metric_consistency": metric_consistency,
    }


# ─── Distribution Analysis ────────────────────────────────────────────────────

def score_distribution_analysis(results: list[dict]) -> dict:
    """
    Analyze score distributions for anomaly detection.
    Flags metrics that show unusual distributions (ceiling/floor effects, bimodal).
    """
    analysis = {}
    metrics = ["relevance", "toxicity", "bias", "asr", "faithfulness"]

    for metric in metrics:
        vals = [r[metric] for r in results if r.get(metric) is not None]
        if len(vals) < 5:
            continue

        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        sorted_vals = sorted(vals)

        # Quartiles
        n = len(sorted_vals)
        q1 = sorted_vals[n // 4]
        q3 = sorted_vals[3 * n // 4]

        # Ceiling/floor effects (>80% of values at max or min)
        at_ceiling = sum(1 for v in vals if v >= 0.95) / n
        at_floor = sum(1 for v in vals if v <= 0.05) / n

        flags = []
        if at_ceiling > 0.8:
            flags.append("ceiling_effect")
        if at_floor > 0.8:
            flags.append("floor_effect")
        if std < 0.01 and n > 10:
            flags.append("near_constant")

        analysis[metric] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(sorted_vals[0], 4),
            "max": round(sorted_vals[-1], 4),
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(q3 - q1, 4),
            "at_ceiling_pct": round(at_ceiling, 3),
            "at_floor_pct": round(at_floor, 3),
            "flags": flags,
            "n": n,
        }

    return analysis


# ─── Per-category deep dive ───────────────────────────────────────────────────

def category_deep_dive(results: list[dict]) -> dict:
    """Per-category breakdown of all metrics."""
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        by_cat.setdefault(cat, []).append(r)

    breakdown = {}
    for cat, cat_results in by_cat.items():
        cat_stats = {}
        for metric in ["relevance", "toxicity", "bias", "asr", "latency_ms", "token_efficiency"]:
            vals = [r[metric] for r in cat_results if r.get(metric) is not None]
            if vals:
                cat_stats[metric] = {
                    "mean": round(statistics.mean(vals), 4),
                    "std": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
                    "n": len(vals),
                }
        cat_stats["error_rate"] = round(
            sum(1 for r in cat_results if r.get("error")) / len(cat_results), 4
        )
        breakdown[cat] = cat_stats

    return breakdown
