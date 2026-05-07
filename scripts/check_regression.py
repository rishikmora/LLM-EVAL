#!/usr/bin/env python3
"""CI/CD regression checker — fails with exit code 1 if regressions detected."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    from modules.tracker import list_runs, load_run_results, compute_run_stats, compare_runs
    import yaml
    with open(Path(__file__).parent.parent / "config" / "eval_config.yaml") as f:
        cfg = yaml.safe_load(f)
    db = Path(cfg["database"]["path"])
    runs = list_runs(db)
    if len(runs) < 2:
        print("[Regression] Need >= 2 runs to check regression. SKIP.")
        sys.exit(0)
    current = load_run_results(runs[0], db)
    previous = load_run_results(runs[1], db)
    sc = compute_run_stats(current)
    sp = compute_run_stats(previous)
    comparison = compare_runs(sc, sp, current, previous)
    regressions = [m for m, c in comparison.items()
                   if c.get("should_alert") and c["direction"] == "degraded"
                   and m in ("faithfulness", "relevance", "toxicity", "asr")]
    if regressions:
        print(f"[Regression] ✗ REGRESSIONS DETECTED: {regressions}")
        for m in regressions:
            c = comparison[m]
            print(f"  {m}: delta={c['delta']:+.4f}  p={c['p_value']:.4f}")
        sys.exit(1)
    else:
        print(f"[Regression] ✓ No significant regressions. ({runs[0]} vs {runs[1]})")
        sys.exit(0)

if __name__ == "__main__":
    main()
