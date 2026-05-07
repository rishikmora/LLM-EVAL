#!/usr/bin/env python3
"""
run_eval.py — Production CLI for LLM Eval & Red-Teaming Framework
70 LPA Research-Grade Edition

Commands:
  python run_eval.py --suite adversarial --n 5
  python run_eval.py --suite full --n 10
  python run_eval.py --calibrate [--set-baseline]
  python run_eval.py --list-runs
  python run_eval.py --stats <run_id>
  python run_eval.py --compare <run_a> <run_b>
  python run_eval.py --cost <run_id>
  python run_eval.py --annotate <run_id>
"""
import argparse, asyncio, json, os, sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from modules.prompt_generator import generate_prompt_dataset
from modules.evaluator import run_evaluation, load_config
from modules.tracker import track_run, load_run_results, list_runs, compare_runs, compute_run_stats
from modules.alerts import fire_alerts
from modules.cost_intelligence import compute_cost_analytics, print_cost_report, record_run_tokens

DATA_DIR = Path(__file__).parent / "data"

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  LLM EVAL & RED-TEAMING FRAMEWORK — Production Grade           ║
║  Calibration · Stats · Annotation · Agentic · Cost Intel       ║
╚══════════════════════════════════════════════════════════════════╝""")

async def cmd_calibrate(api_key, cfg, set_baseline=False):
    from modules.evaluator import GeminiClient
    from modules.calibration import run_calibration
    client = GeminiClient(api_key, cfg)
    return await run_calibration(client, cfg.get("calibration",{}).get("benchmark_version","v1.0"), cfg, set_baseline)

async def cmd_run(args, cfg, api_key):
    suite = getattr(args,"suite","adversarial")
    n = getattr(args,"n",None) or cfg["evaluation"]["prompts_per_category"]
    if cfg.get("evaluation",{}).get("run_calibration_first",True):
        print("\n[Runner] Step 1/4: Judge calibration...")
        cal = await cmd_calibrate(api_key, cfg, False)
        if not cal["calibration_passed"]:
            print(f"[Runner] ⚠ Calibration drift on {len(cal.get('drift_alerts',[]))} metric(s)")
            if cfg.get("calibration",{}).get("block_on_calibration_fail",False):
                print("[Runner] Aborting — calibration failed"); return
    print(f"\n[Runner] Step 2/4: Generating prompts (suite={suite}, n={n})...")
    prompts = generate_prompt_dataset(
        n_per_category=n if suite in ("full","adversarial") else 0,
        include_benign=suite in ("full","benign"),
        include_rag=suite in ("full","rag"),
    )
    if not prompts: print("[Runner] No prompts generated."); return
    from modules.cost_intelligence import check_budget
    cost_db = DATA_DIR / cfg["database"].get("cost_db","costs.db")
    budget = check_budget(cost_db, cfg["model"]["target"], cfg)
    if budget["budget_exceeded"]: print(f"[Runner] ⚠ Daily budget exceeded. Aborting."); return
    if budget["warning"]: print(f"[Runner] ⚠ Budget {budget['pct_used']:.1%} used today")
    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    DATA_DIR.mkdir(exist_ok=True)
    cats = {}
    for p in prompts: cats[p["category"]] = cats.get(p["category"],0) + 1
    print(f"[Runner] Run: {run_id} | {len(prompts)} prompts")
    for cat,count in sorted(cats.items()): print(f"  {cat:35s} {count}")
    print(f"\n[Runner] Step 3/4: Evaluating...")
    results = await run_evaluation(api_key, prompts, run_id, cfg)
    out = DATA_DIR / f"results_{run_id}.jsonl"
    with open(out,"w") as f:
        for r in results: f.write(json.dumps(r,default=str)+"\n")
    total_in = sum(r.get("input_tokens",0) or 0 for r in results)
    total_out = sum(r.get("output_tokens",0) or 0 for r in results)
    record_run_tokens(cost_db, cfg["model"]["target"], total_in, total_out)
    print(f"\n[Runner] Step 4/4: Tracking & analysis...")
    tracking = track_run(run_id, results, cfg)
    stats = tracking["stats"]; violations = tracking["violations"]
    cost_analytics = compute_cost_analytics(results, cfg["model"]["target"])
    ann_db = DATA_DIR / cfg["database"].get("annotation_db","annotations.db")
    eval_db = Path(cfg["database"]["path"])
    try:
        from modules.annotation import create_annotation_tasks
        n_tasks = create_annotation_tasks(run_id, eval_db, ann_db, cfg.get("annotation",{}).get("sample_size_per_run",10))
        if n_tasks > 0: print(f"[Runner] {n_tasks} annotation tasks queued")
    except Exception as e: pass
    fire_alerts(run_id, violations, stats, cfg)
    agentic_summary = None
    if cfg.get("evaluation",{}).get("run_agentic_eval",False) or suite=="agentic":
        from modules.evaluator import GeminiClient
        from modules.agentic_eval import run_agentic_eval
        client = GeminiClient(api_key, cfg)
        ag = await run_agentic_eval(client, cfg)
        agentic_summary = ag["summary"]
        with open(DATA_DIR/f"agentic_{run_id}.json","w") as f: json.dump(ag,f,indent=2,default=str)
    print("\n"+"═"*65)
    print(f"RUN SUMMARY: {run_id}")
    print("═"*65)
    for metric,s in stats.items():
        if isinstance(s,dict) and "mean" in s:
            print(f"  {metric:20s}: {s['mean']:.4f}  CI=[{s['ci_low']:.4f},{s['ci_high']:.4f}]  n={s['n']}")
    if "latency_p95" in stats: print(f"  {'latency_p95':20s}: {stats['latency_p95']:.0f}ms")
    asr_clf = stats.get("asr_classifier",{})
    if asr_clf.get("roc_auc") is not None:
        print(f"\n  ASR Classifier: ROC-AUC={asr_clf['roc_auc']:.4f} ({asr_clf.get('auc_interpretation','')})  F1={asr_clf.get('f1',0):.4f}")
    if violations:
        print(f"\n  ⚠ {len(violations)} violation(s):")
        for v in violations: print(f"    [{v['severity']}] {v['metric']}={v['value']:.3f}")
    else: print("\n  ✓ All metrics within thresholds")
    print_cost_report(cost_analytics)
    if agentic_summary:
        print("\n  Agentic Eval:")
        for k,v in agentic_summary.items():
            if isinstance(v,float): print(f"    {k:35s}: {v:.4f}")
    print(f"\n[Runner] Results → {out}")
    print("[Runner] Dashboard: streamlit run modules/dashboard.py")

def cmd_list_runs(cfg):
    db = Path(cfg["database"]["path"])
    runs = list_runs(db)
    if not runs: print("No runs."); return
    print(f"\n{'Run ID':40s} {'Prompts':>8} {'Errors':>8}")
    print("─"*60)
    for rid in runs:
        res = load_run_results(rid,db); errors = sum(1 for r in res if r.get("error"))
        print(f"{rid:40s} {len(res):>8} {errors:>8}")

def cmd_stats(run_id, cfg):
    db = Path(cfg["database"]["path"])
    results = load_run_results(run_id, db)
    if not results: print(f"No results for: {run_id}"); return
    tracking = track_run(run_id, results, cfg)
    stats = tracking["stats"]
    print(f"\n=== Stats: {run_id} ({len(results)} prompts) ===")
    for metric,s in stats.items():
        if isinstance(s,dict) and "mean" in s:
            print(f"  {metric:20s}: {s['mean']:.4f}  CI=[{s['ci_low']:.4f},{s['ci_high']:.4f}]  std={s['std']:.4f}")
    if "latency_p95" in stats: print(f"\n  P50={stats.get('latency_p50',0):.0f}ms  P95={stats.get('latency_p95',0):.0f}ms")
    print("\n  ASR by category:")
    for cat,asr in stats.get("asr_by_category",{}).items():
        print(f"    {cat:30s}: {asr:.3f} {'█'*int(asr*20)}")
    clf = stats.get("asr_classifier",{})
    if clf.get("roc_auc") is not None:
        print(f"\n  ROC-AUC={clf['roc_auc']:.4f} ({clf.get('auc_interpretation','')})  F1={clf.get('f1',0):.4f}")

def cmd_compare(run_a, run_b, cfg):
    db = Path(cfg["database"]["path"])
    ra = load_run_results(run_a,db); rb = load_run_results(run_b,db)
    if not ra or not rb: print("One or both runs not found."); return
    sa = compute_run_stats(ra); sb = compute_run_stats(rb)
    comparison = compare_runs(sa,sb,ra,rb)
    print(f"\n=== Comparison: {run_a} vs {run_b} ===")
    print(f"{'Metric':20s} {'Δ':>8} {'Direction':>10} {'p-value':>10} {'Sig':>5} {'Alert':>6}")
    print("─"*65)
    for metric,c in comparison.items():
        print(f"  {metric:18s} {c['delta']:+8.4f} {c['direction']:>10} {c['p_value']:>10.4f} "
              f"{'yes':>5 if c['significant'] else 'no':>5} {'⚠':>6 if c.get('should_alert') else '':>6}")

def cmd_cost(run_id, cfg):
    db = Path(cfg["database"]["path"])
    results = load_run_results(run_id, db)
    if not results: print(f"No results for: {run_id}"); return
    print_cost_report(compute_cost_analytics(results, cfg["model"]["target"]))

def cmd_annotate(run_id, cfg):
    db = Path(cfg["database"]["path"])
    ann_db = DATA_DIR / cfg["database"].get("annotation_db","annotations.db")
    from modules.annotation import create_annotation_tasks, get_pending_tasks, compute_agreement_report
    n = create_annotation_tasks(run_id, db, ann_db, sample_size=20)
    print(f"Created {n} annotation tasks. Total pending: {len(get_pending_tasks(ann_db))}")
    report = compute_agreement_report(ann_db)
    if report.get("n_annotations",0) > 0:
        print(f"\nAgreement ({report['n_annotations']} annotations):")
        for m,s in report.get("metrics",{}).items():
            print(f"  {m}: kappa={s['kappa']:.4f} ({s['kappa_interpretation']})  disagreements={s['high_disagreement_count']}")
    else:
        print("No annotations yet. Run: streamlit run modules/dashboard.py")

def main():
    print_banner()
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["adversarial","benign","rag","full","agentic"], default="adversarial")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--set-baseline", action="store_true")
    parser.add_argument("--list-runs", action="store_true")
    parser.add_argument("--stats", type=str)
    parser.add_argument("--compare", nargs=2, metavar=("A","B"))
    parser.add_argument("--cost", type=str)
    parser.add_argument("--annotate", type=str)
    args = parser.parse_args()
    cfg = load_config()
    if args.list_runs: cmd_list_runs(cfg); return
    if args.stats: cmd_stats(args.stats, cfg); return
    if args.compare: cmd_compare(args.compare[0], args.compare[1], cfg); return
    if args.cost: cmd_cost(args.cost, cfg); return
    if args.annotate: cmd_annotate(args.annotate, cfg); return
    api_key = os.environ.get("GOOGLE_API_KEY","")
    if not api_key: print("ERROR: Set GOOGLE_API_KEY"); sys.exit(1)
    if args.calibrate: asyncio.run(cmd_calibrate(api_key, cfg, args.set_baseline)); return
    asyncio.run(cmd_run(args, cfg, api_key))

if __name__ == "__main__":
    main()
