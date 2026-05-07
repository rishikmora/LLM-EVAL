"""
Module 05 — Production Streamlit Dashboard (70 LPA Edition)
Tabs: Overview · Metrics · Red-Team · Calibration · Annotation · Cost · Agentic · Raw
Run: streamlit run modules/dashboard.py
"""
import json, sqlite3, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import yaml

from modules.tracker import load_run_results, list_runs, compute_run_stats, compare_runs, check_thresholds

DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG_PATH = Path(__file__).parent.parent / "config" / "eval_config.yaml"

st.set_page_config(page_title="LLM Eval Dashboard", page_icon="🔬", layout="wide")
st.markdown("""<style>
.stApp{background:#0A0E1A;color:#E8EAF0}
h1,h2,h3{color:#00FFB3!important}
div[data-testid="metric-container"]{background:rgba(0,255,179,0.05);border:1px solid rgba(0,255,179,0.2);border-radius:8px;padding:8px}
.stTabs [data-baseweb="tab"]{color:#888}
.stTabs [aria-selected="true"]{color:#00FFB3!important}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_config():
    with open(CONFIG_PATH) as f: return yaml.safe_load(f)

@st.cache_data(ttl=30)
def get_all_results(db_path):
    path = Path(db_path)
    if not path.exists(): return {}
    conn = sqlite3.connect(path); conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM eval_results ORDER BY timestamp DESC").fetchall()
    conn.close()
    res = {}
    for row in [dict(r) for r in rows]:
        res.setdefault(row["run_id"],[]).append(row)
    return res

def mcol(name):
    return {"faithfulness":"#00FFB3","relevance":"#00C8FF","toxicity":"#FF4466",
            "bias":"#FFB800","asr":"#FF6B35","latency_ms":"#34D399","token_efficiency":"#F472B6"}.get(name,"#888")

def thresh_status(metric, value, config):
    t = config["thresholds"]
    checks = {"faithfulness":(t["faithfulness_min"],"min"),"relevance":(t["relevance_min"],"min"),
              "toxicity":(t["toxicity_max"],"max"),"bias":(t["bias_max"],"max"),"asr":(t["asr_max"],"max")}
    if metric not in checks: return "◆","#888"
    threshold, op = checks[metric]
    ok = value >= threshold if op=="min" else value <= threshold
    return ("✓","#00FFB3") if ok else ("✗","#FF4466")

def main():
    cfg = load_config()
    db_path = cfg["database"]["path"]
    ann_db = DATA_DIR / cfg["database"].get("annotation_db","annotations.db")
    cal_db = DATA_DIR / cfg["database"].get("calibration_db","calibration.db")
    cost_db = DATA_DIR / cfg["database"].get("cost_db","costs.db")

    with st.sidebar:
        st.markdown("## 🔬 LLM Eval")
        st.markdown(f"**Model:** `{cfg['model']['target']}`")
        st.markdown(f"**Judge:** `{cfg['model']['judge']}`")
        st.markdown("---")
        all_results = get_all_results(db_path)
        runs = sorted(all_results.keys(), reverse=True)
        if not runs:
            st.warning("No runs found.\nRun `python run_eval.py --suite adversarial --n 5`")
            st.stop()
        selected_run = st.selectbox("Select Run", runs)
        st.markdown("---")
        compare_mode = st.checkbox("Compare with another run")
        compare_run = None
        if compare_mode and len(runs) > 1:
            compare_run = st.selectbox("Compare against", [r for r in runs if r != selected_run])
        st.markdown("---")
        annotator_id = st.text_input("Annotator ID", "analyst_1")
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear(); st.rerun()
        st.markdown(f"**Runs:** {len(runs)}  **Total evals:** {sum(len(v) for v in all_results.values())}")

    results = all_results.get(selected_run, [])
    if not results: st.error(f"No results: {selected_run}"); st.stop()

    stats = compute_run_stats(results)
    violations = check_thresholds(stats, cfg)

    # Header
    st.markdown("# LLM Eval & Red-Teaming Dashboard")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(f"**Run:** `{selected_run[:20]}`")
    c2.markdown(f"**Prompts:** {len(results)}")
    c3.markdown(f"**Errors:** {sum(1 for r in results if r.get('error'))}")
    c4.markdown(f"**Violations:** {'🔴 '+str(len(violations)) if violations else '🟢 0'}")
    asr_clf = stats.get("asr_classifier",{})
    c5.markdown(f"**ROC-AUC:** {asr_clf.get('roc_auc','N/A')}")

    if violations:
        with st.expander(f"⚠ {len(violations)} Threshold Violation(s)", expanded=True):
            for v in violations:
                color = "#FF4466" if v["severity"]=="CRITICAL" else "#FFB800"
                ci_low = v.get('ci_low') or 0; ci_high = v.get('ci_high') or 0
                st.markdown(f"<span style='color:{color}'>**[{v['severity']}]** {v['metric'].upper()} = {v['value']:.3f}  CI=[{ci_low:.3f},{ci_high:.3f}]  (threshold: {'≥' if v['op']=='min' else '≤'} {v['threshold']})</span>", unsafe_allow_html=True)

    st.markdown("---")

    tabs = st.tabs(["📊 Metrics","🔴 Red-Team","📈 Trends","🔬 Comparison","🎯 Calibration","🏷️ Annotation","💰 Cost","🤖 Agentic","📋 Raw"])
    tab_metrics, tab_redteam, tab_trends, tab_compare, tab_cal, tab_ann, tab_cost, tab_agentic, tab_raw = tabs

    # ── Tab 1: Metrics ────────────────────────────────────────────────────
    with tab_metrics:
        st.markdown("### Core Metrics (95% Bootstrap CI)")
        cols = st.columns(4)
        for i, metric in enumerate(["faithfulness","relevance","toxicity","bias","asr","token_efficiency"]):
            if metric not in stats: continue
            s = stats[metric]; emoji,color = thresh_status(metric, s["mean"], cfg)
            with cols[i%4]:
                st.metric(f"{emoji} {metric.replace('_',' ').title()}", f"{s['mean']:.3f}",
                          f"CI [{s['ci_low']:.3f},{s['ci_high']:.3f}]", delta_color="off")
        lat_cols = st.columns(3)
        for col,key in zip(lat_cols,["latency_p50","latency_p95","latency_p99"]):
            if key in stats: col.metric(key.replace("_"," ").upper(), f"{stats[key]:.0f}ms")

        if asr_clf.get("roc_auc") is not None:
            st.markdown("### ASR Classifier Performance")
            clf_cols = st.columns(4)
            clf_cols[0].metric("ROC-AUC", f"{asr_clf.get('roc_auc',0):.4f}", asr_clf.get('auc_interpretation',''))
            clf_cols[1].metric("F1", f"{asr_clf.get('f1',0):.4f}")
            clf_cols[2].metric("Precision", f"{asr_clf.get('precision',0):.4f}")
            clf_cols[3].metric("Recall", f"{asr_clf.get('recall',0):.4f}")

        df = pd.DataFrame(results)
        col_a, col_b = st.columns(2)
        with col_a:
            metric_sel = st.selectbox("Distribution", ["relevance","toxicity","bias","asr","faithfulness"])
            vals = df[metric_sel].dropna()
            if len(vals) > 0:
                fig = go.Figure(go.Histogram(x=vals, nbinsx=20, marker_color=mcol(metric_sel), opacity=0.8))
                if metric_sel in stats:
                    s = stats[metric_sel]
                    fig.add_vline(x=s["mean"], line_color="white", line_dash="dash", annotation_text=f"μ={s['mean']:.3f}")
                    fig.add_vrect(x0=s["ci_low"],x1=s["ci_high"], fillcolor="rgba(255,255,255,0.05)", line_width=0)
                fig.update_layout(template="plotly_dark",paper_bgcolor="#0A0E1A",plot_bgcolor="#0A0E1A",
                                  title=f"{metric_sel} Distribution",height=300,margin=dict(t=40,b=20))
                st.plotly_chart(fig, use_container_width=True)
        with col_b:
            lats = df["latency_ms"].dropna()
            if len(lats) > 0:
                fig2 = go.Figure(go.Histogram(x=lats, nbinsx=20, marker_color="#34D399", opacity=0.8))
                if "latency_p95" in stats:
                    fig2.add_vline(x=stats["latency_p95"], line_color="#FF4466", annotation_text=f"P95={stats['latency_p95']:.0f}ms")
                fig2.update_layout(template="plotly_dark",paper_bgcolor="#0A0E1A",plot_bgcolor="#0A0E1A",
                                   title="Latency Distribution (ms)",height=300,margin=dict(t=40,b=20))
                st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 2: Red-Team ───────────────────────────────────────────────────
    with tab_redteam:
        df = pd.DataFrame(results)
        st.markdown("#### ASR by Category")
        asr_by_cat = stats.get("asr_by_category",{})
        if asr_by_cat:
            cats = list(asr_by_cat.keys()); vals = list(asr_by_cat.values())
            fig_asr = go.Figure(go.Bar(x=cats,y=vals,
                marker_color=["#FF4466" if v>0.05 else "#00FFB3" for v in vals],
                text=[f"{v:.1%}" for v in vals], textposition="outside"))
            fig_asr.add_hline(y=0.05, line_color="yellow", line_dash="dash", annotation_text="5% threshold")
            fig_asr.update_layout(template="plotly_dark",paper_bgcolor="#0A0E1A",plot_bgcolor="#0A0E1A",height=320)
            st.plotly_chart(fig_asr, use_container_width=True)

        # ROC curve
        if asr_clf.get("roc_auc") is not None and asr_clf.get("roc_auc") is not None:
            col_roc, col_tox = st.columns(2)
            with col_roc:
                st.markdown(f"#### ASR ROC Curve (AUC={asr_clf.get('roc_auc',0):.4f})")
                adv = [r for r in results if r.get("priority") in ("CRITICAL","HIGH","MEDIUM")]
                if len(adv) >= 5:
                    from modules.statistical_analysis import compute_roc_auc
                    y_true = [1 if (r.get("asr") or 0)>=0.5 else 0 for r in adv]
                    y_score = [r.get("asr") or 0 for r in adv]
                    if 0 < sum(y_true) < len(y_true):
                        roc = compute_roc_auc(y_true, y_score)
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(x=roc["fpr"],y=roc["tpr"],mode="lines",line=dict(color="#00FFB3",width=2),name=f"ROC AUC={roc['auc']:.4f}"))
                        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(color="#555",dash="dash"),name="Random"))
                        fig_roc.update_layout(template="plotly_dark",paper_bgcolor="#0A0E1A",plot_bgcolor="#0A0E1A",
                                              xaxis_title="FPR",yaxis_title="TPR",height=300)
                        st.plotly_chart(fig_roc, use_container_width=True)
            with col_tox:
                st.markdown("#### Toxicity by Category")
                tox = df.groupby("category")["toxicity"].mean().dropna().sort_values(ascending=False)
                if len(tox)>0:
                    fig_tox = go.Figure(go.Bar(x=tox.index,y=tox.values,
                        marker_color=["#FF4466" if v>0.10 else "#FFB800" if v>0.05 else "#00FFB3" for v in tox.values]))
                    fig_tox.add_hline(y=0.10,line_color="red",line_dash="dash")
                    fig_tox.update_layout(template="plotly_dark",paper_bgcolor="#0A0E1A",plot_bgcolor="#0A0E1A",height=300)
                    st.plotly_chart(fig_tox, use_container_width=True)

        adv_df = df[df["priority"].isin(["CRITICAL","HIGH","MEDIUM"])]
        if not adv_df.empty:
            st.markdown("#### Adversarial Results")
            disp = ["category","priority","asr","toxicity","bias","asr_reason"]
            avail = [c for c in disp if c in adv_df.columns]
            st.dataframe(adv_df[avail].sort_values("asr",ascending=False).head(50), use_container_width=True)

    # ── Tab 3: Trends ─────────────────────────────────────────────────────
    with tab_trends:
        all_res = get_all_results(db_path)
        run_stats_all = {rid: compute_run_stats(res) for rid,res in sorted(all_res.items())}
        if len(run_stats_all) > 1:
            metrics_sel = st.multiselect("Metrics",["faithfulness","relevance","toxicity","bias","asr"],
                                          default=["relevance","asr","toxicity"])
            fig_t = go.Figure()
            run_ids = sorted(run_stats_all.keys())
            for metric in metrics_sel:
                ms = [run_stats_all[r].get(metric,{}).get("mean") for r in run_ids]
                ls = [run_stats_all[r].get(metric,{}).get("ci_low") for r in run_ids]
                hs = [run_stats_all[r].get(metric,{}).get("ci_high") for r in run_ids]
                valid = [(r,m,l,h) for r,m,l,h in zip(run_ids,ms,ls,hs) if m is not None]
                if valid:
                    xs,ms2,ls2,hs2 = zip(*valid); c = mcol(metric)
                    fig_t.add_trace(go.Scatter(x=list(xs),y=list(ms2),name=metric,line=dict(color=c,width=2),mode="lines+markers"))
                    r,g,b = int(c[1:3],16),int(c[3:5],16),int(c[5:7],16)
                    fig_t.add_trace(go.Scatter(x=list(xs)+list(xs)[::-1],y=list(hs2)+list(ls2)[::-1],
                        fill="toself",fillcolor=f"rgba({r},{g},{b},0.12)",line=dict(color="rgba(255,255,255,0)"),showlegend=False))
            fig_t.update_layout(template="plotly_dark",paper_bgcolor="#0A0E1A",plot_bgcolor="#0A0E1A",
                                title="Metric Trends (95% CI bands)",height=400,xaxis_tickangle=-45)
            st.plotly_chart(fig_t, use_container_width=True)

            if asr_clf.get("roc_auc") is not None:
                roc_aucs = [run_stats_all[r].get("asr_classifier",{}).get("roc_auc") for r in run_ids]
                valid_roc = [(r,v) for r,v in zip(run_ids,roc_aucs) if v is not None]
                if len(valid_roc) > 1:
                    xr,yr = zip(*valid_roc)
                    fig_roc_t = go.Figure(go.Scatter(x=list(xr),y=list(yr),mode="lines+markers",line=dict(color="#FF6B35",width=2)))
                    fig_roc_t.update_layout(template="plotly_dark",paper_bgcolor="#0A0E1A",plot_bgcolor="#0A0E1A",
                                            title="ASR ROC-AUC Trend",height=280,yaxis_range=[0,1])
                    st.plotly_chart(fig_roc_t, use_container_width=True)
        else:
            st.info("Need ≥ 2 runs for trend analysis.")

    # ── Tab 4: Comparison ─────────────────────────────────────────────────
    with tab_compare:
        if compare_run and compare_run in all_results:
            res_b = all_results[compare_run]; stats_b = compute_run_stats(res_b)
            comparison = compare_runs(stats, stats_b, results, res_b)
            st.markdown(f"**A:** `{selected_run}` vs **B:** `{compare_run}`")
            rows = []
            for m,c in comparison.items():
                rows.append({"Metric":m,"Δ (B-A)":f"{c['delta']:+.4f}","Direction":c["direction"],
                             "p-value":c["p_value"],"Significant":"✓" if c["significant"] else "–",
                             "Alert":"⚠" if c.get("should_alert") else ""})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            metrics_r = ["faithfulness","relevance","toxicity","bias","asr"]
            va = [stats.get(m,{}).get("mean",0) for m in metrics_r]
            vb = [stats_b.get(m,{}).get("mean",0) for m in metrics_r]
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(r=va,theta=metrics_r,fill="toself",name=selected_run[:20],line_color="#00FFB3"))
            fig_r.add_trace(go.Scatterpolar(r=vb,theta=metrics_r,fill="toself",name=compare_run[:20],line_color="#00C8FF",opacity=0.7))
            fig_r.update_layout(template="plotly_dark",paper_bgcolor="#0A0E1A",polar=dict(bgcolor="#0A0E1A"),height=380)
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("Enable 'Compare with another run' in the sidebar.")

    # ── Tab 5: Calibration ────────────────────────────────────────────────
    with tab_cal:
        st.markdown("### Judge Calibration Status")
        if not cal_db.exists():
            st.warning("No calibration data yet.\nRun: `python run_eval.py --calibrate`\nTo set baseline: `python run_eval.py --calibrate --set-baseline`")
        else:
            conn = sqlite3.connect(cal_db); conn.row_factory = sqlite3.Row
            runs_cal = conn.execute("SELECT DISTINCT run_timestamp FROM calibration_runs ORDER BY run_timestamp DESC LIMIT 10").fetchall()
            conn.close()
            if not runs_cal:
                st.info("No calibration runs found. Run `python run_eval.py --calibrate --set-baseline` first.")
            else:
                latest_ts = runs_cal[0][0]
                conn = sqlite3.connect(cal_db); conn.row_factory = sqlite3.Row
                latest = conn.execute("SELECT * FROM calibration_runs WHERE run_timestamp=?",(latest_ts,)).fetchall()
                baselines = conn.execute("SELECT * FROM calibration_baselines ORDER BY created_at DESC").fetchall()
                conn.close()
                latest_df = pd.DataFrame([dict(r) for r in latest])
                st.markdown(f"**Latest calibration:** `{latest_ts[:19]}`")
                if not latest_df.empty:
                    summary = latest_df.groupby("metric").agg(
                        mean_score=("actual_score","mean"),
                        in_range_pct=("in_range","mean"),
                        n=("actual_score","count")
                    ).reset_index()
                    baselines_df = pd.DataFrame([dict(r) for r in baselines])
                    if not baselines_df.empty:
                        latest_baseline = baselines_df.sort_values("created_at").groupby("metric").last().reset_index()
                        summary = summary.merge(latest_baseline[["metric","baseline_mean"]], on="metric", how="left")
                        summary["drift"] = (summary["mean_score"] - summary["baseline_mean"]).abs().round(4)
                        summary["status"] = summary["drift"].apply(lambda d: "⚠ DRIFT" if d>0.05 else "✓ OK" if pd.notna(d) else "No baseline")
                    st.dataframe(summary, use_container_width=True)
                if not baselines_df.empty if 'baselines_df' in dir() else False:
                    fig_cal = go.Figure()
                    for metric in latest_df["metric"].unique():
                        m_data = latest_df[latest_df["metric"]==metric]
                        fig_cal.add_trace(go.Box(y=m_data["actual_score"],name=metric,marker_color=mcol(metric)))
                    fig_cal.update_layout(template="plotly_dark",paper_bgcolor="#0A0E1A",plot_bgcolor="#0A0E1A",
                                          title="Score Distribution on Calibration Benchmark",height=350)
                    st.plotly_chart(fig_cal, use_container_width=True)
        st.markdown("---")
        st.markdown("**Commands:**")
        st.code("python run_eval.py --calibrate             # Check for drift\npython run_eval.py --calibrate --set-baseline  # Record new baseline")

    # ── Tab 6: Annotation ─────────────────────────────────────────────────
    with tab_ann:
        st.markdown("### Human Annotation Queue")
        from modules.annotation import render_annotation_ui, compute_agreement_report
        render_annotation_ui(ann_db, annotator_id)
        st.markdown("---")
        st.markdown("### Human-LLM Agreement Report")
        report = compute_agreement_report(ann_db)
        if report.get("status") in ("no_data","no_annotations"):
            st.info("No annotations yet. Use the queue above to annotate responses.")
        else:
            st.metric("Total Annotations", report.get("n_annotations",0))
            if report.get("metrics"):
                rows = []
                for m,s in report["metrics"].items():
                    rows.append({"Metric":m,"Cohen's Kappa":s["kappa"],"Interpretation":s["kappa_interpretation"],
                                 "Mean Δ":s["mean_delta"],"Disagreements":s["high_disagreement_count"]})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                kappas = {m: s["kappa"] for m,s in report["metrics"].items()}
                if kappas:
                    fig_k = go.Figure(go.Bar(x=list(kappas.keys()),y=list(kappas.values()),
                        marker_color=["#00FFB3" if v>0.6 else "#FFB800" if v>0.4 else "#FF4466" for v in kappas.values()]))
                    fig_k.add_hline(y=0.6,line_dash="dash",line_color="#00FFB3",annotation_text="Strong agreement")
                    fig_k.add_hline(y=0.4,line_dash="dash",line_color="#FFB800",annotation_text="Moderate")
                    fig_k.update_layout(template="plotly_dark",paper_bgcolor="#0A0E1A",plot_bgcolor="#0A0E1A",
                                        title="Cohen's Kappa (Human vs LLM Judge)",height=300,yaxis_range=[-1,1])
                    st.plotly_chart(fig_k, use_container_width=True)
            if report.get("disagreements"):
                st.markdown("**High-Disagreement Cases (delta > 0.3):**")
                st.dataframe(pd.DataFrame(report["disagreements"]), use_container_width=True)

    # ── Tab 7: Cost Intelligence ──────────────────────────────────────────
    with tab_cost:
        st.markdown("### Cost Intelligence")
        from modules.cost_intelligence import compute_cost_analytics, get_daily_usage, check_budget
        analytics = compute_cost_analytics(results, cfg["model"]["target"])
        if analytics:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Cost (USD)", f"${analytics['total_cost_usd']:.6f}")
            c2.metric("Total Cost (INR)", f"₹{analytics['total_cost_inr']:.4f}")
            c3.metric("Cost/Prompt", f"${analytics['cost_per_prompt_usd']:.8f}")
            c4.metric("Free Tier", "✓ In limit" if analytics["within_free_tier"] else "✗ Exceeded")
            budget = check_budget(cost_db, cfg["model"]["target"], cfg)
            st.progress(min(budget["pct_used"], 1.0), text=f"Daily budget: {budget['pct_used']:.1%} used ({budget['used']:,}/{budget['budget']:,} tokens)")
            cats_cost = analytics.get("category_breakdown",{})
            if cats_cost:
                burn_data = analytics.get("token_burn_by_category",{})
                fig_burn = go.Figure(go.Bar(x=list(burn_data.keys()),y=list(burn_data.values()),
                    marker_color="#00C8FF",text=[f"{v:,}" for v in burn_data.values()],textposition="outside"))
                fig_burn.update_layout(template="plotly_dark",paper_bgcolor="#0A0E1A",plot_bgcolor="#0A0E1A",
                                       title="Token Burn by Category",height=320)
                st.plotly_chart(fig_burn, use_container_width=True)
            st.markdown("**Category Cost Breakdown:**")
            cost_rows = [{"Category":cat,"Prompts":d["prompt_count"],"Input Tokens":d["input_tokens"],
                          "Cost USD":f"${d['cost_usd']:.6f}","Cost INR":f"₹{d['cost_inr']:.4f}"}
                         for cat,d in cats_cost.items()]
            st.dataframe(pd.DataFrame(cost_rows).sort_values("Cost USD",ascending=False), use_container_width=True)
            roi_c1, roi_c2 = st.columns(2)
            roi_c1.metric("Quality ROI (relevance/$)", f"{analytics['roi_quality_per_dollar']:.1f}")
            roi_c2.metric("Safety ROI (1-ASR/$)", f"{analytics['roi_safety_per_dollar']:.1f}")

    # ── Tab 8: Agentic ────────────────────────────────────────────────────
    with tab_agentic:
        st.markdown("### Agentic Evaluation Results")
        agentic_files = sorted(DATA_DIR.glob("agentic_*.json"), reverse=True)
        if not agentic_files:
            st.info("No agentic evaluation results yet.\nRun: `python run_eval.py --suite agentic`\nOr set `run_agentic_eval: true` in eval_config.yaml")
            st.markdown("**Agentic metrics that will appear here:**")
            for m in ["Tool Selection Accuracy","Parameter Correctness","Planning Success Rate",
                      "Memory Retention Rate","CoT Consistency Rate","Hallucination-Free Rate"]:
                st.markdown(f"- {m}")
        else:
            with open(agentic_files[0]) as f: ag = json.load(f)
            summary = ag.get("summary",{})
            st.markdown(f"**Latest:** `{agentic_files[0].name}`")
            metric_map = {
                "tool_selection_accuracy": "Tool Selection Accuracy",
                "param_correctness": "Param Correctness",
                "planning_success_rate": "Planning Success Rate",
                "memory_retention_rate": "Memory Retention",
                "cot_consistency_rate": "CoT Consistency",
                "hallucination_free_rate": "Hallucination-Free Rate",
            }
            ag_cols = st.columns(3)
            for i,(k,label) in enumerate(metric_map.items()):
                val = summary.get(k)
                if val is not None:
                    ag_cols[i%3].metric(label, f"{val:.4f}" if isinstance(val,float) else str(val))
            results_ag = ag.get("results",[])
            if results_ag:
                fig_ag = go.Figure(go.Bar(
                    x=[r.get("test_id","") for r in results_ag],
                    y=[1 if r.get("tool_correct") or r.get("memory_retained") or r.get("cot_consistent") or r.get("planning_score",0)>0.5 else 0 for r in results_ag],
                    marker_color=["#00FFB3" if r.get("tool_correct") or r.get("memory_retained") or r.get("answer_correct") else "#FF4466" for r in results_ag],
                ))
                fig_ag.update_layout(template="plotly_dark",paper_bgcolor="#0A0E1A",plot_bgcolor="#0A0E1A",
                                     title="Agentic Test Results (1=pass, 0=fail)",height=300,yaxis_range=[0,1.2])
                st.plotly_chart(fig_ag, use_container_width=True)
                st.dataframe(pd.DataFrame(results_ag)[["test_id","category","error"] if "error" in pd.DataFrame(results_ag).columns else ["test_id","category"]].head(20), use_container_width=True)

    # ── Tab 9: Raw ────────────────────────────────────────────────────────
    with tab_raw:
        df = pd.DataFrame(results)
        c1,c2,c3 = st.columns(3)
        cat_filter = c1.multiselect("Category", sorted(df["category"].unique()) if "category" in df else [], default=[])
        errors_only = c2.checkbox("Errors only")
        asr_only = c3.checkbox("ASR > 0 only")
        filtered = df.copy()
        if cat_filter: filtered = filtered[filtered["category"].isin(cat_filter)]
        if errors_only: filtered = filtered[filtered.get("error","").notna() if "error" in filtered else filtered.index == -1]
        if asr_only and "asr" in filtered: filtered = filtered[filtered["asr"] > 0]
        display_cols = ["prompt_id","category","priority","relevance","toxicity","bias","asr","faithfulness","latency_ms","error"]
        avail = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[avail].reset_index(drop=True), use_container_width=True, height=450)
        if st.button("📥 Export CSV"):
            st.download_button("Download", filtered.to_csv(index=False), f"{selected_run}.csv", "text/csv")

if __name__ == "__main__":
    main()
