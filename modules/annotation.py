"""
Module 07 — Human Annotation Pipeline (Tier 1 Enhancement)

Implements:
  - Annotation task queue (samples from eval DB)
  - DB storage for human labels
  - Cohen's Kappa between human and LLM judge
  - Disagreement detection (delta > 0.3)
  - Streamlit annotation UI (importable widget)
"""

import json
import math
import sqlite3
import statistics
from pathlib import Path
from datetime import datetime
from typing import Optional
import yaml

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config" / "eval_config.yaml"
DATA_DIR = ROOT / "data"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─── Annotation DB ────────────────────────────────────────────────────────────

def init_annotation_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS annotation_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            prompt_id TEXT,
            category TEXT,
            prompt_text TEXT,
            model_response TEXT,
            llm_relevance REAL,
            llm_toxicity REAL,
            llm_bias REAL,
            llm_asr REAL,
            status TEXT DEFAULT 'pending',
            assigned_to TEXT,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS human_annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER,
            annotator_id TEXT,
            relevance_score REAL,
            toxicity_score REAL,
            bias_score REAL,
            asr_score REAL,
            overall_quality INTEGER,
            notes TEXT,
            annotated_at TEXT,
            FOREIGN KEY (task_id) REFERENCES annotation_tasks(id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_task_status ON annotation_tasks(status)")
    conn.commit()
    conn.close()


# ─── Task queue management ────────────────────────────────────────────────────

def create_annotation_tasks(
    run_id: str,
    eval_db_path: Path,
    annotation_db_path: Path,
    sample_size: int = 20,
    focus_categories: Optional[list[str]] = None,
) -> int:
    """Sample eval results and create annotation tasks. Returns count created."""
    init_annotation_db(annotation_db_path)

    eval_conn = sqlite3.connect(eval_db_path)
    eval_conn.row_factory = sqlite3.Row

    query = "SELECT * FROM eval_results WHERE run_id = ? AND error IS NULL"
    params = [run_id]
    if focus_categories:
        placeholders = ",".join("?" * len(focus_categories))
        query += f" AND category IN ({placeholders})"
        params.extend(focus_categories)
    query += " ORDER BY RANDOM() LIMIT ?"
    params.append(sample_size)

    rows = eval_conn.execute(query, params).fetchall()
    eval_conn.close()

    ann_conn = sqlite3.connect(annotation_db_path)
    created = 0
    for row in rows:
        r = dict(row)
        # Check not already queued
        existing = ann_conn.execute(
            "SELECT id FROM annotation_tasks WHERE run_id=? AND prompt_id=?",
            (r["run_id"], r["prompt_id"])
        ).fetchone()
        if existing:
            continue
        ann_conn.execute("""
            INSERT INTO annotation_tasks (
                run_id, prompt_id, category, prompt_text, model_response,
                llm_relevance, llm_toxicity, llm_bias, llm_asr, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
        """, (
            r["run_id"], r["prompt_id"], r["category"],
            r.get("prompt_text", ""), r.get("response", ""),
            r.get("relevance"), r.get("toxicity"), r.get("bias"), r.get("asr"),
            datetime.utcnow().isoformat(),
        ))
        created += 1
    ann_conn.commit()
    ann_conn.close()
    return created


def get_pending_tasks(annotation_db_path: Path, limit: int = 5) -> list[dict]:
    if not annotation_db_path.exists():
        return []
    conn = sqlite3.connect(annotation_db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM annotation_tasks WHERE status='pending' ORDER BY created_at ASC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_annotation(
    annotation_db_path: Path,
    task_id: int,
    annotator_id: str,
    relevance: float,
    toxicity: float,
    bias: float,
    asr: float,
    overall_quality: int,
    notes: str = "",
):
    conn = sqlite3.connect(annotation_db_path)
    conn.execute("""
        INSERT INTO human_annotations (
            task_id, annotator_id, relevance_score, toxicity_score,
            bias_score, asr_score, overall_quality, notes, annotated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (task_id, annotator_id, relevance, toxicity, bias, asr,
          overall_quality, notes, datetime.utcnow().isoformat()))
    conn.execute(
        "UPDATE annotation_tasks SET status='completed', assigned_to=? WHERE id=?",
        (annotator_id, task_id)
    )
    conn.commit()
    conn.close()


# ─── Agreement metrics ────────────────────────────────────────────────────────

def cohen_kappa_continuous(
    human_scores: list[float],
    llm_scores: list[float],
    bins: int = 5,
) -> float:
    """
    Cohen's Kappa adapted for continuous scores by binning into ordinal categories.
    Bins: [0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
    """
    if len(human_scores) != len(llm_scores) or len(human_scores) < 2:
        return 0.0

    def to_bin(v: float) -> int:
        return min(int(v * bins), bins - 1)

    human_bins = [to_bin(s) for s in human_scores]
    llm_bins = [to_bin(s) for s in llm_scores]

    n = len(human_bins)
    # Observed agreement
    p_o = sum(1 for h, l in zip(human_bins, llm_bins) if h == l) / n

    # Expected agreement
    human_dist = [human_bins.count(i) / n for i in range(bins)]
    llm_dist = [llm_bins.count(i) / n for i in range(bins)]
    p_e = sum(h * l for h, l in zip(human_dist, llm_dist))

    if p_e == 1.0:
        return 1.0
    return round((p_o - p_e) / (1 - p_e), 4)


def compute_agreement_report(annotation_db_path: Path) -> dict:
    """Compute human-LLM agreement across all completed annotations."""
    if not annotation_db_path.exists():
        return {"status": "no_data"}

    conn = sqlite3.connect(annotation_db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT h.relevance_score, h.toxicity_score, h.bias_score, h.asr_score,
               t.llm_relevance, t.llm_toxicity, t.llm_bias, t.llm_asr,
               t.category, t.prompt_id
        FROM human_annotations h
        JOIN annotation_tasks t ON h.task_id = t.id
        WHERE t.status = 'completed'
    """).fetchall()
    conn.close()

    if not rows:
        return {"status": "no_annotations"}

    data = [dict(r) for r in rows]

    metrics = [
        ("relevance", "relevance_score", "llm_relevance"),
        ("toxicity", "toxicity_score", "llm_toxicity"),
        ("bias", "bias_score", "llm_bias"),
        ("asr", "asr_score", "llm_asr"),
    ]

    report = {"n_annotations": len(data), "metrics": {}, "disagreements": []}

    for metric, human_col, llm_col in metrics:
        pairs = [
            (r[human_col], r[llm_col])
            for r in data
            if r[human_col] is not None and r[llm_col] is not None
        ]
        if not pairs:
            continue

        human_vals, llm_vals = zip(*pairs)
        kappa = cohen_kappa_continuous(list(human_vals), list(llm_vals))
        deltas = [abs(h - l) for h, l in pairs]
        mean_delta = statistics.mean(deltas)
        disagreements = [(r, d) for r, d in zip(data, deltas) if d > 0.3]

        report["metrics"][metric] = {
            "kappa": kappa,
            "kappa_interpretation": (
                "strong" if kappa > 0.6 else
                "moderate" if kappa > 0.4 else
                "weak"
            ),
            "mean_delta": round(mean_delta, 4),
            "n_pairs": len(pairs),
            "high_disagreement_count": len(disagreements),
        }

        for r, delta in disagreements[:5]:  # top 5 disagreements
            report["disagreements"].append({
                "metric": metric,
                "prompt_id": r["prompt_id"],
                "category": r["category"],
                "human_score": r[human_col],
                "llm_score": r[llm_col],
                "delta": round(delta, 4),
            })

    return report


# ─── Streamlit annotation widget (import in dashboard) ───────────────────────

def render_annotation_ui(annotation_db_path: Path, annotator_id: str = "anonymous"):
    """Render the annotation interface. Call from within a Streamlit app."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not available")
        return

    st.markdown("### 🏷️ Human Annotation Queue")

    tasks = get_pending_tasks(annotation_db_path, limit=3)
    if not tasks:
        st.success("✓ No pending annotation tasks.")
        return

    st.markdown(f"**{len(tasks)} task(s) pending.** Annotate a sample:")

    task = tasks[0]
    with st.expander(f"Task #{task['id']} — {task['category']}", expanded=True):
        st.markdown(f"**Prompt:** {task['prompt_text'][:300]}")
        st.markdown(f"**Model Response:** {task['model_response'][:500]}")
        st.markdown("---")
        st.markdown("**LLM Judge scores (for reference):**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Relevance", f"{task['llm_relevance']:.2f}" if task['llm_relevance'] else "N/A")
        col2.metric("Toxicity", f"{task['llm_toxicity']:.2f}" if task['llm_toxicity'] else "N/A")
        col3.metric("Bias", f"{task['llm_bias']:.2f}" if task['llm_bias'] else "N/A")
        col4.metric("ASR", f"{task['llm_asr']:.2f}" if task['llm_asr'] else "N/A")

        st.markdown("**Your scores (0.0 = best/safe, 1.0 = worst/harmful for safety metrics):**")
        c1, c2, c3, c4 = st.columns(4)
        h_relevance = c1.slider("Relevance", 0.0, 1.0, 0.8, 0.05, key=f"rel_{task['id']}")
        h_toxicity = c2.slider("Toxicity", 0.0, 1.0, 0.0, 0.05, key=f"tox_{task['id']}")
        h_bias = c3.slider("Bias", 0.0, 1.0, 0.0, 0.05, key=f"bias_{task['id']}")
        h_asr = c4.slider("ASR", 0.0, 1.0, 0.0, 0.05, key=f"asr_{task['id']}")
        quality = st.radio("Overall quality", [1, 2, 3, 4, 5], horizontal=True, index=3,
                           key=f"qual_{task['id']}")
        notes = st.text_area("Notes (optional)", key=f"notes_{task['id']}")

        if st.button("✓ Submit Annotation", key=f"submit_{task['id']}"):
            save_annotation(
                annotation_db_path, task["id"], annotator_id,
                h_relevance, h_toxicity, h_bias, h_asr, quality, notes
            )
            st.success("Annotation saved!")
            st.rerun()
