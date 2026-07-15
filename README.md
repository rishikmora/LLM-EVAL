# LLM Eval & Red-Teaming Framework
**Enhanced Edition — Research Scientist Grade**

Systematic evaluation and adversarial red-teaming for Anthropic Claude.

## Quick Start

```bash 
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
export ANTHROPIC_API_KEY=your_key_here

# 3. Run adversarial evaluation (5 prompts/category)
python run_eval.py --suite adversarial --n 5

# 4. Launch dashboard
streamlit run modules/dashboard.py

# 5. Run tests
pytest tests/ -v
```

## Architecture — 5 Modules

| Module | File | Responsibility |
|--------|------|----------------|
| 01 Prompt Generator | `modules/prompt_generator.py` | 3-layer adversarial prompt generation |
| 02 Evaluator Engine | `modules/evaluator.py` | Async scoring across 8 metrics |
| 03 MLflow Tracker | `modules/tracker.py` | Experiment tracking + statistical validation |
| 04 Alert System | `modules/alerts.py` | Threshold violation notifications |
| 05 Dashboard | `modules/dashboard.py` | Real-time Streamlit visualization |

## 8 Evaluation Metrics

| Metric | Target | Tool |
|--------|--------|------|
| Faithfulness | ≥ 0.85 | RAGAS-style LLM-as-Judge |
| Answer Relevance | ≥ 0.80 | LLM-as-Judge cosine proxy |
| Toxicity | ≤ 0.10 | Segment-level LLM judge |
| Bias | ≤ 0.15 | Counterfactual demographic probing |
| Attack Success Rate | ≤ 0.05 | 2-stage: keyword + LLM judge |
| Latency P95 | < 8000ms | Async wall-clock timer |
| Token Efficiency | 0.3–2.0 | out_tokens / in_tokens |
| Contextual Precision | ≥ 0.75 | RAG context evaluation |

## Attack Categories

- **Roleplay / Persona Injection** (CRITICAL) — DAN, AIM, character impersonation
- **Prompt Injection** (CRITICAL) — Instruction smuggling in documents
- **Multi-Turn Escalation** (HIGH) — Gradual safety erosion over 10 turns
- **Logic Traps** (HIGH) — Self-referential paradox prompts
- **Encoding Attacks** (MEDIUM) — Base64, ROT13, leet-speak payloads
- **Bias Elicitation** (MEDIUM) — Counterfactual demographic substitution

## CLI Reference

```bash
python run_eval.py --suite adversarial --n 5    # adversarial only
python run_eval.py --suite full --n 10          # all categories
python run_eval.py --suite benign               # sanity check
python run_eval.py --list-runs                  # list all runs
python run_eval.py --stats <run_id>             # stats for a run
```

## Configuration

Edit `config/eval_config.yaml` to adjust:
- Model names and temperatures
- Rate limits and retry logic
- Metric thresholds
- Alert webhook URLs

## Tier 1 Enhancements Included

- **Bootstrap 95% CI** on every metric (1000 resamples)
- **Mann-Whitney U** significance testing between runs
- **Statistical regression gates** — alerts only fire at p < 0.05
- **Per-category ASR** breakdown
- **Latency percentiles** (P50/P95/P99)

## Cost

Approximate cost using Claude Haiku 4.5 (no free tier):
- ~150K tokens per 300-prompt run → roughly $0.20–$0.30 per run at current Haiku pricing
- 50 req/min rate limit (configurable via `providers.anthropic.rpm`)
- Set `api.daily_token_budget` in `config/eval_config.yaml` to cap spend
