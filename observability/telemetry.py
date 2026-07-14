"""
observability/telemetry.py — Advanced Observability Stack

Implements:
  - OpenTelemetry traces for every LLM call
  - Prometheus metrics (counters, histograms, gauges)
  - Structured logging with trace correlation
  - Langfuse-compatible trace export
  - Evaluation drift detection
  - Real-time token usage and latency tracking
"""
from __future__ import annotations
import json, os, time, threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"


# ─── Span / Trace model ───────────────────────────────────────────────────────

@dataclass
class Span:
    trace_id: str
    span_id: str
    name: str
    start_time: float           # perf_counter() timestamp — used for ordering/duration
    end_time: float = 0.0       # perf_counter() timestamp — used for ordering/duration
    attributes: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    status: str = "ok"     # ok | error
    parent_span_id: str = ""
    wall_clock_start: float = 0.0  # real epoch time.time(), for export/display only

    @property
    def duration_ms(self): return round((self.end_time - self.start_time) * 1000, 2)

    def add_event(self, name: str, **attrs):
        self.events.append({"name": name, "timestamp": time.time(), "attributes": attrs})

    def set_error(self, msg: str):
        self.status = "error"
        self.attributes["error"] = msg

    def to_dict(self):
        return {"trace_id": self.trace_id, "span_id": self.span_id, "name": self.name,
                "duration_ms": self.duration_ms, "status": self.status,
                "attributes": self.attributes, "events": self.events}


class TraceContext:
    """Thread-local trace context for correlation."""
    _local = threading.local()

    @classmethod
    def current_trace_id(cls) -> str:
        return getattr(cls._local, "trace_id", "")

    @classmethod
    def set_trace_id(cls, trace_id: str):
        cls._local.trace_id = trace_id


# ─── Prometheus Metrics ───────────────────────────────────────────────────────

class MetricsRegistry:
    """
    Prometheus-compatible metrics registry.
    Exposes /metrics endpoint when used with FastAPI.
    """
    def __init__(self):
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = defaultdict(float)
        self._histogram_buckets = [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

    # Counters
    def inc(self, name: str, value: float = 1.0, labels: Optional[dict] = None):
        key = self._key(name, labels)
        self._counters[key] += value

    # Histograms
    def observe(self, name: str, value: float, labels: Optional[dict] = None):
        key = self._key(name, labels)
        self._histograms[key].append(value)

    # Gauges
    def set_gauge(self, name: str, value: float, labels: Optional[dict] = None):
        key = self._key(name, labels)
        self._gauges[key] = value

    def _key(self, name: str, labels: Optional[dict]) -> str:
        if not labels: return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def exposition(self) -> str:
        """Prometheus text exposition format."""
        lines = [f"# LLM Eval Metrics — {datetime.utcnow().isoformat()}"]

        for key, val in sorted(self._counters.items()):
            lines.append(f"llm_eval_{key}_total {val}")

        for key, vals in sorted(self._histograms.items()):
            if not vals: continue
            import statistics as st
            lines.append(f"llm_eval_{key}_count {len(vals)}")
            lines.append(f"llm_eval_{key}_sum {sum(vals):.2f}")
            lines.append(f"llm_eval_{key}_mean {st.mean(vals):.2f}")
            p95 = sorted(vals)[int(len(vals) * 0.95)]
            lines.append(f"llm_eval_{key}_p95 {p95:.2f}")

        for key, val in sorted(self._gauges.items()):
            lines.append(f"llm_eval_{key} {val}")

        return "\n".join(lines)

    def snapshot(self) -> dict:
        """JSON snapshot for dashboard."""
        import statistics as st
        snap = {"counters": dict(self._counters), "gauges": dict(self._gauges), "histograms": {}}
        for key, vals in self._histograms.items():
            if not vals: continue
            sorted_vals = sorted(vals)
            snap["histograms"][key] = {
                "count": len(vals), "mean": round(st.mean(vals), 2),
                "p50": sorted_vals[len(vals)//2],
                "p95": sorted_vals[int(len(vals)*0.95)],
                "p99": sorted_vals[min(int(len(vals)*0.99), len(vals)-1)],
            }
        return snap


# ─── Global metrics registry ──────────────────────────────────────────────────

metrics = MetricsRegistry()


# ─── Telemetry Collector ──────────────────────────────────────────────────────

class TelemetryCollector:
    """
    Central telemetry hub. Records spans, exports to:
    - Local JSONL file (always)
    - Prometheus exposition (on demand)
    - Langfuse (if configured)
    - OpenTelemetry OTLP (if configured)
    """
    def __init__(self):
        self._spans: deque[Span] = deque(maxlen=10_000)
        self._trace_log = DATA_DIR / "traces.jsonl"
        DATA_DIR.mkdir(exist_ok=True)
        self._drift_window: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    # ── Span management ───────────────────────────────────────────────────

    def start_span(self, name: str, trace_id: str = "", parent: str = "",
                   **attrs) -> Span:
        import uuid
        tid = trace_id or TraceContext.current_trace_id() or str(uuid.uuid4())[:8]
        sid = str(uuid.uuid4())[:8]
        # perf_counter() is monotonic and high-resolution, guaranteeing
        # end_time > start_time even for very fast spans (plain time.time()
        # can have ~15ms resolution on Windows and collide on quick calls).
        # wall_clock_start keeps a real epoch timestamp for export/display.
        span = Span(trace_id=tid, span_id=sid, name=name,
                    start_time=time.perf_counter(), attributes=attrs, parent_span_id=parent,
                    wall_clock_start=time.time())
        return span

    def end_span(self, span: Span, **attrs):
        span.end_time = time.perf_counter()
        span.attributes.update(attrs)
        self._spans.append(span)
        self._flush_span(span)
        return span

    def _flush_span(self, span: Span):
        with open(self._trace_log, "a") as f:
            f.write(json.dumps(span.to_dict()) + "\n")

    # ── LLM call instrumentation ───────────────────────────────────────────

    def record_llm_call(self, provider: str, model: str, prompt_tokens: int,
                        completion_tokens: int, latency_ms: float,
                        success: bool = True, cached: bool = False,
                        run_id: str = "", category: str = ""):
        metrics.inc("llm_requests", labels={"provider": provider, "model": model,
                                             "success": str(success), "cached": str(cached)})
        metrics.observe("llm_latency_ms", latency_ms, labels={"provider": provider})
        metrics.observe("llm_input_tokens", prompt_tokens, labels={"provider": provider})
        metrics.observe("llm_output_tokens", completion_tokens, labels={"provider": provider})
        metrics.set_gauge("llm_cache_hit_rate", metrics._counters.get(
            'llm_requests{cached="True",model="'+model+'",provider="'+provider+'",success="True"}', 0) /
            max(metrics._counters.get('llm_requests{cached="False",model="'+model+'",provider="'+provider+'",success="True"}', 1), 1))

    def record_eval_result(self, run_id: str, category: str, metric: str, value: float):
        """Record a metric score and check for drift."""
        metrics.observe(f"eval_{metric}", value, labels={"category": category})
        self._drift_window[metric].append(value)

    def record_attack_result(self, category: str, success: bool):
        metrics.inc("attack_attempts", labels={"category": category})
        if success:
            metrics.inc("attack_successes", labels={"category": category})

    # ── Drift detection ────────────────────────────────────────────────────

    def detect_drift(self, metric: str, window_size: int = 20,
                     threshold: float = 0.05) -> dict:
        """Compare recent vs historical scores to detect distribution drift."""
        vals = list(self._drift_window[metric])
        if len(vals) < window_size * 2:
            return {"status": "insufficient_data", "n": len(vals)}
        import statistics
        recent = vals[-window_size:]
        historical = vals[:-window_size]
        recent_mean = statistics.mean(recent)
        hist_mean = statistics.mean(historical)
        drift = abs(recent_mean - hist_mean)
        return {
            "metric": metric, "drift": round(drift, 4),
            "recent_mean": round(recent_mean, 4), "historical_mean": round(hist_mean, 4),
            "drift_detected": drift > threshold, "threshold": threshold,
            "window_size": window_size,
        }

    def drift_report(self) -> dict:
        return {metric: self.detect_drift(metric)
                for metric in ["faithfulness", "relevance", "toxicity", "bias", "asr"]}

    # ── Query traces ───────────────────────────────────────────────────────

    def get_recent_spans(self, n: int = 100) -> list[dict]:
        return [s.to_dict() for s in list(self._spans)[-n:]]

    def get_trace(self, trace_id: str) -> list[dict]:
        return [s.to_dict() for s in self._spans if s.trace_id == trace_id]

    # ── Langfuse export ────────────────────────────────────────────────────

    def export_to_langfuse(self, span: Span):
        """Export span as Langfuse generation event (requires LANGFUSE_* env vars)."""
        pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if not pk or not sk: return

        try:
            import httpx
            # start_time/end_time are perf_counter() values (monotonic, used
            # for accurate duration) — real calendar timestamps for export
            # come from wall_clock_start plus the measured duration.
            wall_start = span.wall_clock_start or time.time()
            wall_end = wall_start + (span.end_time - span.start_time)
            payload = {
                "id": span.span_id, "traceId": span.trace_id, "name": span.name,
                "startTime": datetime.utcfromtimestamp(wall_start).isoformat() + "Z",
                "endTime": datetime.utcfromtimestamp(wall_end).isoformat() + "Z",
                "metadata": span.attributes,
            }
            httpx.post(f"{host}/api/public/generations", json=payload,
                       auth=(pk, sk), timeout=5)
        except Exception:
            pass  # Non-blocking

    # ── Prometheus exposition ──────────────────────────────────────────────

    def prometheus_exposition(self) -> str:
        return metrics.exposition()

    def metrics_snapshot(self) -> dict:
        return metrics.snapshot()


# ─── Global telemetry instance ────────────────────────────────────────────────

telemetry = TelemetryCollector()


# ─── Decorator for instrumentation ───────────────────────────────────────────

def instrument(span_name: str):
    """Decorator to auto-instrument async functions with telemetry spans."""
    import functools, uuid
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            trace_id = str(uuid.uuid4())[:8]
            span = telemetry.start_span(span_name, trace_id=trace_id)
            try:
                result = await fn(*args, **kwargs)
                telemetry.end_span(span, status="ok")
                return result
            except Exception as e:
                span.set_error(str(e))
                telemetry.end_span(span)
                raise
        return wrapper
    return decorator