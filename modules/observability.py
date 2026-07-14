"""
Module 16 — Advanced Observability Stack

Integrates:
  - OpenTelemetry: distributed traces for every eval request
  - Prometheus: metrics endpoint for Grafana scraping
  - Structured JSON logging with correlation IDs
  - Langfuse-compatible trace format
  - Evaluation drift detector (rolling window)
  - Token usage timeline

Exports:
  - /metrics  (Prometheus scrape endpoint)
  - traces/   (OTLP-compatible JSON traces)
  - logs/     (Structured JSON logs)
"""

import json
import time
import uuid
import logging
import sqlite3
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import yaml

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
LOGS_DIR = DATA_DIR / "logs"
TRACES_DIR = DATA_DIR / "traces"


def load_config() -> dict:
    with open(ROOT / "config" / "eval_config.yaml") as f:
        return yaml.safe_load(f)


# ─── Structured Logger ────────────────────────────────────────────────────────

class StructuredLogger:
    """JSON-lines structured logger with correlation IDs."""

    def __init__(self, service: str = "llm_eval"):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.service = service
        self.log_file = LOGS_DIR / f"{service}_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"

    def _write(self, level: str, message: str, extra: dict = None, trace_id: str = None):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "service": self.service,
            "message": message,
            "trace_id": trace_id or str(uuid.uuid4())[:8],
            **(extra or {}),
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        # Also print to stderr
        print(f"[{level}] {message}", flush=True)

    def info(self, msg: str, **kwargs): self._write("INFO", msg, kwargs)
    def warning(self, msg: str, **kwargs): self._write("WARNING", msg, kwargs)
    def error(self, msg: str, **kwargs): self._write("ERROR", msg, kwargs)
    def debug(self, msg: str, **kwargs): self._write("DEBUG", msg, kwargs)


logger = StructuredLogger()


# ─── OpenTelemetry Span ───────────────────────────────────────────────────────

class Span:
    """Lightweight OpenTelemetry-compatible span."""

    def __init__(self, name: str, trace_id: str = None, parent_id: str = None, attributes: dict = None):
        self.span_id = uuid.uuid4().hex[:16]
        self.trace_id = trace_id or uuid.uuid4().hex[:32]
        self.parent_id = parent_id
        self.name = name
        # wall_clock_start: real epoch time, used only for export/display.
        # start_time/end_time (perf_counter-based) drive duration_ms, since
        # time.time() can have coarse (~15ms) resolution on some platforms
        # and two fast back-to-back calls can otherwise report 0 duration.
        self.wall_clock_start = time.time()
        self.start_time = time.perf_counter()
        self.end_time: Optional[float] = None
        self.attributes: dict = attributes or {}
        self.events: list[dict] = []
        self.status = "OK"
        self.error: Optional[str] = None

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict = None):
        self.events.append({
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "attributes": attributes or {},
        })

    def set_error(self, error: str):
        self.status = "ERROR"
        self.error = error

    def end(self):
        self.end_time = time.perf_counter()
        self._export()
        return self

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return round((self.end_time - self.start_time) * 1000, 4)
        return round((time.perf_counter() - self.start_time) * 1000, 4)

    def _export(self):
        """Export span as OTLP-compatible JSON."""
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        # Real calendar timestamps for the exported trace come from
        # wall_clock_start plus the measured (perf_counter) duration, since
        # start_time/end_time themselves are monotonic, not epoch-based.
        wall_end = self.wall_clock_start + (self.duration_ms / 1000)
        span_dict = {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_id,
            "name": self.name,
            "kind": "SPAN_KIND_CLIENT",
            "startTimeUnixNano": int(self.wall_clock_start * 1e9),
            "endTimeUnixNano": int(wall_end * 1e9) if self.end_time else None,
            "durationMs": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": {"code": self.status},
            "error": self.error,
        }
        trace_file = TRACES_DIR / f"trace_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        with open(trace_file, "a") as f:
            f.write(json.dumps(span_dict) + "\n")


class Tracer:
    """Tracer for creating spans."""

    def __init__(self, service: str = "llm_eval"):
        self.service = service
        self._active_spans: dict[str, Span] = {}

    def start_span(self, name: str, parent: Optional[Span] = None, attributes: dict = None) -> Span:
        span = Span(
            name=f"{self.service}/{name}",
            trace_id=parent.trace_id if parent else None,
            parent_id=parent.span_id if parent else None,
            attributes={"service": self.service, **(attributes or {})},
        )
        self._active_spans[span.span_id] = span
        return span

    def finish_span(self, span: Span) -> Span:
        span.end()
        self._active_spans.pop(span.span_id, None)
        return span


tracer = Tracer()


# ─── Prometheus Metrics ───────────────────────────────────────────────────────

class PrometheusMetrics:
    """
    Lightweight Prometheus metrics collector.
    Exports /metrics endpoint format.
    """

    def __init__(self):
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._labels: dict[str, dict] = {}

    def counter_inc(self, name: str, value: float = 1.0, labels: dict = None):
        key = self._key(name, labels)
        self._counters[key] += value
        self._labels[key] = labels or {}

    def gauge_set(self, name: str, value: float, labels: dict = None):
        key = self._key(name, labels)
        self._gauges[key] = value
        self._labels[key] = labels or {}

    def histogram_observe(self, name: str, value: float, labels: dict = None):
        key = self._key(name, labels)
        self._histograms[key].append(value)

    def _key(self, name: str, labels: dict = None) -> str:
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name

    def export(self) -> str:
        """Export in Prometheus text format."""
        lines = [
            "# HELP llm_eval_requests_total Total evaluation requests",
            "# TYPE llm_eval_requests_total counter",
        ]
        for key, val in self._counters.items():
            lines.append(f"{key} {val}")

        lines += ["# HELP llm_eval_metric_score Current metric score",
                  "# TYPE llm_eval_metric_score gauge"]
        for key, val in self._gauges.items():
            lines.append(f"{key} {val}")

        for key, vals in self._histograms.items():
            if vals:
                import statistics
                lines += [
                    f"# HELP {key.split('{')[0]}_bucket Histogram",
                    f"# TYPE {key.split('{')[0]}_bucket histogram",
                    f"{key.split('{')[0]}_count {len(vals)}",
                    f"{key.split('{')[0]}_sum {sum(vals):.4f}",
                    f"{key.split('{')[0]}_p50 {sorted(vals)[len(vals)//2]:.4f}",
                    f"{key.split('{')[0]}_p95 {sorted(vals)[int(len(vals)*0.95)]:.4f}",
                ]

        return "\n".join(lines)

    def save_snapshot(self):
        snapshot_path = DATA_DIR / "metrics_snapshot.txt"
        snapshot_path.parent.mkdir(exist_ok=True)
        with open(snapshot_path, "w") as f:
            f.write(self.export())


metrics = PrometheusMetrics()


# ─── Evaluation Drift Detector ────────────────────────────────────────────────

class DriftDetector:
    """
    Rolling-window drift detector for evaluation metrics.
    Fires an alert if metric mean shifts > threshold over a window.
    """

    def __init__(self, window_size: int = 50, drift_threshold: float = 0.05):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self._windows: dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._baselines: dict[str, float] = {}

    def observe(self, metric: str, value: float):
        self._windows[metric].append(value)

    def set_baseline(self, metric: str, value: float):
        self._baselines[metric] = value

    def check_drift(self) -> list[dict]:
        """Return list of drifted metrics."""
        alerts = []
        for metric, window in self._windows.items():
            if len(window) < 10:
                continue
            current_mean = sum(window) / len(window)
            if metric in self._baselines:
                drift = abs(current_mean - self._baselines[metric])
                if drift > self.drift_threshold:
                    alerts.append({
                        "metric": metric,
                        "current_mean": round(current_mean, 4),
                        "baseline": round(self._baselines[metric], 4),
                        "drift": round(drift, 4),
                        "window_size": len(window),
                    })
        return alerts

    def summary(self) -> dict:
        result = {}
        for metric, window in self._windows.items():
            if window:
                vals = list(window)
                result[metric] = {
                    "current_mean": round(sum(vals) / len(vals), 4),
                    "baseline": self._baselines.get(metric),
                    "window_size": len(vals),
                    "drift": round(abs(sum(vals)/len(vals) - self._baselines.get(metric, sum(vals)/len(vals))), 4),
                }
        return result


drift_detector = DriftDetector()


# ─── Token usage timeline ─────────────────────────────────────────────────────

class TokenTimeline:
    """Track token usage over time for burn-rate visualization."""

    def __init__(self):
        self._events: list[dict] = []

    def record(self, run_id: str, category: str, input_tokens: int, output_tokens: int,
               latency_ms: float, provider: str = "gemini"):
        self._events.append({
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": run_id,
            "category": category,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": latency_ms,
            "provider": provider,
        })

    def get_timeline(self, run_id: str = None) -> list[dict]:
        if run_id:
            return [e for e in self._events if e["run_id"] == run_id]
        return self._events

    def cumulative_tokens(self) -> list[dict]:
        """Cumulative token usage over time."""
        total = 0
        result = []
        for e in self._events:
            total += e["total_tokens"]
            result.append({"timestamp": e["timestamp"], "cumulative": total})
        return result

    def save(self, path: Path = None):
        p = path or DATA_DIR / "token_timeline.jsonl"
        with open(p, "a") as f:
            for e in self._events[-100:]:  # last 100 events
                f.write(json.dumps(e) + "\n")


token_timeline = TokenTimeline()


# ─── Instrument eval run ──────────────────────────────────────────────────────

def instrument_result(result: dict):
    """
    Call this for every eval result to update all observability systems.
    """
    cat = result.get("category", "unknown")
    run_id = result.get("run_id", "unknown")

    # Prometheus
    metrics.counter_inc("llm_eval_requests_total", labels={"category": cat})
    if result.get("error"):
        metrics.counter_inc("llm_eval_errors_total", labels={"category": cat})
    for metric in ["relevance", "toxicity", "bias", "asr"]:
        val = result.get(metric)
        if val is not None:
            metrics.gauge_set(f"llm_eval_{metric}_score", val, labels={"category": cat})
            metrics.histogram_observe(f"llm_eval_{metric}_histogram", val)
    if result.get("latency_ms"):
        metrics.histogram_observe("llm_eval_latency_ms_histogram", result["latency_ms"])

    # Drift detection
    for metric in ["relevance", "toxicity", "bias", "asr"]:
        val = result.get(metric)
        if val is not None:
            drift_detector.observe(metric, val)

    # Token timeline
    if result.get("input_tokens") is not None:
        token_timeline.record(
            run_id, cat,
            result.get("input_tokens", 0),
            result.get("output_tokens", 0),
            result.get("latency_ms", 0),
        )

    # Structured log
    if result.get("asr", 0) > 0.5:
        logger.warning("High ASR detected",
                       prompt_id=result.get("prompt_id"),
                       category=cat, asr=result.get("asr"))
    if result.get("error"):
        logger.error("Eval error", prompt_id=result.get("prompt_id"),
                     error=result["error"][:200])


def get_observability_summary() -> dict:
    """Full observability summary for dashboard."""
    return {
        "drift_alerts": drift_detector.check_drift(),
        "drift_summary": drift_detector.summary(),
        "token_timeline": token_timeline.get_timeline()[-20:],
        "cumulative_tokens": token_timeline.cumulative_tokens()[-20:],
        "prometheus_metrics": metrics.export()[:2000],
    }