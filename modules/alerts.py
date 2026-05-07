"""
Module 04 — Alert System
Sends notifications when metrics cross thresholds.
Supports webhook (Slack/Discord) and email (optional).
"""

import json
import smtplib
from pathlib import Path
from datetime import datetime
from typing import Optional
import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config" / "eval_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def format_alert_message(run_id: str, violations: list[dict], stats: dict) -> str:
    lines = [
        f"🚨 LLM Eval Alert — Run: {run_id}",
        f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"Violations: {len(violations)}",
        "",
    ]
    for v in violations:
        emoji = "🔴" if v["severity"] == "CRITICAL" else "🟡"
        lines.append(
            f"{emoji} [{v['severity']}] {v['metric'].upper()} = {v['value']:.3f} "
            f"(threshold: {'≥' if v['op'] == 'min' else '≤'} {v['threshold']})"
        )

    lines.append("")
    lines.append("── Current Means ──")
    for metric, s in stats.items():
        if isinstance(s, dict) and "mean" in s:
            lines.append(f"  {metric}: {s['mean']:.4f} [{s['ci_low']:.4f}, {s['ci_high']:.4f}]")

    return "\n".join(lines)


def send_webhook(url: str, message: str) -> bool:
    """Send to Slack or Discord webhook."""
    try:
        import urllib.request
        payload = json.dumps({"text": message, "content": message}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status < 300
    except Exception as e:
        print(f"[Alerts] Webhook failed: {e}")
        return False


def send_email(to: str, subject: str, body: str) -> bool:
    """Send email alert via SMTP (requires env vars SMTP_HOST, SMTP_USER, SMTP_PASS)."""
    import os
    host = os.environ.get("SMTP_HOST", "")
    user = os.environ.get("SMTP_USER", "")
    password = os.environ.get("SMTP_PASS", "")
    if not all([host, user, password, to]):
        print("[Alerts] Email not configured. Set SMTP_HOST, SMTP_USER, SMTP_PASS env vars.")
        return False
    try:
        with smtplib.SMTP_SSL(host, 465) as smtp:
            smtp.login(user, password)
            msg = f"Subject: {subject}\nFrom: {user}\nTo: {to}\n\n{body}"
            smtp.sendmail(user, to, msg)
        return True
    except Exception as e:
        print(f"[Alerts] Email failed: {e}")
        return False


def fire_alerts(
    run_id: str,
    violations: list[dict],
    stats: dict,
    config: Optional[dict] = None,
) -> dict:
    cfg = config or load_config()
    alert_cfg = cfg.get("alerts", {})

    if not alert_cfg.get("enabled", False):
        print("[Alerts] Alerts disabled in config. Skipping.")
        return {"sent": False, "reason": "disabled"}

    if not violations:
        print("[Alerts] No violations to alert on.")
        return {"sent": False, "reason": "no_violations"}

    message = format_alert_message(run_id, violations, stats)
    print(f"\n[Alerts] Firing alert for {len(violations)} violation(s):")
    print(message)

    results = {}

    webhook_url = alert_cfg.get("webhook_url", "")
    if webhook_url:
        ok = send_webhook(webhook_url, message)
        results["webhook"] = "sent" if ok else "failed"

    email = alert_cfg.get("email", "")
    if email:
        ok = send_email(email, f"[LLM Eval] {len(violations)} threshold violation(s) in {run_id}", message)
        results["email"] = "sent" if ok else "failed"

    # Always log to file
    log_path = Path(cfg["database"]["path"]).parent / "alerts.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "violations": violations,
            "channels": results,
        }) + "\n")

    return {"sent": True, "results": results}


if __name__ == "__main__":
    # Demo: fire a test alert
    test_violations = [
        {"metric": "faithfulness", "value": 0.65, "threshold": 0.70, "op": "min", "label": "critical", "severity": "CRITICAL"},
        {"metric": "asr", "value": 0.12, "threshold": 0.05, "op": "max", "label": "warning", "severity": "WARNING"},
    ]
    test_stats = {
        "faithfulness": {"mean": 0.65, "ci_low": 0.60, "ci_high": 0.70},
        "asr": {"mean": 0.12, "ci_low": 0.08, "ci_high": 0.16},
    }
    cfg = load_config()
    cfg["alerts"]["enabled"] = True  # force for demo
    fire_alerts("demo_run_001", test_violations, test_stats, cfg)
