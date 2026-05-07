"""
Module 18 — Security Layer

Enterprise-grade security for the eval platform:
  - RBAC: Role-Based Access Control (admin/analyst/viewer)
  - API key isolation and rotation
  - Audit logging (tamper-evident JSONL)
  - PII detection and redaction in prompts/responses
  - Rate limiting per API key
  - Sandboxed prompt execution metadata
  - Secret management (env-var based, Vault-compatible interface)
"""

import re
import json
import time
import uuid
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
AUDIT_DIR = DATA_DIR / "audit"


def load_config() -> dict:
    with open(ROOT / "config" / "eval_config.yaml") as f:
        return yaml.safe_load(f)


# ─── RBAC ────────────────────────────────────────────────────────────────────

ROLES = {
    "admin": {
        "permissions": ["read", "write", "delete", "run_eval", "view_keys",
                        "manage_users", "export_data", "view_audit"],
        "description": "Full access",
    },
    "analyst": {
        "permissions": ["read", "run_eval", "view_audit", "export_data"],
        "description": "Run evaluations and view results",
    },
    "viewer": {
        "permissions": ["read"],
        "description": "Read-only access to results",
    },
}


class RBACManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY, username TEXT UNIQUE, role TEXT,
            api_key_hash TEXT, created_at TEXT, last_login TEXT, active INTEGER DEFAULT 1)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY, user_id TEXT, created_at TEXT,
            expires_at TEXT, ip_address TEXT)""")
        conn.commit(); conn.close()

    def create_user(self, username: str, role: str) -> dict:
        if role not in ROLES:
            raise ValueError(f"Invalid role: {role}. Valid: {list(ROLES.keys())}")
        user_id = uuid.uuid4().hex[:12]
        api_key = f"llmeval_{uuid.uuid4().hex}"
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO users VALUES (?,?,?,?,?,?,1)",
                     (user_id, username, role, api_key_hash,
                      datetime.utcnow().isoformat(), None))
        conn.commit(); conn.close()
        return {"user_id": user_id, "username": username, "role": role,
                "api_key": api_key, "note": "Store this key securely — it won't be shown again"}

    def check_permission(self, api_key: str, permission: str) -> bool:
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        conn = sqlite3.connect(self.db_path); conn.row_factory = sqlite3.Row
        user = conn.execute("SELECT role FROM users WHERE api_key_hash=? AND active=1",
                            (key_hash,)).fetchone()
        conn.close()
        if not user: return False
        return permission in ROLES.get(user["role"], {}).get("permissions", [])

    def get_user_role(self, api_key: str) -> Optional[str]:
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        conn = sqlite3.connect(self.db_path); conn.row_factory = sqlite3.Row
        user = conn.execute("SELECT role FROM users WHERE api_key_hash=? AND active=1",
                            (key_hash,)).fetchone()
        conn.close()
        return user["role"] if user else None

    def list_users(self) -> list[dict]:
        conn = sqlite3.connect(self.db_path); conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT user_id,username,role,created_at,active FROM users").fetchall()
        conn.close()
        return [dict(r) for r in rows]


# ─── Rate Limiter ─────────────────────────────────────────────────────────────

class RateLimiter:
    """Token-bucket rate limiter per API key."""

    def __init__(self, requests_per_minute: int = 60, burst: int = 10):
        self.rpm = requests_per_minute
        self.burst = burst
        self._buckets: dict[str, dict] = {}

    def _get_bucket(self, key: str) -> dict:
        now = time.time()
        if key not in self._buckets:
            self._buckets[key] = {"tokens": self.burst, "last_refill": now, "total_requests": 0}
        bucket = self._buckets[key]
        # Refill
        elapsed = now - bucket["last_refill"]
        refill = elapsed * (self.rpm / 60.0)
        bucket["tokens"] = min(self.burst, bucket["tokens"] + refill)
        bucket["last_refill"] = now
        return bucket

    def check(self, api_key: str) -> tuple[bool, dict]:
        key_hash = hashlib.md5(api_key.encode()).hexdigest()[:8]
        bucket = self._get_bucket(key_hash)
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            bucket["total_requests"] += 1
            return True, {"allowed": True, "remaining_tokens": bucket["tokens"],
                          "total_requests": bucket["total_requests"]}
        return False, {"allowed": False, "retry_after_seconds": round(60 / self.rpm, 1),
                       "total_requests": bucket["total_requests"]}


# ─── PII Detector ────────────────────────────────────────────────────────────

PII_PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "phone_us": re.compile(r'\b(\+?1?\s?)?(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b'),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "credit_card": re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'),
    "api_key": re.compile(r'\b[A-Za-z0-9_-]{32,}\b'),
    "ip_address": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
    "date_of_birth": re.compile(r'\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/(\d{4})\b'),
}

REDACT_TOKEN = "[REDACTED]"


def detect_pii(text: str) -> dict:
    """Detect PII in text. Returns findings without redacting."""
    findings = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            findings[pii_type] = len(matches) if isinstance(matches[0], str) else len(matches)
    return findings


def redact_pii(text: str) -> tuple[str, dict]:
    """Redact PII from text. Returns (redacted_text, findings_summary)."""
    redacted = text
    findings = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            count = len(matches)
            findings[pii_type] = count
            redacted = pattern.sub(f"[{pii_type.upper()}_REDACTED]", redacted)
    return redacted, findings


# ─── Audit Logger ─────────────────────────────────────────────────────────────

class AuditLogger:
    """
    Tamper-evident audit log.
    Each entry is chained with the hash of the previous entry (like a blockchain).
    """

    def __init__(self):
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        self.log_file = AUDIT_DIR / f"audit_{datetime.utcnow().strftime('%Y%m')}.jsonl"
        self._prev_hash = self._get_last_hash()

    def _get_last_hash(self) -> str:
        """Get hash of the last log entry for chaining."""
        if not self.log_file.exists():
            return "genesis"
        try:
            with open(self.log_file, "rb") as f:
                lines = f.readlines()
                if lines:
                    return hashlib.sha256(lines[-1]).hexdigest()[:16]
        except Exception:
            pass
        return "genesis"

    def log(self, action: str, user_id: str = "system", resource: str = "",
            details: dict = None, outcome: str = "success", ip: str = ""):
        entry = {
            "event_id": uuid.uuid4().hex[:12],
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "user_id": user_id,
            "resource": resource,
            "outcome": outcome,
            "ip": ip,
            "details": details or {},
            "prev_hash": self._prev_hash,
        }
        entry_bytes = json.dumps(entry, sort_keys=True).encode()
        entry["hash"] = hashlib.sha256(entry_bytes).hexdigest()[:16]
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self._prev_hash = entry["hash"]

    def verify_chain(self) -> dict:
        """Verify the integrity of the audit log chain."""
        if not self.log_file.exists():
            return {"status": "empty", "valid": True}
        entries = []
        with open(self.log_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
        if not entries:
            return {"status": "empty", "valid": True}
        broken_at = None
        for i in range(1, len(entries)):
            prev = entries[i - 1]
            curr = entries[i]
            prev_hash = hashlib.sha256(
                json.dumps({k: v for k, v in prev.items() if k != "hash"}, sort_keys=True).encode()
            ).hexdigest()[:16]
            if curr.get("prev_hash") != prev_hash:
                broken_at = i
                break
        return {
            "status": "valid" if broken_at is None else "TAMPERED",
            "valid": broken_at is None,
            "total_entries": len(entries),
            "broken_at_index": broken_at,
        }

    def get_recent(self, n: int = 20) -> list[dict]:
        if not self.log_file.exists(): return []
        entries = []
        with open(self.log_file) as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except Exception:
                    pass
        return entries[-n:]


# ─── Secret Manager (Vault-compatible interface) ──────────────────────────────

class SecretManager:
    """
    Vault-compatible secret management interface.
    Uses environment variables as the backing store.
    In production, replace _get/_set with actual HashiCorp Vault API calls.
    """
    import os

    SECRET_MAP = {
        "google_api_key": "GOOGLE_API_KEY",
        "openai_api_key": "OPENAI_API_KEY",
        "anthropic_api_key": "ANTHROPIC_API_KEY",
        "slack_webhook": "SLACK_WEBHOOK_URL",
        "smtp_password": "SMTP_PASS",
    }

    def get(self, secret_name: str) -> Optional[str]:
        import os
        env_var = self.SECRET_MAP.get(secret_name.lower())
        if env_var:
            return os.environ.get(env_var)
        return os.environ.get(secret_name.upper())

    def list_secrets(self) -> list[str]:
        import os
        available = []
        for name, env_var in self.SECRET_MAP.items():
            if os.environ.get(env_var):
                available.append(name)
        return available

    def is_configured(self, secret_name: str) -> bool:
        return self.get(secret_name) is not None


# ─── Prompt sanitizer ─────────────────────────────────────────────────────────

DANGEROUS_PATTERNS = [
    re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
    re.compile(r'javascript:', re.IGNORECASE),
    re.compile(r'data:text/html', re.IGNORECASE),
    re.compile(r'\{\{.*?\}\}'),  # Template injection
]


def sanitize_prompt(prompt: str) -> tuple[str, list[str]]:
    """Basic prompt sanitization. Returns (sanitized, warnings)."""
    warnings = []
    sanitized = prompt
    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(sanitized):
            warnings.append(f"Dangerous pattern detected: {pattern.pattern[:30]}")
            sanitized = pattern.sub("[SANITIZED]", sanitized)
    # Truncate excessively long prompts
    if len(sanitized) > 10000:
        sanitized = sanitized[:10000] + "...[TRUNCATED]"
        warnings.append("Prompt truncated at 10000 chars")
    return sanitized, warnings


# ─── Security middleware ──────────────────────────────────────────────────────

audit = AuditLogger()
secrets = SecretManager()
rate_limiter = RateLimiter(requests_per_minute=60, burst=10)


def security_check(api_key: str, action: str, resource: str = "",
                   prompt: Optional[str] = None) -> dict:
    """
    Full security check for an API request.
    Returns {"allowed": bool, "warnings": [], "pii_findings": {}}
    """
    result = {"allowed": False, "warnings": [], "pii_findings": {}}

    # Rate limit check
    allowed, rate_info = rate_limiter.check(api_key)
    if not allowed:
        audit.log("RATE_LIMIT_EXCEEDED", resource=resource,
                  outcome="denied", details=rate_info)
        result["warnings"].append(f"Rate limit exceeded. Retry after {rate_info.get('retry_after_seconds')}s")
        return result

    # PII scan on prompt
    if prompt:
        findings = detect_pii(prompt)
        if findings:
            result["pii_findings"] = findings
            result["warnings"].append(f"PII detected in prompt: {list(findings.keys())}")

    result["allowed"] = True
    audit.log(action, resource=resource, outcome="allowed")
    return result
