"""Persistence layer for AssignmentRouter outcomes."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from typing import Any, Dict, List

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9\-_]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9\-_\.]{20,}"),
    re.compile(r"OPENAI_API_KEY\s*=\s*\S+"),
    re.compile(r"ANTHROPIC_API_KEY\s*=\s*\S+"),
    re.compile(r"DEEPSEEK_API_KEY\s*=\s*\S+"),
]


def compute_task_signature(goal_text: str) -> str:
    """SHA-256 of normalised goal text.

    TODO: replace with embedding similarity when history is large enough.
    """
    normalized = " ".join(goal_text.lower().strip().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def _redact_string(value: str) -> str:
    for pattern in _SECRET_PATTERNS:
        value = pattern.sub("[REDACTED]", value)
    return value


def sanitize_for_storage(record: Dict[str, Any]) -> Dict[str, Any]:
    """Redact API keys and secrets from a record before writing to disk."""
    out: Dict[str, Any] = {}
    for k, v in record.items():
        if isinstance(v, str):
            out[k] = _redact_string(v)
        elif isinstance(v, dict):
            out[k] = sanitize_for_storage(v)
        elif isinstance(v, list):
            out[k] = [_redact_string(i) if isinstance(i, str) else i for i in v]
        else:
            out[k] = v
    return out


def load_assignment_outcomes(path: str) -> List[Dict[str, Any]]:
    """Load outcomes list from JSON file. Returns [] if missing or corrupt."""
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return []


def save_assignment_outcome(path: str, record: Dict[str, Any]) -> None:
    """Atomically append a sanitized record to the outcomes file."""
    safe = sanitize_for_storage(record)
    outcomes = load_assignment_outcomes(path)
    outcomes.append(safe)
    _atomic_write(path, outcomes)


def _atomic_write(path: str, data: Any) -> None:
    dir_ = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
