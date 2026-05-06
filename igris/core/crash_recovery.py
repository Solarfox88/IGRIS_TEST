"""Crash recovery for IGRIS_GPT.

Provides structured crash handling:
- Redacted stacktrace capture
- JSON/Markdown crash report generation
- Timeline event recording
- ``last_known_good_state`` tracking
- Failure categorisation
- Suggested remediation

Every unhandled exception in a mission/loop cycle should be routed through
:func:`handle_crash` which produces a :class:`CrashReport` and persists it.
"""

from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from igris.core.safety import redact_secrets


# ---------------------------------------------------------------------------
# Failure categories
# ---------------------------------------------------------------------------

FAILURE_CATEGORIES: Dict[str, str] = {
    "import_error": "Missing or broken dependency",
    "connection_error": "Network / service unreachable",
    "timeout_error": "Operation timed out",
    "permission_error": "Insufficient permissions",
    "file_not_found": "Required file or path missing",
    "config_error": "Configuration invalid or incomplete",
    "json_error": "Malformed JSON data",
    "llm_error": "LLM provider failure or bad response",
    "git_error": "Git operation failed",
    "test_failure": "Test execution failure",
    "validation_error": "Input or schema validation failure",
    "unknown": "Unclassified failure",
}

REMEDIATION_SUGGESTIONS: Dict[str, str] = {
    "import_error": 'Run: pip install -e ".[dev]"',
    "connection_error": "Check network connectivity and service status (Ollama, Docker, etc.)",
    "timeout_error": "Increase timeout or check service responsiveness",
    "permission_error": "Check file/directory permissions and ownership",
    "file_not_found": "Verify file paths and project root configuration",
    "config_error": "Check .env and config/config.sample.json for required fields",
    "json_error": "Validate JSON syntax in the affected file",
    "llm_error": "Check LLM provider status (Ollama running? API key valid?)",
    "git_error": "Check git status, branch, and remote configuration",
    "test_failure": "Run tests locally and inspect output: python -m pytest -q",
    "validation_error": "Check input data against expected schema",
    "unknown": "Inspect the redacted stacktrace for details",
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class CrashReport:
    """Structured crash report."""
    id: str = field(default_factory=lambda: f"crash-{uuid.uuid4().hex[:12]}")
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    failure_category: str = "unknown"
    failure_description: str = ""
    exception_type: str = ""
    redacted_stacktrace: str = ""
    mission_id: Optional[str] = None
    task_id: Optional[str] = None
    action_id: Optional[str] = None
    trace_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    last_known_good_state: Optional[Dict[str, Any]] = None
    suggested_remediation: str = ""
    severity: str = "error"  # warning | error | critical

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "failure_category": self.failure_category,
            "failure_description": redact_secrets(self.failure_description),
            "exception_type": self.exception_type,
            "redacted_stacktrace": self.redacted_stacktrace,
            "mission_id": self.mission_id,
            "task_id": self.task_id,
            "action_id": self.action_id,
            "trace_id": self.trace_id,
            "context": {k: redact_secrets(str(v)) for k, v in self.context.items()},
            "last_known_good_state": self.last_known_good_state,
            "suggested_remediation": self.suggested_remediation,
            "severity": self.severity,
        }

    def to_markdown(self) -> str:
        lines = [
            "# IGRIS Crash Report",
            f"**ID:** {self.id}",
            f"**Timestamp:** {self.timestamp}",
            f"**Severity:** {self.severity}",
            f"**Category:** {self.failure_category} — {FAILURE_CATEGORIES.get(self.failure_category, 'Unknown')}",
            f"**Exception:** {self.exception_type}",
            "",
            "## Description",
            redact_secrets(self.failure_description),
            "",
            "## Redacted Stacktrace",
            "```",
            self.redacted_stacktrace,
            "```",
            "",
        ]
        if self.suggested_remediation:
            lines.extend(["## Suggested Remediation", self.suggested_remediation, ""])
        if self.last_known_good_state:
            lines.extend([
                "## Last Known Good State",
                "```json",
                json.dumps(self.last_known_good_state, indent=2, default=str),
                "```",
                "",
            ])
        ids = []
        if self.mission_id:
            ids.append(f"Mission: {self.mission_id}")
        if self.task_id:
            ids.append(f"Task: {self.task_id}")
        if self.action_id:
            ids.append(f"Action: {self.action_id}")
        if self.trace_id:
            ids.append(f"Trace: {self.trace_id}")
        if ids:
            lines.extend(["## Trace IDs", "\n".join(f"- {i}" for i in ids), ""])
        return "\n".join(lines)

    def to_timeline_event(self) -> Dict[str, Any]:
        """Return a dict suitable for appending to the IGRIS timeline."""
        return {
            "type": "crash",
            "title": f"Crash: {self.failure_category}",
            "detail": redact_secrets(self.failure_description)[:500],
            "severity": self.severity,
            "crash_id": self.id,
            "mission_id": self.mission_id,
            "task_id": self.task_id,
            "trace_id": self.trace_id,
        }


# ---------------------------------------------------------------------------
# Failure categorisation
# ---------------------------------------------------------------------------

def classify_exception(exc: BaseException) -> str:
    """Classify an exception into a failure category."""
    type_name = type(exc).__name__.lower()
    msg = str(exc).lower()

    if "import" in type_name or "module" in type_name:
        return "import_error"
    if any(k in type_name for k in ("connection", "socket", "urlopen", "http")):
        return "connection_error"
    if "timeout" in type_name or "timeout" in msg:
        return "timeout_error"
    if "permission" in type_name:
        return "permission_error"
    if "filenotfound" in type_name or "no such file" in msg:
        return "file_not_found"
    if any(k in msg for k in ("config", "configuration", "missing key", "required field")):
        return "config_error"
    if "json" in type_name or "json" in msg:
        return "json_error"
    if any(k in msg for k in ("ollama", "openai", "llm", "model not found", "api key")):
        return "llm_error"
    if "git" in msg and any(k in msg for k in ("fatal", "error", "failed")):
        return "git_error"
    if any(k in msg for k in ("assert", "test fail")):
        return "test_failure"
    if "validat" in type_name or "validat" in msg:
        return "validation_error"
    return "unknown"


def redact_stacktrace(tb: str) -> str:
    """Redact secrets from a stacktrace string."""
    return redact_secrets(tb)


# ---------------------------------------------------------------------------
# Last known good state
# ---------------------------------------------------------------------------

def _good_state_path(project_root: Optional[str] = None) -> Path:
    root = Path(project_root) if project_root else Path(os.environ.get("PROJECT_ROOT", "."))
    d = root / ".igris" / "recovery"
    d.mkdir(parents=True, exist_ok=True)
    return d / "last_known_good_state.json"


def save_good_state(
    state: Dict[str, Any],
    project_root: Optional[str] = None,
) -> None:
    """Persist the last known good state."""
    path = _good_state_path(project_root)
    state["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def load_good_state(project_root: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load the last known good state, or ``None`` if not saved."""
    path = _good_state_path(project_root)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Crash report persistence
# ---------------------------------------------------------------------------

def _crash_dir(project_root: Optional[str] = None) -> Path:
    root = Path(project_root) if project_root else Path(os.environ.get("PROJECT_ROOT", "."))
    d = root / ".igris" / "recovery" / "crashes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_crash_report(report: CrashReport, project_root: Optional[str] = None) -> Path:
    """Persist a crash report as JSON."""
    d = _crash_dir(project_root)
    path = d / f"{report.id}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    # Also write markdown
    md_path = d / f"{report.id}.md"
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    return path


def list_crash_reports(
    project_root: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """List recent crash reports (newest first)."""
    d = _crash_dir(project_root)
    files = sorted(d.glob("crash-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    reports: List[Dict[str, Any]] = []
    for f in files[:limit]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            reports.append(data)
        except Exception:
            pass
    return reports


def get_crash_report(crash_id: str, project_root: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load a specific crash report by ID."""
    d = _crash_dir(project_root)
    path = d / f"{crash_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def handle_crash(
    exc: BaseException,
    *,
    mission_id: Optional[str] = None,
    task_id: Optional[str] = None,
    action_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    project_root: Optional[str] = None,
) -> CrashReport:
    """Handle an exception: classify, build report, persist.

    Returns the :class:`CrashReport` which callers can use to create a
    timeline event or return in API responses.
    """
    category = classify_exception(exc)
    raw_tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    redacted_tb = redact_stacktrace("".join(raw_tb))
    good_state = load_good_state(project_root)

    report = CrashReport(
        failure_category=category,
        failure_description=str(exc),
        exception_type=type(exc).__name__,
        redacted_stacktrace=redacted_tb,
        mission_id=mission_id,
        task_id=task_id,
        action_id=action_id,
        trace_id=trace_id or f"trace-{uuid.uuid4().hex[:8]}",
        context=context or {},
        last_known_good_state=good_state,
        suggested_remediation=REMEDIATION_SUGGESTIONS.get(category, ""),
    )
    save_crash_report(report, project_root)
    return report
