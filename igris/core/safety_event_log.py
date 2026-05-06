"""Safety Event Log for IGRIS_GPT — Epic #42.

Logs every safety-relevant event: blocks, approvals, risk decisions,
rollback requirements, escalations.  Provides query/filter capabilities
for audit and debugging.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from igris.core.safety import redact_secrets


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

SAFETY_EVENT_TYPES = (
    "action_blocked",
    "action_approved",
    "risk_decision",
    "rollback_required",
    "rollback_applied",
    "escalation",
    "secret_detected",
    "policy_violation",
    "approval_requested",
    "approval_granted",
    "approval_denied",
)


# ---------------------------------------------------------------------------
# Safety event
# ---------------------------------------------------------------------------

@dataclass
class SafetyEvent:
    """A single safety event."""
    id: str = field(default_factory=lambda: f"sev-{uuid.uuid4().hex[:8]}")
    type: str = "risk_decision"
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    mission_id: str = ""
    action_id: str = ""
    trace_id: str = ""
    risk_level: str = ""
    approval_mode: str = ""
    decision: str = ""  # allowed | blocked | escalated
    reason: str = ""
    detail: str = ""
    severity: str = "info"  # info | warning | error | critical

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "timestamp": self.timestamp,
            "mission_id": self.mission_id,
            "action_id": self.action_id,
            "trace_id": self.trace_id,
            "risk_level": self.risk_level,
            "approval_mode": self.approval_mode,
            "decision": self.decision,
            "reason": redact_secrets(self.reason),
            "detail": redact_secrets(self.detail),
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyEvent":
        return cls(
            id=data.get("id", f"sev-{uuid.uuid4().hex[:8]}"),
            type=data.get("type", "risk_decision"),
            timestamp=data.get("timestamp", ""),
            mission_id=data.get("mission_id", ""),
            action_id=data.get("action_id", ""),
            trace_id=data.get("trace_id", ""),
            risk_level=data.get("risk_level", ""),
            approval_mode=data.get("approval_mode", ""),
            decision=data.get("decision", ""),
            reason=data.get("reason", ""),
            detail=data.get("detail", ""),
            severity=data.get("severity", "info"),
        )


# ---------------------------------------------------------------------------
# Safety Event Log
# ---------------------------------------------------------------------------

class SafetyEventLog:
    """Persistent safety event log."""

    def __init__(self, project_root: Optional[str] = None):
        import os
        root = Path(project_root) if project_root else Path(os.environ.get("PROJECT_ROOT", "."))
        self._log_dir = root / ".igris" / "safety" / "events"
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _save_event(self, event: SafetyEvent) -> Path:
        path = self._log_dir / f"{event.id}.json"
        path.write_text(json.dumps(event.to_dict(), indent=2, default=str), encoding="utf-8")
        return path

    def log_block(
        self,
        action_id: str,
        risk_level: str,
        reason: str,
        mission_id: str = "",
        trace_id: str = "",
        detail: str = "",
    ) -> SafetyEvent:
        """Log an action that was blocked."""
        event = SafetyEvent(
            type="action_blocked",
            mission_id=mission_id,
            action_id=action_id,
            trace_id=trace_id,
            risk_level=risk_level,
            decision="blocked",
            reason=reason,
            detail=detail,
            severity="warning",
        )
        self._save_event(event)
        return event

    def log_approval(
        self,
        action_id: str,
        risk_level: str,
        approval_mode: str,
        reason: str = "",
        mission_id: str = "",
        trace_id: str = "",
    ) -> SafetyEvent:
        """Log an action that was approved."""
        event = SafetyEvent(
            type="action_approved",
            mission_id=mission_id,
            action_id=action_id,
            trace_id=trace_id,
            risk_level=risk_level,
            approval_mode=approval_mode,
            decision="allowed",
            reason=reason,
            severity="info",
        )
        self._save_event(event)
        return event

    def log_risk_decision(
        self,
        action_id: str,
        risk_level: str,
        decision: str,
        reason: str = "",
        mission_id: str = "",
        trace_id: str = "",
    ) -> SafetyEvent:
        """Log a risk classification decision."""
        event = SafetyEvent(
            type="risk_decision",
            mission_id=mission_id,
            action_id=action_id,
            trace_id=trace_id,
            risk_level=risk_level,
            decision=decision,
            reason=reason,
            severity="info" if decision == "allowed" else "warning",
        )
        self._save_event(event)
        return event

    def log_rollback_required(
        self,
        action_id: str,
        reason: str,
        mission_id: str = "",
        trace_id: str = "",
    ) -> SafetyEvent:
        """Log that a rollback is required for an action."""
        event = SafetyEvent(
            type="rollback_required",
            mission_id=mission_id,
            action_id=action_id,
            trace_id=trace_id,
            decision="rollback_needed",
            reason=reason,
            severity="warning",
        )
        self._save_event(event)
        return event

    def log_escalation(
        self,
        action_id: str,
        reason: str,
        risk_level: str = "critical",
        mission_id: str = "",
        trace_id: str = "",
    ) -> SafetyEvent:
        """Log a safety escalation."""
        event = SafetyEvent(
            type="escalation",
            mission_id=mission_id,
            action_id=action_id,
            trace_id=trace_id,
            risk_level=risk_level,
            decision="escalated",
            reason=reason,
            severity="critical",
        )
        self._save_event(event)
        return event

    def log_secret_detected(
        self,
        action_id: str,
        detail: str = "",
        mission_id: str = "",
        trace_id: str = "",
    ) -> SafetyEvent:
        """Log detection of a secret in output or diff."""
        event = SafetyEvent(
            type="secret_detected",
            mission_id=mission_id,
            action_id=action_id,
            trace_id=trace_id,
            decision="blocked",
            reason="Secret-like content detected",
            detail=detail,
            severity="error",
        )
        self._save_event(event)
        return event

    def log_policy_violation(
        self,
        action_id: str,
        violation: str,
        mission_id: str = "",
        trace_id: str = "",
    ) -> SafetyEvent:
        """Log a policy violation."""
        event = SafetyEvent(
            type="policy_violation",
            mission_id=mission_id,
            action_id=action_id,
            trace_id=trace_id,
            decision="blocked",
            reason=violation,
            severity="error",
        )
        self._save_event(event)
        return event

    # -- Query --

    def list_events(
        self,
        event_type: Optional[str] = None,
        mission_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List safety events with optional filters."""
        events: List[Dict[str, Any]] = []
        for fp in sorted(self._log_dir.glob("sev-*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                if event_type and data.get("type") != event_type:
                    continue
                if mission_id and data.get("mission_id") != mission_id:
                    continue
                if severity and data.get("severity") != severity:
                    continue
                events.append(data)
                if len(events) >= limit:
                    break
            except Exception:
                continue
        return events

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific safety event."""
        path = self._log_dir / f"{event_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def count_blocks(self, mission_id: str = "") -> int:
        """Count blocked events, optionally for a specific mission."""
        count = 0
        for fp in self._log_dir.glob("sev-*.json"):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                if data.get("decision") == "blocked":
                    if mission_id and data.get("mission_id") != mission_id:
                        continue
                    count += 1
            except Exception:
                continue
        return count

    def get_summary(self, mission_id: str = "") -> Dict[str, Any]:
        """Get summary statistics of safety events."""
        events = self.list_events(mission_id=mission_id or None, limit=1000)
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        by_decision: Dict[str, int] = {}
        for e in events:
            by_type[e.get("type", "unknown")] = by_type.get(e.get("type", "unknown"), 0) + 1
            by_severity[e.get("severity", "info")] = by_severity.get(e.get("severity", "info"), 0) + 1
            by_decision[e.get("decision", "")] = by_decision.get(e.get("decision", ""), 0) + 1
        return {
            "total_events": len(events),
            "by_type": by_type,
            "by_severity": by_severity,
            "by_decision": by_decision,
            "mission_id": mission_id or "all",
        }
