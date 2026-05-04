"""Operational diagnostics for IGRIS_GPT.

Detects unhealthy patterns in task execution, memory, and loop behavior:
- Task starvation (pending tasks not progressing)
- Observation loops (same families repeated without progress)
- Blocked task accumulation
- Family failure health (families with high failure rates)
- Recovery escalation (repeated failures requiring attention)

Inspired by IGRIS_DEVIN diagnostics.py but adapted to IGRIS_GPT's architecture.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from igris.core import decision_memory
from igris.core.safety import redact_secrets


# ---------------------------------------------------------------------------
# Diagnostic result models
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticFinding:
    """A single diagnostic finding."""
    category: str  # starvation | observation_loop | blocked_accumulation | family_failure | recovery_escalation
    severity: str  # info | warning | critical
    title: str
    detail: str
    affected_items: List[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "title": self.title,
            "detail": redact_secrets(self.detail),
            "affected_items": self.affected_items,
            "recommendation": self.recommendation,
        }


@dataclass
class DiagnosticReport:
    """Full diagnostic report."""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    findings: List[DiagnosticFinding] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "finding_count": len(self.findings),
            "has_critical": any(f.severity == "critical" for f in self.findings),
            "has_warning": any(f.severity == "warning" for f in self.findings),
        }


# ---------------------------------------------------------------------------
# Diagnostic checks
# ---------------------------------------------------------------------------

STARVATION_THRESHOLD_SECONDS = 300  # 5 minutes pending without progress


def check_task_starvation(tasks: List[Dict[str, Any]]) -> List[DiagnosticFinding]:
    """Detect pending tasks that have been waiting too long."""
    findings: List[DiagnosticFinding] = []
    now = time.time()
    pending = [t for t in tasks if t.get("status") == "pending"]

    if not pending:
        return findings

    stale = []
    for t in pending:
        created = t.get("created_at", "")
        if not created:
            continue
        try:
            ct = time.mktime(time.strptime(created, "%Y-%m-%dT%H:%M:%SZ"))
            age = now - ct
            if age > STARVATION_THRESHOLD_SECONDS:
                stale.append(t)
        except (ValueError, OverflowError):
            pass

    if stale:
        severity = "critical" if len(stale) >= 5 else "warning"
        findings.append(DiagnosticFinding(
            category="starvation",
            severity=severity,
            title=f"{len(stale)} task(s) starving",
            detail=f"{len(stale)} pending task(s) have been waiting for over {STARVATION_THRESHOLD_SECONDS}s without being selected.",
            affected_items=[str(t.get("id", "?")) for t in stale],
            recommendation="Check task selection logic. High-priority starving tasks may need manual intervention.",
        ))

    if len(pending) > 10:
        findings.append(DiagnosticFinding(
            category="starvation",
            severity="warning",
            title=f"{len(pending)} pending tasks accumulated",
            detail=f"Large backlog of {len(pending)} pending tasks. Agent may not be processing fast enough.",
            affected_items=[],
            recommendation="Consider prioritizing or archiving low-priority tasks.",
        ))

    return findings


def check_observation_loop(
    timeline_events: List[Dict[str, Any]],
    window: int = 20,
) -> List[DiagnosticFinding]:
    """Detect observation loops where same families repeat without progress."""
    findings: List[DiagnosticFinding] = []

    recent = timeline_events[-window:] if len(timeline_events) > window else timeline_events
    if len(recent) < 5:
        return findings

    family_counts: Dict[str, int] = {}
    for ev in recent:
        detail = ev.get("detail", "") or ev.get("title", "")
        ev_type = ev.get("type", "")
        if ev_type in ("task", "loop", "command"):
            family = ev.get("family", "") or _extract_family(detail)
            if family:
                family_counts[family] = family_counts.get(family, 0) + 1

    for fam, count in family_counts.items():
        if count >= window * 0.5:
            findings.append(DiagnosticFinding(
                category="observation_loop",
                severity="warning",
                title=f"Observation loop detected: '{fam}'",
                detail=f"Family '{fam}' appeared {count} times in last {window} events. Possible loop without progress.",
                affected_items=[fam],
                recommendation=f"Consider marking '{fam}' as saturated or switching strategy.",
            ))

    return findings


def check_blocked_accumulation(tasks: List[Dict[str, Any]]) -> List[DiagnosticFinding]:
    """Detect accumulation of blocked tasks."""
    findings: List[DiagnosticFinding] = []
    blocked = [t for t in tasks if t.get("status") == "blocked"]

    if len(blocked) >= 3:
        severity = "critical" if len(blocked) >= 5 else "warning"
        reasons = {}
        for t in blocked:
            r = t.get("blocked_reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1

        top_reasons = sorted(reasons.items(), key=lambda x: -x[1])[:3]
        reason_str = "; ".join(f"{r}: {c}" for r, c in top_reasons)

        findings.append(DiagnosticFinding(
            category="blocked_accumulation",
            severity=severity,
            title=f"{len(blocked)} blocked task(s)",
            detail=f"{len(blocked)} tasks are blocked. Top reasons: {reason_str}",
            affected_items=[str(t.get("id", "?")) for t in blocked],
            recommendation="Investigate blocked tasks. Common causes: missing dependencies, safety violations, repeated failures.",
        ))

    return findings


def check_family_failure_health(
    project_root: Optional[str] = None,
) -> List[DiagnosticFinding]:
    """Check failure rates per family from decision memory."""
    findings: List[DiagnosticFinding] = []

    failures = decision_memory.get_recent_failures(limit=50, project_root=project_root)
    decisions = decision_memory.get_recent_decisions(limit=50, project_root=project_root)

    family_failures: Dict[str, int] = {}
    family_decisions: Dict[str, int] = {}

    for f in failures:
        fam = f.get("family", "other")
        family_failures[fam] = family_failures.get(fam, 0) + 1

    for d in decisions:
        fam = d.get("family", "other")
        family_decisions[fam] = family_decisions.get(fam, 0) + 1

    for fam, fail_count in family_failures.items():
        total = family_decisions.get(fam, 0) + fail_count
        if total >= 3 and fail_count / total > 0.5:
            findings.append(DiagnosticFinding(
                category="family_failure",
                severity="warning" if fail_count < 5 else "critical",
                title=f"Family '{fam}' high failure rate",
                detail=f"Family '{fam}': {fail_count} failures out of {total} attempts ({fail_count*100//total}% failure rate).",
                affected_items=[fam],
                recommendation=f"Investigate root cause for '{fam}' failures. Consider saturation or strategy change.",
            ))

    return findings


def check_recovery_escalation(
    project_root: Optional[str] = None,
) -> List[DiagnosticFinding]:
    """Detect repeated recovery attempts that aren't resolving issues."""
    findings: List[DiagnosticFinding] = []

    failures = decision_memory.get_recent_failures(limit=30, project_root=project_root)
    saturated = decision_memory.get_saturated_families(project_root=project_root)

    if len(failures) >= 10:
        findings.append(DiagnosticFinding(
            category="recovery_escalation",
            severity="critical" if len(failures) >= 20 else "warning",
            title=f"{len(failures)} recent failures",
            detail=f"{len(failures)} failures recorded recently. Recovery attempts may not be effective.",
            affected_items=[],
            recommendation="Review failure patterns. Consider manual intervention or strategy reset.",
        ))

    if len(saturated) >= 3:
        findings.append(DiagnosticFinding(
            category="recovery_escalation",
            severity="warning",
            title=f"{len(saturated)} families saturated",
            detail=f"Saturated families: {', '.join(saturated[:5])}. Agent has limited action space.",
            affected_items=saturated[:5],
            recommendation="Consider resetting saturation for families that have been addressed.",
        ))

    return findings


# ---------------------------------------------------------------------------
# Full diagnostic run
# ---------------------------------------------------------------------------

def run_diagnostics(
    tasks: List[Dict[str, Any]],
    timeline_events: List[Dict[str, Any]],
    project_root: Optional[str] = None,
) -> DiagnosticReport:
    """Run all diagnostic checks and produce a report."""
    all_findings: List[DiagnosticFinding] = []

    all_findings.extend(check_task_starvation(tasks))
    all_findings.extend(check_observation_loop(timeline_events))
    all_findings.extend(check_blocked_accumulation(tasks))
    all_findings.extend(check_family_failure_health(project_root=project_root))
    all_findings.extend(check_recovery_escalation(project_root=project_root))

    # Build summary
    categories: Dict[str, int] = {}
    severities: Dict[str, int] = {}
    for f in all_findings:
        categories[f.category] = categories.get(f.category, 0) + 1
        severities[f.severity] = severities.get(f.severity, 0) + 1

    pending = [t for t in tasks if t.get("status") == "pending"]
    blocked = [t for t in tasks if t.get("status") == "blocked"]
    running = [t for t in tasks if t.get("status") == "running"]
    completed = [t for t in tasks if t.get("status") == "completed"]

    summary = {
        "total_tasks": len(tasks),
        "pending": len(pending),
        "running": len(running),
        "completed": len(completed),
        "blocked": len(blocked),
        "categories": categories,
        "severities": severities,
        "healthy": len(all_findings) == 0,
    }

    return DiagnosticReport(findings=all_findings, summary=summary)


def get_diagnostic_summary(
    tasks: List[Dict[str, Any]],
    timeline_events: List[Dict[str, Any]],
    project_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Quick summary for dashboard display."""
    report = run_diagnostics(tasks, timeline_events, project_root=project_root)
    d = report.to_dict()
    return {
        "healthy": d["summary"]["healthy"],
        "finding_count": d["finding_count"],
        "has_critical": d["has_critical"],
        "has_warning": d["has_warning"],
        "categories": d["summary"]["categories"],
        "task_stats": {
            "total": d["summary"]["total_tasks"],
            "pending": d["summary"]["pending"],
            "running": d["summary"]["running"],
            "completed": d["summary"]["completed"],
            "blocked": d["summary"]["blocked"],
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_family(text: str) -> str:
    """Try to extract a family name from event text."""
    text_lower = text.lower()
    families = ["test", "code", "deploy", "analyze", "config", "doc", "security", "infra"]
    for f in families:
        if f in text_lower:
            return f
    return ""
