"""
Behavior Tracker — IGRIS Supervisor-First Autonomy Policy (issue #147).

The supervisor is the FIRST and PRIMARY responsible for quality of every run.
This module provides:

  BehaviorRecord  — a single classified observation (blocking or non-blocking)
  BehaviorTracker — collects records across a run, runs self-audit, opens issues

Design principles (from #147):
- Blocking defects  → repair cycle → PR → retry (or escalate if budget exhausted)
- Non-blocking, safe → fix in a dedicated PR within repair budget
- Non-blocking, unsafe now → open issue with evidence, severity, run_id
- Supervisor misses a defect → separate issue against supervisor capability

Advisory-only: this module NEVER blocks a run or changes loop decisions.
It records observations; the supervisor decides what to do.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Behavior codes (mirrors FAILURE_ERROR_CODES in self_repair_supervisor.py)
# ---------------------------------------------------------------------------

#: Agent-loop behaviors
AGENT_BEHAVIORS: Dict[str, str] = {
    "E001": "wrong_file_edit",
    "E002": "reasoning_loop_no_progress",   # same action repeated ≥3 times
    "E003": "no_finish_on_convergence",      # loop didn't call finish/blocked at convergence
    "E004": "duplicate_insertion",
    "E005": "brittle_test",                  # assert True / assert == 200 on stub
    "E006": "incomplete_report",             # final report missing facts
    "E007": "no_rollback_after_ast_failure",
    "E008": "no_diff_repair",                # reasoning stopped without clean diff
}

#: Supervisor-self behaviors
SUPERVISOR_BEHAVIORS: Dict[str, str] = {
    "E010": "wrong_failure_classification",
    "E011": "no_issue_for_non_blocking_defect",   # defect observed but no issue opened
    "E012": "no_repair_for_retryable_bug",
    "E013": "success_without_verification",        # completed without smoke/pytest evidence
    "E014": "diagnostics_hidden_or_truncated",
    "E015": "dirty_workspace_after_blocked",       # workspace not cleaned after blocked run
    "E016": "no_escalation_at_budget_exhaustion",
    "E017": "stage_scope_leak",
    "E018": "repair_without_progress",             # repair cycles used, same failure persists
}

ALL_BEHAVIOR_CODES = {**AGENT_BEHAVIORS, **SUPERVISOR_BEHAVIORS}
BEHAVIOR_BY_NAME = {v: k for k, v in ALL_BEHAVIOR_CODES.items()}


@dataclass
class BehaviorRecord:
    """A single classified observation during a supervised run."""
    code: str                    # e.g. "E002"
    name: str                    # e.g. "reasoning_loop_no_progress"
    detail: str                  # human-readable description
    severity: str = "low"        # low | medium | high | critical
    blocking: bool = False
    stage_id: str = ""
    evidence: str = ""           # snippet / log fragment
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    issue_url: str = ""          # filled if a GitHub issue was opened


@dataclass
class SelfAuditResult:
    """Result of the supervisor self-audit at end of run."""
    missed_behaviors: List[str] = field(default_factory=list)
    opened_issues: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class BehaviorTracker:
    """
    Collect, classify, and act on behavioral observations during a supervised run.

    Usage (inside SelfRepairSupervisor):
        tracker = BehaviorTracker(run_id=run.run_id)
        tracker.record("E002", "reasoning repeated read_file 4× without progress", severity="medium")
        ...
        audit = tracker.self_audit(run, project_root)
    """

    def __init__(self, run_id: str = "", issue_number: Optional[int] = None) -> None:
        self.run_id = run_id
        self.issue_number = issue_number
        self.records: List[BehaviorRecord] = []
        self._auto_open = os.getenv("IGRIS_AUTO_OPEN_DEFECT_ISSUES", "false").lower() == "true"

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        code_or_name: str,
        detail: str,
        severity: str = "low",
        blocking: bool = False,
        stage_id: str = "",
        evidence: str = "",
    ) -> BehaviorRecord:
        """Record a behavior observation."""
        # Normalise code/name
        if code_or_name in ALL_BEHAVIOR_CODES:
            code = code_or_name
            name = ALL_BEHAVIOR_CODES[code]
        elif code_or_name in BEHAVIOR_BY_NAME:
            name = code_or_name
            code = BEHAVIOR_BY_NAME[name]
        else:
            code = "E999"
            name = code_or_name

        rec = BehaviorRecord(
            code=code, name=name, detail=detail,
            severity=severity, blocking=blocking,
            stage_id=stage_id, evidence=evidence[:500],
        )
        self.records.append(rec)
        logger.debug("BehaviorTracker [%s] %s: %s", code, name, detail[:100])
        return rec

    def non_blocking(self) -> List[BehaviorRecord]:
        return [r for r in self.records if not r.blocking]

    def blocking(self) -> List[BehaviorRecord]:
        return [r for r in self.records if r.blocking]

    def by_severity(self, severity: str) -> List[BehaviorRecord]:
        return [r for r in self.records if r.severity == severity]

    # ------------------------------------------------------------------
    # Self-audit
    # ------------------------------------------------------------------

    def self_audit(
        self,
        *,
        run_status: str,
        failure_class: str,
        repair_cycles_used: int,
        smoke_ran: bool,
        pytest_ran: bool,
        workspace_dirty: bool,
        escalation_budget_exhausted: bool,
        escalation_was_called: bool,
        completion_mode: str = "",
        project_root: str = "",
    ) -> SelfAuditResult:
        """
        Run supervisor self-audit at end of run.

        Detects cases where the supervisor SHOULD have acted but didn't,
        and records them as additional E01x behaviors.
        """
        result = SelfAuditResult()

        # E013: success declared without smoke/pytest
        if run_status == "completed" and not smoke_ran and not pytest_ran:
            r = self.record(
                "E013", "Run completed but neither smoke nor pytest ran — outcome unverified",
                severity="high",
            )
            result.missed_behaviors.append(r.name)
            result.notes.append("Add smoke/pytest step before declaring completion")

        # E015: dirty workspace after blocked run
        if run_status in ("blocked", "interrupted") and workspace_dirty:
            r = self.record(
                "E015", f"Workspace dirty after {run_status} run — branch/files not cleaned",
                severity="medium",
            )
            result.missed_behaviors.append(r.name)
            result.notes.append("Supervisor must clean workspace after blocked/interrupted runs")

        # E016: no escalation at budget exhaustion
        if escalation_budget_exhausted and not escalation_was_called and run_status != "completed":
            r = self.record(
                "E016", "Repair budget exhausted but escalation API was not called",
                severity="high",
            )
            result.missed_behaviors.append(r.name)
            result.notes.append("Call /api/supervisor/escalate when repair budget is exhausted")

        # E018: repair cycles used without progress (same failure_class persists)
        if repair_cycles_used > 0 and failure_class and run_status == "blocked":
            r = self.record(
                "E018", f"Repair cycles={repair_cycles_used} used but run still blocked with {failure_class}",
                severity="medium",
            )
            result.missed_behaviors.append(r.name)

        # E008: reasoning stopped without clean diff
        if completion_mode in ("degraded", "no_diff_repair", "stopped") and run_status == "completed":
            r = self.record(
                "E008", f"Reasoning stopped without clean finish (mode={completion_mode})",
                severity="medium",
            )
            result.missed_behaviors.append(r.name)
            result.notes.append("Review diff carefully — reasoning did not converge")

        # E011: non-blocking defects observed but no issue was opened
        non_blocking_high = [
            r for r in self.non_blocking()
            if r.severity in ("high", "critical") and not r.issue_url
        ]
        if non_blocking_high and not self._auto_open:
            result.notes.append(
                f"{len(non_blocking_high)} high-severity non-blocking defects not tracked in issues. "
                "Set IGRIS_AUTO_OPEN_DEFECT_ISSUES=true to auto-open."
            )

        # Auto-open issues for non-blocking defects if configured
        if self._auto_open and project_root and self.non_blocking():
            opened = self._open_defect_issues(project_root)
            result.opened_issues.extend(opened)

        return result

    # ------------------------------------------------------------------
    # GitHub issue auto-opening
    # ------------------------------------------------------------------

    def _open_defect_issues(self, project_root: str) -> List[str]:
        """Open a GitHub issue for each high-severity non-blocking defect not yet tracked."""
        opened: List[str] = []
        for rec in self.non_blocking():
            if rec.severity not in ("high", "critical"):
                continue
            if rec.issue_url:
                continue
            url = self._open_single_issue(rec, project_root)
            if url:
                rec.issue_url = url
                opened.append(url)
        return opened

    def _open_single_issue(self, rec: BehaviorRecord, project_root: str) -> str:
        body = (
            f"## Supervisor-detected non-blocking defect\n\n"
            f"**Code**: `{rec.code}` — `{rec.name}`  \n"
            f"**Severity**: {rec.severity}  \n"
            f"**Run ID**: `{self.run_id}`  \n"
            f"**Issue**: #{self.issue_number or 'N/A'}  \n\n"
            f"### Detail\n{rec.detail}\n\n"
            f"### Evidence\n```\n{rec.evidence or 'none'}\n```\n\n"
            f"*Auto-opened by IGRIS BehaviorTracker — supervisor-first autonomy policy (#147)*"
        )
        title = f"[supervisor-defect] {rec.name}: {rec.detail[:60]}"
        try:
            proc = subprocess.run(
                ["gh", "issue", "create", "--title", title, "--body", body,
                 "--label", "supervisor-defect,autonomy"],
                capture_output=True, text=True, timeout=30, cwd=project_root,
            )
            if proc.returncode == 0:
                url = proc.stdout.strip()
                logger.info("BehaviorTracker opened issue: %s", url)
                return url
        except Exception as exc:
            logger.warning("BehaviorTracker failed to open issue: %s", exc)
        return ""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "issue_number": self.issue_number,
            "total": len(self.records),
            "blocking_count": len(self.blocking()),
            "non_blocking_count": len(self.non_blocking()),
            "records": [
                {
                    "code": r.code, "name": r.name, "detail": r.detail,
                    "severity": r.severity, "blocking": r.blocking,
                    "stage_id": r.stage_id, "timestamp": r.timestamp,
                    "issue_url": r.issue_url,
                }
                for r in self.records
            ],
        }

    def summary(self) -> str:
        if not self.records:
            return "no behaviors recorded"
        counts: Dict[str, int] = {}
        for r in self.records:
            counts[r.name] = counts.get(r.name, 0) + 1
        parts = [f"{name}×{n}" for name, n in sorted(counts.items())]
        return f"{len(self.records)} behaviors: {', '.join(parts)}"
