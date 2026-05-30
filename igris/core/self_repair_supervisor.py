"""Autonomous self-repair supervisor for controlled rank missions.

The supervisor coordinates an IGRIS rank attempt and bounded infrastructure
repair cycles. It does not expose free-form shell execution: the default
backend runs fixed argv commands only, and tests can inject a fake backend.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import shlex
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

from igris.core.safety import redact_secrets
from igris.core.failure_memory import FailureMemory, FailureRisk
from igris.core.acceptance_gate import check_acceptance_evidence

# AssignmentRouter — lazy import to avoid circular deps at module load
_assignment_router_available = False
try:
    from igris.core.assignment_router import AssignmentRequest, AssignmentDecision, AssignmentRouter
    from igris.core.assignment_outcomes import compute_task_signature, save_assignment_outcome
    _assignment_router_available = True
except ImportError:
    pass

# MissionBrain Advisory — lazy import, monitoring-only, never blocks run (#914)
_selected_advisory_available = False
try:
    from igris.agent.mission.selected_advisory import (
        enrich_cycle_selected as _enrich_cycle_selected,
        make_selected_monitoring_config as _make_selected_monitoring_config,
    )
    _selected_advisory_available = True
except ImportError:
    pass


REPAIRABLE_FAILURES = {
    "pytest_failure",
    "reasoning_loop_blocked",
    "max_steps",
    "ask_user",
    "missing_tests",
    "missing_ui_visibility",
    "wrong_file_edit",
    "infrastructure_bug",
    "invalid_bootstrap",
    "syntax_error",
    "semantic_incomplete",
    "test_runner_timeout",
}

FAILURE_ERROR_CODES = {
    "pytest_failure": "E001",
    "missing_tests": "E002",
    "syntax_error": "E003",
    "wrong_file_edit": "E004",
    "reasoning_loop_blocked": "E005",
    "max_steps": "E006",
    "ask_user": "E007",
    "infrastructure_bug": "E008",
    "invalid_bootstrap": "E009",
    "semantic_incomplete": "E010",
    "test_runner_timeout": "E011",
    "decomposition_required": "E012",
    "capability_ceiling_reached": "E013",
    "execution_budget_exceeded": "E014",
    "workspace_dirty": "E015",
    "destructive_diff": "E016",
    "missing_ui_visibility": "E017",
}

# Signals that accumulate across repair cycles to indicate model capability limits.
# Decomposition is triggered when any single signal reaches CAPABILITY_LIMIT_THRESHOLD,
# OR when the combined total of all signals reaches it (mixed-failure capability wall).
CAPABILITY_LIMIT_SIGNALS = frozenset({"reasoning_timeout", "pytest_hang", "no_diff_repair"})
CAPABILITY_LIMIT_THRESHOLD = 2

# Pre-flight mission planning: a lightweight read-only reasoning pass that
# estimates complexity and recommends decomposition BEFORE any code is written.
PLANNING_MAX_STEPS = 20
PLANNING_TIMEOUT_SECONDS = 60

# Required fields in a valid IGRIS decomposition response.
DECOMPOSITION_REQUIRED_FIELDS = (
    "why_too_large",
    "sub_missions",
    "first_sub_mission",
    "human_approval_required",
)

RETRYABLE_REPAIR_FAILURES = {
    "reasoning_loop_blocked",
    "missing_ui_visibility",
    "missing_tests",
    "wrong_file_edit",
    "max_steps",
    "syntax_error",
    "pytest_failure",
}

UNSAFE_STATUS_PREFIXES = (
    "?? .env",
    "?? .venv",
    "?? .pytest_cache",
    "?? __pycache__",
    "?? .igris",
)

WRITE_ACTION_TYPES = frozenset({
    "write_file",
    "insert_after",
    "insert_before",
    "replace_range",
    "append_file",
    "apply_patch",
})

AUDIT_STATUSES = {
    "audit-new",
    "audit-reviewed",
    "audit-fixed",
    "audit-deferred",
    "audit-false-positive",
}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _safe_redact(value: Any) -> str:
    return redact_secrets(_safe_text(value))


def _failure_error_code(failure_class: str) -> str:
    return FAILURE_ERROR_CODES.get(str(failure_class or "").strip(), "E999")


def _command_detail(result: "CommandResult") -> str:
    parts = []
    if result.output:
        parts.append(_safe_text(result.output).rstrip())
    if result.error:
        parts.append(_safe_text(result.error).rstrip())
    return "\n".join(part for part in parts if part)


def _infer_targeted_tests(goal: str, explicit_targets: List[str]) -> List[str]:
    targets = list(explicit_targets)
    seen = set(targets)
    for match in re.findall(r"tests/test_[A-Za-z0-9_]+\.py", goal):
        if match not in seen:
            targets.append(match)
            seen.add(match)
    return targets


def _infer_dry_run(data: Dict[str, Any]) -> bool:
    if "dry_run" in data:
        return bool(data.get("dry_run"))
    return not (
        bool(data.get("allow_github_pr", False))
        or bool(data.get("allow_merge_if_green", False))
    )


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def _parse_issue_number(explicit: Any, goal: str = "") -> int:
    """Extract issue number from explicit value or goal string (e.g. '#614').

    Returns 0 if not found or invalid.
    """
    try:
        if explicit:
            n = int(explicit)
            if n > 0:
                return n
    except (TypeError, ValueError):
        pass
    # Fallback: parse first #NNN from goal string
    m = re.search(r"#(\d+)", goal)
    if m:
        try:
            return int(m.group(1))
        except (TypeError, ValueError):
            pass
    return 0


@dataclass
class CommandResult:
    success: bool = False
    output: str = ""
    error: str = ""
    returncode: int = 0
    # Telemetry fields set by call_api_helper for helper A/B test tracking
    helper_model: str = ""
    helper_ab_active: bool = False
    helper_ab_alt_model: str = ""
    # Shadow mode fields (Epic #445)
    helper_ab_shadow_mode: bool = False
    helper_primary_score: float = 0.0
    helper_alt_score: float = 0.0
    helper_primary_cost_usd: float = 0.0
    helper_alt_cost_usd: float = 0.0
    helper_primary_latency_ms: int = 0
    helper_alt_latency_ms: int = 0
    helper_switch_recommendation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": _safe_redact(self.output),
            "error": _safe_redact(self.error),
            "returncode": self.returncode,
        }


@dataclass
class RankSupervisorConfig:
    goal: str
    rank_id: str = "rank"
    max_rank_attempts: int = 2  # Issue #710: default ≥ 2; env-overridable via IGRIS_MAX_RANK_ATTEMPTS
    max_repair_cycles: int = 2
    allow_github_pr: bool = False
    allow_merge_if_green: bool = False
    service_restart_command: str = ""
    required_smoke_endpoints: List[str] = field(default_factory=list)
    targeted_tests: List[str] = field(default_factory=list)
    dry_run: bool = True
    defer_service_restart: bool = False
    # Idle timeout: kill the pytest subprocess only when it produces *no output*
    # for this many seconds.  A healthy (but slow) suite keeps printing dots, so
    # the timer resets on every line; only a hung/stuck process is killed.
    # 300s (5 min) accommodates individual slow integration tests that may take
    # 2-3 min without printing, while still catching genuinely hung processes.
    test_timeout_seconds: int = int(os.getenv("IGRIS_TEST_RUNNER_TIMEOUT_SECONDS", "300"))
    # Absolute ceiling: kill unconditionally after this many seconds regardless
    # of output activity (safety net against infinite-loop tests).
    test_hard_cap_seconds: int = 3600
    reasoning_timeout_seconds: int = 300
    allow_api_escalation: bool = False
    max_api_escalations_per_run: int = 0
    max_api_budget_usd: float = 0.0
    max_tokens_per_escalation: int = 600
    api_helper_model: str = "gpt-5.4-mini"
    enable_mission_planning: bool = False
    allow_auto_subissues: bool = _as_bool(os.getenv("IGRIS_ALLOW_AUTO_SUBISSUES_DEFAULT"), True)
    enable_semantic_gate: bool = True
    allow_roadmap_autoselect: bool = False
    api_helper_mode: str = ""
    # Depth counter incremented each time a child run is spawned via auto-chain.
    # Guards against infinite cascade: parent→child→grandchild→... stops at depth 2.
    autochain_depth: int = 0
    no_diff_steps_max: int = 20
    # Cross-run history: populated by the watchdog from _issue_failures so the
    # assignment router knows how many prior attempts have been made for this
    # issue.  Enables hard_debugging escalation (→ gpu_reasoning → VastAI) on
    # repeated failures instead of always starting from code_reasoning.
    prior_attempts: int = 0
    # Aggregated capability_signals from the last failed run for this issue.
    # Merged with the current run's signals before the initial AssignmentRequest
    # so that accumulated no_diff_repair / reasoning_timeout counts survive
    # across watchdog cycles.
    prior_capability_signals: Dict[str, int] = field(default_factory=dict)
    # Issue #730 — force re-validation of baseline cache even on SHA hit
    force_revalidate_baseline: bool = False
    # Issue #615 — issue number for pre-run dependency validation (0 = not set)
    issue_number: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RankSupervisorConfig":
        return cls(
            goal=str(data.get("goal", "")),
            rank_id=str(data.get("rank_id", "rank")),
            max_rank_attempts=max(1, int(data.get(
                "max_rank_attempts",
                int(os.getenv("IGRIS_MAX_RANK_ATTEMPTS", str(cls.max_rank_attempts))),
            ))),
            max_repair_cycles=max(0, int(data.get("max_repair_cycles", cls.max_repair_cycles))),
            allow_github_pr=_as_bool(data.get("allow_github_pr"), False),
            allow_merge_if_green=_as_bool(data.get("allow_merge_if_green"), False),
            service_restart_command=str(data.get("service_restart_command", "")),
            required_smoke_endpoints=list(data.get("required_smoke_endpoints", [])),
            targeted_tests=_infer_targeted_tests(
                str(data.get("goal", "")),
                list(data.get("targeted_tests", [])),
            ),
            dry_run=_infer_dry_run(data),
            defer_service_restart=_as_bool(data.get("defer_service_restart"), False),
            test_timeout_seconds=max(30, int(data.get("test_timeout_seconds", 300))),
            test_hard_cap_seconds=max(60, int(data.get("test_hard_cap_seconds", 3600))),
            reasoning_timeout_seconds=max(30, int(
                data.get("reasoning_timeout_seconds")
                or os.environ.get("IGRIS_REASONING_TIMEOUT_SECONDS")
                or 300
            )),
            allow_api_escalation=_as_bool(data.get("allow_api_escalation"), False),
            max_api_escalations_per_run=max(0, int(data.get("max_api_escalations_per_run", 0))),
            max_api_budget_usd=max(0.0, float(data.get("max_api_budget_usd", 0.0))),
            max_tokens_per_escalation=max(64, int(data.get("max_tokens_per_escalation", 600))),
            api_helper_model=str(data.get("api_helper_model", "gpt-5.4-mini")),
            enable_mission_planning=_as_bool(data.get("enable_mission_planning"), False),
            allow_auto_subissues=_as_bool(
                data.get("allow_auto_subissues"),
                _as_bool(os.getenv("IGRIS_ALLOW_AUTO_SUBISSUES_DEFAULT"), True),
            ),
            enable_semantic_gate=_as_bool(data.get("enable_semantic_gate"), True),
            allow_roadmap_autoselect=_as_bool(data.get("allow_roadmap_autoselect"), False),
            api_helper_mode=str(data.get("api_helper_mode", "")),
            autochain_depth=max(0, int(data.get("autochain_depth", 0) or data.get("_autochain_depth", 0))),
            no_diff_steps_max=max(1, int(data.get("no_diff_steps_max", 20))),
            prior_attempts=max(0, int(data.get("prior_attempts", 0))),
            prior_capability_signals=dict(data.get("prior_capability_signals") or {}),
            # Issue #730 — force baseline re-validation even on SHA hit
            force_revalidate_baseline=_as_bool(data.get("force_revalidate_baseline"), False),
            # Issue #615 — issue number for dependency pre-check
            issue_number=_parse_issue_number(data.get("issue_number", 0), str(data.get("goal", ""))),
        )


@dataclass
class MissionStage:
    stage_id: str
    goal: str
    required: bool
    allowed_file_families: List[str]
    acceptance_criteria: List[str]
    validation: List[str]
    rollback_policy: str
    preserved_progress_policy: str
    failure_classification: List[str]
    repair_strategy: str
    report_entry: str


@dataclass
class MissionPlan:
    mode: str
    stages: List[MissionStage]


@dataclass
class SupervisorEvent:
    phase: str
    status: str
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    audit_status: str = "audit-new"
    audit_reviewed_by: str = ""
    audit_reviewed_at: str = ""
    audit_review_id: str = ""
    audit_scope_hash: str = ""
    audit_next_review_after: str = ""
    audit_resolution_pr: str = ""
    audit_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "status": self.status,
            "detail": _safe_redact(self.detail),
            "data": {k: _safe_redact(v) for k, v in self.data.items()},
            "timestamp": self.timestamp,
            "audit_status": self.audit_status,
            "audit_reviewed_by": self.audit_reviewed_by,
            "audit_reviewed_at": self.audit_reviewed_at,
            "audit_review_id": self.audit_review_id,
            "audit_scope_hash": self.audit_scope_hash,
            "audit_next_review_after": self.audit_next_review_after,
            "audit_resolution_pr": self.audit_resolution_pr,
            "audit_notes": _safe_redact(self.audit_notes),
        }


@dataclass
class SupervisorRun:
    run_id: str
    rank_id: str
    status: str = "running"
    outcome: str = ""
    failure_class: str = ""
    branch: str = ""
    repair_cycles_used: int = 0
    max_repair_cycles: int = 0
    api_escalations_used: int = 0
    api_escalations_failed_unconfigured: int = 0
    api_budget_used_usd: float = 0.0
    max_api_escalations_per_run: int = 0
    max_api_budget_usd: float = 0.0
    events: List[SupervisorEvent] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)
    audit_resolver: Any = None
    update_hook: Any = None
    cancel_requested: bool = False
    cancel_reason: str = ""
    # Capability-limit tracking: maps signal name → count across all attempts/repairs.
    capability_signals: Dict[str, int] = field(default_factory=dict)
    # Decomposition produced by IGRIS when capability_limit is detected.
    decomposition: Optional[Dict[str, Any]] = None
    # Pre-flight scope assessment produced by the mission planning pass.
    mission_scope: Optional[Dict[str, Any]] = None
    # Goal string copied from config so it's available in terminal callbacks.
    goal: str = ""
    # Semantic acceptance gate result (set by the gate, survives report overwrites).
    acceptance_evidence: Optional[Dict[str, Any]] = None
    # Cost-policy execution strategy telemetry.
    strategy_used: str = ""
    same_failure_count: int = 0
    last_repair_failure: str = ""
    execution_budget_used_usd: float = 0.0
    autorun_child_run_id: str = ""
    autorun_policy: str = ""
    autorun_skipped_reason: str = ""
    # MBOP Phase 1 intake — set before supervisor.run() so _rank_initial_context can read it (#1040)
    mbop_intake: Any = None  # Optional[MBOPIntakeResult]
    # Supervisor-first autonomy policy (#147)
    completion_mode: str = ""        # set at end of run; read by MBOP Phase 11
    behavior_tracker: Any = None     # BehaviorTracker instance; created in _worker

    def add(self, phase: str, status: str, detail: str = "", **data: Any) -> None:
        event = SupervisorEvent(phase=phase, status=status, detail=detail, data=data)
        if callable(self.audit_resolver):
            self.audit_resolver(event)
        self.events.append(event)
        if callable(self.update_hook):
            self.update_hook(self)

    def touch(self) -> None:
        if callable(self.update_hook):
            self.update_hook(self)

    @property
    def started_at(self) -> Optional[float]:
        """Unix timestamp of the first event, or None if no events exist."""
        return self.events[0].timestamp if self.events else None

    @property
    def last_updated_at(self) -> Optional[float]:
        """Unix timestamp of the most recent event, or None if no events exist."""
        return self.events[-1].timestamp if self.events else None

    def is_zombie(self, threshold_seconds: float = 1800.0) -> bool:
        """Return True if the run is stuck: status is 'running' but no new event
        has been recorded in the last threshold_seconds.

        A long-running but active session is not a zombie — only one that has
        stopped producing events (no actions, no updates) for an extended period.
        """
        import time
        if self.status not in ("running", "cancelling"):
            return False
        last = self.last_updated_at
        if last is None:
            return False
        return (time.time() - last) > threshold_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "rank_id": self.rank_id,
            "status": self.status,
            "outcome": self.outcome,
            "failure_class": self.failure_class,
            "branch": self.branch,
            "repair_cycles_used": self.repair_cycles_used,
            "max_repair_cycles": self.max_repair_cycles,
            "api_escalations_used": self.api_escalations_used,
            "api_escalations_failed_unconfigured": self.api_escalations_failed_unconfigured,
            "api_budget_used_usd": round(self.api_budget_used_usd, 6),
            "max_api_escalations_per_run": self.max_api_escalations_per_run,
            "max_api_budget_usd": round(self.max_api_budget_usd, 6),
            "events": [e.to_dict() for e in self.events],
            "report": self.report,
            "cancel_requested": bool(self.cancel_requested),
            "cancel_reason": _safe_redact(self.cancel_reason),
            "capability_signals": dict(self.capability_signals),
            "decomposition": self.decomposition,
            "mission_scope": self.mission_scope,
            "goal": self.goal,
            "strategy_used": self.strategy_used,
            "same_failure_count": self.same_failure_count,
            "execution_budget_used_usd": round(self.execution_budget_used_usd, 6),
            "autorun_child_run_id": self.autorun_child_run_id,
            "autorun_policy": self.autorun_policy,
            "autorun_skipped_reason": self.autorun_skipped_reason,
            "completion_mode": self.completion_mode,
        }


class SupervisorBackend(Protocol):
    def git_status(self) -> CommandResult: ...
    def git_log_head(self) -> CommandResult: ...
    def create_branch(self, branch: str) -> CommandResult: ...
    def run_reasoning(self, goal: str, max_steps: int, initial_context: Dict[str, Any], timeout: int = 300, task_type: str = "code_reasoning", preferred_profile: Optional[str] = None) -> Dict[str, Any]: ...
    def git_diff_stat(self) -> CommandResult: ...
    def git_diff(self) -> CommandResult: ...
    def run_tests(self, targets: Optional[List[str]] = None, timeout: int = 120, hard_cap: int = 3600) -> CommandResult: ...
    def run_test_diagnostics(self, timeout: int = 120) -> CommandResult: ...
    def smoke(self, endpoints: List[str], restart_command: str = "") -> CommandResult: ...
    def commit(self, message: str, files: Optional[List[str]] = None) -> CommandResult: ...
    def push_branch(self, branch: str) -> CommandResult: ...
    def open_pr(self, branch: str, title: str, body: str) -> CommandResult: ...
    def wait_ci(self) -> CommandResult: ...
    def merge_pr(self) -> CommandResult: ...
    def pull_main(self) -> CommandResult: ...
    def create_issue(self, title: str, body: str) -> CommandResult: ...
    def update_issue(self, issue_url: str, comment_body: str) -> CommandResult: ...
    def fetch_issue(self, issue_url: str) -> CommandResult: ...
    def restore_dangerous_diff(self) -> CommandResult: ...
    def restore_paths(self, paths: List[str]) -> CommandResult: ...
    def checkout_main(self) -> CommandResult: ...
    def delete_stale_rank_branches(self) -> CommandResult: ...
    def call_api_helper(self, packet: Dict[str, Any], model: str, max_tokens: int, timeout: int = 45) -> CommandResult: ...
    def api_helper_is_configured(self) -> bool: ...


class LocalSupervisorBackend:
    """Governed local backend using fixed argv commands only."""

    # LLM provider credentials forwarded to reasoning subprocesses when
    # forward_credentials=True — allows ModelOrchestrator inside the worker
    # to reach cloud providers instead of falling back to Ollama only.
    _REASONING_CREDENTIAL_ALLOWLIST: frozenset = frozenset({
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "ANTHROPIC_API_KEY",
        "IGRIS_API_HELPER_COMMAND",
        "IGRIS_API_HELPER_MODE",
        "IGRIS_API_HELPER_PROVIDER",
        "IGRIS_API_HELPER_MODEL",
        "IGRIS_EXECUTION_STRONG_MODEL",
        "IGRIS_EXECUTION_FALLBACK_MODEL",
        "IGRIS_ENABLE_CODEX_DIRECT_EXECUTION",
    })

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)

    def _subprocess_env(self, *, clean_for_tests: bool = False, forward_credentials: bool = False) -> Dict[str, str]:
        if not clean_for_tests:
            env = os.environ.copy()
            env["IGRIS_SUPERVISOR_CHILD"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            env.pop("PYTEST_CURRENT_TEST", None)
            return env
        allowlist: set = {
            "HOME",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "LOGNAME",
            "PATH",
            "PYTHONPATH",
            "SHELL",
            "TERM",
            "TMPDIR",
            "TZ",
            "USER",
        }
        if forward_credentials:
            allowlist = allowlist | LocalSupervisorBackend._REASONING_CREDENTIAL_ALLOWLIST
        env = {
            key: value
            for key, value in os.environ.items()
            if key in allowlist and value
        }
        env.setdefault("PATH", os.defpath)
        env["IGRIS_SUPERVISOR_CHILD"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _run(
        self,
        cmd: List[str],
        timeout: int = 120,
        *,
        input_text: Optional[str] = None,
        clean_env: bool = False,
        forward_credentials: bool = False,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> CommandResult:
        try:
            env = self._subprocess_env(clean_for_tests=clean_env, forward_credentials=forward_credentials)
            if extra_env:
                env.update(extra_env)
            proc = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                env=env,
                input=input_text,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
                start_new_session=True,
                close_fds=True,
            )
            return CommandResult(
                success=proc.returncode == 0,
                output=proc.stdout,
                error=proc.stderr,
                returncode=proc.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            output = _safe_text(exc.stdout)
            error = _safe_text(exc.stderr) or "Command timed out"
            return CommandResult(False, output, error, 124)
        except OSError as exc:
            return CommandResult(False, "", str(exc), 1)

    def _run_adaptive(
        self,
        cmd: List[str],
        *,
        idle_timeout: int = 120,
        hard_cap: int = 3600,
        clean_env: bool = False,
    ) -> CommandResult:
        """Run a command with activity-based dynamic timeout.

        The subprocess is killed only when:
        - no output (stdout OR stderr) has been produced for ``idle_timeout``
          seconds — the process is considered hung/stuck, OR
        - total wall-clock time exceeds ``hard_cap`` seconds — absolute
          safety net against infinite-loop tests.

        A healthy long-running command (e.g. pytest printing progress dots)
        continuously resets the idle timer and will never be killed by it.
        """
        env = self._subprocess_env(clean_for_tests=clean_env)
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            return CommandResult(False, "", str(exc), 1)

        out_q: queue.Queue = queue.Queue()
        err_q: queue.Queue = queue.Queue()

        def _reader(pipe, q: queue.Queue) -> None:
            try:
                for line in pipe:
                    q.put(line)
            finally:
                q.put(None)  # sentinel — pipe closed

        threading.Thread(target=_reader, args=(proc.stdout, out_q), daemon=True).start()
        threading.Thread(target=_reader, args=(proc.stderr, err_q), daemon=True).start()

        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
        start = time.monotonic()
        last_active = start
        out_alive = err_alive = True
        kill_reason: Optional[str] = None

        while out_alive or err_alive:
            now = time.monotonic()
            if now - start >= hard_cap:
                kill_reason = f"hard cap {hard_cap}s exceeded"
                break
            if now - last_active >= idle_timeout:
                kill_reason = f"no output for {idle_timeout}s (idle timeout)"
                break

            drained = False
            for q_pipe, parts, name in (
                (out_q, stdout_parts, "out"),
                (err_q, stderr_parts, "err"),
            ):
                while True:
                    try:
                        chunk = q_pipe.get_nowait()
                    except queue.Empty:
                        break
                    if chunk is None:
                        if name == "out":
                            out_alive = False
                        else:
                            err_alive = False
                    else:
                        parts.append(chunk)
                        last_active = time.monotonic()
                        drained = True

            if not drained:
                time.sleep(0.05)

        if kill_reason:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                try:
                    proc.kill()
                except OSError:
                    pass
            proc.wait()
            return CommandResult(
                False,
                "".join(stdout_parts),
                f"Command killed: {kill_reason}",
                124,
            )

        proc.wait()
        return CommandResult(
            success=proc.returncode == 0,
            output="".join(stdout_parts),
            error="".join(stderr_parts),
            returncode=proc.returncode,
        )

    # Seconds without a heartbeat update before the worker is considered stale.
    # Workers write every 30s; 120s allows ~3 missed beats before we kill early.
    _HEARTBEAT_STALE_SECONDS = 120

    def _run_with_heartbeat_monitor(
        self,
        cmd: List[str],
        timeout: int,
        *,
        input_text: str,
        heartbeat_path: str,
        stale_threshold: int = _HEARTBEAT_STALE_SECONDS,
        forward_credentials: bool = False,
    ) -> CommandResult:
        """Run a subprocess, killing it early if its heartbeat file goes stale.

        Uses Popen so we can poll both process state and the heartbeat file
        simultaneously, rather than blocking on subprocess.run() for the full
        timeout when the worker silently hangs.
        """
        env = self._subprocess_env(forward_credentials=forward_credentials)
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            return CommandResult(False, "", str(exc), 1)

        stdout_parts: List[str] = []
        stderr_parts: List[str] = []

        def _read_pipe(pipe: Any, parts: List[str]) -> None:
            try:
                for line in pipe:
                    parts.append(line)
            except (OSError, ValueError):
                pass

        def _write_stdin() -> None:
            try:
                proc.stdin.write(input_text)  # type: ignore[union-attr]
            except OSError:
                pass
            finally:
                try:
                    proc.stdin.close()  # type: ignore[union-attr]
                except OSError:
                    pass

        threading.Thread(target=_write_stdin, daemon=True).start()
        out_thread = threading.Thread(target=_read_pipe, args=(proc.stdout, stdout_parts), daemon=True)
        err_thread = threading.Thread(target=_read_pipe, args=(proc.stderr, stderr_parts), daemon=True)
        out_thread.start()
        err_thread.start()

        start_mono = time.monotonic()
        last_hb_at: Optional[float] = None
        kill_reason = ""

        def _kill(reason: str) -> None:
            nonlocal kill_reason
            kill_reason = reason
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                try:
                    proc.kill()
                except OSError:
                    pass

        _POLL = 5
        while True:
            time.sleep(_POLL)
            elapsed = time.monotonic() - start_mono

            if proc.poll() is not None:
                break

            if elapsed >= timeout:
                _kill("Reasoning timeout")
                break

            # Read heartbeat_at from the progress file.
            try:
                with open(heartbeat_path) as _f:
                    hb = json.load(_f)
                hb_at = float(hb.get("heartbeat_at", 0))
                if hb_at > 0:
                    last_hb_at = hb_at
            except (OSError, json.JSONDecodeError, ValueError, KeyError):
                pass

            now = time.time()
            if last_hb_at is not None and (now - last_hb_at) > stale_threshold:
                _kill(
                    f"Reasoning worker heartbeat stale "
                    f"({int(now - last_hb_at)}s since last update)"
                )
                break
            elif last_hb_at is None and elapsed > stale_threshold + 60:
                _kill("Reasoning worker never wrote a heartbeat; possible crash or startup failure")
                break

        proc.wait()
        out_thread.join(timeout=5)
        err_thread.join(timeout=5)

        try:
            os.unlink(heartbeat_path)
        except OSError:
            pass

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts) or kill_reason
        if kill_reason:
            return CommandResult(False, stdout, stderr, 124)
        return CommandResult(proc.returncode == 0, stdout, stderr, proc.returncode)

    def git_status(self) -> CommandResult:
        return self._run(["git", "status", "--short"], timeout=10)

    def git_log_head(self) -> CommandResult:
        return self._run(["git", "log", "-1", "--oneline"], timeout=10)

    def create_branch(self, branch: str) -> CommandResult:
        if branch in {"main", "master"} or branch.startswith("-"):
            return CommandResult(False, "", "Refusing unsafe branch name", 2)
        return self._run(["git", "checkout", "-b", branch], timeout=30)

    def run_reasoning(
        self,
        goal: str,
        max_steps: int,
        initial_context: Dict[str, Any],
        timeout: int = 300,
        task_type: str = "code_reasoning",
        preferred_profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        import tempfile
        hb_fd, heartbeat_path = tempfile.mkstemp(prefix="igris_reasoning_hb_", suffix=".json")
        os.close(hb_fd)
        payload = json.dumps({
            "project_root": str(self.project_root),
            "goal": goal,
            "max_steps": max_steps,
            "initial_context": initial_context,
            "task_type": task_type,
            "preferred_profile": preferred_profile,
            "heartbeat_path": heartbeat_path,
        })
        result = self._run_with_heartbeat_monitor(
            [str(self.project_root / ".venv/bin/python"), "-m", "igris.core.supervisor_reasoning_worker"],
            timeout=timeout,
            input_text=payload,
            heartbeat_path=heartbeat_path,
            forward_credentials=True,
        )
        if result.success:
            try:
                return json.loads(result.output)
            except json.JSONDecodeError:
                return {
                    "status": "blocked",
                    "stop_reason": "invalid_reasoning_output",
                    "files_modified": [],
                    "final_summary": _command_detail(result) or "Reasoning worker returned invalid JSON",
                }
        return {
            "status": "blocked",
            "stop_reason": "reasoning_timeout" if result.returncode == 124 else "blocked",
            "files_modified": [],
            "final_summary": _command_detail(result) or "Reasoning worker failed",
        }

    def git_diff_stat(self) -> CommandResult:
        # Use `git diff HEAD --stat` instead of bare `git diff --stat` so that
        # newly-created files that have been staged (git add) are included.
        # Bare `git diff --stat` only shows unstaged changes to *tracked* files;
        # `git diff HEAD --stat` captures all changes vs HEAD (staged or unstaged).
        return self._run(["git", "diff", "HEAD", "--stat"], timeout=10)

    def git_diff(self) -> CommandResult:
        # Mirror the broader HEAD-relative view used by git_diff_stat.
        return self._run(["git", "diff", "HEAD"], timeout=10)

    def run_tests(
        self,
        targets: Optional[List[str]] = None,
        timeout: int = 120,
        hard_cap: int = 3600,
        exclude_slow: bool = False,
    ) -> CommandResult:
        cmd = [str(self.project_root / ".venv/bin/python"), "-m", "pytest", "-q"]
        if exclude_slow:
            cmd.extend(["-m", "not slow"])
        if targets:
            cmd.extend(targets)
        return self._run_adaptive(cmd, idle_timeout=timeout, hard_cap=hard_cap, clean_env=True)

    def run_test_diagnostics(self, timeout: int = 120) -> CommandResult:
        cmd = [
            str(self.project_root / ".venv/bin/python"),
            "-m",
            "pytest",
            "-x",
            "-vv",
        ]
        return self._run_adaptive(cmd, idle_timeout=timeout, hard_cap=timeout * 5, clean_env=True)

    def smoke(self, endpoints: List[str], restart_command: str = "") -> CommandResult:
        if restart_command:
            allowed = {"sudo -n systemctl restart igris"}
            if restart_command not in allowed:
                return CommandResult(False, "", "Restart command is not allowlisted", 126)
            restart = self._run(["sudo", "-n", "systemctl", "restart", "igris"], timeout=30)
            if not restart.success:
                return restart
        outputs: List[str] = []
        for endpoint in endpoints:
            result = self._run(["curl", "-fsS", endpoint], timeout=15)
            outputs.append(result.output or result.error)
            if not result.success:
                return CommandResult(False, "\n".join(outputs), result.error, result.returncode)
            if not _smoke_output_is_valid(endpoint, result.output):
                return CommandResult(
                    False,
                    "\n".join(outputs),
                    f"Invalid bootstrap response for {endpoint}",
                    1,
                )
        return CommandResult(True, "\n".join(outputs), "", 0)

    def commit(self, message: str, files: Optional[List[str]] = None) -> CommandResult:
        if files:
            add = self._run(["git", "add", *files], timeout=30)
            if not add.success:
                return add
        result = self._run(["git", "commit", "-m", message], timeout=60)
        combined = (result.output or "") + (result.error or "")
        if not result.success and (
            "nothing to commit" not in combined
            and "not staged" in combined
        ):
            # Stage all tracked modified files and retry once.
            self._run(["git", "add", "-u"], timeout=30)
            result = self._run(["git", "commit", "-m", message], timeout=60)
            combined = (result.output or "") + (result.error or "")
        if not result.success and "nothing to commit" in combined:
            # Last-resort: stage ALL changes (including untracked in allowed dirs)
            # in case the earlier file-specific add missed something.
            self._run(["git", "add", "-A", "--", "igris", "tests", "docs"], timeout=30)
            result = self._run(["git", "commit", "-m", message], timeout=60)
        return result

    def push_branch(self, branch: str) -> CommandResult:
        if branch in {"main", "master"} or branch.startswith("-"):
            return CommandResult(False, "", "Refusing push to protected/unsafe branch", 2)
        return self._run(["git", "push", "origin", branch], timeout=120)

    def open_pr(self, branch: str, title: str, body: str) -> CommandResult:
        return self._run(["gh", "pr", "create", "--base", "main", "--head", branch, "--title", title, "--body", body], timeout=120)

    def wait_ci(self) -> CommandResult:
        return self._run(["gh", "pr", "checks", "--watch"], timeout=900)

    def merge_pr(self) -> CommandResult:
        return self._run(["gh", "pr", "merge", "--squash", "--delete-branch"], timeout=120)

    def pull_main(self) -> CommandResult:
        checkout = self._run(["git", "checkout", "main"], timeout=30)
        if not checkout.success:
            return checkout
        return self._run(["git", "pull", "--rebase", "origin", "main"], timeout=120)

    def update_issue(self, issue_url: str, comment_body: str) -> CommandResult:
        return self._run(
            ["gh", "issue", "comment", issue_url, "--body", comment_body],
            timeout=60,
        )

    def create_issue(self, title: str, body: str) -> CommandResult:
        listed = self._run(
            ["gh", "issue", "list", "--state", "open", "--limit", "200", "--json", "title,url"],
            timeout=120,
        )
        if listed.success:
            try:
                open_issues = json.loads(listed.output or "[]")
                for issue in open_issues:
                    if str(issue.get("title", "")) == title:
                        return CommandResult(True, str(issue.get("url", "")), "", 0)
            except json.JSONDecodeError:
                pass
        return self._run(["gh", "issue", "create", "--title", title, "--body", body], timeout=120)

    def fetch_issue(self, issue_url: str) -> CommandResult:
        return self._run(
            ["gh", "issue", "view", issue_url, "--json", "title,body,number"],
            timeout=60,
        )

    def restore_dangerous_diff(self) -> CommandResult:
        restore = self._run(["git", "restore", "--worktree", "--staged", "."], timeout=60)
        if not restore.success:
            return restore
        # Remove untracked files left by a failed supervised branch.
        # Scope covers all common project directories (not just igris/tests/docs).
        clean = self._run(["git", "clean", "-fd", "."], timeout=60)
        if not clean.success:
            return clean
        return CommandResult(True, restore.output + clean.output, "", 0)

    def checkout_main(self) -> CommandResult:
        """Switch back to main branch after a blocked/cancelled run."""
        return self._run(["git", "checkout", "main"], timeout=30)

    def delete_stale_rank_branches(self) -> CommandResult:
        """Delete all local rank-* branches left by previous supervised runs."""
        list_result = self._run(
            ["git", "branch", "--list", "rank-*"],
            timeout=15,
        )
        if not list_result.success:
            return list_result
        branches = [b.strip().lstrip("* ") for b in list_result.output.splitlines() if b.strip()]
        if not branches:
            return CommandResult(True, "no rank branches to delete", "", 0)
        deleted, errors = [], []
        for branch in branches:
            r = self._run(["git", "branch", "-D", branch], timeout=15)
            if r.success:
                deleted.append(branch)
            else:
                errors.append(branch)
        msg = f"deleted={deleted} errors={errors}"
        return CommandResult(True, msg, "", 0)

    def restore_paths(self, paths: List[str]) -> CommandResult:
        selected = []
        for raw in paths:
            path = str(raw or "").strip()
            if not path:
                continue
            if path.startswith("-") or path.startswith("/") or ".." in path.split("/"):
                return CommandResult(False, "", f"Refusing unsafe restore path: {path}", 2)
            selected.append(path)
        if not selected:
            return CommandResult(True, "", "", 0)
        restore = self._run(["git", "restore", "--worktree", "--staged", "--", *selected], timeout=60)
        if not restore.success:
            return restore
        clean = self._run(["git", "clean", "-f", "--", *selected], timeout=60)
        if not clean.success:
            return clean
        return CommandResult(True, (restore.output or "") + (clean.output or ""), "", 0)

    def call_api_helper(
        self,
        packet: Dict[str, Any],
        model: str,
        max_tokens: int,
        timeout: int = 45,
        mode: str = "",
    ) -> CommandResult:
        helper_command = str(os.getenv("IGRIS_API_HELPER_COMMAND", "")).strip()
        if not helper_command:
            return CommandResult(False, "", "API helper command is not configured.", 2)
        cmd = shlex.split(helper_command)
        if not cmd:
            return CommandResult(False, "", "API helper command is empty after parsing.", 2)

        # Shadow mode A/B (Epic #445): always call primary; call shadow in parallel for scoring.
        # Primary result is always returned — shadow is never used for decisions.
        alt_model = str(os.getenv("IGRIS_API_HELPER_ALT_MODEL", "")).strip()
        ab_enabled = (
            bool(alt_model)
            and str(os.getenv("IGRIS_ENABLE_HELPER_AB_TEST", "false")).lower() == "true"
        )
        shadow_mode = str(os.getenv("IGRIS_HELPER_AB_SHADOW_MODE", "true")).lower() != "false"

        # Call primary
        import time as _time
        primary_env: Dict[str, str] = {}
        if mode:
            primary_env["IGRIS_API_HELPER_MODE"] = mode
        primary_payload = json.dumps({"model": model, "max_tokens": max_tokens, "packet": packet})
        t0 = _time.monotonic()
        result = self._run(cmd, timeout=timeout, input_text=primary_payload, extra_env=primary_env or None)
        result.helper_primary_latency_ms = int((_time.monotonic() - t0) * 1000)
        result.helper_model = model
        result.helper_ab_active = ab_enabled
        result.helper_ab_alt_model = alt_model if ab_enabled else ""

        # Launch shadow call if enabled
        if ab_enabled and shadow_mode and alt_model:
            self._run_shadow_helper(
                cmd=cmd,
                alt_model=alt_model,
                packet=packet,
                max_tokens=max_tokens,
                timeout=timeout,
                primary_result=result,
            )

        return result

    def _run_shadow_helper(
        self,
        *,
        cmd,
        alt_model: str,
        packet,
        max_tokens: int,
        timeout: int,
        primary_result: "CommandResult",
    ) -> None:
        """Call shadow helper and score both. Non-fatal — never changes primary output."""
        try:
            import time as _time
            import json as _json
            from igris.core.helper_ab_eval import (
                score_helper_response,
                make_ab_record,
                save_ab_result,
                is_safe_to_switch,
                load_ab_results,
            )
            alt_provider = str(os.getenv("IGRIS_API_HELPER_ALT_PROVIDER", "deepseek")).strip()
            shadow_env = {
                "IGRIS_API_HELPER_MODE": "auto",
                "IGRIS_API_HELPER_PROVIDER": alt_provider,
                # Override model so _resolve_model doesn't forward the Codex name to DeepSeek
                "IGRIS_API_HELPER_MODEL": alt_model,
                "IGRIS_HELPER_AB_ARM": "alt",
            }
            shadow_payload = _json.dumps({"model": alt_model, "max_tokens": max_tokens, "packet": packet})
            t0 = _time.monotonic()
            shadow_result = self._run(cmd, timeout=timeout, input_text=shadow_payload, extra_env=shadow_env)
            alt_latency_ms = int((_time.monotonic() - t0) * 1000)

            try:
                primary_parsed = _json.loads(primary_result.output) if primary_result.output else {}
            except _json.JSONDecodeError:
                primary_parsed = {}
            try:
                alt_parsed = _json.loads(shadow_result.output) if shadow_result.output else {}
            except _json.JSONDecodeError:
                alt_parsed = {}

            empty_case: Dict[str, Any] = {}
            primary_score_r = score_helper_response(primary_parsed, empty_case)
            alt_score_r = score_helper_response(alt_parsed, empty_case)
            primary_cost = float(primary_parsed.get("estimated_cost_usd", 0.0))
            alt_cost = float(alt_parsed.get("estimated_cost_usd", 0.0))

            record = make_ab_record(
                case_id=str(packet.get("failure_class", "unknown")),
                primary_model=primary_result.helper_model,
                alt_model=alt_model,
                primary_score=primary_score_r["total"],
                alt_score=alt_score_r["total"],
                primary_breakdown=primary_score_r["breakdown"],
                alt_breakdown=alt_score_r["breakdown"],
                primary_cost_usd=primary_cost,
                alt_cost_usd=alt_cost,
                primary_latency_ms=primary_result.helper_primary_latency_ms,
                alt_latency_ms=alt_latency_ms,
                source="organic_run",
                # Model identity from provider responses
                primary_requested_model=str(primary_parsed.get("api_helper_model_requested", "") or ""),
                primary_resolved_model=str(primary_parsed.get("api_helper_model_resolved", "") or ""),
                primary_provider_response_model=str(primary_parsed.get("model", "") or ""),
                primary_served_model=str(primary_parsed.get("model", "") or ""),
                primary_provider=str(primary_parsed.get("api_helper_provider", "") or ""),
                alt_requested_model=str(alt_parsed.get("api_helper_model_requested", alt_model) or ""),
                alt_resolved_model=str(alt_parsed.get("api_helper_model_resolved", "") or ""),
                alt_provider_response_model=str(alt_parsed.get("model", "") or ""),
                alt_served_model=str(alt_parsed.get("model", "") or ""),
                alt_provider=str(alt_parsed.get("api_helper_provider", alt_provider) or ""),
                api_helper_mode=str(primary_parsed.get("api_helper_mode", "") or ""),
            )
            ab_path = str(os.getenv("IGRIS_HELPER_AB_RESULTS_PATH", ".igris/helper_ab_results.json"))
            save_ab_result(record, ab_path)
            all_records = load_ab_results(ab_path)
            sw_report = is_safe_to_switch(all_records)
            safe = sw_report["safe_to_switch"]

            primary_result.helper_ab_shadow_mode = True
            primary_result.helper_primary_score = primary_score_r["total"]
            primary_result.helper_alt_score = alt_score_r["total"]
            primary_result.helper_primary_cost_usd = primary_cost
            primary_result.helper_alt_cost_usd = alt_cost
            primary_result.helper_alt_latency_ms = alt_latency_ms
            primary_result.helper_switch_recommendation = safe
        except Exception:
            pass  # shadow mode is non-fatal

    def api_helper_is_configured(self) -> bool:
        """Return True when IGRIS_API_HELPER_COMMAND env var is set and non-empty."""
        return bool(str(os.getenv("IGRIS_API_HELPER_COMMAND", "")).strip())


def classify_failure(
    reasoning_result: Optional[Dict[str, Any]] = None,
    diff: str = "",
    targeted_tests: Optional[CommandResult] = None,
    full_tests: Optional[CommandResult] = None,
    smoke: Optional[CommandResult] = None,
) -> str:
    reasoning_text = ""
    if reasoning_result:
        reasoning_text = "\n".join(
            str(reasoning_result.get(key, ""))
            for key in ("final_summary", "error", "stop_reason")
        )
    text = "\n".join([
        reasoning_text,
        diff or "",
        targeted_tests.error if targeted_tests else "",
        targeted_tests.output if targeted_tests else "",
        full_tests.error if full_tests else "",
        full_tests.output if full_tests else "",
    ])
    if _has_destructive_diff(diff):
        return "destructive_diff"
    if _is_llm_provider_unavailable(reasoning_text):
        return "infrastructure_bug"
    if targeted_tests and targeted_tests.returncode == 124:
        return "test_runner_timeout"
    if full_tests and full_tests.returncode == 124:
        return "test_runner_timeout"
    if targeted_tests and not targeted_tests.success:
        if _is_missing_test_target_error(targeted_tests):
            return "missing_tests"
        return "pytest_failure"
    if full_tests and not full_tests.success:
        return "pytest_failure"
    if reasoning_result:
        stop = str(reasoning_result.get("stop_reason", ""))
        status = str(reasoning_result.get("status", ""))
        if (
            "Python AST validation failed" in reasoning_text
            or "SyntaxError" in reasoning_text
            or "invalid syntax" in reasoning_text
        ):
            return "syntax_error"
        if stop in {"reasoning_timeout", "budget_exceeded", "no_diff_repair"}:
            return "reasoning_loop_blocked"
        if stop == "max_steps":
            return "max_steps"
        if stop == "ask_user":
            return "ask_user"
        if status == "blocked" or stop == "blocked":
            return "reasoning_loop_blocked"
        files = reasoning_result.get("files_modified") or []
        if "test" in str(reasoning_result.get("goal", "")).lower() and not any("test" in f for f in files):
            return "missing_tests"
    if "SyntaxError" in text or "invalid syntax" in text:
        return "syntax_error"
    if smoke and not smoke.success:
        smoke_text = "\n".join([smoke.output or "", smoke.error or ""]).lower()
        if "bootstrap" in smoke_text or "invalid bootstrap" in smoke_text:
            return "invalid_bootstrap"
        return "infrastructure_bug"
    return "infrastructure_bug"


def _extract_failed_pytest_nodes(text: str) -> List[str]:
    nodes = re.findall(r"FAILED\s+([^\s]+::[^\s]+)", text or "")
    if not nodes:
        nodes = re.findall(r"(tests/[A-Za-z0-9_./-]+\.py)", text or "")
    seen: set[str] = set()
    out: List[str] = []
    for node in nodes:
        key = str(node).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _parse_pytest_collection_error(pytest_output: str) -> Optional[Dict[str, Any]]:
    """Parse pytest output to extract actionable collection error details.

    Returns a dict with error_type and context keys, or None if no known
    collection error is detected.  Supported patterns:

    * ``ImportError: cannot import name 'X' from 'Y'``
    * ``ImportError: cannot import name 'X'``  (no module qualifier)
    * ``ModuleNotFoundError: No module named 'X'``
    * ``AttributeError: module 'X' has no attribute 'Y'``
    * Generic ERROR during collection with no test selected (``no tests ran``)
    """
    if not pytest_output:
        return None

    text = pytest_output

    # Pattern 1: ImportError: cannot import name 'Symbol' from 'module.path'
    m = re.search(
        r"ImportError: cannot import name ['\"]([^'\"]+)['\"] from ['\"]([^'\"]+)['\"]",
        text,
    )
    if m:
        return {
            "error_type": "missing_symbol",
            "missing_symbol": m.group(1),
            "source_module": m.group(2),
        }

    # Pattern 1b: ImportError: cannot import name 'Symbol' (no 'from' clause)
    m = re.search(r"ImportError: cannot import name ['\"]([^'\"]+)['\"]", text)
    if m:
        # Try to infer the module from the collection path
        mod_m = re.search(r"from ([a-zA-Z0-9_.]+) import", text)
        return {
            "error_type": "missing_symbol",
            "missing_symbol": m.group(1),
            "source_module": mod_m.group(1) if mod_m else "",
        }

    # Pattern 2: ModuleNotFoundError: No module named 'X'
    m = re.search(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", text)
    if m:
        return {
            "error_type": "missing_module",
            "missing_module": m.group(1),
        }

    # Pattern 3: AttributeError: module 'X' has no attribute 'Y'
    m = re.search(
        r"AttributeError: module ['\"]([^'\"]+)['\"] has no attribute ['\"]([^'\"]+)['\"]",
        text,
    )
    if m:
        return {
            "error_type": "missing_symbol",
            "missing_symbol": m.group(2),
            "source_module": m.group(1),
        }

    # Pattern 4: generic collection error — EEE / no tests ran / ERROR collecting
    if re.search(r"(ERROR collecting|no tests ran|= no tests ran =|EEE)", text):
        # Extract the test file that failed collection
        file_m = re.search(r"ERROR collecting (tests/[^\s]+\.py)", text)
        return {
            "error_type": "collection_error",
            "failing_test_file": file_m.group(1) if file_m else "",
        }

    return None


def _baseline_failure_is_transient(baseline: CommandResult, diagnostics: Optional[CommandResult]) -> bool:
    if baseline.returncode == 124:
        return True
    if diagnostics and diagnostics.returncode == 124:
        return True
    text = "\n".join([
        baseline.output or "",
        baseline.error or "",
        diagnostics.output if diagnostics else "",
        diagnostics.error if diagnostics else "",
    ]).lower()
    transient_markers = (
        "keyboardinterrupt",
        "timed out",
        "timeout",
        "connection reset",
        "connection refused",
        "temporarily unavailable",
        "resource temporarily unavailable",
        "no space left on device",
    )
    return any(marker in text for marker in transient_markers)


def _allow_unrelated_vastai_baseline_failures(
    goal: str,
    baseline: CommandResult,
    diagnostics: Optional[CommandResult],
) -> bool:
    diag_text = "\n".join([diagnostics.output or "", diagnostics.error or ""]) if diagnostics else ""
    failed_nodes = _extract_failed_pytest_nodes(
        "\n".join([baseline.output or "", baseline.error or "", diag_text])
    )
    if not failed_nodes:
        return False
    if any("/test_vastai_" not in node for node in failed_nodes):
        return False
    goal_l = (goal or "").lower()
    goal_is_vastai = any(token in goal_l for token in ("vast", "gpu", "v100", "3090", "4090", "ollama"))
    return not goal_is_vastai


def _baseline_cache_path(project_root: str) -> Path:
    return Path(project_root) / ".igris" / "baseline_cache.json"


# ---------------------------------------------------------------------------
# Issue #626 — Delta baseline: detect pre-existing failures vs new regressions
# ---------------------------------------------------------------------------

def _known_failures_path(project_root: str) -> Path:
    return Path(project_root) / ".igris" / "known_baseline_failures.json"


def _load_known_baseline_failures(project_root: str, main_sha: str) -> Optional[List[str]]:
    """Return the list of test nodes known to fail on *main_sha*, or None if not cached."""
    path = _known_failures_path(project_root)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if str(data.get("main_sha", "")).strip() == str(main_sha).strip():
            return list(data.get("failed_nodes", []))
    except Exception:
        pass
    return None


def _save_known_baseline_failures(
    project_root: str, main_sha: str, failed_nodes: List[str]
) -> None:
    """Persist the set of pre-existing failures for *main_sha*."""
    path = _known_failures_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {
        "main_sha": str(main_sha),
        "failed_nodes": list(failed_nodes),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def _get_main_sha(project_root: str) -> str:
    """Return the current SHA of origin/main (or main if origin/main is absent)."""
    import subprocess as _sp
    for ref in ("origin/main", "main"):
        r = _sp.run(
            ["git", "rev-parse", ref],
            capture_output=True, text=True, cwd=str(project_root),
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    return ""


def _diff_vs_main_is_empty(project_root: str, main_sha: str) -> bool:
    """True when HEAD has no diff relative to main (branch == main, no new commits)."""
    import subprocess as _sp
    if not main_sha:
        return False
    r = _sp.run(
        ["git", "diff", "--quiet", main_sha, "HEAD"],
        capture_output=True, cwd=str(project_root),
    )
    return r.returncode == 0


def _delta_baseline_failures(
    branch_failures: List[str], known_failures: List[str]
) -> List[str]:
    """Return failures present in *branch_failures* but NOT in *known_failures*.

    These are genuine regressions introduced by the current branch.
    """
    known_set = set(known_failures)
    return [f for f in branch_failures if f not in known_set]


def _load_valid_baseline_cache(
    project_root: str, head_sha: str, force_revalidate: bool = False
) -> Optional[Dict[str, Any]]:
    """Load a valid baseline cache entry, or return None on miss.

    Issue #730: also sets ``_miss_reason`` on the returned payload (or on a
    dummy dict when returning None) so callers can emit a ``baseline_revalidation``
    event with the reason for the miss.
    """
    ttl = max(60, int(os.getenv("IGRIS_BASELINE_CACHE_SECONDS", "1800")))
    path = _baseline_cache_path(project_root)
    if force_revalidate:
        return None  # caller will emit baseline_revalidation event with reason="force_revalidate"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if str(payload.get("head_sha", "")).strip() != str(head_sha).strip():
        payload["_miss_reason"] = "sha_changed"
        return None
    checked_at = float(payload.get("checked_at", 0.0) or 0.0)
    if checked_at <= 0:
        return None
    if (time.time() - checked_at) > ttl:
        # Issue #730 — cache stale due to age; surface this as a revalidation event
        _stale_age_s = round(time.time() - checked_at, 0)
        payload["_miss_reason"] = "stale"
        payload["_stale_age_s"] = _stale_age_s
        return None
    if not bool(payload.get("baseline_ok", False)):
        return None
    return payload


def _save_baseline_cache(project_root: str, head_sha: str, policy: str = "strict") -> None:
    path = _baseline_cache_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "head_sha": str(head_sha),
        "checked_at": float(time.time()),
        "baseline_ok": True,
        "policy": str(policy),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _baseline_sanity_targets(project_root: str) -> List[str]:
    raw = str(os.getenv("IGRIS_BASELINE_TEST_TARGETS", "")).strip()
    if raw:
        return [t for t in raw.split() if t.strip()]
    defaults = [
        "tests/test_health_readiness.py",
        "tests/test_rank_status.py",
    ]
    root = Path(project_root)
    return [t for t in defaults if (root / t).exists()]


def _has_immediately_dangerous_diff(diff: str) -> bool:
    """Fast pre-test check for diffs that would definitely break the app.

    Only catches two categories that cannot possibly be recovered by the test suite:
      1. Dangerous file tokens (.env, .venv, __pycache__, etc.)
      2. Structural deletions of def create_app or class bodies

    Import-deletion detection is left to _has_destructive_diff (used post-test via
    classify_failure), allowing the test suite to be the primary safety net.
    """
    # Use path-level matching (same logic as _has_destructive_diff) to avoid false
    # positives when ".env" appears in diff content rather than as a changed path.
    _dangerous_exact = {".env"}
    _dangerous_prefix = (".venv/", "__pycache__/", ".pytest_cache/", ".igris/")
    paths = _diff_changed_paths(diff)
    for path in paths:
        if path in _dangerous_exact or any(path.startswith(p) for p in _dangerous_prefix):
            return True
    if paths and all(path.startswith("tests/") for path in paths):
        return False
    python_removed_lines: List[str] = []
    python_added_lines: List[str] = []
    has_diff_headers = "diff --git " in diff
    if not has_diff_headers:
        for line in diff.splitlines():
            if line.startswith("-") and not line.startswith("---"):
                python_removed_lines.append(line)
            elif line.startswith("+") and not line.startswith("+++"):
                python_added_lines.append(line)
    else:
        current_path = ""
        for line in diff.splitlines():
            if line.startswith("diff --git "):
                parts = line.split()
                current_path = parts[3][2:] if len(parts) >= 4 and parts[3].startswith("b/") else ""
                continue
            if not current_path.endswith(".py"):
                continue
            if line.startswith("-") and not line.startswith("---"):
                python_removed_lines.append(line)
            elif line.startswith("+") and not line.startswith("+++"):
                python_added_lines.append(line)
    # Cross-reference: a structural token in a removed line is only dangerous when
    # the same token does NOT appear in any added line (modification vs. deletion).
    structural = ("def create_app", "class ")
    added_text = "\n".join(python_added_lines)
    for line in python_removed_lines:
        for token in structural:
            if token in line and token not in added_text:
                return True
    return False


def _has_destructive_diff(diff: str) -> bool:
    paths = _diff_changed_paths(diff)
    # .env exact match catches the secrets file; prefix match catches venv/cache dirs.
    # Substring matching on the raw diff is intentionally avoided to prevent false
    # positives on safe template files like .env.example.
    _dangerous_exact = {".env"}
    _dangerous_prefix = (".venv/", "__pycache__/", ".pytest_cache/", ".igris/")
    for path in paths:
        if path in _dangerous_exact or any(path.startswith(p) for p in _dangerous_prefix):
            return True
    if paths and all(path.startswith("tests/") for path in paths):
        return False

    python_removed_lines: List[str] = []
    has_diff_headers = "diff --git " in diff
    if not has_diff_headers:
        python_removed_lines = [
            line for line in diff.splitlines()
            if line.startswith("-") and not line.startswith("---")
        ]
    else:
        current_path = ""
        for line in diff.splitlines():
            if line.startswith("diff --git "):
                parts = line.split()
                if len(parts) >= 4:
                    current_path = parts[3][2:] if parts[3].startswith("b/") else parts[3]
                else:
                    current_path = ""
                continue
            if not (current_path.endswith(".py") and line.startswith("-") and not line.startswith("---")):
                continue
            python_removed_lines.append(line)

    # Structural deletions (app factory, class bodies) are always destructive.
    structural = ("def create_app", "class ")
    if any(any(token in line for token in structural) for line in python_removed_lines):
        return True

    # Import deletions: only destructive when an import is truly removed (not
    # reorganised).  Reorganisation removes and re-adds the same names, so the
    # module/symbol appears in an added import line.  We compare against added
    # import lines only (not all added text) to avoid false matches.
    def _extract_import_names(raw: str) -> List[str]:
        tokens = raw.lstrip("-+ \t").split()
        if not tokens:
            return []
        if tokens[0] == "from" and len(tokens) >= 4:
            return [t.rstrip(",") for t in tokens[3:] if t not in ("as", "(", ")")]
        if tokens[0] == "import":
            return [t.rstrip(",").split(".")[0] for t in tokens[1:] if t != "as"]
        return []

    added_import_names: set = set()
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++") and "import " in line:
            for name in _extract_import_names(line):
                added_import_names.add(name)

    import_removed_lines = [l for l in python_removed_lines if "import " in l]
    for removed in import_removed_lines:
        names = _extract_import_names(removed)
        # If NONE of the removed names are re-added, it's a true deletion.
        if names and not any(name in added_import_names for name in names):
            return True

    return False


def _has_invalid_fastapi_bootstrap_diff(diff: str) -> bool:
    paths = _diff_changed_paths(diff)
    if paths and "igris/web/server.py" not in paths:
        return False

    lowered = diff.lower()
    if "return jsonresponse" in lowered and "def create_app" in lowered:
        return True

    ui_card_route = "@app.get('/api/rank/ui-card')" in lowered or '@app.get("/api/rank/ui-card")' in lowered
    if "def run_app" in lowered and ui_card_route:
        return True

    if lowered.count("@app.get('/api/rank/ui-card')") > 1:
        return True
    if lowered.count('@app.get("/api/rank/ui-card")') > 1:
        return True

    bootstrap_routes = ("/api/health", "/api/readiness", "/api/ping")
    if any(route in lowered for route in bootstrap_routes) and "return jsonresponse" in lowered:
        return True

    return False


def _is_missing_test_target_error(result: Optional["CommandResult"]) -> bool:
    if not result:
        return False
    text = "\n".join([result.output or "", result.error or ""]).lower()
    return "file or directory not found" in text and "tests/test_" in text


def _is_llm_provider_unavailable(text: str) -> bool:
    lowered = (text or "").lower()
    return (
        "no suitable llm provider available" in lowered
        or "llm unavailable" in lowered
    )


def _required_endpoint_from_goal(goal: str) -> str:
    match = re.search(r"/api/[a-z0-9_/-]+", goal.lower())
    if not match:
        return ""
    return match.group(0)


def _is_valid_missing_tests_repair_diff(diff: str, goal: str) -> bool:
    paths = _diff_changed_paths(diff)
    if not paths:
        return False
    if not all(path.startswith("tests/") for path in paths):
        return False

    lowered = diff.lower()
    if "test_client(" in lowered:
        return False
    if "testclient(create_app())" not in lowered and "create_app()" not in lowered:
        return False

    required_endpoint = _required_endpoint_from_goal(goal)
    endpoints_found = set(re.findall(r"/api/[a-z0-9_/-]+", lowered))
    if required_endpoint:
        if required_endpoint not in endpoints_found:
            return False
        if any(endpoint != required_endpoint for endpoint in endpoints_found):
            return False
    if "/dashboard" in lowered:
        return False
    return True


def _has_flask_test_client_in_diff(diff: str) -> bool:
    """Return True when the diff *adds* Flask-style ``app.test_client()`` calls.

    FastAPI app objects have no ``test_client()`` method; using it causes
    ``AttributeError`` at pytest collection time (EEE errors).  This helper
    is used during ``pytest_failure`` repair validation to reject such diffs
    early so the repair cycle retries with explicit FastAPI TestClient guidance.
    Only lines that are *added* (starting with '+' but not '+++') are checked.
    """
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            if "test_client(" in line.lower():
                return True
    return False


def _is_valid_ui_test_diff(diff: str) -> bool:
    """Return True when a UI test diff stays minimal and exact.

    The UI rank task should use a read-only test against ``/api/rank/ui-card``.
    We reject diffs that introduce request bodies, alternate verbs, or app
    import patterns that have historically produced unstable bootstrap errors.
    """

    if "tests/test_rank_ui_card.py" not in diff:
        return True

    lowered = diff.lower()
    required_get = (
        'client.get("/api/rank/ui-card")' in lowered
        or "client.get('/api/rank/ui-card')" in lowered
    )
    required_factory = "testclient(create_app())" in lowered or "create_app()" in lowered
    forbidden_tokens = (
        "client.post(",
        "client.put(",
        "client.patch(",
        "client.delete(",
        "client.request(",
        "body(",
        "json=",
        "data=",
        "from igris.web.server import app",
        "response.json()['data']",
        'response.json()["data"]',
        "response.json().get('data')",
        'response.json().get("data")',
        "assert 'data' in response.json()",
        'assert "data" in response.json()',
    )
    if not required_get or not required_factory:
        return False
    return not any(token in lowered for token in forbidden_tokens)


def _diff_changed_paths(diff: str) -> List[str]:
    paths: List[str] = []
    for line in diff.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        path = parts[3]
        if path.startswith("b/"):
            path = path[2:]
        paths.append(path)
    return paths


def _diff_sections_by_path(diff: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    current_path = ""
    current_lines: List[str] = []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            if current_path:
                sections[current_path] = "\n".join(current_lines)
            parts = line.split()
            if len(parts) >= 4:
                current_path = parts[3][2:] if parts[3].startswith("b/") else parts[3]
            else:
                current_path = ""
            current_lines = [line]
            continue
        if current_path:
            current_lines.append(line)
    if current_path:
        sections[current_path] = "\n".join(current_lines)
    return sections


def _changed_paths_between_diffs(before_diff: str, after_diff: str) -> Set[str]:
    before_sections = _diff_sections_by_path(before_diff)
    after_sections = _diff_sections_by_path(after_diff)
    changed: Set[str] = set()
    for path in set(before_sections.keys()).union(after_sections.keys()):
        if before_sections.get(path, "") != after_sections.get(path, ""):
            changed.add(path)
    return changed


def _normalize_candidate_path(path: str) -> str:
    normalized = str(path or "").strip().strip("'\"")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("a/") or normalized.startswith("b/"):
        normalized = normalized[2:]
    return normalized


def _extract_attempted_write_paths(reasoning_result: Dict[str, Any]) -> List[str]:
    paths: Set[str] = set()
    steps = reasoning_result.get("steps") or []
    for raw_step in steps:
        if not isinstance(raw_step, dict):
            continue
        action_type = str(raw_step.get("action_type", "")).strip()
        if action_type not in WRITE_ACTION_TYPES:
            continue
        params = raw_step.get("parameters") or {}
        if isinstance(params, dict):
            for key in ("path", "file_path", "file", "target_path"):
                candidate = params.get(key)
                if not isinstance(candidate, str):
                    continue
                normalized = _normalize_candidate_path(candidate)
                if normalized:
                    paths.add(normalized)
        for text_key in ("error", "result_summary"):
            text = str(raw_step.get(text_key, "") or "")
            for match in re.findall(r"['\"]([A-Za-z0-9_./-]+\.[A-Za-z0-9_]+)['\"]", text):
                normalized = _normalize_candidate_path(match)
                if normalized and not normalized.startswith(("http://", "https://")):
                    paths.add(normalized)
    for text in [
        str(reasoning_result.get("final_summary", "") or ""),
        str(reasoning_result.get("error", "") or ""),
    ]:
        for match in re.findall(r"['\"]([A-Za-z0-9_./-]+\.[A-Za-z0-9_]+)['\"]", text):
            normalized = _normalize_candidate_path(match)
            if normalized and not normalized.startswith(("http://", "https://")):
                paths.add(normalized)
    for error in reasoning_result.get("errors") or []:
        text = str(error or "")
        for match in re.findall(r"['\"]([A-Za-z0-9_./-]+\.[A-Za-z0-9_]+)['\"]", text):
            normalized = _normalize_candidate_path(match)
            if normalized and not normalized.startswith(("http://", "https://")):
                paths.add(normalized)
    return sorted(paths)


def _is_product_only_ui_task_diff(diff: str) -> bool:
    """Return True when a repair diff only changes UI rank product files."""

    paths = _diff_changed_paths(diff)
    if not paths:
        return False

    product_prefixes = (
        "igris/web/templates/",
        "igris/web/static/js/",
        "igris/web/static/css/",
    )
    product_paths = {
        "igris/web/server.py",
        "tests/test_rank_ui_card.py",
        "tests/test_dashboard_tabs.py",
        "tests/test_guided_actions.py",
    }
    return all(path in product_paths or path.startswith(product_prefixes) for path in paths)


def _has_ui_surface_change(diff: str) -> bool:
    paths = _diff_changed_paths(diff)
    if not paths:
        return False
    ui_prefixes = (
        "igris/web/templates/",
        "igris/web/static/js/",
        "igris/web/static/css/",
    )
    return any(path.startswith(ui_prefixes) for path in paths)


def _touches_rank_ui_contract_files(diff: str) -> bool:
    paths = _diff_changed_paths(diff)
    if not paths:
        return False
    protected = {
        "igris/web/server.py",
        "tests/test_rank_ui_card.py",
    }
    return any(path in protected for path in paths)


def _smoke_output_is_valid(endpoint: str, output: str) -> bool:
    text = output.strip()
    if not text:
        return False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False

    if endpoint.endswith("/api/health"):
        return payload.get("status") == "ok" and "version" in payload
    if endpoint.endswith("/api/readiness"):
        expected = ("project_root_exists", "project_root_is_dir", "templates", "static", "agents_registered")
        return all(payload.get(key) is True for key in expected)
    if endpoint.endswith("/api/ping"):
        return payload.get("pong") is True
    return True


class SelfRepairSupervisor:
    def __init__(self, project_root: str, backend: Optional[SupervisorBackend] = None):
        self.project_root = project_root
        self.backend = backend or LocalSupervisorBackend(project_root)
        self._audit_path = Path(project_root) / ".igris" / "supervisor_audit.json"
        self._audit_index = self._load_audit_index()
        self._runs_path = Path(project_root) / ".igris" / "supervisor_runs.json"
        self._runs_lock = threading.RLock()
        self._runs_index = self._load_runs_index()
        self._failure_memory = FailureMemory(
            store_path=Path(project_root) / ".igris" / "failure_patterns.json"
        )
        # Issue #722 — mark zombie 'running' runs as 'interrupted' on startup
        self._startup_cleanup_zombie_runs()
        # Issue #733 — delete stale rank_pending.patch left by a crashed run
        self._startup_cleanup_stale_patch()

    def _startup_cleanup_zombie_runs(self) -> None:
        """Mark any run with status='running' as 'interrupted' on supervisor init.

        When the server restarts, runs that were active at shutdown are stuck
        forever as 'running'.  We detect them here by checking that their PID
        is not the current process (or has no PID at all) and transition them
        to 'interrupted' so the UI is not misleading.  (Issue #722)

        Parallel-run fix: runs already registered in the module-level RUN_STORE
        are actively managed by this process — skip them.  This allows multiple
        concurrent supervised runs (one SelfRepairSupervisor per run) without
        each new instantiation cancelling the others.
        """
        _logger = logging.getLogger("igris.supervisor.startup")
        current_pid = os.getpid()
        interrupted_ids = []
        with self._runs_lock:
            for run_id, record in self._runs_index.items():
                status = str(record.get("status", "")).strip().lower()
                if status not in ("running", "cancelling"):
                    continue
                # Skip runs that are already in the in-memory store — they are
                # live runs managed by this process (parallel multi-run support).
                if run_id in RUN_STORE:
                    continue
                run_pid = record.get("pid")
                if run_pid is not None and int(run_pid) == current_pid:
                    continue  # Started by this process — still live
                record["status"] = "interrupted"
                record["interrupted_at"] = time.time()
                record.setdefault("events", []).append({
                    "phase": "startup_cleanup",
                    "status": "interrupted",
                    "detail": f"Run was stuck as '{status}' on server restart; marked interrupted.",
                    "ts": time.time(),
                })
                interrupted_ids.append(run_id)
            if interrupted_ids:
                self._persist_runs_index()
        if interrupted_ids:
            _logger.warning(
                "Startup cleanup: %d zombie run(s) marked interrupted: %s",
                len(interrupted_ids), interrupted_ids,
            )

    def _startup_cleanup_stale_patch(self) -> None:
        """Delete rank_pending.patch left over from a crashed run.  (Issue #733)"""
        _logger = logging.getLogger("igris.supervisor.startup")
        patch_path = Path(self.project_root) / ".igris" / "rank_pending.patch"
        if patch_path.exists():
            try:
                patch_path.unlink()
                _logger.warning(
                    "Startup cleanup: removed stale rank_pending.patch "
                    "(leftover from previous crashed run)."
                )
            except OSError as exc:
                _logger.warning("Startup cleanup: could not remove stale patch: %s", exc)

    def _load_audit_index(self) -> Dict[str, Dict[str, Any]]:
        try:
            if not self._audit_path.exists():
                return {}
            payload = json.loads(self._audit_path.read_text(encoding="utf-8"))
            records = payload.get("records", {}) if isinstance(payload, dict) else {}
            if not isinstance(records, dict):
                return {}
            return {str(k): dict(v) for k, v in records.items() if isinstance(k, str) and isinstance(v, dict)}
        except (OSError, json.JSONDecodeError):
            return {}

    def _persist_audit_index(self) -> None:
        try:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            self._audit_path.write_text(json.dumps({"records": self._audit_index}, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            return

    def _load_runs_index(self) -> Dict[str, Dict[str, Any]]:
        try:
            if not self._runs_path.exists():
                return {}
            payload = json.loads(self._runs_path.read_text(encoding="utf-8"))
            runs = payload.get("runs", {}) if isinstance(payload, dict) else {}
            if not isinstance(runs, dict):
                return {}
            return {
                str(k): dict(v)
                for k, v in runs.items()
                if isinstance(k, str) and isinstance(v, dict)
            }
        except (OSError, json.JSONDecodeError):
            return {}

    def _persist_runs_index(self) -> None:
        try:
            self._runs_path.parent.mkdir(parents=True, exist_ok=True)
            # Issue #729 — rotate supervisor_runs.json if it exceeds size cap
            try:
                from igris.core.file_rotation import rotate_if_needed
                rotate_if_needed(self._runs_path)
            except Exception:
                pass
            payload = {"runs": self._runs_index}
            self._runs_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            return

    @staticmethod
    def _timestamp_to_iso(ts: Optional[float]) -> str:
        if ts is None:
            return ""
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
        except (TypeError, ValueError, OSError):
            return ""

    def _persisted_run_record(self, run: SupervisorRun) -> Dict[str, Any]:
        snapshot = summarize_supervised_run(run)
        payload = run.to_dict()
        events = payload.get("events") or []
        first_event_ts = events[0].get("timestamp") if events else None
        last_event = events[-1] if events else {}
        current_stage = str(snapshot.get("current_stage", "")).strip()
        failed_stage = str(snapshot.get("failed_stage", "")).strip()
        latest_event_summary = {
            "phase": str(last_event.get("phase", "")),
            "status": str(last_event.get("status", "")),
            "detail": str(last_event.get("detail", ""))[:500],
            "timestamp": last_event.get("timestamp"),
        }
        report_data = self._sanitize_escalation_value(payload.get("report") or {})
        cancelled_reason = str(report_data.get("cancelled_reason", "") or payload.get("cancel_reason", "") or "")
        record = {
            "run_id": payload.get("run_id", ""),
            "rank_id": payload.get("rank_id", ""),
            "issue_number": _parse_issue_number(
                (payload.get("report") or {}).get("issue_number", 0),
                str(payload.get("goal", "")),
            ) or None,
            "status": payload.get("status", ""),
            "outcome": payload.get("outcome", ""),
            "branch": payload.get("branch", ""),
            "current_stage": current_stage,
            "failed_stage": failed_stage,
            "failure_class": payload.get("failure_class", ""),
            "repair_cycles_used": int(payload.get("repair_cycles_used", 0) or 0),
            "max_repair_cycles": int(payload.get("max_repair_cycles", 0) or 0),
            "api_escalations_used": int(payload.get("api_escalations_used", 0) or 0),
            "api_escalations_failed_unconfigured": int(payload.get("api_escalations_failed_unconfigured", 0) or 0),
            "max_api_escalations_per_run": int(payload.get("max_api_escalations_per_run", 0) or 0),
            "api_budget_used_usd": round(_safe_float(payload.get("api_budget_used_usd", 0.0)), 6),
            "max_api_budget_usd": round(_safe_float(payload.get("max_api_budget_usd", 0.0)), 6),
            "escalation_issue_url": str(snapshot.get("escalation_issue_url", "")),
            "latest_event": latest_event_summary,
            "created_at": self._timestamp_to_iso(first_event_ts) or self._timestamp_now_iso(),
            "updated_at": self._timestamp_to_iso(snapshot.get("updated_at")) or self._timestamp_now_iso(),
            "final_report": report_data,
            "blocked_reason": str(report_data.get("blocked_reason", "")),
            "cancelled_reason": cancelled_reason,
            "next_action": str(snapshot.get("next_action", "")),
            "resolved_failure": bool((payload.get("report") or {}).get("resolved_failure", False)),
            "degraded_completion": bool((payload.get("report") or {}).get("degraded_completion", False)),
            "degraded_completion_reason": str((payload.get("report") or {}).get("degraded_completion_reason", "")),
            "state_conflict": bool(snapshot.get("state_conflict", False)),
            "warning": str(snapshot.get("warning", "")),
        }
        return _enforce_completion_failure_invariant(record)

    def _persist_run_snapshot(self, run: SupervisorRun) -> None:
        record = self._persisted_run_record(run)
        with self._runs_lock:
            self._runs_index[str(run.run_id)] = record
            self._persist_runs_index()

    def _configure_run_tracking(self, run: SupervisorRun, config: RankSupervisorConfig) -> None:
        run.audit_resolver = self._resolve_event_audit
        run.update_hook = self._persist_run_snapshot
        run.max_repair_cycles = config.max_repair_cycles
        run.max_api_escalations_per_run = config.max_api_escalations_per_run
        run.max_api_budget_usd = round(config.max_api_budget_usd, 6)
        run.goal = config.goal

    def _cancel_if_requested(
        self,
        run: SupervisorRun,
        *,
        mission_plan: Optional[MissionPlan] = None,
        stage_statuses: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[SupervisorRun]:
        if not bool(run.cancel_requested):
            return None
        reason = str(run.cancel_reason or "Cancelled by user").strip() or "Cancelled by user"
        return self._cancelled(
            run,
            reason,
            mission_plan=mission_plan,
            stage_statuses=stage_statuses,
            cleanup_workspace=True,
        )

    @staticmethod
    def _sanitize_escalation_value(value: Any) -> Any:
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for key, raw in value.items():
                lowered = str(key).lower()
                if any(token in lowered for token in ("secret", "token", "password", "api_key", "authorization")):
                    out[str(key)] = "[redacted]"
                else:
                    out[str(key)] = SelfRepairSupervisor._sanitize_escalation_value(raw)
            return out
        if isinstance(value, list):
            return [SelfRepairSupervisor._sanitize_escalation_value(item) for item in value][:50]
        if isinstance(value, (bool, int, float)) or value is None:
            return value
        text = _safe_redact(value)
        text = re.sub(r"\bsk-[A-Za-z0-9_-]{6,}\b", "***REDACTED***", text)
        return text[:2000] + ("...(truncated)" if len(text) > 2000 else "")

    def _event_scope_hash(self, event: SupervisorEvent) -> str:
        canonical = {
            "phase": event.phase,
            "status": event.status,
            "detail": _safe_redact(event.detail),
            "data": self._sanitize_escalation_value(event.data),
        }
        return sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def _timestamp_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _timestamp_is_due(next_review_after: str) -> bool:
        if not str(next_review_after or "").strip():
            return True
        try:
            due = datetime.fromisoformat(str(next_review_after).replace("Z", "+00:00"))
        except ValueError:
            return True
        return datetime.now(timezone.utc) >= due

    def _resolve_event_audit(self, event: SupervisorEvent) -> None:
        scope_hash = self._event_scope_hash(event)
        event.audit_scope_hash = scope_hash
        entry = self._audit_index.get(scope_hash, {})
        prior = str(entry.get("audit_status", "")).strip()
        if prior in {"audit-reviewed", "audit-fixed", "audit-false-positive"}:
            event.audit_status = prior
        elif prior == "audit-deferred" and not self._timestamp_is_due(str(entry.get("audit_next_review_after", ""))):
            event.audit_status = "audit-deferred"
        else:
            event.audit_status = "audit-new"
        event.audit_reviewed_by = str(entry.get("audit_reviewed_by", ""))
        event.audit_reviewed_at = str(entry.get("audit_reviewed_at", ""))
        event.audit_review_id = str(entry.get("audit_review_id", "")) or scope_hash[:12]
        event.audit_next_review_after = str(entry.get("audit_next_review_after", ""))
        event.audit_resolution_pr = str(entry.get("audit_resolution_pr", ""))
        event.audit_notes = str(entry.get("audit_notes", ""))

    def record_audit_checkpoint(
        self,
        scope_hash: str,
        *,
        audit_status: str,
        reviewed_by: str = "supervisor",
        review_id: str = "",
        next_review_after: str = "",
        resolution_pr: str = "",
        notes: str = "",
    ) -> None:
        if audit_status not in AUDIT_STATUSES:
            raise ValueError(f"Unsupported audit status: {audit_status}")
        normalized_hash = str(scope_hash or "").strip()
        if not normalized_hash:
            raise ValueError("scope_hash is required")
        self._audit_index[normalized_hash] = {
            "audit_status": audit_status,
            "audit_reviewed_by": str(reviewed_by or ""),
            "audit_reviewed_at": self._timestamp_now_iso(),
            "audit_review_id": str(review_id or normalized_hash[:12]),
            "audit_scope_hash": normalized_hash,
            "audit_next_review_after": str(next_review_after or ""),
            "audit_resolution_pr": str(resolution_pr or ""),
            "audit_notes": _safe_redact(notes or ""),
        }
        self._persist_audit_index()

    def _build_api_escalation_packet(
        self,
        run: SupervisorRun,
        config: RankSupervisorConfig,
        *,
        failure: str,
        cycle: int,
        stage_statuses: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        recent_events = []
        for event in run.events[-10:]:
            recent_events.append({
                "phase": event.phase,
                "status": event.status,
                "detail": _safe_redact(event.detail),
                "data": self._sanitize_escalation_value(event.data),
                "audit_status": event.audit_status,
                "audit_scope_hash": event.audit_scope_hash,
            })
        packet = {
            "run_id": run.run_id,
            "rank_id": run.rank_id,
            "branch": run.branch,
            "goal": self._sanitize_escalation_value(config.goal),
            "failure_class": failure,
            "repair_cycle": cycle,
            "repair_cycles_used": run.repair_cycles_used,
            "mission_orchestration_mode": "staged" if stage_statuses else "single-stage-or-unknown",
            "stage_statuses": self._sanitize_escalation_value(stage_statuses or {}),
            "recent_events": recent_events,
            "policy": {
                "helper_output_is_advice_not_authority": True,
                "must_not_complete_product_manually": True,
                "no_secrets": True,
                "sanitized_logs_only": True,
            },
        }
        return self._sanitize_escalation_value(packet)

    @staticmethod
    def _validate_helper_response(payload: Any) -> Tuple[bool, Dict[str, Any], str]:
        if not isinstance(payload, dict):
            return False, {}, "helper response is not a JSON object"
        required = [
            "diagnosis",
            "likely_supervisor_gap",
            "suggested_repair_strategy",
            "suggested_tests",
            "risk",
            "confidence",
            "requires_human_or_codex_audit",
            "must_not_complete_product_manually",
        ]
        missing = [key for key in required if key not in payload]
        normalized = {
            "diagnosis": payload.get("diagnosis", ""),
            "likely_supervisor_gap": payload.get("likely_supervisor_gap", ""),
            "suggested_repair_strategy": payload.get("suggested_repair_strategy", ""),
            "suggested_tests": payload.get("suggested_tests", []),
            "risk": payload.get("risk", "unknown"),
            "confidence": payload.get("confidence", 0),
            "requires_human_or_codex_audit": bool(payload.get("requires_human_or_codex_audit", False)),
            "must_not_complete_product_manually": bool(payload.get("must_not_complete_product_manually", False)),
            # Execution-plan fields (optional, backward-compatible).
            # advice_only is always True — helper is never an authority.
            "advice_only": True,
            "execution_plan": str(payload.get("execution_plan", "") or ""),
            "file_targets": list(payload.get("file_targets", []) or []),
            "operations": list(payload.get("operations", []) or []),
            "acceptance_matrix": list(payload.get("acceptance_matrix", []) or []),
            "required_tests": list(payload.get("required_tests", []) or []),
            "do_not_do": list(payload.get("do_not_do", []) or []),
            "retry_focus": str(payload.get("retry_focus", "") or ""),
        }
        if missing:
            return False, normalized, f"missing required helper fields: {', '.join(missing)}"
        return True, normalized, ""

    def _maybe_api_escalate(
        self,
        run: SupervisorRun,
        config: RankSupervisorConfig,
        *,
        failure: str,
        cycle: int,
        stage_statuses: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not config.allow_api_escalation:
            run.add("api_escalation", "skipped", "API escalation disabled by config.")
            return None
        if run.api_escalations_used >= config.max_api_escalations_per_run:
            run.add("api_escalation", "skipped", "API escalation call budget exhausted.", budget_type="calls")
            return None
        if run.api_budget_used_usd >= config.max_api_budget_usd:
            run.add("api_escalation", "skipped", "API escalation USD budget exhausted.", budget_type="usd")
            return None

        # Pre-flight: if the helper is not configured, skip without consuming
        # call budget.  The operator may have set allow_api_escalation=True but
        # not yet provided IGRIS_API_HELPER_COMMAND; burning budget here is
        # unhelpful and misleads the UI into showing api=N/api=N as exhausted.
        if not self.backend.api_helper_is_configured():
            run.api_escalations_failed_unconfigured += 1
            run.add(
                "api_escalation",
                "not_configured",
                "API helper command is not configured (IGRIS_API_HELPER_COMMAND unset); "
                "escalation skipped without consuming call budget.",
                budget_type="unconfigured",
            )
            return None
        # Resolve effective mode: config field takes precedence over env var so
        # that per-run config can override the operator .env setting.
        effective_mode = config.api_helper_mode.strip() or os.getenv("IGRIS_API_HELPER_MODE", "").strip() or "auto"
        is_codex_only = effective_mode == "codex_only"

        packet = self._build_api_escalation_packet(run, config, failure=failure, cycle=cycle, stage_statuses=stage_statuses)
        run.add(
            "api_escalation_request",
            "running",
            "Calling API helper for advisory diagnosis and recovery plan.",
            model=config.api_helper_model,
            max_tokens=config.max_tokens_per_escalation,
            api_helper_mode=effective_mode,
            api_helper_model_requested=config.api_helper_model,
            codex_only=is_codex_only,
            packet=packet,
        )
        result = self.backend.call_api_helper(
            packet,
            model=config.api_helper_model,
            max_tokens=config.max_tokens_per_escalation,
            timeout=min(60, config.reasoning_timeout_seconds),
            mode=effective_mode,
        )
        # Only count as a used escalation when the helper was actually called
        # (configured, regardless of whether it succeeded).
        run.api_escalations_used += 1
        if not result.success:
            run.add("api_escalation_response", "failure", _command_detail(result),
                    api_helper_mode=effective_mode, codex_only=is_codex_only)
            return None
        try:
            raw_payload = json.loads(result.output or "{}")
        except json.JSONDecodeError:
            run.add("api_escalation_response", "failure", "API helper returned invalid JSON.",
                    api_helper_mode=effective_mode, codex_only=is_codex_only)
            return None
        valid, advice, error = self._validate_helper_response(raw_payload)
        if not valid:
            run.add("api_escalation_response", "failure", error,
                    api_helper_mode=effective_mode, codex_only=is_codex_only,
                    payload=self._sanitize_escalation_value(raw_payload))
            return None
        try:
            estimated_cost_usd = max(0.0, float(raw_payload.get("estimated_cost_usd", 0.0)))
        except (TypeError, ValueError):
            estimated_cost_usd = 0.0
        run.api_budget_used_usd += estimated_cost_usd
        # Extract observability fields from helper response
        model_resolved = str(raw_payload.get("api_helper_model_resolved", raw_payload.get("model", "")))
        helper_provider = str(raw_payload.get("api_helper_provider", ""))
        run.add(
            "api_escalation_response",
            "success",
            "API helper advice received and recorded.",
            advice=self._sanitize_escalation_value(advice),
            estimated_cost_usd=estimated_cost_usd,
            helper_is_authority=False,
            api_helper_mode=effective_mode,
            api_helper_provider=helper_provider,
            api_helper_model_requested=config.api_helper_model,
            api_helper_model_resolved=model_resolved,
            helper_model=result.helper_model or config.api_helper_model,
            helper_alt_model=result.helper_ab_alt_model,
            helper_ab_active=result.helper_ab_active,
            helper_ab_shadow_mode=result.helper_ab_shadow_mode,
            helper_primary_score=result.helper_primary_score,
            helper_alt_score=result.helper_alt_score,
            helper_alt_used_for_decision=False,
            helper_switch_recommendation=result.helper_switch_recommendation,
            codex_only=is_codex_only,
        )
        return advice

    # ------------------------------------------------------------------
    # Cost-policy execution strategy helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_max_same_failure_retries() -> int:
        """Max consecutive same-failure repairs before escalating to strong model."""
        try:
            return max(1, int(os.getenv("IGRIS_MAX_SAME_FAILURE_RETRIES", "2") or "2"))
        except (ValueError, TypeError):
            return 2

    @staticmethod
    def _get_max_cost_per_run() -> float:
        """USD cap per supervised run; 0 means unlimited."""
        try:
            return max(0.0, float(os.getenv("IGRIS_MAX_COST_PER_RUN", "0") or "0"))
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _is_codex_direct_execution_enabled() -> bool:
        """Experimental Codex direct execution — off by default.

        Only active when IGRIS_ENABLE_CODEX_DIRECT_EXECUTION=true.
        Uses its own budget: IGRIS_MAX_CODEX_DIRECT_BUDGET_USD (default 0 = disabled).
        """
        return os.getenv("IGRIS_ENABLE_CODEX_DIRECT_EXECUTION", "").lower() in ("true", "1", "yes")

    @staticmethod
    def _get_max_codex_direct_budget_usd() -> float:
        """USD cap for experimental codex direct execution; 0 means disabled."""
        try:
            return max(0.0, float(os.getenv("IGRIS_MAX_CODEX_DIRECT_BUDGET_USD", "0") or "0"))
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _get_max_cost_per_issue() -> float:
        """USD cap per issue; 0 means unlimited. Not yet enforced cross-run."""
        try:
            return max(0.0, float(os.getenv("IGRIS_MAX_COST_PER_ISSUE", "0") or "0"))
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _strategy_for_repair(
        run: SupervisorRun,
        has_execution_plan: bool,
    ) -> Tuple[str, Optional[str]]:
        """Return (strategy_name, preferred_profile) for the next repair cycle.

        Rules:
        - No execution plan → no strategy override (return empty/None).
        - same_failure_count < threshold → mini strategy (cheap, helper-guided).
        - same_failure_count >= threshold → strong strategy (gpt-4o escalation).

        Codex direct execution is experimental and NOT selected here; it requires
        IGRIS_ENABLE_CODEX_DIRECT_EXECUTION=true and is handled separately.
        """
        if not has_execution_plan:
            return "", None
        max_retries = SelfRepairSupervisor._get_max_same_failure_retries()
        if run.same_failure_count >= max_retries:
            return "helper_advice_then_gpt4o_execution", "strong_execution"
        return "helper_advice_then_mini_execution", "mini_execution"

    @staticmethod
    def _check_execution_budget(run: SupervisorRun) -> Optional[str]:
        """Return failure_class string if execution budget is exceeded, else None."""
        max_per_run = SelfRepairSupervisor._get_max_cost_per_run()
        if max_per_run > 0 and run.execution_budget_used_usd >= max_per_run:
            return "execution_budget_exceeded"
        return None

    @staticmethod
    def _build_telemetry_fragment(
        time_to_first_diff_s: Optional[float],
        no_diff_count: int,
        decompose_count: int,
        attempt_outcomes: List[str],
        total_attempts: int,
    ) -> Dict[str, Any]:
        """Build execution-effectiveness telemetry fragment for run.report (Issue #715)."""
        denom = max(total_attempts, 1)
        return {
            "time_to_first_diff_s": time_to_first_diff_s,
            "no_diff_rate": round(no_diff_count / denom, 4),
            "decompose_rate": round(decompose_count / denom, 4),
            "attempt_outcomes": list(attempt_outcomes),
        }

    @staticmethod
    def _repair_issue_already_created(run: SupervisorRun, failure: str) -> bool:
        for event in run.events:
            if event.phase != "repair_issue":
                continue
            if event.status != "success":
                continue
            if str(event.data.get("failure_class", "")) == failure:
                return True
        return False

    @staticmethod
    def _goal_requires_backend_change(goal: str) -> bool:
        lowered = goal.lower()
        return any(token in lowered for token in ("backend", "api", "endpoint", "/api/"))

    @staticmethod
    def _goal_requires_docs_or_config(goal: str) -> bool:
        lowered = goal.lower()
        return any(token in lowered for token in ("docs", "documentation", "readme", "config"))

    @staticmethod
    def _goal_requires_tests(goal: str) -> bool:
        lowered = goal.lower()
        return any(token in lowered for token in ("test", "pytest", "coverage"))

    def _mission_is_non_trivial(self, config: RankSupervisorConfig) -> bool:
        score = 0
        if self._goal_requires_backend_change(config.goal):
            score += 1
        if self._goal_requires_ui_visibility(config.goal):
            score += 1
        if self._goal_requires_tests(config.goal) or bool(config.targeted_tests):
            score += 1
        if self._goal_requires_docs_or_config(config.goal):
            score += 1
        lowered = config.goal.lower()
        if any(token in lowered for token in ("e2e", "end-to-end", "workflow", "ci", "merge", "restart", "smoke")):
            score += 1
        if len(config.targeted_tests) > 1:
            score += 1
        return score >= 4

    def _build_mission_plan(self, config: RankSupervisorConfig) -> MissionPlan:
        if not self._mission_is_non_trivial(config):
            return MissionPlan(
                mode="single-stage",
                stages=[
                    MissionStage(
                        stage_id="single_stage_execution",
                        goal="Execute the mission as one bounded stage with validation gates.",
                        required=True,
                        allowed_file_families=["igris/", "tests/", "docs/"],
                        acceptance_criteria=["Goal-aligned diff exists and validation gates are green."],
                        validation=["targeted tests (if configured)", "full pytest", "smoke endpoints"],
                        rollback_policy="Restore only for unsafe/off-contract diffs.",
                        preserved_progress_policy="Keep valid edits unless unsafe/off-contract.",
                        failure_classification=list(sorted(REPAIRABLE_FAILURES)),
                        repair_strategy="Use repair cycle scoped to detected failure class.",
                        report_entry="Single-stage execution status and validation results.",
                    )
                ],
            )

        stages: List[MissionStage] = [
            MissionStage(
                stage_id="understand_locate",
                goal="Understand mission scope and locate relevant files before edits.",
                required=True,
                allowed_file_families=[],
                acceptance_criteria=["Relevant files identified and mission constraints captured."],
                validation=["context initialized"],
                rollback_policy="No rollback needed; no edits expected.",
                preserved_progress_policy="Always preserve stage output metadata.",
                failure_classification=["reasoning_loop_blocked", "max_steps", "ask_user"],
                repair_strategy=(
                    "On reasoning_loop_blocked: reduce scope, request minimal change only. "
                    "Cycle 2+: escalate to API helper with full reasoning context."
                ),
                report_entry="Understanding/locating stage result.",
            ),
            MissionStage(
                stage_id="backend_api_change",
                goal="Implement backend/API changes required by the mission.",
                required=self._goal_requires_backend_change(config.goal),
                allowed_file_families=["igris/web/server.py", "igris/core/"],
                acceptance_criteria=["Mission-owned backend/API behavior implemented."],
                validation=["target endpoint reachable in tests/smoke"],
                rollback_policy="Restore only unsafe/off-contract backend edits.",
                preserved_progress_policy="Preserve validated backend edits through later stage repairs.",
                failure_classification=["wrong_file_edit", "syntax_error", "reasoning_loop_blocked"],
                repair_strategy=(
                    "Stage-scoped repair that keeps validated earlier stages. "
                    "On reasoning_loop_blocked: reduce scope, request minimal change only. "
                    "Cycle 2+: escalate to API helper with full reasoning context."
                ),
                report_entry="Backend/API stage result.",
            ),
            MissionStage(
                stage_id="backend_tests",
                goal="Add or update backend tests required by the mission.",
                required=self._goal_requires_tests(config.goal) or bool(config.targeted_tests),
                allowed_file_families=["tests/test_"],
                acceptance_criteria=["Backend test coverage for mission endpoint exists."],
                validation=["targeted pytest files"],
                rollback_policy="Restore only unsafe/off-contract test edits.",
                preserved_progress_policy="Preserve validated backend tests on later failures.",
                failure_classification=["missing_tests", "wrong_file_edit", "pytest_failure", "test_runner_timeout"],
                repair_strategy="Repair missing/invalid tests without deleting validated code.",
                report_entry="Backend tests stage result.",
            ),
            MissionStage(
                stage_id="ui_dashboard_change",
                goal="Apply minimal non-destructive UI/dashboard visibility changes when required.",
                required=self._goal_requires_ui_visibility(config.goal),
                allowed_file_families=[
                    "igris/web/templates/",
                    "igris/web/static/js/",
                    "igris/web/static/css/",
                ],
                acceptance_criteria=["UI/dashboard visibility for mission objective is present."],
                validation=["UI/dashboard smoke checks"],
                rollback_policy="Restore only unsafe/off-contract UI edits.",
                preserved_progress_policy="Do not discard validated backend/test stages on UI failure.",
                failure_classification=["missing_ui_visibility", "wrong_file_edit", "pytest_failure", "test_runner_timeout"],
                repair_strategy="Repair only UI stage scope while preserving validated prior stages.",
                report_entry="UI/dashboard stage result.",
            ),
            MissionStage(
                stage_id="ui_dashboard_tests",
                goal="Add or update UI/dashboard smoke tests when UI stage is required.",
                required=self._goal_requires_ui_visibility(config.goal),
                allowed_file_families=["tests/test_"],
                acceptance_criteria=["UI/dashboard smoke tests cover the new visibility signal."],
                validation=["targeted ui/dashboard tests"],
                rollback_policy="Restore only unsafe/off-contract test edits.",
                preserved_progress_policy="Preserve validated backend/UI changes.",
                failure_classification=["pytest_failure", "wrong_file_edit", "missing_tests", "test_runner_timeout"],
                repair_strategy="Stage-scoped test repair only; do not rewrite unrelated tests.",
                report_entry="UI/dashboard tests stage result.",
            ),
            MissionStage(
                stage_id="docs_config_update",
                goal="Update docs/config only when mission explicitly requires it.",
                required=False,
                allowed_file_families=["docs/", "README", "pyproject.toml", "requirements"],
                acceptance_criteria=["Docs/config aligned with delivered behavior when relevant."],
                validation=["diff review for docs/config scope"],
                rollback_policy="Skip restore for not-applicable stage.",
                preserved_progress_policy="No-op skip is valid when stage is not relevant.",
                failure_classification=["wrong_file_edit"],
                repair_strategy="Skip with explanation when not applicable; otherwise minimal patch.",
                report_entry="Docs/config stage result.",
            ),
            MissionStage(
                stage_id="targeted_tests",
                goal="Run targeted tests for mission-owned files.",
                required=bool(config.targeted_tests),
                allowed_file_families=[],
                acceptance_criteria=["Targeted tests green."],
                validation=["pytest -q <targets>"],
                rollback_policy="No restore for test execution itself.",
                preserved_progress_policy="Preserve validated code when targeted tests fail and repair tests only.",
                failure_classification=["missing_tests", "pytest_failure", "test_runner_timeout"],
                repair_strategy="Repair targeted failures with stage-scoped cycle.",
                report_entry="Targeted tests stage result.",
            ),
            MissionStage(
                stage_id="full_pytest",
                goal="Run full pytest for repository-wide safety.",
                required=True,
                allowed_file_families=[],
                acceptance_criteria=["Full pytest green."],
                validation=["pytest -q"],
                rollback_policy="No restore for test execution itself.",
                preserved_progress_policy="Preserve validated stages and repair only failing scope.",
                failure_classification=["pytest_failure", "test_runner_timeout"],
                repair_strategy="Repair failing scope while keeping validated progress.",
                report_entry="Full pytest stage result.",
            ),
            MissionStage(
                stage_id="pr_ci_merge",
                goal="Complete PR/CI/merge workflow when enabled.",
                required=not config.dry_run,
                allowed_file_families=[],
                acceptance_criteria=["PR opened, CI green, merged when allowed."],
                validation=["gh pr checks --watch"],
                rollback_policy="No rollback for disabled workflow; mark skipped with reason.",
                preserved_progress_policy="Preserve validated branch content.",
                failure_classification=["infrastructure_bug"],
                repair_strategy="Retry delivery actions only after code validation is green.",
                report_entry="PR/CI/merge stage result.",
            ),
            MissionStage(
                stage_id="post_merge_runtime",
                goal="Pull main, restart runtime and run live smoke when enabled.",
                required=not config.dry_run,
                allowed_file_families=[],
                acceptance_criteria=["Post-merge smoke green on refreshed runtime."],
                validation=["required smoke endpoints"],
                rollback_policy="Block completion if runtime smoke fails.",
                preserved_progress_policy="Preserve merged code; classify runtime failures separately.",
                failure_classification=["infrastructure_bug", "invalid_bootstrap"],
                repair_strategy="Repair runtime/bootstrap and rerun smoke.",
                report_entry="Post-merge runtime stage result.",
            ),
            MissionStage(
                stage_id="final_report",
                goal="Emit truthful final report with per-stage statuses.",
                required=True,
                allowed_file_families=[],
                acceptance_criteria=["All required stages are green before completed status."],
                validation=["stage status audit"],
                rollback_policy="Never mark completed if required stages are missing/failed.",
                preserved_progress_policy="Stage statuses remain visible even when blocked.",
                failure_classification=["infrastructure_bug"],
                repair_strategy="Report blocked/repair honestly with stage diagnostics.",
                report_entry="Final report stage result.",
            ),
        ]
        return MissionPlan(mode="staged", stages=stages)

    @staticmethod
    def _stage_status_template(stage: MissionStage) -> Dict[str, Any]:
        return {
            "stage_id": stage.stage_id,
            "goal": stage.goal,
            "required": stage.required,
            "allowed_file_families": list(stage.allowed_file_families),
            "acceptance_criteria": list(stage.acceptance_criteria),
            "validation": list(stage.validation),
            "rollback_policy": stage.rollback_policy,
            "preserved_progress_policy": stage.preserved_progress_policy,
            "failure_classification": list(stage.failure_classification),
            "repair_strategy": stage.repair_strategy,
            "report_entry": stage.report_entry,
            "status": "pending",
            "detail": "",
            "no_op": False,
            "non_blocking_behaviors": [],
        }

    def _init_stage_statuses(self, plan: MissionPlan) -> Dict[str, Dict[str, Any]]:
        return {stage.stage_id: self._stage_status_template(stage) for stage in plan.stages}

    def _set_stage_status(
        self,
        run: SupervisorRun,
        statuses: Dict[str, Dict[str, Any]],
        stage_id: str,
        status: str,
        detail: str,
        *,
        no_op: bool = False,
    ) -> None:
        if stage_id not in statuses:
            return
        entry = statuses[stage_id]
        entry["status"] = status
        entry["detail"] = detail
        entry["no_op"] = bool(no_op)
        run.add(
            "mission_stage",
            status,
            detail,
            stage_id=stage_id,
            required=entry.get("required", False),
            no_op=bool(no_op),
        )

    def _track_non_blocking_behavior(
        self,
        run: SupervisorRun,
        statuses: Dict[str, Dict[str, Any]],
        stage_id: str,
        code: str,
        detail: str,
    ) -> None:
        if stage_id not in statuses:
            return
        entry = statuses[stage_id]
        behaviors = entry.setdefault("non_blocking_behaviors", [])
        payload = {"code": code, "detail": detail}
        behaviors.append(payload)
        run.add(
            "mission_stage_behavior",
            "tracked",
            detail,
            stage_id=stage_id,
            behavior_code=code,
            blocking=False,
        )

    @staticmethod
    def _required_stages_green(
        statuses: Dict[str, Dict[str, Any]],
        *,
        include_final_report: bool = False,
        exclude_stage_ids: Optional[Set[str]] = None,
    ) -> bool:
        excluded = exclude_stage_ids or set()
        for stage_id, entry in statuses.items():
            if stage_id in excluded:
                continue
            if not entry.get("required", False):
                continue
            if stage_id == "final_report" and not include_final_report:
                continue
            if entry.get("status") not in {"success", "skipped"}:
                return False
        return True

    @staticmethod
    def _compute_degraded_completion(
        *,
        completion_mode: str,
        runtime_refresh_required: bool,
        post_merge_smoke_success: bool,
        smoke_was_applicable: bool,
        failure_class: str,
        stage_statuses: Optional[Dict[str, Dict[str, Any]]],
    ) -> Tuple[bool, str]:
        """Return ``(degraded_completion, degraded_completion_reason)``.

        A completion is **clean** (degraded=False) when all of the following hold:
        - ``failure_class`` is empty
        - all required stages are green (or there is no stage system)
        - when a post-merge smoke was applicable (merge was actually attempted),
          it either passed or ``runtime_refresh_required`` was False

        Any condition that is not met makes the completion *degraded*, and an
        explicit human-readable reason string is always returned alongside
        ``degraded=True`` so that callers and the UI can surface the cause.

        ``smoke_was_applicable`` must be True only when a merge was actually
        executed (not dry-run, not merge-disabled).  When smoke is not
        applicable (dry-run / merge skipped), the ``runtime_refresh_required``
        flag is irrelevant to delivery quality and must not trigger degraded.

        Note: ``completion_mode == "verified_diff"`` alone is *not* a
        degradation signal — it describes the reasoning path, not delivery
        quality.  Only genuine delivery failures (failed stages, unconfirmed
        smoke, non-empty failure_class) trigger degraded.
        """
        reasons: List[str] = []
        required_all_green = (
            stage_statuses is None
            or SelfRepairSupervisor._required_stages_green(stage_statuses)
        )
        if failure_class:
            reasons.append(f"failure_class set: {failure_class}")
        if not required_all_green:
            reasons.append("not all required stages passed")
        if smoke_was_applicable and runtime_refresh_required and not post_merge_smoke_success:
            reasons.append(
                "post-merge smoke deferred; runtime refresh required but smoke not confirmed"
            )
        return bool(reasons), "; ".join(reasons)

    @staticmethod
    def _stage_status_list(statuses: Dict[str, Dict[str, Any]], plan: MissionPlan) -> List[Dict[str, Any]]:
        ordered: List[Dict[str, Any]] = []
        for stage in plan.stages:
            entry = dict(statuses.get(stage.stage_id, {}))
            if entry:
                ordered.append(entry)
        return ordered

    def _stage_is_already_satisfied(self, stage: MissionStage, config: RankSupervisorConfig) -> bool:
        if stage.stage_id == "understand_locate":
            return True
        if stage.stage_id == "backend_api_change":
            endpoint = _required_endpoint_from_goal(config.goal)
            if not endpoint:
                return False
            server_path = Path(self.project_root) / "igris/web/server.py"
            if not server_path.exists():
                return False
            try:
                content = server_path.read_text(encoding="utf-8").lower()
            except OSError:
                return False
            return endpoint in content
        if stage.stage_id in {"backend_tests", "ui_dashboard_tests", "targeted_tests"}:
            if not config.targeted_tests:
                return stage.stage_id == "targeted_tests"
            return all((Path(self.project_root) / target).exists() for target in config.targeted_tests)
        if stage.stage_id == "ui_dashboard_change":
            if not self._goal_requires_ui_visibility(config.goal):
                return True
            endpoint = _required_endpoint_from_goal(config.goal).replace("/", "-").strip("-")
            index_path = Path(self.project_root) / "igris/web/templates/index.html"
            if not index_path.exists():
                return False
            try:
                content = index_path.read_text(encoding="utf-8").lower()
            except OSError:
                return False
            return endpoint in content if endpoint else ("rank" in content and "dashboard" in content)
        return False

    @staticmethod
    def _ui_stage_hard_forbidden_paths(
        statuses: Dict[str, Dict[str, Any]],
        config: RankSupervisorConfig,
    ) -> Set[str]:
        forbidden: Set[str] = set()
        if statuses.get("backend_api_change", {}).get("status") == "success":
            forbidden.add("igris/web/server.py")
        if statuses.get("backend_tests", {}).get("status") == "success":
            for target in config.targeted_tests:
                normalized = str(target or "").strip()
                if normalized:
                    forbidden.add(normalized)
        return forbidden

    def _ui_stage_retry_goal(
        self,
        *,
        base_goal: str,
        stage: MissionStage,
        hard_forbidden: Set[str],
        retry_attempt: int,
        invalid_paths: List[str],
    ) -> str:
        policy_lines = [
            "UI-only recovery policy:",
            "- Do not modify igris/web/server.py.",
            "- Do not modify validated backend endpoint contract files or validated backend tests.",
            "- Search existing UI/dashboard files first.",
            "- Modify only UI/dashboard files under igris/web/templates/, igris/web/static/js/, igris/web/static/css/.",
            "- Add minimal Rank S UI/dashboard visibility and update relevant UI/dashboard tests in their stage.",
        ]
        forbidden_line = ", ".join(sorted(hard_forbidden)) or "igris/web/server.py"
        retry_line = ""
        if retry_attempt > 0:
            retry_line = (
                f"\nRetry attempt {retry_attempt}: previous wrong_file_edit touched: "
                f"{', '.join(invalid_paths) or 'unknown paths'}."
            )
        return (
            f"{base_goal}\n\n"
            f"[stage:{stage.stage_id}] {stage.goal}\n"
            f"Allowed file families: {', '.join(stage.allowed_file_families) or 'mission-owned minimal scope'}.\n"
            f"Acceptance criteria: {'; '.join(stage.acceptance_criteria)}\n"
            + "\n".join(policy_lines)
            + f"\nHard-forbidden paths for this stage: {forbidden_line}."
            + retry_line
        )

    def _restore_ui_stage_scope(
        self,
        run: SupervisorRun,
        stage: MissionStage,
        changed_paths: Set[str],
        observed_paths: List[str],
    ) -> Tuple[bool, List[str]]:
        candidates = set(changed_paths)
        for path in observed_paths:
            if path:
                candidates.add(path)
        restore_paths = sorted(
            path for path in candidates
            if self._path_in_allowed_family(path, stage.allowed_file_families)
        )
        restore = self.backend.restore_paths(restore_paths)
        run.add(
            "ui_stage_restore",
            "success" if restore.success else "failure",
            "Restoring UI-stage scoped edits after wrong_file_edit.",
            restored_paths=restore_paths,
        )
        return restore.success, restore_paths

    @staticmethod
    def _path_in_allowed_family(path: str, families: List[str]) -> bool:
        for family in families:
            if family.endswith("/"):
                if path.startswith(family):
                    return True
                continue
            if family.endswith("test_"):
                if path.startswith(family):
                    return True
                continue
            if family in {"README", "pyproject.toml", "requirements"}:
                if path == family or path.startswith(f"{family}."):
                    return True
                continue
            if path == family or path.startswith(family):
                return True
        return False

    def _validate_new_stage_paths(
        self,
        stage: MissionStage,
        before_paths: Set[str],
        after_paths: Set[str],
        touched_files: List[str],
        changed_paths: Optional[Set[str]] = None,
    ) -> Tuple[bool, str]:
        if not stage.allowed_file_families:
            return True, ""
        paths_to_check = set(after_paths - before_paths)
        if changed_paths:
            paths_to_check.update(path for path in changed_paths if path)
        for touched in touched_files:
            normalized = str(touched or "").strip()
            if not normalized:
                continue
            if normalized.startswith("./"):
                normalized = normalized[2:]
            if normalized.startswith("b/"):
                normalized = normalized[2:]
            paths_to_check.add(normalized)
        candidate_paths = sorted(path for path in paths_to_check if path)
        if not candidate_paths:
            return True, ""
        invalid = [
            path for path in candidate_paths
            if not self._path_in_allowed_family(path, stage.allowed_file_families)
        ]
        if not invalid:
            return True, ""
        return False, ", ".join(invalid)

    def _execute_staged_reasoning(
        self,
        run: SupervisorRun,
        config: RankSupervisorConfig,
        plan: MissionPlan,
        statuses: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], str, bool]:
        aggregated_files: List[str] = []
        summaries: List[str] = []
        stage_failure = ""
        runtime_refresh_required = False
        last_status = "finished"
        last_stop_reason = "finish"
        loop_ids: List[str] = []

        for stage in plan.stages:
            if stage.stage_id in {"targeted_tests", "full_pytest", "pr_ci_merge", "post_merge_runtime", "final_report"}:
                continue

            if not stage.required and stage.stage_id == "docs_config_update" and not self._goal_requires_docs_or_config(config.goal):
                self._set_stage_status(
                    run,
                    statuses,
                    stage.stage_id,
                    "skipped",
                    "Optional docs/config stage skipped: mission does not require docs/config updates.",
                    no_op=True,
                )
                continue

            if not stage.required:
                self._set_stage_status(
                    run,
                    statuses,
                    stage.stage_id,
                    "skipped",
                    "Stage is optional for this mission and was skipped.",
                    no_op=True,
                )
                continue

            current_status = statuses.get(stage.stage_id, {}).get("status")
            if current_status in {"success", "skipped"}:
                self._set_stage_status(
                    run,
                    statuses,
                    stage.stage_id,
                    "success",
                    "Stage already validated; preserving progress across attempts.",
                    no_op=True,
                )
                continue

            if self._stage_is_already_satisfied(stage, config):
                self._set_stage_status(
                    run,
                    statuses,
                    stage.stage_id,
                    "success",
                    "Stage already satisfied; marked complete as no-op.",
                    no_op=True,
                )
                continue

            if stage.stage_id == "understand_locate":
                self._set_stage_status(
                    run,
                    statuses,
                    stage.stage_id,
                    "success",
                    "Mission scope classified and relevant files located.",
                    no_op=True,
                )
                continue

            ui_retry_attempt = 0
            ui_retry_budget = 2
            ui_retry_invalid_paths: List[str] = []
            ui_seen_paths: Set[str] = set()
            ui_hard_forbidden = (
                self._ui_stage_hard_forbidden_paths(statuses, config)
                if stage.stage_id == "ui_dashboard_change"
                else set()
            )
            while True:
                before_diff = self.backend.git_diff()
                before_paths = set(_diff_changed_paths(before_diff.output))
                stage_goal = (
                    self._ui_stage_retry_goal(
                        base_goal=config.goal,
                        stage=stage,
                        hard_forbidden=ui_hard_forbidden,
                        retry_attempt=ui_retry_attempt,
                        invalid_paths=ui_retry_invalid_paths,
                    )
                    if stage.stage_id == "ui_dashboard_change"
                    else (
                        f"{config.goal}\n\n"
                        f"[stage:{stage.stage_id}] {stage.goal}\n"
                        f"Allowed file families: {', '.join(stage.allowed_file_families) or 'mission-owned minimal scope'}.\n"
                        f"Acceptance criteria: {'; '.join(stage.acceptance_criteria)}"
                    )
                )
                stage_context = self._rank_initial_context(config, run=run)
                stage_context.update({
                    "mission_orchestration_mode": plan.mode,
                    "mission_stage_id": stage.stage_id,
                    "mission_stage_goal": stage.goal,
                    "mission_stage_allowed_file_families": stage.allowed_file_families,
                    "mission_stage_acceptance_criteria": stage.acceptance_criteria,
                    "mission_stage_validation": stage.validation,
                    "mission_stage_repair_strategy": stage.repair_strategy,
                })
                rank_task_type = (
                    "semantic_repair"
                    if stage.stage_id and "endpoint" in str(stage.stage_id).lower()
                    else "code_reasoning"
                )
                run.add(
                    "rank_reasoning",
                    "running",
                    f"Running staged mission reasoning: {stage.stage_id}",
                    stage_id=stage.stage_id,
                    timeout_seconds=config.reasoning_timeout_seconds,
                    task_type=rank_task_type,
                )
                result = self.backend.run_reasoning(
                    stage_goal,
                    max_steps=140,
                    initial_context=stage_context,
                    timeout=config.reasoning_timeout_seconds,
                    task_type=rank_task_type,
                )
                status = str(result.get("status", ""))
                stop_reason = str(result.get("stop_reason", ""))
                files_modified = list(result.get("files_modified") or [])
                attempted_write_paths = _extract_attempted_write_paths(result)
                if files_modified:
                    for path in files_modified:
                        if path not in aggregated_files:
                            aggregated_files.append(path)
                summaries.append(f"[{stage.stage_id}] {result.get('final_summary', '')}")
                loop_id = str(result.get("loop_id", ""))
                if loop_id:
                    loop_ids.append(loop_id)
                run.add(
                    "rank_reasoning",
                    status,
                    result.get("final_summary", ""),
                    stage_id=stage.stage_id,
                    loop_id=loop_id,
                    stop_reason=stop_reason,
                    files_modified=files_modified,
                    attempted_write_paths=attempted_write_paths,
                    orchestrator_used=result.get("orchestrator_used", False),
                    reasoning_execution_provider=result.get("reasoning_execution_provider", ""),
                    reasoning_execution_model=result.get("reasoning_execution_model", ""),
                    reasoning_execution_profile=result.get("reasoning_execution_profile", ""),
                    local_model_available=result.get("local_model_available", False),
                )
                after_diff = self.backend.git_diff()
                after_paths = set(_diff_changed_paths(after_diff.output))
                changed_paths = _changed_paths_between_diffs(before_diff.output, after_diff.output)
                observed_paths = list(dict.fromkeys(files_modified + attempted_write_paths + sorted(changed_paths)))
                valid_paths, invalid_paths = self._validate_new_stage_paths(
                    stage,
                    before_paths,
                    after_paths,
                    observed_paths,
                    changed_paths=changed_paths,
                )
                if not valid_paths:
                    invalid_path_list = [path.strip() for path in invalid_paths.split(",") if path.strip()]
                    if stage.stage_id == "ui_dashboard_change":
                        for path in observed_paths:
                            if self._path_in_allowed_family(path, stage.allowed_file_families):
                                ui_seen_paths.add(path)
                        if any(path in ui_hard_forbidden for path in invalid_path_list):
                            ui_retry_invalid_paths = invalid_path_list
                            restored_ok, restored_paths = self._restore_ui_stage_scope(
                                run,
                                stage,
                                changed_paths,
                                observed_paths,
                            )
                            if not restored_ok:
                                self._set_stage_status(
                                    run,
                                    statuses,
                                    stage.stage_id,
                                    "failure",
                                    "UI-stage recovery failed while restoring stage-local edits.",
                                )
                                stage_failure = "wrong_file_edit"
                                last_status = status or "blocked"
                                last_stop_reason = stop_reason or "blocked"
                                break
                            if ui_retry_attempt < ui_retry_budget:
                                ui_retry_attempt += 1
                                run.add(
                                    "ui_stage_retry",
                                    "running",
                                    "Retrying ui_dashboard_change with UI-only constraints after stage-local wrong_file_edit.",
                                    retry_attempt=ui_retry_attempt,
                                    retry_budget=ui_retry_budget,
                                    hard_forbidden=sorted(ui_hard_forbidden),
                                    restored_paths=restored_paths,
                                    invalid_paths=invalid_path_list,
                                )
                                continue
                            searched = sorted(ui_seen_paths) or ["(none)"]
                            self._set_stage_status(
                                run,
                                statuses,
                                stage.stage_id,
                                "failure",
                                "UI-only recovery exhausted after repeated wrong_file_edit attempts. "
                                f"UI files searched/touched: {', '.join(searched)}. "
                                f"Attempted forbidden edits: {', '.join(sorted(set(ui_retry_invalid_paths)) or ['(none)'])}.",
                            )
                            stage_failure = "wrong_file_edit"
                            last_status = status or "blocked"
                            last_stop_reason = stop_reason or "blocked"
                            break
                    self._set_stage_status(
                        run,
                        statuses,
                        stage.stage_id,
                        "failure",
                        f"Stage touched out-of-scope files: {invalid_paths}",
                    )
                    stage_failure = "wrong_file_edit"
                    last_status = status or "blocked"
                    last_stop_reason = stop_reason or "blocked"
                    break
                # Guard: a required stage with allowed_file_families that produced no
                # diff and is not pre-satisfied must be treated as failure, not success.
                # Without this check _validate_new_stage_paths returns (True, "") for
                # empty candidate_paths, and the stage falls through to success —
                # a false positive ("stage required segnato success senza evidence").
                # Only fires when status == "finished": timeout/blocked paths are handled
                # by the existing `if status != "finished":` block below.
                if (
                    status == "finished"
                    and stage.required
                    and stage.allowed_file_families
                    and not changed_paths
                    and not observed_paths
                    and not self._stage_is_already_satisfied(stage, config)
                ):
                    self._set_stage_status(
                        run,
                        statuses,
                        stage.stage_id,
                        "failure",
                        f"Required stage '{stage.stage_id}' produced no file changes "
                        "and is not already satisfied.",
                    )
                    stage_failure = "reasoning_loop_blocked"
                    last_status = status or "blocked"
                    last_stop_reason = stop_reason or "no_change"
                    break
                runtime_refresh_required = runtime_refresh_required or any(str(path).startswith("igris/") for path in files_modified)
                if status != "finished":
                    if (
                        stage.stage_id == "ui_dashboard_change"
                        and stop_reason in {"reasoning_timeout", "budget_exceeded"}
                    ):
                        has_ui_diff = _has_ui_surface_change(after_diff.output)
                        stage_satisfied = self._stage_is_already_satisfied(stage, config)
                        if has_ui_diff or stage_satisfied:
                            self._track_non_blocking_behavior(
                                run,
                                statuses,
                                stage.stage_id,
                                "ui_stage_timeout_accepted",
                                "UI stage timed out but validated UI visibility evidence was present; accepting stage with degraded status.",
                            )
                            self._set_stage_status(
                                run,
                                statuses,
                                stage.stage_id,
                                "success",
                                "UI stage accepted after timeout because mission-owned UI visibility evidence is present.",
                            )
                            last_status = "finished"
                            last_stop_reason = stop_reason or "reasoning_timeout"
                            break
                    if status in {"blocked", "error", "stopped"} or stop_reason in {"blocked", "ask_user", "max_steps", "reasoning_timeout", "budget_exceeded"}:
                        self._set_stage_status(
                            run,
                            statuses,
                            stage.stage_id,
                            "failure",
                            result.get("final_summary", "") or f"Stage {stage.stage_id} did not finish cleanly.",
                        )
                        stage_failure = classify_failure(reasoning_result=result)
                        last_status = status
                        last_stop_reason = stop_reason
                        break
                    self._track_non_blocking_behavior(
                        run,
                        statuses,
                        stage.stage_id,
                        "degraded_reasoning",
                        f"Stage {stage.stage_id} accepted with degraded reasoning status {status}/{stop_reason}.",
                    )
                self._set_stage_status(
                    run,
                    statuses,
                    stage.stage_id,
                    "success",
                    result.get("final_summary", "") or f"Stage {stage.stage_id} completed.",
                )
                last_status = status or "finished"
                last_stop_reason = stop_reason or "finish"
                break
            if stage_failure:
                break

        aggregated = {
            "status": last_status,
            "stop_reason": last_stop_reason,
            "files_modified": aggregated_files,
            "final_summary": "\n".join(part for part in summaries if part).strip(),
            "loop_id": ",".join(loop_ids),
            "goal": config.goal,
        }
        return aggregated, stage_failure, runtime_refresh_required

    def _run_preflight_phase(
        self,
        run: Optional["SupervisorRun"],
        config: RankSupervisorConfig,
    ) -> "Tuple[SupervisorRun, Optional[Dict[str, Any]]]":
        """Phase 1: init, git, baseline, smoke, assignment routing, mission plan.

        Returns (run, None) when blocked or cancelled, (run, ctx) on success.
        ctx keys: mission_plan, stage_statuses, assignment_decision, restart_command.
        """
        run = run or SupervisorRun(run_id=uuid.uuid4().hex[:12], rank_id=config.rank_id)
        self._configure_run_tracking(run, config)
        run.add("start", "running", "Supervisor started", dry_run=config.dry_run)
        # Validate API escalation helper config at run start so problems are
        # visible immediately rather than discovered mid-repair-cycle.
        if config.allow_api_escalation and config.max_api_escalations_per_run > 0:
            if not self.backend.api_helper_is_configured():
                run.add(
                    "api_escalation_config",
                    "not_configured",
                    "API escalation is enabled (allow_api_escalation=True) but "
                    "IGRIS_API_HELPER_COMMAND is not set. Escalation calls will be "
                    "skipped without consuming call budget.",
                    allow_api_escalation=config.allow_api_escalation,
                    max_api_escalations_per_run=config.max_api_escalations_per_run,
                )
        cancelled = self._cancel_if_requested(run)
        if cancelled is not None:
            return cancelled, None
        restart_command = config.service_restart_command
        if config.defer_service_restart and restart_command:
            run.add(
                "service_restart",
                "deferred",
                "Service restart deferred because this supervised run is owned by the API process.",
                command=restart_command,
            )
            restart_command = ""

        status = self.backend.git_status()
        run.add("git_status", "success" if status.success else "failure", _command_detail(status))
        cancelled = self._cancel_if_requested(run)
        if cancelled is not None:
            return cancelled, None
        if not status.success:
            return self._blocked(run, "infrastructure_bug", "Unable to read git status"), None
        # Ignore untracked files (lines starting with "??") — they don't conflict with
        # git checkout/merge and are often leftover artefacts from previous runs.
        tracked_dirty = "\n".join(
            line for line in status.output.splitlines() if line and not line.startswith("??")
        ).strip()
        if tracked_dirty:
            return self._blocked(run, "workspace_dirty", "Workspace is not clean"), None

        # Issue #615 — pre-run dependency validator
        if config.issue_number:
            try:
                from igris.core.dependency_checker import DependencyChecker
                _dep_checker = DependencyChecker(str(self.project_root))
                _dep_ok, _dep_unsat = _dep_checker.check(config.issue_number)
                run.add(
                    "dependency_check",
                    "satisfied" if _dep_ok else "blocked",
                    f"Issue #{config.issue_number}: deps {'all satisfied' if _dep_ok else f'unsatisfied: {_dep_unsat}'}",
                    issue_number=config.issue_number,
                    unsatisfied=_dep_unsat,
                )
                if not _dep_ok:
                    return self._blocked(
                        run,
                        "dependency_not_satisfied",
                        f"Issue #{config.issue_number} has unsatisfied dependencies: {_dep_unsat}. "
                        "Close or merge dependent issues first.",
                    ), None
            except Exception as _dep_exc:
                # Dep check is best-effort: log but never block on error
                run.add("dependency_check", "error", f"dep check error (non-fatal): {_dep_exc}")

        head = self.backend.git_log_head()
        run.add("git_head", "success" if head.success else "failure", _command_detail(head))
        head_sha = str((head.output or "").strip().split()[0] if head.success and (head.output or "").strip() else "")

        cache_hit = (
            _load_valid_baseline_cache(
                str(self.project_root), head_sha,
                force_revalidate=config.force_revalidate_baseline,
            )
            if head_sha else None
        )
        if config.force_revalidate_baseline:
            run.add("baseline_revalidation", "triggered",
                    "Baseline cache bypassed due to force_revalidate_baseline=True",
                    reason="force_revalidate")
        if cache_hit:
            run.add(
                "baseline_tests",
                "skipped",
                "Reusing cached baseline result for current HEAD.",
                head_sha=head_sha,
                checked_at=float(cache_hit.get("checked_at", 0.0) or 0.0),
                policy=str(cache_hit.get("policy", "strict")),
            )
        else:
            baseline_targets = _baseline_sanity_targets(str(self.project_root))
            run.add(
                "baseline_tests",
                "running",
                "Running baseline sanity pytest",
                timeout_seconds=config.test_timeout_seconds,
                exclude_slow=True,
                targets=baseline_targets,
            )
            baseline = self.backend.run_tests(
                baseline_targets or None,
                timeout=config.test_timeout_seconds,
                hard_cap=config.test_hard_cap_seconds,
                exclude_slow=True,
            )
            run.add("baseline_tests", "success" if baseline.success else "failure", _command_detail(baseline))
            cancelled = self._cancel_if_requested(run)
            if cancelled is not None:
                return cancelled, None
            if not baseline.success:
                run.add(
                    "baseline_diagnostics",
                    "running",
                    "Running first-failure pytest diagnostics",
                    timeout_seconds=min(config.test_timeout_seconds, 180),
                )
                diagnostics = self.backend.run_test_diagnostics(
                    timeout=min(config.test_timeout_seconds, 180),
                )
                run.add(
                    "baseline_diagnostics",
                    "success" if diagnostics.success else "failure",
                    _command_detail(diagnostics),
                )
                if _allow_unrelated_vastai_baseline_failures(config.goal, baseline, diagnostics):
                    run.add(
                        "baseline_gate",
                        "warning",
                        "Proceeding despite unrelated baseline failures in VastAI test suite",
                        policy="allow_unrelated_vastai_baseline_failures",
                    )
                    if head_sha:
                        try:
                            _save_baseline_cache(str(self.project_root), head_sha, policy="allow_unrelated_vastai")
                        except OSError:
                            pass
                elif _baseline_failure_is_transient(baseline, diagnostics):
                    return self._blocked(run, "infra_timeout", "Baseline tests timed out or transient infra error"), None
                else:
                    # Issue #626 — delta baseline: only block on NEW failures, not pre-existing ones.
                    _diag_text = "\n".join([
                        diagnostics.output or "", diagnostics.error or "",
                    ]) if diagnostics else ""
                    _branch_failures = _extract_failed_pytest_nodes(
                        "\n".join([baseline.output or "", baseline.error or "", _diag_text])
                    )
                    _main_sha = _get_main_sha(str(self.project_root))
                    _known = _load_known_baseline_failures(str(self.project_root), _main_sha) if _main_sha else None

                    if _known is not None:
                        # We have a record of pre-existing failures — compute delta.
                        _delta = _delta_baseline_failures(_branch_failures, _known)
                        if not _delta:
                            # All failures are pre-existing — proceed.
                            run.add(
                                "baseline_gate", "warning",
                                f"All {len(_branch_failures)} baseline failure(s) are pre-existing on "
                                f"main ({_main_sha[:8]}) — proceeding.",
                                policy="preexisting_failures",
                                preexisting_count=len(_branch_failures),
                                delta_count=0,
                            )
                            if head_sha:
                                try:
                                    _save_baseline_cache(
                                        str(self.project_root), head_sha,
                                        policy="preexisting_failures",
                                    )
                                except OSError:
                                    pass
                        else:
                            return self._blocked(
                                run, "pytest_failure",
                                f"Baseline tests introduced {len(_delta)} new failure(s) "
                                f"not present on main: {_delta[:5]}",
                            ), None
                    elif _main_sha and (head_sha == _main_sha or _diff_vs_main_is_empty(str(self.project_root), _main_sha)):
                        # Running on main itself (or branch identical to main) — record as pre-existing.
                        _save_known_baseline_failures(str(self.project_root), _main_sha, _branch_failures)
                        run.add(
                            "baseline_gate", "warning",
                            f"Recorded {len(_branch_failures)} pre-existing failure(s) for "
                            f"main {_main_sha[:8]} — proceeding without blocking.",
                            policy="recording_preexisting_failures",
                            known_count=len(_branch_failures),
                        )
                        if head_sha:
                            try:
                                _save_baseline_cache(
                                    str(self.project_root), head_sha,
                                    policy="preexisting_failures",
                                )
                            except OSError:
                                pass
                    else:
                        # Unknown failures on a diverged branch — block conservatively.
                        return self._blocked(run, "pytest_failure", "Baseline tests failed"), None
            elif head_sha:
                try:
                    _save_baseline_cache(str(self.project_root), head_sha, policy="strict")
                except OSError:
                    pass

        run.add("baseline_smoke", "running", "Running baseline smoke")
        smoke = self.backend.smoke(config.required_smoke_endpoints, restart_command)
        run.add("baseline_smoke", "success" if smoke.success else "failure", _command_detail(smoke))
        cancelled = self._cancel_if_requested(run)
        if cancelled is not None:
            return cancelled, None
        if not smoke.success:
            return self._blocked(run, "infrastructure_bug", "Baseline smoke failed"), None

        # Consult failure memory before any attempt — surface historical risk
        # so the operator can see early if similar goals have failed before.
        failure_risk = self._failure_memory.check(config.goal)
        run.add(
            "failure_memory",
            "checked",
            f"Failure memory check: risk={failure_risk.risk_level} "
            f"similar_failures={failure_risk.similar_count}",
            risk_level=failure_risk.risk_level,
            similar_count=failure_risk.similar_count,
            dominant_failure=failure_risk.dominant_failure,
            notes=failure_risk.notes,
        )

        # Pre-flight assignment routing: decide role/profile/strategy before any attempt.
        assignment_decision: Optional[Any] = None
        if _assignment_router_available:
            try:
                _outcomes_path = str(Path(self.project_root) / ".igris" / "assignment_outcomes.json")
                _router = AssignmentRouter(outcomes_path=_outcomes_path)
                # Merge prior capability_signals (from last failed run, passed by
                # the watchdog) with any signals already accumulated in this run.
                # This preserves no_diff_repair / reasoning_timeout counts across
                # watchdog cycles so the router can escalate to hard_debugging
                # (→ gpu_reasoning → VastAI) after repeated cross-run failures.
                _merged_signals: Dict[str, int] = dict(config.prior_capability_signals)
                for _sig, _cnt in run.capability_signals.items():
                    _merged_signals[_sig] = _merged_signals.get(_sig, 0) + _cnt
                _req = AssignmentRequest(
                    goal_text=config.goal,
                    risk_level="medium",
                    failure_class="",
                    capability_signals=_merged_signals,
                    prior_attempts=config.prior_attempts,
                    local_model_available=True,
                    budget_remaining_usd=float(config.max_api_budget_usd) or 10.0,
                    required_tests=list(config.targeted_tests),
                    is_repair=False,
                    outcomes_path=_outcomes_path,
                )
                assignment_decision = _router.decide(_req)
                forced_planner_profile = str(
                    os.getenv("IGRIS_ROLE_PLANNER_PROFILE", "mini_execution")
                ).strip() or "mini_execution"
                if (
                    assignment_decision is not None
                    and str(getattr(assignment_decision, "task_type", "")) == "memory_system"
                    and forced_planner_profile
                ):
                    prev_profile = str(getattr(assignment_decision, "preferred_profile", "") or "")
                    if prev_profile != forced_planner_profile:
                        assignment_decision.preferred_profile = forced_planner_profile
                        run.add(
                            "assignment_routing_override",
                            "success",
                            f"Planner profile override applied for initial rank path: "
                            f"{prev_profile or 'unset'} -> {forced_planner_profile}",
                            task_type=str(getattr(assignment_decision, "task_type", "")),
                            previous_profile=prev_profile,
                            forced_profile=forced_planner_profile,
                        )
                run.add(
                    "assignment_routing",
                    "success",
                    (
                        f"role={assignment_decision.agent_role} "
                        f"type={assignment_decision.task_type} "
                        f"profile={assignment_decision.preferred_profile} "
                        f"strategy={assignment_decision.execution_strategy} "
                        f"p={assignment_decision.estimated_success_probability:.2f} "
                        f"history={assignment_decision.history_matches}"
                    ),
                    **assignment_decision.to_dict(),
                )
            except Exception as _exc:
                run.add("assignment_routing", "skipped", f"AssignmentRouter error: {_exc}")

        mission_plan = self._build_mission_plan(config)
        stage_statuses = self._init_stage_statuses(mission_plan)
        run.add(
            "mission_plan",
            "success",
            "Mission execution strategy planned.",
            mode=mission_plan.mode,
            stage_ids=[stage.stage_id for stage in mission_plan.stages],
        )

        force_preemptive_decomposition = (
            str(os.getenv("IGRIS_FORCE_PREEMPTIVE_DECOMPOSITION", "true")).strip().lower() != "false"
        )
        if (
            force_preemptive_decomposition
            and self._goal_needs_preflight_decomposition(config.goal)
            and config.allow_auto_subissues
            and not config.dry_run
            and config.autochain_depth <= SelfRepairSupervisor._MAX_AUTOCHAIN_DEPTH
        ):
            run.add(
                "mission_planning",
                "decomposition_required",
                "Pre-emptive decomposition for large mission (policy shortcut) before long reasoning loops.",
            )
            decomposition = self._ask_igris_decompose(run, config)
            return self._blocked_decomposition_required(
                run,
                "preemptive_large_mission",
                "Large mission routed to decomposition-first execution policy.",
                decomposition,
                config=config,
                mission_plan=mission_plan,
                stage_statuses=stage_statuses,
            ), None

        # Pre-flight planning: read-only scope analysis before first attempt.
        # If the planning pass recommends decomposition, block proactively rather
        # than discovering the same thing after 3 failed repair cycles.
        if config.enable_mission_planning or self._goal_needs_preflight_decomposition(config.goal):
            scope = self._plan_mission(run, config)
            if scope and scope.get("decomposition_recommended"):
                run.add(
                    "mission_planning",
                    "decomposition_required",
                    f"Pre-flight planning recommends decomposition before any attempt: "
                    f"{scope.get('decomposition_reason', 'mission too large for single attempt')}",
                )
                decomposition = self._ask_igris_decompose(run, config)
                return self._blocked_decomposition_required(
                    run,
                    "pre_flight_planning",
                    (
                        f"Pre-flight planning detected scope too large for single attempt: "
                        f"{scope.get('decomposition_reason', 'see mission_scope in report')}"
                    ),
                    decomposition,
                    config=config,
                    mission_plan=mission_plan,
                    stage_statuses=stage_statuses,
                ), None

        return run, {
            "mission_plan": mission_plan,
            "stage_statuses": stage_statuses,
            "assignment_decision": assignment_decision,
            "restart_command": restart_command,
        }

    def _maybe_autoselect_next_roadmap(
        self,
        run: "SupervisorRun",
        config: RankSupervisorConfig,
    ) -> None:
        """Select and persist the next roadmap target after a completed run."""
        if not config.allow_roadmap_autoselect:
            return
        _next = self._select_next_roadmap_issue(config)
        if not _next:
            return

        # Issue #616 — skip candidates whose dependencies are unsatisfied
        try:
            from igris.core.dependency_checker import DependencyChecker
            _dep_checker = DependencyChecker(str(self.project_root))
            _dep_ok, _dep_unsat = _dep_checker.check(_next["number"])
            if not _dep_ok:
                run.add(
                    "watchdog_dependency_skip",
                    "skipped",
                    f"Roadmap candidate #{_next['number']} skipped: unsatisfied deps {_dep_unsat}",
                    issue_number=_next["number"],
                    unsatisfied_deps=_dep_unsat,
                )
                return
        except Exception as _dep_exc:
            # Best-effort — never block roadmap autoselect on dep check error
            run.add("watchdog_dependency_skip", "error",
                    f"dep check error (non-fatal): {_dep_exc}", issue_number=_next["number"])

        run.add(
            "roadmap_next_target",
            "selected",
            f"Next roadmap target: #{_next['number']} — {_next.get('title', '')}",
            issue_number=_next["number"],
            issue_title=_next.get("title", ""),
        )
        try:
            _hint_path = Path(self.project_root) / ".igris" / "next_roadmap_target.json"
            _hint_path.parent.mkdir(parents=True, exist_ok=True)
            _hint_path.write_text(
                json.dumps(
                    {
                        "issue_number": _next["number"],
                        "issue_title": _next.get("title", ""),
                        "selected_at": time.time(),
                        "selected_by_run": run.run_id,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as _e:
            run.add("roadmap_next_target", "write_failed", str(_e))


    def _run_rank_loop(
        self,
        run: SupervisorRun,
        config: RankSupervisorConfig,
        *,
        mission_plan: MissionPlan,
        stage_statuses: Dict[str, Dict[str, Any]],
        assignment_decision: Optional[Any],
        restart_command: str,
    ) -> SupervisorRun:
        """Phase 2: rank attempt loop and finalization."""
        repair_cycles = 0
        attempt = 1
        attempt_limit = config.max_rank_attempts
        final_validation_extension_used = False
        # Issue #715 — write-first / execution-effectiveness telemetry
        _no_diff_count: int = 0
        _time_to_first_diff_s: Optional[float] = None
        _attempt_start_time: float = time.time()
        _attempt_outcomes: List[str] = []
        _decompose_count: int = 0
        while attempt <= attempt_limit:
            cancelled = self._cancel_if_requested(run, mission_plan=mission_plan, stage_statuses=stage_statuses)
            if cancelled is not None:
                return cancelled
            branch = f"rank-{config.rank_id.lower()}-{int(time.time())}-{attempt}"
            run.branch = branch
            # Always start rank branch from latest main so every run has all committed fixes.
            _pre_checkout = self.backend.checkout_main()
            if not _pre_checkout.success:
                run.add("rank_branch_pre_checkout", "warning", f"Could not checkout main before branch creation: {_command_detail(_pre_checkout)}")
            branch_result = self.backend.create_branch(branch)
            run.add("rank_branch", "success" if branch_result.success else "failure", _command_detail(branch_result), branch=branch)
            cancelled = self._cancel_if_requested(run, mission_plan=mission_plan, stage_statuses=stage_statuses)
            if cancelled is not None:
                return cancelled
            if not branch_result.success:
                return self._blocked(run, "infrastructure_bug", "Could not create rank branch")

            stage_failure = ""
            if mission_plan.mode == "staged":
                reasoning, stage_failure, runtime_refresh_required = self._execute_staged_reasoning(
                    run,
                    config,
                    mission_plan,
                    stage_statuses,
                )
                reasoning_status = str(reasoning.get("status", ""))
                stop_reason = str(reasoning.get("stop_reason", ""))
                modified_files = list(reasoning.get("files_modified") or [])
            else:
                self._set_stage_status(
                    run,
                    stage_statuses,
                    "single_stage_execution",
                    "running",
                    "Running supervised rank reasoning as single stage.",
                )
                _routed_profile = (
                    assignment_decision.preferred_profile
                    if assignment_decision is not None
                    else None
                )
                _routed_task_type = (
                    assignment_decision.task_type
                    if assignment_decision is not None
                    else "code_reasoning"
                )
                max_reasoning_steps = max(40, int(os.getenv("IGRIS_RANK_MAX_STEPS", "120")))
                reasoning_timeout = config.reasoning_timeout_seconds
                # Profile-aware timeout adjustment:
                # Strong cloud models (DeepSeek V4 Pro, GPT-4o) take ~40-60s per step.
                # At 900s limit → only ~18 steps → never enough for full implementation.
                # Boost timeout for strong profiles; cap for local profiles.
                _STRONG_PROFILES = {"strong_execution", "strong_cloud_reasoning", "gpu_reasoning"}
                _LOCAL_PROFILES_SET = {"local_light", "local_coder", "mini_execution"}
                _profile = (_routed_profile or "")
                if _profile in _STRONG_PROFILES:
                    # Strong models need more time: env var or 2.5× the base timeout.
                    reasoning_timeout = int(os.getenv(
                        "IGRIS_STRONG_REASONING_TIMEOUT_SECONDS",
                        str(max(reasoning_timeout * 3, 2400)),
                    ))
                elif self._goal_needs_preflight_decomposition(config.goal) and _profile in _LOCAL_PROFILES_SET:
                    # Cap local-profile timeout on large missions — phi4-mini spins without progress.
                    reasoning_timeout = min(
                        reasoning_timeout,
                        int(os.getenv("IGRIS_LARGE_MISSION_REASONING_TIMEOUT", "240")),
                    )
                # Log the actual adjusted timeout — event was previously logged before
                # the profile-aware adjustment, showing 900s even when strong models
                # would use 2700s. Now logged AFTER adjustment for accurate audit trail.
                run.add(
                    "rank_reasoning",
                    "running",
                    "Running supervised rank reasoning",
                    timeout_seconds=reasoning_timeout,
                )
                reasoning = self.backend.run_reasoning(
                    config.goal,
                    max_steps=max_reasoning_steps,
                    initial_context=self._rank_initial_context(config, run=run),
                    timeout=reasoning_timeout,
                    task_type=_routed_task_type,
                    preferred_profile=_routed_profile,
                )
                reasoning_status = str(reasoning.get("status", ""))
                stop_reason = str(reasoning.get("stop_reason", ""))
                modified_files = list(reasoning.get("files_modified") or [])
                runtime_refresh_required = any(
                    str(path).startswith("igris/")
                    for path in modified_files
                )
                run.add(
                    "rank_reasoning",
                    reasoning_status,
                    reasoning.get("final_summary", ""),
                    loop_id=reasoning.get("loop_id", ""),
                    stop_reason=stop_reason,
                    files_modified=modified_files,
                    steps_completed=reasoning.get("steps_completed", 0),
                    orchestrator_used=reasoning.get("orchestrator_used", False),
                    reasoning_execution_provider=reasoning.get("reasoning_execution_provider", ""),
                    reasoning_execution_model=reasoning.get("reasoning_execution_model", ""),
                    reasoning_execution_profile=reasoning.get("reasoning_execution_profile", ""),
                    execution_provider=reasoning.get("reasoning_execution_provider", ""),
                    execution_model=reasoning.get("reasoning_execution_model", ""),
                    local_model_available=reasoning.get("local_model_available", False),
                )
                if reasoning_status == "finished":
                    self._set_stage_status(
                        run,
                        stage_statuses,
                        "single_stage_execution",
                        "success",
                        "Single-stage reasoning completed.",
                    )
                else:
                    self._set_stage_status(
                        run,
                        stage_statuses,
                        "single_stage_execution",
                        "failure",
                        f"Single-stage reasoning ended with {reasoning_status}/{stop_reason}.",
                    )
                    # Short-circuit the full_pytest validation when the reasoning
                    # loop itself signals "no_diff_repair" AND produced no files.
                    # In that case running 3000+ tests on an unchanged tree is pure
                    # waste — the staged path already does this inside
                    # _execute_staged_reasoning; we mirror the behaviour here.
                    # For other stop reasons (reasoning_timeout, max_steps, blocked…)
                    # we leave stage_failure empty so the normal repair → decomposition
                    # escalation path is preserved.
                    if stop_reason == "no_diff_repair" and not modified_files:
                        stage_failure = "reasoning_loop_blocked"
                        # Record capability signal immediately so the decomposition
                        # decision threshold is still updated even though the normal
                        # classify_failure path (inside `if not failure:`) is skipped.
                        self._record_capability_signal(run, "no_diff_repair")

            ui_visibility_required = self._goal_requires_ui_visibility(config.goal)
            ui_card_contract_goal = self._goal_targets_rank_ui_card(config.goal)
            ui_visibility_changed = self._has_ui_visibility_change(modified_files)
            run.add(
                "rank_reasoning",
                reasoning_status,
                reasoning.get("final_summary", ""),
                loop_id=reasoning.get("loop_id", ""),
                stop_reason=stop_reason,
                files_modified=modified_files,
                steps_completed=reasoning.get("steps_completed", 0),
                orchestrator_used=reasoning.get("orchestrator_used", False),
                reasoning_execution_provider=reasoning.get("reasoning_execution_provider", ""),
                reasoning_execution_model=reasoning.get("reasoning_execution_model", ""),
                reasoning_execution_profile=reasoning.get("reasoning_execution_profile", ""),
                local_model_available=reasoning.get("local_model_available", False),
                ui_visibility_required=ui_visibility_required,
                ui_visibility_changed=ui_visibility_changed,
                mission_orchestration_mode=mission_plan.mode,
            )
            if (
                stage_failure == "reasoning_loop_blocked"
                and stop_reason == "no_diff_repair"
                and not modified_files
            ):
                triggering_signal = self._detect_capability_limit(run)
                if triggering_signal:
                    return self._handle_capability_limit(
                        run, triggering_signal, config, mission_plan, stage_statuses,
                        cleanup_workspace=True,
                    )
            cancelled = self._cancel_if_requested(run, mission_plan=mission_plan, stage_statuses=stage_statuses)
            if cancelled is not None:
                return cancelled

            if stage_failure == "reasoning_loop_blocked" and stop_reason == "no_diff_repair" and not modified_files:
                diff_stat = CommandResult(True, "")
                diff = CommandResult(True, "")
                run.add("diff_stat", "skipped", "Skipped diff collection after no_diff_repair with no modified files.")
            else:
                diff_stat = self.backend.git_diff_stat()
                diff = self.backend.git_diff()
                run.add("diff_stat", "success" if diff_stat.success else "failure", _command_detail(diff_stat))
            # Issue #715 — track whether this attempt produced any file changes.
            _has_diff_this_attempt: bool = bool(diff_stat.output.strip())
            if _has_diff_this_attempt and _time_to_first_diff_s is None:
                _time_to_first_diff_s = round(time.time() - _attempt_start_time, 1)
            if not _has_diff_this_attempt:
                _no_diff_count += 1
            # Persist the full diff to disk immediately so _complete_rank can
            # recover if the working tree is unexpectedly reverted before the
            # commit (e.g. watchdog cleanup racing during branch transitions).
            if diff_stat.output.strip() and diff.output.strip():
                try:
                    _patch_path = Path(self.project_root) / ".igris" / "rank_pending.patch"
                    _patch_path.parent.mkdir(parents=True, exist_ok=True)
                    _patch_path.write_text(diff.output, encoding="utf-8")
                    run.add("diff_patch_saved", "success", str(_patch_path))
                except Exception as _pe:
                    run.add("diff_patch_saved", "failure", str(_pe))
            if (
                ui_visibility_required
                and not ui_visibility_changed
                and _has_ui_surface_change(diff.output)
            ):
                ui_visibility_changed = True
                run.add(
                    "ui_visibility",
                    "success",
                    "UI visibility inferred from validated diff paths",
                    inferred_from_diff=True,
                )
                if mission_plan.mode == "staged":
                    self._track_non_blocking_behavior(
                        run,
                        stage_statuses,
                        "ui_dashboard_change",
                        "ui_visibility_inferred_from_diff",
                        "UI visibility metadata was inferred from diff paths.",
                    )
            ui_contract_locked = (
                ui_card_contract_goal
                and ui_visibility_required
                and self._rank_ui_card_contract_satisfied()
                and self._rank_ui_visibility_signal_present()
            )
            if (
                ui_contract_locked
                and _touches_rank_ui_contract_files(diff.output)
                and not _has_ui_surface_change(diff.output)
            ):
                restore = self.backend.restore_dangerous_diff()
                run.add(
                    "rank_restore",
                    "success" if restore.success else "failure",
                    "Protected UI contract files were modified despite already satisfied objective.",
                )
                if not restore.success:
                    return self._blocked(
                        run,
                        "infrastructure_bug",
                        "Could not restore unsupported edits to satisfied UI contract files",
                    )
                return self._complete_noop(
                    run,
                    completion_mode="already_satisfied",
                    runtime_refresh_required=runtime_refresh_required,
                    detail=(
                        "Restored unsupported edits to satisfied UI contract files; "
                        "completed as verified no-op."
                    ),
                    post_merge_smoke=False,
                    mission_plan=mission_plan,
                    stage_statuses=stage_statuses,
                )
            targeted = CommandResult(True, "Targeted tests skipped")
            full = CommandResult(True, "Full pytest skipped")
            final_smoke = CommandResult(True, "Final smoke skipped")
            failure = stage_failure or ""
            if not failure:
                # Issue #731 — gate is only active when enable_semantic_gate=True.
                # Disabling the semantic gate (e.g. in tests) skips both the
                # pre-apply quality gate and the post-reasoning acceptance gate.
                should_gate = (
                    config.enable_semantic_gate
                    and (reasoning_status == "finished" or stop_reason == "finish")
                    and (bool(modified_files) or bool((diff.output or "").strip()))
                )
                if should_gate:
                    run.add("quality_gate_preapply", "running", "Running pre-apply quality gate")
                    gate_ok, gate_reasons = self._preapply_quality_gate(config.goal, diff.output, modified_files)
                    if gate_ok:
                        run.add("quality_gate_preapply", "success", "Pre-apply quality gate passed")
                    else:
                        failure = "semantic_incomplete"
                        run.add(
                            "quality_gate_preapply",
                            "failure",
                            "Pre-apply quality gate failed",
                            reasons=gate_reasons,
                            error_code=_failure_error_code("semantic_incomplete"),
                        )
                else:
                    run.add(
                        "quality_gate_preapply",
                        "skipped",
                        "Skipped pre-apply quality gate (no candidate patch from finished reasoning).",
                    )

            if failure:
                run.add(
                    "validation_short_circuit",
                    "running",
                    f"Skipping attempt validation because required stage failed: {failure}",
                )
                run.add("targeted_tests", "skipped", "Skipped because a required stage failed before validation.")
                run.add("full_pytest", "skipped", "Skipped because a required stage failed before validation.")
                run.add("smoke", "skipped", "Skipped because a required stage failed before validation.")
                if "targeted_tests" in stage_statuses:
                    self._set_stage_status(
                        run,
                        stage_statuses,
                        "targeted_tests",
                        "skipped",
                        "Skipped because a required implementation stage failed.",
                    )
                if "full_pytest" in stage_statuses:
                    self._set_stage_status(
                        run,
                        stage_statuses,
                        "full_pytest",
                        "skipped",
                        "Skipped because a required implementation stage failed.",
                    )
            elif _has_immediately_dangerous_diff(diff.output):
                # Dangerous tokens (.env, .venv) or structural deletions (def create_app,
                # class) — restore without running tests as these would definitely break
                # the app regardless of what else the model added.
                run.add("safety", "blocked", "Immediately dangerous diff detected (tokens or structural deletion)")
                self.backend.restore_dangerous_diff()
                failure = "destructive_diff"
            else:
                if config.targeted_tests:
                    self._set_stage_status(
                        run,
                        stage_statuses,
                        "targeted_tests",
                        "running",
                        "Running targeted pytest validation.",
                    )
                    run.add(
                        "targeted_tests",
                        "running",
                        "Running targeted pytest",
                        targets=" ".join(config.targeted_tests),
                        timeout_seconds=config.test_timeout_seconds,
                    )
                    targeted = self.backend.run_tests(
                        config.targeted_tests,
                        timeout=config.test_timeout_seconds,
                        hard_cap=config.test_hard_cap_seconds,
                    )
                else:
                    targeted = CommandResult(True, "No targeted tests configured")
                    if mission_plan.mode == "staged" and "targeted_tests" in stage_statuses:
                        self._set_stage_status(
                            run,
                            stage_statuses,
                            "targeted_tests",
                            "skipped",
                            "No targeted tests configured for this mission.",
                            no_op=True,
                        )
                self._set_stage_status(
                    run,
                    stage_statuses,
                    "full_pytest",
                    "running",
                    "Running full pytest validation.",
                )
                full_targets: Optional[List[str]] = None
                full_validation_mode = "full"
                # Structural policy: during local supervised execution (no PR/merge),
                # avoid paying full-suite cost on every attempt. Use a stable
                # validation suite focused on service health + rank contract.
                if not config.allow_github_pr and not config.allow_merge_if_green:
                    full_targets = _baseline_sanity_targets(str(self.project_root))
                    full_validation_mode = "sanity"
                run.add(
                    "full_pytest",
                    "running",
                    "Running full pytest (-m 'not slow')" if full_validation_mode == "full" else "Running validation sanity suite",
                    timeout_seconds=config.test_timeout_seconds,
                    exclude_slow=True,
                    targets=full_targets or [],
                    validation_mode=full_validation_mode,
                )
                full = self.backend.run_tests(
                    full_targets or None,
                    timeout=config.test_timeout_seconds,
                    hard_cap=config.test_hard_cap_seconds,
                    exclude_slow=True,
                )
                run.add("smoke", "running", "Running final smoke")
                final_smoke = self.backend.smoke(config.required_smoke_endpoints, restart_command)
                run.add("targeted_tests", "success" if targeted.success else "failure", _command_detail(targeted))
                run.add("full_pytest", "success" if full.success else "failure", _command_detail(full))
                run.add("smoke", "success" if final_smoke.success else "failure", _command_detail(final_smoke))
                if config.targeted_tests:
                    self._set_stage_status(
                        run,
                        stage_statuses,
                        "targeted_tests",
                        "success" if targeted.success else "failure",
                        "Targeted tests passed." if targeted.success else "Targeted tests failed.",
                    )
                self._set_stage_status(
                    run,
                    stage_statuses,
                    "full_pytest",
                    "success" if full.success else "failure",
                    "Full pytest passed." if full.success else "Full pytest failed.",
                )
            already_satisfied_noop = (
                not failure
                and self._ui_noop_completion_eligible(
                    config,
                    diff_stat,
                    targeted,
                    full,
                    final_smoke,
                )
            )
            if already_satisfied_noop:
                return self._complete_noop(
                    run,
                    completion_mode="already_satisfied",
                    runtime_refresh_required=runtime_refresh_required,
                    detail="Rank objective already satisfied; completed as verified no-op.",
                    post_merge_smoke=final_smoke.success,
                    mission_plan=mission_plan,
                    stage_statuses=stage_statuses,
                )
            required_stages_complete = self._required_stages_green(
                stage_statuses,
                exclude_stage_ids={"pr_ci_merge", "post_merge_runtime", "final_report"},
            )
            staged_noop_completion = (
                mission_plan.mode == "staged"
                and not failure
                and not diff_stat.output.strip()
                and targeted.success
                and full.success
                and final_smoke.success
                and required_stages_complete
            )
            if staged_noop_completion:
                return self._complete_noop(
                    run,
                    completion_mode="already_satisfied",
                    runtime_refresh_required=runtime_refresh_required,
                    detail="All required staged mission phases were already satisfied; completed as verified no-op.",
                    post_merge_smoke=final_smoke.success,
                    mission_plan=mission_plan,
                    stage_statuses=stage_statuses,
            )
            rank_passed = self._rank_passed(reasoning, diff_stat, targeted, full, final_smoke)
            if not failure:
                if rank_passed:
                    if ui_visibility_required and not ui_visibility_changed:
                        failure = "missing_ui_visibility"
                else:
                    failure = classify_failure(reasoning, diff.output, targeted, full, final_smoke)
                    # Record reasoning_timeout signal when the model timed out, hit
                    # budget, or explicitly refused the task — all indicate capability
                    # limit that should trigger decomposition after N occurrences.
                    if failure == "reasoning_loop_blocked" and stop_reason in {
                        "reasoning_timeout", "budget_exceeded", "blocked",
                    }:
                        self._record_capability_signal(run, "reasoning_timeout")
                    if failure == "reasoning_loop_blocked" and stop_reason == "no_diff_repair":
                        self._record_capability_signal(run, "no_diff_repair")
            # Record pytest_hang when the full test subprocess was killed for
            # producing no output (idle timeout) — repeated hangs indicate the
            # model's change consistently breaks the test suite in a way it
            # cannot self-repair.
            if not full.success and "Command killed:" in (full.error or ""):
                self._record_capability_signal(run, "pytest_hang")
            triggering_signal_early = self._should_fast_track_capability_limit(
                run,
                failure,
            )
            if triggering_signal_early:
                run.add(
                    "capability_ceiling",
                    "detected",
                    (
                        "Capability limit reached during active attempt; "
                        "fast-tracking decomposition before exhausting repair budget."
                    ),
                    triggering_signal=triggering_signal_early,
                    capability_signals=dict(run.capability_signals),
                    failure_class=failure,
                )
                return self._handle_capability_limit(
                    run,
                    triggering_signal_early,
                    config,
                    mission_plan,
                    stage_statuses,
                    cleanup_workspace=True,
                )
            if not failure and mission_plan.mode == "staged" and not required_stages_complete:
                failure = "reasoning_loop_blocked"
                run.add(
                    "stage_gate",
                    "blocked",
                    "Required mission stage not completed; refusing completed status.",
                )
            reasoning_text = "\n".join(
                str(reasoning.get(key, ""))
                for key in ("final_summary", "error", "stop_reason")
            )
            if failure == "infrastructure_bug" and _is_llm_provider_unavailable(reasoning_text):
                # LLM unavailability is a structural capability wall — no repair cycle
                # will succeed without a working model.  Decompose immediately so the
                # auto-chain can hand the task to a sub-mission that may reach a cloud
                # provider, rather than blocking the entire run indefinitely.
                self._record_capability_signal(run, "reasoning_timeout")
                decomposition = self._ask_igris_decompose(run, config)
                return self._blocked_decomposition_required(
                    run,
                    "reasoning_timeout",
                    "LLM unavailable — decomposing to sub-mission for capable model",
                    decomposition,
                    config=config,
                    mission_plan=mission_plan,
                    stage_statuses=stage_statuses,
                    cleanup_workspace=True,
                )

            if not failure and rank_passed:
                completion_mode = "direct"
                if reasoning_status != "finished" or stop_reason != "finish":
                    completion_mode = "verified_diff"
                    run.add(
                        "completion",
                        "degraded",
                        "Rank completed by verification after reasoning did not finish cleanly",
                        mode=completion_mode,
                        stop_reason=stop_reason,
                    )
                    if mission_plan.mode == "staged":
                        stage_id = "single_stage_execution"
                        self._track_non_blocking_behavior(
                            run,
                            stage_statuses,
                            stage_id,
                            "verified_diff_completion",
                            "Completed by validated diff despite non-clean reasoning stop.",
                        )

                # --- Semantic acceptance gate ---
                # Verify the diff is a genuine implementation, not a stub.
                if config.enable_semantic_gate:
                    # When targeted tests pass, include their file paths so the gate
                    # knows test coverage exists even if a repair cycle only touched
                    # the implementation file (tests were already written earlier and
                    # restored workspace left them absent from files_modified).
                    gate_files = list(modified_files)
                    if targeted.success and config.targeted_tests:
                        for tf in config.targeted_tests:
                            if tf not in gate_files:
                                gate_files.append(tf)
                    acceptance = check_acceptance_evidence(
                        config.goal,
                        diff.output,
                        gate_files,
                    )
                    # Store on the run object so it survives any subsequent run.report overwrites.
                    run.acceptance_evidence = {
                        "passed": acceptance.passed,
                        "found_evidence": acceptance.found_evidence,
                        "missing_evidence": acceptance.missing_evidence,
                        "required_endpoints": acceptance.required_endpoints,
                    }
                    if not acceptance.passed:
                        run.add(
                            "semantic_check",
                            "incomplete",
                            "Mission acceptance gate failed: implementation appears to be a stub. "
                            + "; ".join(acceptance.missing_evidence),
                            missing_evidence=acceptance.missing_evidence,
                            found_evidence=acceptance.found_evidence,
                            required_endpoints=acceptance.required_endpoints,
                        )
                        failure = "semantic_incomplete"
                        rank_passed = False
                    else:
                        run.add(
                            "semantic_check",
                            "passed",
                            "Mission acceptance gate passed.",
                            found_evidence=acceptance.found_evidence,
                            required_endpoints=acceptance.required_endpoints,
                        )

                if rank_passed:
                    self._persist_assignment_outcome(run, self.project_root, assignment_decision)
                    _done = self._complete_rank(
                        run,
                        config,
                        branch,
                        completion_mode=completion_mode,
                        runtime_refresh_required=runtime_refresh_required,
                        mission_plan=mission_plan,
                        stage_statuses=stage_statuses,
                    )
                    self._maybe_autoselect_next_roadmap(run, config)
                    return _done

            run.failure_class = failure
            run.add("failure", "classified", failure, error_code=_failure_error_code(failure))
            # Issue #715 — record per-attempt outcome for telemetry.
            _attempt_outcomes.append(
                "no_diff" if not _has_diff_this_attempt else (failure or "failed")
            )
            if failure not in REPAIRABLE_FAILURES or repair_cycles >= config.max_repair_cycles:
                triggering_signal = self._detect_capability_limit(run)
                if triggering_signal:
                    return self._handle_capability_limit(
                        run, triggering_signal, config, mission_plan, stage_statuses,
                        cleanup_workspace=True,
                    )
                # Issue #715 — no_diff_terminal_report: surface when attempts consistently
                # produced no file changes so callers know the agent is structurally stuck.
                if _no_diff_count >= 1:
                    run.add(
                        "no_diff_terminal_report",
                        "blocked",
                        f"Attempt(s) produced no diff (no_diff_count={_no_diff_count}). "
                        "Stopping to avoid further wasted cycles.",
                        no_diff_count=_no_diff_count,
                        attempt=attempt,
                    )
                _telemetry = self._build_telemetry_fragment(
                    _time_to_first_diff_s, _no_diff_count, _decompose_count,
                    _attempt_outcomes, attempt,
                )
                _done = self._blocked(
                    run,
                    failure,
                    "Rank failed and repair budget is exhausted or not repairable",
                    mission_plan=mission_plan,
                    stage_statuses=stage_statuses,
                    cleanup_workspace=True,
                )
                _done.report.update(_telemetry)
                return _done

            repair_cycles += 1
            run.repair_cycles_used = repair_cycles
            if not self._repair_cycle(
                run,
                config,
                failure,
                repair_cycles,
                preserve_validated_progress=mission_plan.mode == "staged",
                stage_statuses=stage_statuses if mission_plan.mode == "staged" else None,
            ):
                triggering_signal = self._detect_capability_limit(run)
                if triggering_signal:
                    return self._handle_capability_limit(
                        run, triggering_signal, config, mission_plan, stage_statuses,
                        cleanup_workspace=True,
                    )
                return self._blocked(
                    run,
                    failure,
                    "Repair cycle failed validation",
                    mission_plan=mission_plan,
                    stage_statuses=stage_statuses,
                    cleanup_workspace=True,
                )
            if attempt == attempt_limit:
                if repair_cycles < config.max_repair_cycles:
                    attempt_limit += 1
                    run.add(
                        "rank_attempt_extension",
                        "running",
                        "Extending rank attempts after successful repair on final configured attempt.",
                        attempt_limit=attempt_limit,
                        repair_cycles_used=repair_cycles,
                    )
                elif not final_validation_extension_used:
                    attempt_limit += 1
                    final_validation_extension_used = True
                    run.add(
                        "rank_attempt_extension",
                        "running",
                        "Granting one final validation attempt after successful repair at repair budget limit.",
                        attempt_limit=attempt_limit,
                        repair_cycles_used=repair_cycles,
                        final_validation_only=True,
                    )
            attempt += 1

        triggering_signal = self._detect_capability_limit(run)
        if triggering_signal:
            self._persist_assignment_outcome(run, self.project_root, assignment_decision)
            return self._handle_capability_limit(
                run, triggering_signal, config, mission_plan, stage_statuses,
                cleanup_workspace=True,
            )
        self._persist_assignment_outcome(run, self.project_root, assignment_decision)
        _done = self._blocked(
            run,
            run.failure_class or "max_rank_attempts",
            "Rank attempts exhausted",
            mission_plan=mission_plan,
            stage_statuses=stage_statuses,
            cleanup_workspace=True,
        )
        # Issue #715 — append execution-effectiveness telemetry to the final report.
        _done.report.update(self._build_telemetry_fragment(
            _time_to_first_diff_s, _no_diff_count, _decompose_count,
            _attempt_outcomes, max(attempt - 1, 1),
        ))
        self._maybe_autoselect_next_roadmap(run, config)
        return _done


    def _select_next_roadmap_issue(
        self, config: "RankSupervisorConfig"
    ) -> Optional[Dict[str, Any]]:
        """Query GitHub for next roadmap issue. Only for root runs (autochain_depth=0)."""
        if config.autochain_depth != 0:
            return None
        try:
            import subprocess as _sub
            result = _sub.run(
                ["gh", "issue", "list", "--label", "roadmap", "--state", "open",
                 "--json", "number,title,labels", "--limit", "200"],
                capture_output=True, text=True, cwd=self.project_root, timeout=30,
            )
            if result.returncode != 0:
                return None
            issues = json.loads(result.stdout or "[]")
        except Exception:
            return None

        EPIC_SKIP = ("epic", "phase", "milestone", "overview", "arch", "design")

        def _is_epic(issue: Dict[str, Any]) -> bool:
            title = (issue.get("title") or "").lower()
            labels = [l.get("name", "").lower() for l in (issue.get("labels") or [])]
            # Use word-boundary matching to avoid false positives from substrings:
            # e.g. "arch" must not match "hierarchy", "phase" must not match "phase-2bis" label.
            # We check labels by membership (exact element), not substring.
            _is_epic_title = any(
                re.search(r"\b" + k + r"\b", title) for k in EPIC_SKIP
            )
            return _is_epic_title or "epic" in labels

        def _priority(issue: Dict[str, Any]) -> tuple:
            labels = [l.get("name", "").lower() for l in (issue.get("labels") or [])]
            p = 99
            if any(x in labels for x in ("p1", "priority: high", "priority:high")):
                p = 1
            elif any(x in labels for x in ("p2", "priority: medium", "priority:medium")):
                p = 2
            return (p, issue.get("number", 9999))

        candidates = [i for i in issues if not _is_epic(i)]
        if not candidates:
            return None
        candidates.sort(key=_priority)
        return candidates[0]

    def _rank_passed(
        self,
        reasoning: Dict[str, Any],
        diff_stat: CommandResult,
        targeted: CommandResult,
        full: CommandResult,
        smoke: CommandResult,
    ) -> bool:
        has_diff = bool(diff_stat.output.strip())
        stop_reason = str(reasoning.get("stop_reason", ""))
        files_modified: List[str] = list(reasoning.get("files_modified") or [])
        # `git diff --stat` only shows tracked file changes. When the reasoning worker
        # creates NEW files (untracked), they won't appear in the diff even though real
        # work was done. Detect this by checking whether the reported modified paths
        # actually exist on disk — if they do, treat it as a valid diff.
        if not has_diff and files_modified:
            has_diff = any(
                (Path(self.project_root) / f).exists()
                for f in files_modified
            )
        delivered_changes = bool(files_modified) or (
            has_diff and stop_reason == "reasoning_timeout"
        )
        reasoning_finished = reasoning.get("status") == "finished"
        return (
            (reasoning_finished or delivered_changes)
            and delivered_changes
            and has_diff
            and targeted.success
            and full.success
            and smoke.success
        )

    def _ui_noop_completion_eligible(
        self,
        config: RankSupervisorConfig,
        diff_stat: CommandResult,
        targeted: CommandResult,
        full: CommandResult,
        smoke: CommandResult,
    ) -> bool:
        if not self._goal_requires_ui_visibility(config.goal):
            return False
        if not self._goal_targets_rank_ui_card(config.goal):
            return False
        if diff_stat.output.strip():
            return False
        if not (targeted.success and full.success and smoke.success):
            return False
        return self._rank_ui_card_contract_satisfied() and self._rank_ui_visibility_signal_present()

    def _rank_initial_context(
        self,
        config: RankSupervisorConfig,
        run: Optional["SupervisorRun"] = None,
    ) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "rank_test": config.rank_id,
            "project_root": self.project_root,
            "must_not_push_directly_to_main": True,
            "must_not_merge_if_tests_fail": True,
            "suppress_human_gate": True,
            "must_not_ask_user": True,
            "supervised": True,
            "expected_endpoint_file": "igris/web/server.py",
            "safe_edit_policy": (
                "For existing large files use insert_after, insert_before, "
                "replace_range or append_file. Never full-file write server.py."
            ),
            "fastapi_test_policy": (
                "API tests must import create_app from igris.web.server and use "
                "TestClient(create_app()). Do not import app from igris.web.server."
            ),
            "implementation_quality_policy": (
                "Write real implementation code, not stubs. "
                "Do NOT add '# Placeholder', '# TODO', '# FIXME', or 'pass' in the "
                "function body. Do NOT return empty dicts, empty lists, or empty strings "
                "as field values. The implementation will be rejected by the semantic "
                "acceptance gate if stub patterns are detected."
            ),
        }
        if self._goal_prefers_tool_first(config.goal):
            context["tool_first_policy"] = (
                "For broad analysis/mapping tasks, prefer deterministic tool-first work. "
                "Collect compact facts with ripgrep/python scripts, then reason on the summary "
                "instead of loading many files into LLM context."
            )
            context["tool_first_snapshot"] = self._build_tool_first_snapshot()
        if self._goal_requires_ui_visibility(config.goal):
            ui_card_contract_goal = self._goal_targets_rank_ui_card(config.goal)
            ui_contract_satisfied = ui_card_contract_goal and self._rank_ui_card_contract_satisfied()
            context["must_add_ui_visibility"] = True
            context["ui_visibility_policy"] = (
                "If the goal requires UI/dashboard visibility, modify a UI surface "
                "such as igris/web/templates/index.html, igris/web/static/js/app.js, "
                "or igris/web/static/css/style.css. Backend-only changes are not enough."
            )
            context["ui_contract_already_satisfied"] = ui_contract_satisfied
            if ui_contract_satisfied:
                context["ui_contract_policy"] = (
                    "The /api/rank/ui-card contract is already satisfied in "
                    "igris/web/server.py. Do not modify this route. Focus only on "
                    "minimal UI/dashboard visibility edits and related UI checks."
                )
            if ui_card_contract_goal:
                context["ui_test_policy"] = (
                    "UI tests must stay minimal and exact. Do not add placeholder routes, "
                    "commented example paths, or unrelated assertions. Test the exact "
                    "required endpoint plus the minimal UI/dashboard visibility signal. "
                    "For /api/rank/ui-card, only assert the contract keys app, rank, status, "
                    "and capability. Do not assert extra JSON keys such as data."
                )
            else:
                context["ui_test_policy"] = (
                    "UI/dashboard tests must stay minimal and exact for this mission. "
                    "Validate only the required endpoint contract and the requested "
                    "visibility signal. Do not add placeholder routes or unrelated assertions."
                )
        for target in config.targeted_tests:
            if target.startswith("tests/test_") and target.endswith(".py"):
                target_path = Path(self.project_root) / target
                if target_path.exists():
                    context["targeted_test_file_exists"] = target
                    context["targeted_test_policy"] = (
                        f"{target} already exists. Edit this file in place when needed. "
                        "Do not rediscover tests/ and do not recreate the file."
                    )
                else:
                    context["must_create_test_file"] = target
                    context["anti_loop_instruction"] = (
                        f"Do not repeat test discovery after tests/ is known. "
                        f"Create {target} directly."
                    )
                break

        # --- Inject MBOP Phase 1 intake fields (#1040) ---
        # These let the reasoning loop know the exact target file/module and acceptance
        # criteria from the start, eliminating blind find_files exploration.
        _intake = getattr(run, "mbop_intake", None) if run is not None else None
        if _intake is not None:
            try:
                if getattr(_intake, "what", ""):
                    context["mbop_what"] = str(_intake.what)[:300]
                if getattr(_intake, "where", ""):
                    context["mbop_where"] = str(_intake.where)[:300]
                if getattr(_intake, "why", ""):
                    context["mbop_why"] = str(_intake.why)[:300]
                if getattr(_intake, "acceptance_criteria", []):
                    context["mbop_acceptance_criteria"] = list(_intake.acceptance_criteria[:10])
                if getattr(_intake, "constraints", []):
                    context["mbop_constraints"] = list(_intake.constraints[:5])
                if getattr(_intake, "extraction_ok", False):
                    context["mbop_intake_ok"] = True
            except Exception:
                pass  # best-effort — never block the run

        # --- Inject MBOP Phase 10-11 prior-run lessons for the same issue (BUG2 fix) ---
        # Phases 10-11 fire after supervisor.run() returns, so they're not available
        # during the CURRENT run's repair cycle.  However, for SUBSEQUENT runs on the
        # same issue, these lessons prevent repeating the same mistakes.
        if config.issue_number:
            try:
                from igris.core.mbop_log import read_for_issue
                prior_events = read_for_issue(str(self.project_root), config.issue_number)
                # Extract most recent Phase 11 lessons and Phase 10 criteria_missing
                prior_lessons: list = []
                prior_criteria_missing: list = []
                for ev in reversed(prior_events):
                    if ev.get("phase") == "mbop_phase11_post_task_eval" and not prior_lessons:
                        extra = ev.get("extra", {}) or {}
                        lessons_raw = extra.get("lessons", [])
                        prior_lessons = [str(l) for l in lessons_raw if l][:5]
                for ev in reversed(prior_events):
                    if ev.get("phase") == "mbop_phase10_satisfaction_gate" and not prior_criteria_missing:
                        extra = ev.get("extra", {}) or {}
                        prior_criteria_missing = [str(c) for c in extra.get("criteria_missing", []) if c][:5]
                        break
                if prior_lessons:
                    context["mbop_prior_lessons"] = prior_lessons
                if prior_criteria_missing:
                    context["mbop_prior_criteria_missing"] = prior_criteria_missing
            except Exception:
                pass  # best-effort — never block the run

        return context

    @staticmethod
    def _goal_prefers_tool_first(goal: str) -> bool:
        text = (goal or "").lower()
        markers = ("analyze", "analysis", "compare", "mapping", "blueprint", "gap", "inventory", "logs", "roadmap")
        return any(m in text for m in markers)

    def _build_tool_first_snapshot(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {"project_root": str(self.project_root), "file_count": 0, "top_dirs": []}
        try:
            proc = subprocess.run(
                ["rg", "--files"],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=8,
                check=False,
            )
            files = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
            snapshot["file_count"] = len(files)
            counts: Dict[str, int] = {}
            for rel in files:
                top = rel.split("/", 1)[0]
                counts[top] = counts.get(top, 0) + 1
            snapshot["top_dirs"] = [
                {"name": name, "files": count}
                for name, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:8]
            ]
        except Exception:
            return snapshot
        return snapshot

    @staticmethod
    def _preapply_quality_gate(goal: str, diff_text: str, files_modified: List[str]) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        lowered_goal = (goal or "").lower()
        lowered_diff = (diff_text or "").lower()
        if any(tok in lowered_goal for tok in ("test", "pytest")) and not any("test" in str(path).lower() for path in files_modified):
            reasons.append("goal_mentions_tests_but_no_test_file_touched")
        if any(marker in lowered_diff for marker in ("# placeholder", "# todo", "# fixme", "\n+pass\n", "+    pass", "+        pass")):
            reasons.append("stub_pattern_detected_in_diff")
        return (len(reasons) == 0), reasons

    def _rank_ui_card_contract_satisfied(self) -> bool:
        server_path = Path(self.project_root) / "igris/web/server.py"
        if not server_path.exists():
            return False
        try:
            content = server_path.read_text(encoding="utf-8")
        except OSError:
            return False

        route_present = (
            "@app.get('/api/rank/ui-card')" in content
            or '@app.get("/api/rank/ui-card")' in content
        )
        if not route_present:
            return False

        def _has_pair(key: str, value: str) -> bool:
            return (
                f"'{key}': '{value}'" in content
                or f'"{key}": "{value}"' in content
            )

        return all(
            _has_pair(key, value)
            for key, value in (
                ("app", "IGRIS_GPT"),
                ("rank", "A++"),
                ("status", "ok"),
                ("capability", "ui-visible-supervised"),
            )
        )

    def _rank_ui_visibility_signal_present(self) -> bool:
        index_path = Path(self.project_root) / "igris/web/templates/index.html"
        if not index_path.exists():
            return False
        try:
            content = index_path.read_text(encoding="utf-8").lower()
        except OSError:
            return False
        return "rank-ui-card" in content and "ui-visible-supervised" in content

    @staticmethod
    def _goal_requires_ui_visibility(goal: str) -> bool:
        import re as _re
        lowered = goal.lower()
        # Use word-boundary matching to avoid false positives from substrings
        # (e.g. Italian "qui" contains "ui", "visible" matches "visibility" correctly).
        _UI_PATTERNS = _re.compile(
            r"\b(ui|dashboard|frontend)\b|visib",
            _re.IGNORECASE,
        )
        return bool(_UI_PATTERNS.search(lowered))

    @staticmethod
    def _goal_targets_rank_ui_card(goal: str) -> bool:
        lowered = goal.lower()
        if "/api/rank/ui-card" in lowered:
            return True
        if "ui-card" in lowered or "ui card" in lowered:
            return True
        return "rank card" in lowered and "ui" in lowered

    @staticmethod
    def _build_reasoning_loop_repair_prompt(
        stage_id: str,
        goal: str,
        previous_reasoning_output: str,
        repair_cycle: int,
    ) -> str:
        """
        Costruisce un repair goal progressivo per reasoning_loop_blocked.

        Ciclo 1: semplifica il task, output minimo, approccio incrementale.
        Ciclo 2+: suddividi nel componente più piccolo risolvibile, ignora ottimizzazioni.
        """
        _ = previous_reasoning_output
        if repair_cycle <= 1:
            return (
                f"{goal} "
                f"(REPAIR CYCLE {repair_cycle} — previous attempt on stage '{stage_id}' "
                f"timed out or exceeded reasoning budget. "
                f"Focus ONLY on the minimal change needed. "
                f"Do not optimize, refactor, or add features beyond what the goal strictly requires. "
                f"Use an incremental approach: implement the smallest complete unit first, verify it, then stop. "
                f"Prioritise writing code and tests over exploration. Keep edits minimal, do not push.)"
            )
        return (
            f"{goal} "
            f"(REPAIR CYCLE {repair_cycle} — previous attempts on stage '{stage_id}' "
            f"were repeatedly blocked. "
            f"Break the task down to its single smallest resolvable sub-component. "
            f"Ignore all non-critical aspects, optimisations, and edge cases. "
            f"Implement only what is strictly necessary to satisfy the goal, nothing more. "
            f"Do not push.)"
        )

    @staticmethod
    def _build_wrong_file_edit_repair_prompt(
        stage_id: str,
        goal: str,
        wrong_paths: List[str],
        allowed_families: List[str],
        repair_cycle: int,
    ) -> str:
        """
        Costruisce un repair goal per wrong_file_edit che:
        1. Elenca i file modificati fuori scope
        2. Elenca i file consentiti per lo stage
        3. Chiede di ripetere SOLO la modifica consentita
        4. Al ciclo 2+: aggiunge vincolo hard
        """
        wrong_list = "\n".join(f"  - {p}" for p in wrong_paths) if wrong_paths else "  - (unknown paths)"
        allowed_list = (
            "\n".join(f"  - {fam}" for fam in allowed_families)
            if allowed_families
            else "  - (mission-owned minimal scope)"
        )
        prompt = (
            f"{goal} "
            f"(REPAIR CYCLE {repair_cycle} — previous attempt on stage '{stage_id}' "
            f"modified files outside the allowed scope.\n"
            f"Files wrongly modified:\n{wrong_list}\n"
            f"Allowed file families:\n{allowed_list}\n"
            f"You MUST only modify files belonging to the allowed families listed above. "
            f"Repeat your edit but restrict ALL changes to the allowed files. "
            f"Do not touch any file outside the allowed families.)"
        )
        if repair_cycle >= 2:
            prompt += (
                " If you cannot complete the task within the allowed files, "
                "output ONLY the changes to allowed files and stop. "
                "Do not modify any file outside the allowed list under any circumstance."
            )
        return prompt

    @staticmethod
    def _has_ui_visibility_change(files_modified: List[str]) -> bool:
        ui_markers = (
            "igris/web/templates/",
            "igris/web/static/js/",
            "igris/web/static/css/",
            ".html",
            ".js",
            ".css",
        )
        for path in files_modified:
            if any(marker in path for marker in ui_markers):
                return True
        return False

    @staticmethod
    def _targeted_test_file(config: RankSupervisorConfig) -> str:
        for candidate in config.targeted_tests:
            if candidate.startswith("tests/test_") and candidate.endswith(".py"):
                return candidate
        return ""

    def _synthetic_missing_tests_diff(self, config: RankSupervisorConfig) -> str:
        target = self._targeted_test_file(config)
        if not target:
            return ""
        target_path = Path(self.project_root) / target
        if not target_path.exists():
            return ""
        try:
            content = target_path.read_text(encoding="utf-8")
        except OSError:
            return ""
        lines = [
            f"diff --git a/{target} b/{target}",
            "new file mode 100644",
            "--- /dev/null",
            f"+++ b/{target}",
        ]
        for line in content.splitlines():
            lines.append(f"+{line}")
        if not content.endswith("\n"):
            lines.append("\\ No newline at end of file")
        return "\n".join(lines) + "\n"

    def _re_scaffold_targeted_test_if_missing(
        self,
        run: SupervisorRun,
        config: RankSupervisorConfig,
    ) -> bool:
        target = self._targeted_test_file(config)
        if not target:
            return False
        target_path = Path(self.project_root) / target
        if target_path.exists():
            return False

        scaffold = self._scaffold_missing_tests_target(config)
        run.add("repair_scaffold", "success" if scaffold.success else "failure", _command_detail(scaffold))
        if not scaffold.success:
            return False

        synthetic_diff = self._synthetic_missing_tests_diff(config)
        if not synthetic_diff or not _is_valid_missing_tests_repair_diff(synthetic_diff, config.goal):
            restore = self.backend.restore_dangerous_diff()
            run.add(
                "repair_restore",
                "success" if restore.success else "failure",
                "Post-restore targeted test scaffold was invalid; restored.",
            )
            return False

        run.add(
            "repair_scaffold_diff",
            "success",
            "Synthesized missing-tests diff from post-restore scaffold file.",
            synthesized_untracked=True,
        )
        return True

    def _preserve_targeted_tests_after_restore_retry(
        self,
        run: SupervisorRun,
        config: RankSupervisorConfig,
        failure: str,
    ) -> None:
        if failure not in {"missing_tests", "pytest_failure"}:
            return
        if self._re_scaffold_targeted_test_if_missing(run, config):
            run.add(
                "repair_completion",
                "degraded",
                "Re-scaffolded targeted tests after restore-based retry path.",
            )

    def _scaffold_missing_tests_target(self, config: RankSupervisorConfig) -> CommandResult:
        target = self._targeted_test_file(config)
        if not target:
            return CommandResult(False, "", "No targeted test path configured for missing-tests scaffold", 2)

        endpoint = _required_endpoint_from_goal(config.goal)
        if not endpoint:
            return CommandResult(False, "", "No API endpoint found in goal for missing-tests scaffold", 2)

        test_slug = endpoint.strip("/").replace("/", "_").replace("-", "_").lower()
        test_slug = re.sub(r"[^a-z0-9_]+", "_", test_slug).strip("_")
        if not test_slug:
            test_slug = "mission_endpoint"

        # Scaffold a placeholder test that accepts both 200 (implemented) and 404/405
        # (endpoint not yet added).  A hard assert==200 on an unimplemented endpoint
        # wastes the full pytest run (~16 min) and leaves workspace dirty.
        content = (
            "from fastapi.testclient import TestClient\n\n"
            "from igris.web.server import create_app\n\n\n"
            f"def test_{test_slug}():\n"
            "    client = TestClient(create_app())\n"
            f"    response = client.get(\"{endpoint}\")\n"
            "    # Accept 200 (implemented) or 404/405 (scaffold placeholder — not yet implemented).\n"
            "    # A 5xx error would indicate a real problem and is not accepted.\n"
            "    assert response.status_code in (200, 404, 405), (\n"
            f"        f\"Unexpected status {{response.status_code}} for '{endpoint}' — \"\n"
            "        \"expected 200 (implemented) or 404/405 (not yet implemented)\"\n"
            "    )\n"
        )

        target_path = Path(self.project_root) / target
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return CommandResult(False, "", str(exc), 1)

        return CommandResult(True, f"Scaffolded {target} for {endpoint}", "", 0)

    def _repair_cycle(
        self,
        run: SupervisorRun,
        config: RankSupervisorConfig,
        failure: str,
        cycle: int,
        *,
        preserve_validated_progress: bool = False,
        stage_statuses: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> bool:
        title = f"{config.rank_id}: supervised repair for {failure}"
        body = f"Supervisor detected {failure} during run {run.run_id}."

        def _restore_or_preserve(detail: str, *, force_restore: bool = False) -> bool:
            if preserve_validated_progress and not force_restore:
                run.add(
                    "repair_restore",
                    "skipped",
                    f"{detail} Progress preserved because stage orchestration validated earlier stages.",
                    preserved_progress=True,
                )
                return True
            restore_result = self.backend.restore_dangerous_diff()
            run.add("repair_restore", "success" if restore_result.success else "failure", detail or _command_detail(restore_result))
            return restore_result.success

        if config.allow_github_pr and not config.dry_run:
            if self._repair_issue_already_created(run, failure):
                run.add(
                    "repair_issue",
                    "skipped",
                    "Repair issue already exists for this run/failure",
                    failure_class=failure,
                )
            else:
                issue = self.backend.create_issue(title, body)
                run.add(
                    "repair_issue",
                    "success" if issue.success else "failure",
                    _command_detail(issue),
                    failure_class=failure,
                )
        else:
            run.add("repair_issue", "dry_run", title, failure_class=failure)
        helper_advice = self._maybe_api_escalate(
            run,
            config,
            failure=failure,
            cycle=cycle,
            stage_statuses=stage_statuses,
        )
        high_risk_advice = False
        if helper_advice:
            risk = str(helper_advice.get("risk", "unknown")).lower()
            try:
                confidence = float(helper_advice.get("confidence", 0) or 0)
            except (TypeError, ValueError):
                confidence = 0.0
            high_risk_advice = (
                risk in {"high", "critical"}
                or confidence < 0.5
                or bool(helper_advice.get("requires_human_or_codex_audit", False))
                or not bool(helper_advice.get("must_not_complete_product_manually", False))
            )
        _wrong_paths: List[str] = []
        _allowed: List[str] = []

        if failure == "reasoning_loop_blocked":
            repair_goal = self._build_reasoning_loop_repair_prompt(
                stage_id=getattr(run, "stage_id", "unknown"),
                goal=config.goal,
                previous_reasoning_output="",
                repair_cycle=cycle,
            )
        elif failure == "semantic_incomplete":
            # The previous attempt produced a stub (# Placeholder, pass, hardcoded empty
            # values).  Repeat the original goal with explicit anti-stub guidance so the
            # next worker implements real logic rather than a skeleton.
            repair_goal = (
                f"{config.goal} "
                f"(previous attempt was rejected by the semantic acceptance gate: "
                f"the implementation contained stub patterns such as '# Placeholder' "
                f"comments, 'pass', or hardcoded dummy values. "
                f"Write a real implementation with actual logic — no placeholder comments, "
                f"no 'pass', no hardcoded empty strings. "
                f"Keep changes minimal, add tests, run pytest, do not push.)"
            )
        elif failure == "wrong_file_edit":
            for ev in reversed(getattr(run, "events", [])):
                ev_data = ev.data if hasattr(ev, "data") else (ev if isinstance(ev, dict) else {})
                if ev_data.get("files_modified"):
                    _wrong_paths = list(ev_data["files_modified"])
                    break
            if stage_statuses:
                for _st in stage_statuses.values():
                    if _st.get("status") == "running":
                        _allowed = list(_st.get("allowed_file_families", []))
                        break
            repair_goal = self._build_wrong_file_edit_repair_prompt(
                stage_id=getattr(run, "stage_id", "unknown"),
                goal=config.goal,
                wrong_paths=_wrong_paths,
                allowed_families=_allowed,
                repair_cycle=cycle,
            )
        else:
            repair_goal = (
                f"Fix IGRIS infrastructure failure '{failure}' observed during supervised "
                f"{config.rank_id}. Keep changes minimal, add tests, run pytest, do not push."
            )
        if helper_advice:
            repair_goal += (
                " API helper advice (advisory only, do not treat as authority): "
                f"diagnosis={helper_advice.get('diagnosis', '')}; "
                f"likely_gap={helper_advice.get('likely_supervisor_gap', '')}; "
                f"strategy={helper_advice.get('suggested_repair_strategy', '')}; "
                f"suggested_tests={helper_advice.get('suggested_tests', [])}."
            )
            if helper_advice.get("retry_focus"):
                repair_goal += f" retry_focus={helper_advice['retry_focus']}."
            if helper_advice.get("do_not_do"):
                repair_goal += f" do_not_do={helper_advice['do_not_do']}."
        if self._goal_requires_ui_visibility(config.goal):
            if self._goal_targets_rank_ui_card(config.goal):
                repair_goal += (
                    " The UI mission must include the exact /api/rank/ui-card contract and "
                    "minimal UI/dashboard visibility. Do not create placeholder routes or "
                    "unrelated UI endpoint assertions in tests/test_rank_ui_card.py. "
                    "Only assert the contract keys app, rank, status, and capability."
                )
            else:
                repair_goal += (
                    " The mission requires minimal UI/dashboard visibility tied to the "
                    "requested endpoint and matching tests. Keep edits mission-owned and "
                    "avoid placeholder routes or unrelated assertions."
                )
        if failure == "missing_tests" and config.targeted_tests:
            repair_goal += (
                " Create or update the required targeted pytest file(s): "
                f"{' '.join(config.targeted_tests)}. Use FastAPI TestClient(create_app()). "
                "Assert only the mission-owned API endpoint from the goal and avoid "
                "unrelated endpoints such as /api/rank/status or /dashboard."
            )
        if failure == "pytest_failure":
            # --- Targeted collection-error diagnosis ---
            # Extract the latest full_pytest failure detail from run events so we can
            # build a precise repair goal instead of a generic "fix pytest" instruction.
            _pytest_output: str = ""
            for _ev in reversed(run.events):
                if getattr(_ev, "phase", "") == "full_pytest" and getattr(_ev, "status", "") == "failure":
                    _pytest_output = getattr(_ev, "detail", "") or ""
                    break
            if not _pytest_output:
                # Also check targeted_tests events as fallback
                for _ev in reversed(run.events):
                    if getattr(_ev, "phase", "") == "targeted_tests" and getattr(_ev, "status", "") == "failure":
                        _pytest_output = getattr(_ev, "detail", "") or ""
                        break

            _collection_err = _parse_pytest_collection_error(_pytest_output) if _pytest_output else None

            if _collection_err:
                _err_type = _collection_err.get("error_type", "")
                if _err_type == "missing_symbol":
                    _sym = _collection_err.get("missing_symbol", "")
                    _mod = _collection_err.get("source_module", "")
                    repair_goal = (
                        f"Fix pytest collection ImportError: the test suite tries to import "
                        f"'{_sym}' from '{_mod}' but that symbol does not exist there. "
                        f"Steps: (1) read the failing test file(s) to understand how '{_sym}' "
                        f"is used; (2) implement '{_sym}' in '{_mod}' (or the correct module) "
                        f"with the exact API the tests expect; (3) run pytest and confirm "
                        f"collection succeeds and all tests pass. "
                        f"Keep changes minimal — do not refactor unrelated code."
                    )
                elif _err_type == "missing_module":
                    _missing_mod = _collection_err.get("missing_module", "")
                    repair_goal = (
                        f"Fix pytest collection ModuleNotFoundError: module '{_missing_mod}' "
                        f"is imported by the test suite but cannot be found. "
                        f"Steps: (1) check if '{_missing_mod}' is a project module that needs "
                        f"to be created or if it is a missing dependency; (2) if it is a "
                        f"project module, create it with the minimum API required by the tests; "
                        f"(3) if it is a third-party package, add it to requirements and "
                        f"install it; (4) run pytest and confirm collection succeeds. "
                        f"Keep changes minimal."
                    )
                elif _err_type == "collection_error":
                    _tf = _collection_err.get("failing_test_file", "")
                    repair_goal = (
                        f"Fix pytest collection error{' in ' + _tf if _tf else ''}. "
                        f"The test collection phase failed (EEE / no tests ran). "
                        f"Steps: (1) run 'python -m pytest --collect-only' to reproduce the "
                        f"exact error; (2) read the failing test file{'  ' + _tf if _tf else ''} "
                        f"and the module(s) it imports; (3) fix the root cause (missing class, "
                        f"wrong import path, syntax error, etc.); (4) run pytest and confirm "
                        f"all tests are collected and pass. Keep changes minimal."
                    )

            # Always append the FastAPI test-client reminder
            repair_goal += (
                " CRITICAL — this is a FastAPI application. Any test file MUST use "
                "'from fastapi.testclient import TestClient' and instantiate the client as "
                "'client = TestClient(create_app())'. Do NOT use the Flask-style "
                "'app.test_client()' — FastAPI app objects have no test_client() method and "
                "its use causes AttributeError at pytest collection time (EEE errors). "
                "If existing test files contain 'test_client(' they must be rewritten to "
                "use 'TestClient(create_app())' from fastapi.testclient. "
                "Verify that target API endpoints exist in igris/web/server.py before writing "
                "tests for them; if they are missing, add the endpoint implementation first."
            )
        # Track same-failure count: increment when the same failure class recurs.
        if failure and failure == run.last_repair_failure:
            run.same_failure_count += 1
        else:
            run.same_failure_count = 0
        run.last_repair_failure = failure

        # Issue #715 — adaptive retry ladder: when the same failure recurs, emit a
        # strategy_switch event so that the reasoning worker can use a different approach.
        if run.same_failure_count >= 1:
            run.add(
                "adaptive_retry",
                "strategy_switch",
                f"Same failure '{failure}' recurred {run.same_failure_count + 1} time(s); "
                "switching to focused single-file task type for this repair cycle.",
                attempt=cycle,
                task_type="single_file_single_test",
                same_failure_count=run.same_failure_count,
            )

        # Execution budget guard — checked before spending reasoning resources.
        budget_failure = self._check_execution_budget(run)
        if budget_failure:
            run.add(
                "execution_budget",
                "exceeded",
                f"Execution budget exceeded (IGRIS_MAX_COST_PER_RUN); aborting repair.",
                failure_class=budget_failure,
                execution_budget_used_usd=run.execution_budget_used_usd,
                max_cost_per_run=self._get_max_cost_per_run(),
            )
            run.failure_class = budget_failure
            run.status = "failed"
            return False

        repair_context = self._rank_initial_context(config, run=run)
        repair_context.update({
            "repair_cycle": cycle,
            "failure_class": failure,
            "supervised_repair": True,
            "repair_goal": repair_goal,
            "api_helper_advice": helper_advice or {},
            "api_helper_advisory_only": True,
        })
        if failure == "wrong_file_edit" and _wrong_paths:
            repair_context["constraint_wrong_file_history"] = {
                "previous_wrong_paths": _wrong_paths,
                "allowed_families": _allowed,
                "instruction": (
                    "You MUST only modify files in allowed_families. "
                    "Any edit outside this list will be reverted and counted as a failure."
                ),
            }

        # Determine execution strategy when helper has provided an execution plan.
        has_execution_plan = bool(helper_advice and str(helper_advice.get("execution_plan", "")).strip())
        strategy, strategy_profile = self._strategy_for_repair(run, has_execution_plan)
        if strategy:
            run.strategy_used = strategy
            # Inject structured plan fields so the reasoning worker can use them.
            repair_context.update({
                "execution_plan": helper_advice.get("execution_plan", ""),
                "file_targets": helper_advice.get("file_targets", []),
                "operations": helper_advice.get("operations", []),
                "acceptance_matrix": helper_advice.get("acceptance_matrix", []),
                "required_tests": helper_advice.get("required_tests", []),
                "do_not_do": helper_advice.get("do_not_do", []),
                "retry_focus": helper_advice.get("retry_focus", ""),
                "helper_advice_strategy": strategy,
                "helper_advice_only": True,
            })

        # Escalate to cloud-first execution for repeated semantic failures or
        # when the local model is unavailable (would otherwise silently degrade
        # to deterministic fallback producing empty/stub output).
        repair_task_type = "code_reasoning"
        repair_profile: Optional[str] = None
        if failure in {"semantic_incomplete", "stub_detected", "reasoning_loop_blocked"}:
            repair_task_type = "semantic_repair"
        elif failure in {"missing_tests", "pytest_failure"} and cycle > 1:
            repair_task_type = "code_generation"
        elif failure == "max_steps":
            # Reasoning hit the step ceiling without making progress — the cheap model
            # couldn't complete the task. Escalate to strong_execution (gpt-4o) so the
            # repair attempt uses a more capable model rather than repeating the same failure.
            repair_profile = "strong_execution"
        # Issue #715 — adaptive retry ladder: same failure recurring → switch to a
        # focused single-file strategy so the model works on one file at a time.
        if run.same_failure_count >= 1 and repair_task_type == "code_reasoning":
            repair_task_type = "single_file_single_test"
        # Strategy profile takes precedence, then env override, then task default.
        if strategy_profile:
            repair_profile = strategy_profile
        env_profile = os.environ.get("IGRIS_EXECUTION_PREFERRED_PROFILE", "")
        if env_profile and not strategy_profile:
            repair_profile = env_profile
        run.add(
            "repair_reasoning",
            "running",
            f"Starting repair reasoning cycle {cycle}",
            task_type=repair_task_type,
            preferred_profile=repair_profile,
            failure_class=failure,
            error_code=_failure_error_code(failure),
            strategy_used=strategy or "",
            helper_model=config.api_helper_model if helper_advice else "",
            has_execution_plan=has_execution_plan,
            same_failure_count=run.same_failure_count,
        )
        # Strong models need extended repair timeout (same logic as main reasoning).
        _repair_timeout = config.reasoning_timeout_seconds
        _STRONG_PROFILES = {"strong_execution", "strong_cloud_reasoning", "gpu_reasoning"}
        if (repair_profile or "") in _STRONG_PROFILES or repair_task_type in ("semantic_repair", "endpoint_implementation"):
            _repair_timeout = int(os.getenv(
                "IGRIS_STRONG_REASONING_TIMEOUT_SECONDS",
                str(max(_repair_timeout * 3, 2400)),
            ))
        result = self.backend.run_reasoning(
            repair_goal,
            max_steps=160,
            initial_context=repair_context,
            timeout=_repair_timeout,
            task_type=repair_task_type,
            preferred_profile=repair_profile,
        )
        # Accumulate execution cost for budget tracking.
        try:
            step_cost = float(result.get("estimated_cost", 0) or 0)
        except (TypeError, ValueError):
            step_cost = 0.0
        run.execution_budget_used_usd += step_cost
        run.add(
            "repair_reasoning",
            str(result.get("status", "")),
            result.get("final_summary", ""),
            orchestrator_used=result.get("orchestrator_used", False),
            reasoning_execution_provider=result.get("reasoning_execution_provider", ""),
            reasoning_execution_model=result.get("reasoning_execution_model", ""),
            reasoning_execution_profile=result.get("reasoning_execution_profile", ""),
            local_model_available=result.get("local_model_available", False),
            strategy_used=strategy or "",
            execution_model=result.get("reasoning_execution_model", ""),
            task_type=repair_task_type,
            prompt_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
            estimated_cost=step_cost,
            same_failure_count=run.same_failure_count,
        )
        # Record a reasoning_timeout signal when repair reasoning times out, hits
        # budget, or explicitly refuses — all indicate the model cannot make progress.
        if str(result.get("stop_reason", "")) in {"reasoning_timeout", "budget_exceeded", "blocked"}:
            self._record_capability_signal(run, "reasoning_timeout")
        # Record a capability signal when repair also hits max_steps — the escalated
        # model (strong_execution) exhausted its step budget too, confirming a
        # capability ceiling rather than a transient model availability issue.
        if str(result.get("stop_reason", "")) == "max_steps" and repair_profile == "strong_execution":
            self._record_capability_signal(run, "max_steps_ceiling")
        # Restore working tree when reasoning timed out. Partial changes from an
        # interrupted worker are unreliable (e.g. broken imports in test files) and
        # must never reach repair_tests. For pytest_failure, try to re-scaffold the
        # targeted test to preserve mission progress. For all other failures, return
        # False so the outer loop can detect capability limits and trigger decomposition.
        if str(result.get("stop_reason", "")) == "reasoning_timeout":
            _restore_or_preserve(
                "Repair reasoning timed out; restoring working tree to prevent broken "
                "partial changes from reaching repair_tests.",
                force_restore=True,
            )
            if failure == "pytest_failure" and self._re_scaffold_targeted_test_if_missing(run, config):
                run.add(
                    "repair_completion",
                    "degraded",
                    "Restored failed pytest repair and re-scaffolded targeted tests to preserve mission progress.",
                )
                return True
            return False
        diff_stat = self.backend.git_diff_stat()
        diff = self.backend.git_diff()
        run.add("repair_diff_stat", "success" if diff_stat.success else "failure", _command_detail(diff_stat))
        if not diff_stat.success:
            return _restore_or_preserve(_command_detail(diff_stat), force_restore=True) and False
        if _has_destructive_diff(diff.output):
            if not _restore_or_preserve("Destructive repair diff rejected; restoring.", force_restore=True):
                return False
            if failure in RETRYABLE_REPAIR_FAILURES:
                self._preserve_targeted_tests_after_restore_retry(run, config, failure)
                run.add(
                    "repair_retry",
                    "running",
                    "Destructive repair diff was rejected; retrying with remaining budget.",
                    failure_class="destructive_diff",
                )
                return True
            return False
        if _has_invalid_fastapi_bootstrap_diff(diff.output):
            if not _restore_or_preserve(
                "Invalid FastAPI bootstrap diff rejected before repair validation",
                force_restore=True,
            ):
                return False
            run.add(
                "repair_retry",
                "running",
                "Invalid FastAPI bootstrap diff was rejected; retrying with remaining budget.",
                failure_class="invalid_bootstrap",
            )
            self._preserve_targeted_tests_after_restore_retry(run, config, failure)
            return True
        if failure == "missing_tests" and not _is_valid_missing_tests_repair_diff(diff.output, config.goal):
            if not _restore_or_preserve(
                "Missing-tests repair diff rejected before validation",
                force_restore=True,
            ):
                return False
            scaffold = self._scaffold_missing_tests_target(config)
            run.add("repair_scaffold", "success" if scaffold.success else "failure", _command_detail(scaffold))
            if scaffold.success:
                diff_stat = self.backend.git_diff_stat()
                diff = self.backend.git_diff()
                run.add("repair_scaffold_diff", "success" if diff_stat.success else "failure", _command_detail(diff_stat))
                if not diff.output.strip():
                    synthetic_diff = self._synthetic_missing_tests_diff(config)
                    if synthetic_diff:
                        diff = CommandResult(True, synthetic_diff, "", 0)
                        run.add(
                            "repair_scaffold_diff",
                            "success",
                            "Synthesized missing-tests diff from untracked scaffold file.",
                            synthesized_untracked=True,
                        )
            if (
                not scaffold.success
                or not _is_valid_missing_tests_repair_diff(diff.output, config.goal)
            ):
                if scaffold.success:
                    _restore_or_preserve(
                        "Scaffolded missing-tests diff was invalid; restored.",
                        force_restore=True,
                    )
                run.add(
                    "repair_retry",
                    "running",
                    "Missing-tests repair diff was rejected; retrying with remaining budget.",
                    failure_class="wrong_file_edit",
                )
                return True
        ui_visibility_goal = self._goal_requires_ui_visibility(config.goal)
        ui_card_goal = self._goal_targets_rank_ui_card(config.goal)
        if ui_visibility_goal and ui_card_goal and _is_product_only_ui_task_diff(diff.output):
            allow_safe_ui_repair = (
                _has_ui_surface_change(diff.output)
                and failure in {"missing_ui_visibility", "reasoning_loop_blocked"}
            )
            if failure == "pytest_failure":
                allow_safe_ui_repair = True
            if not allow_safe_ui_repair:
                if not _restore_or_preserve(
                    "Product-only UI task diff rejected before repair validation",
                    force_restore=True,
                ):
                    return False
                run.add(
                    "repair_retry",
                    "running",
                    "Product-only UI task diff was rejected; retrying with remaining budget.",
                    failure_class="wrong_file_edit",
                )
                self._preserve_targeted_tests_after_restore_retry(run, config, failure)
                return True
        if ui_visibility_goal and ui_card_goal and not _is_valid_ui_test_diff(diff.output):
            if not _restore_or_preserve(
                "Invalid UI test diff rejected before repair validation",
                force_restore=True,
            ):
                return False
            run.add(
                "repair_retry",
                "running",
                "Invalid UI test diff was rejected; retrying with remaining budget.",
                failure_class="wrong_file_edit",
            )
            self._preserve_targeted_tests_after_restore_retry(run, config, failure)
            return True
        # For pytest_failure repairs: reject diffs that introduce Flask-style test_client()
        # calls.  FastAPI apps have no test_client() method; using it causes AttributeError
        # at collection time (EEE errors).  Detecting and rejecting this pattern early avoids
        # repeated repair cycles that add the wrong client without making progress.
        if failure == "pytest_failure" and _has_flask_test_client_in_diff(diff.output):
            if not _restore_or_preserve(
                "Repair diff uses Flask-style test_client() which is incompatible with "
                "this FastAPI application; restoring and retrying with FastAPI TestClient "
                "guidance.",
                force_restore=True,
            ):
                return False
            run.add(
                "repair_retry",
                "running",
                "Flask test_client() detected in repair diff for FastAPI app; "
                "diff rejected. Retrying with explicit FastAPI TestClient(create_app()) guidance.",
                failure_class="wrong_file_edit",
            )
            self._preserve_targeted_tests_after_restore_retry(run, config, failure)
            return True
        if not diff.output.strip():
            # Count repairs that produce no diff — a model that cannot propose any
            # change after multiple cycles has hit a capability wall.
            self._record_capability_signal(run, "no_diff_repair")
            _restore_or_preserve("Repair produced no validated diff; restoring working tree state.")
            if failure == "pytest_failure" and self._re_scaffold_targeted_test_if_missing(run, config):
                run.add(
                    "repair_completion",
                    "degraded",
                    "No-diff pytest repair restored and re-scaffolded targeted tests; continuing rank attempts.",
                )
                return True
            if failure in RETRYABLE_REPAIR_FAILURES:
                self._preserve_targeted_tests_after_restore_retry(run, config, failure)
                run.add(
                    "repair_retry",
                    "running",
                    "Repair reasoning produced no validated diff; retrying with remaining budget.",
                    failure_class=failure,
                )
                return True
            return False
        run.add(
            "repair_tests",
            "running",
            "Running repair validation pytest (-m 'not slow')",
            timeout_seconds=config.test_timeout_seconds,
            exclude_slow=True,
        )
        tests = self.backend.run_tests(timeout=config.test_timeout_seconds, hard_cap=config.test_hard_cap_seconds, exclude_slow=True)
        run.add("repair_tests", "success" if tests.success else "failure", _command_detail(tests))
        if not tests.success and "Command killed:" in (tests.error or ""):
            # Repair validation also hung — counts against the same capability-limit budget.
            self._record_capability_signal(run, "pytest_hang")
        if not tests.success:
            if failure == "missing_tests" and _is_valid_missing_tests_repair_diff(diff.output, config.goal):
                run.add(
                    "repair_completion",
                    "degraded",
                    "Preserved valid missing-tests scaffold despite failing full pytest; continuing rank attempts.",
                )
                return True
            _restore_or_preserve("Repair validation failed; restoring unless preserving validated stage progress.")
            if failure == "pytest_failure" and self._re_scaffold_targeted_test_if_missing(run, config):
                run.add(
                    "repair_completion",
                    "degraded",
                    "Restored failed pytest repair and re-scaffolded targeted tests to preserve mission progress.",
                )
                return True
            if failure in RETRYABLE_REPAIR_FAILURES:
                self._preserve_targeted_tests_after_restore_retry(run, config, failure)
                run.add(
                    "repair_retry",
                    "running",
                    "Repair validation failed; retrying with remaining budget.",
                    failure_class=failure,
                )
                return True
            return False
        if str(result.get("status", "")) != "finished":
            run.add(
                "repair_completion",
                "degraded",
                "Repair reasoning did not finish cleanly but the validated diff was accepted.",
                stop_reason=result.get("stop_reason", ""),
                files_modified=result.get("files_modified", []),
            )
        if high_risk_advice:
            run.add(
                "repair_high_risk_validation",
                "running",
                "High-risk helper advice detected; running stronger validation smoke.",
            )
            strong_smoke = self.backend.smoke(config.required_smoke_endpoints, "")
            run.add(
                "repair_high_risk_validation",
                "success" if strong_smoke.success else "failure",
                _command_detail(strong_smoke),
            )
            if not strong_smoke.success:
                _restore_or_preserve("High-risk advisory validation smoke failed; restoring.", force_restore=True)
                if failure in RETRYABLE_REPAIR_FAILURES:
                    run.add(
                        "repair_retry",
                        "running",
                        "High-risk advisory smoke failed; retrying with remaining budget.",
                        failure_class="infrastructure_bug",
                    )
                    return True
                return False
        return True

    def run(
        self,
        config: RankSupervisorConfig,
        run: Optional[SupervisorRun] = None,
    ) -> SupervisorRun:
        """Thin orchestrator — delegates to preflight and rank-loop phases."""
        # Issue #540 — create WorkSession for this run (best-effort, never blocks)
        _work_session = None
        try:
            from igris.core.work_session import WorkSession as _WS
            _work_session = _WS.create(goal=config.goal, mission_id=None)
        except Exception:
            pass

        run, ctx = self._run_preflight_phase(run, config)
        if ctx is None:
            if _work_session is not None:
                try:
                    _work_session.remember(str(self.project_root))
                except Exception:
                    pass
            return run

        result = self._run_rank_loop(run, config, **ctx)

        if _work_session is not None:
            try:
                _commands = [
                    {"action_type": e.phase, "outcome": e.status, "duration_ms": 0.0}
                    for e in result.events
                    if e.phase not in {"start", "queued"}
                ]
                _work_session.remember(str(self.project_root), commands_run=_commands)
            except Exception:
                pass

        return result

    def _stage_report_fragment(
        self,
        mission_plan: Optional[MissionPlan],
        stage_statuses: Optional[Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        if not mission_plan or not stage_statuses:
            return {}
        return {
            "mission_orchestration": {
                "mode": mission_plan.mode,
                "stages": self._stage_status_list(stage_statuses, mission_plan),
            }
        }

    @staticmethod
    def _api_escalation_report_fragment(run: SupervisorRun) -> Dict[str, Any]:
        return {
            "api_escalation": {
                "calls_used": run.api_escalations_used,
                "calls_failed_unconfigured": run.api_escalations_failed_unconfigured,
                "budget_used_usd": round(run.api_budget_used_usd, 6),
            }
        }

    def _complete_rank(
        self,
        run: SupervisorRun,
        config: RankSupervisorConfig,
        branch: str,
        *,
        completion_mode: str = "direct",
        runtime_refresh_required: bool = False,
        mission_plan: Optional[MissionPlan] = None,
        stage_statuses: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> SupervisorRun:
        cancelled = self._cancel_if_requested(run, mission_plan=mission_plan, stage_statuses=stage_statuses)
        if cancelled is not None:
            return cancelled
        restart_command = config.service_restart_command if not config.defer_service_restart else ""
        post_merge_smoke: Optional[CommandResult] = None
        manual_remaining = ""
        if stage_statuses and "pr_ci_merge" in stage_statuses:
            self._set_stage_status(run, stage_statuses, "pr_ci_merge", "running", "Executing PR/CI/merge workflow.")
        if config.dry_run:
            manual_remaining = "delivery skipped by dry_run"
            run.add("github", "dry_run", "Commit/PR/merge skipped by dry_run")
            if stage_statuses and "pr_ci_merge" in stage_statuses:
                self._set_stage_status(
                    run,
                    stage_statuses,
                    "pr_ci_merge",
                    "skipped",
                    "PR/CI/merge skipped because dry_run is enabled.",
                    no_op=True,
                )
            if stage_statuses and "post_merge_runtime" in stage_statuses:
                self._set_stage_status(
                    run,
                    stage_statuses,
                    "post_merge_runtime",
                    "skipped",
                    "Post-merge runtime checks skipped because dry_run is enabled.",
                    no_op=True,
                )
        else:
            commit = self.backend.commit(f"feat: complete supervised {config.rank_id}", ["igris", "tests"])
            run.add("commit", "success" if commit.success else "failure", _command_detail(commit))
            if not commit.success and "nothing to commit" in (commit.output or "") + (commit.error or ""):
                # Working tree unexpectedly clean — attempt patch-based recovery.
                # The diff was saved to disk before tests ran; apply it now.
                _patch_path = Path(self.project_root) / ".igris" / "rank_pending.patch"
                if _patch_path.exists():
                    run.add("commit_patch_recovery", "running", "Applying saved diff patch to recover working tree")
                    apply = self.backend._run(
                        ["git", "apply", "--index", str(_patch_path)],
                        timeout=30,
                    )
                    run.add(
                        "commit_patch_recovery",
                        "success" if apply.success else "failure",
                        _command_detail(apply),
                    )
                    if apply.success:
                        commit = self.backend.commit(
                            f"feat: complete supervised {config.rank_id}", None
                        )
                        run.add(
                            "commit",
                            "success" if commit.success else "failure",
                            _command_detail(commit),
                        )
                    try:
                        _patch_path.unlink(missing_ok=True)
                    except Exception:
                        pass
            elif commit.success:
                try:
                    (Path(self.project_root) / ".igris" / "rank_pending.patch").unlink(missing_ok=True)
                except Exception:
                    pass
            if not commit.success:
                return self._blocked(
                    run,
                    "infrastructure_bug",
                    "Commit failed",
                    mission_plan=mission_plan,
                    stage_statuses=stage_statuses,
                )
            if config.allow_github_pr:
                push = self.backend.push_branch(branch)
                run.add("push", "success" if push.success else "failure", _command_detail(push))
                pr = self.backend.open_pr(branch, f"feat: supervised {config.rank_id}", self._pr_body(run))
                run.add("pr", "success" if pr.success else "failure", _command_detail(pr))
                ci = self.backend.wait_ci()
                run.add("ci", "success" if ci.success else "failure", _command_detail(ci))
                if stage_statuses and "pr_ci_merge" in stage_statuses:
                    self._set_stage_status(
                        run,
                        stage_statuses,
                        "pr_ci_merge",
                        "success" if ci.success else "failure",
                        "PR workflow green." if ci.success else "PR workflow failed or CI not green.",
                    )
                if config.allow_merge_if_green and ci.success:
                    merge = self.backend.merge_pr()
                    run.add("merge", "success" if merge.success else "failure", _command_detail(merge))
                    pull = self.backend.pull_main()
                    run.add("pull_main", "success" if pull.success else "failure", _command_detail(pull))
                    if config.defer_service_restart and runtime_refresh_required:
                        run.add(
                            "post_merge_smoke",
                            "deferred",
                            "Post-merge smoke deferred until the runtime is restarted and refreshed.",
                            runtime_refresh_required=True,
                        )
                        if stage_statuses and "post_merge_runtime" in stage_statuses:
                            self._set_stage_status(
                                run,
                                stage_statuses,
                                "post_merge_runtime",
                                "skipped",
                                "Post-merge runtime smoke deferred until runtime refresh.",
                                no_op=True,
                            )
                    else:
                        post_merge_smoke = self.backend.smoke(
                            config.required_smoke_endpoints,
                            restart_command,
                        )
                        run.add(
                            "post_merge_smoke",
                            "success" if post_merge_smoke.success else "failure",
                            _command_detail(post_merge_smoke),
                        )
                        if stage_statuses and "post_merge_runtime" in stage_statuses:
                            self._set_stage_status(
                                run,
                                stage_statuses,
                                "post_merge_runtime",
                                "success" if post_merge_smoke.success else "failure",
                                "Post-merge runtime smoke passed."
                                if post_merge_smoke.success
                                else "Post-merge runtime smoke failed.",
                            )
                        if not post_merge_smoke.success:
                            run.status = "blocked"
                            run.outcome = "Blocked"
                            run.failure_class = "infrastructure_bug"
                            _deg, _deg_reason = self._compute_degraded_completion(
                                completion_mode=completion_mode,
                                runtime_refresh_required=runtime_refresh_required,
                                post_merge_smoke_success=False,
                                smoke_was_applicable=True,  # smoke ran (but failed)
                                failure_class=run.failure_class,
                                stage_statuses=stage_statuses,
                            )
                            run.report = {
                                "autonomous": True,
                                "manual_remaining": "post-merge verification failed",
                                "completion_mode": completion_mode,
                                "degraded_completion": _deg,
                                "degraded_completion_reason": _deg_reason,
                                "post_merge_smoke": False,
                                "runtime_refresh_required": runtime_refresh_required,
                            }
                            if run.acceptance_evidence is not None:
                                run.report["acceptance_evidence"] = run.acceptance_evidence
                            run.report.update(self._api_escalation_report_fragment(run))
                            run.report.update(self._stage_report_fragment(mission_plan, stage_statuses))
                            run.touch()
                            return run
                elif config.allow_merge_if_green and not ci.success:
                    manual_remaining = "merge skipped because CI is not green"
                    if stage_statuses and "post_merge_runtime" in stage_statuses:
                        self._set_stage_status(
                            run,
                            stage_statuses,
                            "post_merge_runtime",
                            "skipped",
                            "Post-merge runtime checks skipped because merge did not occur.",
                            no_op=True,
                        )
                else:
                    manual_remaining = "merge disabled by config"
                    if stage_statuses and "post_merge_runtime" in stage_statuses:
                        self._set_stage_status(
                            run,
                            stage_statuses,
                            "post_merge_runtime",
                            "skipped",
                            "Post-merge runtime checks skipped because merge is disabled by config.",
                            no_op=True,
                        )
            else:
                manual_remaining = "GitHub PR/merge workflow disabled by config"
                if stage_statuses and "pr_ci_merge" in stage_statuses:
                    self._set_stage_status(
                        run,
                        stage_statuses,
                        "pr_ci_merge",
                        "skipped",
                        "PR workflow disabled by config.",
                        no_op=True,
                    )
                if stage_statuses and "post_merge_runtime" in stage_statuses:
                    self._set_stage_status(
                        run,
                        stage_statuses,
                        "post_merge_runtime",
                        "skipped",
                        "Post-merge runtime checks skipped because PR workflow is disabled.",
                        no_op=True,
                    )
        run.status = "completed"
        run.outcome = "Completed"
        if stage_statuses and "final_report" in stage_statuses:
            if self._required_stages_green(stage_statuses):
                self._set_stage_status(run, stage_statuses, "final_report", "success", "All required mission stages are green.")
            else:
                self._set_stage_status(run, stage_statuses, "final_report", "failure", "Required mission stage is missing or failed.")
                run.status = "blocked"
                run.outcome = "Blocked"
                run.failure_class = "infrastructure_bug"
        post_smoke_success = False if post_merge_smoke is None else post_merge_smoke.success
        # post_merge_smoke is only non-None when a merge was actually executed
        smoke_applicable = post_merge_smoke is not None
        degraded, degraded_reason = self._compute_degraded_completion(
            completion_mode=completion_mode,
            runtime_refresh_required=runtime_refresh_required,
            post_merge_smoke_success=post_smoke_success,
            smoke_was_applicable=smoke_applicable,
            failure_class=run.failure_class,
            stage_statuses=stage_statuses,
        )
        run.completion_mode = completion_mode  # (#147) expose for MBOP Phase 11
        run.report = {
            "autonomous": True,
            "manual_remaining": manual_remaining,
            "completion_mode": completion_mode,
            "degraded_completion": degraded,
            "degraded_completion_reason": degraded_reason,
            "post_merge_smoke": post_smoke_success,
            "runtime_refresh_required": runtime_refresh_required,
        }
        if run.acceptance_evidence is not None:
            run.report["acceptance_evidence"] = run.acceptance_evidence
        run.report.update(self._api_escalation_report_fragment(run))
        run.report.update(self._stage_report_fragment(mission_plan, stage_statuses))
        run.touch()
        return run

    def _complete_noop(
        self,
        run: SupervisorRun,
        *,
        completion_mode: str,
        runtime_refresh_required: bool,
        detail: str,
        post_merge_smoke: bool,
        mission_plan: Optional[MissionPlan] = None,
        stage_statuses: Optional[Dict[str, Dict[str, Any]]] = None,
        exclude_stage_ids: Optional[Set[str]] = None,
    ) -> SupervisorRun:
        # No-op completions never execute delivery stages; always exclude them so
        # the required-stages check does not reject a valid no-op because
        # pr_ci_merge / post_merge_runtime were not reached.
        _noop_exclude = (exclude_stage_ids or set()) | {
            "pr_ci_merge",
            "post_merge_runtime",
            "final_report",
        }
        run.completion_mode = completion_mode  # (#147)
        run.add("completion", "degraded", detail, mode=completion_mode)
        run.status = "completed"
        run.outcome = "Completed"
        if stage_statuses and "final_report" in stage_statuses:
            if self._required_stages_green(stage_statuses, exclude_stage_ids=_noop_exclude):
                self._set_stage_status(run, stage_statuses, "final_report", "success", "No-op completion validated with required stages satisfied.")
            else:
                self._set_stage_status(run, stage_statuses, "final_report", "failure", "No-op completion rejected: required stage missing.")
                run.status = "blocked"
                run.outcome = "Blocked"
                run.failure_class = "reasoning_loop_blocked"
        run.report = {
            "autonomous": True,
            "manual_remaining": "",
            "completion_mode": completion_mode,
            "degraded_completion": True,
            "degraded_completion_reason": (
                f"no-op completion ({completion_mode}): mission goal already satisfied; "
                "no delivery actions performed in this run"
            ),
            "post_merge_smoke": post_merge_smoke,
            "runtime_refresh_required": runtime_refresh_required,
            "no_op_completion": True,
        }
        run.report.update(self._api_escalation_report_fragment(run))
        run.report.update(self._stage_report_fragment(mission_plan, stage_statuses))
        run.touch()
        return run

    def _cleanup_blocked_workspace(self, run: SupervisorRun) -> None:
        run.add(
            "blocked_workspace_cleanup",
            "running",
            "Ensuring blocked run does not leave dirty workspace state.",
        )
        status_before = self.backend.git_status()
        run.add(
            "blocked_workspace_state",
            "success" if status_before.success else "failure",
            _command_detail(status_before),
            dirty=bool(status_before.output.strip()) if status_before.success else False,
        )
        if not status_before.success:
            run.add(
                "blocked_workspace_cleanup",
                "failure",
                "Unable to read git status for blocked workspace cleanup.",
            )
            return
        if not status_before.output.strip():
            run.add(
                "blocked_workspace_cleanup",
                "success",
                "Workspace already clean at blocked exit.",
                no_op=True,
            )
            return
        diff_stat = self.backend.git_diff_stat()
        run.add(
            "blocked_workspace_diff",
            "success" if diff_stat.success else "failure",
            _command_detail(diff_stat),
        )
        restore = self.backend.restore_dangerous_diff()
        run.add(
            "blocked_workspace_cleanup",
            "success" if restore.success else "failure",
            _command_detail(restore),
        )
        # Switch back to main so rank branch is not the current HEAD
        checkout_result = self.backend.checkout_main()
        run.add(
            "blocked_workspace_checkout_main",
            "success" if checkout_result.success else "failure",
            _command_detail(checkout_result),
        )
        # Delete stale rank-* local branches left by supervised runs
        branch_cleanup = self.backend.delete_stale_rank_branches()
        run.add(
            "blocked_workspace_branch_cleanup",
            "success" if branch_cleanup.success else "failure",
            _command_detail(branch_cleanup),
        )
        status_after = self.backend.git_status()
        run.add(
            "blocked_workspace_state",
            "success" if status_after.success else "failure",
            _command_detail(status_after),
            dirty=bool(status_after.output.strip()) if status_after.success else False,
            after_cleanup=True,
        )

    def _cleanup_cancelled_workspace(self, run: SupervisorRun) -> None:
        run.add(
            "cancel_workspace_cleanup",
            "running",
            "Ensuring cancelled run leaves tracked workspace state.",
        )
        status_before = self.backend.git_status()
        dirty_before = bool(status_before.output.strip()) if status_before.success else False
        run.add(
            "cancel_workspace_state",
            "success" if status_before.success else "failure",
            _command_detail(status_before),
            dirty=dirty_before,
            before_cleanup=True,
        )
        if not status_before.success or not dirty_before:
            run.add(
                "cancel_workspace_cleanup",
                "skipped",
                "Workspace already clean or status unavailable; no restore executed.",
            )
            return
        restore = self.backend.restore_dangerous_diff()
        run.add(
            "cancel_workspace_cleanup",
            "success" if restore.success else "failure",
            _command_detail(restore),
        )
        status_after = self.backend.git_status()
        run.add(
            "cancel_workspace_state",
            "success" if status_after.success else "failure",
            _command_detail(status_after),
            dirty=bool(status_after.output.strip()) if status_after.success else False,
            after_cleanup=True,
        )

    def _cancelled(
        self,
        run: SupervisorRun,
        reason: str,
        *,
        mission_plan: Optional[MissionPlan] = None,
        stage_statuses: Optional[Dict[str, Dict[str, Any]]] = None,
        cleanup_workspace: bool = True,
    ) -> SupervisorRun:
        run.cancel_requested = True
        run.cancel_reason = reason
        run.status = "cancelled"
        run.outcome = "Cancelled"
        run.failure_class = "user_cancelled"
        run.add("cancelled", "cancelled", reason)
        if cleanup_workspace:
            self._cleanup_cancelled_workspace(run)
        if stage_statuses and "final_report" in stage_statuses:
            self._set_stage_status(run, stage_statuses, "final_report", "failure", "Run cancelled by user.")
        run.report = {"autonomous": False, "cancelled_reason": reason, "blocked_reason": reason}
        run.report.update(self._api_escalation_report_fragment(run))
        run.report.update(self._stage_report_fragment(mission_plan, stage_statuses))
        run.touch()
        return run

    # ------------------------------------------------------------------
    # Capability-limit detection and mission decomposition
    # ------------------------------------------------------------------

    @staticmethod
    def _record_capability_signal(run: SupervisorRun, signal: str) -> None:
        run.capability_signals[signal] = run.capability_signals.get(signal, 0) + 1

    @staticmethod
    def _detect_capability_limit(run: SupervisorRun) -> Optional[str]:
        """Return the triggering signal if capability limit reached, or None.

        Fires when any single signal reaches CAPABILITY_LIMIT_THRESHOLD, or when
        the combined total of all distinct signals reaches the threshold (mixed-failure
        capability wall — e.g. one reasoning_timeout + one no_diff_repair).
        """
        for signal, count in run.capability_signals.items():
            if count >= CAPABILITY_LIMIT_THRESHOLD:
                return signal
        if sum(run.capability_signals.values()) >= CAPABILITY_LIMIT_THRESHOLD:
            return max(run.capability_signals, key=run.capability_signals.get)
        return None

    @staticmethod
    def _is_structural_ceiling(run: SupervisorRun, triggering_signal: str) -> bool:
        """Return True when the capability limit is structural, not transient.

        Only `max_steps_ceiling` (strong_execution profile exhausted its step budget)
        qualifies as a true structural ceiling.  Pure `reasoning_timeout` or
        `no_diff_repair` signals indicate transient failures where decomposition may
        still help — these go through the normal decompose path.
        """
        return run.capability_signals.get("max_steps_ceiling", 0) >= 1

    @staticmethod
    def _should_fast_track_capability_limit(
        run: SupervisorRun,
        failure: str,
    ) -> Optional[str]:
        """Return capability signal when we should decompose immediately."""
        if failure not in {
            "reasoning_loop_blocked",
            "pytest_failure",
            "test_runner_timeout",
            "infrastructure_bug",
        }:
            return None
        return SelfRepairSupervisor._detect_capability_limit(run)

    def _handle_capability_limit(
        self,
        run: SupervisorRun,
        triggering_signal: str,
        config: "RankSupervisorConfig",
        mission_plan: Optional["MissionPlan"],
        stage_statuses: Optional[Dict[str, Dict[str, Any]]],
        *,
        cleanup_workspace: bool = True,
    ) -> SupervisorRun:
        """Block with `capability_ceiling_reached` or `decomposition_required`.

        When the ceiling is structural (strong model already exhausted), skip the
        expensive decompose LLM call and emit `capability_ceiling_reached` so the
        watchdog can skip the issue immediately after the first failure.
        """
        if self._is_structural_ceiling(run, triggering_signal):
            run.add(
                "capability_ceiling",
                "detected",
                f"Structural capability ceiling confirmed ({triggering_signal} × "
                f"{run.capability_signals.get(triggering_signal, 0)}); "
                "no stronger model available — skipping decomposition call.",
                triggering_signal=triggering_signal,
                capability_signals=dict(run.capability_signals),
            )
            return self._blocked(
                run,
                "capability_ceiling_reached",
                (
                    f"Capability ceiling reached ({triggering_signal} × "
                    f"{run.capability_signals.get(triggering_signal, 0)}); "
                    "model cannot make further progress on this mission."
                ),
                mission_plan=mission_plan,
                stage_statuses=stage_statuses,
                cleanup_workspace=cleanup_workspace,
            )
        decomposition = self._ask_igris_decompose(run, config)
        return self._blocked_decomposition_required(
            run,
            triggering_signal,
            (
                f"Capability limit detected ({triggering_signal} × "
                f"{run.capability_signals[triggering_signal]}); "
                "mission requires decomposition."
            ),
            decomposition,
            config=config,
            mission_plan=mission_plan,
            stage_statuses=stage_statuses,
            cleanup_workspace=cleanup_workspace,
        )

    def _plan_mission(
        self, run: SupervisorRun, config: RankSupervisorConfig
    ) -> Dict[str, Any]:
        """Pre-flight read-only reasoning pass: estimate scope and flag if
        decomposition is needed BEFORE any code is written.

        Returns a MissionScope dict (may be empty on planning failure — the run
        proceeds normally in that case so planning never blocks a mission).
        """
        planning_goal = (
            "PLANNING PASS — read-only analysis only, do NOT modify any files.\n\n"
            f"Mission goal: {config.goal}\n\n"
            "Analyse the codebase and output ONLY valid JSON with these fields:\n"
            "- files_to_touch: list of file paths you would need to modify\n"
            "- estimated_complexity: 'low', 'medium', or 'high'\n"
            "- decomposition_recommended: true if the mission is too large for a single attempt\n"
            "- decomposition_reason: one sentence explaining why (if recommended)\n"
            "- safe_entry_point: the smallest first concrete step\n"
            "- risks: list of strings describing potential pitfalls\n\n"
            "Output ONLY the JSON object, nothing else."
        )
        run.add(
            "mission_planning",
            "running",
            "Running pre-flight mission scope analysis (read-only)",
            max_steps=PLANNING_MAX_STEPS,
            timeout_seconds=PLANNING_TIMEOUT_SECONDS,
        )
        planner_profile = str(
            os.getenv("IGRIS_ROLE_PLANNER_PROFILE", "mini_execution")
        ).strip() or "mini_execution"
        planner_task_type = str(
            os.getenv("IGRIS_ROLE_PLANNER_TASK_TYPE", "code_reasoning")
        ).strip() or "code_reasoning"
        result = self.backend.run_reasoning(
            planning_goal,
            max_steps=PLANNING_MAX_STEPS,
            initial_context={"read_only": True, "planning_pass": True},
            timeout=PLANNING_TIMEOUT_SECONDS,
            task_type=planner_task_type,
            preferred_profile=planner_profile,
        )
        raw = _safe_redact(
            result.get("final_summary") or result.get("output") or ""
        )
        scope: Dict[str, Any] = {}
        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                scope = json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            scope = {"raw_output": raw}
        run.add(
            "mission_planning",
            "success" if scope and "estimated_complexity" in scope else "partial",
            (
                f"Planning complete. complexity={scope.get('estimated_complexity', '?')} "
                f"decomposition_recommended={scope.get('decomposition_recommended', False)}"
            ),
            estimated_complexity=scope.get("estimated_complexity", "unknown"),
            decomposition_recommended=bool(scope.get("decomposition_recommended", False)),
            files_to_touch=list(scope.get("files_to_touch") or []),
        )
        run.mission_scope = scope
        run.report["mission_scope"] = scope
        run.report["mission_planning_profile"] = planner_profile
        run.report["mission_planning_task_type"] = planner_task_type

        # M3 — Model-aware escalation: when the local model says the mission is
        # high-complexity AND the operator has configured API escalation, ask the
        # helper for strategic advice BEFORE the first attempt.  This is purely
        # advisory: advice is recorded in run events but never blocks the run.
        if (
            scope.get("estimated_complexity") == "high"
            and config.allow_api_escalation
            and config.max_api_escalations_per_run > 0
        ):
            run.add(
                "model_aware_escalation",
                "running",
                "High complexity detected during planning — requesting advisory strategy from API helper.",
                complexity="high",
            )
            advice = self._maybe_api_escalate(
                run,
                config,
                failure="high_complexity_planning",
                cycle=0,
            )
            if advice:
                run.add(
                    "model_aware_escalation",
                    "success",
                    f"Planning-phase advisory received. strategy: "
                    f"{str(advice.get('suggested_repair_strategy',''))[:120]}",
                    confidence=advice.get("confidence"),
                    risk=advice.get("risk"),
                )
                # Surface escalation hints in the mission scope so they're
                # visible alongside planning output.
                scope["escalation_strategy_hint"] = advice.get("suggested_repair_strategy", "")
                scope["escalation_risk"] = advice.get("risk", "")
                run.mission_scope = scope
                run.report["mission_scope"] = scope
            else:
                run.add(
                    "model_aware_escalation",
                    "skipped",
                    "Planning-phase escalation skipped (helper not configured or budget exhausted).",
                )

        return scope

    # Short prompt template for the local decomposition attempt (max_steps=15).
    _DECOMP_SHORT_PROMPT = (
        "DECOMPOSE — no code, output JSON only.\n"
        "Mission: '{goal}'\n"
        "Signals: {signals}\n\n"
        "Rules:\n"
        "- Each sub-mission touches at most 1-2 files and is implementable in <40 reasoning steps.\n"
        "- Prefer 4-8 atomic sub-missions over 2-3 large ones.\n"
        "- First sub-mission must be self-contained (no deps on later ones).\n"
        "- Include concrete file paths and function names in each goal.\n\n"
        "Output ONLY:\n"
        '{{"why_too_large":"<reason>","sub_missions":[{{"title":"<t>","goal":"<g>","risk_level":"low"}}],"first_sub_mission":"<t>","human_approval_required":false}}'
    )

    def _ask_igris_decompose(
        self, run: SupervisorRun, config: RankSupervisorConfig
    ) -> Dict[str, Any]:
        """Ask IGRIS to decompose a too-large mission into sub-missions.

        Uses a fallback chain:
          1. Local reasoning short-prompt (max_steps=15)
          2. API helper (if configured and budget allows)
          3. Deterministic fallback (always succeeds)
        """
        signals = dict(run.capability_signals)

        # --- emit decomposition_request event (same as before) ---
        context = self._rank_initial_context(config, run=run)
        context.update({
            "decomposition_required": True,
            "capability_limit_signals": signals,
            "repair_cycles_used": run.repair_cycles_used,
            "max_repair_cycles": run.max_repair_cycles,
        })
        run.add(
            "decomposition_request",
            "running",
            f"Asking IGRIS to decompose mission. signals={signals}",
            capability_signals=signals,
            original_goal=_safe_redact(config.goal),
        )

        # --- 1. Local short-prompt attempt ---
        short_prompt = self._DECOMP_SHORT_PROMPT.format(
            goal=_safe_redact(config.goal),
            signals=signals,
        )
        result = self.backend.run_reasoning(
            short_prompt,
            max_steps=15,
            initial_context=context,
            timeout=config.reasoning_timeout_seconds,
        )
        raw = _safe_redact(
            result.get("final_summary") or result.get("output") or ""
        )
        decomposition: Dict[str, Any] = {}
        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                decomposition = json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            decomposition = {}

        fields_missing = [f for f in DECOMPOSITION_REQUIRED_FIELDS if f not in decomposition]

        if not fields_missing:
            # Local reasoning succeeded
            decomposition["generated_by"] = "local_reasoning"
        else:
            prefer_deterministic = (
                str(os.getenv("IGRIS_PREFER_DETERMINISTIC_DECOMPOSITION", "true")).strip().lower() != "false"
            )
            if prefer_deterministic and self._goal_needs_preflight_decomposition(config.goal):
                decomposition = self._deterministic_decompose_fallback(config.goal, signals)
                fields_missing = []
            else:
                # --- 2. API helper attempt ---
                api_result = self._api_helper_decompose(run, config, signals)
                if api_result is not None:
                    decomposition = api_result
                    fields_missing = [f for f in DECOMPOSITION_REQUIRED_FIELDS if f not in decomposition]
                else:
                    # --- 3. Deterministic fallback ---
                    decomposition = self._deterministic_decompose_fallback(config.goal, signals)
                    fields_missing = []

        fields_present = [f for f in DECOMPOSITION_REQUIRED_FIELDS if f in decomposition]
        fields_missing_final = [f for f in DECOMPOSITION_REQUIRED_FIELDS if f not in decomposition]
        decomposition["_fields_present"] = fields_present
        decomposition["_fields_missing"] = fields_missing_final
        decomposition["_capability_signals"] = signals

        run.add(
            "decomposition_response",
            "success" if not fields_missing_final else "fallback",
            (
                f"IGRIS decomposition generated via {decomposition.get('generated_by','unknown')}. "
                f"present={fields_present} missing={fields_missing_final}"
            ),
            fields_present=fields_present,
            fields_missing=fields_missing_final,
            generated_by=decomposition.get("generated_by", "unknown"),
        )
        run.decomposition = decomposition
        return decomposition

    def _api_helper_decompose(
        self,
        run: "SupervisorRun",
        config: "RankSupervisorConfig",
        signals: Dict[str, int],
    ) -> Optional[Dict[str, Any]]:
        """Try to obtain a decomposition from the API helper.

        Returns a decomposition dict with generated_by='api_helper' on success,
        or None if the helper is not available, budget is exhausted, or the
        response is invalid.
        """
        # Budget check
        if run.api_escalations_used >= config.max_api_escalations_per_run:
            return None

        if not self.backend.api_helper_is_configured():
            run.add(
                "decomposition_api",
                "not_configured",
                "API helper not configured; skipping decomposition escalation.",
            )
            return None

        packet: Dict[str, Any] = {
            "task": "decomposition",
            "goal": _safe_redact(config.goal),
            "signals": signals,
            "run_id": run.run_id,
            "decomposition_guidance": (
                "Prefer 4-8 atomic sub-missions over 2-3 large ones. "
                "Each sub-mission should touch at most 1-2 files and be implementable "
                "in fewer than 40 reasoning steps. Include concrete file paths and "
                "function names in each goal. The first sub-mission must be "
                "self-contained with no dependencies on later ones."
            ),
        }
        run.add(
            "decomposition_api_request",
            "running",
            "Calling API helper for decomposition.",
        )
        api_result = self.backend.call_api_helper(
            packet,
            model=config.api_helper_model,
            max_tokens=512,
            timeout=45,
        )
        run.api_escalations_used += 1

        if not api_result.success:
            run.add(
                "decomposition_api_response",
                "failure",
                f"API helper decomposition failed: {_safe_redact(api_result.error)}",
            )
            return None

        # Parse response
        resp: Dict[str, Any] = {}
        try:
            resp = json.loads(api_result.output)
        except (json.JSONDecodeError, ValueError):
            pass

        why = resp.get("why_too_large", "")
        subs = resp.get("sub_missions")
        first = resp.get("first_sub_mission", "")

        if (
            why and isinstance(why, str)
            and subs and isinstance(subs, list) and len(subs) > 0
            and isinstance(first, str)
        ):
            decomp: Dict[str, Any] = {
                "why_too_large": _safe_redact(why),
                "sub_missions": subs,
                "first_sub_mission": _safe_redact(first),
                "human_approval_required": bool(resp.get("human_approval_required", True)),
                "generated_by": "api_helper",
            }
            run.add(
                "decomposition_api_response",
                "success",
                "API helper returned valid decomposition.",
            )
            return decomp

        run.add(
            "decomposition_api_response",
            "partial",
            "API helper returned incomplete decomposition; falling back.",
        )
        return None

    @staticmethod
    def _deterministic_decompose_fallback(
        goal: str,
        signals: Dict[str, int],
    ) -> Dict[str, Any]:
        """Always produce a syntactically complete decomposition from the goal text.

        Parsing strategy (in order of priority):
        1. Numbered/bulleted list items in the goal (\\n- / \\n* / \\n1. / \\n2. etc.)
        2. Semicolon-separated clauses (;)
        3. Semantic split: if goal mentions endpoint/API → 2 sub-missions
           (backend implementation + test coverage). Never split on '.' or ','
           because those are sentence/decimal separators, not list boundaries.
        4. Last resort: treat the entire goal as a single sub-mission.
        """

        def _infer_risk(text: str) -> str:
            t = text.lower()
            if any(k in t for k in ("zombie", "orphan", "delete", "destroy", "drop")):
                return "high"
            if any(k in t for k in ("report", "badge", "endpoint", "api", "dashboard")):
                return "medium"
            return "low"

        def _infer_file_scopes(text: str) -> List[str]:
            t = text.lower()
            if any(k in t for k in ("endpoint", "api", "server", "route")):
                return ["igris/web/server.py", "igris/core/"]
            if any(k in t for k in ("dashboard", "badge", "ui", "card")):
                return ["igris/web/static/**", "igris/web/templates/**"]
            # Memory Tree specific layers — most precise match first
            if any(k in t for k in ("memory_content_store", "content_store")):
                return ["igris/core/memory_content_store.py", "tests/test_memory_content_store.py"]
            if any(k in t for k in ("memory_scorer", "memoryscorer")):
                return ["igris/core/memory_scorer.py", "tests/test_memory_scorer.py"]
            if any(k in t for k in ("topic_tree", "topictree", "global_digest", "globaldigest")):
                return [
                    "igris/core/memory_topic_tree.py",
                    "igris/core/memory_global_digest.py",
                    "tests/test_memory_topic_tree.py",
                ]
            if any(k in t for k in ("memory tree", "memory_tree", "memorytree",
                                     "memory chunker", "memory_chunker")):
                return [
                    "igris/core/memory_chunker.py",
                    "igris/core/memory_graph.py",
                    "igris/core/",
                    "tests/",
                ]
            # Broader memory/hierarchy patterns
            if any(k in t for k in ("memory", "chunk", "score", "topic", "global", "hierarchy")):
                return ["igris/core/", "tests/"]
            if "test" in t:
                return ["tests/"]
            if any(k in t for k in ("supervisor", "repair")):
                return ["igris/core/self_repair_supervisor.py"]
            # #913: fallback is now igris/core/ + tests/ instead of igris/**
            # A broad igris/** scope caused no_diff_repair loops because Igris
            # could not determine which file to edit.
            return ["igris/core/", "tests/"]

        def _infer_test_targets(text: str) -> List[str]:
            # Extract explicit test file paths like tests/test_foo.py
            matches = re.findall(r"tests/[\w/]+\.py", text)
            if matches:
                return matches
            if "test" in text.lower():
                return ["tests/"]
            return []

        def _make_sub(
            title: str,
            goal_text: str,
            *,
            explicit_file_scopes: Optional[List[str]] = None,
            explicit_acceptance_criteria: Optional[List[str]] = None,
            explicit_tests: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            """Build a sub-mission dict.

            When explicit_* params are provided they take precedence over inference.
            Anti-loop guard (#913): if both scopes and criteria are generic the sub-
            mission is flagged human_approval_required=True so a human can refine them
            before Igris runs — preventing the no_diff_repair → fallback loop.
            """
            safe = _safe_redact(goal_text)
            scopes = (
                explicit_file_scopes
                if explicit_file_scopes is not None
                else _infer_file_scopes(goal_text)
            )
            criteria = (
                explicit_acceptance_criteria
                if explicit_acceptance_criteria is not None
                else [f"{title} implemented and validated"]
            )
            tests = (
                explicit_tests
                if explicit_tests is not None
                else _infer_test_targets(goal_text)
            )
            # Anti-loop guard: broad scope + generic criterion → require human review.
            # This prevents Igris from entering a no_diff_repair loop where it cannot
            # determine which file to edit or how to verify success.
            _broad_scopes = {"igris/core/", "tests/", "igris/**"}
            _scope_is_broad = set(scopes) <= _broad_scopes
            _criteria_generic = criteria == [f"{title} implemented and validated"]
            human_approval = _scope_is_broad and _criteria_generic
            return {
                "title": title[:60],
                "goal": safe,
                "dependencies": [],
                "acceptance_criteria": criteria,
                "allowed_file_scopes": scopes,
                "tests": tests,
                "risk_level": _infer_risk(goal_text),
                "human_approval_required": human_approval,
            }

        safe_goal = _safe_redact(str(goal))

        # --- Strategy 1: explicit bulleted/numbered list items ---
        bullet_parts = re.split(r"\n\s*(?:[-*]|\d+\.)\s+", safe_goal)
        # Only use bullet split if it produced ≥2 meaningful items (each ≥30 chars)
        bullet_items = [p.strip() for p in bullet_parts if len(p.strip()) >= 30]
        if len(bullet_items) >= 2:
            components = bullet_items[:4]
            sub_missions = [_make_sub(c[:50].capitalize(), c) for c in components]
            why = (
                f"Mission contains {len(components)} explicit list items requiring separate "
                f"reasoning passes. Signals: {signals}"
            )
            return {
                "why_too_large": _safe_redact(why),
                "sub_missions": sub_missions,
                "first_sub_mission": sub_missions[0]["title"],
                "human_approval_required": True,
                "generated_by": "deterministic_fallback",
            }

        # --- Strategy 2: semicolon-separated clauses ---
        semi_parts = [p.strip() for p in safe_goal.split(";") if len(p.strip()) >= 30]
        if len(semi_parts) >= 2:
            components = semi_parts[:4]
            sub_missions = [_make_sub(c[:50].capitalize(), c) for c in components]
            why = (
                f"Mission contains {len(components)} semicolon-delimited components. "
                f"Signals: {signals}"
            )
            return {
                "why_too_large": _safe_redact(why),
                "sub_missions": sub_missions,
                "first_sub_mission": sub_missions[0]["title"],
                "human_approval_required": True,
                "generated_by": "deterministic_fallback",
            }

        # --- Strategy 3: semantic split for endpoint/API missions ---
        gl = safe_goal.lower()
        is_endpoint_mission = any(k in gl for k in ("endpoint", "/api/", "get /", "post /", "put /"))
        has_test_requirement = "test" in gl
        if is_endpoint_mission:
            # Sub-mission 1: backend implementation
            impl_goal = (
                f"{safe_goal.rstrip('.')}. "
                "Focus on implementing the backend endpoint logic only. "
                "Do not write tests in this sub-mission."
            )
            # Sub-mission 2: test coverage (only if tests mentioned)
            sub_missions = [_make_sub("Backend endpoint implementation", impl_goal)]
            if has_test_requirement:
                test_files = _infer_test_targets(safe_goal) or ["tests/"]
                test_goal = (
                    f"Add comprehensive test coverage for: {safe_goal[:200]}. "
                    f"Write tests in {', '.join(test_files)}. "
                    "Assume the endpoint is already implemented."
                )
                sub_missions.append(_make_sub("Test coverage", test_goal))
            return {
                "why_too_large": _safe_redact(
                    f"Endpoint mission split into {len(sub_missions)} semantic sub-missions "
                    f"(implementation + tests). Signals: {signals}"
                ),
                "sub_missions": sub_missions,
                "first_sub_mission": sub_missions[0]["title"],
                "human_approval_required": True,
                "generated_by": "deterministic_fallback",
            }

        # --- Strategy 4: semantic split for memory-tree hierarchy missions ---
        # #913: Rewrote from 4 generic sub-missions to 5 explicit, bounded steps.
        # Each step has: precise file_scopes, verifiable acceptance_criteria, explicit tests.
        # Step 0 is read-only (architecture plan) and gates the implementation steps.
        # human_approval_required=True so a human confirms the plan before Igris runs code.
        is_memory_tree_mission = (
            "memory tree" in gl
            and any(k in gl for k in ("chunk", "score", "topic", "global", "pipeline", "hierarchy"))
        )
        if is_memory_tree_mission:
            sub_missions = [
                # Step 0 — read-only architecture plan (no production code written)
                _make_sub(
                    "MemoryTree Step 0: architecture verification",
                    (
                        "Read-only architecture pass for Memory Tree hierarchy (issue #536). "
                        "Read igris/core/memory_chunker.py, igris/core/memory_graph.py, "
                        "igris/core/memory_content_store.py (if it exists). "
                        "Identify which of the 4 layers (ContentStore, Scorer, TopicTree, GlobalDigest) "
                        "are missing or stub-only. "
                        "Output a JSON plan to .igris/memory_tree_plan.json listing each layer with: "
                        "layer_name, target_file, status (missing|stub|complete), "
                        "first_function_to_implement. Do not write any production code."
                    ),
                    explicit_file_scopes=[
                        "igris/core/memory_chunker.py",
                        "igris/core/memory_graph.py",
                        "igris/core/memory_content_store.py",
                        ".igris/memory_tree_plan.json",
                    ],
                    explicit_acceptance_criteria=[
                        ".igris/memory_tree_plan.json exists and is valid JSON",
                        "Each layer entry has status in {missing, stub, complete}",
                        "No production code written or modified in this step",
                    ],
                    explicit_tests=[],
                ),
                # Step 1 — MemoryContentStore: raw storage layer
                _make_sub(
                    "MemoryTree Step 1: MemoryContentStore",
                    (
                        "Implement igris/core/memory_content_store.py for Memory Tree hierarchy (issue #536). "
                        "Create MemoryContentStore class with methods: "
                        "store(chunk_id: str, content: str, metadata: dict) -> None, "
                        "retrieve(chunk_id: str) -> dict, "
                        "list_ids() -> List[str]. "
                        "Use SQLite via the pattern in igris/core/memory_graph.py — no new dependencies. "
                        "Write tests/test_memory_content_store.py with ≥3 unit tests covering "
                        "store, retrieve, list_ids. All tests must pass with: "
                        "pytest tests/test_memory_content_store.py"
                    ),
                    explicit_file_scopes=[
                        "igris/core/memory_content_store.py",
                        "tests/test_memory_content_store.py",
                    ],
                    explicit_acceptance_criteria=[
                        "igris/core/memory_content_store.py exists and is not a stub",
                        "MemoryContentStore.store(), .retrieve(), .list_ids() all implemented",
                        "tests/test_memory_content_store.py has ≥3 test functions",
                        "pytest tests/test_memory_content_store.py exits with code 0",
                    ],
                    explicit_tests=["tests/test_memory_content_store.py"],
                ),
                # Step 2 — MemoryScorer: keyword-based relevance scoring
                _make_sub(
                    "MemoryTree Step 2: MemoryScorer",
                    (
                        "Implement igris/core/memory_scorer.py for Memory Tree hierarchy (issue #536). "
                        "Create MemoryScorer class with methods: "
                        "score(chunk_id: str, query: str) -> float, "
                        "rank(chunk_ids: List[str], query: str) -> List[Tuple[str, float]] "
                        "(sorted descending by score). "
                        "Use keyword overlap or TF-IDF — no external ML dependencies. "
                        "Write tests/test_memory_scorer.py with ≥3 unit tests verifying "
                        "score range [0,1] and rank ordering. All tests must pass with: "
                        "pytest tests/test_memory_scorer.py"
                    ),
                    explicit_file_scopes=[
                        "igris/core/memory_scorer.py",
                        "tests/test_memory_scorer.py",
                    ],
                    explicit_acceptance_criteria=[
                        "igris/core/memory_scorer.py exists and is not a stub",
                        "MemoryScorer.score() returns float in [0.0, 1.0]",
                        "MemoryScorer.rank() returns list sorted by score descending",
                        "tests/test_memory_scorer.py has ≥3 test functions",
                        "pytest tests/test_memory_scorer.py exits with code 0",
                    ],
                    explicit_tests=["tests/test_memory_scorer.py"],
                ),
                # Step 3 — TopicTree + GlobalDigest: grouping and synthesis
                _make_sub(
                    "MemoryTree Step 3: TopicTree and GlobalDigest",
                    (
                        "Implement igris/core/memory_topic_tree.py for Memory Tree hierarchy (issue #536). "
                        "Create TopicTree class with: "
                        "add_chunk(chunk_id: str, topic_key: str) -> None, "
                        "get_topic(topic_key: str) -> List[str], "
                        "list_topics() -> List[str]. "
                        "Create GlobalDigest class (same file or igris/core/memory_global_digest.py) with: "
                        "summarize(topic_keys: List[str]) -> str, "
                        "refresh() -> None. "
                        "Write tests/test_memory_topic_tree.py with ≥4 unit tests. "
                        "All tests must pass with: pytest tests/test_memory_topic_tree.py"
                    ),
                    explicit_file_scopes=[
                        "igris/core/memory_topic_tree.py",
                        "igris/core/memory_global_digest.py",
                        "tests/test_memory_topic_tree.py",
                    ],
                    explicit_acceptance_criteria=[
                        "TopicTree.add_chunk(), .get_topic(), .list_topics() all implemented",
                        "GlobalDigest.summarize() and .refresh() both implemented",
                        "tests/test_memory_topic_tree.py has ≥4 test functions",
                        "pytest tests/test_memory_topic_tree.py exits with code 0",
                    ],
                    explicit_tests=["tests/test_memory_topic_tree.py"],
                ),
                # Step 4 — Retrieval integration in memory_graph.py + feature flag
                _make_sub(
                    "MemoryTree Step 4: retrieval integration",
                    (
                        "Integrate Memory Tree layers into igris/core/memory_graph.py (issue #536). "
                        "Add retrieve_tree(query: str, top_k: int = 5) -> List[dict] method that: "
                        "(1) fetches chunk_ids from MemoryContentStore.list_ids(), "
                        "(2) scores via MemoryScorer.rank(chunk_ids, query), "
                        "(3) groups top results by topic via TopicTree, "
                        "(4) returns top_k results as list of dicts with keys: "
                        "chunk_id, content, score, topic. "
                        "Gate the method behind IGRIS_MEMORY_TREE_ENABLED env var (default '0'): "
                        "when disabled return [] immediately (no exception). "
                        "Write tests/test_memory_tree_integration.py with ≥3 integration tests "
                        "covering: enabled path returns results, disabled path returns [], "
                        "result dicts have required keys. "
                        "All tests must pass with: pytest tests/test_memory_tree_integration.py"
                    ),
                    explicit_file_scopes=[
                        "igris/core/memory_graph.py",
                        "tests/test_memory_tree_integration.py",
                    ],
                    explicit_acceptance_criteria=[
                        "memory_graph.py has retrieve_tree() method",
                        "IGRIS_MEMORY_TREE_ENABLED=0 → retrieve_tree() returns []",
                        "IGRIS_MEMORY_TREE_ENABLED=1 → retrieve_tree() returns list of dicts",
                        "Each result dict has keys: chunk_id, content, score, topic",
                        "tests/test_memory_tree_integration.py has ≥3 test functions",
                        "pytest tests/test_memory_tree_integration.py exits with code 0",
                    ],
                    explicit_tests=["tests/test_memory_tree_integration.py"],
                ),
            ]
            return {
                "why_too_large": _safe_redact(
                    f"Memory Tree hierarchy mission requires staged implementation across 5 bounded steps "
                    f"(Step 0: read-only architecture plan → Steps 1-4: ContentStore, Scorer, "
                    f"TopicTree+GlobalDigest, retrieval integration). Signals: {signals}"
                ),
                "sub_missions": sub_missions,
                "first_sub_mission": sub_missions[0]["title"],
                "human_approval_required": True,
                "generated_by": "deterministic_fallback",
            }

        # --- Strategy 5: single sub-mission (whole goal, scoped) ---
        sub_missions = [_make_sub("Complete mission", safe_goal)]
        return {
            "why_too_large": _safe_redact(
                f"Mission could not be structurally decomposed; presented as single bounded "
                f"sub-mission for focused retry. Signals: {signals}"
            ),
            "sub_missions": sub_missions,
            "first_sub_mission": sub_missions[0]["title"],
            "human_approval_required": True,
            "generated_by": "deterministic_fallback",
        }

    _DESTRUCTIVE_KEYWORDS = frozenset({
        "drop", "delete", "destroy", "wipe", "format", "truncate",
        "rm -rf", "reset --hard", "force push", "force-push",
        "sudo", "kubectl apply", "terraform apply", "deploy production",
        "database migration", "data migration",
    })

    # Maximum nesting level for auto-chained sub-missions.
    # At this depth the policy must NOT create GitHub issues — doing so would
    # produce orphaned issues that can never be auto-run.
    _MAX_AUTOCHAIN_DEPTH: int = 2

    @staticmethod
    def _goal_needs_preflight_decomposition(goal: str) -> bool:
        text = (goal or "").lower()
        strong_markers = (
            "memory tree",
            "hierarchy",
            "pipeline",
            "roadmap",
            "phase-2bis",
            "chunk",
            "topic",
            "global",
            "decompose",
        )
        score = sum(1 for marker in strong_markers if marker in text)
        return score >= 3 or len(text) >= 220

    @staticmethod
    def _decomposition_policy(
        decomposition: Dict[str, Any],
        config: "RankSupervisorConfig",
    ) -> str:
        """Decide how to handle a valid decomposition.

        Returns one of:
          "auto_create_subissues"       — safe, GitHub enabled, create issues automatically
          "request_human_approval"      — unsafe or GitHub disabled
          "block_unsafe_decomposition"  — secret/destructive content detected
        """
        # Require valid structure
        fields_missing = decomposition.get("_fields_missing", [])
        sub_missions = decomposition.get("sub_missions") or []
        if fields_missing or not sub_missions:
            return "request_human_approval"

        # Require explicit opt-in to autonomous sub-issue creation
        if not config.allow_auto_subissues or config.dry_run:
            return "request_human_approval"

        # Do not block safe sub-issue creation at max autochain depth.
        # Depth limits are enforced by _autorun_guards for child execution, so
        # decomposition can still progress without requiring a manual approval
        # deadlock on large missions.

        # Check for destructive/secret/dangerous content
        all_text = " ".join([
            str(decomposition.get("why_too_large", "")),
            str(decomposition.get("first_sub_mission", "")),
            *[
                " ".join([
                    str(s.get("title", "")),
                    str(s.get("goal", "")),
                    *[str(c) for c in (s.get("acceptance_criteria") or [])],
                ])
                for s in sub_missions
            ],
        ]).lower()

        # Check for secret patterns (raw or already-redacted by _safe_redact)
        secret_re = re.compile(r"sk-[A-Za-z0-9_\-]{3,}[A-Za-z0-9]{10,}|bearer\s+\S{20,}", re.I)
        if secret_re.search(all_text) or "***redacted***" in all_text:
            return "block_unsafe_decomposition"

        # Check for destructive keywords
        if any(kw in all_text for kw in SelfRepairSupervisor._DESTRUCTIVE_KEYWORDS):
            return "request_human_approval"

        return "auto_create_subissues"

    def _auto_create_subissues(
        self,
        run: SupervisorRun,
        config: RankSupervisorConfig,
        decomposition: Dict[str, Any],
        triggering_signal: str,
    ) -> List[str]:
        """Create one GitHub issue per sub_mission and return list of created URLs.

        Each issue body includes parent run context, generated_by, risk, scopes.
        After creating all issues, posts a summary comment on the parent issue (if
        a parent URL can be inferred from config.goal or run.report).
        """
        sub_missions = decomposition.get("sub_missions") or []
        generated_by = decomposition.get("generated_by", "unknown")
        why_too_large = _safe_redact(str(decomposition.get("why_too_large", "")))
        first_sub = decomposition.get("first_sub_mission", "")

        # Collect parent issue labels to propagate to sub-issues (roadmap, P*, phase-*).
        # This ensures the watchdog can discover sub-issues the same way it finds parent
        # roadmap issues.  Best-effort: if we can't read the parent labels, we proceed
        # without them rather than blocking sub-issue creation.
        _parent_inherit_labels: List[str] = []
        try:
            import re as _re, subprocess as _subp2
            _parent_num_m = _re.search(r"#(\d+)", config.goal or "")
            if _parent_num_m:
                _parent_num = int(_parent_num_m.group(1))
                _pl = _subp2.run(
                    ["gh", "issue", "view", str(_parent_num), "--json", "labels"],
                    capture_output=True, text=True, cwd=self.project_root, timeout=15,
                )
                if _pl.returncode == 0:
                    import json as _json2
                    _raw_labels = _json2.loads(_pl.stdout or "{}").get("labels", [])
                    for _lbl in _raw_labels:
                        _n = (_lbl.get("name") or "").lower()
                        if _n in ("roadmap", "created-by:igris") or _n.startswith("p") and len(_n) == 2 and _n[1].isdigit() or _n.startswith("phase-"):
                            _parent_inherit_labels.append(_lbl.get("name", _n))
        except Exception:
            pass  # Label propagation is best-effort

        created_urls: List[str] = []
        run.add(
            "subissue_creation",
            "running",
            f"Creating {len(sub_missions)} sub-issue(s) from decomposition.",
            count=len(sub_missions),
            generated_by=generated_by,
        )

        # Build a set of existing open issue titles to deduplicate sub-missions.
        # Fixes #613: IGRIS was creating identical sub-missions on every decomposition.
        existing_open_titles: set = set()
        try:
            import subprocess as _subp
            _existing = _subp.run(
                ["gh", "issue", "list", "--state", "open", "--limit", "50",
                 "--json", "number,title"],
                capture_output=True, text=True, cwd=self.project_root, timeout=20,
            )
            if _existing.returncode == 0:
                import json as _json
                for _issue in _json.loads(_existing.stdout or "[]"):
                    existing_open_titles.add((_issue.get("title") or "").lower().strip())
        except Exception:
            pass  # Dedup is best-effort; if it fails, allow creation to proceed

        for i, sub in enumerate(sub_missions):
            title = _safe_redact(str(sub.get("title", f"Sub-task {i+1}")))
            goal_text = _safe_redact(str(sub.get("goal", "")))
            risk = str(sub.get("risk_level", "medium"))
            scopes = sub.get("allowed_file_scopes") or []
            tests = sub.get("tests") or []
            criteria = sub.get("acceptance_criteria") or []
            deps = sub.get("dependencies") or []

            # Dedup: skip sub-mission if an open issue with same title already exists
            if title.lower().strip() in existing_open_titles:
                # Find existing URL to include in created_urls so autochain works
                try:
                    _found = _subp.run(
                        ["gh", "issue", "list", "--state", "open", "--search", title,
                         "--json", "number,title", "--limit", "5"],
                        capture_output=True, text=True, cwd=self.project_root, timeout=20,
                    )
                    if _found.returncode == 0:
                        for _fi in _json.loads(_found.stdout or "[]"):
                            if (_fi.get("title") or "").lower().strip() == title.lower().strip():
                                _repo_url = "https://github.com/Solarfox88/IGRIS_GPT"
                                _existing_url = f"{_repo_url}/issues/{_fi['number']}"
                                created_urls.append(_existing_url)
                                run.add(
                                    "subissue_dedup",
                                    "skipped",
                                    f"Sub-mission {i+1} already exists: {title}",
                                    index=i + 1,
                                    title=title,
                                    url=_existing_url,
                                    reason="dedup:title_match",
                                )
                                break
                except Exception:
                    pass
                continue

            scopes_md = "\n".join(f"- `{s}`" for s in scopes) if scopes else "_not specified_"
            tests_md = "\n".join(f"- `{t}`" for t in tests) if tests else "_not specified_"
            criteria_md = "\n".join(f"- {c}" for c in criteria) if criteria else "_not specified_"
            deps_md = ", ".join(deps) if deps else "none"

            body = (
                f"## Sub-mission {i+1} of {len(sub_missions)}\n\n"
                f"**Goal:** {goal_text}\n\n"
                f"**Risk level:** {risk}\n"
                f"**Dependencies:** {deps_md}\n\n"
                f"### Acceptance criteria\n{criteria_md}\n\n"
                f"### File scopes\n{scopes_md}\n\n"
                f"### Test targets\n{tests_md}\n\n"
                f"---\n"
                f"**Parent run:** `{run.run_id}` (rank `{run.rank_id}`)\n"
                f"**Decomposition source:** `{generated_by}`\n"
                f"**Trigger signal:** `{triggering_signal}`\n"
                f"**Why original mission was too large:** {why_too_large}\n"
                f"**Original goal:** {_safe_redact(config.goal)}\n"
            )

            result = self.backend.create_issue(title, body)
            if result.success:
                url = result.output.strip()
                created_urls.append(url)
                run.add(
                    "subissue_created",
                    "success",
                    f"Created sub-issue: {title}",
                    index=i + 1,
                    title=title,
                    url=url,
                    risk=risk,
                )
                # Propagate parent roadmap/priority/phase labels so the watchdog
                # can discover and schedule sub-issues automatically.
                # Always add "no-decompose" so the watchdog knows this is a leaf
                # sub-issue that must be implemented directly, not decomposed again.
                _sub_labels = list(_parent_inherit_labels)
                if "no-decompose" not in _sub_labels:
                    _sub_labels.append("no-decompose")
                # Also add depends-on-NNN labels for dependencies between sub-issues
                for _dep in deps:
                    # deps may be issue URLs or "Sub-task N" style references
                    import re as _re2
                    _dep_num = _re2.search(r"#?(\d+)", str(_dep))
                    if _dep_num:
                        _sub_labels.append(f"depends-on-{_dep_num.group(1)}")
                try:
                    import subprocess as _subp3
                    _subp3.run(
                        ["gh", "issue", "edit", url, "--add-label",
                         ",".join(_sub_labels)],
                        capture_output=True, text=True,
                        cwd=self.project_root, timeout=20,
                    )
                except Exception:
                    pass  # Label application is best-effort
            else:
                run.add(
                    "subissue_created",
                    "failure",
                    f"Failed to create sub-issue: {title}",
                    index=i + 1,
                    title=title,
                    error=_safe_redact(result.error),
                )

        # Post summary comment on parent issue if we can infer the URL.
        parent_url = self._infer_parent_issue_url(config.goal)
        if parent_url and created_urls:
            sub_list = "\n".join(
                f"- {url} — {sub.get('title','?')}"
                for url, sub in zip(created_urls, sub_missions)
            )
            comment = (
                f"## Decomposition sub-issues created\n\n"
                f"Run `{run.run_id}` produced {len(created_urls)} sub-issue(s) "
                f"via `{generated_by}`:\n\n{sub_list}\n\n"
                f"First sub-mission to run: **{_safe_redact(first_sub)}**"
            )
            comment_result = self.backend.update_issue(parent_url, comment)
            run.add(
                "parent_issue_updated",
                "success" if comment_result.success else "failure",
                "Posted sub-issue summary to parent issue.",
                parent_url=parent_url,
                sub_count=len(created_urls),
            )

        run.add(
            "subissue_creation",
            "success" if created_urls else "failure",
            f"Sub-issue creation complete. Created {len(created_urls)}/{len(sub_missions)}.",
            created_count=len(created_urls),
            total=len(sub_missions),
            urls=created_urls,
        )
        return created_urls

    @staticmethod
    def _infer_parent_issue_url(goal: str) -> Optional[str]:
        """Extract a GitHub issue URL from the goal string if present."""
        m = re.search(r"https://github\.com/[^\s\)\"']+/issues/\d+", goal)
        return m.group(0) if m else None

    @staticmethod
    def _autorun_guards(
        run: "SupervisorRun",
        config: "RankSupervisorConfig",
        decomposition: Dict[str, Any],
        created_urls: List[str],
    ) -> Tuple[bool, str]:
        """Check all guards before auto-queuing a child run.

        Returns (ok, reason) — if ok is False, reason explains why autorun is skipped.
        """
        if not config.allow_auto_subissues:
            return False, "allow_auto_subissues=False"
        if config.dry_run:
            return False, "dry_run=True"
        # Cascade depth guard: stop auto-chaining after _MAX_AUTOCHAIN_DEPTH levels
        if config.autochain_depth >= SelfRepairSupervisor._MAX_AUTOCHAIN_DEPTH:
            return False, f"max_autochain_depth: depth={config.autochain_depth}>={SelfRepairSupervisor._MAX_AUTOCHAIN_DEPTH}"
        if not created_urls:
            return False, "no_sub_issue_urls"
        approval = decomposition.get("approval_status", "")
        if approval != "auto_approved_by_policy":
            return False, f"approval_status={approval!r} (not auto_approved)"
        if decomposition.get("decomposition_cycle_detected"):
            return False, "decomposition_cycle_detected"

        # Anti-loop: first sub-issue must not be the same as any URL referenced in parent goal
        first_url = created_urls[0]
        parent_goal_lower = config.goal.lower()
        # Extract issue numbers from first_url and goal
        import re as _re
        first_num_m = _re.search(r"/issues/(\d+)", first_url)
        if first_num_m:
            first_num = first_num_m.group(1)
            if f"/issues/{first_num}" in parent_goal_lower or f"#{first_num}" in parent_goal_lower:
                return False, f"anti_loop: sub-issue #{first_num} matches parent goal"

        # Check if a run for this sub-issue URL is already active
        from igris.core.self_repair_supervisor import RUN_LOCK, RUN_STORE
        with RUN_LOCK:
            active_runs = list(RUN_STORE.values())
        for r in active_runs:
            if r.run_id == run.run_id:
                continue
            if r.status in ("running", "cancelling"):
                if first_url in r.goal or first_url in str(r.report):
                    return False, f"sub_issue_already_running: run_id={r.run_id}"

        # Budget check (0 means unlimited)
        max_cost_per_run = SelfRepairSupervisor._get_max_cost_per_run()
        if max_cost_per_run > 0 and run.execution_budget_used_usd >= max_cost_per_run:
            return False, f"budget_exceeded: {run.execution_budget_used_usd:.4f}>={max_cost_per_run:.4f}"

        return True, ""

    def _autorun_first_subissue(
        self,
        run: "SupervisorRun",
        config: "RankSupervisorConfig",
        decomposition: Dict[str, Any],
        created_urls: List[str],
        triggering_signal: str,
    ) -> Optional[str]:
        """Fetch the first sub-issue from GitHub and queue a child supervised run.

        Returns the child run_id on success, None if skipped or failed.
        """
        from igris.core.self_repair_supervisor import start_supervised_rank_async
        import re as _re

        ok, skip_reason = self._autorun_guards(run, config, decomposition, created_urls)
        if not ok:
            run.add(
                "submission_autorun_skipped",
                "skipped",
                f"Auto-run skipped: {skip_reason}",
                reason=skip_reason,
            )
            run.autorun_skipped_reason = skip_reason
            run.report.update({
                "autorun_policy": "skipped",
                "autorun_skipped_reason": skip_reason,
            })
            return None

        first_url = created_urls[0]
        run.add(
            "submission_autorun_queued",
            "running",
            f"Fetching sub-issue to prepare child run: {_safe_redact(first_url)}",
            sub_issue_url=_safe_redact(first_url),
        )

        # Fetch sub-issue data from GitHub
        fetch_result = self.backend.fetch_issue(first_url)
        if not fetch_result.success:
            reason = f"fetch_issue_failed: {_safe_redact(fetch_result.error)[:120]}"
            run.add("submission_autorun_skipped", "failure", reason, sub_issue_url=_safe_redact(first_url))
            run.autorun_skipped_reason = reason
            run.report.update({"autorun_policy": "skipped", "autorun_skipped_reason": reason})
            return None

        try:
            issue_data = json.loads(fetch_result.output or "{}")
        except json.JSONDecodeError:
            issue_data = {}

        issue_title = _safe_redact(str(issue_data.get("title", "") or ""))
        issue_body = _safe_redact(str(issue_data.get("body", "") or ""))
        issue_number = issue_data.get("number", "")

        # Build goal from sub-issue title + body (first 2000 chars of body)
        body_excerpt = issue_body[:2000].strip()
        child_goal = f"{issue_title}\n\n{body_excerpt}" if body_excerpt else issue_title
        if not child_goal.strip():
            child_goal = decomposition.get("first_sub_mission", first_url)

        # Derive child rank_id from parent + issue number
        child_rank_id = f"{run.rank_id}-sub{issue_number}" if issue_number else f"{run.rank_id}-sub1"

        # Inherit parent config but override goal and rank_id
        child_data: Dict[str, Any] = {
            "goal": child_goal,
            "rank_id": child_rank_id,
            "dry_run": False,
            "max_rank_attempts": config.max_rank_attempts,
            "max_repair_cycles": config.max_repair_cycles,
            "allow_github_pr": config.allow_github_pr,
            "allow_merge_if_green": config.allow_merge_if_green,
            "service_restart_command": config.service_restart_command,
            "required_smoke_endpoints": list(config.required_smoke_endpoints),
            "test_timeout_seconds": config.test_timeout_seconds,
            "test_hard_cap_seconds": config.test_hard_cap_seconds,
            "reasoning_timeout_seconds": config.reasoning_timeout_seconds,
            "allow_api_escalation": config.allow_api_escalation,
            "max_api_escalations_per_run": config.max_api_escalations_per_run,
            "max_api_budget_usd": config.max_api_budget_usd,
            "max_tokens_per_escalation": config.max_tokens_per_escalation,
            "api_helper_model": config.api_helper_model,
            "enable_mission_planning": config.enable_mission_planning,
            "allow_auto_subissues": config.allow_auto_subissues,
            "enable_semantic_gate": config.enable_semantic_gate,
            "api_helper_mode": config.api_helper_mode,
            # Increment depth so grandchild hits max_autochain_depth guard
            "autochain_depth": config.autochain_depth + 1,
            # Mark so child knows its parent
            "_parent_run_id": run.run_id,
            "_parent_sub_issue_url": first_url,
            "_parent_triggering_signal": triggering_signal,
        }

        try:
            child_run = start_supervised_rank_async(child_data, project_root=str(self.project_root))
            child_run_id = child_run.run_id
        except Exception as exc:
            reason = f"child_run_start_failed: {_safe_redact(str(exc))[:120]}"
            run.add("submission_autorun_skipped", "failure", reason, sub_issue_url=_safe_redact(first_url))
            run.autorun_skipped_reason = reason
            run.report.update({"autorun_policy": "skipped", "autorun_skipped_reason": reason})
            return None

        run.autorun_child_run_id = child_run_id
        run.autorun_policy = "auto_create_subissues"
        run.add(
            "submission_autorun_run_id",
            "success",
            f"Child run {child_run_id} queued for sub-issue {_safe_redact(first_url)}",
            child_run_id=child_run_id,
            sub_issue_url=_safe_redact(first_url),
            sub_issue_title=issue_title,
            child_rank_id=child_rank_id,
        )
        run.report.update({
            "next_subissue_url": _safe_redact(first_url),
            "next_subissue_number": str(issue_number),
            "autorun_child_run_id": child_run_id,
            "autorun_policy": "auto_create_subissues",
            "autorun_skipped_reason": "",
        })
        return child_run_id


    def _run_decomposed_parallel(
        self,
        sub_goals: List[str],
        base_max_steps: int = 20,
        preferred_profile: Optional[str] = None,
    ) -> List[dict]:
        """Run decomposed sub-goals in parallel and return run_reasoning-like dicts."""
        from igris.core.parallel_task_runner import ParallelTask, ParallelTaskRunner

        tasks = [
            ParallelTask(
                task_id=f"sub_{i}",
                goal=goal,
                max_steps=base_max_steps,
                preferred_profile=preferred_profile,
            )
            for i, goal in enumerate(sub_goals)
        ]
        runner = ParallelTaskRunner(self.project_root, max_concurrent=3)
        parallel_results = runner.run_sync(tasks)
        return [
            pr.result.to_dict() if pr.result is not None else {"status": "error", "error": pr.error}
            for pr in parallel_results
        ]

    def _blocked_decomposition_required(
        self,
        run: SupervisorRun,
        triggering_signal: str,
        detail: str,
        decomposition: Dict[str, Any],
        *,
        config: Optional[RankSupervisorConfig] = None,
        mission_plan: Optional[MissionPlan] = None,
        stage_statuses: Optional[Dict[str, Dict[str, Any]]] = None,
        cleanup_workspace: bool = False,
    ) -> SupervisorRun:
        """Block the run with failure_class='decomposition_required' and attach the
        IGRIS-generated decomposition to the run report and durable storage."""
        run = self._blocked(
            run,
            "decomposition_required",
            detail,
            mission_plan=mission_plan,
            stage_statuses=stage_statuses,
            cleanup_workspace=cleanup_workspace,
        )
        first_sub = _safe_redact(str(decomposition.get("first_sub_mission", "")))

        # Evaluate policy to decide whether to auto-create sub-issues
        # If no config was provided, default to requesting human approval.
        if config is not None:
            policy = self._decomposition_policy(decomposition, config)
        else:
            policy = "request_human_approval"

        run.add(
            "decomposition_policy",
            "evaluated",
            f"Decomposition policy: {policy}",
            policy=policy,
            allow_auto_subissues=config.allow_auto_subissues if config is not None else False,
            allow_github_pr=config.allow_github_pr if config is not None else False,
            dry_run=config.dry_run if config is not None else True,
        )

        created_urls: List[str] = []
        if policy == "auto_create_subissues":
            created_urls = self._auto_create_subissues(run, config, decomposition, triggering_signal)
            if created_urls:
                next_action = f"run:{first_sub}" if first_sub else "queued:first_sub_mission"
            else:
                # All issue creations failed — fall back to manual approval
                next_action = "request_approval:decomposition"
        elif policy == "block_unsafe_decomposition":
            run.add(
                "decomposition_policy",
                "blocked_unsafe",
                "Decomposition contains unsafe content (secret/destructive); human approval required.",
            )
            next_action = "request_approval:decomposition"
        else:  # "request_human_approval"
            next_action = "request_approval:decomposition"

        # Redact any strings inside sub_missions for safety.
        safe_decomposition: Dict[str, Any] = {}
        for k, v in decomposition.items():
            if isinstance(v, str):
                safe_decomposition[k] = _safe_redact(v)
            elif isinstance(v, list) and all(isinstance(i, dict) for i in v):
                safe_decomposition[k] = [
                    {ik: _safe_redact(iv) if isinstance(iv, str) else iv
                     for ik, iv in item.items()}
                    for item in v
                ]
            else:
                safe_decomposition[k] = v
        safe_decomposition["sub_issue_urls"] = created_urls if policy == "auto_create_subissues" else []
        safe_decomposition["policy"] = policy
        safe_decomposition["allow_auto_subissues"] = (
            config.allow_auto_subissues if config is not None else False
        )
        safe_decomposition["next_action"] = next_action
        # Resolve the approval ambiguity: if the policy already auto-approved the
        # decomposition (sub-issues created autonomously), human review is not needed
        # and the original human_approval_required=True from the LLM response is
        # overridden.  For all other policies human_approval_required keeps its
        # original value so callers can gate correctly.
        if policy == "auto_create_subissues" and created_urls:
            safe_decomposition["human_approval_required"] = False
            safe_decomposition["auto_approved_by_policy"] = True
            safe_decomposition["approval_status"] = "auto_approved_by_policy"
        else:
            safe_decomposition.setdefault("auto_approved_by_policy", False)
            safe_decomposition.setdefault("approval_status", "pending_human_approval")
        run.report.update({
            "decomposition_required": True,
            "capability_limit_signal": triggering_signal,
            "next_action": next_action,
            "decomposition": safe_decomposition,
        })
        run.decomposition = safe_decomposition
        run.touch()

        # Auto-queue child run on first sub-issue if policy approved it
        if policy == "auto_create_subissues" and created_urls and config is not None:
            self._autorun_first_subissue(run, config, safe_decomposition, created_urls, triggering_signal)
            run.touch()

        return run

    # ------------------------------------------------------------------

    def _blocked(
        self,
        run: SupervisorRun,
        failure: str,
        detail: str,
        *,
        mission_plan: Optional[MissionPlan] = None,
        stage_statuses: Optional[Dict[str, Dict[str, Any]]] = None,
        cleanup_workspace: bool = False,
    ) -> SupervisorRun:
        run.status = "blocked"
        run.outcome = "Blocked"
        run.failure_class = failure
        run.completion_mode = f"blocked/{failure}"  # (#147)
        run.add("blocked", "blocked", detail)
        if cleanup_workspace:
            self._cleanup_blocked_workspace(run)
        if stage_statuses and "final_report" in stage_statuses:
            self._set_stage_status(run, stage_statuses, "final_report", "failure", f"Run blocked: {failure}.")
        run.report = {"autonomous": False, "blocked_reason": detail}
        if run.acceptance_evidence is not None:
            run.report["acceptance_evidence"] = run.acceptance_evidence
        run.report.update(self._api_escalation_report_fragment(run))
        run.report.update(self._stage_report_fragment(mission_plan, stage_statuses))
        run.touch()
        # Issue #733 — ensure rank_pending.patch is cleaned up on any blocked/failure path
        try:
            _stale_patch = Path(self.project_root) / ".igris" / "rank_pending.patch"
            if _stale_patch.exists():
                _stale_patch.unlink(missing_ok=True)
                run.add("patch_cleanup", "success", "rank_pending.patch removed on blocked run")
        except Exception:
            pass
        # Record capability-related failures so future runs can learn from history.
        # Skip infrastructure/baseline failures — they're environment issues, not
        # capability limits, and would pollute similarity matching.
        _SKIP_MEMORY_CLASSES = frozenset({"pytest_failure", "workspace_dirty", "infrastructure_bug", "test_runner_timeout"})
        if failure not in _SKIP_MEMORY_CLASSES and hasattr(self, "_failure_memory"):
            try:
                self._failure_memory.record(
                    goal=getattr(run, "goal", "") or "",
                    failure_class=failure,
                    capability_signals=dict(run.capability_signals),
                    repair_cycles=run.repair_cycles_used,
                )
            except Exception:
                pass
        # Issue #914 — MissionBrain Advisory diagnostic (monitoring-only).
        # Computes a recovery recommendation for failed/blocked runs without
        # surfacing it in reports (should_emit=False, is_gate=False).
        # Wrapped in bare except so it can NEVER block or modify run outcome.
        if _selected_advisory_available:
            try:
                _goal_status = "partial" if run.repair_cycles_used > 0 else "failed"
                _adv_cycle = {
                    "cycle_id": getattr(run, "run_id", "unknown"),
                    "current_loop_decision": "blocked",
                    "mission_brain_decision": _goal_status,
                    "report_type": "diagnostic",
                    "failure_class": failure,
                    "capability_signals": dict(run.capability_signals),
                }
                _adv_cfg = _make_selected_monitoring_config(include_blocked=True)
                _adv_result = _enrich_cycle_selected(_adv_cycle, config=_adv_cfg)
                run.add(
                    "advisory_diagnostic",
                    "computed",
                    "MissionBrain Advisory diagnostic computed (monitoring-only, not surfaced)",
                    combined_status=_adv_result.get(
                        "bridge_diagnostics", {}
                    ).get("combined_status", "unknown"),
                    template_used=_adv_result.get("_advisory_template_used", "none"),
                    advisory_surfaced=False,
                )
            except Exception:
                pass  # advisory monitoring must never block or alter run outcome
        return run

    def _persist_blocked_outcome(
        self,
        run: "SupervisorRun",
        assignment_decision: Any,
    ) -> None:
        self._persist_assignment_outcome(run, self.project_root, assignment_decision)

    @staticmethod
    def _persist_assignment_outcome(
        run: "SupervisorRun",
        project_root: Any,
        assignment_decision: Any,
    ) -> None:
        """Append assignment outcome record for historical learning. No-op if unavailable."""
        if not _assignment_router_available or assignment_decision is None:
            return
        try:
            outcomes_path = str(Path(project_root) / ".igris" / "assignment_outcomes.json")
            total_cost = run.execution_budget_used_usd + run.api_budget_used_usd
            attempts = run.repair_cycles_used + 1
            cost_per_success = (
                round(total_cost / attempts, 6)
                if run.status == "completed" and attempts > 0
                else None
            )
            record = {
                "task_signature": compute_task_signature(getattr(run, "goal", "") or ""),
                "goal_excerpt": (getattr(run, "goal", "") or "")[:200],
                "agent_role": assignment_decision.agent_role,
                "task_type": assignment_decision.task_type,
                "preferred_profile": assignment_decision.preferred_profile,
                "execution_strategy": assignment_decision.execution_strategy,
                "model_used": assignment_decision.preferred_model,
                "fallback_model_path": list(assignment_decision.fallback_model_path),
                "outcome": run.status,
                "failure_class": run.failure_class,
                "capability_signals": dict(run.capability_signals),
                "cost_usd": total_cost,
                "execution_cost_usd": run.execution_budget_used_usd,
                "helper_cost_usd": run.api_budget_used_usd,
                "cost_per_success": cost_per_success,
                "attempts": attempts,
                "execution_provider": "",
                "execution_model": "",
                "created_at": run.created_at.isoformat() if hasattr(run, "created_at") and run.created_at else "",
            }
            save_assignment_outcome(outcomes_path, record)
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).warning("Failed to persist assignment outcome: %s", exc)

    def _pr_body(self, run: SupervisorRun) -> str:
        lines = [
            "## Summary",
            f"- Supervised rank run `{run.run_id}` completed.",
            "",
            "## Safety",
            "- Full pytest passed before merge consideration.",
            "- No direct push to main.",
        ]
        # Append "Closes #N" so GitHub auto-closes the issue on merge.
        goal = getattr(run, "goal", "") or ""
        _m = re.search(r"#(\d+)", goal)
        if _m:
            lines += ["", f"Closes #{_m.group(1)}"]
        return "\n".join(lines)


RUN_STORE: Dict[str, SupervisorRun] = {}
RUN_LOCK = threading.RLock()


def start_supervised_rank(data: Dict[str, Any], project_root: str) -> SupervisorRun:
    config = RankSupervisorConfig.from_dict(data)
    supervisor = SelfRepairSupervisor(project_root=project_root)
    run = SupervisorRun(run_id=uuid.uuid4().hex[:12], rank_id=config.rank_id)
    if hasattr(supervisor, "_configure_run_tracking"):
        supervisor._configure_run_tracking(run, config)
    elif hasattr(supervisor, "_resolve_event_audit"):
        run.audit_resolver = getattr(supervisor, "_resolve_event_audit")
    run = supervisor.run(config, run=run)
    with RUN_LOCK:
        RUN_STORE[run.run_id] = run
    return run


def start_supervised_rank_async(data: Dict[str, Any], project_root: str) -> SupervisorRun:
    """Create a run immediately and execute it in a background worker.

    MBOP integration (#936): wraps the worker with Phases 1, 9, 10, 11, 12.
    - Phase 1 (Intake): reads GitHub issue before run starts.
    - Phases 9–12: quality gate, satisfaction gate, eval, next-step after completion.
    MBOP hooks are best-effort: any failure is logged but never crashes the run.
    """
    payload = dict(data)
    payload["defer_service_restart"] = True
    config = RankSupervisorConfig.from_dict(payload)
    # mbop_enforce_quality_gate: opt-in per-issue to enforce QG (default: advisory-only)
    mbop_enforce_qg = bool(data.get("mbop_enforce_quality_gate", False))
    supervisor = SelfRepairSupervisor(project_root=project_root)
    run = SupervisorRun(run_id=uuid.uuid4().hex[:12], rank_id=config.rank_id)
    if hasattr(supervisor, "_configure_run_tracking"):
        supervisor._configure_run_tracking(run, config)
    elif hasattr(supervisor, "_resolve_event_audit"):
        run.audit_resolver = getattr(supervisor, "_resolve_event_audit")
    run.add("queued", "running", "Supervisor run accepted for background execution")
    with RUN_LOCK:
        RUN_STORE[run.run_id] = run

    def _worker() -> None:
        import time as _time
        _run_start = _time.time()

        # --- (#147) Initialise BehaviorTracker for this run ---
        try:
            from igris.core.behavior_tracker import BehaviorTracker
            run.behavior_tracker = BehaviorTracker(
                run_id=run.run_id,
                issue_number=config.issue_number,
            )
        except Exception:  # noqa: BLE001
            pass  # best-effort — never block the run

        # --- MBOP Phase 1: Intake (pre-run) ---
        _mbop_intake = None
        try:
            from igris.core.mbop_runner import mbop_pre_run
            _mbop_intake = mbop_pre_run(
                issue_number=config.issue_number,
                project_root=project_root,
                run_add_fn=run.add,
                run_id=run.run_id,
            )
        except Exception:  # noqa: BLE001
            pass  # best-effort — never block the run

        # --- MBOP Phase 2: Pre-flight ---
        try:
            from igris.core.mbop_runner import _persist_event as _mbop_persist
            _mbop_persist(
                project_root, run.run_id, config.issue_number,
                "mbop_phase2_preflight", "running",
                f"MBOP Phase 2 Pre-flight: #{config.issue_number} | "
                f"deps={'checking' if config.issue_number else 'skip'} env=ok"
            )
        except Exception:
            pass

        # --- MBOP Phase 3: Mission Planning ---
        try:
            _mbop_persist(
                project_root, run.run_id, config.issue_number,
                "mbop_phase3_planning", "running",
                f"MBOP Phase 3 Mission Planning: #{config.issue_number} | "
                f"goal={str(config.goal)[:80]}"
            )
        except Exception:
            pass

        # --- Store MBOP intake on run so _rank_initial_context can inject it (#1040) ---
        if _mbop_intake is not None:
            run.mbop_intake = _mbop_intake

        # --- Main supervisor run ---
        try:
            supervisor.run(config, run=run)
        except Exception as exc:
            run.status = "blocked"
            run.outcome = "Blocked"
            run.failure_class = "supervisor_bug"
            run.add("exception", "blocked", str(exc))
            run.report = {"autonomous": False, "blocked_reason": "Supervisor worker crashed"}
            run.touch()

        # --- MBOP Phases 4–8: post-run intermediates (based on run outcome) ---
        try:
            _repair_cycles = getattr(run, "repair_cycles_used", 0)
            _failure_class = str(getattr(run, "failure_class", "") or "")
            _run_status = str(getattr(run, "status", "") or "")
            # Phase 4: Implementation outcome
            _mbop_persist(
                project_root, run.run_id, config.issue_number,
                "mbop_phase4_implementation",
                "done" if _run_status == "completed" else "blocked",
                f"MBOP Phase 4 Implementation: #{config.issue_number} | "
                f"status={_run_status} failure_class={_failure_class}",
                extra={"failure_class": _failure_class, "run_status": _run_status}
            )
            # Phase 5: Testing
            _test_ran = any(
                getattr(e, "phase", e.get("phase", "") if isinstance(e, dict) else "") in
                {"pytest_run", "test_run", "pytest_result"}
                for e in getattr(run, "events", [])
            )
            _mbop_persist(
                project_root, run.run_id, config.issue_number,
                "mbop_phase5_testing",
                "ran" if _test_ran else "skipped",
                f"MBOP Phase 5 Testing: #{config.issue_number} | "
                f"pytest={'ran' if _test_ran else 'skipped'}"
            )
            # Phase 6: Review
            _mbop_persist(
                project_root, run.run_id, config.issue_number,
                "mbop_phase6_review",
                "done",
                f"MBOP Phase 6 Review: #{config.issue_number} | "
                f"repair_cycles={_repair_cycles}"
            )
            # Phase 7: Repair
            _mbop_persist(
                project_root, run.run_id, config.issue_number,
                "mbop_phase7_repair",
                f"cycles={_repair_cycles}" if _repair_cycles > 0 else "none",
                f"MBOP Phase 7 Repair: #{config.issue_number} | "
                f"cycles_used={_repair_cycles}",
                extra={"repair_cycles_used": _repair_cycles}
            )
            # Phase 8: Completion check
            _mbop_persist(
                project_root, run.run_id, config.issue_number,
                "mbop_phase8_completion_check",
                "pass" if _run_status == "completed" else "fail",
                f"MBOP Phase 8 Completion Check: #{config.issue_number} | "
                f"final_status={_run_status}"
            )
        except Exception:
            pass

        # --- MBOP Phases 9–12: post-run hooks ---
        try:
            from igris.core.mbop_runner import mbop_post_run, MBOPIntakeResult
            _intake = _mbop_intake if _mbop_intake is not None else MBOPIntakeResult(
                issue_number=config.issue_number
            )
            mbop_post_run(
                run=run,
                intake=_intake,
                project_root=project_root,
                run_start_ts=_run_start,
                enforce_quality_gate=mbop_enforce_qg,
                run_id=run.run_id,
            )
        except Exception:  # noqa: BLE001
            pass  # best-effort — never crash after supervisor completed

        # --- (#147) Supervisor self-audit post-run ---
        try:
            if run.behavior_tracker is not None:
                _run_status = str(getattr(run, "status", "") or "")
                _failure_class = str(getattr(run, "failure_class", "") or "")
                _repair_cycles = int(getattr(run, "repair_cycles_used", 0) or 0)
                _completion_mode = str(getattr(run, "completion_mode", "") or "")
                _escalations_used = int(getattr(run, "api_escalations_used", 0) or 0)
                _max_escalations = int(getattr(run, "max_api_escalations_per_run", 0) or 0)
                _budget_exhausted = bool(
                    _max_escalations > 0 and _escalations_used >= _max_escalations
                )
                _report = dict(getattr(run, "report", {}) or {})
                _smoke_ran = bool(_report.get("post_merge_smoke") is not None)
                _pytest_evidence = any(
                    e.phase in ("full_pytest", "targeted_tests", "baseline_tests")
                    and e.status in ("success", "failure")
                    for e in getattr(run, "events", [])
                )
                # Workspace dirty = git status has uncommitted changes
                _workspace_dirty = False
                try:
                    _gs = subprocess.run(
                        ["git", "status", "--porcelain"],
                        capture_output=True, text=True, cwd=project_root, timeout=10,
                    )
                    _workspace_dirty = bool(_gs.stdout.strip())
                except Exception:
                    pass
                audit = run.behavior_tracker.self_audit(
                    run_status=_run_status,
                    failure_class=_failure_class,
                    repair_cycles_used=_repair_cycles,
                    smoke_ran=_smoke_ran,
                    pytest_ran=_pytest_evidence,
                    workspace_dirty=_workspace_dirty,
                    escalation_budget_exhausted=_budget_exhausted,
                    escalation_was_called=_escalations_used > 0,
                    completion_mode=_completion_mode,
                    project_root=project_root,
                )
                audit_summary = run.behavior_tracker.summary()
                run.add(
                    "supervisor_self_audit", "done",
                    f"Self-audit complete: {audit_summary}",
                    missed_behaviors=audit.missed_behaviors[:10],
                    opened_issues=audit.opened_issues,
                    notes=audit.notes[:5],
                    behavior_log=run.behavior_tracker.to_dict(),
                )
        except Exception:  # noqa: BLE001
            pass  # best-effort — never crash after supervisor completed

    thread = threading.Thread(
        target=_worker,
        name=f"rank-supervisor-{run.run_id}",
        daemon=True,
    )
    thread.start()
    return run


def get_supervised_run(run_id: str) -> Optional[SupervisorRun]:
    with RUN_LOCK:
        return RUN_STORE.get(run_id)


def cancel_supervised_run(run_id: str, project_root: str, reason: str = "Cancelled by user") -> Optional[SupervisorRun]:
    with RUN_LOCK:
        run = RUN_STORE.get(run_id)
    if run is None:
        return None

    current_status = str(run.status or "").strip().lower()
    if _is_terminal_status(current_status):
        return run

    cancel_reason = str(reason or "Cancelled by user").strip() or "Cancelled by user"
    run.cancel_requested = True
    run.cancel_reason = cancel_reason
    if current_status != "cancelling":
        run.status = "cancelling"
        run.add("cancel_request", "running", cancel_reason, requested_by="api")
    else:
        run.add("cancel_request", "running", cancel_reason, requested_by="api", duplicate=True)

    supervisor = SelfRepairSupervisor(project_root=project_root)
    if hasattr(supervisor, "_configure_run_tracking"):
        supervisor._configure_run_tracking(run, RankSupervisorConfig.from_dict({"goal": "", "rank_id": run.rank_id}))
    return supervisor._cancelled(run, cancel_reason, cleanup_workspace=True)


def list_supervised_runs() -> List[SupervisorRun]:
    with RUN_LOCK:
        return list(RUN_STORE.values())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _timestamp_sort_key(value: Any) -> float:
    numeric = _safe_float(value, default=float("nan"))
    if numeric == numeric:  # not NaN
        return numeric
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _stage_summary_from_run_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    stages = (
        ((payload.get("report") or {}).get("mission_orchestration") or {}).get("stages")
        or []
    )
    counts = {
        "success": 0,
        "failure": 0,
        "pending": 0,
        "running": 0,
        "skipped": 0,
        "unknown": 0,
    }
    failed_stage_ids: List[str] = []
    pending_stage_ids: List[str] = []
    for stage in stages:
        status = str((stage or {}).get("status", "")).strip().lower()
        stage_id = str((stage or {}).get("stage_id", "")).strip()
        if status in counts:
            counts[status] += 1
        else:
            counts["unknown"] += 1
        if status == "failure" and stage_id:
            failed_stage_ids.append(stage_id)
        if status in {"pending", "running"} and stage_id:
            pending_stage_ids.append(stage_id)
    return {
        "counts": counts,
        "failed_stage_ids": failed_stage_ids,
        "pending_stage_ids": pending_stage_ids,
        "total": len(stages),
    }


def _audit_counts_from_events(events: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {status: 0 for status in sorted(AUDIT_STATUSES)}
    counts["unknown"] = 0
    for event in events:
        status = str((event or {}).get("audit_status", "")).strip().lower()
        if status in counts:
            counts[status] += 1
        else:
            counts["unknown"] += 1
    return counts


def _extract_issue_url_from_text(text: str) -> str:
    match = re.search(r"https://github\.com/[^\s]+/issues/\d+", text or "")
    return match.group(0) if match else ""


TERMINAL_RUN_STATUSES = {"completed", "blocked", "failed", "crashed", "cancelled", "interrupted"}


def _is_terminal_status(status: Any) -> bool:
    return str(status or "").strip().lower() in TERMINAL_RUN_STATUSES


def _run_has_resolved_failure(record: Dict[str, Any]) -> bool:
    report = record.get("final_report")
    if not isinstance(report, dict):
        report = {}
    return bool(
        report.get("resolved_failure")
        or report.get("degraded_completion")
        or record.get("resolved_failure")
        or record.get("degraded_completion")
    )


def _enforce_completion_failure_invariant(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(record)
    status = str(normalized.get("status", "")).strip().lower()
    failure_class = str(normalized.get("failure_class", "")).strip()
    if status == "completed" and failure_class and not _run_has_resolved_failure(normalized):
        normalized["state_conflict"] = True
        normalized["warning"] = (
            "Completed run has failure_class without resolved/degraded completion flag."
        )
    else:
        normalized["state_conflict"] = bool(normalized.get("state_conflict", False))
        normalized["warning"] = str(normalized.get("warning", "") or "")
    return normalized


def _reconcile_run_records(
    in_memory: Dict[str, Dict[str, Any]],
    persisted: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    run_ids = set(in_memory.keys()) | set(persisted.keys())
    for run_id in run_ids:
        memory_record = in_memory.get(run_id)
        persisted_record = persisted.get(run_id)
        if memory_record is None and persisted_record is not None:
            chosen = dict(persisted_record)
            # A persisted run that is no longer in the in-memory store was
            # interrupted by a service restart — promote it to a terminal
            # status so it never reappears as a ghost active run.
            if str(chosen.get("status", "")).strip().lower() == "running":
                chosen["status"] = "interrupted"
                chosen.setdefault(
                    "warning",
                    "Run was interrupted by a service restart and is no longer active.",
                )
        elif persisted_record is None and memory_record is not None:
            chosen = dict(memory_record)
        else:
            assert memory_record is not None and persisted_record is not None
            mem_updated = _timestamp_sort_key(memory_record.get("updated_at", 0.0))
            per_updated = _timestamp_sort_key(persisted_record.get("updated_at", 0.0))
            chosen = dict(memory_record if mem_updated >= per_updated else persisted_record)
            mem_status = str(memory_record.get("status", "")).strip().lower()
            per_status = str(persisted_record.get("status", "")).strip().lower()
            if mem_status != per_status:
                if _is_terminal_status(per_status) and per_updated >= mem_updated:
                    chosen = dict(persisted_record)
                elif _is_terminal_status(mem_status) and mem_updated >= per_updated:
                    chosen = dict(memory_record)
                chosen["state_conflict"] = True
                chosen["warning"] = (
                    f"State conflict between in-memory({mem_status}) and durable({per_status})."
                )
        merged[run_id] = _enforce_completion_failure_invariant(chosen)
    return merged


def summarize_supervised_run(run: SupervisorRun) -> Dict[str, Any]:
    payload = run.to_dict()
    events = payload.get("events") or []
    started_at = events[0]["timestamp"] if events else None
    updated_at = events[-1]["timestamp"] if events else None
    last_event = events[-1] if events else {}
    current_stage = ""
    for event in reversed(events):
        phase = str((event or {}).get("phase", ""))
        status = str((event or {}).get("status", ""))
        data = (event or {}).get("data") or {}
        if phase == "rank_reasoning" and status == "running":
            current_stage = str(data.get("stage_id", "")).strip()
            break
        if phase == "mission_stage" and status == "running":
            current_stage = str(data.get("stage_id", "")).strip()
            break
    stage_summary = _stage_summary_from_run_dict(payload)
    audit_counts = _audit_counts_from_events(events)
    failed_stage = (stage_summary.get("failed_stage_ids") or [""])[0]
    escalation_issue_url = ""
    for event in reversed(events):
        if str((event or {}).get("phase", "")).strip() != "repair_issue":
            continue
        detail = str((event or {}).get("detail", ""))
        escalation_issue_url = _extract_issue_url_from_text(detail)
        if escalation_issue_url:
            break

    next_action = "monitor"
    if payload.get("status") == "running":
        next_action = f"wait:{current_stage}" if current_stage else "wait:next_event"
    elif payload.get("status") == "blocked":
        failure = str(payload.get("failure_class", "")).strip() or "blocked"
        next_action = f"review:{failure}"
    elif payload.get("status") == "completed":
        next_action = "done"

    summary = {
        "run_id": payload.get("run_id", ""),
        "rank_id": payload.get("rank_id", ""),
        "status": payload.get("status", ""),
        "outcome": payload.get("outcome", ""),
        "failure_class": payload.get("failure_class", ""),
        "branch": payload.get("branch", ""),
        "repair_cycles_used": int(payload.get("repair_cycles_used", 0) or 0),
        "max_repair_cycles": int(payload.get("max_repair_cycles", 0) or 0),
        "api_escalations_used": int(payload.get("api_escalations_used", 0) or 0),
        "api_escalations_failed_unconfigured": int(payload.get("api_escalations_failed_unconfigured", 0) or 0),
        "max_api_escalations_per_run": int(payload.get("max_api_escalations_per_run", 0) or 0),
        "api_budget_used_usd": round(_safe_float(payload.get("api_budget_used_usd", 0.0)), 6),
        "max_api_budget_usd": round(_safe_float(payload.get("max_api_budget_usd", 0.0)), 6),
        "current_stage": current_stage,
        "failed_stage": failed_stage,
        "escalation_issue_url": escalation_issue_url,
        "stage_summary": stage_summary,
        "audit_summary": {
            "counts": audit_counts,
            "next_review_due_count": sum(
                1
                for event in events
                if str((event or {}).get("audit_status", "")).strip().lower() == "audit-deferred"
                and SelfRepairSupervisor._timestamp_is_due(
                    str((event or {}).get("audit_next_review_after", ""))
                )
            ),
        },
        "last_event": {
            "phase": str(last_event.get("phase", "")),
            "status": str(last_event.get("status", "")),
            "timestamp": last_event.get("timestamp"),
            "audit_status": str(last_event.get("audit_status", "")),
        },
        "started_at": started_at,
        "updated_at": updated_at,
        "resolved_failure": bool((payload.get("report") or {}).get("resolved_failure", False)),
        "degraded_completion": bool((payload.get("report") or {}).get("degraded_completion", False)),
        "degraded_completion_reason": str((payload.get("report") or {}).get("degraded_completion_reason", "")),
        "cancelled_reason": str((payload.get("report") or {}).get("cancelled_reason", "") or payload.get("cancel_reason", "")),
        "next_action": next_action,
    }
    return _enforce_completion_failure_invariant(summary)


def list_active_supervised_runs() -> List[SupervisorRun]:
    with RUN_LOCK:
        return [run for run in RUN_STORE.values() if run.status == "running"]


def list_active_supervised_run_summaries(project_root: str) -> List[Dict[str, Any]]:
    in_memory_active: Dict[str, Dict[str, Any]] = {}
    with RUN_LOCK:
        for run in RUN_STORE.values():
            if str(run.status).strip().lower() != "running":
                continue
            in_memory_active[str(run.run_id)] = summarize_supervised_run(run)
    persisted = {
        str(item.get("run_id", "")): dict(item)
        for item in _load_persisted_recent_runs(project_root)
        if str(item.get("run_id", "")).strip()
    }
    reconciled = _reconcile_run_records(in_memory_active, persisted)
    active = [
        record for record in reconciled.values()
        if str(record.get("status", "")).strip().lower() == "running"
    ]
    active.sort(key=lambda item: _timestamp_sort_key(item.get("updated_at", 0.0)), reverse=True)
    return active


def _load_persisted_recent_runs(project_root: str) -> List[Dict[str, Any]]:
    runs_path = Path(project_root) / ".igris" / "supervisor_runs.json"
    if not runs_path.exists():
        return []
    try:
        payload = json.loads(runs_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    runs = payload.get("runs", {}) if isinstance(payload, dict) else {}
    if not isinstance(runs, dict):
        return []
    out: List[Dict[str, Any]] = []
    for run_id, raw in runs.items():
        if not isinstance(run_id, str) or not isinstance(raw, dict):
            continue
        out.append(
            {
                "run_id": run_id,
                "rank_id": str(raw.get("rank_id", "")),
                "status": str(raw.get("status", "")),
                "outcome": str(raw.get("outcome", "")),
                "branch": str(raw.get("branch", "")),
                "current_stage": str(raw.get("current_stage", "")),
                "failed_stage": str(raw.get("failed_stage", "")),
                "failure_class": str(raw.get("failure_class", "")),
                "repair_cycles_used": int(raw.get("repair_cycles_used", 0) or 0),
                "max_repair_cycles": int(raw.get("max_repair_cycles", 0) or 0),
                "api_escalations_used": int(raw.get("api_escalations_used", 0) or 0),
                "api_escalations_failed_unconfigured": int(raw.get("api_escalations_failed_unconfigured", 0) or 0),
                "max_api_escalations_per_run": int(raw.get("max_api_escalations_per_run", 0) or 0),
                "api_budget_used_usd": round(_safe_float(raw.get("api_budget_used_usd", 0.0)), 6),
                "max_api_budget_usd": round(_safe_float(raw.get("max_api_budget_usd", 0.0)), 6),
                "escalation_issue_url": str(raw.get("escalation_issue_url", "")),
                "latest_event": raw.get("latest_event", {}) if isinstance(raw.get("latest_event"), dict) else {},
                "updated_at": str(raw.get("updated_at", "")),
                "created_at": str(raw.get("created_at", "")),
                "blocked_reason": _safe_redact(raw.get("blocked_reason", "")),
                "cancelled_reason": _safe_redact(raw.get("cancelled_reason", "")),
                "next_action": str(raw.get("next_action", "")),
                "resolved_failure": bool(raw.get("resolved_failure", False)),
                "degraded_completion": bool(raw.get("degraded_completion", False)),
                "degraded_completion_reason": str(raw.get("degraded_completion_reason", "")),
                "state_conflict": bool(raw.get("state_conflict", False)),
                "warning": str(raw.get("warning", "")),
            }
        )
    out.sort(key=lambda item: _timestamp_sort_key(item.get("updated_at", "")), reverse=True)
    return [_enforce_completion_failure_invariant(item) for item in out[:20]]


def get_supervisor_audit_summary(project_root: str) -> Dict[str, Any]:
    in_memory_events: List[Dict[str, Any]] = []
    recent_runs: Dict[str, Dict[str, Any]] = {}
    with RUN_LOCK:
        for run in RUN_STORE.values():
            summary = summarize_supervised_run(run)
            events = (run.to_dict().get("events") or [])
            in_memory_events.extend(events)
            run_id = str(summary.get("run_id", "")).strip()
            if run_id:
                recent_runs[run_id] = summary

    persisted_recent_runs = {
        str(item.get("run_id", "")): dict(item)
        for item in _load_persisted_recent_runs(project_root)
        if str(item.get("run_id", "")).strip()
    }
    merged_recent = _reconcile_run_records(recent_runs, persisted_recent_runs)
    merged_recent_runs = sorted(
        merged_recent.values(),
        key=lambda item: _timestamp_sort_key(item.get("updated_at", 0.0)),
        reverse=True,
    )[:5]
    in_memory_counts = _audit_counts_from_events(in_memory_events)

    persisted_counts = {status: 0 for status in sorted(AUDIT_STATUSES)}
    persisted_counts["unknown"] = 0
    persisted_total = 0
    deferred_due_count = 0
    audit_path = Path(project_root) / ".igris" / "supervisor_audit.json"
    if audit_path.exists():
        try:
            payload = json.loads(audit_path.read_text(encoding="utf-8"))
            records = payload.get("records", {}) if isinstance(payload, dict) else {}
            if isinstance(records, dict):
                for entry in records.values():
                    if not isinstance(entry, dict):
                        continue
                    persisted_total += 1
                    status = str(entry.get("audit_status", "")).strip().lower()
                    if status in persisted_counts:
                        persisted_counts[status] += 1
                    else:
                        persisted_counts["unknown"] += 1
                    if (
                        status == "audit-deferred"
                        and SelfRepairSupervisor._timestamp_is_due(
                            str(entry.get("audit_next_review_after", ""))
                        )
                    ):
                        deferred_due_count += 1
        except (OSError, json.JSONDecodeError):
            pass

    return {
        "audit_file": str(audit_path),
        "audit_file_exists": audit_path.exists(),
        "in_memory": {
            "event_count": len(in_memory_events),
            "counts": in_memory_counts,
        },
        "persisted": {
            "record_count": persisted_total,
            "counts": persisted_counts,
            "deferred_due_count": deferred_due_count,
        },
        "recent_runs": merged_recent_runs,
    }
