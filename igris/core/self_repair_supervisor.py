"""Autonomous self-repair supervisor for controlled rank missions.

The supervisor coordinates an IGRIS rank attempt and bounded infrastructure
repair cycles. It does not expose free-form shell execution: the default
backend runs fixed argv commands only, and tests can inject a fake backend.
"""

from __future__ import annotations

import json
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


@dataclass
class CommandResult:
    success: bool = False
    output: str = ""
    error: str = ""
    returncode: int = 0

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
    max_rank_attempts: int = 1
    max_repair_cycles: int = 0
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
    test_timeout_seconds: int = 300
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
    allow_auto_subissues: bool = False
    enable_semantic_gate: bool = True
    api_helper_mode: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RankSupervisorConfig":
        return cls(
            goal=str(data.get("goal", "")),
            rank_id=str(data.get("rank_id", "rank")),
            max_rank_attempts=max(1, int(data.get("max_rank_attempts", 1))),
            max_repair_cycles=max(0, int(data.get("max_repair_cycles", 0))),
            allow_github_pr=bool(data.get("allow_github_pr", False)),
            allow_merge_if_green=bool(data.get("allow_merge_if_green", False)),
            service_restart_command=str(data.get("service_restart_command", "")),
            required_smoke_endpoints=list(data.get("required_smoke_endpoints", [])),
            targeted_tests=_infer_targeted_tests(
                str(data.get("goal", "")),
                list(data.get("targeted_tests", [])),
            ),
            dry_run=_infer_dry_run(data),
            defer_service_restart=bool(data.get("defer_service_restart", False)),
            test_timeout_seconds=max(30, int(data.get("test_timeout_seconds", 300))),
            test_hard_cap_seconds=max(60, int(data.get("test_hard_cap_seconds", 3600))),
            reasoning_timeout_seconds=max(30, int(
                data.get("reasoning_timeout_seconds")
                or os.environ.get("IGRIS_REASONING_TIMEOUT_SECONDS")
                or 300
            )),
            allow_api_escalation=bool(data.get("allow_api_escalation", False)),
            max_api_escalations_per_run=max(0, int(data.get("max_api_escalations_per_run", 0))),
            max_api_budget_usd=max(0.0, float(data.get("max_api_budget_usd", 0.0))),
            max_tokens_per_escalation=max(64, int(data.get("max_tokens_per_escalation", 600))),
            api_helper_model=str(data.get("api_helper_model", "gpt-5.4-mini")),
            enable_mission_planning=bool(data.get("enable_mission_planning", False)),
            allow_auto_subissues=bool(data.get("allow_auto_subissues", False)),
            enable_semantic_gate=bool(data.get("enable_semantic_gate", True)),
            api_helper_mode=str(data.get("api_helper_mode", "")),
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
    def restore_dangerous_diff(self) -> CommandResult: ...
    def restore_paths(self, paths: List[str]) -> CommandResult: ...
    def call_api_helper(self, packet: Dict[str, Any], model: str, max_tokens: int, timeout: int = 45) -> CommandResult: ...
    def api_helper_is_configured(self) -> bool: ...


class LocalSupervisorBackend:
    """Governed local backend using fixed argv commands only."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)

    def _subprocess_env(self, *, clean_for_tests: bool = False) -> Dict[str, str]:
        if not clean_for_tests:
            env = os.environ.copy()
            env["IGRIS_SUPERVISOR_CHILD"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            env.pop("PYTEST_CURRENT_TEST", None)
            return env
        allowlist = {
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
        extra_env: Optional[Dict[str, str]] = None,
    ) -> CommandResult:
        try:
            env = self._subprocess_env(clean_for_tests=clean_env)
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
        payload = json.dumps({
            "project_root": str(self.project_root),
            "goal": goal,
            "max_steps": max_steps,
            "initial_context": initial_context,
            "task_type": task_type,
            "preferred_profile": preferred_profile,
        })
        result = self._run(
            [str(self.project_root / ".venv/bin/python"), "-m", "igris.core.supervisor_reasoning_worker"],
            timeout=timeout,
            input_text=payload,
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
        return self._run(["git", "diff", "--stat"], timeout=10)

    def git_diff(self) -> CommandResult:
        return self._run(["git", "diff"], timeout=10)

    def run_tests(self, targets: Optional[List[str]] = None, timeout: int = 120, hard_cap: int = 3600) -> CommandResult:
        cmd = [str(self.project_root / ".venv/bin/python"), "-m", "pytest", "-q"]
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
        return self._run(["git", "commit", "-m", message], timeout=60)

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

    def restore_dangerous_diff(self) -> CommandResult:
        restore = self._run(["git", "restore", "--worktree", "--staged", "."], timeout=60)
        if not restore.success:
            return restore
        # Remove untracked source/test/docs files left by a failed supervised
        # branch. The argv is fixed and scoped; no force push or main mutation.
        clean = self._run(["git", "clean", "-fd", "--", "igris", "tests", "docs"], timeout=60)
        if not clean.success:
            return clean
        return CommandResult(True, restore.output + clean.output, "", 0)

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
        payload = json.dumps({"model": model, "max_tokens": max_tokens, "packet": packet})
        # Forward mode to helper via env var so the subprocess enforces codex_only policy.
        # We only override if the caller explicitly requested a mode — otherwise let the
        # process-level env (set by the operator in .env) take precedence.
        extra_env: Dict[str, str] = {}
        if mode:
            extra_env["IGRIS_API_HELPER_MODE"] = mode
        return self._run(cmd, timeout=timeout, input_text=payload, extra_env=extra_env or None)

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
        if stop in {"reasoning_timeout", "budget_exceeded"}:
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


def _has_immediately_dangerous_diff(diff: str) -> bool:
    """Fast pre-test check for diffs that would definitely break the app.

    Only catches two categories that cannot possibly be recovered by the test suite:
      1. Dangerous file tokens (.env, .venv, __pycache__, etc.)
      2. Structural deletions of def create_app or class bodies

    Import-deletion detection is left to _has_destructive_diff (used post-test via
    classify_failure), allowing the test suite to be the primary safety net.
    """
    dangerous_tokens = [".env", ".venv", "__pycache__", ".pytest_cache", ".igris"]
    if any(token in diff for token in dangerous_tokens):
        return True
    paths = _diff_changed_paths(diff)
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
                current_path = parts[3][2:] if len(parts) >= 4 and parts[3].startswith("b/") else ""
                continue
            if not (current_path.endswith(".py") and line.startswith("-") and not line.startswith("---")):
                continue
            python_removed_lines.append(line)
    structural = ("def create_app", "class ")
    return any(any(token in line for token in structural) for line in python_removed_lines)


def _has_destructive_diff(diff: str) -> bool:
    dangerous_tokens = [".env", ".venv", "__pycache__", ".pytest_cache", ".igris"]
    if any(token in diff for token in dangerous_tokens):
        return True
    paths = _diff_changed_paths(diff)
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
            codex_only=is_codex_only,
        )
        return advice

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
                repair_strategy="Retry with explicit stage scope and stricter instructions.",
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
                repair_strategy="Stage-scoped repair that keeps validated earlier stages.",
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
                failure_classification=["missing_tests", "wrong_file_edit", "pytest_failure"],
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
                failure_classification=["missing_ui_visibility", "wrong_file_edit", "pytest_failure"],
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
                failure_classification=["pytest_failure", "wrong_file_edit", "missing_tests"],
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
                failure_classification=["missing_tests", "pytest_failure"],
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
                failure_classification=["pytest_failure"],
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
                stage_context = self._rank_initial_context(config)
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

    def run(
        self,
        config: RankSupervisorConfig,
        run: Optional[SupervisorRun] = None,
    ) -> SupervisorRun:
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
            return cancelled
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
            return cancelled
        if not status.success:
            return self._blocked(run, "infrastructure_bug", "Unable to read git status")
        if status.output.strip():
            return self._blocked(run, "workspace_dirty", "Workspace is not clean")

        head = self.backend.git_log_head()
        run.add("git_head", "success" if head.success else "failure", _command_detail(head))

        run.add(
            "baseline_tests",
            "running",
            "Running baseline pytest",
            timeout_seconds=config.test_timeout_seconds,
        )
        baseline = self.backend.run_tests(timeout=config.test_timeout_seconds, hard_cap=config.test_hard_cap_seconds)
        run.add("baseline_tests", "success" if baseline.success else "failure", _command_detail(baseline))
        cancelled = self._cancel_if_requested(run)
        if cancelled is not None:
            return cancelled
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
            return self._blocked(run, "pytest_failure", "Baseline tests failed")

        run.add("baseline_smoke", "running", "Running baseline smoke")
        smoke = self.backend.smoke(config.required_smoke_endpoints, restart_command)
        run.add("baseline_smoke", "success" if smoke.success else "failure", _command_detail(smoke))
        cancelled = self._cancel_if_requested(run)
        if cancelled is not None:
            return cancelled
        if not smoke.success:
            return self._blocked(run, "infrastructure_bug", "Baseline smoke failed")

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

        mission_plan = self._build_mission_plan(config)
        stage_statuses = self._init_stage_statuses(mission_plan)
        run.add(
            "mission_plan",
            "success",
            "Mission execution strategy planned.",
            mode=mission_plan.mode,
            stage_ids=[stage.stage_id for stage in mission_plan.stages],
        )

        # Pre-flight planning: read-only scope analysis before first attempt.
        # If the planning pass recommends decomposition, block proactively rather
        # than discovering the same thing after 3 failed repair cycles.
        if config.enable_mission_planning:
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
                )

        repair_cycles = 0
        attempt = 1
        attempt_limit = config.max_rank_attempts
        final_validation_extension_used = False
        while attempt <= attempt_limit:
            cancelled = self._cancel_if_requested(run, mission_plan=mission_plan, stage_statuses=stage_statuses)
            if cancelled is not None:
                return cancelled
            branch = f"rank-{config.rank_id.lower()}-{int(time.time())}-{attempt}"
            run.branch = branch
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
                run.add(
                    "rank_reasoning",
                    "running",
                    "Running supervised rank reasoning",
                    timeout_seconds=config.reasoning_timeout_seconds,
                )
                reasoning = self.backend.run_reasoning(
                    config.goal,
                    max_steps=220,
                    initial_context=self._rank_initial_context(config),
                    timeout=config.reasoning_timeout_seconds,
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
                ui_visibility_required=ui_visibility_required,
                ui_visibility_changed=ui_visibility_changed,
                mission_orchestration_mode=mission_plan.mode,
            )
            cancelled = self._cancel_if_requested(run, mission_plan=mission_plan, stage_statuses=stage_statuses)
            if cancelled is not None:
                return cancelled

            diff_stat = self.backend.git_diff_stat()
            diff = self.backend.git_diff()
            run.add("diff_stat", "success" if diff_stat.success else "failure", _command_detail(diff_stat))
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
                run.add(
                    "full_pytest",
                    "running",
                    "Running full pytest",
                    timeout_seconds=config.test_timeout_seconds,
                )
                full = self.backend.run_tests(timeout=config.test_timeout_seconds, hard_cap=config.test_hard_cap_seconds)
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
                    # Record reasoning_timeout signal when the model timed out without
                    # producing a usable diff — a strong indicator of capability limit.
                    if failure == "reasoning_loop_blocked" and stop_reason in {
                        "reasoning_timeout", "budget_exceeded"
                    }:
                        self._record_capability_signal(run, "reasoning_timeout")
            # Record pytest_hang when the full test subprocess was killed for
            # producing no output (idle timeout) — repeated hangs indicate the
            # model's change consistently breaks the test suite in a way it
            # cannot self-repair.
            if not full.success and "Command killed:" in (full.error or ""):
                self._record_capability_signal(run, "pytest_hang")
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
                return self._blocked(
                    run,
                    "infrastructure_bug",
                    "No suitable LLM provider available",
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
                    return self._complete_rank(
                        run,
                        config,
                        branch,
                        completion_mode=completion_mode,
                        runtime_refresh_required=runtime_refresh_required,
                        mission_plan=mission_plan,
                        stage_statuses=stage_statuses,
                    )

            run.failure_class = failure
            run.add("failure", "classified", failure)
            if failure not in REPAIRABLE_FAILURES or repair_cycles >= config.max_repair_cycles:
                triggering_signal = self._detect_capability_limit(run)
                if triggering_signal:
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
                        cleanup_workspace=True,
                    )
                return self._blocked(
                    run,
                    failure,
                    "Rank failed and repair budget is exhausted or not repairable",
                    mission_plan=mission_plan,
                    stage_statuses=stage_statuses,
                    cleanup_workspace=True,
                )

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
                cleanup_workspace=True,
            )
        return self._blocked(
            run,
            run.failure_class or "max_rank_attempts",
            "Rank attempts exhausted",
            mission_plan=mission_plan,
            stage_statuses=stage_statuses,
            cleanup_workspace=True,
        )

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
        delivered_changes = bool(reasoning.get("files_modified")) or (
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

    def _rank_initial_context(self, config: RankSupervisorConfig) -> Dict[str, Any]:
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
        return context

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
        lowered = goal.lower()
        return any(token in lowered for token in ("ui", "dashboard", "frontend", "visible"))

    @staticmethod
    def _goal_targets_rank_ui_card(goal: str) -> bool:
        lowered = goal.lower()
        if "/api/rank/ui-card" in lowered:
            return True
        if "ui-card" in lowered or "ui card" in lowered:
            return True
        return "rank card" in lowered and "ui" in lowered

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

        content = (
            "from fastapi.testclient import TestClient\n\n"
            "from igris.web.server import create_app\n\n\n"
            f"def test_{test_slug}():\n"
            "    client = TestClient(create_app())\n"
            f"    response = client.get(\"{endpoint}\")\n"
            "    assert response.status_code == 200\n"
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
        if failure == "reasoning_loop_blocked":
            # A reasoning_loop_blocked failure means the worker timed out or hit its
            # step limit — there is no product-level infrastructure bug to fix.
            # Repeat the original mission goal so the next worker actually attempts it.
            repair_goal = (
                f"{config.goal} "
                f"(previous attempt timed out or was blocked at step limit — "
                f"continue from the beginning, prioritise writing code and tests over "
                f"exploration, keep edits minimal, do not push)"
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
        repair_context = self._rank_initial_context(config)
        repair_context.update({
            "repair_cycle": cycle,
            "failure_class": failure,
            "supervised_repair": True,
            "repair_goal": repair_goal,
            "api_helper_advice": helper_advice or {},
            "api_helper_advisory_only": True,
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
        env_profile = os.environ.get("IGRIS_EXECUTION_PREFERRED_PROFILE", "")
        if env_profile:
            repair_profile = env_profile
        run.add(
            "repair_reasoning",
            "running",
            f"Starting repair reasoning cycle {cycle}",
            task_type=repair_task_type,
            preferred_profile=repair_profile,
            failure_class=failure,
        )
        result = self.backend.run_reasoning(
            repair_goal,
            max_steps=160,
            initial_context=repair_context,
            timeout=config.reasoning_timeout_seconds,
            task_type=repair_task_type,
            preferred_profile=repair_profile,
        )
        run.add(
            "repair_reasoning",
            str(result.get("status", "")),
            result.get("final_summary", ""),
            orchestrator_used=result.get("orchestrator_used", False),
            reasoning_execution_provider=result.get("reasoning_execution_provider", ""),
            reasoning_execution_model=result.get("reasoning_execution_model", ""),
            reasoning_execution_profile=result.get("reasoning_execution_profile", ""),
            local_model_available=result.get("local_model_available", False),
        )
        # Record a reasoning_timeout signal when repair reasoning itself times out —
        # that also indicates the model cannot make progress on this mission.
        if str(result.get("stop_reason", "")) in {"reasoning_timeout", "budget_exceeded"}:
            self._record_capability_signal(run, "reasoning_timeout")
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
            "Running repair validation pytest",
            timeout_seconds=config.test_timeout_seconds,
        )
        tests = self.backend.run_tests(timeout=config.test_timeout_seconds, hard_cap=config.test_hard_cap_seconds)
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
        result = self.backend.run_reasoning(
            planning_goal,
            max_steps=PLANNING_MAX_STEPS,
            initial_context={"read_only": True, "planning_pass": True},
            timeout=PLANNING_TIMEOUT_SECONDS,
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
        context = self._rank_initial_context(config)
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
            if "test" in t:
                return ["tests/"]
            if any(k in t for k in ("supervisor", "repair")):
                return ["igris/core/self_repair_supervisor.py"]
            return ["igris/**"]

        def _infer_test_targets(text: str) -> List[str]:
            # Extract explicit test file paths like tests/test_foo.py
            matches = re.findall(r"tests/[\w/]+\.py", text)
            if matches:
                return matches
            if "test" in text.lower():
                return ["tests/"]
            return []

        def _make_sub(title: str, goal_text: str) -> Dict[str, Any]:
            safe = _safe_redact(goal_text)
            return {
                "title": title[:60],
                "goal": safe,
                "dependencies": [],
                "acceptance_criteria": [f"{title} implemented and validated"],
                "allowed_file_scopes": _infer_file_scopes(goal_text),
                "tests": _infer_test_targets(goal_text),
                "risk_level": _infer_risk(goal_text),
                "human_approval_required": False,
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

        # --- Strategy 4: single sub-mission (whole goal, scoped) ---
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

        created_urls: List[str] = []
        run.add(
            "subissue_creation",
            "running",
            f"Creating {len(sub_missions)} sub-issue(s) from decomposition.",
            count=len(sub_missions),
            generated_by=generated_by,
        )

        for i, sub in enumerate(sub_missions):
            title = _safe_redact(str(sub.get("title", f"Sub-task {i+1}")))
            goal_text = _safe_redact(str(sub.get("goal", "")))
            risk = str(sub.get("risk_level", "medium"))
            scopes = sub.get("allowed_file_scopes") or []
            tests = sub.get("tests") or []
            criteria = sub.get("acceptance_criteria") or []
            deps = sub.get("dependencies") or []

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
        # Record capability-related failures so future runs can learn from history.
        # Skip infrastructure/baseline failures — they're environment issues, not
        # capability limits, and would pollute similarity matching.
        _SKIP_MEMORY_CLASSES = frozenset({"pytest_failure", "workspace_dirty", "infrastructure_bug"})
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
        return run

    def _pr_body(self, run: SupervisorRun) -> str:
        return "\n".join([
            "## Summary",
            f"- Supervised rank run `{run.run_id}` completed.",
            "",
            "## Safety",
            "- Full pytest passed before merge consideration.",
            "- No direct push to main.",
        ])


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
    """Create a run immediately and execute it in a background worker."""
    payload = dict(data)
    payload["defer_service_restart"] = True
    config = RankSupervisorConfig.from_dict(payload)
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
        try:
            supervisor.run(config, run=run)
        except Exception as exc:
            run.status = "blocked"
            run.outcome = "Blocked"
            run.failure_class = "supervisor_bug"
            run.add("exception", "blocked", str(exc))
            run.report = {"autonomous": False, "blocked_reason": "Supervisor worker crashed"}
            run.touch()

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
