"""Autonomous self-repair supervisor for controlled rank missions.

The supervisor coordinates an IGRIS rank attempt and bounded infrastructure
repair cycles. It does not expose free-form shell execution: the default
backend runs fixed argv commands only, and tests can inject a fake backend.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from igris.core.safety import redact_secrets


REPAIRABLE_FAILURES = {
    "pytest_failure",
    "reasoning_loop_blocked",
    "max_steps",
    "ask_user",
    "missing_tests",
    "wrong_file_edit",
    "infrastructure_bug",
    "syntax_error",
}

UNSAFE_STATUS_PREFIXES = (
    "?? .env",
    "?? .venv",
    "?? .pytest_cache",
    "?? __pycache__",
    "?? .igris",
)


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
    test_timeout_seconds: int = 240
    reasoning_timeout_seconds: int = 300

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
            targeted_tests=list(data.get("targeted_tests", [])),
            dry_run=bool(data.get("dry_run", True)),
            defer_service_restart=bool(data.get("defer_service_restart", False)),
            test_timeout_seconds=max(30, int(data.get("test_timeout_seconds", 240))),
            reasoning_timeout_seconds=max(30, int(data.get("reasoning_timeout_seconds", 300))),
        )


@dataclass
class SupervisorEvent:
    phase: str
    status: str
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "status": self.status,
            "detail": _safe_redact(self.detail),
            "data": {k: _safe_redact(v) for k, v in self.data.items()},
            "timestamp": self.timestamp,
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
    events: List[SupervisorEvent] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)

    def add(self, phase: str, status: str, detail: str = "", **data: Any) -> None:
        self.events.append(SupervisorEvent(phase=phase, status=status, detail=detail, data=data))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "rank_id": self.rank_id,
            "status": self.status,
            "outcome": self.outcome,
            "failure_class": self.failure_class,
            "branch": self.branch,
            "repair_cycles_used": self.repair_cycles_used,
            "events": [e.to_dict() for e in self.events],
            "report": self.report,
        }


class SupervisorBackend(Protocol):
    def git_status(self) -> CommandResult: ...
    def git_log_head(self) -> CommandResult: ...
    def create_branch(self, branch: str) -> CommandResult: ...
    def run_reasoning(self, goal: str, max_steps: int, initial_context: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]: ...
    def git_diff_stat(self) -> CommandResult: ...
    def git_diff(self) -> CommandResult: ...
    def run_tests(self, targets: Optional[List[str]] = None, timeout: int = 240) -> CommandResult: ...
    def run_test_diagnostics(self, timeout: int = 120) -> CommandResult: ...
    def smoke(self, endpoints: List[str], restart_command: str = "") -> CommandResult: ...
    def commit(self, message: str, files: Optional[List[str]] = None) -> CommandResult: ...
    def push_branch(self, branch: str) -> CommandResult: ...
    def open_pr(self, branch: str, title: str, body: str) -> CommandResult: ...
    def wait_ci(self) -> CommandResult: ...
    def merge_pr(self) -> CommandResult: ...
    def pull_main(self) -> CommandResult: ...
    def create_issue(self, title: str, body: str) -> CommandResult: ...
    def restore_dangerous_diff(self) -> CommandResult: ...


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
    ) -> CommandResult:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                env=self._subprocess_env(clean_for_tests=clean_env),
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
    ) -> Dict[str, Any]:
        payload = json.dumps({
            "project_root": str(self.project_root),
            "goal": goal,
            "max_steps": max_steps,
            "initial_context": initial_context,
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

    def run_tests(self, targets: Optional[List[str]] = None, timeout: int = 240) -> CommandResult:
        cmd = [str(self.project_root / ".venv/bin/python"), "-m", "pytest", "-q"]
        if targets:
            cmd.extend(targets)
        return self._run(cmd, timeout=timeout, clean_env=True)

    def run_test_diagnostics(self, timeout: int = 120) -> CommandResult:
        cmd = [
            str(self.project_root / ".venv/bin/python"),
            "-m",
            "pytest",
            "-x",
            "-vv",
        ]
        return self._run(cmd, timeout=timeout, clean_env=True)

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

    def create_issue(self, title: str, body: str) -> CommandResult:
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


def classify_failure(
    reasoning_result: Optional[Dict[str, Any]] = None,
    diff: str = "",
    targeted_tests: Optional[CommandResult] = None,
    full_tests: Optional[CommandResult] = None,
    smoke: Optional[CommandResult] = None,
) -> str:
    text = "\n".join([
        diff or "",
        targeted_tests.error if targeted_tests else "",
        targeted_tests.output if targeted_tests else "",
        full_tests.error if full_tests else "",
        full_tests.output if full_tests else "",
    ])
    if _has_destructive_diff(diff):
        return "destructive_diff"
    if reasoning_result:
        stop = str(reasoning_result.get("stop_reason", ""))
        status = str(reasoning_result.get("status", ""))
        if stop == "reasoning_timeout":
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
    if targeted_tests and not targeted_tests.success:
        return "pytest_failure"
    if full_tests and not full_tests.success:
        return "pytest_failure"
    if smoke and not smoke.success:
        return "infrastructure_bug"
    return "infrastructure_bug"


def _has_destructive_diff(diff: str) -> bool:
    dangerous_tokens = [".env", ".venv", "__pycache__", ".pytest_cache", ".igris"]
    if any(token in diff for token in dangerous_tokens):
        return True
    removed_lines = [line for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")]
    critical = ("def create_app", "class ", "import ")
    return any(any(token in line for token in critical) for line in removed_lines)


class SelfRepairSupervisor:
    def __init__(self, project_root: str, backend: Optional[SupervisorBackend] = None):
        self.project_root = project_root
        self.backend = backend or LocalSupervisorBackend(project_root)

    def run(
        self,
        config: RankSupervisorConfig,
        run: Optional[SupervisorRun] = None,
    ) -> SupervisorRun:
        run = run or SupervisorRun(run_id=uuid.uuid4().hex[:12], rank_id=config.rank_id)
        run.add("start", "running", "Supervisor started", dry_run=config.dry_run)
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
        baseline = self.backend.run_tests(timeout=config.test_timeout_seconds)
        run.add("baseline_tests", "success" if baseline.success else "failure", _command_detail(baseline))
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
        if not smoke.success:
            return self._blocked(run, "infrastructure_bug", "Baseline smoke failed")

        repair_cycles = 0
        for attempt in range(1, config.max_rank_attempts + 1):
            branch = f"rank-{config.rank_id.lower()}-{int(time.time())}-{attempt}"
            run.branch = branch
            branch_result = self.backend.create_branch(branch)
            run.add("rank_branch", "success" if branch_result.success else "failure", _command_detail(branch_result), branch=branch)
            if not branch_result.success:
                return self._blocked(run, "infrastructure_bug", "Could not create rank branch")

            run.add(
                "rank_reasoning",
                "running",
                "Running supervised rank reasoning",
                timeout_seconds=config.reasoning_timeout_seconds,
            )
            reasoning = self.backend.run_reasoning(
                config.goal,
                max_steps=220,
                initial_context={
                    "rank_test": config.rank_id,
                    "project_root": self.project_root,
                    "must_not_push_directly_to_main": True,
                    "must_not_merge_if_tests_fail": True,
                    "suppress_human_gate": True,
                    "supervised": True,
                },
                timeout=config.reasoning_timeout_seconds,
            )
            run.add("rank_reasoning", str(reasoning.get("status", "")), reasoning.get("final_summary", ""), loop_id=reasoning.get("loop_id", ""))

            diff_stat = self.backend.git_diff_stat()
            diff = self.backend.git_diff()
            run.add("diff_stat", "success" if diff_stat.success else "failure", _command_detail(diff_stat))

            if _has_destructive_diff(diff.output):
                run.add("safety", "blocked", "Destructive diff detected")
                self.backend.restore_dangerous_diff()
                failure = "destructive_diff"
            else:
                if config.targeted_tests:
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
                    )
                else:
                    targeted = CommandResult(True, "No targeted tests configured")
                run.add(
                    "full_pytest",
                    "running",
                    "Running full pytest",
                    timeout_seconds=config.test_timeout_seconds,
                )
                full = self.backend.run_tests(timeout=config.test_timeout_seconds)
                run.add("smoke", "running", "Running final smoke")
                final_smoke = self.backend.smoke(config.required_smoke_endpoints, restart_command)
                run.add("targeted_tests", "success" if targeted.success else "failure", _command_detail(targeted))
                run.add("full_pytest", "success" if full.success else "failure", _command_detail(full))
                run.add("smoke", "success" if final_smoke.success else "failure", _command_detail(final_smoke))
                if self._rank_passed(reasoning, diff_stat, targeted, full, final_smoke):
                    return self._complete_rank(run, config, branch)
                failure = classify_failure(reasoning, diff.output, targeted, full, final_smoke)

            run.failure_class = failure
            run.add("failure", "classified", failure)
            if failure not in REPAIRABLE_FAILURES or repair_cycles >= config.max_repair_cycles:
                return self._blocked(run, failure, "Rank failed and repair budget is exhausted or not repairable")

            repair_cycles += 1
            run.repair_cycles_used = repair_cycles
            if not self._repair_cycle(run, config, failure, repair_cycles):
                return self._blocked(run, failure, "Repair cycle failed validation")

        return self._blocked(run, run.failure_class or "max_rank_attempts", "Rank attempts exhausted")

    def _rank_passed(
        self,
        reasoning: Dict[str, Any],
        diff_stat: CommandResult,
        targeted: CommandResult,
        full: CommandResult,
        smoke: CommandResult,
    ) -> bool:
        return (
            reasoning.get("status") == "finished"
            and reasoning.get("files_modified")
            and bool(diff_stat.output.strip())
            and targeted.success
            and full.success
            and smoke.success
        )

    def _repair_cycle(self, run: SupervisorRun, config: RankSupervisorConfig, failure: str, cycle: int) -> bool:
        title = f"{config.rank_id}: supervised repair for {failure}"
        body = f"Supervisor detected {failure} during run {run.run_id}."
        if config.allow_github_pr and not config.dry_run:
            issue = self.backend.create_issue(title, body)
            run.add("repair_issue", "success" if issue.success else "failure", _command_detail(issue))
        else:
            run.add("repair_issue", "dry_run", title)
        repair_goal = (
            f"Fix IGRIS infrastructure failure '{failure}' observed during supervised "
            f"{config.rank_id}. Keep changes minimal, add tests, run pytest, do not push."
        )
        result = self.backend.run_reasoning(
            repair_goal,
            max_steps=160,
            initial_context={"repair_cycle": cycle, "failure_class": failure, "supervised_repair": True},
            timeout=config.reasoning_timeout_seconds,
        )
        run.add("repair_reasoning", str(result.get("status", "")), result.get("final_summary", ""))
        if result.get("status") != "finished":
            restore = self.backend.restore_dangerous_diff()
            run.add("repair_restore", "success" if restore.success else "failure", _command_detail(restore))
            return False
        run.add(
            "repair_tests",
            "running",
            "Running repair validation pytest",
            timeout_seconds=config.test_timeout_seconds,
        )
        tests = self.backend.run_tests(timeout=config.test_timeout_seconds)
        run.add("repair_tests", "success" if tests.success else "failure", _command_detail(tests))
        if not tests.success:
            restore = self.backend.restore_dangerous_diff()
            run.add("repair_restore", "success" if restore.success else "failure", _command_detail(restore))
            return False
        return True

    def _complete_rank(self, run: SupervisorRun, config: RankSupervisorConfig, branch: str) -> SupervisorRun:
        if config.dry_run:
            run.add("github", "dry_run", "Commit/PR/merge skipped by dry_run")
        else:
            commit = self.backend.commit(f"feat: complete supervised {config.rank_id}", ["igris", "tests"])
            run.add("commit", "success" if commit.success else "failure", _command_detail(commit))
            if not commit.success:
                return self._blocked(run, "infrastructure_bug", "Commit failed")
            if config.allow_github_pr:
                push = self.backend.push_branch(branch)
                run.add("push", "success" if push.success else "failure", _command_detail(push))
                pr = self.backend.open_pr(branch, f"feat: supervised {config.rank_id}", self._pr_body(run))
                run.add("pr", "success" if pr.success else "failure", _command_detail(pr))
                ci = self.backend.wait_ci()
                run.add("ci", "success" if ci.success else "failure", _command_detail(ci))
                if config.allow_merge_if_green and ci.success:
                    merge = self.backend.merge_pr()
                    run.add("merge", "success" if merge.success else "failure", _command_detail(merge))
                    pull = self.backend.pull_main()
                    run.add("pull_main", "success" if pull.success else "failure", _command_detail(pull))
        run.status = "completed"
        run.outcome = "Completed"
        run.report = {"autonomous": True, "manual_remaining": "real GitHub merge is gated unless enabled"}
        return run

    def _blocked(self, run: SupervisorRun, failure: str, detail: str) -> SupervisorRun:
        run.status = "blocked"
        run.outcome = "Blocked"
        run.failure_class = failure
        run.add("blocked", "blocked", detail)
        run.report = {"autonomous": False, "blocked_reason": detail}
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
    run = supervisor.run(config)
    with RUN_LOCK:
        RUN_STORE[run.run_id] = run
    return run


def start_supervised_rank_async(data: Dict[str, Any], project_root: str) -> SupervisorRun:
    """Create a run immediately and execute it in a background worker."""
    payload = dict(data)
    payload["defer_service_restart"] = True
    config = RankSupervisorConfig.from_dict(payload)
    run = SupervisorRun(run_id=uuid.uuid4().hex[:12], rank_id=config.rank_id)
    run.add("queued", "running", "Supervisor run accepted for background execution")
    with RUN_LOCK:
        RUN_STORE[run.run_id] = run

    def _worker() -> None:
        try:
            supervisor = SelfRepairSupervisor(project_root=project_root)
            supervisor.run(config, run=run)
        except Exception as exc:
            run.status = "blocked"
            run.outcome = "Blocked"
            run.failure_class = "supervisor_bug"
            run.add("exception", "blocked", str(exc))
            run.report = {"autonomous": False, "blocked_reason": "Supervisor worker crashed"}

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


def list_supervised_runs() -> List[SupervisorRun]:
    with RUN_LOCK:
        return list(RUN_STORE.values())
