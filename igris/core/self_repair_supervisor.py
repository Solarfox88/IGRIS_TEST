"""Autonomous self-repair supervisor for controlled rank missions.

The supervisor coordinates an IGRIS rank attempt and bounded infrastructure
repair cycles. It does not expose free-form shell execution: the default
backend runs fixed argv commands only, and tests can inject a fake backend.
"""

from __future__ import annotations

import json
import os
import re
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
    "missing_ui_visibility",
    "wrong_file_edit",
    "infrastructure_bug",
    "invalid_bootstrap",
    "syntax_error",
}

RETRYABLE_REPAIR_FAILURES = {
    "reasoning_loop_blocked",
    "missing_ui_visibility",
    "missing_tests",
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
            targeted_tests=_infer_targeted_tests(
                str(data.get("goal", "")),
                list(data.get("targeted_tests", [])),
            ),
            dry_run=_infer_dry_run(data),
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
    if smoke and not smoke.success:
        smoke_text = "\n".join([smoke.output or "", smoke.error or ""]).lower()
        if "bootstrap" in smoke_text or "invalid bootstrap" in smoke_text:
            return "invalid_bootstrap"
        return "infrastructure_bug"
    return "infrastructure_bug"


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

    critical = ("def create_app", "class ", "import ")
    return any(any(token in line for token in critical) for line in python_removed_lines)


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
        attempt = 1
        attempt_limit = config.max_rank_attempts
        while attempt <= attempt_limit:
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
            )

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
                )
            targeted = CommandResult(True, "Targeted tests skipped")
            full = CommandResult(True, "Full pytest skipped")
            final_smoke = CommandResult(True, "Final smoke skipped")
            failure = ""

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
                )
            rank_passed = self._rank_passed(reasoning, diff_stat, targeted, full, final_smoke)
            if not failure:
                if rank_passed:
                    if ui_visibility_required and not ui_visibility_changed:
                        failure = "missing_ui_visibility"
                else:
                    failure = classify_failure(reasoning, diff.output, targeted, full, final_smoke)
            reasoning_text = "\n".join(
                str(reasoning.get(key, ""))
                for key in ("final_summary", "error", "stop_reason")
            )
            if failure == "infrastructure_bug" and _is_llm_provider_unavailable(reasoning_text):
                return self._blocked(run, "infrastructure_bug", "No suitable LLM provider available")

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
                return self._complete_rank(
                    run,
                    config,
                    branch,
                    completion_mode=completion_mode,
                    runtime_refresh_required=runtime_refresh_required,
                )

            run.failure_class = failure
            run.add("failure", "classified", failure)
            if failure not in REPAIRABLE_FAILURES or repair_cycles >= config.max_repair_cycles:
                return self._blocked(run, failure, "Rank failed and repair budget is exhausted or not repairable")

            repair_cycles += 1
            run.repair_cycles_used = repair_cycles
            if not self._repair_cycle(run, config, failure, repair_cycles):
                return self._blocked(run, failure, "Repair cycle failed validation")
            if attempt == attempt_limit and repair_cycles < config.max_repair_cycles:
                attempt_limit += 1
                run.add(
                    "rank_attempt_extension",
                    "running",
                    "Extending rank attempts after successful repair on final configured attempt.",
                    attempt_limit=attempt_limit,
                    repair_cycles_used=repair_cycles,
                )
            attempt += 1

        return self._blocked(run, run.failure_class or "max_rank_attempts", "Rank attempts exhausted")

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

    def _repair_cycle(self, run: SupervisorRun, config: RankSupervisorConfig, failure: str, cycle: int) -> bool:
        title = f"{config.rank_id}: supervised repair for {failure}"
        body = f"Supervisor detected {failure} during run {run.run_id}."
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
        repair_goal = (
            f"Fix IGRIS infrastructure failure '{failure}' observed during supervised "
            f"{config.rank_id}. Keep changes minimal, add tests, run pytest, do not push."
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
        repair_context = self._rank_initial_context(config)
        repair_context.update({
            "repair_cycle": cycle,
            "failure_class": failure,
            "supervised_repair": True,
            "repair_goal": repair_goal,
        })
        result = self.backend.run_reasoning(
            repair_goal,
            max_steps=160,
            initial_context=repair_context,
            timeout=config.reasoning_timeout_seconds,
        )
        run.add("repair_reasoning", str(result.get("status", "")), result.get("final_summary", ""))
        diff_stat = self.backend.git_diff_stat()
        diff = self.backend.git_diff()
        run.add("repair_diff_stat", "success" if diff_stat.success else "failure", _command_detail(diff_stat))
        if not diff_stat.success:
            restore = self.backend.restore_dangerous_diff()
            run.add("repair_restore", "success" if restore.success else "failure", _command_detail(restore))
            return False
        if _has_destructive_diff(diff.output):
            restore = self.backend.restore_dangerous_diff()
            run.add("repair_restore", "success" if restore.success else "failure", _command_detail(restore))
            if failure in {"reasoning_loop_blocked", "missing_ui_visibility", "missing_tests", "max_steps"}:
                run.add(
                    "repair_retry",
                    "running",
                    "Destructive repair diff was rejected; retrying with remaining budget.",
                    failure_class="destructive_diff",
                )
                return True
            return False
        if _has_invalid_fastapi_bootstrap_diff(diff.output):
            restore = self.backend.restore_dangerous_diff()
            run.add(
                "repair_restore",
                "success" if restore.success else "failure",
                "Invalid FastAPI bootstrap diff rejected before repair validation",
            )
            run.add(
                "repair_retry",
                "running",
                "Invalid FastAPI bootstrap diff was rejected; retrying with remaining budget.",
                failure_class="invalid_bootstrap",
            )
            return True
        if failure == "missing_tests" and not _is_valid_missing_tests_repair_diff(diff.output, config.goal):
            restore = self.backend.restore_dangerous_diff()
            run.add(
                "repair_restore",
                "success" if restore.success else "failure",
                "Missing-tests repair diff rejected before validation",
            )
            if not restore.success:
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
                    restore = self.backend.restore_dangerous_diff()
                    run.add(
                        "repair_restore",
                        "success" if restore.success else "failure",
                        "Scaffolded missing-tests diff was invalid; restored.",
                    )
                run.add(
                    "repair_retry",
                    "running",
                    "Missing-tests repair diff was rejected; retrying with remaining budget.",
                    failure_class="wrong_file_edit",
                )
                return True
        if self._goal_requires_ui_visibility(config.goal) and _is_product_only_ui_task_diff(diff.output):
            allow_safe_ui_repair = (
                _has_ui_surface_change(diff.output)
                and failure in {"missing_ui_visibility", "reasoning_loop_blocked"}
            )
            if failure == "pytest_failure":
                allow_safe_ui_repair = True
            if not allow_safe_ui_repair:
                restore = self.backend.restore_dangerous_diff()
                run.add(
                    "repair_restore",
                    "success" if restore.success else "failure",
                    "Product-only UI task diff rejected before repair validation",
                )
                run.add(
                    "repair_retry",
                    "running",
                    "Product-only UI task diff was rejected; retrying with remaining budget.",
                    failure_class="wrong_file_edit",
                )
                return True
        if self._goal_requires_ui_visibility(config.goal) and not _is_valid_ui_test_diff(diff.output):
            restore = self.backend.restore_dangerous_diff()
            run.add(
                "repair_restore",
                "success" if restore.success else "failure",
                "Invalid UI test diff rejected before repair validation",
            )
            run.add(
                "repair_retry",
                "running",
                "Invalid UI test diff was rejected; retrying with remaining budget.",
                failure_class="wrong_file_edit",
            )
            return True
        if not diff.output.strip():
            restore = self.backend.restore_dangerous_diff()
            run.add("repair_restore", "success" if restore.success else "failure", _command_detail(restore))
            if failure in RETRYABLE_REPAIR_FAILURES:
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
        tests = self.backend.run_tests(timeout=config.test_timeout_seconds)
        run.add("repair_tests", "success" if tests.success else "failure", _command_detail(tests))
        if not tests.success:
            if failure == "missing_tests" and _is_valid_missing_tests_repair_diff(diff.output, config.goal):
                run.add(
                    "repair_completion",
                    "degraded",
                    "Preserved valid missing-tests scaffold despite failing full pytest; continuing rank attempts.",
                )
                return True
            restore = self.backend.restore_dangerous_diff()
            run.add("repair_restore", "success" if restore.success else "failure", _command_detail(restore))
            if failure in RETRYABLE_REPAIR_FAILURES:
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
        return True

    def _complete_rank(
        self,
        run: SupervisorRun,
        config: RankSupervisorConfig,
        branch: str,
        *,
        completion_mode: str = "direct",
        runtime_refresh_required: bool = False,
    ) -> SupervisorRun:
        restart_command = config.service_restart_command if not config.defer_service_restart else ""
        post_merge_smoke: Optional[CommandResult] = None
        manual_remaining = ""
        if config.dry_run:
            manual_remaining = "delivery skipped by dry_run"
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
                    if config.defer_service_restart and runtime_refresh_required:
                        run.add(
                            "post_merge_smoke",
                            "deferred",
                            "Post-merge smoke deferred until the runtime is restarted and refreshed.",
                            runtime_refresh_required=True,
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
                        if not post_merge_smoke.success:
                            run.status = "blocked"
                            run.outcome = "Blocked"
                            run.failure_class = "infrastructure_bug"
                            run.report = {
                                "autonomous": True,
                                "manual_remaining": "post-merge verification failed",
                                "completion_mode": completion_mode,
                                "degraded_completion": completion_mode != "direct" or runtime_refresh_required,
                                "post_merge_smoke": False,
                                "runtime_refresh_required": runtime_refresh_required,
                            }
                            return run
                elif config.allow_merge_if_green and not ci.success:
                    manual_remaining = "merge skipped because CI is not green"
                else:
                    manual_remaining = "merge disabled by config"
            else:
                manual_remaining = "GitHub PR/merge workflow disabled by config"
        run.status = "completed"
        run.outcome = "Completed"
        run.report = {
            "autonomous": True,
            "manual_remaining": manual_remaining,
            "completion_mode": completion_mode,
            "degraded_completion": completion_mode != "direct" or runtime_refresh_required,
            "post_merge_smoke": False if post_merge_smoke is None else post_merge_smoke.success,
            "runtime_refresh_required": runtime_refresh_required,
        }
        return run

    def _complete_noop(
        self,
        run: SupervisorRun,
        *,
        completion_mode: str,
        runtime_refresh_required: bool,
        detail: str,
        post_merge_smoke: bool,
    ) -> SupervisorRun:
        run.add("completion", "degraded", detail, mode=completion_mode)
        run.status = "completed"
        run.outcome = "Completed"
        run.report = {
            "autonomous": True,
            "manual_remaining": "",
            "completion_mode": completion_mode,
            "degraded_completion": True,
            "post_merge_smoke": post_merge_smoke,
            "runtime_refresh_required": runtime_refresh_required,
            "no_op_completion": True,
        }
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
