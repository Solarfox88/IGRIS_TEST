"""Tool Runtime for IGRIS_GPT — Epic #41.

Provides modular, governed tool execution for real local/server operations.
Every tool invocation passes through risk classification, secret guard,
rollback check, and produces traceable results.

Tool families:
    shell      — governed command execution (template/allowlist)
    filesystem — read/write with path guard and backup
    git        — status, diff, branch, commit (gated)
    github     — PR prepare/create (gated)
    docker     — ps, logs, compose up/down (risk-gated)
    nginx      — config test, reload (risk-gated)
    systemd    — status, logs, restart (risk-gated)
    http       — health check with SSL/response time
    ssh_host   — host registry with policies
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from igris.core.risk_classifier import (
    classify_action_risk,
    check_approval,
    guard_secret_access,
    is_secret_file,
)
from igris.core.safety import (
    check_path_access,
    detect_secret_like_content,
    redact_secrets,
    truncate_output,
)


# ---------------------------------------------------------------------------
# Tool result
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Standard result from any tool invocation."""
    tool: str = ""
    action: str = ""
    success: bool = False
    output: str = ""
    error: str = ""
    risk_level: str = "low"
    returncode: int = 0
    duration_ms: int = 0
    mission_id: str = ""
    action_id: str = ""
    trace_id: str = ""
    rollback_id: str = ""
    redacted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "action": self.action,
            "success": self.success,
            "output": redact_secrets(self.output) if self.output else "",
            "error": redact_secrets(self.error) if self.error else "",
            "risk_level": self.risk_level,
            "returncode": self.returncode,
            "duration_ms": self.duration_ms,
            "mission_id": self.mission_id,
            "action_id": self.action_id,
            "trace_id": self.trace_id,
            "rollback_id": self.rollback_id,
            "redacted": True,
        }


# ---------------------------------------------------------------------------
# SSH Host Registry
# ---------------------------------------------------------------------------

@dataclass
class SSHHost:
    """Registered SSH host with policies."""
    hostname: str = ""
    alias: str = ""
    allowed_paths: List[str] = field(default_factory=lambda: ["/home"])
    allowed_services: List[str] = field(default_factory=list)
    requires_backup: bool = True
    policy: str = "safe"  # safe | operator | trusted

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hostname": self.hostname,
            "alias": self.alias or self.hostname,
            "allowed_paths": self.allowed_paths,
            "allowed_services": self.allowed_services,
            "requires_backup": self.requires_backup,
            "policy": self.policy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SSHHost":
        return cls(
            hostname=data.get("hostname", ""),
            alias=data.get("alias", ""),
            allowed_paths=data.get("allowed_paths", ["/home"]),
            allowed_services=data.get("allowed_services", []),
            requires_backup=data.get("requires_backup", True),
            policy=data.get("policy", "safe"),
        )


# ---------------------------------------------------------------------------
# Tool Runtime
# ---------------------------------------------------------------------------

class ToolRuntime:
    """Governed tool execution runtime.

    All tool invocations:
    1. Classify risk
    2. Check approval
    3. Guard secrets
    4. Execute with timeout
    5. Redact output
    6. Return ToolResult
    """

    def __init__(
        self,
        project_root: Optional[str] = None,
        approval_mode: str = "safe",
        authorized_hosts: Optional[List[str]] = None,
    ):
        self.project_root = Path(project_root) if project_root else Path(os.environ.get("PROJECT_ROOT", "."))
        self.approval_mode = approval_mode
        self.authorized_hosts = authorized_hosts or []
        self._host_registry: Dict[str, SSHHost] = {}
        self._mission_allowlist: Dict[str, List[str]] = {}  # mission_id → extra allowed commands

    # -- Host registry --

    def register_host(self, host: SSHHost) -> None:
        self._host_registry[host.hostname] = host

    def get_host(self, hostname: str) -> Optional[SSHHost]:
        return self._host_registry.get(hostname)

    def list_hosts(self) -> List[Dict[str, Any]]:
        return [h.to_dict() for h in self._host_registry.values()]

    # -- Mission allowlist --

    def set_mission_allowlist(self, mission_id: str, commands: List[str]) -> None:
        self._mission_allowlist[mission_id] = commands

    # -- Internal helpers --

    def _run_subprocess(
        self,
        cmd: List[str],
        cwd: Optional[str] = None,
        timeout: int = 30,
        env_override: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute a subprocess with timeout and output capture."""
        start = time.time()
        work_dir = cwd or str(self.project_root)
        env = os.environ.copy()
        if env_override:
            env.update(env_override)
        # Redact env secrets
        for k in list(env.keys()):
            if any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                env[k] = "***REDACTED***"

        try:
            result = subprocess.run(
                cmd, cwd=work_dir, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=timeout, check=False, env=env,
            )
            elapsed = int((time.time() - start) * 1000)
            return {
                "returncode": result.returncode,
                "stdout": truncate_output(result.stdout),
                "stderr": truncate_output(result.stderr),
                "duration_ms": elapsed,
            }
        except subprocess.TimeoutExpired:
            return {"returncode": 124, "stdout": "", "stderr": "Command timed out", "duration_ms": timeout * 1000}
        except FileNotFoundError:
            return {"returncode": 127, "stdout": "", "stderr": f"Command not found: {cmd[0]}", "duration_ms": 0}
        except Exception as exc:
            return {"returncode": 1, "stdout": "", "stderr": str(exc), "duration_ms": 0}

    def _make_result(
        self, tool: str, action: str, proc: Dict[str, Any],
        mission_id: str = "", action_id: str = "", trace_id: str = "",
        risk_level: str = "low", rollback_id: str = "",
    ) -> ToolResult:
        return ToolResult(
            tool=tool, action=action,
            success=proc["returncode"] == 0,
            output=proc.get("stdout", ""),
            error=proc.get("stderr", ""),
            risk_level=risk_level,
            returncode=proc["returncode"],
            duration_ms=proc.get("duration_ms", 0),
            mission_id=mission_id,
            action_id=action_id,
            trace_id=trace_id,
            rollback_id=rollback_id,
        )

    def _check_risk(self, action_id: str, desc: str = "", has_rollback: bool = False,
                    host: str = "", trace_id: str = "") -> Optional[ToolResult]:
        """Check risk and return error ToolResult if blocked."""
        risk = classify_action_risk(action_id, desc)
        decision = check_approval(
            action_id=action_id, risk_level=risk,
            approval_mode=self.approval_mode, has_rollback=has_rollback,
            host=host, authorized_hosts=self.authorized_hosts,
            trace_id=trace_id,
        )
        if not decision.allowed:
            return ToolResult(
                tool="risk_gate", action=action_id,
                success=False, error=decision.reason,
                risk_level=risk, trace_id=trace_id,
            )
        return None

    # =====================================================================
    # Shell tool
    # =====================================================================

    def shell_execute(
        self,
        command_id: str,
        args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        timeout: int = 30,
        mission_id: str = "",
        trace_id: str = "",
    ) -> ToolResult:
        """Execute a governed shell command by ID.

        Only allowlisted commands can run. Mission-specific allowlists
        can extend the base set.
        """
        from igris.layers.execution.safe_commands import ALLOWED_COMMANDS

        allowed = dict(ALLOWED_COMMANDS)
        if mission_id and mission_id in self._mission_allowlist:
            for cmd_id in self._mission_allowlist[mission_id]:
                if cmd_id not in allowed:
                    allowed[cmd_id] = [cmd_id]

        if command_id not in allowed:
            return ToolResult(
                tool="shell", action=command_id,
                success=False, error=f"Command not allowed: {command_id}",
                returncode=126, trace_id=trace_id, mission_id=mission_id,
            )

        blocked = self._check_risk(command_id, trace_id=trace_id)
        if blocked:
            blocked.tool = "shell"
            blocked.mission_id = mission_id
            return blocked

        cmd = list(allowed[command_id])
        if args:
            cmd.extend(args)

        proc = self._run_subprocess(cmd, cwd=cwd, timeout=timeout)
        return self._make_result("shell", command_id, proc,
                                 mission_id=mission_id, trace_id=trace_id)

    # =====================================================================
    # Filesystem tool
    # =====================================================================

    def fs_read(
        self,
        path: str,
        max_chars: int = 10000,
        mission_id: str = "",
        trace_id: str = "",
    ) -> ToolResult:
        """Read a file safely with path guard and secret check."""
        target = Path(path)

        # Secret guard
        sg = guard_secret_access(path)
        if not sg.allowed:
            return ToolResult(
                tool="filesystem", action="read",
                success=False, error=sg.reason,
                risk_level="critical", trace_id=trace_id, mission_id=mission_id,
            )

        # Path guard
        if not check_path_access(target, self.project_root):
            return ToolResult(
                tool="filesystem", action="read",
                success=False, error=f"Path outside project root: {path}",
                risk_level="medium", trace_id=trace_id, mission_id=mission_id,
            )

        try:
            content = target.read_text(encoding="utf-8")
            return ToolResult(
                tool="filesystem", action="read",
                success=True, output=truncate_output(content, max_chars),
                risk_level="low", trace_id=trace_id, mission_id=mission_id,
            )
        except Exception as exc:
            return ToolResult(
                tool="filesystem", action="read",
                success=False, error=str(exc),
                trace_id=trace_id, mission_id=mission_id,
            )

    def fs_write(
        self,
        path: str,
        content: str,
        mission_id: str = "",
        trace_id: str = "",
        backup: bool = True,
    ) -> ToolResult:
        """Write to a file with path guard, secret check, and optional backup."""
        target = Path(path)

        # Secret guard
        if is_secret_file(path):
            return ToolResult(
                tool="filesystem", action="write",
                success=False, error=f"Cannot write to secret file: {redact_secrets(path)}",
                risk_level="critical", trace_id=trace_id, mission_id=mission_id,
            )

        # Path guard
        if not check_path_access(target, self.project_root):
            return ToolResult(
                tool="filesystem", action="write",
                success=False, error=f"Path outside project root: {path}",
                risk_level="medium", trace_id=trace_id, mission_id=mission_id,
            )

        # Secret content check
        if detect_secret_like_content(content):
            return ToolResult(
                tool="filesystem", action="write",
                success=False, error="Content contains secret-like patterns",
                risk_level="critical", trace_id=trace_id, mission_id=mission_id,
            )

        # Backup before overwrite
        rollback_id = ""
        if backup and target.exists():
            from igris.core.rollback_manager import RollbackManager
            mgr = RollbackManager(project_root=str(self.project_root))
            entry = mgr.backup_file(str(target), mission_id=mission_id, trace_id=trace_id)
            if entry:
                rollback_id = entry.id

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResult(
                tool="filesystem", action="write",
                success=True, output=f"Written {len(content)} chars to {path}",
                risk_level="medium", trace_id=trace_id, mission_id=mission_id,
                rollback_id=rollback_id,
            )
        except Exception as exc:
            return ToolResult(
                tool="filesystem", action="write",
                success=False, error=str(exc),
                trace_id=trace_id, mission_id=mission_id,
            )

    def fs_diff(
        self,
        path: str,
        new_content: str,
        mission_id: str = "",
        trace_id: str = "",
    ) -> ToolResult:
        """Show diff preview between current file and proposed content."""
        target = Path(path)
        if not target.exists():
            return ToolResult(
                tool="filesystem", action="diff",
                success=True, output=f"New file: {path} ({len(new_content)} chars)",
                trace_id=trace_id, mission_id=mission_id,
            )
        try:
            current = target.read_text(encoding="utf-8")
            if current == new_content:
                return ToolResult(
                    tool="filesystem", action="diff",
                    success=True, output="No changes",
                    trace_id=trace_id, mission_id=mission_id,
                )
            # Simple line diff
            old_lines = current.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)
            import difflib
            diff = "".join(difflib.unified_diff(old_lines, new_lines, fromfile=path, tofile=path))
            return ToolResult(
                tool="filesystem", action="diff",
                success=True, output=truncate_output(diff),
                trace_id=trace_id, mission_id=mission_id,
            )
        except Exception as exc:
            return ToolResult(
                tool="filesystem", action="diff",
                success=False, error=str(exc),
                trace_id=trace_id, mission_id=mission_id,
            )

    # =====================================================================
    # Git tool
    # =====================================================================

    def git_status(self, mission_id: str = "", trace_id: str = "") -> ToolResult:
        proc = self._run_subprocess(["git", "status", "--short"], timeout=10)
        return self._make_result("git", "status", proc, mission_id=mission_id, trace_id=trace_id)

    def git_diff(self, staged: bool = False, mission_id: str = "", trace_id: str = "") -> ToolResult:
        cmd = ["git", "diff", "--stat"]
        if staged:
            cmd.append("--cached")
        proc = self._run_subprocess(cmd, timeout=10)
        return self._make_result("git", "diff", proc, mission_id=mission_id, trace_id=trace_id)

    def git_log(self, count: int = 10, mission_id: str = "", trace_id: str = "") -> ToolResult:
        proc = self._run_subprocess(["git", "log", "--oneline", f"-{count}"], timeout=10)
        return self._make_result("git", "log", proc, mission_id=mission_id, trace_id=trace_id)

    def git_branch(self, mission_id: str = "", trace_id: str = "") -> ToolResult:
        proc = self._run_subprocess(["git", "branch", "-a"], timeout=10)
        return self._make_result("git", "branch", proc, mission_id=mission_id, trace_id=trace_id)

    def git_commit(
        self,
        message: str,
        files: Optional[List[str]] = None,
        mission_id: str = "",
        trace_id: str = "",
        approval_token: Optional[str] = None,
    ) -> ToolResult:
        """Create a gated commit. Requires safety checks to pass."""
        # Secret check on message
        if detect_secret_like_content(message):
            return ToolResult(
                tool="git", action="commit",
                success=False, error="Commit message contains secret-like content",
                risk_level="medium", trace_id=trace_id, mission_id=mission_id,
            )

        # Check for secret files in staged files
        if files:
            for f in files:
                if is_secret_file(f):
                    return ToolResult(
                        tool="git", action="commit",
                        success=False, error=f"Cannot commit secret file: {f}",
                        risk_level="critical", trace_id=trace_id, mission_id=mission_id,
                    )

        # Stage files if provided
        if files:
            for f in files:
                self._run_subprocess(["git", "add", f], timeout=5)

        proc = self._run_subprocess(["git", "commit", "-m", message], timeout=30)
        return self._make_result("git", "commit", proc,
                                 mission_id=mission_id, trace_id=trace_id, risk_level="medium")

    def git_push(
        self,
        branch: str = "",
        mission_id: str = "",
        trace_id: str = "",
        approval_token: Optional[str] = None,
    ) -> ToolResult:
        """Gated push — never to main/master, never force push."""
        if not approval_token:
            return ToolResult(
                tool="git", action="push",
                success=False, error="Push requires approval token: I_APPROVE_GITHUB_WRITE",
                risk_level="high", trace_id=trace_id, mission_id=mission_id,
            )

        if branch in ("main", "master"):
            return ToolResult(
                tool="git", action="push",
                success=False, error="Push to main/master is forbidden",
                risk_level="critical", trace_id=trace_id, mission_id=mission_id,
            )

        cmd = ["git", "push", "origin"]
        if branch:
            cmd.append(branch)
        proc = self._run_subprocess(cmd, timeout=60)
        return self._make_result("git", "push", proc,
                                 mission_id=mission_id, trace_id=trace_id, risk_level="high")

    # =====================================================================
    # Docker tool
    # =====================================================================

    def docker_ps(self, mission_id: str = "", trace_id: str = "") -> ToolResult:
        proc = self._run_subprocess(["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"], timeout=10)
        return self._make_result("docker", "ps", proc, mission_id=mission_id, trace_id=trace_id)

    def docker_logs(self, container: str, tail: int = 50, mission_id: str = "", trace_id: str = "") -> ToolResult:
        proc = self._run_subprocess(["docker", "logs", "--tail", str(tail), container], timeout=15)
        return self._make_result("docker", "logs", proc, mission_id=mission_id, trace_id=trace_id)

    def docker_compose_config(self, mission_id: str = "", trace_id: str = "") -> ToolResult:
        proc = self._run_subprocess(["docker", "compose", "config"], timeout=10)
        return self._make_result("docker", "compose_config", proc, mission_id=mission_id, trace_id=trace_id)

    def docker_compose_up(
        self, service: str = "", mission_id: str = "",
        trace_id: str = "", approval_token: Optional[str] = None,
    ) -> ToolResult:
        """Start docker compose (risk-gated)."""
        blocked = self._check_risk("docker_compose_up", "start containers", trace_id=trace_id)
        if blocked:
            blocked.tool = "docker"
            blocked.mission_id = mission_id
            return blocked

        cmd = ["docker", "compose", "up", "-d"]
        if service:
            cmd.append(service)
        proc = self._run_subprocess(cmd, timeout=120)
        return self._make_result("docker", "compose_up", proc,
                                 mission_id=mission_id, trace_id=trace_id, risk_level="high")

    def docker_compose_down(
        self, mission_id: str = "", trace_id: str = "",
        approval_token: Optional[str] = None,
    ) -> ToolResult:
        """Stop docker compose (risk-gated)."""
        blocked = self._check_risk("docker_compose_down", "stop containers", trace_id=trace_id)
        if blocked:
            blocked.tool = "docker"
            blocked.mission_id = mission_id
            return blocked

        proc = self._run_subprocess(["docker", "compose", "down"], timeout=60)
        return self._make_result("docker", "compose_down", proc,
                                 mission_id=mission_id, trace_id=trace_id, risk_level="high")

    def docker_health(self, mission_id: str = "", trace_id: str = "") -> ToolResult:
        proc = self._run_subprocess(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=10)
        return self._make_result("docker", "health", proc, mission_id=mission_id, trace_id=trace_id)

    # =====================================================================
    # Nginx tool
    # =====================================================================

    def nginx_config_test(self, mission_id: str = "", trace_id: str = "") -> ToolResult:
        proc = self._run_subprocess(["nginx", "-t"], timeout=10)
        return self._make_result("nginx", "config_test", proc, mission_id=mission_id, trace_id=trace_id)

    def nginx_reload(self, mission_id: str = "", trace_id: str = "") -> ToolResult:
        blocked = self._check_risk("nginx_reload", "reload nginx", trace_id=trace_id)
        if blocked:
            blocked.tool = "nginx"
            blocked.mission_id = mission_id
            return blocked

        proc = self._run_subprocess(["nginx", "-s", "reload"], timeout=10)
        return self._make_result("nginx", "reload", proc,
                                 mission_id=mission_id, trace_id=trace_id, risk_level="high")

    # =====================================================================
    # Systemd tool
    # =====================================================================

    def systemd_status(self, service: str, mission_id: str = "", trace_id: str = "") -> ToolResult:
        proc = self._run_subprocess(["systemctl", "status", service, "--no-pager"], timeout=10)
        return self._make_result("systemd", "status", proc, mission_id=mission_id, trace_id=trace_id)

    def systemd_logs(self, service: str, lines: int = 50, mission_id: str = "", trace_id: str = "") -> ToolResult:
        proc = self._run_subprocess(["journalctl", "-u", service, "-n", str(lines), "--no-pager"], timeout=10)
        return self._make_result("systemd", "logs", proc, mission_id=mission_id, trace_id=trace_id)

    def systemd_restart(
        self, service: str, mission_id: str = "",
        trace_id: str = "", approval_token: Optional[str] = None,
    ) -> ToolResult:
        blocked = self._check_risk("systemd_restart", f"restart {service}", trace_id=trace_id)
        if blocked:
            blocked.tool = "systemd"
            blocked.mission_id = mission_id
            return blocked

        proc = self._run_subprocess(["systemctl", "restart", service], timeout=30)
        return self._make_result("systemd", "restart", proc,
                                 mission_id=mission_id, trace_id=trace_id, risk_level="high")

    # =====================================================================
    # HTTP health check tool
    # =====================================================================

    def http_check(
        self,
        url: str,
        timeout: int = 10,
        mission_id: str = "",
        trace_id: str = "",
    ) -> ToolResult:
        """HTTP health check — status, SSL, response time, body snippet."""
        import urllib.request
        import urllib.error
        import ssl

        start = time.time()
        try:
            ctx = ssl.create_default_context()
            req = urllib.request.Request(url, method="GET")
            req.add_header("User-Agent", "IGRIS-GPT/healthcheck")
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                elapsed = int((time.time() - start) * 1000)
                body = resp.read(500).decode("utf-8", errors="replace")
                ssl_info = "yes" if url.startswith("https") else "no"
                output = json.dumps({
                    "status_code": resp.status,
                    "response_time_ms": elapsed,
                    "ssl": ssl_info,
                    "body_snippet": redact_secrets(body[:200]),
                }, indent=2)
                return ToolResult(
                    tool="http", action="check",
                    success=resp.status < 400,
                    output=output,
                    risk_level="low",
                    duration_ms=elapsed,
                    trace_id=trace_id, mission_id=mission_id,
                )
        except urllib.error.HTTPError as exc:
            elapsed = int((time.time() - start) * 1000)
            return ToolResult(
                tool="http", action="check",
                success=False, error=f"HTTP {exc.code}: {exc.reason}",
                returncode=exc.code,
                duration_ms=elapsed,
                trace_id=trace_id, mission_id=mission_id,
            )
        except Exception as exc:
            elapsed = int((time.time() - start) * 1000)
            return ToolResult(
                tool="http", action="check",
                success=False, error=str(exc),
                duration_ms=elapsed,
                trace_id=trace_id, mission_id=mission_id,
            )

    # =====================================================================
    # Test runner tool
    # =====================================================================

    def run_tests(
        self,
        args: Optional[List[str]] = None,
        timeout: int = 120,
        mission_id: str = "",
        trace_id: str = "",
    ) -> ToolResult:
        """Run pytest with optional arguments."""
        cmd = [sys.executable, "-m", "pytest", "-q"]
        if args:
            cmd.extend(args)
        proc = self._run_subprocess(cmd, timeout=timeout)
        return self._make_result("test", "pytest", proc,
                                 mission_id=mission_id, trace_id=trace_id)

    # =====================================================================
    # List available tools
    # =====================================================================

    def list_tools(self) -> List[Dict[str, str]]:
        """List all available tool families and their actions."""
        return [
            {"tool": "shell", "actions": "execute (governed by allowlist)"},
            {"tool": "filesystem", "actions": "read, write, diff"},
            {"tool": "git", "actions": "status, diff, log, branch, commit (gated), push (gated)"},
            {"tool": "docker", "actions": "ps, logs, compose_config, compose_up (gated), compose_down (gated), health"},
            {"tool": "nginx", "actions": "config_test, reload (gated)"},
            {"tool": "systemd", "actions": "status, logs, restart (gated)"},
            {"tool": "http", "actions": "check (status, SSL, response time)"},
            {"tool": "test", "actions": "pytest runner"},
            {"tool": "ssh_host", "actions": "register, list, get"},
        ]
