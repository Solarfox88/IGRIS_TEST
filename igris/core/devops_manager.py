"""DevOps Manager — Epic #1076.

Host registry (persist/load from JSON), server policy enforcement,
deploy patterns with preflight/postcheck, and HTTP smoke-test evidence.

Designed to be imported by igris.web.routers.routes_10 for the /api/devops/*
endpoints.  All operations are best-effort: each step records its own
outcome so that partial failures are visible rather than silent.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class HostConfig:
    """A registered deployment host with its policy."""

    hostname: str
    alias: str = ""
    policy: str = "safe"           # safe | operator | trusted
    allowed_paths: List[str] = field(default_factory=lambda: ["/home"])
    allowed_services: List[str] = field(default_factory=list)
    requires_backup: bool = True
    health_url: str = ""           # URL for post-deploy health check

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HostConfig":
        return cls(
            hostname=data.get("hostname", ""),
            alias=data.get("alias", ""),
            policy=data.get("policy", "safe"),
            allowed_paths=data.get("allowed_paths", ["/home"]),
            allowed_services=data.get("allowed_services", []),
            requires_backup=data.get("requires_backup", True),
            health_url=data.get("health_url", ""),
        )


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------

# What actions are permitted per policy tier
_POLICY_ACTIONS: Dict[str, List[str]] = {
    "safe": ["status", "logs", "health", "list"],
    "operator": ["status", "logs", "health", "list", "restart", "deploy"],
    "trusted": ["status", "logs", "health", "list", "restart", "deploy", "shell", "backup"],
}

_VALID_POLICIES = ("safe", "operator", "trusted")


def check_action_allowed(policy: str, action: str) -> Dict[str, Any]:
    """Return whether *action* is permitted under *policy*."""
    allowed_actions = _POLICY_ACTIONS.get(policy, _POLICY_ACTIONS["safe"])
    allowed = action in allowed_actions
    return {
        "policy": policy,
        "action": action,
        "allowed": allowed,
        "reason": (
            f"action '{action}' is permitted under policy '{policy}'"
            if allowed
            else f"action '{action}' is not permitted under policy '{policy}'; "
                 f"allowed: {allowed_actions}"
        ),
        "allowed_actions": allowed_actions,
    }


# ---------------------------------------------------------------------------
# DevOpsManager
# ---------------------------------------------------------------------------

class DevOpsManager:
    """Manages host registry, deploy flows, and smoke tests."""

    #: Relative path inside project root for the host registry JSON.
    _REGISTRY_FILE = ".igris/devops_hosts.json"

    def __init__(self, project_root: str) -> None:
        self.project_root = Path(project_root)
        self._registry_path = self.project_root / self._REGISTRY_FILE
        self._hosts: Dict[str, HostConfig] = {}
        self._load_registry()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_registry(self) -> None:
        """Load hosts from the on-disk JSON file (if present)."""
        if self._registry_path.exists():
            try:
                raw = json.loads(self._registry_path.read_text(encoding="utf-8"))
                for entry in raw.get("hosts", []):
                    h = HostConfig.from_dict(entry)
                    self._hosts[h.hostname] = h
            except Exception:
                pass  # corrupt file → start with empty registry

    def _save_registry(self) -> None:
        """Persist the host registry to disk."""
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"hosts": [h.to_dict() for h in self._hosts.values()]}
        self._registry_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Host registry API
    # ------------------------------------------------------------------

    def list_hosts(self) -> List[Dict[str, Any]]:
        """Return all registered hosts as dicts."""
        return [h.to_dict() for h in self._hosts.values()]

    def get_host(self, hostname: str) -> Optional[HostConfig]:
        """Return a registered host or None."""
        return self._hosts.get(hostname)

    def register_host(self, config: HostConfig) -> Dict[str, Any]:
        """Register (or update) a host.  Persists immediately."""
        if config.policy not in _VALID_POLICIES:
            return {
                "registered": False,
                "hostname": config.hostname,
                "error": f"invalid policy '{config.policy}'; must be one of {_VALID_POLICIES}",
            }
        self._hosts[config.hostname] = config
        self._save_registry()
        return {"registered": True, "hostname": config.hostname, "policy": config.policy}

    def remove_host(self, hostname: str) -> Dict[str, Any]:
        """Remove a host from the registry."""
        if hostname not in self._hosts:
            return {"removed": False, "hostname": hostname, "error": "host not found"}
        del self._hosts[hostname]
        self._save_registry()
        return {"removed": True, "hostname": hostname}

    def check_policy(self, hostname: str, action: str) -> Dict[str, Any]:
        """Check whether *action* is allowed on *hostname*."""
        host = self._hosts.get(hostname)
        if host is None:
            return {
                "allowed": False,
                "hostname": hostname,
                "action": action,
                "reason": f"host '{hostname}' is not registered",
            }
        result = check_action_allowed(host.policy, action)
        result["hostname"] = hostname
        return result

    # ------------------------------------------------------------------
    # Preflight check
    # ------------------------------------------------------------------

    def run_preflight(
        self,
        hostname: Optional[str] = None,
        min_disk_pct_free: int = 10,
    ) -> Dict[str, Any]:
        """Run pre-deploy preflight checks locally.

        Checks:
        - Disk space (df on project root volume)
        - Git working tree state (clean / dirty)
        - Service reachability (nc port 7778)

        Returns a dict with individual check results and an overall ``ok`` flag.
        """
        checks: Dict[str, Any] = {}
        ts = time.time()

        # 1. Disk space
        try:
            _r = subprocess.run(
                ["df", "-P", str(self.project_root)],
                capture_output=True, text=True, timeout=5,
            )
            if _r.returncode == 0:
                lines = _r.stdout.strip().splitlines()
                if len(lines) >= 2:
                    parts = lines[1].split()
                    use_pct_str = parts[4].rstrip("%") if len(parts) > 4 else "100"
                    used_pct = int(use_pct_str)
                    free_pct = 100 - used_pct
                    checks["disk"] = {
                        "ok": free_pct >= min_disk_pct_free,
                        "used_pct": used_pct,
                        "free_pct": free_pct,
                        "min_free_required": min_disk_pct_free,
                    }
                else:
                    checks["disk"] = {"ok": False, "error": "could not parse df output"}
            else:
                checks["disk"] = {"ok": False, "error": _r.stderr.strip()[:200]}
        except Exception as exc:
            checks["disk"] = {"ok": False, "error": str(exc)[:200]}

        # 2. Git working-tree state
        try:
            _g = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
                cwd=str(self.project_root),
            )
            is_clean = _g.returncode == 0 and not _g.stdout.strip()
            checks["git"] = {"ok": True, "clean": is_clean, "dirty": not is_clean}
        except Exception as exc:
            checks["git"] = {"ok": False, "error": str(exc)[:200]}

        # 3. IGRIS service reachability
        try:
            _nc = subprocess.run(
                ["nc", "-z", "-w", "2", "localhost", "7778"],
                capture_output=True, timeout=5,
            )
            reachable = _nc.returncode == 0
            checks["service"] = {
                "ok": True,  # non-blocking: just report, don't fail preflight
                "reachable": reachable,
                "port": 7778,
            }
        except Exception as exc:
            checks["service"] = {"ok": True, "reachable": False, "error": str(exc)[:200]}

        overall_ok = all(c.get("ok", False) for c in checks.values())
        return {
            "ok": overall_ok,
            "hostname": hostname or "localhost",
            "timestamp": ts,
            "checks": checks,
        }

    # ------------------------------------------------------------------
    # Postcheck
    # ------------------------------------------------------------------

    def run_postcheck(
        self,
        hostname: Optional[str] = None,
        health_url: str = "",
    ) -> Dict[str, Any]:
        """Post-deploy verification.

        Checks:
        - Service reachability (port 7778)
        - HTTP health endpoint (if health_url given)

        Returns a dict with results and an overall ``ok`` flag.
        """
        checks: Dict[str, Any] = {}

        # Service port
        try:
            _nc = subprocess.run(
                ["nc", "-z", "-w", "3", "localhost", "7778"],
                capture_output=True, timeout=6,
            )
            checks["service"] = {
                "ok": _nc.returncode == 0,
                "port": 7778,
                "reachable": _nc.returncode == 0,
            }
        except Exception as exc:
            checks["service"] = {"ok": False, "error": str(exc)[:200]}

        # HTTP health endpoint
        if health_url:
            smoke = self.run_smoke_test(health_url)
            checks["http_health"] = {
                "ok": smoke["ok"],
                "url": health_url,
                "status_code": smoke.get("status_code"),
                "response_time_ms": smoke.get("response_time_ms"),
            }

        overall_ok = all(c.get("ok", False) for c in checks.values())
        return {
            "ok": overall_ok,
            "hostname": hostname or "localhost",
            "checks": checks,
        }

    # ------------------------------------------------------------------
    # Deploy flow
    # ------------------------------------------------------------------

    def run_deploy(
        self,
        strategy: str = "git_pull_restart",
        hostname: Optional[str] = None,
        health_url: str = "",
        dry_run: bool = False,
        min_disk_pct_free: int = 10,
    ) -> Dict[str, Any]:
        """Execute a deploy cycle: preflight → action → postcheck.

        Supported strategies:
        - ``git_pull_restart``: git pull then systemctl restart igris
        - ``dry_run``: preflight only, no action

        Returns a full deploy report.
        """
        report: Dict[str, Any] = {
            "strategy": strategy,
            "hostname": hostname or "localhost",
            "dry_run": dry_run,
            "timestamp": time.time(),
        }

        # Preflight
        preflight = self.run_preflight(hostname=hostname, min_disk_pct_free=min_disk_pct_free)
        report["preflight"] = preflight
        if not preflight["ok"]:
            report["deployed"] = False
            report["abort_reason"] = "preflight failed"
            return report

        if dry_run or strategy == "dry_run":
            report["deployed"] = False
            report["note"] = "dry_run: preflight passed, no action taken"
            return report

        # Deploy action
        action_result: Dict[str, Any] = {}
        if strategy == "git_pull_restart":
            try:
                _pull = subprocess.run(
                    ["git", "pull", "--ff-only"],
                    capture_output=True, text=True, timeout=60,
                    cwd=str(self.project_root),
                )
                action_result["git_pull"] = {
                    "ok": _pull.returncode == 0,
                    "output": _pull.stdout.strip()[:500] + _pull.stderr.strip()[:200],
                }
            except Exception as exc:
                action_result["git_pull"] = {"ok": False, "error": str(exc)[:200]}

            if action_result.get("git_pull", {}).get("ok", False):
                try:
                    _restart = subprocess.run(
                        ["systemctl", "restart", "igris"],
                        capture_output=True, text=True, timeout=30,
                    )
                    action_result["restart"] = {
                        "ok": _restart.returncode == 0,
                        "output": _restart.stderr.strip()[:300],
                    }
                except Exception as exc:
                    action_result["restart"] = {"ok": False, "error": str(exc)[:200]}
        else:
            action_result["error"] = f"unknown strategy: {strategy}"

        report["action"] = action_result
        action_ok = all(v.get("ok", False) for v in action_result.values() if isinstance(v, dict))
        report["deployed"] = action_ok

        # Postcheck (only if action succeeded)
        if action_ok:
            # Brief pause to allow service to come up
            time.sleep(2)
            postcheck = self.run_postcheck(hostname=hostname, health_url=health_url)
            report["postcheck"] = postcheck
            report["postcheck_ok"] = postcheck["ok"]
        else:
            report["postcheck"] = None
            report["postcheck_ok"] = False

        return report

    # ------------------------------------------------------------------
    # Smoke test
    # ------------------------------------------------------------------

    def run_smoke_test(self, url: str = "") -> Dict[str, Any]:
        """HTTP smoke test: GET *url* and return evidence.

        Falls back to ``http://localhost:7778/api/ping`` if no URL is given.
        Never raises — all failures are captured in the result dict.
        """
        target = url.strip() or "http://localhost:7778/api/ping"
        start = time.time()
        result: Dict[str, Any] = {
            "url": target,
            "ok": False,
            "status_code": None,
            "response_time_ms": None,
            "body_preview": "",
            "timestamp": start,
        }
        try:
            req = urllib.request.Request(
                target,
                headers={"User-Agent": "IGRIS-DevOps-Smoke/1.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read(4096).decode("utf-8", errors="replace")
                elapsed_ms = int((time.time() - start) * 1000)
                result.update(
                    ok=200 <= resp.status < 400,
                    status_code=resp.status,
                    response_time_ms=elapsed_ms,
                    body_preview=body[:500],
                )
        except urllib.error.HTTPError as exc:
            elapsed_ms = int((time.time() - start) * 1000)
            result.update(
                ok=False,
                status_code=exc.code,
                response_time_ms=elapsed_ms,
                error=str(exc)[:200],
            )
        except Exception as exc:
            elapsed_ms = int((time.time() - start) * 1000)
            result.update(
                ok=False,
                response_time_ms=elapsed_ms,
                error=str(exc)[:200],
            )
        return result
