"""Risk Classifier for IGRIS_GPT — Epic #42.

Classifies actions by risk level and enforces approval policies.

Risk levels:
    low     — read/status/test
    medium  — write workspace/install local/restart dev
    high    — deploy/nginx/systemd/docker down/push
    critical — delete/db migration/DNS/firewall/secrets/production

Approval modes:
    safe       — only low/medium automatic
    operator   — high automatic only if rollback present
    trusted    — more autonomy on authorized hosts
    manual-critical — critical always requires confirmation
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from igris.core.safety import redact_secrets


# ---------------------------------------------------------------------------
# Risk levels
# ---------------------------------------------------------------------------

RISK_LEVELS = ("low", "medium", "high", "critical")

RISK_DESCRIPTIONS = {
    "low": "Read-only, status checks, test runs — no side effects",
    "medium": "Write workspace files, install local packages, restart dev server",
    "high": "Deploy, nginx/systemd changes, docker down, push to remote",
    "critical": "Delete data, DB migrations, DNS/firewall changes, secrets, production destructive",
}


# ---------------------------------------------------------------------------
# Action risk patterns
# ---------------------------------------------------------------------------

_LOW_PATTERNS = [
    "read", "list", "status", "health", "info", "check", "version",
    "diff", "log", "cat", "head", "tail", "grep", "find", "ls",
    "test", "pytest", "verify", "lint", "type_check",
    "plan", "explain", "describe", "analyze", "inspect",
]

_MEDIUM_PATTERNS = [
    "write", "edit", "create_file", "patch", "install",
    "pip_install", "npm_install", "restart_dev",
    "git_add", "git_commit", "git_branch",
    "mkdir", "cp", "mv", "touch",
]

_HIGH_PATTERNS = [
    "deploy", "push", "git_push", "publish",
    "nginx_reload", "systemd_restart", "docker_down",
    "docker_compose_up", "docker_compose_down",
    "service_restart", "service_stop",
]

_CRITICAL_PATTERNS = [
    "delete", "rm_rf", "drop", "truncate",
    "db_migrate", "db_drop", "db_delete",
    "dns_update", "firewall_change",
    "write_env", "write_secrets", "expose_secret",
    "force_push", "push_main", "push_master",
    "auto_merge", "production_deploy",
    "destroy", "provision",
]


def classify_action_risk(action_id: str, action_description: str = "") -> str:
    """Classify an action's risk level based on its ID and description."""
    combined = f"{action_id} {action_description}".lower()

    for pattern in _CRITICAL_PATTERNS:
        if pattern in combined:
            return "critical"

    for pattern in _HIGH_PATTERNS:
        if pattern in combined:
            return "high"

    for pattern in _MEDIUM_PATTERNS:
        if pattern in combined:
            return "medium"

    return "low"


# ---------------------------------------------------------------------------
# Approval modes
# ---------------------------------------------------------------------------

APPROVAL_MODES = ("safe", "operator", "trusted", "manual-critical")

APPROVAL_MODE_DESCRIPTIONS = {
    "safe": "Only low and medium actions run automatically",
    "operator": "High actions allowed if rollback is present",
    "trusted": "More autonomy on authorized hosts",
    "manual-critical": "Critical actions always require human confirmation",
}


@dataclass
class ApprovalDecision:
    """Result of an approval check."""
    allowed: bool
    risk_level: str
    approval_mode: str
    requires_rollback: bool = False
    requires_confirmation: bool = False
    reason: str = ""
    action_id: str = ""
    trace_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "risk_level": self.risk_level,
            "approval_mode": self.approval_mode,
            "requires_rollback": self.requires_rollback,
            "requires_confirmation": self.requires_confirmation,
            "reason": self.reason,
            "action_id": self.action_id,
            "trace_id": self.trace_id,
        }


def check_approval(
    action_id: str,
    risk_level: str,
    approval_mode: str = "safe",
    has_rollback: bool = False,
    host: str = "",
    authorized_hosts: Optional[List[str]] = None,
    approval_token: Optional[str] = None,
    trace_id: str = "",
) -> ApprovalDecision:
    """Check whether an action is approved under the given policy.

    Returns an ApprovalDecision indicating whether the action can proceed.
    """
    if risk_level not in RISK_LEVELS:
        risk_level = "critical"  # Unknown → safest default
    if approval_mode not in APPROVAL_MODES:
        approval_mode = "manual-critical"

    # Low is always allowed
    if risk_level == "low":
        return ApprovalDecision(
            allowed=True, risk_level=risk_level, approval_mode=approval_mode,
            reason="Low risk — always allowed", action_id=action_id, trace_id=trace_id,
        )

    # Medium: allowed in all modes
    if risk_level == "medium":
        return ApprovalDecision(
            allowed=True, risk_level=risk_level, approval_mode=approval_mode,
            reason="Medium risk — allowed in current mode",
            action_id=action_id, trace_id=trace_id,
        )

    # High
    if risk_level == "high":
        if approval_mode == "safe":
            return ApprovalDecision(
                allowed=False, risk_level=risk_level, approval_mode=approval_mode,
                requires_confirmation=True,
                reason="High risk action rejected in safe mode",
                action_id=action_id, trace_id=trace_id,
            )
        if approval_mode == "operator":
            if has_rollback:
                return ApprovalDecision(
                    allowed=True, risk_level=risk_level, approval_mode=approval_mode,
                    requires_rollback=True,
                    reason="High risk allowed — rollback present",
                    action_id=action_id, trace_id=trace_id,
                )
            return ApprovalDecision(
                allowed=False, risk_level=risk_level, approval_mode=approval_mode,
                requires_rollback=True, requires_confirmation=True,
                reason="High risk rejected — no rollback in operator mode",
                action_id=action_id, trace_id=trace_id,
            )
        if approval_mode == "trusted":
            if authorized_hosts and host in authorized_hosts:
                return ApprovalDecision(
                    allowed=True, risk_level=risk_level, approval_mode=approval_mode,
                    reason=f"High risk allowed — {host} is authorized",
                    action_id=action_id, trace_id=trace_id,
                )
            return ApprovalDecision(
                allowed=False, risk_level=risk_level, approval_mode=approval_mode,
                requires_confirmation=True,
                reason=f"High risk rejected — {host} not in authorized hosts",
                action_id=action_id, trace_id=trace_id,
            )
        # manual-critical: high still needs confirmation
        return ApprovalDecision(
            allowed=False, risk_level=risk_level, approval_mode=approval_mode,
            requires_confirmation=True,
            reason="High risk — requires confirmation in manual-critical mode",
            action_id=action_id, trace_id=trace_id,
        )

    # Critical: always requires confirmation unless token provided
    if risk_level == "critical":
        if approval_token:
            return ApprovalDecision(
                allowed=True, risk_level=risk_level, approval_mode=approval_mode,
                reason="Critical action approved with explicit token",
                action_id=action_id, trace_id=trace_id,
            )
        return ApprovalDecision(
            allowed=False, risk_level=risk_level, approval_mode=approval_mode,
            requires_confirmation=True,
            reason="Critical action — requires explicit approval token",
            action_id=action_id, trace_id=trace_id,
        )

    return ApprovalDecision(
        allowed=False, risk_level=risk_level, approval_mode=approval_mode,
        reason="Unknown risk level", action_id=action_id, trace_id=trace_id,
    )


# ---------------------------------------------------------------------------
# Secret guard
# ---------------------------------------------------------------------------

_SECRET_FILE_PATTERNS = [
    re.compile(r"\.env($|\.)"),
    re.compile(r"\.secret"),
    re.compile(r"credentials"),
    re.compile(r"service_account"),
    re.compile(r"id_rsa"),
    re.compile(r"id_ed25519"),
    re.compile(r"\.pem$"),
    re.compile(r"\.key$"),
]


def is_secret_file(path: str) -> bool:
    """Check if a file path refers to a secrets file."""
    name = path.lower().rsplit("/", 1)[-1] if "/" in path else path.lower()
    for pat in _SECRET_FILE_PATTERNS:
        if pat.search(name):
            return True
    return False


def guard_secret_access(path: str, action: str = "read") -> ApprovalDecision:
    """Block access to secret files."""
    if is_secret_file(path):
        return ApprovalDecision(
            allowed=False,
            risk_level="critical",
            approval_mode="manual-critical",
            reason=f"Access to secret file blocked: {redact_secrets(path)}",
            action_id=f"{action}:{path}",
        )
    return ApprovalDecision(
        allowed=True,
        risk_level="low",
        approval_mode="safe",
        reason="File is not a secret",
        action_id=f"{action}:{path}",
    )
