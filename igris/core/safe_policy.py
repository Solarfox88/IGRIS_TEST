"""Strict safety policy for command execution.

Second-level safety check after command_id allowlist.
Validates context, rate limits, and execution conditions
before allowing a command to run.

Inspired by IGRIS_DECO safe_policy.py but adapted for IGRIS_GPT.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from igris.core.safety import redact_secrets
from igris.layers.execution.safe_commands import ALLOWED_COMMANDS


# ---------------------------------------------------------------------------
# Policy result
# ---------------------------------------------------------------------------

@dataclass
class PolicyDecision:
    """Result of a safety policy check."""
    allowed: bool
    command_id: str
    reason: str
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "command_id": self.command_id,
            "reason": redact_secrets(self.reason),
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

_execution_log: List[Dict[str, Any]] = []
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 20  # max executions per window
BURST_LIMIT = 5  # max same command in window


def _record_execution(command_id: str) -> None:
    """Record a command execution for rate limiting."""
    _execution_log.append({
        "command_id": command_id,
        "timestamp": time.time(),
    })
    # Prune old entries
    cutoff = time.time() - RATE_LIMIT_WINDOW * 2
    _execution_log[:] = [e for e in _execution_log if e["timestamp"] > cutoff]


def _check_rate_limit(command_id: str) -> Optional[str]:
    """Check if command would exceed rate limits. Returns reason if blocked."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    recent = [e for e in _execution_log if e["timestamp"] > window_start]

    if len(recent) >= RATE_LIMIT_MAX:
        return f"Rate limit exceeded: {len(recent)} executions in last {RATE_LIMIT_WINDOW}s (max {RATE_LIMIT_MAX})"

    same_cmd = [e for e in recent if e["command_id"] == command_id]
    if len(same_cmd) >= BURST_LIMIT:
        return f"Burst limit exceeded: '{command_id}' executed {len(same_cmd)} times in last {RATE_LIMIT_WINDOW}s (max {BURST_LIMIT})"

    return None


def reset_rate_limits() -> None:
    """Reset rate limit counters (for testing)."""
    _execution_log.clear()


# ---------------------------------------------------------------------------
# Blocked patterns
# ---------------------------------------------------------------------------

BLOCKED_COMMAND_IDS = frozenset({
    "git_push", "git_force_push", "rm_rf", "delete_all",
    "shell_exec", "eval", "sudo",
})

DESTRUCTIVE_KEYWORDS = frozenset({
    "push", "force", "delete", "remove", "drop", "truncate",
    "reset --hard", "clean -fd",
})


# ---------------------------------------------------------------------------
# Policy check
# ---------------------------------------------------------------------------

def check_command_policy(
    command_id: str,
    context: Optional[Dict[str, Any]] = None,
) -> PolicyDecision:
    """Apply strict safety policy to a command execution request.

    Checks:
    1. command_id is in ALLOWED_COMMANDS allowlist
    2. command_id is not in blocked list
    3. No destructive keywords in command_id
    4. Rate limiting not exceeded
    5. Context validation (if provided)
    """
    checks_passed: List[str] = []
    checks_failed: List[str] = []

    # Check 1: allowlist
    if command_id in ALLOWED_COMMANDS:
        checks_passed.append("allowlist")
    else:
        checks_failed.append("allowlist")
        return PolicyDecision(
            allowed=False,
            command_id=command_id,
            reason=f"Command '{command_id}' not in allowlist",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )

    # Check 2: blocked list
    if command_id in BLOCKED_COMMAND_IDS:
        checks_failed.append("blocked_list")
        return PolicyDecision(
            allowed=False,
            command_id=command_id,
            reason=f"Command '{command_id}' is explicitly blocked",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )
    checks_passed.append("blocked_list")

    # Check 3: destructive keywords
    cmd_lower = command_id.lower()
    for kw in DESTRUCTIVE_KEYWORDS:
        if kw in cmd_lower:
            checks_failed.append("destructive_check")
            return PolicyDecision(
                allowed=False,
                command_id=command_id,
                reason=f"Command contains destructive keyword: '{kw}'",
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )
    checks_passed.append("destructive_check")

    # Check 4: rate limit
    rate_reason = _check_rate_limit(command_id)
    if rate_reason:
        checks_failed.append("rate_limit")
        return PolicyDecision(
            allowed=False,
            command_id=command_id,
            reason=rate_reason,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )
    checks_passed.append("rate_limit")

    # Check 5: context validation
    if context:
        project_root = context.get("project_root", "")
        if project_root and ".." in project_root:
            checks_failed.append("context_validation")
            return PolicyDecision(
                allowed=False,
                command_id=command_id,
                reason="Suspicious project_root with path traversal",
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )
    checks_passed.append("context_validation")

    # All passed — record execution
    _record_execution(command_id)

    return PolicyDecision(
        allowed=True,
        command_id=command_id,
        reason="All safety checks passed",
        checks_passed=checks_passed,
        checks_failed=checks_failed,
    )


def get_policy_status() -> Dict[str, Any]:
    """Return current policy configuration and status."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    recent = [e for e in _execution_log if e["timestamp"] > window_start]

    return {
        "allowed_commands": sorted(ALLOWED_COMMANDS.keys()),
        "blocked_commands": sorted(BLOCKED_COMMAND_IDS),
        "rate_limit_window": RATE_LIMIT_WINDOW,
        "rate_limit_max": RATE_LIMIT_MAX,
        "burst_limit": BURST_LIMIT,
        "recent_executions": len(recent),
        "remaining_capacity": max(0, RATE_LIMIT_MAX - len(recent)),
    }
