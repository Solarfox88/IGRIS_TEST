"""Command Risk Engine v2 — Epic #63.

Governs shell access with multi-level risk classification:
    1. Structured Tool preferred
    2. Template parametrized as second option
    3. Raw shell proposal only as escape hatch, gated

Pipeline for raw shell proposals:
    parse_command → deterministic_classify → contextual_policy →
    llm_risk_review (via Model Orchestrator) → decision

Risk classes: LOW, MEDIUM, HIGH, CRITICAL, UNKNOWN

The LLM Risk Reviewer is advisory only — final decision is always
made by IGRIS Policy Engine, never by the LLM.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from igris.core.safety import redact_secrets


# ---------------------------------------------------------------------------
# Risk levels
# ---------------------------------------------------------------------------

RISK_LEVELS = ("low", "medium", "high", "critical", "unknown")


# ---------------------------------------------------------------------------
# Shell parser — recognize command structure and dangerous patterns
# ---------------------------------------------------------------------------

# Dangerous command patterns (deterministic blocklist)
_SUDO_RE = re.compile(r"\b(sudo|su)\b")
_RM_RE = re.compile(r"\brm\b.*(-r|-f|-rf|--recursive|--force)")
_DELETE_RE = re.compile(r"\b(unlink|rmdir|shred)\b")
_CHMOD_RE = re.compile(r"\b(chmod|chown)\b")
_SYSTEMCTL_RE = re.compile(r"\b(systemctl|service|journalctl)\b")
_DOCKER_RE = re.compile(r"\b(docker|docker-compose|docker compose)\b")
_NGINX_RE = re.compile(r"\b(nginx|apache2?|httpd|certbot)\b")
_PKG_RE = re.compile(r"\b(apt|apt-get|dpkg|pip|pip3|npm|pnpm|yarn|cargo)\b")
_GIT_DANGER_RE = re.compile(r"\bgit\b.*\b(push|reset|clean|force)\b")
_FORCE_PUSH_RE = re.compile(r"\bgit\b.*\bpush\b.*(-f|--force|--force-with-lease)")
_CURL_PIPE_RE = re.compile(r"\b(curl|wget)\b.*\|\s*(bash|sh|zsh|python|perl)")
_PIPE_RE = re.compile(r"\|")
_REDIRECT_RE = re.compile(r"[>]{1,2}")
_SUBSHELL_RE = re.compile(r"[\$]\(|`")
_CHAIN_RE = re.compile(r"&&|\|\|")
_ABS_PATH_RE = re.compile(r"(?:^|\s)/(?:etc|usr|var|root|boot|proc|sys|dev)/")
_WILDCARD_RE = re.compile(r"\*")
_NETWORK_RE = re.compile(r"\b(curl|wget|nc|ncat|netcat|ssh|scp|rsync|telnet|ftp)\b")
_DB_RE = re.compile(r"\b(mysql|psql|mongo|redis-cli|sqlite3)\b.*\b(DROP|DELETE|TRUNCATE|ALTER|MIGRATE)\b", re.IGNORECASE)
_DB_CMD_RE = re.compile(r"\b(mysql|psql|mongo|redis-cli|sqlite3)\b")
_FIREWALL_RE = re.compile(r"\b(iptables|ufw|firewalld|firewall-cmd|nftables)\b")
_DNS_RE = re.compile(r"\b(dig|nslookup|host|resolvectl)\b.*\b(update|set|add|delete)\b", re.IGNORECASE)
_ENV_RE = re.compile(r"(\.env|\.secrets|\.pem|\.key|id_rsa|credentials|token|password|api[._]key)", re.IGNORECASE)
_SECRET_ACCESS_RE = re.compile(r"\b(cat|less|more|head|tail|grep|awk|sed)\b.*\.(env|secret|pem|key)", re.IGNORECASE)


@dataclass
class ParsedCommand:
    """Parsed representation of a shell command."""
    raw: str = ""
    executable: str = ""
    args: List[str] = field(default_factory=list)
    has_sudo: bool = False
    has_rm: bool = False
    has_delete: bool = False
    has_chmod: bool = False
    has_systemctl: bool = False
    has_docker: bool = False
    has_nginx: bool = False
    has_package_manager: bool = False
    has_git_danger: bool = False
    has_force_push: bool = False
    has_curl_pipe: bool = False
    has_pipe: bool = False
    has_redirect: bool = False
    has_subshell: bool = False
    has_chain: bool = False
    has_abs_path: bool = False
    has_wildcard: bool = False
    has_network: bool = False
    has_db: bool = False
    has_db_destructive: bool = False
    has_firewall: bool = False
    has_dns_modify: bool = False
    has_env_access: bool = False
    has_secret_access: bool = False
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw": redact_secrets(self.raw),
            "executable": self.executable,
            "has_sudo": self.has_sudo,
            "has_rm": self.has_rm,
            "has_chmod": self.has_chmod,
            "has_systemctl": self.has_systemctl,
            "has_docker": self.has_docker,
            "has_nginx": self.has_nginx,
            "has_package_manager": self.has_package_manager,
            "has_git_danger": self.has_git_danger,
            "has_force_push": self.has_force_push,
            "has_curl_pipe": self.has_curl_pipe,
            "has_pipe": self.has_pipe,
            "has_redirect": self.has_redirect,
            "has_subshell": self.has_subshell,
            "has_chain": self.has_chain,
            "has_abs_path": self.has_abs_path,
            "has_wildcard": self.has_wildcard,
            "has_network": self.has_network,
            "has_db": self.has_db,
            "has_db_destructive": self.has_db_destructive,
            "has_firewall": self.has_firewall,
            "has_dns_modify": self.has_dns_modify,
            "has_env_access": self.has_env_access,
            "has_secret_access": self.has_secret_access,
            "flags": self.flags,
        }


def parse_command(raw: str) -> ParsedCommand:
    """Parse a shell command string into structured representation.

    Epic #1072 fix: uses shlex.split() instead of .split() so quoted arguments
    (e.g. 'git commit -m "fix: my message"') are parsed correctly.  Falls back
    to simple whitespace split on shlex.ValueError (e.g. unterminated quotes).
    """
    import shlex as _shlex
    cmd = ParsedCommand(raw=raw)
    if not raw or not raw.strip():
        return cmd

    try:
        parts = _shlex.split(raw)
    except ValueError:
        # Unterminated quote or other shell syntax error — use naive split
        parts = raw.strip().split()
    cmd.executable = parts[0] if parts else ""
    cmd.args = parts[1:] if len(parts) > 1 else []

    cmd.has_sudo = bool(_SUDO_RE.search(raw))
    cmd.has_rm = bool(_RM_RE.search(raw))
    cmd.has_delete = bool(_DELETE_RE.search(raw))
    cmd.has_chmod = bool(_CHMOD_RE.search(raw))
    cmd.has_systemctl = bool(_SYSTEMCTL_RE.search(raw))
    cmd.has_docker = bool(_DOCKER_RE.search(raw))
    cmd.has_nginx = bool(_NGINX_RE.search(raw))
    cmd.has_package_manager = bool(_PKG_RE.search(raw))
    cmd.has_git_danger = bool(_GIT_DANGER_RE.search(raw))
    cmd.has_force_push = bool(_FORCE_PUSH_RE.search(raw))
    cmd.has_curl_pipe = bool(_CURL_PIPE_RE.search(raw))
    cmd.has_pipe = bool(_PIPE_RE.search(raw))
    cmd.has_redirect = bool(_REDIRECT_RE.search(raw))
    cmd.has_subshell = bool(_SUBSHELL_RE.search(raw))
    cmd.has_chain = bool(_CHAIN_RE.search(raw))
    cmd.has_abs_path = bool(_ABS_PATH_RE.search(raw))
    cmd.has_wildcard = bool(_WILDCARD_RE.search(raw))
    cmd.has_network = bool(_NETWORK_RE.search(raw))
    cmd.has_db = bool(_DB_CMD_RE.search(raw))
    cmd.has_db_destructive = bool(_DB_RE.search(raw))
    cmd.has_firewall = bool(_FIREWALL_RE.search(raw))
    cmd.has_dns_modify = bool(_DNS_RE.search(raw))
    cmd.has_env_access = bool(_ENV_RE.search(raw))
    cmd.has_secret_access = bool(_SECRET_ACCESS_RE.search(raw))

    # Collect flags
    for p in cmd.flags_list():
        cmd.flags.append(p)

    return cmd


def _flags_list(self) -> List[str]:
    """List all detected flags."""
    flags = []
    for attr in dir(self):
        if attr.startswith("has_") and getattr(self, attr, False):
            flags.append(attr.replace("has_", ""))
    return flags


ParsedCommand.flags_list = _flags_list


# ---------------------------------------------------------------------------
# Deterministic risk classifier
# ---------------------------------------------------------------------------

def classify_command_risk(parsed: ParsedCommand) -> str:
    """Classify risk level deterministically from parsed command.

    Returns: low | medium | high | critical | unknown
    """
    # CRITICAL — always blocked or requires explicit confirmation
    if parsed.has_force_push:
        return "critical"
    if parsed.has_curl_pipe:
        return "critical"
    if parsed.has_rm and parsed.has_sudo:
        return "critical"
    if parsed.has_rm and parsed.has_wildcard:
        return "critical"
    if parsed.has_db_destructive:
        return "critical"
    if parsed.has_firewall:
        return "critical"
    if parsed.has_dns_modify:
        return "critical"
    if parsed.has_secret_access:
        return "critical"

    # HIGH — requires rollback/policy
    if parsed.has_sudo:
        return "high"
    if parsed.has_rm:
        return "high"
    if parsed.has_delete:
        return "high"
    if parsed.has_systemctl:
        return "high"
    if parsed.has_docker:
        return "high"
    if parsed.has_nginx:
        return "high"
    if parsed.has_git_danger:
        return "high"
    if parsed.has_abs_path:
        return "high"
    if parsed.has_env_access:
        return "high"

    # MEDIUM — review recommended
    if parsed.has_package_manager:
        return "medium"
    if parsed.has_network:
        return "medium"
    if parsed.has_redirect:
        return "medium"
    if parsed.has_subshell:
        return "medium"
    if parsed.has_chmod:
        return "medium"
    if parsed.has_db:
        return "medium"
    if parsed.has_chain and parsed.has_pipe:
        return "medium"

    # LOW — safe read-only operations
    safe_executables = {
        "ls", "cat", "head", "tail", "wc", "echo", "pwd", "whoami",
        "date", "uname", "hostname", "env", "printenv",
        "grep", "rg", "find", "which", "type", "file",
        "git", "python", "python3", "node", "ruby",
        "pytest", "jest", "mocha", "cargo",
    }
    if parsed.executable in safe_executables and not parsed.has_pipe and not parsed.has_redirect:
        return "low"

    # UNKNOWN — needs LLM review
    return "unknown"


# ---------------------------------------------------------------------------
# LLM Risk Reviewer output
# ---------------------------------------------------------------------------

@dataclass
class RiskReviewResult:
    """Output of the LLM Risk Reviewer."""
    risk_assessment: str = "unknown"  # low | medium | high | critical | unknown
    reasons: List[str] = field(default_factory=list)
    affected_paths: List[str] = field(default_factory=list)
    affected_services: List[str] = field(default_factory=list)
    requires_rollback: bool = False
    recommended_prechecks: List[str] = field(default_factory=list)
    recommended_postchecks: List[str] = field(default_factory=list)
    safer_alternative: Optional[str] = None
    should_execute: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_assessment": self.risk_assessment,
            "reasons": self.reasons,
            "affected_paths": self.affected_paths,
            "affected_services": self.affected_services,
            "requires_rollback": self.requires_rollback,
            "recommended_prechecks": self.recommended_prechecks,
            "recommended_postchecks": self.recommended_postchecks,
            "safer_alternative": self.safer_alternative,
            "should_execute": self.should_execute,
        }


# ---------------------------------------------------------------------------
# Safety Event Log
# ---------------------------------------------------------------------------

@dataclass
class SafetyEvent:
    """Record of a risk engine evaluation."""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    command: str = ""
    parsed_flags: List[str] = field(default_factory=list)
    deterministic_risk: str = "unknown"
    llm_risk: str = ""
    final_risk: str = "unknown"
    decision: str = "blocked"  # allowed | blocked | needs_approval
    reason: str = ""
    review_result: Optional[Dict[str, Any]] = None
    mission_id: str = ""
    trace_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "command": redact_secrets(self.command),
            "parsed_flags": self.parsed_flags,
            "deterministic_risk": self.deterministic_risk,
            "llm_risk": self.llm_risk,
            "final_risk": self.final_risk,
            "decision": self.decision,
            "reason": redact_secrets(self.reason),
            "review_result": self.review_result,
            "mission_id": self.mission_id,
            "trace_id": self.trace_id,
        }


# ---------------------------------------------------------------------------
# Command Risk Engine
# ---------------------------------------------------------------------------

class CommandRiskEngine:
    """Multi-level risk classification and governance for shell commands.

    Policy hierarchy:
        1. Structured Tool → always prefer
        2. Template parametrized → safer than raw
        3. Raw shell proposal → escape hatch, fully gated

    For raw shell proposals:
        parse → deterministic classify → LLM review (MEDIUM+) → policy decision

    Epic #1072 improvements:
        - Contextual policy: higher risk threshold in production environments
        - Destructive pre-check: explicit check before any destructive command
        - Dry-run mode: evaluate without executing; return would-execute result
    """

    #: Known destructive command patterns
    DESTRUCTIVE_PATTERNS = re.compile(
        r"\b(rm\b.*(-r|-f|-rf)|DROP\s+TABLE|TRUNCATE\s+TABLE|git\s+clean|"
        r"git\s+reset\s+--hard|mkfs|dd\s+if=|shred|wipefs)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        project_root: Optional[str] = None,
        use_llm_reviewer: bool = True,
        environment: str = "dev",  # Epic #1072: "dev" | "staging" | "production"
        dry_run: bool = False,      # Epic #1072: if True, never executes
    ):
        import os
        self.project_root = project_root or os.environ.get("PROJECT_ROOT", ".")
        self.use_llm_reviewer = use_llm_reviewer
        self.environment = environment
        self.dry_run = dry_run
        self._event_log: List[SafetyEvent] = []
        # Epic #1072 — Precheck/postcheck hook registries
        # Each hook is callable(command: str) -> Optional[str]
        # Returning a non-None string means "block with this reason"
        self._prechecks: List[Any] = []
        self._postchecks: List[Any] = []

    def register_precheck(self, fn: Any) -> None:
        """Register a precheck hook called before command evaluation.

        Epic #1072 — Precheck hooks run before the deterministic classifier.
        A hook returns a string (block reason) or None (allow to proceed).
        Hooks are called in registration order; first block wins.

        Example:
            engine.register_precheck(lambda cmd: "blocked" if "sudo" in cmd else None)
        """
        self._prechecks.append(fn)

    def register_postcheck(self, fn: Any) -> None:
        """Register a postcheck hook called after evaluation, before returning.

        Epic #1072 — Postcheck hooks can veto an 'allowed' decision. They
        receive the command and the SafetyEvent produced so far and return
        a block reason string or None.

        Example:
            engine.register_postcheck(lambda cmd, evt: "no prod writes" if evt.final_risk == "high" else None)
        """
        self._postchecks.append(fn)

    def _run_prechecks(self, command: str, event: "SafetyEvent") -> Optional[str]:
        """Run all registered precheck hooks. Return first block reason or None."""
        for hook in self._prechecks:
            try:
                reason = hook(command)
                if reason is not None:
                    return str(reason)
            except Exception as exc:
                import logging as _logging
                _logging.getLogger("igris.risk.precheck").warning(
                    "precheck hook %r raised: %s", hook, exc
                )
        return None

    def _run_postchecks(self, command: str, event: "SafetyEvent") -> Optional[str]:
        """Run all registered postcheck hooks. Return first block reason or None."""
        for hook in self._postchecks:
            try:
                reason = hook(command, event)
                if reason is not None:
                    return str(reason)
            except Exception as exc:
                import logging as _logging
                _logging.getLogger("igris.risk.postcheck").warning(
                    "postcheck hook %r raised: %s", hook, exc
                )
        return None

    def get_rollback_suggestion(self, command: str, event: "SafetyEvent") -> str:
        """Return a human-readable rollback suggestion for a CRITICAL/HIGH command.

        Epic #1072 — Binding rollback hints let operators know how to undo
        a command if it runs and causes problems.
        """
        cmd_lower = command.lower()
        if "rm " in cmd_lower or "rmdir" in cmd_lower:
            return "Restore from the last git commit or backup: `git checkout -- .`"
        if "drop table" in cmd_lower or "truncate table" in cmd_lower:
            return "Restore from the last database snapshot or `pg_dump` backup."
        if "git reset --hard" in cmd_lower:
            return "Use `git reflog` to find the pre-reset SHA and `git reset --hard <SHA>`."
        if "git clean" in cmd_lower:
            return "Untracked files cannot be recovered after `git clean -f`. Restore from backup."
        if "dd if=" in cmd_lower or "mkfs" in cmd_lower:
            return "Disk write is irreversible. Restore from full disk backup or snapshot."
        if event.final_risk in ("critical", "high"):
            return (
                f"Command classified as {event.final_risk}. Take a snapshot or backup before running. "
                "To rollback: restore from the most recent backup of affected resources."
            )
        return ""

    def is_destructive(self, command: str) -> bool:
        """Epic #1072 — Pre-check: return True if command matches destructive patterns.

        This is a fast, deterministic check run before full evaluation.
        Destructive commands always get at least 'high' risk in production.
        """
        return bool(self.DESTRUCTIVE_PATTERNS.search(command))

    def evaluate_command(
        self,
        command: str,
        context: str = "",
        mission_id: str = "",
        trace_id: str = "",
        cwd: Optional[str] = None,
    ) -> Tuple[SafetyEvent, RiskReviewResult]:
        """Evaluate a raw shell command through the full risk pipeline.

        Returns (SafetyEvent, RiskReviewResult).

        Epic #1072:
        - If dry_run=True, returns a would-execute event with decision="dry_run"
        - If environment="production" and command is destructive → escalate to critical
        - Destructive pre-check always fires before LLM review
        """
        event = SafetyEvent(
            command=command,
            mission_id=mission_id,
            trace_id=trace_id,
        )

        # Epic #1072 — run precheck hooks before any evaluation
        precheck_block = self._run_prechecks(command, event)
        if precheck_block:
            event.decision = "blocked"
            event.reason = f"precheck: {precheck_block}"
            event.final_risk = "high"
            event.deterministic_risk = "high"
            self._event_log.append(event)
            return event, RiskReviewResult()

        # Epic #1072 — dry-run mode: classify but never gate execution
        if self.dry_run:
            parsed = parse_command(command)
            det_risk = classify_command_risk(parsed)
            event.parsed_flags = parsed.flags_list()
            event.deterministic_risk = det_risk
            event.final_risk = det_risk
            event.decision = "dry_run"
            event.reason = f"dry_run mode: would classify as {det_risk}"
            self._event_log.append(event)
            return event, RiskReviewResult()

        # Epic #1072 — Contextual policy: cwd-based escalation.
        # Commands run outside the project root (e.g. /etc, /var, /home/other)
        # are escalated to at least 'medium' to flag unexpected scope.
        # Commands run in system directories are escalated to 'high'.
        if cwd:
            import os as _os
            _cwd_resolved = _os.path.realpath(str(cwd))
            _proj_resolved = _os.path.realpath(str(self.project_root))
            _in_project = _cwd_resolved.startswith(_proj_resolved)
            _system_dirs = ("/etc", "/usr", "/var", "/bin", "/sbin", "/lib", "/boot", "/sys", "/proc")
            _in_system = any(_cwd_resolved.startswith(d) for d in _system_dirs)
            if _in_system:
                event.reason = (
                    f"Command cwd is a system directory ({_cwd_resolved!r}); escalating risk."
                )
                event.decision = "blocked"
                event.deterministic_risk = "high"
                event.final_risk = "high"
                self._event_log.append(event)
                return event, RiskReviewResult()
            if not _in_project:
                # Outside project root — log warning but don't block by default
                context = context + f" [cwd={_cwd_resolved!r} is outside project root]"

        # 1. Parse command
        parsed = parse_command(command)
        event.parsed_flags = parsed.flags_list()

        # 2. Deterministic classification
        det_risk = classify_command_risk(parsed)
        event.deterministic_risk = det_risk

        # Epic #1072 — Destructive pre-check: escalate in production
        if self.is_destructive(command):
            if self.environment == "production":
                det_risk = "critical"
                event.deterministic_risk = "critical"
                event.reason = (
                    f"Destructive command blocked in production environment: {command[:100]}"
                )
                event.decision = "blocked"
                event.final_risk = "critical"
                self._event_log.append(event)
                return event, RiskReviewResult()
            elif self.environment == "staging" and det_risk not in ("high", "critical"):
                # Escalate destructive commands in staging to at least high
                det_risk = "high"
                event.deterministic_risk = "high"

        # 3. LLM review for MEDIUM, HIGH, UNKNOWN
        review = RiskReviewResult()
        if det_risk in ("medium", "high", "unknown") and self.use_llm_reviewer:
            review = self._llm_review(command, parsed, context, det_risk)
            event.llm_risk = review.risk_assessment

        # 4. Final risk = max(deterministic, llm) for safety
        event.final_risk = self._resolve_final_risk(det_risk, event.llm_risk)

        # Epic #1072 — Contextual policy: production blocks HIGH (not just CRITICAL)
        if self.environment == "production" and event.final_risk in ("high", "critical"):
            event.decision = "blocked"
            event.reason = (
                f"Blocked in production environment (risk={event.final_risk}): "
                + (", ".join(review.reasons) or "policy")
            )
            self._event_log.append(event)
            return event, review

        # 5. Standard policy decision
        event.decision, event.reason = self._apply_policy(
            event.final_risk, parsed, review,
        )
        event.review_result = review.to_dict()

        # Epic #1072 — run postcheck hooks (can veto 'allowed' or 'needs_approval')
        if event.decision in ("allowed", "needs_approval"):
            postcheck_block = self._run_postchecks(command, event)
            if postcheck_block:
                event.decision = "blocked"
                event.reason = f"postcheck: {postcheck_block}"

        # 6. Log event
        self._event_log.append(event)

        return event, review

    def evaluate_template(
        self,
        template_id: str,
        parameters: Dict[str, str],
        mission_id: str = "",
        trace_id: str = "",
    ) -> Tuple[SafetyEvent, RiskReviewResult]:
        """Evaluate a parametrized shell template.

        Templates are safer than raw commands — validated parameters only.
        """
        rendered = self._render_template(template_id, parameters)
        event, review = self.evaluate_command(
            command=rendered,
            context=f"Template: {template_id}",
            mission_id=mission_id,
            trace_id=trace_id,
        )
        # Templates get a risk reduction (one level down from raw)
        if event.final_risk == "high":
            event.final_risk = "medium"
            event.reason = f"Template risk reduced: {event.reason}"
        elif event.final_risk == "medium":
            event.final_risk = "low"
            event.reason = f"Template risk reduced: {event.reason}"
        # Re-evaluate policy with reduced risk
        event.decision, policy_reason = self._apply_policy(
            event.final_risk, parse_command(rendered), review,
        )
        if "Template" not in event.reason:
            event.reason = f"Template risk reduced: {policy_reason}"
        self._event_log.append(event)
        return event, review

    def get_event_log(self) -> List[Dict[str, Any]]:
        """Get safety event log."""
        return [e.to_dict() for e in self._event_log]

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent safety events."""
        return [e.to_dict() for e in self._event_log[-limit:]]

    # -- Internal --

    def _llm_review(
        self,
        command: str,
        parsed: ParsedCommand,
        context: str,
        det_risk: str,
    ) -> RiskReviewResult:
        """Request LLM risk review via Model Orchestrator.

        The LLM reviewer is advisory only. IGRIS Policy Engine makes
        the final decision.
        """
        try:
            from igris.core.model_orchestrator import ModelOrchestrator
            orch = ModelOrchestrator()

            prompt = (
                f"You are a security reviewer. Evaluate this shell command:\n"
                f"Command: {redact_secrets(command)}\n"
                f"Deterministic risk: {det_risk}\n"
                f"Detected flags: {', '.join(parsed.flags_list())}\n"
                f"Context: {context}\n\n"
                f"Respond with JSON:\n"
                f'{{"risk_assessment": "medium|high|critical|unknown", '
                f'"reasons": [], "affected_paths": [], "affected_services": [], '
                f'"requires_rollback": true/false, '
                f'"recommended_prechecks": [], "recommended_postchecks": [], '
                f'"safer_alternative": null, "should_execute": false}}'
            )

            result = orch.complete(
                task_type="risk_review",
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a security-focused command risk reviewer.",
                json_mode=True,
                preferred_profile="risk_reviewer",
            )

            if result.success and result.text:
                return self._parse_review_response(result.text)

        except Exception:
            pass

        # Fallback: conservative review
        return RiskReviewResult(
            risk_assessment=det_risk,
            reasons=[f"LLM review unavailable, using deterministic: {det_risk}"],
            requires_rollback=det_risk in ("high", "critical"),
            should_execute=det_risk == "low",
        )

    def _parse_review_response(self, text: str) -> RiskReviewResult:
        """Parse LLM risk review JSON response."""
        import json
        try:
            data = json.loads(text)
            return RiskReviewResult(
                risk_assessment=data.get("risk_assessment", "unknown"),
                reasons=data.get("reasons", []),
                affected_paths=data.get("affected_paths", []),
                affected_services=data.get("affected_services", []),
                requires_rollback=data.get("requires_rollback", False),
                recommended_prechecks=data.get("recommended_prechecks", []),
                recommended_postchecks=data.get("recommended_postchecks", []),
                safer_alternative=data.get("safer_alternative"),
                should_execute=data.get("should_execute", False),
            )
        except (json.JSONDecodeError, TypeError):
            return RiskReviewResult(
                risk_assessment="unknown",
                reasons=["Failed to parse LLM review response"],
                should_execute=False,
            )

    @staticmethod
    def _resolve_final_risk(deterministic: str, llm: str) -> str:
        """Resolve final risk level — always take the higher one.

        If LLM review was not performed (llm is empty), use deterministic only.
        """
        if not llm:
            return deterministic
        order = {"low": 0, "medium": 1, "high": 2, "critical": 3, "unknown": 2}
        d_score = order.get(deterministic, 2)
        l_score = order.get(llm, -1)
        if l_score > d_score:
            return llm
        return deterministic

    @staticmethod
    def _apply_policy(
        risk: str,
        parsed: ParsedCommand,
        review: RiskReviewResult,
    ) -> Tuple[str, str]:
        """Apply policy engine to determine final decision.

        Returns (decision, reason).
        Decision: allowed | blocked | needs_approval
        """
        # CRITICAL — always blocked
        if risk == "critical":
            return "blocked", f"Critical risk: {', '.join(review.reasons) or 'deterministic block'}"

        # HIGH — needs approval + rollback
        if risk == "high":
            if review.requires_rollback:
                return "needs_approval", f"High risk, requires rollback: {', '.join(review.reasons) or 'high risk'}"
            return "needs_approval", f"High risk: {', '.join(review.reasons) or 'high risk command'}"

        # MEDIUM — allowed with logging
        if risk == "medium":
            return "allowed", f"Medium risk, logged: {', '.join(review.reasons) or 'standard medium'}"

        # LOW — always allowed
        if risk == "low":
            return "allowed", "Low risk: safe command"

        # UNKNOWN — needs approval
        return "needs_approval", f"Unknown risk: {', '.join(review.reasons) or 'unrecognized command'}"

    @staticmethod
    def _render_template(template_id: str, parameters: Dict[str, str]) -> str:
        """Render a shell template with safe parameters."""
        templates = {
            "pip_install": "pip install {package}",
            "npm_install": "npm install {package}",
            "pytest_run": "python -m pytest {path} -v",
            "git_status": "git status",
            "git_diff": "git diff {path}",
            "cat_file": "cat {path}",
            "ls_dir": "ls -la {path}",
            "docker_ps": "docker ps",
            "systemctl_status": "systemctl status {service}",
        }
        template = templates.get(template_id, template_id)
        try:
            return template.format(**parameters)
        except (KeyError, IndexError):
            return template
