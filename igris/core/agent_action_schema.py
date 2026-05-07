"""Agent Action Schema for IGRIS_GPT — Epic #58.

Defines the structured contract between LLM reasoning and IGRIS execution.
Every LLM proposal must conform to this schema before it can be validated,
risk-classified, and executed by the Tool Runtime.

Action types:
    search_code       — search for patterns in codebase
    find_files        — find files by name/pattern
    list_directory    — list directory contents
    read_file_range   — read specific lines from a file
    write_file        — write/create a file
    propose_patch     — propose a code patch (diff-style)
    apply_patch       — apply a previously validated patch
    run_tests         — execute test suite
    git_status        — check git status
    git_diff          — view git diff
    shell_template    — run a pre-approved command template
    raw_shell_proposal — propose an arbitrary shell command (gated)
    http_check        — HTTP health/status check
    update_plan       — update the mission plan
    record_memory     — record a decision/lesson in memory
    ask_user          — request human input
    finish            — declare mission/task complete
    blocked           — declare inability to proceed

Agent roles (from Agent Registry):
    coordinator, planner, researcher, coder, tester, reviewer,
    devops, security_guard, memory_manager, cost_guardian, reporter
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from igris.core.safety import redact_secrets


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_TYPES = (
    "search_code",
    "find_files",
    "list_directory",
    "read_file_range",
    "write_file",
    "insert_after",
    "insert_before",
    "replace_range",
    "append_file",
    "propose_patch",
    "apply_patch",
    "run_tests",
    "git_status",
    "git_diff",
    "shell_template",
    "raw_shell_proposal",
    "http_check",
    "update_plan",
    "record_memory",
    "ask_user",
    "finish",
    "blocked",
)

AGENT_ROLES = (
    "coordinator",
    "planner",
    "researcher",
    "coder",
    "tester",
    "reviewer",
    "devops",
    "security_guard",
    "memory_manager",
    "cost_guardian",
    "reporter",
)

RISK_HINTS = ("low", "medium", "high", "critical", "unknown")

# Actions that are always read-only / no side effects
READ_ONLY_ACTIONS = frozenset({
    "search_code", "find_files", "list_directory", "read_file_range",
    "git_status", "git_diff", "http_check", "update_plan",
    "record_memory", "ask_user", "finish", "blocked",
})

# Actions that modify filesystem or state
WRITE_ACTIONS = frozenset({
    "write_file", "insert_after", "insert_before", "replace_range", "append_file",
    "propose_patch", "apply_patch", "run_tests",
    "shell_template", "raw_shell_proposal",
})

# Actions requiring Command Risk Engine review
RISK_GATED_ACTIONS = frozenset({
    "raw_shell_proposal", "shell_template", "apply_patch", "write_file",
    "insert_after", "insert_before", "replace_range", "append_file",
})

# Role → allowed action types mapping
ROLE_ALLOWED_ACTIONS: Dict[str, frozenset] = {
    "coordinator": frozenset(ACTION_TYPES),
    "planner": frozenset({
        "search_code", "find_files", "list_directory", "read_file_range",
        "git_status", "git_diff", "update_plan", "record_memory",
        "ask_user", "finish", "blocked",
    }),
    "researcher": frozenset({
        "search_code", "find_files", "list_directory", "read_file_range",
        "git_status", "git_diff", "http_check", "record_memory",
        "ask_user", "finish", "blocked",
    }),
    "coder": frozenset({
        "search_code", "find_files", "list_directory", "read_file_range",
        "write_file", "insert_after", "insert_before", "replace_range", "append_file",
        "propose_patch", "apply_patch", "run_tests",
        "git_status", "git_diff", "record_memory",
        "ask_user", "finish", "blocked",
    }),
    "tester": frozenset({
        "search_code", "find_files", "list_directory", "read_file_range",
        "run_tests", "git_status", "git_diff", "http_check",
        "record_memory", "ask_user", "finish", "blocked",
    }),
    "reviewer": frozenset({
        "search_code", "find_files", "list_directory", "read_file_range",
        "git_status", "git_diff", "record_memory",
        "ask_user", "finish", "blocked",
    }),
    "devops": frozenset({
        "search_code", "find_files", "list_directory", "read_file_range",
        "write_file", "insert_after", "insert_before", "replace_range", "append_file",
        "shell_template", "raw_shell_proposal",
        "run_tests", "git_status", "git_diff", "http_check",
        "record_memory", "ask_user", "finish", "blocked",
    }),
    "security_guard": frozenset({
        "search_code", "find_files", "list_directory", "read_file_range",
        "git_status", "git_diff", "record_memory",
        "ask_user", "finish", "blocked",
    }),
    "memory_manager": frozenset({
        "search_code", "find_files", "read_file_range",
        "record_memory", "ask_user", "finish", "blocked",
    }),
    "cost_guardian": frozenset({
        "record_memory", "ask_user", "finish", "blocked",
    }),
    "reporter": frozenset({
        "search_code", "find_files", "read_file_range",
        "git_status", "git_diff", "record_memory",
        "ask_user", "finish", "blocked",
    }),
}


# ---------------------------------------------------------------------------
# Agent Action dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentAction:
    """A structured action proposed by the LLM reasoning loop.

    Every field is validated before execution. The LLM produces this
    structure; IGRIS validates, risk-classifies, and executes it.
    """

    mode: str = "researcher"  # agent role
    action_type: str = "blocked"
    reason: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_effect: str = ""
    risk_hint: str = "low"
    confidence: float = 0.5
    required_preconditions: List[str] = field(default_factory=list)
    success_check: Dict[str, Any] = field(default_factory=dict)
    fallback_if_blocked: Optional[str] = None

    # Metadata (set by IGRIS, not LLM)
    action_id: str = field(default_factory=lambda: f"act-{uuid.uuid4().hex[:8]}")
    mission_id: str = ""
    trace_id: str = ""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "mode": self.mode,
            "action_type": self.action_type,
            "reason": redact_secrets(self.reason),
            "parameters": _redact_params(self.parameters),
            "expected_effect": redact_secrets(self.expected_effect),
            "risk_hint": self.risk_hint,
            "confidence": self.confidence,
            "required_preconditions": self.required_preconditions,
            "success_check": self.success_check,
            "fallback_if_blocked": self.fallback_if_blocked,
            "mission_id": self.mission_id,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentAction":
        return cls(
            mode=data.get("mode", "researcher"),
            action_type=data.get("action_type", "blocked"),
            reason=data.get("reason", ""),
            parameters=data.get("parameters", {}),
            expected_effect=data.get("expected_effect", ""),
            risk_hint=data.get("risk_hint", "low"),
            confidence=data.get("confidence", 0.5),
            required_preconditions=data.get("required_preconditions", []),
            success_check=data.get("success_check", {}),
            fallback_if_blocked=data.get("fallback_if_blocked"),
            action_id=data.get("action_id", f"act-{uuid.uuid4().hex[:8]}"),
            mission_id=data.get("mission_id", ""),
            trace_id=data.get("trace_id", ""),
            timestamp=data.get("timestamp", ""),
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of validating an AgentAction."""
    valid: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_action: Optional[AgentAction] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }
        if self.sanitized_action:
            d["sanitized_action"] = self.sanitized_action.to_dict()
        return d


def validate_action(action: AgentAction) -> ValidationResult:
    """Validate an agent action against the schema contract.

    Checks:
    - action_type is known
    - mode/role is known
    - risk_hint is valid
    - confidence is in [0, 1]
    - role is allowed to perform the action
    - required parameters are present for the action type
    - no secret content in parameters
    """
    errors: List[str] = []
    warnings: List[str] = []

    # action_type
    if action.action_type not in ACTION_TYPES:
        errors.append(f"Unknown action_type: {action.action_type}")

    # mode/role
    if action.mode not in AGENT_ROLES:
        errors.append(f"Unknown agent role: {action.mode}")

    # risk_hint
    if action.risk_hint not in RISK_HINTS:
        warnings.append(f"Unknown risk_hint '{action.risk_hint}', defaulting to 'unknown'")
        action.risk_hint = "unknown"

    # confidence
    if not isinstance(action.confidence, (int, float)):
        warnings.append("confidence must be numeric, defaulting to 0.5")
        action.confidence = 0.5
    elif action.confidence < 0 or action.confidence > 1:
        warnings.append(f"confidence {action.confidence} clamped to [0,1]")
        action.confidence = max(0.0, min(1.0, action.confidence))

    # Role permission check
    if action.mode in AGENT_ROLES and action.action_type in ACTION_TYPES:
        allowed = ROLE_ALLOWED_ACTIONS.get(action.mode, frozenset())
        if action.action_type not in allowed:
            errors.append(
                f"Role '{action.mode}' is not allowed to perform '{action.action_type}'"
            )

    # Required parameters per action type
    param_errors = _validate_parameters(action.action_type, action.parameters)
    errors.extend(param_errors)

    # Secret content check in parameters
    secret_warnings = _check_secret_content(action.parameters)
    if secret_warnings:
        errors.extend(secret_warnings)

    # Reason is required
    if not action.reason or not action.reason.strip():
        warnings.append("Action has no reason — LLM should explain why")

    sanitized = action if not errors else None
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        sanitized_action=sanitized,
    )


def _validate_parameters(action_type: str, params: Dict[str, Any]) -> List[str]:
    """Validate required parameters for each action type."""
    errors: List[str] = []

    required_params: Dict[str, List[str]] = {
        "search_code": ["pattern"],
        "find_files": ["pattern"],
        "list_directory": ["path"],
        "read_file_range": ["path"],
        "write_file": ["path", "content"],
        "insert_after": ["path", "anchor", "content"],
        "insert_before": ["path", "anchor", "content"],
        "replace_range": ["path", "start", "end", "content"],
        "append_file": ["path", "content"],
        "propose_patch": ["files"],
        "apply_patch": ["patch_id"],
        "run_tests": [],
        "git_status": [],
        "git_diff": [],
        "shell_template": ["template_id"],
        "raw_shell_proposal": ["command"],
        "http_check": ["url"],
        "update_plan": ["updates"],
        "record_memory": ["event_type", "content"],
        "ask_user": ["question"],
        "finish": ["summary"],
        "blocked": ["reason"],
    }

    reqs = required_params.get(action_type, [])
    for req in reqs:
        if req not in params:
            errors.append(f"Missing required parameter '{req}' for action '{action_type}'")

    return errors


def _check_secret_content(params: Dict[str, Any]) -> List[str]:
    """Check if parameters contain secret-like content."""
    from igris.core.safety import detect_secret_like_content

    issues: List[str] = []
    for key, value in params.items():
        if isinstance(value, str) and detect_secret_like_content(value):
            issues.append(f"Parameter '{key}' appears to contain secret content")
    return issues


def _redact_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Redact secret content from parameters for logging."""
    redacted: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, str):
            redacted[k] = redact_secrets(v)
        elif isinstance(v, list):
            redacted[k] = [redact_secrets(str(i)) if isinstance(i, str) else i for i in v]
        elif isinstance(v, dict):
            redacted[k] = _redact_params(v)
        else:
            redacted[k] = v
    return redacted


# ---------------------------------------------------------------------------
# Action routing — maps action_type to execution category
# ---------------------------------------------------------------------------

ACTION_ROUTING = {
    # Navigation / read-only
    "search_code": "code_navigation",
    "find_files": "code_navigation",
    "list_directory": "code_navigation",
    "read_file_range": "code_navigation",
    # Modification
    "write_file": "tool_runtime",
    "insert_after": "tool_runtime",
    "insert_before": "tool_runtime",
    "replace_range": "tool_runtime",
    "append_file": "tool_runtime",
    "propose_patch": "tool_runtime",
    "apply_patch": "tool_runtime",
    # Testing
    "run_tests": "tool_runtime",
    # Git
    "git_status": "tool_runtime",
    "git_diff": "tool_runtime",
    # Shell
    "shell_template": "command_risk_engine",
    "raw_shell_proposal": "command_risk_engine",
    # HTTP
    "http_check": "tool_runtime",
    # Planning
    "update_plan": "mission_controller",
    "record_memory": "memory",
    # Human
    "ask_user": "human_gate",
    # Terminal
    "finish": "terminal",
    "blocked": "terminal",
}


def get_action_route(action_type: str) -> str:
    """Get the execution category for an action type."""
    return ACTION_ROUTING.get(action_type, "unknown")


# ---------------------------------------------------------------------------
# JSON schema for LLM output validation
# ---------------------------------------------------------------------------

ACTION_JSON_SCHEMA = {
    "type": "object",
    "required": ["mode", "action_type", "reason", "parameters"],
    "properties": {
        "mode": {
            "type": "string",
            "enum": list(AGENT_ROLES),
            "description": "Current agent role/mode",
        },
        "action_type": {
            "type": "string",
            "enum": list(ACTION_TYPES),
            "description": "Type of action to perform",
        },
        "reason": {
            "type": "string",
            "description": "Why this action is needed",
        },
        "parameters": {
            "type": "object",
            "description": "Action-specific parameters",
        },
        "expected_effect": {
            "type": "string",
            "description": "What this action should achieve",
        },
        "risk_hint": {
            "type": "string",
            "enum": list(RISK_HINTS),
            "description": "LLM's assessment of risk level",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "LLM confidence in this action (0-1)",
        },
        "required_preconditions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Conditions that must be true before execution",
        },
        "success_check": {
            "type": "object",
            "description": "How to verify the action succeeded",
        },
        "fallback_if_blocked": {
            "type": ["string", "null"],
            "description": "Alternative action_type if this one is blocked",
        },
    },
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Parse LLM output into AgentAction
# ---------------------------------------------------------------------------

def parse_llm_action(raw_output: str) -> tuple[Optional[AgentAction], List[str]]:
    """Parse raw LLM output into a validated AgentAction.

    Returns (action, errors). If parsing fails, action is None.
    """
    errors: List[str] = []

    # Strip markdown code fences if present
    text = raw_output.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, [f"Invalid JSON from LLM: {e}"]

    if not isinstance(data, dict):
        return None, ["LLM output is not a JSON object"]

    # Required fields
    for field_name in ("mode", "action_type", "reason", "parameters"):
        if field_name not in data:
            errors.append(f"Missing required field: {field_name}")

    if errors:
        return None, errors

    action = AgentAction.from_dict(data)
    validation = validate_action(action)

    if not validation.valid:
        return None, validation.errors

    return validation.sanitized_action, validation.warnings


# ---------------------------------------------------------------------------
# Agent Registry (minimal — single loop with role mode)
# ---------------------------------------------------------------------------

@dataclass
class AgentRegistryEntry:
    """An agent role definition in the registry."""
    role: str
    description: str
    allowed_actions: frozenset
    tool_families: List[str] = field(default_factory=list)
    safety_notes: str = ""
    max_risk_level: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "description": self.description,
            "allowed_actions": sorted(self.allowed_actions),
            "tool_families": self.tool_families,
            "safety_notes": self.safety_notes,
            "max_risk_level": self.max_risk_level,
        }


# Default registry entries
AGENT_REGISTRY: Dict[str, AgentRegistryEntry] = {
    "coordinator": AgentRegistryEntry(
        role="coordinator",
        description="Holds mission focus, plan, step tracking. Can delegate to any role.",
        allowed_actions=ROLE_ALLOWED_ACTIONS["coordinator"],
        tool_families=["mission", "plan", "state", "report"],
        safety_notes="Can block drift and reassign roles",
        max_risk_level="high",
    ),
    "planner": AgentRegistryEntry(
        role="planner",
        description="Decomposes goals into plans, evaluates preconditions, no risky execution.",
        allowed_actions=ROLE_ALLOWED_ACTIONS["planner"],
        tool_families=["goap", "state", "memory"],
        safety_notes="No risky execution allowed",
        max_risk_level="low",
    ),
    "researcher": AgentRegistryEntry(
        role="researcher",
        description="Explores repo, docs, logs, server facts. Read-only by default.",
        allowed_actions=ROLE_ALLOWED_ACTIONS["researcher"],
        tool_families=["search", "read", "grep", "logs"],
        safety_notes="Read-only, no write operations",
        max_risk_level="low",
    ),
    "coder": AgentRegistryEntry(
        role="coder",
        description="Modifies code and files in workspace. No system/server tools.",
        allowed_actions=ROLE_ALLOWED_ACTIONS["coder"],
        tool_families=["read", "write", "patch", "test"],
        safety_notes="No system/server tools, workspace only",
        max_risk_level="medium",
    ),
    "tester": AgentRegistryEntry(
        role="tester",
        description="Executes tests, interprets failures, proposes verifications.",
        allowed_actions=ROLE_ALLOWED_ACTIONS["tester"],
        tool_families=["run_tests", "logs", "read"],
        safety_notes="Write only if authorized",
        max_risk_level="low",
    ),
    "reviewer": AgentRegistryEntry(
        role="reviewer",
        description="Reviews diff, quality, regressions, secrets. Can block delivery.",
        allowed_actions=ROLE_ALLOWED_ACTIONS["reviewer"],
        tool_families=["git_diff", "tests", "secret_scan"],
        safety_notes="Can block delivery on quality issues",
        max_risk_level="low",
    ),
    "devops": AgentRegistryEntry(
        role="devops",
        description="Deploy, server, nginx, Docker, systemd, SSL operations.",
        allowed_actions=ROLE_ALLOWED_ACTIONS["devops"],
        tool_families=["ssh", "docker", "nginx", "systemd", "http"],
        safety_notes="Rollback required for high risk actions",
        max_risk_level="high",
    ),
    "security_guard": AgentRegistryEntry(
        role="security_guard",
        description="Evaluates risk, secrets, policy, paths, commands. Can block any action.",
        allowed_actions=ROLE_ALLOWED_ACTIONS["security_guard"],
        tool_families=["risk", "secrets", "policy"],
        safety_notes="Can block any action on security grounds",
        max_risk_level="low",
    ),
    "memory_manager": AgentRegistryEntry(
        role="memory_manager",
        description="Saves/retrieves lessons, failures, patterns. No shell execution.",
        allowed_actions=ROLE_ALLOWED_ACTIONS["memory_manager"],
        tool_families=["memory", "retrieval"],
        safety_notes="No shell execution",
        max_risk_level="low",
    ),
    "cost_guardian": AgentRegistryEntry(
        role="cost_guardian",
        description="Selects provider, manages budget, escalates costs.",
        allowed_actions=ROLE_ALLOWED_ACTIONS["cost_guardian"],
        tool_families=["routing", "cost"],
        safety_notes="Can block expensive model usage",
        max_risk_level="low",
    ),
    "reporter": AgentRegistryEntry(
        role="reporter",
        description="Produces final reports, artifacts, next steps. No execution.",
        allowed_actions=ROLE_ALLOWED_ACTIONS["reporter"],
        tool_families=["reports", "logs", "artifacts"],
        safety_notes="No execution, report-only",
        max_risk_level="low",
    ),
}


def get_registry_entry(role: str) -> Optional[AgentRegistryEntry]:
    """Get an agent registry entry by role name."""
    return AGENT_REGISTRY.get(role)


def list_registry() -> List[Dict[str, Any]]:
    """List all registered agent roles."""
    return [entry.to_dict() for entry in AGENT_REGISTRY.values()]


def is_action_allowed_for_role(role: str, action_type: str) -> bool:
    """Check if a role is allowed to perform an action type."""
    entry = AGENT_REGISTRY.get(role)
    if not entry:
        return False
    return action_type in entry.allowed_actions
