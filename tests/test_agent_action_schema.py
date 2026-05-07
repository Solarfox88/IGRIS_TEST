"""Tests for Agent Action Schema — Epic #58.

Validates the core contract between LLM reasoning and IGRIS execution.
"""

import json
import pytest

from igris.core.agent_action_schema import (
    ACTION_TYPES,
    AGENT_ROLES,
    RISK_HINTS,
    READ_ONLY_ACTIONS,
    WRITE_ACTIONS,
    RISK_GATED_ACTIONS,
    ROLE_ALLOWED_ACTIONS,
    ACTION_ROUTING,
    ACTION_JSON_SCHEMA,
    AgentAction,
    ValidationResult,
    validate_action,
    parse_llm_action,
    get_action_route,
    AgentRegistryEntry,
    AGENT_REGISTRY,
    get_registry_entry,
    list_registry,
    is_action_allowed_for_role,
)


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

class TestSchemaConstants:
    """Verify schema constants are well-defined."""

    def test_action_types_are_tuple(self):
        assert isinstance(ACTION_TYPES, tuple)
        assert len(ACTION_TYPES) == 22  # +4 safe-edit actions (#76)

    def test_all_required_action_types_present(self):
        required = {
            "search_code", "find_files", "list_directory", "read_file_range",
            "write_file", "insert_after", "insert_before", "replace_range", "append_file",
            "propose_patch", "apply_patch", "run_tests",
            "git_status", "git_diff", "shell_template", "raw_shell_proposal",
            "http_check", "update_plan", "record_memory",
            "ask_user", "finish", "blocked",
        }
        assert required == set(ACTION_TYPES)

    def test_agent_roles_are_tuple(self):
        assert isinstance(AGENT_ROLES, tuple)
        assert len(AGENT_ROLES) == 11

    def test_all_required_roles_present(self):
        required = {
            "coordinator", "planner", "researcher", "coder", "tester",
            "reviewer", "devops", "security_guard", "memory_manager",
            "cost_guardian", "reporter",
        }
        assert required == set(AGENT_ROLES)

    def test_risk_hints(self):
        assert set(RISK_HINTS) == {"low", "medium", "high", "critical", "unknown"}

    def test_read_only_actions_are_subset(self):
        assert READ_ONLY_ACTIONS.issubset(set(ACTION_TYPES))

    def test_write_actions_are_subset(self):
        assert WRITE_ACTIONS.issubset(set(ACTION_TYPES))

    def test_no_overlap_read_write(self):
        assert READ_ONLY_ACTIONS.isdisjoint(WRITE_ACTIONS)

    def test_all_actions_are_read_or_write(self):
        assert READ_ONLY_ACTIONS | WRITE_ACTIONS == set(ACTION_TYPES)

    def test_risk_gated_are_write_actions(self):
        assert RISK_GATED_ACTIONS.issubset(WRITE_ACTIONS)

    def test_every_action_has_route(self):
        for at in ACTION_TYPES:
            assert at in ACTION_ROUTING, f"No route for {at}"

    def test_json_schema_has_required_fields(self):
        assert "type" in ACTION_JSON_SCHEMA
        assert ACTION_JSON_SCHEMA["type"] == "object"
        assert "required" in ACTION_JSON_SCHEMA
        assert set(ACTION_JSON_SCHEMA["required"]) == {"mode", "action_type", "reason", "parameters"}


# ---------------------------------------------------------------------------
# AgentAction creation and serialization
# ---------------------------------------------------------------------------

class TestAgentAction:
    """Test AgentAction dataclass."""

    def test_default_creation(self):
        a = AgentAction()
        assert a.mode == "researcher"
        assert a.action_type == "blocked"
        assert a.confidence == 0.5

    def test_to_dict(self):
        a = AgentAction(
            mode="coder",
            action_type="write_file",
            reason="test reason",
            parameters={"path": "foo.py", "content": "print('hi')"},
        )
        d = a.to_dict()
        assert d["mode"] == "coder"
        assert d["action_type"] == "write_file"
        assert "action_id" in d
        assert "timestamp" in d

    def test_from_dict(self):
        data = {
            "mode": "tester",
            "action_type": "run_tests",
            "reason": "check if tests pass",
            "parameters": {"target": "tests/"},
        }
        a = AgentAction.from_dict(data)
        assert a.mode == "tester"
        assert a.action_type == "run_tests"
        assert a.parameters["target"] == "tests/"

    def test_roundtrip(self):
        a = AgentAction(
            mode="devops",
            action_type="http_check",
            reason="check health",
            parameters={"url": "http://localhost:7778/api/health"},
            confidence=0.9,
        )
        d = a.to_dict()
        a2 = AgentAction.from_dict(d)
        assert a2.mode == a.mode
        assert a2.action_type == a.action_type
        assert a2.confidence == a.confidence

    def test_secret_redaction_in_params(self):
        fake_key = "sk-" + "a" * 30  # 33 chars total, matches sk-[A-Za-z0-9]{20,}
        a = AgentAction(
            mode="coder",
            action_type="write_file",
            reason="test",
            parameters={"path": "foo.py", "content": fake_key},
        )
        d = a.to_dict()
        # The secret should be redacted in the output
        assert fake_key not in d["parameters"]["content"]
        assert "REDACTED" in d["parameters"]["content"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    """Test action validation."""

    def test_valid_action(self):
        a = AgentAction(
            mode="coder",
            action_type="read_file_range",
            reason="check code",
            parameters={"path": "server.py"},
        )
        result = validate_action(a)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_unknown_action_type(self):
        a = AgentAction(
            mode="coder",
            action_type="hack_server",
            reason="test",
            parameters={},
        )
        result = validate_action(a)
        assert result.valid is False
        assert any("Unknown action_type" in e for e in result.errors)

    def test_unknown_role(self):
        a = AgentAction(
            mode="superadmin",
            action_type="read_file_range",
            reason="test",
            parameters={"path": "foo.py"},
        )
        result = validate_action(a)
        assert result.valid is False
        assert any("Unknown agent role" in e for e in result.errors)

    def test_invalid_risk_hint(self):
        a = AgentAction(
            mode="coder",
            action_type="read_file_range",
            reason="test",
            parameters={"path": "foo.py"},
            risk_hint="extreme",
        )
        result = validate_action(a)
        # Should warn and default to unknown
        assert result.valid is True
        assert any("Unknown risk_hint" in w for w in result.warnings)

    def test_confidence_clamped(self):
        a = AgentAction(
            mode="coder",
            action_type="read_file_range",
            reason="test",
            parameters={"path": "foo.py"},
            confidence=1.5,
        )
        result = validate_action(a)
        assert result.valid is True
        assert any("clamped" in w for w in result.warnings)

    def test_missing_required_parameters(self):
        a = AgentAction(
            mode="coder",
            action_type="write_file",
            reason="test",
            parameters={"path": "foo.py"},  # missing "content"
        )
        result = validate_action(a)
        assert result.valid is False
        assert any("Missing required parameter 'content'" in e for e in result.errors)

    def test_role_permission_denied(self):
        # Planner should not be allowed to write_file
        a = AgentAction(
            mode="planner",
            action_type="write_file",
            reason="test",
            parameters={"path": "foo.py", "content": "hello"},
        )
        result = validate_action(a)
        assert result.valid is False
        assert any("not allowed" in e for e in result.errors)

    def test_secret_in_params_blocked(self):
        fake_key = "sk-" + "x" * 30
        a = AgentAction(
            mode="coder",
            action_type="write_file",
            reason="test",
            parameters={"path": "foo.py", "content": f"OPENAI_API_KEY={fake_key}"},
        )
        result = validate_action(a)
        assert result.valid is False
        assert any("secret" in e.lower() for e in result.errors)

    def test_empty_reason_warning(self):
        a = AgentAction(
            mode="coder",
            action_type="read_file_range",
            reason="",
            parameters={"path": "foo.py"},
        )
        result = validate_action(a)
        assert result.valid is True
        assert any("reason" in w.lower() for w in result.warnings)

    def test_finish_requires_summary(self):
        a = AgentAction(
            mode="reporter",
            action_type="finish",
            reason="done",
            parameters={},  # missing "summary"
        )
        result = validate_action(a)
        assert result.valid is False
        assert any("summary" in e for e in result.errors)

    def test_blocked_requires_reason(self):
        a = AgentAction(
            mode="coder",
            action_type="blocked",
            reason="stuck",
            parameters={},  # missing "reason" in params
        )
        result = validate_action(a)
        assert result.valid is False
        assert any("reason" in e for e in result.errors)

    def test_ask_user_requires_question(self):
        a = AgentAction(
            mode="coordinator",
            action_type="ask_user",
            reason="need info",
            parameters={},  # missing "question"
        )
        result = validate_action(a)
        assert result.valid is False
        assert any("question" in e for e in result.errors)


# ---------------------------------------------------------------------------
# LLM output parsing
# ---------------------------------------------------------------------------

class TestParseLLMAction:
    """Test parsing raw LLM output into actions."""

    def test_valid_json(self):
        raw = json.dumps({
            "mode": "coder",
            "action_type": "read_file_range",
            "reason": "check code",
            "parameters": {"path": "server.py"},
        })
        action, issues = parse_llm_action(raw)
        assert action is not None
        assert action.action_type == "read_file_range"

    def test_json_with_code_fence(self):
        raw = '```json\n{"mode": "coder", "action_type": "git_status", "reason": "check", "parameters": {}}\n```'
        action, issues = parse_llm_action(raw)
        assert action is not None
        assert action.action_type == "git_status"

    def test_invalid_json(self):
        raw = "This is not JSON at all"
        action, issues = parse_llm_action(raw)
        assert action is None
        assert any("Invalid JSON" in e for e in issues)

    def test_missing_required_field(self):
        raw = json.dumps({"mode": "coder", "action_type": "read_file_range"})
        action, issues = parse_llm_action(raw)
        assert action is None
        assert any("Missing required field" in e for e in issues)

    def test_non_object_json(self):
        raw = json.dumps([1, 2, 3])
        action, issues = parse_llm_action(raw)
        assert action is None
        assert any("not a JSON object" in e for e in issues)


# ---------------------------------------------------------------------------
# Action routing
# ---------------------------------------------------------------------------

class TestActionRouting:
    """Test action type routing."""

    def test_navigation_routes(self):
        for at in ("search_code", "find_files", "list_directory", "read_file_range"):
            assert get_action_route(at) == "code_navigation"

    def test_tool_runtime_routes(self):
        for at in ("write_file", "propose_patch", "apply_patch", "run_tests", "git_status", "git_diff", "http_check"):
            assert get_action_route(at) == "tool_runtime"

    def test_risk_engine_routes(self):
        for at in ("shell_template", "raw_shell_proposal"):
            assert get_action_route(at) == "command_risk_engine"

    def test_terminal_routes(self):
        assert get_action_route("finish") == "terminal"
        assert get_action_route("blocked") == "terminal"

    def test_human_route(self):
        assert get_action_route("ask_user") == "human_gate"

    def test_unknown_action(self):
        assert get_action_route("nonexistent") == "unknown"


# ---------------------------------------------------------------------------
# Agent Registry
# ---------------------------------------------------------------------------

class TestAgentRegistry:
    """Test the Agent Registry."""

    def test_all_roles_registered(self):
        for role in AGENT_ROLES:
            assert role in AGENT_REGISTRY

    def test_registry_entry_to_dict(self):
        entry = get_registry_entry("coder")
        assert entry is not None
        d = entry.to_dict()
        assert d["role"] == "coder"
        assert "write_file" in d["allowed_actions"]

    def test_list_registry(self):
        entries = list_registry()
        assert len(entries) == len(AGENT_ROLES)
        roles = {e["role"] for e in entries}
        assert roles == set(AGENT_ROLES)

    def test_unknown_role(self):
        assert get_registry_entry("nonexistent") is None

    def test_coordinator_has_all_actions(self):
        entry = get_registry_entry("coordinator")
        assert entry is not None
        assert entry.allowed_actions == frozenset(ACTION_TYPES)

    def test_security_guard_no_write(self):
        entry = get_registry_entry("security_guard")
        assert entry is not None
        assert "write_file" not in entry.allowed_actions
        assert "raw_shell_proposal" not in entry.allowed_actions

    def test_is_action_allowed(self):
        assert is_action_allowed_for_role("coder", "write_file") is True
        assert is_action_allowed_for_role("planner", "write_file") is False
        assert is_action_allowed_for_role("nonexistent", "read_file_range") is False

    def test_role_max_risk_levels(self):
        # Researcher should be low risk
        entry = get_registry_entry("researcher")
        assert entry is not None
        assert entry.max_risk_level == "low"
        # DevOps can be high risk
        entry = get_registry_entry("devops")
        assert entry is not None
        assert entry.max_risk_level == "high"


# ---------------------------------------------------------------------------
# No shell / no secret enforcement
# ---------------------------------------------------------------------------

class TestSafetyEnforcement:
    """Verify safety constraints are built into the schema."""

    def test_raw_shell_is_risk_gated(self):
        assert "raw_shell_proposal" in RISK_GATED_ACTIONS

    def test_shell_template_is_risk_gated(self):
        assert "shell_template" in RISK_GATED_ACTIONS

    def test_raw_shell_routes_to_risk_engine(self):
        assert get_action_route("raw_shell_proposal") == "command_risk_engine"

    def test_researcher_cannot_raw_shell(self):
        assert not is_action_allowed_for_role("researcher", "raw_shell_proposal")

    def test_planner_cannot_raw_shell(self):
        assert not is_action_allowed_for_role("planner", "raw_shell_proposal")

    def test_security_guard_cannot_raw_shell(self):
        assert not is_action_allowed_for_role("security_guard", "raw_shell_proposal")

    def test_devops_can_raw_shell(self):
        assert is_action_allowed_for_role("devops", "raw_shell_proposal")

    def test_coordinator_can_raw_shell(self):
        assert is_action_allowed_for_role("coordinator", "raw_shell_proposal")
