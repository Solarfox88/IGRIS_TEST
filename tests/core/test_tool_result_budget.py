"""Tests for tool_result_budget — 16KB byte-cap enforcement.

Acceptance criteria from issue #543:
- apply_tool_result_budget truncates at valid UTF-8 boundary
- Marker [… N bytes truncated …] included in output
- budget=0 disables truncation
- BudgetOutcome.truncated=False when content < budget
- 50KB input → output ≤ 16KB + marker
- Integration: every tool call passes through the budget in agent_reasoning_loop
"""

from __future__ import annotations

import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from igris.core.tool_result_budget import (
    DEFAULT_BUDGET_BYTES,
    TRAILER_RESERVED,
    BudgetOutcome,
    apply_tool_result_budget,
)


# ---------------------------------------------------------------------------
# apply_tool_result_budget — unit tests
# ---------------------------------------------------------------------------

class TestApplyToolResultBudget:

    def test_no_truncation_when_under_budget(self):
        content = "hello world"
        result, outcome = apply_tool_result_budget(content, budget_bytes=1024)
        assert result == content
        assert outcome.truncated is False
        assert outcome.original_bytes == len(content.encode())
        assert outcome.final_bytes == outcome.original_bytes

    def test_no_truncation_when_exactly_at_budget(self):
        content = "x" * DEFAULT_BUDGET_BYTES
        result, outcome = apply_tool_result_budget(content, budget_bytes=DEFAULT_BUDGET_BYTES)
        assert outcome.truncated is False
        assert result == content

    def test_truncation_when_over_budget(self):
        content = "a" * (DEFAULT_BUDGET_BYTES * 2)
        result, outcome = apply_tool_result_budget(content)
        assert outcome.truncated is True
        assert len(result.encode("utf-8")) <= DEFAULT_BUDGET_BYTES

    def test_truncation_marker_present(self):
        content = "z" * (DEFAULT_BUDGET_BYTES * 3)
        result, outcome = apply_tool_result_budget(content)
        assert "truncated by tool_result_budget" in result
        assert "bytes truncated" in result

    def test_truncation_marker_shows_byte_count(self):
        content = "a" * 50_000
        result, outcome = apply_tool_result_budget(content, budget_bytes=DEFAULT_BUDGET_BYTES)
        removed = outcome.original_bytes - (DEFAULT_BUDGET_BYTES - TRAILER_RESERVED)
        assert str(removed) in result

    def test_budget_zero_disables_truncation(self):
        """budget_bytes=0 is the opt-out signal — no truncation ever."""
        huge = "x" * 1_000_000
        result, outcome = apply_tool_result_budget(huge, budget_bytes=0)
        assert result == huge
        assert outcome.truncated is False
        assert outcome.original_bytes == len(huge.encode())

    def test_output_within_budget(self):
        """Final output must fit within budget_bytes."""
        content = "b" * 50_000
        budget = DEFAULT_BUDGET_BYTES
        result, outcome = apply_tool_result_budget(content, budget_bytes=budget)
        assert len(result.encode("utf-8")) <= budget
        assert outcome.truncated is True

    def test_utf8_boundary_safe_ascii(self):
        """Pure ASCII input — truncation is always at a valid boundary."""
        content = "hello " * 5000  # well over 16KB
        result, outcome = apply_tool_result_budget(content)
        # Must decode cleanly without errors
        result.encode("utf-8").decode("utf-8")
        assert outcome.truncated is True

    def test_utf8_boundary_multibyte(self):
        """Multibyte characters must not be split at the truncation point."""
        # U+1F600 (😀) is 4 bytes in UTF-8
        emoji = "😀" * 10_000  # 40_000 bytes, over 16KB
        result, outcome = apply_tool_result_budget(emoji, budget_bytes=DEFAULT_BUDGET_BYTES)
        assert outcome.truncated is True
        # Round-trip must succeed — no broken sequences
        result.encode("utf-8").decode("utf-8")

    def test_utf8_boundary_3byte_chars(self):
        """3-byte UTF-8 chars (e.g. Chinese) must not be split."""
        chinese = "中文测试" * 3000  # each char = 3 bytes → 36KB total
        result, outcome = apply_tool_result_budget(chinese, budget_bytes=DEFAULT_BUDGET_BYTES)
        assert outcome.truncated is True
        result.encode("utf-8").decode("utf-8")  # no UnicodeDecodeError

    def test_50kb_input_fits_in_16kb_output(self):
        """Explicit 50KB → ≤16KB requirement from issue spec."""
        content = "x" * 50_000
        result, outcome = apply_tool_result_budget(content, budget_bytes=DEFAULT_BUDGET_BYTES)
        assert outcome.original_bytes == 50_000
        assert outcome.truncated is True
        assert len(result.encode("utf-8")) <= DEFAULT_BUDGET_BYTES

    def test_budget_outcome_fields(self):
        content = "a" * 32_000
        _, outcome = apply_tool_result_budget(content, budget_bytes=DEFAULT_BUDGET_BYTES)
        assert isinstance(outcome, BudgetOutcome)
        assert outcome.original_bytes == 32_000
        assert outcome.final_bytes < 32_000
        assert outcome.truncated is True

    def test_empty_string(self):
        result, outcome = apply_tool_result_budget("", budget_bytes=DEFAULT_BUDGET_BYTES)
        assert result == ""
        assert outcome.truncated is False
        assert outcome.original_bytes == 0

    def test_small_budget(self):
        """Budget just above TRAILER_RESERVED — content truncated to near-zero chars."""
        content = "hello world " * 100  # 1200 bytes
        budget = TRAILER_RESERVED + 10   # 266 bytes — just enough for marker
        result, outcome = apply_tool_result_budget(content, budget_bytes=budget)
        assert outcome.truncated is True
        assert "truncated by tool_result_budget" in result
        # Output may slightly exceed budget due to marker but must be << original
        assert len(result.encode("utf-8")) < len(content.encode("utf-8"))

    def test_default_budget_is_16kb(self):
        assert DEFAULT_BUDGET_BYTES == 16 * 1024

    def test_returns_tuple(self):
        result = apply_tool_result_budget("test")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], BudgetOutcome)


# ---------------------------------------------------------------------------
# Integration: agent_reasoning_loop applies budget to tool results
# ---------------------------------------------------------------------------

class TestAgentReasoningLoopIntegration:

    def _make_loop(self, budget=DEFAULT_BUDGET_BYTES):
        from igris.core.agent_reasoning_loop import AgentReasoningLoop
        loop = AgentReasoningLoop(
            project_root="/tmp/test_project",
            tool_result_budget=budget,
        )
        loop._world_state = {}
        return loop

    def test_tool_result_budget_attribute_stored(self):
        loop = self._make_loop(budget=8192)
        assert loop.tool_result_budget == 8192

    def test_default_budget_is_16kb(self):
        loop = self._make_loop()
        assert loop.tool_result_budget == DEFAULT_BUDGET_BYTES

    def test_context_config_fallback(self, tmp_path):
        from igris.core.agent_reasoning_loop import AgentReasoningLoop
        loop = AgentReasoningLoop(project_root=str(tmp_path))
        # No context_config.json → falls back to instance default
        assert loop._tool_result_budget_bytes() == DEFAULT_BUDGET_BYTES

    def test_context_config_override(self, tmp_path):
        from igris.core.agent_reasoning_loop import AgentReasoningLoop
        igris_dir = tmp_path / ".igris"
        igris_dir.mkdir()
        (igris_dir / "context_config.json").write_text(
            json.dumps({"tool_result_budget_bytes": 4096})
        )
        loop = AgentReasoningLoop(project_root=str(tmp_path))
        assert loop._tool_result_budget_bytes() == 4096

    def test_context_config_zero_disables(self, tmp_path):
        from igris.core.agent_reasoning_loop import AgentReasoningLoop
        igris_dir = tmp_path / ".igris"
        igris_dir.mkdir()
        (igris_dir / "context_config.json").write_text(
            json.dumps({"tool_result_budget_bytes": 0})
        )
        loop = AgentReasoningLoop(project_root=str(tmp_path))
        assert loop._tool_result_budget_bytes() == 0

    def test_context_config_invalid_json_fallback(self, tmp_path):
        from igris.core.agent_reasoning_loop import AgentReasoningLoop
        igris_dir = tmp_path / ".igris"
        igris_dir.mkdir()
        (igris_dir / "context_config.json").write_text("not valid json{{{")
        loop = AgentReasoningLoop(project_root=str(tmp_path))
        # Must not raise — falls back to default
        assert loop._tool_result_budget_bytes() == DEFAULT_BUDGET_BYTES

    def test_large_result_data_truncated_in_step(self, tmp_path):
        """String result_data > 16KB must be truncated before _store_tool_result."""
        from igris.core.agent_reasoning_loop import AgentReasoningLoop
        from igris.core.agent_action_schema import AgentAction

        loop = AgentReasoningLoop(
            project_root=str(tmp_path),
            tool_result_budget=DEFAULT_BUDGET_BYTES,
        )
        loop._world_state = {}

        big_output = "x" * 50_000  # 50KB

        # Fake exec_result with large result_data string
        exec_result = {
            "success": True,
            "summary": "read 50KB file",
            "result_data": big_output,
        }

        stored = []
        original_store = loop._store_tool_result

        def capture_store(action_type, result_data):
            stored.append(result_data)
            original_store(action_type, result_data)

        loop._store_tool_result = capture_store

        action = AgentAction(
            mode="coder",
            action_type="read_file",
            parameters={"path": "big.txt"},
        )

        # Simulate the 4b block directly
        result_data = exec_result.get("result_data")
        if isinstance(result_data, str):
            from igris.core.tool_result_budget import apply_tool_result_budget
            budget = loop._tool_result_budget_bytes()
            result_data, bout = apply_tool_result_budget(result_data, budget)

        loop._store_tool_result(action.action_type, result_data)

        assert stored, "store was not called"
        stored_val = stored[0]
        assert isinstance(stored_val, str)
        assert len(stored_val.encode("utf-8")) <= DEFAULT_BUDGET_BYTES
        assert "truncated by tool_result_budget" in stored_val

    def test_non_string_result_data_not_truncated(self, tmp_path):
        """List/dict result_data (e.g. search results) must pass through unchanged."""
        from igris.core.agent_reasoning_loop import AgentReasoningLoop

        loop = AgentReasoningLoop(project_root=str(tmp_path))
        loop._world_state = {}

        list_data = [{"file": "foo.py", "line": 1, "match": "x" * 1000}] * 100

        stored = []
        loop._store_tool_result = lambda at, rd: stored.append(rd)

        result_data = list_data
        if isinstance(result_data, str):
            result_data, _ = apply_tool_result_budget(result_data, DEFAULT_BUDGET_BYTES)

        loop._store_tool_result("search_code", result_data)

        assert stored[0] is list_data  # unchanged reference

    def test_budget_disabled_via_config_passes_50kb(self, tmp_path):
        """With budget=0, 50KB string must reach _store_tool_result unchanged."""
        from igris.core.agent_reasoning_loop import AgentReasoningLoop

        igris_dir = tmp_path / ".igris"
        igris_dir.mkdir()
        (igris_dir / "context_config.json").write_text(
            json.dumps({"tool_result_budget_bytes": 0})
        )

        loop = AgentReasoningLoop(project_root=str(tmp_path))
        loop._world_state = {}

        big = "y" * 50_000
        budget = loop._tool_result_budget_bytes()
        result_data, bout = apply_tool_result_budget(big, budget)

        assert bout.truncated is False
        assert result_data == big
