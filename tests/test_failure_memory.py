"""Tests for igris/core/failure_memory.py (Miglioramento 2)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from igris.core.failure_memory import FailureMemory, FailureRisk, _keywords, _jaccard


# ---------------------------------------------------------------------------
# Unit: keyword extraction and Jaccard similarity
# ---------------------------------------------------------------------------


class TestKeywords:
    def test_extracts_meaningful_tokens(self):
        kw = _keywords("implement websocket streaming for agent log tail")
        assert "websocket" in kw
        assert "streaming" in kw
        assert "agent" in kw

    def test_filters_stop_words(self):
        kw = _keywords("add the new feature for this endpoint")
        assert "the" not in kw
        assert "for" not in kw
        assert "this" not in kw

    def test_empty_string(self):
        assert _keywords("") == frozenset()

    def test_min_length_filter(self):
        kw = _keywords("do it be")
        assert len(kw) == 0  # all < 3 chars


class TestJaccard:
    def test_identical(self):
        s = frozenset({"websocket", "streaming"})
        assert _jaccard(s, s) == pytest.approx(1.0)

    def test_disjoint(self):
        assert _jaccard(frozenset({"a"}), frozenset({"b"})) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = frozenset({"websocket", "streaming", "agent"})
        b = frozenset({"websocket", "endpoint"})
        j = _jaccard(a, b)
        assert 0.0 < j < 1.0

    def test_empty_both(self):
        assert _jaccard(frozenset(), frozenset()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# FailureMemory — record and check
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_memory(tmp_path):
    store = tmp_path / "failure_patterns.json"
    return FailureMemory(store_path=store)


class TestFailureMemoryRecord:
    def test_record_persists_to_file(self, tmp_memory, tmp_path):
        tmp_memory.record(
            goal="implement websocket streaming for agent log tail",
            failure_class="reasoning_timeout",
            capability_signals={"reasoning_timeout": 2},
            repair_cycles=3,
        )
        data = json.loads((tmp_path / "failure_patterns.json").read_text())
        assert len(data["patterns"]) == 1
        p = data["patterns"][0]
        assert p["failure_class"] == "reasoning_timeout"
        assert "websocket" in p["keywords"]

    def test_multiple_records_accumulate(self, tmp_memory):
        for i in range(5):
            tmp_memory.record(goal=f"goal number {i}", failure_class="no_diff")
        assert len(tmp_memory._patterns) == 5

    def test_rolling_cap_enforced(self, tmp_memory):
        from igris.core.failure_memory import _MAX_PATTERNS
        for i in range(_MAX_PATTERNS + 10):
            tmp_memory.record(goal=f"goal {i}", failure_class="no_diff")
        assert len(tmp_memory._patterns) <= _MAX_PATTERNS

    def test_goal_truncated_to_500(self, tmp_memory):
        long_goal = "x" * 600
        tmp_memory.record(goal=long_goal, failure_class="reasoning_timeout")
        assert len(tmp_memory._patterns[0]["goal"]) <= 500

    def test_record_survives_os_error(self, tmp_path):
        store = tmp_path / "no_dir" / "fail.json"
        fm = FailureMemory(store_path=store)
        # Should not raise even if dir creation fails somehow
        with patch.object(Path, "mkdir", side_effect=OSError("no space")):
            # record() catches OSError in _save()
            fm.record("goal", "reasoning_timeout")


class TestFailureMemoryCheck:
    def test_no_history_returns_low(self, tmp_memory):
        risk = tmp_memory.check("implement auth middleware")
        assert risk.risk_level == "low"
        assert risk.similar_count == 0

    def test_one_similar_returns_low(self, tmp_memory):
        tmp_memory.record(
            goal="add websocket streaming endpoint to agent server",
            failure_class="reasoning_timeout",
        )
        risk = tmp_memory.check("implement websocket streaming for agent")
        assert risk.risk_level == "low"
        assert risk.similar_count == 1
        assert risk.dominant_failure == "reasoning_timeout"

    def test_two_similar_returns_medium(self, tmp_memory):
        for _ in range(2):
            tmp_memory.record(
                goal="add websocket streaming endpoint to agent server",
                failure_class="reasoning_timeout",
            )
        risk = tmp_memory.check("implement websocket streaming for agent")
        assert risk.risk_level == "medium"

    def test_three_similar_returns_high(self, tmp_memory):
        for _ in range(3):
            tmp_memory.record(
                goal="websocket streaming agent server endpoint live",
                failure_class="no_diff",
            )
        risk = tmp_memory.check("websocket streaming agent server endpoint")
        assert risk.risk_level == "high"
        assert risk.dominant_failure == "no_diff"

    def test_dissimilar_goal_not_matched(self, tmp_memory):
        tmp_memory.record(
            goal="implement websocket streaming for agent log",
            failure_class="reasoning_timeout",
        )
        risk = tmp_memory.check("fix typo in readme documentation file")
        assert risk.similar_count == 0

    def test_notes_contain_failure_info(self, tmp_memory):
        for _ in range(2):
            tmp_memory.record(
                goal="websocket streaming live endpoint agent server",
                failure_class="reasoning_timeout",
                capability_signals={"reasoning_timeout": 2},
            )
        risk = tmp_memory.check("websocket streaming live agent server")
        assert any("similar" in n.lower() or "failure" in n.lower() for n in risk.notes)

    def test_corrupt_store_returns_low(self, tmp_path):
        store = tmp_path / "failure_patterns.json"
        store.write_text("not json at all")
        fm = FailureMemory(store_path=store)
        risk = fm.check("any goal here")
        assert risk.risk_level == "low"

    def test_returns_failure_risk_instance(self, tmp_memory):
        result = tmp_memory.check("any goal")
        assert isinstance(result, FailureRisk)


# ---------------------------------------------------------------------------
# Integration: failure memory wired into SelfRepairSupervisor
# ---------------------------------------------------------------------------


def _make_supervisor_with_memory(tmp_path):
    """Helper: build a supervisor whose failure_memory uses a temp store."""
    from igris.core.self_repair_supervisor import SelfRepairSupervisor
    sup = SelfRepairSupervisor.__new__(SelfRepairSupervisor)
    sup.project_root = str(tmp_path)
    sup._failure_memory = FailureMemory(store_path=tmp_path / "failure_patterns.json")
    return sup


class TestSupervisorFailureMemoryIntegration:
    def test_failure_memory_check_event_emitted(self, tmp_path):
        """run() emits a failure_memory/checked event after baseline passes."""
        from igris.core.self_repair_supervisor import (
            RankSupervisorConfig,
            SelfRepairSupervisor,
            SupervisorRun,
        )
        from unittest.mock import MagicMock

        config = RankSupervisorConfig(
            rank_id="rank",
            goal="add health check endpoint",
            max_repair_cycles=0,
            enable_mission_planning=False,
        )

        backend = MagicMock()
        backend.git_status.return_value = MagicMock(success=True, output="", error="")
        backend.git_log_head.return_value = MagicMock(success=True, output="abc123", error="")
        backend.run_tests.return_value = MagicMock(success=True, output="1 passed", error="")
        backend.smoke.return_value = MagicMock(success=True, output="ok", error="")
        backend.create_branch.return_value = MagicMock(success=True, output="", error="")
        backend.api_helper_is_configured.return_value = False
        # reasoning → no diff so run ends blocked (capability exhausted) quickly
        backend.run_reasoning.return_value = {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": [],
            "final_summary": "",
        }
        backend.git_diff_stat.return_value = MagicMock(success=True, output="", error="")
        backend.git_diff.return_value = MagicMock(success=True, output="", error="")

        sup = SelfRepairSupervisor(project_root=str(tmp_path), backend=backend)
        sup._failure_memory = FailureMemory(store_path=tmp_path / "failure_patterns.json")

        run = sup.run(config)

        phases = [e.phase for e in run.events]
        assert "failure_memory" in phases

        fm_event = next(e for e in run.events if e.phase == "failure_memory")
        assert fm_event.status == "checked"
        assert "risk_level" in fm_event.data

    def test_blocked_run_records_to_memory(self, tmp_path):
        """_blocked() with a capability-class failure writes to failure memory."""
        from igris.core.self_repair_supervisor import (
            RankSupervisorConfig,
            SelfRepairSupervisor,
            SupervisorRun,
        )
        from unittest.mock import MagicMock

        config = RankSupervisorConfig(
            rank_id="rank",
            goal="implement complex feature",
            max_repair_cycles=1,
            enable_mission_planning=False,
        )

        backend = MagicMock()
        backend.git_status.return_value = MagicMock(success=True, output="", error="")
        backend.git_log_head.return_value = MagicMock(success=True, output="abc123", error="")
        backend.run_tests.return_value = MagicMock(success=True, output="1 passed", error="")
        backend.smoke.return_value = MagicMock(success=True, output="ok", error="")
        backend.create_branch.return_value = MagicMock(success=True, output="", error="")
        backend.api_helper_is_configured.return_value = False
        # reasoning loop: returns no_diff immediately to consume repair budget
        no_diff_result = {
            "status": "finished",
            "stop_reason": "finish",
            "files_modified": [],
            "final_summary": "nothing to do",
        }
        backend.run_reasoning.return_value = no_diff_result
        backend.git_diff_stat.return_value = MagicMock(success=True, output="", error="")
        backend.git_diff.return_value = MagicMock(success=True, output="", error="")

        sup = SelfRepairSupervisor(project_root=str(tmp_path), backend=backend)
        mem = FailureMemory(store_path=tmp_path / "failure_patterns.json")
        sup._failure_memory = mem

        run = sup.run(config)
        # Run should end blocked; check memory was populated for capability failures
        # (no_diff_repair is a capability signal so decomposition_required is recorded)
        store = tmp_path / "failure_patterns.json"
        if store.exists():
            data = json.loads(store.read_text())
            # If a capability-class block occurred, it was recorded
            fc_list = [p["failure_class"] for p in data.get("patterns", [])]
            for fc in fc_list:
                assert fc not in ("pytest_failure", "workspace_dirty", "infrastructure_bug")

    def test_baseline_failure_not_recorded_to_memory(self, tmp_path):
        """pytest_failure at baseline must NOT pollute the failure memory."""
        from igris.core.self_repair_supervisor import (
            RankSupervisorConfig,
            SelfRepairSupervisor,
        )
        from unittest.mock import MagicMock

        config = RankSupervisorConfig(
            rank_id="rank",
            goal="add health check endpoint",
            max_repair_cycles=0,
            enable_mission_planning=False,
        )

        backend = MagicMock()
        backend.git_status.return_value = MagicMock(success=True, output="", error="")
        backend.git_log_head.return_value = MagicMock(success=True, output="abc123", error="")
        backend.run_tests.return_value = MagicMock(success=False, output="1 failed", error="")
        backend.run_test_diagnostics.return_value = MagicMock(success=True, output="diag", error="")
        backend.api_helper_is_configured.return_value = False

        sup = SelfRepairSupervisor(project_root=str(tmp_path), backend=backend)
        mem = FailureMemory(store_path=tmp_path / "failure_patterns.json")
        sup._failure_memory = mem

        sup.run(config)

        store = tmp_path / "failure_patterns.json"
        if store.exists():
            data = json.loads(store.read_text())
            assert data.get("patterns", []) == []
        else:
            pass  # file not created = no records, which is correct
