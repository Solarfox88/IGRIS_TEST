"""Tests for Epic #1078 — Clean decomposition quality.

Validates deduplication check (title + goal-hash), AC validation/generation,
max sub-issue count enforcement, and title hygiene normalization.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, call
from igris.core.self_repair_supervisor import SelfRepairSupervisor, SupervisorRun, RankSupervisorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_supervisor(project_root="/tmp"):
    backend = MagicMock()
    backend.create_issue.return_value = MagicMock(
        success=True, output="https://github.com/Solarfox88/IGRIS_GPT/issues/999", error=""
    )
    backend.run_reasoning.return_value = {
        "status": "finished", "final_summary": "", "orchestrator_used": False,
        "stop_reason": "finish", "estimated_cost": 0.0,
    }
    return SelfRepairSupervisor(project_root=project_root, backend=backend)


def _make_run():
    return SupervisorRun(run_id="test-run-1078", rank_id="rank-test")


def _make_config():
    return RankSupervisorConfig(rank_id="rank-test", goal="Fix issue #123: implement memory tree")


_UNSET = object()  # sentinel for "not provided" vs explicitly []


def _sub(title="MemoryTree: implement chunk contracts", goal="Implement chunk contract validation",
         criteria=_UNSET, tests=_UNSET, scopes=_UNSET, deps=None, risk="medium"):
    return {
        "title": title,
        "goal": goal,
        "acceptance_criteria": (
            ["All chunk contracts validated", "pytest passes", "No regressions"]
            if criteria is _UNSET else criteria
        ),
        "tests": (["tests/test_memory_tree.py"] if tests is _UNSET else tests),
        "allowed_file_scopes": (["igris/core/memory_tree.py"] if scopes is _UNSET else scopes),
        "dependencies": deps or [],
        "risk_level": risk,
    }


def _run_auto_create(sup, run, config, sub_missions):
    """Call _auto_create_subissues with patched subprocess."""
    decomposition = {
        "sub_missions": sub_missions,
        "generated_by": "test",
        "why_too_large": "test reason",
        "first_sub_mission": sub_missions[0]["title"] if sub_missions else "",
    }

    # Patch subprocess.run to avoid real gh calls
    mock_status = MagicMock(returncode=1, stdout="[]")  # no existing issues
    mock_labels = MagicMock(returncode=1, stdout="{}")

    with patch("subprocess.run", return_value=mock_status):
        return sup._auto_create_subissues(run, config, decomposition, "test_signal")


# ---------------------------------------------------------------------------
# Test: Max sub-issue count enforcement
# ---------------------------------------------------------------------------

class TestMaxSubissueCount:
    """Max sub-issue cap prevents noisy decompositions."""

    def test_cap_applied_when_exceeded(self):
        """When more than 12 sub-missions, only first 12 are created."""
        sup = _make_supervisor()
        run = _make_run()
        config = _make_config()

        sub_missions = [_sub(title=f"Task: step {i}", goal=f"Implement step {i} with unique work") for i in range(20)]

        with patch.dict(os.environ, {"IGRIS_MAX_SUBISSUES_PER_DECOMPOSITION": "12"}):
            _run_auto_create(sup, run, config, sub_missions)

        capped_events = [e for e in run.events if e.phase == "subissue_creation" and e.status == "capped"]
        assert len(capped_events) == 1
        assert capped_events[0].data["original_count"] == 20
        assert capped_events[0].data["cap"] == 12

        # Only 12 create_issue calls
        assert sup.backend.create_issue.call_count <= 12

    def test_no_cap_when_within_limit(self):
        """When sub-missions <= cap, no capping event is emitted."""
        sup = _make_supervisor()
        run = _make_run()
        config = _make_config()

        sub_missions = [_sub(title=f"Task: step {i}", goal=f"Implement step {i} precisely") for i in range(5)]

        with patch.dict(os.environ, {"IGRIS_MAX_SUBISSUES_PER_DECOMPOSITION": "12"}):
            _run_auto_create(sup, run, config, sub_missions)

        capped_events = [e for e in run.events if e.phase == "subissue_creation" and e.status == "capped"]
        assert len(capped_events) == 0

    def test_cap_configurable_via_env(self):
        """IGRIS_MAX_SUBISSUES_PER_DECOMPOSITION controls the cap."""
        sup = _make_supervisor()
        run = _make_run()
        config = _make_config()

        sub_missions = [_sub(title=f"Task: step {i}", goal=f"Implement step {i} for real") for i in range(6)]

        with patch.dict(os.environ, {"IGRIS_MAX_SUBISSUES_PER_DECOMPOSITION": "3"}):
            _run_auto_create(sup, run, config, sub_missions)

        capped_events = [e for e in run.events if e.phase == "subissue_creation" and e.status == "capped"]
        assert len(capped_events) == 1
        assert capped_events[0].data["cap"] == 3


# ---------------------------------------------------------------------------
# Test: AC validation — generate when missing or vague
# ---------------------------------------------------------------------------

class TestAcValidation:
    """Acceptance criteria are generated when missing or vague."""

    def test_ac_generated_when_empty(self):
        """When criteria is [], auto-generation fires."""
        sup = _make_supervisor()
        run = _make_run()
        config = _make_config()

        sub_missions = [_sub(criteria=[])]
        _run_auto_create(sup, run, config, sub_missions)

        ac_events = [e for e in run.events if e.phase == "subissue_ac_generated"]
        assert len(ac_events) == 1
        assert ac_events[0].data["ac_count"] >= 3

    def test_ac_generated_when_vague(self):
        """When criteria contains only '_not specified_', generation fires."""
        sup = _make_supervisor()
        run = _make_run()
        config = _make_config()

        sub_missions = [_sub(criteria=["_not specified_", "TBD", "N/A"])]
        _run_auto_create(sup, run, config, sub_missions)

        ac_events = [e for e in run.events if e.phase == "subissue_ac_generated"]
        assert len(ac_events) == 1

    def test_ac_not_generated_when_sufficient(self):
        """When 3+ valid criteria exist, no auto-generation event fires."""
        sup = _make_supervisor()
        run = _make_run()
        config = _make_config()

        sub_missions = [_sub(criteria=[
            "All chunk contracts validated with schemas",
            "pytest passes with zero failures",
            "No regressions in existing memory tests",
        ])]
        _run_auto_create(sup, run, config, sub_missions)

        ac_events = [e for e in run.events if e.phase == "subissue_ac_generated"]
        assert len(ac_events) == 0

    def test_generated_acs_reference_tests(self):
        """Generated ACs include test target reference when tests are present."""
        sup = _make_supervisor()
        run = _make_run()
        config = _make_config()

        sub_missions = [_sub(criteria=[], tests=["tests/test_foo.py"])]
        _run_auto_create(sup, run, config, sub_missions)

        # Check that the created issue body contains the test reference
        body = sup.backend.create_issue.call_args[0][1]
        assert "test_foo.py" in body or "tests/test_foo.py" in body

    def test_generated_acs_include_no_regression_criterion(self):
        """Generated ACs always include a no-regression criterion."""
        sup = _make_supervisor()
        run = _make_run()
        config = _make_config()

        sub_missions = [_sub(criteria=[])]
        _run_auto_create(sup, run, config, sub_missions)

        body = sup.backend.create_issue.call_args[0][1]
        # No-regression criterion should appear in the issue body
        assert "no regression" in body.lower() or "no new" in body.lower() or "pytest passes" in body.lower()


# ---------------------------------------------------------------------------
# Test: Title hygiene normalization
# ---------------------------------------------------------------------------

class TestTitleHygiene:
    """Vague and auto-generated titles are normalized."""

    def _get_events_of(self, run, phase):
        return [e for e in run.events if e.phase == phase]

    def test_vague_title_normalized(self):
        """Title starting with 'sub-task ' is normalized."""
        sup = _make_supervisor()
        run = _make_run()
        config = _make_config()

        sub_missions = [_sub(title="sub-task 3", goal="Implement something real here")]
        _run_auto_create(sup, run, config, sub_missions)

        hygiene_events = self._get_events_of(run, "subissue_title_hygiene")
        assert len(hygiene_events) == 1
        assert hygiene_events[0].data["original_title"] == "sub-task 3"
        assert "normalized_title" in hygiene_events[0].data

    def test_generic_github_title_normalized(self):
        """'Implement github issue #...' title is normalized."""
        sup = _make_supervisor()
        run = _make_run()
        config = _make_config()

        sub_missions = [_sub(title="Implement github issue #456 into codebase", goal="Real goal here")]
        _run_auto_create(sup, run, config, sub_missions)

        hygiene_events = self._get_events_of(run, "subissue_title_hygiene")
        assert len(hygiene_events) == 1

    def test_glob_title_normalized(self):
        """Title containing 'igris/**' is normalized."""
        sup = _make_supervisor()
        run = _make_run()
        config = _make_config()

        sub_missions = [_sub(title="igris/** refactoring", goal="Real refactoring goal here")]
        _run_auto_create(sup, run, config, sub_missions)

        hygiene_events = self._get_events_of(run, "subissue_title_hygiene")
        assert len(hygiene_events) == 1

    def test_good_title_not_normalized(self):
        """A properly formatted title is not touched."""
        sup = _make_supervisor()
        run = _make_run()
        config = _make_config()

        sub_missions = [_sub(title="MemoryTree: implement chunk contracts")]
        _run_auto_create(sup, run, config, sub_missions)

        hygiene_events = self._get_events_of(run, "subissue_title_hygiene")
        assert len(hygiene_events) == 0


# ---------------------------------------------------------------------------
# Test: Goal-hash deduplication
# ---------------------------------------------------------------------------

class TestGoalHashDedup:
    """Sub-missions with identical goal text are deduplicated."""

    def test_duplicate_goal_skipped(self):
        """Two sub-missions with identical goal text → second is skipped."""
        sup = _make_supervisor()
        run = _make_run()
        config = _make_config()

        shared_goal = "Implement chunk contract validation in memory tree module"
        sub_missions = [
            _sub(title="MemoryTree: chunk validation", goal=shared_goal),
            _sub(title="MemoryTree: chunk validation part 2", goal=shared_goal),  # same goal
        ]
        _run_auto_create(sup, run, config, sub_missions)

        # Second sub-mission should be skipped due to goal-hash dedup
        dedup_events = [
            e for e in run.events
            if e.phase == "subissue_dedup" and e.status == "skipped"
            and e.data.get("reason") == "dedup:goal_hash_match"
        ]
        assert len(dedup_events) >= 1
        # Only one create_issue call (not two)
        assert sup.backend.create_issue.call_count == 1

    def test_different_goals_not_deduplicated(self):
        """Two sub-missions with different goals are both created."""
        sup = _make_supervisor()
        run = _make_run()
        config = _make_config()

        sub_missions = [
            _sub(title="Task A: implement foo", goal="Implement foo feature in the foo module"),
            _sub(title="Task B: implement bar", goal="Implement bar feature in the bar module"),
        ]
        _run_auto_create(sup, run, config, sub_missions)

        dedup_events = [
            e for e in run.events
            if e.phase == "subissue_dedup" and e.data.get("reason") == "dedup:goal_hash_match"
        ]
        assert len(dedup_events) == 0
        assert sup.backend.create_issue.call_count == 2
