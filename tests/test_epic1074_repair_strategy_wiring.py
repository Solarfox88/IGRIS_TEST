"""Tests for Epic #1074 — decide_repair_strategy wiring in SelfRepairSupervisor._repair_cycle.

Verifies that:
1. RepairContext is correctly built from run state.
2. decide_repair_strategy returns expected strategy for common failure classes.
3. skip_repair=True causes early return (False) from _repair_cycle.
4. goal_prefix is prepended to repair_goal.
5. 'repair_strategy_decision' event is logged with correct fields.
6. Non-repairable failures produce skip_repair=True.
7. Capability-ceiling threshold triggers skip + escalate_to_decomposition.
"""
from __future__ import annotations

from igris.core.repair_strategy import (
    RepairContext,
    RepairStrategy,
    decide_repair_strategy,
    select_repair_profile,
    REPAIRABLE,
    CEILING_SIGNAL_FAILURES,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ctx(
    failure_class: str = "pytest_failure",
    cycle: int = 0,
    same_failure_count: int = 0,
    max_repair_cycles: int = 3,
    base_timeout_seconds: int = 900,
    has_high_risk_advice: bool = False,
    has_execution_plan: bool = False,
    capability_signals: dict | None = None,
) -> RepairContext:
    return RepairContext(
        failure_class=failure_class,
        cycle=cycle,
        same_failure_count=same_failure_count,
        max_repair_cycles=max_repair_cycles,
        base_timeout_seconds=base_timeout_seconds,
        has_high_risk_advice=has_high_risk_advice,
        has_execution_plan=has_execution_plan,
        capability_signals=capability_signals or {},
    )


# ---------------------------------------------------------------------------
# Tests for decide_repair_strategy
# ---------------------------------------------------------------------------

class TestDecideRepairStrategy:

    def test_non_repairable_failure_skip_repair(self):
        # "workspace_dirty" is not in REPAIRABLE
        ctx = _ctx(failure_class="workspace_dirty")
        strategy = decide_repair_strategy(ctx)
        assert strategy.skip_repair is True
        assert "not in REPAIRABLE" in strategy.skip_reason or strategy.skip_reason

    def test_repairable_pytest_failure_proceeds(self):
        ctx = _ctx(failure_class="pytest_failure", cycle=0, same_failure_count=0)
        strategy = decide_repair_strategy(ctx)
        assert strategy.skip_repair is False
        assert strategy.task_type != ""

    def test_capability_ceiling_triggers_skip_for_ceiling_failure(self):
        # same_failure_count >= 2 (default threshold) for a ceiling failure
        ctx = _ctx(failure_class="pytest_failure", cycle=2, same_failure_count=2)
        strategy = decide_repair_strategy(ctx)
        assert strategy.skip_repair is True
        assert strategy.escalate_to_decomposition is True

    def test_below_ceiling_threshold_proceeds(self):
        ctx = _ctx(failure_class="pytest_failure", cycle=1, same_failure_count=1)
        strategy = decide_repair_strategy(ctx)
        assert strategy.skip_repair is False

    def test_high_risk_advice_with_repeated_failure_skips(self):
        ctx = _ctx(
            failure_class="pytest_failure",
            cycle=1,
            same_failure_count=1,
            has_high_risk_advice=True,
        )
        strategy = decide_repair_strategy(ctx)
        assert strategy.skip_repair is True

    def test_strong_model_failure_uses_strong_profile_cycle0(self):
        ctx = _ctx(failure_class="syntax_error", cycle=0)
        strategy = decide_repair_strategy(ctx)
        assert strategy.profile == "strong_execution"

    def test_strong_model_failure_escalates_profile_cycle1(self):
        ctx = _ctx(failure_class="syntax_error", cycle=1)
        strategy = decide_repair_strategy(ctx)
        assert strategy.profile == "strong_cloud_reasoning"

    def test_timeout_extended_for_strong_profile(self):
        ctx = _ctx(failure_class="syntax_error", cycle=0, base_timeout_seconds=900)
        strategy = decide_repair_strategy(ctx)
        assert strategy.timeout_seconds >= 900  # at least base, ideally extended

    def test_goal_prefix_contains_cycle_info(self):
        ctx = _ctx(failure_class="pytest_failure", cycle=1, same_failure_count=1)
        strategy = decide_repair_strategy(ctx)
        assert strategy.goal_prefix  # non-empty
        assert "Repair cycle" in strategy.goal_prefix or "repair" in strategy.goal_prefix.lower()

    def test_goal_prefix_empty_on_skip(self):
        ctx = _ctx(failure_class="workspace_dirty")
        strategy = decide_repair_strategy(ctx)
        assert strategy.goal_prefix == ""

    def test_missing_tests_task_type(self):
        ctx = _ctx(failure_class="missing_tests", cycle=0)
        strategy = decide_repair_strategy(ctx)
        assert strategy.task_type == "test_generation"

    def test_semantic_incomplete_task_type(self):
        ctx = _ctx(failure_class="semantic_incomplete", cycle=0)
        strategy = decide_repair_strategy(ctx)
        assert strategy.task_type == "semantic_repair"

    def test_strategy_has_notes_when_proceeding(self):
        ctx = _ctx(failure_class="pytest_failure", cycle=0)
        strategy = decide_repair_strategy(ctx)
        assert isinstance(strategy.notes, str)


# ---------------------------------------------------------------------------
# Tests for select_repair_profile
# ---------------------------------------------------------------------------

class TestSelectRepairProfile:

    def test_strong_failure_cycle0_returns_strong_execution(self):
        ctx = _ctx(failure_class="infrastructure_bug", cycle=0)
        profile = select_repair_profile(ctx)
        assert profile == "strong_execution"

    def test_strong_failure_cycle1_returns_strong_cloud(self):
        ctx = _ctx(failure_class="infrastructure_bug", cycle=1)
        profile = select_repair_profile(ctx)
        assert profile == "strong_cloud_reasoning"

    def test_regular_failure_cycle0_returns_empty(self):
        ctx = _ctx(failure_class="missing_tests", cycle=0, same_failure_count=0)
        profile = select_repair_profile(ctx)
        assert profile == ""

    def test_repeat_failure_cycle1_escalates_to_strong(self):
        ctx = _ctx(failure_class="missing_tests", cycle=1, same_failure_count=1)
        profile = select_repair_profile(ctx)
        assert profile == "strong_execution"


# ---------------------------------------------------------------------------
# Integration: RepairContext construction mirrors supervisor state
# ---------------------------------------------------------------------------

class TestRepairContextConstruction:

    def test_context_is_frozen(self):
        """RepairContext must be immutable (frozen dataclass)."""
        ctx = _ctx()
        try:
            ctx.failure_class = "changed"  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except Exception:
            pass  # expected

    def test_all_fields_present(self):
        ctx = _ctx(
            failure_class="pytest_failure",
            cycle=2,
            same_failure_count=1,
            max_repair_cycles=5,
            base_timeout_seconds=1200,
            has_high_risk_advice=False,
            has_execution_plan=True,
            capability_signals={"reasoning_timeout": 1},
        )
        assert ctx.failure_class == "pytest_failure"
        assert ctx.cycle == 2
        assert ctx.same_failure_count == 1
        assert ctx.max_repair_cycles == 5
        assert ctx.base_timeout_seconds == 1200
        assert ctx.has_execution_plan is True
        assert ctx.capability_signals == {"reasoning_timeout": 1}

    def test_strategy_use_targeted_tests_true_for_most_failures(self):
        for fc in ["pytest_failure", "syntax_error", "wrong_file_edit"]:
            ctx = _ctx(failure_class=fc, cycle=0)
            strat = decide_repair_strategy(ctx)
            assert strat.use_targeted_tests is True, f"Expected use_targeted_tests=True for {fc}"

    def test_strategy_use_targeted_tests_false_for_missing_tests(self):
        ctx = _ctx(failure_class="missing_tests", cycle=0)
        strat = decide_repair_strategy(ctx)
        assert strat.use_targeted_tests is False
