"""Tests for #871 — shadow monitoring calibrated taxonomy.

Verifies:
- classify_mismatch_calibrated() returns correct class for every goal_class
- aggregate_shadow_cycles_calibrated() produces calibrated fields
- legacy fallback preserved for unknown goal_classes
- agreement cycles return empty string from classify_mismatch_calibrated
- constants are correctly defined
- calibrated classes are stable strings (no typos)
"""
from __future__ import annotations

import pytest

from igris.agent.mission.shadow_monitoring import (
    MISMATCH_CLASS_AMBIGUOUS_CONTEXT,
    MISMATCH_CLASS_LEGACY,
    MISMATCH_CLASS_SCOPE_MISMATCH,
    _GOAL_CLASSES_AMBIGUOUS_CONTEXT,
    _GOAL_CLASSES_SCOPE_MISMATCH,
    aggregate_shadow_cycles_calibrated,
    classify_mismatch_calibrated,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCOPE_MISMATCH_GOAL_CLASSES = [
    "policy_check",
    "risk_assessment",
    "loop_coherence",
    "planning",
    "test_coverage",
    "completion_boundary",
    "goal_decomposition",
    "git_safety",
    "verification",
    "memory_saturation",
    "regression_detection",
    "dependency_check",
    "simple_verification",
    "multi_step_complex",
]

AMBIGUOUS_CONTEXT_GOAL_CLASSES = [
    "ambiguous_goal",
    "empty_context",
    "conflicting_signals",
]


def _disagreeing_cycle(goal_class: str) -> dict:
    return {
        "cycle_id": f"c_{goal_class}",
        "goal_class": goal_class,
        "mission_brain_decision": "partial",
        "current_loop_decision": "failed",
        "agreement": False,
        "mismatch_class": MISMATCH_CLASS_LEGACY,
        "risk_introduced_candidate": False,
        "potential_false_completed": False,
        "potential_critical_false_completed": False,
    }


def _agreeing_cycle(goal_class: str) -> dict:
    return {
        "cycle_id": f"ca_{goal_class}",
        "goal_class": goal_class,
        "mission_brain_decision": "failed",
        "current_loop_decision": "failed",
        "agreement": True,
        "mismatch_class": "",
        "risk_introduced_candidate": False,
        "potential_false_completed": False,
        "potential_critical_false_completed": False,
    }


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_scope_mismatch_class_string(self):
        assert MISMATCH_CLASS_SCOPE_MISMATCH == "scope_mismatch_goal_vs_run_assessment"

    def test_ambiguous_context_class_string(self):
        assert MISMATCH_CLASS_AMBIGUOUS_CONTEXT == "expected_divergence_ambiguous_context"

    def test_legacy_class_string(self):
        assert MISMATCH_CLASS_LEGACY == "safe_more_optimistic_mission_brain"

    def test_scope_mismatch_goal_classes_is_frozenset(self):
        assert isinstance(_GOAL_CLASSES_SCOPE_MISMATCH, frozenset)

    def test_ambiguous_context_goal_classes_is_frozenset(self):
        assert isinstance(_GOAL_CLASSES_AMBIGUOUS_CONTEXT, frozenset)

    def test_scope_mismatch_has_expected_count(self):
        assert len(_GOAL_CLASSES_SCOPE_MISMATCH) == 14

    def test_ambiguous_context_has_expected_count(self):
        assert len(_GOAL_CLASSES_AMBIGUOUS_CONTEXT) == 3

    def test_no_overlap_between_goal_class_sets(self):
        overlap = _GOAL_CLASSES_SCOPE_MISMATCH & _GOAL_CLASSES_AMBIGUOUS_CONTEXT
        assert overlap == frozenset()


# ---------------------------------------------------------------------------
# Test classify_mismatch_calibrated — scope mismatch classes
# ---------------------------------------------------------------------------

class TestClassifyMismatchCalibratedScopeMismatch:
    @pytest.mark.parametrize("goal_class", SCOPE_MISMATCH_GOAL_CLASSES)
    def test_scope_mismatch_goal_class(self, goal_class: str):
        cycle = _disagreeing_cycle(goal_class)
        result = classify_mismatch_calibrated(cycle)
        assert result == MISMATCH_CLASS_SCOPE_MISMATCH, (
            f"Expected scope_mismatch for goal_class={goal_class!r}, got {result!r}"
        )


# ---------------------------------------------------------------------------
# Test classify_mismatch_calibrated — ambiguous context classes
# ---------------------------------------------------------------------------

class TestClassifyMismatchCalibratedAmbiguousContext:
    @pytest.mark.parametrize("goal_class", AMBIGUOUS_CONTEXT_GOAL_CLASSES)
    def test_ambiguous_context_goal_class(self, goal_class: str):
        cycle = _disagreeing_cycle(goal_class)
        result = classify_mismatch_calibrated(cycle)
        assert result == MISMATCH_CLASS_AMBIGUOUS_CONTEXT, (
            f"Expected ambiguous_context for goal_class={goal_class!r}, got {result!r}"
        )


# ---------------------------------------------------------------------------
# Test classify_mismatch_calibrated — edge cases
# ---------------------------------------------------------------------------

class TestClassifyMismatchCalibratedEdgeCases:
    def test_agreeing_cycle_returns_empty_string(self):
        cycle = _agreeing_cycle("planning")
        result = classify_mismatch_calibrated(cycle)
        assert result == ""

    def test_agreement_true_even_with_disagreeing_goal_class(self):
        cycle = _disagreeing_cycle("policy_check")
        cycle["agreement"] = True
        result = classify_mismatch_calibrated(cycle)
        assert result == ""

    def test_unknown_goal_class_falls_back_to_legacy(self):
        cycle = _disagreeing_cycle("totally_unknown_class_xyz")
        result = classify_mismatch_calibrated(cycle)
        assert result == MISMATCH_CLASS_LEGACY

    def test_empty_goal_class_falls_back_to_legacy(self):
        cycle = _disagreeing_cycle("")
        result = classify_mismatch_calibrated(cycle)
        assert result == MISMATCH_CLASS_LEGACY

    def test_none_goal_class_falls_back_to_legacy(self):
        cycle = {
            "cycle_id": "c_none",
            "goal_class": None,
            "mission_brain_decision": "partial",
            "current_loop_decision": "failed",
            "agreement": False,
        }
        result = classify_mismatch_calibrated(cycle)
        assert result == MISMATCH_CLASS_LEGACY

    def test_missing_goal_class_falls_back_to_legacy(self):
        cycle = {
            "cycle_id": "c_missing",
            "mission_brain_decision": "partial",
            "current_loop_decision": "failed",
            "agreement": False,
        }
        result = classify_mismatch_calibrated(cycle)
        assert result == MISMATCH_CLASS_LEGACY


# ---------------------------------------------------------------------------
# Test aggregate_shadow_cycles_calibrated
# ---------------------------------------------------------------------------

class TestAggregateShadowCyclesCalibrated:
    def _build_batch(self):
        """Build 20 cycles matching the #869/#870 dataset profile."""
        scope_classes = SCOPE_MISMATCH_GOAL_CLASSES  # 14 unique, use 17 total
        ambiguous_classes = AMBIGUOUS_CONTEXT_GOAL_CLASSES  # 3 total
        cycles = []
        # 17 scope mismatch cycles
        for i, gc in enumerate(scope_classes[:14]):
            cycles.append(_disagreeing_cycle(gc))
        # 3 extra scope mismatch (reuse some)
        cycles.append(_disagreeing_cycle("planning"))
        cycles.append(_disagreeing_cycle("verification"))
        cycles.append(_disagreeing_cycle("policy_check"))
        # 3 ambiguous context cycles
        for gc in ambiguous_classes:
            cycles.append(_disagreeing_cycle(gc))
        assert len(cycles) == 20
        return cycles

    def test_returns_dict(self):
        result = aggregate_shadow_cycles_calibrated(self._build_batch())
        assert isinstance(result, dict)

    def test_calibration_applied_sentinel(self):
        result = aggregate_shadow_cycles_calibrated(self._build_batch())
        assert result["calibration_applied"] is True

    def test_taxonomy_version(self):
        result = aggregate_shadow_cycles_calibrated(self._build_batch())
        assert result["taxonomy_version"] == "calibrated_v1"

    def test_calibrated_disagreement_by_class_present(self):
        result = aggregate_shadow_cycles_calibrated(self._build_batch())
        assert "calibrated_disagreement_by_class" in result

    def test_calibrated_dominant_mismatch_classes_present(self):
        result = aggregate_shadow_cycles_calibrated(self._build_batch())
        assert "calibrated_dominant_mismatch_classes" in result

    def test_scope_mismatch_count(self):
        result = aggregate_shadow_cycles_calibrated(self._build_batch())
        assert result["scope_mismatch_count"] == 17

    def test_ambiguous_context_count(self):
        result = aggregate_shadow_cycles_calibrated(self._build_batch())
        assert result["ambiguous_context_count"] == 3

    def test_legacy_unclassified_count_zero(self):
        result = aggregate_shadow_cycles_calibrated(self._build_batch())
        assert result["legacy_unclassified_count"] == 0

    def test_base_fields_preserved(self):
        result = aggregate_shadow_cycles_calibrated(self._build_batch())
        for field in (
            "total_shadow_cycles", "agreement_rate", "disagreement_rate",
            "risk_introduced_candidates", "potential_critical_false_completed",
            "rollback_path_status",
        ):
            assert field in result, f"Missing base field: {field}"

    def test_total_shadow_cycles(self):
        result = aggregate_shadow_cycles_calibrated(self._build_batch())
        assert result["total_shadow_cycles"] == 20

    def test_agreement_rate_zero_for_all_disagreeing(self):
        result = aggregate_shadow_cycles_calibrated(self._build_batch())
        assert result["agreement_rate"] == 0.0

    def test_dominant_mismatch_first_is_scope_mismatch(self):
        result = aggregate_shadow_cycles_calibrated(self._build_batch())
        dominant = result["calibrated_dominant_mismatch_classes"]
        assert len(dominant) >= 1
        assert dominant[0] == MISMATCH_CLASS_SCOPE_MISMATCH

    def test_empty_cycles(self):
        result = aggregate_shadow_cycles_calibrated([])
        assert result["total_shadow_cycles"] == 0
        assert result["scope_mismatch_count"] == 0
        assert result["ambiguous_context_count"] == 0
        assert result["calibration_applied"] is True

    def test_all_agreeing_cycles(self):
        cycles = [_agreeing_cycle("planning") for _ in range(5)]
        result = aggregate_shadow_cycles_calibrated(cycles)
        assert result["scope_mismatch_count"] == 0
        assert result["ambiguous_context_count"] == 0
        assert result["agreement_rate"] == 1.0

    def test_mixed_agreeing_and_disagreeing(self):
        cycles = (
            [_agreeing_cycle("planning")] * 5
            + [_disagreeing_cycle("policy_check")] * 5
        )
        result = aggregate_shadow_cycles_calibrated(cycles)
        assert result["scope_mismatch_count"] == 5
        assert result["ambiguous_context_count"] == 0
        assert result["agreement_rate"] == 0.5
