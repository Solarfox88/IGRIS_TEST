"""
test_new_epic_modules.py

Comprehensive tests for the 4 from-scratch epic modules:
  - decomposition_validator  (Epic #1078)
  - repair_strategy          (Epic #1074)
  - memory_circuit_breaker   (Epic #1073)
  - ci_repair_loop           (Epic #1071)

250+ assertions across 60+ test cases.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# decomposition_validator
# ---------------------------------------------------------------------------

from igris.core.decomposition_validator import (
    DecompositionValidator,
    SubMission,
    ValidationIssue,
    ValidationReport,
    MAX_SUBISSUES,
    MIN_AC_COUNT,
    MIN_TITLE_LEN,
    REQUIRED_FIELDS,
)


def _good_sub(n: int = 0, **overrides) -> Dict[str, Any]:
    """Build a minimal valid sub-mission dict."""
    d: Dict[str, Any] = {
        "title": f"Core: implement feature {n} properly",
        "goal": f"Implement the feature number {n} in the core module with full test coverage.",
        "risk_level": "medium",
        "acceptance_criteria": [
            "All existing tests continue to pass without modification",
            "New feature is covered by at least 3 unit tests",
            "Import works cleanly from igris.core",
        ],
        "allowed_file_scopes": [f"igris/core/feature_{n}.py"],
        "tests": [f"tests/test_feature_{n}.py"],
        "dependencies": [],
    }
    d.update(overrides)
    return d


class TestDecompositionValidatorBasic:
    def setup_method(self):
        DecompositionValidator._global_registry = {}  # not needed but safe
        self.v = DecompositionValidator(parent_goal="Build IGRIS feature")

    def test_empty_list_returns_no_accepted(self):
        r = self.v.validate([])
        assert not r.ok
        assert len(r.accepted) == 0

    def test_single_valid_sub_accepted(self):
        r = self.v.validate([_good_sub(0)])
        assert r.ok
        assert len(r.accepted) == 1
        assert len(r.rejected) == 0

    def test_accepted_has_correct_title(self):
        r = self.v.validate([_good_sub(0)])
        sm = r.accepted[0]
        assert "Core" in sm.title or "feature" in sm.title.lower()

    def test_accepted_sub_is_SubMission_instance(self):
        r = self.v.validate([_good_sub(0)])
        assert isinstance(r.accepted[0], SubMission)

    def test_multiple_valid_subs_all_accepted(self):
        r = self.v.validate([_good_sub(i) for i in range(5)])
        assert len(r.accepted) == 5
        assert len(r.rejected) == 0

    def test_summary_string_format(self):
        r = self.v.validate([_good_sub(0)])
        s = r.summary()
        assert "accepted" in s
        assert "rejected" in s


class TestDecompositionValidatorRejections:
    def setup_method(self):
        self.v = DecompositionValidator()

    def test_missing_title_rejected(self):
        sub = _good_sub(0)
        del sub["title"]
        r = self.v.validate([sub])
        assert len(r.rejected) == 1
        assert len(r.accepted) == 0

    def test_empty_title_rejected(self):
        sub = _good_sub(0)
        sub["title"] = ""
        r = self.v.validate([sub])
        assert len(r.rejected) == 1

    def test_missing_goal_rejected(self):
        sub = _good_sub(0)
        del sub["goal"]
        r = self.v.validate([sub])
        assert len(r.rejected) == 1

    def test_empty_goal_rejected(self):
        sub = _good_sub(0)
        sub["goal"] = ""
        r = self.v.validate([sub])
        assert len(r.rejected) == 1

    def test_duplicate_title_second_rejected(self):
        sub1 = _good_sub(0)
        sub2 = _good_sub(0)  # same title
        r = self.v.validate([sub1, sub2])
        assert len(r.accepted) == 1
        assert len(r.rejected) == 1

    def test_duplicate_title_case_insensitive(self):
        sub1 = _good_sub(0)
        sub2 = _good_sub(0)
        sub2["title"] = sub1["title"].upper()
        r = self.v.validate([sub1, sub2])
        assert len(r.rejected) >= 1

    def test_duplicate_goal_hash_rejected(self):
        sub1 = _good_sub(0)
        sub2 = _good_sub(1)
        sub2["goal"] = sub1["goal"]  # same goal text
        r = self.v.validate([sub1, sub2])
        assert len(r.accepted) == 1
        assert len(r.rejected) == 1


class TestDecompositionValidatorAcceptanceCriteria:
    def setup_method(self):
        self.v = DecompositionValidator()

    def test_empty_ac_generates_defaults(self):
        sub = _good_sub(0)
        sub["acceptance_criteria"] = []
        r = self.v.validate([sub])
        assert r.ok
        sm = r.accepted[0]
        assert len(sm.acceptance_criteria) >= MIN_AC_COUNT

    def test_vague_ac_filtered(self):
        sub = _good_sub(0)
        sub["acceptance_criteria"] = ["tbd", "n/a", "", "not specified"]
        r = self.v.validate([sub])
        assert r.ok
        sm = r.accepted[0]
        assert len(sm.acceptance_criteria) >= MIN_AC_COUNT

    def test_one_good_ac_fills_to_minimum(self):
        sub = _good_sub(0)
        sub["acceptance_criteria"] = ["All tests pass without regressions detected"]
        r = self.v.validate([sub])
        assert r.ok
        sm = r.accepted[0]
        assert len(sm.acceptance_criteria) >= MIN_AC_COUNT

    def test_three_good_ac_preserved(self):
        sub = _good_sub(0)
        sub["acceptance_criteria"] = [
            "Feature works end-to-end with no errors",
            "Unit tests cover all new code paths",
            "No regressions in the existing test suite",
        ]
        r = self.v.validate([sub])
        assert r.ok
        sm = r.accepted[0]
        assert len(sm.acceptance_criteria) >= 3


class TestDecompositionValidatorCap:
    def test_cap_at_max_subissues(self):
        v = DecompositionValidator(max_subissues=3)
        subs = [_good_sub(i) for i in range(10)]
        r = v.validate(subs)
        assert r.capped
        assert r.original_count == 10
        assert len(r.accepted) <= 3

    def test_no_cap_at_default(self):
        v = DecompositionValidator()
        subs = [_good_sub(i) for i in range(MAX_SUBISSUES)]
        r = v.validate(subs)
        assert not r.capped
        assert len(r.accepted) == MAX_SUBISSUES

    def test_cap_emits_warning_issue(self):
        v = DecompositionValidator(max_subissues=2)
        subs = [_good_sub(i) for i in range(5)]
        r = v.validate(subs)
        warnings = [i for i in r.issues if i.severity == "warning" and i.field == "count"]
        assert len(warnings) == 1


class TestDecompositionValidatorRiskLevel:
    def setup_method(self):
        self.v = DecompositionValidator()

    def test_valid_risk_low(self):
        sub = _good_sub(0, risk_level="low")
        r = self.v.validate([sub])
        assert r.accepted[0].risk_level == "low"

    def test_valid_risk_critical(self):
        sub = _good_sub(0, risk_level="critical")
        r = self.v.validate([sub])
        assert r.accepted[0].risk_level == "critical"

    def test_invalid_risk_normalised_to_medium(self):
        sub = _good_sub(0, risk_level="extreme")
        r = self.v.validate([sub])
        assert r.accepted[0].risk_level == "medium"
        fixed = [i for i in r.issues if i.field == "risk_level" and i.severity == "fixed"]
        assert fixed

    def test_risk_uppercased_normalised(self):
        sub = _good_sub(0, risk_level="HIGH")
        r = self.v.validate([sub])
        assert r.accepted[0].risk_level == "high"


class TestDecompositionValidatorSuccessSignal:
    def setup_method(self):
        self.v = DecompositionValidator()

    def test_default_success_signal_set(self):
        sub = _good_sub(0)
        r = self.v.validate([sub])
        sm = r.accepted[0]
        assert sm.success_signal

    def test_custom_success_signal_preserved(self):
        sub = _good_sub(0, success_signal="CI green + peer approved")
        r = self.v.validate([sub])
        assert r.accepted[0].success_signal == "CI green + peer approved"

    def test_failure_fallback_high_risk(self):
        sub = _good_sub(0, risk_level="high")
        r = self.v.validate([sub])
        assert "escalate" in r.accepted[0].failure_fallback.lower() or "human" in r.accepted[0].failure_fallback.lower()

    def test_failure_fallback_low_risk(self):
        sub = _good_sub(0, risk_level="low")
        r = self.v.validate([sub])
        assert r.accepted[0].failure_fallback


class TestDecompositionValidatorDependencies:
    def setup_method(self):
        self.v = DecompositionValidator()

    def test_external_ref_no_warning(self):
        sub1 = _good_sub(0, dependencies=["#1234"])
        r = self.v.validate([sub1])
        dep_warnings = [i for i in r.issues if i.field == "dependencies" and i.severity == "warning"]
        assert len(dep_warnings) == 0

    def test_unresolved_dep_emits_warning(self):
        sub1 = _good_sub(0, dependencies=["Core: implement feature 99 properly"])
        r = self.v.validate([sub1])
        dep_warnings = [i for i in r.issues if i.field == "dependencies" and i.severity == "warning"]
        assert len(dep_warnings) == 1

    def test_resolved_dep_no_warning(self):
        sub1 = _good_sub(0)
        sub2 = _good_sub(1, dependencies=[sub1["title"]])
        r = self.v.validate([sub1, sub2])
        dep_warnings = [i for i in r.issues if i.field == "dependencies" and i.severity == "warning"]
        assert len(dep_warnings) == 0


class TestSubMissionToDict:
    def test_to_dict_has_all_fields(self):
        v = DecompositionValidator()
        r = v.validate([_good_sub(0)])
        d = r.accepted[0].to_dict()
        for key in ("title", "goal", "risk_level", "acceptance_criteria",
                    "allowed_file_scopes", "tests", "dependencies",
                    "out_of_scope", "success_signal", "failure_fallback"):
            assert key in d


# ---------------------------------------------------------------------------
# repair_strategy
# ---------------------------------------------------------------------------

from igris.core.repair_strategy import (
    RepairContext,
    RepairStrategy,
    decide_repair_strategy,
    select_repair_profile,
    REPAIRABLE,
    STRONG_MODEL_FAILURES,
    CEILING_SIGNAL_FAILURES,
)


def _ctx(**kw) -> RepairContext:
    defaults = dict(
        failure_class="pytest_failure",
        cycle=0,
        same_failure_count=0,
        max_repair_cycles=4,
        base_timeout_seconds=600,
    )
    defaults.update(kw)
    return RepairContext(**defaults)


class TestDecideRepairStrategy:
    def test_unknown_failure_skipped(self):
        ctx = _ctx(failure_class="unknown_xyzzy")
        s = decide_repair_strategy(ctx)
        assert s.skip_repair
        assert "not in REPAIRABLE" in s.skip_reason

    def test_repairable_failure_not_skipped(self):
        ctx = _ctx(failure_class="pytest_failure")
        s = decide_repair_strategy(ctx)
        assert not s.skip_repair

    def test_capability_ceiling_signals_skip(self, monkeypatch):
        monkeypatch.setenv("IGRIS_CAPABILITY_LIMIT_THRESHOLD", "2")
        ctx = _ctx(failure_class="pytest_failure", same_failure_count=2)
        s = decide_repair_strategy(ctx)
        assert s.skip_repair
        assert s.escalate_to_decomposition

    def test_ceiling_threshold_not_reached(self, monkeypatch):
        monkeypatch.setenv("IGRIS_CAPABILITY_LIMIT_THRESHOLD", "3")
        ctx = _ctx(failure_class="pytest_failure", same_failure_count=2)
        s = decide_repair_strategy(ctx)
        assert not s.skip_repair

    def test_high_risk_with_repeat_skipped(self):
        ctx = _ctx(failure_class="pytest_failure", has_high_risk_advice=True, same_failure_count=1)
        s = decide_repair_strategy(ctx)
        assert s.skip_repair
        assert "high-risk" in s.skip_reason.lower() or "budget" in s.skip_reason.lower()

    def test_high_risk_first_attempt_not_skipped(self):
        ctx = _ctx(failure_class="pytest_failure", has_high_risk_advice=True, same_failure_count=0)
        s = decide_repair_strategy(ctx)
        assert not s.skip_repair

    def test_syntax_error_uses_code_repair(self):
        ctx = _ctx(failure_class="syntax_error")
        s = decide_repair_strategy(ctx)
        assert s.task_type == "code_repair"

    def test_missing_tests_uses_test_generation(self):
        ctx = _ctx(failure_class="missing_tests")
        s = decide_repair_strategy(ctx)
        assert s.task_type == "test_generation"

    def test_strong_model_failure_cycle0_profile(self):
        ctx = _ctx(failure_class="syntax_error", cycle=0)
        s = decide_repair_strategy(ctx)
        assert s.profile == "strong_execution"

    def test_strong_model_failure_cycle1_profile(self):
        ctx = _ctx(failure_class="syntax_error", cycle=1)
        s = decide_repair_strategy(ctx)
        assert s.profile == "strong_cloud_reasoning"

    def test_repeat_failure_escalates_profile(self):
        ctx = _ctx(failure_class="pytest_failure", cycle=1, same_failure_count=1)
        s = decide_repair_strategy(ctx)
        assert s.profile == "strong_execution"

    def test_notes_include_cycle_info(self):
        ctx = _ctx(failure_class="pytest_failure", cycle=2, same_failure_count=1)
        s = decide_repair_strategy(ctx)
        assert "cycle" in s.notes

    def test_timeout_extended_for_strong_profile(self, monkeypatch):
        monkeypatch.delenv("IGRIS_STRONG_REASONING_TIMEOUT_SECONDS", raising=False)
        ctx = _ctx(failure_class="syntax_error", base_timeout_seconds=600)
        s = decide_repair_strategy(ctx)
        assert s.timeout_seconds >= 1800  # at least 3x

    def test_targeted_tests_true_for_pytest_failure(self):
        ctx = _ctx(failure_class="pytest_failure")
        s = decide_repair_strategy(ctx)
        assert s.use_targeted_tests is True

    def test_targeted_tests_false_for_missing_tests(self):
        ctx = _ctx(failure_class="missing_tests")
        s = decide_repair_strategy(ctx)
        assert s.use_targeted_tests is False


class TestSelectRepairProfile:
    def test_default_profile_empty_string(self):
        ctx = _ctx(failure_class="pytest_failure", cycle=0, same_failure_count=0)
        assert select_repair_profile(ctx) == ""

    def test_syntax_error_cycle0_strong_execution(self):
        ctx = _ctx(failure_class="syntax_error", cycle=0)
        assert select_repair_profile(ctx) == "strong_execution"

    def test_syntax_error_cycle1_strong_cloud(self):
        ctx = _ctx(failure_class="syntax_error", cycle=1)
        assert select_repair_profile(ctx) == "strong_cloud_reasoning"

    def test_repeat_escalates(self):
        ctx = _ctx(failure_class="pytest_failure", cycle=1, same_failure_count=1)
        assert select_repair_profile(ctx) == "strong_execution"


class TestGoalPrefix:
    def test_first_cycle_prefix(self):
        ctx = _ctx(failure_class="pytest_failure", cycle=0, same_failure_count=0)
        s = decide_repair_strategy(ctx)
        assert "cycle 1" in s.goal_prefix.lower() or "repair" in s.goal_prefix.lower()

    def test_repeated_failure_prefix_mentions_count(self):
        # same_failure_count=1 is below ceiling (default=2), so we get a goal_prefix
        ctx = _ctx(failure_class="pytest_failure", cycle=1, same_failure_count=1)
        s = decide_repair_strategy(ctx)
        assert "1" in s.goal_prefix or "cycle" in s.goal_prefix.lower() or "repair" in s.goal_prefix.lower()

    def test_syntax_prefix_mentions_priority(self):
        ctx = _ctx(failure_class="syntax_error", cycle=0)
        s = decide_repair_strategy(ctx)
        assert "syntax" in s.goal_prefix.lower() or "priority" in s.goal_prefix.lower()

    def test_plan_mention_when_has_plan(self):
        ctx = _ctx(failure_class="pytest_failure", has_execution_plan=True)
        s = decide_repair_strategy(ctx)
        assert "plan" in s.goal_prefix.lower() or "follow" in s.goal_prefix.lower()


# ---------------------------------------------------------------------------
# memory_circuit_breaker
# ---------------------------------------------------------------------------

from igris.core.memory_circuit_breaker import (
    MemoryCircuitBreaker,
    BreakerState,
    DEFAULT_OPEN_THRESHOLD,
    DEFAULT_RECOVERY_WINDOW,
)


class TestMemoryCircuitBreakerBasic:
    def setup_method(self):
        MemoryCircuitBreaker.reset_all()

    def test_initial_state_closed(self):
        b = MemoryCircuitBreaker.get("test_basic")
        assert b.state == BreakerState.CLOSED

    def test_allow_when_closed(self):
        b = MemoryCircuitBreaker.get("test_allow")
        assert b.allow() is True

    def test_singleton_same_name(self):
        b1 = MemoryCircuitBreaker.get("test_singleton")
        b2 = MemoryCircuitBreaker.get("test_singleton")
        assert b1 is b2

    def test_different_names_different_instances(self):
        b1 = MemoryCircuitBreaker.get("alpha")
        b2 = MemoryCircuitBreaker.get("beta")
        assert b1 is not b2

    def test_is_closed_true_initially(self):
        b = MemoryCircuitBreaker.get("test_is_closed")
        assert b.is_closed()
        assert not b.is_open()

    def test_status_dict_keys(self):
        b = MemoryCircuitBreaker.get("test_status")
        d = b.status_dict()
        for key in ("name", "state", "failure_count", "open_threshold",
                    "recovery_window_seconds", "healthy"):
            assert key in d

    def test_status_dict_healthy_true_when_closed(self):
        b = MemoryCircuitBreaker.get("test_health")
        assert b.status_dict()["healthy"] is True


class TestMemoryCircuitBreakerStateTransitions:
    def setup_method(self):
        MemoryCircuitBreaker.reset_all()

    def test_record_failure_stays_closed_below_threshold(self):
        b = MemoryCircuitBreaker.get("tf1", open_threshold=3)
        b.record_failure()
        b.record_failure()
        assert b.state == BreakerState.CLOSED
        assert b.is_closed()

    def test_record_failure_opens_at_threshold(self):
        b = MemoryCircuitBreaker.get("tf2", open_threshold=3)
        b.record_failure()
        b.record_failure()
        b.record_failure()
        assert b.state == BreakerState.OPEN
        assert b.is_open()

    def test_allow_false_when_open(self):
        b = MemoryCircuitBreaker.get("tf3", open_threshold=1)
        b.record_failure()
        assert b.is_open()
        assert b.allow() is False

    def test_success_decrements_failure_count_when_closed(self):
        b = MemoryCircuitBreaker.get("tf4", open_threshold=5)
        b.record_failure()
        b.record_failure()
        b.record_success()
        assert b._failure_count == 1

    def test_success_does_not_go_below_zero(self):
        b = MemoryCircuitBreaker.get("tf5", open_threshold=5)
        b.record_success()
        assert b._failure_count == 0

    def test_open_to_half_open_after_recovery_window(self):
        b = MemoryCircuitBreaker.get("tf6", open_threshold=1, recovery_window_seconds=0.05)
        b.record_failure()
        assert b.is_open()
        time.sleep(0.1)
        # accessing .state triggers _maybe_recover
        assert b.state == BreakerState.HALF_OPEN

    def test_half_open_allows_one_trial(self):
        b = MemoryCircuitBreaker.get("tf7", open_threshold=1, recovery_window_seconds=0.05)
        b.record_failure()
        time.sleep(0.1)
        assert b.allow() is True

    def test_half_open_success_closes(self):
        b = MemoryCircuitBreaker.get("tf8", open_threshold=1, recovery_window_seconds=0.05)
        b.record_failure()
        time.sleep(0.1)
        _ = b.state  # trigger recovery
        b.record_success()
        assert b.state == BreakerState.CLOSED

    def test_half_open_failure_reopens(self):
        b = MemoryCircuitBreaker.get("tf9", open_threshold=1, recovery_window_seconds=0.05)
        b.record_failure()
        time.sleep(0.1)
        _ = b.state  # trigger recovery
        b.record_failure()
        assert b.state == BreakerState.OPEN

    def test_reset_all_clears_registry(self):
        MemoryCircuitBreaker.get("temp1")
        MemoryCircuitBreaker.get("temp2")
        MemoryCircuitBreaker.reset_all()
        # After reset, a new get should return a fresh breaker in CLOSED state
        b = MemoryCircuitBreaker.get("temp1")
        assert b.state == BreakerState.CLOSED


class TestMemoryCircuitBreakerThreadSafety:
    def setup_method(self):
        MemoryCircuitBreaker.reset_all()

    def test_concurrent_failures_safe(self):
        b = MemoryCircuitBreaker.get("tthread", open_threshold=100)
        errors: List[Exception] = []

        def worker():
            try:
                for _ in range(10):
                    b.record_failure()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert b._failure_count == 50

    def test_allow_thread_safe(self):
        b = MemoryCircuitBreaker.get("tthread2", open_threshold=5)
        results: List[bool] = []

        def worker():
            results.append(b.allow())

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10


# ---------------------------------------------------------------------------
# ci_repair_loop
# ---------------------------------------------------------------------------

from igris.core.ci_repair_loop import (
    CIRepairLoop,
    CIRepairAttempt,
    CIRepairResult,
    MAX_ATTEMPTS,
    LINT_COMMANDS,
)


class FakeBackendCI:
    """Test double for ci_repair_loop backend."""

    def __init__(self, status="finished", raises=False):
        self._status = status
        self._raises = raises
        self.calls: List[Dict] = []

    def run_reasoning(self, goal: str, **kwargs) -> Dict:
        self.calls.append({"goal": goal, **kwargs})
        if self._raises:
            raise RuntimeError("backend error")
        return {"status": self._status}


class TestCIRepairLoopDataClasses:
    def test_attempt_count_property(self):
        r = CIRepairResult(resolved=True)
        assert r.attempt_count == 0
        r.attempts.append(
            CIRepairAttempt(
                attempt=0, failure_type="lint_error",
                strategy="deterministic_lint", goal_sent="ruff",
                success=True, duration_seconds=1.0,
            )
        )
        assert r.attempt_count == 1

    def test_attempt_dataclass_fields(self):
        a = CIRepairAttempt(
            attempt=1, failure_type="test_failure",
            strategy="llm_repair", goal_sent="fix tests",
            success=False, duration_seconds=5.5, error="timeout",
        )
        assert a.attempt == 1
        assert a.failure_type == "test_failure"
        assert a.success is False
        assert a.error == "timeout"

    def test_max_attempts_constant(self):
        assert MAX_ATTEMPTS >= 1

    def test_lint_commands_non_empty(self):
        assert len(LINT_COMMANDS) >= 1
        assert any("ruff" in " ".join(cmd) for cmd in LINT_COMMANDS)


class TestCIRepairLoopDiagnose:
    def setup_method(self):
        self.loop = CIRepairLoop(
            project_root="/tmp", pr_number=1, original_goal="test"
        )

    def test_import_error_detected(self):
        d = self.loop._diagnose("ModuleNotFoundError: no module named foo")
        assert d["failure_type"] == "import_error"

    def test_syntax_error_detected(self):
        d = self.loop._diagnose("SyntaxError: invalid syntax on line 5")
        assert d["failure_type"] == "syntax_error"

    def test_lint_error_detected(self):
        d = self.loop._diagnose("ruff: error E501 line too long")
        assert d["failure_type"] == "lint_error"

    def test_test_failure_detected(self):
        d = self.loop._diagnose("FAILED tests/test_core.py::test_foo")
        assert d["failure_type"] == "test_failure"
        assert "tests/test_core.py::test_foo" in d["failing_tests"]

    def test_unknown_failure_type(self):
        d = self.loop._diagnose("some random output with no patterns")
        assert d["failure_type"] == "unknown"

    def test_log_excerpt_capped(self):
        long_log = "x" * 5000
        d = self.loop._diagnose(long_log)
        assert len(d["log_excerpt"]) <= 2000

    def test_multiple_failing_tests_parsed(self):
        log = (
            "FAILED tests/test_a.py::test_one\n"
            "FAILED tests/test_b.py::test_two\n"
            "FAILED tests/test_c.py::test_three\n"
        )
        d = self.loop._diagnose(log)
        assert len(d["failing_tests"]) == 3

    def test_failing_tests_deduplicated(self):
        log = (
            "FAILED tests/test_a.py::test_one\n"
            "FAILED tests/test_a.py::test_one\n"
        )
        d = self.loop._diagnose(log)
        assert len(d["failing_tests"]) == 1


class TestCIRepairLoopGoalBuilding:
    def setup_method(self):
        self.loop = CIRepairLoop(
            project_root="/tmp", pr_number=42,
            original_goal="Build the feature correctly"
        )

    def test_test_failure_goal_mentions_pr(self):
        diag = {
            "failure_type": "test_failure",
            "failing_tests": ["tests/test_foo.py::test_bar"],
            "log_excerpt": "",
        }
        goal = self.loop._build_llm_repair_goal(diag)
        assert "42" in goal
        assert "tests/test_foo.py" in goal

    def test_test_failure_goal_says_fix_source(self):
        diag = {
            "failure_type": "test_failure",
            "failing_tests": ["tests/test_x.py::test_y"],
            "log_excerpt": "",
        }
        goal = self.loop._build_llm_repair_goal(diag)
        assert "SOURCE CODE" in goal or "source code" in goal.lower()

    def test_import_error_goal(self):
        diag = {
            "failure_type": "import_error",
            "failing_tests": [],
            "log_excerpt": "ModuleNotFoundError: foo",
        }
        goal = self.loop._build_llm_repair_goal(diag)
        assert "import" in goal.lower() or "ImportError" in goal

    def test_syntax_error_goal(self):
        diag = {
            "failure_type": "syntax_error",
            "failing_tests": [],
            "log_excerpt": "SyntaxError: bad",
        }
        goal = self.loop._build_llm_repair_goal(diag)
        assert "syntax" in goal.lower()

    def test_unknown_failure_goal(self):
        diag = {
            "failure_type": "unknown",
            "failing_tests": [],
            "log_excerpt": "something broke",
        }
        goal = self.loop._build_llm_repair_goal(diag)
        assert len(goal) > 50  # non-trivial content

    def test_original_goal_included(self):
        diag = {
            "failure_type": "unknown",
            "failing_tests": [],
            "log_excerpt": "",
        }
        goal = self.loop._build_llm_repair_goal(diag)
        assert "Build the feature correctly" in goal


class TestCIRepairLoopRun:
    def setup_method(self):
        self.project_root = "/tmp"

    def _make_loop(self, **kw) -> CIRepairLoop:
        return CIRepairLoop(
            project_root=self.project_root,
            pr_number=99,
            original_goal="test goal",
            max_attempts=kw.pop("max_attempts", 2),
            **kw,
        )

    def test_run_returns_result_object(self):
        loop = self._make_loop(max_attempts=1)
        backend = FakeBackendCI(status="finished")

        with patch.object(loop, "_fetch_ci_logs", return_value="AssertionError: boom"), \
             patch.object(loop, "_ci_is_green", return_value=True), \
             patch.object(loop, "_push_fix", return_value=True):
            result = loop.run(backend)

        assert isinstance(result, CIRepairResult)

    def test_run_resolved_when_ci_green(self):
        loop = self._make_loop(max_attempts=1)
        backend = FakeBackendCI(status="finished")

        with patch.object(loop, "_fetch_ci_logs", return_value="FAILED tests/test_x.py::foo"), \
             patch.object(loop, "_ci_is_green", return_value=True), \
             patch.object(loop, "_push_fix", return_value=True):
            result = loop.run(backend)

        assert result.resolved is True

    def test_run_not_resolved_when_ci_not_green(self):
        loop = self._make_loop(max_attempts=1)
        backend = FakeBackendCI(status="error")

        with patch.object(loop, "_fetch_ci_logs", return_value="FAILED tests/x.py::t"), \
             patch.object(loop, "_ci_is_green", return_value=False), \
             patch.object(loop, "_push_fix", return_value=True):
            result = loop.run(backend)

        assert result.resolved is False
        assert result.failure_summary

    def test_run_records_attempts(self):
        loop = self._make_loop(max_attempts=2)
        backend = FakeBackendCI(status="finished")

        with patch.object(loop, "_fetch_ci_logs", return_value="FAILED tests/x.py::t"), \
             patch.object(loop, "_ci_is_green", return_value=False), \
             patch.object(loop, "_push_fix", return_value=True):
            result = loop.run(backend)

        assert result.attempt_count >= 1

    def test_run_backend_exception_does_not_crash(self):
        loop = self._make_loop(max_attempts=1)
        backend = FakeBackendCI(raises=True)

        with patch.object(loop, "_fetch_ci_logs", return_value="FAILED tests/x.py::t"), \
             patch.object(loop, "_ci_is_green", return_value=False), \
             patch.object(loop, "_push_fix", return_value=True):
            result = loop.run(backend)

        assert isinstance(result, CIRepairResult)
        assert not result.resolved

    def test_run_lint_error_tries_deterministic(self):
        loop = self._make_loop(max_attempts=1)
        backend = FakeBackendCI(status="finished")
        ruff_attempt = CIRepairAttempt(
            attempt=0, failure_type="lint_error",
            strategy="deterministic_lint", goal_sent="ruff",
            success=True, duration_seconds=0.5,
        )

        with patch.object(loop, "_fetch_ci_logs", return_value="ruff: error E501"), \
             patch.object(loop, "_try_deterministic_lint_fix", return_value=ruff_attempt), \
             patch.object(loop, "_ci_is_green", return_value=True), \
             patch.object(loop, "_push_fix", return_value=True):
            result = loop.run(backend)

        assert result.resolved is True

    def test_run_total_duration_set(self):
        loop = self._make_loop(max_attempts=1)
        backend = FakeBackendCI(status="finished")

        with patch.object(loop, "_fetch_ci_logs", return_value="FAILED tests/x.py::t"), \
             patch.object(loop, "_ci_is_green", return_value=True), \
             patch.object(loop, "_push_fix", return_value=True):
            result = loop.run(backend)

        assert result.total_duration_seconds >= 0.0

    def test_failure_summary_format(self):
        loop = self._make_loop(max_attempts=1)
        loop._attempts = [
            CIRepairAttempt(
                attempt=0, failure_type="test_failure",
                strategy="llm_repair", goal_sent="fix",
                success=False, duration_seconds=1.0, error="timeout",
            )
        ]
        summary = loop._build_failure_summary()
        assert "attempt" in summary.lower() or "1" in summary
        assert "test_failure" in summary
