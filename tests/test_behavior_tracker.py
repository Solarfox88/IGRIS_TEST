"""Unit tests for BehaviorTracker — supervisor-first autonomy policy (#147)."""
import pytest
from igris.core.behavior_tracker import (
    BehaviorTracker,
    BehaviorRecord,
    SelfAuditResult,
    ALL_BEHAVIOR_CODES,
    BEHAVIOR_BY_NAME,
)


def make_tracker(**kwargs):
    return BehaviorTracker(run_id="test-run", issue_number=147, **kwargs)


# ---------------------------------------------------------------------------
# Record / classify
# ---------------------------------------------------------------------------

def test_record_by_code():
    bt = make_tracker()
    r = bt.record("E001", "edited the wrong file")
    assert r.code == "E001"
    assert r.name == "wrong_file_edit"
    assert len(bt.records) == 1


def test_record_by_name():
    bt = make_tracker()
    r = bt.record("reasoning_loop_no_progress", "same action 4×")
    assert r.code == "E002"
    assert r.name == "reasoning_loop_no_progress"


def test_record_unknown():
    bt = make_tracker()
    r = bt.record("mystery_behavior", "unknown")
    assert r.code == "E999"


def test_blocking_non_blocking():
    bt = make_tracker()
    bt.record("E001", "wrong file", blocking=True)
    bt.record("E002", "no progress", blocking=False)
    assert len(bt.blocking()) == 1
    assert len(bt.non_blocking()) == 1


def test_by_severity():
    bt = make_tracker()
    bt.record("E001", "d1", severity="high")
    bt.record("E002", "d2", severity="low")
    assert len(bt.by_severity("high")) == 1
    assert len(bt.by_severity("low")) == 1


def test_summary_empty():
    bt = make_tracker()
    assert "no behaviors" in bt.summary()


def test_summary_with_records():
    bt = make_tracker()
    bt.record("E001", "a")
    bt.record("E001", "b")
    bt.record("E002", "c")
    s = bt.summary()
    assert "wrong_file_edit×2" in s
    assert "reasoning_loop_no_progress×1" in s


# ---------------------------------------------------------------------------
# Self-audit — dirty workspace
# ---------------------------------------------------------------------------

def _audit(bt, **overrides):
    defaults = dict(
        run_status="blocked",
        failure_class="reasoning_loop_blocked",
        repair_cycles_used=0,
        smoke_ran=False,
        pytest_ran=True,
        workspace_dirty=False,
        escalation_budget_exhausted=False,
        escalation_was_called=False,
        completion_mode="",
        project_root="",
    )
    defaults.update(overrides)
    return bt.self_audit(**defaults)


def test_audit_dirty_workspace_after_blocked():
    bt = make_tracker()
    result = _audit(bt, run_status="blocked", workspace_dirty=True)
    assert "dirty_workspace_after_blocked" in result.missed_behaviors


def test_audit_clean_workspace_no_flag():
    bt = make_tracker()
    result = _audit(bt, run_status="blocked", workspace_dirty=False)
    assert "dirty_workspace_after_blocked" not in result.missed_behaviors


# ---------------------------------------------------------------------------
# Self-audit — success without verification
# ---------------------------------------------------------------------------

def test_audit_success_without_verification():
    bt = make_tracker()
    result = _audit(bt, run_status="completed", smoke_ran=False, pytest_ran=False)
    assert "success_without_verification" in result.missed_behaviors


def test_audit_success_with_pytest_ok():
    bt = make_tracker()
    result = _audit(bt, run_status="completed", smoke_ran=False, pytest_ran=True)
    assert "success_without_verification" not in result.missed_behaviors


# ---------------------------------------------------------------------------
# Self-audit — no escalation at budget exhaustion
# ---------------------------------------------------------------------------

def test_audit_escalation_not_called_when_budget_exhausted():
    bt = make_tracker()
    result = _audit(
        bt, run_status="blocked",
        escalation_budget_exhausted=True,
        escalation_was_called=False,
    )
    assert "no_escalation_at_budget_exhaustion" in result.missed_behaviors


def test_audit_no_flag_when_escalation_called():
    bt = make_tracker()
    result = _audit(
        bt, run_status="blocked",
        escalation_budget_exhausted=True,
        escalation_was_called=True,
    )
    assert "no_escalation_at_budget_exhaustion" not in result.missed_behaviors


# ---------------------------------------------------------------------------
# Self-audit — repair without progress
# ---------------------------------------------------------------------------

def test_audit_repair_without_progress():
    bt = make_tracker()
    result = _audit(
        bt, run_status="blocked",
        repair_cycles_used=2,
        failure_class="reasoning_loop_blocked",
    )
    assert "repair_without_progress" in result.missed_behaviors


def test_audit_no_repair_flag_on_completed():
    bt = make_tracker()
    result = _audit(bt, run_status="completed", repair_cycles_used=2, failure_class="")
    assert "repair_without_progress" not in result.missed_behaviors


# ---------------------------------------------------------------------------
# Self-audit — degraded completion
# ---------------------------------------------------------------------------

def test_audit_degraded_completion_lesson():
    bt = make_tracker()
    result = _audit(bt, run_status="completed", completion_mode="degraded")
    assert "no_diff_repair" in result.missed_behaviors


def test_audit_clean_completion_no_lesson():
    bt = make_tracker()
    result = _audit(bt, run_status="completed", completion_mode="verified_diff")
    assert "no_diff_repair" not in result.missed_behaviors


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def test_to_dict():
    bt = make_tracker()
    bt.record("E001", "detail", severity="high")
    d = bt.to_dict()
    assert d["total"] == 1
    assert d["records"][0]["code"] == "E001"
    assert d["run_id"] == "test-run"
    assert d["issue_number"] == 147


# ---------------------------------------------------------------------------
# SupervisorRun fields
# ---------------------------------------------------------------------------

def test_supervisor_run_has_completion_mode():
    from igris.core.self_repair_supervisor import SupervisorRun
    r = SupervisorRun(run_id="x", rank_id="rank")
    assert hasattr(r, "completion_mode")
    assert r.completion_mode == ""


def test_supervisor_run_has_behavior_tracker():
    from igris.core.self_repair_supervisor import SupervisorRun
    r = SupervisorRun(run_id="x", rank_id="rank")
    assert hasattr(r, "behavior_tracker")
    assert r.behavior_tracker is None
