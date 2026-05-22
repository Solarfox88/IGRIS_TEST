from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from igris.core.smw_pr_review import (
    PRReviewRequest, PRReviewResult,
    _is_high_risk, load_review_results, review_pr, save_review_result,
)
from igris.core.smw_weak_signals import (
    _load_runs,
    _ts,
    detect_cost_drift,
    detect_decomposition_inflation,
    detect_fix_not_sticky,
    detect_model_overkill,
    detect_repair_cycle_saturation,
    detect_systemic_capability_gap,
    get_weak_signal_summary,
    run_all_detectors,
)


def _iso(delta_days: float = 0) -> str:
    """Return ISO UTC timestamp offset by delta_days from now."""
    dt = datetime.now(timezone.utc) - timedelta(days=delta_days)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _req(**kw):
    base = dict(
        pr_number=1, pr_title="t", pr_diff="d", issue_description="i",
        changed_files=["a"], ci_passed=True, run_id="r",
        last_failure_class="", repair_cycles_used=0, max_repair_cycles=3,
        capability_signals={},
    )
    base.update(kw)
    return PRReviewRequest(**base)


# PR review tests ─────────────────────────────────────────────────────────────

def test_is_high_risk_wrong_file_edit():
    assert _is_high_risk(_req(last_failure_class="wrong_file_edit"))


def test_is_high_risk_normal():
    assert not _is_high_risk(_req())


@patch("igris.core.smw_pr_review._call_deepseek_review")
def test_review_pr_approved(mock_call):
    mock_call.return_value = {"approved": True, "confidence": 0.9, "concerns": [], "suggestion": "ok"}
    r = asyncio.run(review_pr(_req(), "."))
    assert r.approved and r.confidence == 0.9


@patch("igris.core.smw_pr_review._call_deepseek_review")
def test_review_pr_second_opinion(mock_call):
    mock_call.side_effect = [
        {"approved": True, "confidence": 0.5, "concerns": [], "suggestion": "a"},
        {"approved": True, "confidence": 0.8, "concerns": [], "suggestion": "b"},
    ]
    r = asyncio.run(review_pr(_req(), "."))
    assert mock_call.call_count == 2
    assert r.confidence == 0.8


@patch("igris.core.smw_pr_review._call_codex_tiebreaker")
@patch("igris.core.smw_pr_review._call_deepseek_review")
def test_review_pr_tiebreaker(mock_call, mock_tie):
    mock_call.side_effect = [
        {"approved": True, "confidence": 0.5, "concerns": [], "suggestion": "a"},
        {"approved": False, "confidence": 0.7, "concerns": [], "suggestion": "b"},
    ]
    mock_tie.return_value = {"approved": True, "confidence": 0.75, "concerns": [], "suggestion": "t"}
    r = asyncio.run(review_pr(_req(), "."))
    assert r.tiebreaker_used


@patch("igris.core.smw_pr_review._call_deepseek_review", side_effect=RuntimeError("down"))
def test_review_pr_fail_open(_):
    r = asyncio.run(review_pr(_req(), "."))
    assert r.approved and r.confidence == 0.3


def test_save_and_load_review_result():
    with tempfile.TemporaryDirectory() as td:
        rr = PRReviewResult(7, True, 0.9, "m", [], "", time.time(), False)
        save_review_result(rr, td)
        got = load_review_results(td)
        assert got[0].pr_number == 7


# _load_runs correctness tests ────────────────────────────────────────────────

def test_load_runs_empty_dir():
    with tempfile.TemporaryDirectory() as td:
        assert _load_runs(td) == []


def test_load_runs_returns_list():
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(f"{td}/.igris", exist_ok=True)
        payload = {"runs": {
            "r1": {"run_id": "r1", "created_at": _iso(2), "status": "done"},
            "r2": {"run_id": "r2", "created_at": _iso(1), "status": "open"},
        }}
        with open(f"{td}/.igris/supervisor_runs.json", "w") as f:
            json.dump(payload, f)
        runs = _load_runs(td)
        assert isinstance(runs, list) and len(runs) == 2


def test_load_runs_sorted_ascending():
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(f"{td}/.igris", exist_ok=True)
        payload = {"runs": {
            "r2": {"run_id": "r2", "created_at": _iso(1)},
            "r1": {"run_id": "r1", "created_at": _iso(2)},
        }}
        with open(f"{td}/.igris/supervisor_runs.json", "w") as f:
            json.dump(payload, f)
        runs = _load_runs(td)
        assert runs[0]["run_id"] == "r1" and runs[1]["run_id"] == "r2"


# Detector tests using correct field names ────────────────────────────────────

def test_detect_model_overkill_triggered():
    runs = [{"api_escalations_used": 1, "max_api_escalations_per_run": 5} for _ in range(14)]
    runs += [{"api_escalations_used": 0} for _ in range(6)]
    assert detect_model_overkill(runs) is not None


def test_detect_model_overkill_not_triggered():
    runs = [{"api_escalations_used": 1, "max_api_escalations_per_run": 5} for _ in range(8)]
    runs += [{"api_escalations_used": 0} for _ in range(12)]
    assert detect_model_overkill(runs) is None


def test_detect_model_overkill_ceiling_excluded():
    runs = [{"api_escalations_used": 5, "max_api_escalations_per_run": 5} for _ in range(20)]
    assert detect_model_overkill(runs) is None


def test_detect_repair_cycle_saturation():
    runs = [{"repair_cycles_used": 3, "max_repair_cycles": 3} for _ in range(8)]
    runs += [{"repair_cycles_used": 1, "max_repair_cycles": 3} for _ in range(2)]
    assert detect_repair_cycle_saturation(runs) is not None


def test_detect_systemic_gap_uses_run_id():
    runs = [{"failure_class": "wrong_file_edit", "run_id": f"run{i}"} for i in range(4)]
    assert detect_systemic_capability_gap(runs) is not None


def test_detect_systemic_gap_not_triggered():
    runs = [{"failure_class": "wrong_file_edit", "run_id": f"run{i}"} for i in range(2)]
    assert detect_systemic_capability_gap(runs) is None


def test_detect_decomposition_uses_failure_class():
    runs = [{"failure_class": "decomposition_required"} for _ in range(6)]
    runs += [{"failure_class": "other"}]
    assert detect_decomposition_inflation(runs) is not None


def test_detect_cost_drift_uses_created_at():
    # this_week: 1 day ago (value 5.0); prev_week: 10 days ago (value 1.0)
    # a=5.0, b=1.0, b*1.3=1.3 → a > 1.3 → signal fires
    runs = [
        {"created_at": _iso(1), "api_budget_used_usd": 5.0},
        {"created_at": _iso(10), "api_budget_used_usd": 1.0},
    ]
    assert detect_cost_drift(runs) is not None


def test_detect_fix_not_sticky_uses_branch():
    runs = [
        {"branch": "igris/mission-abc", "status": "done",
         "updated_at": _iso(0.1), "created_at": _iso(0.1)},
        {"branch": "igris/mission-abc", "status": "open",
         "created_at": _iso(0.05)},
    ]
    assert detect_fix_not_sticky(runs, ".") is not None


def test_run_all_detectors_no_signals():
    with tempfile.TemporaryDirectory() as td:
        assert run_all_detectors(td) == []


def test_run_all_detectors_with_supervisor_format():
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(f"{td}/.igris", exist_ok=True)
        payload = {"runs": {
            f"r{i}": {
                "run_id": f"r{i}", "created_at": _iso(i + 1),
                "status": "done", "failure_class": "", "api_budget_used_usd": 0.1,
                "api_escalations_used": 0, "max_api_escalations_per_run": 3,
                "repair_cycles_used": 1, "max_repair_cycles": 3,
                "branch": f"igris/mission-{i}",
            }
            for i in range(5)
        }}
        with open(f"{td}/.igris/supervisor_runs.json", "w") as f:
            json.dump(payload, f)
        signals = run_all_detectors(td)
        assert isinstance(signals, list)


def test_get_weak_signal_summary_keys():
    with tempfile.TemporaryDirectory() as td:
        out = get_weak_signal_summary(td)
        assert "weak_signals_active" in out and "metrics" in out
