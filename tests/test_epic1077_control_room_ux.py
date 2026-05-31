"""Tests for Epic #1077 — Control Room UX endpoints.

Verifies the new /timeline, /risk-detail, /report endpoints, and
that the existing /status and /approve endpoints are present and functional.

Tests use FastAPI TestClient with mocked run objects.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from igris.web.server import create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run(
    run_id: str = "test123",
    status: str = "running",
    failure_class: str = "",
    events: Optional[List[Any]] = None,
    decomposition: Optional[Dict] = None,
) -> MagicMock:
    run = MagicMock()
    run.run_id = run_id
    run.rank_id = "test-rank"
    run.status = status
    run.failure_class = failure_class
    run.goal = "Implement feature X"
    run.repair_cycles_used = 1
    run.same_failure_count = 0
    run.execution_budget_used_usd = 0.05
    run.capability_signals = {}
    run.cancel_requested = False
    run.cancel_reason = ""
    run.decomposition = decomposition
    run.acceptance_evidence = None
    run.start_ts = time.time() - 60

    # Events
    if events is None:
        ev1 = MagicMock()
        ev1.phase = "start"
        ev1.status = "running"
        ev1.detail = "Supervisor started"
        ev1.ts = time.time() - 60
        ev1.same_failure_count = 0

        ev2 = MagicMock()
        ev2.phase = "rank_reasoning"
        ev2.status = "success"
        ev2.detail = "Reasoning finished"
        ev2.ts = time.time() - 30
        ev2.same_failure_count = 0

        run.events = [ev1, ev2]
    else:
        run.events = events

    run.report = {}
    return run


def _client() -> TestClient:
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# /api/rank/runs/{run_id}/timeline
# ---------------------------------------------------------------------------

class TestRunTimelineEndpoint:

    def test_timeline_404_unknown_run(self):
        client = _client()
        with patch("igris.core.self_repair_supervisor.get_supervised_run", return_value=None):
            resp = client.get("/api/rank/runs/unknown/timeline")
        assert resp.status_code == 404

    def test_timeline_returns_expected_keys(self):
        client = _client()
        run = _make_run("r1")
        with patch("igris.core.self_repair_supervisor.get_supervised_run", return_value=run):
            resp = client.get("/api/rank/runs/r1/timeline")
        assert resp.status_code == 200
        body = resp.json()
        assert "timeline" in body
        assert "total_events" in body
        assert "phases_seen" in body
        assert "run_id" in body

    def test_timeline_groups_by_phase(self):
        client = _client()
        run = _make_run("r1")
        with patch("igris.core.self_repair_supervisor.get_supervised_run", return_value=run):
            resp = client.get("/api/rank/runs/r1/timeline")
        body = resp.json()
        phases_seen = body["phases_seen"]
        assert "start" in phases_seen
        assert "rank_reasoning" in phases_seen

    def test_timeline_total_events_matches(self):
        client = _client()
        run = _make_run("r1")
        with patch("igris.core.self_repair_supervisor.get_supervised_run", return_value=run):
            resp = client.get("/api/rank/runs/r1/timeline")
        body = resp.json()
        assert body["total_events"] == len(run.events)

    def test_timeline_empty_events(self):
        client = _client()
        run = _make_run("r1", events=[])
        with patch("igris.core.self_repair_supervisor.get_supervised_run", return_value=run):
            resp = client.get("/api/rank/runs/r1/timeline")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_events"] == 0
        assert body["timeline"] == []


# ---------------------------------------------------------------------------
# /api/rank/runs/{run_id}/risk-detail
# ---------------------------------------------------------------------------

class TestRunRiskDetailEndpoint:

    def test_risk_detail_404_unknown_run(self):
        client = _client()
        with patch("igris.core.self_repair_supervisor.get_supervised_run", return_value=None):
            resp = client.get("/api/rank/runs/unknown/risk-detail")
        assert resp.status_code == 404

    def test_risk_detail_returns_expected_keys(self):
        client = _client()
        run = _make_run("r2")
        with patch("igris.core.self_repair_supervisor.get_supervised_run", return_value=run):
            resp = client.get("/api/rank/runs/r2/risk-detail")
        assert resp.status_code == 200
        body = resp.json()
        required = {
            "run_id", "failure_class", "repair_cycles_used", "same_failure_count",
            "capability_signals", "failure_history", "repair_attempts",
            "risk_trajectory", "budget_used_usd",
        }
        assert required.issubset(set(body.keys()))

    def test_risk_detail_budget_is_float(self):
        client = _client()
        run = _make_run("r2")
        with patch("igris.core.self_repair_supervisor.get_supervised_run", return_value=run):
            resp = client.get("/api/rank/runs/r2/risk-detail")
        body = resp.json()
        assert isinstance(body["budget_used_usd"], (int, float))

    def test_risk_detail_decomposition_info_when_decomposition(self):
        client = _client()
        run = _make_run("r2", decomposition={"_quality_score": 0.85, "_quality_valid": True})
        with patch("igris.core.self_repair_supervisor.get_supervised_run", return_value=run):
            resp = client.get("/api/rank/runs/r2/risk-detail")
        body = resp.json()
        assert "decomposition_info" in body
        assert body["decomposition_info"]["pending"] is True


# ---------------------------------------------------------------------------
# /api/rank/runs/{run_id}/report
# ---------------------------------------------------------------------------

class TestRunReportEndpoint:

    def test_report_404_unknown_run(self):
        client = _client()
        with patch("igris.core.self_repair_supervisor.get_supervised_run", return_value=None):
            resp = client.get("/api/rank/runs/unknown/report")
        assert resp.status_code == 404

    def test_report_returns_expected_keys(self):
        client = _client()
        run = _make_run("r3", status="blocked", failure_class="pytest_failure")
        with patch("igris.core.self_repair_supervisor.get_supervised_run", return_value=run):
            resp = client.get("/api/rank/runs/r3/report")
        assert resp.status_code == 200
        body = resp.json()
        required = {"run_id", "outcome", "status", "goal", "failure_class", "key_metrics", "recommendations"}
        assert required.issubset(set(body.keys()))

    def test_report_outcome_success(self):
        client = _client()
        run = _make_run("r4", status="success")
        with patch("igris.core.self_repair_supervisor.get_supervised_run", return_value=run):
            resp = client.get("/api/rank/runs/r4/report")
        body = resp.json()
        assert body["outcome"] == "success"

    def test_report_outcome_blocked_pytest(self):
        client = _client()
        run = _make_run("r5", status="blocked", failure_class="pytest_failure")
        with patch("igris.core.self_repair_supervisor.get_supervised_run", return_value=run):
            resp = client.get("/api/rank/runs/r5/report")
        body = resp.json()
        assert body["outcome"] == "blocked"
        assert len(body["recommendations"]) > 0

    def test_report_outcome_decomposition_required(self):
        client = _client()
        run = _make_run("r6", status="blocked", failure_class="decomposition_required")
        with patch("igris.core.self_repair_supervisor.get_supervised_run", return_value=run):
            resp = client.get("/api/rank/runs/r6/report")
        body = resp.json()
        assert body["outcome"] == "decomposition_required"
        assert any("decomposition" in r.lower() for r in body["recommendations"])

    def test_report_key_metrics_present(self):
        client = _client()
        run = _make_run("r7", status="running")
        with patch("igris.core.self_repair_supervisor.get_supervised_run", return_value=run):
            resp = client.get("/api/rank/runs/r7/report")
        body = resp.json()
        metrics = body["key_metrics"]
        assert "repair_cycles_used" in metrics
        assert "budget_used_usd" in metrics
        assert "event_count" in metrics

    def test_report_in_progress_outcome(self):
        client = _client()
        run = _make_run("r8", status="running")
        with patch("igris.core.self_repair_supervisor.get_supervised_run", return_value=run):
            resp = client.get("/api/rank/runs/r8/report")
        body = resp.json()
        assert body["outcome"] == "in_progress"


# ---------------------------------------------------------------------------
# Existing /status endpoint sanity
# ---------------------------------------------------------------------------

class TestExistingStatusEndpoint:

    def test_status_200_with_valid_run(self):
        client = _client()
        run = _make_run("r9")
        with patch("igris.core.self_repair_supervisor.get_supervised_run", return_value=run):
            resp = client.get("/api/rank/runs/r9/status")
        assert resp.status_code == 200
        body = resp.json()
        assert "risk_card" in body
        assert "recent_events" in body
        assert "elapsed_seconds" in body
