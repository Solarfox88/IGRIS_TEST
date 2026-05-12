from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from igris.core import self_repair_supervisor as sup
from igris.core.self_repair_supervisor import SupervisorEvent, SupervisorRun
from igris.web.server import CONFIG, create_app


@pytest.fixture()
def isolated_run_store():
    with sup.RUN_LOCK:
        backup = dict(sup.RUN_STORE)
        sup.RUN_STORE.clear()
    try:
        yield
    finally:
        with sup.RUN_LOCK:
            sup.RUN_STORE.clear()
            sup.RUN_STORE.update(backup)


@pytest.fixture()
def client():
    return TestClient(create_app())


def _seed_run(*, run_id: str, status: str = "running") -> SupervisorRun:
    run = SupervisorRun(run_id=run_id, rank_id="S-full-e2e")
    run.status = status
    run.failure_class = "reasoning_loop_blocked" if status == "blocked" else ""
    run.repair_cycles_used = 2
    run.api_escalations_used = 1
    run.api_budget_used_usd = 0.42
    run.report = {
        "mission_orchestration": {
            "mode": "staged",
            "stages": [
                {"stage_id": "backend_api_change", "status": "success"},
                {"stage_id": "backend_tests", "status": "failure"},
                {"stage_id": "ui_dashboard_change", "status": "pending"},
            ],
        }
    }
    run.events.append(
        SupervisorEvent(
            phase="rank_reasoning",
            status="running" if status == "running" else "blocked",
            detail="token=sk-secret-should-not-leak",
            data={"stage_id": "backend_tests", "token": "sk-abc123456"},
            audit_status="audit-new",
            audit_review_id="review-1",
            audit_scope_hash="scope-1",
        )
    )
    run.events.append(
        SupervisorEvent(
            phase="repair_issue",
            status="success",
            detail="https://github.com/Solarfox88/IGRIS_GPT/issues/314",
            data={},
            audit_status="audit-reviewed",
            audit_review_id="review-issue",
            audit_scope_hash="scope-issue",
        )
    )
    if status == "running":
        run.events.append(
            SupervisorEvent(
                phase="mission_stage",
                status="running",
                detail="still progressing",
                data={"stage_id": "backend_tests"},
                audit_status="audit-reviewed",
                audit_review_id="review-2",
                audit_scope_hash="scope-2",
            )
        )
    return run


def test_run_summary_endpoint_returns_compact_fields(client, isolated_run_store):
    run = _seed_run(run_id="abc123def456", status="running")
    with sup.RUN_LOCK:
        sup.RUN_STORE[run.run_id] = run

    r = client.get(f"/api/rank/runs/{run.run_id}/summary")
    assert r.status_code == 200
    data = r.json()
    for key in (
        "run_id",
        "status",
        "current_stage",
        "failed_stage",
        "repair_cycles_used",
        "api_escalations_used",
        "api_budget_used_usd",
        "escalation_issue_url",
        "stage_summary",
        "audit_summary",
        "next_action",
    ):
        assert key in data
    # compact response should not contain full events dump
    assert "events" not in data


def test_active_endpoint_includes_only_running_runs(client, isolated_run_store):
    run_active = _seed_run(run_id="run-active-001", status="running")
    run_blocked = _seed_run(run_id="run-blocked-001", status="blocked")
    with sup.RUN_LOCK:
        sup.RUN_STORE[run_active.run_id] = run_active
        sup.RUN_STORE[run_blocked.run_id] = run_blocked

    r = client.get("/api/rank/runs/active")
    assert r.status_code == 200
    runs = r.json().get("runs", [])
    ids = {item.get("run_id") for item in runs}
    assert run_active.run_id in ids
    assert run_blocked.run_id not in ids


def test_stage_summary_contains_success_failure_pending(client, isolated_run_store):
    run = _seed_run(run_id="stage-summary-001", status="running")
    with sup.RUN_LOCK:
        sup.RUN_STORE[run.run_id] = run

    r = client.get(f"/api/rank/runs/{run.run_id}/summary")
    assert r.status_code == 200
    counts = r.json()["stage_summary"]["counts"]
    assert counts["success"] == 1
    assert counts["failure"] == 1
    assert counts["pending"] == 1
    assert r.json()["failed_stage"] == "backend_tests"


def test_summary_includes_escalation_issue_url(client, isolated_run_store):
    run = _seed_run(run_id="issue-url-001", status="running")
    with sup.RUN_LOCK:
        sup.RUN_STORE[run.run_id] = run

    r = client.get(f"/api/rank/runs/{run.run_id}/summary")
    assert r.status_code == 200
    assert r.json()["escalation_issue_url"].endswith("/issues/314")


def test_audit_summary_counts_statuses(client, isolated_run_store, tmp_path, monkeypatch):
    run = _seed_run(run_id="audit-summary-001", status="running")
    run.events.append(
        SupervisorEvent(
            phase="repair_issue",
            status="success",
            detail="fixed",
            data={},
            audit_status="audit-fixed",
            audit_review_id="review-fixed",
            audit_scope_hash="scope-fixed",
        )
    )
    with sup.RUN_LOCK:
        sup.RUN_STORE[run.run_id] = run

    audit_dir = tmp_path / ".igris"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "supervisor_audit.json").write_text(
        json.dumps(
            {
                "records": {
                    "h1": {"audit_status": "audit-new"},
                    "h2": {"audit_status": "audit-reviewed"},
                    "h3": {"audit_status": "audit-deferred", "audit_next_review_after": "2000-01-01T00:00:00+00:00"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(CONFIG, "project_root", Path(tmp_path))

    r = client.get("/api/rank/audit/summary")
    assert r.status_code == 200
    data = r.json()
    assert data["in_memory"]["counts"]["audit-new"] >= 1
    assert data["in_memory"]["counts"]["audit-reviewed"] >= 1
    assert data["in_memory"]["counts"]["audit-fixed"] >= 1
    assert data["persisted"]["counts"]["audit-new"] == 1
    assert data["persisted"]["counts"]["audit-reviewed"] == 1
    assert data["persisted"]["counts"]["audit-deferred"] == 1
    assert data["persisted"]["deferred_due_count"] == 1


def test_ui_contains_supervisor_monitor_label(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Supervisor Monitor" in r.text
    assert 'id="dash-supervisor-monitor"' in r.text
    assert 'id="btn-refresh-supervisor-monitor"' in r.text
    assert "Loading supervisor runs..." in r.text


def test_ui_js_contains_supervisor_monitor_states(client):
    r = client.get("/static/js/app.js")
    assert r.status_code == 200
    js = r.text
    assert "No active supervisor runs. Start a supervised mission or view recent audit history." in js
    assert "Supervisor monitor unavailable:" in js
    assert "Loading supervisor runs..." in js
    assert '"/api/rank/runs/active"' in js
    assert '"/api/rank/audit/summary"' in js
    assert "btn-refresh-supervisor-monitor" in js
    assert "run.run_id" in js
    assert "run.rank_id" in js
    assert "run.current_stage" in js
    assert "run.failed_stage" in js
    assert "run.failure_class" in js
    assert "run.repair_cycles_used" in js
    assert "run.api_escalations_used" in js
    assert "run.api_budget_used_usd" in js
    assert "run.next_action" in js


def test_summary_endpoint_does_not_expose_secrets(client, isolated_run_store):
    run = _seed_run(run_id="secret-redaction-001", status="running")
    with sup.RUN_LOCK:
        sup.RUN_STORE[run.run_id] = run

    r = client.get(f"/api/rank/runs/{run.run_id}/summary")
    assert r.status_code == 200
    text = r.text.lower()
    assert "sk-secret" not in text


def test_audit_summary_includes_recent_runs(client, isolated_run_store):
    run = _seed_run(run_id="recent-runs-001", status="running")
    with sup.RUN_LOCK:
        sup.RUN_STORE[run.run_id] = run

    r = client.get("/api/rank/audit/summary")
    assert r.status_code == 200
    data = r.json()
    assert "recent_runs" in data
    assert any(item.get("run_id") == "recent-runs-001" for item in data["recent_runs"])
