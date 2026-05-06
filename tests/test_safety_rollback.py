"""Tests for Epic #42 — Safety/Rollback/Autonomy Policy.

Covers risk_classifier, rollback_manager, safety_event_log.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from igris.core.risk_classifier import (
    APPROVAL_MODES,
    RISK_LEVELS,
    ApprovalDecision,
    check_approval,
    classify_action_risk,
    guard_secret_access,
    is_secret_file,
)
from igris.core.rollback_manager import (
    RollbackEntry,
    RollbackManager,
)
from igris.core.safety_event_log import (
    SAFETY_EVENT_TYPES,
    SafetyEvent,
    SafetyEventLog,
)


# ===========================================================================
# Risk Classifier
# ===========================================================================


class TestClassifyActionRisk:
    def test_read_is_low(self):
        assert classify_action_risk("read_file") == "low"

    def test_status_is_low(self):
        assert classify_action_risk("status_check") == "low"

    def test_test_is_low(self):
        assert classify_action_risk("pytest_run") == "low"

    def test_write_is_medium(self):
        assert classify_action_risk("write_file") == "medium"

    def test_install_is_medium(self):
        assert classify_action_risk("pip_install") == "medium"

    def test_deploy_is_high(self):
        assert classify_action_risk("deploy_staging") == "high"

    def test_push_is_high(self):
        assert classify_action_risk("git_push") == "high"

    def test_docker_down_is_high(self):
        assert classify_action_risk("docker_compose_down") == "high"

    def test_delete_is_critical(self):
        assert classify_action_risk("delete_database") == "critical"

    def test_force_push_is_critical(self):
        assert classify_action_risk("force_push") == "critical"

    def test_push_main_is_critical(self):
        assert classify_action_risk("push_main") == "critical"

    def test_db_migrate_is_critical(self):
        assert classify_action_risk("db_migrate") == "critical"

    def test_unknown_is_low(self):
        assert classify_action_risk("unknown_action_xyz") == "low"

    def test_description_used(self):
        assert classify_action_risk("action", "deploy to production") == "high"

    def test_write_secrets_critical(self):
        assert classify_action_risk("write_secrets") == "critical"


class TestCheckApproval:
    def test_low_always_allowed(self):
        d = check_approval("read", "low", "safe")
        assert d.allowed is True

    def test_medium_allowed_in_safe(self):
        d = check_approval("write", "medium", "safe")
        assert d.allowed is True

    def test_high_blocked_in_safe(self):
        d = check_approval("deploy", "high", "safe")
        assert d.allowed is False
        assert d.requires_confirmation is True

    def test_high_allowed_in_operator_with_rollback(self):
        d = check_approval("deploy", "high", "operator", has_rollback=True)
        assert d.allowed is True

    def test_high_blocked_in_operator_without_rollback(self):
        d = check_approval("deploy", "high", "operator", has_rollback=False)
        assert d.allowed is False
        assert d.requires_rollback is True

    def test_high_allowed_in_trusted_authorized_host(self):
        d = check_approval("deploy", "high", "trusted", host="server1", authorized_hosts=["server1"])
        assert d.allowed is True

    def test_high_blocked_in_trusted_unauthorized_host(self):
        d = check_approval("deploy", "high", "trusted", host="unknown")
        assert d.allowed is False

    def test_critical_blocked_without_token(self):
        d = check_approval("delete", "critical", "safe")
        assert d.allowed is False
        assert d.requires_confirmation is True

    def test_critical_allowed_with_token(self):
        d = check_approval("delete", "critical", "safe", approval_token="I_APPROVE")
        assert d.allowed is True

    def test_unknown_risk_defaults_critical(self):
        d = check_approval("x", "unknown_level", "safe")
        assert d.allowed is False

    def test_trace_id_propagated(self):
        d = check_approval("read", "low", "safe", trace_id="trace-123")
        assert d.trace_id == "trace-123"


class TestSecretGuard:
    def test_env_file_blocked(self):
        assert is_secret_file(".env") is True
        d = guard_secret_access(".env")
        assert d.allowed is False

    def test_env_local_blocked(self):
        assert is_secret_file(".env.local") is True

    def test_credentials_blocked(self):
        assert is_secret_file("credentials.json") is True

    def test_id_rsa_blocked(self):
        assert is_secret_file("id_rsa") is True

    def test_pem_blocked(self):
        assert is_secret_file("server.pem") is True

    def test_normal_file_allowed(self):
        assert is_secret_file("main.py") is False
        d = guard_secret_access("main.py")
        assert d.allowed is True

    def test_path_with_directory(self):
        assert is_secret_file("/home/user/.env") is True


class TestApprovalDecision:
    def test_to_dict(self):
        d = ApprovalDecision(allowed=True, risk_level="low", approval_mode="safe")
        data = d.to_dict()
        assert data["allowed"] is True
        assert data["risk_level"] == "low"


# ===========================================================================
# Rollback Manager
# ===========================================================================


class TestRollbackEntry:
    def test_to_dict(self):
        e = RollbackEntry(type="file_backup", original_path="/tmp/x.py")
        d = e.to_dict()
        assert d["id"].startswith("rb-")
        assert d["type"] == "file_backup"

    def test_roundtrip(self):
        e = RollbackEntry(type="state_snapshot", state_data={"v": 1})
        d = e.to_dict()
        e2 = RollbackEntry.from_dict(d)
        assert e2.type == "state_snapshot"

    def test_secret_redacted(self):
        e = RollbackEntry(original_path="sk-1234567890abcdef1234567890abcdef")
        d = e.to_dict()
        assert "sk-" not in d["original_path"]


class TestRollbackManager:
    def test_backup_file(self, tmp_path):
        src = tmp_path / "test.txt"
        src.write_text("hello", encoding="utf-8")
        mgr = RollbackManager(project_root=str(tmp_path))
        entry = mgr.backup_file(str(src), mission_id="m1")
        assert entry is not None
        assert entry.type == "file_backup"
        assert Path(entry.backup_path).exists()

    def test_backup_nonexistent(self, tmp_path):
        mgr = RollbackManager(project_root=str(tmp_path))
        entry = mgr.backup_file("/nonexistent/file.txt")
        assert entry is None

    def test_apply_rollback(self, tmp_path):
        src = tmp_path / "data.txt"
        src.write_text("original", encoding="utf-8")
        mgr = RollbackManager(project_root=str(tmp_path))
        entry = mgr.backup_file(str(src))
        # Modify the file
        src.write_text("modified", encoding="utf-8")
        assert src.read_text() == "modified"
        # Rollback
        success = mgr.apply_file_rollback(entry.id)
        assert success is True
        assert src.read_text() == "original"

    def test_apply_rollback_nonexistent(self, tmp_path):
        mgr = RollbackManager(project_root=str(tmp_path))
        assert mgr.apply_file_rollback("rb-nonexistent") is False

    def test_save_diff_snapshot(self, tmp_path):
        mgr = RollbackManager(project_root=str(tmp_path))
        entry = mgr.save_diff_snapshot("diff --git a/x.py", mission_id="m1")
        assert entry.type == "diff_snapshot"
        assert entry.state_data is not None

    def test_save_state_snapshot(self, tmp_path):
        mgr = RollbackManager(project_root=str(tmp_path))
        entry = mgr.save_state_snapshot({"tests_passing": True}, mission_id="m1")
        assert entry.type == "state_snapshot"

    def test_verify_rollback_applicable(self, tmp_path):
        src = tmp_path / "v.txt"
        src.write_text("v", encoding="utf-8")
        mgr = RollbackManager(project_root=str(tmp_path))
        entry = mgr.backup_file(str(src))
        result = mgr.verify_rollback_applicable(entry.id)
        assert result["applicable"] is True

    def test_verify_nonexistent(self, tmp_path):
        mgr = RollbackManager(project_root=str(tmp_path))
        result = mgr.verify_rollback_applicable("rb-nope")
        assert result["applicable"] is False

    def test_verify_already_applied(self, tmp_path):
        src = tmp_path / "a.txt"
        src.write_text("a", encoding="utf-8")
        mgr = RollbackManager(project_root=str(tmp_path))
        entry = mgr.backup_file(str(src))
        mgr.apply_file_rollback(entry.id)
        result = mgr.verify_rollback_applicable(entry.id)
        assert result["applicable"] is False

    def test_list_entries(self, tmp_path):
        src = tmp_path / "l.txt"
        src.write_text("l", encoding="utf-8")
        mgr = RollbackManager(project_root=str(tmp_path))
        mgr.backup_file(str(src), mission_id="m1")
        mgr.save_state_snapshot({"x": 1}, mission_id="m2")
        entries = mgr.list_entries()
        assert len(entries) == 2

    def test_list_filter_by_mission(self, tmp_path):
        src = tmp_path / "f.txt"
        src.write_text("f", encoding="utf-8")
        mgr = RollbackManager(project_root=str(tmp_path))
        mgr.backup_file(str(src), mission_id="m1")
        mgr.save_state_snapshot({"x": 1}, mission_id="m2")
        entries = mgr.list_entries(mission_id="m1")
        assert len(entries) == 1

    def test_get_entry(self, tmp_path):
        mgr = RollbackManager(project_root=str(tmp_path))
        entry = mgr.save_state_snapshot({"v": 1})
        loaded = mgr.get_entry(entry.id)
        assert loaded is not None

    def test_has_rollback_for_action(self, tmp_path):
        src = tmp_path / "h.txt"
        src.write_text("h", encoding="utf-8")
        mgr = RollbackManager(project_root=str(tmp_path))
        mgr.backup_file(str(src), action_id="act-1")
        assert mgr.has_rollback_for_action("act-1") is True
        assert mgr.has_rollback_for_action("act-nope") is False


# ===========================================================================
# Safety Event Log
# ===========================================================================


class TestSafetyEvent:
    def test_to_dict(self):
        e = SafetyEvent(type="action_blocked", risk_level="high")
        d = e.to_dict()
        assert d["id"].startswith("sev-")
        assert d["type"] == "action_blocked"

    def test_from_dict(self):
        e = SafetyEvent(type="escalation", severity="critical")
        d = e.to_dict()
        e2 = SafetyEvent.from_dict(d)
        assert e2.severity == "critical"

    def test_secret_redacted(self):
        e = SafetyEvent(reason="key=sk-1234567890abcdef1234567890abcdef")
        d = e.to_dict()
        assert "sk-" not in d["reason"]


class TestSafetyEventLog:
    def test_log_block(self, tmp_path):
        log = SafetyEventLog(project_root=str(tmp_path))
        ev = log.log_block("deploy", "high", "Blocked in safe mode", mission_id="m1")
        assert ev.type == "action_blocked"
        assert ev.decision == "blocked"

    def test_log_approval(self, tmp_path):
        log = SafetyEventLog(project_root=str(tmp_path))
        ev = log.log_approval("read", "low", "safe")
        assert ev.type == "action_approved"

    def test_log_risk_decision(self, tmp_path):
        log = SafetyEventLog(project_root=str(tmp_path))
        ev = log.log_risk_decision("write", "medium", "allowed")
        assert ev.type == "risk_decision"

    def test_log_rollback_required(self, tmp_path):
        log = SafetyEventLog(project_root=str(tmp_path))
        ev = log.log_rollback_required("deploy", "No rollback present")
        assert ev.type == "rollback_required"

    def test_log_escalation(self, tmp_path):
        log = SafetyEventLog(project_root=str(tmp_path))
        ev = log.log_escalation("delete_db", "Needs human approval")
        assert ev.severity == "critical"

    def test_log_secret_detected(self, tmp_path):
        log = SafetyEventLog(project_root=str(tmp_path))
        ev = log.log_secret_detected("commit", detail="API key in diff")
        assert ev.type == "secret_detected"

    def test_log_policy_violation(self, tmp_path):
        log = SafetyEventLog(project_root=str(tmp_path))
        ev = log.log_policy_violation("force_push", "Force push not allowed")
        assert ev.type == "policy_violation"

    def test_list_events(self, tmp_path):
        log = SafetyEventLog(project_root=str(tmp_path))
        log.log_block("a1", "high", "blocked")
        log.log_approval("a2", "low", "safe")
        events = log.list_events()
        assert len(events) == 2

    def test_list_filter_by_type(self, tmp_path):
        log = SafetyEventLog(project_root=str(tmp_path))
        log.log_block("a1", "high", "blocked")
        log.log_approval("a2", "low", "safe")
        events = log.list_events(event_type="action_blocked")
        assert len(events) == 1

    def test_list_filter_by_mission(self, tmp_path):
        log = SafetyEventLog(project_root=str(tmp_path))
        log.log_block("a1", "high", "blocked", mission_id="m1")
        log.log_block("a2", "high", "blocked", mission_id="m2")
        events = log.list_events(mission_id="m1")
        assert len(events) == 1

    def test_list_filter_by_severity(self, tmp_path):
        log = SafetyEventLog(project_root=str(tmp_path))
        log.log_escalation("x", "critical thing")
        log.log_approval("y", "low", "safe")
        events = log.list_events(severity="critical")
        assert len(events) == 1

    def test_get_event(self, tmp_path):
        log = SafetyEventLog(project_root=str(tmp_path))
        ev = log.log_block("a1", "high", "test")
        loaded = log.get_event(ev.id)
        assert loaded is not None
        assert loaded["id"] == ev.id

    def test_get_nonexistent(self, tmp_path):
        log = SafetyEventLog(project_root=str(tmp_path))
        assert log.get_event("sev-nope") is None

    def test_count_blocks(self, tmp_path):
        log = SafetyEventLog(project_root=str(tmp_path))
        log.log_block("a1", "high", "b1")
        log.log_block("a2", "high", "b2")
        log.log_approval("a3", "low", "safe")
        assert log.count_blocks() == 2

    def test_count_blocks_by_mission(self, tmp_path):
        log = SafetyEventLog(project_root=str(tmp_path))
        log.log_block("a1", "high", "b1", mission_id="m1")
        log.log_block("a2", "high", "b2", mission_id="m2")
        assert log.count_blocks(mission_id="m1") == 1

    def test_summary(self, tmp_path):
        log = SafetyEventLog(project_root=str(tmp_path))
        log.log_block("a1", "high", "b1")
        log.log_approval("a2", "low", "safe")
        log.log_escalation("a3", "esc")
        summary = log.get_summary()
        assert summary["total_events"] == 3
        assert "by_type" in summary
        assert "by_severity" in summary


# ===========================================================================
# API integration
# ===========================================================================


class TestSafetyAPI:
    @pytest.fixture
    def client(self):
        from igris.web.server import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        return TestClient(app)

    def test_classify_risk(self, client):
        resp = client.post("/api/safety/classify-risk", json={
            "action_id": "git_push",
        })
        assert resp.status_code == 200
        assert resp.json()["risk_level"] == "high"

    def test_check_approval_safe_low(self, client):
        resp = client.post("/api/safety/check-approval", json={
            "action_id": "read", "risk_level": "low", "approval_mode": "safe",
        })
        assert resp.json()["allowed"] is True

    def test_check_approval_safe_high(self, client):
        resp = client.post("/api/safety/check-approval", json={
            "action_id": "deploy", "risk_level": "high", "approval_mode": "safe",
        })
        assert resp.json()["allowed"] is False

    def test_guard_secret(self, client):
        resp = client.post("/api/safety/guard-secret", json={"path": ".env"})
        assert resp.json()["allowed"] is False

    def test_guard_normal_file(self, client):
        resp = client.post("/api/safety/guard-secret", json={"path": "main.py"})
        assert resp.json()["allowed"] is True

    def test_rollback_save_state(self, client):
        resp = client.post("/api/rollback/save-state", json={
            "state": {"tests": True}, "mission_id": "m1",
        })
        assert resp.status_code == 200
        assert resp.json()["type"] == "state_snapshot"

    def test_rollback_list(self, client):
        resp = client.get("/api/rollback/entries")
        assert resp.status_code == 200
        assert "entries" in resp.json()

    def test_rollback_entry_not_found(self, client):
        resp = client.get("/api/rollback/entries/rb-nonexistent")
        assert resp.status_code == 404

    def test_safety_events_list(self, client):
        resp = client.get("/api/safety/events")
        assert resp.status_code == 200
        assert "events" in resp.json()

    def test_safety_event_not_found(self, client):
        resp = client.get("/api/safety/events/sev-nonexistent")
        assert resp.status_code == 404

    def test_safety_summary(self, client):
        resp = client.get("/api/safety/summary")
        assert resp.status_code == 200
        assert "total_events" in resp.json()
