"""Tests for Epic #1076 — DevOps/VPS operator + browser evidence.

Covers:
1. DevOpsManager host registry (list, register, remove, policy check).
2. Preflight checks (disk, git, service).
3. Deploy dry-run flow (preflight + no action).
4. HTTP smoke test.
5. API endpoints: /api/devops/hosts, /api/devops/hosts/{h}/policy,
   /api/devops/preflight, /api/devops/deploy (dry_run), /api/devops/smoke.
6. Existing /api/devops/health, /api/devops/deploy-status endpoints.

All subprocess calls are mocked so tests run without system tools or network.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from igris.core.devops_manager import (
    DevOpsManager,
    HostConfig,
    check_action_allowed,
)
from igris.web.server import create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mgr(tmp_path) -> DevOpsManager:
    return DevOpsManager(str(tmp_path))


def _client() -> TestClient:
    app = create_app()
    return TestClient(app)


def _mock_subprocess_ok(stdout: str = "", returncode: int = 0):
    """Return a MagicMock that mimics subprocess.CompletedProcess."""
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = ""
    return m


# ---------------------------------------------------------------------------
# check_action_allowed (pure function)
# ---------------------------------------------------------------------------

class TestCheckActionAllowed:

    def test_safe_policy_allows_status(self):
        result = check_action_allowed("safe", "status")
        assert result["allowed"] is True

    def test_safe_policy_blocks_deploy(self):
        result = check_action_allowed("safe", "deploy")
        assert result["allowed"] is False

    def test_operator_allows_deploy(self):
        result = check_action_allowed("operator", "deploy")
        assert result["allowed"] is True

    def test_trusted_allows_shell(self):
        result = check_action_allowed("trusted", "shell")
        assert result["allowed"] is True

    def test_unknown_policy_falls_back_to_safe(self):
        result = check_action_allowed("unknown_policy", "deploy")
        assert result["allowed"] is False

    def test_result_contains_reason(self):
        result = check_action_allowed("operator", "restart")
        assert "reason" in result
        assert isinstance(result["reason"], str)

    def test_allowed_actions_listed_in_result(self):
        result = check_action_allowed("operator", "deploy")
        assert "allowed_actions" in result
        assert isinstance(result["allowed_actions"], list)


# ---------------------------------------------------------------------------
# DevOpsManager — host registry
# ---------------------------------------------------------------------------

class TestDevOpsManagerRegistry:

    def test_empty_registry_on_new_instance(self, tmp_path):
        mgr = _mgr(tmp_path)
        assert mgr.list_hosts() == []

    def test_register_host_returns_registered(self, tmp_path):
        mgr = _mgr(tmp_path)
        host = HostConfig(hostname="vps1.example.com", policy="operator")
        result = mgr.register_host(host)
        assert result["registered"] is True
        assert result["hostname"] == "vps1.example.com"

    def test_list_hosts_after_register(self, tmp_path):
        mgr = _mgr(tmp_path)
        mgr.register_host(HostConfig(hostname="vps2.example.com"))
        hosts = mgr.list_hosts()
        assert len(hosts) == 1
        assert hosts[0]["hostname"] == "vps2.example.com"

    def test_registry_persists_to_disk(self, tmp_path):
        mgr = _mgr(tmp_path)
        mgr.register_host(HostConfig(hostname="vps3.example.com", policy="trusted"))
        # Load a new instance — should read from disk
        mgr2 = _mgr(tmp_path)
        assert len(mgr2.list_hosts()) == 1
        assert mgr2.list_hosts()[0]["hostname"] == "vps3.example.com"

    def test_invalid_policy_not_registered(self, tmp_path):
        mgr = _mgr(tmp_path)
        result = mgr.register_host(HostConfig(hostname="bad.example.com", policy="godmode"))
        assert result["registered"] is False
        assert "error" in result

    def test_remove_existing_host(self, tmp_path):
        mgr = _mgr(tmp_path)
        mgr.register_host(HostConfig(hostname="vps4.example.com"))
        result = mgr.remove_host("vps4.example.com")
        assert result["removed"] is True
        assert mgr.list_hosts() == []

    def test_remove_unknown_host_returns_error(self, tmp_path):
        mgr = _mgr(tmp_path)
        result = mgr.remove_host("nonexistent.example.com")
        assert result["removed"] is False
        assert "error" in result

    def test_get_host_returns_none_when_missing(self, tmp_path):
        mgr = _mgr(tmp_path)
        assert mgr.get_host("nope") is None

    def test_get_host_returns_config(self, tmp_path):
        mgr = _mgr(tmp_path)
        mgr.register_host(HostConfig(hostname="vps5.example.com", policy="safe"))
        h = mgr.get_host("vps5.example.com")
        assert h is not None
        assert h.hostname == "vps5.example.com"


# ---------------------------------------------------------------------------
# DevOpsManager — check_policy
# ---------------------------------------------------------------------------

class TestDevOpsManagerPolicy:

    def test_unregistered_host_denied(self, tmp_path):
        mgr = _mgr(tmp_path)
        result = mgr.check_policy("unknown.host", "deploy")
        assert result["allowed"] is False
        assert "not registered" in result["reason"]

    def test_registered_safe_host_denied_deploy(self, tmp_path):
        mgr = _mgr(tmp_path)
        mgr.register_host(HostConfig(hostname="vps.safe", policy="safe"))
        result = mgr.check_policy("vps.safe", "deploy")
        assert result["allowed"] is False

    def test_registered_operator_host_allowed_deploy(self, tmp_path):
        mgr = _mgr(tmp_path)
        mgr.register_host(HostConfig(hostname="vps.op", policy="operator"))
        result = mgr.check_policy("vps.op", "deploy")
        assert result["allowed"] is True

    def test_policy_result_contains_hostname(self, tmp_path):
        mgr = _mgr(tmp_path)
        mgr.register_host(HostConfig(hostname="vps.h", policy="trusted"))
        result = mgr.check_policy("vps.h", "shell")
        assert result["hostname"] == "vps.h"


# ---------------------------------------------------------------------------
# DevOpsManager — preflight
# ---------------------------------------------------------------------------

class TestDevOpsManagerPreflight:

    def _disk_stdout(self, used_pct: int = 40) -> str:
        """Fake `df -P` output."""
        return (
            "Filesystem     1024-blocks  Used Available Capacity Mounted on\n"
            f"/dev/sda1       100000000 {used_pct}000000 {100 - used_pct}000000    {used_pct}% /\n"
        )

    def test_preflight_ok_when_disk_has_space(self, tmp_path):
        with (
            patch("igris.core.devops_manager.subprocess.run") as mock_run,
        ):
            # df -> ok, git status -> clean, nc -> reachable
            mock_run.side_effect = [
                _mock_subprocess_ok(self._disk_stdout(40)),  # df
                _mock_subprocess_ok(""),                      # git status
                _mock_subprocess_ok("", returncode=0),       # nc
            ]
            mgr = _mgr(tmp_path)
            result = mgr.run_preflight()

        assert "checks" in result
        assert result["checks"]["disk"]["ok"] is True
        assert result["checks"]["git"]["clean"] is True

    def test_preflight_disk_fails_when_full(self, tmp_path):
        with patch("igris.core.devops_manager.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _mock_subprocess_ok(self._disk_stdout(96)),  # df — 4% free
                _mock_subprocess_ok(""),
                _mock_subprocess_ok("", returncode=0),
            ]
            mgr = _mgr(tmp_path)
            result = mgr.run_preflight(min_disk_pct_free=10)

        assert result["checks"]["disk"]["ok"] is False

    def test_preflight_returns_overall_ok_key(self, tmp_path):
        with patch("igris.core.devops_manager.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _mock_subprocess_ok(self._disk_stdout(40)),
                _mock_subprocess_ok(""),
                _mock_subprocess_ok("", returncode=0),
            ]
            mgr = _mgr(tmp_path)
            result = mgr.run_preflight()

        assert "ok" in result
        assert isinstance(result["ok"], bool)

    def test_preflight_hostname_in_result(self, tmp_path):
        with patch("igris.core.devops_manager.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _mock_subprocess_ok(self._disk_stdout(40)),
                _mock_subprocess_ok(""),
                _mock_subprocess_ok("", returncode=0),
            ]
            mgr = _mgr(tmp_path)
            result = mgr.run_preflight(hostname="prod.server")

        assert result["hostname"] == "prod.server"

    def test_preflight_dirty_git_recorded(self, tmp_path):
        with patch("igris.core.devops_manager.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _mock_subprocess_ok(self._disk_stdout(40)),
                _mock_subprocess_ok("M  dirty_file.py"),   # git status dirty
                _mock_subprocess_ok("", returncode=0),
            ]
            mgr = _mgr(tmp_path)
            result = mgr.run_preflight()

        assert result["checks"]["git"]["dirty"] is True


# ---------------------------------------------------------------------------
# DevOpsManager — deploy (dry_run)
# ---------------------------------------------------------------------------

class TestDevOpsManagerDeploy:

    def _disk_ok(self):
        return (
            "Filesystem     1024-blocks  Used Available Capacity Mounted on\n"
            "/dev/sda1       100000000 40000000 60000000    40% /\n"
        )

    def test_dry_run_returns_deployed_false(self, tmp_path):
        with patch("igris.core.devops_manager.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _mock_subprocess_ok(self._disk_ok()),
                _mock_subprocess_ok(""),
                _mock_subprocess_ok("", returncode=0),
            ]
            mgr = _mgr(tmp_path)
            result = mgr.run_deploy(strategy="dry_run", dry_run=True)

        assert result["deployed"] is False
        assert result["dry_run"] is True

    def test_deploy_report_has_preflight(self, tmp_path):
        with patch("igris.core.devops_manager.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _mock_subprocess_ok(self._disk_ok()),
                _mock_subprocess_ok(""),
                _mock_subprocess_ok("", returncode=0),
            ]
            mgr = _mgr(tmp_path)
            result = mgr.run_deploy(dry_run=True)

        assert "preflight" in result
        assert isinstance(result["preflight"], dict)

    def test_deploy_aborts_if_preflight_fails(self, tmp_path):
        with patch("igris.core.devops_manager.subprocess.run") as mock_run:
            # disk returns 99% used → preflight fail
            mock_run.side_effect = [
                _mock_subprocess_ok(
                    "Filesystem 1024-blocks Used Available Capacity Mounted on\n"
                    "/dev/sda1  100000000 99000000 1000000    99% /\n"
                ),
                _mock_subprocess_ok(""),
                _mock_subprocess_ok("", returncode=0),
            ]
            mgr = _mgr(tmp_path)
            result = mgr.run_deploy(min_disk_pct_free=10)

        assert result["deployed"] is False
        assert "abort_reason" in result

    def test_deploy_dry_run_flag_via_strategy(self, tmp_path):
        with patch("igris.core.devops_manager.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _mock_subprocess_ok(self._disk_ok()),
                _mock_subprocess_ok(""),
                _mock_subprocess_ok("", returncode=0),
            ]
            mgr = _mgr(tmp_path)
            # Pass dry_run as a strategy keyword
            result = mgr.run_deploy(strategy="dry_run")

        assert result["deployed"] is False


# ---------------------------------------------------------------------------
# DevOpsManager — smoke test
# ---------------------------------------------------------------------------

class TestDevOpsManagerSmoke:

    def test_smoke_returns_required_keys(self, tmp_path):
        mgr = _mgr(tmp_path)
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.status = 200
            mock_resp.read.return_value = b'{"ok": true}'
            mock_open.return_value = mock_resp
            result = mgr.run_smoke_test(url="http://localhost:7778/api/ping")

        assert "url" in result
        assert "ok" in result
        assert "status_code" in result
        assert "response_time_ms" in result

    def test_smoke_ok_true_on_200(self, tmp_path):
        mgr = _mgr(tmp_path)
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.status = 200
            mock_resp.read.return_value = b"pong"
            mock_open.return_value = mock_resp
            result = mgr.run_smoke_test("http://localhost:7778/api/ping")

        assert result["ok"] is True
        assert result["status_code"] == 200

    def test_smoke_ok_false_on_network_error(self, tmp_path):
        mgr = _mgr(tmp_path)
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            result = mgr.run_smoke_test("http://localhost:9999/nope")

        assert result["ok"] is False
        assert "error" in result

    def test_smoke_uses_default_url(self, tmp_path):
        mgr = _mgr(tmp_path)
        with patch("urllib.request.urlopen", side_effect=OSError("no server")):
            result = mgr.run_smoke_test(url="")

        assert "localhost:7778" in result["url"]


# ---------------------------------------------------------------------------
# API endpoints — /api/devops/hosts
# ---------------------------------------------------------------------------

class TestDevOpsHostsEndpoints:

    def test_list_hosts_returns_list(self, tmp_path):
        client = _client()
        with patch("igris.core.devops_manager.DevOpsManager._load_registry"):
            with patch("igris.core.devops_manager.DevOpsManager.list_hosts", return_value=[]):
                resp = client.get("/api/devops/hosts")
        assert resp.status_code == 200
        assert "hosts" in resp.json()

    def test_register_host_via_api(self, tmp_path):
        client = _client()
        with patch("igris.core.devops_manager.DevOpsManager.register_host",
                   return_value={"registered": True, "hostname": "vps.test", "policy": "operator"}):
            resp = client.post("/api/devops/hosts", json={"hostname": "vps.test", "policy": "operator"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["registered"] is True

    def test_register_host_missing_hostname_422(self):
        client = _client()
        resp = client.post("/api/devops/hosts", json={"policy": "safe"})
        assert resp.status_code == 422

    def test_policy_check_endpoint(self):
        client = _client()
        with patch("igris.core.devops_manager.DevOpsManager.check_policy",
                   return_value={"allowed": True, "hostname": "vps.test", "action": "deploy",
                                  "policy": "operator", "reason": "ok", "allowed_actions": []}):
            resp = client.get("/api/devops/hosts/vps.test/policy?action=deploy")
        assert resp.status_code == 200
        assert "allowed" in resp.json()

    def test_remove_host_404_unknown(self):
        client = _client()
        with patch("igris.core.devops_manager.DevOpsManager.remove_host",
                   return_value={"removed": False, "error": "host not found", "hostname": "x"}):
            resp = client.delete("/api/devops/hosts/x")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# API endpoints — /api/devops/preflight, /api/devops/deploy, /api/devops/smoke
# ---------------------------------------------------------------------------

class TestDevOpsOperationalEndpoints:

    def _preflight_ok(self):
        return {
            "ok": True, "hostname": "localhost", "timestamp": 1.0,
            "checks": {
                "disk": {"ok": True, "used_pct": 40, "free_pct": 60},
                "git": {"ok": True, "clean": True, "dirty": False},
                "service": {"ok": True, "reachable": True, "port": 7778},
            },
        }

    def test_preflight_endpoint_returns_ok(self):
        client = _client()
        with patch("igris.core.devops_manager.DevOpsManager.run_preflight",
                   return_value=self._preflight_ok()):
            resp = client.post("/api/devops/preflight", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert "ok" in body
        assert "checks" in body

    def test_deploy_endpoint_dry_run(self):
        client = _client()
        with patch("igris.core.devops_manager.DevOpsManager.run_deploy",
                   return_value={"deployed": False, "dry_run": True,
                                  "strategy": "git_pull_restart", "preflight": self._preflight_ok(),
                                  "note": "dry_run", "hostname": "localhost", "timestamp": 1.0}):
            resp = client.post("/api/devops/deploy", json={"dry_run": True})
        assert resp.status_code == 200
        body = resp.json()
        assert body["dry_run"] is True
        assert "preflight" in body

    def test_smoke_endpoint_returns_ok_key(self):
        client = _client()
        with patch("igris.core.devops_manager.DevOpsManager.run_smoke_test",
                   return_value={"url": "http://localhost:7778/api/ping", "ok": True,
                                  "status_code": 200, "response_time_ms": 12,
                                  "body_preview": "pong", "timestamp": 1.0}):
            resp = client.get("/api/devops/smoke")
        assert resp.status_code == 200
        body = resp.json()
        assert "ok" in body
        assert "status_code" in body
        assert "response_time_ms" in body


# ---------------------------------------------------------------------------
# Existing endpoints sanity
# ---------------------------------------------------------------------------

class TestExistingDevOpsEndpoints:

    def test_health_endpoint_200(self):
        client = _client()
        import subprocess as _sp
        with patch("subprocess.run") as mock_run:
            # disk
            m1 = _mock_subprocess_ok(
                "Filesystem Size Used Avail Use% Mounted\n/dev/sda1 100G 40G 60G 40% /\n"
            )
            # memory
            m2 = _mock_subprocess_ok("Mem: 16G 8G 8G 0B 1G 7G\n")
            # nc igris port
            m3 = _mock_subprocess_ok("", returncode=0)
            mock_run.side_effect = [m1, m2, m3]
            resp = client.get("/api/devops/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert "checks" in body

    def test_deploy_status_endpoint_200(self):
        client = _client()
        with patch("subprocess.run") as mock_run:
            m1 = _mock_subprocess_ok("main")
            m2 = _mock_subprocess_ok("abc1234 commit message\n")
            m3 = _mock_subprocess_ok("")
            mock_run.side_effect = [m1, m2, m3]
            resp = client.get("/api/devops/deploy-status")
        assert resp.status_code == 200
        body = resp.json()
        assert "branch" in body or "error" in body  # either works
