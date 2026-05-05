"""Tests for Epic #41 — Real Local/Server Tool Runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from igris.core.tool_runtime import SSHHost, ToolResult, ToolRuntime


# ===========================================================================
# ToolResult
# ===========================================================================


class TestToolResult:
    def test_to_dict(self):
        r = ToolResult(tool="shell", action="ls", success=True, output="file.py")
        d = r.to_dict()
        assert d["tool"] == "shell"
        assert d["success"] is True
        assert d["redacted"] is True

    def test_secret_redacted(self):
        r = ToolResult(output="key=sk-1234567890abcdef1234567890abcdef")
        d = r.to_dict()
        assert "sk-" not in d["output"]

    def test_error_redacted(self):
        r = ToolResult(error="token=ghp_1234567890abcdefghij")
        d = r.to_dict()
        assert "ghp_" not in d["error"]


# ===========================================================================
# SSHHost
# ===========================================================================


class TestSSHHost:
    def test_to_dict(self):
        h = SSHHost(hostname="server1.example.com", policy="operator")
        d = h.to_dict()
        assert d["hostname"] == "server1.example.com"
        assert d["policy"] == "operator"

    def test_from_dict(self):
        h = SSHHost.from_dict({"hostname": "x", "allowed_services": ["nginx"]})
        assert h.hostname == "x"
        assert "nginx" in h.allowed_services

    def test_alias_defaults(self):
        h = SSHHost(hostname="host1")
        d = h.to_dict()
        assert d["alias"] == "host1"


# ===========================================================================
# Shell tool
# ===========================================================================


class TestShellTool:
    def test_allowed_command(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.shell_execute("git_status")
        # May fail if not a git repo, but should not be "not allowed"
        assert result.tool == "shell"
        assert "not allowed" not in result.error

    def test_blocked_command(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.shell_execute("rm_everything")
        assert result.success is False
        assert "not allowed" in result.error

    def test_mission_allowlist(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path))
        rt.set_mission_allowlist("m1", ["custom_cmd"])
        # still won't find binary, but won't be "not allowed"
        result = rt.shell_execute("custom_cmd", mission_id="m1")
        assert "not allowed" not in result.error


# ===========================================================================
# Filesystem tool
# ===========================================================================


class TestFilesystemTool:
    def test_read_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.fs_read(str(f))
        assert result.success is True
        assert "hello world" in result.output

    def test_read_secret_blocked(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("SECRET=x", encoding="utf-8")
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.fs_read(str(f))
        assert result.success is False
        assert result.risk_level == "critical"

    def test_read_outside_root(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.fs_read("/etc/passwd")
        assert result.success is False

    def test_write_file(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path))
        target = str(tmp_path / "new.txt")
        result = rt.fs_write(target, "new content")
        assert result.success is True
        assert Path(target).read_text() == "new content"

    def test_write_secret_file_blocked(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.fs_write(str(tmp_path / ".env"), "SECRET=x")
        assert result.success is False

    def test_write_secret_content_blocked(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.fs_write(
            str(tmp_path / "safe.txt"),
            "API_KEY=sk-1234567890abcdef1234567890abcdef",
        )
        assert result.success is False

    def test_write_with_backup(self, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("original", encoding="utf-8")
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.fs_write(str(f), "modified", backup=True)
        assert result.success is True
        assert result.rollback_id  # Should have created a rollback entry

    def test_diff_no_changes(self, tmp_path):
        f = tmp_path / "same.txt"
        f.write_text("same", encoding="utf-8")
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.fs_diff(str(f), "same")
        assert "No changes" in result.output

    def test_diff_with_changes(self, tmp_path):
        f = tmp_path / "diff.txt"
        f.write_text("line1\n", encoding="utf-8")
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.fs_diff(str(f), "line1\nline2\n")
        assert "+line2" in result.output

    def test_diff_new_file(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.fs_diff(str(tmp_path / "nonexistent.txt"), "new content")
        assert "New file" in result.output


# ===========================================================================
# Git tool
# ===========================================================================


class TestGitTool:
    def test_git_status(self, tmp_path):
        # Init a git repo
        os.system(f"cd {tmp_path} && git init -q && git commit --allow-empty -m init -q")
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.git_status()
        assert result.tool == "git"

    def test_git_log(self, tmp_path):
        os.system(f"cd {tmp_path} && git init -q && git commit --allow-empty -m init -q")
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.git_log(count=5)
        assert result.tool == "git"

    def test_git_branch(self, tmp_path):
        os.system(f"cd {tmp_path} && git init -q && git commit --allow-empty -m init -q")
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.git_branch()
        assert result.tool == "git"

    def test_git_commit_secret_message_blocked(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.git_commit("API_KEY=sk-1234567890abcdef1234567890abcdef")
        assert result.success is False

    def test_git_commit_secret_file_blocked(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.git_commit("fix", files=[".env"])
        assert result.success is False

    def test_git_push_no_approval(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.git_push(branch="feature/x")
        assert result.success is False
        assert "approval" in result.error.lower()

    def test_git_push_main_blocked(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.git_push(branch="main", approval_token="I_APPROVE_GITHUB_WRITE")
        assert result.success is False
        assert "forbidden" in result.error.lower()

    def test_git_push_master_blocked(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.git_push(branch="master", approval_token="I_APPROVE_GITHUB_WRITE")
        assert result.success is False


# ===========================================================================
# Docker tool
# ===========================================================================


class TestDockerTool:
    def test_docker_ps(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.docker_ps()
        assert result.tool == "docker"

    def test_docker_health(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.docker_health()
        assert result.tool == "docker"

    def test_docker_compose_down_risk_gated(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path), approval_mode="safe")
        result = rt.docker_compose_down()
        assert result.success is False  # Blocked in safe mode

    def test_docker_compose_up_risk_gated(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path), approval_mode="safe")
        result = rt.docker_compose_up()
        assert result.success is False


# ===========================================================================
# Nginx tool
# ===========================================================================


class TestNginxTool:
    def test_nginx_reload_blocked_in_safe(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path), approval_mode="safe")
        result = rt.nginx_reload()
        assert result.success is False


# ===========================================================================
# Systemd tool
# ===========================================================================


class TestSystemdTool:
    def test_systemd_restart_blocked_in_safe(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path), approval_mode="safe")
        result = rt.systemd_restart("nginx")
        assert result.success is False


# ===========================================================================
# HTTP check tool
# ===========================================================================


class TestHTTPCheckTool:
    def test_http_check_invalid_url(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.http_check("http://localhost:99999/nonexistent")
        assert result.success is False

    def test_http_check_result_structure(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.http_check("http://localhost:99999")
        d = result.to_dict()
        assert d["tool"] == "http"
        assert d["action"] == "check"


# ===========================================================================
# Test runner tool
# ===========================================================================


class TestRunnerTool:
    def test_run_tests(self, tmp_path):
        # Create a minimal test
        (tmp_path / "test_x.py").write_text("def test_pass(): assert True\n", encoding="utf-8")
        rt = ToolRuntime(project_root=str(tmp_path))
        result = rt.run_tests(args=[str(tmp_path / "test_x.py")], timeout=30)
        assert result.tool == "test"
        assert result.success is True


# ===========================================================================
# Host registry
# ===========================================================================


class TestHostRegistry:
    def test_register_and_list(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path))
        rt.register_host(SSHHost(hostname="server1", policy="operator"))
        hosts = rt.list_hosts()
        assert len(hosts) == 1
        assert hosts[0]["hostname"] == "server1"

    def test_get_host(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path))
        rt.register_host(SSHHost(hostname="server2"))
        h = rt.get_host("server2")
        assert h is not None

    def test_get_nonexistent(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path))
        assert rt.get_host("nope") is None


# ===========================================================================
# Risk gating integration
# ===========================================================================


class TestRiskGating:
    def test_safe_mode_blocks_high(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path), approval_mode="safe")
        result = rt.docker_compose_down()
        assert result.success is False
        assert "safe mode" in result.error.lower() or "risk" in result.error.lower()

    def test_operator_mode_with_rollback(self, tmp_path):
        rt = ToolRuntime(project_root=str(tmp_path), approval_mode="operator")
        # compose_down has no rollback → should still block
        result = rt.docker_compose_down()
        assert result.success is False


# ===========================================================================
# API integration
# ===========================================================================


class TestToolRuntimeAPI:
    @pytest.fixture
    def client(self):
        from igris.web.server import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        return TestClient(app)

    def test_list_tools(self, client):
        resp = client.get("/api/tools")
        assert resp.status_code == 200
        assert "tools" in resp.json()

    def test_shell_blocked(self, client):
        resp = client.post("/api/tools/shell/execute", json={"command_id": "rm_all"})
        assert resp.status_code == 200
        assert resp.json()["success"] is False

    def test_shell_git_status(self, client):
        resp = client.post("/api/tools/shell/execute", json={"command_id": "git_status"})
        assert resp.status_code == 200
        assert resp.json()["tool"] == "shell"

    def test_fs_read_secret_blocked(self, client):
        resp = client.post("/api/tools/fs/read", json={"path": ".env"})
        assert resp.json()["success"] is False

    def test_fs_diff(self, client):
        resp = client.post("/api/tools/fs/diff", json={"path": "/tmp/nofile.txt", "new_content": "x"})
        assert resp.status_code == 200

    def test_git_status(self, client):
        resp = client.get("/api/tools/git/status")
        assert resp.status_code == 200
        assert resp.json()["tool"] == "git"

    def test_git_diff(self, client):
        resp = client.get("/api/tools/git/diff")
        assert resp.status_code == 200

    def test_git_log(self, client):
        resp = client.get("/api/tools/git/log")
        assert resp.status_code == 200

    def test_git_branch(self, client):
        resp = client.get("/api/tools/git/branch")
        assert resp.status_code == 200

    def test_http_check(self, client):
        resp = client.post("/api/tools/http/check", json={"url": "http://localhost:99999"})
        assert resp.status_code == 200
        assert resp.json()["success"] is False

    def test_hosts_list(self, client):
        resp = client.get("/api/tools/hosts")
        assert resp.status_code == 200

    def test_host_register(self, client):
        resp = client.post("/api/tools/hosts/register", json={
            "hostname": "test-server", "policy": "safe",
        })
        assert resp.status_code == 200
        assert resp.json()["registered"]["hostname"] == "test-server"
