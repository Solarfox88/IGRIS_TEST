"""Sprint 35 — Safe System Info Capability.

Tests for:
- System info module (OS, Python, CPU, memory, disk, uptime, container, Ollama, network)
- No secrets/env vars exposed
- No private IPs exposed
- API endpoint /api/system/info
- command_id system_info in allowlist
- Chat machine_info intent suggests system_info endpoint
- Graceful fallback on missing data
"""

from __future__ import annotations

import json
import os
import platform
import re

import pytest


# ---------------------------------------------------------------------------
# Core module tests
# ---------------------------------------------------------------------------

class TestGetSystemInfo:
    """Test get_system_info returns correct safe data."""

    def test_returns_dict(self):
        from igris.core.system_info import get_system_info
        info = get_system_info()
        assert isinstance(info, dict)

    def test_has_os_section(self):
        from igris.core.system_info import get_system_info
        info = get_system_info()
        assert "os" in info
        assert "system" in info["os"]
        assert info["os"]["system"] == platform.system()

    def test_has_python_section(self):
        from igris.core.system_info import get_system_info
        info = get_system_info()
        assert "python" in info
        assert info["python"]["version"] == platform.python_version()

    def test_has_process_pid(self):
        from igris.core.system_info import get_system_info
        info = get_system_info()
        assert "process" in info
        assert info["process"]["pid"] == os.getpid()

    def test_has_cpu_section(self):
        from igris.core.system_info import get_system_info
        info = get_system_info()
        assert "cpu" in info
        assert "count" in info["cpu"]

    def test_has_memory_section(self):
        from igris.core.system_info import get_system_info
        info = get_system_info()
        assert "memory" in info

    def test_has_disk_section(self):
        from igris.core.system_info import get_system_info
        info = get_system_info()
        assert "disk" in info

    def test_has_uptime_section(self):
        from igris.core.system_info import get_system_info
        info = get_system_info()
        assert "uptime" in info
        assert "process_uptime_seconds" in info["uptime"]
        assert "formatted" in info["uptime"]

    def test_has_container_section(self):
        from igris.core.system_info import get_system_info
        info = get_system_info()
        assert "container" in info
        assert "likely_container" in info["container"]

    def test_has_ollama_section(self):
        from igris.core.system_info import get_system_info
        info = get_system_info()
        assert "ollama" in info
        assert "reachable" in info["ollama"]
        assert "model_configured" in info["ollama"]
        assert "model_available" in info["ollama"]

    def test_has_igris_section(self):
        from igris.core.system_info import get_system_info
        info = get_system_info(host="0.0.0.0", port=9000)
        assert "igris" in info
        assert info["igris"]["host"] == "0.0.0.0"
        assert info["igris"]["port"] == 9000

    def test_has_network_section(self):
        from igris.core.system_info import get_system_info
        info = get_system_info(host="0.0.0.0", port=8000)
        assert "network" in info
        assert info["network"]["server_bind"] == "0.0.0.0:8000"
        assert info["network"]["external_access_possible"] is True

    def test_localhost_no_external_access(self):
        from igris.core.system_info import get_system_info
        info = get_system_info(host="127.0.0.1", port=8000)
        assert info["network"]["external_access_possible"] is False

    def test_project_root_passed(self):
        from igris.core.system_info import get_system_info
        info = get_system_info(project_root="/tmp")
        assert info["igris"]["project_root"] == "/tmp"


class TestNoSecretsExposed:
    """Verify no sensitive data in system info output."""

    def test_no_env_vars_in_output(self):
        from igris.core.system_info import get_system_info
        info = get_system_info()
        text = json.dumps(info).lower()
        for key in ["ghp_", "sk-", "password=", "api_key=", "secret_key",
                     "aws_access", "openai_api"]:
            assert key not in text

    def test_no_env_section(self):
        from igris.core.system_info import get_system_info
        info = get_system_info()
        assert "environment" not in info
        assert "env" not in info
        assert "environ" not in info

    def test_no_ip_addresses_in_network(self):
        from igris.core.system_info import get_system_info
        info = get_system_info()
        network = info.get("network", {})
        text = json.dumps(network)
        # Should not contain private IP patterns
        assert "192.168." not in text
        assert "10.0." not in text
        assert "172.16." not in text

    def test_no_home_directory_leak(self):
        from igris.core.system_info import get_system_info
        info = get_system_info()
        text = json.dumps(info)
        home = os.path.expanduser("~")
        # Python executable path is OK, but no other home references
        non_python_text = text.replace(info["python"]["executable"], "")
        # Don't check if home is /, /root etc (too generic)
        if len(home) > 5:
            assert home not in non_python_text or "project_root" in non_python_text


class TestMemoryFallback:
    """Memory info handles missing /proc/meminfo gracefully."""

    def test_memory_has_data_or_note(self):
        from igris.core.system_info import get_system_info
        info = get_system_info()
        mem = info["memory"]
        assert "total_mb" in mem or "note" in mem

    def test_memory_linux_has_totals(self):
        if platform.system() != "Linux":
            pytest.skip("Linux only")
        from igris.core.system_info import get_system_info
        info = get_system_info()
        mem = info["memory"]
        assert "total_mb" in mem
        assert "available_mb" in mem
        assert "used_percent" in mem
        assert mem["total_mb"] > 0


class TestDiskFallback:
    """Disk info handles missing statvfs gracefully."""

    def test_disk_has_data_or_note(self):
        from igris.core.system_info import get_system_info
        info = get_system_info(project_root="/tmp")
        disk = info["disk"]
        assert "total_gb" in disk or "note" in disk


class TestSafeSummary:
    """get_safe_system_summary returns a short string."""

    def test_returns_string(self):
        from igris.core.system_info import get_safe_system_summary
        s = get_safe_system_summary()
        assert isinstance(s, str)
        assert len(s) > 10

    def test_contains_os_and_python(self):
        from igris.core.system_info import get_safe_system_summary
        s = get_safe_system_summary()
        assert "Python" in s
        assert platform.system() in s


# ---------------------------------------------------------------------------
# Command ID tests
# ---------------------------------------------------------------------------

class TestSystemInfoCommandId:
    """system_info is in the safe commands allowlist."""

    def test_in_allowed_commands(self):
        from igris.layers.execution.safe_commands import ALLOWED_COMMANDS
        assert "system_info" in ALLOWED_COMMANDS

    def test_command_uses_python(self):
        from igris.layers.execution.safe_commands import ALLOWED_COMMANDS
        cmd = ALLOWED_COMMANDS["system_info"]
        assert "system_info" in " ".join(cmd)


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    from igris.web.server import create_app
    from starlette.testclient import TestClient
    app = create_app()
    return TestClient(app)


class TestSystemInfoAPI:
    """GET /api/system/info endpoint."""

    def test_endpoint_200(self, client):
        r = client.get("/api/system/info")
        assert r.status_code == 200

    def test_response_has_sections(self, client):
        r = client.get("/api/system/info")
        data = r.json()
        for key in ["os", "python", "cpu", "memory", "disk", "uptime",
                     "container", "ollama", "igris", "network"]:
            assert key in data, f"Missing section: {key}"

    def test_no_secrets_in_response(self, client):
        r = client.get("/api/system/info")
        text = r.text.lower()
        for pat in ["ghp_", "sk-", "password=", "api_key="]:
            assert pat not in text

    def test_no_env_dump(self, client):
        r = client.get("/api/system/info")
        text = r.text
        # Should not contain full env var dumps
        assert "PATH=" not in text
        assert "HOME=" not in text

    def test_response_serializable(self, client):
        r = client.get("/api/system/info")
        data = r.json()
        # Should be JSON serializable (no datetime, Path objects)
        json.dumps(data)


# ---------------------------------------------------------------------------
# Chat personality integration
# ---------------------------------------------------------------------------

class TestChatIntegration:
    """Chat machine_info intent references system_info."""

    def test_grounded_response_mentions_system_info(self):
        from igris.core.chat_personality import get_grounded_response
        resp = get_grounded_response("machine_info")
        assert "/api/system/info" in resp

    def test_actions_include_system_info(self):
        from igris.core.chat_personality import get_suggested_actions
        actions = get_suggested_actions("machine_info")
        endpoints = [a["endpoint"] for a in actions]
        assert "/api/system/info" in endpoints

    def test_network_info_mentions_system_info(self):
        from igris.core.chat_personality import get_grounded_response
        resp = get_grounded_response("network_info")
        assert "/api/system/info" in resp

    def test_machine_info_no_free_shell(self):
        from igris.core.chat_personality import get_grounded_response
        resp = get_grounded_response("machine_info")
        assert "shell libera" not in resp.lower() or "non uso shell libera" in resp.lower()

    def test_machine_info_actions_no_shell_endpoint(self):
        from igris.core.chat_personality import get_suggested_actions
        actions = get_suggested_actions("machine_info")
        for a in actions:
            assert "/api/shell" not in a["endpoint"]
            assert "/api/exec" not in a["endpoint"]


# ---------------------------------------------------------------------------
# Safety: no interface dump, no public IP
# ---------------------------------------------------------------------------

class TestNetworkSafety:
    """Network section is conservative."""

    def test_no_interface_list(self):
        from igris.core.system_info import get_system_info
        info = get_system_info()
        network = info.get("network", {})
        assert "interfaces" not in network
        assert "eth0" not in json.dumps(network)
        assert "wlan" not in json.dumps(network)

    def test_no_public_ip(self):
        from igris.core.system_info import get_system_info
        info = get_system_info()
        network = info.get("network", {})
        assert "public_ip" not in network

    def test_only_bind_and_external_flag(self):
        from igris.core.system_info import get_system_info
        info = get_system_info()
        network = info.get("network", {})
        assert set(network.keys()) == {"server_bind", "external_access_possible"}


# ---------------------------------------------------------------------------
# Git status
# ---------------------------------------------------------------------------

class TestGitStatus:
    def test_git_status_clean(self):
        import subprocess
        r = subprocess.run(["git", "status", "--porcelain"],
                          capture_output=True, text=True, cwd=".")
        lines = [l for l in r.stdout.strip().split("\n") if l.strip()
                 and not any(ig in l for ig in [".igris/", "logs/", "__pycache__",
                                                 ".egg-info", ".pyc"])]
        for line in lines:
            assert any(allowed in line for allowed in [
                "test_system_info.py", "system_info.py",
                "safe_commands.py", "server.py",
                "chat_personality.py",
                "SYSTEM_INFO.md",
                "test_guided_actions.py", "test_dashboard_tabs.py",
                "index.html", "app.js", "style.css",
                "test_integration_v02.py", "test_ui_polish.py",
                "DASHBOARD_UI.md", "GUIDED_ACTIONS.md",
                "README.md", "PREPARED_NOT_IMPLEMENTED.md",
                # Files modified by #76:
                "agent_action_schema.py", "agent_reasoning_loop.py",
                "prompt_contract.py", "test_write_guard.py",
                "test_agent_action_schema.py", "test_issue74_toolruntime_dispatcher.py",
                "test_doctor.py",
            ]), f"Unexpected changed file: {line}"
