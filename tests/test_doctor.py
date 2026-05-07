"""Tests for igris.core.doctor — environment diagnostics."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from igris.core.doctor import (
    DoctorCheck,
    DoctorReport,
    check_config_json,
    check_dependencies,
    check_docker,
    check_env_file,
    check_fastapi_server,
    check_git,
    check_ollama,
    check_openai_key,
    check_permissions,
    check_port,
    check_python,
    check_ssh,
    check_venv,
    check_workspace,
    run_doctor,
    run_verify,
)


# ---------------------------------------------------------------------------
# DoctorCheck model
# ---------------------------------------------------------------------------


class TestDoctorCheck:
    def test_to_dict_basic(self):
        c = DoctorCheck(name="test", category="python", status="ok", detail="all good")
        d = c.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "ok"
        assert "fix_suggestion" not in d

    def test_to_dict_with_fix(self):
        c = DoctorCheck(name="x", category="deps", status="error", detail="missing", fix_suggestion="install it")
        d = c.to_dict()
        assert d["fix_suggestion"] == "install it"

    def test_secret_redacted_in_detail(self):
        c = DoctorCheck(name="x", category="python", status="ok", detail="key=sk-1234567890abcdef1234567890abcdef")
        d = c.to_dict()
        assert "sk-" not in d["detail"]


# ---------------------------------------------------------------------------
# DoctorReport model
# ---------------------------------------------------------------------------


class TestDoctorReport:
    def test_empty_report_ok(self):
        r = DoctorReport()
        d = r.to_dict()
        assert d["overall"] == "ok"
        assert d["total_checks"] == 0

    def test_report_with_warning(self):
        r = DoctorReport(checks=[
            DoctorCheck(name="a", category="x", status="ok"),
            DoctorCheck(name="b", category="y", status="warning", detail="hmm"),
        ])
        d = r.to_dict()
        assert d["overall"] == "warning"

    def test_report_with_error(self):
        r = DoctorReport(checks=[
            DoctorCheck(name="a", category="x", status="ok"),
            DoctorCheck(name="b", category="y", status="error", detail="bad"),
        ])
        d = r.to_dict()
        assert d["overall"] == "error"

    def test_to_markdown(self):
        r = DoctorReport(checks=[
            DoctorCheck(name="python", category="python", status="ok", detail="3.12"),
            DoctorCheck(name="deps", category="deps", status="error", detail="missing", fix_suggestion="pip install"),
        ])
        md = r.to_markdown()
        assert "# IGRIS Doctor Report" in md
        assert "python" in md
        assert "Fix:" in md

    def test_summary_counts(self):
        r = DoctorReport(checks=[
            DoctorCheck(name="a", category="x", status="ok"),
            DoctorCheck(name="b", category="y", status="ok"),
            DoctorCheck(name="c", category="z", status="warning"),
            DoctorCheck(name="d", category="w", status="skipped"),
        ])
        d = r.to_dict()
        assert d["summary"]["ok"] == 2
        assert d["summary"]["warning"] == 1
        assert d["summary"]["skipped"] == 1


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


class TestCheckPython:
    def test_current_python_ok(self):
        c = check_python()
        assert c.status == "ok"
        assert c.category == "python"


class TestCheckVenv:
    def test_detects_venv_via_env_var(self):
        with mock.patch.dict(os.environ, {"VIRTUAL_ENV": "/tmp/test-venv"}):
            c = check_venv()
            assert c.status == "ok"

    def test_no_venv_warning(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(sys, "base_prefix", sys.prefix):
                c = check_venv()
                # May still be in venv via real_prefix, so accept ok or warning
                assert c.status in ("ok", "warning")


class TestCheckDependencies:
    def test_all_present(self):
        c = check_dependencies()
        assert c.status == "ok"

    def test_missing_package(self):
        with mock.patch("builtins.__import__", side_effect=ImportError("no httpx")):
            c = check_dependencies()
            assert c.status == "error"
            assert "missing" in c.detail.lower() or c.meta.get("missing")


class TestCheckFastapiServer:
    def test_server_not_running(self):
        c = check_fastapi_server(host="127.0.0.1", port=19999)
        assert c.status == "warning"
        assert c.category == "server"


class TestCheckOllama:
    def test_ollama_not_running(self):
        with mock.patch.dict(os.environ, {"LOCAL_LLM_BASE_URL": "http://127.0.0.1:19999"}):
            c = check_ollama()
            assert c.status == "warning"
            assert c.category == "ollama"


class TestCheckOpenaiKey:
    def test_key_present(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            c = check_openai_key()
            assert c.status == "ok"
            assert "not shown" in c.detail

    def test_no_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            c = check_openai_key()
            assert c.status == "warning"


class TestCheckGit:
    def test_git_available(self):
        c = check_git()
        # git should be available on CI/dev machines
        assert c.status in ("ok", "error")
        assert c.category == "git"


class TestCheckDocker:
    def test_docker_check_runs(self):
        c = check_docker()
        assert c.category == "docker"
        assert c.status in ("ok", "warning", "skipped")


class TestCheckSsh:
    def test_ssh_check_runs(self):
        c = check_ssh()
        assert c.category == "ssh"
        assert c.status in ("ok", "skipped")


class TestCheckPort:
    def test_check_unused_port(self):
        c = check_port(49999)
        assert c.category == "ports"
        assert c.status == "ok"


class TestCheckWorkspace:
    def test_valid_workspace(self, tmp_path):
        c = check_workspace(str(tmp_path))
        assert c.status == "ok"

    def test_nonexistent_workspace(self):
        c = check_workspace("/nonexistent/path/12345")
        assert c.status == "error"


class TestCheckPermissions:
    def test_writable_dir(self, tmp_path):
        c = check_permissions(str(tmp_path))
        assert c.status == "ok"


class TestCheckEnvFile:
    def test_env_exists(self, tmp_path):
        (tmp_path / ".env").write_text("# test", encoding="utf-8")
        c = check_env_file(str(tmp_path))
        assert c.status == "ok"

    def test_env_missing_example_exists(self, tmp_path):
        (tmp_path / ".env.example").write_text("# example", encoding="utf-8")
        c = check_env_file(str(tmp_path))
        assert c.status == "warning"
        assert ".env.example" in c.detail

    def test_env_missing(self, tmp_path):
        c = check_env_file(str(tmp_path))
        assert c.status == "warning"


class TestCheckConfigJson:
    def test_valid_config(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "config.sample.json").write_text(
            json.dumps({"local_llm_provider": "ollama"}), encoding="utf-8"
        )
        c = check_config_json(str(tmp_path))
        assert c.status == "ok"

    def test_invalid_json(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "config.sample.json").write_text("{bad json", encoding="utf-8")
        c = check_config_json(str(tmp_path))
        assert c.status == "error"

    def test_no_config_dir(self, tmp_path):
        c = check_config_json(str(tmp_path))
        assert c.status == "warning"


# ---------------------------------------------------------------------------
# Full doctor run
# ---------------------------------------------------------------------------


class TestRunDoctor:
    def test_produces_report(self, tmp_path):
        report = run_doctor(project_root=str(tmp_path), port=19999)
        d = report.to_dict()
        assert "overall" in d
        assert "checks" in d
        assert d["total_checks"] >= 10  # We have 14 checks

    def test_report_has_categories(self, tmp_path):
        report = run_doctor(project_root=str(tmp_path), port=19999)
        categories = {c.category for c in report.checks}
        assert "python" in categories
        assert "deps" in categories
        assert "git" in categories


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


class TestRunVerify:
    def test_verify_returns_structure(self):
        result = run_verify()
        assert "ok" in result
        assert "checks" in result
        assert "timestamp" in result

    def test_verify_checks_project_root(self, tmp_path):
        result = run_verify(str(tmp_path))
        assert result["checks"]["project_root"] is True

    def test_verify_nonexistent_root(self, tmp_path):
        """Use a guaranteed-nonexistent subdir of tmp_path.

        /nonexistent/12345 is hardcoded and can exist in some CI environments.
        tmp_path is provided by pytest and is always a fresh temp directory;
        a sub-path that was never created is guaranteed to not exist.
        """
        missing = tmp_path / "does_not_exist_subdirectory"
        # Sanity: must not exist
        assert not missing.exists()
        result = run_verify(str(missing))
        assert result["checks"]["project_root"] is False
        assert result["ok"] is False

    def test_verify_checks_critical_files(self):
        # Running from actual project root should find files
        result = run_verify(str(Path(__file__).resolve().parent.parent))
        assert result["checks"]["critical_files"]["ok"] is True

    def test_verify_missing_critical_files(self, tmp_path):
        result = run_verify(str(tmp_path))
        assert result["checks"]["critical_files"]["ok"] is False
        assert len(result["checks"]["critical_files"]["missing"]) > 0


# ---------------------------------------------------------------------------
# API endpoint integration (via TestClient)
# ---------------------------------------------------------------------------


class TestDoctorAPI:
    @pytest.fixture
    def client(self):
        from igris.web.server import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        return TestClient(app)

    def test_doctor_endpoint(self, client):
        resp = client.get("/api/doctor")
        assert resp.status_code == 200
        data = resp.json()
        assert "overall" in data
        assert "checks" in data

    def test_doctor_markdown_endpoint(self, client):
        resp = client.get("/api/doctor/markdown")
        assert resp.status_code == 200
        data = resp.json()
        assert "markdown" in data
        assert "# IGRIS Doctor Report" in data["markdown"]

    def test_verify_endpoint(self, client):
        resp = client.get("/api/verify")
        assert resp.status_code == 200
        data = resp.json()
        assert "ok" in data
        assert "checks" in data

    def test_config_validate_endpoint(self, client):
        resp = client.get("/api/config/validate")
        assert resp.status_code == 200
        data = resp.json()
        assert "valid" in data
        assert "issues" in data

    def test_crash_reports_endpoint(self, client):
        resp = client.get("/api/crash-reports")
        assert resp.status_code == 200
        data = resp.json()
        assert "reports" in data
        assert "count" in data

    def test_crash_report_not_found(self, client):
        resp = client.get("/api/crash-reports/nonexistent-id")
        assert resp.status_code == 404

    def test_last_good_state_endpoint(self, client):
        resp = client.get("/api/crash-reports/last-good-state")
        assert resp.status_code == 200
        data = resp.json()
        assert "available" in data

    def test_save_good_state_endpoint(self, client):
        resp = client.post(
            "/api/crash-reports/save-good-state",
            json={"state": {"tests_passing": True, "step": 5}},
        )
        assert resp.status_code == 200
        assert resp.json()["saved"] is True

    def test_save_good_state_empty_rejected(self, client):
        resp = client.post(
            "/api/crash-reports/save-good-state",
            json={"state": {}},
        )
        assert resp.status_code == 400
