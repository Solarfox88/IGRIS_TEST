"""Tests for UI Operations Polish (Sprint 9).

Verifies HTML structure, JS doesn't call missing endpoints,
no free shell input, aria labels present, and tab structure.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from igris.web.server import create_app


@pytest.fixture
def client(tmp_path):
    from igris.models.config import CONFIG
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    (root / ".igris" / "tasks").mkdir(parents=True)
    (root / ".igris" / "timeline").mkdir(parents=True)
    (root / ".igris" / "missions").mkdir(parents=True)
    (root / ".igris" / "memory").mkdir(parents=True)
    (root / ".igris" / "a2a" / "tasks").mkdir(parents=True)
    os.environ["PROJECT_ROOT"] = str(root)
    os.environ["WORKSPACE_ROOT"] = str(root)
    CONFIG.project_root = Path(str(root))
    return TestClient(create_app())


class TestTabStructure:
    def test_all_tabs_present(self, client):
        r = client.get("/")
        assert r.status_code == 200
        html = r.text
        expected_tabs = [
            "mission", "terminal", "files", "git", "tests", "logs",
            "agent", "tasks", "safety", "cost", "a2a", "memory", "loop", "patches",
        ]
        for tab in expected_tabs:
            assert f'data-tab="{tab}"' in html, f"Tab '{tab}' missing"

    def test_aria_labels_on_tabs(self, client):
        r = client.get("/")
        html = r.text
        assert html.count('role="tab"') >= 14

    def test_tab_panes_present(self, client):
        r = client.get("/")
        html = r.text
        expected_panes = [
            "tab-mission", "tab-terminal", "tab-files", "tab-git",
            "tab-tests", "tab-logs", "tab-agent", "tab-tasks",
            "tab-safety", "tab-cost", "tab-a2a", "tab-memory",
            "tab-loop", "tab-patches",
        ]
        for pane in expected_panes:
            assert f'id="{pane}"' in html, f"Pane '{pane}' missing"


class TestNoFreeShell:
    def test_no_shell_input_in_html(self, client):
        r = client.get("/")
        html = r.text
        assert "shell" not in html.lower() or "free shell" not in html.lower()

    def test_no_push_endpoint(self, client):
        r = client.post("/api/git/push")
        assert r.status_code == 404 or r.status_code == 405


class TestRefreshButtons:
    def test_safety_refresh(self, client):
        r = client.get("/")
        assert 'id="btn-refresh-safety"' in r.text

    def test_cost_refresh(self, client):
        r = client.get("/")
        assert 'id="btn-refresh-cost"' in r.text

    def test_a2a_refresh(self, client):
        r = client.get("/")
        assert 'id="btn-refresh-a2a"' in r.text

    def test_mission_refresh(self, client):
        r = client.get("/")
        assert 'id="btn-refresh-missions"' in r.text

    def test_memory_refresh(self, client):
        r = client.get("/")
        assert 'id="btn-refresh-memory"' in r.text

    def test_loop_refresh(self, client):
        r = client.get("/")
        assert 'id="btn-refresh-loop"' in r.text

    def test_timeline_refresh(self, client):
        r = client.get("/")
        assert 'id="btn-refresh-timeline"' in r.text


class TestCostTabEnhanced:
    def test_availability_section(self, client):
        r = client.get("/")
        assert 'id="cost-availability"' in r.text

    def test_budget_section(self, client):
        r = client.get("/")
        assert 'id="cost-budget"' in r.text

    def test_estimate_section(self, client):
        r = client.get("/")
        assert 'id="cost-estimate"' in r.text

    def test_estimate_button(self, client):
        r = client.get("/")
        assert 'id="btn-estimate-route"' in r.text


class TestA2ATabEnhanced:
    def test_a2a_tasks_section(self, client):
        r = client.get("/")
        assert 'id="a2a-tasks"' in r.text


class TestJsNoMissingEndpoints:
    def test_static_js_served(self, client):
        r = client.get("/static/js/app.js")
        assert r.status_code == 200

    def test_js_no_free_shell(self, client):
        r = client.get("/static/js/app.js")
        js = r.text
        assert "/api/shell" not in js
        assert "exec(" not in js or "function exec" not in js

    def test_css_served(self, client):
        r = client.get("/static/css/style.css")
        assert r.status_code == 200


class TestMobileCSS:
    def test_css_has_media_queries(self, client):
        r = client.get("/static/css/style.css")
        css = r.text
        assert "@media" in css
        assert "max-width:768px" in css
        assert "max-width:480px" in css

    def test_no_horizontal_overflow_class(self, client):
        r = client.get("/static/css/style.css")
        css = r.text
        assert "overflow-x:auto" in css
