"""Sprint 34 — Information Architecture: Simplify Tabs and Mission Control Dashboard.

Tests for:
- Tab reduction: 14 tabs → 7 primary tabs
- Sub-tab navigation within grouped tabs
- Dashboard grid with health/readiness/diagnostics/loop cards
- Dashboard summary endpoint /api/dashboard/summary
- All original functionality preserved in new grouping
- CSS for dashboard grid, sub-tabs, responsive
- JS sub-tab switching, dashboard extras loading
- No broken element IDs (all original IDs still present)
"""

from __future__ import annotations

import re

import pytest


# ---------------------------------------------------------------------------
# HTML structure tests
# ---------------------------------------------------------------------------

class TestTabReduction:
    """14 tabs reduced to 7 primary tabs."""

    def test_only_7_primary_tabs(self):
        import pathlib
        html = pathlib.Path("igris/web/templates/index.html").read_text()
        tabs = re.findall(r'data-tab="([^"]+)"', html)
        assert len(tabs) == 7
        expected = {"dashboard", "code", "tasks", "terminal", "memory", "safety", "advanced"}
        assert set(tabs) == expected

    def test_no_old_standalone_tabs(self):
        import pathlib
        html = pathlib.Path("igris/web/templates/index.html").read_text()
        tabs = re.findall(r'data-tab="([^"]+)"', html)
        old_tabs = {"mission", "files", "git", "tests", "logs", "agent", "cost", "a2a", "loop", "patches"}
        for t in old_tabs:
            assert t not in tabs, f"Old tab '{t}' still exists as primary"


class TestSubTabs:
    """Grouped tabs have sub-tab navigation."""

    def test_code_has_subtabs(self):
        import pathlib
        html = pathlib.Path("igris/web/templates/index.html").read_text()
        assert 'data-subtab="code-files"' in html
        assert 'data-subtab="code-git"' in html
        assert 'data-subtab="code-patches"' in html

    def test_tasks_has_subtabs(self):
        import pathlib
        html = pathlib.Path("igris/web/templates/index.html").read_text()
        assert 'data-subtab="tasks-list"' in html
        assert 'data-subtab="tasks-loop"' in html

    def test_terminal_has_subtabs(self):
        import pathlib
        html = pathlib.Path("igris/web/templates/index.html").read_text()
        assert 'data-subtab="terminal-cmd"' in html
        assert 'data-subtab="terminal-tests"' in html

    def test_memory_has_subtabs(self):
        import pathlib
        html = pathlib.Path("igris/web/templates/index.html").read_text()
        assert 'data-subtab="memory-data"' in html
        assert 'data-subtab="memory-timeline"' in html

    def test_safety_has_subtabs(self):
        import pathlib
        html = pathlib.Path("igris/web/templates/index.html").read_text()
        assert 'data-subtab="safety-status"' in html
        assert 'data-subtab="safety-cost"' in html

    def test_advanced_has_subtabs(self):
        import pathlib
        html = pathlib.Path("igris/web/templates/index.html").read_text()
        assert 'data-subtab="adv-a2a"' in html
        assert 'data-subtab="adv-logs"' in html


class TestDashboardStructure:
    """Dashboard has grid cards and enhanced content."""

    def test_dashboard_tab_exists(self):
        import pathlib
        html = pathlib.Path("igris/web/templates/index.html").read_text()
        assert 'id="tab-dashboard"' in html

    def test_dashboard_grid(self):
        import pathlib
        html = pathlib.Path("igris/web/templates/index.html").read_text()
        assert "dashboard-grid" in html
        assert "dash-card" in html

    def test_dashboard_health_card(self):
        import pathlib
        html = pathlib.Path("igris/web/templates/index.html").read_text()
        assert 'id="dash-health"' in html
        assert "System Health" in html

    def test_dashboard_readiness_card(self):
        import pathlib
        html = pathlib.Path("igris/web/templates/index.html").read_text()
        assert 'id="dash-readiness"' in html

    def test_dashboard_diagnostics_card(self):
        import pathlib
        html = pathlib.Path("igris/web/templates/index.html").read_text()
        assert 'id="dash-diagnostics"' in html
        assert "dash-diagnostics-summary" in html

    def test_dashboard_loop_card(self):
        import pathlib
        html = pathlib.Path("igris/web/templates/index.html").read_text()
        assert 'id="dash-loop-summary"' in html
        assert "dash-loop-info" in html

    def test_dashboard_has_decision_reports(self):
        import pathlib
        html = pathlib.Path("igris/web/templates/index.html").read_text()
        assert 'id="dash-reports"' in html
        assert "Decision Reports" in html

    def test_dashboard_has_missions(self):
        import pathlib
        html = pathlib.Path("igris/web/templates/index.html").read_text()
        assert 'id="mission-list"' in html
        assert 'id="mission-form"' in html


class TestOriginalElementsPreserved:
    """All original element IDs still exist for backward compatibility."""

    REQUIRED_IDS = [
        "mission-health", "mission-readiness", "mission-context",
        "mission-list", "mission-detail", "mission-graph",
        "terminal-commands", "terminal-output",
        "file-tree", "file-preview",
        "git-info", "git-branches", "git-diff", "git-safety",
        "git-commit-proposal", "git-pr-summary",
        "test-output",
        "logs-output",
        "timeline-list",
        "task-list", "teacher-output",
        "safety-info", "reports-list",
        "cost-availability", "cost-budget", "cost-summary",
        "cost-estimate", "routing-explain",
        "a2a-card", "a2a-capabilities", "a2a-tasks",
        "memory-constraints", "memory-decisions", "memory-failures",
        "loop-status", "loop-recent",
        "patches-list", "patch-detail", "patch-diff", "patch-actions",
    ]

    @pytest.mark.parametrize("elem_id", REQUIRED_IDS)
    def test_element_id_preserved(self, elem_id):
        import pathlib
        html = pathlib.Path("igris/web/templates/index.html").read_text()
        assert f'id="{elem_id}"' in html, f"Element #{elem_id} missing from HTML"


# ---------------------------------------------------------------------------
# CSS tests
# ---------------------------------------------------------------------------

class TestDashboardCSS:
    """CSS supports dashboard grid and sub-tabs."""

    def test_dashboard_grid_css(self):
        import pathlib
        css = pathlib.Path("igris/web/static/css/style.css").read_text()
        assert ".dashboard-grid" in css
        assert "grid-template-columns" in css

    def test_dash_card_css(self):
        import pathlib
        css = pathlib.Path("igris/web/static/css/style.css").read_text()
        assert ".dash-card" in css

    def test_sub_tab_css(self):
        import pathlib
        css = pathlib.Path("igris/web/static/css/style.css").read_text()
        assert ".sub-tab-bar" in css
        assert ".sub-tab" in css
        assert ".sub-tab.active" in css
        assert ".sub-tab-pane" in css

    def test_sub_tab_pane_hidden_by_default(self):
        import pathlib
        css = pathlib.Path("igris/web/static/css/style.css").read_text()
        assert ".sub-tab-pane{display:none}" in css

    def test_sub_tab_pane_active_visible(self):
        import pathlib
        css = pathlib.Path("igris/web/static/css/style.css").read_text()
        assert ".sub-tab-pane.active{display:block}" in css

    def test_mobile_dashboard_single_column(self):
        import pathlib
        css = pathlib.Path("igris/web/static/css/style.css").read_text()
        assert ".dashboard-grid{grid-template-columns:1fr}" in css


# ---------------------------------------------------------------------------
# JS tests
# ---------------------------------------------------------------------------

class TestDashboardJS:
    """JS handles sub-tab switching and dashboard extras."""

    def test_js_has_subtab_handler(self):
        import pathlib
        js = pathlib.Path("igris/web/static/js/app.js").read_text()
        assert ".sub-tab" in js
        assert "data-subtab" in js or "dataset.subtab" in js

    def test_js_loads_dashboard_extras(self):
        import pathlib
        js = pathlib.Path("igris/web/static/js/app.js").read_text()
        assert "loadDashboardExtras" in js

    def test_js_populates_diagnostics(self):
        import pathlib
        js = pathlib.Path("igris/web/static/js/app.js").read_text()
        assert "dash-diagnostics-summary" in js

    def test_js_populates_loop_info(self):
        import pathlib
        js = pathlib.Path("igris/web/static/js/app.js").read_text()
        assert "dash-loop-info" in js

    def test_js_populates_decision_reports(self):
        import pathlib
        js = pathlib.Path("igris/web/static/js/app.js").read_text()
        assert "dash-reports" in js

    def test_js_auto_refresh_uses_dashboard(self):
        import pathlib
        js = pathlib.Path("igris/web/static/js/app.js").read_text()
        assert '"dashboard"' in js


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    from igris.web.server import create_app
    from starlette.testclient import TestClient
    app = create_app()
    return TestClient(app)


class TestDashboardAPI:
    """Dashboard summary endpoint."""

    def test_dashboard_summary_200(self, client):
        r = client.get("/api/dashboard/summary")
        assert r.status_code == 200

    def test_dashboard_summary_has_sections(self, client):
        r = client.get("/api/dashboard/summary")
        data = r.json()
        assert "health" in data
        assert "diagnostics" in data
        assert "loop" in data
        assert "tab_layout" in data
        assert data["health"]["status"] == "ok"

    def test_dashboard_tab_layout(self, client):
        r = client.get("/api/dashboard/summary")
        layout = r.json()["tab_layout"]
        assert "primary" in layout
        assert len(layout["primary"]) == 7
        assert "dashboard" in layout["primary"]
        assert "code" in layout["primary"]

    def test_dashboard_grouped_tabs(self, client):
        r = client.get("/api/dashboard/summary")
        grouped = r.json()["tab_layout"]["grouped"]
        assert "code" in grouped
        assert "files" in grouped["code"]
        assert "git" in grouped["code"]
        assert "patches" in grouped["code"]

    def test_ui_loads_with_new_tabs(self, client):
        r = client.get("/")
        assert r.status_code == 200
        text = r.text
        assert "Dashboard" in text
        assert "data-tab" in text

    def test_no_secrets_in_dashboard(self, client):
        r = client.get("/api/dashboard/summary")
        text = r.text.lower()
        for pat in ["ghp_", "sk-", "password=", "api_key="]:
            assert pat not in text

    def test_git_status_clean(self):
        import subprocess
        r = subprocess.run(["git", "status", "--porcelain"],
                          capture_output=True, text=True, cwd=".")
        lines = [l for l in r.stdout.strip().split("\n") if l.strip()
                 and not any(ig in l for ig in [".igris/", "logs/", "__pycache__",
                                                 ".egg-info", ".pyc"])]
        for line in lines:
            assert any(allowed in line for allowed in [
                "test_dashboard_tabs.py", "index.html",
                "app.js", "style.css", "server.py",
                "DASHBOARD_UI.md",
                "test_guided_actions.py", "test_integration_v02.py",
                "test_ui_polish.py", "chat_personality.py",
                "chat_engine.py", "chat_streaming.py",
                "GUIDED_ACTIONS.md",
                "system_info.py", "safe_commands.py",
                "test_system_info.py", "SYSTEM_INFO.md",
                "README.md", "PREPARED_NOT_IMPLEMENTED.md",
                # Files modified by #76:
                "agent_action_schema.py", "agent_reasoning_loop.py",
                "prompt_contract.py", "test_write_guard.py",
                "test_agent_action_schema.py", "test_issue74_toolruntime_dispatcher.py",
                "test_doctor.py",
            ]), f"Unexpected changed file: {line}"
