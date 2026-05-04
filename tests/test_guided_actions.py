"""Sprint 33 — Guided Actions and Intent-to-Action Suggestions.

Tests for:
- SuggestedAction dataclass and serialization
- Intent → actions mapping for all intents
- Actions map only to existing safe endpoints (no free shell)
- API endpoints: /api/chat/actions, /api/chat/actions/{intent}, /api/chat/intent
- Chat engine and streaming include suggested_actions
- No secrets in any action
- No unsafe endpoints (no /api/shell, no /api/exec)
- Gated actions marked correctly (approval_required)
- UI renders action cards (CSS and JS)
- Mobile responsive action cards
"""

from __future__ import annotations

import json
import re

import pytest


# ---------------------------------------------------------------------------
# Unit tests — chat_personality suggested actions
# ---------------------------------------------------------------------------

class TestSuggestedActionDataclass:
    """SuggestedAction serialization."""

    def test_action_to_dict_basic(self):
        from igris.core.chat_personality import SuggestedAction
        a = SuggestedAction("Test", "desc", "/api/test")
        d = a.to_dict()
        assert d["label"] == "Test"
        assert d["description"] == "desc"
        assert d["endpoint"] == "/api/test"
        assert d["method"] == "GET"
        assert d["risk"] == "safe"
        assert d["approval_required"] is False
        assert "command_id" not in d
        assert "payload" not in d

    def test_action_to_dict_with_optional_fields(self):
        from igris.core.chat_personality import SuggestedAction
        a = SuggestedAction("X", "y", "/api/x", method="POST",
                           command_id="run_tests", payload={"key": "val"})
        d = a.to_dict()
        assert d["method"] == "POST"
        assert d["command_id"] == "run_tests"
        assert d["payload"] == {"key": "val"}

    def test_action_gated_flag(self):
        from igris.core.chat_personality import SuggestedAction
        a = SuggestedAction("PR", "create", "/api/github/pr/create",
                           risk="gated", approval_required=True)
        d = a.to_dict()
        assert d["approval_required"] is True
        assert d["risk"] == "gated"


class TestIntentToActions:
    """Each intent maps to a list of safe actions."""

    ALL_INTENTS = [
        "machine_info", "network_info", "github_access", "capabilities",
        "testing", "git_local", "patching", "missions", "memory", "shell_request",
    ]

    def test_all_intents_have_actions(self):
        from igris.core.chat_personality import get_suggested_actions
        for intent in self.ALL_INTENTS:
            actions = get_suggested_actions(intent)
            assert len(actions) > 0, f"No actions for intent {intent}"

    def test_unknown_intent_returns_empty(self):
        from igris.core.chat_personality import get_suggested_actions
        assert get_suggested_actions("nonexistent") == []

    @pytest.mark.parametrize("intent", ALL_INTENTS)
    def test_actions_have_required_fields(self, intent):
        from igris.core.chat_personality import get_suggested_actions
        for action in get_suggested_actions(intent):
            assert "label" in action
            assert "description" in action
            assert "endpoint" in action
            assert "method" in action
            assert action["method"] in ("GET", "POST")

    @pytest.mark.parametrize("intent", ALL_INTENTS)
    def test_no_unsafe_endpoints(self, intent):
        """No action should point to shell/exec/free-command endpoints."""
        from igris.core.chat_personality import get_suggested_actions
        unsafe = {"/api/shell", "/api/exec", "/api/cmd", "/api/run"}
        for action in get_suggested_actions(intent):
            assert action["endpoint"] not in unsafe, \
                f"Unsafe endpoint {action['endpoint']} in {intent}"

    @pytest.mark.parametrize("intent", ALL_INTENTS)
    def test_no_secrets_in_actions(self, intent):
        from igris.core.chat_personality import get_suggested_actions
        secret_patterns = ["ghp_", "sk-", "password", "token", "secret", "api_key"]
        text = json.dumps(get_suggested_actions(intent)).lower()
        for pat in secret_patterns:
            assert pat not in text, f"Secret pattern '{pat}' in actions for {intent}"

    def test_machine_info_actions(self):
        from igris.core.chat_personality import get_suggested_actions
        actions = get_suggested_actions("machine_info")
        labels = [a["label"] for a in actions]
        assert "Show Status" in labels
        assert "Show Readiness" in labels

    def test_github_actions_have_gated(self):
        from igris.core.chat_personality import get_suggested_actions
        actions = get_suggested_actions("github_access")
        gated = [a for a in actions if a.get("approval_required")]
        assert len(gated) >= 1, "GitHub should have at least one gated action"

    def test_testing_actions_include_run_tests(self):
        from igris.core.chat_personality import get_suggested_actions
        actions = get_suggested_actions("testing")
        labels = [a["label"] for a in actions]
        assert "Run Tests" in labels

    def test_shell_request_redirects_to_safe(self):
        """Shell request should suggest safe alternatives, not free shell."""
        from igris.core.chat_personality import get_suggested_actions
        actions = get_suggested_actions("shell_request")
        endpoints = [a["endpoint"] for a in actions]
        assert "/api/shell" not in endpoints
        assert "/api/exec" not in endpoints
        labels = [a["label"] for a in actions]
        assert "Show Available Commands" in labels

    def test_memory_actions(self):
        from igris.core.chat_personality import get_suggested_actions
        actions = get_suggested_actions("memory")
        labels = [a["label"] for a in actions]
        assert "Show Failures" in labels
        assert "Show Decisions" in labels
        assert "Show Saturation" in labels

    def test_missions_actions(self):
        from igris.core.chat_personality import get_suggested_actions
        actions = get_suggested_actions("missions")
        labels = [a["label"] for a in actions]
        assert "List Missions" in labels


class TestGetAllSafeActions:
    """get_all_safe_actions returns grouped actions."""

    def test_returns_dict_of_intents(self):
        from igris.core.chat_personality import get_all_safe_actions
        result = get_all_safe_actions()
        assert isinstance(result, dict)
        assert len(result) >= 10
        for intent, actions in result.items():
            assert isinstance(actions, list)
            assert len(actions) > 0

    def test_no_secrets_in_full_dump(self):
        from igris.core.chat_personality import get_all_safe_actions
        text = json.dumps(get_all_safe_actions()).lower()
        for pat in ["ghp_", "sk-", "password=", "api_key="]:
            assert pat not in text


class TestChatEngineIncludesActions:
    """Chat engine returns suggested_actions for grounded intents."""

    def test_grounded_response_has_actions(self):
        from igris.core.chat_engine import chat
        result = chat("dammi info sulla macchina")
        assert result.get("intent_detected") == "machine_info"
        actions = result.get("suggested_actions", [])
        assert len(actions) > 0
        assert any(a["label"] == "Show Status" for a in actions)

    def test_non_intent_has_no_actions(self):
        from igris.core.chat_engine import chat
        result = chat("ciao come stai")
        assert result.get("suggested_actions") is None or result.get("suggested_actions", []) == []

    def test_capabilities_intent_actions(self):
        from igris.core.chat_engine import chat
        result = chat("cosa puoi fare?")
        actions = result.get("suggested_actions", [])
        assert len(actions) > 0

    def test_github_intent_actions(self):
        from igris.core.chat_engine import chat
        result = chat("riesci a vedere il mio GitHub?")
        actions = result.get("suggested_actions", [])
        assert len(actions) > 0
        labels = [a["label"] for a in actions]
        assert "Show Git Status" in labels


class TestChatStreamIncludesActions:
    """Chat streaming includes suggested_actions in metadata."""

    def test_stream_grounded_has_actions(self):
        from igris.core.chat_streaming import chat_stream_sync
        chunks = chat_stream_sync("dammi info sulla macchina")
        assert len(chunks) > 0
        last_chunk = chunks[-1]
        meta = last_chunk.metadata
        actions = meta.get("suggested_actions", [])
        assert len(actions) > 0

    def test_stream_non_intent_no_actions(self):
        from igris.core.chat_streaming import chat_stream_sync
        chunks = chat_stream_sync("ciao")
        if chunks:
            last_chunk = chunks[-1]
            meta = last_chunk.metadata
            actions = meta.get("suggested_actions", [])
            # Non-intent may have empty actions or no key at all
            assert actions is None or isinstance(actions, list)


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    from igris.web.server import create_app
    from starlette.testclient import TestClient
    app = create_app()
    return TestClient(app)


class TestActionsAPI:
    """API endpoint tests for guided actions."""

    def test_get_all_actions(self, client):
        r = client.get("/api/chat/actions")
        assert r.status_code == 200
        data = r.json()
        assert "actions" in data
        assert len(data["actions"]) >= 10

    def test_get_actions_by_intent(self, client):
        r = client.get("/api/chat/actions/machine_info")
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "machine_info"
        assert len(data["actions"]) > 0

    def test_get_actions_unknown_intent_404(self, client):
        r = client.get("/api/chat/actions/nonexistent_xyz")
        assert r.status_code == 404

    def test_intent_endpoint_includes_actions(self, client):
        r = client.post("/api/chat/intent", json={"message": "dammi info sulla macchina"})
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] == "machine_info"
        assert "suggested_actions" in data
        assert len(data["suggested_actions"]) > 0

    def test_intent_no_match_empty_actions(self, client):
        r = client.post("/api/chat/intent", json={"message": "ciao come stai oggi"})
        assert r.status_code == 200
        data = r.json()
        assert data["intent"] is None
        assert data["suggested_actions"] == []

    def test_session_message_includes_actions(self, client):
        # Create session
        sr = client.post("/api/sessions")
        sid = sr.json()["id"]
        # Send message with known intent
        r = client.post(f"/api/sessions/{sid}/messages",
                       json={"message": "cosa puoi fare?"})
        assert r.status_code == 200
        data = r.json()
        assert "suggested_actions" in data
        assert len(data["suggested_actions"]) > 0

    def test_session_message_no_intent_empty_actions(self, client):
        sr = client.post("/api/sessions")
        sid = sr.json()["id"]
        r = client.post(f"/api/sessions/{sid}/messages",
                       json={"message": "ciao mondo"})
        assert r.status_code == 200
        data = r.json()
        actions = data.get("suggested_actions", [])
        assert actions is None or isinstance(actions, list)

    def test_no_secrets_in_actions_api(self, client):
        r = client.get("/api/chat/actions")
        text = r.text.lower()
        for pat in ["ghp_", "sk-", "password=", "api_key="]:
            assert pat not in text

    def test_all_intents_via_api(self, client):
        intents = ["machine_info", "network_info", "github_access",
                   "capabilities", "testing", "git_local", "patching",
                   "missions", "memory", "shell_request"]
        for intent in intents:
            r = client.get(f"/api/chat/actions/{intent}")
            assert r.status_code == 200, f"Failed for {intent}"


# ---------------------------------------------------------------------------
# UI / CSS / JS tests
# ---------------------------------------------------------------------------

class TestUIActionCards:
    """CSS and JS support for action cards."""

    def test_css_has_action_card_styles(self):
        import pathlib
        css = pathlib.Path("igris/web/static/css/style.css").read_text()
        assert ".action-card" in css
        assert ".suggested-actions" in css
        assert ".action-gated" in css
        assert ".action-loading" in css

    def test_css_mobile_action_card(self):
        import pathlib
        css = pathlib.Path("igris/web/static/css/style.css").read_text()
        assert ".action-card .action-label" in css
        assert ".action-card .action-desc" in css

    def test_js_has_handle_action_click(self):
        import pathlib
        js = pathlib.Path("igris/web/static/js/app.js").read_text()
        assert "handleActionClick" in js

    def test_js_renders_actions_in_addmsg(self):
        import pathlib
        js = pathlib.Path("igris/web/static/js/app.js").read_text()
        assert "suggested-actions" in js
        assert "action-card" in js

    def test_js_passes_actions_in_meta(self):
        import pathlib
        js = pathlib.Path("igris/web/static/js/app.js").read_text()
        assert "suggested_actions" in js
        assert "meta.actions" in js or "actions: r.data.suggested_actions" in js

    def test_js_action_click_calls_api(self):
        import pathlib
        js = pathlib.Path("igris/web/static/js/app.js").read_text()
        assert "api(" in js
        assert "btn.dataset.endpoint" in js

    def test_js_no_xss_in_action_labels(self):
        """Action labels are escaped via escapeHtml."""
        import pathlib
        js = pathlib.Path("igris/web/static/js/app.js").read_text()
        assert "escapeHtml(act.label)" in js
        assert "escapeHtml(act.description)" in js

    def test_js_shows_approval_badge(self):
        import pathlib
        js = pathlib.Path("igris/web/static/js/app.js").read_text()
        assert "action-gated" in js
        assert "requires approval" in js

    def test_ui_loads(self, client):
        r = client.get("/")
        assert r.status_code == 200
        text = r.text
        assert "chat" in text.lower()

    def test_git_status_clean(self):
        import subprocess
        r = subprocess.run(["git", "status", "--porcelain"],
                          capture_output=True, text=True, cwd=".")
        lines = [l for l in r.stdout.strip().split("\n") if l.strip()
                 and not any(ig in l for ig in [".igris/", "logs/", "__pycache__",
                                                 ".egg-info", ".pyc"])]
        # Only our new/changed files should show
        for line in lines:
            assert any(allowed in line for allowed in [
                "test_guided_actions.py", "chat_personality.py",
                "server.py", "app.js", "style.css", "chat_engine.py",
                "chat_streaming.py", "GUIDED_ACTIONS.md",
                "index.html", "DASHBOARD_UI.md",
                "test_integration_v02.py", "test_ui_polish.py",
                "test_dashboard_tabs.py",
            ]), f"Unexpected changed file: {line}"
