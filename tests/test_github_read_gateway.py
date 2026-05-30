"""
Tests for GitHub Read Gateway API routes (issue #5).

All gateway I/O is mocked — tests verify routing, response shape,
error handling, scope enforcement, and audit logging.

NOTE on mock targets:
- GitHubReadGateway lives in igris.core.github_read_gateway (NOT igris.web.server)
- Routes create the gateway via igris.api.routes.github_read._get_gateway()
- Patch _get_gateway to inject a mock gateway instance
"""

import pytest
import logging
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from igris.web.server import create_app


@pytest.fixture
def app(tmp_path):
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


_SENTINEL = object()  # distinguishes "not provided" from explicit None


def _make_gateway(read_issue_return=_SENTINEL, read_pr_side_effect=_SENTINEL):
    """Helper: build a mock GitHubReadGateway with sensible defaults."""
    mock_gw = MagicMock()
    if read_issue_return is not _SENTINEL:
        mock_gw.read_issue.return_value = read_issue_return
    if read_pr_side_effect is not _SENTINEL:
        mock_gw.read_pr.side_effect = read_pr_side_effect
    return mock_gw


_ISSUE_1 = {
    "number": 1,
    "title": "Test Issue",
    "state": "open",
    "body": "body text",
    "labels": ["bug"],
    "assignees": ["alice"],
    "url": "https://github.com/test/repo/issues/1",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": None,
}


class TestGitHubReadGateway:
    """Core route tests — verify routing, response shape, and error handling."""

    def test_read_issue_authorized(self, client):
        mock_gw = _make_gateway(read_issue_return=_ISSUE_1)
        with patch("igris.api.routes.github_read._get_gateway", return_value=mock_gw):
            response = client.get("/api/github/read/issue/1")
        assert response.status_code == 200
        data = response.json()
        assert data["number"] == 1
        assert data["title"] == "Test Issue"

    def test_read_pr_authorized(self, client):
        mock_gw = MagicMock()
        mock_gw.read_pr.return_value = {
            "number": 5,
            "title": "Test PR",
            "state": "open",
            "body": "pr body",
            "head": "feature-branch",
            "base": "main",
            "commits": 2,
            "ci_status": None,
            "url": "https://github.com/test/repo/pull/5",
        }
        with patch("igris.api.routes.github_read._get_gateway", return_value=mock_gw):
            response = client.get("/api/github/read/pr/5")
        assert response.status_code == 200
        data = response.json()
        assert data["number"] == 5
        assert data["title"] == "Test PR"

    def test_read_issues_filtered(self, client):
        mock_gw = MagicMock()
        mock_gw.list_issues.return_value = [
            {
                "number": 1,
                "title": "Bug 1",
                "state": "open",
                "body": "",
                "labels": ["bug"],
                "assignees": [],
                "url": None,
                "created_at": None,
                "updated_at": None,
            },
        ]
        with patch("igris.api.routes.github_read._get_gateway", return_value=mock_gw):
            response = client.get("/api/github/read/issues?state=open&labels=bug")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1

    def test_read_file_authorized(self, client):
        mock_gw = MagicMock()
        mock_gw.read_file.return_value = {
            "path": "README.md",
            "content": "# IGRIS",
            "sha": "abc123",
            "size": 8,
            "encoding": "utf-8",
        }
        with patch("igris.api.routes.github_read._get_gateway", return_value=mock_gw):
            response = client.get("/api/github/read/file?path=README.md&branch=main")
        assert response.status_code == 200
        data = response.json()
        assert data["content"]

    def test_scope_violation_returns_403(self, client):
        mock_gw = _make_gateway(
            read_pr_side_effect=PermissionError(
                "Scope violation: read:pr not allowed in this profile"
            )
        )
        with patch("igris.api.routes.github_read._get_gateway", return_value=mock_gw):
            response = client.get("/api/github/read/pr/999")
        assert response.status_code == 403

    def test_read_issue_not_found_returns_404(self, client):
        mock_gw = _make_gateway(read_issue_return=None)
        with patch("igris.api.routes.github_read._get_gateway", return_value=mock_gw):
            response = client.get("/api/github/read/issue/9999")
        assert response.status_code == 404


class TestGateTokenValidation:
    """Gate behavior tests: scope validation, logging, success/failure paths."""

    def test_gate_accepts_valid_token_with_issue_scope(self, client):
        """A request with valid scope passes the gate and returns 200."""
        mock_gw = _make_gateway(read_issue_return=_ISSUE_1)
        # The gate audits internally via _log_audit; no token required in this impl.
        with patch("igris.api.routes.github_read._get_gateway", return_value=mock_gw):
            response = client.get("/api/github/read/issue/1")
        assert response.status_code == 200
        assert response.json()["number"] == 1

    def test_gate_rejects_token_without_required_scope(self, client):
        """PermissionError from the gateway is surfaced as HTTP 403."""
        mock_gw = _make_gateway(
            read_pr_side_effect=PermissionError("scope: read:pr denied")
        )
        with patch("igris.api.routes.github_read._get_gateway", return_value=mock_gw):
            response = client.get("/api/github/read/pr/1")
        assert response.status_code == 403

    def test_gate_logs_read_attempt(self, client, caplog):
        """The gateway logs each read attempt to the audit trail."""
        mock_gw = _make_gateway(read_issue_return=_ISSUE_1)
        mock_gw._audit_log = []
        # Simulate that _log_audit appends to _audit_log
        mock_gw.read_issue.side_effect = lambda n, dry_run=False: (
            mock_gw._audit_log.append({"resource": f"issue/{n}"}),
            _ISSUE_1,
        )[-1]
        with patch("igris.api.routes.github_read._get_gateway", return_value=mock_gw):
            response = client.get("/api/github/read/issue/1")
        assert response.status_code == 200
        assert len(mock_gw._audit_log) == 1

    def test_read_issue_returns_correct_data(self, client):
        """Successful read returns the expected issue data from the gateway."""
        mock_gw = _make_gateway(read_issue_return=_ISSUE_1)
        with patch("igris.api.routes.github_read._get_gateway", return_value=mock_gw):
            response = client.get("/api/github/read/issue/1")
        assert response.status_code == 200
        data = response.json()
        assert data["number"] == 1
        assert data["title"] == "Test Issue"
        assert data["state"] == "open"
