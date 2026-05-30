"""
Tests for GitHub Read Gateway API routes (issue #5).

All gateway I/O is mocked — tests verify routing, response shape,
and error handling without hitting real GitHub APIs.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from igris.web.server import create_app


@pytest.fixture
def app(tmp_path):
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


class TestGitHubReadGateway:

    def test_read_issue_authorized(self, client):
        mock_gw = MagicMock()
        mock_gw.read_issue.return_value = {
            "number": 1,
            "title": "Test Issue",
            "state": "open",
            "body": "body text",
            "labels": [],
            "assignees": [],
            "url": "https://github.com/test/repo/issues/1",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": None,
        }
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
        mock_gw = MagicMock()
        mock_gw.read_pr.side_effect = PermissionError(
            "Scope violation: read:pr not allowed in this profile"
        )
        with patch("igris.api.routes.github_read._get_gateway", return_value=mock_gw):
            response = client.get("/api/github/read/pr/999")
        assert response.status_code == 403

    def test_read_issue_not_found_returns_404(self, client):
        mock_gw = MagicMock()
        mock_gw.read_issue.return_value = None
        with patch("igris.api.routes.github_read._get_gateway", return_value=mock_gw):
            response = client.get("/api/github/read/issue/9999")
        assert response.status_code == 404
