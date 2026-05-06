"""API tests for Code Navigation endpoints — Epic #59.

Tests the API contract: endpoints exist, return correct structure,
handle errors. Does not depend on specific repo content.
"""

import pytest
from fastapi.testclient import TestClient

from igris.web.server import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestNavSearchCodeAPI:
    """Test POST /api/nav/search-code."""

    def test_search_code_returns_200(self, client):
        resp = client.post("/api/nav/search-code", json={"pattern": "def"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["tool"] == "search_code"
        assert isinstance(data["success"], bool)
        assert isinstance(data["total_count"], int)
        assert isinstance(data["returned_count"], int)

    def test_search_invalid_regex(self, client):
        resp = client.post("/api/nav/search-code", json={"pattern": "[invalid"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "Invalid regex" in data["error"]


class TestNavFindFilesAPI:
    """Test POST /api/nav/find-files."""

    def test_find_files_returns_200(self, client):
        resp = client.post("/api/nav/find-files", json={"pattern": "*.py"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["success"], bool)
        assert isinstance(data["total_count"], int)

    def test_find_nonexistent(self, client):
        resp = client.post("/api/nav/find-files", json={"pattern": "*.xyznonexistent987"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 0


class TestNavListDirectoryAPI:
    """Test POST /api/nav/list-directory."""

    def test_list_returns_200(self, client):
        resp = client.post("/api/nav/list-directory", json={"path": ".", "depth": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["success"], bool)
        assert isinstance(data["returned_count"], int)

    def test_list_nonexistent(self, client):
        resp = client.post("/api/nav/list-directory", json={"path": "nonexistent_dir_12345"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False


class TestNavReadFileRangeAPI:
    """Test POST /api/nav/read-file-range."""

    def test_read_returns_200(self, client):
        resp = client.post("/api/nav/read-file-range", json={"path": "README.md", "start": 1, "end": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["success"], bool)

    def test_read_env_blocked(self, client):
        resp = client.post("/api/nav/read-file-range", json={"path": ".env"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False

    def test_read_nonexistent_file(self, client):
        resp = client.post("/api/nav/read-file-range", json={"path": "nonexistent_file_xyz.py"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False


class TestNavRepoMapAPI:
    """Test GET /api/nav/repo-map."""

    def test_repo_map_returns_200(self, client):
        resp = client.get("/api/nav/repo-map")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["success"], bool)
        assert "data" in data


class TestNavFindSymbolAPI:
    """Test POST /api/nav/find-symbol."""

    def test_find_symbol_returns_200(self, client):
        resp = client.post("/api/nav/find-symbol", json={"symbol": "create_app"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["success"], bool)
        assert isinstance(data["total_count"], int)

    def test_find_symbol_nonexistent(self, client):
        resp = client.post("/api/nav/find-symbol", json={"symbol": "xyznonexistent987654"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 0
