"""Tests for /api/github/admin routes.

All mutating endpoints default to dry_run=True and return a dry_run
response — no real GitHub API calls are made.  Auth is tested both
without credentials (anonymous → 403) and with admin override via
FastAPI dependency_overrides.
"""

import pytest
from fastapi.testclient import TestClient

from igris.web.server import create_app
from igris.core.authorization import get_current_user


_ANON_USER = {"user_id": "anonymous", "scopes": ["read"], "trust_level": "readonly"}
_ADMIN_USER = {"user_id": "test_admin", "scopes": ["admin"], "trust_level": "admin"}


@pytest.fixture(scope="module")
def anon_client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def admin_client() -> TestClient:
    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: _ADMIN_USER
    return TestClient(app)


# ---------------------------------------------------------------------------
# Unauthenticated (anonymous) — all scope-gated endpoints return 403
# ---------------------------------------------------------------------------

class TestGitHubAdminUnauthenticated:

    def test_add_collaborator_requires_scope(self, anon_client):
        r = anon_client.post("/api/github/admin/collaborator/add",
                             json={"repo": "test/repo", "username": "user"})
        assert r.status_code == 403

    def test_create_repo_requires_scope(self, anon_client):
        r = anon_client.post("/api/github/admin/repo/create", json={"name": "new-repo"})
        assert r.status_code == 403

    def test_delete_repo_requires_scope(self, anon_client):
        r = anon_client.post("/api/github/admin/repo/delete", json={"repo": "test/repo"})
        assert r.status_code == 403

    def test_audit_log_requires_scope(self, anon_client):
        r = anon_client.get("/api/github/admin/audit-log")
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# Admin user — all dry_run endpoints return 200 with dry_run status
# ---------------------------------------------------------------------------

class TestGitHubAdminDryRun:

    def test_add_collaborator_dry_run(self, admin_client):
        r = admin_client.post("/api/github/admin/collaborator/add",
                              json={"repo": "owner/repo", "username": "alice", "dry_run": True})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "dry_run"
        assert "alice" in data["message"]

    def test_remove_collaborator_dry_run(self, admin_client):
        r = admin_client.post("/api/github/admin/collaborator/remove",
                              json={"repo": "owner/repo", "username": "alice", "dry_run": True})
        assert r.status_code == 200
        assert r.json()["status"] == "dry_run"

    def test_branch_protection_dry_run(self, admin_client):
        r = admin_client.post("/api/github/admin/branch-protection/set",
                              json={"repo": "owner/repo", "branch": "main", "dry_run": True})
        assert r.status_code == 200
        assert r.json()["status"] == "dry_run"

    def test_secret_set_dry_run(self, admin_client):
        r = admin_client.post("/api/github/admin/secret/set",
                              json={"repo": "owner/repo", "name": "MY_SECRET",
                                    "value": "s3cr3t", "dry_run": True})
        assert r.status_code == 200
        assert r.json()["status"] == "dry_run"

    def test_repo_create_dry_run(self, admin_client):
        r = admin_client.post("/api/github/admin/repo/create",
                              json={"name": "new-repo", "dry_run": True})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "dry_run"
        assert "new-repo" in data["message"]

    def test_repo_delete_dry_run(self, admin_client):
        r = admin_client.post("/api/github/admin/repo/delete",
                              json={"repo": "owner/repo", "dry_run": True})
        assert r.status_code == 200
        assert r.json()["status"] == "dry_run"

    def test_audit_log_returns_list(self, admin_client):
        r = admin_client.get("/api/github/admin/audit-log")
        assert r.status_code == 200
        assert "audit_log" in r.json()
        assert isinstance(r.json()["audit_log"], list)

    def test_repo_info(self, admin_client):
        r = admin_client.get("/api/github/admin/repo/info?repo=owner/repo")
        assert r.status_code == 200
