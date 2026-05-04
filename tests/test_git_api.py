"""Tests for Git API endpoints."""

import os

from fastapi.testclient import TestClient

from igris.web.server import create_app


def _client(tmp_path):
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    os.environ["PROJECT_ROOT"] = str(root)
    os.environ["WORKSPACE_ROOT"] = str(root)
    return TestClient(create_app())


def test_git_diff_endpoint(tmp_path):
    client = _client(tmp_path)
    r = client.get("/api/git/diff")
    assert r.status_code == 200
    data = r.json()
    assert "diff" in data or "error" in data


def test_git_diff_staged(tmp_path):
    client = _client(tmp_path)
    r = client.get("/api/git/diff?staged=true")
    assert r.status_code == 200


def test_git_diff_stat(tmp_path):
    client = _client(tmp_path)
    r = client.get("/api/git/diff/stat")
    assert r.status_code == 200


def test_git_branches(tmp_path):
    client = _client(tmp_path)
    r = client.get("/api/git/branches")
    assert r.status_code == 200
    data = r.json()
    assert "branches" in data or "error" in data


def test_git_create_branch_no_name(tmp_path):
    client = _client(tmp_path)
    r = client.post("/api/git/branch", json={"name": ""})
    assert r.status_code == 400


def test_git_safety_check(tmp_path):
    client = _client(tmp_path)
    r = client.get("/api/git/safety-check")
    assert r.status_code == 200
    data = r.json()
    assert "safe" in data or "error" in data


def test_git_commit_proposal_no_message(tmp_path):
    client = _client(tmp_path)
    r = client.post("/api/git/commit-proposal", json={"message": ""})
    assert r.status_code == 400


def test_git_commit_proposal(tmp_path):
    client = _client(tmp_path)
    r = client.post("/api/git/commit-proposal", json={"message": "test: proposal"})
    assert r.status_code == 200
    data = r.json()
    assert "message" in data
    assert "safe" in data
    assert "warnings" in data


def test_git_pr_summary(tmp_path):
    client = _client(tmp_path)
    r = client.get("/api/git/pr-summary")
    assert r.status_code == 200


def test_no_push_endpoint(tmp_path):
    client = _client(tmp_path)
    r = client.post("/api/git/push")
    assert r.status_code in (404, 405)


def test_no_secrets_in_diff(tmp_path):
    client = _client(tmp_path)
    r = client.get("/api/git/diff")
    body = r.text
    assert "sk-" not in body
    assert "OPENAI_API_KEY" not in body
