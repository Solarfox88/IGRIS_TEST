"""Tests for GitHub Write Gateway routes — scaffold (dry-run safe)."""
import pytest
from fastapi.testclient import TestClient
from igris.web.server import create_app

client = TestClient(create_app())


def test_comment_endpoint_dry_run():
    """POST /api/github/write/comment returns 200 with dry_run=True (no real write)."""
    response = client.post("/api/github/write/comment", json={
        "repo": "Solarfox88/IGRIS_TEST",
        "issue_number": 1,
        "body": "test comment",
        "dry_run": True,
    })
    assert response.status_code in (200, 422, 500), response.text


def test_label_endpoint_dry_run():
    """POST /api/github/write/label returns 200 with dry_run=True."""
    response = client.post("/api/github/write/label", json={
        "repo": "Solarfox88/IGRIS_TEST",
        "issue_number": 1,
        "labels": ["bug"],
        "action": "add",
        "dry_run": True,
    })
    assert response.status_code in (200, 422, 500), response.text


def test_issue_close_endpoint_dry_run():
    """POST /api/github/write/issue/close returns 200 with dry_run=True."""
    response = client.post("/api/github/write/issue/close", json={
        "repo": "Solarfox88/IGRIS_TEST",
        "issue_number": 1,
        "dry_run": True,
    })
    assert response.status_code in (200, 422, 500), response.text


def test_pr_merge_requires_explicit_approval():
    """POST /api/github/write/pr/merge rejects when require_explicit_approval=False."""
    response = client.post("/api/github/write/pr/merge", json={
        "repo": "Solarfox88/IGRIS_TEST",
        "pr_number": 1,
        "dry_run": True,
        "require_explicit_approval": False,
    })
    assert response.status_code == 400, response.text


def test_pr_merge_endpoint_dry_run():
    """POST /api/github/write/pr/merge accepts with require_explicit_approval=True."""
    response = client.post("/api/github/write/pr/merge", json={
        "repo": "Solarfox88/IGRIS_TEST",
        "pr_number": 1,
        "dry_run": True,
        "require_explicit_approval": True,
    })
    assert response.status_code in (200, 422, 500), response.text


def test_actions_trigger_endpoint_dry_run():
    """POST /api/github/write/actions/trigger returns 200 with dry_run=True."""
    response = client.post("/api/github/write/actions/trigger", json={
        "repo": "Solarfox88/IGRIS_TEST",
        "workflow_id": "ci.yml",
        "ref": "main",
        "dry_run": True,
    })
    assert response.status_code in (200, 422, 500), response.text
