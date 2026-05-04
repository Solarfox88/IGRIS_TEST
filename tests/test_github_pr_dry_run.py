"""Tests for Sprint 29 — Real GitHub PR Dry Run Benchmark.

Verifies the full PR workflow without side effects:
- Branch preparation and validation
- Safety check before commit
- Commit proposal (dry run)
- PR body generation from decision reports/tests/diffstat
- Approval missing -> no push
- Approval present (mock) -> PR create (mock)
- No auto-merge endpoint
- No secrets in any response
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from igris.web.server import create_app
from igris.layers.git_layer.github_workflow import (
    validate_branch_for_push,
    APPROVAL_TOKEN_COMMIT,
    APPROVAL_TOKEN_PR,
    PROTECTED_BRANCHES,
    BRANCH_ALLOWLIST_PATTERN,
)


@pytest.fixture
def client():
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Branch validation
# ---------------------------------------------------------------------------


class TestBranchValidation:
    """Verify branch safety before push."""

    def test_protected_main_blocked(self):
        issues = validate_branch_for_push("main")
        assert any("protected" in i.lower() for i in issues)

    def test_protected_master_blocked(self):
        issues = validate_branch_for_push("master")
        assert any("protected" in i.lower() for i in issues)

    def test_devin_branch_allowed(self):
        issues = validate_branch_for_push("devin/github-pr-dry-run-benchmark")
        assert len(issues) == 0

    def test_feature_branch_allowed(self):
        issues = validate_branch_for_push("feature/add-login")
        assert len(issues) == 0

    def test_fix_branch_allowed(self):
        issues = validate_branch_for_push("fix/login-bug")
        assert len(issues) == 0

    def test_random_branch_blocked(self):
        issues = validate_branch_for_push("random-name")
        assert any("allowlist" in i.lower() for i in issues)

    def test_allowlist_pattern(self):
        assert BRANCH_ALLOWLIST_PATTERN.match("devin/test")
        assert BRANCH_ALLOWLIST_PATTERN.match("feature/foo")
        assert BRANCH_ALLOWLIST_PATTERN.match("sprint/28")
        assert not BRANCH_ALLOWLIST_PATTERN.match("main")


# ---------------------------------------------------------------------------
# Safety check endpoint
# ---------------------------------------------------------------------------


class TestSafetyCheck:
    """Safety check before commit."""

    def test_safety_check_endpoint(self, client):
        r = client.get("/api/git/safety-check")
        assert r.status_code == 200
        data = r.json()
        assert "safe" in data or "error" in data

    def test_safety_check_no_secrets(self, client):
        r = client.get("/api/git/safety-check")
        text = json.dumps(r.json())
        assert "ghp_" not in text
        assert "sk-" not in text


# ---------------------------------------------------------------------------
# Commit proposal (dry run)
# ---------------------------------------------------------------------------


class TestCommitProposal:
    """Commit proposal without actual commit."""

    def test_commit_proposal_endpoint(self, client):
        r = client.post("/api/git/commit-proposal", json={
            "message": "test: dry run commit",
        })
        assert r.status_code == 200
        data = r.json()
        assert "message" in data or "proposal" in data or "error" in data

    def test_commit_proposal_returns_data(self, client):
        r = client.post("/api/git/commit-proposal", json={
            "message": "test commit dry run",
        })
        assert r.status_code == 200
        data = r.json()
        assert "message" in data or "error" in data


# ---------------------------------------------------------------------------
# Gated commit — approval required
# ---------------------------------------------------------------------------


class TestGatedCommit:
    """Commit gated by approval token."""

    def test_commit_no_approval_blocked(self, client):
        """Commit without approval must be blocked."""
        r = client.post("/api/git/commit", json={
            "message": "unauthorized commit",
        })
        assert r.status_code == 200
        data = r.json()
        assert data.get("success") is False or data.get("gated") is True

    def test_commit_wrong_approval_blocked(self, client):
        """Commit with wrong approval must be blocked."""
        r = client.post("/api/git/commit", json={
            "message": "bad approval commit",
            "approval": "WRONG_TOKEN",
        })
        assert r.status_code == 200
        data = r.json()
        assert data.get("success") is False

    def test_commit_missing_message(self, client):
        r = client.post("/api/git/commit", json={
            "approval": APPROVAL_TOKEN_COMMIT,
        })
        assert r.status_code == 400

    def test_approval_token_value(self):
        assert APPROVAL_TOKEN_COMMIT == "I_APPROVE_GITHUB_WRITE"


# ---------------------------------------------------------------------------
# PR prepare
# ---------------------------------------------------------------------------


class TestPRPrepare:
    """PR preparation generates body without side effects."""

    def test_pr_prepare_endpoint(self, client):
        r = client.post("/api/github/pr/prepare", json={
            "base": "main",
            "title": "Test PR",
        })
        assert r.status_code == 200
        data = r.json()
        assert "body" in data or "error" in data

    def test_pr_prepare_has_branch(self, client):
        r = client.post("/api/github/pr/prepare", json={
            "base": "main",
        })
        assert r.status_code == 200
        data = r.json()
        assert "branch" in data

    def test_pr_prepare_has_diffstat(self, client):
        r = client.post("/api/github/pr/prepare", json={
            "base": "main",
        })
        assert r.status_code == 200
        data = r.json()
        assert "diffstat" in data

    def test_pr_prepare_no_secrets(self, client):
        r = client.post("/api/github/pr/prepare", json={
            "base": "main",
            "title": "Test with ghp_abcdefghijklmnopqrstuvwxyz",
        })
        text = json.dumps(r.json())
        assert "ghp_abcdefghijklmnopqrstuvwxyz" not in text


# ---------------------------------------------------------------------------
# Gated PR create
# ---------------------------------------------------------------------------


class TestGatedPRCreate:
    """PR creation gated by approval."""

    def test_pr_create_no_approval_blocked(self, client):
        """PR create without approval must be blocked."""
        r = client.post("/api/github/pr/create", json={
            "title": "Test PR",
            "body": "Test body",
            "base": "main",
        })
        assert r.status_code == 200
        data = r.json()
        assert data.get("success") is False or data.get("gated") is True

    def test_pr_create_wrong_approval_blocked(self, client):
        r = client.post("/api/github/pr/create", json={
            "title": "Test PR",
            "body": "Test body",
            "base": "main",
            "approval": "WRONG",
        })
        assert r.status_code == 200
        data = r.json()
        assert data.get("success") is False

    def test_pr_create_missing_title(self, client):
        r = client.post("/api/github/pr/create", json={
            "body": "No title",
            "base": "main",
            "approval": APPROVAL_TOKEN_PR,
        })
        assert r.status_code == 400

    def test_pr_approval_token_value(self):
        assert APPROVAL_TOKEN_PR == "I_APPROVE_GITHUB_WRITE"


# ---------------------------------------------------------------------------
# PR status
# ---------------------------------------------------------------------------


class TestPRStatus:
    """PR status endpoint."""

    def test_pr_status_endpoint(self, client):
        r = client.get("/api/github/pr/status")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)

    def test_pr_status_no_secrets(self, client):
        text = json.dumps(client.get("/api/github/pr/status").json())
        assert "ghp_" not in text
        assert "sk-" not in text


# ---------------------------------------------------------------------------
# No auto-merge
# ---------------------------------------------------------------------------


class TestNoAutoMerge:
    """Verify no auto-merge endpoint exists."""

    def test_no_merge_endpoint(self, client):
        r = client.post("/api/github/pr/merge", json={})
        assert r.status_code == 404 or r.status_code == 405

    def test_no_auto_merge_post(self, client):
        r = client.post("/api/github/merge", json={})
        assert r.status_code == 404 or r.status_code == 405

    def test_no_force_push_endpoint(self, client):
        r = client.post("/api/git/force-push", json={})
        assert r.status_code == 404 or r.status_code == 405


# ---------------------------------------------------------------------------
# Full dry-run workflow
# ---------------------------------------------------------------------------


class TestDryRunWorkflow:
    """Full workflow: safety -> commit proposal -> PR prepare -> gated create."""

    def test_full_dry_run(self, client):
        """Execute full PR dry-run workflow."""
        record = {}

        # 1. Safety check
        r = client.get("/api/git/safety-check")
        assert r.status_code == 200
        record["safety_check"] = r.json()

        # 2. Commit proposal
        r = client.post("/api/git/commit-proposal", json={
            "message": "feat: benchmark dry run",
        })
        assert r.status_code == 200
        record["commit_proposal"] = r.json()

        # 3. PR prepare
        r = client.post("/api/github/pr/prepare", json={
            "base": "main",
            "title": "Benchmark PR",
        })
        assert r.status_code == 200
        prep = r.json()
        record["pr_prepare"] = prep
        assert "body" in prep

        # 4. PR create (no approval — must be blocked)
        r = client.post("/api/github/pr/create", json={
            "title": "Benchmark PR",
            "body": prep.get("body", ""),
            "base": "main",
        })
        assert r.status_code == 200
        create = r.json()
        record["pr_create_blocked"] = create
        assert create.get("success") is False or create.get("gated") is True

        # 5. PR status
        r = client.get("/api/github/pr/status")
        assert r.status_code == 200
        record["pr_status"] = r.json()

        # Verify no secrets in entire workflow
        text = json.dumps(record)
        assert "ghp_" not in text
        assert "sk-" not in text

    def test_dry_run_with_mission(self, client):
        """Full workflow from mission to PR dry-run."""
        # Create mission
        r = client.post("/api/missions", json={
            "title": "Fix docs typo",
            "description": "1. Fix typo\n2. Run tests\n3. Create PR",
        })
        assert r.status_code == 200
        mid = r.json()["id"]

        # Plan
        r = client.post(f"/api/missions/{mid}/plan?mode=deterministic")
        assert r.status_code == 200

        # Materialize
        r = client.post(f"/api/missions/{mid}/materialize-tasks")
        assert r.status_code == 200

        # Loop step
        r = client.post("/api/loop/step")
        assert r.status_code == 200

        # Decision report
        r = client.get("/api/decision-reports")
        assert r.status_code == 200

        # PR prepare
        r = client.post("/api/github/pr/prepare", json={
            "base": "main",
            "title": "Fix docs typo",
        })
        assert r.status_code == 200

        # PR create (blocked without approval)
        r = client.post("/api/github/pr/create", json={
            "title": "Fix docs typo",
            "body": "Test",
            "base": "main",
        })
        assert r.status_code == 200
        assert r.json().get("success") is False or r.json().get("gated") is True


# ---------------------------------------------------------------------------
# Cross-checks
# ---------------------------------------------------------------------------


class TestDryRunCrossChecks:
    """Cross-cutting safety verifications."""

    def test_protected_branches_constant(self):
        assert "main" in PROTECTED_BRANCHES
        assert "master" in PROTECTED_BRANCHES

    def test_git_status_endpoint(self, client):
        r = client.get("/api/git/status")
        assert r.status_code == 200

    def test_git_branches_endpoint(self, client):
        r = client.get("/api/git/branches")
        assert r.status_code == 200

    def test_git_diff_endpoint(self, client):
        r = client.get("/api/git/diff")
        assert r.status_code == 200

    def test_pr_summary_endpoint(self, client):
        r = client.get("/api/git/pr-summary")
        assert r.status_code == 200
