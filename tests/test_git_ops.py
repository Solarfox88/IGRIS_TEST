"""Tests for igris.layers.git_layer.git_ops module."""

import os
import subprocess

import pytest

from igris.layers.git_layer.git_ops import (
    sanitize_branch_name,
    is_git_repo,
    get_diff,
    get_diff_stat,
    list_branches,
    detect_runtime_artifacts_in_changes,
    detect_secret_like_changes,
    check_staging_safety,
    create_commit_proposal,
    pre_commit_safety_check,
    generate_commit_message,
    generate_pr_summary,
)


@pytest.fixture
def git_project(tmp_path, monkeypatch):
    """Create a temporary git repository."""
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path), capture_output=True, check=True)
    (tmp_path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "README.md"], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), capture_output=True, check=True)
    return tmp_path


@pytest.fixture
def non_git_project(tmp_path, monkeypatch):
    """Create a temporary non-git directory."""
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    return tmp_path


class TestSanitizeBranchName:
    def test_normal_name(self):
        assert sanitize_branch_name("feature/my-branch") == "feature/my-branch"

    def test_spaces_replaced(self):
        assert sanitize_branch_name("my branch name") == "my-branch-name"

    def test_special_chars(self):
        assert sanitize_branch_name("feat!@#$%test") == "feat-test"

    def test_empty(self):
        assert sanitize_branch_name("") == "unnamed-branch"

    def test_dots_dashes(self):
        assert sanitize_branch_name("v1.2.3") == "v1.2.3"

    def test_consecutive_dashes(self):
        result = sanitize_branch_name("a---b")
        assert "--" not in result


class TestIsGitRepo:
    def test_git_repo(self, git_project):
        assert is_git_repo(cwd=git_project) is True

    def test_non_repo(self, non_git_project):
        assert is_git_repo(cwd=non_git_project) is False


class TestGetDiff:
    def test_diff_returns_dict(self, git_project):
        result = get_diff()
        assert "diff" in result
        assert "staged" in result
        assert "secret_detected" in result

    def test_staged_diff(self, git_project):
        result = get_diff(staged=True)
        assert result["staged"] is True

    def test_non_git_returns_error(self, non_git_project):
        result = get_diff()
        assert "error" in result

    def test_diff_redacts_secrets(self, git_project):
        (git_project / "conf.txt").write_text("API_KEY=sk-abc123def456ghi789jkl012mno345pqr\n")
        subprocess.run(["git", "add", "conf.txt"], cwd=str(git_project), capture_output=True)
        result = get_diff(staged=True)
        assert result["secret_detected"] is True
        assert "sk-abc123" not in result["diff"]


class TestDiffStat:
    def test_returns_stat(self, git_project):
        result = get_diff_stat()
        assert "unstaged" in result
        assert "staged" in result

    def test_non_git(self, non_git_project):
        result = get_diff_stat()
        assert "error" in result


class TestListBranches:
    def test_list_branches(self, git_project):
        result = list_branches()
        assert "branches" in result
        assert "current" in result
        assert len(result["branches"]) > 0

    def test_non_git(self, non_git_project):
        result = list_branches()
        assert "error" in result


class TestStagingSafety:
    def test_safe_files(self):
        result = check_staging_safety(["docs/readme.md", "src/main.py"])
        assert len(result.safe_files) == 2
        assert len(result.blocked_files) == 0

    def test_runtime_artifact_blocked(self):
        result = check_staging_safety(["__pycache__/foo.pyc", "docs/ok.md"])
        assert "__pycache__/foo.pyc" in result.blocked_files
        assert "docs/ok.md" in result.safe_files

    def test_sensitive_filename_blocked(self):
        result = check_staging_safety([".env", "src/app.py"])
        assert ".env" in result.blocked_files
        assert "src/app.py" in result.safe_files

    def test_secret_key_file_blocked(self):
        result = check_staging_safety(["api_key.json"])
        assert "api_key.json" in result.blocked_files


class TestCommitProposal:
    def test_proposal_creation(self, git_project):
        proposal = create_commit_proposal("test: add tests")
        assert proposal.message == "test: add tests"

    def test_proposal_with_no_staged(self, git_project):
        proposal = create_commit_proposal("test: nothing staged")
        assert "No files to commit" in proposal.warnings or len(proposal.files) == 0


class TestPreCommitSafety:
    def test_safety_check(self, git_project):
        result = pre_commit_safety_check()
        assert "safe" in result
        assert "warnings" in result
        assert "staged_files" in result

    def test_non_git(self, non_git_project):
        result = pre_commit_safety_check()
        assert result["safe"] is False


class TestGenerateCommitMessage:
    def test_simple_message(self):
        msg = generate_commit_message("feat: add feature")
        assert msg == "feat: add feature"

    def test_with_summary(self):
        msg = generate_commit_message("fix: bug", "Fixed the null pointer")
        assert "fix: bug" in msg
        assert "Fixed the null pointer" in msg


class TestPrSummary:
    def test_pr_summary_non_git(self, non_git_project):
        result = generate_pr_summary(base_branch="main")
        assert "error" in result

    def test_pr_summary_same_branch(self, git_project):
        result = generate_pr_summary(base_branch="master")
        assert "error" in result or "branch" in result


class TestDetectArtifacts:
    def test_no_crash(self, git_project):
        result = detect_runtime_artifacts_in_changes()
        assert isinstance(result, list)

    def test_non_git(self, non_git_project):
        result = detect_runtime_artifacts_in_changes()
        assert result == []


class TestDetectSecrets:
    def test_no_crash(self, git_project):
        result = detect_secret_like_changes()
        assert isinstance(result, list)

    def test_non_git(self, non_git_project):
        result = detect_secret_like_changes()
        assert result == []
