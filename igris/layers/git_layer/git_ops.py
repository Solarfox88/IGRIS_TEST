"""
Controlled Git operations — safety-first, no auto-push.

Functions for diff viewing, branch management, staging, commit proposals,
and pre-commit safety checks. No push endpoint is exposed.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from igris.core.safety import (
    detect_secret_like_content,
    is_runtime_artifact,
    is_sensitive_filename,
    redact_secrets,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    root = Path(os.environ.get("PROJECT_ROOT", "."))
    if root.exists() and root.is_dir():
        return root
    return Path.cwd()


def _run_git(args: List[str], cwd: Path | None = None) -> str:
    cwd = cwd or _repo_root()
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _run_git_full(args: List[str], cwd: Path | None = None) -> Dict[str, str]:
    """Run git command returning stdout, stderr and returncode."""
    cwd = cwd or _repo_root()
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": "1"}
    return {
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "returncode": str(result.returncode),
    }


def is_git_repo(cwd: Path | None = None) -> bool:
    """Check if the working directory is a git repository."""
    cwd = cwd or _repo_root()
    r = _run_git_full(["rev-parse", "--is-inside-work-tree"], cwd)
    return r["returncode"] == "0"


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def get_diff(staged: bool = False, redact: bool = True) -> Dict[str, object]:
    """Get the working tree or staged diff."""
    if not is_git_repo():
        return {"error": "Not a git repository", "diff": ""}
    args = ["diff"]
    if staged:
        args.append("--staged")
    raw = _run_git(args)
    secret_detected = False
    if raw and detect_secret_like_content(raw):
        secret_detected = True
    diff_text = redact_secrets(raw) if (redact and raw) else raw
    return {
        "diff": diff_text,
        "staged": staged,
        "secret_detected": secret_detected,
        "lines": len(raw.splitlines()) if raw else 0,
    }


def get_diff_stat() -> Dict[str, object]:
    """Get a summary of changes (diffstat)."""
    if not is_git_repo():
        return {"error": "Not a git repository", "stat": ""}
    stat = _run_git(["diff", "--stat"])
    staged_stat = _run_git(["diff", "--staged", "--stat"])
    return {"unstaged": stat, "staged": staged_stat}


# ---------------------------------------------------------------------------
# Branch
# ---------------------------------------------------------------------------

_BRANCH_SANITIZE = re.compile(r"[^a-zA-Z0-9/_\-.]")


def sanitize_branch_name(name: str) -> str:
    """Sanitize a branch name to only allow safe characters."""
    sanitized = _BRANCH_SANITIZE.sub("-", name.strip())
    sanitized = re.sub(r"-{2,}", "-", sanitized)
    sanitized = sanitized.strip("-.")
    if not sanitized:
        sanitized = "unnamed-branch"
    return sanitized


def create_branch(name: str) -> Dict[str, object]:
    """Create and checkout a new branch (sanitized name)."""
    if not is_git_repo():
        return {"success": False, "error": "Not a git repository"}
    safe_name = sanitize_branch_name(name)
    r = _run_git_full(["checkout", "-b", safe_name])
    if r["returncode"] != "0":
        return {"success": False, "error": r["stderr"], "branch": safe_name}
    return {"success": True, "branch": safe_name}


def list_branches() -> Dict[str, object]:
    """List local branches."""
    if not is_git_repo():
        return {"error": "Not a git repository", "branches": []}
    raw = _run_git(["branch", "--list"])
    branches = []
    current = ""
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("* "):
            current = line[2:].strip()
            branches.append(current)
        elif line:
            branches.append(line)
    return {"branches": branches, "current": current}


# ---------------------------------------------------------------------------
# Runtime artifact & secret detection for staging
# ---------------------------------------------------------------------------

@dataclass
class StagingSafetyResult:
    """Result of pre-stage safety checks."""
    safe_files: List[str] = field(default_factory=list)
    blocked_files: List[str] = field(default_factory=list)
    secret_files: List[str] = field(default_factory=list)
    runtime_artifacts: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


def detect_runtime_artifacts_in_changes() -> List[str]:
    """Detect runtime artifacts in git changed files."""
    if not is_git_repo():
        return []
    status = _run_git(["status", "--short"])
    artifacts = []
    for line in status.splitlines():
        if len(line) < 4:
            continue
        file_path = line[3:].strip().strip('"')
        if is_runtime_artifact(Path(file_path)):
            artifacts.append(file_path)
    return artifacts


def detect_secret_like_changes() -> List[str]:
    """Detect files with secret-like content in uncommitted changes."""
    if not is_git_repo():
        return []
    diff = _run_git(["diff"])
    staged_diff = _run_git(["diff", "--staged"])
    secret_files = []
    for diff_text in [diff, staged_diff]:
        if not diff_text:
            continue
        current_file = ""
        for line in diff_text.splitlines():
            if line.startswith("diff --git"):
                parts = line.split(" b/")
                if len(parts) > 1:
                    current_file = parts[-1]
            elif line.startswith("+") and not line.startswith("+++"):
                if detect_secret_like_content(line[1:]):
                    if current_file and current_file not in secret_files:
                        secret_files.append(current_file)
    return secret_files


def check_staging_safety(files: List[str]) -> StagingSafetyResult:
    """Check if files are safe to stage."""
    result = StagingSafetyResult()
    for f in files:
        p = Path(f)
        if is_runtime_artifact(p):
            result.runtime_artifacts.append(f)
            result.blocked_files.append(f)
            result.reasons.append(f"Runtime artifact: {f}")
            continue
        if is_sensitive_filename(p.name):
            result.blocked_files.append(f)
            result.reasons.append(f"Sensitive filename: {f}")
            continue
        result.safe_files.append(f)
    return result


# ---------------------------------------------------------------------------
# Stage files
# ---------------------------------------------------------------------------

def stage_files(files: List[str]) -> Dict[str, object]:
    """Stage specific files after safety check."""
    if not is_git_repo():
        return {"success": False, "error": "Not a git repository"}
    if not files:
        return {"success": False, "error": "No files specified"}

    safety = check_staging_safety(files)
    if not safety.safe_files:
        return {
            "success": False,
            "error": "No safe files to stage",
            "blocked": safety.blocked_files,
            "reasons": safety.reasons,
        }

    r = _run_git_full(["add", "--"] + safety.safe_files)
    if r["returncode"] != "0":
        return {"success": False, "error": r["stderr"]}
    return {
        "success": True,
        "staged": safety.safe_files,
        "blocked": safety.blocked_files,
        "reasons": safety.reasons,
    }


# ---------------------------------------------------------------------------
# Commit proposal & gated commit
# ---------------------------------------------------------------------------

@dataclass
class CommitProposal:
    """A proposed commit with safety metadata."""
    message: str
    files: List[str]
    safe: bool
    warnings: List[str] = field(default_factory=list)
    blocked_files: List[str] = field(default_factory=list)
    secret_files: List[str] = field(default_factory=list)
    runtime_artifacts: List[str] = field(default_factory=list)


def generate_commit_message(title: str, changes_summary: str = "") -> str:
    """Generate a conventional commit message."""
    msg = title.strip()
    if changes_summary:
        msg += f"\n\n{changes_summary.strip()}"
    return msg


def pre_commit_safety_check() -> Dict[str, object]:
    """Run safety checks before committing."""
    if not is_git_repo():
        return {"safe": False, "error": "Not a git repository", "warnings": []}

    warnings: List[str] = []
    secrets = detect_secret_like_changes()
    if secrets:
        warnings.append(f"Secret-like content detected in: {', '.join(secrets)}")

    artifacts = detect_runtime_artifacts_in_changes()
    if artifacts:
        warnings.append(f"Runtime artifacts found: {', '.join(artifacts)}")

    staged = _run_git(["diff", "--staged", "--name-only"])
    staged_files = [f for f in staged.splitlines() if f.strip()]
    if not staged_files:
        warnings.append("No files staged for commit")

    for f in staged_files:
        if is_sensitive_filename(Path(f).name):
            warnings.append(f"Sensitive file staged: {f}")

    safe = len(warnings) == 0 and len(staged_files) > 0
    return {
        "safe": safe,
        "warnings": warnings,
        "staged_files": staged_files,
        "secret_files": secrets,
        "runtime_artifacts": artifacts,
    }


def create_commit_proposal(
    message: str,
    files: Optional[List[str]] = None,
) -> CommitProposal:
    """Create a commit proposal (does NOT actually commit)."""
    if not is_git_repo():
        return CommitProposal(
            message=message, files=[], safe=False,
            warnings=["Not a git repository"],
        )

    if files is None:
        staged = _run_git(["diff", "--staged", "--name-only"])
        files = [f for f in staged.splitlines() if f.strip()]

    safety = check_staging_safety(files)
    secrets = detect_secret_like_changes()
    artifacts = detect_runtime_artifacts_in_changes()

    warnings: List[str] = []
    if safety.blocked_files:
        warnings.extend(safety.reasons)
    if secrets:
        warnings.append(f"Secret-like content in: {', '.join(secrets)}")
    if artifacts:
        warnings.append(f"Runtime artifacts: {', '.join(artifacts)}")
    if not files:
        warnings.append("No files to commit")

    safe = len(warnings) == 0 and len(safety.safe_files) > 0
    return CommitProposal(
        message=message,
        files=safety.safe_files,
        safe=safe,
        warnings=warnings,
        blocked_files=safety.blocked_files,
        secret_files=secrets,
        runtime_artifacts=artifacts,
    )


def execute_commit(message: str, gate_override: bool = False) -> Dict[str, object]:
    """Execute a commit if safety checks pass or gate_override is True.

    By default commit is only allowed when pre_commit_safety_check() is safe.
    """
    if not is_git_repo():
        return {"success": False, "error": "Not a git repository"}

    safety = pre_commit_safety_check()
    if not safety["safe"] and not gate_override:
        return {
            "success": False,
            "error": "Commit blocked by safety checks",
            "warnings": safety["warnings"],
        }

    r = _run_git_full(["commit", "-m", message])
    if r["returncode"] != "0":
        return {"success": False, "error": r["stderr"]}

    commit_hash = _run_git(["rev-parse", "--short", "HEAD"])
    return {
        "success": True,
        "commit": commit_hash,
        "message": message,
        "warnings": safety.get("warnings", []),
    }


# ---------------------------------------------------------------------------
# PR summary generation
# ---------------------------------------------------------------------------

def generate_pr_summary(base_branch: str = "main") -> Dict[str, object]:
    """Generate a PR summary comparing current branch to base."""
    if not is_git_repo():
        return {"error": "Not a git repository"}

    current = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    if current == base_branch:
        return {"error": f"Already on {base_branch}, nothing to compare"}

    log = _run_git(["log", f"{base_branch}..HEAD", "--oneline"])
    stat = _run_git(["diff", f"{base_branch}...HEAD", "--stat"])
    diff_summary = _run_git(["diff", f"{base_branch}...HEAD", "--shortstat"])

    commits = [line for line in log.splitlines() if line.strip()] if log else []
    return {
        "branch": current,
        "base": base_branch,
        "commits": commits,
        "commit_count": len(commits),
        "stat": stat,
        "summary": diff_summary,
    }
