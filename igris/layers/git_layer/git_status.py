"""
Functions for retrieving read‑only git information.

These helpers use subprocess calls to `git` to determine the current branch,
clean/dirty status, changed files and the most recent commit hash.  They
operate relative to the configured project root.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional

from igris.models.config import CONFIG


class GitInfo:
    def __init__(self, branch: Optional[str], remote: Optional[str], dirty: bool, changed: List[str], head: Optional[str]):
        self.branch = branch
        self.remote = remote
        self.dirty = dirty
        self.changed = changed
        self.head = head


def _run_git(args: List[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(CONFIG.project_root),
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def get_git_info() -> GitInfo:
    """Gather information about the git repository located at project_root."""
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    remote = _run_git(["remote", "get-url", "origin"])
    status_output = _run_git(["status", "--porcelain"])
    dirty = bool(status_output)
    changed = [line[3:] for line in status_output.splitlines() if line] if status_output else []
    head = _run_git(["rev-parse", "--short", "HEAD"])
    return GitInfo(
        branch=branch or None,
        remote=remote or None,
        dirty=dirty,
        changed=changed,
        head=head or None,
    )