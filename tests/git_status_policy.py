"""Shared git-status policy checks for tests.

The goal is to catch dirty runtime artifacts and sensitive files without
requiring every legitimate task file to be added to a static allowlist.
"""

from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path


BLOCKED_PARTS = {
    ".git",
    ".igris",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "logs",
}

BLOCKED_NAMES = {
    ".env",
    ".env.local",
    ".envrc",
}

BLOCKED_SUFFIXES = {
    ".log",
    ".pyc",
    ".pyo",
    ".sqlite",
    ".sqlite3",
    ".tmp",
}

ALLOWED_PATTERNS = (
    "*.md",
    "*.toml",
    "*.json",
    "*.yaml",
    "*.yml",
    "docs/**",
    "igris/**/*.py",
    "igris/web/static/**",
    "igris/web/templates/**",
    "scripts/**",
    "tests/*.py",
    "tests/test_*.py",
    "tests/fixtures/**",
)


def _path_from_porcelain(line: str) -> str:
    text = line.strip()
    parts = text.split(maxsplit=1)
    path = parts[1].strip() if len(parts) == 2 else text[2:].strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1].strip()
    return path


def is_allowed_git_status_line(line: str) -> bool:
    """Return True for legitimate source/test/doc changes."""
    path_text = _path_from_porcelain(line)
    path = Path(path_text)
    parts = set(path.parts)
    name = path.name

    if parts & BLOCKED_PARTS:
        return False
    if name in BLOCKED_NAMES:
        return False
    if name.endswith("~") or name.endswith(".bak") or name.endswith(".swp"):
        return False
    if path.suffix in BLOCKED_SUFFIXES:
        return False
    if "secret" in name.lower() or "token" in name.lower():
        return False

    return any(fnmatch.fnmatch(path_text, pattern) for pattern in ALLOWED_PATTERNS)


def assert_git_status_policy() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=".",
        check=False,
    )
    lines = [line for line in result.stdout.strip().split("\n") if line.strip()]
    unexpected = [line for line in lines if not is_allowed_git_status_line(line)]
    assert unexpected == [], f"Unexpected changed files: {unexpected}"
