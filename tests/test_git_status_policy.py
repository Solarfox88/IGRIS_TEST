"""Tests for policy-based git status validation."""

from __future__ import annotations

from tests.git_status_policy import is_allowed_git_status_line


def test_new_rank_test_file_is_allowed():
    assert is_allowed_git_status_line("?? tests/test_version_info.py")


def test_runtime_artifacts_are_blocked():
    blocked = [
        "?? .env",
        "?? logs/igris.log",
        "?? .pytest_cache/v/cache/nodeids",
        "?? .igris/runtime.json",
        "?? .venv/bin/python",
        "?? igris/web/__pycache__/server.cpython-312.pyc",
        "?? tmp-output.log",
    ]

    for line in blocked:
        assert is_allowed_git_status_line(line) is False


def test_legitimate_source_test_and_docs_paths_are_allowed():
    allowed = [
        " M igris/web/server.py",
        "?? tests/git_status_policy.py",
        " M tests/test_dashboard_tabs.py",
        "?? tests/test_new_rank_case.py",
        " M docs/OPERATIONAL_BASELINE.md",
        " M README.md",
    ]

    for line in allowed:
        assert is_allowed_git_status_line(line) is True
