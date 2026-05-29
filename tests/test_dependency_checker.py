"""Tests for igris/core/dependency_checker.py (issues #614 / #819)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from igris.core.dependency_checker import (
    DependencyChecker,
    load_dep_file,
    parse_depends_on_labels,
    save_dep_file,
)


# ---------------------------------------------------------------------------
# parse_depends_on_labels
# ---------------------------------------------------------------------------

class TestParseDependsOnLabels:
    def test_plain_string_labels(self):
        labels = ["depends-on-614", "bug", "depends-on-522"]
        deps = parse_depends_on_labels(labels)
        assert sorted(deps) == [522, 614]

    def test_dict_labels(self):
        labels = [{"name": "depends-on-800"}, {"name": "enhancement"}]
        deps = parse_depends_on_labels(labels)
        assert deps == [800]

    def test_empty_list_returns_empty(self):
        assert parse_depends_on_labels([]) == []

    def test_no_dep_labels_returns_empty(self):
        assert parse_depends_on_labels(["bug", "good first issue"]) == []

    def test_deduplicates(self):
        deps = parse_depends_on_labels(["depends-on-100", "depends-on-100"])
        assert deps == [100]

    def test_mixed_plain_and_dict(self):
        labels = ["depends-on-10", {"name": "depends-on-20"}]
        deps = parse_depends_on_labels(labels)
        assert sorted(deps) == [10, 20]


# ---------------------------------------------------------------------------
# save_dep_file / load_dep_file
# ---------------------------------------------------------------------------

class TestDepFile:
    def test_save_and_load(self, tmp_path):
        mapping = {614: [522, 523], 819: [614]}
        save_dep_file(str(tmp_path), mapping)
        loaded = load_dep_file(str(tmp_path))
        assert loaded["614"] == [522, 523]
        assert loaded["819"] == [614]

    def test_load_missing_returns_empty(self, tmp_path):
        assert load_dep_file(str(tmp_path)) == {}

    def test_load_corrupt_returns_empty(self, tmp_path):
        (tmp_path / ".igris").mkdir()
        (tmp_path / ".igris" / "dependencies.json").write_text("NOT JSON")
        assert load_dep_file(str(tmp_path)) == {}

    def test_save_deduplicates_values(self, tmp_path):
        save_dep_file(str(tmp_path), {1: [5, 5, 5]})
        loaded = load_dep_file(str(tmp_path))
        assert loaded["1"] == [5]

    def test_atomic_write(self, tmp_path):
        # Verify no .tmp leftover
        save_dep_file(str(tmp_path), {1: [2]})
        assert not list(tmp_path.rglob("*.tmp"))


# ---------------------------------------------------------------------------
# DependencyChecker.check()
# ---------------------------------------------------------------------------

class TestDependencyCheckerCheck:
    def _checker(self, tmp_path, labels_map=None, issue_states=None, pr_merged=None):
        """Build a DependencyChecker with mocked label/state responses."""
        states = issue_states or {}
        merged = pr_merged or {}

        def gh_labels(issue_number):
            return labels_map.get(issue_number, []) if labels_map else []

        checker = DependencyChecker(str(tmp_path), gh_labels_fn=gh_labels)

        with patch("igris.core.dependency_checker._gh_issue_state", side_effect=lambda root, n: states.get(n)):
            with patch("igris.core.dependency_checker._gh_pr_merged", side_effect=lambda root, n: merged.get(n)):
                return checker, states, merged

    def test_no_deps_always_ok(self, tmp_path):
        checker = DependencyChecker(str(tmp_path), gh_labels_fn=lambda n: [])
        with patch("igris.core.dependency_checker._gh_issue_state", return_value=None):
            with patch("igris.core.dependency_checker._gh_pr_merged", return_value=None):
                ok, unsat = checker.check(999)
        assert ok is True
        assert unsat == []

    def test_closed_dep_satisfies(self, tmp_path):
        def labels(n):
            return ["depends-on-100"] if n == 200 else []

        checker = DependencyChecker(str(tmp_path), gh_labels_fn=labels)
        with patch("igris.core.dependency_checker._gh_issue_state", side_effect=lambda r, n: "closed" if n == 100 else "open"):
            with patch("igris.core.dependency_checker._gh_pr_merged", return_value=None):
                ok, unsat = checker.check(200)
        assert ok is True
        assert unsat == []

    def test_open_dep_blocks(self, tmp_path):
        def labels(n):
            return ["depends-on-100"] if n == 200 else []

        checker = DependencyChecker(str(tmp_path), gh_labels_fn=labels)
        with patch("igris.core.dependency_checker._gh_issue_state", return_value="open"):
            with patch("igris.core.dependency_checker._gh_pr_merged", return_value=False):
                ok, unsat = checker.check(200)
        assert ok is False
        assert 100 in unsat

    def test_merged_pr_satisfies(self, tmp_path):
        def labels(n):
            return ["depends-on-55"] if n == 60 else []

        checker = DependencyChecker(str(tmp_path), gh_labels_fn=labels)
        with patch("igris.core.dependency_checker._gh_issue_state", return_value="open"):
            with patch("igris.core.dependency_checker._gh_pr_merged", side_effect=lambda r, n: True if n == 55 else False):
                ok, unsat = checker.check(60)
        assert ok is True

    def test_dep_from_file_when_no_labels(self, tmp_path):
        save_dep_file(str(tmp_path), {300: [200]})
        checker = DependencyChecker(str(tmp_path), gh_labels_fn=lambda n: [])
        with patch("igris.core.dependency_checker._gh_issue_state", side_effect=lambda r, n: "closed" if n == 200 else "open"):
            with patch("igris.core.dependency_checker._gh_pr_merged", return_value=None):
                ok, unsat = checker.check(300)
        assert ok is True

    def test_labels_take_priority_over_dep_file(self, tmp_path):
        # dep file says 300 depends on 999 (open), but label says 300 depends on 200 (closed)
        save_dep_file(str(tmp_path), {300: [999]})

        def labels(n):
            return ["depends-on-200"] if n == 300 else []

        checker = DependencyChecker(str(tmp_path), gh_labels_fn=labels)
        with patch("igris.core.dependency_checker._gh_issue_state", side_effect=lambda r, n: "closed" if n == 200 else "open"):
            with patch("igris.core.dependency_checker._gh_pr_merged", return_value=None):
                ok, unsat = checker.check(300)
        assert ok is True  # label-based dep (200) is closed, so ok

    def test_multiple_deps_all_must_be_satisfied(self, tmp_path):
        def labels(n):
            return ["depends-on-10", "depends-on-20"] if n == 30 else []

        checker = DependencyChecker(str(tmp_path), gh_labels_fn=labels)
        with patch("igris.core.dependency_checker._gh_issue_state", side_effect=lambda r, n: "closed" if n == 10 else "open"):
            with patch("igris.core.dependency_checker._gh_pr_merged", return_value=None):
                ok, unsat = checker.check(30)
        assert ok is False
        assert 20 in unsat
        assert 10 not in unsat


# ---------------------------------------------------------------------------
# Circular dependency
# ---------------------------------------------------------------------------

class TestDependencyCheckerCircular:
    def test_circular_dep_does_not_deadlock(self, tmp_path):
        """A -> B -> A must not recurse infinitely; result is False (unsatisfied)."""
        # Both A (500) and B (501) are 'open'
        def labels(n):
            if n == 500:
                return ["depends-on-501"]
            if n == 501:
                return ["depends-on-500"]
            return []

        checker = DependencyChecker(str(tmp_path), gh_labels_fn=labels)
        with patch("igris.core.dependency_checker._gh_issue_state", return_value="open"):
            with patch("igris.core.dependency_checker._gh_pr_merged", return_value=False):
                ok, unsat = checker.check(500)
        # Should not hang; result is blocked (neither is closed)
        assert ok is False

    def test_has_circular_dependency_detects_cycle(self, tmp_path):
        def labels(n):
            if n == 500:
                return ["depends-on-501"]
            if n == 501:
                return ["depends-on-500"]
            return []

        checker = DependencyChecker(str(tmp_path), gh_labels_fn=labels)
        assert checker.has_circular_dependency(500) is True

    def test_has_circular_dependency_false_when_linear(self, tmp_path):
        def labels(n):
            return ["depends-on-200"] if n == 300 else []

        checker = DependencyChecker(str(tmp_path), gh_labels_fn=labels)
        assert checker.has_circular_dependency(300) is False

    def test_self_dependency_does_not_deadlock(self, tmp_path):
        """An issue that depends on itself should not loop."""
        def labels(n):
            return ["depends-on-700"]

        checker = DependencyChecker(str(tmp_path), gh_labels_fn=labels)
        with patch("igris.core.dependency_checker._gh_issue_state", return_value="open"):
            with patch("igris.core.dependency_checker._gh_pr_merged", return_value=False):
                ok, unsat = checker.check(700)
        assert ok is False  # it depends on itself which is open
