"""DependencyChecker — dependency model and launch-gate rules for IGRIS.

Part of GitHub issues #614 / #819: feat(supervisor): Dependency checker.
Fase 2bis — Gap 7.

When IGRIS picks an issue to work on, this module checks whether all declared
dependencies are satisfied (merged PR or closed issue) before allowing the run
to start.

Dependencies are declared in two ways (in priority order):
1. Issue labels: ``depends-on-NNN`` (e.g. ``depends-on-614``)
2. Fallback: ``.igris/dependencies.json`` mapping  {"issue_number": [dep, dep, ...]}

Usage from self_repair_supervisor (before starting a run)::

    checker = DependencyChecker(project_root, gh_client)
    ok, unsatisfied = checker.check(issue_number)
    if not ok:
        return self._blocked(run, "dependency_not_satisfied",
                             f"deps not ready: {unsatisfied}")
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("igris.dependency_checker")

_DEP_FILE = ".igris/dependencies.json"


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def parse_depends_on_labels(labels: List[Any]) -> List[int]:
    """Extract dependency issue numbers from a list of label strings/dicts.

    Accepts both plain strings and GitHub label objects (dicts with "name" key).
    """
    deps: List[int] = []
    for label in labels:
        name = label if isinstance(label, str) else label.get("name", "")
        m = re.match(r"^depends-on-(\d+)$", str(name).strip())
        if m:
            deps.append(int(m.group(1)))
    return sorted(set(deps))


# ---------------------------------------------------------------------------
# Dependency file helpers
# ---------------------------------------------------------------------------

def _dep_file_path(project_root: str) -> Path:
    return Path(project_root) / _DEP_FILE


def load_dep_file(project_root: str) -> Dict[str, List[int]]:
    """Return {str(issue_number): [dep_issue_number, ...]} from disk."""
    path = _dep_file_path(project_root)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): [int(d) for d in v] for k, v in data.items()}
    except Exception:
        return {}


def save_dep_file(project_root: str, mapping: Dict[int, List[int]]) -> None:
    """Persist {issue_number: [dep, ...]} to .igris/dependencies.json."""
    path = _dep_file_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {str(k): sorted(set(v)) for k, v in mapping.items()}
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, str(path))


# ---------------------------------------------------------------------------
# GitHub state helpers
# ---------------------------------------------------------------------------

def _gh_issue_state(project_root: str, issue_number: int) -> Optional[str]:
    """Return 'open' | 'closed' | None via gh CLI."""
    try:
        r = subprocess.run(
            ["gh", "issue", "view", str(issue_number), "--json", "state,stateReason"],
            capture_output=True, text=True, cwd=project_root, timeout=10,
        )
        if r.returncode == 0:
            data = json.loads(r.stdout)
            return str(data.get("state", "")).lower()
    except Exception:
        pass
    return None


def _gh_pr_merged(project_root: str, issue_number: int) -> Optional[bool]:
    """Return True if a PR for the issue is merged, False if not, None on error."""
    try:
        # Try treating issue_number as a PR number directly
        r = subprocess.run(
            ["gh", "pr", "view", str(issue_number), "--json", "state,merged"],
            capture_output=True, text=True, cwd=project_root, timeout=10,
        )
        if r.returncode == 0:
            data = json.loads(r.stdout)
            return bool(data.get("merged", False))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------

class DependencyChecker:
    """Check whether all dependencies of an issue are satisfied.

    Args:
        project_root: Absolute path to the IGRIS project root.
        gh_labels_fn: Optional callable(issue_number) -> List[label] that returns
            issue labels. Used in tests to inject mock labels without a real CLI.
    """

    def __init__(
        self,
        project_root: str,
        gh_labels_fn: Optional[Any] = None,
    ) -> None:
        self._root = project_root
        self._gh_labels_fn = gh_labels_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_satisfied(self, issue_number: int, _visited: Optional[Set[int]] = None) -> bool:
        """Return True if issue is closed or its PR is merged."""
        if _visited is None:
            _visited = set()
        return self._is_satisfied_internal(issue_number, _visited)

    def check(self, issue_number: int) -> Tuple[bool, List[int]]:
        """Check all declared dependencies for issue_number.

        Returns:
            (ok, unsatisfied_deps) — ok is True if all deps satisfied.
        """
        deps = self._get_deps(issue_number)
        if not deps:
            return True, []

        visited: Set[int] = set()
        unsatisfied = [d for d in deps if not self._is_dep_satisfied(d, set(visited))]
        return len(unsatisfied) == 0, unsatisfied

    def has_circular_dependency(self, issue_number: int) -> bool:
        """Return True if issue_number has a circular dependency chain."""
        return self._has_cycle(issue_number, set())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_labels(self, issue_number: int) -> List[Any]:
        if self._gh_labels_fn is not None:
            try:
                return list(self._gh_labels_fn(issue_number))
            except Exception:
                return []
        try:
            r = subprocess.run(
                ["gh", "issue", "view", str(issue_number), "--json", "labels"],
                capture_output=True, text=True, cwd=self._root, timeout=10,
            )
            if r.returncode == 0:
                data = json.loads(r.stdout)
                return data.get("labels", [])
        except Exception:
            pass
        return []

    def _get_deps(self, issue_number: int) -> List[int]:
        """Return list of dependency issue numbers for issue_number."""
        labels = self._get_labels(issue_number)
        from_labels = parse_depends_on_labels(labels)
        if from_labels:
            return from_labels
        # Fallback to dep file
        dep_map = load_dep_file(self._root)
        return dep_map.get(str(issue_number), [])

    def _is_satisfied_internal(self, issue_number: int, visited: Set[int]) -> bool:
        """Check if a single issue is closed or its PR merged."""
        state = _gh_issue_state(self._root, issue_number)
        if state == "closed":
            return True
        merged = _gh_pr_merged(self._root, issue_number)
        if merged is True:
            return True
        return False

    def _is_dep_satisfied(self, dep_number: int, visited: Set[int]) -> bool:
        """Recursively check if dep_number and its own deps are all satisfied.

        The visited set prevents infinite loops on circular dependency chains.
        """
        # Circular dependency guard — treat as satisfied to avoid deadlock
        if dep_number in visited:
            return True
        visited = visited | {dep_number}

        if not self._is_satisfied_internal(dep_number, visited):
            return False

        # Also check transitive deps
        sub_deps = self._get_deps(dep_number)
        return all(self._is_dep_satisfied(d, visited) for d in sub_deps)

    def _has_cycle(self, issue_number: int, visited: Set[int]) -> bool:
        """DFS cycle detection in dependency graph."""
        if issue_number in visited:
            return True
        visited = visited | {issue_number}
        for dep in self._get_deps(issue_number):
            if self._has_cycle(dep, visited):
                return True
        return False
