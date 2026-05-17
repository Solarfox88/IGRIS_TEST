"""Tests for igris/core/acceptance_gate.py — semantic stub rejection gate."""

from __future__ import annotations

import pytest

from igris.core.acceptance_gate import (
    AcceptanceResult,
    check_acceptance_evidence,
    extract_required_endpoints,
    _added_lines,
    _is_stub_diff,
)


# ---------------------------------------------------------------------------
# extract_required_endpoints
# ---------------------------------------------------------------------------


class TestExtractRequiredEndpoints:
    def test_extracts_api_path(self):
        goal = "Implement GET /api/diagnostics/session-resume endpoint in server.py"
        eps = extract_required_endpoints(goal)
        assert "/api/diagnostics/session-resume" in eps

    def test_extracts_multiple_paths(self):
        goal = "Add /api/health and /api/status endpoints"
        eps = extract_required_endpoints(goal)
        assert "/api/health" in eps
        assert "/api/status" in eps

    def test_ignores_noise(self):
        goal = "Fix bug in the code. No endpoint specified."
        eps = extract_required_endpoints(goal)
        assert eps == []

    def test_extracts_path_with_segment(self):
        goal = "Implement /api/rank/runs endpoint"
        eps = extract_required_endpoints(goal)
        assert any("/api/rank/runs" in ep for ep in eps)

    def test_does_not_extract_file_path_fragment(self):
        """File paths embedded in 'in igris/web/server.py' must not be extracted as endpoints.

        Regression: 'igris/web/server.py' produced '/web/server' as a false endpoint,
        causing 'Required endpoint /web/server not found in diff' failures.
        """
        goal = (
            "Implement GET /api/diagnostics/session-resume endpoint "
            "in igris/web/server.py. Add tests in tests/test_session_resume_endpoint.py."
        )
        eps = extract_required_endpoints(goal)
        assert "/web/server" not in eps, f"'/web/server' must not be extracted from file path: {eps}"
        assert "/test_session_resume_endpoint" not in eps, (
            f"'/test_session_resume_endpoint' must not be extracted: {eps}"
        )
        assert "/api/diagnostics/session-resume" in eps, (
            f"Real endpoint must still be extracted: {eps}"
        )

    def test_does_not_extract_py_extension_path(self):
        """Paths ending in .py or other file extensions must be excluded."""
        goal = "Modify igris/core/supervisor.py to add /api/health endpoint"
        eps = extract_required_endpoints(goal)
        assert not any(ep.endswith(".py") for ep in eps), f"No .py paths should be extracted: {eps}"
        assert "/api/health" in eps


# ---------------------------------------------------------------------------
# _is_stub_diff
# ---------------------------------------------------------------------------


class TestIsStubDiff:
    def _make_added(self, body: str) -> str:
        return "\n".join(f"+{line}" for line in body.splitlines())

    def test_todo_comment_is_stub(self):
        diff = self._make_added("    # TODO: implement this")
        assert _is_stub_diff(diff) is not None

    def test_placeholder_comment_is_stub(self):
        diff = self._make_added("    # placeholder logic here")
        assert _is_stub_diff(diff) is not None

    def test_logic_to_comment_is_stub(self):
        diff = self._make_added("    # Logic to gather diagnostics data")
        assert _is_stub_diff(diff) is not None

    def test_pass_only_body_is_stub(self):
        diff = self._make_added("    pass")
        assert _is_stub_diff(diff) is not None

    def test_three_empty_list_values_is_stub(self):
        diff = self._make_added(
            "    return {'zombie_runs': [], 'active_runs': [], 'stale_branches': []}"
        )
        assert _is_stub_diff(diff) is not None

    def test_real_logic_is_not_stub(self):
        diff = self._make_added(
            "    runs = list_supervised_runs()\n"
            "    zombie = [r for r in runs if r['status']=='running' and _is_zombie(r)]\n"
            "    return {'zombie_runs': zombie}"
        )
        assert _is_stub_diff(diff) is None

    def test_single_empty_list_not_flagged(self):
        diff = self._make_added(
            "    errors = []\n"
            "    for item in items:\n"
            "        errors.append(item.validate())\n"
            "    return errors"
        )
        assert _is_stub_diff(diff) is None


# ---------------------------------------------------------------------------
# check_acceptance_evidence — endpoint coverage
# ---------------------------------------------------------------------------


def _make_diff(server_added: str, test_added: str = "") -> str:
    """Build a minimal unified diff with server.py and optionally a test file."""
    parts = [
        "diff --git a/igris/web/server.py b/igris/web/server.py",
        "--- a/igris/web/server.py",
        "+++ b/igris/web/server.py",
        "@@ -1,3 +1,10 @@",
    ]
    parts += [f"+{line}" for line in server_added.splitlines()]
    if test_added:
        parts += [
            "diff --git a/tests/test_session.py b/tests/test_session.py",
            "--- /dev/null",
            "+++ b/tests/test_session.py",
            "@@ -0,0 +1,10 @@",
        ]
        parts += [f"+{line}" for line in test_added.splitlines()]
    return "\n".join(parts)


_GOAL = "Implement GET /api/diagnostics/session-resume endpoint"

_REAL_SERVER = """
@app.get('/api/diagnostics/session-resume')
async def session_resume():
    runs = list_supervised_runs()
    zombie = [r for r in runs if _is_zombie(r)]
    return JSONResponse(content={'zombie_runs': zombie, 'active_runs': [r for r in runs if not _is_zombie(r)]})
"""

_STUB_SERVER = """
@app.get('/api/diagnostics/session-resume')
async def session_resume():
    # Logic to gather diagnostics data
    return JSONResponse(content={'zombie_runs': [], 'active_runs': [], 'stale_branches': []})
"""

_TEST_COVERAGE = """
def test_session_resume():
    resp = client.get('/api/diagnostics/session-resume')
    assert resp.status_code == 200
    data = resp.json()
    assert 'zombie_runs' in data
    assert isinstance(data['zombie_runs'], list)
"""


class TestCheckAcceptanceEvidence:
    def test_real_impl_with_test_passes(self):
        diff = _make_diff(_REAL_SERVER, _TEST_COVERAGE)
        result = check_acceptance_evidence(
            _GOAL, diff, ["igris/web/server.py", "tests/test_session.py"]
        )
        assert result.passed
        assert not result.missing_evidence

    def test_stub_endpoint_fails(self):
        diff = _make_diff(_STUB_SERVER, _TEST_COVERAGE)
        result = check_acceptance_evidence(
            _GOAL, diff, ["igris/web/server.py", "tests/test_session.py"]
        )
        assert not result.passed
        assert any("Stub" in m or "stub" in m.lower() for m in result.missing_evidence)

    def test_endpoint_without_test_fails(self):
        diff = _make_diff(_REAL_SERVER)  # no test file in diff
        result = check_acceptance_evidence(
            _GOAL, diff, ["igris/web/server.py"]
        )
        assert not result.passed
        assert any("test" in m.lower() for m in result.missing_evidence)

    def test_no_endpoint_in_diff_fails(self):
        # Diff adds an unrelated function, not the required endpoint
        diff = _make_diff("def helper(): return 42")
        result = check_acceptance_evidence(
            _GOAL, diff, ["igris/web/server.py"]
        )
        assert not result.passed
        assert any("/api/diagnostics/session-resume" in m for m in result.missing_evidence)

    def test_goal_without_endpoint_passes_through(self):
        # Generic goal — no endpoint to check, no stubs detected
        diff = _make_diff("x = 1 + 2\ny = x * 3")
        result = check_acceptance_evidence(
            "Fix arithmetic bug in calculation module",
            diff,
            ["igris/core/calc.py"],
        )
        assert result.passed
        assert result.required_endpoints == []

    def test_hardcoded_empty_list_stub_rejected(self):
        stub = """
@app.get('/api/diagnostics/session-resume')
async def session_resume():
    return {'zombie_runs': [], 'active_runs': [], 'stale_branches': [], 'open_supervisor_issues': []}
"""
        diff = _make_diff(stub)
        result = check_acceptance_evidence(
            _GOAL, diff, ["igris/web/server.py"]
        )
        assert not result.passed

    def test_report_contains_found_and_missing(self):
        diff = _make_diff(_REAL_SERVER, _TEST_COVERAGE)
        result = check_acceptance_evidence(
            _GOAL, diff, ["igris/web/server.py", "tests/test_session.py"]
        )
        assert isinstance(result.found_evidence, list)
        assert isinstance(result.missing_evidence, list)
        assert isinstance(result.required_endpoints, list)
        assert "/api/diagnostics/session-resume" in result.required_endpoints

    def test_empty_diff_passes_gate(self):
        # Empty diff is already caught by _rank_passed; gate should not double-fire
        result = check_acceptance_evidence(_GOAL, "", ["igris/web/server.py"])
        assert result.passed
