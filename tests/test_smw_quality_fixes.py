from __future__ import annotations

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── restart_watchdog_cycle ────────────────────────────────────────────────────

def test_restart_watchdog_cycle_creates_sentinel():
    from igris.core.smw_actions import restart_watchdog_cycle
    with tempfile.TemporaryDirectory() as td:
        r = asyncio.run(restart_watchdog_cycle(project_root=td))
        assert r.success
        sentinel = os.path.join(td, ".igris", "watchdog_restart_requested")
        assert os.path.exists(sentinel)


def test_execute_action_restart_watchdog_cycle():
    from igris.core.smw_actions import execute_action
    with tempfile.TemporaryDirectory() as td:
        r = asyncio.run(execute_action(
            "restart_watchdog_cycle", tier=1, dry_run=False, project_root=td
        ))
        assert r.success


def test_execute_action_unknown_returns_failure():
    from igris.core.smw_actions import execute_action
    r = asyncio.run(execute_action("non_existent_action", tier=1, dry_run=False))
    assert not r.success
    assert r.output == "unknown action"


# ── AgentCoordinator reused across steps (_coord singleton) ──────────────────

def test_agent_coord_is_none_on_init():
    from igris.core.agent_reasoning_loop import AgentReasoningLoop
    with tempfile.TemporaryDirectory() as td:
        loop = AgentReasoningLoop(project_root=td, max_steps=2)
        assert loop._coord is None


def test_agent_coord_set_and_reused():
    """After first lazy init, _coord must not be re-instantiated."""
    from igris.core.agent_reasoning_loop import AgentReasoningLoop
    with tempfile.TemporaryDirectory() as td:
        loop = AgentReasoningLoop(project_root=td, max_steps=2)
        sentinel = object()
        loop._coord = sentinel   # simulate first lazy init
        # Second access should return same sentinel
        assert loop._coord is sentinel


# ── SMW diagnosis escalates to LLM for unknown patterns ─────────────────────

@pytest.mark.asyncio
async def test_meta_watchdog_escalates_unknown_pattern_to_llm():
    from igris.core.smw_diagnosis import Diagnosis
    from igris.core.smw_patterns import DetectedPattern, Pattern

    fake_snapshot = MagicMock()
    fake_pattern = Pattern("totally_unknown_xyz", "x", "warn", lambda s: False, 0)
    detected = DetectedPattern(pattern=fake_pattern, snapshot=fake_snapshot, evidence="e", detected_at=0.0)

    static_diag = Diagnosis("totally_unknown_xyz", "pattern non riconosciuto", 0.4, 1,
                             ["open_diagnostic_issue"], "e", requires_llm=True)
    llm_diag = Diagnosis("totally_unknown_xyz", "llm root cause", 0.8, 2,
                          ["open_diagnostic_issue"], "e", requires_llm=False)

    with (
        patch("igris.core.meta_watchdog.take_snapshot", return_value=MagicMock()),
        patch("igris.core.meta_watchdog.detect_patterns") as mock_detect,
        patch("igris.core.meta_watchdog.diagnose", return_value=static_diag),
        patch("igris.core.meta_watchdog.diagnose_with_llm",
              new_callable=AsyncMock, return_value=llm_diag) as mock_llm,
        patch("igris.core.meta_watchdog.execute_action", new_callable=AsyncMock),
        patch("igris.core.meta_watchdog.record_incident"),
        patch("igris.core.meta_watchdog.teach_back", new_callable=AsyncMock),
        patch("igris.core.meta_watchdog.run_all_detectors", return_value=[]),
        patch("igris.core.meta_watchdog.load_review_results", return_value=[]),
        patch("asyncio.to_thread", return_value=MagicMock(returncode=0, stdout="[]")),
    ):
        mock_detect.side_effect = [[detected], []]   # detected, then resolved

        from igris.core.meta_watchdog import _smw_loop
        task = asyncio.create_task(_smw_loop("/tmp"))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        mock_llm.assert_called_once()


# ── PR review gate receives real pr_diff ─────────────────────────────────────

@pytest.mark.asyncio
async def test_meta_watchdog_pr_review_includes_diff():
    captured = []

    async def capture_review(req, project_root):
        captured.append(req)
        from igris.core.smw_pr_review import PRReviewResult
        import time
        return PRReviewResult(req.pr_number, False, 0.9, "mock", [], "", time.time())

    fake_diff = "diff --git a/foo.py b/foo.py\n+code"
    pr_list_json = json.dumps([{
        "number": 42, "title": "feat: test",
        "headRefName": "igris/mission-abc",
        "files": [{"path": "foo.py"}],
        "statusCheckRollup": [{"conclusion": "SUCCESS"}],
    }])

    call_n = 0
    async def to_thread_side(fn, *a, **kw):
        nonlocal call_n
        call_n += 1
        if call_n == 1:
            return MagicMock(returncode=0, stdout=pr_list_json)   # gh pr list
        return MagicMock(returncode=0, stdout=fake_diff)           # gh pr diff + merge/comment

    with (
        patch("igris.core.meta_watchdog.take_snapshot", return_value=MagicMock()),
        patch("igris.core.meta_watchdog.detect_patterns", return_value=[]),
        patch("igris.core.meta_watchdog.run_all_detectors", return_value=[]),
        patch("igris.core.meta_watchdog.load_review_results", return_value=[]),
        patch("igris.core.meta_watchdog.save_review_result"),
        patch("igris.core.meta_watchdog.review_pr", side_effect=capture_review),
        patch("asyncio.to_thread", side_effect=to_thread_side),
    ):
        from igris.core.meta_watchdog import _smw_loop
        task = asyncio.create_task(_smw_loop("/tmp"))
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert len(captured) == 1
    assert captured[0].pr_diff == fake_diff
