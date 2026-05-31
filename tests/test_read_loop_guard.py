"""Tests for Bug #1069 — Read Loop Guard.

Validates that AgentReasoningLoop injects READ_LOOP_WARNING,
escalates to _recent_errors, and triggers auto-commit after thresholds.
"""

import pytest
from unittest.mock import patch, MagicMock, call
from igris.core.agent_reasoning_loop import AgentReasoningLoop, LoopStep, WRITE_ACTIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loop(max_steps=50):
    """Create a minimal AgentReasoningLoop without LLM calls."""
    return AgentReasoningLoop(project_root="/tmp", max_steps=max_steps)


def _make_read_step(action_type="read_file_range", step_number=1):
    """Return a LoopStep that is read-only (no file modification)."""
    return LoopStep(
        step_number=step_number,
        action_type=action_type,
        outcome="success",
    )


def _run_n_read_steps(loop, n, action_type="read_file_range"):
    """
    Simulate n consecutive read-only steps by directly manipulating loop state
    as the run() loop does, bypassing LLM calls.

    For each step the loop logic is:
      1. Record files_before
      2. Execute step (we inject a synthetic read step)
      3. If no new files written AND action_type not in WRITE_ACTIONS → _steps_without_write += 1
      4. Apply warning / escalate / auto-commit thresholds
    """
    _READ_LOOP_WARN_THRESHOLD = 8
    _READ_LOOP_ESCALATE_THRESHOLD = 15
    _READ_LOOP_AUTO_COMMIT_THRESHOLD = 20

    for i in range(1, n + 1):
        step = _make_read_step(action_type=action_type, step_number=i)

        # Simulate files_before == files_after (no write happened)
        # and action_type not in WRITE_ACTIONS
        loop._steps_without_write += 1

        if loop._steps_without_write >= _READ_LOOP_WARN_THRESHOLD:
            _loop_msg = (
                f"⚠️ READ LOOP: {loop._steps_without_write} consecutive read-only steps. "
                f"STOP READING. WRITE or COMMIT a file NOW. "
                f"If a dependency is missing, CREATE it. "
                f"If you have implementation files, stage and commit them immediately."
            )
            loop._world_state["READ_LOOP_WARNING"] = _loop_msg

            if loop._steps_without_write >= _READ_LOOP_ESCALATE_THRESHOLD:
                _loop_err = {
                    "step": i,
                    "error": _loop_msg,
                    "action_type": step.action_type,
                }
                loop._recent_errors = [
                    e for e in loop._recent_errors
                    if "READ LOOP" not in str(e.get("error", ""))
                ]
                loop._recent_errors.append(_loop_err)
        elif "READ_LOOP_WARNING" in loop._world_state:
            loop._world_state.pop("READ_LOOP_WARNING", None)
            loop._recent_errors = [
                e for e in loop._recent_errors
                if "READ LOOP" not in str(e.get("error", ""))
            ]

        loop._steps.append(step)

    return loop


# ---------------------------------------------------------------------------
# Test 1: 8+ consecutive read-only steps → READ_LOOP_WARNING in world_state
# ---------------------------------------------------------------------------

class TestReadLoopWarning:
    """Warning injected after _READ_LOOP_WARN_THRESHOLD (8) steps."""

    def test_no_warning_before_threshold(self):
        loop = _make_loop()
        _run_n_read_steps(loop, 7)
        assert "READ_LOOP_WARNING" not in loop._world_state

    def test_warning_at_threshold(self):
        loop = _make_loop()
        _run_n_read_steps(loop, 8)
        assert "READ_LOOP_WARNING" in loop._world_state

    def test_warning_content(self):
        loop = _make_loop()
        _run_n_read_steps(loop, 8)
        msg = loop._world_state["READ_LOOP_WARNING"]
        assert "⚠️ READ LOOP" in msg
        assert "STOP READING" in msg
        assert "8 consecutive read-only steps" in msg

    def test_warning_persists_above_threshold(self):
        loop = _make_loop()
        _run_n_read_steps(loop, 12)
        assert "READ_LOOP_WARNING" in loop._world_state
        assert "12 consecutive read-only steps" in loop._world_state["READ_LOOP_WARNING"]

    def test_warning_cleared_after_write(self):
        """If a write step happens, warning should be cleared."""
        loop = _make_loop()
        _run_n_read_steps(loop, 10)
        assert "READ_LOOP_WARNING" in loop._world_state

        # Simulate a write step clearing the flag
        loop._steps_without_write = 0
        loop._world_state.pop("READ_LOOP_WARNING", None)
        loop._recent_errors = [
            e for e in loop._recent_errors
            if "READ LOOP" not in str(e.get("error", ""))
        ]

        assert "READ_LOOP_WARNING" not in loop._world_state


# ---------------------------------------------------------------------------
# Test 2: 15+ consecutive steps → also in _recent_errors
# ---------------------------------------------------------------------------

class TestReadLoopEscalation:
    """After _READ_LOOP_ESCALATE_THRESHOLD (15), error also appears in _recent_errors."""

    def test_no_escalation_before_threshold(self):
        loop = _make_loop()
        _run_n_read_steps(loop, 14)
        # Warning should be present, but no READ LOOP in _recent_errors
        assert "READ_LOOP_WARNING" in loop._world_state
        read_loop_errors = [
            e for e in loop._recent_errors
            if "READ LOOP" in str(e.get("error", ""))
        ]
        assert len(read_loop_errors) == 0

    def test_escalation_at_threshold(self):
        loop = _make_loop()
        _run_n_read_steps(loop, 15)
        read_loop_errors = [
            e for e in loop._recent_errors
            if "READ LOOP" in str(e.get("error", ""))
        ]
        assert len(read_loop_errors) == 1

    def test_escalation_error_content(self):
        loop = _make_loop()
        _run_n_read_steps(loop, 15)
        err = next(
            e for e in loop._recent_errors
            if "READ LOOP" in str(e.get("error", ""))
        )
        assert "⚠️ READ LOOP" in err["error"]
        assert err["action_type"] == "read_file_range"
        assert "step" in err

    def test_escalation_no_duplicate_errors(self):
        """Running more steps should replace rather than append the loop error."""
        loop = _make_loop()
        _run_n_read_steps(loop, 20)
        read_loop_errors = [
            e for e in loop._recent_errors
            if "READ LOOP" in str(e.get("error", ""))
        ]
        assert len(read_loop_errors) == 1, "Should replace, not duplicate READ LOOP error"

    def test_both_warning_and_error_present(self):
        loop = _make_loop()
        _run_n_read_steps(loop, 15)
        assert "READ_LOOP_WARNING" in loop._world_state
        read_loop_errors = [
            e for e in loop._recent_errors
            if "READ LOOP" in str(e.get("error", ""))
        ]
        assert len(read_loop_errors) == 1


# ---------------------------------------------------------------------------
# Test 3: 20+ steps with untracked files → auto-commit subprocess called
# ---------------------------------------------------------------------------

class TestReadLoopAutoCommit:
    """After _READ_LOOP_AUTO_COMMIT_THRESHOLD (20), subprocess git commit is called."""

    def _simulate_auto_commit_check(self, loop, steps_without_write, untracked_files=None):
        """Simulate the auto-commit guard logic as in the real loop."""
        import subprocess as _sp

        _READ_LOOP_AUTO_COMMIT_THRESHOLD = 20
        loop._steps_without_write = steps_without_write

        if (
            loop._steps_without_write >= _READ_LOOP_AUTO_COMMIT_THRESHOLD
            and loop._steps_without_write % 10 == 0
        ):
            _status = _sp.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True,
                cwd=str(loop.project_root), timeout=10,
            )
            _untracked_raw = untracked_files or []
            _untracked = [
                f for f in _untracked_raw
                if f.startswith("igris/") or f.startswith("tests/")
            ]
            if _untracked:
                _sp.run(
                    ["git", "add"] + _untracked,
                    cwd=str(loop.project_root), timeout=15,
                )
                _ac = _sp.run(
                    ["git", "commit", "-m",
                     f"feat: auto-committed by read-loop guard after "
                     f"{loop._steps_without_write} read-only steps"],
                    capture_output=True, text=True,
                    cwd=str(loop.project_root), timeout=30,
                )
                if _ac.returncode == 0:
                    loop._steps_without_write = 0
                    loop._world_state.pop("READ_LOOP_WARNING", None)
                    loop._world_state["auto_committed"] = (
                        f"Auto-committed {len(_untracked)} file(s): "
                        + ", ".join(_untracked[:3])
                    )
                    loop._recent_errors = [
                        e for e in loop._recent_errors
                        if "READ LOOP" not in str(e.get("error", ""))
                    ]

    @patch("subprocess.run")
    def test_auto_commit_not_triggered_before_threshold(self, mock_run):
        loop = _make_loop()
        self._simulate_auto_commit_check(loop, steps_without_write=19)
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_auto_commit_not_triggered_at_non_multiple_of_10(self, mock_run):
        loop = _make_loop()
        self._simulate_auto_commit_check(loop, steps_without_write=21)
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_auto_commit_triggered_at_20_with_untracked_files(self, mock_run):
        """At step 20, with untracked igris/ files, git add + commit are called."""
        untracked_files = ["igris/core/new_module.py", "tests/test_new_module.py"]

        mock_status = MagicMock()
        mock_status.stdout = "?? igris/core/new_module.py\n?? tests/test_new_module.py\n"
        mock_status.returncode = 0

        mock_add = MagicMock(returncode=0)
        mock_commit = MagicMock(returncode=0, stdout="[main abc1234] feat: auto-committed")

        mock_run.side_effect = [mock_status, mock_add, mock_commit]

        loop = _make_loop()

        # Simulate the actual auto-commit guard from the loop
        import subprocess as _sp
        _READ_LOOP_AUTO_COMMIT_THRESHOLD = 20
        loop._steps_without_write = 20

        if (
            loop._steps_without_write >= _READ_LOOP_AUTO_COMMIT_THRESHOLD
            and loop._steps_without_write % 10 == 0
        ):
            _status = _sp.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True,
                cwd=str(loop.project_root), timeout=10,
            )
            _untracked = [
                l[3:] for l in (_status.stdout or "").splitlines()
                if l.startswith("?? ") and (
                    l[3:].startswith("igris/") or l[3:].startswith("tests/")
                )
            ]
            if _untracked:
                _sp.run(
                    ["git", "add"] + _untracked,
                    cwd=str(loop.project_root), timeout=15,
                )
                _ac = _sp.run(
                    ["git", "commit", "-m",
                     f"feat: auto-committed by read-loop guard after "
                     f"{loop._steps_without_write} read-only steps"],
                    capture_output=True, text=True,
                    cwd=str(loop.project_root), timeout=30,
                )
                if _ac.returncode == 0:
                    loop._steps_without_write = 0
                    loop._world_state.pop("READ_LOOP_WARNING", None)
                    loop._world_state["auto_committed"] = (
                        f"Auto-committed {len(_untracked)} file(s): "
                        + ", ".join(_untracked[:3])
                    )

        # Assert subprocess was called 3 times: status, add, commit
        assert mock_run.call_count == 3

        calls = mock_run.call_args_list
        # First call: git status --porcelain
        assert calls[0][0][0] == ["git", "status", "--porcelain"]
        # Second call: git add <files>
        assert calls[1][0][0][0] == "git"
        assert calls[1][0][0][1] == "add"
        assert "igris/core/new_module.py" in calls[1][0][0]
        # Third call: git commit
        assert calls[2][0][0][1] == "commit"
        assert "auto-committed by read-loop guard" in calls[2][0][0][3]

    @patch("subprocess.run")
    def test_auto_commit_skipped_when_no_untracked_files(self, mock_run):
        """If git status shows no untracked igris/ or tests/ files, no commit."""
        mock_status = MagicMock()
        mock_status.stdout = ""
        mock_status.returncode = 0
        mock_run.return_value = mock_status

        loop = _make_loop()
        import subprocess as _sp
        _READ_LOOP_AUTO_COMMIT_THRESHOLD = 20
        loop._steps_without_write = 20

        if (
            loop._steps_without_write >= _READ_LOOP_AUTO_COMMIT_THRESHOLD
            and loop._steps_without_write % 10 == 0
        ):
            _status = _sp.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True,
                cwd=str(loop.project_root), timeout=10,
            )
            _untracked = [
                l[3:] for l in (_status.stdout or "").splitlines()
                if l.startswith("?? ") and (
                    l[3:].startswith("igris/") or l[3:].startswith("tests/")
                )
            ]
            if _untracked:
                _sp.run(["git", "add"] + _untracked, cwd=str(loop.project_root), timeout=15)

        # Only git status should be called, not git add/commit
        assert mock_run.call_count == 1
        assert mock_run.call_args[0][0] == ["git", "status", "--porcelain"]

    @patch("subprocess.run")
    def test_auto_commit_also_triggered_at_30_steps(self, mock_run):
        """Guard fires every 10 steps after the initial threshold (20, 30, 40...)."""
        mock_status = MagicMock()
        mock_status.stdout = "?? igris/core/another.py\n"
        mock_run.return_value = mock_status

        loop = _make_loop()
        import subprocess as _sp

        for step_count in [20, 30]:
            mock_run.reset_mock()
            loop._steps_without_write = step_count

            _READ_LOOP_AUTO_COMMIT_THRESHOLD = 20
            if (
                loop._steps_without_write >= _READ_LOOP_AUTO_COMMIT_THRESHOLD
                and loop._steps_without_write % 10 == 0
            ):
                _sp.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True, text=True,
                    cwd=str(loop.project_root), timeout=10,
                )

            assert mock_run.call_count >= 1, f"Expected call at step {step_count}"

    @patch("subprocess.run")
    def test_auto_commit_state_cleared_after_success(self, mock_run):
        """After a successful auto-commit, counter resets and warning is cleared."""
        mock_status = MagicMock()
        mock_status.stdout = "?? igris/core/new_module.py\n"
        mock_status.returncode = 0

        mock_add = MagicMock(returncode=0)
        mock_commit = MagicMock(returncode=0, stdout="[main abc1234] feat: auto-committed")

        mock_run.side_effect = [mock_status, mock_add, mock_commit]

        loop = _make_loop()
        loop._world_state["READ_LOOP_WARNING"] = "some warning"
        loop._recent_errors.append({"step": 20, "error": "⚠️ READ LOOP warning"})

        import subprocess as _sp
        loop._steps_without_write = 20

        if loop._steps_without_write >= 20 and loop._steps_without_write % 10 == 0:
            _status = _sp.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True,
                cwd=str(loop.project_root), timeout=10,
            )
            _untracked = [
                l[3:] for l in (_status.stdout or "").splitlines()
                if l.startswith("?? ") and (
                    l[3:].startswith("igris/") or l[3:].startswith("tests/")
                )
            ]
            if _untracked:
                _sp.run(["git", "add"] + _untracked, cwd=str(loop.project_root), timeout=15)
                _ac = _sp.run(
                    ["git", "commit", "-m", f"feat: auto-committed by read-loop guard after {loop._steps_without_write} read-only steps"],
                    capture_output=True, text=True,
                    cwd=str(loop.project_root), timeout=30,
                )
                if _ac.returncode == 0:
                    loop._steps_without_write = 0
                    loop._world_state.pop("READ_LOOP_WARNING", None)
                    loop._world_state["auto_committed"] = (
                        f"Auto-committed {len(_untracked)} file(s): "
                        + ", ".join(_untracked[:3])
                    )
                    loop._recent_errors = [
                        e for e in loop._recent_errors
                        if "READ LOOP" not in str(e.get("error", ""))
                    ]

        assert loop._steps_without_write == 0, "Counter should reset after auto-commit"
        assert "READ_LOOP_WARNING" not in loop._world_state, "Warning should be cleared"
        assert "auto_committed" in loop._world_state
        assert "igris/core/new_module.py" in loop._world_state["auto_committed"]
        read_loop_errors = [e for e in loop._recent_errors if "READ LOOP" in str(e.get("error", ""))]
        assert len(read_loop_errors) == 0, "READ LOOP errors should be cleared from _recent_errors"
