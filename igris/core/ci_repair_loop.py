"""
ci_repair_loop.py — Epic #1071

Devin-level CI repair loop for DeliveryWorkflow.

When CI fails on a PR, this module orchestrates:
  1. Fetch CI logs (via `gh run view --log`)
  2. Diagnose the failure (structured: import_error / syntax_error / lint / test)
  3. Attempt a deterministic fix if possible (ruff --fix, ruff format)
  4. If deterministic fix fails or not applicable, build a targeted LLM goal
  5. Record the attempt and result
  6. Repeat up to MAX_ATTEMPTS times

Usage:
    loop = CIRepairLoop(project_root, pr_number=123, original_goal="...")
    result = loop.run(backend)
    if result.resolved:
        print("CI fixed!")
    else:
        print("Could not fix:", result.failure_summary)
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

_log = logging.getLogger("igris.ci_repair_loop")

MAX_ATTEMPTS: int = 3
LINT_COMMANDS: List[List[str]] = [
    ["ruff", "check", "--fix", "."],
    ["ruff", "format", "."],
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CIRepairAttempt:
    """Record of one CI repair attempt."""
    attempt: int
    failure_type: str
    strategy: str           # "deterministic_lint" | "llm_repair" | "skip"
    goal_sent: str
    success: bool
    duration_seconds: float
    error: str = ""
    files_fixed: List[str] = field(default_factory=list)


@dataclass
class CIRepairResult:
    """Final result of the CI repair loop."""
    resolved: bool
    attempts: List[CIRepairAttempt] = field(default_factory=list)
    failure_summary: str = ""
    total_duration_seconds: float = 0.0

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CIRepairLoop:
    """Orchestrates CI failure diagnosis and repair for a PR."""

    def __init__(
        self,
        project_root: str,
        pr_number: int,
        original_goal: str,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self.project_root = project_root
        self.pr_number = pr_number
        self.original_goal = original_goal
        self.max_attempts = max_attempts
        self._attempts: List[CIRepairAttempt] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, backend: Any, wait_for_ci: Optional[Callable] = None) -> CIRepairResult:
        """Run the CI repair loop.

        Args:
            backend: supervisor backend with run_reasoning() method
            wait_for_ci: optional callable(pr_number) → CIStatus (defaults
                         to polling gh pr checks)

        Returns CIRepairResult with resolved=True if CI becomes green.
        """
        start = time.monotonic()

        for attempt_num in range(self.max_attempts):
            _log.info(
                "CIRepairLoop: attempt %d/%d for PR #%d",
                attempt_num + 1, self.max_attempts, self.pr_number,
            )

            # 1. Fetch and diagnose CI failure
            log_text = self._fetch_ci_logs()
            diagnosis = self._diagnose(log_text)
            failure_type = diagnosis.get("failure_type", "unknown")

            _log.info(
                "CIRepairLoop: diagnosed %r — failing tests: %d",
                failure_type, len(diagnosis.get("failing_tests", [])),
            )

            # 2. Try deterministic fix for lint errors
            if failure_type == "lint_error":
                attempt = self._try_deterministic_lint_fix(attempt_num)
                self._attempts.append(attempt)
                if attempt.success:
                    # Push fix and check CI
                    self._push_fix("ci-repair: fix lint errors")
                    if self._ci_is_green(wait_for_ci):
                        break
                    continue

            # 3. Build targeted LLM goal
            repair_goal = self._build_llm_repair_goal(diagnosis)
            attempt_start = time.monotonic()

            try:
                result = backend.run_reasoning(
                    repair_goal,
                    max_steps=60,
                    initial_context={
                        "pr_number": self.pr_number,
                        "failure_type": failure_type,
                        "failing_tests": diagnosis.get("failing_tests", []),
                        "diagnosis": diagnosis,
                    },
                    timeout=600,
                    task_type="code_repair",
                    preferred_profile=None,
                )
                success = str(result.get("status", "")) == "finished"
                attempt = CIRepairAttempt(
                    attempt=attempt_num,
                    failure_type=failure_type,
                    strategy="llm_repair",
                    goal_sent=repair_goal[:500],
                    success=success,
                    duration_seconds=round(time.monotonic() - attempt_start, 1),
                    error=result.get("final_summary", "")[:300] if not success else "",
                )
            except Exception as exc:
                attempt = CIRepairAttempt(
                    attempt=attempt_num,
                    failure_type=failure_type,
                    strategy="llm_repair",
                    goal_sent=repair_goal[:500],
                    success=False,
                    duration_seconds=round(time.monotonic() - attempt_start, 1),
                    error=str(exc)[:200],
                )
                _log.warning("CIRepairLoop: LLM repair raised: %s", exc)

            self._attempts.append(attempt)

            if attempt.success and self._ci_is_green(wait_for_ci):
                break

        resolved = bool(self._attempts) and self._ci_is_green(wait_for_ci)
        summary = "" if resolved else self._build_failure_summary()

        return CIRepairResult(
            resolved=resolved,
            attempts=self._attempts,
            failure_summary=summary,
            total_duration_seconds=round(time.monotonic() - start, 1),
        )

    # ------------------------------------------------------------------
    # CI log fetching
    # ------------------------------------------------------------------

    def _fetch_ci_logs(self) -> str:
        """Fetch CI failure logs for the PR via gh CLI."""
        try:
            # Get the latest failed run ID
            runs = subprocess.run(
                ["gh", "pr", "checks", str(self.pr_number),
                 "--json", "name,state,detailsUrl"],
                cwd=self.project_root, capture_output=True, text=True, timeout=30,
            )
            if runs.returncode != 0:
                return runs.stderr[:500]

            # Fetch the logs of the first failed check
            log_result = subprocess.run(
                ["gh", "run", "list", "--limit", "1",
                 "--json", "databaseId,conclusion"],
                cwd=self.project_root, capture_output=True, text=True, timeout=30,
            )
            if log_result.returncode == 0:
                import json
                run_list = json.loads(log_result.stdout or "[]")
                if run_list:
                    run_id = run_list[0].get("databaseId")
                    if run_id:
                        log_fetch = subprocess.run(
                            ["gh", "run", "view", str(run_id), "--log-failed"],
                            cwd=self.project_root, capture_output=True, text=True, timeout=60,
                        )
                        return (log_fetch.stdout or "") + (log_fetch.stderr or "")
        except Exception as exc:
            _log.warning("CIRepairLoop._fetch_ci_logs: %s", exc)
        return ""

    # ------------------------------------------------------------------
    # Diagnosis
    # ------------------------------------------------------------------

    def _diagnose(self, log_text: str) -> Dict[str, Any]:
        """Classify CI failure from log text."""
        import re
        failure_type = "unknown"
        failing_tests: List[str] = []

        if "ImportError" in log_text or "ModuleNotFoundError" in log_text:
            failure_type = "import_error"
        elif "SyntaxError" in log_text or "IndentationError" in log_text:
            failure_type = "syntax_error"
        elif "ruff" in log_text.lower() and ("error" in log_text.lower() or "warning" in log_text.lower()):
            failure_type = "lint_error"
        elif "FAILED tests/" in log_text or "AssertionError" in log_text:
            failure_type = "test_failure"
            pattern = re.compile(r"FAILED\s+(tests/[^\s:]+(?:::[^\s]+)?)")
            seen: Dict[str, bool] = {}
            for m in pattern.finditer(log_text):
                t = m.group(1)
                if t not in seen:
                    seen[t] = True
                    failing_tests.append(t)

        return {
            "failure_type": failure_type,
            "failing_tests": failing_tests,
            "log_excerpt": log_text[:2000],
        }

    # ------------------------------------------------------------------
    # Deterministic lint fix
    # ------------------------------------------------------------------

    def _try_deterministic_lint_fix(self, attempt: int) -> CIRepairAttempt:
        """Run ruff --fix + ruff format deterministically."""
        start = time.monotonic()
        files_fixed: List[str] = []

        for cmd in LINT_COMMANDS:
            try:
                r = subprocess.run(
                    cmd, cwd=self.project_root,
                    capture_output=True, text=True, timeout=60,
                )
                if r.returncode == 0:
                    files_fixed.extend(
                        line.strip() for line in r.stdout.splitlines()
                        if line.strip() and not line.startswith("Found")
                    )
            except Exception as exc:
                _log.warning("CIRepairLoop._try_deterministic_lint_fix %s: %s", cmd, exc)

        success = bool(files_fixed)
        return CIRepairAttempt(
            attempt=attempt,
            failure_type="lint_error",
            strategy="deterministic_lint",
            goal_sent="ruff --fix + ruff format",
            success=success,
            duration_seconds=round(time.monotonic() - start, 1),
            files_fixed=files_fixed,
        )

    # ------------------------------------------------------------------
    # LLM repair goal building
    # ------------------------------------------------------------------

    def _build_llm_repair_goal(self, diagnosis: Dict[str, Any]) -> str:
        """Build a focused repair goal for the LLM."""
        failure_type = diagnosis.get("failure_type", "unknown")
        failing_tests = diagnosis.get("failing_tests", [])
        log_excerpt = diagnosis.get("log_excerpt", "")[:800]

        header = (
            f"CI REPAIR — PR #{self.pr_number}\n"
            f"Failure type: {failure_type}\n\n"
        )

        if failure_type == "test_failure" and failing_tests:
            tests_str = "\n".join(f"  - {t}" for t in failing_tests[:10])
            return (
                f"{header}"
                f"The following tests are failing in CI:\n{tests_str}\n\n"
                f"Fix the SOURCE CODE (not the tests) to make these tests pass.\n"
                f"Do NOT modify any file in tests/.\n"
                f"Do NOT introduce new functionality beyond what is needed.\n\n"
                f"Original mission context:\n{self.original_goal[:400]}"
            )
        elif failure_type == "import_error":
            return (
                f"{header}"
                f"There is an ImportError or ModuleNotFoundError in CI.\n"
                f"Check recently added/modified files for missing imports, "
                f"missing __init__.py, or incorrect package paths.\n\n"
                f"CI log excerpt:\n{log_excerpt}\n\n"
                f"Original mission context:\n{self.original_goal[:400]}"
            )
        elif failure_type == "syntax_error":
            return (
                f"{header}"
                f"There is a SyntaxError or IndentationError in CI.\n"
                f"Check recently modified Python files for syntax issues.\n\n"
                f"CI log excerpt:\n{log_excerpt}\n\n"
                f"Original mission context:\n{self.original_goal[:400]}"
            )
        else:
            return (
                f"{header}"
                f"CI failed with the following log excerpt:\n{log_excerpt}\n\n"
                f"Investigate and fix the root cause without breaking passing tests.\n"
                f"Original mission context:\n{self.original_goal[:400]}"
            )

    # ------------------------------------------------------------------
    # CI green check
    # ------------------------------------------------------------------

    def _ci_is_green(self, wait_fn: Optional[Callable]) -> bool:
        """Poll CI status via wait_fn or gh pr checks."""
        if wait_fn:
            try:
                status = wait_fn(self.pr_number)
                return getattr(status, "status", "") == "green"
            except Exception:
                return False

        try:
            import json
            r = subprocess.run(
                ["gh", "pr", "checks", str(self.pr_number),
                 "--json", "name,state"],
                cwd=self.project_root, capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                return False
            checks = json.loads(r.stdout or "[]")
            if not checks:
                return False
            return all(
                c.get("state", "") in ("SUCCESS", "success", "NEUTRAL", "neutral", "SKIPPED")
                for c in checks
            )
        except Exception:
            return False

    def _push_fix(self, message: str) -> bool:
        """Commit and push any staged/modified files."""
        try:
            subprocess.run(["git", "add", "-u"], cwd=self.project_root, check=True, capture_output=True)
            r = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.project_root, capture_output=True, text=True,
            )
            if r.returncode != 0:
                return False
            subprocess.run(["git", "push"], cwd=self.project_root, check=True, capture_output=True)
            return True
        except Exception as exc:
            _log.warning("CIRepairLoop._push_fix: %s", exc)
            return False

    def _build_failure_summary(self) -> str:
        """Build a human-readable failure summary."""
        lines = [f"CIRepairLoop failed after {len(self._attempts)} attempt(s):"]
        for a in self._attempts:
            lines.append(
                f"  [{a.attempt+1}] {a.strategy} / {a.failure_type} — "
                f"success={a.success} — {a.error[:100]}"
            )
        return "\n".join(lines)
