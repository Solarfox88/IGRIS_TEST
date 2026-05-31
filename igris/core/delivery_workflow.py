from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_log = logging.getLogger("igris.delivery")

# Epic #1071 — Stale branch age threshold in seconds (default: 14 days)
STALE_BRANCH_AGE_SECONDS = int(14 * 24 * 3600)


@dataclass
class CIStatus:
    status: str
    failed_jobs: List[str]
    logs_url: str
    # Epic #1071 — structured failure diagnosis
    failure_type: str = ""
    failing_tests: List[str] = field(default_factory=list)
    log_excerpt: str = ""


@dataclass
class BranchHygieneReport:
    """Epic #1071 — Result of branch hygiene check."""
    branch: str
    is_stale: bool
    age_days: float
    last_commit_ts: float
    recommendation: str  # "ok" | "warn" | "delete"


class DeliveryWorkflow:
    def __init__(self, project_root: str) -> None:
        self.project_root = project_root
        self._fix_attempts: Dict[str, int] = {}

    def create_mission_branch(self, mission_id: str) -> str:
        branch = f"igris/mission-{mission_id[:8]}"
        subprocess.run(["git", "checkout", "-b", branch], cwd=self.project_root, check=True, capture_output=True)
        return branch

    def commit_staged(self, message: str, files: List[str]) -> bool:
        for f in files:
            subprocess.run(["git", "add", f], cwd=self.project_root, check=True)
        result = subprocess.run(["git", "commit", "-m", message], cwd=self.project_root, capture_output=True, text=True)
        return result.returncode == 0

    def open_pr(self, branch: str, title: str, body: str, closes_issues: List[int]) -> str:
        full_body = body
        if closes_issues:
            full_body += "\n\n" + " ".join(f"Closes #{n}" for n in closes_issues)
        result = subprocess.run(["gh", "pr", "create", "--title", title, "--body", full_body, "--head", branch, "--base", "main"], cwd=self.project_root, capture_output=True, text=True, check=True)
        return result.stdout.strip()

    def wait_for_ci(self, pr_number: int, timeout: int = 600, poll: int = 30) -> CIStatus:
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = subprocess.run(["gh", "pr", "checks", str(pr_number), "--json", "name,status,conclusion"], cwd=self.project_root, capture_output=True, text=True)
            if result.returncode != 0:
                time.sleep(poll)
                continue
            checks = json.loads(result.stdout or "[]")
            if not checks:
                time.sleep(poll)
                continue
            if [c for c in checks if c.get("status") != "completed"]:
                time.sleep(poll)
                continue
            failed = [c for c in checks if c.get("conclusion") not in ("success", "skipped", "neutral")]
            if not failed:
                return CIStatus("green", [], "")
            return CIStatus("red", [c["name"] for c in failed], "")
        return CIStatus("timeout", [], "")

    def fix_ci_loop(self, pr_number: int, max_attempts: int = 3) -> bool:
        for attempt in range(max_attempts):
            ci = self.wait_for_ci(pr_number)
            if ci.status == "green":
                self._record_ci_fix_success(pr_number, attempt + 1, ci.failed_jobs)
                return True
            if ci.status != "red":
                break

            diagnosis = self._diagnose_ci_failure(pr_number, ci.failed_jobs)
            if not diagnosis:
                break
            fixed = self._apply_ci_fix(diagnosis)
            if not fixed:
                break
            push_ok = self._push_fix_commit(f"fix(ci): repair {', '.join(ci.failed_jobs[:3])} [attempt {attempt + 1}]")
            if not push_ok:
                break

        self._record_weak_signals()
        return False

    def _record_ci_fix_success(self, pr_number: int, attempts: int, failed_jobs: List[str]) -> None:
        try:
            from igris.core.memory_graph import MemoryGraph

            MemoryGraph(self.project_root).add_node(
                "lesson",
                {"event_type": "ci_fix_success", "pr_number": pr_number, "attempts": attempts, "failed_jobs": failed_jobs},
            )
        except Exception:
            pass

    def _record_weak_signals(self) -> None:
        try:
            from igris.core.smw_weak_signals import run_all_detectors, save_weak_signals

            signals = run_all_detectors(self.project_root)
            save_weak_signals(signals, self.project_root)
        except Exception:
            pass

    def _diagnose_ci_failure(self, pr_number: int, failed_jobs: List[str]) -> Optional[dict]:
        result = subprocess.run(
            ["gh", "run", "list", "--json", "databaseId,status,conclusion,name", "--pr", str(pr_number), "--limit", "1"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        runs = json.loads(result.stdout or "[]")
        if not runs:
            return None
        run_id = runs[0].get("databaseId")
        if not run_id:
            return None
        log_result = subprocess.run(
            ["gh", "run", "view", str(run_id), "--log-failed"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
        )
        log_text = log_result.stdout[:6000] if log_result.returncode == 0 else ""
        if not log_text:
            return None
        failure_type = "unknown"
        if "ImportError" in log_text or "ModuleNotFoundError" in log_text:
            failure_type = "import_error"
        elif "FAILED tests/" in log_text or "AssertionError" in log_text:
            failure_type = "test_failure"
        elif "SyntaxError" in log_text:
            failure_type = "syntax_error"
        elif "ruff" in log_text.lower() or "flake8" in log_text.lower():
            failure_type = "lint_error"
        return {"run_id": run_id, "failed_jobs": failed_jobs, "failure_type": failure_type, "log_excerpt": log_text}

    def _apply_ci_fix(self, diagnosis: dict) -> bool:
        failure_type = diagnosis.get("failure_type", "unknown")
        if failure_type == "lint_error":
            result = subprocess.run(
                ["python", "-m", "ruff", "check", "--fix", "."],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
            subprocess.run(["git", "add", "-u"], cwd=self.project_root, capture_output=True)
            return result.returncode == 0
        if failure_type == "test_failure":
            try:
                from igris.core.memory_graph import MemoryGraph

                MemoryGraph(self.project_root).add_node(
                    "lesson",
                    {
                        "event_type": "ci_test_failure_needs_llm",
                        "log_excerpt": str(diagnosis.get("log_excerpt", ""))[:1000],
                        "failed_jobs": diagnosis.get("failed_jobs", []),
                    },
                    confidence=0.7,
                )
            except Exception:
                pass
            return False
        return False

    def _push_fix_commit(self, message: str) -> bool:
        status = subprocess.run(
            ["git", "diff", "--cached", "--name-only"], cwd=self.project_root, capture_output=True, text=True
        )
        if not status.stdout.strip():
            return False
        commit = subprocess.run(["git", "commit", "-m", message], cwd=self.project_root, capture_output=True, text=True)
        if commit.returncode != 0:
            return False
        push = subprocess.run(["git", "push"], cwd=self.project_root, capture_output=True, text=True)
        return push.returncode == 0

    def update_issue(self, issue_number: int, comment: str) -> bool:
        result = subprocess.run(["gh", "issue", "comment", str(issue_number), "--body", comment], cwd=self.project_root, capture_output=True, text=True)
        return result.returncode == 0

    def merge_pr(self, pr_number: int) -> bool:
        result = subprocess.run(["gh", "pr", "merge", str(pr_number), "--squash", "--auto"], cwd=self.project_root, capture_output=True, text=True)
        return result.returncode == 0

    def verify_and_unsaturate(self, family: str) -> None:
        try:
            from igris.core.memory_graph import MemoryGraph
            MemoryGraph(self.project_root).unsaturate_family(family)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Epic #1071 — CI failure diagnosis
    # ------------------------------------------------------------------

    def parse_failing_tests(self, log_text: str) -> List[str]:
        """Extract failing test names from pytest output.

        Recognises patterns like:
          FAILED tests/test_foo.py::TestBar::test_baz
          FAILED tests/test_foo.py::test_func
        Returns a deduplicated list of test node IDs.
        """
        pattern = re.compile(r"FAILED\s+(tests/[^\s:]+(?:::[^\s]+)?)")
        found = pattern.findall(log_text)
        seen: Dict[str, bool] = {}
        result = []
        for item in found:
            if item not in seen:
                seen[item] = True
                result.append(item)
        return result

    def diagnose_ci_failure_structured(self, log_text: str, failed_jobs: List[str]) -> dict:
        """Return a structured CI failure diagnosis from raw log text.

        Epic #1071 — enriches the existing diagnosis with parsed test names
        and a human-readable summary so the repair loop can target specific
        failing tests rather than retrying the whole suite.
        """
        failure_type = "unknown"
        failing_tests: List[str] = []

        if "ImportError" in log_text or "ModuleNotFoundError" in log_text:
            failure_type = "import_error"
        elif "SyntaxError" in log_text:
            failure_type = "syntax_error"
        elif "ruff" in log_text.lower() or "flake8" in log_text.lower():
            failure_type = "lint_error"
        elif "FAILED tests/" in log_text or "AssertionError" in log_text:
            failure_type = "test_failure"
            failing_tests = self.parse_failing_tests(log_text)

        if failure_type == "test_failure" and failing_tests:
            summary = (
                f"CI test failure: {len(failing_tests)} test(s) failed — "
                + ", ".join(failing_tests[:5])
            )
        elif failure_type != "unknown":
            summary = f"CI failure type: {failure_type} in job(s): {', '.join(failed_jobs[:3])}"
        else:
            summary = f"CI failure: {len(failed_jobs)} job(s) failed; review logs for details"

        _log.info("diagnose_ci_failure_structured: type=%s, failing_tests=%d", failure_type, len(failing_tests))
        return {
            "failure_type": failure_type,
            "failing_tests": failing_tests,
            "failed_jobs": failed_jobs,
            "log_excerpt": log_text[:2000],
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Epic #1071 — Branch hygiene check
    # ------------------------------------------------------------------

    def check_branch_hygiene(self, branch: str) -> BranchHygieneReport:
        """Check if *branch* is stale (older than STALE_BRANCH_AGE_SECONDS).

        Uses `git log` to find the last commit timestamp. Returns a
        BranchHygieneReport with staleness status and recommendation.
        """
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%ct", branch],
                cwd=self.project_root,
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return BranchHygieneReport(
                    branch=branch, is_stale=False, age_days=0.0,
                    last_commit_ts=0.0, recommendation="ok",
                )
            last_commit_ts = float(result.stdout.strip())
            age_seconds = time.time() - last_commit_ts
            age_days = age_seconds / 86400
            is_stale = age_seconds > STALE_BRANCH_AGE_SECONDS

            if is_stale:
                recommendation = "delete" if age_days > 30 else "warn"
                _log.warning(
                    "check_branch_hygiene: branch %r is stale (%.1f days old)",
                    branch, age_days,
                )
            else:
                recommendation = "ok"

            return BranchHygieneReport(
                branch=branch,
                is_stale=is_stale,
                age_days=round(age_days, 1),
                last_commit_ts=last_commit_ts,
                recommendation=recommendation,
            )
        except Exception as exc:
            _log.warning("check_branch_hygiene: failed for branch %r: %s", branch, exc)
            return BranchHygieneReport(
                branch=branch, is_stale=False, age_days=0.0,
                last_commit_ts=0.0, recommendation="ok",
            )

    # ------------------------------------------------------------------
    # Epic #1071 — PR review gate (wait for CI before merge)
    # ------------------------------------------------------------------

    def pr_review_gate(
        self,
        pr_number: int,
        *,
        require_green_ci: bool = True,
        timeout: int = 600,
    ) -> Tuple[bool, str]:
        """Block until CI passes (or times out) before allowing merge.

        Returns (True, "green") if CI passes within timeout, or
        (False, reason) if CI fails or times out.

        Epic #1071 — prevents merging PRs with failing CI.
        """
        if not require_green_ci:
            _log.info("pr_review_gate: CI gate bypassed (require_green_ci=False)")
            return True, "bypassed"

        ci = self.wait_for_ci(pr_number, timeout=timeout)
        if ci.status == "green":
            _log.info("pr_review_gate: CI is green for PR #%d", pr_number)
            return True, "green"
        elif ci.status == "timeout":
            _log.warning("pr_review_gate: CI timed out for PR #%d after %ds", pr_number, timeout)
            return False, f"ci_timeout_after_{timeout}s"
        else:
            _log.warning(
                "pr_review_gate: CI is red for PR #%d, failed jobs: %s",
                pr_number, ci.failed_jobs,
            )
            return False, f"ci_red: {', '.join(ci.failed_jobs[:5])}"

    def merge_pr_after_ci(self, pr_number: int, timeout: int = 600) -> Tuple[bool, str]:
        """Merge only after CI passes. Returns (success, reason)."""
        gate_ok, gate_reason = self.pr_review_gate(pr_number, timeout=timeout)
        if not gate_ok:
            return False, f"pr_review_gate_failed: {gate_reason}"
        merged = self.merge_pr(pr_number)
        return merged, "merged" if merged else "merge_failed"

    # ------------------------------------------------------------------
    # Epic #1071 — Branch cleanup and LLM-guided CI repair
    # ------------------------------------------------------------------

    def delete_merged_branch(self, branch: str, *, remote: bool = True) -> bool:
        """Delete a branch after successful merge.

        Deletes both the remote tracking ref and the local branch.
        Returns True if deletion succeeded (or branch did not exist).
        """
        success = True
        if remote:
            try:
                r = subprocess.run(
                    ["git", "push", "origin", "--delete", branch],
                    cwd=self.project_root, capture_output=True, text=True, timeout=30,
                )
                if r.returncode != 0 and "remote ref does not exist" not in r.stderr:
                    _log.warning("delete_merged_branch: remote delete failed for %r: %s", branch, r.stderr[:200])
                    success = False
            except Exception as exc:
                _log.warning("delete_merged_branch: remote delete exception for %r: %s", branch, exc)
                success = False

        # Delete local branch (may not exist if we only have the remote)
        try:
            r = subprocess.run(
                ["git", "branch", "-d", branch],
                cwd=self.project_root, capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0 and "not found" not in r.stderr and "error: branch" not in r.stderr:
                _log.warning("delete_merged_branch: local delete failed for %r: %s", branch, r.stderr[:200])
        except Exception as exc:
            _log.warning("delete_merged_branch: local delete exception for %r: %s", branch, exc)

        _log.info("delete_merged_branch: %r remote=%s success=%s", branch, remote, success)
        return success

    def merge_with_cleanup(
        self,
        pr_number: int,
        branch: str,
        *,
        timeout: int = 600,
        delete_branch: bool = True,
    ) -> Tuple[bool, str]:
        """Full merge pipeline: CI gate → merge → branch cleanup.

        Returns (success, reason). On merge success, deletes branch if
        delete_branch=True. Logs hygiene report for the branch before merge.
        """
        # Log hygiene for visibility before deleting
        hygiene = self.check_branch_hygiene(branch)
        _log.info(
            "merge_with_cleanup: branch %r — age=%.1fd recommendation=%s",
            branch, hygiene.age_days, hygiene.recommendation,
        )

        merged, reason = self.merge_pr_after_ci(pr_number, timeout=timeout)
        if not merged:
            return False, reason

        if delete_branch:
            self.delete_merged_branch(branch)

        return True, "merged_and_cleaned"

    def suggest_ci_repair_goal(self, diagnosis: dict, original_goal: str) -> str:
        """Build a targeted repair goal string for LLM-guided CI repair.

        Epic #1071 — rather than retrying the full original goal, this
        produces a focused goal string that tells the repair reasoning loop
        exactly which tests are failing and what to fix.

        The returned string can be passed to backend.run_reasoning as the goal.
        """
        failure_type = diagnosis.get("failure_type", "unknown")
        failing_tests = diagnosis.get("failing_tests", [])
        failed_jobs = diagnosis.get("failed_jobs", [])
        summary = diagnosis.get("summary", "")

        if failure_type == "test_failure" and failing_tests:
            test_list = ", ".join(f"`{t}`" for t in failing_tests[:5])
            return (
                f"CI repair: fix the following failing tests in the current branch.\n\n"
                f"Failing tests: {test_list}\n\n"
                f"Do NOT change any test file. Fix only the source code to make these tests pass.\n"
                f"Original mission context: {original_goal[:300]}"
            )
        elif failure_type == "lint_error":
            return (
                f"CI repair: fix lint/style errors in the current branch.\n"
                f"Run `ruff check --fix` and `ruff format` on changed files.\n"
                f"Original mission context: {original_goal[:300]}"
            )
        elif failure_type == "import_error":
            return (
                f"CI repair: fix ImportError/ModuleNotFoundError in the current branch.\n"
                f"Failed jobs: {', '.join(failed_jobs[:3])}\n"
                f"Check missing imports, __init__.py files, and package structure.\n"
                f"Original mission context: {original_goal[:300]}"
            )
        elif failure_type == "syntax_error":
            return (
                f"CI repair: fix SyntaxError/IndentationError in the current branch.\n"
                f"Check recently edited Python files for syntax problems.\n"
                f"Original mission context: {original_goal[:300]}"
            )
        else:
            return (
                f"CI repair: CI failed with the following summary: {summary}\n"
                f"Failed jobs: {', '.join(failed_jobs[:3])}\n"
                f"Investigate and fix the root cause. Do not break passing tests.\n"
                f"Original mission context: {original_goal[:300]}"
            )
