from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CIStatus:
    status: str
    failed_jobs: List[str]
    logs_url: str


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
