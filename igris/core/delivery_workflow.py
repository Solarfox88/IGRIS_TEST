from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List


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
        key = str(pr_number)
        for attempt in range(max_attempts):
            ci = self.wait_for_ci(pr_number)
            if ci.status == "green":
                try:
                    from igris.core.memory_graph import MemoryGraph
                    MemoryGraph(self.project_root).add_node("lesson", {"event_type": "ci_fix_success", "pr_number": pr_number, "attempts": attempt + 1, "failed_jobs": ci.failed_jobs})
                except Exception:
                    pass
                return True
            if ci.status != "red":
                break
            self._fix_attempts[key] = self._fix_attempts.get(key, 0) + 1
            if self._fix_attempts[key] > max_attempts:
                break
        try:
            from igris.core.smw_weak_signals import run_all_detectors, save_weak_signals
            signals = run_all_detectors(self.project_root)
            save_weak_signals(signals, self.project_root)
        except Exception:
            pass
        return False

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
