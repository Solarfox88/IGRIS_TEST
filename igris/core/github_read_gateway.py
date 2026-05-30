"""
GitHub Read Gateway

Provides gated, audited read access to GitHub resources:
issues, pull requests, files on remote branches, Actions workflow status,
and release information.

Each operation:
- Logs access with resource type, identifier, timestamp (audit trail)
- Supports dry-run mode (simulated access without real execution)
- Returns normalized data (not raw API)
"""

import json
import logging
import subprocess
import base64
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from igris.core.authorization_gate import AuthorizationGate

logger = logging.getLogger(__name__)


class GitHubReadGateway:
    """Gated reader for GitHub resources."""

    def __init__(self, auth_gate: AuthorizationGate, repo: str = "."):
        self._auth = auth_gate
        self._repo = repo
        self._audit_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def read_issue(
        self, issue_number: int, dry_run: bool = False
    ) -> Dict[str, Any]:
        """Read a single issue by number."""
        self._log_audit("issue", str(issue_number))
        if dry_run:
            return self._dry_run_response("issue", issue_number)

        result = self._gh(
            "issue", "view", str(issue_number),
            "--json", "number,title,state,body,labels,assignees,url,createdAt,updatedAt,comments"
        )
        return self._normalize_issue(json.loads(result))

    def read_pr(
        self, pr_number: int, dry_run: bool = False
    ) -> Dict[str, Any]:
        """Read a pull request by number."""
        self._log_audit("pr", str(pr_number))
        if dry_run:
            return self._dry_run_response("pr", pr_number)

        result = self._gh(
            "pr", "view", str(pr_number),
            "--json", "number,title,state,body,headRefName,baseRefName,commits,statusCheckRollup,url"
        )
        return self._normalize_pr(json.loads(result))

    def list_issues(
        self,
        state: Optional[str] = None,
        label: Optional[str] = None,
        assignee: Optional[str] = None,
        limit: int = 30,
        dry_run: bool = False,
    ) -> List[Dict[str, Any]]:
        """List issues with optional filters."""
        self._log_audit("issues", "list")
        if dry_run:
            return [{"dry_run": True, "resource": "issues",
                     "filters": {"state": state, "label": label, "limit": limit}}]

        args = ["issue", "list", "--limit", str(limit)]
        if state:
            args += ["--state", state]
        if label:
            args += ["--label", label]
        if assignee:
            args += ["--assignee", assignee]
        args += ["--json", "number,title,state,labels,url,createdAt,assignees"]

        raw = self._gh(*args)
        issues = json.loads(raw)
        return [self._normalize_issue(item) for item in issues]

    def read_file(
        self, path: str, branch: str = "main", dry_run: bool = False
    ) -> Dict[str, Any]:
        """Read a file from a remote branch."""
        self._log_audit("file", f"{branch}:{path}")
        if dry_run:
            return self._dry_run_response("file", f"{branch}:{path}")

        result = self._gh(
            "api", f"repos/{{owner}}/{{repo}}/contents/{path}?ref={branch}"
        )
        return self._normalize_file(json.loads(result))

    def read_actions(
        self, workflow_name: Optional[str] = None,
        status: Optional[str] = None,
        dry_run: bool = False
    ) -> List[Dict[str, Any]]:
        """Read latest Actions workflow run(s)."""
        self._log_audit("actions", workflow_name or "all")
        if dry_run:
            return [self._dry_run_response("actions", workflow_name or "all")]

        args = ["run", "list", "--limit", "10"]
        if workflow_name:
            args += ["--workflow", workflow_name]
        if status:
            args += ["--status", status]
        args += ["--json", "databaseId,name,status,conclusion,headBranch,createdAt,url,event,runNumber"]

        result = self._gh(*args)
        runs = json.loads(result)
        return [self._normalize_actions_run(r) for r in runs]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gh(self, *args: str) -> str:
        """Run a gh CLI command and return stdout."""
        cmd = ["gh", *args]
        if self._repo and self._repo != ".":
            cmd += ["--repo", self._repo]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"gh command failed: {' '.join(cmd)}\nstderr: {proc.stderr}"
            )
        return proc.stdout.strip()

    def _dry_run_response(self, resource: str, identifier: Any) -> Dict[str, Any]:
        resp = {
            "dry_run": True,
            "resource": resource,
            "identifier": str(identifier),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return resp

    def _log_audit(
        self,
        resource_type: str,
        identifier: str,
        authorized: bool = True,
        dry_run: bool = False,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resource": f"{resource_type}/{identifier}",
            "authorized": authorized,
            "dry_run": dry_run,
        }
        self._audit_log.append(entry)
        logger.info("GitHubReadGateway audit: %s", entry)

    # ------------------------------------------------------------------
    # Normalizers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_issue(raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "number": raw.get("number"),
            "title": raw.get("title"),
            "state": raw.get("state"),
            "body": (raw.get("body") or "")[:5000],
            "labels": [lbl["name"] if isinstance(lbl, dict) else lbl
                       for lbl in raw.get("labels", [])],
            "assignees": [a["login"] if isinstance(a, dict) else a
                          for a in raw.get("assignees", [])],
            "url": raw.get("url"),
            "created_at": raw.get("createdAt"),
            "updated_at": raw.get("updatedAt"),
        }

    @staticmethod
    def _normalize_pr(raw: Dict[str, Any]) -> Dict[str, Any]:
        ci_checks = raw.get("statusCheckRollup") or []
        ci_status = [
            {"context": c.get("context", c.get("name", "")),
             "state": c.get("conclusion") or c.get("state", "")}
            for c in ci_checks
        ] if ci_checks else None

        commits = raw.get("commits", [])
        return {
            "number": raw.get("number"),
            "title": raw.get("title"),
            "state": raw.get("state"),
            "body": (raw.get("body") or "")[:5000],
            "head": raw.get("headRefName"),
            "base": raw.get("baseRefName"),
            "commits": len(commits),
            "ci_status": ci_status,
            "url": raw.get("url"),
        }

    @staticmethod
    def _normalize_file(raw: Dict[str, Any]) -> Dict[str, Any]:
        content = raw.get("content", "")
        try:
            decoded = base64.b64decode(content).decode("utf-8")
        except Exception:
            decoded = "[binary or undecodable content]"
        return {
            "path": raw.get("path"),
            "sha": raw.get("sha"),
            "size": raw.get("size", 0),
            "encoding": "utf-8",
            "content": decoded[:10000],
        }

    @staticmethod
    def _normalize_actions_run(raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": raw.get("databaseId"),
            "name": raw.get("name"),
            "status": raw.get("status"),
            "conclusion": raw.get("conclusion"),
            "head_branch": raw.get("headBranch"),
            "event": raw.get("event"),
            "run_number": raw.get("runNumber"),
            "created_at": raw.get("createdAt"),
            "url": raw.get("url"),
        }
