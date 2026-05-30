import logging
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class GitHubAdminGateway:
    """
    Triple-gated gateway for GitHub administrative operations.

    All operations require:
    1. AuthorizationGate: scope 'admin' required
    2. JudgmentLayer: operation must be approved by risk assessment
    3. HumanApproval: explicit out-of-band confirmation

    Every attempt (successful or denied) is logged.
    Dry-run is the default mode.
    """

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.audit_log: list[dict] = []

    def _log(self, action: str, target: str, status: str, details: dict = None):
        """Record an audit entry for every attempt."""
        entry = {
            "id": str(uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "target": target,
            "status": status,
            "details": details or {},
            "dry_run": self.dry_run,
        }
        self.audit_log.append(entry)
        logger.info(f"AUDIT: {entry}")
        return entry

    # ------------------------------------------------------------------
    # Authorization & Approval gates (stubs for integration)
    # ------------------------------------------------------------------
    def check_authorization(self, scope: str) -> bool:
        """Verify the caller holds the required scope."""
        # Placeholder: real implementation will check JWT / token
        return scope == "admin"

    def judgment_layer(self, action: str, target: str) -> bool:
        """Risk assessment: allow or deny."""
        # Placeholder: integrate with risk engine
        return True

    def require_human_approval(self, ticket_id: str) -> bool:
        """Block until human operator approves out-of-band."""
        # Placeholder: actual implementation sends Slack/email and waits
        return True

    # ------------------------------------------------------------------
    # Operations – all require triple gate
    # ------------------------------------------------------------------
    def _triple_gate(self, action: str, target: str) -> bool:
        """Run authorization, judgment, and approval."""
        if not self.check_authorization("admin"):
            self._log(action, target, "DENIED_AUTH")
            return False
        if not self.judgment_layer(action, target):
            self._log(action, target, "DENIED_JUDGMENT")
            return False
        if not self.require_human_approval(f"admin-{action}-{target}"):
            self._log(action, target, "DENIED_APPROVAL")
            return False
        return True

    def add_collaborator(self, repo: str, username: str, permission: str = "push") -> dict:
        """Add a collaborator to a repository."""
        action = "collaborator.add"
        if not self._triple_gate(action, repo):
            return {"success": False, "reason": "Gate denied"}
        if self.dry_run:
            self._log(action, repo, "DRY_RUN", {"username": username, "permission": permission})
            return {"success": True, "dry_run": True, "changes": {"add": {"username": username, "permission": permission}}}
        # TODO: actual GitHub API call
        self._log(action, repo, "EXECUTED", {"username": username, "permission": permission})
        return {"success": True, "dry_run": False, "result": {"username": username, "permission": permission}}

    def remove_collaborator(self, repo: str, username: str) -> dict:
        """Remove a collaborator from a repository."""
        action = "collaborator.remove"
        if not self._triple_gate(action, repo):
            return {"success": False, "reason": "Gate denied"}
        if self.dry_run:
            self._log(action, repo, "DRY_RUN", {"username": username})
            return {"success": True, "dry_run": True, "changes": {"remove": {"username": username}}}
        self._log(action, repo, "EXECUTED", {"username": username})
        return {"success": True, "dry_run": False, "result": {"username": username}}

    def set_branch_protection(self, repo: str, branch: str, rules: dict) -> dict:
        """Configure branch protection rules."""
        action = "branch-protection.set"
        if not self._triple_gate(action, repo):
            return {"success": False, "reason": "Gate denied"}
        if self.dry_run:
            self._log(action, repo, "DRY_RUN", {"branch": branch, "rules": rules})
            return {"success": True, "dry_run": True, "changes": {"branch": branch, "rules": rules}}
        self._log(action, repo, "EXECUTED", {"branch": branch, "rules": rules})
        return {"success": True, "dry_run": False, "result": {"branch": branch, "rules": rules}}

    def set_secret(self, repo: str, secret_name: str, secret_value: str) -> dict:
        """Set a repository secret. Write-only: never returns the value."""
        action = "secret.set"
        if not self._triple_gate(action, repo):
            return {"success": False, "reason": "Gate denied"}
        # Hashing for audit trail – never store plaintext
        import hashlib
        secret_hash = hashlib.sha256(secret_value.encode()).hexdigest()
        if self.dry_run:
            self._log(action, repo, "DRY_RUN", {"secret_name": secret_name, "secret_hash": secret_hash})
            return {"success": True, "dry_run": True, "changes": {"set": {"secret_name": secret_name}}}
        self._log(action, repo, "EXECUTED", {"secret_name": secret_name, "secret_hash": secret_hash})
        return {"success": True, "dry_run": False, "result": {"secret_name": secret_name}}

    def get_repo_info(self, repo: str) -> dict:
        """Read repository metadata (no secrets)."""
        action = "repo.info"
        if not self._triple_gate(action, repo):
            return {"success": False, "reason": "Gate denied"}
        if self.dry_run:
            self._log(action, repo, "DRY_RUN")
            return {"success": True, "dry_run": True, "info": {"repo": repo, "description": "(dry run)"}}
        # TODO: actual GitHub API call
        self._log(action, repo, "EXECUTED")
        return {"success": True, "dry_run": False, "info": {"repo": repo, "description": "sample"}}

    def create_repo(self, name: str, description: str = "", private: bool = True) -> dict:
        """Create a new GitHub repository."""
        action = "repo.create"
        if not self._triple_gate(action, name):
            return {"success": False, "reason": "Gate denied"}
        if self.dry_run:
            self._log(action, name, "DRY_RUN", {"description": description, "private": private})
            return {"success": True, "dry_run": True, "changes": {"create": {"name": name, "private": private}}}
        self._log(action, name, "EXECUTED", {"private": private})
        return {"success": True, "dry_run": False, "result": {"name": name, "private": private}}

    def delete_repo(self, repo: str, double_confirm: bool = False) -> dict:
        """Delete a repository. Requires double confirmation."""
        action = "repo.delete"
        if not double_confirm:
            self._log(action, repo, "DENIED_DOUBLE_CONFIRM")
            return {"success": False, "reason": "Double confirmation required", "double_confirm_required": True}
        if not self._triple_gate(action, repo):
            return {"success": False, "reason": "Gate denied"}
        if self.dry_run:
            self._log(action, repo, "DRY_RUN")
            return {"success": True, "dry_run": True, "changes": {"delete": {"repo": repo}}}
        self._log(action, repo, "EXECUTED")
        return {"success": True, "dry_run": False, "result": {"deleted": repo}}

    def get_audit_log(self) -> list:
        """Return all audit log entries."""
        return self.audit_log.copy()
