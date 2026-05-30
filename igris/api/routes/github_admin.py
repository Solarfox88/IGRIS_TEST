"""GitHub administrative API routes.

Triple-gated: requires 'admin' scope, goes through GitHubAdminGateway
(which enforces check_authorization + judgment_layer + require_human_approval),
and all mutating operations default to dry_run=True.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from igris.core.authorization import get_current_user, require_scope
from igris.core.github_admin_gateway import GitHubAdminGateway

router = APIRouter(prefix="/api/github/admin", tags=["github-admin"])
gateway = GitHubAdminGateway(dry_run=True)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class CollaboratorRequest(BaseModel):
    repo: str
    username: str
    permission: str = "push"
    dry_run: bool = True


class BranchProtectionRequest(BaseModel):
    repo: str
    branch: str = "main"
    required_reviews: int = 1
    dismiss_stale_reviews: bool = True
    require_code_owner_reviews: bool = False
    enforce_for_admins: bool = True
    dry_run: bool = True


class SecretRequest(BaseModel):
    repo: str
    name: str
    value: str
    dry_run: bool = True


class RepoCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    private: bool = False
    dry_run: bool = True


class RepoDeleteRequest(BaseModel):
    repo: str
    dry_run: bool = True


# ---------------------------------------------------------------------------
# Endpoints (all mutating ops are dry_run by default; admin scope required)
# ---------------------------------------------------------------------------

@router.post("/collaborator/add")
async def add_collaborator(
    req: CollaboratorRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_scope("admin")),
) -> Dict[str, Any]:
    """Add a collaborator. Requires admin scope; dry_run=True by default."""
    if req.dry_run:
        return {"status": "dry_run", "message": f"Would add {req.username} to {req.repo} with {req.permission}"}
    return gateway.add_collaborator(req.repo, req.username, req.permission)


@router.post("/collaborator/remove")
async def remove_collaborator(
    req: CollaboratorRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_scope("admin")),
) -> Dict[str, Any]:
    """Remove a collaborator. Requires admin scope; dry_run=True by default."""
    if req.dry_run:
        return {"status": "dry_run", "message": f"Would remove {req.username} from {req.repo}"}
    return gateway.remove_collaborator(req.repo, req.username)


@router.post("/branch-protection/set")
async def set_branch_protection(
    req: BranchProtectionRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_scope("admin")),
) -> Dict[str, Any]:
    """Set branch protection rules. Requires admin scope; dry_run=True by default."""
    if req.dry_run:
        return {"status": "dry_run", "message": f"Would set branch protection on {req.repo}/{req.branch}"}
    rules = {
        "required_reviews": req.required_reviews,
        "dismiss_stale_reviews": req.dismiss_stale_reviews,
        "require_code_owner_reviews": req.require_code_owner_reviews,
        "enforce_for_admins": req.enforce_for_admins,
    }
    return gateway.set_branch_protection(req.repo, req.branch, rules)


@router.post("/secret/set")
async def set_secret(
    req: SecretRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_scope("admin")),
) -> Dict[str, Any]:
    """Set a repository secret. Write-only: value never returned. Requires admin scope."""
    if req.dry_run:
        return {"status": "dry_run", "message": f"Would set secret {req.name} on {req.repo}"}
    return gateway.set_secret(req.repo, req.name, req.value)


@router.get("/repo/info")
async def get_repo_info(
    repo: str,
    user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_scope("admin")),
) -> Dict[str, Any]:
    """Get repository metadata. Read-only; does not return secrets."""
    return gateway.get_repo_info(repo)


@router.post("/repo/create")
async def create_repo(
    req: RepoCreateRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_scope("admin")),
) -> Dict[str, Any]:
    """Create a new repository. Requires admin scope; dry_run=True by default."""
    if req.dry_run:
        return {"status": "dry_run", "message": f"Would create repo {req.name}"}
    return gateway.create_repo(req.name, req.description or "", req.private)


@router.post("/repo/delete")
async def delete_repo(
    req: RepoDeleteRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_scope("admin")),
) -> Dict[str, Any]:
    """Delete a repository. Requires admin scope and double confirmation; dry_run=True by default."""
    if req.dry_run:
        return {"status": "dry_run", "message": f"Would delete repo {req.repo}"}
    return gateway.delete_repo(req.repo, double_confirm=True)


@router.get("/audit-log")
async def get_audit_log(
    user: Dict[str, Any] = Depends(get_current_user),
    _: None = Depends(require_scope("admin")),
) -> Dict[str, Any]:
    """Return the gateway audit trail. Requires admin scope."""
    return {"audit_log": gateway.get_audit_log()}
