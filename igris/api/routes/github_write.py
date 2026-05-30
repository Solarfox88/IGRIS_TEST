"""
GitHub Write Gateway API Routes
Endpoints for gated GitHub write operations: comment, label, issue management, PR merge, actions trigger.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from igris.core.github_write_gateway import GitHubWriteGateway, GitHubWriteResult
from igris.core.authorization_gate import AuthorizationGate
from igris.core.judgment_layer import JudgmentLayer

router = APIRouter(prefix="/api/github/write", tags=["github-write"])

logger = logging.getLogger(__name__)


# --- Request/Response models ---

class CommentRequest(BaseModel):
    repo: str
    issue_number: int
    body: str
    dry_run: bool = True
    mission_id: Optional[str] = None
    run_id: Optional[str] = None

class CommentResponse(BaseModel):
    status: str
    message: str
    action: str = "comment"
    dry_run: bool
    judgment_advisory: Optional[str] = None

class LabelRequest(BaseModel):
    repo: str
    issue_number: int
    labels: List[str]
    action: str = "add"  # "add" or "remove"
    dry_run: bool = True
    mission_id: Optional[str] = None
    run_id: Optional[str] = None

class LabelResponse(BaseModel):
    status: str
    message: str
    action: str = "label"
    dry_run: bool
    judgment_advisory: Optional[str] = None

class IssueCloseRequest(BaseModel):
    repo: str
    issue_number: int
    comment: Optional[str] = None
    dry_run: bool = True
    mission_id: Optional[str] = None
    run_id: Optional[str] = None

class IssueCloseResponse(BaseModel):
    status: str
    message: str
    action: str = "issue/close"
    dry_run: bool
    judgment_advisory: Optional[str] = None

class IssueCreateRequest(BaseModel):
    repo: str
    title: str
    body: str
    labels: List[str] = Field(default_factory=list)
    assignees: List[str] = Field(default_factory=list)
    dry_run: bool = True
    mission_id: Optional[str] = None
    run_id: Optional[str] = None

class IssueCreateResponse(BaseModel):
    status: str
    message: str
    action: str = "issue/create"
    dry_run: bool
    issue_url: Optional[str] = None
    judgment_advisory: Optional[str] = None

class PrMergeRequest(BaseModel):
    repo: str
    pr_number: int
    method: str = "merge"  # "merge", "squash", "rebase"
    dry_run: bool = True
    require_explicit_approval: bool = True
    mission_id: Optional[str] = None
    run_id: Optional[str] = None

class PrMergeResponse(BaseModel):
    status: str
    message: str
    action: str = "pr/merge"
    dry_run: bool
    judgment_advisory: Optional[str] = None

class ActionTriggerRequest(BaseModel):
    repo: str
    workflow_id: str
    ref: str = "main"
    inputs: Optional[Dict[str, Any]] = None
    dry_run: bool = True
    mission_id: Optional[str] = None
    run_id: Optional[str] = None

class ActionTriggerResponse(BaseModel):
    status: str
    message: str
    action: str = "actions/trigger"
    dry_run: bool
    judgment_advisory: Optional[str] = None


def _make_url(repo: str, resource: str, number: int) -> str:
    """Build a GitHub URL string for use with gh CLI."""
    return f"https://github.com/{repo}/{resource}/{number}"


def _get_gateway(dry_run: bool = True) -> GitHubWriteGateway:
    auth_gate = AuthorizationGate()
    judgment = JudgmentLayer()
    return GitHubWriteGateway(auth_gate=auth_gate, judgment_layer=judgment, dry_run=dry_run)


def _result_to_response(result: GitHubWriteResult) -> dict:
    advisory = None
    if result.judgment and hasattr(result.judgment, "risk_level"):
        advisory = f"risk={result.judgment.risk_level}"
    return {
        "status": "ok" if result.success else "error",
        "message": result.output or result.error or "",
        "dry_run": result.dry_run,
        "judgment_advisory": advisory,
    }


@router.post("/comment", response_model=CommentResponse)
async def add_comment(request: CommentRequest):
    """Add a comment to an issue or PR (gated, dry-run by default)."""
    try:
        gw = _get_gateway(dry_run=request.dry_run)
        issue_url = _make_url(request.repo, "issues", request.issue_number)
        result = gw.comment(issue_url=issue_url, body=request.body, context={
            "mission_id": request.mission_id, "run_id": request.run_id,
        })
        return CommentResponse(**_result_to_response(result))
    except Exception as e:
        logger.exception("Failed to add comment")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/label", response_model=LabelResponse)
async def manage_label(request: LabelRequest):
    """Add or remove labels on an issue/PR (gated, dry-run by default)."""
    try:
        gw = _get_gateway(dry_run=request.dry_run)
        issue_url = _make_url(request.repo, "issues", request.issue_number)
        if request.action == "add":
            result = gw.add_label(issue_url=issue_url, labels=request.labels, context={
                "mission_id": request.mission_id, "run_id": request.run_id,
            })
        else:
            result = gw.remove_label(issue_url=issue_url, labels=request.labels, context={
                "mission_id": request.mission_id, "run_id": request.run_id,
            })
        return LabelResponse(**_result_to_response(result))
    except Exception as e:
        logger.exception("Failed to manage labels")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/issue/close", response_model=IssueCloseResponse)
async def close_issue(request: IssueCloseRequest):
    """Close an issue with optional comment (gated, dry-run by default)."""
    try:
        gw = _get_gateway(dry_run=request.dry_run)
        issue_url = _make_url(request.repo, "issues", request.issue_number)
        result = gw.close_issue(issue_url=issue_url, comment=request.comment or "", context={
            "mission_id": request.mission_id, "run_id": request.run_id,
        })
        return IssueCloseResponse(**_result_to_response(result))
    except Exception as e:
        logger.exception("Failed to close issue")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/issue/create", response_model=IssueCreateResponse)
async def create_issue(request: IssueCreateRequest):
    """Open a new issue (gated, dry-run by default)."""
    try:
        gw = _get_gateway(dry_run=request.dry_run)
        result = gw.create_issue(
            title=request.title,
            body=request.body,
            labels=request.labels,
            assignees=request.assignees,
            context={"mission_id": request.mission_id, "run_id": request.run_id},
        )
        r = _result_to_response(result)
        r["issue_url"] = result.output if result.success and not result.dry_run else None
        return IssueCreateResponse(**r)
    except Exception as e:
        logger.exception("Failed to create issue")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pr/merge", response_model=PrMergeResponse)
async def merge_pr(request: PrMergeRequest):
    """Merge a pull request (requires explicit approval; destructive)."""
    try:
        if not request.require_explicit_approval:
            raise HTTPException(status_code=400, detail="Merge requires require_explicit_approval=true")
        gw = _get_gateway(dry_run=request.dry_run)
        pr_url = _make_url(request.repo, "pull", request.pr_number)
        result = gw.merge_pr(pr_url=pr_url, method=request.method, context={
            "mission_id": request.mission_id, "run_id": request.run_id,
        })
        return PrMergeResponse(**_result_to_response(result))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to merge PR")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/actions/trigger", response_model=ActionTriggerResponse)
async def trigger_workflow(request: ActionTriggerRequest):
    """Trigger a GitHub Actions workflow (gated, dry-run by default)."""
    try:
        gw = _get_gateway(dry_run=request.dry_run)
        result = gw.trigger_action(
            workflow=request.workflow_id,
            ref=request.ref,
            inputs=request.inputs,
            context={"mission_id": request.mission_id, "run_id": request.run_id},
        )
        return ActionTriggerResponse(**_result_to_response(result))
    except Exception as e:
        logger.exception("Failed to trigger workflow")
        raise HTTPException(status_code=500, detail=str(e))
