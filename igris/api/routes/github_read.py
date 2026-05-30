"""
GitHub Read Gateway API Routes
Endpoints for gated GitHub read operations: issue, PR, issues list, file, actions.
All reads are scope-checked via AuthorizationGate and logged for audit.
"""

import os
from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
from pydantic import BaseModel

from igris.core.github_read_gateway import GitHubReadGateway
from igris.core.authorization_gate import AuthorizationGate

router = APIRouter(prefix="/api/github/read", tags=["github-read"])

# Project root for AuthorizationGate
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_gateway() -> GitHubReadGateway:
    """Create a GitHubReadGateway with an AuthorizationGate."""
    gate = AuthorizationGate(_PROJECT_ROOT)
    return GitHubReadGateway(gate)


class IssueResponse(BaseModel):
    number: Optional[int]
    title: Optional[str]
    body: Optional[str]
    state: Optional[str]
    labels: List[str] = []
    assignees: List[str] = []
    url: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]


class PRResponse(BaseModel):
    number: Optional[int]
    title: Optional[str]
    body: Optional[str]
    state: Optional[str]
    head: Optional[str]
    base: Optional[str]
    commits: Optional[int]
    ci_status: Optional[list]
    url: Optional[str]


class FileResponse(BaseModel):
    path: Optional[str]
    content: str
    sha: Optional[str]
    size: Optional[int]
    encoding: str = "utf-8"


class ActionRunResponse(BaseModel):
    id: Optional[int]
    name: Optional[str]
    status: Optional[str]
    conclusion: Optional[str]
    head_branch: Optional[str]
    event: Optional[str]
    run_number: Optional[int]
    created_at: Optional[str]
    url: Optional[str]


@router.get("/issue/{issue_number}", response_model=IssueResponse)
async def get_issue(
    issue_number: int,
    dry_run: bool = Query(False, description="Simulate read without actual access"),
):
    """Read a single GitHub issue by number."""
    try:
        gateway = _get_gateway()
        issue = gateway.read_issue(issue_number, dry_run=dry_run)
        if not issue:
            raise HTTPException(status_code=404, detail="Issue not found")
        return issue
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"GitHub CLI error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal gateway error: {type(e).__name__}")


@router.get("/pr/{pr_number}", response_model=PRResponse)
async def get_pr(
    pr_number: int,
    dry_run: bool = Query(False, description="Simulate read without actual access"),
):
    """Read a single GitHub pull request by number."""
    try:
        gateway = _get_gateway()
        pr = gateway.read_pr(pr_number, dry_run=dry_run)
        if not pr:
            raise HTTPException(status_code=404, detail="PR not found")
        return pr
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"GitHub CLI error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal gateway error: {type(e).__name__}")


@router.get("/issues", response_model=List[IssueResponse])
async def list_issues(
    state: str = Query("open", description="Filter by state: open, closed, all"),
    labels: Optional[str] = Query(None, description="Comma-separated labels"),
    assignee: Optional[str] = Query(None, description="Filter by assignee"),
    limit: int = Query(30, description="Max results"),
    dry_run: bool = Query(False, description="Simulate read without actual access"),
):
    """List GitHub issues with optional filters."""
    try:
        gateway = _get_gateway()
        issues = gateway.list_issues(
            state=state,
            label=labels,
            assignee=assignee,
            limit=limit,
            dry_run=dry_run,
        )
        return issues
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"GitHub CLI error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal gateway error: {type(e).__name__}")


@router.get("/file", response_model=FileResponse)
async def get_file(
    path: str = Query(..., description="File path within the repository"),
    branch: Optional[str] = Query("main", description="Branch name"),
    dry_run: bool = Query(False, description="Simulate read without actual access"),
):
    """Read a file from a remote branch."""
    try:
        gateway = _get_gateway()
        file_data = gateway.read_file(path, branch=branch or "main", dry_run=dry_run)
        if not file_data:
            raise HTTPException(status_code=404, detail="File not found")
        return file_data
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"GitHub CLI error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal gateway error: {type(e).__name__}")


@router.get("/actions", response_model=List[ActionRunResponse])
async def get_actions(
    workflow: Optional[str] = Query(None, description="Filter by workflow name"),
    status: Optional[str] = Query(None, description="Filter by status: completed, in_progress, queued"),
    dry_run: bool = Query(False, description="Simulate read without actual access"),
):
    """Get GitHub Actions workflow runs with optional filters."""
    try:
        gateway = _get_gateway()
        runs = gateway.read_actions(workflow_name=workflow, status=status, dry_run=dry_run)
        return runs
    except HTTPException:
        raise
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"GitHub CLI error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal gateway error: {type(e).__name__}")
