"""
Generic report models.

These models are used to serialise data returned by various API endpoints,
such as test results, git status or routing explanations.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


class GitStatusResponse(BaseModel):
    branch: Optional[str]
    remote: Optional[str]
    dirty: bool
    changed: List[str]
    head: Optional[str]


class TestRunResponse(BaseModel):
    success: bool
    stdout: str
    stderr: str


class StatusResponse(BaseModel):
    provider: str
    model: str
    cost: Optional[float] = None
    safe: bool