"""
FastAPI dependency helpers for IGRIS authorization.

Wraps the existing AuthorizationGate / identity-resolver layer into
FastAPI-compatible dependency functions (get_current_user, require_scope).

Design:
- In production, a real auth token should be validated here.
- In dev/supervised mode (no Authorization header), a read-only anonymous
  profile is returned so that dry-run endpoints still respond.
- Scope enforcement is done by require_scope(); admin-only endpoints
  will return 403 when the caller does not hold the required scope.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import Depends, HTTPException, Request


# ---------------------------------------------------------------------------
# Current-user dependency
# ---------------------------------------------------------------------------

async def get_current_user(request: Request) -> Dict[str, Any]:
    """Extract caller identity from the request.

    Currently returns a simple dict; replace with JWT/session validation
    for production use.  In supervised/dry-run contexts the user is always
    treated as an internal operator with read-only trust.
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        # TODO: validate token against identity store
        return {"user_id": "token_user", "scopes": ["read"], "trust_level": "operator"}

    # No token — anonymous read-only context (dev / dry-run)
    return {"user_id": "anonymous", "scopes": ["read"], "trust_level": "readonly"}


# ---------------------------------------------------------------------------
# Scope-gate dependency factory
# ---------------------------------------------------------------------------

def require_scope(scope: str):
    """Return a FastAPI dependency that enforces a required scope.

    Raises HTTP 403 if the caller does not hold *scope* in their profile.
    Usage::

        @router.post("/admin/action")
        async def action(user = Depends(get_current_user),
                         _ = Depends(require_scope("admin"))):
            ...
    """
    async def _checker(user: Dict[str, Any] = Depends(get_current_user)) -> None:
        user_scopes = user.get("scopes", [])
        if scope not in user_scopes and "admin" not in user_scopes:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient scope. Required: '{scope}'.",
            )
    return _checker
