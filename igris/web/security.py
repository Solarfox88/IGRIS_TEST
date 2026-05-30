"""Issue #727 — IGRIS API security: CORS restriction, API-key auth, rate limiting.

Import and apply via ``apply_security_middleware(app)`` in ``create_app()``.

Environment variables
---------------------
IGRIS_API_KEY               If set, all /api/* requests must include
                            ``X-API-Key: <value>`` header.  Empty string
                            disables auth entirely (default: disabled).
IGRIS_ALLOWED_ORIGINS       Comma-separated list of allowed CORS origins.
                            Default: ``http://localhost:7778,http://127.0.0.1:7778``
IGRIS_RATE_LIMIT            Max requests/min per IP for standard endpoints.
                            Default: 60.
IGRIS_RATE_LIMIT_DESTRUCTIVE
                            Max requests/min per IP for destructive endpoints
                            (rank runs, vast, integration).  Default: 10.
"""

from __future__ import annotations

import collections
import os
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# Endpoints that bypass API-key auth entirely.
_AUTH_EXEMPT_PATHS_EXACT: frozenset = frozenset({"/", "/health"})
_AUTH_EXEMPT_PREFIXES: tuple = ("/docs", "/openapi.json", "/redoc", "/static/")

# Path prefixes that count as "destructive" and use the lower rate limit.
_DESTRUCTIVE_PATH_PREFIXES: tuple = (
    "/api/rank/runs",
    "/api/vast/",
    "/api/integration/run-mission",
)


def apply_security_middleware(app: FastAPI) -> None:
    """Attach CORS, API-key auth, and rate-limiting middleware to *app*.

    Safe to call multiple times (middleware is applied once per call, so
    only call this once from ``create_app()``).
    """
    _api_key: str = os.getenv("IGRIS_API_KEY", "")
    _auth_enabled: bool = bool(_api_key)
    _allowed_origins: list = [
        o.strip()
        for o in os.getenv(
            "IGRIS_ALLOWED_ORIGINS",
            "http://localhost:7778,http://127.0.0.1:7778",
        ).split(",")
        if o.strip()
    ]
    _rate_standard: int = max(1, int(os.getenv("IGRIS_RATE_LIMIT", "60")))
    _rate_destructive: int = max(1, int(os.getenv("IGRIS_RATE_LIMIT_DESTRUCTIVE", "10")))

    # 1 — CORS (must be added before the custom middleware so it runs first)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
    )

    # 2 — Rate limiting + API-key auth in a single middleware pass
    _rate_buckets: Any = collections.defaultdict(list)

    @app.middleware("http")
    async def _igris_security(request: Request, call_next):  # type: ignore[misc]
        path: str = request.url.path
        client_ip: str = (
            (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
            or (request.client.host if request.client else "unknown")
        )

        # ---- Rate limiting ----
        now = time.time()
        _rate_buckets[client_ip] = [t for t in _rate_buckets[client_ip] if now - t < 60]
        # GET requests to /api/rank/runs/{id} are read-only status checks — do not
        # classify them as destructive even though the path prefix matches.
        _method = request.method.upper()
        is_destructive = (
            any(path.startswith(p) for p in _DESTRUCTIVE_PATH_PREFIXES)
            and _method not in ("GET", "HEAD", "OPTIONS")
        )
        limit = _rate_destructive if is_destructive else _rate_standard
        if len(_rate_buckets[client_ip]) >= limit:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded: max {limit} req/min "
                        f"({'destructive' if is_destructive else 'standard'} endpoint)."
                    )
                },
            )
        _rate_buckets[client_ip].append(now)

        # ---- API-key authentication ----
        if _auth_enabled:
            exempt = (
                path in _AUTH_EXEMPT_PATHS_EXACT
                or any(path.startswith(pfx) for pfx in _AUTH_EXEMPT_PREFIXES)
            )
            if not exempt:
                provided = request.headers.get("X-API-Key", "")
                if provided != _api_key:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Unauthorized: missing or invalid X-API-Key header."},
                    )

        return await call_next(request)
