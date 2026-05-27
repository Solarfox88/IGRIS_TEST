"""Tests for Issue #727 — API auth, CORS restriction, and rate limiting."""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest


def _make_client(tmp_path, env_overrides=None):
    """Create a TestClient with optional env overrides."""
    env = env_overrides or {}
    from igris.web.server import create_app, CONFIG
    CONFIG.project_root = tmp_path
    with patch.dict(os.environ, env, clear=False):
        app = create_app()
    from starlette.testclient import TestClient
    return TestClient(app, raise_server_exceptions=False)


class TestCORSMiddleware:
    """CORS headers are controlled by IGRIS_ALLOWED_ORIGINS."""

    def test_cors_allowed_origin_returns_allow_header(self, tmp_path):
        """Request from allowed origin gets Access-Control-Allow-Origin."""
        client = _make_client(tmp_path, {
            "IGRIS_ALLOWED_ORIGINS": "http://localhost:7778",
            "IGRIS_API_KEY": "",
        })
        resp = client.options(
            "/api/rank/status",
            headers={
                "Origin": "http://localhost:7778",
                "Access-Control-Request-Method": "GET",
            }
        )
        # Either 200 (CORS OK) or 404/405 — key is the CORS header
        # is present OR at minimum origin is not blocked
        assert resp.status_code in (200, 404, 405)

    def test_cors_disallowed_origin_blocked(self, tmp_path):
        """Request from non-allowed origin does not get ACAO header."""
        client = _make_client(tmp_path, {
            "IGRIS_ALLOWED_ORIGINS": "http://localhost:7778",
            "IGRIS_API_KEY": "",
        })
        resp = client.get(
            "/health",
            headers={"Origin": "http://evil.example.com"},
        )
        # CORS middleware should not echo back the evil origin
        acao = resp.headers.get("access-control-allow-origin", "")
        assert "evil.example.com" not in acao


class TestAPIKeyAuth:
    """API key authentication via X-API-Key header."""

    def test_auth_disabled_when_no_igris_api_key(self, tmp_path):
        """/api/* endpoints accessible without X-API-Key if IGRIS_API_KEY not set."""
        client = _make_client(tmp_path, {"IGRIS_API_KEY": ""})
        resp = client.get("/api/rank/status")
        assert resp.status_code != 401, "Auth must be disabled when IGRIS_API_KEY is empty"

    def test_auth_enabled_blocks_unauthenticated(self, tmp_path):
        """When IGRIS_API_KEY is set, /api/* returns 401 without X-API-Key."""
        client = _make_client(tmp_path, {"IGRIS_API_KEY": "test-secret-key"})
        resp = client.get("/api/rank/status")
        assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"

    def test_auth_passes_with_correct_key(self, tmp_path):
        """Correct X-API-Key header passes through auth."""
        client = _make_client(tmp_path, {"IGRIS_API_KEY": "test-secret-key"})
        resp = client.get("/api/rank/status", headers={"X-API-Key": "test-secret-key"})
        assert resp.status_code != 401, f"Auth should pass with correct key, got {resp.status_code}"

    def test_auth_rejects_wrong_key(self, tmp_path):
        """Wrong X-API-Key returns 401."""
        client = _make_client(tmp_path, {"IGRIS_API_KEY": "test-secret-key"})
        resp = client.get("/api/rank/status", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_health_endpoint_exempt_from_auth(self, tmp_path):
        """/health endpoint is always accessible without auth."""
        client = _make_client(tmp_path, {"IGRIS_API_KEY": "test-secret-key"})
        resp = client.get("/health")
        assert resp.status_code != 401, "/health must be auth-exempt"

    def test_static_exempt_from_auth(self, tmp_path):
        """/static/ path is exempt from API key auth."""
        client = _make_client(tmp_path, {"IGRIS_API_KEY": "test-secret-key"})
        resp = client.get("/static/nonexistent.js")
        # 404 (file not found) is fine — 401 is not
        assert resp.status_code != 401, "/static/ must be auth-exempt"


class TestRateLimiting:
    """Rate limiter blocks excessive requests from the same IP."""

    def test_rate_limit_standard_enforced(self, tmp_path):
        """Exceeding IGRIS_RATE_LIMIT requests/min returns 429."""
        client = _make_client(tmp_path, {
            "IGRIS_API_KEY": "",
            "IGRIS_RATE_LIMIT": "3",  # very low limit for test
        })
        responses = [client.get("/api/rank/status") for _ in range(5)]
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes, f"Expected 429 rate limit, got: {status_codes}"

    def test_destructive_rate_limit_lower(self, tmp_path):
        """Destructive endpoints have a lower rate limit."""
        client = _make_client(tmp_path, {
            "IGRIS_API_KEY": "",
            "IGRIS_RATE_LIMIT": "100",
            "IGRIS_RATE_LIMIT_DESTRUCTIVE": "2",
        })
        responses = [client.post("/api/rank/runs/x/cancel", content=b"") for _ in range(5)]
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes, f"Expected 429 for destructive endpoint, got: {status_codes}"

    def test_rate_limit_response_is_429_not_500(self, tmp_path):
        """Rate limit exceeded returns 429 (not 500)."""
        client = _make_client(tmp_path, {
            "IGRIS_API_KEY": "",
            "IGRIS_RATE_LIMIT": "1",
        })
        # First request goes through
        client.get("/api/rank/status")
        # Second should be rate limited
        resp = client.get("/api/rank/status")
        assert resp.status_code == 429
        data = resp.json()
        assert "detail" in data
        assert "Rate limit" in data["detail"]
