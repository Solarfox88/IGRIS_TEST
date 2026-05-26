"""Tests for Vast.ai real API integration — search, provision, destroy.

Verifies:
- _vastai_request HTTP helper (success, HTTP error, network error)
- search_offers: real filtering by VRAM/cost, sort, max 10
- provision: selects cheapest offer, real instance_id from API response
- provision: uses specified offer_id when provided
- destroy: calls DELETE on correct instance_id, handles API error
- No API key in any response (extended coverage)
- All safety gates still enforced with real API path
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import urllib.error
import json

import pytest

from igris.layers.advisory import vastai_manager as _mod
from igris.layers.advisory.vastai_manager import (
    APPROVAL_TOKEN,
    VastAIManager,
    _vastai_request,
)
from igris.models.config import CONFIG


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manager():
    return VastAIManager()


@pytest.fixture
def mock_api(monkeypatch):
    """Patch _vastai_request for offline tests."""
    calls = []

    def _fake(method, path, api_key, payload=None, timeout=20):
        calls.append({"method": method, "path": path, "payload": payload})
        if method == "GET" and "/bundles" in path:
            return {
                "offers": [
                    # cheapest, enough VRAM
                    {"id": 1001, "gpu_name": "Tesla V100", "gpu_ram": 32768,
                     "num_gpus": 1, "dph_total": 0.021, "cuda_max_good": 12.2,
                     "disk_space": 50, "reliability2": 0.95, "rentable": True,
                     "geolocation": "US"},
                    # second cheapest
                    {"id": 1002, "gpu_name": "RTX 3090", "gpu_ram": 24576,
                     "num_gpus": 1, "dph_total": 0.055, "cuda_max_good": 12.1,
                     "disk_space": 50, "reliability2": 0.92, "rentable": True,
                     "geolocation": "IT"},
                    # over budget (>0.50)
                    {"id": 1003, "gpu_name": "A100", "gpu_ram": 81920,
                     "num_gpus": 1, "dph_total": 1.20, "cuda_max_good": 12.4,
                     "disk_space": 100, "reliability2": 0.99, "rentable": True,
                     "geolocation": "US"},
                    # insufficient VRAM for deepseek-r1:32b (needs 24GB)
                    {"id": 1004, "gpu_name": "RTX 3060", "gpu_ram": 12288,
                     "num_gpus": 1, "dph_total": 0.05, "cuda_max_good": 12.0,
                     "disk_space": 50, "reliability2": 0.90, "rentable": True,
                     "geolocation": "EU"},
                ]
            }
        if method == "PUT" and "/asks/" in path:
            return {"id": 88001}
        if method == "DELETE" and "/instances/" in path:
            return {"success": True}
        return {}

    monkeypatch.setattr(_mod, "_vastai_request", _fake)
    return calls


# ---------------------------------------------------------------------------
# _vastai_request HTTP helper
# ---------------------------------------------------------------------------

class TestVastaiRequestHelper:

    def test_success_parses_json(self, monkeypatch):
        """Good response → parsed dict returned."""
        fake_body = json.dumps({"ok": True}).encode()
        fake_resp = MagicMock()
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = MagicMock(return_value=False)
        fake_resp.read.return_value = fake_body

        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = _vastai_request("GET", "/test/", "key123")
        assert result == {"ok": True}

    def test_http_error_raises_runtime(self, monkeypatch):
        """HTTP error → RuntimeError with code."""
        err = urllib.error.HTTPError(
            url="https://x", code=401, msg="Unauthorized",
            hdrs={}, fp=None,
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(RuntimeError, match="401"):
                _vastai_request("GET", "/test/", "bad-key")

    def test_network_error_raises_runtime(self, monkeypatch):
        """Network failure → RuntimeError."""
        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            with pytest.raises(RuntimeError, match="timeout"):
                _vastai_request("GET", "/test/", "key")

    def test_authorization_header_set(self, monkeypatch):
        """Bearer token is always in request headers."""
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["auth"] = req.get_header("Authorization")
            fake_resp = MagicMock()
            fake_resp.__enter__ = lambda s: s
            fake_resp.__exit__ = MagicMock(return_value=False)
            fake_resp.read.return_value = b"{}"
            return fake_resp

        with patch("urllib.request.urlopen", fake_urlopen):
            _vastai_request("GET", "/test/", "mykey42")
        assert captured["auth"] == "Bearer mykey42"

    def test_delete_no_payload(self, monkeypatch):
        """DELETE with no payload sends no body."""
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["data"] = req.data
            fake_resp = MagicMock()
            fake_resp.__enter__ = lambda s: s
            fake_resp.__exit__ = MagicMock(return_value=False)
            fake_resp.read.return_value = b"{}"
            return fake_resp

        with patch("urllib.request.urlopen", fake_urlopen):
            _vastai_request("DELETE", "/instances/999/", "key")
        assert captured["data"] is None


# ---------------------------------------------------------------------------
# search_offers — real API filtering
# ---------------------------------------------------------------------------

class TestSearchOffersRealApi:

    def test_filters_by_vram(self, manager, mock_api):
        """Offers with insufficient VRAM are excluded."""
        with patch.object(CONFIG.vastai, "api_key", "test-key"):
            result = manager.search_offers(model="deepseek-r1:32b", max_cost=0.50)
        ids = [o["id"] for o in result.offers]
        assert 1004 not in ids  # RTX 3060 — 12GB, not enough for 32b

    def test_filters_by_cost(self, manager, mock_api):
        """Offers over budget are excluded."""
        with patch.object(CONFIG.vastai, "api_key", "test-key"):
            result = manager.search_offers(model="deepseek-r1:32b", max_cost=0.50)
        ids = [o["id"] for o in result.offers]
        assert 1003 not in ids  # A100 at $1.20/h — over budget

    def test_sorted_by_price_ascending(self, manager, mock_api):
        """Results are sorted cheapest first."""
        with patch.object(CONFIG.vastai, "api_key", "test-key"):
            result = manager.search_offers(model="deepseek-r1:32b", max_cost=0.50)
        costs = [o["cost_per_hour"] for o in result.offers]
        assert costs == sorted(costs)

    def test_max_10_results(self, manager, monkeypatch):
        """Returns at most 10 offers."""
        big_response = {
            "offers": [
                {"id": i, "gpu_name": "RTX 3090", "gpu_ram": 24576,
                 "num_gpus": 1, "dph_total": 0.05 + i * 0.001,
                 "cuda_max_good": 12.0, "disk_space": 50, "reliability2": 0.95,
                 "rentable": True, "geolocation": "US"}
                for i in range(20)
            ]
        }
        monkeypatch.setattr(_mod, "_vastai_request", lambda *a, **kw: big_response)
        with patch.object(CONFIG.vastai, "api_key", "test-key"):
            result = manager.search_offers(model="deepseek-r1:32b", max_cost=1.0)
        assert len(result.offers) <= 10

    def test_vram_gb_converted_correctly(self, manager, mock_api):
        """gpu_ram (MB) converted to vram_gb correctly."""
        with patch.object(CONFIG.vastai, "api_key", "test-key"):
            result = manager.search_offers(model="deepseek-r1:32b", max_cost=0.50)
        v100 = next(o for o in result.offers if o["id"] == 1001)
        assert v100["vram_gb"] == 32.0  # 32768 MB → 32 GB

    def test_api_error_returns_error_result(self, manager, monkeypatch):
        """API failure returns OfferResult with error, not exception."""
        monkeypatch.setattr(
            _mod, "_vastai_request",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("API down"))
        )
        with patch.object(CONFIG.vastai, "api_key", "test-key"):
            result = manager.search_offers()
        assert result.error
        assert "API down" in result.error

    def test_no_api_key_returns_error(self, manager):
        """No key → error without making network call."""
        with patch.object(CONFIG.vastai, "api_key", ""):
            result = manager.search_offers()
        assert result.error
        assert "api_key" in result.error.lower() or "not configured" in result.error.lower()


# ---------------------------------------------------------------------------
# provision — real API flow
# ---------------------------------------------------------------------------

class TestProvisionRealApi:

    def test_selects_cheapest_offer_automatically(self, manager, mock_api):
        """When no offer_id given, picks cheapest from search."""
        with patch.object(CONFIG.vastai, "api_key", "test-key"):
            result = manager.provision(approval=APPROVAL_TOKEN)
        assert result["success"] is True
        # Should have called PUT /asks/1001/ (cheapest valid offer)
        put_calls = [c for c in mock_api if c["method"] == "PUT"]
        assert len(put_calls) == 1
        assert "/asks/1001/" in put_calls[0]["path"]

    def test_uses_specified_offer_id(self, manager, mock_api):
        """When offer_id given, uses it directly without searching."""
        with patch.object(CONFIG.vastai, "api_key", "test-key"):
            result = manager.provision(approval=APPROVAL_TOKEN, offer_id=1002)
        assert result["success"] is True
        put_calls = [c for c in mock_api if c["method"] == "PUT"]
        assert "/asks/1002/" in put_calls[0]["path"]
        # No GET /bundles/ call when offer_id is explicit
        get_calls = [c for c in mock_api if c["method"] == "GET"]
        assert len(get_calls) == 0

    def test_instance_id_from_api_response(self, manager, mock_api):
        """instance_id comes from API response, not a mock string."""
        with patch.object(CONFIG.vastai, "api_key", "test-key"):
            result = manager.provision(approval=APPROVAL_TOKEN)
        assert result["success"] is True
        assert result["instance"]["instance_id"] == "88001"

    def test_provision_api_error_returns_failure(self, manager, monkeypatch):
        """API error during provision → success=False, no exception."""
        def _fail(method, path, api_key, payload=None, timeout=20):
            if method == "GET":
                return {"offers": [{"id": 9, "gpu_name": "RTX 3090",
                                    "gpu_ram": 24576, "num_gpus": 1,
                                    "dph_total": 0.05, "cuda_max_good": 12.0,
                                    "disk_space": 50, "reliability2": 0.95,
                                    "rentable": True, "geolocation": "US"}]}
            raise RuntimeError("Instance creation failed")

        monkeypatch.setattr(_mod, "_vastai_request", _fail)
        with patch.object(CONFIG.vastai, "api_key", "test-key"):
            result = manager.provision(approval=APPROVAL_TOKEN)
        assert result["success"] is False
        assert "Instance creation failed" in result.get("error", "")

    def test_no_offers_found_returns_failure(self, manager, monkeypatch):
        """Empty offer list → success=False."""
        monkeypatch.setattr(
            _mod, "_vastai_request", lambda *a, **kw: {"offers": []}
        )
        with patch.object(CONFIG.vastai, "api_key", "test-key"):
            result = manager.provision(approval=APPROVAL_TOKEN)
        assert result["success"] is False

    def test_provision_payload_includes_label(self, manager, mock_api):
        """Provision request includes an igris label."""
        with patch.object(CONFIG.vastai, "api_key", "test-key"):
            manager.provision(approval=APPROVAL_TOKEN, model="deepseek-r1:32b")
        put_calls = [c for c in mock_api if c["method"] == "PUT"]
        label = put_calls[0]["payload"].get("label", "")
        assert "igris" in label.lower()


# ---------------------------------------------------------------------------
# destroy — real API flow
# ---------------------------------------------------------------------------

class TestDestroyRealApi:

    def test_calls_delete_on_correct_instance(self, manager, mock_api):
        """DELETE is sent to the correct instance_id."""
        with patch.object(CONFIG.vastai, "api_key", "test-key"):
            manager.provision(approval=APPROVAL_TOKEN)
            result = manager.destroy(approval=APPROVAL_TOKEN)
        assert result["success"] is True
        delete_calls = [c for c in mock_api if c["method"] == "DELETE"]
        assert len(delete_calls) == 1
        assert "/instances/88001/" in delete_calls[0]["path"]

    def test_destroy_api_error_returns_failure(self, manager, monkeypatch):
        """API error during destroy → success=False, no exception."""
        def _mixed(method, path, api_key, payload=None, timeout=20):
            if method == "GET":
                return {"offers": [{"id": 9, "gpu_name": "RTX 3090",
                                    "gpu_ram": 24576, "num_gpus": 1,
                                    "dph_total": 0.05, "cuda_max_good": 12.0,
                                    "disk_space": 50, "reliability2": 0.95,
                                    "rentable": True, "geolocation": "US"}]}
            if method == "PUT":
                return {"id": 77777}
            raise RuntimeError("Delete failed on remote")

        monkeypatch.setattr(_mod, "_vastai_request", _mixed)
        with patch.object(CONFIG.vastai, "api_key", "test-key"):
            manager.provision(approval=APPROVAL_TOKEN)
            result = manager.destroy(approval=APPROVAL_TOKEN)
        assert result["success"] is False
        assert "Delete failed" in result.get("error", "")

    def test_destroyed_instance_id_in_response(self, manager, mock_api):
        """Response includes the destroyed instance ID."""
        with patch.object(CONFIG.vastai, "api_key", "test-key"):
            manager.provision(approval=APPROVAL_TOKEN)
            result = manager.destroy(approval=APPROVAL_TOKEN)
        assert result["destroyed_instance"] == "88001"

    def test_instance_status_set_to_destroyed(self, manager, mock_api):
        """Internal state updated to 'destroyed' after destroy."""
        with patch.object(CONFIG.vastai, "api_key", "test-key"):
            manager.provision(approval=APPROVAL_TOKEN)
            manager.destroy(approval=APPROVAL_TOKEN)
        assert manager._instance.status == "destroyed"


# ---------------------------------------------------------------------------
# Safety: no key exposure in any response
# ---------------------------------------------------------------------------

class TestNoKeyExposure:

    def test_search_result_no_key(self, manager, mock_api):
        with patch.object(CONFIG.vastai, "api_key", "super-secret-key"):
            result = manager.search_offers()
            raw = str(result.to_dict())
        assert "super-secret-key" not in raw

    def test_provision_result_no_key(self, manager, mock_api):
        with patch.object(CONFIG.vastai, "api_key", "super-secret-key"):
            result = manager.provision(approval=APPROVAL_TOKEN)
        assert "super-secret-key" not in str(result)

    def test_config_endpoint_no_key(self):
        from fastapi.testclient import TestClient
        from igris.web.server import create_app
        client = TestClient(create_app())
        with patch.object(CONFIG.vastai, "api_key", "top-secret-key"):
            r = client.get("/api/vastai/config")
        assert "top-secret-key" not in r.text
        assert r.json().get("api_key_present") is True


# ---------------------------------------------------------------------------
# Safety gates still enforced with real API path
# ---------------------------------------------------------------------------

class TestSafetyGatesWithRealApi:

    def test_provision_requires_approval_even_with_key(self, manager, mock_api):
        with patch.object(CONFIG.vastai, "api_key", "test-key"):
            result = manager.provision(approval="WRONG")
        assert result["success"] is False
        assert result.get("approval_required") is True
        # No API calls made
        assert len(mock_api) == 0

    def test_provision_disabled_mode_blocks_api(self, manager, mock_api):
        with patch.object(CONFIG.vastai, "api_key", "test-key"), \
             patch.object(CONFIG.vastai, "mode", "disabled"):
            result = manager.provision(approval=APPROVAL_TOKEN)
        assert result["success"] is False
        assert len(mock_api) == 0  # no API call attempted

    def test_provision_budget_gate_before_api(self, manager, mock_api):
        """Over-budget model blocked before any API call."""
        with patch.object(CONFIG.vastai, "api_key", "test-key"), \
             patch.object(CONFIG.vastai, "max_hourly_cost", 0.50):
            result = manager.provision(approval=APPROVAL_TOKEN, model="deepseek-r1:70b")
        assert result["success"] is False
        assert "budget" in result.get("error", "").lower() or "exceeds" in result.get("error", "").lower()
        assert len(mock_api) == 0

    def test_destroy_requires_approval_even_with_key(self, manager, mock_api):
        with patch.object(CONFIG.vastai, "api_key", "test-key"):
            manager.provision(approval=APPROVAL_TOKEN)
            mock_api.clear()
            result = manager.destroy(approval="")
        assert result["success"] is False
        assert len(mock_api) == 0  # no DELETE called without approval
