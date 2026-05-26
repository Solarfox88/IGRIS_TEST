"""Tests — ModelOrchestrator routes hard tasks to Vast.ai GPU automatically.

Verifies:
- Hard task types (hard_debugging, security_review, architecture_review)
  map to the gpu_reasoning profile
- vastai_ollama is first in the gpu_reasoning provider chain
- When instance is ready, orchestrator uses it directly (no cloud fallback)
- When instance is not ready, orchestrator falls through to deepseek_strong
  AND triggers auto-provision in the background
- auto_provision_for_orchestrator() respects all safety gates:
  VASTAI_AUTO_PROVISION=false, mode=disabled, no API key, budget
- Singleton _SHARED_MANAGER is used by both orchestrator and web server
- No API key leaks in any orchestrator response
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import threading
import time

import pytest

from igris.core.model_orchestrator import (
    ModelOrchestrator,
    TASK_PROFILE_MAP,
    OrchestratorResult,
    ProviderConfig,
)
from igris.layers.advisory import vastai_manager as _vast_mod
from igris.layers.advisory.vastai_manager import (
    VastAIManager,
    VastInstance,
    _SHARED_MANAGER,
    APPROVAL_TOKEN,
)
import igris.layers.advisory.vastai_fleet as _fleet_mod
from igris.models.config import CONFIG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ready_instance(model: str = "deepseek-r1:32b") -> VastInstance:
    """Return a VastInstance that is fully ready."""
    return VastInstance(
        instance_id="77001",
        status="running",
        model=model,
        instance_host="165.10.20.30",
        ollama_port=32768,
        ready=True,
    )


def _make_ollama_result(text: str = "analysis complete") -> OrchestratorResult:
    return OrchestratorResult(
        text=text,
        provider="ollama",
        model="deepseek-r1:32b",
        profile="gpu_reasoning",
        success=True,
    )


# ---------------------------------------------------------------------------
# Profile / chain structure
# ---------------------------------------------------------------------------

class TestGpuReasoningProfile:

    def test_hard_debugging_maps_to_gpu_reasoning(self):
        assert TASK_PROFILE_MAP["hard_debugging"] == "gpu_reasoning"

    def test_security_review_maps_to_gpu_reasoning(self):
        assert TASK_PROFILE_MAP["security_review"] == "gpu_reasoning"

    def test_architecture_review_maps_to_gpu_reasoning(self):
        assert TASK_PROFILE_MAP["architecture_review"] == "gpu_reasoning"

    def test_gpu_reasoning_chain_starts_with_vastai(self):
        orch = ModelOrchestrator()
        chain = orch._get_provider_chain("gpu_reasoning")
        assert chain[0] == "vastai_ollama"

    def test_gpu_reasoning_chain_has_cloud_fallback(self):
        orch = ModelOrchestrator()
        chain = orch._get_provider_chain("gpu_reasoning")
        # Must have at least one strong cloud provider as fallback
        assert any(p in chain for p in ("deepseek_strong", "anthropic", "openai_strong"))

    def test_vastai_ollama_provider_registered(self):
        orch = ModelOrchestrator()
        assert "vastai_ollama" in orch.providers

    def test_vastai_ollama_model_is_deepseek_r1(self):
        orch = ModelOrchestrator()
        p = orch.providers["vastai_ollama"]
        assert "deepseek-r1" in p.model

    def test_easy_tasks_not_on_gpu_reasoning(self):
        """Chat/code tasks must not trigger GPU provisioning."""
        for task_type in ("chat", "code_reasoning", "classification", "synthesis"):
            assert TASK_PROFILE_MAP.get(task_type) != "gpu_reasoning", (
                f"{task_type} should not use gpu_reasoning"
            )


# ---------------------------------------------------------------------------
# Routing: instance ready → uses Vast.ai directly
# ---------------------------------------------------------------------------

class TestOrchestratorUsesVastaiWhenReady:

    def test_hard_task_uses_vastai_when_ready(self):
        """When fleet has a ready endpoint, hard_debugging → vastai_ollama, no cloud call."""
        orch = ModelOrchestrator()

        with patch.object(_fleet_mod._SHARED_FLEET, "get_ready_endpoint",
                          return_value="http://165.10.20.30:32768"):
            with patch.object(orch, "_call_ollama", return_value=_make_ollama_result()) as mock_call:
                result = orch.complete(
                    task_type="hard_debugging",
                    messages=[{"role": "user", "content": "debug this crash"}],
                )

        assert result.success is True
        assert mock_call.called
        # Provider base_url was updated to the Vast.ai endpoint
        assert "165.10.20.30" in orch.providers["vastai_ollama"].base_url

    def test_security_review_uses_vastai_when_ready(self):
        orch = ModelOrchestrator()

        with patch.object(_fleet_mod._SHARED_FLEET, "get_ready_endpoint",
                          return_value="http://165.10.20.30:32768"):
            with patch.object(orch, "_call_ollama", return_value=_make_ollama_result("secure")) as mc:
                result = orch.complete(
                    task_type="security_review",
                    messages=[{"role": "user", "content": "review auth code"}],
                )

        assert result.success is True
        assert mc.called

    def test_vastai_used_before_cloud_when_ready(self):
        """vastai_ollama must be tried before deepseek_strong when fleet has endpoint."""
        orch = ModelOrchestrator()
        call_order = []

        def track_call(provider, *args, **kwargs):
            call_order.append(provider.name)
            return _make_ollama_result()

        with patch.object(_fleet_mod._SHARED_FLEET, "get_ready_endpoint",
                          return_value="http://1.2.3.4:11434"):
            with patch.object(orch, "_call_provider", side_effect=track_call):
                orch.complete(
                    task_type="architecture_review",
                    messages=[{"role": "user", "content": "review arch"}],
                )

        assert call_order[0] == "vastai_ollama"


# ---------------------------------------------------------------------------
# Routing: instance not ready → fallback + auto-provision triggered
# ---------------------------------------------------------------------------

class TestOrchestratorFallbackAndAutoProvision:

    def test_falls_through_to_cloud_when_not_ready(self):
        """If Vast.ai not ready, call goes to deepseek_strong transparently."""
        orch = ModelOrchestrator()

        with patch.object(_vast_mod, "_SHARED_MANAGER") as mock_mgr:
            mock_mgr.get_ollama_endpoint.return_value = None
            mock_mgr.auto_provision_for_orchestrator.return_value = False

            cloud_result = OrchestratorResult(
                text="cloud analysis", provider="deepseek_strong",
                model="deepseek-v4-pro", success=True,
            )
            with patch.object(orch, "_call_openai_compatible", return_value=cloud_result):
                import os
                with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
                    result = orch.complete(
                        task_type="hard_debugging",
                        messages=[{"role": "user", "content": "debug"}],
                    )

        assert result.success is True
        assert result.provider in ("deepseek_strong", "anthropic", "openai_strong", "deterministic_fallback")

    def test_auto_provision_triggered_when_not_ready(self):
        """When no GPU instance and auto_provision=True, provision is triggered."""
        orch = ModelOrchestrator()

        with patch.object(_vast_mod, "_SHARED_MANAGER") as mock_mgr:
            mock_mgr.get_ollama_endpoint.return_value = None
            mock_mgr.auto_provision_for_orchestrator.return_value = True

            # Cloud call succeeds for this round
            cloud_result = OrchestratorResult(
                text="cloud", provider="deepseek_strong", model="deepseek-v4-pro", success=True,
            )
            with patch.object(orch, "_call_openai_compatible", return_value=cloud_result):
                import os
                with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
                    orch.complete(
                        task_type="hard_debugging",
                        messages=[{"role": "user", "content": "debug"}],
                    )

        # auto_provision_for_orchestrator must have been called
        mock_mgr.auto_provision_for_orchestrator.assert_called_once()

    def test_auto_provision_not_called_for_easy_tasks(self):
        """Easy tasks must never trigger GPU provisioning."""
        orch = ModelOrchestrator()

        with patch.object(_vast_mod, "_SHARED_MANAGER") as mock_mgr:
            mock_mgr.get_ollama_endpoint.return_value = None
            mock_mgr.auto_provision_for_orchestrator.return_value = False

            with patch.object(orch, "_call_ollama") as mock_ollama:
                mock_ollama.return_value = OrchestratorResult(text="ok", provider="ollama", success=True)
                orch.complete(
                    task_type="chat",
                    messages=[{"role": "user", "content": "hello"}],
                )

        mock_mgr.auto_provision_for_orchestrator.assert_not_called()


# ---------------------------------------------------------------------------
# auto_provision_for_orchestrator safety gates
# ---------------------------------------------------------------------------

class TestAutoProvisionSafetyGates:

    def test_auto_provision_disabled_flag(self):
        """VASTAI_AUTO_PROVISION=false → returns False, no API call."""
        mgr = VastAIManager()
        with patch.object(CONFIG.vastai, "auto_provision", False):
            result = mgr.auto_provision_for_orchestrator()
        assert result is False

    def test_auto_provision_no_api_key(self):
        """No API key → returns False."""
        mgr = VastAIManager()
        with patch.object(CONFIG.vastai, "auto_provision", True), \
             patch.object(CONFIG.vastai, "api_key", ""):
            result = mgr.auto_provision_for_orchestrator()
        assert result is False

    def test_auto_provision_mode_disabled(self):
        """mode=disabled → returns False."""
        mgr = VastAIManager()
        with patch.object(CONFIG.vastai, "auto_provision", True), \
             patch.object(CONFIG.vastai, "api_key", "test-key"), \
             patch.object(CONFIG.vastai, "mode", "disabled"):
            result = mgr.auto_provision_for_orchestrator()
        assert result is False

    def test_auto_provision_over_budget(self):
        """Model over budget → returns False."""
        mgr = VastAIManager()
        with patch.object(CONFIG.vastai, "auto_provision", True), \
             patch.object(CONFIG.vastai, "api_key", "test-key"), \
             patch.object(CONFIG.vastai, "mode", "on_demand"), \
             patch.object(CONFIG.vastai, "max_hourly_cost", 0.05):
            # deepseek-r1:32b estimated at $0.30/h > $0.05 budget
            result = mgr.auto_provision_for_orchestrator(model="deepseek-r1:32b")
        assert result is False

    def test_auto_provision_anti_duplicate(self):
        """Already provisioning → returns True without new API call."""
        mgr = VastAIManager()
        mgr._instance = VastInstance(instance_id="existing", status="provisioning")
        with patch.object(CONFIG.vastai, "auto_provision", True), \
             patch.object(CONFIG.vastai, "api_key", "test-key"), \
             patch.object(CONFIG.vastai, "mode", "on_demand"):
            with patch.object(_vast_mod, "_vastai_request") as mock_req:
                result = mgr.auto_provision_for_orchestrator()
        assert result is True
        mock_req.assert_not_called()  # no new API call

    def test_auto_provision_running_instance_no_duplicate(self):
        """Already running → returns True, no new provision."""
        mgr = VastAIManager()
        mgr._instance = VastInstance(instance_id="running-one", status="running", ready=True)
        with patch.object(CONFIG.vastai, "auto_provision", True), \
             patch.object(CONFIG.vastai, "api_key", "test-key"), \
             patch.object(CONFIG.vastai, "mode", "on_demand"):
            with patch.object(_vast_mod, "_vastai_request") as mock_req:
                result = mgr.auto_provision_for_orchestrator()
        assert result is True
        mock_req.assert_not_called()

    def test_auto_provision_success_starts_background_thread(self):
        """Successful provision kicks off a background polling thread."""
        mgr = VastAIManager()
        thread_started = []

        _fake_offers = {"offers": [
            {"id": 5001, "gpu_name": "RTX PRO 6000 WS", "gpu_ram": 24576,
             "num_gpus": 1, "dph_total": 0.05, "cuda_max_good": 13.1,
             "disk_space": 50, "reliability2": 0.95, "rentable": True,
             "geolocation": "US"}
        ]}
        _fake_provision = {"id": 9999}

        def _fake_req(method, path, api_key, payload=None, timeout=20):
            if method == "GET":
                return _fake_offers
            if method == "PUT":
                return _fake_provision
            return {}

        orig_thread_init = threading.Thread.__init__

        def _track_thread(self_t, *args, **kwargs):
            thread_started.append(kwargs.get("name", "?"))
            orig_thread_init(self_t, *args, **kwargs)
            # Override run so it doesn't actually poll
            self_t.run = lambda: None

        with patch.object(CONFIG.vastai, "auto_provision", True), \
             patch.object(CONFIG.vastai, "api_key", "test-key"), \
             patch.object(CONFIG.vastai, "mode", "on_demand"), \
             patch.object(CONFIG.vastai, "max_hourly_cost", 3.00), \
             patch.object(_vast_mod, "_vastai_request", _fake_req), \
             patch.object(threading.Thread, "__init__", _track_thread):
            result = mgr.auto_provision_for_orchestrator(model="deepseek-r1:32b")

        assert result is True
        assert mgr._instance is not None
        assert mgr._instance.status == "provisioning"
        assert any("vastai-poll" in t for t in thread_started)


# ---------------------------------------------------------------------------
# get_ollama_endpoint
# ---------------------------------------------------------------------------

class TestGetOllamaEndpoint:

    def test_returns_none_when_no_instance(self):
        mgr = VastAIManager()
        assert mgr.get_ollama_endpoint() is None

    def test_returns_none_when_provisioning(self):
        mgr = VastAIManager()
        mgr._instance = VastInstance(
            instance_id="x", status="provisioning", ready=False,
        )
        assert mgr.get_ollama_endpoint() is None

    def test_returns_endpoint_when_ready(self):
        mgr = VastAIManager()
        mgr._instance = _make_ready_instance()
        ep = mgr.get_ollama_endpoint()
        assert ep == "http://165.10.20.30:32768"

    def test_endpoint_uses_custom_port(self):
        mgr = VastAIManager()
        mgr._instance = VastInstance(
            instance_id="y", status="running", ready=True,
            instance_host="10.0.0.5", ollama_port=49200,
        )
        assert mgr.get_ollama_endpoint() == "http://10.0.0.5:49200"

    def test_returns_none_if_destroyed(self):
        mgr = VastAIManager()
        mgr._instance = VastInstance(
            instance_id="z", status="destroyed", ready=True,
            instance_host="1.2.3.4", ollama_port=11434,
        )
        assert mgr.get_ollama_endpoint() is None


# ---------------------------------------------------------------------------
# Singleton shared between orchestrator and web server
# ---------------------------------------------------------------------------

class TestSharedManagerSingleton:

    def test_shared_manager_is_vastai_manager(self):
        assert isinstance(_SHARED_MANAGER, VastAIManager)

    def test_orchestrator_uses_shared_fleet(self):
        """_check_vastai_available now uses _SHARED_FLEET.get_ready_endpoint()."""
        orch = ModelOrchestrator()

        with patch.object(_fleet_mod._SHARED_FLEET, "get_ready_endpoint",
                          return_value=None) as mock_ep:
            with patch.object(_vast_mod, "_SHARED_MANAGER") as mock_mgr:
                mock_mgr.auto_provision_for_orchestrator.return_value = False
                provider = orch.providers["vastai_ollama"]
                orch._check_vastai_available(provider)

        # Fleet's get_ready_endpoint was called — confirms orchestrator uses fleet
        mock_ep.assert_called_once()

    def test_state_visible_across_references(self):
        """Provision state set on _SHARED_MANAGER is visible via get_ollama_endpoint."""
        # Save and restore
        original_instance = _SHARED_MANAGER._instance
        try:
            _SHARED_MANAGER._instance = _make_ready_instance()
            ep = _SHARED_MANAGER.get_ollama_endpoint()
            assert ep is not None
            assert "165.10.20.30" in ep
        finally:
            _SHARED_MANAGER._instance = original_instance


# ---------------------------------------------------------------------------
# _probe_ollama
# ---------------------------------------------------------------------------

class TestProbeOllama:

    def test_returns_true_on_200(self):
        fake_resp = MagicMock()
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = MagicMock(return_value=False)
        fake_resp.status = 200
        with patch("urllib.request.urlopen", return_value=fake_resp):
            assert VastAIManager._probe_ollama("1.2.3.4", 11434) is True

    def test_returns_false_on_connection_error(self):
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            assert VastAIManager._probe_ollama("1.2.3.4", 11434) is False

    def test_returns_false_on_timeout(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            assert VastAIManager._probe_ollama("1.2.3.4", 11434) is False


# ---------------------------------------------------------------------------
# No API key in orchestrator responses
# ---------------------------------------------------------------------------

class TestNoKeyExposureInOrchestrator:

    def test_result_no_api_key(self):
        orch = ModelOrchestrator()
        with patch.object(_vast_mod, "_SHARED_MANAGER") as mock_mgr:
            mock_mgr.get_ollama_endpoint.return_value = "http://1.2.3.4:11434"
            mock_mgr._instance = _make_ready_instance()
            with patch.object(CONFIG.vastai, "api_key", "super-secret-key"):
                with patch.object(orch, "_call_ollama", return_value=_make_ollama_result()):
                    result = orch.complete(
                        task_type="hard_debugging",
                        messages=[{"role": "user", "content": "debug"}],
                    )
        assert "super-secret-key" not in str(result.to_dict())

    def test_history_no_api_key(self):
        orch = ModelOrchestrator()
        with patch.object(_vast_mod, "_SHARED_MANAGER") as mock_mgr:
            mock_mgr.get_ollama_endpoint.return_value = "http://1.2.3.4:11434"
            mock_mgr._instance = _make_ready_instance()
            with patch.object(CONFIG.vastai, "api_key", "super-secret-key"):
                with patch.object(orch, "_call_ollama", return_value=_make_ollama_result()):
                    orch.complete(
                        task_type="security_review",
                        messages=[{"role": "user", "content": "review"}],
                    )
        history = str(orch.get_history())
        assert "super-secret-key" not in history
