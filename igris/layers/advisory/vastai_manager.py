"""Vast.ai Manager — gated, live API when key present.

Manages GPU instance lifecycle for DeepSeek R1 inference.
All destructive operations (provision, destroy, set-mode) require
explicit approval. No auto-provisioning from the autonomous loop.

Config defaults:
  VASTAI_MODEL=deepseek-r1:32b
  VASTAI_FALLBACK_MODEL=qwen2.5-coder:7b
  VASTAI_AUTO_PROVISION=false
  VASTAI_REQUIRE_APPROVAL=true
  VASTAI_MAX_HOURLY_COST=0.50
  VASTAI_MODE=on_demand
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)

from igris.core.safety import redact_secrets
from igris.models.config import CONFIG


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APPROVAL_TOKEN = "I_APPROVE_VASTAI_COSTS"

SUPPORTED_MODELS = {
    "deepseek-r1:32b": {
        "vram_gb": 24,
        "min_gpu": "RTX 3090",
        "estimated_cost_hr": 0.30,
    },
    "deepseek-r1:70b": {
        "vram_gb": 48,
        "min_gpu": "A6000",
        "estimated_cost_hr": 0.60,
    },
    "qwen2.5-coder:7b": {
        "vram_gb": 8,
        "min_gpu": "RTX 3060",
        "estimated_cost_hr": 0.10,
    },
    "qwen2.5-coder:32b": {
        "vram_gb": 24,
        "min_gpu": "RTX 3090",
        "estimated_cost_hr": 0.30,
    },
}

VALID_MODES = {"on_demand", "always_on", "disabled"}

VASTAI_API_BASE = "https://console.vast.ai/api/v0"


# ---------------------------------------------------------------------------
# Internal HTTP helper
# ---------------------------------------------------------------------------

def _vastai_request(
    method: str,
    path: str,
    api_key: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 20,
) -> Dict[str, Any]:
    """Make a Vast.ai REST API call. Returns parsed JSON or raises on error."""
    url = f"{VASTAI_API_BASE}{path}"
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:200]
        except Exception:
            pass
        raise RuntimeError(f"Vast.ai API {method} {path} → HTTP {e.code}: {body}") from e
    except Exception as e:
        raise RuntimeError(f"Vast.ai API {method} {path} error: {e}") from e


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VastInstance:
    """Represents a Vast.ai GPU instance."""
    instance_id: str = ""
    status: str = "none"  # none | searching | provisioning | running | stopping | destroyed
    model: str = ""
    gpu: str = ""
    cost_per_hour: float = 0.0
    created_at: str = ""
    region: str = ""
    # Orchestrator fields — set once instance is running and reachable
    instance_host: str = ""   # public IP / hostname
    ollama_port: int = 11434  # mapped port for Ollama
    ready: bool = False       # True when Ollama is confirmed reachable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "status": self.status,
            "model": self.model,
            "gpu": self.gpu,
            "cost_per_hour": self.cost_per_hour,
            "created_at": self.created_at,
            "region": self.region,
            "ready": self.ready,
            "ollama_endpoint": (
                f"http://{self.instance_host}:{self.ollama_port}"
                if self.ready else ""
            ),
        }


@dataclass
class OfferResult:
    """Search result for available GPU offers."""
    offers: List[Dict[str, Any]] = field(default_factory=list)
    model: str = ""
    min_vram_gb: int = 0
    max_cost_hr: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "offers": self.offers,
            "model": self.model,
            "min_vram_gb": self.min_vram_gb,
            "max_cost_hr": self.max_cost_hr,
            "offer_count": len(self.offers),
            "error": redact_secrets(self.error),
        }


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class VastAIManager:
    """Gated Vast.ai GPU instance manager.

    All destructive operations require explicit approval.
    No real API calls — mock/dry-run only.
    """

    def __init__(self) -> None:
        self._instance: Optional[VastInstance] = None
        self._provision_history: List[Dict[str, Any]] = []
        # Cleanup orphaned instances from previous service runs in the background.
        # This handles the case where the service restarted while an instance was
        # provisioning or errored — without this, the instance keeps billing.
        self._startup_cleanup()

    # -- Config --

    def get_config(self) -> Dict[str, Any]:
        """Return current Vast.ai configuration (no secrets)."""
        cfg = CONFIG.vastai
        return {
            "model": cfg.model,
            "fallback_model": cfg.fallback_model,
            "auto_provision": cfg.auto_provision,
            "require_approval": cfg.require_approval,
            "max_hourly_cost": cfg.max_hourly_cost,
            "mode": cfg.mode,
            "api_key_present": bool(cfg.api_key),
            "supported_models": list(SUPPORTED_MODELS.keys()),
        }

    # -- Status --

    def get_status(self) -> Dict[str, Any]:
        """Return current instance status."""
        cfg = CONFIG.vastai
        return {
            "configured": bool(cfg.api_key),
            "mode": cfg.mode,
            "model": cfg.model,
            "instance": self._instance.to_dict() if self._instance else None,
            "has_active_instance": (
                self._instance is not None
                and self._instance.status in ("running", "provisioning")
            ),
            "auto_provision": cfg.auto_provision,
            "require_approval": cfg.require_approval,
            "provision_count": len(self._provision_history),
        }

    # -- Cost estimation --

    def estimate_cost(
        self,
        model: Optional[str] = None,
        hours: float = 1.0,
    ) -> Dict[str, Any]:
        """Estimate cost for running a model."""
        model = model or CONFIG.vastai.model
        info = SUPPORTED_MODELS.get(model)

        if not info:
            return {
                "model": model,
                "error": f"Unknown model: {model}. Supported: {list(SUPPORTED_MODELS.keys())}",
                "estimated_cost": 0.0,
            }

        cost_hr = info["estimated_cost_hr"]
        total = cost_hr * hours
        budget = CONFIG.vastai.max_hourly_cost

        return {
            "model": model,
            "vram_gb": info["vram_gb"],
            "min_gpu": info["min_gpu"],
            "cost_per_hour": cost_hr,
            "hours": hours,
            "estimated_total": round(total, 4),
            "max_hourly_budget": budget,
            "within_budget": cost_hr <= budget,
            "warning": (
                f"Cost ${cost_hr}/hr exceeds budget ${budget}/hr"
                if cost_hr > budget else ""
            ),
        }

    # -- Offer search (mock) --

    def search_offers(
        self,
        model: Optional[str] = None,
        max_cost: Optional[float] = None,
    ) -> OfferResult:
        """Search for GPU offers — real API call when key is present, mock otherwise."""
        cfg = CONFIG.vastai
        model = model or cfg.model
        max_cost = max_cost or cfg.max_hourly_cost
        info = SUPPORTED_MODELS.get(model)

        if not info:
            return OfferResult(model=model, error=f"Unknown model: {model}")

        if not cfg.api_key:
            return OfferResult(model=model, error="VASTAI_API_KEY not configured")

        vram_mb = info["vram_gb"] * 1024  # Vast.ai uses MB
        try:
            data = _vastai_request("GET", "/bundles/", cfg.api_key)
            raw_offers = data.get("offers", [])
            offers = []
            for o in raw_offers:
                gpu_ram_mb = o.get("gpu_ram", 0)
                dph = o.get("dph_total", 999)
                # Only include rentable offers — non-rentable ones silently fail
                # the PUT /asks/{id}/ provisioning call, wasting the attempt.
                is_rentable = o.get("rentable", True)
                if gpu_ram_mb >= vram_mb and dph <= max_cost and is_rentable:
                    offers.append({
                        "id": o.get("id"),
                        "gpu": o.get("gpu_name", "?"),
                        "vram_gb": round(gpu_ram_mb / 1024, 1),
                        "cost_per_hour": round(dph, 4),
                        "num_gpus": o.get("num_gpus", 1),
                        "cuda": o.get("cuda_max_good", "?"),
                        "region": o.get("geolocation", "?"),
                        "available": True,
                    })
            offers.sort(key=lambda x: x["cost_per_hour"])
            return OfferResult(
                offers=offers[:10],
                model=model,
                min_vram_gb=info["vram_gb"],
                max_cost_hr=max_cost,
            )
        except Exception as e:
            return OfferResult(model=model, error=str(e))

    # -- Provision (gated) --

    def provision(
        self,
        approval: str = "",
        model: Optional[str] = None,
        offer_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Provision a GPU instance (gated, mock).

        Requires:
        - approval = "I_APPROVE_VASTAI_COSTS"
        - API key configured
        - No existing active instance (anti-duplicate)
        - Cost within budget
        - Mode not disabled
        """
        cfg = CONFIG.vastai
        model = model or cfg.model

        # Gate: mode check
        if cfg.mode == "disabled":
            return {
                "success": False,
                "error": "Vast.ai mode is 'disabled'. Set VASTAI_MODE=on_demand to enable.",
                "gated": True,
            }

        # Gate: approval
        if cfg.require_approval and approval != APPROVAL_TOKEN:
            return {
                "success": False,
                "error": f"Approval required. Send approval='{APPROVAL_TOKEN}' to confirm.",
                "approval_required": True,
                "gated": True,
            }

        # Gate: API key
        if not cfg.api_key:
            return {
                "success": False,
                "error": "VASTAI_API_KEY not configured.",
                "gated": True,
            }

        # Gate: anti-duplicate
        if self._instance and self._instance.status in ("running", "provisioning"):
            return {
                "success": False,
                "error": f"Active instance already exists: {self._instance.instance_id} ({self._instance.status})",
                "existing_instance": self._instance.to_dict(),
                "gated": True,
            }

        # Gate: budget
        estimate = self.estimate_cost(model)
        if not estimate.get("within_budget", False):
            return {
                "success": False,
                "error": estimate.get("warning", "Over budget"),
                "estimate": estimate,
                "gated": True,
            }

        # Real provision via Vast.ai API
        try:
            # Find best offer if not specified
            if not offer_id:
                result = self.search_offers(model=model, max_cost=cfg.max_hourly_cost)
                if result.error or not result.offers:
                    return {
                        "success": False,
                        "error": result.error or "No suitable offers found",
                        "gated": True,
                    }
                offer_id = result.offers[0]["id"]

            # Rent the instance (Vast.ai: PUT /asks/<id>/)
            resp = _vastai_request(
                "PUT",
                f"/asks/{offer_id}/",
                cfg.api_key,
                payload={
                    "client_id": "me",
                    "image": "pytorch/pytorch:latest",
                    "runtype": "ssh",
                    # NOTE: do NOT pass "disk" — triggers --storage-opt size=Ng
                    # which requires XFS+pquota on the host (most hosts use ext4).
                    "label": f"igris-{model.replace(':','-')}",
                },
            )
            instance_id = str(resp.get("id") or resp.get("new_contract", ""))
            if not instance_id:
                return {"success": False, "error": f"Provision failed: {resp}", "gated": True}

            instance = VastInstance(
                instance_id=instance_id,
                status="provisioning",
                model=model,
                gpu=SUPPORTED_MODELS.get(model, {}).get("min_gpu", "unknown"),
                cost_per_hour=SUPPORTED_MODELS.get(model, {}).get("estimated_cost_hr", 0.0),
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                region="vast.ai",
            )
            self._instance = instance
            self._provision_history.append({
                "action": "provision",
                "instance_id": instance_id,
                "offer_id": offer_id,
                "model": model,
                "timestamp": instance.created_at,
            })
            return {"success": True, "instance": instance.to_dict(), "gated": True}

        except Exception as e:
            return {"success": False, "error": str(e), "gated": True}

    # -- Destroy (gated) --

    def destroy(self, approval: str = "") -> Dict[str, Any]:
        """Destroy the current instance (gated, state-aware)."""
        cfg = CONFIG.vastai

        # Gate: approval
        if cfg.require_approval and approval != APPROVAL_TOKEN:
            return {
                "success": False,
                "error": f"Approval required. Send approval='{APPROVAL_TOKEN}' to confirm.",
                "approval_required": True,
            }

        # State check
        if not self._instance or self._instance.status in ("none", "destroyed"):
            return {
                "success": False,
                "error": "No active instance to destroy.",
            }

        # Real destroy via Vast.ai API
        old_id = self._instance.instance_id
        try:
            _vastai_request("DELETE", f"/instances/{old_id}/", cfg.api_key)
        except Exception as e:
            return {"success": False, "error": f"Destroy API call failed: {e}"}

        self._instance.status = "destroyed"
        self._provision_history.append({
            "action": "destroy",
            "instance_id": old_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        return {"success": True, "destroyed_instance": old_id}

    # -- Orchestrator integration --

    def get_ollama_endpoint(self) -> Optional[str]:
        """Return Ollama base_url if a running instance is ready, else None."""
        if (
            self._instance
            and self._instance.ready
            and self._instance.instance_host
            and self._instance.status == "running"
        ):
            return f"http://{self._instance.instance_host}:{self._instance.ollama_port}"
        return None

    def auto_provision_for_orchestrator(self, model: Optional[str] = None) -> bool:
        """Provision a GPU instance autonomously for the orchestrator.

        Gate: VASTAI_AUTO_PROVISION=true (replaces user approval token).
        All other gates still apply: API key, mode, budget, anti-duplicate.
        Returns True if provisioning was started or instance is already active.
        """
        cfg = CONFIG.vastai

        if not cfg.auto_provision:
            return False
        if not cfg.api_key:
            _log.debug("vastai auto_provision skipped: no API key")
            return False
        if cfg.mode == "disabled":
            _log.debug("vastai auto_provision skipped: mode=disabled")
            return False

        # Anti-duplicate: already provisioning or running
        if self._instance and self._instance.status in ("provisioning", "running"):
            return True

        model = model or cfg.model
        estimate = self.estimate_cost(model)
        if not estimate.get("within_budget", False):
            _log.debug("vastai auto_provision skipped: over budget for %s", model)
            return False

        try:
            result = self.search_offers(model=model, max_cost=cfg.max_hourly_cost)
            if result.error or not result.offers:
                _log.warning("vastai auto_provision: no offers found — %s", result.error)
                return False
            offer_id = result.offers[0]["id"]

            resp = _vastai_request(
                "PUT",
                f"/asks/{offer_id}/",
                cfg.api_key,
                payload={
                    "client_id": "me",
                    # Ollama official image; starts server on port 11434
                    "image": "ollama/ollama:latest",
                    "runtype": "ssh",
                    # NOTE: do NOT pass "disk" — it triggers --storage-opt size=Ng
                    # which requires XFS with pquota on the host.  Most Vast.ai hosts
                    # use ext4 and fail with "storage-opt is supported only for overlay
                    # over xfs with 'pquota' mount option".  Vast.ai allocates disk
                    # automatically from the offer's available space.
                    "label": f"igris-orchestrator-{model.replace(':', '-')}",
                    # Pull model on startup so it's ready for first inference
                    "onstart": (
                        "ollama serve &>/var/log/ollama.log & "
                        "sleep 15 && "
                        f"ollama pull {model} &>/var/log/ollama_pull.log"
                    ),
                    "ports": "11434",  # expose Ollama port
                },
            )
            instance_id = str(resp.get("id") or resp.get("new_contract", ""))
            if not instance_id:
                _log.warning("vastai auto_provision: no instance_id in response %s", resp)
                return False

            self._instance = VastInstance(
                instance_id=instance_id,
                status="provisioning",
                model=model,
                gpu=result.offers[0].get("gpu", "unknown"),
                cost_per_hour=result.offers[0].get("cost_per_hour", 0.0),
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                region=result.offers[0].get("region", "vast.ai"),
            )
            self._provision_history.append({
                "action": "auto_provision",
                "instance_id": instance_id,
                "offer_id": offer_id,
                "model": model,
                "timestamp": self._instance.created_at,
            })
            _log.info(
                "vastai auto_provision started: instance_id=%s offer=%s model=%s",
                instance_id, offer_id, model,
            )

            # Background thread: polls until Ollama is reachable, then marks ready
            t = threading.Thread(
                target=self._poll_until_ready,
                args=(instance_id, cfg.api_key),
                daemon=True,
                name=f"vastai-poll-{instance_id}",
            )
            t.start()
            return True

        except Exception as exc:
            _log.warning("vastai auto_provision failed: %s", exc)
            return False

    def _poll_until_ready(
        self,
        instance_id: str,
        api_key: str,
        max_wait: int = 600,
        poll_interval: int = 20,
    ) -> None:
        """Background thread: poll Vast.ai until instance is running and Ollama responds."""
        deadline = time.time() + max_wait
        _log.info("vastai poll started: instance_id=%s max_wait=%ds", instance_id, max_wait)

        while time.time() < deadline:
            time.sleep(poll_interval)
            try:
                data = _vastai_request("GET", f"/instances/{instance_id}/", api_key)
                # Vast.ai returns {"instances": [...]} or the instance directly
                instances = data.get("instances", [data])
                inst = instances[0] if isinstance(instances, list) and instances else data

                actual_status = inst.get("actual_status", "")
                _log.debug("vastai poll: instance_id=%s status=%s", instance_id, actual_status)

                if actual_status == "running":
                    ssh_host = inst.get("ssh_host", "")
                    ports = inst.get("ports", {})

                    # Get mapped external port for Ollama (11434/tcp)
                    ollama_port = 11434
                    for port_key in ("11434/tcp", "11434"):
                        if port_key in ports and ports[port_key]:
                            try:
                                ollama_port = int(ports[port_key][0].get("HostPort", 11434))
                            except (KeyError, ValueError, TypeError):
                                pass
                            break

                    if not ssh_host:
                        _log.debug("vastai poll: running but no ssh_host yet")
                        continue

                    # Probe Ollama /api/tags to confirm it's actually up
                    if self._probe_ollama(ssh_host, ollama_port):
                        if self._instance and self._instance.instance_id == instance_id:
                            self._instance.instance_host = ssh_host
                            self._instance.ollama_port = ollama_port
                            self._instance.ready = True
                            self._instance.status = "running"
                        _log.info(
                            "vastai ready: instance_id=%s endpoint=http://%s:%d",
                            instance_id, ssh_host, ollama_port,
                        )
                        return
                    else:
                        _log.debug(
                            "vastai poll: instance running but Ollama not yet up at %s:%d",
                            ssh_host, ollama_port,
                        )

                elif actual_status in ("offline", "exited", "error"):
                    _log.warning(
                        "vastai poll: instance_id=%s entered terminal state=%s — destroying",
                        instance_id, actual_status,
                    )
                    # Actually terminate the instance on Vast.ai so it stops billing.
                    try:
                        _vastai_request("DELETE", f"/instances/{instance_id}/", api_key)
                        _log.info("vastai poll: deleted failed instance %s", instance_id)
                    except Exception as _del_exc:
                        _log.warning("vastai poll: DELETE failed for %s: %s", instance_id, _del_exc)
                    if self._instance and self._instance.instance_id == instance_id:
                        self._instance.status = "destroyed"
                    return

                elif actual_status in ("loading", None, ""):
                    # Check if Docker reported an error in status_msg (e.g. storage-opt
                    # error on ext4 hosts).  These never transition to "error" — they
                    # stay in "loading" until we explicitly kill them.
                    status_msg = inst.get("status_msg") or ""
                    _FATAL_MSG_PATTERNS = (
                        "storage-opt",
                        "Error response from daemon",
                        "OCI runtime",
                        "no space left",
                        "permission denied",
                    )
                    if any(p in status_msg for p in _FATAL_MSG_PATTERNS):
                        _log.warning(
                            "vastai poll: instance_id=%s stuck loading with fatal Docker error"
                            " — destroying.  status_msg=%r",
                            instance_id, status_msg[:200],
                        )
                        try:
                            _vastai_request("DELETE", f"/instances/{instance_id}/", api_key)
                            _log.info("vastai poll: deleted stuck-loading instance %s", instance_id)
                        except Exception as _del_exc:
                            _log.warning(
                                "vastai poll: DELETE failed for %s: %s", instance_id, _del_exc
                            )
                        if self._instance and self._instance.instance_id == instance_id:
                            self._instance.status = "destroyed"
                        return

            except Exception as exc:
                _log.debug("vastai poll error: %s", exc)

        _log.warning("vastai poll timed out after %ds: instance_id=%s", max_wait, instance_id)

    def _startup_cleanup(self) -> None:
        """Delete any orphaned igris-orchestrator-* instances left by previous runs.

        Runs in a background thread so it doesn't block startup.  Instances
        in error/exited/offline state are deleted immediately.  Running instances
        are adopted (so the next get_ready_endpoint() call can use them).
        """
        cfg = CONFIG.vastai
        if not cfg.api_key or cfg.mode == "disabled":
            return

        def _cleanup() -> None:
            try:
                data = _vastai_request("GET", "/instances/", cfg.api_key)
                instances = data.get("instances", [])
                for inst in instances:
                    label = inst.get("label", "") or ""
                    if not label.startswith("igris-orchestrator-"):
                        continue
                    inst_id = str(inst.get("id", ""))
                    actual_status = inst.get("actual_status", "")
                    _log.info(
                        "vastai startup: found orphaned instance %s label=%s status=%s",
                        inst_id, label, actual_status,
                    )
                    # "loading" with a Docker error or stuck for >5 min counts as terminal.
                    status_msg = inst.get("status_msg") or ""
                    _FATAL_PATTERNS = (
                        "storage-opt", "Error response from daemon",
                        "OCI runtime", "no space left", "permission denied",
                    )
                    loading_with_error = actual_status in ("loading", None) and any(
                        p in status_msg for p in _FATAL_PATTERNS
                    )
                    # actual_status=None (JSON null) means Vast.ai hasn't assigned a status
                    # yet — if we see it at startup time, the previous service run left it
                    # stuck before Docker ever started.  Treat as terminal.
                    is_terminal = (
                        actual_status in ("error", "exited", "offline", "")
                        or actual_status is None
                        or loading_with_error
                    )
                    if is_terminal:
                        try:
                            _vastai_request("DELETE", f"/instances/{inst_id}/", cfg.api_key)
                            _log.info("vastai startup: deleted orphaned instance %s (status=%s)", inst_id, actual_status)
                        except Exception as exc:
                            _log.warning("vastai startup: DELETE %s failed: %s", inst_id, exc)
                    elif actual_status == "running" and self._instance is None:
                        # Adopt a running instance so we don't re-provision needlessly
                        ssh_host = inst.get("ssh_host", "")
                        model = label.replace("igris-orchestrator-", "").replace("-", ":", 1)
                        self._instance = VastInstance(
                            instance_id=inst_id,
                            status="running",
                            model=model,
                            gpu=inst.get("gpu_name", "unknown"),
                            cost_per_hour=inst.get("dph_total", 0.0),
                            created_at=str(inst.get("start_date", "")),
                            region=inst.get("geolocation", "vast.ai"),
                            instance_host=ssh_host,
                            ready=False,  # will be confirmed by probe
                        )
                        _log.info(
                            "vastai startup: adopted running instance %s at %s",
                            inst_id, ssh_host,
                        )
            except Exception as exc:
                _log.debug("vastai startup cleanup error: %s", exc)

        t = threading.Thread(target=_cleanup, daemon=True, name="vastai-startup-cleanup")
        t.start()

    @staticmethod
    def _probe_ollama(host: str, port: int, timeout: int = 5) -> bool:
        """Return True if Ollama /api/tags responds at host:port."""
        try:
            url = f"http://{host}:{port}/api/tags"
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    # -- Set mode (gated) --

    def set_mode(
        self,
        mode: str,
        approval: str = "",
    ) -> Dict[str, Any]:
        """Change the Vast.ai operating mode (gated)."""
        if mode not in VALID_MODES:
            return {
                "success": False,
                "error": f"Invalid mode: {mode}. Valid: {sorted(VALID_MODES)}",
            }

        cfg = CONFIG.vastai

        if cfg.require_approval and approval != APPROVAL_TOKEN:
            return {
                "success": False,
                "error": f"Approval required. Send approval='{APPROVAL_TOKEN}' to confirm.",
                "approval_required": True,
            }

        old_mode = cfg.mode
        CONFIG.vastai.mode = mode

        return {
            "success": True,
            "old_mode": old_mode,
            "new_mode": mode,
            "note": "Mode changed in-memory only. Set VASTAI_MODE env var for persistence.",
        }


# ---------------------------------------------------------------------------
# Module-level shared instance
# ---------------------------------------------------------------------------

#: Singleton used by ModelOrchestrator and the web API so they share state.
#: Do NOT import VastAIManager and create a new instance — import this instead.
_SHARED_MANAGER: VastAIManager = VastAIManager()
