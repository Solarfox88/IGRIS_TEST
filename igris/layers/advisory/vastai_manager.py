"""Vast.ai Manager — gated, mock-safe, no real API calls in CI.

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

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "status": self.status,
            "model": self.model,
            "gpu": self.gpu,
            "cost_per_hour": self.cost_per_hour,
            "created_at": self.created_at,
            "region": self.region,
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
        """Search for GPU offers (mock — no real API calls)."""
        cfg = CONFIG.vastai
        model = model or cfg.model
        max_cost = max_cost or cfg.max_hourly_cost
        info = SUPPORTED_MODELS.get(model)

        if not info:
            return OfferResult(
                model=model,
                error=f"Unknown model: {model}",
            )

        if not cfg.api_key:
            return OfferResult(
                model=model,
                error="VASTAI_API_KEY not configured",
            )

        # Mock offers (no real API call)
        mock_offers = [
            {
                "id": "mock-offer-001",
                "gpu": info["min_gpu"],
                "vram_gb": info["vram_gb"],
                "cost_per_hour": info["estimated_cost_hr"],
                "region": "us-east",
                "available": True,
                "note": "MOCK — no real Vast.ai API call made",
            },
            {
                "id": "mock-offer-002",
                "gpu": info["min_gpu"],
                "vram_gb": info["vram_gb"],
                "cost_per_hour": round(info["estimated_cost_hr"] * 0.9, 4),
                "region": "eu-west",
                "available": True,
                "note": "MOCK — no real Vast.ai API call made",
            },
        ]

        # Filter by budget
        offers = [o for o in mock_offers if o["cost_per_hour"] <= max_cost]

        return OfferResult(
            offers=offers,
            model=model,
            min_vram_gb=info["vram_gb"],
            max_cost_hr=max_cost,
        )

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

        # Mock provision (no real API call)
        instance = VastInstance(
            instance_id=f"mock-{int(time.time())}",
            status="provisioning",
            model=model,
            gpu=SUPPORTED_MODELS.get(model, {}).get("min_gpu", "unknown"),
            cost_per_hour=SUPPORTED_MODELS.get(model, {}).get("estimated_cost_hr", 0.0),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            region="mock-region",
        )
        self._instance = instance
        self._provision_history.append({
            "action": "provision",
            "instance_id": instance.instance_id,
            "model": model,
            "timestamp": instance.created_at,
            "mock": True,
        })

        return {
            "success": True,
            "instance": instance.to_dict(),
            "note": "MOCK — no real Vast.ai instance created. No costs incurred.",
            "gated": True,
        }

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

        # Mock destroy
        old_id = self._instance.instance_id
        self._instance.status = "destroyed"
        self._provision_history.append({
            "action": "destroy",
            "instance_id": old_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mock": True,
        })

        return {
            "success": True,
            "destroyed_instance": old_id,
            "note": "MOCK — no real instance destroyed.",
        }

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
