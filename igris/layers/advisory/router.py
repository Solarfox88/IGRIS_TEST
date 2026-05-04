"""
Provider routing logic.

The router determines which language model provider to use based on availability
and cost.  It uses the configuration to decide between the local provider
(Ollama), a fallback provider (OpenAI) or (in the future) Vast.ai.  The
`explain_routing` function returns a human‑readable explanation for the last
decision.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any

from igris.models.config import CONFIG


class Provider(str):
    LOCAL = "local"
    FALLBACK = "fallback"
    VASTAI = "vastai"


_last_provider: Optional[Tuple[str, str]] = None
# Maintain a history of provider decisions for cost/routing summary.
# Each entry is a dict with provider, model and reason fields.
_provider_history: List[Dict[str, Any]] = []

# Budget configuration (per-session, not persistent)
_budget_config: Dict[str, Any] = {
    "max_session_cost": 10.0,  # USD
    "warn_threshold": 0.8,  # warn at 80% of budget
    "cost_per_local_call": 0.0,
    "cost_per_fallback_call": 0.003,
    "cost_per_vastai_call": 0.01,
}

# Estimated cost per provider per call (USD)
COST_ESTIMATES: Dict[str, float] = {
    "local": 0.0,
    "ollama": 0.0,
    "deterministic": 0.0,
    "fallback": 0.003,
    "openai": 0.003,
    "vastai": 0.01,
}


def choose_provider(for_task: str = "chat") -> Tuple[str, str]:
    """Return the name and model of the chosen provider and record the choice.

    Currently the logic is simplistic: always use the local provider.  If the
    fallback API key is missing the fallback provider will not be chosen.
    This function records the last choice for reporting in both `_last_provider` and
    `_provider_history`.  In future versions this logic can consider factors
    such as task complexity, model capabilities, latency and cost budgets.
    """
    global _last_provider, _provider_history
    # Determine provider; default to local
    provider = Provider.LOCAL
    model = CONFIG.local_llm.model
    reason = "Using local provider because it is low cost and sufficient for the task."
    # TODO: add logic for fallback or vastai here based on CONFIG and availability
    _last_provider = (provider, model)
    # Record history entry
    _provider_history.append({
        "provider": provider,
        "model": model,
        "reason": reason,
        "estimated_cost": COST_ESTIMATES.get(provider, 0.0),
    })
    return _last_provider


def explain_routing() -> str:
    """Explain why the last provider was chosen."""
    if _last_provider is None:
        return "No provider has been chosen yet.  Messages have not been sent."
    provider, model = _last_provider
    if provider == Provider.LOCAL:
        return (
            f"Using local provider {CONFIG.local_llm.provider} with model {model} because it is low cost and sufficient for the task."
        )
    elif provider == Provider.FALLBACK:
        return (
            f"Using fallback provider {CONFIG.fallback_llm.provider} with model {model} because the local model was unable to answer."
        )
    elif provider == Provider.VASTAI:
        return (
            f"Using Vast.ai instance with model {model} due to high compute requirements."
        )
    return "Unknown provider choice."


def record_chat_routing(
    provider: str, model: str, reason: str,
    latency_ms: int = 0, fallback_used: bool = False,
    estimated_cost: float = 0.0, task_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """Record a chat routing decision in the history."""
    global _last_provider
    import time as _time
    _last_provider = (provider, model)
    _provider_history.append({
        "provider": provider,
        "model": model,
        "reason": reason,
        "latency_ms": latency_ms,
        "fallback_used": fallback_used,
        "estimated_cost": estimated_cost,
        "task_id": task_id,
        "session_id": session_id,
        "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
    })


def get_history() -> List[Dict[str, Any]]:
    """Return a copy of the provider history.

    Each entry in the history is a dict with keys: provider, model and reason.
    """
    return list(_provider_history)


def cost_summary() -> Dict[str, Any]:
    """Return a summary of provider usage with cost estimates."""
    providers: Dict[str, int] = {}
    local_calls = 0
    fallback_calls = 0
    estimated_cost_total = 0.0
    for entry in _provider_history:
        p = entry["provider"]
        providers.setdefault(p, 0)
        providers[p] += 1
        if p in ("local", "ollama", "deterministic"):
            local_calls += 1
        else:
            fallback_calls += 1
        estimated_cost_total += entry.get("estimated_cost", 0.0)

    avail = check_availability()
    budget = get_budget_status()
    return {
        "total_calls": len(_provider_history),
        "providers": providers,
        "local_calls": local_calls,
        "fallback_calls": fallback_calls,
        "estimated_cost_total": round(estimated_cost_total, 6),
        "last_provider": _last_provider[0] if _last_provider else None,
        "fallback_available": avail["openai"]["available"],
        "ollama_available": avail["ollama"]["available"],
        "vast_available": avail["vastai"]["available"],
        "budget": budget,
    }


def check_availability() -> Dict[str, Any]:
    """Check provider availability without exposing API keys."""
    from igris.core.chat_engine import check_ollama_available
    ollama_ok = check_ollama_available()
    openai_key = bool(CONFIG.fallback_llm.api_key)
    vast_key = bool(CONFIG.vastai.api_key)
    return {
        "ollama": {
            "available": ollama_ok,
            "provider": CONFIG.local_llm.provider,
            "model": CONFIG.local_llm.model,
            "cost_per_call": COST_ESTIMATES.get("local", 0.0),
        },
        "openai": {
            "available": openai_key,
            "provider": CONFIG.fallback_llm.provider,
            "model": CONFIG.fallback_llm.model,
            "key_present": openai_key,
            "cost_per_call": COST_ESTIMATES.get("fallback", 0.003),
        },
        "vastai": {
            "available": vast_key,
            "key_present": vast_key,
            "cost_per_call": COST_ESTIMATES.get("vastai", 0.01),
            "auto_provision": False,
        },
    }


def get_budget_config() -> Dict[str, Any]:
    """Return current budget configuration (no secrets)."""
    return dict(_budget_config)


def set_budget_config(
    max_session_cost: Optional[float] = None,
    warn_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Update budget configuration."""
    if max_session_cost is not None and max_session_cost > 0:
        _budget_config["max_session_cost"] = max_session_cost
    if warn_threshold is not None and 0 < warn_threshold <= 1.0:
        _budget_config["warn_threshold"] = warn_threshold
    return dict(_budget_config)


def get_budget_status() -> Dict[str, Any]:
    """Return current budget status with usage percentage."""
    total_cost = sum(e.get("estimated_cost", 0.0) for e in _provider_history)
    max_cost = _budget_config["max_session_cost"]
    usage_pct = (total_cost / max_cost * 100) if max_cost > 0 else 0
    warn_at = _budget_config["warn_threshold"] * 100
    return {
        "spent": round(total_cost, 6),
        "max_session_cost": max_cost,
        "usage_percent": round(usage_pct, 2),
        "warning": usage_pct >= warn_at,
        "exceeded": usage_pct >= 100,
    }


def estimate_route(
    task_type: str = "chat",
    complexity: str = "low",
) -> Dict[str, Any]:
    """Estimate which provider would be used and the cost, without executing."""
    avail = check_availability()
    # Simple routing logic: prefer local, fallback if complex
    if complexity in ("high",) and avail["openai"]["available"]:
        provider = "fallback"
        model = CONFIG.fallback_llm.model
        reason = "High complexity task, using fallback provider"
        est_cost = COST_ESTIMATES.get("fallback", 0.003)
    elif avail["ollama"]["available"]:
        provider = "local"
        model = CONFIG.local_llm.model
        reason = "Local provider available and sufficient"
        est_cost = COST_ESTIMATES.get("local", 0.0)
    elif avail["openai"]["available"]:
        provider = "fallback"
        model = CONFIG.fallback_llm.model
        reason = "Local unavailable, using fallback"
        est_cost = COST_ESTIMATES.get("fallback", 0.003)
    else:
        provider = "none"
        model = ""
        reason = "No providers available"
        est_cost = 0.0

    budget = get_budget_status()
    return {
        "recommended_provider": provider,
        "model": model,
        "reason": reason,
        "estimated_cost": est_cost,
        "budget_remaining": round(budget["max_session_cost"] - budget["spent"], 6),
        "would_exceed_budget": budget["spent"] + est_cost > budget["max_session_cost"],
        "availability": {k: v["available"] for k, v in avail.items()},
    }