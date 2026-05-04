"""
Provider routing logic.

The router determines which language model provider to use based on availability
and cost.  It uses the configuration to decide between the local provider
(Ollama), a fallback provider (OpenAI) or (in the future) Vast.ai.  The
`explain_routing` function returns a human‑readable explanation for the last
decision.
"""

from __future__ import annotations

from typing import Optional, Tuple

from igris.models.config import CONFIG


class Provider(str):
    LOCAL = "local"
    FALLBACK = "fallback"
    VASTAI = "vastai"


_last_provider: Optional[Tuple[str, str]] = None


def choose_provider(for_task: str = "chat") -> Tuple[str, str]:
    """Return the name and model of the chosen provider.

    Currently the logic is simplistic: always use the local provider.  If the
    fallback API key is missing the fallback provider will not be chosen.
    This function records the last choice for reporting.
    """
    global _last_provider
    # Always choose local provider for the MVP
    _last_provider = (Provider.LOCAL, CONFIG.local_llm.model)
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