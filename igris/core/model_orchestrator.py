"""Model Orchestrator for IGRIS_GPT — Epic #58.

All LLM usage must go through this orchestrator. No component should
call OpenAI, DeepSeek, Anthropic, Ollama or any provider directly.

The orchestrator:
- Selects model/provider based on task type, role, risk, budget, context size
- Uses local models when sufficient
- Uses cheap cloud providers when convenient
- Uses strong models for hard debugging, architecture, security review
- Degrades honestly when no suitable model is available
- Records cost, latency, provider, fallback, outcome
- Supports any OpenAI-compatible provider
- Circuit breaker: marks providers unavailable after repeated failures
- Retry with exponential backoff before falling through to next provider

Profiles:
    deterministic          — no LLM, safety/policy/routing
    local_light            — chat, synthesis, simple classification (Ollama)
    local_coder            — code reasoning if hardware allows
    cheap_cloud_reasoning  — coding/reasoning economical (DeepSeek API etc.)
    strong_cloud_reasoning — hard debugging, architecture, critical review
    risk_reviewer          — risk analysis for medium/high/unknown commands
    embedding_memory       — semantic retrieval (future)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from igris.core.safety import redact_secrets
from igris.models.config import CONFIG


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

# States: CLOSED (normal), OPEN (failing, skip), HALF_OPEN (testing recovery)
_CB_CLOSED = "closed"
_CB_OPEN = "open"
_CB_HALF_OPEN = "half_open"

# Defaults
_CB_FAILURE_THRESHOLD = 3      # consecutive failures to trip
_CB_RECOVERY_TIMEOUT = 30.0    # seconds before trying again (half-open)
_CB_SUCCESS_THRESHOLD = 1      # successes in half-open to close

# Retry defaults
_RETRY_MAX_ATTEMPTS = 2        # retries per provider before moving to next
_RETRY_BASE_DELAY = 0.5        # seconds, doubles each retry


@dataclass
class CircuitBreakerState:
    """Per-provider circuit breaker state."""
    state: str = _CB_CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    success_count_half_open: int = 0
    last_error: str = ""

    # Configurable thresholds
    failure_threshold: int = _CB_FAILURE_THRESHOLD
    recovery_timeout: float = _CB_RECOVERY_TIMEOUT
    success_threshold: int = _CB_SUCCESS_THRESHOLD

    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == _CB_HALF_OPEN:
            self.success_count_half_open += 1
            if self.success_count_half_open >= self.success_threshold:
                self.state = _CB_CLOSED
                self.failure_count = 0
                self.success_count_half_open = 0
        elif self.state == _CB_CLOSED:
            self.failure_count = 0

    def record_failure(self, error: str = "") -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        self.last_error = error
        if self.state == _CB_HALF_OPEN:
            # Failed during recovery attempt — reopen
            self.state = _CB_OPEN
            self.success_count_half_open = 0
        elif self.failure_count >= self.failure_threshold:
            self.state = _CB_OPEN

    def is_available(self) -> bool:
        """Check if the provider should be tried."""
        if self.state == _CB_CLOSED:
            return True
        if self.state == _CB_HALF_OPEN:
            return True
        # OPEN — check if recovery timeout has elapsed
        elapsed = time.monotonic() - self.last_failure_time
        if elapsed >= self.recovery_timeout:
            self.state = _CB_HALF_OPEN
            self.success_count_half_open = 0
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_error": self.last_error,
            "available": self.is_available(),
        }


class CircuitBreakerRegistry:
    """Registry of per-provider circuit breakers."""

    def __init__(
        self,
        failure_threshold: int = _CB_FAILURE_THRESHOLD,
        recovery_timeout: float = _CB_RECOVERY_TIMEOUT,
        success_threshold: int = _CB_SUCCESS_THRESHOLD,
    ):
        self._breakers: Dict[str, CircuitBreakerState] = {}
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold

    def get(self, provider_name: str) -> CircuitBreakerState:
        if provider_name not in self._breakers:
            self._breakers[provider_name] = CircuitBreakerState(
                failure_threshold=self._failure_threshold,
                recovery_timeout=self._recovery_timeout,
                success_threshold=self._success_threshold,
            )
        return self._breakers[provider_name]

    def is_available(self, provider_name: str) -> bool:
        return self.get(provider_name).is_available()

    def record_success(self, provider_name: str) -> None:
        self.get(provider_name).record_success()

    def record_failure(self, provider_name: str, error: str = "") -> None:
        self.get(provider_name).record_failure(error)

    def reset(self, provider_name: str) -> None:
        """Manually reset a circuit breaker (e.g. after known recovery)."""
        if provider_name in self._breakers:
            self._breakers[provider_name] = CircuitBreakerState(
                failure_threshold=self._failure_threshold,
                recovery_timeout=self._recovery_timeout,
                success_threshold=self._success_threshold,
            )

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        self._breakers.clear()

    def status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all known breakers."""
        return {name: cb.to_dict() for name, cb in self._breakers.items()}


# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------

MODEL_PROFILES = (
    "deterministic",
    "local_light",
    "local_coder",
    "cheap_cloud_reasoning",
    "strong_cloud_reasoning",
    "endpoint_implementation",
    "risk_reviewer",
    "embedding_memory",
    # Cost-policy profiles: mini for helper-guided cheap execution, strong for gpt-4o escalation
    "mini_execution",
    "strong_execution",
)

# Task type → recommended profile mapping
TASK_PROFILE_MAP: Dict[str, str] = {
    "chat": "local_light",
    "classification": "local_light",
    "synthesis": "local_light",
    "code_reasoning": "cheap_cloud_reasoning",
    "code_generation": "cheap_cloud_reasoning",
    "patch_generation": "cheap_cloud_reasoning",
    "plan_generation": "cheap_cloud_reasoning",
    # Cloud-first profiles for tasks that repeatedly fail or require real implementations
    "semantic_repair": "endpoint_implementation",
    "endpoint_implementation": "endpoint_implementation",
    "stub_repair": "endpoint_implementation",
    "risk_review": "risk_reviewer",
    "architecture_review": "strong_cloud_reasoning",
    "security_review": "strong_cloud_reasoning",
    "hard_debugging": "strong_cloud_reasoning",
    "embedding": "embedding_memory",
    "safety_check": "deterministic",
    "policy_check": "deterministic",
    "routing": "deterministic",
    # Cost-policy task types driven by helper advice strategy
    "mini_execution": "mini_execution",
    "strong_execution": "strong_execution",
}


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    name: str = ""
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""  # env var name, never the actual key
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    max_context: int = 4096
    supports_json_mode: bool = False
    is_local: bool = False
    available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
            "max_context": self.max_context,
            "supports_json_mode": self.supports_json_mode,
            "is_local": self.is_local,
            "available": self.available,
        }


# Default provider configurations
def _build_default_providers() -> Dict[str, ProviderConfig]:
    """Build default provider configs from environment."""
    import os
    providers: Dict[str, ProviderConfig] = {}

    # Ollama (local)
    ollama_url = CONFIG.local_llm.base_url
    ollama_model = CONFIG.local_llm.model
    providers["ollama"] = ProviderConfig(
        name="ollama",
        base_url=str(ollama_url),
        model=str(ollama_model),
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        max_context=4096,
        is_local=True,
    )

    # OpenAI (mini) — model resolved from:
    #   1. IGRIS_EXECUTION_FALLBACK_MODEL env var (execution-specific override)
    #   2. CONFIG.fallback_llm.model (from FALLBACK_LLM_MODEL in .env)
    #   3. gpt-4o-mini (safe default)
    openai_model = (
        os.environ.get("IGRIS_EXECUTION_FALLBACK_MODEL")
        or CONFIG.fallback_llm.model
        or "gpt-4o-mini"
    )
    providers["openai"] = ProviderConfig(
        name="openai",
        base_url="https://api.openai.com/v1",
        model=str(openai_model),
        api_key_env="OPENAI_API_KEY",
        cost_per_1k_input=0.15,
        cost_per_1k_output=0.60,
        max_context=128000,
        supports_json_mode=True,
    )

    # OpenAI (strong) — used by strong_execution profile; model resolved from:
    #   1. IGRIS_EXECUTION_STRONG_MODEL env var
    #   2. gpt-4o (safe default for strong escalation)
    openai_strong_model = os.environ.get("IGRIS_EXECUTION_STRONG_MODEL") or "gpt-4o"
    providers["openai_strong"] = ProviderConfig(
        name="openai_strong",
        base_url="https://api.openai.com/v1",
        model=str(openai_strong_model),
        api_key_env="OPENAI_API_KEY",
        cost_per_1k_input=2.50,
        cost_per_1k_output=10.00,
        max_context=128000,
        supports_json_mode=True,
    )

    # DeepSeek (cheap cloud reasoning)
    providers["deepseek"] = ProviderConfig(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        cost_per_1k_input=0.014,
        cost_per_1k_output=0.028,
        max_context=64000,
        supports_json_mode=True,
    )

    # Anthropic (strong cloud)
    providers["anthropic"] = ProviderConfig(
        name="anthropic",
        base_url="https://api.anthropic.com/v1",
        model="claude-sonnet-4-20250514",
        api_key_env="ANTHROPIC_API_KEY",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        max_context=200000,
        supports_json_mode=True,
    )

    return providers


# ---------------------------------------------------------------------------
# Orchestrator result
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorResult:
    """Result from Model Orchestrator."""
    text: str = ""
    provider: str = ""
    model: str = ""
    profile: str = ""
    fallback_used: bool = False
    fallback_reason: str = ""
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0
    success: bool = False
    error: str = ""
    request_id: str = field(default_factory=lambda: f"req-{uuid.uuid4().hex[:8]}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": redact_secrets(self.text) if self.text else "",
            "provider": self.provider,
            "model": self.model,
            "profile": self.profile,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost": self.estimated_cost,
            "success": self.success,
            "error": self.error,
            "request_id": self.request_id,
        }


# ---------------------------------------------------------------------------
# Model Orchestrator
# ---------------------------------------------------------------------------

class ModelOrchestrator:
    """Central orchestrator for all LLM interactions.

    Usage:
        orchestrator = ModelOrchestrator()
        result = orchestrator.complete(
            task_type="code_reasoning",
            messages=[{"role": "user", "content": "..."}],
            system_prompt="...",
        )

    Features:
        - Provider chain with priority ordering per profile
        - Circuit breaker per provider (trips after N failures, recovers after timeout)
        - Retry with exponential backoff per provider attempt
        - Cost tracking and call history
    """

    def __init__(
        self,
        providers: Optional[Dict[str, ProviderConfig]] = None,
        circuit_breaker: Optional[CircuitBreakerRegistry] = None,
        retry_max_attempts: int = _RETRY_MAX_ATTEMPTS,
        retry_base_delay: float = _RETRY_BASE_DELAY,
    ):
        self.providers = providers or _build_default_providers()
        self._circuit_breaker = circuit_breaker or CircuitBreakerRegistry()
        self._retry_max_attempts = retry_max_attempts
        self._retry_base_delay = retry_base_delay
        self._history: List[Dict[str, Any]] = []
        self._total_cost: float = 0.0
        self._call_count: int = 0

    def complete(
        self,
        task_type: str,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        preferred_profile: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        json_mode: bool = False,
        timeout: float = 30.0,
    ) -> OrchestratorResult:
        """Send a completion request through the orchestrator.

        Selects the best provider based on task_type, availability, and budget.
        Falls back through provider chain on failure.
        """
        profile = preferred_profile or TASK_PROFILE_MAP.get(task_type, "local_light")

        # Deterministic profile — no LLM needed
        if profile == "deterministic":
            result = OrchestratorResult(
                text="",
                provider="deterministic",
                model="none",
                profile="deterministic",
                success=True,
            )
            self._record_call(result, task_type)
            return result

        # Build provider priority chain
        chain = self._get_provider_chain(profile)

        t0 = time.monotonic()
        last_error = ""

        for i, provider_name in enumerate(chain):
            provider = self.providers.get(provider_name)
            if not provider or not provider.available:
                continue

            if not self._check_provider_available(provider):
                continue

            # Circuit breaker check
            if not self._circuit_breaker.is_available(provider_name):
                last_error = (
                    f"{provider_name} circuit breaker open: "
                    f"{self._circuit_breaker.get(provider_name).last_error}"
                )
                continue

            # Retry loop with exponential backoff
            attempt_error = ""
            for attempt in range(self._retry_max_attempts + 1):
                if attempt > 0:
                    delay = self._retry_base_delay * (2 ** (attempt - 1))
                    time.sleep(delay)

                try:
                    result = self._call_provider(
                        provider, messages, system_prompt,
                        max_tokens, temperature, json_mode, timeout,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    result.profile = profile
                    result.latency_ms = elapsed
                    result.fallback_used = i > 0
                    if i > 0:
                        result.fallback_reason = last_error or "primary unavailable"

                    # Record success in circuit breaker
                    self._circuit_breaker.record_success(provider_name)
                    self._record_call(result, task_type)
                    return result

                except Exception as e:
                    attempt_error = str(e)
                    continue

            # All retries exhausted for this provider
            self._circuit_breaker.record_failure(provider_name, attempt_error)
            last_error = attempt_error

        # All providers failed — deterministic fallback
        elapsed = int((time.monotonic() - t0) * 1000)
        result = OrchestratorResult(
            text="",
            provider="deterministic_fallback",
            model="none",
            profile=profile,
            fallback_used=True,
            fallback_reason=last_error or "no provider available",
            latency_ms=elapsed,
            success=False,
            error="All providers unavailable",
        )
        self._record_call(result, task_type)
        return result

    def _get_provider_chain(self, profile: str) -> List[str]:
        """Get ordered provider chain for a profile."""
        chains: Dict[str, List[str]] = {
            "local_light": ["ollama", "deepseek", "openai"],
            "local_coder": ["ollama", "deepseek", "openai"],
            "cheap_cloud_reasoning": ["deepseek", "openai", "ollama"],
            "strong_cloud_reasoning": ["anthropic", "openai", "deepseek"],
            # Cloud-first: never starts with Ollama — used for endpoint/API implementation
            # and repeated semantic failures where local model repeatedly produces stubs.
            "endpoint_implementation": ["openai", "anthropic", "deepseek"],
            "risk_reviewer": ["deepseek", "openai", "ollama"],
            "embedding_memory": ["ollama", "openai"],
            # Cost-policy profiles for helper-guided execution strategy
            "mini_execution": ["openai", "deepseek"],
            "strong_execution": ["openai_strong", "openai", "deepseek"],
        }
        return chains.get(profile, ["ollama", "openai"])

    def _check_provider_available(self, provider: ProviderConfig) -> bool:
        """Check if a provider is configured and reachable."""
        import os

        if provider.is_local:
            return True  # Will fail at call time if not running

        if provider.api_key_env:
            key = os.environ.get(provider.api_key_env, "")
            if not key:
                return False

        return True

    def _call_provider(
        self,
        provider: ProviderConfig,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
        timeout: float,
    ) -> OrchestratorResult:
        """Call a specific provider."""
        if provider.is_local and provider.name == "ollama":
            return self._call_ollama(provider, messages, system_prompt, timeout)
        else:
            return self._call_openai_compatible(
                provider, messages, system_prompt,
                max_tokens, temperature, json_mode, timeout,
            )

    def _call_ollama(
        self,
        provider: ProviderConfig,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        timeout: float,
    ) -> OrchestratorResult:
        """Call Ollama API."""
        import urllib.request
        import urllib.error

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        url = f"{provider.base_url.rstrip('/')}/api/chat"
        payload = json.dumps({
            "model": provider.model,
            "messages": full_messages,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data.get("message", {}).get("content", "")
                return OrchestratorResult(
                    text=content,
                    provider="ollama",
                    model=provider.model,
                    success=bool(content),
                )
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                ValueError, TimeoutError, ConnectionError) as e:
            raise RuntimeError(f"Ollama call failed: {e}") from e

    def _call_openai_compatible(
        self,
        provider: ProviderConfig,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
        timeout: float,
    ) -> OrchestratorResult:
        """Call an OpenAI-compatible API (OpenAI, DeepSeek, etc.)."""
        import os
        import urllib.request
        import urllib.error

        api_key = os.environ.get(provider.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"No API key for {provider.name}")

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        url = f"{provider.base_url.rstrip('/')}/chat/completions"
        body: Dict[str, Any] = {
            "model": provider.model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode and provider.supports_json_mode:
            body["response_format"] = {"type": "json_object"}

        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                content = choices[0]["message"]["content"] if choices else ""
                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                cost = (
                    (input_tokens / 1000) * provider.cost_per_1k_input +
                    (output_tokens / 1000) * provider.cost_per_1k_output
                )
                return OrchestratorResult(
                    text=content,
                    provider=provider.name,
                    model=provider.model,
                    success=bool(content),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    estimated_cost=round(cost, 6),
                )
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                ValueError, TimeoutError, ConnectionError) as e:
            raise RuntimeError(f"{provider.name} call failed: {e}") from e

    def _record_call(self, result: OrchestratorResult, task_type: str) -> None:
        """Record a call in history for cost tracking."""
        self._call_count += 1
        self._total_cost += result.estimated_cost
        self._history.append({
            "request_id": result.request_id,
            "task_type": task_type,
            "provider": result.provider,
            "model": result.model,
            "profile": result.profile,
            "success": result.success,
            "latency_ms": result.latency_ms,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "estimated_cost": result.estimated_cost,
            "fallback_used": result.fallback_used,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    # -- Public accessors --

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        return {
            "total_cost": round(self._total_cost, 6),
            "call_count": self._call_count,
            "history_count": len(self._history),
        }

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent call history."""
        return list(reversed(self._history[-limit:]))

    def list_providers(self) -> List[Dict[str, Any]]:
        """List all configured providers (no secrets)."""
        return [p.to_dict() for p in self.providers.values()]

    def get_profiles(self) -> Dict[str, str]:
        """Get task type → profile mapping."""
        return dict(TASK_PROFILE_MAP)

    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get circuit breaker status for all providers."""
        return self._circuit_breaker.status()

    def reset_circuit_breaker(self, provider_name: Optional[str] = None) -> None:
        """Reset circuit breaker(s). If provider_name is None, reset all."""
        if provider_name:
            self._circuit_breaker.reset(provider_name)
        else:
            self._circuit_breaker.reset_all()
