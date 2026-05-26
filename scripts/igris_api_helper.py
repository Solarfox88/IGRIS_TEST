#!/usr/bin/env python3
"""
IGRIS API Helper — external advisory escalation script.

Called by the SelfRepairSupervisor when IGRIS_API_HELPER_COMMAND is configured
and allow_api_escalation=True.  Reads a sanitized escalation packet from stdin,
calls the configured helper model API, and returns structured JSON advice.

The output is ADVISORY ONLY.  The supervisor uses it as additional context for
repair planning — it never bypasses safety gates, tests, or CI.

Input (stdin): JSON object
  {
    "model":      str,   # e.g. "codex-mini-latest"
    "max_tokens": int,
    "packet":     dict   # sanitized escalation context from supervisor
  }

Output (stdout): JSON object
  {
    "ok":                            bool,
    "model":                         str,
    "api_helper_mode":               str,   # "codex_only"|"auto"
    "api_helper_provider":           str,   # "openai"|"anthropic"
    "api_helper_model_requested":    str,
    "api_helper_model_resolved":     str,
    "codex_only":                    bool,
    "summary":                       str,
    "diagnosis":                     str,
    "likely_supervisor_gap":         str,
    "suggested_repair_strategy":     str,
    "suggested_tests":               list[str],
    "risk":                          str,   # "low"|"medium"|"high"
    "risk_notes":                    list[str],
    "do_not_do":                     list[str],
    "confidence":                    float,  # 0.0-1.0
    "requires_human_or_codex_audit": bool,
    "must_not_complete_product_manually": bool,
    "estimated_cost_usd":            float
  }

On any error the script prints a safe JSON error object to stdout and exits 1.
Secrets are never printed or logged.

Environment variables:
  IGRIS_API_HELPER_MODE      "codex_only" | "auto" (default: "auto")
  IGRIS_API_HELPER_PROVIDER  "openai" | "anthropic" — forces provider in auto mode
  IGRIS_API_HELPER_MODEL     Required in codex_only mode; optional override in auto mode
  IGRIS_OPENAI_API_KEY       OpenAI key (preferred over OPENAI_API_KEY)
  OPENAI_API_KEY             OpenAI key fallback
  IGRIS_ANTHROPIC_API_KEY    Anthropic key (preferred over ANTHROPIC_API_KEY)
  ANTHROPIC_API_KEY          Anthropic key fallback
  IGRIS_HELPER_TIMEOUT       API call timeout seconds (default: 45)

Codex-only mode (IGRIS_API_HELPER_MODE=codex_only):
  - Provider must be OpenAI; Anthropic is rejected
  - IGRIS_API_HELPER_MODEL must be set explicitly — no fallback to gpt-4o-mini
  - Missing model → error_code=codex_not_configured
  - Missing key   → error_code=codex_not_configured
  - API call fail → error_code=codex_unavailable
  - No silent fallbacks to any other model or provider
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{3,}[A-Za-z0-9]{10,}", re.ASCII),
    re.compile(r"anthropic[_-]?api[_-]?key\s*[=:]\s*\S+", re.IGNORECASE),
    re.compile(r"openai[_-]?api[_-]?key\s*[=:]\s*\S+", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}", re.ASCII),
]


def _redact(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def _safe_error(msg: str, exit_code: int = 1, error_code: str = "") -> None:
    """Emit a safe JSON error to stdout and exit."""
    _mode = os.environ.get("IGRIS_API_HELPER_MODE", "auto").strip() or "auto"
    payload: Dict[str, Any] = {
        "ok": False,
        "model": "",
        "api_helper_mode": _mode,
        "api_helper_provider": "",
        "api_helper_model_requested": "",
        "api_helper_model_resolved": "",
        "codex_only": _mode == "codex_only",
        "summary": "",
        "diagnosis": _redact(str(msg)),
        "likely_supervisor_gap": "",
        "suggested_repair_strategy": "",
        "execution_plan": [],
        "acceptance_matrix": [],
        "suggested_tests": [],
        "risk": "unknown",
        "risk_notes": [],
        "do_not_do": [],
        "confidence": 0.0,
        "requires_human_or_codex_audit": True,
        "must_not_complete_product_manually": True,
        "estimated_cost_usd": 0.0,
        "error": _redact(str(msg)),
    }
    if error_code:
        payload["error_code"] = error_code
    print(json.dumps(payload))
    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------------------

def _resolve_mode() -> str:
    """Return normalised helper mode: 'codex_only' or 'auto'."""
    raw = os.environ.get("IGRIS_API_HELPER_MODE", "").strip().lower()
    if raw == "codex_only":
        return "codex_only"
    return "auto"


# ---------------------------------------------------------------------------
# API key resolution — never print the key
# ---------------------------------------------------------------------------

def _resolve_key() -> Tuple[str, str]:
    """Return (provider, key) for auto mode, or raise RuntimeError.

    If IGRIS_API_HELPER_PROVIDER is set, forces that provider.
    Otherwise priority: Anthropic -> OpenAI -> DeepSeek.
    """
    forced = os.environ.get("IGRIS_API_HELPER_PROVIDER", "").strip().lower()
    if forced == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if key:
            return "deepseek", key
        raise RuntimeError(
            "IGRIS_API_HELPER_PROVIDER=deepseek but no DEEPSEEK_API_KEY configured."
        )
    if forced == "anthropic":
        for var in ("IGRIS_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"):
            key = os.environ.get(var, "").strip()
            if key:
                return "anthropic", key
        raise RuntimeError(
            "IGRIS_API_HELPER_PROVIDER=anthropic but no ANTHROPIC_API_KEY configured."
        )
    if forced == "openai":
        for var in ("IGRIS_OPENAI_API_KEY", "OPENAI_API_KEY"):
            key = os.environ.get(var, "").strip()
            if key:
                return "openai", key
        raise RuntimeError(
            "IGRIS_API_HELPER_PROVIDER=openai but no OPENAI_API_KEY configured."
        )
    # Auto mode: Anthropic -> OpenAI -> DeepSeek
    for var in ("IGRIS_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"):
        key = os.environ.get(var, "").strip()
        if key:
            return "anthropic", key
    for var in ("IGRIS_OPENAI_API_KEY", "OPENAI_API_KEY"):
        key = os.environ.get(var, "").strip()
        if key:
            return "openai", key
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return "deepseek", key
    raise RuntimeError(
        "No API key configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY."
    )


def _resolve_key_codex_only() -> Tuple[str, str]:
    """Return ('openai', key) in codex_only mode, or raise with error_code.

    Only OpenAI keys are accepted. Anthropic keys are ignored.
    Raises RuntimeError with error_code embedded in message on failure.
    """
    for var in ("IGRIS_OPENAI_API_KEY", "OPENAI_API_KEY"):
        key = os.environ.get(var, "").strip()
        if key:
            return "openai", key
    raise RuntimeError(
        "codex_not_configured: No OpenAI API key found. "
        "Set OPENAI_API_KEY or IGRIS_OPENAI_API_KEY for codex_only mode. "
        "Anthropic keys are not accepted in codex_only mode."
    )


def _resolve_model(requested: str, provider: str) -> str:
    """Auto-mode model resolution.

    Use IGRIS_API_HELPER_MODEL override if set, otherwise use requested
    model or provider default (gpt-4o-mini / claude-haiku).
    """
    override = os.environ.get("IGRIS_API_HELPER_MODEL", "").strip()
    if override:
        return override
    if requested and requested not in ("gpt-5.4-mini", ""):
        return requested
    # Sensible defaults per provider
    return "claude-haiku-4-5-20251001" if provider == "anthropic" else "gpt-4o-mini"


def _resolve_model_codex_only(requested: str) -> str:
    """Codex-only model resolution.

    IGRIS_API_HELPER_MODEL is required AND must contain 'codex'.
    No fallback to gpt-4o-mini, claude-haiku, or any other model.
    Raises RuntimeError with error_code embedded in message on failure.
    """
    model = os.environ.get("IGRIS_API_HELPER_MODEL", "").strip()
    if not model:
        raise RuntimeError(
            "codex_not_configured: IGRIS_API_HELPER_MODEL is not set. "
            "In codex_only mode you must explicitly configure the Codex model name. "
            "No fallback to gpt-4o-mini or any other default is allowed."
        )
    if not _is_codex_model(model):
        raise RuntimeError(
            f"codex_not_configured: IGRIS_API_HELPER_MODEL={model!r} does not appear to be a "
            "Codex model (model name must contain 'codex'). In codex_only mode only Codex "
            "models are accepted."
        )
    return model


def _is_codex_model(model: str) -> bool:
    """Return True when the model name identifies a Codex model."""
    return "codex" in model.lower()


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_DECOMPOSITION_SYSTEM_PROMPT = """You are a mission decomposition assistant for IGRIS.
A mission was too large for the local model to complete in one reasoning pass.
Your job is to decompose it into 2-4 smaller sub-missions.

Output ONLY a valid JSON object with exactly these fields:
{
  "why_too_large": "<one sentence: root cause>",
  "sub_missions": [
    {
      "title": "<short title>",
      "goal": "<concrete goal>",
      "risk_level": "low|medium|high"
    }
  ],
  "first_sub_mission": "<title of first sub-mission to run>",
  "human_approval_required": false
}

No markdown. No explanation. Only the JSON."""

_SYSTEM_PROMPT = """You are an advisory assistant for IGRIS, an autonomous coding agent.
IGRIS's supervisor is blocked on a repair task and is asking for diagnostic advice.

Your role is ADVISORY ONLY. You must never:
- Claim to have executed code or tests
- Bypass any safety, test, or CI requirement
- Complete the product manually or generate final code
- Override the supervisor's authority

Respond ONLY with a valid JSON object containing exactly these fields:
{
  "ok": true,
  "summary": "<one sentence summary>",
  "diagnosis": "<what is likely wrong>",
  "likely_supervisor_gap": "<what the supervisor may be missing>",
  "suggested_repair_strategy": "<concrete next step for the supervisor>",
  "suggested_tests": ["<test 1>", "<test 2>"],
  "risk": "<low|medium|high>",
  "risk_notes": ["<risk note>"],
  "do_not_do": ["<thing to avoid>"],
  "confidence": 0.7,
  "requires_human_or_codex_audit": false,
  "must_not_complete_product_manually": true,
  "estimated_cost_usd": 0.001
}

Output ONLY the JSON. No markdown, no explanation outside the JSON."""


# ---------------------------------------------------------------------------
# Anthropic call
# ---------------------------------------------------------------------------

def _call_anthropic(key: str, model: str, max_tokens: int, context: str, timeout: int, system_prompt: str = _SYSTEM_PROMPT) -> Tuple[str, float]:
    try:
        import anthropic as _anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    client = _anthropic.Anthropic(api_key=key, timeout=float(timeout))
    msg = client.messages.create(
        model=model,
        max_tokens=max(64, min(max_tokens, 4096)),
        system=system_prompt,
        messages=[{"role": "user", "content": context}],
    )
    text = "".join(
        block.text for block in msg.content if hasattr(block, "text")
    )
    input_tokens = getattr(msg.usage, "input_tokens", 0)
    output_tokens = getattr(msg.usage, "output_tokens", 0)
    # Rough cost: $0.25/M input + $1.25/M output for Haiku
    cost = (input_tokens * 0.25 + output_tokens * 1.25) / 1_000_000
    return text, cost


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------

def _call_openai(key: str, model: str, max_tokens: int, context: str, timeout: int, system_prompt: str = _SYSTEM_PROMPT) -> Tuple[str, float]:
    try:
        import openai as _openai
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    client = _openai.OpenAI(api_key=key, timeout=float(timeout))
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ],
    }
    bounded_tokens = max(64, min(max_tokens, 4096))
    if model.lower().startswith("gpt-5"):
        kwargs["max_completion_tokens"] = bounded_tokens
    else:
        kwargs["max_tokens"] = bounded_tokens
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    input_tokens = getattr(usage, "prompt_tokens", 0)
    output_tokens = getattr(usage, "completion_tokens", 0)
    # Rough cost: $0.15/M input + $0.60/M output (gpt-4o-mini tier)
    cost = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
    return text, cost


def _call_openai_responses(key: str, model: str, max_tokens: int, context: str, timeout: int, system_prompt: str = _SYSTEM_PROMPT) -> Tuple[str, float]:
    """Call OpenAI /v1/responses (Responses API) — required for Codex models.

    Codex models (e.g. gpt-5.3-codex) are not chat models and reject
    /v1/chat/completions with "This is not a chat model".  The Responses API
    accepts them via POST /v1/responses with 'instructions' + 'input'.

    Uses urllib.request (stdlib) so there is no dependency on the openai SDK
    version supporting this endpoint.
    """
    _RESPONSES_URL = "https://api.openai.com/v1/responses"
    payload = json.dumps({
        "model": model,
        "instructions": system_prompt,
        "input": context,
        "max_output_tokens": max(64, min(max_tokens, 4096)),
    }).encode()
    req = urllib.request.Request(
        _RESPONSES_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode(errors="replace")[:300]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {exc.code}: {_redact(body)}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"URL error: {_redact(str(exc.reason))}")

    data = json.loads(raw)

    # Extract text: prefer top-level 'output_text', fall back to iterating output items
    text = str(data.get("output_text", ""))
    if not text:
        for item in data.get("output", []):
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text += part.get("text", "")
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if isinstance(part, dict) and part.get("type") in ("text", "output_text"):
                        text += part.get("text", "")

    usage = data.get("usage", {})
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    # Same rough rate as _call_openai
    cost = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
    return text, cost


def _call_deepseek(key: str, model: str, max_tokens: int, context: str, timeout: int, system_prompt: str = _SYSTEM_PROMPT) -> Tuple[str, float]:
    """Call DeepSeek via OpenAI-compatible /v1/chat/completions."""
    _DEEPSEEK_BASE = "https://api.deepseek.com/v1"
    payload = json.dumps({
        "model": model,
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": max(64, min(max_tokens, 8192)),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ],
    }).encode()
    req = urllib.request.Request(
        f"{_DEEPSEEK_BASE}/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode(errors="replace")[:300]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {exc.code}: {_redact(body)}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"URL error: {_redact(str(exc.reason))}")
    data = json.loads(raw)
    msg = data["choices"][0]["message"]
    text = str(msg.get("content") or "").strip()
    if not text:
        text = str(msg.get("reasoning_content") or "").strip()
    usage = data.get("usage", {})
    input_tokens = int(usage.get("prompt_tokens", 0))
    output_tokens = int(usage.get("completion_tokens", 0))
    # DeepSeek V4 Pro: $0.435/$0.87 per 1M
    cost = (input_tokens * 0.435 + output_tokens * 0.87) / 1_000_000
    return text, cost


# ---------------------------------------------------------------------------
# Parse and validate response
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = (
    "diagnosis",
    "likely_supervisor_gap",
    "suggested_repair_strategy",
    "suggested_tests",
    "risk",
    "requires_human_or_codex_audit",
    "must_not_complete_product_manually",
)


def _extract_first_json_object(text: str) -> Optional[str]:
    """Return the first complete JSON object in text using a brace counter.

    The greedy regex approach fails when the model appends extra text with
    braces after the JSON block — this handles nested objects and strings
    containing braces/quotes correctly.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _coerce_confidence(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip().lower()
        mapping = {
            "low": 0.25,
            "medium": 0.5,
            "moderate": 0.5,
            "high": 0.8,
            "very_high": 0.9,
            "very-high": 0.9,
        }
        if s in mapping:
            return mapping[s]
        try:
            return float(s)
        except ValueError:
            return 0.0
    return 0.0


def _quality_score_response(payload: Dict[str, Any]) -> float:
    """Heuristic quality score for advisory payload, 0.0-1.0."""
    strategy = str(payload.get("suggested_repair_strategy", "")).strip()
    diagnosis = str(payload.get("diagnosis", "")).strip()
    tests = payload.get("suggested_tests", [])
    conf = _coerce_confidence(payload.get("confidence", 0.0))
    has_tests = isinstance(tests, list) and len([t for t in tests if str(t).strip()]) >= 2
    has_detail = len(strategy) >= 80 and len(diagnosis) >= 80
    score = 0.35
    if has_detail:
        score += 0.35
    if has_tests:
        score += 0.20
    score += min(0.10, conf * 0.10)
    return min(1.0, score)


def _passes_quality_gate(payload: Dict[str, Any], floor: float) -> bool:
    tests = payload.get("suggested_tests", [])
    strategy = str(payload.get("suggested_repair_strategy", "")).strip()
    diagnosis = str(payload.get("diagnosis", "")).strip()
    tests_ok = isinstance(tests, list) and len([t for t in tests if str(t).strip()]) >= 2
    text_ok = len(strategy) >= 80 and len(diagnosis) >= 80
    return tests_ok and text_ok and (_quality_score_response(payload) >= floor)


def _call_deepseek_quality_pass(
    key: str,
    model: str,
    timeout: int,
    context: str,
    current_payload: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], float]:
    """Ask DeepSeek to improve an already parsed advisory JSON deterministically."""
    _DEEPSEEK_BASE = "https://api.deepseek.com/v1"
    improve_system = (
        "You are improving an advisory JSON for IGRIS. "
        "Return ONLY valid JSON object with the exact same schema keys as input. "
        "Make diagnosis and suggested_repair_strategy concrete and actionable. "
        "Provide at least 2 suggested_tests. Keep safety constraints."
    )
    improve_user = (
        "Context:\n"
        + context
        + "\n\nCurrent JSON to improve:\n"
        + json.dumps(current_payload, ensure_ascii=True)
    )
    payload = json.dumps({
        "model": model,
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": 1400,
        "messages": [
            {"role": "system", "content": improve_system},
            {"role": "user", "content": improve_user},
        ],
    }).encode()
    req = urllib.request.Request(
        f"{_DEEPSEEK_BASE}/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
        raw = json.loads(resp.read())
    msg = raw["choices"][0]["message"]
    text = str(msg.get("content") or "").strip() or str(msg.get("reasoning_content") or "").strip()
    parsed_text = _extract_first_json_object(text)
    if not parsed_text:
        return None, 0.0
    improved = json.loads(parsed_text)
    usage = raw.get("usage", {})
    input_tokens = int(usage.get("prompt_tokens", 0))
    output_tokens = int(usage.get("completion_tokens", 0))
    extra_cost = (input_tokens * 0.435 + output_tokens * 0.87) / 1_000_000
    return improved, extra_cost


def _parse_response(
    raw: str,
    model: str,
    cost: float,
    *,
    mode: str = "auto",
    provider: str = "",
    model_requested: str = "",
) -> Dict[str, Any]:
    text = _redact(raw.strip())
    extracted = _extract_first_json_object(text)
    _obs = {
        "api_helper_mode": mode,
        "api_helper_provider": provider,
        "api_helper_model_requested": model_requested,
        "api_helper_model_resolved": model,
        "codex_only": mode == "codex_only",
    }
    if not extracted:
        return {
            "ok": True,
            "model": model,
            **_obs,
            "error": "helper returned no JSON object",
            "summary": "Helper returned non-JSON output; using degraded advisory fallback.",
            "diagnosis": f"Raw helper output (truncated): {text[:300]}",
            "likely_supervisor_gap": "Output contract mismatch (expected JSON object).",
            "suggested_repair_strategy": "Retry with stricter JSON mode and preserve original failure context.",
            "execution_plan": [],
            "acceptance_matrix": [],
            "suggested_tests": [],
            "risk": "high",
            "risk_notes": ["Degraded parser fallback used due to non-JSON helper output."],
            "do_not_do": [],
            "confidence": 0.1,
            "requires_human_or_codex_audit": True,
            "must_not_complete_product_manually": True,
            "estimated_cost_usd": cost,
        }
    try:
        payload = json.loads(extracted)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "model": model,
            **_obs,
            "error": f"JSON parse error: {exc}",
            "summary": "",
            "diagnosis": "",
            "likely_supervisor_gap": "",
            "suggested_repair_strategy": "",
            "execution_plan": [],
            "acceptance_matrix": [],
            "suggested_tests": [],
            "risk": "unknown",
            "risk_notes": [],
            "do_not_do": [],
            "confidence": 0.0,
            "requires_human_or_codex_audit": True,
            "must_not_complete_product_manually": True,
            "estimated_cost_usd": cost,
        }
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    result: Dict[str, Any] = {
        "ok": True,
        "model": model,
        **_obs,
        "summary": str(payload.get("summary", "")),
        "diagnosis": str(payload.get("diagnosis", "")),
        "likely_supervisor_gap": str(payload.get("likely_supervisor_gap", "")),
        "suggested_repair_strategy": str(payload.get("suggested_repair_strategy", "")),
        "execution_plan": list(payload.get("execution_plan") or []),
        "acceptance_matrix": list(payload.get("acceptance_matrix") or []),
        "suggested_tests": list(payload.get("suggested_tests") or []),
        "risk": str(payload.get("risk", "unknown")),
        "risk_notes": list(payload.get("risk_notes") or []),
        "do_not_do": list(payload.get("do_not_do") or []),
        "confidence": _coerce_confidence(payload.get("confidence", 0.0)),
        "requires_human_or_codex_audit": bool(payload.get("requires_human_or_codex_audit", False)),
        "must_not_complete_product_manually": bool(payload.get("must_not_complete_product_manually", True)),
        "estimated_cost_usd": float(payload.get("estimated_cost_usd", cost)),
    }
    if missing:
        result["error"] = f"missing fields: {', '.join(missing)}"
        if not result["summary"]:
            result["summary"] = "Helper JSON parsed with missing fields; defaults applied."
        if not result["diagnosis"]:
            result["diagnosis"] = "Helper response omitted one or more required diagnostic fields."
        if not result["likely_supervisor_gap"]:
            result["likely_supervisor_gap"] = "Model contract compliance drift."
        if not result["suggested_repair_strategy"]:
            result["suggested_repair_strategy"] = "Retry once with stricter JSON instructions and validate schema."
        if not result["risk"] or result["risk"] == "unknown":
            result["risk"] = "medium"
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Read stdin
    try:
        raw_input = sys.stdin.read()
    except Exception as exc:
        _safe_error(f"failed to read stdin: {exc}")

    # Parse input
    try:
        data = json.loads(raw_input)
    except json.JSONDecodeError as exc:
        _safe_error(f"invalid JSON on stdin: {exc}")

    model_requested = str(data.get("model", "")).strip()
    max_tokens = int(data.get("max_tokens", 600))
    packet = data.get("packet", {})
    timeout = int(os.environ.get("IGRIS_HELPER_TIMEOUT", "45"))

    # Resolve mode — this determines key/model resolution strategy
    mode = _resolve_mode()
    is_codex_only = (mode == "codex_only")

    # Resolve provider and API key
    if is_codex_only:
        try:
            provider, api_key = _resolve_key_codex_only()
        except RuntimeError as exc:
            err = str(exc)
            error_code = "codex_not_configured" if "codex_not_configured" in err else "codex_unavailable"
            _safe_error(err, error_code=error_code)
    else:
        try:
            provider, api_key = _resolve_key()
        except RuntimeError as exc:
            _safe_error(str(exc))

    # Resolve model
    if is_codex_only:
        try:
            model = _resolve_model_codex_only(model_requested)
        except RuntimeError as exc:
            err = str(exc)
            error_code = "codex_not_configured" if "codex_not_configured" in err else "codex_unavailable"
            _safe_error(err, error_code=error_code)
    else:
        model = _resolve_model(model_requested, provider)

    # Detect decomposition task and build appropriate context
    is_decomposition = packet.get("task") == "decomposition"

    if is_decomposition:
        context_parts = [
            f"goal: {_redact(str(packet.get('goal', ''))[:500])}",
            f"signals: {packet.get('signals', {})}",
            f"run_id: {packet.get('run_id', '')}",
        ]
        system_prompt = _DECOMPOSITION_SYSTEM_PROMPT
    else:
        context_parts = [
            f"failure_class: {packet.get('failure_class', 'unknown')}",
            f"goal: {_redact(str(packet.get('goal', ''))[:500])}",
            f"repair_cycles_used: {packet.get('repair_cycles_used', 0)}",
            f"capability_signals: {packet.get('capability_signals', {})}",
        ]
        if packet.get("events"):
            recent = packet["events"][-5:]
            context_parts.append(
                "recent_events: " + json.dumps([
                    {k: _redact(str(v)) for k, v in e.items()
                     if k in ("phase", "status", "detail")}
                    for e in recent
                ])
            )
        system_prompt = _SYSTEM_PROMPT

    context = "\n".join(context_parts)

    # Call API — Codex models must use /v1/responses, not /v1/chat/completions
    try:
        if provider == "anthropic":
            raw_response, cost = _call_anthropic(api_key, model, max_tokens, context, timeout, system_prompt)
        elif provider == "deepseek":
            raw_response, cost = _call_deepseek(api_key, model, max_tokens, context, timeout, system_prompt)
        elif _is_codex_model(model) or model.lower().startswith("gpt-5"):
            raw_response, cost = _call_openai_responses(api_key, model, max_tokens, context, timeout, system_prompt)
        else:
            raw_response, cost = _call_openai(api_key, model, max_tokens, context, timeout, system_prompt)
    except Exception as exc:
        err_msg = f"API call failed: {_redact(str(exc))}"
        if is_codex_only:
            _safe_error(err_msg, error_code="codex_unavailable")
        else:
            _safe_error(err_msg)

    # Observability envelope — included in every response
    _obs = {
        "api_helper_mode": mode,
        "api_helper_provider": provider,
        "api_helper_model_requested": model_requested,
        "api_helper_model_resolved": model,
        "codex_only": is_codex_only,
    }

    # Handle decomposition response separately
    if is_decomposition:
        decomp: Dict[str, Any] = {}
        try:
            extracted_decomp = _extract_first_json_object(raw_response)
            if extracted_decomp:
                decomp = json.loads(extracted_decomp)
        except (json.JSONDecodeError, AttributeError):
            pass
        print(json.dumps({
            "ok": bool(decomp.get("why_too_large") and decomp.get("sub_missions")),
            "model": model,
            **_obs,
            "why_too_large": _redact(str(decomp.get("why_too_large", ""))),
            "sub_missions": [
                {k: _redact(str(v)) if isinstance(v, str) else v for k, v in s.items()}
                for s in (decomp.get("sub_missions") or [])
            ],
            "first_sub_mission": _redact(str(decomp.get("first_sub_mission", ""))),
            "human_approval_required": bool(decomp.get("human_approval_required", True)),
            "estimated_cost_usd": cost,
        }))
        sys.exit(0)

    result = _parse_response(
        raw_response,
        model,
        cost,
        mode=mode,
        provider=provider,
        model_requested=model_requested,
    )
    # DeepSeek quality boost: one deterministic refinement pass for weak outputs.
    if provider == "deepseek" and isinstance(result, dict):
        model_l = model.lower()
        is_flash = "flash" in model_l
        default_floor = "0.92" if is_flash else "0.94"
        default_passes = "2"
        floor_env = "IGRIS_DEEPSEEK_FLASH_QUALITY_FLOOR" if is_flash else "IGRIS_DEEPSEEK_PRO_QUALITY_FLOOR"
        passes_env = "IGRIS_DEEPSEEK_FLASH_QUALITY_PASSES" if is_flash else "IGRIS_DEEPSEEK_PRO_QUALITY_PASSES"
        # Priority: model-specific env > global env > model default.
        quality_floor = float(
            os.environ.get(
                floor_env,
                os.environ.get("IGRIS_DEEPSEEK_QUALITY_FLOOR", default_floor),
            )
        )
        max_passes = int(
            os.environ.get(
                passes_env,
                os.environ.get("IGRIS_DEEPSEEK_QUALITY_PASSES", default_passes),
            )
        )
        applied = False
        for _ in range(max(0, max_passes)):
            if _passes_quality_gate(result, quality_floor):
                break
            try:
                improved_payload, extra_cost = _call_deepseek_quality_pass(
                    api_key, model, timeout, context, result
                )
                if not isinstance(improved_payload, dict):
                    break
                improved = _parse_response(
                    json.dumps(improved_payload),
                    model,
                    float(result.get("estimated_cost_usd", cost)) + extra_cost,
                    mode=mode,
                    provider=provider,
                    model_requested=model_requested,
                )
                if _quality_score_response(improved) >= _quality_score_response(result):
                    result = improved
                    applied = True
            except Exception:
                break
        result["quality_boost_applied"] = applied
        result["quality_score"] = _quality_score_response(result)
        result["quality_gate_passed"] = _passes_quality_gate(result, quality_floor)
        if not result["quality_gate_passed"]:
            result["requires_human_or_codex_audit"] = True
            result["risk"] = "high"
            notes = list(result.get("risk_notes") or [])
            notes.append("DeepSeek quality gate not fully met after refinement passes.")
            result["risk_notes"] = notes
    print(json.dumps(result))
    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
