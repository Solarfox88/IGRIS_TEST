"""LLM-based patch generation for IGRIS_GPT.

Generates patch proposals using LLM, but NEVER auto-applies them.
All generated patches go through the existing patch proposal workflow:

    LLM draft -> patch proposal -> validation -> diff preview -> gated apply

Safety rules:
- Output is always a proposal, never applied automatically
- Schema validated JSON output
- No secrets in generated content
- No binary files, .env, path traversal
- No unsafe file operations
- Deterministic fallback when LLM unavailable
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from igris.core.chat_engine import chat
from igris.core.safety import (
    redact_secrets,
    detect_secret_like_content,
    is_sensitive_filename,
    check_path_access,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCKED_PATHS = {".env", ".git", ".igris", "node_modules", "__pycache__"}
BLOCKED_EXTENSIONS = {
    ".pem", ".key", ".p12", ".pfx", ".pyc", ".pyo",
    ".so", ".dll", ".exe", ".bin", ".whl",
    ".png", ".jpg", ".jpeg", ".gif", ".ico",
}

MAX_FILES_PER_PATCH = 5
MAX_CONTENT_LENGTH = 50_000

# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

PATCH_GENERATION_PROMPT = """You are a code patch generator for IGRIS_GPT.

Given a task description, generate a patch proposal as JSON.

Output ONLY valid JSON — no markdown, no code fences.

Schema:
{
  "files": [
    {
      "path": "relative/path/to/file.py",
      "action": "create" | "modify",
      "after": "full file content after change",
      "reason": "why this change is needed"
    }
  ],
  "description": "overall description of the patch",
  "risk": "low" | "medium" | "high"
}

Rules:
- Generate ONLY safe, readable code
- No secrets, API keys, or credentials in generated content
- No .env files, no binary files
- No path traversal (no ../)
- Keep patches minimal and focused
- Each file must have valid content
- Maximum 5 files per patch
- action is "create" for new files, "modify" for existing ones
"""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_path(path: str) -> Optional[str]:
    """Validate a file path. Returns error message or None."""
    if not path:
        return "Empty file path"
    if ".." in path:
        return f"Path traversal detected: {path}"
    parts = path.split("/")
    for part in parts:
        if part in BLOCKED_PATHS:
            return f"Blocked directory: {part}"
    ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
    if ext in BLOCKED_EXTENSIONS:
        return f"Blocked file extension: {ext}"
    if is_sensitive_filename(path):
        return f"Sensitive filename: {path}"
    return None


def _validate_content(content: str) -> Optional[str]:
    """Validate file content. Returns error message or None."""
    if not content:
        return None
    if len(content) > MAX_CONTENT_LENGTH:
        return f"Content too large: {len(content)} chars (max {MAX_CONTENT_LENGTH})"
    has_secrets = detect_secret_like_content(content)
    if has_secrets:
        return "Secret-like content detected"
    return None


def validate_patch_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate LLM-generated patch output.

    Returns dict with 'valid', 'errors', and cleaned 'files'.
    """
    errors: List[str] = []
    clean_files: List[Dict[str, Any]] = []

    if not isinstance(data, dict):
        return {"valid": False, "errors": ["Output is not a JSON object"], "files": []}

    files = data.get("files", [])
    if not isinstance(files, list):
        return {"valid": False, "errors": ["'files' is not an array"], "files": []}

    if len(files) == 0:
        return {"valid": False, "errors": ["No files in patch"], "files": []}

    if len(files) > MAX_FILES_PER_PATCH:
        errors.append(f"Too many files: {len(files)} (max {MAX_FILES_PER_PATCH})")
        files = files[:MAX_FILES_PER_PATCH]

    for i, f in enumerate(files):
        if not isinstance(f, dict):
            errors.append(f"File entry {i} is not an object")
            continue

        path = f.get("path", "")
        action = f.get("action", "create")
        after = f.get("after", "")
        reason = f.get("reason", "")

        path_err = _validate_path(path)
        if path_err:
            errors.append(f"File {i} ({path}): {path_err}")
            continue

        content_err = _validate_content(after)
        if content_err:
            errors.append(f"File {i} ({path}): {content_err}")
            continue

        if action not in ("create", "modify"):
            errors.append(f"File {i} ({path}): invalid action '{action}'")
            continue

        clean_files.append({
            "path": path,
            "action": action,
            "after": redact_secrets(after) if after else "",
            "reason": redact_secrets(reason) if reason else "",
        })

    return {
        "valid": len(errors) == 0 and len(clean_files) > 0,
        "errors": errors,
        "files": clean_files,
    }


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------


def _deterministic_patch(task_title: str, task_description: str) -> Dict[str, Any]:
    """Generate a placeholder patch proposal without LLM."""
    return {
        "files": [],
        "description": f"Deterministic placeholder for: {task_title}",
        "risk": "low",
        "generated_by": "deterministic",
        "note": "LLM unavailable — no patch generated. Create manually via patch proposal workflow.",
    }


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response text."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return None
    return None


def generate_patch(
    task_title: str,
    task_description: str = "",
    context: str = "",
) -> Dict[str, Any]:
    """Generate a patch proposal using LLM.

    Returns a proposal dict — NEVER applies the patch.
    Falls back to deterministic if LLM fails.
    """
    t0 = time.monotonic()

    prompt = f"Task: {task_title}\n"
    if task_description:
        prompt += f"Description: {task_description}\n"
    if context:
        prompt += f"Context:\n{context}\n"
    prompt += "\nGenerate a minimal, safe patch proposal."

    try:
        response = chat(message=prompt, system_prompt=PATCH_GENERATION_PROMPT)
        text = response.get("text", "")

        data = _extract_json(text)
        if data is None:
            result = _deterministic_patch(task_title, task_description)
            result["fallback_reason"] = "Failed to parse LLM JSON output"
            result["latency_ms"] = int((time.monotonic() - t0) * 1000)
            return result

        validation = validate_patch_output(data)
        latency_ms = int((time.monotonic() - t0) * 1000)

        if not validation["valid"]:
            result = _deterministic_patch(task_title, task_description)
            result["fallback_reason"] = f"Validation failed: {'; '.join(validation['errors'])}"
            result["latency_ms"] = latency_ms
            return result

        return {
            "files": validation["files"],
            "description": redact_secrets(data.get("description", task_title)),
            "risk": data.get("risk", "medium") if data.get("risk") in ("low", "medium", "high") else "medium",
            "generated_by": "llm",
            "provider": response.get("provider", "unknown"),
            "model": response.get("model", "unknown"),
            "latency_ms": latency_ms,
            "proposal_only": True,
        }

    except Exception:
        result = _deterministic_patch(task_title, task_description)
        result["fallback_reason"] = "LLM call failed"
        result["latency_ms"] = int((time.monotonic() - t0) * 1000)
        return result
