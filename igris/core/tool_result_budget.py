"""tool_result_budget — 16 KB byte-cap enforcement for tool outputs.

Truncates tool result content that exceeds a configurable byte limit before
it is injected into the LLM context, preventing context explosion from large
file reads, test outputs, or search results.

Usage:
    from igris.core.tool_result_budget import apply_tool_result_budget

    content, outcome = apply_tool_result_budget(raw_output)
    if outcome.truncated:
        log.debug("truncated %d → %d bytes", outcome.original_bytes, outcome.final_bytes)

Config (optional, .igris/context_config.json):
    {"tool_result_budget_bytes": 16384}   # default
    {"tool_result_budget_bytes": 0}       # disable truncation
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_BUDGET_BYTES: int = 16 * 1024   # 16 KB
TRAILER_RESERVED: int = 256             # bytes reserved for the truncation marker


@dataclass
class BudgetOutcome:
    """Result of a budget application."""
    original_bytes: int
    final_bytes: int
    truncated: bool


def apply_tool_result_budget(
    content: str,
    budget_bytes: int = DEFAULT_BUDGET_BYTES,
) -> tuple[str, BudgetOutcome]:
    """Truncate *content* to *budget_bytes* at a valid UTF-8 boundary.

    Rules:
    - If ``budget_bytes == 0``: truncation disabled, content returned as-is.
    - If ``len(content.encode()) <= budget_bytes``: returned unchanged.
    - Otherwise: truncated to ``budget_bytes - TRAILER_RESERVED`` bytes
      (decoded safely at a UTF-8 boundary), then a marker appended:
      ``[… N bytes truncated by tool_result_budget …]``

    Returns:
        (final_content, BudgetOutcome)
    """
    raw = content.encode("utf-8")
    original_bytes = len(raw)

    if budget_bytes == 0 or original_bytes <= budget_bytes:
        return content, BudgetOutcome(original_bytes, original_bytes, False)

    cap = max(0, budget_bytes - TRAILER_RESERVED)
    truncated_raw = raw[:cap]
    # Decode at a valid UTF-8 boundary — ignore incomplete multibyte sequence
    truncated_str = truncated_raw.decode("utf-8", errors="ignore")
    removed = original_bytes - len(truncated_raw)
    trailer = f"\n[… {removed} bytes truncated by tool_result_budget …]"
    final = truncated_str + trailer
    return final, BudgetOutcome(original_bytes, len(final.encode("utf-8")), True)
