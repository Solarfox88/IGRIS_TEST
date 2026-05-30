"""Context Manager for IGRIS_GPT — Epic #60.

Decides what the LLM sees at each reasoning step. Builds a token-budget-aware
context packet containing:

    - Mission/goal description
    - Relevant file contents (scored by relevance)
    - Recent actions and their results
    - Recent errors and test output (summarized)
    - Memory/lessons retrieved for the current task
    - World state summary

The Context Manager does NOT call LLM providers directly. It produces a
context packet that is passed to the Model Orchestrator.

Key features:
    - Token budget management (configurable per profile)
    - File relevance scoring based on mission keywords
    - History condensation for long-running missions
    - Secret redaction on all context output
    - Graceful degradation when components are unavailable
    - Role-specific context filtering
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from igris.core.safety import redact_secrets


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default token budgets per profile (approximate char count, ~4 chars/token)
TOKEN_BUDGETS: Dict[str, int] = {
    "local_light": 8000,       # ~2K tokens
    "local_coder": 16000,      # ~4K tokens
    "cheap_cloud_reasoning": 64000,   # ~16K tokens
    "strong_cloud_reasoning": 200000,  # ~50K tokens
    "risk_reviewer": 16000,    # ~4K tokens
    "default": 16000,
}

# Maximum items in each context section
MAX_RECENT_ACTIONS = 10
MAX_RECENT_ERRORS = 5
MAX_MEMORY_ITEMS = 10
MAX_FILE_SNIPPETS = 8

# Chars reserved for system prompt + schema + overhead
RESERVED_CHARS = 4000


# ---------------------------------------------------------------------------
# Context packet
# ---------------------------------------------------------------------------

@dataclass
class ContextPacket:
    """A structured context packet for the Model Orchestrator.

    Built by the Context Manager, consumed by the Reasoning Loop.
    """

    mission_context: str = ""
    state_context: str = ""
    file_context: str = ""
    recent_actions: str = ""
    error_context: str = ""
    memory_context: str = ""
    role: str = "coder"
    budget_chars: int = 16000
    used_chars: int = 0
    truncated_sections: List[str] = field(default_factory=list)
    build_time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_context": redact_secrets(self.mission_context),
            "state_context": redact_secrets(self.state_context),
            "file_context": redact_secrets(self.file_context),
            "recent_actions": redact_secrets(self.recent_actions),
            "error_context": redact_secrets(self.error_context),
            "memory_context": redact_secrets(self.memory_context),
            "role": self.role,
            "budget_chars": self.budget_chars,
            "used_chars": self.used_chars,
            "truncated_sections": self.truncated_sections,
            "build_time_ms": self.build_time_ms,
        }

    @property
    def total_chars(self) -> int:
        return (
            len(self.mission_context) +
            len(self.state_context) +
            len(self.file_context) +
            len(self.recent_actions) +
            len(self.error_context) +
            len(self.memory_context)
        )


# ---------------------------------------------------------------------------
# File relevance scoring
# ---------------------------------------------------------------------------

@dataclass
class ScoredFile:
    """A file with its relevance score."""
    path: str = ""
    score: float = 0.0
    snippet: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "score": self.score,
            "snippet": redact_secrets(self.snippet),
            "reason": self.reason,
        }


def score_file_relevance(
    file_path: str,
    keywords: List[str],
    recent_files: List[str],
    error_files: List[str],
) -> float:
    """Score a file's relevance to the current task.

    Scoring factors:
    - Keyword match in path (+0.3 per keyword)
    - Recently accessed (+0.2)
    - Mentioned in errors (+0.4)
    - Test file if testing task (+0.2)
    - Config/entry file bonus (+0.1)
    """
    score = 0.0
    path_lower = file_path.lower()

    for kw in keywords:
        if kw.lower() in path_lower:
            score += 0.3

    if file_path in recent_files:
        score += 0.2

    if file_path in error_files:
        score += 0.4

    # Bonus for common entry points
    entry_names = {"server.py", "main.py", "app.py", "__init__.py",
                   "conftest.py", "setup.py", "pyproject.toml"}
    for name in entry_names:
        if path_lower.endswith(name):
            score += 0.1
            break

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# History condenser
# ---------------------------------------------------------------------------

def condense_actions(
    actions: List[Dict[str, Any]],
    max_items: int = MAX_RECENT_ACTIONS,
) -> str:
    """Condense action history into a compact summary.

    Keeps the most recent actions in detail, summarizes older ones.
    """
    if not actions:
        return "No recent actions."

    if len(actions) <= max_items:
        lines = []
        for i, act in enumerate(actions):
            step = act.get("step", i + 1)
            action_type = act.get("action_type", "unknown")
            outcome = act.get("outcome", "unknown")
            reason = act.get("reason", "")
            line = f"Step {step}: {action_type} → {outcome}"
            if reason:
                line += f" ({reason[:80]})"
            lines.append(line)
        return "\n".join(lines)

    # Summarize old actions, detail recent ones
    old_count = len(actions) - max_items
    old_summary = f"[{old_count} earlier actions summarized: "
    old_types = {}
    for act in actions[:old_count]:
        at = act.get("action_type", "unknown")
        old_types[at] = old_types.get(at, 0) + 1
    old_summary += ", ".join(f"{v}x {k}" for k, v in old_types.items())
    old_summary += "]\n"

    recent_lines = []
    for i, act in enumerate(actions[old_count:]):
        step = act.get("step", old_count + i + 1)
        action_type = act.get("action_type", "unknown")
        outcome = act.get("outcome", "unknown")
        reason = act.get("reason", "")
        line = f"Step {step}: {action_type} → {outcome}"
        if reason:
            line += f" ({reason[:80]})"
        recent_lines.append(line)

    return old_summary + "\n".join(recent_lines)


def summarize_errors(
    errors: List[Dict[str, Any]],
    max_items: int = MAX_RECENT_ERRORS,
) -> str:
    """Summarize recent errors/test failures.

    Extracts the key information: file, line, message, type.
    """
    if not errors:
        return "No recent errors."

    lines = []
    for err in errors[-max_items:]:
        err_type = err.get("type", "error")
        message = err.get("message", "")
        file_path = err.get("file", "")
        line_num = err.get("line", "")
        loc = f"{file_path}:{line_num}" if file_path else ""
        line = f"[{err_type}] {loc}: {message[:200]}" if loc else f"[{err_type}] {message[:200]}"
        lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Context Manager
# ---------------------------------------------------------------------------

class ContextManager:
    """Builds token-budget-aware context packets for the reasoning loop.

    Usage:
        ctx = ContextManager(project_root="/path/to/repo")
        packet = ctx.build_context(
            goal="Add /api/ping endpoint with tests",
            role="coder",
            profile="cheap_cloud_reasoning",
        )
    """

    def __init__(self, project_root: Optional[str] = None):
        import os
        self.project_root = project_root or os.environ.get("PROJECT_ROOT", ".")

    def build_context(
        self,
        goal: str = "",
        role: str = "coder",
        profile: str = "default",
        mission_id: str = "",
        mission_status: str = "",
        world_state: Optional[Dict[str, Any]] = None,
        recent_actions: Optional[List[Dict[str, Any]]] = None,
        recent_errors: Optional[List[Dict[str, Any]]] = None,
        memory_items: Optional[List[Dict[str, Any]]] = None,
        relevant_files: Optional[List[str]] = None,
        file_snippets: Optional[Dict[str, str]] = None,
        keywords: Optional[List[str]] = None,
    ) -> ContextPacket:
        """Build a complete context packet for the reasoning loop.

        Args:
            goal: Current mission/task goal
            role: Agent role (from Agent Registry)
            profile: Model profile (determines token budget)
            mission_id: Current mission ID
            mission_status: Current mission status
            world_state: Current world state dict
            recent_actions: List of recent action dicts
            recent_errors: List of recent error dicts
            memory_items: Retrieved memory/lessons
            relevant_files: List of file paths to include
            file_snippets: Pre-loaded file contents {path: content}
            keywords: Keywords for file relevance scoring

        Returns:
            ContextPacket ready for Model Orchestrator
        """
        t0 = time.monotonic()

        budget = TOKEN_BUDGETS.get(profile, TOKEN_BUDGETS["default"])
        available = budget - RESERVED_CHARS
        packet = ContextPacket(role=role, budget_chars=budget)
        truncated: List[str] = []

        # Issue #524 — load section budget multipliers from weighter (best-effort)
        _section_multipliers: Dict[str, float] = {}
        try:
            from igris.core.context_section_weighter import ContextSectionWeighter
            _weighter = ContextSectionWeighter(self.project_root or ".")
            _section_multipliers = _weighter.get_budget_multipliers()
        except Exception:
            pass

        def _weighted_budget(base: int, section: str) -> int:
            return max(0, int(base * _section_multipliers.get(section, 1.0)))

        # 1. Mission context (always included, highest priority — not weighted)
        mission_text = self._build_mission_context(goal, mission_id, mission_status)
        packet.mission_context = self._fit(mission_text, available, "mission")
        available -= len(packet.mission_context)
        if len(mission_text) > len(packet.mission_context):
            truncated.append("mission_context")

        # 2. Error context (high priority — errors drive next action)
        error_text = summarize_errors(recent_errors or [])
        error_budget = _weighted_budget(min(available // 4, 4000), "error_context")
        packet.error_context = self._fit(error_text, error_budget, "errors")
        available -= len(packet.error_context)
        if len(error_text) > len(packet.error_context):
            truncated.append("error_context")

        # 3. Recent actions (important for avoiding loops)
        action_text = condense_actions(recent_actions or [])
        action_budget = _weighted_budget(min(available // 4, 4000), "recent_actions")
        packet.recent_actions = self._fit(action_text, action_budget, "actions")
        available -= len(packet.recent_actions)
        if len(action_text) > len(packet.recent_actions):
            truncated.append("recent_actions")

        # 4. State context
        state_text = self._build_state_context(world_state or {})
        state_budget = _weighted_budget(min(available // 5, 2000), "state_context")
        packet.state_context = self._fit(state_text, state_budget, "state")
        available -= len(packet.state_context)
        if len(state_text) > len(packet.state_context):
            truncated.append("state_context")

        # 5. Memory context
        graph_items = list(memory_items or [])
        try:
            from igris.core.memory_graph import MemoryGraph
            mg = MemoryGraph(self.project_root or ".")
            graph_items.extend(mg.get_lessons_for_goal(goal, limit=5))
            graph_items.extend(mg.query_by_intent(goal, node_type="project_fact", limit=3))
            recipe = mg.get_command_recipe(goal)
            if recipe:
                graph_items.append(recipe)
        except Exception:
            pass
        memory_text = self._build_memory_context(graph_items)
        memory_budget = _weighted_budget(min(available // 4, 3000), "memory_context")
        packet.memory_context = self._fit(memory_text, memory_budget, "memory")
        available -= len(packet.memory_context)
        if len(memory_text) > len(packet.memory_context):
            truncated.append("memory_context")

        # 6. File context (uses remaining budget)
        file_text = self._build_file_context(
            relevant_files=relevant_files or [],
            file_snippets=file_snippets or {},
            keywords=keywords or [],
            recent_files=[a.get("file", "") for a in (recent_actions or []) if a.get("file")],
            error_files=[e.get("file", "") for e in (recent_errors or []) if e.get("file")],
            budget=max(available, 0),
        )
        packet.file_context = self._fit(file_text, max(available, 0), "files")
        if len(file_text) > len(packet.file_context) or getattr(self, "_file_context_truncated", False):
            truncated.append("file_context")

        packet.truncated_sections = truncated
        packet.used_chars = packet.total_chars
        packet.build_time_ms = int((time.monotonic() - t0) * 1000)

        return packet

    def _build_mission_context(
        self,
        goal: str,
        mission_id: str,
        mission_status: str,
    ) -> str:
        """Build mission context string."""
        parts = []
        if goal:
            parts.append(f"Goal: {goal}")
        if mission_id:
            parts.append(f"Mission: {mission_id}")
        if mission_status:
            parts.append(f"Status: {mission_status}")
        return "\n".join(parts) if parts else "No active mission."

    def _build_state_context(self, world_state: Dict[str, Any]) -> str:
        """Build world state context string with operative directives (#1041).

        High-signal keys (discovered_files, search_matched_files, MBOP intake,
        anti_repeat) are rendered as explicit instructions. Remaining keys are
        condensed to avoid token overflow on local profiles (cap: 2500 chars).
        """
        if not world_state:
            return "No state information available."

        sections: List[str] = []

        # --- MBOP intake block (highest priority — tells the model WHAT and WHERE) ---
        mbop_parts: List[str] = []
        mbop_what = str(world_state.get("mbop_what", "") or "").strip()
        mbop_where = str(world_state.get("mbop_where", "") or "").strip()
        mbop_why = str(world_state.get("mbop_why", "") or "").strip()
        mbop_acs = world_state.get("mbop_acceptance_criteria") or []
        if mbop_what:
            mbop_parts.append(f"  WHAT: {mbop_what[:200]}")
        if mbop_where:
            mbop_parts.append(f"  WHERE (target file/module): {mbop_where[:200]}")
        if mbop_why:
            mbop_parts.append(f"  WHY: {mbop_why[:200]}")
        if mbop_acs:
            ac_str = "; ".join(str(a)[:100] for a in mbop_acs[:5])
            mbop_parts.append(f"  ACCEPTANCE CRITERIA: {ac_str}")
        if mbop_parts:
            sections.append("## MISSION INTAKE\n" + "\n".join(mbop_parts))

        # --- Anti-repeat warning ---
        if world_state.get("anti_repeat_triggered"):
            diag = str(world_state.get("anti_repeat_diagnosis", "") or "")[:200]
            sections.append(
                f"⚠ ANTI-REPEAT GUARD TRIGGERED: {diag}\n"
                "→ Do NOT repeat the same find/search. Read or modify the files you found."
            )

        # --- Discovered files directive ---
        discovered = world_state.get("discovered_files")
        if discovered:
            files_str = ", ".join(str(f) for f in (discovered[:10] if isinstance(discovered, list) else [str(discovered)]))
            sections.append(
                f"DISCOVERED FILES (→ read or modify these next, do NOT search again):\n  {files_str}"
            )

        # --- Search matches directive ---
        search_matches = world_state.get("search_matched_files")
        if search_matches and isinstance(search_matches, list) and search_matches:
            matches_str = ", ".join(str(f) for f in search_matches[:10])
            sections.append(
                f"SEARCH MATCHES (→ read these files, do NOT search again):\n  {matches_str}"
            )

        # --- Last tool result summary ---
        last_tool = world_state.get("last_tool_result")
        if last_tool and isinstance(last_tool, dict):
            action_type = last_tool.get("action_type", "")
            data = last_tool.get("data")
            if action_type and data is not None:
                data_preview = str(data)[:150]
                sections.append(
                    f"LAST TOOL RESULT ({action_type}): {data_preview}\n"
                    "→ Use this result. Do NOT repeat the same action."
                )

        # --- Condensed remaining state (skip MBOP and directive keys already shown) ---
        _skip_keys = {
            "discovered_files", "search_matched_files", "last_tool_result",
            "anti_repeat_triggered", "anti_repeat_diagnosis", "anti_repeat_retryable",
            "mbop_what", "mbop_where", "mbop_why", "mbop_acceptance_criteria",
        }
        # Also skip tool_result_history when at budget risk (keep max 2 recent)
        _tool_history = world_state.get("tool_result_history")
        compact_lines: List[str] = []
        for k, v in world_state.items():
            if k in _skip_keys or k == "tool_result_history":
                continue
            val_str = str(v)[:120]
            compact_lines.append(f"{k}: {val_str}")

        # Add abbreviated tool history (2 entries max to save tokens)
        if _tool_history and isinstance(_tool_history, list) and len(_tool_history) > 0:
            recent = _tool_history[-2:]
            hist_parts = []
            for entry in recent:
                if isinstance(entry, dict):
                    at = entry.get("action_type", "?")
                    d = str(entry.get("data", ""))[:60]
                    hist_parts.append(f"{at}:{d}")
            if hist_parts:
                compact_lines.append(f"tool_result_history (last 2): {' | '.join(hist_parts)}")

        if compact_lines:
            sections.append("STATE:\n" + "\n".join(compact_lines))

        result = "\n\n".join(sections) if sections else "No state information available."

        # Hard cap for local profiles: 2500 chars
        if len(result) > 2500:
            result = result[:2480] + "\n... [state truncated]"

        return result

    def _build_memory_context(self, memory_items: List[Dict[str, Any]]) -> str:
        """Build memory context from retrieved items."""
        if not memory_items:
            return "No relevant memory."

        lines = []
        for item in memory_items[-MAX_MEMORY_ITEMS:]:
            event_type = item.get("event_type", "lesson")
            content = item.get("content", item.get("description", ""))
            if content:
                lines.append(f"[{event_type}] {str(content)[:200]}")
        return "\n".join(lines) if lines else "No relevant memory."

    def _build_file_context(
        self,
        relevant_files: List[str],
        file_snippets: Dict[str, str],
        keywords: List[str],
        recent_files: List[str],
        error_files: List[str],
        budget: int,
    ) -> str:
        """Build file context within budget.

        Scores files by relevance and includes the most relevant
        ones within the budget. Sets ``self._file_context_truncated``
        when content is cut.
        """
        self._file_context_truncated = False

        if not relevant_files and not file_snippets:
            return "No files loaded."

        # Score and sort files
        scored: List[ScoredFile] = []
        all_paths = set(relevant_files) | set(file_snippets.keys())

        for path in all_paths:
            s = score_file_relevance(path, keywords, recent_files, error_files)
            snippet = file_snippets.get(path, "")
            scored.append(ScoredFile(
                path=path,
                score=s,
                snippet=snippet,
                reason="keyword" if any(k.lower() in path.lower() for k in keywords) else "related",
            ))

        # Sort by score descending
        scored.sort(key=lambda x: x.score, reverse=True)

        # Build context within budget
        parts = []
        used = 0
        included = 0
        total_content_size = sum(len(sf.snippet) for sf in scored if sf.snippet)

        for sf in scored[:MAX_FILE_SNIPPETS]:
            if sf.snippet:
                entry = f"--- {sf.path} ---\n{redact_secrets(sf.snippet)}\n"
            else:
                entry = f"--- {sf.path} (no content loaded) ---\n"

            if used + len(entry) > budget:
                self._file_context_truncated = True
                # Try to truncate the snippet
                remaining = budget - used - len(f"--- {sf.path} ---\n\n") - 20
                if remaining > 100 and sf.snippet:
                    truncated_content = redact_secrets(sf.snippet[:remaining])
                    entry = f"--- {sf.path} ---\n{truncated_content}\n... [truncated]\n"
                else:
                    break

            parts.append(entry)
            used += len(entry)
            included += 1

        if included < len(scored):
            self._file_context_truncated = True

        if not parts:
            return "No files loaded."

        return "\n".join(parts)

    def _fit(self, text: str, budget: int, section: str) -> str:
        """Fit text into budget, truncating if necessary."""
        if not text:
            return ""
        if len(text) <= budget:
            return text
        if budget <= 0:
            return ""
        return text[:budget - 20] + f"\n... [{section} truncated]"

    # -- Convenience methods --

    def build_context_for_navigation(
        self,
        goal: str,
        keywords: Optional[List[str]] = None,
    ) -> ContextPacket:
        """Build a lightweight context for code navigation tasks."""
        return self.build_context(
            goal=goal,
            role="researcher",
            profile="local_light",
            keywords=keywords or goal.split(),
        )

    def build_context_for_coding(
        self,
        goal: str,
        file_snippets: Optional[Dict[str, str]] = None,
        recent_actions: Optional[List[Dict[str, Any]]] = None,
        recent_errors: Optional[List[Dict[str, Any]]] = None,
        keywords: Optional[List[str]] = None,
    ) -> ContextPacket:
        """Build a full context for coding tasks."""
        return self.build_context(
            goal=goal,
            role="coder",
            profile="cheap_cloud_reasoning",
            file_snippets=file_snippets or {},
            recent_actions=recent_actions or [],
            recent_errors=recent_errors or [],
            keywords=keywords or goal.split(),
        )

    def build_context_for_testing(
        self,
        goal: str,
        test_output: str = "",
        recent_errors: Optional[List[Dict[str, Any]]] = None,
    ) -> ContextPacket:
        """Build context for test execution and analysis."""
        errors = recent_errors or []
        if test_output and not errors:
            errors = [{"type": "test_output", "message": test_output}]
        return self.build_context(
            goal=goal,
            role="tester",
            profile="cheap_cloud_reasoning",
            recent_errors=errors,
            keywords=["test", "pytest", "assert"] + goal.split(),
        )

    def get_budget_info(self, profile: str = "default") -> Dict[str, Any]:
        """Get token budget information for a profile."""
        budget = TOKEN_BUDGETS.get(profile, TOKEN_BUDGETS["default"])
        return {
            "profile": profile,
            "total_budget_chars": budget,
            "reserved_chars": RESERVED_CHARS,
            "available_chars": budget - RESERVED_CHARS,
            "approximate_tokens": budget // 4,
        }
