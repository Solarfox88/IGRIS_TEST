"""Prompt Contract for IGRIS_GPT — Epic #58.

Defines the system prompt template and few-shot examples that teach
the LLM how to interact with IGRIS through structured actions.

The prompt contract is the interface between human-readable goals
and machine-executable action schemas. It tells the LLM:
- What it can do (action types)
- How to format actions (JSON schema)
- What role it is in (agent mode)
- What context it has (files, errors, history)
- When to stop (finish/blocked/ask_user)
- What it must never do (shell free, secrets, unsafe)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from igris.core.agent_action_schema import (
    ACTION_TYPES,
    AGENT_ROLES,
    RISK_HINTS,
    AGENT_REGISTRY,
)


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

REASONING_LOOP_SYSTEM_PROMPT = """You are IGRIS, an AI engineering operator. You work on real repositories, machines, and servers.

## Your Role
You are currently operating as: {role}
{role_description}

## How You Work
You propose ONE structured action at a time as JSON. IGRIS validates, risk-classifies, and executes your action. You then observe the result and propose the next action.

You NEVER execute commands directly. You ALWAYS propose through the action schema.

## File Editing Policy (CRITICAL — read before any write)
When editing **existing** source files (.py, .js, .ts, .html, .css, .md, .json, .yaml, etc.):
- NEVER use `write_file` to replace a large file with a small snippet.
  A file with hundreds of lines must NOT be replaced with 4 lines.
- For **small additions** to existing files: use `insert_after`, `insert_before`, or `append_file`.
- For **targeted replacements**: use `replace_range` (specify exact start/end line numbers).
- For **full file rewrites**: use `write_file` ONLY when the new content is a complete, valid replacement
  (i.e. new content ≥ 30% of current file size). Otherwise IGRIS will block it.
- `write_file` is safe for **new files** (file does not exist yet).
- When in doubt, use `propose_patch` to preview the diff first.

Available safe-edit actions:
  insert_after(path, anchor, content)   — add lines after the anchor line
  insert_before(path, anchor, content)  — add lines before the anchor line
  replace_range(path, start, end, content) — replace lines start..end (1-based)
  append_file(path, content)            — add lines at end of file
  propose_patch(path, content)          — preview diff without writing

## Action Schema
Respond with exactly one JSON object (no markdown, no explanation outside JSON):

{{
  "mode": "{role}",
  "action_type": "<one of the available action types>",
  "reason": "<why this action is needed for the current goal>",
  "parameters": {{<action-specific parameters>}},
  "expected_effect": "<what this action should achieve>",
  "risk_hint": "<low|medium|high|critical|unknown>",
  "confidence": <0.0 to 1.0>,
  "required_preconditions": [<conditions that must be true>],
  "success_check": {{<how to verify success>}},
  "fallback_if_blocked": "<alternative action_type or null>"
}}

## Available Action Types
{action_types_doc}

## Rules
1. Always respond with valid JSON only — no text outside the JSON object.
2. Always include "reason" explaining why this specific action is needed.
3. Use "risk_hint" honestly — if unsure, say "unknown".
4. If you need information, use read/search actions BEFORE modifying anything.
5. If you cannot proceed, use "blocked" with a clear reason.
6. If you need human input, use "ask_user" with a specific question.
7. When the goal is achieved, use "finish" with a summary.
8. NEVER include secrets, API keys, passwords, or tokens in your output.
9. NEVER propose actions that read .env files or secret stores.
10. Prefer structured tools over raw shell commands.
11. If you must use shell, prefer "shell_template" over "raw_shell_proposal".
12. For "raw_shell_proposal", the command will be analyzed by the Command Risk Engine before execution.
13. CONSUME BEFORE EXPLORING: If discovered_files, search_matched_files, or last_tool_result
    appear in Current State, your NEXT action MUST read or act on those results. Do NOT issue
    another find_files or search_code when you already have results waiting to be consumed.
14. PROGRESSIVE EXECUTION: Follow this sequence for every task and never skip steps backward:
    find/search → read → modify/create → run_tests → finish.
    After reading a file, your next action modifies it or creates the required artifact — NOT
    another search. After writing code, run tests. After tests pass, finish.

## Current Mission
{mission_context}

## Current State
{state_context}

## Recent Actions and Results
{recent_actions}

## Available Context
{file_context}
{examples_context}"""


# ---------------------------------------------------------------------------
# Action type documentation for the prompt
# ---------------------------------------------------------------------------

ACTION_TYPE_DOCS: Dict[str, str] = {
    "search_code": 'Search for patterns in code. Params: {{"pattern": "...", "path": "..." (optional)}}',
    "find_files": 'Find files by name/glob pattern. Params: {{"pattern": "..."}}',
    "list_directory": 'List directory contents. Params: {{"path": "...", "depth": 1 (optional)}}',
    "read_file_range": 'Read specific lines from a file. Params: {{"path": "...", "start": 1 (optional), "end": 50 (optional)}}',
    "write_file": 'Write/create a NEW file or FULL replacement (≥30% of existing size). Params: {{"path": "...", "content": "..."}}',
    "insert_after": 'Insert lines after anchor. Params: {{"path": "...", "anchor": "...", "content": "..."}}',
    "insert_before": 'Insert lines before anchor. Params: {{"path": "...", "anchor": "...", "content": "..."}}',
    "replace_range": 'Replace explicit line range (1-based). Params: {{"path": "...", "start": 10, "end": 15, "content": "..."}}',
    "append_file": 'Append content to end of file. Params: {{"path": "...", "content": "..."}}',
    "propose_patch": 'Propose a code patch. Params: {{"files": [{{"path": "...", "action": "modify", "content": "...", "original": "..."}}]}}',
    "apply_patch": 'Apply a validated patch. Params: {{"patch_id": "..."}}',
    "run_tests": 'Run test suite. Params: {{"target": "..." (optional), "args": "..." (optional)}}',
    "git_status": "Check git status. Params: {}",
    "git_diff": 'View git diff. Params: {{"target": "..." (optional)}}',
    "shell_template": 'Run a pre-approved command template. Params: {{"template_id": "...", "args": {{}} (optional)}}',
    "raw_shell_proposal": 'Propose a shell command (gated by risk engine). Params: {{"command": "...", "cwd": "..." (optional)}}',
    "http_check": 'HTTP health check. Params: {{"url": "...", "expected_status": 200 (optional)}}',
    "update_plan": 'Update mission plan. Params: {{"updates": "..."}}',
    "record_memory": 'Record a lesson/decision. Params: {{"event_type": "...", "content": "..."}}',
    "ask_user": 'Ask the human a question. Params: {{"question": "..."}}',
    "finish": 'Declare task complete. Params: {{"summary": "...", "files_modified": [...] (optional), "tests_passed": true/false (optional)}}',
    "blocked": 'Declare inability to proceed. Params: {{"reason": "...", "attempted": [...] (optional)}}',
}


def _build_action_types_doc(role: str) -> str:
    """Build action types documentation filtered by role permissions."""
    entry = AGENT_REGISTRY.get(role)
    allowed = entry.allowed_actions if entry else set(ACTION_TYPES)

    lines = []
    for at in ACTION_TYPES:
        if at in allowed:
            doc = ACTION_TYPE_DOCS.get(at, "No documentation")
            lines.append(f"- {at}: {doc}")
        else:
            lines.append(f"- {at}: NOT AVAILABLE for your role")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build prompt
# ---------------------------------------------------------------------------

def build_reasoning_prompt(
    role: str = "coder",
    mission_context: str = "No active mission.",
    state_context: str = "No state information available.",
    recent_actions: str = "No recent actions.",
    file_context: str = "No files loaded.",
    examples_context: str = "",
) -> str:
    """Build the complete system prompt for the reasoning loop.

    Args:
        role: Current agent role from the registry
        mission_context: Description of current mission/goal
        state_context: Current world state summary
        recent_actions: Recent action history and results
        file_context: Relevant file contents / code context
        examples_context: Optional CoT step-by-step examples (#1043 fix)

    Returns:
        Complete system prompt string
    """
    entry = AGENT_REGISTRY.get(role)
    role_desc = entry.description if entry else "General-purpose agent"

    action_types_doc = _build_action_types_doc(role)

    return REASONING_LOOP_SYSTEM_PROMPT.format(
        role=role,
        role_description=role_desc,
        action_types_doc=action_types_doc,
        mission_context=mission_context,
        state_context=state_context,
        recent_actions=recent_actions,
        file_context=file_context,
        examples_context=("\n" + examples_context) if examples_context else "",
    )


# ---------------------------------------------------------------------------
# Few-shot examples for 10+ scenarios
# ---------------------------------------------------------------------------

EXAMPLE_SCENARIOS: List[Dict[str, Any]] = [
    {
        "scenario": "Read a file to understand code structure",
        "goal": "Understand how FastAPI routes are defined",
        "action": {
            "mode": "researcher",
            "action_type": "read_file_range",
            "reason": "Need to understand existing route patterns before adding new endpoint",
            "parameters": {"path": "igris/web/server.py", "start": 1, "end": 80},
            "expected_effect": "See FastAPI app creation and route registration pattern",
            "risk_hint": "low",
            "confidence": 0.9,
            "required_preconditions": [],
            "success_check": {"has_content": True},
            "fallback_if_blocked": "find_files",
        },
    },
    {
        "scenario": "Search for a specific pattern in code",
        "goal": "Find where API endpoints are registered",
        "action": {
            "mode": "researcher",
            "action_type": "search_code",
            "reason": "Need to find all @app.get and @app.post decorators",
            "parameters": {"pattern": "@app\\.(get|post|put|delete)", "path": "igris/"},
            "expected_effect": "List of all API endpoint definitions",
            "risk_hint": "low",
            "confidence": 0.85,
            "required_preconditions": [],
            "success_check": {"has_results": True},
            "fallback_if_blocked": "find_files",
        },
    },
    {
        "scenario": "Write a new endpoint",
        "goal": "Add /api/ping endpoint",
        "action": {
            "mode": "coder",
            "action_type": "write_file",
            "reason": "Adding /api/ping endpoint that returns {\"pong\": true}",
            "parameters": {
                "path": "igris/web/server.py",
                "content": "# ... modified content with new endpoint ...",
            },
            "expected_effect": "New /api/ping endpoint available",
            "risk_hint": "medium",
            "confidence": 0.8,
            "required_preconditions": ["file_read: igris/web/server.py"],
            "success_check": {"endpoint_exists": "/api/ping"},
            "fallback_if_blocked": "propose_patch",
        },
    },
    {
        "scenario": "Run tests after code change",
        "goal": "Verify code changes pass tests",
        "action": {
            "mode": "tester",
            "action_type": "run_tests",
            "reason": "Verify that the new /api/ping endpoint works and existing tests pass",
            "parameters": {"target": "tests/", "args": "-q"},
            "expected_effect": "All tests pass including new ping test",
            "risk_hint": "low",
            "confidence": 0.7,
            "required_preconditions": ["code_written"],
            "success_check": {"exit_code": 0},
            "fallback_if_blocked": None,
        },
    },
    {
        "scenario": "Check git status before committing",
        "goal": "Verify working tree state",
        "action": {
            "mode": "coder",
            "action_type": "git_status",
            "reason": "Check what files were modified before proposing commit",
            "parameters": {},
            "expected_effect": "See list of modified/new files",
            "risk_hint": "low",
            "confidence": 0.95,
            "required_preconditions": [],
            "success_check": {},
            "fallback_if_blocked": None,
        },
    },
    {
        "scenario": "Use shell template for safe operation",
        "goal": "Install a Python dependency",
        "action": {
            "mode": "devops",
            "action_type": "shell_template",
            "reason": "Need to install pytest-asyncio for async test support",
            "parameters": {"template_id": "pip_install", "args": {"package": "pytest-asyncio"}},
            "expected_effect": "pytest-asyncio installed in virtual environment",
            "risk_hint": "medium",
            "confidence": 0.85,
            "required_preconditions": ["venv_active"],
            "success_check": {"exit_code": 0},
            "fallback_if_blocked": "ask_user",
        },
    },
    {
        "scenario": "Raw shell proposal (gated)",
        "goal": "Check disk space on server",
        "action": {
            "mode": "devops",
            "action_type": "raw_shell_proposal",
            "reason": "Need to check available disk space to ensure enough room for deployment",
            "parameters": {"command": "df -h /", "cwd": "/"},
            "expected_effect": "Disk usage information for root partition",
            "risk_hint": "low",
            "confidence": 0.9,
            "required_preconditions": [],
            "success_check": {"exit_code": 0},
            "fallback_if_blocked": "http_check",
        },
    },
    {
        "scenario": "HTTP health check",
        "goal": "Verify service is running",
        "action": {
            "mode": "devops",
            "action_type": "http_check",
            "reason": "Verify the FastAPI server is responding after restart",
            "parameters": {"url": "http://localhost:7778/api/health", "expected_status": 200},
            "expected_effect": "Server responds with 200 OK",
            "risk_hint": "low",
            "confidence": 0.8,
            "required_preconditions": ["service_started"],
            "success_check": {"status_code": 200},
            "fallback_if_blocked": None,
        },
    },
    {
        "scenario": "Record a lesson in memory",
        "goal": "Remember that test file needs specific import",
        "action": {
            "mode": "memory_manager",
            "action_type": "record_memory",
            "reason": "Recording that test_server.py needs httpx TestClient import",
            "parameters": {
                "event_type": "lesson",
                "content": "FastAPI tests in this project use httpx.AsyncClient, not TestClient",
            },
            "expected_effect": "Lesson stored for future reference",
            "risk_hint": "low",
            "confidence": 0.95,
            "required_preconditions": [],
            "success_check": {},
            "fallback_if_blocked": None,
        },
    },
    {
        "scenario": "Finish a completed task",
        "goal": "Report task completion",
        "action": {
            "mode": "reporter",
            "action_type": "finish",
            "reason": "All objectives met: endpoint added, test written, all tests pass",
            "parameters": {
                "summary": "Added /api/ping endpoint returning {\"pong\": true}. Added test_ping.py with 3 tests. All 1596 tests pass.",
                "files_modified": ["igris/web/server.py", "tests/test_ping.py"],
                "tests_passed": True,
            },
            "expected_effect": "Mission marked as complete with final report",
            "risk_hint": "low",
            "confidence": 0.95,
            "required_preconditions": ["all_tests_pass"],
            "success_check": {},
            "fallback_if_blocked": None,
        },
    },
    {
        "scenario": "Blocked — cannot proceed",
        "goal": "Report inability to continue",
        "action": {
            "mode": "coder",
            "action_type": "blocked",
            "reason": "Cannot find the correct route registration pattern — server.py uses an unusual structure",
            "parameters": {
                "reason": "Route registration pattern not recognized. Need human guidance on where to add the endpoint.",
                "attempted": ["read_file_range server.py", "search_code @app.get", "search_code router"],
            },
            "expected_effect": "Mission paused pending human assistance",
            "risk_hint": "low",
            "confidence": 0.3,
            "required_preconditions": [],
            "success_check": {},
            "fallback_if_blocked": "ask_user",
        },
    },
    {
        "scenario": "Ask user for clarification",
        "goal": "Need human input on ambiguous requirement",
        "action": {
            "mode": "coordinator",
            "action_type": "ask_user",
            "reason": "The requirement says 'add endpoint' but doesn't specify response format",
            "parameters": {
                "question": "Should /api/ping return {\"pong\": true} or {\"status\": \"ok\", \"pong\": true}?",
            },
            "expected_effect": "Human provides clarification on response format",
            "risk_hint": "low",
            "confidence": 0.5,
            "required_preconditions": [],
            "success_check": {},
            "fallback_if_blocked": None,
        },
    },
]


# ---------------------------------------------------------------------------
# Chain-of-thought multi-step examples (#1043)
# These show complete thought→action→result sequences, not isolated actions.
# ---------------------------------------------------------------------------

CHAIN_OF_THOUGHT_EXAMPLES: List[Dict[str, Any]] = [
    {
        "scenario": "Add a new API endpoint end-to-end (5-step chain)",
        "goal": "Add /api/version-info endpoint with tests",
        "chain": [
            {
                "step": 1,
                "thought": "I need to add /api/version-info. First I should understand the existing route patterns in server.py.",
                "action": {
                    "mode": "coder",
                    "action_type": "read_file_range",
                    "reason": "Read server.py to understand existing endpoint pattern before adding new one",
                    "parameters": {"path": "igris/web/server.py", "start": 1, "end": 60},
                    "expected_effect": "See how existing endpoints are defined",
                    "risk_hint": "low",
                    "confidence": 0.9,
                    "required_preconditions": [],
                    "success_check": {"has_content": True},
                    "fallback_if_blocked": "search_code",
                },
                "observed_result": "server.py loaded — endpoints use @router.get decorator, router included at line 45",
            },
            {
                "step": 2,
                "thought": "I now know the pattern. I will insert the /api/version-info endpoint after the last @router.get block using insert_after.",
                "action": {
                    "mode": "coder",
                    "action_type": "insert_after",
                    "reason": "Add /api/version-info endpoint using the existing router pattern discovered in step 1",
                    "parameters": {
                        "path": "igris/web/server.py",
                        "anchor": "@router.get(\"/api/status\")",
                        "content": "\n@router.get(\"/api/version-info\")\ndef version_info():\n    return {\"version\": \"1.0\", \"name\": \"igris\"}\n",
                    },
                    "expected_effect": "/api/version-info route added to server.py",
                    "risk_hint": "medium",
                    "confidence": 0.85,
                    "required_preconditions": ["read: igris/web/server.py"],
                    "success_check": {"file_modified": "igris/web/server.py"},
                    "fallback_if_blocked": "propose_patch",
                },
                "observed_result": "insert_after succeeded — server.py now contains /api/version-info",
            },
            {
                "step": 3,
                "thought": "Endpoint added. Now I need a test file. I'll write tests/test_version_info.py using FastAPI TestClient.",
                "action": {
                    "mode": "coder",
                    "action_type": "write_file",
                    "reason": "Create test file for /api/version-info using TestClient(create_app())",
                    "parameters": {
                        "path": "tests/test_version_info.py",
                        "content": "from fastapi.testclient import TestClient\nfrom igris.web.server import create_app\n\nclient = TestClient(create_app())\n\ndef test_version_info_status():\n    r = client.get(\"/api/version-info\")\n    assert r.status_code == 200\n\ndef test_version_info_keys():\n    r = client.get(\"/api/version-info\")\n    data = r.json()\n    assert \"version\" in data\n",
                    },
                    "expected_effect": "tests/test_version_info.py created with endpoint coverage",
                    "risk_hint": "low",
                    "confidence": 0.9,
                    "required_preconditions": ["code_written: igris/web/server.py"],
                    "success_check": {"file_exists": "tests/test_version_info.py"},
                    "fallback_if_blocked": None,
                },
                "observed_result": "tests/test_version_info.py created successfully",
            },
            {
                "step": 4,
                "thought": "Code and tests written. Run pytest to verify everything passes.",
                "action": {
                    "mode": "tester",
                    "action_type": "run_tests",
                    "reason": "Verify the new endpoint and test file work correctly",
                    "parameters": {"target": "tests/test_version_info.py", "args": "-v"},
                    "expected_effect": "2 tests pass for /api/version-info",
                    "risk_hint": "low",
                    "confidence": 0.85,
                    "required_preconditions": ["file_written: tests/test_version_info.py"],
                    "success_check": {"exit_code": 0},
                    "fallback_if_blocked": None,
                },
                "observed_result": "2 passed in 0.8s — tests pass",
            },
            {
                "step": 5,
                "thought": "All done. Endpoint added, tests pass. Declare completion.",
                "action": {
                    "mode": "reporter",
                    "action_type": "finish",
                    "reason": "Goal achieved: /api/version-info endpoint added and tested",
                    "parameters": {
                        "summary": "Added /api/version-info endpoint to igris/web/server.py. Created tests/test_version_info.py with 2 tests. All tests pass.",
                        "files_modified": ["igris/web/server.py", "tests/test_version_info.py"],
                        "tests_passed": True,
                    },
                    "expected_effect": "Mission marked complete",
                    "risk_hint": "low",
                    "confidence": 0.98,
                    "required_preconditions": ["tests_pass"],
                    "success_check": {},
                    "fallback_if_blocked": None,
                },
                "observed_result": "Mission complete",
            },
        ],
    },
    {
        "scenario": "Fix ImportError / missing symbol (4-step chain)",
        "goal": "Fix pytest collection error: cannot import 'MemoryRetriever' from 'igris.core.long_term_memory'",
        "chain": [
            {
                "step": 1,
                "thought": "The test tries to import MemoryRetriever from igris.core.long_term_memory. I need to read the test to understand the expected interface.",
                "action": {
                    "mode": "coder",
                    "action_type": "search_code",
                    "reason": "Find all usages of MemoryRetriever in tests to understand its interface",
                    "parameters": {"pattern": "MemoryRetriever", "path": "tests/"},
                    "expected_effect": "See how MemoryRetriever is instantiated and used",
                    "risk_hint": "low",
                    "confidence": 0.9,
                    "required_preconditions": [],
                    "success_check": {"has_results": True},
                    "fallback_if_blocked": "find_files",
                },
                "observed_result": "tests/test_long_term_memory.py line 15: mr = MemoryRetriever(memory); mr.retrieve_recent('domain', limit=5)",
            },
            {
                "step": 2,
                "thought": "Interface is clear: MemoryRetriever(memory) with retrieve_recent(). Now read long_term_memory.py to find the right insertion point.",
                "action": {
                    "mode": "coder",
                    "action_type": "read_file_range",
                    "reason": "Read long_term_memory.py to find insertion point for MemoryRetriever class",
                    "parameters": {"path": "igris/core/long_term_memory.py", "start": 1, "end": 50},
                    "expected_effect": "Locate end of LongTermMemory class to append MemoryRetriever after it",
                    "risk_hint": "low",
                    "confidence": 0.9,
                    "required_preconditions": [],
                    "success_check": {"has_content": True},
                    "fallback_if_blocked": None,
                },
                "observed_result": "File loaded — LongTermMemory class ends at line 298, file ends at line 300",
            },
            {
                "step": 3,
                "thought": "I have the interface and the insertion point. Append MemoryRetriever class after LongTermMemory.",
                "action": {
                    "mode": "coder",
                    "action_type": "append_file",
                    "reason": "Add MemoryRetriever class that tests expect to find in this module",
                    "parameters": {
                        "path": "igris/core/long_term_memory.py",
                        "content": "\n\nclass MemoryRetriever:\n    def __init__(self, memory):\n        self._memory = memory\n\n    def retrieve_recent(self, domain, limit=10):\n        return self._memory.get_entries(domain, limit=limit)\n\n    def retrieve_contextual(self, domain, query='', limit=10):\n        if query:\n            return self._memory.search(query, domains=[domain], limit=limit)\n        return self._memory.get_entries(domain, limit=limit)\n",
                    },
                    "expected_effect": "MemoryRetriever class appended to long_term_memory.py",
                    "risk_hint": "low",
                    "confidence": 0.95,
                    "required_preconditions": ["read: igris/core/long_term_memory.py"],
                    "success_check": {"file_modified": "igris/core/long_term_memory.py"},
                    "fallback_if_blocked": "insert_after",
                },
                "observed_result": "append_file succeeded",
            },
            {
                "step": 4,
                "thought": "Class added. Run tests to confirm collection error is resolved and tests pass.",
                "action": {
                    "mode": "tester",
                    "action_type": "run_tests",
                    "reason": "Verify ImportError is resolved and all related tests pass",
                    "parameters": {"target": "tests/test_long_term_memory.py", "args": "-v"},
                    "expected_effect": "All tests in test_long_term_memory.py pass",
                    "risk_hint": "low",
                    "confidence": 0.9,
                    "required_preconditions": ["code_written: igris/core/long_term_memory.py"],
                    "success_check": {"exit_code": 0},
                    "fallback_if_blocked": None,
                },
                "observed_result": "10 passed — ImportError resolved, all tests pass",
            },
        ],
    },
    {
        "scenario": "Replace a buggy function (4-step chain)",
        "goal": "Fix rate limiter that crashes on OPTIONS requests",
        "chain": [
            {
                "step": 1,
                "thought": "I need to find the rate limiter code. Search for the rate limiting logic in the security module.",
                "action": {
                    "mode": "coder",
                    "action_type": "search_code",
                    "reason": "Find rate limiter implementation to understand the crash location",
                    "parameters": {"pattern": "is_destructive", "path": "igris/"},
                    "expected_effect": "Find the line that incorrectly classifies OPTIONS requests",
                    "risk_hint": "low",
                    "confidence": 0.9,
                    "required_preconditions": [],
                    "success_check": {"has_results": True},
                    "fallback_if_blocked": "find_files",
                },
                "observed_result": "igris/web/security.py:45: is_destructive = path.startswith(p) (missing method check)",
            },
            {
                "step": 2,
                "thought": "Found it at security.py:45. Read lines 40-55 to see the full logic before replacing.",
                "action": {
                    "mode": "coder",
                    "action_type": "read_file_range",
                    "reason": "Read the buggy is_destructive block to write an exact replacement",
                    "parameters": {"path": "igris/web/security.py", "start": 40, "end": 55},
                    "expected_effect": "See exact lines to replace",
                    "risk_hint": "low",
                    "confidence": 0.9,
                    "required_preconditions": [],
                    "success_check": {"has_content": True},
                    "fallback_if_blocked": None,
                },
                "observed_result": "Lines 40-55 loaded — is_destructive missing method filter at line 43",
            },
            {
                "step": 3,
                "thought": "I know exactly what to fix: add `and _method not in ('GET','HEAD','OPTIONS')`. Use replace_range for surgical fix.",
                "action": {
                    "mode": "coder",
                    "action_type": "replace_range",
                    "reason": "Fix is_destructive to exclude safe HTTP methods, preventing OPTIONS crash",
                    "parameters": {
                        "path": "igris/web/security.py",
                        "start": 43,
                        "end": 44,
                        "content": "    is_destructive = (\n        any(path.startswith(p) for p in _DESTRUCTIVE_PATH_PREFIXES)\n        and _method not in (\"GET\", \"HEAD\", \"OPTIONS\")\n    )\n",
                    },
                    "expected_effect": "OPTIONS requests no longer classified as destructive",
                    "risk_hint": "medium",
                    "confidence": 0.9,
                    "required_preconditions": ["read: igris/web/security.py lines 40-55"],
                    "success_check": {"file_modified": "igris/web/security.py"},
                    "fallback_if_blocked": "propose_patch",
                },
                "observed_result": "replace_range succeeded",
            },
            {
                "step": 4,
                "thought": "Fix applied. Run the security tests to confirm.",
                "action": {
                    "mode": "tester",
                    "action_type": "run_tests",
                    "reason": "Verify rate limiter fix does not break existing security tests",
                    "parameters": {"target": "tests/test_security.py", "args": "-v"},
                    "expected_effect": "All security tests pass",
                    "risk_hint": "low",
                    "confidence": 0.85,
                    "required_preconditions": ["code_written: igris/web/security.py"],
                    "success_check": {"exit_code": 0},
                    "fallback_if_blocked": None,
                },
                "observed_result": "All tests pass — rate limiter fix confirmed",
            },
        ],
    },
]


def get_example_scenarios() -> List[Dict[str, Any]]:
    """Return all example scenarios for documentation and testing."""
    return EXAMPLE_SCENARIOS


def get_cot_examples() -> List[Dict[str, Any]]:
    """Return chain-of-thought multi-step examples (#1043)."""
    return CHAIN_OF_THOUGHT_EXAMPLES
