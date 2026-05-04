"""Tests for context-enriched chat (Sprint 17)."""

from __future__ import annotations

import pytest

from igris.core.chat_context import (
    build_chat_context,
    build_context_system_prompt,
    get_context_summary,
)
from igris.core.task_engine import TaskEngine


class TestBuildChatContext:
    def test_returns_sections(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        (tmp_path / ".igris" / "tasks").mkdir(parents=True)
        (tmp_path / ".igris" / "timeline").mkdir(parents=True)
        ctx = build_chat_context(project_root=str(tmp_path))
        assert "sections" in ctx
        assert "timestamp" in ctx
        assert "missions" in ctx["sections"]
        assert "tasks" in ctx["sections"]
        assert "memory" in ctx["sections"]
        assert "git" in ctx["sections"]
        assert "patches" in ctx["sections"]
        assert "cost" in ctx["sections"]
        assert "project_state" in ctx["sections"]

    def test_tasks_section(self, tmp_path):
        (tmp_path / ".igris" / "tasks").mkdir(parents=True)
        (tmp_path / ".igris" / "timeline").mkdir(parents=True)
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        import os
        from igris.models.config import CONFIG
        from pathlib import Path
        os.environ["PROJECT_ROOT"] = str(tmp_path)
        CONFIG.project_root = Path(str(tmp_path))

        te = TaskEngine()
        te.create_task("test task")
        ctx = build_chat_context(task_engine=te, project_root=str(tmp_path))
        tasks = ctx["sections"]["tasks"]
        assert tasks["total"] == 1
        assert tasks["pending"] == 1

    def test_memory_section(self, tmp_path):
        from igris.core import decision_memory
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        (tmp_path / ".igris" / "tasks").mkdir(parents=True)
        (tmp_path / ".igris" / "timeline").mkdir(parents=True)
        pr = str(tmp_path)
        decision_memory.record_saturation("deploy", reason="test", project_root=pr)
        ctx = build_chat_context(project_root=pr)
        mem = ctx["sections"]["memory"]
        assert "deploy" in mem["saturated_families"]
        assert "deploy" in mem["avoid_families"]


class TestBuildContextSystemPrompt:
    def test_returns_string(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        (tmp_path / ".igris" / "tasks").mkdir(parents=True)
        (tmp_path / ".igris" / "timeline").mkdir(parents=True)
        prompt = build_context_system_prompt(project_root=str(tmp_path))
        assert isinstance(prompt, str)
        assert "IGRIS_GPT" in prompt
        assert "CURRENT PROJECT CONTEXT" in prompt

    def test_no_secrets_in_prompt(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        (tmp_path / ".igris" / "tasks").mkdir(parents=True)
        (tmp_path / ".igris" / "timeline").mkdir(parents=True)
        prompt = build_context_system_prompt(project_root=str(tmp_path))
        assert "sk-" not in prompt
        assert "ghp_" not in prompt

    def test_includes_safety_note(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        (tmp_path / ".igris" / "tasks").mkdir(parents=True)
        (tmp_path / ".igris" / "timeline").mkdir(parents=True)
        prompt = build_context_system_prompt(project_root=str(tmp_path))
        assert "do NOT execute" in prompt.lower() or "proper workflow" in prompt.lower()


class TestContextSummary:
    def test_summary_structure(self, tmp_path):
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        (tmp_path / ".igris" / "tasks").mkdir(parents=True)
        (tmp_path / ".igris" / "timeline").mkdir(parents=True)
        summary = get_context_summary(project_root=str(tmp_path))
        assert "timestamp" in summary
        assert "missions_active" in summary
        assert "tasks_pending" in summary
        assert "tasks_blocked" in summary
        assert "memory_avoid_families" in summary
        assert "git_branch" in summary
        assert "patches_pending" in summary
        assert "provider" in summary
        assert "cooling_down" in summary

    def test_summary_with_tasks(self, tmp_path):
        (tmp_path / ".igris" / "tasks").mkdir(parents=True)
        (tmp_path / ".igris" / "timeline").mkdir(parents=True)
        (tmp_path / ".igris" / "memory").mkdir(parents=True)
        import os
        from igris.models.config import CONFIG
        from pathlib import Path
        os.environ["PROJECT_ROOT"] = str(tmp_path)
        CONFIG.project_root = Path(str(tmp_path))

        te = TaskEngine()
        te.create_task("task 1")
        te.create_task("task 2")
        summary = get_context_summary(task_engine=te, project_root=str(tmp_path))
        assert summary["tasks_pending"] == 2
