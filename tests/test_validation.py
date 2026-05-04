"""Tests for igris.layers.validation.validator."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from igris.layers.validation.validator import (
    ValidationResult,
    get_validation,
    get_validations_for_task,
    map_validation_to_status,
    validate_success_criteria,
    validate_task_completion,
)
from igris.models.task import Task, TaskStatus


@pytest.fixture
def project_dir(tmp_path: Path) -> str:
    (tmp_path / ".igris" / "validations").mkdir(parents=True, exist_ok=True)
    return str(tmp_path)


def _make_task(
    task_id: int = 1,
    desc: str = "test task",
    criteria: List[str] = None,
) -> Task:
    return Task(
        id=task_id,
        description=desc,
        success_criteria=criteria or [],
    )


class TestValidateTaskCompletion:
    def test_no_criteria_no_manual_reason(self, project_dir: str) -> None:
        task = _make_task(criteria=[])
        result = validate_task_completion(task, project_root=project_dir)
        assert not result.valid
        assert result.overall_status == "needs_review"
        assert "no success criteria" in result.reason.lower()

    def test_no_criteria_with_manual_reason(self, project_dir: str) -> None:
        task = _make_task(criteria=[])
        result = validate_task_completion(
            task, manual_completion_reason="Verified manually",
            project_root=project_dir,
        )
        assert result.valid
        assert result.overall_status == "completed"
        assert result.manual_completion_reason == "Verified manually"

    def test_test_criterion_with_passing_report(self, project_dir: str) -> None:
        task = _make_task(criteria=["All tests pass"])
        reports = [{"command_id": "run_tests", "success": True}]
        result = validate_task_completion(
            task, reports=reports, project_root=project_dir,
        )
        assert result.valid
        assert result.overall_status == "completed"

    def test_test_criterion_with_failing_report(self, project_dir: str) -> None:
        task = _make_task(criteria=["pytest passes green"])
        reports = [{"command_id": "run_tests", "success": False, "stderr_truncated": "2 failed"}]
        result = validate_task_completion(
            task, reports=reports, project_root=project_dir,
        )
        assert not result.valid
        assert result.overall_status == "blocked"

    def test_file_criterion_with_existing_file(self, project_dir: str) -> None:
        Path(project_dir, "docs", "readme.md").parent.mkdir(parents=True, exist_ok=True)
        Path(project_dir, "docs", "readme.md").write_text("# Readme")
        task = _make_task(criteria=["Create file docs/readme.md"])
        result = validate_task_completion(
            task, project_root=project_dir,
        )
        assert result.valid
        assert result.overall_status == "completed"

    def test_file_criterion_missing_file(self, project_dir: str) -> None:
        task = _make_task(criteria=["Create file docs/missing.md"])
        result = validate_task_completion(
            task, project_root=project_dir,
        )
        assert not result.valid

    def test_file_criterion_in_files_changed(self, project_dir: str) -> None:
        task = _make_task(criteria=["Add file src/main.py"])
        result = validate_task_completion(
            task, files_changed=["src/main.py"],
            project_root=project_dir,
        )
        assert result.valid

    def test_generic_criterion_needs_review(self, project_dir: str) -> None:
        task = _make_task(criteria=["Performance improved by 20%"])
        result = validate_task_completion(
            task, project_root=project_dir,
        )
        assert not result.valid
        assert result.overall_status == "needs_review"
        assert "manual verification" in result.reason.lower()

    def test_mixed_criteria(self, project_dir: str) -> None:
        task = _make_task(criteria=["All tests pass", "Performance improved"])
        reports = [{"command_id": "run_tests", "success": True}]
        result = validate_task_completion(
            task, reports=reports, project_root=project_dir,
        )
        assert not result.valid
        assert result.overall_status == "needs_review"
        assert "1 met" in result.reason

    def test_manual_override(self, project_dir: str) -> None:
        task = _make_task(criteria=["Performance improved"])
        result = validate_task_completion(
            task, manual_completion_reason="Checked locally",
            project_root=project_dir,
        )
        assert result.valid
        assert "manually overridden" in result.reason.lower()


class TestValidationPersistence:
    def test_saved_and_retrieved(self, project_dir: str) -> None:
        task = _make_task()
        result = validate_task_completion(task, project_root=project_dir)
        retrieved = get_validation(result.validation_id, project_root=project_dir)
        assert retrieved is not None
        assert retrieved.task_id == task.id

    def test_get_validations_for_task(self, project_dir: str) -> None:
        t1 = _make_task(task_id=1, criteria=["test passes"])
        t2 = _make_task(task_id=2, criteria=["test passes"])
        validate_task_completion(t1, project_root=project_dir)
        validate_task_completion(t1, project_root=project_dir)
        validate_task_completion(t2, project_root=project_dir)
        results = get_validations_for_task(1, project_root=project_dir)
        assert len(results) == 2


class TestValidationResult:
    def test_to_dict(self) -> None:
        r = ValidationResult(valid=True, task_id=1, overall_status="completed")
        d = r.to_dict()
        assert d["valid"] is True
        assert d["task_id"] == 1

    def test_from_dict(self) -> None:
        d = {"valid": True, "task_id": 1, "overall_status": "completed", "reason": "OK"}
        r = ValidationResult.from_dict(d)
        assert r.valid
        assert r.task_id == 1

    def test_secret_redacted(self) -> None:
        r = ValidationResult(reason="API_KEY=sk-secrettest1234567890123456")
        d = r.to_dict()
        assert "sk-secrettest1234567890123456" not in d["reason"]

    def test_roundtrip(self) -> None:
        r = ValidationResult(valid=True, task_id=5, overall_status="completed", reason="All good")
        d = r.to_dict()
        r2 = ValidationResult.from_dict(d)
        assert r2.valid == r.valid
        assert r2.task_id == r.task_id


class TestMapValidationToStatus:
    def test_completed(self) -> None:
        r = ValidationResult(overall_status="completed")
        assert map_validation_to_status(r) == "completed"

    def test_needs_review(self) -> None:
        r = ValidationResult(overall_status="needs_review")
        assert map_validation_to_status(r) == "pending"

    def test_blocked(self) -> None:
        r = ValidationResult(overall_status="blocked")
        assert map_validation_to_status(r) == "blocked"


class TestValidateSuccessCriteria:
    def test_empty_criteria(self) -> None:
        results = validate_success_criteria([], [], [])
        assert results == []

    def test_test_criterion_no_reports(self) -> None:
        results = validate_success_criteria(["tests pass"], [], [])
        assert len(results) == 1
        assert results[0]["met"] is False

    def test_file_criterion_found(self) -> None:
        results = validate_success_criteria(
            ["Create docs/api.md"], [], ["docs/api.md"]
        )
        assert results[0]["met"] is True
