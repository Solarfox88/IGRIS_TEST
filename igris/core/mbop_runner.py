"""
Mission Brain Operating Protocol (MBOP) — supervisor integration hooks.

Issue: #936 / MBOP-wiring
Purpose: Wire MBOP phases 1, 9, 10, 11, 12 into the IGRIS supervisor execution
         loop so that every supervised run is structured, gated, and evaluated.

Design principles:
- ADVISORY-ONLY: MBOP hooks never change runtime loop decisions for active runs.
- BEST-EFFORT: any MBOP hook failure is logged but never crashes the supervisor.
- NO AUTO-EXECUTION: MBOP never triggers actions without IGRIS going through its
  own supervisor loop first.
- NO MANDATORY GATE (by default): quality/satisfaction gates are advisory; they
  change the run outcome to "degraded" rather than blocking unconditionally.
  Set mbop_enforce_quality_gate=True in config to make quality-gate failures
  flip the run to "blocked" — this is opt-in per-issue.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class MBOPIntakeResult:
    """Structured intake extracted from a GitHub issue."""
    issue_number: int = 0
    what: str = ""
    where: str = ""
    why: str = ""
    constraints: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    operating_mode: str = "compact"  # compact | full
    raw_body: str = ""
    extraction_ok: bool = False


@dataclass
class MBOPQualityGateResult:
    """Result of the post-completion quality gate (Phase 9)."""
    passed: bool = False
    pytest_ran: bool = False
    pytest_passed: bool = False
    stub_patterns_found: List[str] = field(default_factory=list)
    test_files_checked: List[str] = field(default_factory=list)
    evidence: str = ""
    error: str = ""


@dataclass
class MBOPSatisfactionGateResult:
    """Result of the satisfaction gate (Phase 10)."""
    passed: bool = False
    criteria_checked: List[str] = field(default_factory=list)
    criteria_covered: List[str] = field(default_factory=list)
    criteria_missing: List[str] = field(default_factory=list)
    evidence: str = ""
    error: str = ""


@dataclass
class MBOPEvalResult:
    """Post-task evaluation summary (Phase 11)."""
    summary: str = ""
    lessons: List[str] = field(default_factory=list)
    follow_up_issues: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 1 — Intake
# ---------------------------------------------------------------------------

def mbop_phase1_intake(issue_number: int, project_root: str) -> MBOPIntakeResult:
    """Read GitHub issue and extract structured MBOP intake."""
    result = MBOPIntakeResult(issue_number=issue_number)
    if not issue_number:
        return result
    try:
        proc = subprocess.run(
            ["gh", "issue", "view", str(issue_number), "--json", "title,body,labels"],
            capture_output=True, text=True, timeout=15, cwd=project_root,
        )
        if proc.returncode != 0:
            return result
        import json as _json
        data = _json.loads(proc.stdout)
        body = data.get("body") or ""
        title = data.get("title") or ""
        labels = [lbl.get("name", "") for lbl in data.get("labels", [])]
        result.raw_body = body
        result.operating_mode = "full" if "full" in " ".join(labels).lower() else "compact"
        result.what = _extract_section(body, ["### What", "**What**", "what"]) or title
        result.where = _extract_section(body, ["### Where", "**Where**", "where"])
        result.why = _extract_section(body, ["### Why", "**Why**", "why"])
        result.constraints = _extract_list_section(body, ["### Constraints", "**Constraints**"])
        result.acceptance_criteria = _extract_acceptance_criteria(body)
        result.extraction_ok = True
    except Exception:  # noqa: BLE001
        pass
    return result


def _extract_section(body: str, headers: List[str]) -> str:
    for header in headers:
        idx = body.find(header)
        if idx == -1:
            continue
        start = idx + len(header)
        rest = body[start:]
        match = re.search(r"\n#{1,4} |\n\*\*", rest)
        chunk = rest[: match.start()] if match else rest
        text = chunk.strip()
        if text and text != "_not specified_":
            return text[:500]
    return ""


def _extract_list_section(body: str, headers: List[str]) -> List[str]:
    text = _extract_section(body, headers)
    items = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("-", "*", "+")):
            item = stripped.lstrip("-*+ ").strip()
            if item:
                items.append(item)
    return items


def _extract_acceptance_criteria(body: str) -> List[str]:
    criteria = []
    for line in body.splitlines():
        stripped = line.strip()
        m = re.match(r"-\s*\[[ xX]\]\s*(.+)", stripped)
        if m:
            ac_text = m.group(1).strip()
            if ac_text and not ac_text.lower().startswith("_"):
                criteria.append(ac_text)
    return criteria[:20]


# ---------------------------------------------------------------------------
# Phase 9 — Quality Gate
# ---------------------------------------------------------------------------

_STUB_PATTERNS = [
    "# placeholder", "# todo", "# fixme", "# hack",
    "raise notimplementederror", "pass  # stub", "... # stub",
    # Brittle test patterns — tests that pass trivially without real verification
    "assert true",          # assert True always passes
    "assert false",         # assert False always fails; stub placeholder
    "assert 1 == 1",        # tautology
    "assert none is none",  # tautology
    "return none  # stub",
    "return {}  # stub",
    "return []  # stub",
]
# Regex-based brittle patterns applied per-line in test files
_STUB_REGEX_PATTERNS = [
    re.compile(r"assert\s+\w.*==\s*200\b"),     # assert x == 200 (status code stub)
    re.compile(r"assert\s+response\.ok\b"),      # assert response.ok without any context check
    re.compile(r"assert\s+True\b"),              # assert True (always passes)
    re.compile(r"assert\s+\d+\s*==\s*\d+\b"),   # assert literal == literal (tautology)
]
_MAX_PYTEST_SECONDS = 120


def mbop_phase9_quality_gate(
    project_root: str,
    modified_files: List[str],
    run_pytest: bool = True,
    diff_text: str = "",
) -> MBOPQualityGateResult:
    """Run post-completion quality gate (Phase 9)."""
    result = MBOPQualityGateResult()
    root = Path(project_root)

    stub_found: List[str] = []
    for rel_path in modified_files:
        full = root / rel_path
        if not full.exists() or not rel_path.endswith(".py"):
            continue
        try:
            content = full.read_text(encoding="utf-8", errors="replace")
            content_lower = content.lower()
            for pat in _STUB_PATTERNS:
                if pat in content_lower:
                    stub_found.append(f"{rel_path}:{pat}")
            # Regex patterns on test files only (avoid false positives in production code)
            if "test" in rel_path.lower():
                for line_no, line in enumerate(content.splitlines(), 1):
                    for rpat in _STUB_REGEX_PATTERNS:
                        if rpat.search(line):
                            stub_found.append(f"{rel_path}:{line_no}:{rpat.pattern}")
        except OSError:
            pass
    result.stub_patterns_found = stub_found

    # --- Destructive patch detection ---
    # If a file loses far more lines than it gains, flag it as potentially destructive.
    # Example: supervisor_reasoning_worker.py lost 163 lines and gained 117 — a net -46
    # where most of the original implementation was deleted.
    if diff_text:
        for rel_path in modified_files:
            if not rel_path.endswith(".py"):
                continue
            lines_added = sum(
                1 for ln in diff_text.splitlines()
                if ln.startswith("+") and not ln.startswith("+++")
                and rel_path.split("/")[-1] in diff_text.split(rel_path)[0][-200:]
            )
            lines_removed = sum(
                1 for ln in diff_text.splitlines()
                if ln.startswith("-") and not ln.startswith("---")
                and rel_path.split("/")[-1] in diff_text.split(rel_path)[0][-200:]
            )
            if lines_removed > 50 and lines_added < lines_removed * 0.6:
                stub_found.append(f"{rel_path}:destructive_rewrite (removed={lines_removed} added={lines_added})")
    result.stub_patterns_found = stub_found

    test_files = [f for f in modified_files if re.search(r"test.*\.py$|\.py.*test", f)]
    result.test_files_checked = test_files

    if run_pytest and test_files:
        try:
            import sys as _sys
            _venv_pytest = Path(project_root) / ".venv" / "bin" / "pytest"
            if _venv_pytest.exists():
                _pytest_cmd = [str(_venv_pytest)]
            else:
                _pytest_cmd = [_sys.executable, "-m", "pytest"]
                _check = subprocess.run(
                    [_sys.executable, "-m", "pytest", "--version"],
                    capture_output=True, text=True, timeout=5, cwd=project_root,
                )
                if _check.returncode != 0:
                    result.error = "pytest not available"
                    result.evidence = "pytest not found — skipped"
                    result.passed = len(stub_found) == 0
                    return result
            cmd = _pytest_cmd + ["--tb=short", "-q", "--no-header"] + test_files
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=_MAX_PYTEST_SECONDS, cwd=project_root)
            result.pytest_ran = True
            result.pytest_passed = proc.returncode == 0
            result.evidence = (proc.stdout + proc.stderr)[-1000:]
        except subprocess.TimeoutExpired:
            result.pytest_ran = True
            result.pytest_passed = False
            result.evidence = f"pytest timed out after {_MAX_PYTEST_SECONDS}s"
        except Exception as exc:  # noqa: BLE001
            result.error = f"pytest error: {exc}"
    elif not test_files:
        result.evidence = "no test files in diff — pytest skipped"
    else:
        result.evidence = "pytest disabled by config"

    stub_ok = len(stub_found) == 0
    pytest_ok = (not result.pytest_ran) or result.pytest_passed
    result.passed = stub_ok and pytest_ok
    if not result.passed:
        reasons = []
        if not stub_ok:
            reasons.append(f"stub patterns: {stub_found[:3]}")
        if result.pytest_ran and not result.pytest_passed:
            reasons.append("pytest FAIL")
        result.evidence = "; ".join(reasons) + " | " + result.evidence
    return result


# ---------------------------------------------------------------------------
# Phase 10 — Satisfaction Gate
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "should", "must", "shall", "when", "then", "given", "that", "have", "been",
    "with", "from", "into", "will", "this", "there", "their", "which", "about",
    "would", "could", "other", "more", "also", "than", "these", "those",
}


def mbop_phase10_satisfaction_gate(
    intake: MBOPIntakeResult,
    diff_text: str,
    commit_message: str,
) -> MBOPSatisfactionGateResult:
    """Check that ACs from intake appear in the diff/commit (heuristic). Advisory-only."""
    result = MBOPSatisfactionGateResult()
    criteria = intake.acceptance_criteria
    if not criteria:
        # No structured ACs: advisory pass (not vacuously true — issue may have implicit ACs)
        result.passed = True
        result.evidence = "no structured ACs in issue — satisfaction gate advisory (not verified)"
        return result
    haystack = (diff_text + "\n" + commit_message).lower()
    for ac in criteria:
        result.criteria_checked.append(ac)
        keywords = [w.lower() for w in re.findall(r"\b\w{5,}\b", ac) if w.lower() not in _STOP_WORDS]
        if not keywords:
            keywords = [w.lower() for w in re.findall(r"\b\w{3,}\b", ac)]
        covered = any(kw in haystack for kw in keywords[:5])
        if covered:
            result.criteria_covered.append(ac)
        else:
            result.criteria_missing.append(ac)
    total = len(criteria)
    covered_count = len(result.criteria_covered)
    result.passed = covered_count >= max(1, total // 2)
    result.evidence = f"{covered_count}/{total} ACs keyword-matched in diff"
    return result


# ---------------------------------------------------------------------------
# Phase 11 — Post-Task Evaluation
# ---------------------------------------------------------------------------

def mbop_phase11_post_task_eval(
    intake: MBOPIntakeResult,
    quality: MBOPQualityGateResult,
    satisfaction: MBOPSatisfactionGateResult,
    run_duration_seconds: float,
    failure_class: str = "",
    run_status: str = "",
    completion_mode: str = "",
) -> MBOPEvalResult:
    """Generate a brief post-task evaluation summary (Phase 11)."""
    lessons = []
    if quality.stub_patterns_found:
        lessons.append(f"Stubs detected in output: {quality.stub_patterns_found[:2]}")
    if quality.pytest_ran and not quality.pytest_passed:
        lessons.append("Tests failed at completion — needs re-run")
    if not quality.test_files_checked:
        lessons.append("No test files in diff — tests not verified")
    if satisfaction.criteria_missing:
        lessons.append(f"ACs not addressed: {satisfaction.criteria_missing[:2]}")
    if failure_class:
        lessons.append(f"failure_class={failure_class}")
    # Detect degraded completions: reasoning stopped without clean finish
    if run_status == "completed" and completion_mode in ("degraded", "no_diff_repair", "stopped"):
        lessons.append(f"reasoning stopped without clean finish (mode={completion_mode}) — review diff carefully")
    elif run_status in ("blocked", "interrupted") and not failure_class:
        lessons.append(f"run ended with status={run_status} but no failure_class recorded")
    qg = "PASS" if quality.passed else "FAIL"
    sg = "PASS" if satisfaction.passed else "ADVISORY"
    summary = (
        f"Issue #{intake.issue_number} | QG:{qg} SG:{sg} | "
        f"Duration:{run_duration_seconds:.0f}s | Mode:{intake.operating_mode}"
    )
    return MBOPEvalResult(summary=summary, lessons=lessons)


# ---------------------------------------------------------------------------
# Phase 12 — Next-Step Propagation (decomposition)
# ---------------------------------------------------------------------------

def mbop_phase12_next_step(
    intake: MBOPIntakeResult,
    project_root: str,
    failure_class: str = "",
    open_issues: Optional[List[int]] = None,
) -> List[str]:
    """On decomposition_required, suggest sub-issues. Advisory-only."""
    if failure_class != "decomposition_required":
        return []
    what = intake.what or f"Issue #{intake.issue_number}"
    return [
        f"[MBOP sub] Phase 1 requirements analysis for: {what[:60]}",
        f"[MBOP sub] Phase 2 implementation for: {what[:60]}",
        f"[MBOP sub] Phase 3 tests and verification for: {what[:60]}",
    ]


# ---------------------------------------------------------------------------
# Run helpers — git diff utilities
# ---------------------------------------------------------------------------

def _get_modified_files(project_root: str, base_branch: str = "main") -> List[str]:
    """Get list of files modified vs base branch."""
    for cmd in [
        ["git", "diff", "--name-only", base_branch, "HEAD"],
        ["git", "diff", "--name-only", "HEAD^", "HEAD"],
    ]:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10, cwd=project_root)
            if proc.returncode == 0:
                return [f.strip() for f in proc.stdout.splitlines() if f.strip()]
        except Exception:  # noqa: BLE001
            pass
    return []


def _get_diff_text(project_root: str, base_branch: str = "main") -> str:
    """Get unified diff text vs base branch (truncated)."""
    try:
        proc = subprocess.run(
            ["git", "diff", base_branch, "HEAD"],
            capture_output=True, text=True, timeout=15, cwd=project_root,
        )
        if proc.returncode == 0:
            return proc.stdout[:10000]
    except Exception:  # noqa: BLE001
        pass
    return ""


def _get_last_commit_message(project_root: str) -> str:
    """Get the last commit message."""
    try:
        proc = subprocess.run(
            ["git", "log", "-1", "--pretty=%B"],
            capture_output=True, text=True, timeout=5, cwd=project_root,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


# ---------------------------------------------------------------------------
# Disk persistence — .igris/mbop_events.jsonl
# ---------------------------------------------------------------------------

def _persist_event(
    project_root: str,
    run_id: str,
    issue_number: int,
    phase: str,
    status: str,
    detail: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one MBOP event line to .igris/mbop_events.jsonl. Best-effort, never raises."""
    try:
        import json as _json
        import threading as _threading
        record: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ts_epoch": time.time(),
            "run_id": run_id or "",
            "issue_number": issue_number,
            "phase": phase,
            "status": status,
            "detail": (detail or "")[:500],
        }
        if extra:
            safe: Dict[str, Any] = {}
            for k, v in extra.items():
                try:
                    _json.dumps(v)
                    safe[k] = v
                except (TypeError, ValueError):
                    safe[k] = str(v)
            record["extra"] = safe
        log_path = Path(project_root) / ".igris" / "mbop_events.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = _json.dumps(record, ensure_ascii=False) + "\n"
        # Use a module-level lock to be thread-safe
        if not hasattr(_persist_event, "_lock"):
            _persist_event._lock = _threading.Lock()  # type: ignore[attr-defined]
        with _persist_event._lock:  # type: ignore[attr-defined]
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def mbop_pre_run(
    issue_number: int,
    project_root: str,
    run_add_fn: Any = None,
    run_id: str = "",
) -> MBOPIntakeResult:
    """Execute MBOP Phase 1 (Intake) before the supervisor run.

    Logs to run.events AND to .igris/mbop_events.jsonl.
    Always returns an MBOPIntakeResult — never raises.
    """
    intake = MBOPIntakeResult(issue_number=issue_number)
    try:
        intake = mbop_phase1_intake(issue_number, project_root)
        if intake.extraction_ok:
            detail = (
                f"MBOP Phase 1 Intake: #{issue_number} | "
                f"What: {intake.what[:80]} | "
                f"ACs: {len(intake.acceptance_criteria)} | Mode: {intake.operating_mode}"
            )
            extra: Dict[str, Any] = {
                "what": intake.what[:200], "where": intake.where[:200],
                "why": intake.why[:200], "constraints": intake.constraints[:5],
                "acceptance_criteria": intake.acceptance_criteria[:5],
                "operating_mode": intake.operating_mode,
            }
            if run_add_fn:
                run_add_fn("mbop_phase1_intake", "success", detail,
                           issue_number=issue_number, **extra)
        else:
            detail = (
                f"MBOP Phase 1 Intake: #{issue_number} — "
                "issue not readable (gh CLI unavailable or no issue number)"
            )
            extra = {}
            if run_add_fn:
                run_add_fn("mbop_phase1_intake", "skipped", detail, issue_number=issue_number)
        _persist_event(project_root, run_id, issue_number, "mbop_phase1_intake",
                       "success" if intake.extraction_ok else "skipped", detail, extra or None)
    except Exception as exc:  # noqa: BLE001
        err_detail = f"MBOP intake error (non-fatal): {exc}"
        if run_add_fn:
            try:
                run_add_fn("mbop_phase1_intake", "error", err_detail)
            except Exception:  # noqa: BLE001
                pass
        _persist_event(project_root, run_id, issue_number, "mbop_phase1_intake", "error", err_detail)
    return intake


def mbop_post_run(
    run: Any,
    intake: MBOPIntakeResult,
    project_root: str,
    run_start_ts: float,
    enforce_quality_gate: bool = False,
    run_id: str = "",
) -> None:
    """Execute MBOP Phases 9-12 after the supervisor run.

    Logs to run.events AND to .igris/mbop_events.jsonl.
    Never raises. enforce_quality_gate=True (opt-in) can downgrade to blocked.
    """
    try:
        run_status = getattr(run, "status", "")
        failure_class = getattr(run, "failure_class", "") or ""
        issue_number = intake.issue_number
        duration = time.time() - run_start_ts

        # ---- Phase 9: Quality Gate ----
        modified_files: List[str] = []
        try:
            modified_files = _get_modified_files(project_root)
        except Exception:  # noqa: BLE001
            pass

        # Get diff text early — used by Phase 9 destructive-patch detection and Phase 10
        diff_text_early = ""
        try:
            diff_text_early = _get_diff_text(project_root)
        except Exception:  # noqa: BLE001
            pass

        quality = MBOPQualityGateResult()
        try:
            quality = mbop_phase9_quality_gate(project_root, modified_files, diff_text=diff_text_early)
        except Exception as exc:  # noqa: BLE001
            quality.error = str(exc)

        qg_status = "pass" if quality.passed else "fail"
        qg_detail = (
            f"MBOP Phase 9 Quality Gate: {qg_status.upper()} | "
            f"pytest={'PASS' if quality.pytest_passed else ('FAIL' if quality.pytest_ran else 'skipped')} | "
            f"stubs={quality.stub_patterns_found[:3]} | {quality.evidence[:150]}"
        )
        qg_extra: Dict[str, Any] = {
            "run_status": run_status, "pytest_passed": quality.pytest_passed,
            "pytest_ran": quality.pytest_ran, "stub_patterns": quality.stub_patterns_found[:5],
            "test_files": quality.test_files_checked[:5], "modified_files": modified_files[:10],
            "enforce": enforce_quality_gate,
        }
        try:
            run.add("mbop_phase9_quality_gate", qg_status, qg_detail, **qg_extra)
        except Exception:  # noqa: BLE001
            pass
        _persist_event(project_root, run_id, issue_number, "mbop_phase9_quality_gate", qg_status, qg_detail, qg_extra)

        if not quality.passed and enforce_quality_gate and run_status == "completed":
            enf_detail = (
                "Run downgraded: MBOP Quality Gate FAILED (enforce=True). "
                f"Stubs: {quality.stub_patterns_found[:3]}. "
                f"pytest: {'FAIL' if quality.pytest_ran and not quality.pytest_passed else 'not run'}"
            )
            try:
                run.status = "blocked"
                run.failure_class = "mbop_quality_gate_failed"
                run.outcome = "Blocked — MBOP Quality Gate failed"
                run.add("mbop_quality_gate_enforcement", "blocked", enf_detail)
            except Exception:  # noqa: BLE001
                pass
            _persist_event(project_root, run_id, issue_number, "mbop_quality_gate_enforcement", "blocked", enf_detail)

        # ---- Phase 10: Satisfaction Gate ----
        diff_text, commit_msg = "", ""
        try:
            diff_text = _get_diff_text(project_root)
            commit_msg = _get_last_commit_message(project_root)
        except Exception:  # noqa: BLE001
            pass

        satisfaction = MBOPSatisfactionGateResult()
        try:
            satisfaction = mbop_phase10_satisfaction_gate(intake, diff_text, commit_msg)
        except Exception as exc:  # noqa: BLE001
            satisfaction.error = str(exc)

        sg_status = "pass" if satisfaction.passed else "advisory"
        sg_detail = (
            f"MBOP Phase 10 Satisfaction Gate: {sg_status.upper()} | "
            f"{satisfaction.evidence} | missing={satisfaction.criteria_missing[:3]}"
        )
        sg_extra: Dict[str, Any] = {
            "criteria_checked": satisfaction.criteria_checked[:10],
            "criteria_covered": satisfaction.criteria_covered[:10],
            "criteria_missing": satisfaction.criteria_missing[:10],
            "commit_msg_snippet": commit_msg[:100],
        }
        try:
            run.add("mbop_phase10_satisfaction_gate", sg_status, sg_detail, **sg_extra)
        except Exception:  # noqa: BLE001
            pass
        _persist_event(project_root, run_id, issue_number, "mbop_phase10_satisfaction_gate", sg_status, sg_detail, sg_extra)

        # ---- Phase 11: Post-Task Evaluation ----
        eval_result = MBOPEvalResult()
        try:
            completion_mode = getattr(run, "completion_mode", "") or getattr(run, "degraded_reason", "") or ""
            eval_result = mbop_phase11_post_task_eval(
                intake, quality, satisfaction, duration, failure_class,
                run_status=run_status, completion_mode=completion_mode,
            )
        except Exception:  # noqa: BLE001
            pass
        eval_detail = f"MBOP Phase 11 Post-Task Eval: {eval_result.summary}"
        eval_extra: Dict[str, Any] = {
            "duration_seconds": round(duration, 1), "lessons": eval_result.lessons[:5],
            "quality_gate": qg_status, "satisfaction_gate": sg_status,
            "failure_class": failure_class, "run_status": run_status,
        }
        try:
            run.add("mbop_phase11_post_task_eval", "done", eval_detail, **eval_extra)
        except Exception:  # noqa: BLE001
            pass
        _persist_event(project_root, run_id, issue_number, "mbop_phase11_post_task_eval", "done", eval_detail, eval_extra)

        # ---- Phase 12: Next-Step ----
        try:
            suggestions = mbop_phase12_next_step(intake, project_root, failure_class)
            if suggestions:
                ns_detail = (
                    f"MBOP Phase 12 Next-Step: decomposition_required | suggested: {suggestions[:2]}"
                )
                ns_extra: Dict[str, Any] = {"suggestions": suggestions, "failure_class": failure_class}
                try:
                    run.add("mbop_phase12_next_step", "advisory", ns_detail, **ns_extra)
                except Exception:  # noqa: BLE001
                    pass
                _persist_event(project_root, run_id, issue_number, "mbop_phase12_next_step", "advisory", ns_detail, ns_extra)
        except Exception:  # noqa: BLE001
            pass

    except Exception:  # noqa: BLE001
        pass
