"""LLM-based memory analysis for IGRIS_GPT.

Transforms decision/failure memory from a raw event log into
actionable operational insights:

- Repeated failure pattern detection
- Likely root cause identification
- Recommended avoid families
- Suggested remediation strategies
- Lessons learned extraction

All analysis is advisory-only — it never executes actions.
Uses LLM when available, deterministic fallback otherwise.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from igris.core.decision_memory import (
    _load_events,
    get_recent_decisions,
    get_recent_failures,
    get_saturated_families,
    explain_memory_constraints,
)
from igris.core.chat_engine import chat
from igris.core.safety import redact_secrets


# ---------------------------------------------------------------------------
# Deterministic analysis (fallback)
# ---------------------------------------------------------------------------

def _analyze_failure_patterns(
    project_root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Detect repeated failure patterns from memory."""
    failures = _load_events("failure", project_root)
    if not failures:
        return []

    family_counts: Counter = Counter()
    family_reasons: Dict[str, List[str]] = {}
    family_titles: Dict[str, List[str]] = {}

    for f in failures:
        fam = f.family or "unknown"
        family_counts[fam] += 1
        reason = redact_secrets(f.reason or f.title or "")
        if reason and fam not in family_reasons:
            family_reasons[fam] = []
        if reason:
            family_reasons[fam].append(reason)
        title = redact_secrets(f.title or "")
        if title:
            if fam not in family_titles:
                family_titles[fam] = []
            family_titles[fam].append(title)

    patterns = []
    for fam, count in family_counts.most_common(10):
        if count >= 2:
            reasons = family_reasons.get(fam, [])
            unique_reasons = list(set(reasons[:5]))
            patterns.append({
                "family": fam,
                "failure_count": count,
                "is_repeated": count >= 3,
                "sample_reasons": unique_reasons[:3],
                "sample_titles": list(set(family_titles.get(fam, [])))[:3],
            })

    return patterns


def _identify_root_causes(
    patterns: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Identify likely root causes from failure patterns."""
    causes = []
    for p in patterns:
        fam = p["family"]
        count = p["failure_count"]
        reasons = p.get("sample_reasons", [])

        if count >= 5:
            severity = "critical"
            suggestion = f"Family '{fam}' has {count} failures — consider blocking or restructuring"
        elif count >= 3:
            severity = "high"
            suggestion = f"Family '{fam}' shows repeated failures — investigate root cause"
        else:
            severity = "moderate"
            suggestion = f"Family '{fam}' has occasional failures — monitor"

        cause = {
            "family": fam,
            "severity": severity,
            "likely_cause": _infer_cause(fam, reasons),
            "suggestion": suggestion,
            "evidence_count": count,
        }
        causes.append(cause)

    return causes


def _infer_cause(family: str, reasons: List[str]) -> str:
    """Infer likely cause from family and reasons (deterministic)."""
    combined = " ".join(reasons).lower()

    if "timeout" in combined or "timed out" in combined:
        return "Operation timeouts — possible resource or connectivity issue"
    if "permission" in combined or "denied" in combined:
        return "Permission or access issue"
    if "not found" in combined or "missing" in combined:
        return "Missing dependency or resource"
    if "syntax" in combined or "parse" in combined:
        return "Syntax or parsing error in generated content"
    if "test" in combined or "assert" in combined:
        return "Test failures — code changes may introduce regressions"
    if "conflict" in combined:
        return "Merge or resource conflicts"

    cause_map = {
        "code": "Code generation or modification errors",
        "test": "Test execution failures",
        "deploy": "Deployment pipeline issues",
        "config": "Configuration errors",
        "docs": "Documentation generation issues",
        "analyze": "Analysis step failures",
    }
    return cause_map.get(family, "Unknown — requires manual investigation")


def _suggest_remediations(
    patterns: List[Dict[str, Any]],
    causes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Suggest remediation strategies."""
    remediations = []
    for cause in causes:
        fam = cause["family"]
        severity = cause["severity"]

        strategies = []
        if severity == "critical":
            strategies.append(f"Block family '{fam}' from task selection")
            strategies.append("Review and fix underlying infrastructure")
            strategies.append("Add pre-flight checks before attempting this family")
        elif severity == "high":
            strategies.append(f"Add cooldown period for family '{fam}'")
            strategies.append("Review recent changes that may have caused regressions")
            strategies.append("Add validation gates before execution")
        else:
            strategies.append("Monitor for escalation")
            strategies.append("Review failure logs for actionable patterns")

        remediations.append({
            "family": fam,
            "severity": severity,
            "strategies": strategies,
            "priority": "immediate" if severity == "critical" else (
                "soon" if severity == "high" else "low"
            ),
        })

    return remediations


def _extract_lessons(
    project_root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract lessons learned from memory events."""
    failures = _load_events("failure", project_root)
    decisions = _load_events("decision", project_root)
    remediations = _load_events("remediation", project_root)
    saturations = _load_events("saturation", project_root)

    lessons = []

    # Lesson: families that recovered after remediation
    remediated_families = {r.family for r in remediations if r.outcome == "success"}
    for fam in remediated_families:
        lessons.append({
            "type": "recovery",
            "family": fam,
            "lesson": f"Family '{fam}' recovered after remediation — keep monitoring",
        })

    # Lesson: families with high success rate
    success_by_family: Counter = Counter()
    total_by_family: Counter = Counter()
    for d in decisions:
        if d.family:
            total_by_family[d.family] += 1
            if d.outcome == "success":
                success_by_family[d.family] += 1

    for fam, total in total_by_family.items():
        if total >= 3:
            rate = success_by_family.get(fam, 0) / total
            if rate >= 0.8:
                lessons.append({
                    "type": "strength",
                    "family": fam,
                    "lesson": f"Family '{fam}' has {rate:.0%} success rate ({total} attempts)",
                })
            elif rate <= 0.3:
                lessons.append({
                    "type": "weakness",
                    "family": fam,
                    "lesson": f"Family '{fam}' has only {rate:.0%} success rate — needs improvement",
                })

    # Lesson: persistent saturation
    if saturations:
        saturated_fams = {s.family for s in saturations}
        for fam in saturated_fams:
            if fam not in remediated_families:
                lessons.append({
                    "type": "persistent_block",
                    "family": fam,
                    "lesson": f"Family '{fam}' has been saturated without successful remediation",
                })

    return lessons


# ---------------------------------------------------------------------------
# LLM-based analysis
# ---------------------------------------------------------------------------

LLM_ANALYSIS_PROMPT = """You are analyzing operational memory for IGRIS_GPT.

Given the memory data below, produce a JSON analysis.

Output ONLY valid JSON — no markdown, no code fences.

Schema:
{
  "patterns": [{"family": "str", "description": "str", "severity": "str"}],
  "root_causes": [{"family": "str", "cause": "str", "evidence": "str"}],
  "recommendations": [{"action": "str", "priority": "high|medium|low", "reason": "str"}],
  "lessons": [{"lesson": "str", "type": "insight|warning|positive"}]
}

Rules:
- Be concise and actionable
- No secrets or credentials in output
- Advisory only — never suggest auto-execution
- Focus on patterns, not individual events
"""


def _try_llm_analysis(memory_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Try LLM-based analysis. Returns None on failure."""
    prompt = (
        "Memory data:\n"
        f"{json.dumps(memory_summary, indent=2, default=str)}\n\n"
        "Analyze patterns and provide recommendations."
    )

    try:
        response = chat(message=prompt, system_prompt=LLM_ANALYSIS_PROMPT)
        text = response.get("text", "")

        # Try to extract JSON
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        data = json.loads(text)

        # Validate basic structure
        if not isinstance(data, dict):
            return None

        # Redact any secrets that slipped through
        clean = json.dumps(data, default=str)
        clean = redact_secrets(clean)

        return {
            "llm_analysis": json.loads(clean),
            "provider": response.get("provider", "unknown"),
            "model": response.get("model", "unknown"),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_memory(project_root: Optional[str] = None) -> Dict[str, Any]:
    """Full memory analysis combining LLM and deterministic approaches.

    Returns advisory insights — never executes actions.
    """
    t0 = time.monotonic()

    # Gather data
    patterns = _analyze_failure_patterns(project_root)
    causes = _identify_root_causes(patterns)
    remediations = _suggest_remediations(patterns, causes)
    lessons = _extract_lessons(project_root)
    constraints = explain_memory_constraints(project_root)

    deterministic_result = {
        "failure_patterns": patterns,
        "root_causes": causes,
        "remediations": remediations,
        "lessons": lessons,
        "constraints": constraints,
    }

    # Try LLM analysis
    memory_summary = {
        "recent_failures": get_recent_failures(limit=10, project_root=project_root),
        "recent_decisions": get_recent_decisions(limit=10, project_root=project_root),
        "saturated_families": get_saturated_families(project_root),
        "constraints": constraints,
    }

    llm_result = _try_llm_analysis(memory_summary)

    latency_ms = int((time.monotonic() - t0) * 1000)

    result: Dict[str, Any] = {
        "deterministic": deterministic_result,
        "llm_enhanced": llm_result is not None,
        "latency_ms": latency_ms,
        "advisory_only": True,
    }

    if llm_result:
        result["llm_analysis"] = llm_result["llm_analysis"]
        result["llm_provider"] = llm_result["provider"]
        result["llm_model"] = llm_result["model"]

    return result


def get_analysis_summary(project_root: Optional[str] = None) -> Dict[str, Any]:
    """Compact summary of memory analysis for dashboard/chat use."""
    analysis = analyze_memory(project_root)
    det = analysis["deterministic"]

    total_patterns = len(det["failure_patterns"])
    critical = sum(1 for c in det["root_causes"] if c["severity"] == "critical")
    high = sum(1 for c in det["root_causes"] if c["severity"] == "high")

    return {
        "pattern_count": total_patterns,
        "critical_issues": critical,
        "high_issues": high,
        "saturated_families": det["constraints"].get("saturated_families", []),
        "avoid_families": det["constraints"].get("avoid_families", []),
        "lesson_count": len(det["lessons"]),
        "recommendation": det["constraints"].get("recommendation", ""),
        "llm_enhanced": analysis["llm_enhanced"],
        "advisory_only": True,
    }


def get_lessons_learned(project_root: Optional[str] = None) -> Dict[str, Any]:
    """Extract and return lessons learned from memory."""
    lessons = _extract_lessons(project_root)
    return {
        "lessons": lessons,
        "count": len(lessons),
        "types": list(set(l["type"] for l in lessons)) if lessons else [],
        "advisory_only": True,
    }
