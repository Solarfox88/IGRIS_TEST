from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Dict

from igris.core.smw_actions import execute_action
from igris.core.smw_diagnosis import diagnose, diagnose_with_llm
from igris.core.smw_patterns import detect_patterns
from igris.core.smw_sensors import take_snapshot
from igris.core.smw_teach import Incident, record_incident, teach_back
from igris.core.smw_pr_review import PRReviewRequest, load_review_results, review_pr, save_review_result
from igris.core.smw_weak_signals import run_all_detectors

_SMW_POLL_SECONDS = 120
_SMW_COOLDOWN_PATTERNS: Dict[str, float] = {}


async def _smw_loop(project_root: str) -> None:
    logger = logging.getLogger("igris.smw")
    cycle_count = 0
    while True:
        try:
            snapshot = await take_snapshot(project_root)
            patterns = detect_patterns(snapshot)
            for detected in patterns:
                name = detected.pattern.name
                last = _SMW_COOLDOWN_PATTERNS.get(name, 0)
                if (detected.detected_at - last) < detected.pattern.cooldown_seconds:
                    continue
                _SMW_COOLDOWN_PATTERNS[name] = detected.detected_at
                d = diagnose(detected, project_root)
                if d.requires_llm:
                    try:
                        d = await diagnose_with_llm(detected, snapshot, project_root)
                    except Exception as _llm_exc:
                        logger.warning("SMW LLM diagnosis failed: %s", _llm_exc)
                actions_applied = []
                for action_name in d.recommended_actions:
                    result = await execute_action(action_name, tier=d.recommended_tier, dry_run=(d.confidence < 0.6), project_root=project_root, pattern_name=name, evidence=detected.evidence, actions_tried=actions_applied)
                    actions_applied.append(action_name)
                    logger.info("SMW: action %s => %s", action_name, result.success)
                await asyncio.sleep(1)
                still_active = any(p.pattern.name == name for p in detect_patterns(await take_snapshot(project_root)))
                outcome = "failed" if still_active else "resolved"
                incident = Incident(uuid.uuid4().hex, name, detected.detected_at, None if still_active else asyncio.get_running_loop().time(), d.root_cause, actions_applied, outcome, detected.evidence)
                record_incident(incident, project_root)
                if outcome == "resolved":
                    await teach_back(incident, project_root)
                else:
                    await execute_action("open_diagnostic_issue", tier=2, dry_run=False, project_root=project_root, pattern_name=name, evidence=detected.evidence, actions_tried=actions_applied)
            cycle_count += 1
            if cycle_count % 10 == 0:
                signals = run_all_detectors(project_root)
                for signal in signals:
                    logger.warning("SMW weak signal: %s - %s | action=%s", signal.name, signal.description, signal.recommended_action)
                    if signal.severity == "ACTION_REQUIRED":
                        await execute_action("open_diagnostic_issue", tier=2, dry_run=False, project_root=project_root, pattern_name=signal.name, evidence=signal.description, actions_tried=[])

            try:
                reviewed = {r.pr_number for r in load_review_results(project_root)}
                out = await asyncio.to_thread(__import__("subprocess").run, ["gh", "pr", "list", "--json", "number,title,headRefName,files,statusCheckRollup"], capture_output=True, text=True, cwd=project_root)
                if out.returncode == 0:
                    prs = __import__("json").loads(out.stdout or "[]")
                    for pr in prs:
                        number = int(pr.get("number", 0))
                        if number in reviewed:
                            continue
                        rollup = pr.get("statusCheckRollup") or []
                        ci_green = bool(rollup) and all((c.get("conclusion") in {"SUCCESS", "NEUTRAL", "SKIPPED"}) for c in rollup if isinstance(c, dict))
                        if not ci_green:
                            continue
                        _diff_proc = await asyncio.to_thread(__import__("subprocess").run, ["gh", "pr", "diff", str(number)], capture_output=True, text=True, cwd=project_root)
                        _pr_diff = _diff_proc.stdout[:8000] if _diff_proc.returncode == 0 else ""
                        req = PRReviewRequest(
                            pr_number=number,
                            pr_title=pr.get("title", ""),
                            pr_diff=_pr_diff,
                            issue_description="",
                            changed_files=[f.get("path", "") for f in (pr.get("files") or []) if isinstance(f, dict)],
                            ci_passed=True,
                            run_id="smw",
                            last_failure_class="",
                            repair_cycles_used=0,
                            max_repair_cycles=1,
                            capability_signals={},
                        )
                        rr = await review_pr(req, project_root)
                        save_review_result(rr, project_root)
                        if rr.approved and rr.confidence > 0.8:
                            await asyncio.to_thread(__import__("subprocess").run, ["gh", "pr", "merge", str(number), "--squash", "--delete-branch"], capture_output=True, text=True, cwd=project_root)
                        elif rr.approved and rr.confidence >= 0.5:
                            logger.warning("SMW merging PR #%s with moderate confidence %.2f", number, rr.confidence)
                            await asyncio.to_thread(__import__("subprocess").run, ["gh", "pr", "merge", str(number), "--squash"], capture_output=True, text=True, cwd=project_root)
                        elif (not rr.approved) and rr.confidence > 0.7:
                            await asyncio.to_thread(__import__("subprocess").run, ["gh", "pr", "comment", str(number), "--body", f"SMW blocked merge: {rr.suggestion}\nConcerns: {rr.concerns}"], capture_output=True, text=True, cwd=project_root)
                            await execute_action("open_diagnostic_issue", tier=2, dry_run=False, project_root=project_root, pattern_name="pr_review_blocked", evidence=f"pr#{number}", actions_tried=[])
                        elif rr.tiebreaker_used and rr.confidence < 0.6:
                            await execute_action("open_diagnostic_issue", tier=2, dry_run=False, project_root=project_root, pattern_name="pr_review_discordance", evidence=f"pr#{number}", actions_tried=[])
            except Exception as exc:
                logger.warning("SMW PR review pass failed: %s", exc)
        except Exception as exc:
            logger.warning("SMW error: %s", exc)
        await asyncio.sleep(_SMW_POLL_SECONDS)


def start_smw(project_root: str) -> asyncio.Task:
    return asyncio.create_task(_smw_loop(project_root))
