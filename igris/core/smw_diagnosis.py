from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import List

from igris.core.smw_patterns import DetectedPattern
from igris.core.smw_sensors import SystemSnapshot


@dataclass
class Diagnosis:
    pattern_name: str
    root_cause: str
    confidence: float
    recommended_tier: int
    recommended_actions: List[str]
    evidence: str
    requires_llm: bool


def diagnose(detected: DetectedPattern, project_root: str) -> Diagnosis:
    name = detected.pattern.name
    mapping = {
        "watchdog_cleanup_loop": ("untracked file non coperto da git clean paths", 0.95, 1, ["git_clean_root", "open_diagnostic_issue"], False),
        "port_conflict": ("processo stale su porta 7778", 0.95, 2, ["kill_stale_process", "wait_port_free"], False),
        "watchdog_idle_anomaly": ("watchdog bloccato senza run attivo", 0.8, 2, ["git_clean_root", "check_issue_list", "restart_watchdog_cycle"], False),
        "zombie_run_suspected": ("run zombie in RUN_STORE blocca il watchdog", 0.9, 2, ["restart_watchdog_cycle", "open_diagnostic_issue"], False),
        "untracked_artefact_blocking": ("artefatto non committato da run precedente", 0.9, 1, ["git_clean_root"], False),
        "igris_process_missing": ("processo non in ascolto", 0.9, 3, ["restart_igris_service"], False),
    }
    if name in mapping:
        rc, conf, tier, acts, req = mapping[name]
        return Diagnosis(name, rc, conf, tier, acts, detected.evidence, req)
    return Diagnosis(name, "pattern non riconosciuto", 0.4, 1, ["open_diagnostic_issue"], detected.evidence, True)


async def diagnose_with_llm(detected: DetectedPattern, snapshot: SystemSnapshot, project_root: str) -> Diagnosis:
    cmd = os.getenv("IGRIS_API_HELPER_COMMAND", "")
    if not cmd:
        return diagnose(detected, project_root)
    payload = {"model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"), "max_tokens": 400, "packet": {"pattern": detected.pattern.name, "evidence": detected.evidence, "snapshot": snapshot.__dict__}}
    try:
        proc = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: subprocess.run(shlex.split(cmd), input=json.dumps(payload), capture_output=True, text=True, cwd=project_root, timeout=45),
        )
        if proc.returncode != 0:
            return diagnose(detected, project_root)
        data = json.loads(proc.stdout or "{}")
        return Diagnosis(detected.pattern.name, str(data.get("diagnosis", "llm diagnosis unavailable")), float(data.get("confidence", 0.5)), 2, [str(data.get("suggested_repair_strategy", "open_diagnostic_issue"))], detected.evidence, False)
    except Exception:
        return diagnose(detected, project_root)
