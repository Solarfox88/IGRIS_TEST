from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

from igris.core.smw_sensors import SystemSnapshot


@dataclass
class Pattern:
    name: str
    description: str
    severity: str
    check: Callable[[SystemSnapshot], bool]
    cooldown_seconds: int = 300


@dataclass
class DetectedPattern:
    pattern: Pattern
    snapshot: SystemSnapshot
    detected_at: float
    evidence: str


KNOWN_PATTERNS: List[Pattern] = [
    Pattern("watchdog_cleanup_loop", "Stesso messaggio dirty workspace per >3 cicli consecutivi", "HIGH", lambda s: sum(1 for l in s.recent_log_lines if "dirty workspace detected" in l) >= 3, 600),
    Pattern("port_conflict", "Multipli processi in conflitto sulla porta 7778", "CRITICAL", lambda s: s.port_conflict, 120),
    Pattern("watchdog_idle_anomaly", "Nessun run avviato da >15 min con workspace pulito", "HIGH", lambda s: (not s.active_runs and s.seconds_since_last_run is not None and s.seconds_since_last_run > 900 and not s.tracked_dirty), 900),
    Pattern("zombie_run_suspected", "Run attivo da >2h senza completare — probabile zombie", "HIGH", lambda s: bool(s.active_runs) and s.last_run_started_at is not None and (time.time() - s.last_run_started_at) > 7200, 1800),
    Pattern("model_overkill", "Strong model usato in >60% dei run senza max_steps_ceiling", "LOW", lambda s: s.escalation_rate > 0.6, 3600),
    Pattern("repair_cycle_saturation", "Avg repair cycles >1.8 negli ultimi 20 run", "MEDIUM", lambda s: s.avg_repair_cycles > 1.8, 3600),
    Pattern("untracked_artefact_blocking", "File untracked in root impedisce avvio watchdog", "HIGH", lambda s: bool(s.untracked_files) and not s.active_runs, 300),
    Pattern("igris_process_missing", "Nessun processo IGRIS in ascolto sulla porta 7778", "CRITICAL", lambda s: not s.igris_port_in_use, 60),
]


def detect_patterns(snapshot: SystemSnapshot) -> List[DetectedPattern]:
    out: List[DetectedPattern] = []
    for p in KNOWN_PATTERNS:
        try:
            if p.check(snapshot):
                ev = f"active_runs={len(snapshot.active_runs)} port_in_use={snapshot.igris_port_in_use} dirty={len(snapshot.dirty_files)}"
                out.append(DetectedPattern(pattern=p, snapshot=snapshot, detected_at=time.time(), evidence=ev))
        except Exception:
            continue
    return out


def learn_pattern(name: str, description: str, check_code_str: str, severity: str, project_root: str = ".") -> None:
    p = Path(project_root) / ".igris" / "incident_patterns.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if p.exists():
        try:
            records = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            records = []
    records.append({"name": name, "description": description, "check_code_str": check_code_str, "severity": severity, "learned_at": time.time()})
    p.write_text(json.dumps(records, indent=2), encoding="utf-8")
