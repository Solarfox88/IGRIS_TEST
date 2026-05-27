from __future__ import annotations

import subprocess
from typing import Dict, Iterable, Optional, Set

from igris.agent.mission.mission_schema import Mission, MissionExecutionResult


def execute_mission_actions(
    mission: Mission,
    command_map: Dict[str, str],
    *,
    dry_run: bool = True,
    max_seconds: int = 30,
    previous_commands: Optional[Iterable[str]] = None,
    differentiator: str = "",
) -> Mission:
    """Execute mapped commands and persist execution evidence in mission."""
    seen: Set[str] = set(previous_commands or [])
    results: list[MissionExecutionResult] = []
    for action in mission.actions:
        cmd = command_map.get(action.id, "").strip()
        if not cmd:
            results.append(
                MissionExecutionResult(
                    action_id=action.id,
                    command="",
                    returncode=None,
                    stderr="missing command mapping",
                    success=False,
                )
            )
            continue
        if cmd in seen and not differentiator:
            results.append(
                MissionExecutionResult(
                    action_id=action.id,
                    command=cmd,
                    returncode=None,
                    stderr="blocked blind retry: missing differentiator",
                    success=False,
                )
            )
            continue
        seen.add(cmd)
        if dry_run:
            results.append(
                MissionExecutionResult(
                    action_id=action.id,
                    command=cmd,
                    returncode=0,
                    stdout="dry-run execution simulated",
                    stderr="",
                    success=True,
                    evidence="dry-run",
                )
            )
            continue
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=max_seconds,
            check=False,
        )
        results.append(
            MissionExecutionResult(
                action_id=action.id,
                command=cmd,
                returncode=proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                success=(proc.returncode == 0),
                evidence="process-executed",
            )
        )
    mission.execution_results = results
    return mission

