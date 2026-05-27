from pathlib import Path

from igris.agent.mission import run_mission_pipeline


def test_orchestrator_runs_and_persists_report(tmp_path: Path):
    mission = run_mission_pipeline(
        user_input="Verifica pipeline missione e genera report",
        repo_view={"paths": ["igris/agent/mission"]},
        command_map={"ACT-001": "echo ok"},
        dry_run=True,
        project_root=str(tmp_path),
    )
    assert mission.id
    assert mission.status in {"completed", "partial", "failed"}
    report = tmp_path / ".igris" / "mission_brain" / "reports" / f"{mission.id}.json"
    assert report.exists()

