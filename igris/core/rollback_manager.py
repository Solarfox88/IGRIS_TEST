"""Rollback Manager for IGRIS_GPT — Epic #42.

Provides backup/snapshot/restore for state-modifying operations.
Each rollback entry is linked to a mission/action and verifiable.

Rollback types:
    file_backup    — copies of files before modification
    config_backup  — config/docker-compose/nginx snapshots
    diff_snapshot  — git diff before commit
    state_snapshot — arbitrary JSON state
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from igris.core.safety import redact_secrets


# ---------------------------------------------------------------------------
# Rollback entry
# ---------------------------------------------------------------------------

ROLLBACK_TYPES = ("file_backup", "config_backup", "diff_snapshot", "state_snapshot")


@dataclass
class RollbackEntry:
    """A single rollback record."""
    id: str = field(default_factory=lambda: f"rb-{uuid.uuid4().hex[:8]}")
    type: str = "file_backup"
    description: str = ""
    mission_id: str = ""
    action_id: str = ""
    trace_id: str = ""
    original_path: str = ""
    backup_path: str = ""
    state_data: Optional[Dict[str, Any]] = None
    rollback_command: str = ""
    applicable: bool = True
    applied: bool = False
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "description": self.description,
            "mission_id": self.mission_id,
            "action_id": self.action_id,
            "trace_id": self.trace_id,
            "original_path": redact_secrets(self.original_path),
            "backup_path": redact_secrets(self.backup_path),
            "rollback_command": redact_secrets(self.rollback_command),
            "applicable": self.applicable,
            "applied": self.applied,
            "created_at": self.created_at,
        }
        if self.state_data is not None:
            d["state_data"] = json.loads(redact_secrets(json.dumps(self.state_data, default=str)))
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RollbackEntry":
        return cls(
            id=data.get("id", f"rb-{uuid.uuid4().hex[:8]}"),
            type=data.get("type", "file_backup"),
            description=data.get("description", ""),
            mission_id=data.get("mission_id", ""),
            action_id=data.get("action_id", ""),
            trace_id=data.get("trace_id", ""),
            original_path=data.get("original_path", ""),
            backup_path=data.get("backup_path", ""),
            state_data=data.get("state_data"),
            rollback_command=data.get("rollback_command", ""),
            applicable=data.get("applicable", True),
            applied=data.get("applied", False),
            created_at=data.get("created_at", ""),
        )


# ---------------------------------------------------------------------------
# Rollback Manager
# ---------------------------------------------------------------------------

class RollbackManager:
    """Manages rollback entries for missions and actions."""

    def __init__(self, project_root: Optional[str] = None):
        import os
        self.project_root = Path(project_root) if project_root else Path(os.environ.get("PROJECT_ROOT", "."))
        self._rollback_dir = self.project_root / ".igris" / "rollback"
        self._rollback_dir.mkdir(parents=True, exist_ok=True)
        self._backup_dir = self._rollback_dir / "backups"
        self._backup_dir.mkdir(parents=True, exist_ok=True)

    def _entry_path(self, entry_id: str) -> Path:
        return self._rollback_dir / f"{entry_id}.json"

    def _save_entry(self, entry: RollbackEntry) -> Path:
        path = self._entry_path(entry.id)
        path.write_text(json.dumps(entry.to_dict(), indent=2, default=str), encoding="utf-8")
        return path

    def _load_entry(self, entry_id: str) -> Optional[RollbackEntry]:
        path = self._entry_path(entry_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return RollbackEntry.from_dict(data)
        except Exception:
            return None

    # -- Backup file --

    def backup_file(
        self,
        file_path: str,
        mission_id: str = "",
        action_id: str = "",
        trace_id: str = "",
        description: str = "",
    ) -> Optional[RollbackEntry]:
        """Create a backup of a file before modification."""
        src = Path(file_path)
        if not src.exists():
            return None

        backup_name = f"{uuid.uuid4().hex[:8]}_{src.name}"
        backup_path = self._backup_dir / backup_name
        shutil.copy2(str(src), str(backup_path))

        entry = RollbackEntry(
            type="file_backup",
            description=description or f"Backup of {src.name}",
            mission_id=mission_id,
            action_id=action_id,
            trace_id=trace_id,
            original_path=str(src),
            backup_path=str(backup_path),
            rollback_command=f"cp {backup_path} {src}",
        )
        self._save_entry(entry)
        return entry

    # -- Config backup --

    def backup_config(
        self,
        config_path: str,
        mission_id: str = "",
        action_id: str = "",
        trace_id: str = "",
    ) -> Optional[RollbackEntry]:
        """Backup a configuration file (docker-compose, nginx, etc.)."""
        return self.backup_file(
            config_path,
            mission_id=mission_id,
            action_id=action_id,
            trace_id=trace_id,
            description=f"Config backup: {Path(config_path).name}",
        )

    # -- Diff snapshot --

    def save_diff_snapshot(
        self,
        diff_content: str,
        mission_id: str = "",
        action_id: str = "",
        trace_id: str = "",
        description: str = "",
    ) -> RollbackEntry:
        """Save a git diff snapshot before commit."""
        entry = RollbackEntry(
            type="diff_snapshot",
            description=description or "Git diff snapshot",
            mission_id=mission_id,
            action_id=action_id,
            trace_id=trace_id,
            state_data={"diff": redact_secrets(diff_content)},
            rollback_command="git stash pop or git revert HEAD",
        )
        self._save_entry(entry)
        return entry

    # -- State snapshot --

    def save_state_snapshot(
        self,
        state: Dict[str, Any],
        mission_id: str = "",
        action_id: str = "",
        trace_id: str = "",
        description: str = "",
    ) -> RollbackEntry:
        """Save an arbitrary state snapshot."""
        entry = RollbackEntry(
            type="state_snapshot",
            description=description or "State snapshot",
            mission_id=mission_id,
            action_id=action_id,
            trace_id=trace_id,
            state_data=state,
        )
        self._save_entry(entry)
        return entry

    # -- Rollback --

    def apply_file_rollback(self, entry_id: str) -> bool:
        """Restore a file from backup."""
        entry = self._load_entry(entry_id)
        if not entry:
            return False
        if entry.type != "file_backup":
            return False
        if not entry.applicable:
            return False

        backup = Path(entry.backup_path)
        original = Path(entry.original_path)
        if not backup.exists():
            entry.applicable = False
            self._save_entry(entry)
            return False

        shutil.copy2(str(backup), str(original))
        entry.applied = True
        self._save_entry(entry)
        return True

    # -- Verify rollback --

    def verify_rollback_applicable(self, entry_id: str) -> Dict[str, Any]:
        """Check if a rollback entry can be applied."""
        entry = self._load_entry(entry_id)
        if not entry:
            return {"applicable": False, "reason": "Entry not found"}

        if entry.applied:
            return {"applicable": False, "reason": "Already applied", "entry_id": entry_id}

        if entry.type == "file_backup":
            backup = Path(entry.backup_path)
            if not backup.exists():
                return {"applicable": False, "reason": "Backup file missing", "entry_id": entry_id}
            return {"applicable": True, "entry_id": entry_id, "type": entry.type}

        if entry.type in ("diff_snapshot", "state_snapshot"):
            return {"applicable": True, "entry_id": entry_id, "type": entry.type,
                    "note": "Manual rollback required — review state_data"}

        return {"applicable": entry.applicable, "entry_id": entry_id}

    # -- List / query --

    def list_entries(
        self,
        mission_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List rollback entries, optionally filtered by mission."""
        entries: List[Dict[str, Any]] = []
        for fp in sorted(self._rollback_dir.glob("rb-*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                if mission_id and data.get("mission_id") != mission_id:
                    continue
                entries.append(data)
                if len(entries) >= limit:
                    break
            except Exception:
                continue
        return entries

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific rollback entry."""
        entry = self._load_entry(entry_id)
        if entry:
            return entry.to_dict()
        return None

    def has_rollback_for_action(self, action_id: str) -> bool:
        """Check if a rollback exists for a given action."""
        for fp in self._rollback_dir.glob("rb-*.json"):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                if data.get("action_id") == action_id:
                    return True
            except Exception:
                continue
        return False
