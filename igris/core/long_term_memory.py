"""
Long-term persistent memory with domain index and rolling summary.

Stores memory entries keyed by domain, supports rolling summary generation
for maintaining a concise compressed view of past events.
"""

from __future__ import annotations

import json
import random
import string
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from igris.core.safety import redact_secrets
from igris.models.config import CONFIG


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """A single memory entry with metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    domain: str = ""
    content: Any = field(default_factory=dict)  # supports str or dict
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    tags: List[str] = field(default_factory=list)
    importance: float = 1.0  # 0.0 (low) to 1.0 (high)


@dataclass
class OTPRecord:
    """A one-time password record for gate override."""
    code: str = ""
    user: str = ""
    created_at: float = field(default_factory=time.time)
    ttl: float = 300.0
    used: bool = False
    physically_approved: bool = False

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl


@dataclass
class DomainIndex:
    """Index of memory entries for a given domain."""
    domain: str = ""
    entries: List[str] = field(default_factory=list)  # list of entry ids
    latest_timestamp: float = 0.0
    entry_count: int = 0


@dataclass
class RollingSummary:
    """Rolling summary of memory entries for a domain."""
    domain: str = ""
    summary: str = ""
    last_updated: float = 0.0
    version: int = 0
    entry_count_since_summary: int = 0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LongTermMemory:
    """Persistent, domain-indexed memory with rolling summary."""

    def __init__(
        self,
        base_path: Optional[Path] = None,
        storage_dir: Optional[str] = None,
    ) -> None:
        if storage_dir is not None:
            self._base_path = Path(storage_dir)
        else:
            self._base_path = base_path or Path(CONFIG.igris_dir) / "memory" / "long_term"
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._entries_file = self._base_path / "entries.json"
        self._index_file = self._base_path / "index.json"
        self._summary_file = self._base_path / "summary.json"
        self._entries: Dict[str, MemoryEntry] = {}
        self._index: Dict[str, DomainIndex] = {}
        self._summaries: Dict[str, RollingSummary] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load state from disk."""
        if self._entries_file.exists():
            with open(self._entries_file, "r") as f:
                raw = json.load(f)
                for k, v in raw.items():
                    self._entries[k] = MemoryEntry(**v)
        if self._index_file.exists():
            with open(self._index_file, "r") as f:
                raw = json.load(f)
                for k, v in raw.items():
                    self._index[k] = DomainIndex(**v)
        if self._summary_file.exists():
            with open(self._summary_file, "r") as f:
                raw = json.load(f)
                for k, v in raw.items():
                    self._summaries[k] = RollingSummary(**v)

    def _save(self) -> None:
        """Save state to disk."""
        # Convert dataclasses to dicts, redact content
        entries_dict = {
            eid: asdict(entry)
            for eid, entry in self._entries.items()
        }
        for e in entries_dict.values():
            raw_content = e["content"]
            e["content"] = redact_secrets(str(raw_content)) if isinstance(raw_content, str) else raw_content

        index_dict = {k: asdict(v) for k, v in self._index.items()}
        summary_dict = {k: asdict(v) for k, v in self._summaries.items()}

        with open(self._entries_file, "w") as f:
            json.dump(entries_dict, f, indent=2, default=str)
        with open(self._index_file, "w") as f:
            json.dump(index_dict, f, indent=2, default=str)
        with open(self._summary_file, "w") as f:
            json.dump(summary_dict, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_entry(self, domain: str, content: Any, source: str = "",
                  tags: Optional[List[str]] = None,
                  importance: float = 1.0) -> MemoryEntry:
        """Add a memory entry for a given domain."""
        entry = MemoryEntry(
            domain=domain,
            content=content,
            source=source,
            tags=tags or [],
            importance=importance
        )
        self._entries[entry.id] = entry

        # Update domain index
        if domain not in self._index:
            self._index[domain] = DomainIndex(domain=domain)
        idx = self._index[domain]
        idx.entries.append(entry.id)
        idx.latest_timestamp = max(idx.latest_timestamp, entry.timestamp)
        idx.entry_count += 1

        # Invalidate summary so it will be regenerated
        if domain in self._summaries:
            self._summaries[domain].entry_count_since_summary += 1

        self._save()
        return entry

    def get_entries(self, domain: str,
                    limit: int = 100,
                    offset: int = 0) -> List[MemoryEntry]:
        """Retrieve entries for a domain, ordered by timestamp descending."""
        if domain not in self._index:
            return []
        entry_ids = self._index[domain].entries
        # Sort by timestamp descending
        sorted_ids = sorted(
            entry_ids,
            key=lambda eid: self._entries[eid].timestamp,
            reverse=True
        )
        selected = sorted_ids[offset:offset+limit]
        return [self._entries[eid] for eid in selected]

    def store(
        self,
        domain: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        importance: float = 1.0,
    ) -> MemoryEntry:
        """Store a memory entry (convenience alias for add_entry)."""
        meta = metadata or {}
        return self.add_entry(
            domain=domain,
            content=content,
            source=str(meta.get("source", "")),
            tags=tags or [],
            importance=importance,
        )

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Return a single entry by its ID."""
        return self._entries.get(entry_id)

    def get_domain_index(self) -> Dict[str, List[str]]:
        """Return mapping of domain → list of entry IDs."""
        return {domain: list(idx.entries) for domain, idx in self._index.items()}

    def get_rolling_summary(self, domain: str, max_entries: int = 10) -> List[MemoryEntry]:
        """Return the most recent *max_entries* entries for *domain*."""
        return self.get_entries(domain, limit=max_entries)

    def search(self, query: str,
               domains: Optional[List[str]] = None,
               limit: int = 50) -> List[MemoryEntry]:
        """Search entries by keyword in content (case-insensitive substring)."""
        query_lower = query.lower()
        results: List[MemoryEntry] = []
        for entry in self._entries.values():
            if domains and entry.domain not in domains:
                continue
            if query_lower in str(entry.content).lower():
                results.append(entry)
        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[:limit]

    def search_entries(self, query: str,
                       domains: Optional[List[str]] = None,
                       limit: int = 50) -> List[MemoryEntry]:
        """Alias for search() for backwards compatibility."""
        return self.search(query, domains=domains, limit=limit)

    def generate_summary(self, domain: str, force: bool = False) -> str:
        """Generate or retrieve a rolling summary for a domain.

        In a production system this would use an LLM; here we produce a
        simple concatenation of recent unique tags and a count of entries.
        """
        if domain not in self._index:
            return ""

        curr = self._summaries.get(domain)
        if curr and not force and curr.entry_count_since_summary < 10:
            return curr.summary

        entries = self.get_entries(domain, limit=100)
        if not entries:
            return ""

        # Build a naive summary: most common tags, date range, entry count
        tag_counts: Dict[str, int] = {}
        for e in entries:
            for t in e.tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1
        top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:5]
        tag_str = ", ".join(f"{tag}({cnt})" for tag, cnt in top_tags)

        start_ts = min(e.timestamp for e in entries)
        end_ts = max(e.timestamp for e in entries)
        summary_text = (
            f"Domain: {domain} | Entries: {len(entries)} | "
            f"From: {start_ts:.2f} To: {end_ts:.2f} | "
            f"Top tags: {tag_str}"
        )

        new_summary = RollingSummary(
            domain=domain,
            summary=summary_text,
            last_updated=time.time(),
            version=(curr.version + 1) if curr else 1,
            entry_count_since_summary=0
        )
        self._summaries[domain] = new_summary
        self._save()
        return summary_text

    def get_summary(self, domain: str) -> Optional[str]:
        """Get current summary for a domain if it exists."""
        if domain in self._summaries:
            return self._summaries[domain].summary
        return None

    def delete_entry(self, entry_id: str) -> bool:
        """Delete a specific entry by its ID."""
        if entry_id not in self._entries:
            return False
        entry = self._entries[entry_id]
        domain = entry.domain
        # Remove from index
        if domain in self._index:
            idx = self._index[domain]
            if entry_id in idx.entries:
                idx.entries.remove(entry_id)
                idx.entry_count -= 1
                if idx.entry_count == 0:
                    del self._index[domain]
        # Remove entry
        del self._entries[entry_id]
        self._save()
        return True

    def clear_domain(self, domain: str) -> bool:
        """Remove all entries for a given domain."""
        if domain not in self._index:
            return False
        for eid in self._index[domain].entries:
            del self._entries[eid]
        del self._index[domain]
        if domain in self._summaries:
            del self._summaries[domain]
        self._save()
        return True


# ---------------------------------------------------------------------------
# MemoryRetriever — contextual and recency-based retrieval
# ---------------------------------------------------------------------------

class MemoryRetriever:
    """High-level retriever that wraps a LongTermMemory instance."""

    def __init__(self, memory: LongTermMemory) -> None:
        self._memory = memory

    def retrieve_contextual(
        self,
        domain: str,
        query: str = "",
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Return entries for *domain* that contain *query* (substring search)."""
        if query:
            return self._memory.search(query, domains=[domain], limit=limit)
        return self._memory.get_entries(domain, limit=limit)

    def retrieve_recent(self, domain: str, limit: int = 10) -> List[MemoryEntry]:
        """Return the *limit* most recent entries for *domain*."""
        return self._memory.get_entries(domain, limit=limit)


# ---------------------------------------------------------------------------
# GateOverride — OTP-based physical approval gate
# ---------------------------------------------------------------------------

class GateOverride:
    """OTP-based gate override with audit trail and physical approval support."""

    def __init__(self) -> None:
        self._records: Dict[str, OTPRecord] = {}
        self._audit: List[Dict[str, Any]] = []

    def generate_otp(self, user: str, ttl: float = 300.0) -> str:
        """Generate a 6-digit OTP for *user* with a given TTL in seconds."""
        code = "".join(random.choices(string.digits, k=6))
        record = OTPRecord(code=code, user=user, ttl=ttl)
        self._records[code] = record
        self._audit.append({
            "action": "generate",
            "user": user,
            "code": code,
            "ts": time.time(),
        })
        return code

    def validate_otp(self, code: str) -> bool:
        """Return True if *code* exists and has not expired."""
        record = self._records.get(code)
        if record is None:
            return False
        return not record.is_expired()

    def get_audit_logs(self) -> List[Dict[str, Any]]:
        """Return the full audit trail."""
        return list(self._audit)

    def request_physical_approval(self, code: str) -> Optional[OTPRecord]:
        """Mark a code as pending physical approval and return its record."""
        record = self._records.get(code)
        if record is None:
            return None
        self._audit.append({"action": "physical_approval_requested", "code": code, "ts": time.time()})
        return record

    def approve_physically(self, code: str) -> None:
        """Mark a code as physically approved."""
        record = self._records.get(code)
        if record is not None:
            record.physically_approved = True
            self._audit.append({"action": "physically_approved", "code": code, "ts": time.time()})

    def is_physically_approved(self, code: str) -> bool:
        """Return True if *code* has been physically approved."""
        record = self._records.get(code)
        return record is not None and record.physically_approved
