from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List

from igris.agent.mission.mission_schema import Mission


@dataclass
class MissionLoopState:
    family: str
    semantic_key: str
    count: int
    saturated: bool
    escalation_required: bool


_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for",
    "il", "lo", "la", "i", "gli", "le", "un", "una", "e", "o", "di", "da", "in",
    "su", "per", "con", "del", "della", "dello", "dei", "delle", "al", "alla",
    "now", "quickly", "rapidamente", "subito",
}

_TOKEN_SYNONYMS = {
    "fix": "fix",
    "fixed": "fix",
    "repair": "fix",
    "resolve": "fix",
    "correggi": "fix",
    "sistema": "fix",
    "bug": "bug",
    "error": "bug",
    "errore": "bug",
    "issue": "bug",
    "planner": "planner",
    "planning": "planner",
    "pianificazione": "planner",
    "missione": "mission",
    "mission": "mission",
    "pipeline": "pipeline",
    "verify": "verify",
    "verifica": "verify",
    "test": "test",
    "architecture": "architecture",
    "architettura": "architecture",
}


def classify_mission_family(mission: Mission) -> str:
    summary = (mission.intent_summary or "").lower()
    if "architecture" in summary:
        return "architecture"
    if "diagnosis" in summary:
        return "diagnosis"
    if "verification" in summary:
        return "verification"
    if "code_change" in summary:
        return "code_change"
    if "planning" in summary:
        return "planning"
    return "mixed"


def _extract_intent_type(mission: Mission) -> str:
    decomp = mission.context_snapshot.get("intent_decomposition", {})
    if isinstance(decomp, dict):
        t = str(decomp.get("intent_type") or "").strip().lower()
        if t:
            return t
    summary = (mission.intent_summary or "").strip().lower()
    if summary.startswith("[") and "]" in summary:
        return summary[1:summary.index("]")].split("|")[0].strip() or "mixed"
    return "mixed"


def _normalize_token(token: str) -> str:
    t = token.lower().strip()
    if not t:
        return ""
    t = _TOKEN_SYNONYMS.get(t, t)
    for suffix in ("ing", "ed", "es", "s", "mente", "zione", "zioni"):
        if len(t) > 5 and t.endswith(suffix):
            t = t[: -len(suffix)]
            break
    return t


def _semantic_tokens(text: str) -> List[str]:
    raw = re.findall(r"[a-zA-Z0-9_./-]+", text.lower())
    out: List[str] = []
    for token in raw:
        if token in _STOPWORDS:
            continue
        if token.isdigit():
            continue
        normalized = _normalize_token(token)
        if normalized and normalized not in _STOPWORDS:
            out.append(normalized)
    return out


def semantic_group_for_mission(mission: Mission) -> str:
    intent = _extract_intent_type(mission)
    family = classify_mission_family(mission)
    tokens = _semantic_tokens((mission.user_input or "").strip())
    unique_sorted = sorted(set(tokens))
    basis = f"{intent}|{family}|{' '.join(unique_sorted)}"
    return basis if basis.strip("|") else f"{intent}|{family}|unknown"


def semantic_key_for_mission(mission: Mission) -> str:
    basis = semantic_group_for_mission(mission)
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()
    return digest[:12]


def evaluate_loop_state(
    mission: Mission,
    counters: Dict[str, int],
    *,
    saturation_threshold: int = 3,
    satisfaction_failures: int = 0,
    escalation_threshold: int = 2,
) -> MissionLoopState:
    family = classify_mission_family(mission)
    key = semantic_key_for_mission(mission)
    count = int(counters.get(family, 0))
    saturated = count >= saturation_threshold
    escalation_required = saturated or satisfaction_failures >= escalation_threshold
    return MissionLoopState(
        family=family,
        semantic_key=key,
        count=count,
        saturated=saturated,
        escalation_required=escalation_required,
    )

