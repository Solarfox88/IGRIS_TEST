"""Mission acceptance evidence gate for SelfRepairSupervisor.

After a rank attempt passes baseline tests and produces a non-empty diff,
this gate verifies the implementation is semantically complete — not a stub
that satisfies syntax checks while delivering no real logic.

Public API
----------
check_acceptance_evidence(goal, diff_text, modified_files) -> AcceptanceResult
    Evaluate whether the diff constitutes a genuine implementation.

AcceptanceResult
    .passed: bool
    .missing_evidence: List[str]   — reasons the gate failed
    .found_evidence: List[str]     — positive signals detected
    .required_endpoints: List[str] — endpoint paths extracted from goal
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Stub-pattern detection
# ---------------------------------------------------------------------------

# Patterns that indicate a stub or placeholder body (matched on diff added lines).
_STUB_PATTERNS = [
    # Return a plain dict where ALL values are empty literals
    re.compile(r"^\+\s+return\s+\{(?:\s*['\"][^'\"]+['\"]\s*:\s*(?:\[\]|\{\}|''\s*|\"\")\s*,?\s*)+\}", re.MULTILINE),
    # Placeholder / TODO comments in added lines
    re.compile(r"^\+\s*#\s*(TODO|FIXME|placeholder|Logic to gather|stub|hardcoded)", re.MULTILINE | re.IGNORECASE),
    # Function body is only `pass` or `...`
    re.compile(r"^\+\s+(?:pass|\.\.\.)\s*$", re.MULTILINE),
    # JSONResponse / dict call where ALL key-value pairs are empty literals (single line)
    re.compile(
        r"^\+\s+return\s+\w+\s*\(content\s*=\s*\{(?:\s*['\"][^'\"]+['\"]\s*:\s*(?:\[\s*\]|\{\s*\}|''\s*|\"\"\s*)\s*,?\s*)+\}\s*\)",
        re.MULTILINE,
    ),
]

# Values that, when they appear as the sole value in a returned dict key, signal a stub.
_HARDCODED_EMPTY_VALUES = re.compile(
    r"['\"][^'\"]+['\"]\s*:\s*(?:\[\s*\]|\{\s*\}|''\s*|\"\")",
)


def _is_stub_diff(added_text: str) -> Optional[str]:
    """Return a description of the first stub pattern found, or None if clean."""
    for pat in _STUB_PATTERNS:
        m = pat.search(added_text)
        if m:
            snippet = m.group()[:120].strip()
            return f"Stub pattern detected: {snippet!r}"

    # Heuristic: if all return-dict values for a new function are empty collections,
    # count how many hardcoded-empty assignments appear vs. non-empty ones.
    empty_matches = _HARDCODED_EMPTY_VALUES.findall(added_text)
    if len(empty_matches) >= 3:
        # Three or more empty-value entries in a returned dict strongly suggests a stub.
        return f"Return dict has {len(empty_matches)} hardcoded-empty values: {empty_matches[:3]}"

    return None


# ---------------------------------------------------------------------------
# Endpoint extraction from goal
# ---------------------------------------------------------------------------

# Negative lookbehind ensures we only match '/' that is NOT preceded by a word
# character, so embedded paths like 'igris/web/server.py' do not produce false
# endpoints ('/web/server').
_ENDPOINT_PATTERN = re.compile(r"(?<!\w)(/(?:api/)?[\w\-/]+(?:/\{[\w]+\})*)")

# File extensions that, if present at the end of an extracted path, mean the
# match is a file path reference, not an API endpoint.
_FILE_EXT_RE = re.compile(r"\.\w{1,5}$")


def extract_required_endpoints(goal: str) -> List[str]:
    """Extract API endpoint paths mentioned in the mission goal."""
    raw = _ENDPOINT_PATTERN.findall(goal)
    return [
        ep for ep in raw
        if len(ep) > 3
        and not ep.startswith("//")
        and not _FILE_EXT_RE.search(ep)  # exclude file path fragments
    ]


# ---------------------------------------------------------------------------
# Diff analysis helpers
# ---------------------------------------------------------------------------

def _added_lines(diff_text: str) -> str:
    """Return only the added lines from a unified diff."""
    return "\n".join(
        line for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def _endpoint_in_diff(endpoint: str, diff_text: str) -> bool:
    """Return True if the endpoint path appears among added lines."""
    added = _added_lines(diff_text)
    # Match the path literally (with optional surrounding quotes/parens)
    return endpoint in added


def _endpoint_in_test_files(endpoint: str, modified_files: List[str], diff_text: str) -> bool:
    """Return True if the endpoint appears in added lines of a test file."""
    test_file_in_diff = any(
        f.startswith("tests/") or "test_" in f
        for f in modified_files
    )
    if not test_file_in_diff:
        return False
    # Check added lines in the diff for the endpoint path
    return endpoint in _added_lines(diff_text)


# ---------------------------------------------------------------------------
# AcceptanceResult
# ---------------------------------------------------------------------------

@dataclass
class AcceptanceResult:
    passed: bool
    missing_evidence: List[str] = field(default_factory=list)
    found_evidence: List[str] = field(default_factory=list)
    required_endpoints: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main gate
# ---------------------------------------------------------------------------

def check_acceptance_evidence(
    goal: str,
    diff_text: str,
    modified_files: List[str],
) -> AcceptanceResult:
    """Evaluate whether the diff constitutes a genuine implementation.

    Rules applied (in order):
    1. If goal mentions specific API endpoint paths, each must appear in
       the diff AND in at least one modified test file.
    2. If an endpoint is required but no test file was modified, flag it.
    3. If stub patterns are detected in the added diff lines, flag them.

    A result is only flagged if the goal clearly implies specific deliverables
    (endpoint paths). Generic goals without detectable deliverables pass
    through so that the gate does not over-fire on non-API tasks.
    """
    required_endpoints = extract_required_endpoints(goal)
    missing: List[str] = []
    found: List[str] = []

    if not diff_text.strip():
        # No diff at all — already caught by _rank_passed; gate is irrelevant.
        return AcceptanceResult(passed=True, required_endpoints=required_endpoints)

    added = _added_lines(diff_text)

    # --- Rule 1: stub detection (applies regardless of endpoints) ---
    stub_reason = _is_stub_diff(added)
    if stub_reason:
        missing.append(stub_reason)

    # --- Rules 2+3: endpoint coverage ---
    for ep in required_endpoints:
        if not _endpoint_in_diff(ep, diff_text):
            missing.append(f"Required endpoint '{ep}' not found in diff")
        else:
            found.append(f"Endpoint '{ep}' present in diff")
            if not _endpoint_in_test_files(ep, modified_files, diff_text):
                missing.append(f"Required endpoint '{ep}' has no dedicated test coverage in diff")
            else:
                found.append(f"Endpoint '{ep}' covered by test in diff")

    passed = len(missing) == 0
    return AcceptanceResult(
        passed=passed,
        missing_evidence=missing,
        found_evidence=found,
        required_endpoints=required_endpoints,
    )
