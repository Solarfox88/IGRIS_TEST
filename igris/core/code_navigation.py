"""Code Navigation Tools for IGRIS_GPT — Epic #59.

Provides safe, governed tools for the agent to see and understand
a codebase. All tools enforce path guard, secret guard, and output
limits. Output is structured for consumption by the Context Manager
and Agent Reasoning Loop.

Tools:
    search_code      — search for regex patterns in code files
    find_files       — find files by name/glob pattern
    list_directory   — list directory contents with optional depth
    read_file_range  — read specific line range from a file
    repo_map         — build a lightweight map of the repository
    find_symbol      — simple symbol search (grep-based, no AST)
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from igris.core.safety import (
    check_path_access,
    detect_secret_like_content,
    is_sensitive_filename,
    redact_secrets,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Max results returned per query
MAX_SEARCH_RESULTS = 50
MAX_FILE_RESULTS = 100
MAX_DIR_ENTRIES = 200
MAX_READ_LINES = 500
MAX_REPO_MAP_FILES = 500

# File extensions to search in
CODE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".json",
    ".yaml", ".yml", ".toml", ".cfg", ".ini", ".md", ".txt", ".rst",
    ".sh", ".bash", ".zsh", ".fish",
    ".sql", ".graphql", ".proto",
    ".dockerfile", ".env.example",
    ".xml", ".csv",
})

# Extensions to skip (binary/generated)
SKIP_EXTENSIONS = frozenset({
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin", ".whl",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".avi",
    ".zip", ".tar", ".gz", ".bz2", ".xz",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".pem", ".key", ".p12", ".pfx", ".cert",
    ".egg-info", ".woff", ".woff2", ".ttf", ".eot",
})

# Directories to skip
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", ".tox", ".nox",
    ".venv", "venv", "env",
    ".igris", ".egg-info",
    "dist", "build",
})

# Files to never read (secrets)
SECRET_FILES = frozenset({
    ".env", ".env.local", ".env.production", ".env.development",
    ".env.staging", ".env.test",
    "id_rsa", "id_ed25519", "id_dsa",
    "credentials.json", "service_account.json",
    ".netrc", ".npmrc_with_tokens",
})


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SearchMatch:
    """A single search match."""
    file: str = ""
    line_number: int = 0
    line_content: str = ""
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "line_number": self.line_number,
            "line_content": redact_secrets(self.line_content),
            "context_before": [redact_secrets(l) for l in self.context_before],
            "context_after": [redact_secrets(l) for l in self.context_after],
        }


@dataclass
class NavResult:
    """Standard result from a navigation tool."""
    tool: str = ""
    success: bool = False
    data: Any = None
    error: str = ""
    truncated: bool = False
    total_count: int = 0
    returned_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "tool": self.tool,
            "success": self.success,
            "error": self.error,
            "truncated": self.truncated,
            "total_count": self.total_count,
            "returned_count": self.returned_count,
        }
        if isinstance(self.data, list):
            result["data"] = [
                item.to_dict() if hasattr(item, "to_dict") else item
                for item in self.data
            ]
        elif isinstance(self.data, dict):
            result["data"] = self.data
        elif isinstance(self.data, str):
            result["data"] = redact_secrets(self.data)
        else:
            result["data"] = self.data
        return result


# ---------------------------------------------------------------------------
# Path safety helpers
# ---------------------------------------------------------------------------

def _is_safe_path(path: Path, root: Path) -> bool:
    """Check that path is within root and not a secret file."""
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
        if root_resolved not in resolved.parents and resolved != root_resolved:
            return False
    except Exception:
        return False

    if path.name in SECRET_FILES:
        return False

    return True


def _should_skip_dir(name: str) -> bool:
    """Check if a directory should be skipped."""
    return name in SKIP_DIRS


def _should_skip_file(path: Path) -> bool:
    """Check if a file should be skipped."""
    if path.suffix in SKIP_EXTENSIONS:
        return True
    if path.name in SECRET_FILES:
        return True
    if is_sensitive_filename(path.name):
        return True
    return False


def _is_code_file(path: Path) -> bool:
    """Check if a file is a code file worth searching."""
    if path.suffix in CODE_EXTENSIONS:
        return True
    # Files without extension but with known names
    if path.name in {"Makefile", "Dockerfile", "Procfile", "Vagrantfile",
                      "Rakefile", "Gemfile", "Pipfile", "Brewfile"}:
        return True
    return False


# ---------------------------------------------------------------------------
# Code Navigation Tools
# ---------------------------------------------------------------------------

class CodeNavigator:
    """Safe code navigation tools for the agent.

    All operations are bounded by:
    - Path guard (must be within project root)
    - Secret guard (no .env, no secret files)
    - Output limits (max results, truncation)
    - Secret redaction on all output
    """

    def __init__(self, project_root: Optional[str] = None):
        self.root = Path(project_root) if project_root else Path(
            os.environ.get("PROJECT_ROOT", ".")
        )
        self.root = self.root.resolve()

    def search_code(
        self,
        pattern: str,
        path: Optional[str] = None,
        max_results: int = MAX_SEARCH_RESULTS,
        context_lines: int = 0,
    ) -> NavResult:
        """Search for a regex pattern in code files.

        Args:
            pattern: Regex pattern to search for
            path: Optional subdirectory to search in (relative to root)
            max_results: Maximum number of matches to return
            context_lines: Number of context lines before/after match

        Returns:
            NavResult with list of SearchMatch
        """
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return NavResult(
                tool="search_code",
                success=False,
                error=f"Invalid regex pattern: {e}",
            )

        search_root = self.root
        if path:
            search_root = self.root / path
            if not _is_safe_path(search_root, self.root):
                return NavResult(
                    tool="search_code",
                    success=False,
                    error=f"Path outside project root: {path}",
                )

        matches: List[SearchMatch] = []
        total = 0

        for fpath in self._walk_code_files(search_root):
            try:
                lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            except (OSError, PermissionError):
                continue

            for i, line in enumerate(lines):
                if compiled.search(line):
                    total += 1
                    if len(matches) < max_results:
                        ctx_before = []
                        ctx_after = []
                        if context_lines > 0:
                            start = max(0, i - context_lines)
                            ctx_before = lines[start:i]
                            end = min(len(lines), i + context_lines + 1)
                            ctx_after = lines[i + 1:end]

                        rel_path = str(fpath.relative_to(self.root))
                        matches.append(SearchMatch(
                            file=rel_path,
                            line_number=i + 1,
                            line_content=line,
                            context_before=ctx_before,
                            context_after=ctx_after,
                        ))

        return NavResult(
            tool="search_code",
            success=True,
            data=matches,
            truncated=total > max_results,
            total_count=total,
            returned_count=len(matches),
        )

    def find_files(
        self,
        pattern: str,
        max_results: int = MAX_FILE_RESULTS,
    ) -> NavResult:
        """Find files by name or glob pattern.

        Args:
            pattern: Glob pattern (e.g. "*.py", "test_*.py", "**/*.js")
            max_results: Maximum results to return

        Returns:
            NavResult with list of relative file paths
        """
        results: List[str] = []
        total = 0

        for fpath in self._walk_all_files():
            rel = str(fpath.relative_to(self.root))
            if fnmatch.fnmatch(fpath.name, pattern) or fnmatch.fnmatch(rel, pattern):
                total += 1
                if len(results) < max_results:
                    results.append(rel)

        return NavResult(
            tool="find_files",
            success=True,
            data=results,
            truncated=total > max_results,
            total_count=total,
            returned_count=len(results),
        )

    def list_directory(
        self,
        path: str = ".",
        depth: int = 1,
        max_entries: int = MAX_DIR_ENTRIES,
    ) -> NavResult:
        """List directory contents.

        Args:
            path: Directory path relative to root
            depth: How deep to recurse (1 = immediate children)
            max_entries: Maximum entries to return

        Returns:
            NavResult with list of entries
        """
        target = self.root / path
        if not _is_safe_path(target, self.root):
            return NavResult(
                tool="list_directory",
                success=False,
                error=f"Path outside project root: {path}",
            )

        if not target.is_dir():
            return NavResult(
                tool="list_directory",
                success=False,
                error=f"Not a directory: {path}",
            )

        entries: List[Dict[str, Any]] = []
        total_ref: List[int] = [0]

        self._list_recursive(target, depth, entries, max_entries, total_ref)

        return NavResult(
            tool="list_directory",
            success=True,
            data=entries,
            truncated=len(entries) >= max_entries,
            total_count=total_ref[0],
            returned_count=len(entries),
        )

    def _list_recursive(
        self,
        directory: Path,
        remaining_depth: int,
        entries: List[Dict[str, Any]],
        max_entries: int,
        total_ref: List[int],
    ) -> None:
        """Recursively list directory contents."""
        if remaining_depth <= 0 or len(entries) >= max_entries:
            return

        try:
            children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except (OSError, PermissionError):
            return

        for child in children:
            if len(entries) >= max_entries:
                break

            if child.is_dir():
                if _should_skip_dir(child.name):
                    continue
                rel = str(child.relative_to(self.root))
                entry = {"name": child.name, "type": "directory", "path": rel}
                entries.append(entry)
                total_ref[0] += 1

                if remaining_depth > 1:
                    self._list_recursive(child, remaining_depth - 1, entries, max_entries, total_ref)
            else:
                if _should_skip_file(child):
                    continue
                rel = str(child.relative_to(self.root))
                try:
                    size = child.stat().st_size
                except OSError:
                    size = 0
                entry = {
                    "name": child.name,
                    "type": "file",
                    "path": rel,
                    "size": size,
                }
                entries.append(entry)
                total_ref[0] += 1

    def read_file_range(
        self,
        path: str,
        start: int = 1,
        end: Optional[int] = None,
        max_lines: int = MAX_READ_LINES,
    ) -> NavResult:
        """Read specific lines from a file.

        Args:
            path: File path relative to root
            start: Start line (1-based, inclusive)
            end: End line (1-based, inclusive). None = start + max_lines
            max_lines: Maximum lines to return

        Returns:
            NavResult with file content and metadata
        """
        target = self.root / path
        if not _is_safe_path(target, self.root):
            return NavResult(
                tool="read_file_range",
                success=False,
                error=f"Path outside project root or secret file: {path}",
            )

        if not target.is_file():
            return NavResult(
                tool="read_file_range",
                success=False,
                error=f"File not found: {path}",
            )

        if target.suffix in SKIP_EXTENSIONS:
            return NavResult(
                tool="read_file_range",
                success=False,
                error=f"Binary or unsupported file type: {target.suffix}",
            )

        try:
            all_lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except (OSError, PermissionError) as e:
            return NavResult(
                tool="read_file_range",
                success=False,
                error=f"Cannot read file: {e}",
            )

        total_lines = len(all_lines)
        start_idx = max(0, start - 1)
        if end is not None:
            end_idx = min(total_lines, end)
        else:
            end_idx = min(total_lines, start_idx + max_lines)

        # Enforce max lines
        if end_idx - start_idx > max_lines:
            end_idx = start_idx + max_lines

        selected = all_lines[start_idx:end_idx]
        # Redact secrets in content
        content = "\n".join(redact_secrets(line) for line in selected)

        return NavResult(
            tool="read_file_range",
            success=True,
            data={
                "path": path,
                "content": content,
                "start_line": start_idx + 1,
                "end_line": end_idx,
                "total_lines": total_lines,
            },
            truncated=end_idx < total_lines,
            total_count=total_lines,
            returned_count=end_idx - start_idx,
        )

    def repo_map(self, max_files: int = MAX_REPO_MAP_FILES) -> NavResult:
        """Build a lightweight map of the repository.

        Returns a tree-like structure showing directories and files
        with their types (based on extension).

        Returns:
            NavResult with repo structure
        """
        structure: Dict[str, Any] = {
            "root": str(self.root.name),
            "directories": [],
            "file_count": 0,
            "dir_count": 0,
            "languages": {},
        }

        files_seen = 0
        dirs_seen = set()

        for fpath in self._walk_all_files():
            files_seen += 1
            if files_seen > max_files:
                break

            # Track directory
            parent = str(fpath.parent.relative_to(self.root))
            if parent not in dirs_seen:
                dirs_seen.add(parent)

            # Track language by extension
            ext = fpath.suffix.lower()
            if ext:
                structure["languages"][ext] = structure["languages"].get(ext, 0) + 1

        structure["file_count"] = files_seen
        structure["dir_count"] = len(dirs_seen)
        structure["directories"] = sorted(dirs_seen)[:100]  # top 100 dirs

        return NavResult(
            tool="repo_map",
            success=True,
            data=structure,
            truncated=files_seen >= max_files,
            total_count=files_seen,
            returned_count=files_seen,
        )

    def find_symbol(
        self,
        symbol: str,
        path: Optional[str] = None,
        max_results: int = MAX_SEARCH_RESULTS,
    ) -> NavResult:
        """Find a symbol (function, class, variable) by name.

        Simple grep-based search for common definition patterns.
        Not AST-based — works across languages without parsing.

        Args:
            symbol: Symbol name to search for
            path: Optional subdirectory to search in
            max_results: Maximum results to return

        Returns:
            NavResult with list of SearchMatch for definitions
        """
        # Build pattern for common definition styles
        escaped = re.escape(symbol)
        patterns = [
            rf"^\s*(def|class|async\s+def)\s+{escaped}\s*[\(:]",  # Python
            rf"^\s*(function|const|let|var|export)\s+{escaped}\s*[\(=]",  # JS/TS
            rf"^\s*(pub\s+)?(fn|struct|enum|trait|impl)\s+{escaped}",  # Rust
            rf"^\s*(func|type|var|const)\s+{escaped}",  # Go
        ]

        combined = "|".join(f"(?:{p})" for p in patterns)

        try:
            compiled = re.compile(combined, re.MULTILINE)
        except re.error:
            # Fallback to simple word boundary search
            compiled = re.compile(rf"\b{escaped}\b")

        search_root = self.root
        if path:
            search_root = self.root / path
            if not _is_safe_path(search_root, self.root):
                return NavResult(
                    tool="find_symbol",
                    success=False,
                    error=f"Path outside project root: {path}",
                )

        matches: List[SearchMatch] = []
        total = 0

        for fpath in self._walk_code_files(search_root):
            try:
                lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            except (OSError, PermissionError):
                continue

            for i, line in enumerate(lines):
                if compiled.search(line):
                    total += 1
                    if len(matches) < max_results:
                        rel_path = str(fpath.relative_to(self.root))
                        matches.append(SearchMatch(
                            file=rel_path,
                            line_number=i + 1,
                            line_content=line,
                        ))

        return NavResult(
            tool="find_symbol",
            success=True,
            data=matches,
            truncated=total > max_results,
            total_count=total,
            returned_count=len(matches),
        )

    def discover_fastapi_routes(
        self,
        max_results: int = MAX_SEARCH_RESULTS,
    ) -> NavResult:
        """Discover FastAPI route files and endpoints.

        Prioritises files containing FastAPI(), @app.get, @app.post,
        @app.put, @app.delete, @router, APIRouter and similar patterns.
        Also gives bonus score to known conventional paths like
        ``igris/web/server.py``, ``app/main.py``, etc.

        Returns:
            NavResult with list of dicts:
                {"file": str, "routes": [{"line": int, "content": str}], "score": float}
        """
        fastapi_patterns = re.compile(
            r"(FastAPI\s*\(|APIRouter\s*\(|@app\.(get|post|put|delete|patch|options)"
            r"|@router\.(get|post|put|delete|patch|options)"
            r"|\.include_router|\.add_api_route)",
            re.IGNORECASE,
        )

        # Known conventional FastAPI file paths (relative)
        preferred_paths = {
            "igris/web/server.py",
            "app/main.py",
            "app/server.py",
            "main.py",
            "server.py",
        }

        discoveries: List[Dict[str, Any]] = []

        for fpath in self._walk_code_files(self.root):
            if fpath.suffix != ".py":
                continue

            try:
                lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            except (OSError, PermissionError):
                continue

            route_hits: List[Dict[str, Any]] = []
            for i, line in enumerate(lines):
                if fastapi_patterns.search(line):
                    route_hits.append({
                        "line": i + 1,
                        "content": redact_secrets(line.strip()),
                    })

            if not route_hits:
                continue

            rel = str(fpath.relative_to(self.root))
            score = len(route_hits) * 0.1  # more routes = higher score
            if rel in preferred_paths:
                score += 1.0
            if "server" in rel.lower() or "router" in rel.lower():
                score += 0.3

            discoveries.append({
                "file": rel,
                "routes": route_hits[:20],  # cap per file
                "score": round(score, 2),
            })

        # Sort by score descending
        discoveries.sort(key=lambda d: d["score"], reverse=True)
        total = len(discoveries)
        discoveries = discoveries[:max_results]

        return NavResult(
            tool="discover_fastapi_routes",
            success=True,
            data=discoveries,
            truncated=total > max_results,
            total_count=total,
            returned_count=len(discoveries),
        )

    def discover_tests(
        self,
        max_results: int = MAX_FILE_RESULTS,
    ) -> NavResult:
        """Discover test files and test patterns in the repository.

        Finds files matching common test conventions:
        - test_*.py, *_test.py
        - tests/**/*.py
        - Files containing TestClient, pytest, unittest

        Returns:
            NavResult with list of dicts:
                {"file": str, "type": str, "indicators": [str]}
        """
        test_file_patterns = re.compile(
            r"(^test_.*\.py$|.*_test\.py$|^conftest\.py$)",
        )
        test_dir_names = {"tests", "test", "testing"}

        # Content indicators for test files
        test_content_patterns = re.compile(
            r"(import\s+pytest|from\s+pytest|import\s+unittest"
            r"|from\s+unittest|TestClient|@pytest\.(mark|fixture)"
            r"|class\s+Test\w+|def\s+test_\w+|/api/ping)",
        )

        discoveries: List[Dict[str, Any]] = []

        for fpath in self._walk_code_files(self.root):
            if fpath.suffix != ".py":
                continue

            rel = str(fpath.relative_to(self.root))
            indicators: List[str] = []
            test_type = ""

            # Check filename pattern
            if test_file_patterns.match(fpath.name):
                indicators.append(f"filename:{fpath.name}")
                test_type = "test_file"

            # Check if in a test directory
            parts = Path(rel).parts
            if any(p in test_dir_names for p in parts):
                indicators.append("in_test_directory")
                if not test_type:
                    test_type = "test_dir_file"

            # Check content for test patterns
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")[:5000]
                content_hits = test_content_patterns.findall(content)
                if content_hits:
                    for hit in content_hits[:5]:
                        indicators.append(f"content:{hit}")
                    if not test_type:
                        test_type = "test_content"
            except (OSError, PermissionError):
                pass

            if indicators:
                discoveries.append({
                    "file": rel,
                    "type": test_type,
                    "indicators": indicators,
                })

        total = len(discoveries)
        discoveries = discoveries[:max_results]

        return NavResult(
            tool="discover_tests",
            success=True,
            data=discoveries,
            truncated=total > max_results,
            total_count=total,
            returned_count=len(discoveries),
        )

    # -- Internal helpers --

    def _walk_code_files(self, start: Path) -> List[Path]:
        """Walk directory tree and yield code files."""
        results: List[Path] = []
        if not start.exists():
            return results

        if start.is_file():
            if _is_code_file(start) and not _should_skip_file(start):
                results.append(start)
            return results

        for dirpath, dirnames, filenames in os.walk(start):
            # Filter out skip dirs in-place
            dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]

            for fname in filenames:
                fpath = Path(dirpath) / fname
                if _is_code_file(fpath) and not _should_skip_file(fpath):
                    if _is_safe_path(fpath, self.root):
                        results.append(fpath)

        return results

    def _walk_all_files(self) -> List[Path]:
        """Walk directory tree and yield all non-skip files."""
        results: List[Path] = []

        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]

            for fname in filenames:
                fpath = Path(dirpath) / fname
                if not _should_skip_file(fpath) and _is_safe_path(fpath, self.root):
                    results.append(fpath)

        return results
