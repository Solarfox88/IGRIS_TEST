"""Tests for Code Navigation Tools — Epic #59.

Validates safe, governed code navigation for the agent reasoning loop.
"""

import os
import tempfile
import pytest
from pathlib import Path

from igris.core.code_navigation import (
    CodeNavigator,
    NavResult,
    SearchMatch,
    SECRET_FILES,
    SKIP_DIRS,
    SKIP_EXTENSIONS,
    CODE_EXTENSIONS,
    _is_safe_path,
    _should_skip_dir,
    _should_skip_file,
    _is_code_file,
)


# ---------------------------------------------------------------------------
# Fixture: temporary project directory
# ---------------------------------------------------------------------------

@pytest.fixture
def sandbox(tmp_path):
    """Create a small project structure for testing."""
    # Source files
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("def hello():\n    return 'world'\n\ndef greet(name):\n    return f'Hello {name}'\n")
    (src / "utils.py").write_text("import os\n\ndef get_path():\n    return os.getcwd()\n")
    (src / "config.json").write_text('{"debug": true, "port": 8080}\n')

    # Tests
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_main.py").write_text("def test_hello():\n    assert hello() == 'world'\n")

    # Docs
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "README.md").write_text("# Project\n\nA sample project.\n")

    # Root files
    (tmp_path / "README.md").write_text("# Root README\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")

    # Secret files that should be blocked
    (tmp_path / ".env").write_text("SECRET_KEY=supersecret123456789012345678901234567890\n")
    (tmp_path / ".env.local").write_text("LOCAL_SECRET=abc\n")

    # Binary file that should be skipped
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    # Skip directories
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "main.cpython-312.pyc").write_bytes(b"\x00")

    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n")

    return tmp_path


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

class TestPathSafety:
    """Test path guards and safety checks."""

    def test_safe_path_within_root(self, sandbox):
        assert _is_safe_path(sandbox / "src" / "main.py", sandbox)

    def test_unsafe_path_outside_root(self, sandbox):
        assert not _is_safe_path(Path("/etc/passwd"), sandbox)

    def test_secret_file_blocked(self, sandbox):
        assert not _is_safe_path(sandbox / ".env", sandbox)

    def test_env_local_blocked(self, sandbox):
        assert not _is_safe_path(sandbox / ".env.local", sandbox)

    def test_skip_dir_detection(self):
        assert _should_skip_dir("__pycache__")
        assert _should_skip_dir(".git")
        assert _should_skip_dir("node_modules")
        assert not _should_skip_dir("src")

    def test_skip_file_detection(self, sandbox):
        assert _should_skip_file(sandbox / "image.png")
        assert _should_skip_file(sandbox / ".env")
        assert not _should_skip_file(sandbox / "src" / "main.py")

    def test_code_file_detection(self, sandbox):
        assert _is_code_file(sandbox / "src" / "main.py")
        assert _is_code_file(sandbox / "config.json")
        assert not _is_code_file(sandbox / "image.png")


# ---------------------------------------------------------------------------
# search_code
# ---------------------------------------------------------------------------

class TestSearchCode:
    """Test code search functionality."""

    def test_search_finds_pattern(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.search_code("def hello")
        assert result.success is True
        assert result.total_count >= 1
        matches = result.data
        assert any(m.file == "src/main.py" for m in matches)

    def test_search_with_regex(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.search_code(r"def \w+\(")
        assert result.success is True
        assert result.total_count >= 2  # hello and greet

    def test_search_in_subdirectory(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.search_code("def", path="src")
        assert result.success is True
        # Should only find results in src/
        for m in result.data:
            assert m.file.startswith("src/")

    def test_search_with_context_lines(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.search_code("def hello", context_lines=1)
        assert result.success is True
        assert len(result.data) >= 1
        match = result.data[0]
        assert len(match.context_after) >= 1  # line after def hello

    def test_search_invalid_regex(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.search_code("[invalid")
        assert result.success is False
        assert "Invalid regex" in result.error

    def test_search_outside_root(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.search_code("test", path="/etc")
        assert result.success is False
        assert "outside" in result.error.lower()

    def test_search_no_results(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.search_code("xyznonexistent123")
        assert result.success is True
        assert result.total_count == 0

    def test_search_respects_max_results(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.search_code(".", max_results=2)
        assert result.success is True
        assert result.returned_count <= 2

    def test_search_skips_secret_files(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.search_code("SECRET")
        # Should NOT find anything from .env files
        for m in result.data:
            assert ".env" not in m.file

    def test_search_skips_git_directory(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.search_code("ref:")
        for m in result.data:
            assert ".git" not in m.file

    def test_search_output_redacts_secrets(self, sandbox):
        # Write a file with a secret pattern in code
        code_file = sandbox / "src" / "has_secret.py"
        fake_key = "sk-" + "a" * 30
        code_file.write_text(f'API_KEY = "{fake_key}"\n')
        nav = CodeNavigator(str(sandbox))
        result = nav.search_code("API_KEY")
        assert result.success is True
        for m in result.data:
            d = m.to_dict()
            assert fake_key not in d["line_content"]
            assert "REDACTED" in d["line_content"]


# ---------------------------------------------------------------------------
# find_files
# ---------------------------------------------------------------------------

class TestFindFiles:
    """Test file finding functionality."""

    def test_find_python_files(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.find_files("*.py")
        assert result.success is True
        assert result.total_count >= 3  # main.py, utils.py, test_main.py
        assert any("main.py" in f for f in result.data)

    def test_find_test_files(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.find_files("test_*.py")
        assert result.success is True
        assert any("test_main.py" in f for f in result.data)

    def test_find_markdown_files(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.find_files("*.md")
        assert result.success is True
        assert result.total_count >= 2  # README.md in root and docs

    def test_find_no_secret_files(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.find_files("*.env*")
        # .env files should be filtered out
        for f in result.data:
            assert ".env" not in Path(f).name or ".env.example" in f

    def test_find_no_binary_files(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.find_files("*.png")
        assert result.total_count == 0

    def test_find_respects_max_results(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.find_files("*", max_results=3)
        assert result.returned_count <= 3

    def test_find_nonexistent_pattern(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.find_files("*.nonexistent")
        assert result.success is True
        assert result.total_count == 0


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------

class TestListDirectory:
    """Test directory listing functionality."""

    def test_list_root(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.list_directory(".")
        assert result.success is True
        names = {e["name"] for e in result.data}
        assert "src" in names
        assert "tests" in names
        assert "docs" in names
        assert "README.md" in names

    def test_list_subdirectory(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.list_directory("src")
        assert result.success is True
        names = {e["name"] for e in result.data}
        assert "main.py" in names
        assert "utils.py" in names

    def test_list_with_depth(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.list_directory(".", depth=2)
        assert result.success is True
        # Should include files from subdirectories
        paths = {e.get("path", "") for e in result.data}
        assert any("src" in p for p in paths)

    def test_list_skips_hidden_dirs(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.list_directory(".")
        names = {e["name"] for e in result.data}
        assert "__pycache__" not in names
        assert ".git" not in names

    def test_list_skips_secret_files(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.list_directory(".")
        names = {e["name"] for e in result.data}
        assert ".env" not in names
        assert ".env.local" not in names

    def test_list_includes_file_type_and_size(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.list_directory("src")
        for entry in result.data:
            assert "type" in entry
            assert "name" in entry
            if entry["type"] == "file":
                assert "size" in entry

    def test_list_outside_root_blocked(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.list_directory("/etc")
        assert result.success is False
        assert "outside" in result.error.lower() or "not a directory" in result.error.lower()

    def test_list_nonexistent_directory(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.list_directory("nonexistent")
        assert result.success is False

    def test_list_respects_max_entries(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.list_directory(".", depth=3, max_entries=5)
        assert result.returned_count <= 5


# ---------------------------------------------------------------------------
# read_file_range
# ---------------------------------------------------------------------------

class TestReadFileRange:
    """Test file reading functionality."""

    def test_read_entire_file(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.read_file_range("src/main.py")
        assert result.success is True
        assert "def hello" in result.data["content"]
        assert result.data["total_lines"] >= 4

    def test_read_specific_range(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.read_file_range("src/main.py", start=1, end=2)
        assert result.success is True
        assert result.data["start_line"] == 1
        assert result.data["end_line"] == 2
        assert result.returned_count == 2

    def test_read_nonexistent_file(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.read_file_range("nonexistent.py")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_read_secret_file_blocked(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.read_file_range(".env")
        assert result.success is False
        assert "secret" in result.error.lower() or "outside" in result.error.lower()

    def test_read_binary_file_blocked(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.read_file_range("image.png")
        assert result.success is False
        assert "binary" in result.error.lower() or "unsupported" in result.error.lower()

    def test_read_outside_root_blocked(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.read_file_range("/etc/passwd")
        assert result.success is False

    def test_read_redacts_secrets_in_content(self, sandbox):
        code_file = sandbox / "src" / "config_with_secret.py"
        fake_key = "sk-" + "b" * 30
        code_file.write_text(f'KEY = "{fake_key}"\n')
        nav = CodeNavigator(str(sandbox))
        result = nav.read_file_range("src/config_with_secret.py")
        assert result.success is True
        assert fake_key not in result.data["content"]
        assert "REDACTED" in result.data["content"]

    def test_read_respects_max_lines(self, sandbox):
        # Create a long file
        long_file = sandbox / "src" / "long.py"
        long_file.write_text("\n".join(f"line_{i} = {i}" for i in range(1000)))
        nav = CodeNavigator(str(sandbox))
        result = nav.read_file_range("src/long.py", max_lines=10)
        assert result.success is True
        assert result.returned_count <= 10

    def test_read_output_has_metadata(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.read_file_range("src/main.py")
        assert "path" in result.data
        assert "content" in result.data
        assert "start_line" in result.data
        assert "end_line" in result.data
        assert "total_lines" in result.data


# ---------------------------------------------------------------------------
# repo_map
# ---------------------------------------------------------------------------

class TestRepoMap:
    """Test repository mapping."""

    def test_repo_map_structure(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.repo_map()
        assert result.success is True
        data = result.data
        assert "root" in data
        assert "file_count" in data
        assert "dir_count" in data
        assert "languages" in data
        assert "directories" in data

    def test_repo_map_counts_files(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.repo_map()
        assert result.data["file_count"] >= 5  # main.py, utils.py, test_main.py, README.md, etc.

    def test_repo_map_tracks_languages(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.repo_map()
        langs = result.data["languages"]
        assert ".py" in langs
        assert langs[".py"] >= 3

    def test_repo_map_skips_secrets(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.repo_map()
        dirs = result.data["directories"]
        for d in dirs:
            assert ".git" not in d


# ---------------------------------------------------------------------------
# find_symbol
# ---------------------------------------------------------------------------

class TestFindSymbol:
    """Test symbol finding."""

    def test_find_python_function(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.find_symbol("hello")
        assert result.success is True
        assert result.total_count >= 1
        assert any("def hello" in m.line_content for m in result.data)

    def test_find_python_function_greet(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.find_symbol("greet")
        assert result.success is True
        assert any("def greet" in m.line_content for m in result.data)

    def test_find_in_subdirectory(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.find_symbol("hello", path="src")
        assert result.success is True
        for m in result.data:
            assert m.file.startswith("src/")

    def test_find_nonexistent_symbol(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.find_symbol("xyz_nonexistent_symbol_123")
        assert result.success is True
        assert result.total_count == 0

    def test_find_outside_root_blocked(self, sandbox):
        nav = CodeNavigator(str(sandbox))
        result = nav.find_symbol("test", path="/etc")
        assert result.success is False


# ---------------------------------------------------------------------------
# NavResult serialization
# ---------------------------------------------------------------------------

class TestNavResult:
    """Test NavResult serialization."""

    def test_to_dict_basic(self):
        r = NavResult(tool="test", success=True, data=["a", "b"])
        d = r.to_dict()
        assert d["tool"] == "test"
        assert d["success"] is True
        assert d["data"] == ["a", "b"]

    def test_to_dict_with_search_matches(self):
        m = SearchMatch(file="foo.py", line_number=10, line_content="def test():")
        r = NavResult(tool="search_code", success=True, data=[m])
        d = r.to_dict()
        assert len(d["data"]) == 1
        assert d["data"][0]["file"] == "foo.py"

    def test_to_dict_error(self):
        r = NavResult(tool="test", success=False, error="bad input")
        d = r.to_dict()
        assert d["success"] is False
        assert d["error"] == "bad input"


# ---------------------------------------------------------------------------
# Integration: real repo navigation
# ---------------------------------------------------------------------------

class TestRealRepoNavigation:
    """Test navigation on the actual IGRIS_GPT repo."""

    def test_search_igris_code(self):
        """Can find code patterns in the real repo."""
        nav = CodeNavigator(str(Path(__file__).parent.parent))
        result = nav.search_code("def create_app", max_results=5)
        assert result.success is True
        assert result.total_count >= 1

    def test_find_python_files_real(self):
        """Can find Python files in the real repo."""
        nav = CodeNavigator(str(Path(__file__).parent.parent))
        result = nav.find_files("*.py", max_results=10)
        assert result.success is True
        assert result.total_count >= 10

    def test_read_real_file(self):
        """Can read a real file from the repo."""
        nav = CodeNavigator(str(Path(__file__).parent.parent))
        result = nav.read_file_range("README.md", start=1, end=5)
        assert result.success is True
        assert result.data["content"]

    def test_repo_map_real(self):
        """Can build a map of the real repo."""
        nav = CodeNavigator(str(Path(__file__).parent.parent))
        result = nav.repo_map()
        assert result.success is True
        assert result.data["file_count"] > 20

    def test_env_never_readable(self):
        """Verify .env is never readable even in real repo."""
        nav = CodeNavigator(str(Path(__file__).parent.parent))
        result = nav.read_file_range(".env")
        assert result.success is False

    def test_find_symbol_real(self):
        """Can find real symbols in the repo."""
        nav = CodeNavigator(str(Path(__file__).parent.parent))
        result = nav.find_symbol("create_app")
        assert result.success is True
        assert result.total_count >= 1
