import os
from igris.layers.execution.runner import run_safe_command, CommandError


def test_safe_command_list_files(tmp_path):
    # Create a temporary file in the project root and ensure it appears in ls
    proj_root = tmp_path / "project"
    proj_root.mkdir()
    # Temporarily override PROJECT_ROOT and update CONFIG.project_root
    os.environ["PROJECT_ROOT"] = str(proj_root)
    from igris.models.config import CONFIG as _CONFIG  # reimport to avoid circular
    _CONFIG.project_root = proj_root
    (proj_root / "foo.txt").write_text("hello")
    result = run_safe_command("list_files")
    assert result["returncode"] == 0
    assert "foo.txt" in result["stdout"]


def test_disallowed_command():
    try:
        run_safe_command("rm -rf /")  # type: ignore[arg-type]
    except CommandError as e:
        assert "not allowed" in str(e)