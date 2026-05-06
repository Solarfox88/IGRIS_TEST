"""Environment diagnostics for IGRIS_GPT — ``igris doctor``.

Performs comprehensive checks of the runtime environment:
- Python version & venv
- Dependencies
- FastAPI server reachability
- Ollama / local LLM
- OpenAI key presence (without exposing it)
- Git
- Docker
- SSH
- Port availability
- Workspace/project paths
- Config files (.env existence, config.json)
- Permissions

Every check produces a :class:`DoctorCheck` with status, detail and
optional ``fix_suggestion``.  The full report is a :class:`DoctorReport`.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from igris.core.safety import redact_secrets


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class DoctorCheck:
    """Single environment check result."""
    name: str
    category: str  # python | venv | deps | server | ollama | openai | git | docker | ssh | ports | permissions | config | workspace
    status: str  # ok | warning | error | skipped
    detail: str = ""
    fix_suggestion: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "category": self.category,
            "status": self.status,
            "detail": redact_secrets(self.detail),
        }
        if self.fix_suggestion:
            d["fix_suggestion"] = self.fix_suggestion
        if self.meta:
            d["meta"] = self.meta
        return d


@dataclass
class DoctorReport:
    """Aggregated doctor report."""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    checks: List[DoctorCheck] = field(default_factory=list)
    overall: str = "ok"  # ok | warning | error

    def _compute_overall(self) -> str:
        if any(c.status == "error" for c in self.checks):
            return "error"
        if any(c.status == "warning" for c in self.checks):
            return "warning"
        return "ok"

    def to_dict(self) -> Dict[str, Any]:
        self.overall = self._compute_overall()
        by_status = {"ok": 0, "warning": 0, "error": 0, "skipped": 0}
        for c in self.checks:
            by_status[c.status] = by_status.get(c.status, 0) + 1
        return {
            "timestamp": self.timestamp,
            "overall": self.overall,
            "summary": by_status,
            "checks": [c.to_dict() for c in self.checks],
            "total_checks": len(self.checks),
        }

    def to_markdown(self) -> str:
        self.overall = self._compute_overall()
        lines = [
            "# IGRIS Doctor Report",
            f"**Timestamp:** {self.timestamp}",
            f"**Overall:** {self.overall}",
            "",
            "## Checks",
            "",
        ]
        for c in self.checks:
            icon = {"ok": "+", "warning": "!", "error": "x", "skipped": "-"}.get(c.status, "?")
            lines.append(f"- [{icon}] **{c.name}** ({c.category}): {c.status}")
            if c.detail:
                lines.append(f"  - {redact_secrets(c.detail)}")
            if c.fix_suggestion:
                lines.append(f"  - Fix: {c.fix_suggestion}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_python() -> DoctorCheck:
    """Check Python version and basic info."""
    ver = platform.python_version()
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        return DoctorCheck(
            name="python_version", category="python", status="error",
            detail=f"Python {ver} is below minimum 3.10",
            fix_suggestion="Install Python 3.10+ (e.g. sudo apt install python3.10)",
            meta={"version": ver},
        )
    return DoctorCheck(
        name="python_version", category="python", status="ok",
        detail=f"Python {ver}", meta={"version": ver},
    )


def check_venv() -> DoctorCheck:
    """Check if running inside a virtual environment."""
    in_venv = (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or os.environ.get("VIRTUAL_ENV") is not None
    )
    if in_venv:
        venv_path = os.environ.get("VIRTUAL_ENV", sys.prefix)
        return DoctorCheck(
            name="virtual_env", category="venv", status="ok",
            detail=f"Active venv at {venv_path}",
            meta={"venv_path": venv_path},
        )
    return DoctorCheck(
        name="virtual_env", category="venv", status="warning",
        detail="Not running in a virtual environment",
        fix_suggestion="Create and activate a venv: python -m venv .venv && source .venv/bin/activate",
    )


def check_dependencies() -> DoctorCheck:
    """Check that critical dependencies are importable."""
    missing: List[str] = []
    for pkg in ("fastapi", "uvicorn", "pydantic", "jinja2", "httpx"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        return DoctorCheck(
            name="dependencies", category="deps", status="error",
            detail=f"Missing packages: {', '.join(missing)}",
            fix_suggestion='pip install -e ".[dev]"',
            meta={"missing": missing},
        )
    return DoctorCheck(
        name="dependencies", category="deps", status="ok",
        detail="All critical packages importable",
    )


def check_fastapi_server(host: str = "127.0.0.1", port: int = 8000) -> DoctorCheck:
    """Check if the FastAPI server is reachable."""
    try:
        import urllib.request
        url = f"http://{host}:{port}/api/health"
        req = urllib.request.Request(url, method="GET")
        req.add_header("User-Agent", "IGRIS_GPT/doctor")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                return DoctorCheck(
                    name="fastapi_server", category="server", status="ok",
                    detail=f"Server reachable at {host}:{port}",
                    meta={"url": url},
                )
    except Exception as exc:
        pass
    return DoctorCheck(
        name="fastapi_server", category="server", status="warning",
        detail=f"Server not reachable at {host}:{port}",
        fix_suggestion="Start the server: python -m igris.web.server or bash scripts/start_igris.sh",
    )


def check_ollama() -> DoctorCheck:
    """Check Ollama availability and model presence."""
    base_url = os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        host_part = base_url.replace("http://", "").replace("https://", "")
        parts = host_part.split(":")
        sock_host = parts[0] if parts else "127.0.0.1"
        sock_port = int(parts[1]) if len(parts) > 1 else 11434
        result = sock.connect_ex((sock_host, sock_port))
        sock.close()
        if result != 0:
            return DoctorCheck(
                name="ollama", category="ollama", status="warning",
                detail=f"Ollama not reachable at {base_url}",
                fix_suggestion="Install and start Ollama: curl -fsSL https://ollama.com/install.sh | sh && ollama serve",
                meta={"base_url": base_url},
            )
        import urllib.request
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        req.add_header("User-Agent", "IGRIS_GPT/doctor")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            models = [m.get("name", "") for m in data.get("models", [])]
            model_configured = os.environ.get("LOCAL_LLM_MODEL", "phi4-mini")
            available = any(
                model_configured in m or model_configured.replace("-", "") in m
                for m in models
            )
            if available:
                return DoctorCheck(
                    name="ollama", category="ollama", status="ok",
                    detail=f"Ollama reachable, model '{model_configured}' available",
                    meta={"models_count": len(models), "model_configured": model_configured},
                )
            return DoctorCheck(
                name="ollama", category="ollama", status="warning",
                detail=f"Ollama reachable but model '{model_configured}' not found ({len(models)} models available)",
                fix_suggestion=f"Pull the model: ollama pull {model_configured}",
                meta={"models_count": len(models), "model_configured": model_configured},
            )
    except Exception:
        return DoctorCheck(
            name="ollama", category="ollama", status="warning",
            detail=f"Ollama not reachable at {base_url}",
            fix_suggestion="Install and start Ollama: curl -fsSL https://ollama.com/install.sh | sh && ollama serve",
            meta={"base_url": base_url},
        )


def check_openai_key() -> DoctorCheck:
    """Check whether an OpenAI API key is configured (never expose it)."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return DoctorCheck(
            name="openai_key", category="openai", status="ok",
            detail="OpenAI API key is configured (not shown)",
        )
    return DoctorCheck(
        name="openai_key", category="openai", status="warning",
        detail="No OpenAI API key configured — deterministic fallback will be used",
        fix_suggestion="Set OPENAI_API_KEY in .env if you want OpenAI fallback",
    )


def check_git() -> DoctorCheck:
    """Check git availability and repo status."""
    git_path = shutil.which("git")
    if not git_path:
        return DoctorCheck(
            name="git", category="git", status="error",
            detail="git not found in PATH",
            fix_suggestion="Install git: sudo apt install git",
        )
    try:
        result = subprocess.run(
            ["git", "--version"], capture_output=True, text=True, timeout=5
        )
        version = result.stdout.strip()
        return DoctorCheck(
            name="git", category="git", status="ok",
            detail=version, meta={"path": git_path},
        )
    except Exception as exc:
        return DoctorCheck(
            name="git", category="git", status="error",
            detail=f"git found but error running: {exc}",
        )


def check_docker() -> DoctorCheck:
    """Check Docker availability."""
    docker_path = shutil.which("docker")
    if not docker_path:
        return DoctorCheck(
            name="docker", category="docker", status="skipped",
            detail="Docker not installed (optional)",
            fix_suggestion="Install Docker if needed: https://docs.docker.com/engine/install/",
        )
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return DoctorCheck(
                name="docker", category="docker", status="ok",
                detail=f"Docker server {result.stdout.strip()}",
                meta={"path": docker_path},
            )
        return DoctorCheck(
            name="docker", category="docker", status="warning",
            detail="Docker installed but daemon not reachable",
            fix_suggestion="Start Docker: sudo systemctl start docker",
        )
    except Exception:
        return DoctorCheck(
            name="docker", category="docker", status="warning",
            detail="Docker installed but error checking status",
        )


def check_ssh() -> DoctorCheck:
    """Check SSH client availability."""
    ssh_path = shutil.which("ssh")
    if not ssh_path:
        return DoctorCheck(
            name="ssh", category="ssh", status="skipped",
            detail="SSH client not installed (optional for local-only mode)",
            fix_suggestion="Install SSH: sudo apt install openssh-client",
        )
    return DoctorCheck(
        name="ssh", category="ssh", status="ok",
        detail="SSH client available",
        meta={"path": ssh_path},
    )


def check_port(port: int, host: str = "127.0.0.1") -> DoctorCheck:
    """Check if a port is available (not already bound)."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            return DoctorCheck(
                name=f"port_{port}", category="ports", status="ok",
                detail=f"Port {port} is in use (server may be running)",
                meta={"port": port, "in_use": True},
            )
        return DoctorCheck(
            name=f"port_{port}", category="ports", status="ok",
            detail=f"Port {port} is available",
            meta={"port": port, "in_use": False},
        )
    except Exception:
        return DoctorCheck(
            name=f"port_{port}", category="ports", status="warning",
            detail=f"Could not check port {port}",
        )


def check_workspace(project_root: Optional[str] = None) -> DoctorCheck:
    """Check workspace/project path validity."""
    root = Path(project_root) if project_root else Path(os.environ.get("PROJECT_ROOT", "."))
    root = root.resolve()
    if not root.exists():
        return DoctorCheck(
            name="workspace", category="workspace", status="error",
            detail=f"Project root does not exist: {root}",
            fix_suggestion="Set PROJECT_ROOT to a valid directory",
            meta={"path": str(root)},
        )
    if not root.is_dir():
        return DoctorCheck(
            name="workspace", category="workspace", status="error",
            detail=f"Project root is not a directory: {root}",
            meta={"path": str(root)},
        )
    igris_dir = root / ".igris"
    igris_exists = igris_dir.exists()
    return DoctorCheck(
        name="workspace", category="workspace", status="ok",
        detail=f"Project root OK: {root}",
        meta={"path": str(root), "igris_dir_exists": igris_exists},
    )


def check_permissions(project_root: Optional[str] = None) -> DoctorCheck:
    """Check write permissions on project root and .igris dir."""
    root = Path(project_root) if project_root else Path(os.environ.get("PROJECT_ROOT", "."))
    root = root.resolve()
    if not root.exists():
        return DoctorCheck(
            name="permissions", category="permissions", status="error",
            detail="Project root does not exist — cannot check permissions",
        )
    writable = os.access(root, os.W_OK)
    if not writable:
        return DoctorCheck(
            name="permissions", category="permissions", status="error",
            detail=f"No write permission on {root}",
            fix_suggestion=f"Fix permissions: chmod u+w {root}",
        )
    return DoctorCheck(
        name="permissions", category="permissions", status="ok",
        detail=f"Write access OK on {root}",
    )


def check_env_file(project_root: Optional[str] = None) -> DoctorCheck:
    """Check .env file existence (never read contents)."""
    root = Path(project_root) if project_root else Path(os.environ.get("PROJECT_ROOT", "."))
    env_path = root / ".env"
    if env_path.exists():
        return DoctorCheck(
            name="env_file", category="config", status="ok",
            detail=".env file exists (contents not read)",
        )
    example = root / ".env.example"
    if example.exists():
        return DoctorCheck(
            name="env_file", category="config", status="warning",
            detail=".env file not found but .env.example exists",
            fix_suggestion="Copy .env.example to .env and fill in values: cp .env.example .env",
        )
    return DoctorCheck(
        name="env_file", category="config", status="warning",
        detail=".env file not found",
        fix_suggestion="Create .env from .env.example or set environment variables directly",
    )


def check_config_json(project_root: Optional[str] = None) -> DoctorCheck:
    """Check config JSON file existence and basic validity."""
    root = Path(project_root) if project_root else Path(os.environ.get("PROJECT_ROOT", "."))
    config_dir = root / "config"
    sample = config_dir / "config.sample.json"
    if not config_dir.exists():
        return DoctorCheck(
            name="config_json", category="config", status="warning",
            detail="config/ directory not found",
        )
    if sample.exists():
        try:
            data = json.loads(sample.read_text(encoding="utf-8"))
            return DoctorCheck(
                name="config_json", category="config", status="ok",
                detail="config/config.sample.json is valid JSON",
                meta={"keys": list(data.keys()) if isinstance(data, dict) else []},
            )
        except json.JSONDecodeError as exc:
            return DoctorCheck(
                name="config_json", category="config", status="error",
                detail=f"config/config.sample.json is invalid JSON: {exc}",
                fix_suggestion="Fix the JSON syntax in config/config.sample.json",
            )
    return DoctorCheck(
        name="config_json", category="config", status="warning",
        detail="config/config.sample.json not found",
    )


# ---------------------------------------------------------------------------
# Full doctor run
# ---------------------------------------------------------------------------

def run_doctor(
    project_root: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> DoctorReport:
    """Run all doctor checks and return a full report."""
    report = DoctorReport()
    report.checks.append(check_python())
    report.checks.append(check_venv())
    report.checks.append(check_dependencies())
    report.checks.append(check_fastapi_server(host=host, port=port))
    report.checks.append(check_ollama())
    report.checks.append(check_openai_key())
    report.checks.append(check_git())
    report.checks.append(check_docker())
    report.checks.append(check_ssh())
    report.checks.append(check_port(port))
    report.checks.append(check_workspace(project_root))
    report.checks.append(check_permissions(project_root))
    report.checks.append(check_env_file(project_root))
    report.checks.append(check_config_json(project_root))
    return report


# ---------------------------------------------------------------------------
# Verify — quick smoke check
# ---------------------------------------------------------------------------

def run_verify(project_root: Optional[str] = None) -> Dict[str, Any]:
    """Quick verification of installation essentials.

    Returns a dict with pass/fail for each category plus an overall ``ok``
    flag.
    """
    root = Path(project_root) if project_root else Path(os.environ.get("PROJECT_ROOT", "."))
    root = root.resolve()

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks": {},
        "ok": True,
    }

    # 1. Project root exists
    results["checks"]["project_root"] = root.is_dir()

    # 2. Critical files
    critical_files = [
        "pyproject.toml",
        "igris/__init__.py",
        "igris/web/server.py",
        "igris/models/config.py",
    ]
    missing_files: List[str] = []
    for f in critical_files:
        if not (root / f).exists():
            missing_files.append(f)
    results["checks"]["critical_files"] = {
        "ok": len(missing_files) == 0,
        "missing": missing_files,
    }

    # 3. .igris dir writable
    igris_dir = root / ".igris"
    try:
        igris_dir.mkdir(parents=True, exist_ok=True)
        test_file = igris_dir / ".verify_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        results["checks"]["igris_dir_writable"] = True
    except Exception:
        results["checks"]["igris_dir_writable"] = False

    # 4. Config loadable
    try:
        from igris.models.config import Config
        cfg = Config.load()
        results["checks"]["config_loadable"] = True
    except Exception as exc:
        results["checks"]["config_loadable"] = False

    # 5. Dependencies importable
    dep_ok = True
    for pkg in ("fastapi", "uvicorn", "pydantic"):
        try:
            __import__(pkg)
        except ImportError:
            dep_ok = False
    results["checks"]["dependencies"] = dep_ok

    # Compute overall
    for key, val in results["checks"].items():
        if isinstance(val, dict):
            if not val.get("ok", True):
                results["ok"] = False
        elif val is False:
            results["ok"] = False

    return results
