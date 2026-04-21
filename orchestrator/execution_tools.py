"""
execution_tools.py — Code execution and file access tools for autonomous SDK agents.

These tools give Claude subagents the ability to:
  - run_python: execute arbitrary Python in the project virtualenv, capturing output
  - read_project_file: read any file within the project directory (sandboxed)
  - list_project_files: glob for files within the project directory

Design principles:
  - run_python uses sys.executable so it always runs in the same conda env as the pipeline
  - All file tools are sandboxed to PROJECT_ROOT — paths outside are rejected
  - stdout/stderr are truncated to prevent context overflow (5000 / 2000 chars)
  - run_python working directory is PROJECT_ROOT so all project imports work

Usage (in AgentRunner):
    from orchestrator.execution_tools import run_python, read_project_file, list_project_files
    routes["run_python"]          = run_python
    routes["read_project_file"]   = read_project_file
    routes["list_project_files"]  = list_project_files
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

_STDOUT_LIMIT = 8000   # chars kept from stdout
_STDERR_LIMIT = 3000   # chars kept from stderr


def run_python(code: str, timeout: int = 60) -> dict:
    """
    Execute Python code in the project virtualenv and return stdout/stderr.

    The code runs with PROJECT_ROOT as the working directory, so all
    project imports (from pipelines.*, from mcp_servers.*, etc.) work.

    Args:
        code:    Python source to execute.
        timeout: Wall-clock time limit in seconds (default 60).

    Returns:
        {
            "stdout":     str,   # captured stdout (last 8000 chars)
            "stderr":     str,   # captured stderr (last 3000 chars)
            "returncode": int,   # 0 = success
            "success":    bool,
        }
    """
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        )
        stdout = result.stdout
        stderr = result.stderr
        # Truncate — keep tail (most recent output is most useful)
        if len(stdout) > _STDOUT_LIMIT:
            stdout = f"[...truncated {len(stdout) - _STDOUT_LIMIT} chars...]\n" + stdout[-_STDOUT_LIMIT:]
        if len(stderr) > _STDERR_LIMIT:
            stderr = f"[...truncated {len(stderr) - _STDERR_LIMIT} chars...]\n" + stderr[-_STDERR_LIMIT:]
        return {
            "stdout":     stdout,
            "stderr":     stderr,
            "returncode": result.returncode,
            "success":    result.returncode == 0,
        }

    except subprocess.TimeoutExpired:
        return {
            "stdout":     "",
            "stderr":     f"TimeoutExpired: code exceeded {timeout}s limit",
            "returncode": -1,
            "success":    False,
        }
    except Exception as exc:
        return {
            "stdout":     "",
            "stderr":     f"Execution error: {exc}",
            "returncode": -1,
            "success":    False,
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def read_project_file(relative_path: str) -> dict:
    """
    Read a file within the project directory.

    Paths are resolved relative to PROJECT_ROOT. Traversal outside the
    project (e.g. ../../etc/passwd) is rejected.

    Args:
        relative_path: Path relative to project root, e.g. "data/benchmarks/ibd_upstream_regulators_v1.json"

    Returns:
        {
            "content":  str,   # file contents (first 20000 chars)
            "path":     str,   # resolved absolute path
            "truncated": bool,
            "error":    str | None,
        }
    """
    _CONTENT_LIMIT = 20_000
    try:
        target = (PROJECT_ROOT / relative_path).resolve()
        # Sandbox check
        if not str(target).startswith(str(PROJECT_ROOT)):
            return {
                "content":   "",
                "path":      str(target),
                "truncated": False,
                "error":     f"Access denied: path is outside project root ({PROJECT_ROOT})",
            }
        if not target.exists():
            return {
                "content":   "",
                "path":      str(target),
                "truncated": False,
                "error":     f"File not found: {target}",
            }
        if target.is_dir():
            return {
                "content":   "",
                "path":      str(target),
                "truncated": False,
                "error":     f"Path is a directory, not a file: {target}",
            }
        content = target.read_text(encoding="utf-8", errors="replace")
        truncated = len(content) > _CONTENT_LIMIT
        return {
            "content":   content[:_CONTENT_LIMIT],
            "path":      str(target),
            "truncated": truncated,
            "error":     None,
        }
    except Exception as exc:
        return {
            "content":   "",
            "path":      relative_path,
            "truncated": False,
            "error":     str(exc),
        }


def list_project_files(pattern: str) -> dict:
    """
    Glob for files within the project directory.

    Args:
        pattern: Glob pattern relative to project root, e.g. "data/benchmarks/*.json"
                 or "agents/**/*.py"

    Returns:
        {
            "files":   list[str],  # relative paths from project root
            "count":   int,
            "error":   str | None,
        }
    """
    _MAX_RESULTS = 200
    try:
        # Resolve and sandbox: only return paths inside PROJECT_ROOT
        matches = glob.glob(str(PROJECT_ROOT / pattern), recursive=True)
        rel_paths = []
        for m in sorted(matches):
            resolved = Path(m).resolve()
            if str(resolved).startswith(str(PROJECT_ROOT)):
                rel_paths.append(str(resolved.relative_to(PROJECT_ROOT)))
        truncated = len(rel_paths) > _MAX_RESULTS
        return {
            "files":     rel_paths[:_MAX_RESULTS],
            "count":     len(rel_paths),
            "truncated": truncated,
            "error":     None,
        }
    except Exception as exc:
        return {
            "files":   [],
            "count":   0,
            "truncated": False,
            "error":   str(exc),
        }
