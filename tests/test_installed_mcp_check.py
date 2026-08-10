"""Contracts for the clean-installed MCP smoke checker."""

import asyncio
from pathlib import Path

import pytest

from scripts.check_installed_mcp import (
    _interpreter_path,
    _run,
    _validate_installed_probe,
)


def test_installed_probe_requires_module_under_target_purelib(tmp_path: Path):
    python = tmp_path / "venv" / "bin" / "python"
    purelib = tmp_path / "venv" / "lib" / "site-packages"
    valid = {
        "distribution_version": "0.14.0",
        "executable": str(python),
        "module_file": str(purelib / "stella_mcp" / "__init__.py"),
        "module_version": "0.14.0",
        "purelib": str(purelib),
    }

    _validate_installed_probe(valid, python=python, expected_version="0.14.0")

    checkout_import = {**valid, "module_file": str(tmp_path / "checkout" / "stella_mcp.py")}
    with pytest.raises(AssertionError, match="outside installed purelib"):
        _validate_installed_probe(
            checkout_import, python=python, expected_version="0.14.0"
        )


def test_interpreter_path_preserves_virtual_environment_launcher(tmp_path: Path):
    base = tmp_path / "base-python"
    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    base.touch()
    launcher.symlink_to(base)

    assert _interpreter_path(launcher) == launcher
    assert _interpreter_path(launcher) != launcher.resolve()


def test_protocol_smokes_reuse_one_isolated_server_cwd(monkeypatch, tmp_path: Path):
    calls = []

    async def record_exercise(python, mode, **kwargs):
        calls.append((python, mode, kwargs))

    monkeypatch.setattr("scripts.check_installed_mcp._exercise", record_exercise)
    python = tmp_path / "venv" / "bin" / "python"
    server_cwd = tmp_path / "server-cwd"
    env = {"PATH": "/usr/bin"}

    asyncio.run(
        _run(
            python,
            server_cwd=server_cwd,
            env=env,
            expected_version="0.14.0",
        )
    )

    assert [mode for _, mode, _ in calls] == ["auto", "legacy"]
    assert all(call[2]["server_cwd"] == server_cwd for call in calls)
    assert all(call[2]["env"] == env for call in calls)
