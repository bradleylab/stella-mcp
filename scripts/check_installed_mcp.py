#!/usr/bin/env python3
"""Smoke-test both MCP protocol eras through an installed Python artifact."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mcp import StdioServerParameters
from mcp.client import Client
from mcp.client.stdio import stdio_client
from mcp.types import LATEST_PROTOCOL_VERSION

ROOT = Path(__file__).resolve().parents[1]


def _interpreter_path(value: str | Path) -> Path:
    """Make an interpreter path absolute without resolving a venv symlink."""
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def _subprocess_env() -> dict[str, str]:
    """Return inherited process settings without Python import-path overrides."""
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


def _probe_installed_artifact(
    python: Path, *, cwd: Path, env: dict[str, str]
) -> dict[str, Any]:
    probe = """
import importlib.metadata
import json
import pathlib
import stella_mcp
import sys
import sysconfig

print(json.dumps({
    "distribution_version": importlib.metadata.version("stella-mcp"),
    "executable": sys.executable,
    "module_file": str(pathlib.Path(stella_mcp.__file__).resolve()),
    "module_version": stella_mcp.__version__,
    "purelib": str(pathlib.Path(sysconfig.get_paths()["purelib"]).resolve()),
}))
"""
    completed = subprocess.run(
        [str(python), "-I", "-c", probe],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _validate_installed_probe(
    probe: dict[str, Any], *, python: Path, expected_version: str
) -> None:
    module_file = Path(probe["module_file"]).resolve()
    purelib = Path(probe["purelib"]).resolve()
    if not module_file.is_relative_to(purelib):
        raise AssertionError(
            f"stella_mcp imported from {module_file}, outside installed purelib {purelib}"
        )
    if Path(probe["executable"]).absolute() != python.absolute():
        raise AssertionError(
            f"probe used {probe['executable']}, expected interpreter {python}"
        )
    versions = {probe["distribution_version"], probe["module_version"]}
    if versions != {expected_version}:
        raise AssertionError(
            f"installed version evidence {sorted(versions)} != {expected_version}"
        )


async def _exercise(
    python: Path,
    mode: str,
    *,
    server_cwd: Path,
    env: dict[str, str],
    expected_version: str,
) -> None:
    parameters = StdioServerParameters(
        command=str(python),
        args=["-I", "-m", "stella_mcp.server"],
        cwd=server_cwd,
        env=env,
    )
    async with Client(stdio_client(parameters), mode=mode) as client:
        assert client.server_info is not None
        assert client.server_info.version == expected_version
        tools = await client.list_tools()
        names = [tool.name for tool in tools.tools]
        assert len(names) == 44
        assert names[-2:] == ["create_workspace", "revoke_workspace"]
        assert "code" not in names
        catalog = {tool.name: tool for tool in tools.tools}
        required = set(catalog["build_model"].input_schema.get("required", []))

        if mode == "legacy":
            assert client.protocol_version != LATEST_PROTOCOL_VERSION
            assert "workspace_id" not in required
            workspace: dict[str, str] = {}
        else:
            assert client.protocol_version == LATEST_PROTOCOL_VERSION
            assert "workspace_id" in required
            created = await client.call_tool("create_workspace", {})
            assert not created.is_error
            workspace = {"workspace_id": created.structured_content["workspace_id"]}

        result = await client.call_tool(
            "create_model",
            {"name": f"Installed {mode}", "model_id": "installed", **workspace},
        )
        assert not result.is_error
        assert result.structured_content["model_id"] == "installed"


async def _run(
    python: Path,
    *,
    server_cwd: Path,
    env: dict[str, str],
    expected_version: str,
) -> None:
    await _exercise(
        python,
        "auto",
        server_cwd=server_cwd,
        env=env,
        expected_version=expected_version,
    )
    await _exercise(
        python,
        "legacy",
        server_cwd=server_cwd,
        env=env,
        expected_version=expected_version,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, help="Python from the clean installed environment")
    parser.add_argument(
        "--expected-version",
        default=importlib.metadata.version("stella-mcp"),
        help="Package and server version required from the installed environment",
    )
    args = parser.parse_args(argv)
    python = _interpreter_path(args.python)
    env = _subprocess_env()
    with tempfile.TemporaryDirectory(prefix="stella-installed-smoke-") as temp_dir:
        server_cwd = Path(temp_dir).resolve()
        if server_cwd.is_relative_to(ROOT):
            raise AssertionError(f"server cwd must be outside checkout: {server_cwd}")
        probe = _probe_installed_artifact(python, cwd=server_cwd, env=env)
        _validate_installed_probe(
            probe, python=python, expected_version=args.expected_version
        )
        asyncio.run(
            _run(
                python,
                server_cwd=server_cwd,
                env=env,
                expected_version=args.expected_version,
            )
        )
    print(
        "installed MCP smoke passed from isolated cwd: "
        f"{args.expected_version}; modern + legacy; 44 tools; no Code Mode"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
