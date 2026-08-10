"""End-to-end tests for both protocol eras at the published stdio boundary."""

from __future__ import annotations

import asyncio
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import anyio
import pytest
from mcp import StdioServerParameters
from mcp.client import Client
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import MCPError
from mcp.types import LATEST_PROTOCOL_VERSION

ROOT = Path(__file__).resolve().parents[1]

GROWTH_MODEL = {
    "name": "Protocol Growth",
    "model_id": "stdio_growth",
    "sim_specs": {
        "start": 0,
        "stop": 5,
        "dt": 1,
        "method": "Euler",
        "time_units": "Years",
    },
    "stocks": [{"name": "Population", "initial_value": "100", "units": "people"}],
    "flows": [
        {
            "name": "growth",
            "equation": "Population * growth_rate",
            "to_stock": "Population",
            "units": "people/Year",
        }
    ],
    "auxs": [{"name": "growth_rate", "equation": "0.1", "units": "1/Year"}],
}


async def _exercise_stdio_server(output_path: Path, *, mode: str) -> None:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "stella_mcp.server"],
        cwd=ROOT,
    )

    with anyio.fail_after(30):
        async with Client(stdio_client(parameters), mode=mode) as client:
            assert client.server_info is not None
            assert client.server_info.name == "stella-mcp"
            if mode == "legacy":
                assert client.protocol_version != LATEST_PROTOCOL_VERSION
                workspace: dict[str, str] = {}
            else:
                assert client.protocol_version == LATEST_PROTOCOL_VERSION
                missing = await client.call_tool("build_model", GROWTH_MODEL)
                assert missing.is_error is True
                assert missing.structured_content["error"]["code"] == "workspace_not_found"
                created = await client.call_tool("create_workspace", {})
                assert created.is_error is False
                workspace = {"workspace_id": created.structured_content["workspace_id"]}

            tools = await client.list_tools()
            tool_names = {tool.name for tool in tools.tools}
            tool_catalog = {tool.name: tool for tool in tools.tools}
            assert len(tool_names) == 44
            assert {
                "build_model",
                "validate_model",
                "render_diagram",
                "create_workspace",
            } <= tool_names
            build_required = set(
                tool_catalog["build_model"].input_schema.get("required", [])
            )
            assert "workspace_id" in tool_catalog["build_model"].input_schema[
                "properties"
            ]
            if mode == "legacy":
                assert "workspace_id" not in build_required
            else:
                assert "workspace_id" in build_required
            assert "workspace_id" not in tool_catalog[
                "list_templates"
            ].input_schema.get("properties", {})

            resources = await client.list_resources()
            resource_uris = {str(resource.uri) for resource in resources.resources}
            assert "stella://templates/sir" in resource_uris

            prompts = await client.list_prompts()
            assert [prompt.name for prompt in prompts.prompts] == ["build-stella-model"]
            prompt = await client.get_prompt(
                "build-stella-model",
                {"description": "an exponential population growth model"},
            )
            assert prompt.messages

            built = await client.call_tool("build_model", {**GROWTH_MODEL, **workspace})
            assert built.is_error is False
            assert built.structured_content["model_id"] == "stdio_growth"
            assert built.structured_content["validation"]["passed"] is True

            validated = await client.call_tool(
                "validate_model", {"model_id": "stdio_growth", **workspace}
            )
            assert validated.is_error is False
            assert validated.structured_content == {
                "model_id": "stdio_growth",
                "passed": True,
                "issues": [],
            }

            rendered = await client.call_tool(
                "render_diagram", {"model_id": "stdio_growth", **workspace}
            )
            assert rendered.is_error is False
            assert rendered.structured_content["svg"].startswith("<svg")

            saved = await client.call_tool(
                "save_model",
                {"model_id": "stdio_growth", "filepath": str(output_path), **workspace},
            )
            assert saved.is_error is False
            assert output_path.exists()

            loaded = await client.call_tool(
                "read_model",
                {"filepath": str(output_path), "model_id": "stdio_roundtrip", **workspace},
            )
            assert loaded.is_error is False

            if mode == "legacy":
                session_resources = await client.list_resources()
                session_uris = {str(resource.uri) for resource in session_resources.resources}
                assert "stella://models/stdio_growth" in session_uris
                assert "stella://models/stdio_roundtrip" in session_uris
                legacy_resource = await client.read_resource(
                    "stella://models/stdio_growth"
                )
                ET.fromstring(legacy_resource.contents[0].text)
            else:
                workspace_id = workspace["workspace_id"]
                model_resource_uri = (
                    f"stella://workspaces/{workspace_id}/models/stdio_growth"
                )
                modern_resource = await client.read_resource(model_resource_uri)
                ET.fromstring(modern_resource.contents[0].text)

                current = await client.call_tool(
                    "inspect_model", {"workspace_id": workspace_id}
                )
                assert current.structured_content["model"]["model_id"] == "stdio_roundtrip"

            unknown = await client.call_tool("not_a_stella_tool", {})
            assert unknown.is_error is True
            assert unknown.structured_content["error"]["code"] == "unknown_tool"

            if mode != "legacy":
                revoked = await client.call_tool(
                    "revoke_workspace", {"workspace_id": workspace["workspace_id"]}
                )
                assert revoked.is_error is False
                with pytest.raises(MCPError, match="Internal server error"):
                    await client.read_resource(model_resource_uri)


def test_modern_stdio_lifecycle_and_model_workflow(tmp_path: Path) -> None:
    asyncio.run(_exercise_stdio_server(tmp_path / "modern_growth.stmx", mode="auto"))


def test_legacy_stdio_lifecycle_and_model_workflow(tmp_path: Path) -> None:
    asyncio.run(_exercise_stdio_server(tmp_path / "legacy_growth.stmx", mode="legacy"))
