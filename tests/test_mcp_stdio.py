"""End-to-end tests for the published MCP stdio boundary."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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
    "stocks": [
        {"name": "Population", "initial_value": "100", "units": "people"},
    ],
    "flows": [
        {
            "name": "growth",
            "equation": "Population * growth_rate",
            "to_stock": "Population",
            "units": "people/Year",
        },
    ],
    "auxs": [
        {"name": "growth_rate", "equation": "0.1", "units": "1/Year"},
    ],
}


async def _exercise_stdio_server(output_path: Path) -> None:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "stella_mcp.server"],
        cwd=ROOT,
    )

    with anyio.fail_after(30):
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                initialized = await session.initialize()
                assert initialized.serverInfo.name == "stella-mcp"

                tools = await session.list_tools()
                tool_names = {tool.name for tool in tools.tools}
                assert len(tool_names) == 42
                assert {"build_model", "validate_model", "render_diagram"} <= tool_names

                resources = await session.list_resources()
                resource_uris = {str(resource.uri) for resource in resources.resources}
                assert "stella://templates/sir" in resource_uris

                prompts = await session.list_prompts()
                assert [prompt.name for prompt in prompts.prompts] == ["build-stella-model"]
                prompt = await session.get_prompt(
                    "build-stella-model",
                    {"description": "an exponential population growth model"},
                )
                assert prompt.messages

                built = await session.call_tool("build_model", GROWTH_MODEL)
                assert built.isError is False
                assert built.structuredContent["model_id"] == "stdio_growth"
                assert built.structuredContent["validation"]["passed"] is True

                validated = await session.call_tool(
                    "validate_model",
                    {"model_id": "stdio_growth"},
                )
                assert validated.isError is False
                assert validated.structuredContent == {
                    "model_id": "stdio_growth",
                    "passed": True,
                    "issues": [],
                }

                rendered = await session.call_tool(
                    "render_diagram",
                    {"model_id": "stdio_growth"},
                )
                assert rendered.isError is False
                assert rendered.structuredContent["svg"].startswith("<svg")

                saved = await session.call_tool(
                    "save_model",
                    {"model_id": "stdio_growth", "filepath": str(output_path)},
                )
                assert saved.isError is False
                assert output_path.exists()

                loaded = await session.call_tool(
                    "read_model",
                    {"filepath": str(output_path), "model_id": "stdio_roundtrip"},
                )
                assert loaded.isError is False

                session_resources = await session.list_resources()
                session_uris = {str(resource.uri) for resource in session_resources.resources}
                assert "stella://models/stdio_growth" in session_uris
                assert "stella://models/stdio_roundtrip" in session_uris

                unknown = await session.call_tool("not_a_stella_tool", {})
                assert unknown.isError is True
                assert unknown.structuredContent["error"]["code"] == "unknown_tool"


def test_stdio_lifecycle_and_model_workflow(tmp_path: Path) -> None:
    """A real MCP client should complete the primary local workflow."""
    asyncio.run(_exercise_stdio_server(tmp_path / "protocol_growth.stmx"))
