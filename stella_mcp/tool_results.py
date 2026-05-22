"""Helpers for MCP tool result construction."""

from __future__ import annotations

from typing import Any

from mcp.types import CallToolResult, TextContent


def success_result(text: str, structured: dict[str, Any] | None = None) -> CallToolResult:
    """Return a successful MCP tool result with optional structured content."""
    return CallToolResult(
        isError=False,
        content=[TextContent(type="text", text=text)],
        structuredContent=structured or {},
    )
