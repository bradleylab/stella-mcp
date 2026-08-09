"""Helpers for MCP tool result construction."""

from __future__ import annotations

from typing import Any

from mcp.types import CallToolResult, TextContent


def success_result(text: str, structured: dict[str, Any] | None = None) -> CallToolResult:
    """Return a successful MCP tool result with optional structured content."""
    return CallToolResult(
        is_error=False,
        content=[TextContent(type="text", text=text)],
        structured_content=structured or {},
    )


class BatchItemError(ValueError):
    """A batch tool item failed; carries the failing item's location.

    Batch tools are all-or-nothing, so the error must tell the caller
    exactly which item to fix before retrying the whole batch.
    """

    def __init__(self, stage: str, index: int, item_name: str | None, message: str):
        label = f" ('{item_name}')" if item_name else ""
        super().__init__(f"{stage}[{index}]{label}: {message}")
        self.details: dict[str, Any] = {
            "stage": stage,
            "index": index,
            "item_name": item_name,
        }
