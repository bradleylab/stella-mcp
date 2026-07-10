"""Schemas for model inspection, validation, and deletion tools."""

from __future__ import annotations

from mcp.types import Tool

from .shared import SharedSchemas, build_shared_schemas


def build_tools(shared: SharedSchemas | None = None) -> list[Tool]:
    """Build inspection-domain tool descriptors in public catalog order."""
    model_id_property = (shared or build_shared_schemas()).model_id_property
    return [
        Tool(
            name="list_models",
            description="List all model IDs available in the current session",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="delete_model",
            description=(
                "Remove a model from the current session. Saved .stmx files are "
                "not touched. model_id is required — there is deliberately no "
                "implicit 'delete current model'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Session model ID to remove",
                    },
                },
                "required": ["model_id"],
            },
        ),
        Tool(
            name="inspect_model",
            description=(
                "Return a structured summary of the current model for agent inspection"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "include_validation": {
                        "type": "boolean",
                        "description": "Include validation issues in structured output",
                        "default": True,
                    },
                },
            },
        ),
        Tool(
            name="list_modules",
            description="List modules/groups in the current model",
            inputSchema={
                "type": "object",
                "properties": {"model_id": model_id_property},
            },
        ),
        Tool(
            name="list_connectors",
            description=(
                "List connector metadata (uid, endpoints, angle, routing lock/points)"
            ),
            inputSchema={
                "type": "object",
                "properties": {"model_id": model_id_property},
            },
        ),
        Tool(
            name="validate_model",
            description="Validate the current model for errors and warnings",
            inputSchema={
                "type": "object",
                "properties": {"model_id": model_id_property},
            },
        ),
        Tool(
            name="list_variables",
            description="List all variables (stocks, flows, auxiliaries) in the current model",
            inputSchema={
                "type": "object",
                "properties": {"model_id": model_id_property},
            },
        ),
        Tool(
            name="get_model_xml",
            description="Get the XMILE XML representation of the current model (for preview)",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "auto_layout": {
                        "type": "boolean",
                        "description": "Whether to auto-layout before export",
                        "default": True,
                    },
                    "resolve_layout_violations": {
                        "type": "boolean",
                        "description": (
                            "Whether to run layout crossing/collision post-processing "
                            "before export"
                        ),
                        "default": False,
                    },
                    "compat_mode": {
                        "type": "string",
                        "enum": ["permissive", "strict"],
                        "description": "Compatibility mode for export checks",
                        "default": "permissive",
                    },
                },
            },
        ),
    ]
