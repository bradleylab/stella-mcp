"""MCP server for Stella system dynamics models."""

import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from .tool_handlers import register_tool_handlers
from .tool_schemas import build_tool_definitions
from .xmile import GraphicalFunction, StellaModel


@dataclass
class SessionModels:
    """Model state for a single MCP session."""
    models: dict[str, StellaModel] = field(default_factory=dict)
    current_model_id: str | None = None


# Session-keyed model registry (key is id(server.request_context.session))
_session_models: dict[int, SessionModels] = {}
_GF_TYPES = {"continuous", "discrete"}


# Create MCP server
server = Server("stella-mcp")


def _get_session_key() -> int:
    """Get a stable key for the current MCP session context."""
    try:
        return id(server.request_context.session)
    except LookupError:
        # Fallback for non-request contexts (tests/scripts)
        return -1


def _get_session_models() -> SessionModels:
    """Get or create model state for the current session."""
    session_key = _get_session_key()
    if session_key not in _session_models:
        _session_models[session_key] = SessionModels()
    return _session_models[session_key]


def _set_current_model(model: StellaModel, model_id: str | None = None) -> str:
    """Store model in session and set as current."""
    session_models = _get_session_models()
    resolved_id = model_id or f"model_{uuid.uuid4().hex[:8]}"
    if resolved_id in session_models.models:
        raise ValueError(f"model_id '{resolved_id}' already exists in this session")
    session_models.models[resolved_id] = model
    session_models.current_model_id = resolved_id
    return resolved_id


def get_model(model_id: str | None = None) -> tuple[str, StellaModel]:
    """Get current (or requested) model for this session."""
    session_models = _get_session_models()
    resolved_id = model_id or session_models.current_model_id
    if not resolved_id:
        raise ValueError("No model created in this session. Use create_model first.")
    model = session_models.models.get(resolved_id)
    if model is None:
        raise ValueError(f"Unknown model_id '{resolved_id}' for this session")
    session_models.current_model_id = resolved_id
    return resolved_id, model


def _validate_scale(name: str, data: dict[str, Any]) -> tuple[float, float]:
    """Validate and parse {min,max} scale object."""
    if "min" not in data or "max" not in data:
        raise ValueError(f"{name} requires both min and max")
    min_val = float(data["min"])
    max_val = float(data["max"])
    if not (math.isfinite(min_val) and math.isfinite(max_val)):
        raise ValueError(f"{name} values must be finite numbers")
    if min_val >= max_val:
        raise ValueError(f"{name} must satisfy min < max")
    return min_val, max_val


def build_graphical_function(data: dict | None) -> GraphicalFunction | None:
    """Build a GraphicalFunction from tool input."""
    if not data:
        return None

    ypts_raw = data.get("ypts")
    if not ypts_raw:
        raise ValueError("graphical_function requires non-empty ypts")
    ypts = [float(val) for val in ypts_raw]
    if len(ypts) < 2:
        raise ValueError("graphical_function requires at least 2 ypts")
    if not all(math.isfinite(val) for val in ypts):
        raise ValueError("graphical_function ypts must be finite numbers")

    xscale = data.get("xscale")
    xpts = data.get("xpts")
    if (xscale is None) == (xpts is None):
        raise ValueError("graphical_function requires exactly one of xscale or xpts")

    parsed_xscale = _validate_scale("xscale", xscale) if xscale is not None else None
    parsed_xpts = None
    if xpts is not None:
        parsed_xpts = [float(val) for val in xpts]
        if len(parsed_xpts) < 2:
            raise ValueError("graphical_function requires at least 2 xpts")
        if len(parsed_xpts) != len(ypts):
            raise ValueError("graphical_function xpts and ypts must have the same length")
        if not all(math.isfinite(val) for val in parsed_xpts):
            raise ValueError("graphical_function xpts must be finite numbers")

    yscale = data.get("yscale")
    parsed_yscale = _validate_scale("yscale", yscale) if yscale is not None else None

    gf_type = data.get("type")
    if gf_type is not None:
        gf_type = str(gf_type).lower()
        if gf_type not in _GF_TYPES:
            raise ValueError(f"graphical_function type must be one of {sorted(_GF_TYPES)}")

    return GraphicalFunction(
        ypts=ypts,
        xscale=parsed_xscale,
        xpts=parsed_xpts,
        yscale=parsed_yscale,
        gf_type=gf_type,
    )


def _error_result(
    code: str,
    message: str,
    category: str,
    details: dict[str, Any] | None = None,
) -> CallToolResult:
    """Build a structured MCP tool error result."""
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "category": category,
    }
    if details:
        error.update(details)
    return CallToolResult(
        isError=True,
        content=[TextContent(type="text", text=f"[{code}] {message}")],
        structuredContent={"error": error},
    )


def _classify_error(exc: Exception) -> tuple[str, str]:
    """Map Python exceptions to stable tool error codes/categories."""
    from .simulate import SimulationDependencyError

    message = str(exc)
    if isinstance(exc, SimulationDependencyError):
        return ("sim_dependency_missing", "environment")
    if isinstance(exc, FileNotFoundError):
        return ("not_found", "user_input")
    if isinstance(exc, ValueError):
        if "No model created in this session" in message or "Unknown model_id" in message:
            return ("model_not_found", "user_input")
        return ("invalid_input", "user_input")
    return ("internal_error", "internal")


def _compat_warning_suffix(warnings: list[str]) -> str:
    """Build compact warning suffix for tool text responses."""
    if not warnings:
        return ""
    return (
        f" (compatibility warnings: {len(warnings)}; "
        f"first: {warnings[0]})"
    )


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return build_tool_definitions()


ToolResponse = list[TextContent] | CallToolResult
ToolHandler = Callable[[dict[str, Any]], ToolResponse]
_TOOL_HANDLERS: dict[str, ToolHandler] = {}


def _register_tool_handler(name: str):
    """Register a tool handler function by MCP tool name."""
    def decorator(func: ToolHandler) -> ToolHandler:
        _TOOL_HANDLERS[name] = func
        return func

    return decorator


register_tool_handlers(
    _register_tool_handler,
    get_model=get_model,
    set_current_model=_set_current_model,
    get_session_models=_get_session_models,
    build_graphical_function=build_graphical_function,
    compat_warning_suffix=_compat_warning_suffix,
)

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> ToolResponse:
    """Handle tool calls via handler registry."""
    try:
        handler = _TOOL_HANDLERS.get(name)
        if handler is None:
            return _error_result(
                code="unknown_tool",
                message=f"Unknown tool: {name}",
                category="user_input",
            )
        return handler(arguments)
    except Exception as e:
        code, category = _classify_error(e)
        return _error_result(
            code=code,
            message=str(e),
            category=category,
            details=getattr(e, "details", None),
        )


async def run_server():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Entry point for the MCP server."""
    import asyncio
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
