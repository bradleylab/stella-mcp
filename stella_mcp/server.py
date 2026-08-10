"""MCP server for Stella system dynamics models."""

from __future__ import annotations

import asyncio
import contextvars
import math
from collections.abc import Callable
from typing import Any

from jsonschema import Draft202012Validator
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.types import (
    LATEST_PROTOCOL_VERSION,
    CallToolRequestParams,
    CallToolResult,
    GetPromptRequestParams,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListToolsResult,
    PaginatedRequestParams,
    Prompt,
    ReadResourceRequestParams,
    ReadResourceResult,
    Resource,
    TextContent,
    TextResourceContents,
    Tool,
)

from . import __version__
from .mcp_resources import (
    build_model_prompt,
    list_all_resources,
    list_prompt_definitions,
    read_resource_content,
)
from .session_store import (
    LEGACY_WORKSPACE_ID,
    SessionDeleteResult,
    SessionModelEntry,
    WorkspaceError,
    WorkspaceExpiredError,
    WorkspaceNotFoundError,
    WorkspaceRevokedError,
    WorkspaceStore,
)
from .tool_handlers import register_tool_handlers
from .tool_results import success_result
from .tool_schemas import build_tool_definitions
from .xmile import GraphicalFunction, StellaModel

_workspace_store = WorkspaceStore()
_current_workspace_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "stella_workspace_id", default=LEGACY_WORKSPACE_ID
)
_GF_TYPES = {"continuous", "discrete"}
_WORKSPACE_FREE_TOOLS = {
    "create_workspace",
    "revoke_workspace",
    "list_templates",
    "get_template_info",
}


def _get_workspace_id() -> str:
    """Return the application workspace bound to the current tool call."""
    return _current_workspace_id.get()


def _get_session_key() -> str:
    """Deprecated direct-test alias for the current application workspace."""
    return _get_workspace_id()


def _active_workspace_id() -> str:
    """Normalize the retained direct-test seam without affecting wire routing."""
    value = _get_session_key()
    if isinstance(value, str):
        return value
    workspace_id = f"test_workspace_{value}"
    _workspace_store.ensure_test_workspace(workspace_id)
    return workspace_id


def _list_session_models() -> tuple[SessionModelEntry, ...]:
    return _workspace_store.list(_active_workspace_id())


def _delete_session_model(model_id: str) -> SessionDeleteResult:
    return _workspace_store.delete(_active_workspace_id(), model_id)


def _contains_session_model(model_id: str) -> bool:
    return _workspace_store.contains(_active_workspace_id(), model_id)


def _replace_session_model(model_id: str, model: StellaModel) -> None:
    _workspace_store.replace(_active_workspace_id(), model_id, model)


def _clear_session_store(workspace_id: str | None = None) -> None:
    """Explicit test/lifecycle hook for clearing workspace state."""
    _workspace_store.clear(workspace_id)


def _set_current_model(model: StellaModel, model_id: str | None = None) -> str:
    return _workspace_store.set_current(_active_workspace_id(), model, model_id)


def get_model(model_id: str | None = None) -> tuple[str, StellaModel]:
    return _workspace_store.get(_active_workspace_id(), model_id)


def _validate_scale(name: str, data: dict[str, Any]) -> tuple[float, float]:
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
    error: dict[str, Any] = {"code": code, "message": message, "category": category}
    if details:
        for key, value in details.items():
            if key not in error:
                error[key] = value
    return CallToolResult(
        is_error=True,
        content=[TextContent(type="text", text=f"[{code}] {message}")],
        structured_content={"error": error},
    )


def _classify_error(exc: Exception) -> tuple[str, str]:
    from .equation_parser import StellaReservedIdentifierError
    from .simulate import SimulationDependencyError
    from .xmile_features import UnsupportedModelFeatureError

    if isinstance(exc, UnsupportedModelFeatureError):
        return ("unsupported_model_feature", "compatibility")
    if isinstance(exc, StellaReservedIdentifierError):
        return ("reserved_identifier", "compatibility")
    if isinstance(exc, WorkspaceExpiredError):
        return ("workspace_expired", "workspace")
    if isinstance(exc, WorkspaceRevokedError):
        return ("workspace_revoked", "workspace")
    if isinstance(exc, WorkspaceNotFoundError):
        return ("workspace_not_found", "workspace")
    if isinstance(exc, WorkspaceError):
        return ("invalid_workspace", "workspace")
    message = str(exc)
    if isinstance(exc, SimulationDependencyError):
        return ("sim_dependency_missing", "environment")
    if isinstance(exc, FileNotFoundError):
        return ("not_found", "user_input")
    if isinstance(exc, ValueError):
        if "No model created" in message or "Unknown model_id" in message:
            return ("model_not_found", "user_input")
        return ("invalid_input", "user_input")
    return ("internal_error", "internal")


def _compat_warning_suffix(warnings: list[str]) -> str:
    if not warnings:
        return ""
    return f" (compatibility warnings: {len(warnings)}; first: {warnings[0]})"


ToolResponse = list[TextContent] | CallToolResult
ToolHandler = Callable[[dict[str, Any]], ToolResponse]
_TOOL_HANDLERS: dict[str, ToolHandler] = {}
_OUTPUT_SCHEMAS = {
    tool.name: tool.output_schema for tool in build_tool_definitions()
}


def _register_tool_handler(name: str):
    def decorator(func: ToolHandler) -> ToolHandler:
        _TOOL_HANDLERS[name] = func
        return func

    return decorator


register_tool_handlers(
    _register_tool_handler,
    get_model=get_model,
    set_current_model=_set_current_model,
    list_session_models=_list_session_models,
    delete_session_model=_delete_session_model,
    contains_session_model=_contains_session_model,
    replace_session_model=_replace_session_model,
    build_graphical_function=build_graphical_function,
    compat_warning_suffix=_compat_warning_suffix,
)

# Lifecycle dispatch is handled asynchronously in ``call_tool`` so revocation
# can coordinate with the workspace lock.  Sentinels keep the public registry
# complete for schema/handler parity checks.
_TOOL_HANDLERS["create_workspace"] = lambda arguments: []
_TOOL_HANDLERS["revoke_workspace"] = lambda arguments: []


def _validate_success_result(name: str, result: CallToolResult) -> None:
    """Validate every successful result, including workspace lifecycle tools."""
    schema = _OUTPUT_SCHEMAS.get(name)
    if schema is None:
        raise RuntimeError(f"Tool '{name}' has no output schema")
    Draft202012Validator(schema).validate(result.structured_content)


def _resolve_workspace(
    name: str,
    arguments: dict[str, Any],
    *,
    protocol_version: str,
) -> str:
    """Resolve explicit modern routing or the documented legacy fallback."""
    supplied = arguments.get("workspace_id")
    modern = protocol_version == LATEST_PROTOCOL_VERSION
    if name in _WORKSPACE_FREE_TOOLS:
        return LEGACY_WORKSPACE_ID
    if supplied is None:
        if modern:
            raise WorkspaceNotFoundError(
                "workspace_id is required for stateful MCP 2026-07-28 tool calls"
            )
        return LEGACY_WORKSPACE_ID
    if not isinstance(supplied, str):
        raise WorkspaceNotFoundError("workspace_id must be a string")
    _workspace_store.require(supplied)
    return supplied


async def call_tool(
    name: str,
    arguments: dict[str, Any] | None,
    *,
    protocol_version: str = "legacy",
) -> CallToolResult:
    """Call one registered tool; retained as a direct-test compatibility seam."""
    args = dict(arguments or {})
    try:
        if name == "create_workspace":
            workspace_id = _workspace_store.create(ttl_seconds=args.get("ttl_seconds"))
            result = success_result(
                f"Created workspace {workspace_id}",
                {"workspace_id": workspace_id},
            )
        elif name == "revoke_workspace":
            workspace_id = args.get("workspace_id")
            if not isinstance(workspace_id, str):
                raise WorkspaceNotFoundError("workspace_id is required")
            lock = _workspace_store.lock_for(workspace_id)
            async with lock:
                _workspace_store.revoke(workspace_id)
            result = success_result(
                f"Revoked workspace {workspace_id}",
                {"workspace_id": workspace_id, "revoked": True},
            )
        else:
            handler = _TOOL_HANDLERS.get(name)
            if handler is None:
                return _error_result(
                    "unknown_tool", f"Unknown tool: {name}", "user_input"
                )
            workspace_id = _resolve_workspace(
                name, args, protocol_version=protocol_version
            )
            args.pop("workspace_id", None)
            if name in _WORKSPACE_FREE_TOOLS:
                response = handler(args)
            else:
                lock = _workspace_store.lock_for(workspace_id)
                async with lock:
                    # The workspace may have expired or been revoked while this
                    # call waited behind another operation on the same lock.
                    _workspace_store.require(workspace_id)
                    token = _current_workspace_id.set(workspace_id)
                    try:
                        response = handler(args)
                    finally:
                        _current_workspace_id.reset(token)
            result = (
                response
                if isinstance(response, CallToolResult)
                else CallToolResult(content=response)
            )
        if not result.is_error:
            _validate_success_result(name, result)
        return result
    except Exception as exc:
        code, category = _classify_error(exc)
        internal = category == "internal"
        return _error_result(
            code=code,
            message="Internal server error" if internal else str(exc),
            category=category,
            details=None if internal else getattr(exc, "details", None),
        )


async def _on_list_tools(
    ctx: ServerRequestContext[Any], params: PaginatedRequestParams | None
) -> ListToolsResult:
    del params
    return ListToolsResult(
        tools=build_tool_definitions(
            require_workspace_id=ctx.protocol_version == LATEST_PROTOCOL_VERSION
        )
    )


async def _on_call_tool(
    ctx: ServerRequestContext[Any], params: CallToolRequestParams
) -> CallToolResult:
    return await call_tool(
        params.name,
        params.arguments,
        protocol_version=ctx.protocol_version,
    )


async def _on_list_resources(
    ctx: ServerRequestContext[Any], params: PaginatedRequestParams | None
) -> ListResourcesResult:
    del params
    models = (
        _workspace_store.list(LEGACY_WORKSPACE_ID)
        if ctx.protocol_version != LATEST_PROTOCOL_VERSION
        else ()
    )
    return ListResourcesResult(resources=list_all_resources(models))


async def _on_read_resource(
    ctx: ServerRequestContext[Any], params: ReadResourceRequestParams
) -> ReadResourceResult:
    content, mime_type = read_resource_content(
        str(params.uri),
        workspace_store=_workspace_store,
        legacy_models=(
            _workspace_store.list(LEGACY_WORKSPACE_ID)
            if ctx.protocol_version != LATEST_PROTOCOL_VERSION
            else ()
        ),
    )
    return ReadResourceResult(
        contents=[TextResourceContents(uri=str(params.uri), mime_type=mime_type, text=content)]
    )


async def _on_list_prompts(
    ctx: ServerRequestContext[Any], params: PaginatedRequestParams | None
) -> ListPromptsResult:
    del ctx, params
    return ListPromptsResult(prompts=list_prompt_definitions())


async def _on_get_prompt(
    ctx: ServerRequestContext[Any], params: GetPromptRequestParams
) -> GetPromptResult:
    del ctx
    if params.name != "build-stella-model":
        raise ValueError(f"Unknown prompt '{params.name}'")
    return build_model_prompt((params.arguments or {}).get("description"))


server = Server(
    "stella-mcp",
    version=__version__,
    on_list_tools=_on_list_tools,
    on_call_tool=_on_call_tool,
    on_list_resources=_on_list_resources,
    on_read_resource=_on_read_resource,
    on_list_prompts=_on_list_prompts,
    on_get_prompt=_on_get_prompt,
)


# Direct helpers remain useful to domain tests; they model legacy compatibility
# behavior while the low-level handlers above exercise the real wire boundary.
async def list_tools() -> list[Tool]:
    return build_tool_definitions()


async def list_resources() -> list[Resource]:
    workspace_id = _active_workspace_id()
    return list_all_resources(
        _workspace_store.list(workspace_id), workspace_id=workspace_id
    )


async def read_resource(uri: Any) -> list[TextResourceContents]:
    workspace_id = _active_workspace_id()
    content, mime_type = read_resource_content(
        str(uri),
        workspace_store=_workspace_store,
        legacy_models=(
            _workspace_store.list(workspace_id)
            if workspace_id == LEGACY_WORKSPACE_ID
            else ()
        ),
    )
    return [TextResourceContents(uri=str(uri), mime_type=mime_type, text=content)]


async def list_prompts() -> list[Prompt]:
    return list_prompt_definitions()


async def get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
    if name != "build-stella-model":
        raise ValueError(f"Unknown prompt '{name}'")
    return build_model_prompt((arguments or {}).get("description"))


async def run_server() -> None:
    """Run the dual-era stdio server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
