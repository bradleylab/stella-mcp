"""Compatibility facade for domain-owned MCP tool handlers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .tools import build, inspect, io, modules, simulation
from .tools.shared import (
    HandlerContext,
    SessionModelsLike,
    ToolHandler,
    ToolResponse,
)
from .tools.shared import apply_batch_items as _apply_batch_items
from .xmile import GraphicalFunction, StellaModel

__all__ = [
    "SessionModelsLike",
    "ToolHandler",
    "ToolResponse",
    "_apply_batch_items",
    "register_tool_handlers",
]


def register_tool_handlers(
    register: Callable[[str], Callable[[ToolHandler], ToolHandler]],
    *,
    get_model: Callable[[str | None], tuple[str, StellaModel]],
    set_current_model: Callable[[StellaModel, str | None], str],
    get_session_models: Callable[[], SessionModelsLike],
    build_graphical_function: Callable[
        [dict[str, Any] | None], GraphicalFunction | None
    ],
    compat_warning_suffix: Callable[[list[str]], str],
) -> None:
    """Register every domain while preserving the server-facing API."""
    context = HandlerContext(
        get_model=get_model,
        set_current_model=set_current_model,
        get_session_models=get_session_models,
        build_graphical_function=build_graphical_function,
        compat_warning_suffix=compat_warning_suffix,
    )
    build.register_handlers(register, context)
    modules.register_handlers(register, context)
    io.register_handlers(register, context)
    simulation.register_handlers(register, context)
    inspect.register_handlers(register, context)
