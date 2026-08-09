"""MCP resources and prompts for the Stella server.

Pure helpers (no async, no server object) so the resource catalog, content
resolution, and prompt construction can be unit-tested directly. server.py
wires these into the low-level MCP v2 handlers.

Resource URIs:
- ``stella://templates/{name}`` — a built-in or user template's raw .stmx
- ``stella://workspaces/{workspace_id}/models/{model_id}`` — an explicit
  workspace model's current XMILE export
- ``stella://models/{model_id}`` — the legacy stdio workspace compatibility URI
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from urllib.parse import quote, unquote

from mcp.types import GetPromptResult, Prompt, PromptArgument, PromptMessage, Resource, TextContent

from .session_store import LEGACY_WORKSPACE_ID, SessionModelEntry, WorkspaceStore
from .templates import list_templates as list_available_templates

_TEMPLATE_SCHEME = "stella://templates/"
_MODEL_SCHEME = "stella://models/"
_WORKSPACE_SCHEME = "stella://workspaces/"

BUILD_MODEL_PROMPT = "build-stella-model"


def list_template_resources() -> list[Resource]:
    """One resource per discovered template (builtin + user)."""
    resources: list[Resource] = []
    for info in list_available_templates():
        resources.append(Resource(
            # Percent-encode the name so it round-trips through AnyUrl (which
            # otherwise encodes spaces/unicode and breaks the read lookup).
            uri=f"{_TEMPLATE_SCHEME}{quote(info.name, safe='')}",  # type: ignore[arg-type]
            name=info.name,
            title=info.title or info.name,
            description=info.description or f"{info.source} template",
            mime_type="application/xml",
        ))
    return resources


def list_model_resources(
    session_models: Sequence[SessionModelEntry],
    *,
    workspace_id: str = LEGACY_WORKSPACE_ID,
) -> list[Resource]:
    """One resource per model currently loaded in an explicit workspace."""
    resources: list[Resource] = []
    for entry in session_models:
        uri = (
            f"{_MODEL_SCHEME}{quote(entry.model_id, safe='')}"
            if workspace_id == LEGACY_WORKSPACE_ID
            else (
                f"{_WORKSPACE_SCHEME}{quote(workspace_id, safe='')}/models/"
                f"{quote(entry.model_id, safe='')}"
            )
        )
        resources.append(Resource(
            uri=uri,  # type: ignore[arg-type]
            name=entry.model_id,
            title=entry.model.name,
            description=f"Workspace model '{entry.model_id}' as XMILE",
            mime_type="application/xml",
        ))
    return resources


def list_all_resources(
    session_models: Sequence[SessionModelEntry],
    *,
    workspace_id: str = LEGACY_WORKSPACE_ID,
) -> list[Resource]:
    return list_template_resources() + list_model_resources(
        session_models, workspace_id=workspace_id
    )


def read_resource_content(
    uri: str,
    legacy_models: Sequence[SessionModelEntry] = (),
    *,
    workspace_store: WorkspaceStore | None = None,
) -> tuple[str, str]:
    """Resolve a ``stella://`` URI to (content, mime_type).

    Raises ValueError for unknown schemes or missing resources.
    """
    if uri.startswith(_TEMPLATE_SCHEME):
        # rstrip before unquote: any literal '/' in the name is %2F-encoded,
        # so this only drops an AnyUrl-appended trailing slash.
        name = unquote(uri[len(_TEMPLATE_SCHEME):].rstrip("/"))
        for info in list_available_templates():
            if info.name == name:
                return info.path.read_text(encoding="utf-8"), "application/xml"
        raise ValueError(f"Unknown template resource '{name}'")
    if uri.startswith(_MODEL_SCHEME):
        model_id = unquote(uri[len(_MODEL_SCHEME):].rstrip("/"))
        model = next(
            (entry.model for entry in legacy_models if entry.model_id == model_id),
            None,
        )
        if model is None:
            raise ValueError(f"Unknown model resource '{model_id}'")
        # Export mutates layout state, so render from a copy — a resource
        # read must not rewrite the session model's diagram.
        return copy.deepcopy(model).to_xml(compat_mode="permissive"), "application/xml"
    if uri.startswith(_WORKSPACE_SCHEME):
        if workspace_store is None:
            raise ValueError("Workspace model resources require an explicit workspace store")
        path = uri[len(_WORKSPACE_SCHEME):].rstrip("/").split("/")
        if len(path) != 3 or path[1] != "models":
            raise ValueError(f"Unsupported workspace resource URI '{uri}'")
        workspace_id = unquote(path[0])
        model_id = unquote(path[2])
        model = workspace_store.lookup(workspace_id, model_id)
        return copy.deepcopy(model).to_xml(compat_mode="permissive"), "application/xml"
    raise ValueError(f"Unsupported resource URI '{uri}'")


def list_prompt_definitions() -> list[Prompt]:
    return [Prompt(
        name=BUILD_MODEL_PROMPT,
        title="Build a Stella model",
        description="Guide an agent through building, validating, and saving a Stella model.",
        arguments=[PromptArgument(
            name="description",
            description="Natural-language description of the system to model",
            required=True,
        )],
    )]


def build_model_prompt(description: str | None) -> GetPromptResult:
    target = description.strip() if description else "the system described by the user"
    text = (
        f"Build a Stella system dynamics model of {target}.\n\n"
        "Recommended workflow:\n"
        "1. On MCP 2026-07-28, call create_workspace and include its returned "
        "workspace_id in every stateful call. Legacy stdio clients may omit it.\n"
        "2. Call build_model with a stable model_id and the full set of "
        "stocks, auxiliaries, and flows in one call. Connector sync and "
        "validation run by default, so the response doubles as an inspection.\n"
        "3. Fix any validation errors with update_*, rename_variable, or "
        "delete_variable.\n"
        "4. Extend incrementally with add_variables (batch) or the single-add "
        "tools.\n"
        "5. If the sim extra is installed, call simulate to sanity-check the "
        "model's behavior over time.\n"
        "6. Call render_diagram to inspect the stock-and-flow layout.\n"
        "7. Save with save_model.\n\n"
        "Identify the stocks (accumulations), flows (rates of change), and "
        "auxiliaries (parameters and intermediate calculations) before "
        "calling build_model."
    )
    return GetPromptResult(
        description=f"Workflow for modeling {target}",
        messages=[PromptMessage(role="user", content=TextContent(type="text", text=text))],
    )
