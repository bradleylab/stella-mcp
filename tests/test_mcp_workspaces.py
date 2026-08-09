"""Protocol-facing workspace routing and resource isolation tests."""

from __future__ import annotations

import asyncio
import copy
import xml.etree.ElementTree as ET

import pytest
from mcp.types import LATEST_PROTOCOL_VERSION

from stella_mcp import server as server_mod
from stella_mcp.mcp_resources import list_model_resources, read_resource_content
from stella_mcp.session_store import (
    WorkspaceExpiredError,
    WorkspaceRevokedError,
    WorkspaceStore,
)


def _modern_call(name: str, arguments: dict):
    return asyncio.run(
        server_mod.call_tool(
            name,
            arguments,
            protocol_version=LATEST_PROTOCOL_VERSION,
        )
    )


def test_modern_workspace_handles_isolate_identical_model_ids():
    server_mod._clear_session_store()
    one = _modern_call("create_workspace", {}).structured_content["workspace_id"]
    two = _modern_call("create_workspace", {}).structured_content["workspace_id"]

    _modern_call(
        "create_model",
        {"workspace_id": one, "model_id": "shared", "name": "One"},
    )
    _modern_call(
        "create_model",
        {"workspace_id": two, "model_id": "shared", "name": "Two"},
    )

    first = _modern_call("inspect_model", {"workspace_id": one, "model_id": "shared"})
    second = _modern_call("inspect_model", {"workspace_id": two, "model_id": "shared"})
    assert first.structured_content["model"]["name"] == "One"
    assert second.structured_content["model"]["name"] == "Two"


def test_workspace_resource_uri_routes_without_ambient_context():
    server_mod._clear_session_store()
    workspace_id = _modern_call("create_workspace", {}).structured_content["workspace_id"]
    _modern_call(
        "create_model",
        {"workspace_id": workspace_id, "model_id": "my model", "name": "Resource"},
    )

    entries = server_mod._workspace_store.list(workspace_id)
    [resource] = list_model_resources(entries, workspace_id=workspace_id)
    content, mime_type = read_resource_content(
        str(resource.uri), workspace_store=server_mod._workspace_store
    )

    assert workspace_id in str(resource.uri)
    assert mime_type == "application/xml"
    ET.fromstring(content)


def test_workspace_resource_read_does_not_change_current_model():
    server_mod._clear_session_store()
    workspace_id = _modern_call("create_workspace", {}).structured_content["workspace_id"]
    _modern_call(
        "create_model",
        {"workspace_id": workspace_id, "model_id": "first", "name": "First"},
    )
    _modern_call(
        "create_model",
        {"workspace_id": workspace_id, "model_id": "second", "name": "Second"},
    )

    read_resource_content(
        f"stella://workspaces/{workspace_id}/models/first",
        workspace_store=server_mod._workspace_store,
    )

    current = _modern_call("inspect_model", {"workspace_id": workspace_id})
    assert current.structured_content["model"]["name"] == "Second"


def test_revoked_workspace_cannot_be_reused():
    server_mod._clear_session_store()
    workspace_id = _modern_call("create_workspace", {}).structured_content["workspace_id"]
    revoked = _modern_call("revoke_workspace", {"workspace_id": workspace_id})
    assert revoked.structured_content == {"workspace_id": workspace_id, "revoked": True}

    replay = _modern_call(
        "create_model",
        {"workspace_id": workspace_id, "model_id": "m", "name": "Replay"},
    )
    assert replay.is_error is True
    assert replay.structured_content["error"]["code"] == "workspace_revoked"


def test_waiting_call_rechecks_workspace_lifecycle_after_lock_acquisition():
    server_mod._clear_session_store()
    workspace_id = _modern_call("create_workspace", {}).structured_content["workspace_id"]

    async def exercise():
        lock = server_mod._workspace_store.lock_for(workspace_id)
        await lock.acquire()
        waiting = asyncio.create_task(
            server_mod.call_tool(
                "list_models",
                {"workspace_id": workspace_id},
                protocol_version=LATEST_PROTOCOL_VERSION,
            )
        )
        await asyncio.sleep(0)
        server_mod._workspace_store.revoke(workspace_id)
        lock.release()
        return await waiting

    result = asyncio.run(exercise())

    assert result.is_error is True
    assert result.structured_content["error"]["code"] == "workspace_revoked"


def test_public_routing_preserves_expiry_for_repeated_and_waiting_calls(monkeypatch):
    now = {"value": 10.0}
    store = WorkspaceStore(clock=lambda: now["value"])
    monkeypatch.setattr(server_mod, "_workspace_store", store)
    workspace_id = _modern_call(
        "create_workspace", {"ttl_seconds": 2.0}
    ).structured_content["workspace_id"]

    async def exercise_waiter():
        lock = store.lock_for(workspace_id)
        await lock.acquire()
        waiters = [
            asyncio.create_task(
                server_mod.call_tool(
                    "list_models",
                    {"workspace_id": workspace_id},
                    protocol_version=LATEST_PROTOCOL_VERSION,
                )
            )
            for _ in range(2)
        ]
        await asyncio.sleep(0)
        now["value"] = 12.0
        lock.release()
        return await asyncio.gather(*waiters)

    waiters = asyncio.run(exercise_waiter())
    repeated = _modern_call("list_models", {"workspace_id": workspace_id})

    assert [result.structured_content["error"]["code"] for result in waiters] == [
        "workspace_expired",
        "workspace_expired",
    ]
    assert repeated.structured_content["error"]["code"] == "workspace_expired"


def test_workspace_resource_reads_preserve_expiry_and_revocation_classification():
    now = {"value": 10.0}
    store = WorkspaceStore(clock=lambda: now["value"])
    expired = store.create(ttl_seconds=1.0)
    revoked = store.create()
    store.set_current(expired, server_mod.StellaModel("Expired"), "model")
    store.set_current(revoked, server_mod.StellaModel("Revoked"), "model")
    store.revoke(revoked)
    now["value"] = 11.0

    with pytest.raises(WorkspaceExpiredError):
        read_resource_content(
            f"stella://workspaces/{expired}/models/model",
            workspace_store=store,
        )
    with pytest.raises(WorkspaceExpiredError):
        read_resource_content(
            f"stella://workspaces/{expired}/models/model",
            workspace_store=store,
        )
    with pytest.raises(WorkspaceRevokedError):
        read_resource_content(
            f"stella://workspaces/{revoked}/models/model",
            workspace_store=store,
        )


@pytest.mark.parametrize("tool_name", ["create_workspace", "revoke_workspace"])
def test_lifecycle_successes_use_output_schema_validation(monkeypatch, tool_name):
    server_mod._clear_session_store()
    workspace_id = _modern_call("create_workspace", {}).structured_content["workspace_id"]
    schema = copy.deepcopy(server_mod._OUTPUT_SCHEMAS[tool_name])
    schema["required"].append("review_probe")
    monkeypatch.setitem(server_mod._OUTPUT_SCHEMAS, tool_name, schema)

    arguments = {} if tool_name == "create_workspace" else {"workspace_id": workspace_id}
    result = _modern_call(tool_name, arguments)

    assert result.is_error is True
    assert result.structured_content["error"]["code"] == "internal_error"
