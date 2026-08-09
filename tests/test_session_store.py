"""Focused contracts for application-owned model workspaces."""

import asyncio

import pytest

from stella_mcp.session_store import (
    LEGACY_WORKSPACE_ID,
    WorkspaceExpiredError,
    WorkspaceNotFoundError,
    WorkspaceRevokedError,
    WorkspaceStore,
)
from stella_mcp.xmile import StellaModel


def test_legacy_workspace_is_explicit_and_unknown_ids_fail():
    store = WorkspaceStore()

    assert store.list(LEGACY_WORKSPACE_ID) == ()
    with pytest.raises(WorkspaceNotFoundError, match="Unknown workspace_id"):
        store.list("missing")


def test_workspaces_isolate_identical_model_ids():
    store = WorkspaceStore()
    one = store.create()
    two = store.create()

    store.set_current(one, StellaModel("Workspace One"), "shared")
    store.set_current(two, StellaModel("Workspace Two"), "shared")

    assert store.get(one, "shared")[1].name == "Workspace One"
    assert store.get(two, "shared")[1].name == "Workspace Two"


def test_get_and_set_current_transitions_are_explicit():
    store = WorkspaceStore()
    workspace_id = store.create()
    store.set_current(workspace_id, StellaModel("First"), "first")
    store.set_current(workspace_id, StellaModel("Second"), "second")

    assert [(entry.model_id, entry.current) for entry in store.list(workspace_id)] == [
        ("first", False),
        ("second", True),
    ]

    resolved_id, model = store.get(workspace_id, "first")

    assert resolved_id == "first"
    assert model.name == "First"
    assert [(entry.model_id, entry.current) for entry in store.list(workspace_id)] == [
        ("first", True),
        ("second", False),
    ]


def test_lookup_does_not_change_current_model():
    store = WorkspaceStore()
    workspace_id = store.create()
    store.set_current(workspace_id, StellaModel("First"), "first")
    store.set_current(workspace_id, StellaModel("Second"), "second")

    model = store.lookup(workspace_id, "first")

    assert model.name == "First"
    assert [(entry.model_id, entry.current) for entry in store.list(workspace_id)] == [
        ("first", False),
        ("second", True),
    ]


def test_delete_reports_remaining_state_and_clears_current():
    store = WorkspaceStore()
    workspace_id = store.create()
    store.set_current(workspace_id, StellaModel("First"), "first")
    store.set_current(workspace_id, StellaModel("Second"), "second")

    result = store.delete(workspace_id, "second")

    assert result.deleted == "second"
    assert result.remaining == ("first",)
    assert result.current_model_id is None
    with pytest.raises(ValueError, match="No model created in this workspace"):
        store.get(workspace_id)


def test_revoke_and_expiry_have_distinct_errors():
    now = {"value": 10.0}
    store = WorkspaceStore(clock=lambda: now["value"])
    revoked = store.create()
    expiring = store.create(ttl_seconds=2.0)

    store.revoke(revoked)
    with pytest.raises(WorkspaceRevokedError):
        store.get(revoked)

    now["value"] = 12.0
    with pytest.raises(WorkspaceExpiredError):
        store.get(expiring)
    with pytest.raises(WorkspaceExpiredError):
        store.get(expiring)


def test_expiry_cleanup_retains_bounded_classification():
    now = {"value": 10.0}
    store = WorkspaceStore(clock=lambda: now["value"], tombstone_limit=2)
    workspace_ids = [store.create(ttl_seconds=1.0) for _ in range(3)]

    now["value"] = 11.0
    assert store.cleanup_expired() == 3

    with pytest.raises(WorkspaceNotFoundError):
        store.require(workspace_ids[0])
    for workspace_id in workspace_ids[1:]:
        with pytest.raises(WorkspaceExpiredError):
            store.require(workspace_id)

    store.clear(workspace_ids[-1])
    with pytest.raises(WorkspaceNotFoundError):
        store.require(workspace_ids[-1])


def test_locks_are_workspace_scoped_and_serialize_same_workspace():
    store = WorkspaceStore()
    one = store.create()
    two = store.create()
    assert store.lock_for(one) is not store.lock_for(two)

    events: list[str] = []

    async def exercise() -> None:
        lock = store.lock_for(one)

        async def worker(label: str) -> None:
            async with lock:
                events.append(f"{label}:start")
                await asyncio.sleep(0)
                events.append(f"{label}:end")

        await asyncio.gather(worker("a"), worker("b"))

    asyncio.run(exercise())
    assert events in (
        ["a:start", "a:end", "b:start", "b:end"],
        ["b:start", "b:end", "a:start", "a:end"],
    )


def test_clear_supports_targeted_cleanup_and_full_test_reset():
    store = WorkspaceStore()
    one = store.create()
    two = store.create()
    store.set_current(one, StellaModel("One"), "one")
    store.set_current(two, StellaModel("Two"), "two")

    store.clear(one)
    with pytest.raises(WorkspaceNotFoundError):
        store.list(one)
    assert [entry.model_id for entry in store.list(two)] == ["two"]

    store.clear()
    assert store.list(LEGACY_WORKSPACE_ID) == ()
    with pytest.raises(WorkspaceNotFoundError):
        store.list(two)
