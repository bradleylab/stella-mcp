"""Focused contracts for session-scoped model storage."""

import pytest

from stella_mcp.session_store import (
    FALLBACK_SESSION_KEY,
    SessionStore,
    session_key_for,
)
from stella_mcp.xmile import StellaModel


def test_session_identity_uses_object_identity_and_fallback():
    session = object()

    assert session_key_for(session) == id(session)
    assert session_key_for(None) == FALLBACK_SESSION_KEY


def test_sessions_isolate_identical_model_ids():
    store = SessionStore()
    session_one = object()
    session_two = object()
    key_one = session_key_for(session_one)
    key_two = session_key_for(session_two)

    store.set_current(key_one, StellaModel("Session One"), "shared")
    store.set_current(key_two, StellaModel("Session Two"), "shared")

    assert store.get(key_one, "shared")[1].name == "Session One"
    assert store.get(key_two, "shared")[1].name == "Session Two"


def test_get_and_set_current_transitions_are_explicit():
    store = SessionStore()
    session = object()
    key = session_key_for(session)
    store.set_current(key, StellaModel("First"), "first")
    store.set_current(key, StellaModel("Second"), "second")

    assert [(entry.model_id, entry.current) for entry in store.list(key)] == [
        ("first", False),
        ("second", True),
    ]

    resolved_id, model = store.get(key, "first")

    assert resolved_id == "first"
    assert model.name == "First"
    assert [(entry.model_id, entry.current) for entry in store.list(key)] == [
        ("first", True),
        ("second", False),
    ]


def test_delete_reports_remaining_state_and_clears_current():
    store = SessionStore()
    session = object()
    key = session_key_for(session)
    store.set_current(key, StellaModel("First"), "first")
    store.set_current(key, StellaModel("Second"), "second")

    result = store.delete(key, "second")

    assert result.deleted == "second"
    assert result.remaining == ("first",)
    assert result.current_model_id is None
    with pytest.raises(ValueError, match="No model created in this session"):
        store.get(key)


def test_clear_supports_session_teardown_and_full_test_cleanup():
    store = SessionStore()
    session_one = object()
    session_two = object()
    key_one = session_key_for(session_one)
    key_two = session_key_for(session_two)
    store.set_current(key_one, StellaModel("One"), "one")
    store.set_current(key_two, StellaModel("Two"), "two")

    store.clear(key_one)

    assert store.list(key_one) == ()
    assert [entry.model_id for entry in store.list(key_two)] == ["two"]

    store.clear()

    assert store.list(key_two) == ()
