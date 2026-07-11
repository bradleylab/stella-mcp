"""Session-scoped model storage for the Stella MCP server.

The stdio server identifies a live MCP session by the identity of its session
object. An HTTP transport must either provide its own stable identity or call
``SessionStore.clear(session_key)`` during lifecycle teardown; retaining an
``id(session)`` key after the object is released can allow Python to reuse it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from .xmile import StellaModel

FALLBACK_SESSION_KEY = -1


def session_key_for(session: object | None) -> int:
    """Return the current transport session identity or the test fallback key."""
    return FALLBACK_SESSION_KEY if session is None else id(session)


@dataclass(frozen=True)
class SessionModelEntry:
    """Read-only description of one model registered in a session."""

    model_id: str
    model: StellaModel
    current: bool


@dataclass(frozen=True)
class SessionDeleteResult:
    """State reported after deleting one session model."""

    deleted: str
    remaining: tuple[str, ...]
    current_model_id: str | None


@dataclass
class _SessionState:
    models: dict[str, StellaModel] = field(default_factory=dict)
    current_model_id: str | None = None


class SessionStore:
    """Own model registries and current-model pointers for MCP sessions."""

    def __init__(self) -> None:
        self._sessions: dict[int, _SessionState] = {}

    def _state(self, session_key: int) -> _SessionState:
        return self._sessions.setdefault(session_key, _SessionState())

    def get(
        self,
        session_key: int,
        model_id: str | None = None,
    ) -> tuple[str, StellaModel]:
        """Return a requested or current model and make it current."""
        state = self._state(session_key)
        resolved_id = model_id or state.current_model_id
        if not resolved_id:
            raise ValueError("No model created in this session. Use create_model first.")
        model = state.models.get(resolved_id)
        if model is None:
            raise ValueError(f"Unknown model_id '{resolved_id}' for this session")
        state.current_model_id = resolved_id
        return resolved_id, model

    def set_current(
        self,
        session_key: int,
        model: StellaModel,
        model_id: str | None = None,
    ) -> str:
        """Register a model under a unique ID and make it current."""
        state = self._state(session_key)
        resolved_id = model_id or f"model_{uuid.uuid4().hex[:8]}"
        if resolved_id in state.models:
            raise ValueError(f"model_id '{resolved_id}' already exists in this session")
        state.models[resolved_id] = model
        state.current_model_id = resolved_id
        return resolved_id

    def delete(self, session_key: int, model_id: str) -> SessionDeleteResult:
        """Delete a model and clear the current pointer when it targets that model."""
        state = self._state(session_key)
        if model_id not in state.models:
            raise ValueError(f"Unknown model_id '{model_id}' for this session")
        del state.models[model_id]
        if state.current_model_id == model_id:
            state.current_model_id = None
        return SessionDeleteResult(
            deleted=model_id,
            remaining=tuple(sorted(state.models)),
            current_model_id=state.current_model_id,
        )

    def list(self, session_key: int) -> tuple[SessionModelEntry, ...]:
        """Return session models sorted by ID with current-model metadata."""
        state = self._state(session_key)
        return tuple(
            SessionModelEntry(
                model_id=model_id,
                model=state.models[model_id],
                current=model_id == state.current_model_id,
            )
            for model_id in sorted(state.models)
        )

    def contains(self, session_key: int, model_id: str) -> bool:
        """Return whether a model ID is registered in a session."""
        return model_id in self._state(session_key).models

    def replace(self, session_key: int, model_id: str, model: StellaModel) -> None:
        """Atomically replace an existing model without changing the current pointer."""
        state = self._state(session_key)
        if model_id not in state.models:
            raise ValueError(f"Unknown model_id '{model_id}' for this session")
        state.models[model_id] = model

    def clear(self, session_key: int | None = None) -> None:
        """Clear one session, or every session when no key is provided."""
        if session_key is None:
            self._sessions.clear()
            return
        self._sessions.pop(session_key, None)
