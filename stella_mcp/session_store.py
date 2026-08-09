"""Application-owned model workspaces for the Stella MCP server.

MCP 2026-07-28 requests are stateless at the protocol layer.  Stella therefore
routes mutable model state with an explicit, opaque ``workspace_id``.  The one
reserved legacy workspace exists only for supported pre-2026 stdio clients
whose existing tool calls do not include that argument.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from .xmile import StellaModel

LEGACY_WORKSPACE_ID = "legacy"
DEFAULT_TOMBSTONE_LIMIT = 1024
_TombstoneReason = Literal["expired", "revoked"]


class WorkspaceError(ValueError):
    """Base class for classified workspace routing failures."""


class WorkspaceNotFoundError(WorkspaceError):
    """The supplied workspace identifier is unknown or malformed."""


class WorkspaceExpiredError(WorkspaceError):
    """The supplied workspace has passed its caller-selected expiry."""


class WorkspaceRevokedError(WorkspaceError):
    """The supplied workspace was explicitly revoked."""


@dataclass(frozen=True)
class SessionModelEntry:
    """Read-only description of one model registered in a workspace."""

    model_id: str
    model: StellaModel
    current: bool


@dataclass(frozen=True)
class SessionDeleteResult:
    """State reported after deleting one workspace model."""

    deleted: str
    remaining: tuple[str, ...]
    current_model_id: str | None


@dataclass
class _WorkspaceState:
    models: dict[str, StellaModel] = field(default_factory=dict)
    current_model_id: str | None = None
    expires_at: float | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class WorkspaceStore:
    """Own isolated model registries, lifecycle, and per-workspace locks."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        tombstone_limit: int = DEFAULT_TOMBSTONE_LIMIT,
    ) -> None:
        if tombstone_limit <= 0:
            raise ValueError("tombstone_limit must be greater than zero")
        self._clock = clock
        self._tombstone_limit = tombstone_limit
        self._workspaces: dict[str, _WorkspaceState] = {
            LEGACY_WORKSPACE_ID: _WorkspaceState()
        }
        self._tombstones: OrderedDict[str, _TombstoneReason] = OrderedDict()

    def _remember_tombstone(
        self, workspace_id: str, reason: _TombstoneReason
    ) -> None:
        self._tombstones[workspace_id] = reason
        self._tombstones.move_to_end(workspace_id)
        while len(self._tombstones) > self._tombstone_limit:
            self._tombstones.popitem(last=False)

    def create(self, *, ttl_seconds: float | None = None) -> str:
        """Create an opaque workspace, optionally with a caller-selected TTL."""
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise WorkspaceError("ttl_seconds must be greater than zero")
        workspace_id = f"workspace_{uuid.uuid4().hex}"
        expires_at = None if ttl_seconds is None else self._clock() + ttl_seconds
        self._workspaces[workspace_id] = _WorkspaceState(expires_at=expires_at)
        return workspace_id

    def ensure_test_workspace(self, workspace_id: str) -> None:
        """Create a deterministic compatibility workspace for direct unit tests."""
        if not workspace_id.startswith("test_workspace_"):
            raise WorkspaceError("Only test workspace identifiers may be synthesized")
        self._workspaces.setdefault(workspace_id, _WorkspaceState())

    def require(self, workspace_id: str) -> _WorkspaceState:
        """Resolve an active workspace or raise a stable lifecycle error."""
        if not isinstance(workspace_id, str) or not workspace_id:
            raise WorkspaceNotFoundError("workspace_id must be a non-empty string")
        tombstone = self._tombstones.get(workspace_id)
        if tombstone == "expired":
            raise WorkspaceExpiredError(f"Workspace '{workspace_id}' has expired")
        if tombstone == "revoked":
            raise WorkspaceRevokedError(f"Workspace '{workspace_id}' has been revoked")
        state = self._workspaces.get(workspace_id)
        if state is None:
            raise WorkspaceNotFoundError(f"Unknown workspace_id '{workspace_id}'")
        if state.expires_at is not None and self._clock() >= state.expires_at:
            del self._workspaces[workspace_id]
            self._remember_tombstone(workspace_id, "expired")
            raise WorkspaceExpiredError(f"Workspace '{workspace_id}' has expired")
        return state

    def revoke(self, workspace_id: str) -> None:
        """Revoke a non-legacy workspace and discard its model state."""
        if workspace_id == LEGACY_WORKSPACE_ID:
            raise WorkspaceError("The legacy compatibility workspace cannot be revoked")
        self.require(workspace_id)
        del self._workspaces[workspace_id]
        self._remember_tombstone(workspace_id, "revoked")

    def lock_for(self, workspace_id: str) -> asyncio.Lock:
        """Return the lifecycle-bound serialization lock for a workspace."""
        return self.require(workspace_id).lock

    def get(
        self,
        workspace_id: str,
        model_id: str | None = None,
    ) -> tuple[str, StellaModel]:
        """Return a requested or current model and make it current."""
        state = self.require(workspace_id)
        resolved_id = model_id or state.current_model_id
        if not resolved_id:
            raise ValueError("No model created in this workspace. Use create_model first.")
        model = state.models.get(resolved_id)
        if model is None:
            raise ValueError(f"Unknown model_id '{resolved_id}' for this workspace")
        state.current_model_id = resolved_id
        return resolved_id, model

    def lookup(self, workspace_id: str, model_id: str) -> StellaModel:
        """Return a named model without changing the workspace's current pointer."""
        state = self.require(workspace_id)
        model = state.models.get(model_id)
        if model is None:
            raise ValueError(f"Unknown model_id '{model_id}' for this workspace")
        return model

    def set_current(
        self,
        workspace_id: str,
        model: StellaModel,
        model_id: str | None = None,
    ) -> str:
        """Register a model under a unique ID and make it current."""
        state = self.require(workspace_id)
        resolved_id = model_id or f"model_{uuid.uuid4().hex[:8]}"
        if resolved_id in state.models:
            raise ValueError(f"model_id '{resolved_id}' already exists in this workspace")
        state.models[resolved_id] = model
        state.current_model_id = resolved_id
        return resolved_id

    def delete(self, workspace_id: str, model_id: str) -> SessionDeleteResult:
        """Delete a model and clear the current pointer when it targets that model."""
        state = self.require(workspace_id)
        if model_id not in state.models:
            raise ValueError(f"Unknown model_id '{model_id}' for this workspace")
        del state.models[model_id]
        if state.current_model_id == model_id:
            state.current_model_id = None
        return SessionDeleteResult(
            deleted=model_id,
            remaining=tuple(sorted(state.models)),
            current_model_id=state.current_model_id,
        )

    def list(self, workspace_id: str) -> tuple[SessionModelEntry, ...]:
        """Return workspace models sorted by ID with current-model metadata."""
        state = self.require(workspace_id)
        return tuple(
            SessionModelEntry(
                model_id=model_id,
                model=state.models[model_id],
                current=model_id == state.current_model_id,
            )
            for model_id in sorted(state.models)
        )

    def contains(self, workspace_id: str, model_id: str) -> bool:
        """Return whether a model ID is registered in a workspace."""
        return model_id in self.require(workspace_id).models

    def replace(self, workspace_id: str, model_id: str, model: StellaModel) -> None:
        """Atomically replace an existing model without changing the current pointer."""
        state = self.require(workspace_id)
        if model_id not in state.models:
            raise ValueError(f"Unknown model_id '{model_id}' for this workspace")
        state.models[model_id] = model

    def cleanup_expired(self) -> int:
        """Remove expired workspaces while retaining bounded error classification."""
        now = self._clock()
        expired = [
            workspace_id
            for workspace_id, state in self._workspaces.items()
            if workspace_id != LEGACY_WORKSPACE_ID
            and state.expires_at is not None
            and now >= state.expires_at
        ]
        for workspace_id in expired:
            del self._workspaces[workspace_id]
            self._remember_tombstone(workspace_id, "expired")
        return len(expired)

    def clear(self, workspace_id: str | None = None) -> None:
        """Clear one workspace, or reset all workspaces for tests and shutdown."""
        if workspace_id is None:
            self._workspaces = {LEGACY_WORKSPACE_ID: _WorkspaceState()}
            self._tombstones.clear()
            return
        if workspace_id == LEGACY_WORKSPACE_ID:
            self._workspaces[workspace_id] = _WorkspaceState()
            return
        self._workspaces.pop(workspace_id, None)
        self._tombstones.pop(workspace_id, None)


# Retain the old import name for downstream Python callers while changing its
# semantics from transport sessions to explicit application workspaces.
SessionStore = WorkspaceStore
