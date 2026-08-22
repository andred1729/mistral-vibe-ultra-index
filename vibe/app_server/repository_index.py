from __future__ import annotations

from typing import Literal

from vibe.app_server._model import ProtocolModel

type RepositoryIndexPublicStatus = Literal[
    "building", "complete", "cancelled", "failed"
]
type RepositoryIndexPublicPhase = Literal[
    "idle", "discovering", "chunking", "parsing", "graph", "publishing", "clearing"
]


class RepositoryIndexView(ProtocolModel):
    root: str | None = None
    status: RepositoryIndexPublicStatus
    phase: RepositoryIndexPublicPhase
    dirty: bool
    generation: int | None = None
    file_count: int = 0
    files_processed: int = 0
    files_total: int = 0
    created_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


class RepositoryIndexParams(ProtocolModel):
    session_id: str


class RepositoryIndexStatusResponse(ProtocolModel):
    index: RepositoryIndexView


class RepositoryIndexMutationResponse(RepositoryIndexStatusResponse):
    started: bool


class RepositoryIndexCancelResponse(RepositoryIndexStatusResponse):
    cancelled: bool


class RepositoryIndexUpdatedParams(ProtocolModel):
    session_id: str
    index: RepositoryIndexView


__all__ = [
    "RepositoryIndexCancelResponse",
    "RepositoryIndexMutationResponse",
    "RepositoryIndexParams",
    "RepositoryIndexStatusResponse",
    "RepositoryIndexUpdatedParams",
    "RepositoryIndexView",
]
