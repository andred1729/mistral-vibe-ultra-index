from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import cast

from vibe.app_server._model import ProtocolModel
from vibe.app_server.repository_index import (
    RepositoryIndexCancelResponse,
    RepositoryIndexMutationResponse,
    RepositoryIndexPublicPhase,
    RepositoryIndexPublicStatus,
    RepositoryIndexStatusResponse,
    RepositoryIndexUpdatedParams,
    RepositoryIndexView,
)
from vibe.core.repository_index import (
    RepositoryIndexLifecycle,
    RepositoryIndexUnavailableError,
)
from vibe.observability.logging import logger

type Notify = Callable[[str, ProtocolModel], Awaitable[None]]
type TrackTask = Callable[[asyncio.Task[None]], None]


class RepositoryIndexController:
    def __init__(
        self,
        service: RepositoryIndexLifecycle | None,
        session_id: str,
        notify: Notify,
        track_task: TrackTask | None = None,
    ) -> None:
        self._service = service
        self._session_id = session_id
        self._notify = notify
        self._track_task = track_task or (lambda _task: None)
        self._task: asyncio.Task[None] | None = None

    def status(self) -> RepositoryIndexStatusResponse:
        return RepositoryIndexStatusResponse(index=self._view())

    async def refresh(self) -> RepositoryIndexMutationResponse:
        return await self._start("refresh", self._require_service().ensure_ready)

    async def rebuild(self) -> RepositoryIndexMutationResponse:
        return await self._start("rebuild", self._require_service().rebuild)

    async def clear(self) -> RepositoryIndexMutationResponse:
        # A cleared index cannot remain useful because inference requires a complete
        # generation. Clear and rebuild are therefore one background operation.
        return await self._start("clear", self._require_service().rebuild)

    def cancel(self) -> RepositoryIndexCancelResponse:
        service = self._require_service()
        active = self._task is not None and not self._task.done()
        building = service.state.status.value == "building"
        if active or building:
            service.cancel()
        return RepositoryIndexCancelResponse(
            index=self._view(), cancelled=active or building
        )

    async def _start(
        self, name: str, operation: Callable[[], Awaitable[object]]
    ) -> RepositoryIndexMutationResponse:
        if self._task is not None and not self._task.done():
            return RepositoryIndexMutationResponse(index=self._view(), started=False)
        self._task = asyncio.create_task(
            self._run(name, operation), name=f"repository-index:{name}"
        )
        self._track_task(self._task)
        await asyncio.sleep(0)
        return RepositoryIndexMutationResponse(index=self._view(), started=True)

    async def _run(self, name: str, operation: Callable[[], Awaitable[object]]) -> None:
        try:
            await operation()
        except asyncio.CancelledError:
            raise
        except RepositoryIndexUnavailableError as exc:
            logger.warning("Repository index %s failed: %s", name, exc)
        finally:
            await self._notify(
                "repositoryIndex/updated",
                RepositoryIndexUpdatedParams(
                    session_id=self._session_id, index=self._view()
                ),
            )

    def _view(self) -> RepositoryIndexView:
        service = self._require_service()
        state = service.state
        generation = state.generation
        return RepositoryIndexView(
            root=str(state.root) if state.root is not None else None,
            status=cast("RepositoryIndexPublicStatus", state.status.value),
            phase=cast("RepositoryIndexPublicPhase", state.phase.value),
            dirty=state.dirty,
            generation=generation.id if generation is not None else None,
            file_count=generation.file_count if generation is not None else 0,
            files_processed=state.files_processed,
            files_total=state.files_total,
            created_at=generation.created_at if generation is not None else None,
            completed_at=generation.completed_at if generation is not None else None,
            error=state.error,
        )

    def _require_service(self) -> RepositoryIndexLifecycle:
        if self._service is None:
            raise RuntimeError("Repository index is not available for this session")
        return self._service
