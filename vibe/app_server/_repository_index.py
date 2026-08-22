from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
import time
from typing import cast

from vibe.app_server._model import ProtocolModel
from vibe.app_server.repository_index import (
    RepositoryIndexCancelResponse,
    RepositoryIndexMapEntry,
    RepositoryIndexMapResponse,
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
type RecordEvent = Callable[[str, dict[str, object]], None]


class RepositoryIndexController:
    def __init__(
        self,
        service: RepositoryIndexLifecycle | None,
        session_id: str,
        notify: Notify,
        track_task: TrackTask | None = None,
        record_event: RecordEvent | None = None,
    ) -> None:
        self._service = service
        self._session_id = session_id
        self._notify = notify
        self._track_task = track_task or (lambda _task: None)
        self._record_event = record_event or (lambda _name, _properties: None)
        self._task: asyncio.Task[None] | None = None

    def status(self) -> RepositoryIndexStatusResponse:
        return RepositoryIndexStatusResponse(index=self._view())

    async def compact_map(self) -> RepositoryIndexMapResponse:
        compact = await self._require_service().compact_map()
        return RepositoryIndexMapResponse(
            generation=compact.generation,
            entries=[
                RepositoryIndexMapEntry(
                    path=entry.path,
                    importance=entry.importance,
                    component=entry.component,
                    symbols=list(entry.symbols),
                )
                for entry in compact.entries
            ],
        )

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
        started = time.perf_counter()
        outcome = "success"
        failure_class: str | None = None
        operation_task = asyncio.ensure_future(operation())
        previous: RepositoryIndexView | None = None
        try:
            while not operation_task.done():
                current = self._view()
                if current != previous:
                    await self._notify_state(current)
                    previous = current
                await asyncio.wait({operation_task}, timeout=0.1)
            await operation_task
        except asyncio.CancelledError:
            outcome = "cancelled"
            self._require_service().cancel()
            operation_task.cancel()
            with suppress(asyncio.CancelledError):
                await operation_task
            raise
        except RepositoryIndexUnavailableError as exc:
            outcome = (
                "cancelled"
                if self._require_service().state.status.value == "cancelled"
                else "failure"
            )
            failure_class = type(exc.__cause__ or exc).__name__
            logger.warning("Repository index %s failed: %s", name, exc)
        finally:
            view = self._view()
            await self._notify_state(view)
            self._record_event(
                "vibe.repository_index_operation",
                {
                    "operation": name,
                    "status": outcome,
                    "duration_ms": round((time.perf_counter() - started) * 1000),
                    "file_count": view.file_count,
                    "structural_file_count": view.structural_file_count,
                    "degraded_file_count": view.degraded_file_count,
                    "language_categories": sorted(view.language_counts),
                    "failure_class": failure_class,
                },
            )

    async def _notify_state(self, view: RepositoryIndexView) -> None:
        await self._notify(
            "repositoryIndex/updated",
            RepositoryIndexUpdatedParams(session_id=self._session_id, index=view),
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
            language_counts=(
                generation.language_counts if generation is not None else {}
            ),
            structural_file_count=(
                generation.structural_file_count if generation is not None else 0
            ),
            degraded_file_count=(
                generation.degraded_file_count if generation is not None else 0
            ),
            parse_error_count=(
                generation.parse_error_count if generation is not None else 0
            ),
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
