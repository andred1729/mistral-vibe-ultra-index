from __future__ import annotations

from collections.abc import Callable

from vibe.app_server._model import validate_wire
from vibe.app_server.client_state import ClientSessionState
from vibe.app_server.connection import AppServerResourceConnection
from vibe.app_server.protocol import Notification
from vibe.app_server.repository_index import (
    RepositoryIndexCancelResponse,
    RepositoryIndexMapResponse,
    RepositoryIndexMutationResponse,
    RepositoryIndexParams,
    RepositoryIndexStatusResponse,
    RepositoryIndexUpdatedParams,
    RepositoryIndexView,
)


class RepositoryIndexResource:
    def __init__(
        self, connection: AppServerResourceConnection, state: ClientSessionState
    ) -> None:
        self._connection = connection
        self._state = state
        self._current: RepositoryIndexView | None = None
        self._subscribers: list[Callable[[RepositoryIndexView], None]] = []

    @property
    def current(self) -> RepositoryIndexView | None:
        return self._current

    def subscribe(
        self, callback: Callable[[RepositoryIndexView], None]
    ) -> Callable[[], None]:
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return unsubscribe

    async def status(self) -> RepositoryIndexView:
        client = await self._connection.connect()
        response = validate_wire(
            RepositoryIndexStatusResponse,
            await client.request(
                "repositoryIndex/status",
                RepositoryIndexParams(session_id=self._state.session_id),
            ),
        )
        self._publish(response.index)
        return response.index

    async def refresh(self) -> RepositoryIndexMutationResponse:
        return await self._mutation("repositoryIndex/refresh")

    async def compact_map(self) -> RepositoryIndexMapResponse:
        client = await self._connection.connect()
        return validate_wire(
            RepositoryIndexMapResponse,
            await client.request(
                "repositoryIndex/map",
                RepositoryIndexParams(session_id=self._state.session_id),
            ),
        )

    async def rebuild(self) -> RepositoryIndexMutationResponse:
        return await self._mutation("repositoryIndex/rebuild")

    async def clear(self) -> RepositoryIndexMutationResponse:
        return await self._mutation("repositoryIndex/clear")

    async def cancel(self) -> RepositoryIndexCancelResponse:
        client = await self._connection.connect()
        response = validate_wire(
            RepositoryIndexCancelResponse,
            await client.request(
                "repositoryIndex/cancel",
                RepositoryIndexParams(session_id=self._state.session_id),
            ),
        )
        self._publish(response.index)
        return response

    async def consume_notification(self, notification: Notification) -> bool:
        if notification.method != "repositoryIndex/updated":
            return False
        params = validate_wire(RepositoryIndexUpdatedParams, notification.params)
        if params.session_id != self._state.session_id:
            return False
        self._publish(params.index)
        return True

    async def _mutation(self, method: str) -> RepositoryIndexMutationResponse:
        client = await self._connection.connect()
        response = validate_wire(
            RepositoryIndexMutationResponse,
            await client.request(
                method, RepositoryIndexParams(session_id=self._state.session_id)
            ),
        )
        self._publish(response.index)
        return response

    def _publish(self, view: RepositoryIndexView) -> None:
        if view == self._current:
            return
        self._current = view
        for callback in list(self._subscribers):
            callback(view)


__all__ = ["RepositoryIndexResource"]
