from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Protocol

from vibe.core.repository_index.models import (
    IndexGeneration,
    RepositoryIndexState,
    RepositorySearchMode,
    RepositorySearchResult,
)


class RepositoryIndexReader(Protocol):
    @property
    def state(self) -> RepositoryIndexState: ...

    def inference_scope(self) -> AbstractAsyncContextManager[IndexGeneration]: ...

    async def search(
        self,
        query: str,
        *,
        mode: RepositorySearchMode = RepositorySearchMode.AUTO,
        max_results: int = 20,
    ) -> RepositorySearchResult: ...


class RepositoryIndexLifecycle(RepositoryIndexReader, Protocol):
    async def ensure_ready(self) -> IndexGeneration: ...

    def cancel(self) -> None: ...

    async def rebuild(self) -> IndexGeneration: ...

    async def clear(self) -> None: ...
