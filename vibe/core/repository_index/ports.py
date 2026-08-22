from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Protocol

from vibe.core.repository_index.models import (
    IndexGeneration,
    RepositoryDependencyDirection,
    RepositoryIndexState,
    RepositoryMap,
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
        direction: RepositoryDependencyDirection = (
            RepositoryDependencyDirection.DEPENDENCIES
        ),
        path: str | None = None,
        max_results: int = 20,
    ) -> RepositorySearchResult: ...

    async def compact_map(self, *, max_files: int = 50) -> RepositoryMap: ...


class RepositoryIndexLifecycle(RepositoryIndexReader, Protocol):
    async def ensure_ready(self) -> IndexGeneration: ...

    def cancel(self) -> None: ...

    async def rebuild(self) -> IndexGeneration: ...

    async def clear(self) -> None: ...
