from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
import hashlib
import os
from pathlib import Path
from threading import Event

from vibe.core.repository_index.chunks import (
    RepositoryChangedDuringBuildError,
    build_text_chunks,
)
from vibe.core.repository_index.discovery import (
    DiscoveryCancelledError,
    RepositoryDiscovery,
)
from vibe.core.repository_index.facts import build_file_facts
from vibe.core.repository_index.graph import hydrate_dependency_graph
from vibe.core.repository_index.models import (
    IndexGeneration,
    IndexPhase,
    IndexStatus,
    RepositoryIndexState,
    RepositorySearchMode,
    RepositorySearchResult,
)
from vibe.core.repository_index.store import RepositoryIndexStore

_RETAINED_GENERATIONS = 2
_BUILD_ATTEMPTS = 3


class RepositoryIndexUnavailableError(Exception): ...


def repository_identity(root: Path) -> str:
    normalized = os.path.normcase(str(root.resolve()))
    return hashlib.sha256(normalized.encode()).hexdigest()[:24]


class RepositoryIndexService:
    def __init__(
        self,
        cwd: Path,
        storage_root: Path,
        *,
        discovery: RepositoryDiscovery | None = None,
    ) -> None:
        self._cwd = cwd
        self._storage_root = storage_root
        self._discovery = discovery or RepositoryDiscovery()
        self._root: Path | None = None
        self._store: RepositoryIndexStore | None = None
        self._state = RepositoryIndexState()
        self._build_lock = asyncio.Lock()
        self._pin_lock = asyncio.Lock()
        self._generation_pins: dict[int, int] = {}
        self._inference_generation: ContextVar[IndexGeneration | None] = ContextVar(
            f"repository_index_generation_{id(self)}", default=None
        )
        self._cancel_event = Event()
        self._closed = False

    @property
    def state(self) -> RepositoryIndexState:
        return self._state

    async def ensure_ready(self) -> IndexGeneration:  # noqa: PLR0914, PLR0915
        if self._closed:
            raise RepositoryIndexUnavailableError("Repository index service is closed.")
        async with self._build_lock:  # noqa: PLR1702
            self._cancel_event = Event()
            self._state = self._state.model_copy(
                update={
                    "status": IndexStatus.BUILDING,
                    "phase": IndexPhase.DISCOVERING,
                    "files_processed": 0,
                    "files_total": 0,
                    "error": None,
                }
            )
            try:
                result = None
                chunks = None
                facts = None
                graph = None
                base_generation_id = None
                changed_paths: frozenset[str] | None = None
                for attempt in range(_BUILD_ATTEMPTS):
                    result = await asyncio.to_thread(
                        self._discovery.discover,
                        self._cwd,
                        cancel_event=self._cancel_event,
                    )
                    self._set_progress(
                        IndexPhase.CHUNKING, processed=0, total=len(result.files)
                    )
                    store = self._store_for(result.root)
                    current = await asyncio.to_thread(
                        store.current_generation, result.root
                    )
                    indexed_files = ()
                    chunk_count = 0
                    if current is not None:
                        indexed_files = await asyncio.to_thread(store.files, current.id)
                        chunk_count = await asyncio.to_thread(
                            store.chunk_count, current.id
                        )
                        if (
                            indexed_files == result.files
                            and (not result.files or chunk_count > 0)
                            and current.topology_fingerprint is not None
                        ):
                            self._state = RepositoryIndexState(
                                root=result.root,
                                generation=current,
                                status=IndexStatus.COMPLETE,
                                phase=IndexPhase.IDLE,
                                dirty=False,
                                files_processed=len(result.files),
                                files_total=len(result.files),
                            )
                            return current
                    analysis_reusable = (
                        current is not None
                        and (not result.files or chunk_count > 0)
                        and current.topology_fingerprint is not None
                    )
                    old_files = {file.path: file for file in indexed_files}
                    changed_files = (
                        tuple(
                            file
                            for file in result.files
                            if old_files.get(file.path) is None
                            or old_files[file.path].content_hash != file.content_hash
                            or old_files[file.path].language != file.language
                        )
                        if analysis_reusable
                        else result.files
                    )
                    base_generation_id = (
                        current.id
                        if analysis_reusable and current is not None
                        else None
                    )
                    changed_paths = frozenset(file.path for file in changed_files)
                    try:
                        chunks = await asyncio.to_thread(
                            build_text_chunks,
                            result.root,
                            changed_files,
                            cancel_event=self._cancel_event,
                        )
                        self._set_progress(
                            IndexPhase.PARSING,
                            processed=len(changed_files),
                            total=len(result.files),
                        )
                        cached = await asyncio.to_thread(
                            store.cached_facts, result.files
                        )
                        facts = await asyncio.to_thread(
                            build_file_facts,
                            result.root,
                            result.files,
                            cached,
                            cancel_event=self._cancel_event,
                        )
                        changed_paths |= frozenset(
                            fact.path
                            for fact in facts
                            if (
                                cached_fact := cached.get((
                                    fact.content_hash,
                                    fact.language,
                                ))
                            )
                            is None
                            or cached_fact.parser_version != fact.parser_version
                        )
                        previous_scores = (
                            await asyncio.to_thread(store.file_scores, current.id)
                            if current is not None
                            else ()
                        )
                        self._set_progress(
                            IndexPhase.GRAPH,
                            processed=len(result.files),
                            total=len(result.files),
                        )
                        graph = await asyncio.to_thread(
                            hydrate_dependency_graph,
                            result.files,
                            facts,
                            previous_fingerprint=(
                                current.topology_fingerprint
                                if current is not None
                                else None
                            ),
                            previous_scores=previous_scores,
                            cancel_event=self._cancel_event,
                        )
                        validated = await asyncio.to_thread(
                            self._discovery.discover,
                            self._cwd,
                            cancel_event=self._cancel_event,
                        )
                        if (
                            validated.root != result.root
                            or validated.files != result.files
                        ):
                            if attempt == _BUILD_ATTEMPTS - 1:
                                raise RepositoryChangedDuringBuildError(
                                    "Repository kept changing while the index was built."
                                )
                            self._set_progress(
                                IndexPhase.DISCOVERING, processed=0, total=0
                            )
                            continue
                        break
                    except RepositoryChangedDuringBuildError:
                        if attempt == _BUILD_ATTEMPTS - 1:
                            raise
                if (
                    result is None or chunks is None or facts is None or graph is None
                ):  # pragma: no cover
                    raise RepositoryIndexUnavailableError(
                        "Repository index build produced no snapshot."
                    )
                store = self._store_for(result.root)
                self._set_progress(
                    IndexPhase.PUBLISHING,
                    processed=len(result.files),
                    total=len(result.files),
                )
                generation = await asyncio.to_thread(
                    store.publish,
                    result.root,
                    result.files,
                    chunks,
                    facts,
                    graph,
                    base_generation_id,
                    changed_paths,
                )
                await self._prune(store, result.root)
                self._state = RepositoryIndexState(
                    root=result.root,
                    generation=generation,
                    status=IndexStatus.COMPLETE,
                    phase=IndexPhase.IDLE,
                    dirty=False,
                    files_processed=len(result.files),
                    files_total=len(result.files),
                )
                return generation
            except asyncio.CancelledError:
                self._cancel_event.set()
                self._state = self._state.model_copy(
                    update={
                        "status": IndexStatus.CANCELLED,
                        "phase": IndexPhase.IDLE,
                        "dirty": True,
                        "error": "Repository indexing was cancelled.",
                    }
                )
                raise
            except DiscoveryCancelledError as e:
                self._state = self._state.model_copy(
                    update={
                        "status": IndexStatus.CANCELLED,
                        "phase": IndexPhase.IDLE,
                        "dirty": True,
                        "error": str(e),
                    }
                )
                raise RepositoryIndexUnavailableError(str(e)) from e
            except Exception as e:
                self._state = self._state.model_copy(
                    update={
                        "status": IndexStatus.FAILED,
                        "phase": IndexPhase.IDLE,
                        "error": str(e),
                    }
                )
                raise RepositoryIndexUnavailableError(
                    f"Repository index is unavailable: {e}"
                ) from e

    async def watch(self, stop_event: asyncio.Event) -> None:
        generation = await self.ensure_ready()
        root = generation.root
        from watchfiles import awatch

        async for changes in awatch(root, stop_event=stop_event, debounce=300):
            if not changes or self._closed:
                continue
            self._state = self._state.model_copy(update={"dirty": True})
            try:
                await self.ensure_ready()
            except RepositoryIndexUnavailableError:
                if self._state.status is not IndexStatus.CANCELLED:
                    raise

    @asynccontextmanager
    async def pin_generation(self) -> AsyncIterator[IndexGeneration]:
        generation = await self.ensure_ready()
        async with self._pin_lock:
            self._generation_pins[generation.id] = (
                self._generation_pins.get(generation.id, 0) + 1
            )
        try:
            yield generation
        finally:
            async with self._pin_lock:
                remaining = self._generation_pins[generation.id] - 1
                if remaining:
                    self._generation_pins[generation.id] = remaining
                else:
                    self._generation_pins.pop(generation.id)
            if self._store is not None and self._root is not None:
                await self._prune(self._store, self._root)

    @asynccontextmanager
    async def inference_scope(self) -> AsyncIterator[IndexGeneration]:
        active = self._inference_generation.get()
        if active is not None:
            yield active
            return

        async with self.pin_generation() as generation:
            token = self._inference_generation.set(generation)
            try:
                yield generation
            finally:
                self._inference_generation.reset(token)

    def pinned_generation(self) -> IndexGeneration | None:
        return self._inference_generation.get()

    async def search(
        self,
        query: str,
        *,
        mode: RepositorySearchMode = RepositorySearchMode.AUTO,
        max_results: int = 20,
    ) -> RepositorySearchResult:
        generation = self.pinned_generation()
        if generation is None:
            async with self.inference_scope() as pinned:
                return await self._search_generation(
                    pinned, query, mode=mode, max_results=max_results
                )
        return await self._search_generation(
            generation, query, mode=mode, max_results=max_results
        )

    def cancel(self) -> None:
        self._cancel_event.set()

    async def rebuild(self) -> IndexGeneration:
        await self.clear()
        return await self.ensure_ready()

    async def clear(self) -> None:
        self.cancel()
        async with self._build_lock:
            self._state = self._state.model_copy(
                update={"status": IndexStatus.BUILDING, "phase": IndexPhase.CLEARING}
            )
            if self._store is not None:
                await asyncio.to_thread(self._store.clear)
            self._state = RepositoryIndexState(root=self._root)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.cancel()

    def _store_for(self, root: Path) -> RepositoryIndexStore:
        if self._store is not None and self._root == root:
            return self._store
        self._root = root
        path = self._storage_root / repository_identity(root) / "index.sqlite3"
        self._store = RepositoryIndexStore(path)
        return self._store

    def _set_progress(self, phase: IndexPhase, *, processed: int, total: int) -> None:
        self._state = self._state.model_copy(
            update={"phase": phase, "files_processed": processed, "files_total": total}
        )

    async def _prune(self, store: RepositoryIndexStore, root: Path) -> None:
        async with self._pin_lock:
            protected = frozenset(self._generation_pins)
        await asyncio.to_thread(
            store.prune_generations,
            root,
            keep=_RETAINED_GENERATIONS,
            protected=protected,
        )

    async def _search_generation(
        self,
        generation: IndexGeneration,
        query: str,
        *,
        mode: RepositorySearchMode,
        max_results: int,
    ) -> RepositorySearchResult:
        if self._store is None:
            raise RepositoryIndexUnavailableError(
                "Repository index storage is unavailable."
            )
        try:
            return await asyncio.to_thread(
                self._store.search,
                generation,
                query,
                mode=mode,
                max_results=max_results,
            )
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise RepositoryIndexUnavailableError(
                f"Repository index search is unavailable: {exc}"
            ) from exc
