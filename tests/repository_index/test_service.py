from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Event

from git import Repo
import pytest

from vibe.core.repository_index.discovery import (
    DiscoveryCancelledError,
    RepositoryDiscovery,
)
from vibe.core.repository_index.models import IndexStatus
from vibe.core.repository_index.service import (
    RepositoryIndexService,
    RepositoryIndexUnavailableError,
)


@pytest.mark.asyncio
async def test_builds_only_when_repository_snapshot_changes(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    Repo.init(root, initial_branch="main")
    source = root / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    service = RepositoryIndexService(root, tmp_path / "indexes")

    first = await service.ensure_ready()
    unchanged = await service.ensure_ready()
    source.write_text("value = 2\n", encoding="utf-8")
    changed = await service.ensure_ready()

    assert unchanged.id == first.id
    assert changed.id > first.id
    assert service.state.generation == changed
    assert not service.state.dirty


@pytest.mark.asyncio
async def test_pinned_generation_survives_until_inference_releases_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    Repo.init(root, initial_branch="main")
    source = root / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    service = RepositoryIndexService(root, tmp_path / "indexes")

    async with service.pin_generation() as pinned:
        source.write_text("value = 2\n", encoding="utf-8")
        second = await service.ensure_ready()
        source.write_text("value = 3\n", encoding="utf-8")
        third = await service.ensure_ready()
        assert service._store is not None
        assert service._store.files(pinned.id)

    assert service._store.files(pinned.id) == ()
    assert service._store.files(second.id)
    assert service._store.files(third.id)


@pytest.mark.asyncio
async def test_nested_pins_release_generation_only_after_outer_scope(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    Repo.init(root, initial_branch="main")
    (root / "module.py").write_text("value = 1\n", encoding="utf-8")
    service = RepositoryIndexService(root, tmp_path / "indexes")

    async with service.pin_generation() as outer:
        async with service.pin_generation() as inner:
            assert inner.id == outer.id
        assert service._generation_pins == {outer.id: 1}

    assert service._generation_pins == {}


@pytest.mark.asyncio
async def test_inference_scope_reuses_generation_and_searches_immutable_chunks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    Repo.init(root, initial_branch="main")
    source = root / "module.py"
    source.write_text("class OriginalSymbol:\n    pass\n", encoding="utf-8")
    service = RepositoryIndexService(root, tmp_path / "indexes")

    async with service.inference_scope() as outer:
        first_result = await service.search("OriginalSymbol")
        source.write_text("class ReplacementSymbol:\n    pass\n", encoding="utf-8")
        await service.ensure_ready()
        async with service.inference_scope() as inner:
            pinned_result = await service.search("OriginalSymbol")

        assert inner.id == outer.id
        assert first_result.generation.id == outer.id
        assert pinned_result.generation.id == outer.id
        assert pinned_result.total_matches >= 1

    fresh_result = await service.search("ReplacementSymbol")
    assert fresh_result.generation.id > outer.id
    assert fresh_result.total_matches >= 1


@pytest.mark.asyncio
async def test_non_git_session_fails_with_actionable_diagnostic(tmp_path: Path) -> None:
    service = RepositoryIndexService(tmp_path, tmp_path / "indexes")

    with pytest.raises(RepositoryIndexUnavailableError, match="git repository"):
        await service.ensure_ready()


@pytest.mark.asyncio
async def test_corrupt_published_store_blocks_barrier_with_diagnostic(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    Repo.init(root, initial_branch="main")
    (root / "module.py").write_text("value = 1\n", encoding="utf-8")
    storage = tmp_path / "indexes"
    service = RepositoryIndexService(root, storage)
    await service.ensure_ready()
    assert service._store is not None
    database = service._store.path
    await service.close()
    database.write_bytes(b"not a sqlite database")

    corrupt_service = RepositoryIndexService(root, storage)
    with pytest.raises(RepositoryIndexUnavailableError, match="corrupt"):
        await corrupt_service.ensure_ready()


@pytest.mark.asyncio
async def test_cancel_interrupts_background_discovery(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    Repo.init(root, initial_branch="main")
    started = Event()

    class BlockingDiscovery(RepositoryDiscovery):
        def discover(self, cwd, *, cancel_event=None, on_file=None):
            started.set()
            assert cancel_event is not None
            cancel_event.wait()
            raise DiscoveryCancelledError("cancelled by test")

    service = RepositoryIndexService(
        root, tmp_path / "indexes", discovery=BlockingDiscovery()
    )
    build = asyncio.create_task(service.ensure_ready())
    await asyncio.to_thread(started.wait)

    service.cancel()

    with pytest.raises(RepositoryIndexUnavailableError, match="cancelled by test"):
        await build
    assert service.state.status is IndexStatus.CANCELLED
    assert service.state.dirty


@pytest.mark.asyncio
async def test_revalidates_repository_after_graph_hydration(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    Repo.init(root, initial_branch="main")
    module = root / "module.py"
    module.write_text("value = 1\n", encoding="utf-8")
    discovery = RepositoryDiscovery()

    class ChangingDiscovery(RepositoryDiscovery):
        calls = 0

        def discover(self, cwd, *, cancel_event=None, on_file=None):
            self.calls += 1
            if self.calls == 2:
                module.write_text("value = 2\n", encoding="utf-8")
            return discovery.discover(cwd, cancel_event=cancel_event, on_file=on_file)

    service = RepositoryIndexService(
        root, tmp_path / "indexes", discovery=ChangingDiscovery()
    )

    generation = await service.ensure_ready()
    result = await service.search("value = 2")

    assert generation.id == 1
    assert result.total_matches == 1
