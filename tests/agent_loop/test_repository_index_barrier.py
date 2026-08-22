from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import Mock

from git import Repo
import pytest

from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.repository_index import RepositoryIndexService
from vibe.core.repository_index.models import (
    IndexGeneration,
    IndexStatus,
    RepositoryDependencyDirection,
    RepositoryIndexState,
    RepositorySearchMode,
    RepositorySearchResult,
)
from vibe.core.types import AssistantEvent, BaseEvent


class _RecordingIndexReader:
    def __init__(self, root: Path) -> None:
        self.generation = IndexGeneration(
            id=42,
            root=root,
            status=IndexStatus.COMPLETE,
            file_count=17,
            created_at="now",
            completed_at="now",
        )
        self.active = False
        self.entries = 0

    @property
    def state(self) -> RepositoryIndexState:
        return RepositoryIndexState(
            root=self.generation.root,
            generation=self.generation,
            status=IndexStatus.COMPLETE,
            dirty=False,
        )

    @asynccontextmanager
    async def inference_scope(self) -> AsyncIterator[IndexGeneration]:
        self.entries += 1
        self.active = True
        try:
            yield self.generation
        finally:
            self.active = False

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
    ) -> RepositorySearchResult:
        raise NotImplementedError


@pytest.mark.asyncio
async def test_model_turn_runs_inside_pinned_index_scope_and_refreshes_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = _RecordingIndexReader(tmp_path)
    loop = build_test_agent_loop(cwd=tmp_path, repository_index=reader)
    telemetry = Mock()
    monkeypatch.setattr(loop.telemetry_client, "send_telemetry_event", telemetry)

    async def perform_turn() -> AsyncGenerator[BaseEvent, None]:
        assert reader.active
        system_prompt = loop.messages[0].content or ""
        assert "## Repository index" in system_prompt
        assert "Index generation: 42" in system_prompt
        assert f"Indexed roots: {tmp_path}" in system_prompt
        assert "recommend `/index status`" in system_prompt
        assert "Continue using `repo_search`" in system_prompt
        assert "An `auto`\nquery is a compact locator" in system_prompt
        assert "not an exhaustive map" in system_prompt
        assert (
            "After locating a likely function or symbol, use `impact`" in system_prompt
        )
        assert "related tests, and what an edit could break" in system_prompt
        assert "`direction=dependencies`" in system_prompt
        assert "`direction=dependents`" in system_prompt
        assert (
            "Do not expand broad\ndependency trees before identifying" in system_prompt
        )
        assert "call\n`repo_search` before broad `grep`" not in system_prompt
        yield AssistantEvent(content="indexed")

    monkeypatch.setattr(loop, "_perform_llm_turn", perform_turn)

    events = [event async for event in loop._perform_indexed_llm_turn()]

    assert [event.content for event in events if isinstance(event, AssistantEvent)] == [
        "indexed"
    ]
    assert reader.entries == 1
    assert not reader.active
    telemetry.assert_called_once()
    assert telemetry.call_args.args[0] == "vibe.repository_index_barrier"
    assert telemetry.call_args.args[1]["status"] == "success"


def test_repo_search_is_only_exposed_when_index_reader_is_injected(
    tmp_path: Path,
) -> None:
    without_index = build_test_agent_loop(cwd=tmp_path)
    with_index = build_test_agent_loop(
        cwd=tmp_path, repository_index=_RecordingIndexReader(tmp_path)
    )

    assert "repo_search" not in without_index.tool_manager.available_tools
    assert "repo_search" in with_index.tool_manager.available_tools


@pytest.mark.asyncio
async def test_backend_is_not_called_until_real_index_is_complete_and_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    Repo.init(root, initial_branch="main")
    (root / "module.py").write_text("class IndexedBeforeInference:\n    pass\n")
    service = RepositoryIndexService(root, tmp_path / "indexes")
    backend = FakeBackend(mock_llm_chunk(content="ready"))
    original_complete = backend.complete

    async def assert_index_then_complete(**kwargs: Any):
        generation = service.pinned_generation()
        assert generation is not None
        assert service.state.status is IndexStatus.COMPLETE
        assert service.state.generation == generation
        system_prompt = kwargs["messages"][0].content or ""
        assert f"Index generation: {generation.id}" in system_prompt
        return await original_complete(**kwargs)

    monkeypatch.setattr(backend, "complete", assert_index_then_complete)
    loop = build_test_agent_loop(cwd=root, backend=backend, repository_index=service)

    _ = [event async for event in loop.act("Inspect the repository")]

    assert service.pinned_generation() is None
    assert service.state.status is IndexStatus.COMPLETE
