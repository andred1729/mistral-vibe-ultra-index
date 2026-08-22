from __future__ import annotations

import asyncio
from pathlib import Path

from git import Repo
import pytest

from vibe.app_server._repository_index import RepositoryIndexController
import vibe.app_server._runtime as runtime
from vibe.app_server.protocol import ClientInfo, SessionOptions
from vibe.core.config.harness_files import HarnessFilesManager
from vibe.core.repository_index import IndexStatus, RepositoryIndexService


@pytest.mark.asyncio
async def test_root_blueprint_injects_process_owned_repository_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    Repo.init(root, initial_branch="main")
    (root / "module.py").write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setenv("VIBE_HOME", str(tmp_path / "vibe-home"))
    monkeypatch.setattr(runtime, "setup_tracing", lambda _: None)
    process = runtime.HarnessProcess(HarnessFilesManager(sources=()))

    blueprint = await process.build_root_blueprint(
        SessionOptions(cwd=str(root)), ClientInfo(name="test", version="1")
    )
    loop = blueprint.build()

    assert isinstance(blueprint.repository_index, RepositoryIndexService)
    assert loop.repository_index is blueprint.repository_index
    assert (
        loop.tool_manager.get("repo_search").repository_index is loop.repository_index
    )

    await runtime.close_agent_loop(loop)
    await process.close()


@pytest.mark.asyncio
async def test_harness_process_starts_and_stops_repository_watcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    Repo.init(root, initial_branch="main")
    (root / "module.py").write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setenv("VIBE_HOME", str(tmp_path / "vibe-home"))
    process = runtime.HarnessProcess(HarnessFilesManager(sources=()))
    service = process._repository_index_for(root)

    process._start_repository_index_watcher(root)
    for _ in range(100):
        if service.state.status is IndexStatus.COMPLETE:
            break
        await asyncio.sleep(0.01)

    assert service.state.status is IndexStatus.COMPLETE
    _, watcher = process._repository_index_watchers[root]
    assert not watcher.done()

    await process.close()

    assert watcher.done()
    assert service._closed


@pytest.mark.asyncio
async def test_controller_starts_background_refresh_and_projects_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    Repo.init(root, initial_branch="main")
    (root / "module.py").write_text("value = 1\n", encoding="utf-8")
    service = RepositoryIndexService(root, tmp_path / "index")
    tasks: list[asyncio.Task[None]] = []
    notifications = []

    async def notify(method, payload):
        notifications.append((method, payload))

    controller = RepositoryIndexController(service, "session", notify, tasks.append)

    response = await controller.refresh()
    assert response.started
    assert response.index.status == "building"

    await tasks[0]

    status = controller.status().index
    assert status.status == "complete"
    assert status.generation == 1
    assert status.file_count == 1
    assert notifications[0][0] == "repositoryIndex/updated"
