from __future__ import annotations

from pathlib import Path

from git import Repo
import pytest

from tests.conftest import build_test_agent_loop
from vibe.core.repository_index import (
    RepositoryIndexService,
    RepositorySearchMode,
    RepositorySearchResult,
)


@pytest.mark.asyncio
async def test_repo_search_reads_the_generation_pinned_by_the_agent_loop(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    Repo.init(root, initial_branch="main")
    (root / "service.py").write_text(
        "class HydratedDependencyGraph:\n    pass\n", encoding="utf-8"
    )
    service = RepositoryIndexService(root, tmp_path / "indexes")
    loop = build_test_agent_loop(cwd=root, repository_index=service)
    tool = loop.tool_manager.get("repo_search")

    async with service.inference_scope() as generation:
        results = [
            item
            async for item in tool.invoke(
                query="HydratedDependencyGraph", mode="symbol", limit=5
            )
        ]

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, RepositorySearchResult)
    assert result.generation.id == generation.id
    assert result.total_matches == 1
    assert result.matches[0].path == "service.py"
    assert result.matches[0].generation == generation.id


@pytest.mark.asyncio
async def test_impact_mode_returns_direct_dependents_and_related_tests(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    Repo.init(root, initial_branch="main")
    (root / "pkg" / "service.py").write_text(
        "def shared_service():\n    return 1\n", encoding="utf-8"
    )
    (root / "pkg" / "consumer.py").write_text(
        "from pkg.service import shared_service\nshared_service()\n", encoding="utf-8"
    )
    (root / "tests" / "test_service.py").write_text(
        "from pkg.service import shared_service\n", encoding="utf-8"
    )
    service = RepositoryIndexService(root, tmp_path / "indexes")

    result = await service.search(
        "shared_service", mode=RepositorySearchMode.IMPACT, max_results=20
    )

    relationships = {(match.path, match.relationship) for match in result.matches}
    assert ("pkg/consumer.py", "dependent_via_imports") in relationships
    assert any(path == "tests/test_service.py" for path, _ in relationships)
    assert result.structural_coverage


@pytest.mark.asyncio
async def test_dependency_mode_expands_two_hops_and_path_filters_results(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    (root / "pkg").mkdir(parents=True)
    (root / "apps").mkdir()
    Repo.init(root, initial_branch="main")
    (root / "pkg" / "base.py").write_text(
        "def shared_api():\n    return 1\n", encoding="utf-8"
    )
    (root / "pkg" / "middle.py").write_text(
        "from pkg.base import shared_api\n", encoding="utf-8"
    )
    (root / "apps" / "entry.py").write_text(
        "from pkg.middle import shared_api\n", encoding="utf-8"
    )
    service = RepositoryIndexService(root, tmp_path / "indexes")

    result = await service.search(
        "shared_api", mode=RepositorySearchMode.DEPENDENCY, path="apps", max_results=20
    )

    assert {match.path for match in result.matches} == {"apps/entry.py"}
    assert any(
        match.relationship.startswith("dependency_distance_")
        for match in result.matches
    )
