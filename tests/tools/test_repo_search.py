from __future__ import annotations

from pathlib import Path

from git import Repo
import pytest

from tests.conftest import build_test_agent_loop
from vibe.core.repository_index import (
    RepositoryIndexService,
    RepositoryModuleRole,
    RepositorySearchMode,
    RepositorySearchResult,
)
from vibe.core.tools.builtins.repo_search import RepoSearch, RepoSearchArgs


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
    assert result.matches[0].component == "(root)"
    assert result.matches[0].module_role is RepositoryModuleRole.ROOT
    assert result.groups[0].paths == ("service.py",)
    assert {
        (suggestion.query, suggestion.mode) for suggestion in result.suggestions
    } >= {
        ("HydratedDependencyGraph", RepositorySearchMode.IMPACT),
        ("HydratedDependencyGraph", RepositorySearchMode.DEPENDENCY),
    }

    call_display = RepoSearch.format_call_display(
        RepoSearchArgs(
            query="HydratedDependencyGraph", mode=RepositorySearchMode.SYMBOL, limit=5
        )
    )
    assert call_display.message == "Repo_Search"
    assert "(" not in call_display.summary

    result_display = RepoSearch.format_result_display(result)
    assert result_display.text == "Repo_Search"
    assert result_display.suffix == "1 of 1 matches"


@pytest.mark.asyncio
async def test_impact_mode_returns_direct_dependents_and_related_tests(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    (root / "pkg").mkdir(parents=True)
    (root / "apps").mkdir()
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
    (root / "apps" / "entry.py").write_text(
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
    assert result.dependency_trees == ()
    definition = next(
        match for match in result.matches if match.path == "pkg/service.py"
    )
    assert definition.component == "pkg"
    assert definition.module_role is RepositoryModuleRole.SHARED
    test_match = next(
        match for match in result.matches if match.path == "tests/test_service.py"
    )
    assert test_match.component == "tests"
    assert test_match.module_role is RepositoryModuleRole.TEST
    assert {group.component for group in result.groups} >= {"pkg", "apps", "tests"}


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
    assert result.dependency_trees[0].root == "apps/entry.py"
    rendered_tree = "\n".join(result.dependency_trees[0].lines)
    assert "[imports] pkg/middle.py" in rendered_tree
    assert "[imports] pkg/base.py" in rendered_tree
    assert result.dependency_trees[0].node_count == 3
    assert not result.dependency_trees[0].truncated
