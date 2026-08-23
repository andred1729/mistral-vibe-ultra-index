from __future__ import annotations

from pathlib import Path

from git import Repo
import pytest

from tests.conftest import build_test_agent_loop
from vibe.core.repository_index import (
    RepositoryDependencyDirection,
    RepositoryIndexService,
    RepositoryModuleRole,
    RepositorySearchMode,
    RepositorySearchResult,
)
from vibe.core.tools.builtins.repo_search import RepoSearch, RepoSearchArgs


def test_repo_search_arguments_guide_staged_graph_queries() -> None:
    parameters = RepoSearch.get_parameters()["properties"]
    tool_guidance = RepoSearch.get_full_description()

    assert "compact task or symbol query" in parameters["query"]["description"]
    assert "auto is a compact locator" in parameters["mode"]["description"]
    assert "what could break" in parameters["mode"]["description"]
    assert "known likely locus" in parameters["mode"]["description"]
    assert "candidate anchors" in parameters["mode"]["description"]
    assert "what the locus calls or requires" in parameters["direction"]["description"]
    assert "incoming callers and consumers" in parameters["direction"]["description"]
    assert "bounded two-way neighborhood" in parameters["direction"]["description"]
    assert "not as an exhaustive repository map" in tool_guidance
    assert "Once you identify a likely named function" in tool_guidance
    assert "before returning to broad `grep`/`find`" in tool_guidance
    assert "Do not request broad dependency trees" in tool_guidance


def test_repo_search_formats_ambiguous_anchor_result(tmp_path: Path) -> None:
    result = RepositorySearchResult.model_validate({
        "generation": {
            "id": 1,
            "root": tmp_path,
            "status": "complete",
            "file_count": 2,
            "created_at": "now",
        },
        "query": "run",
        "mode": "dependency",
        "dependency_direction": "dependents",
        "total_matches": 0,
        "matches": [],
        "anchor_resolution": {
            "status": "ambiguous",
            "candidate_count": 2,
            "candidate_anchors": [
                {"path": "first.py", "symbol": "First.run"},
                {"path": "second.py", "symbol": "Second.run"},
            ],
        },
    })

    display = RepoSearch.format_result_display(result)

    assert display.text == "Repo_Search"
    assert display.suffix == "2 candidate anchors"


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
        "def shared_service():\n    return 1\n\n"
        "def unrelated_service():\n    return 2\n",
        encoding="utf-8",
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
    (root / "apps" / "unrelated.py").write_text(
        "from pkg.service import unrelated_service\nunrelated_service()\n",
        encoding="utf-8",
    )
    service = RepositoryIndexService(root, tmp_path / "indexes")

    result = await service.search(
        "shared_service", mode=RepositorySearchMode.IMPACT, max_results=20
    )

    relationships = {(match.path, match.relationship) for match in result.matches}
    assert ("pkg/consumer.py", "dependent_via_imports") in relationships
    assert any(path == "tests/test_service.py" for path, _ in relationships)
    assert all(path != "apps/unrelated.py" for path, _ in relationships)
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
        "from pkg.base import shared_api\nMIDDLE_LAYER_MARKER = True\n",
        encoding="utf-8",
    )
    (root / "apps" / "entry.py").write_text(
        "from pkg.middle import shared_api\n", encoding="utf-8"
    )
    service = RepositoryIndexService(root, tmp_path / "indexes")
    result = await service.search(
        "shared_api",
        mode=RepositorySearchMode.DEPENDENCY,
        direction=RepositoryDependencyDirection.DEPENDENTS,
        path="apps",
        max_results=20,
    )

    assert {match.path for match in result.matches} == {"apps/entry.py"}
    assert result.dependency_direction is RepositoryDependencyDirection.DEPENDENTS
    assert all(
        match.dependency_direction is RepositoryDependencyDirection.DEPENDENTS
        for match in result.matches
    )
    assert result.dependency_trees[0].root == "pkg/base.py"
    assert (
        result.dependency_trees[0].direction is RepositoryDependencyDirection.DEPENDENTS
    )
    rendered_tree = "\n".join(result.dependency_trees[0].lines)
    assert "[← imports] pkg/middle.py" in rendered_tree
    assert "[← imports] apps/entry.py" in rendered_tree
    assert result.dependency_trees[0].node_count == 3
    assert not result.dependency_trees[0].truncated

    dependencies = await service.search(
        "MIDDLE_LAYER_MARKER", mode=RepositorySearchMode.DEPENDENCY, max_results=20
    )
    outgoing = [
        match
        for match in dependencies.matches
        if match.dependency_direction is not None
    ]
    assert {match.path for match in outgoing} == {"pkg/base.py"}
    assert all(
        match.dependency_direction is RepositoryDependencyDirection.DEPENDENCIES
        for match in outgoing
    )
    assert (
        dependencies.dependency_direction is RepositoryDependencyDirection.DEPENDENCIES
    )
    assert "[imports →] pkg/base.py" in "\n".join(
        dependencies.dependency_trees[0].lines
    )

    automatic = await service.search("MIDDLE_LAYER_MARKER", max_results=20)
    assert automatic.dependency_trees == ()

    neighborhood = await service.search(
        "MIDDLE_LAYER_MARKER",
        mode=RepositorySearchMode.DEPENDENCY,
        direction=RepositoryDependencyDirection.BOTH,
        max_results=20,
    )
    graph_matches = [
        match
        for match in neighborhood.matches
        if match.dependency_direction is not None
    ]
    assert {match.path for match in graph_matches} == {"apps/entry.py", "pkg/base.py"}
    assert {match.dependency_direction for match in graph_matches} == {
        RepositoryDependencyDirection.DEPENDENCIES,
        RepositoryDependencyDirection.DEPENDENTS,
    }


@pytest.mark.asyncio
async def test_auto_search_does_not_suggest_impact_from_a_test_definition(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    (root / "tests").mkdir(parents=True)
    Repo.init(root, initial_branch="main")
    (root / "tests" / "test_feature.py").write_text(
        "def test_feature_behavior():\n    pass\n", encoding="utf-8"
    )
    service = RepositoryIndexService(root, tmp_path / "indexes")

    result = await service.search("test_feature_behavior", max_results=20)

    assert all(
        suggestion.mode is not RepositorySearchMode.IMPACT
        for suggestion in result.suggestions
    )


@pytest.mark.asyncio
async def test_symbol_mode_splits_natural_language_terms(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    Repo.init(root, initial_branch="main")
    (root / "query.py").write_text(
        "class FilteredRelation:\n    pass\n\ndef select_related():\n    pass\n",
        encoding="utf-8",
    )
    service = RepositoryIndexService(root, tmp_path / "indexes")

    result = await service.search(
        "FilteredRelation select_related",
        mode=RepositorySearchMode.SYMBOL,
        max_results=20,
    )

    assert {match.symbol for match in result.matches} >= {
        "FilteredRelation",
        "select_related",
    }


@pytest.mark.asyncio
async def test_symbol_mode_falls_back_to_text_and_suggests_split_queries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    Repo.init(root, initial_branch="main")
    (root / "notes.md").write_text(
        "Hydration behavior for filtered relations.\n", encoding="utf-8"
    )
    service = RepositoryIndexService(root, tmp_path / "indexes")

    result = await service.search(
        "hydration missing_term", mode=RepositorySearchMode.SYMBOL, max_results=20
    )

    assert result.matches[0].path == "notes.md"
    assert result.matches[0].relationship == "text_match"

    empty = await service.search(
        "unknown_concept another_unknown",
        mode=RepositorySearchMode.SYMBOL,
        max_results=20,
    )
    assert [(item.query, item.mode) for item in empty.suggestions[:2]] == [
        ("unknown_concept", RepositorySearchMode.AUTO),
        ("another_unknown", RepositorySearchMode.AUTO),
    ]
