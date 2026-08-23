from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.repository_index.models import (
    DiscoveredFile,
    FileGraphScore,
    GraphEdge,
    GraphEdgeKind,
    HydratedGraph,
    IndexChunk,
    IndexStatus,
    RepositoryAnchorStatus,
    RepositoryDependencyDirection,
    RepositorySearchMode,
)
from vibe.core.repository_index.parsers.python import parse_python_facts
from vibe.core.repository_index.store import (
    RepositoryIndexCorruptError,
    RepositoryIndexStore,
)


def _file(path: str, content_hash: str = "hash") -> DiscoveredFile:
    return DiscoveredFile(
        path=path, content_hash=content_hash, language="python", size=10, modified_ns=1
    )


def test_publishes_immutable_generations_and_selects_latest(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    store = RepositoryIndexStore(tmp_path / "index" / "repository.sqlite3")

    first = store.publish(root, [_file("first.py")])
    second = store.publish(root, [_file("second.py")])

    assert first.status is IndexStatus.COMPLETE
    assert first.id < second.id
    assert store.current_generation(root) == second
    assert [file.path for file in store.files(first.id)] == ["first.py"]
    assert [file.path for file in store.files(second.id)] == ["second.py"]


def test_failed_publish_does_not_replace_current_generation(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    store = RepositoryIndexStore(tmp_path / "repository.sqlite3")
    current = store.publish(root, [_file("first.py")])

    with pytest.raises(RepositoryIndexCorruptError, match="publish"):
        store.publish(root, [_file("duplicate.py"), _file("duplicate.py")])

    published = store.current_generation(root)
    assert published is not None
    assert published.id == current.id


def test_rejects_corrupt_storage_and_can_clear_it(tmp_path: Path) -> None:
    path = tmp_path / "repository.sqlite3"
    path.write_bytes(b"not a sqlite database")
    store = RepositoryIndexStore(path)

    with pytest.raises(RepositoryIndexCorruptError, match="corrupt"):
        store.initialize()

    store.clear()
    store.initialize()
    assert store.current_generation(tmp_path) is None


def test_prunes_to_newest_two_generations_while_preserving_pins(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    store = RepositoryIndexStore(tmp_path / "repository.sqlite3")
    first = store.publish(root, [_file("first.py")])
    second = store.publish(root, [_file("second.py")])
    third = store.publish(root, [_file("third.py")])

    assert store.prune_generations(root, keep=2, protected=frozenset({first.id})) == ()
    assert store.files(first.id)

    assert store.prune_generations(root, keep=2) == (first.id,)
    assert store.files(first.id) == ()
    assert store.files(second.id)
    assert store.files(third.id)


def test_search_returns_source_cited_chunks_from_selected_generation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    store = RepositoryIndexStore(tmp_path / "repository.sqlite3")
    first = store.publish(
        root,
        [_file("old.py")],
        [
            IndexChunk(
                path="old.py",
                ordinal=0,
                line_start=10,
                line_end=11,
                content="class PassiveIndex:\n    pass",
            )
        ],
    )
    store.publish(
        root,
        [_file("new.py")],
        [
            IndexChunk(
                path="new.py",
                ordinal=0,
                line_start=1,
                line_end=1,
                content="unrelated = True",
            )
        ],
    )

    result = store.search(first, "PassiveIndex", max_results=10)

    assert result.generation == first
    assert result.total_matches == 1
    assert result.matches[0].path == "old.py"
    assert result.matches[0].line_start == 10
    assert result.matches[0].generation == first.id
    assert "PassiveIndex" in result.matches[0].snippet


def test_text_search_requires_all_terms_while_auto_search_retains_recall(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    store = RepositoryIndexStore(tmp_path / "repository.sqlite3")
    files = [_file("both.py"), _file("alpha.py"), _file("beta.py")]
    generation = store.publish(
        root,
        files,
        [
            IndexChunk(
                path="both.py",
                ordinal=0,
                line_start=1,
                line_end=1,
                content="alpha beta",
            ),
            IndexChunk(
                path="alpha.py", ordinal=0, line_start=1, line_end=1, content="alpha"
            ),
            IndexChunk(
                path="beta.py", ordinal=0, line_start=1, line_end=1, content="beta"
            ),
        ],
    )

    precise = store.search(
        generation, "alpha beta", mode=RepositorySearchMode.TEXT, max_results=10
    )
    broad = store.search(
        generation, "alpha beta", mode=RepositorySearchMode.AUTO, max_results=10
    )

    assert [match.path for match in precise.matches] == ["both.py"]
    assert {match.path for match in broad.matches} == {"alpha.py", "beta.py", "both.py"}


def test_text_search_treats_quoted_terms_as_an_exact_phrase(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    store = RepositoryIndexStore(tmp_path / "repository.sqlite3")
    generation = store.publish(
        root,
        [_file("adjacent.py"), _file("separated.py")],
        [
            IndexChunk(
                path="adjacent.py",
                ordinal=0,
                line_start=1,
                line_end=1,
                content="dependency graph",
            ),
            IndexChunk(
                path="separated.py",
                ordinal=0,
                line_start=1,
                line_end=1,
                content="dependency directed graph",
            ),
        ],
    )

    result = store.search(
        generation, '"dependency graph"', mode=RepositorySearchMode.TEXT, max_results=10
    )

    assert [match.path for match in result.matches] == ["adjacent.py"]


def test_auto_search_expands_identifier_concepts_and_prioritizes_coverage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    store = RepositoryIndexStore(tmp_path / "repository.sqlite3")
    generation = store.publish(
        root,
        [
            _file("api.py"),
            _file("compiler.py"),
            _file("docs/feature.md"),
            _file("tests/test_feature.py"),
        ],
        [
            IndexChunk(
                path="api.py",
                ordinal=0,
                line_start=1,
                line_end=1,
                content="class FilteredRelation: pass",
            ),
            IndexChunk(
                path="compiler.py",
                ordinal=0,
                line_start=1,
                line_end=1,
                content="filtered relation select related compiler hydration",
            ),
            IndexChunk(
                path="docs/feature.md",
                ordinal=0,
                line_start=1,
                line_end=1,
                content="filtered relation select related",
            ),
            IndexChunk(
                path="tests/test_feature.py",
                ordinal=0,
                line_start=1,
                line_end=3,
                content="select related\nselect related\nselect related",
            ),
        ],
    )

    result = store.search(
        generation,
        "FilteredRelation select_related",
        mode=RepositorySearchMode.AUTO,
        max_results=10,
    )

    assert result.matches[0].path == "compiler.py"
    assert len(result.matches) <= 5
    assert len({match.path for match in result.matches}) == len(result.matches)
    assert all(len(match.snippet.encode("utf-8")) <= 400 for match in result.matches)
    assert result.groups == ()
    assert result.dependency_trees == ()


def test_auto_search_suggests_anchored_impact_for_definition(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    store = RepositoryIndexStore(tmp_path / "repository.sqlite3")
    path = "pkg/optimizer.py"
    source = "\n" * 39 + "def optimize_migrations():\n    return 'squash migrations'\n"
    generation = store.publish(
        root,
        [_file(path, "optimizer")],
        [
            IndexChunk(
                path=path,
                ordinal=0,
                line_start=40,
                line_end=41,
                content="def optimize_migrations():\n    return 'squash migrations'",
            )
        ],
        facts=[parse_python_facts(path, "optimizer", source)],
    )

    result = store.search(generation, "squash migrations", max_results=20)

    impact = next(
        suggestion
        for suggestion in result.suggestions
        if suggestion.mode is RepositorySearchMode.IMPACT
    )
    assert impact.query == "optimize_migrations"
    assert "Open pkg/optimizer.py:40 first" in impact.reason
    assert "what could break" in impact.reason
    assert all(
        suggestion.mode is not RepositorySearchMode.DEPENDENCY
        or suggestion.query != "squash migrations"
        for suggestion in result.suggestions
    )


def test_dependency_search_resolves_exact_path_without_lexical_seeds(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    store = RepositoryIndexStore(tmp_path / "repository.sqlite3")
    generation = store.publish(
        root,
        [_file("pkg/service.py", "service"), _file("apps/consumer.py", "consumer")],
        [
            IndexChunk(
                path="pkg/service.py",
                ordinal=0,
                line_start=1,
                line_end=1,
                content="SERVICE = True",
            ),
            IndexChunk(
                path="apps/consumer.py",
                ordinal=0,
                line_start=1,
                line_end=1,
                content="service consumer",
            ),
        ],
        graph=HydratedGraph(
            edges=(
                GraphEdge(
                    source="apps/consumer.py",
                    target="pkg/service.py",
                    kind=GraphEdgeKind.IMPORTS,
                ),
            ),
            scores=(
                FileGraphScore(path="pkg/service.py", importance=0.1, component=0),
                FileGraphScore(path="apps/consumer.py", importance=0.1, component=1),
            ),
            topology_fingerprint="graph",
        ),
    )

    result = store.search(
        generation,
        "pkg/service.py",
        mode=RepositorySearchMode.DEPENDENCY,
        direction=RepositoryDependencyDirection.DEPENDENTS,
        max_results=20,
    )

    assert result.anchor_resolution is not None
    assert result.anchor_resolution.status is RepositoryAnchorStatus.RESOLVED
    assert result.anchor_resolution.resolved_anchor is not None
    assert result.anchor_resolution.resolved_anchor.path == "pkg/service.py"
    assert result.anchor_resolution.candidate_count == 1
    assert result.dependency_trees[0].root == "pkg/service.py"
    assert {match.path for match in result.matches} == {
        "pkg/service.py",
        "apps/consumer.py",
    }


@pytest.mark.parametrize(
    "mode", [RepositorySearchMode.DEPENDENCY, RepositorySearchMode.IMPACT]
)
def test_graph_search_returns_ambiguous_symbol_anchors_without_traversal(
    tmp_path: Path, mode: RepositorySearchMode
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    store = RepositoryIndexStore(tmp_path / "repository.sqlite3")
    sources = {
        "pkg/first.py": "class First:\n    def run(self):\n        return 1\n",
        "pkg/second.py": "class Second:\n    def run(self):\n        return 2\n",
    }
    generation = store.publish(
        root,
        [_file(path, path) for path in sources],
        facts=[
            parse_python_facts(path, path, source) for path, source in sources.items()
        ],
    )

    result = store.search(generation, "run", mode=mode, max_results=20)

    assert result.anchor_resolution is not None
    assert result.anchor_resolution.status is RepositoryAnchorStatus.AMBIGUOUS
    assert result.anchor_resolution.candidate_count == 2
    assert {
        (anchor.path, anchor.symbol)
        for anchor in result.anchor_resolution.candidate_anchors
    } == {("pkg/first.py", "First.run"), ("pkg/second.py", "Second.run")}
    assert result.matches == ()
    assert result.dependency_trees == ()


def test_graph_search_requires_an_exact_anchor(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    store = RepositoryIndexStore(tmp_path / "repository.sqlite3")
    generation = store.publish(
        root,
        [_file("pkg/service.py")],
        [
            IndexChunk(
                path="pkg/service.py",
                ordinal=0,
                line_start=1,
                line_end=1,
                content="migration optimizer service",
            )
        ],
    )

    result = store.search(
        generation,
        "migration optimizer service",
        mode=RepositorySearchMode.DEPENDENCY,
        max_results=20,
    )

    assert result.anchor_resolution is not None
    assert result.anchor_resolution.status is RepositoryAnchorStatus.NEEDS_ANCHOR
    assert result.anchor_resolution.candidate_anchors == ()
    assert result.matches == ()
    assert result.dependency_trees == ()


def test_ambiguous_anchor_candidates_are_bounded_and_report_total(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    store = RepositoryIndexStore(tmp_path / "repository.sqlite3")
    sources = {
        f"pkg/module_{index}.py": f"class Service{index}:\n    def run(self):\n        return {index}\n"
        for index in range(25)
    }
    generation = store.publish(
        root,
        [_file(path, path) for path in sources],
        facts=[
            parse_python_facts(path, path, source) for path, source in sources.items()
        ],
    )

    result = store.search(
        generation, "run", mode=RepositorySearchMode.DEPENDENCY, max_results=20
    )

    assert result.anchor_resolution is not None
    assert result.anchor_resolution.status is RepositoryAnchorStatus.AMBIGUOUS
    assert result.anchor_resolution.candidate_count == 25
    assert len(result.anchor_resolution.candidate_anchors) == 20
    assert result.matches == ()


def test_auto_search_ranks_source_from_beyond_initial_fts_window(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    store = RepositoryIndexStore(tmp_path / "repository.sqlite3")
    documentation = [f"docs/noise_{index:03}.md" for index in range(120)]
    source = "z_implementation.py"
    generation = store.publish(
        root,
        [_file(path, path) for path in (*documentation, source)],
        [
            *(
                IndexChunk(
                    path=path,
                    ordinal=0,
                    line_start=1,
                    line_end=1,
                    content="migration index",
                )
                for path in documentation
            ),
            IndexChunk(
                path=source,
                ordinal=0,
                line_start=1,
                line_end=1,
                content="migration implementation",
            ),
        ],
    )

    result = store.search(generation, "migration index", max_results=10)

    assert result.matches[0].path == source


def test_auto_search_expands_behavior_into_implementation_intent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    store = RepositoryIndexStore(tmp_path / "repository.sqlite3")
    generation = store.publish(
        root,
        [_file("state.py"), _file("optimizer.py")],
        [
            IndexChunk(
                path="state.py",
                ordinal=0,
                line_start=1,
                line_end=1,
                content="migration squashing deprecation warning",
            ),
            IndexChunk(
                path="optimizer.py",
                ordinal=0,
                line_start=1,
                line_end=1,
                content="migration optimizer reduce operations",
            ),
        ],
    )

    result = store.search(generation, "migration squashing", max_results=10)

    assert result.matches[0].path == "optimizer.py"


def test_auto_search_surfaces_tests_related_to_matching_source(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    store = RepositoryIndexStore(tmp_path / "repository.sqlite3")
    source = "pkg/optimizer.py"
    test = "tests/test_related.py"
    generation = store.publish(
        root,
        [_file(source), _file(test)],
        [
            IndexChunk(
                path=source,
                ordinal=0,
                line_start=1,
                line_end=1,
                content="migration optimizer reduce",
            ),
            IndexChunk(
                path=test,
                ordinal=0,
                line_start=1,
                line_end=1,
                content="unrelated test fixture",
            ),
        ],
        graph=HydratedGraph(
            edges=(
                GraphEdge(source=source, target=test, kind=GraphEdgeKind.TESTED_BY),
            ),
            scores=(
                FileGraphScore(path=source, importance=0.5, component=0),
                FileGraphScore(path=test, importance=0.5, component=0),
            ),
            topology_fingerprint="topology",
        ),
    )

    result = store.search(
        generation,
        "migration optimizer",
        mode=RepositorySearchMode.AUTO,
        direction=RepositoryDependencyDirection.DEPENDENCIES,
        max_results=10,
    )

    related_test = next(
        match
        for match in result.matches
        if match.path == test and match.relationship == "tested_by"
    )
    assert related_test.relationship == "tested_by"
    assert related_test.dependency_direction is RepositoryDependencyDirection.DEPENDENTS


def test_search_rejects_absolute_or_parent_path_filters(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    store = RepositoryIndexStore(tmp_path / "repository.sqlite3")
    generation = store.publish(
        root,
        [_file("module.py")],
        [
            IndexChunk(
                path="module.py", ordinal=0, line_start=1, line_end=1, content="needle"
            )
        ],
    )

    for path in ("/tmp", "../secret", r"C:\secret"):
        with pytest.raises(ValueError, match="repository-relative"):
            store.search(generation, "needle", path=path, max_results=10)


def test_auto_search_uses_compact_bounds_without_changing_text_mode(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    store = RepositoryIndexStore(tmp_path / "repository.sqlite3")
    files = [_file(f"file_{index}.py") for index in range(100)]
    chunks = [
        IndexChunk(
            path=file.path,
            ordinal=0,
            line_start=1,
            line_end=1,
            content=f"needle {'x' * 7_900}",
        )
        for file in files
    ]
    generation = store.publish(root, files, chunks)

    automatic = store.search(generation, "needle", max_results=100)
    text = store.search(
        generation, "needle", mode=RepositorySearchMode.TEXT, max_results=10
    )

    encoded_matches = sum(
        len(match.model_dump_json().encode("utf-8")) for match in automatic.matches
    )
    assert encoded_matches <= 48_000
    assert len(automatic.matches) == 5
    assert all(len(match.snippet.encode("utf-8")) <= 400 for match in automatic.matches)
    assert automatic.groups == ()
    assert len(text.matches) == 10
    assert all(len(match.snippet.encode("utf-8")) > 400 for match in text.matches)
    assert text.groups
