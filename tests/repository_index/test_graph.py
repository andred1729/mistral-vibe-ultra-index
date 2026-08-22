from __future__ import annotations

from vibe.core.repository_index.graph import hydrate_dependency_graph
from vibe.core.repository_index.models import (
    DiscoveredFile,
    FileGraphScore,
    GraphEdgeKind,
)
from vibe.core.repository_index.parsers.python import parse_python_facts


def _file(path: str, content_hash: str) -> DiscoveredFile:
    return DiscoveredFile(
        path=path, content_hash=content_hash, language="python", size=1, modified_ns=1
    )


def test_hydrates_internal_import_symbol_call_test_and_external_evidence() -> None:
    files = (
        _file("pkg/a.py", "a"),
        _file("pkg/b.py", "b"),
        _file("tests/test_a.py", "t"),
    )
    facts = (
        parse_python_facts(
            "pkg/a.py",
            "a",
            "from pkg.b import helper\nfrom pydantic import BaseModel\nhelper()\n",
        ),
        parse_python_facts("pkg/b.py", "b", "def helper():\n    return 1\n"),
        parse_python_facts("tests/test_a.py", "t", "from pkg import a\n"),
    )

    graph = hydrate_dependency_graph(files, facts)

    assert any(
        edge.source == "pkg/a.py"
        and edge.target == "pkg/b.py"
        and edge.kind is GraphEdgeKind.IMPORTS
        for edge in graph.edges
    )
    assert any(
        edge.source == "pkg/a.py"
        and edge.target == "symbol:pkg/b.py#helper"
        and edge.kind is GraphEdgeKind.CALLS
        for edge in graph.edges
    )
    assert any(
        edge.source == "pkg/a.py"
        and edge.target == "tests/test_a.py"
        and edge.kind is GraphEdgeKind.TESTED_BY
        for edge in graph.edges
    )
    assert any(
        edge.source == "pkg/a.py"
        and edge.target is None
        and edge.evidence == "pydantic.BaseModel"
        and edge.unresolved
        for edge in graph.edges
    )
    assert abs(sum(score.importance for score in graph.scores) - 1.0) < 1e-9


def test_reuses_scores_when_topology_fingerprint_is_unchanged() -> None:
    files = (_file("a.py", "a"), _file("b.py", "b"))
    facts = (
        parse_python_facts("a.py", "a", "import b\n"),
        parse_python_facts("b.py", "b", "import a\n"),
    )
    first = hydrate_dependency_graph(files, facts)
    sentinel_scores = tuple(
        FileGraphScore(path=score.path, importance=0.5, component=99)
        for score in first.scores
    )

    reused = hydrate_dependency_graph(
        files,
        facts,
        previous_fingerprint=first.topology_fingerprint,
        previous_scores=sentinel_scores,
    )

    assert reused.scores == sentinel_scores
    assert len({score.component for score in first.scores}) == 1
