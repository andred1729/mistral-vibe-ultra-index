from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import PurePosixPath

from vibe.core.repository_index.models import (
    DiscoveredFile,
    FileFacts,
    FileGraphScore,
    GraphEdge,
    GraphEdgeKind,
    HydratedGraph,
    ImportFact,
    ReferenceKind,
)

_PAGERANK_DAMPING = 0.85
_PAGERANK_ITERATIONS = 20


def hydrate_dependency_graph(  # noqa: PLR0914
    files: tuple[DiscoveredFile, ...],
    facts: tuple[FileFacts, ...],
    *,
    previous_fingerprint: str | None = None,
    previous_scores: tuple[FileGraphScore, ...] = (),
) -> HydratedGraph:
    file_paths = {file.path for file in files}
    module_paths = _python_module_paths(file_paths)
    symbol_paths: dict[str, list[tuple[str, str]]] = defaultdict(list)
    edges: list[GraphEdge] = []

    for fact in facts:
        edges.append(
            GraphEdge(source="root", target=fact.path, kind=GraphEdgeKind.CONTAINS)
        )
        for symbol in fact.symbols:
            symbol_id = _symbol_id(fact.path, symbol.qualified_name)
            symbol_paths[symbol.name].append((fact.path, symbol_id))
            edges.append(
                GraphEdge(
                    source=fact.path,
                    target=symbol_id,
                    kind=GraphEdgeKind.DEFINES,
                    line=symbol.line_start,
                    evidence=symbol.qualified_name,
                )
            )

    for fact in facts:
        for imported in fact.imports:
            target = _resolve_import(fact.path, imported, module_paths)
            evidence = _import_evidence(imported)
            edges.append(
                GraphEdge(
                    source=fact.path,
                    target=target,
                    kind=GraphEdgeKind.IMPORTS,
                    line=imported.line,
                    evidence=evidence,
                    unresolved=target is None,
                )
            )
        for reference in fact.references:
            target = _resolve_symbol(fact.path, reference.name, symbol_paths)
            kind = {
                ReferenceKind.REFERENCE: GraphEdgeKind.REFERENCES,
                ReferenceKind.CALL: GraphEdgeKind.CALLS,
                ReferenceKind.INHERITS: GraphEdgeKind.INHERITS,
            }[reference.kind]
            if target is not None or reference.kind is ReferenceKind.INHERITS:
                edges.append(
                    GraphEdge(
                        source=fact.path,
                        target=target,
                        kind=kind,
                        line=reference.line,
                        evidence=reference.name,
                        unresolved=target is None,
                    )
                )

    import_edges = [
        edge
        for edge in edges
        if edge.kind is GraphEdgeKind.IMPORTS
        and edge.target is not None
        and edge.target in file_paths
    ]
    for edge in import_edges:
        imported_path = edge.target
        if imported_path is None:  # pragma: no cover - narrowed above
            continue
        if _is_test_path(edge.source) and not _is_test_path(imported_path):
            edges.append(
                GraphEdge(
                    source=imported_path,
                    target=edge.source,
                    kind=GraphEdgeKind.TESTED_BY,
                    evidence=edge.source,
                )
            )

    deduplicated = tuple(
        sorted({(_edge_key(edge)): edge for edge in edges}.values(), key=_edge_key)
    )
    adjacency = _file_adjacency(file_paths, deduplicated)
    fingerprint = _topology_fingerprint(adjacency)
    if (
        previous_fingerprint == fingerprint
        and {score.path for score in previous_scores} == file_paths
    ):
        scores = previous_scores
    else:
        components = _strongly_connected_components(adjacency)
        importance = _page_rank(adjacency)
        scores = tuple(
            FileGraphScore(
                path=path, importance=importance[path], component=components[path]
            )
            for path in sorted(file_paths)
        )
    return HydratedGraph(
        edges=deduplicated, scores=scores, topology_fingerprint=fingerprint
    )


def _python_module_paths(paths: set[str]) -> dict[str, str]:
    modules: dict[str, str] = {}
    for path in sorted(paths):
        pure = PurePosixPath(path)
        if pure.suffix != ".py":
            continue
        parts = list(pure.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        if parts:
            modules[".".join(parts)] = path
    return modules


def _resolve_import(
    source_path: str, imported: ImportFact, module_paths: dict[str, str]
) -> str | None:
    module = imported.module
    if imported.level:
        source_module = PurePosixPath(source_path).with_suffix("").parts
        package = list(source_module)
        if package and package[-1] != "__init__":
            package.pop()
        trim = imported.level - 1
        if trim:
            package = package[: max(0, len(package) - trim)]
        module = ".".join([*package, *([module] if module else [])])
    candidates = []
    if imported.imported_name and imported.imported_name != "*":
        candidates.append(".".join(filter(None, [module, imported.imported_name])))
    candidates.append(module)
    for candidate in candidates:
        if candidate in module_paths:
            return module_paths[candidate]
    return None


def _resolve_symbol(
    source_path: str, reference: str, symbols: dict[str, list[tuple[str, str]]]
) -> str | None:
    candidates = symbols.get(reference.rsplit(".", maxsplit=1)[-1], [])
    local = [symbol for path, symbol in candidates if path == source_path]
    if len(local) == 1:
        return local[0]
    if len(candidates) == 1:
        return candidates[0][1]
    return None


def _file_adjacency(
    paths: set[str], edges: tuple[GraphEdge, ...]
) -> dict[str, set[str]]:
    adjacency = {path: set() for path in paths}
    for edge in edges:
        if edge.source not in paths or edge.target is None:
            continue
        target = edge.target
        if target.startswith("symbol:"):
            target = target.removeprefix("symbol:").split("#", maxsplit=1)[0]
        if target in paths and target != edge.source:
            adjacency[edge.source].add(target)
    return adjacency


def _page_rank(adjacency: dict[str, set[str]]) -> dict[str, float]:
    if not adjacency:
        return {}
    count = len(adjacency)
    rank = {node: 1.0 / count for node in adjacency}
    inbound: dict[str, set[str]] = {node: set() for node in adjacency}
    for source, targets in adjacency.items():
        for target in targets:
            inbound[target].add(source)
    for _ in range(_PAGERANK_ITERATIONS):
        dangling = sum(rank[node] for node, targets in adjacency.items() if not targets)
        rank = {
            node: (1 - _PAGERANK_DAMPING) / count
            + _PAGERANK_DAMPING * dangling / count
            + _PAGERANK_DAMPING
            * sum(rank[source] / len(adjacency[source]) for source in inbound[node])
            for node in adjacency
        }
    return rank


def _strongly_connected_components(adjacency: dict[str, set[str]]) -> dict[str, int]:
    visited: set[str] = set()
    finish_order: list[str] = []
    for start in sorted(adjacency):
        if start in visited:
            continue
        stack: list[tuple[str, bool]] = [(start, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                finish_order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            stack.extend(
                (target, False) for target in sorted(adjacency[node], reverse=True)
            )

    reverse = {node: set() for node in adjacency}
    for source, targets in adjacency.items():
        for target in targets:
            reverse[target].add(source)
    components: dict[str, int] = {}
    component_id = 0
    for start in reversed(finish_order):
        if start in components:
            continue
        reverse_stack = [start]
        while reverse_stack:
            node = reverse_stack.pop()
            if node in components:
                continue
            components[node] = component_id
            reverse_stack.extend(reverse[node] - components.keys())
        component_id += 1
    return components


def _topology_fingerprint(adjacency: dict[str, set[str]]) -> str:
    payload = "\n".join(
        f"{source}\0{target}"
        for source in sorted(adjacency)
        for target in sorted(adjacency[source])
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _symbol_id(path: str, qualified_name: str) -> str:
    return f"symbol:{path}#{qualified_name}"


def _import_evidence(imported: ImportFact) -> str:
    prefix = "." * imported.level
    suffix = f".{imported.imported_name}" if imported.imported_name else ""
    return f"{prefix}{imported.module}{suffix}"


def _is_test_path(path: str) -> bool:
    pure = PurePosixPath(path)
    return "tests" in pure.parts or pure.name.startswith("test_")


def _edge_key(edge: GraphEdge) -> tuple[str, str, str, str]:
    return (
        edge.source,
        edge.target or "",
        edge.kind.value,
        (edge.evidence or "") if edge.unresolved else "",
    )
