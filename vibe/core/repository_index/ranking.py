from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath

from vibe.core.repository_index.models import (
    RepositoryModuleRole,
    RepositoryQuerySuggestion,
    RepositorySearchGroup,
    RepositorySearchMatch,
    RepositorySearchMode,
)

_MAX_RESULT_BYTES = 48_000
_MAX_SNIPPET_BYTES = 3_000
_SHARED_COMPONENT_THRESHOLD = 2
_MAX_QUERY_SUGGESTIONS = 3


def rank_repository_matches(
    matches: Sequence[RepositorySearchMatch],
    *,
    query: str,
    importance: Mapping[str, float],
    max_results: int,
) -> tuple[RepositorySearchMatch, ...]:
    query_folded = query.casefold()
    unique: dict[tuple[str, int, str], RepositorySearchMatch] = {}
    for match in matches:
        unique.setdefault((match.path, match.line_start, match.relationship), match)

    ordered = sorted(
        unique.values(),
        key=lambda match: (
            -_relationship_score(match.relationship),
            -int(query_folded in match.path.casefold()),
            -importance.get(match.path, 0.0),
            match.path,
            match.line_start,
        ),
    )
    bounded: list[RepositorySearchMatch] = []
    used_bytes = 0
    for match in ordered:
        snippet = _truncate_utf8(match.snippet, _MAX_SNIPPET_BYTES)
        reason = match.score_reason
        file_importance = importance.get(match.path)
        if file_importance is not None:
            reason = f"{reason}, file importance {file_importance:.6f}"
        result = match.model_copy(update={"snippet": snippet, "score_reason": reason})
        result_bytes = len(result.model_dump_json().encode("utf-8"))
        if bounded and used_bytes + result_bytes > _MAX_RESULT_BYTES:
            break
        bounded.append(result)
        used_bytes += result_bytes
        if len(bounded) == max_results:
            break
    return tuple(bounded)


def add_module_boundaries(
    matches: Sequence[RepositorySearchMatch],
    *,
    incoming_components: Mapping[str, frozenset[str]],
) -> tuple[RepositorySearchMatch, ...]:
    return tuple(
        match.model_copy(
            update={
                "component": repository_component(match.path),
                "module_role": _module_role(match.path, incoming_components),
            }
        )
        for match in matches
    )


def group_repository_matches(
    matches: Sequence[RepositorySearchMatch],
) -> tuple[tuple[RepositorySearchMatch, ...], tuple[RepositorySearchGroup, ...]]:
    grouped: dict[str, list[RepositorySearchMatch]] = {}
    for match in matches:
        grouped.setdefault(match.component, []).append(match)

    reordered: list[RepositorySearchMatch] = []
    summaries: list[RepositorySearchGroup] = []
    for component, component_matches in grouped.items():
        reordered.extend(component_matches)
        summaries.append(
            RepositorySearchGroup(
                component=component,
                module_roles=tuple(
                    dict.fromkeys(match.module_role for match in component_matches)
                ),
                match_count=len(component_matches),
                paths=tuple(dict.fromkeys(match.path for match in component_matches)),
            )
        )
    return tuple(reordered), tuple(summaries)


def repository_component(path: str) -> str:
    parent = PurePosixPath(path).parent.as_posix()
    return "(root)" if parent == "." else parent


def suggest_repository_queries(
    *,
    query: str,
    mode: RepositorySearchMode,
    matches: Sequence[RepositorySearchMatch],
    groups: Sequence[RepositorySearchGroup],
) -> tuple[RepositoryQuerySuggestion, ...]:
    suggestions: list[RepositoryQuerySuggestion] = []
    if definition := next(
        (match for match in matches if match.relationship == "defines"), None
    ):
        suggestions.append(
            RepositoryQuerySuggestion(
                query=definition.symbol or definition.snippet,
                mode=RepositorySearchMode.IMPACT,
                reason="Inspect dependents and tests for the strongest definition.",
            )
        )

    if refinement := _refinement_mode(mode):
        suggestions.append(
            RepositoryQuerySuggestion(
                query=query,
                mode=refinement,
                reason=f"Refine the initial evidence with {refinement.value} search.",
            )
        )

    if group := next((item for item in groups if item.component != "(root)"), None):
        suggestions.append(
            RepositoryQuerySuggestion(
                query=query,
                mode=RepositorySearchMode.AUTO,
                path=group.component,
                reason="Focus on the highest-ranked repository component.",
            )
        )

    unique: dict[
        tuple[str, RepositorySearchMode, str | None], RepositoryQuerySuggestion
    ]
    unique = {}
    for suggestion in suggestions:
        unique.setdefault(
            (suggestion.query, suggestion.mode, suggestion.path), suggestion
        )
    return tuple(unique.values())[:_MAX_QUERY_SUGGESTIONS]


def _refinement_mode(mode: RepositorySearchMode) -> RepositorySearchMode | None:
    match mode:
        case RepositorySearchMode.AUTO:
            return RepositorySearchMode.DEPENDENCY
        case RepositorySearchMode.TEXT:
            return RepositorySearchMode.SYMBOL
        case RepositorySearchMode.SYMBOL:
            return RepositorySearchMode.DEPENDENCY
        case RepositorySearchMode.DEPENDENCY:
            return RepositorySearchMode.IMPACT
        case RepositorySearchMode.IMPACT:
            return None


def incoming_component_map(
    edges: Sequence[tuple[str, str]],
) -> dict[str, frozenset[str]]:
    incoming: dict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        source_component = repository_component(source)
        target_component = repository_component(target)
        if source_component != target_component:
            incoming[target].add(source_component)
    return {path: frozenset(components) for path, components in incoming.items()}


def _module_role(
    path: str, incoming_components: Mapping[str, frozenset[str]]
) -> RepositoryModuleRole:
    parts = PurePosixPath(path).parts
    if any(part == "tests" or part.startswith("test_") for part in parts):
        return RepositoryModuleRole.TEST
    if repository_component(path) == "(root)":
        return RepositoryModuleRole.ROOT
    if len(incoming_components.get(path, ())) >= _SHARED_COMPONENT_THRESHOLD:
        return RepositoryModuleRole.SHARED
    return RepositoryModuleRole.FEATURE


def _relationship_score(relationship: str) -> int:
    if relationship == "defines":
        return 100
    if relationship.startswith("dependent_via_") or relationship == "tested_by":
        return 90
    if relationship.startswith("dependency_distance_1"):
        return 80
    if relationship.startswith("dependency_distance_2"):
        return 70
    if relationship in {"call", "inherits", "reference"}:
        return 60
    return 50


def _truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[: max_bytes - 3].decode("utf-8", errors="ignore") + "..."


__all__ = [
    "add_module_boundaries",
    "group_repository_matches",
    "incoming_component_map",
    "rank_repository_matches",
    "repository_component",
    "suggest_repository_queries",
]
