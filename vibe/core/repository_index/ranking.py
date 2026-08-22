from __future__ import annotations

from collections.abc import Mapping, Sequence

from vibe.core.repository_index.models import RepositorySearchMatch

_MAX_RESULT_BYTES = 48_000
_MAX_SNIPPET_BYTES = 3_000


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


__all__ = ["rank_repository_matches"]
