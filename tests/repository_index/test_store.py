from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.repository_index.models import (
    DiscoveredFile,
    IndexChunk,
    IndexStatus,
    RepositorySearchMode,
)
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


def test_search_caps_total_result_bytes(tmp_path: Path) -> None:
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

    result = store.search(generation, "needle", max_results=100)

    encoded_matches = sum(
        len(match.model_dump_json().encode("utf-8")) for match in result.matches
    )
    assert encoded_matches <= 48_000
    assert len(result.matches) < 100
