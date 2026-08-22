from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Event

from vibe.core.repository_index.chunks import RepositoryChangedDuringBuildError
from vibe.core.repository_index.discovery import DiscoveryCancelledError
from vibe.core.repository_index.models import DiscoveredFile, FileFacts
from vibe.core.repository_index.parsers import PYTHON_PARSER_VERSION, parse_python_facts


def build_file_facts(
    root: Path,
    files: tuple[DiscoveredFile, ...],
    cached: dict[tuple[str, str], FileFacts],
    *,
    cancel_event: Event | None = None,
) -> tuple[FileFacts, ...]:
    facts: list[FileFacts] = []
    for file in files:
        if cancel_event is not None and cancel_event.is_set():
            raise DiscoveryCancelledError("Repository parsing was cancelled.")
        cached_fact = cached.get((file.content_hash, file.language))
        expected_version = PYTHON_PARSER_VERSION if file.language == "python" else 0
        if (
            cached_fact is not None
            and cached_fact.language == file.language
            and cached_fact.parser_version == expected_version
        ):
            facts.append(cached_fact.model_copy(update={"path": file.path}))
            continue
        if file.language != "python":
            facts.append(
                FileFacts(
                    path=file.path,
                    content_hash=file.content_hash,
                    language=file.language,
                    parser_version=0,
                )
            )
            continue
        path = root / file.path
        try:
            raw = path.read_bytes()
        except (FileNotFoundError, PermissionError, OSError) as exc:
            raise RepositoryChangedDuringBuildError(
                f"Indexed file changed while parsing: {file.path}"
            ) from exc
        if hashlib.sha256(raw).hexdigest() != file.content_hash:
            raise RepositoryChangedDuringBuildError(
                f"Indexed file changed while parsing: {file.path}"
            )
        facts.append(
            parse_python_facts(
                file.path, file.content_hash, raw.decode("utf-8", errors="replace")
            )
        )
    return tuple(facts)
