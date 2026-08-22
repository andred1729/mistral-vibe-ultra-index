from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Event

from vibe.core.repository_index.discovery import DiscoveryCancelledError
from vibe.core.repository_index.models import DiscoveredFile, IndexChunk

_CHUNK_LINES = 40
_CHUNK_OVERLAP = 5
_MAX_CHUNK_CHARACTERS = 8_000


class RepositoryChangedDuringBuildError(Exception): ...


def build_text_chunks(
    root: Path, files: tuple[DiscoveredFile, ...], *, cancel_event: Event | None = None
) -> tuple[IndexChunk, ...]:
    chunks: list[IndexChunk] = []
    for file in files:
        if cancel_event is not None and cancel_event.is_set():
            raise DiscoveryCancelledError("Repository indexing was cancelled.")
        path = root / file.path
        try:
            raw = path.read_bytes()
        except (FileNotFoundError, PermissionError, OSError) as exc:
            raise RepositoryChangedDuringBuildError(
                f"Indexed file changed while building: {file.path}"
            ) from exc
        if hashlib.sha256(raw).hexdigest() != file.content_hash:
            raise RepositoryChangedDuringBuildError(
                f"Indexed file changed while building: {file.path}"
            )

        lines = raw.decode("utf-8", errors="replace").splitlines()
        if not lines:
            lines = [""]
        step = _CHUNK_LINES - _CHUNK_OVERLAP
        for ordinal, offset in enumerate(range(0, len(lines), step)):
            selected = lines[offset : offset + _CHUNK_LINES]
            chunks.append(
                IndexChunk(
                    path=file.path,
                    ordinal=ordinal,
                    line_start=offset + 1,
                    line_end=offset + len(selected),
                    content="\n".join(selected)[:_MAX_CHUNK_CHARACTERS],
                )
            )
            if offset + _CHUNK_LINES >= len(lines):
                break
    return tuple(chunks)
