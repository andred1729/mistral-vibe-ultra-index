from __future__ import annotations

from collections.abc import Callable
import fnmatch
import hashlib
from pathlib import Path
from threading import Event

from vibe.core.git.repo import GitRepo
from vibe.core.repository_index.models import DiscoveredFile, DiscoveryResult
from vibe.utils.io import read_safe

DEFAULT_MAX_FILE_SIZE = 2 * 1024 * 1024
_BINARY_PROBE_SIZE = 8192
_SENSITIVE_PATTERNS = (
    ".env",
    ".env.*",
    "*.key",
    "*.pem",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "credentials.json",
    "secrets.*",
)


class DiscoveryCancelledError(Exception): ...


class _VibeIgnore:
    def __init__(self, root: Path) -> None:
        path = root / ".vibeignore"
        self._patterns = self._read_patterns(path) if path.is_file() else ()

    @staticmethod
    def _read_patterns(path: Path) -> tuple[tuple[str, bool], ...]:
        patterns: list[tuple[str, bool]] = []
        for raw_line in read_safe(path).text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            include = line.startswith("!")
            pattern = line[1:] if include else line
            if pattern:
                patterns.append((pattern.lstrip("/"), include))
        return tuple(patterns)

    def excludes(self, relative_path: str) -> bool:
        excluded = False
        parts = relative_path.split("/")
        for pattern, include in self._patterns:
            candidate = pattern.rstrip("/")
            matches = fnmatch.fnmatch(relative_path, candidate)
            if "/" not in candidate:
                matches = matches or any(
                    fnmatch.fnmatch(part, candidate) for part in parts
                )
            if pattern.endswith("/"):
                matches = relative_path == candidate or relative_path.startswith(
                    f"{candidate}/"
                )
            if matches:
                excluded = not include
        return excluded


class RepositoryDiscovery:
    def __init__(
        self,
        *,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        sensitive_patterns: tuple[str, ...] = _SENSITIVE_PATTERNS,
    ) -> None:
        self._max_file_size = max_file_size
        self._sensitive_patterns = sensitive_patterns

    def discover(
        self,
        cwd: Path,
        *,
        cancel_event: Event | None = None,
        on_file: Callable[[int], None] | None = None,
    ) -> DiscoveryResult:
        repository = GitRepo.open(cwd)
        try:
            root = repository.working_dir.resolve()
            candidates = repository.indexable_paths()
        finally:
            repository.close()

        vibe_ignore = _VibeIgnore(root)
        files: list[DiscoveredFile] = []
        skipped_binary = 0
        skipped_oversized = 0
        skipped_sensitive = 0
        skipped_symlink = 0

        for candidate in candidates:
            if cancel_event is not None and cancel_event.is_set():
                raise DiscoveryCancelledError("Repository discovery was cancelled.")
            try:
                relative = candidate.relative_to(root).as_posix()
            except ValueError:
                continue
            if self._is_sensitive(relative) or vibe_ignore.excludes(relative):
                skipped_sensitive += 1
                continue
            if candidate.is_symlink():
                skipped_symlink += 1
                continue
            try:
                stat = candidate.stat()
            except (FileNotFoundError, OSError):
                continue
            if not candidate.is_file():
                continue
            if stat.st_size > self._max_file_size:
                skipped_oversized += 1
                continue
            try:
                content = candidate.read_bytes()
            except (PermissionError, OSError):
                continue
            if b"\0" in content[:_BINARY_PROBE_SIZE]:
                skipped_binary += 1
                continue
            files.append(
                DiscoveredFile(
                    path=relative,
                    content_hash=hashlib.sha256(content).hexdigest(),
                    language=_language_for(candidate),
                    size=stat.st_size,
                    modified_ns=stat.st_mtime_ns,
                )
            )
            if on_file is not None:
                on_file(len(files))

        return DiscoveryResult(
            root=root,
            files=tuple(sorted(files, key=lambda item: item.path)),
            skipped_binary=skipped_binary,
            skipped_oversized=skipped_oversized,
            skipped_sensitive=skipped_sensitive,
            skipped_symlink=skipped_symlink,
        )

    def _is_sensitive(self, relative_path: str) -> bool:
        name = Path(relative_path).name
        return any(
            fnmatch.fnmatch(name.casefold(), pattern.casefold())
            for pattern in self._sensitive_patterns
        )


def _language_for(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".py":
        return "python"
    return suffix.removeprefix(".") or "text"
