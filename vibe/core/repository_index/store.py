from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sqlite3
from typing import Any

from vibe.core.repository_index.models import (
    DiscoveredFile,
    FileFacts,
    FileGraphScore,
    GraphEdge,
    GraphEdgeKind,
    HydratedGraph,
    IndexChunk,
    IndexGeneration,
    IndexStatus,
    RepositoryDependencyDirection,
    RepositoryDependencyTree,
    RepositoryMap,
    RepositoryMapEntry,
    RepositorySearchMatch,
    RepositorySearchMode,
    RepositorySearchResult,
)
from vibe.core.repository_index.ranking import (
    add_module_boundaries,
    group_repository_matches,
    incoming_component_map,
    rank_repository_matches,
    repository_query_concepts,
    suggest_repository_queries,
)

_SCHEMA_VERSION = 5
_GRAPH_SCHEMA_VERSION = 3
_STRUCTURED_SEARCH_SCHEMA_VERSION = 4
_CONTENT_FACT_SCHEMA_VERSION = 5
_MAX_SEARCH_RESULTS = 100
_MAX_IMPACT_DEFINITIONS = 5
_MAX_MAP_FILES = 200
_MAX_MAP_SYMBOLS_PER_FILE = 20
_MAX_DEPENDENCY_TREE_ROOTS = 3
_MAX_DEPENDENCY_TREE_DEPTH = 2
_MAX_DEPENDENCY_TREE_NODES = 50


class RepositoryIndexCorruptError(Exception): ...


@dataclass(frozen=True, slots=True)
class _SearchContext:
    incoming_components: dict[str, frozenset[str]]
    dependency_trees: tuple[RepositoryDependencyTree, ...]


class RepositoryIndexStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with closing(self._connect()) as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS roots (
                        id INTEGER PRIMARY KEY,
                        path TEXT NOT NULL UNIQUE,
                        published_generation_id INTEGER
                    );
                    CREATE TABLE IF NOT EXISTS generations (
                        id INTEGER PRIMARY KEY,
                        root_id INTEGER NOT NULL REFERENCES roots(id),
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        completed_at TEXT,
                        file_count INTEGER NOT NULL DEFAULT 0,
                        topology_fingerprint TEXT
                    );
                    CREATE TABLE IF NOT EXISTS files (
                        generation_id INTEGER NOT NULL REFERENCES generations(id),
                        path TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        language TEXT NOT NULL,
                        size INTEGER NOT NULL,
                        modified_ns INTEGER NOT NULL,
                        PRIMARY KEY (generation_id, path)
                    ) WITHOUT ROWID;
                    CREATE INDEX IF NOT EXISTS files_hash_idx
                        ON files(generation_id, content_hash);
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        generation_id UNINDEXED,
                        path,
                        line_start UNINDEXED,
                        line_end UNINDEXED,
                        content,
                        tokenize = 'unicode61'
                    );
                    CREATE TABLE IF NOT EXISTS fact_cache (
                        content_hash TEXT NOT NULL,
                        language TEXT NOT NULL,
                        parser_version INTEGER NOT NULL,
                        payload TEXT NOT NULL,
                        PRIMARY KEY (content_hash, language, parser_version)
                    ) WITHOUT ROWID;
                    CREATE TABLE IF NOT EXISTS generation_facts (
                        generation_id INTEGER NOT NULL REFERENCES generations(id),
                        path TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        PRIMARY KEY (generation_id, path)
                    ) WITHOUT ROWID;
                    CREATE TABLE IF NOT EXISTS symbols (
                        generation_id INTEGER NOT NULL REFERENCES generations(id),
                        path TEXT NOT NULL,
                        name TEXT NOT NULL,
                        qualified_name TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        line_start INTEGER NOT NULL,
                        line_end INTEGER NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS symbols_query_idx
                        ON symbols(generation_id, name, qualified_name);
                    CREATE TABLE IF NOT EXISTS references_index (
                        generation_id INTEGER NOT NULL REFERENCES generations(id),
                        path TEXT NOT NULL,
                        name TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        line INTEGER NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS references_query_idx
                        ON references_index(generation_id, name, kind);
                    CREATE TABLE IF NOT EXISTS imports_index (
                        generation_id INTEGER NOT NULL REFERENCES generations(id),
                        path TEXT NOT NULL,
                        module TEXT NOT NULL,
                        imported_name TEXT,
                        alias TEXT,
                        level INTEGER NOT NULL,
                        line INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS symbol_cache (
                        content_hash TEXT NOT NULL,
                        language TEXT NOT NULL,
                        parser_version INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        qualified_name TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        line_start INTEGER NOT NULL,
                        line_end INTEGER NOT NULL,
                        PRIMARY KEY (
                            content_hash, language, parser_version,
                            qualified_name, line_start
                        )
                    ) WITHOUT ROWID;
                    CREATE INDEX IF NOT EXISTS symbol_cache_query_idx
                        ON symbol_cache(name, qualified_name);
                    CREATE TABLE IF NOT EXISTS reference_cache (
                        content_hash TEXT NOT NULL,
                        language TEXT NOT NULL,
                        parser_version INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        line INTEGER NOT NULL,
                        PRIMARY KEY (
                            content_hash, language, parser_version,
                            name, kind, line
                        )
                    ) WITHOUT ROWID;
                    CREATE INDEX IF NOT EXISTS reference_cache_query_idx
                        ON reference_cache(name, kind);
                    CREATE TABLE IF NOT EXISTS import_cache (
                        content_hash TEXT NOT NULL,
                        language TEXT NOT NULL,
                        parser_version INTEGER NOT NULL,
                        module TEXT NOT NULL,
                        imported_name TEXT,
                        alias TEXT,
                        level INTEGER NOT NULL,
                        line INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS graph_edges (
                        generation_id INTEGER NOT NULL REFERENCES generations(id),
                        source TEXT NOT NULL,
                        target TEXT,
                        kind TEXT NOT NULL,
                        line INTEGER,
                        evidence TEXT,
                        unresolved INTEGER NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS graph_edges_target_idx
                        ON graph_edges(generation_id, target, kind);
                    CREATE INDEX IF NOT EXISTS graph_edges_source_idx
                        ON graph_edges(generation_id, source, kind);
                    CREATE TABLE IF NOT EXISTS file_scores (
                        generation_id INTEGER NOT NULL REFERENCES generations(id),
                        path TEXT NOT NULL,
                        importance REAL NOT NULL,
                        component INTEGER NOT NULL,
                        PRIMARY KEY (generation_id, path)
                    ) WITHOUT ROWID;
                    """
                )
                row = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'schema_version'"
                ).fetchone()
                version = int(row[0]) if row is not None else _SCHEMA_VERSION
                if version < 1 or version > _SCHEMA_VERSION:
                    raise RepositoryIndexCorruptError(
                        f"Unsupported repository index schema version: {version}"
                    )
                connection.execute(
                    "INSERT OR IGNORE INTO metadata(key, value) VALUES('schema_version', ?)",
                    (str(_SCHEMA_VERSION),),
                )
                if version < _SCHEMA_VERSION:
                    if version < _GRAPH_SCHEMA_VERSION:
                        connection.execute(
                            "ALTER TABLE generations ADD COLUMN topology_fingerprint TEXT"
                        )
                    if version < _STRUCTURED_SEARCH_SCHEMA_VERSION:
                        connection.execute(
                            "UPDATE generations SET topology_fingerprint = NULL"
                        )
                    if version < _CONTENT_FACT_SCHEMA_VERSION:
                        connection.execute(
                            "UPDATE generations SET topology_fingerprint = NULL"
                        )
                    connection.execute(
                        "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
                        (str(_SCHEMA_VERSION),),
                    )
                connection.commit()
        except (sqlite3.DatabaseError, ValueError) as e:
            if isinstance(e, RepositoryIndexCorruptError):
                raise
            raise RepositoryIndexCorruptError(
                f"Repository index storage is corrupt: {e}"
            ) from e

    def publish(
        self,
        root: Path,
        files: Sequence[DiscoveredFile],
        chunks: Sequence[IndexChunk] = (),
        facts: Sequence[FileFacts] = (),
        graph: HydratedGraph | None = None,
        base_generation_id: int | None = None,
        changed_paths: frozenset[str] | None = None,
    ) -> IndexGeneration:
        self.initialize()
        try:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                root_path = str(root.resolve())
                connection.execute(
                    "INSERT OR IGNORE INTO roots(path) VALUES(?)", (root_path,)
                )
                root_id = int(
                    connection.execute(
                        "SELECT id FROM roots WHERE path = ?", (root_path,)
                    ).fetchone()[0]
                )
                cursor = connection.execute(
                    """
                    INSERT INTO generations(root_id, status, topology_fingerprint)
                    VALUES(?, ?, ?)
                    """,
                    (
                        root_id,
                        IndexStatus.BUILDING.value,
                        graph.topology_fingerprint if graph is not None else None,
                    ),
                )
                if cursor.lastrowid is None:
                    raise RepositoryIndexCorruptError(
                        "Repository index generation did not receive an ID."
                    )
                generation_id = cursor.lastrowid
                connection.executemany(
                    """
                    INSERT INTO files(
                        generation_id, path, content_hash, language, size, modified_ns
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            generation_id,
                            file.path,
                            file.content_hash,
                            file.language,
                            file.size,
                            file.modified_ns,
                        )
                        for file in files
                    ),
                )
                if base_generation_id is not None:
                    _copy_unchanged_analysis(
                        connection, base_generation_id, generation_id
                    )
                published_facts = (
                    facts
                    if changed_paths is None
                    else tuple(fact for fact in facts if fact.path in changed_paths)
                )
                cache_keys = tuple(
                    (fact.content_hash, fact.language) for fact in published_facts
                )
                for table in (
                    "fact_cache",
                    "symbol_cache",
                    "reference_cache",
                    "import_cache",
                ):
                    connection.executemany(
                        f"DELETE FROM {table} WHERE content_hash = ? AND language = ?",
                        cache_keys,
                    )
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO fact_cache(
                        content_hash, language, parser_version, payload
                    ) VALUES(?, ?, ?, ?)
                    """,
                    (
                        (
                            fact.content_hash,
                            fact.language,
                            fact.parser_version,
                            fact.model_dump_json(),
                        )
                        for fact in published_facts
                    ),
                )
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO symbol_cache(
                        content_hash, language, parser_version, name,
                        qualified_name, kind, line_start, line_end
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            fact.content_hash,
                            fact.language,
                            fact.parser_version,
                            symbol.name,
                            symbol.qualified_name,
                            symbol.kind.value,
                            symbol.line_start,
                            symbol.line_end,
                        )
                        for fact in published_facts
                        for symbol in fact.symbols
                    ),
                )
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO reference_cache(
                        content_hash, language, parser_version, name, kind, line
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            fact.content_hash,
                            fact.language,
                            fact.parser_version,
                            reference.name,
                            reference.kind.value,
                            reference.line,
                        )
                        for fact in published_facts
                        for reference in fact.references
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO import_cache(
                        content_hash, language, parser_version, module,
                        imported_name, alias, level, line
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            fact.content_hash,
                            fact.language,
                            fact.parser_version,
                            imported.module,
                            imported.imported_name,
                            imported.alias,
                            imported.level,
                            imported.line,
                        )
                        for fact in published_facts
                        for imported in fact.imports
                    ),
                )
                if graph is not None:
                    connection.executemany(
                        """
                        INSERT INTO graph_edges(
                            generation_id, source, target, kind, line,
                            evidence, unresolved
                        ) VALUES(?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            (
                                generation_id,
                                edge.source,
                                edge.target,
                                edge.kind.value,
                                edge.line,
                                edge.evidence,
                                edge.unresolved,
                            )
                            for edge in graph.edges
                        ),
                    )
                    connection.executemany(
                        """
                        INSERT INTO file_scores(
                            generation_id, path, importance, component
                        ) VALUES(?, ?, ?, ?)
                        """,
                        (
                            (
                                generation_id,
                                score.path,
                                score.importance,
                                score.component,
                            )
                            for score in graph.scores
                        ),
                    )
                connection.executemany(
                    """
                    INSERT INTO chunks_fts(
                        generation_id, path, line_start, line_end, content
                    ) VALUES(?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            generation_id,
                            chunk.path,
                            chunk.line_start,
                            chunk.line_end,
                            chunk.content,
                        )
                        for chunk in chunks
                    ),
                )
                connection.execute(
                    """
                    UPDATE generations
                    SET status = ?, completed_at = CURRENT_TIMESTAMP, file_count = ?
                    WHERE id = ?
                    """,
                    (IndexStatus.COMPLETE.value, len(files), generation_id),
                )
                connection.execute(
                    "UPDATE roots SET published_generation_id = ? WHERE id = ?",
                    (generation_id, root_id),
                )
                connection.commit()
        except sqlite3.DatabaseError as e:
            raise RepositoryIndexCorruptError(
                f"Failed to publish repository index generation: {e}"
            ) from e
        generation = self.current_generation(root)
        if generation is None:
            raise RepositoryIndexCorruptError(
                "Published repository index generation could not be read."
            )
        return generation

    def current_generation(self, root: Path) -> IndexGeneration | None:
        if not self.path.exists():
            return None
        try:
            with closing(self._connect(read_only=True)) as connection:
                row = connection.execute(
                    """
                    SELECT g.id, r.path, g.status, g.file_count,
                           g.created_at, g.completed_at, g.topology_fingerprint
                    FROM roots AS r
                    JOIN generations AS g ON g.id = r.published_generation_id
                    WHERE r.path = ?
                    """,
                    (str(root.resolve()),),
                ).fetchone()
        except sqlite3.DatabaseError as e:
            raise RepositoryIndexCorruptError(
                f"Repository index storage is corrupt or unreadable: {e}"
            ) from e
        if row is None:
            return None
        return IndexGeneration(
            id=row[0],
            root=Path(row[1]),
            status=IndexStatus(row[2]),
            file_count=row[3],
            created_at=row[4],
            completed_at=row[5],
            topology_fingerprint=row[6],
        )

    def files(self, generation_id: int) -> tuple[DiscoveredFile, ...]:
        try:
            with closing(self._connect(read_only=True)) as connection:
                rows = connection.execute(
                    """
                    SELECT path, content_hash, language, size, modified_ns
                    FROM files WHERE generation_id = ? ORDER BY path
                    """,
                    (generation_id,),
                ).fetchall()
        except sqlite3.DatabaseError as e:
            raise RepositoryIndexCorruptError(
                f"Repository index storage is corrupt or unreadable: {e}"
            ) from e
        return tuple(
            DiscoveredFile(
                path=row[0],
                content_hash=row[1],
                language=row[2],
                size=row[3],
                modified_ns=row[4],
            )
            for row in rows
        )

    def chunk_count(self, generation_id: int) -> int:
        try:
            with closing(self._connect(read_only=True)) as connection:
                return int(
                    connection.execute(
                        "SELECT COUNT(*) FROM chunks_fts WHERE generation_id = ?",
                        (generation_id,),
                    ).fetchone()[0]
                )
        except sqlite3.DatabaseError as exc:
            raise RepositoryIndexCorruptError(
                f"Repository index storage is corrupt or unreadable: {exc}"
            ) from exc

    def cached_facts(
        self, files: Sequence[DiscoveredFile]
    ) -> dict[tuple[str, str], FileFacts]:
        self.initialize()
        keys = {(file.content_hash, file.language) for file in files}
        if not keys:
            return {}
        try:
            with closing(self._connect(read_only=True)) as connection:
                rows = connection.execute(
                    "SELECT content_hash, language, payload FROM fact_cache"
                ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise RepositoryIndexCorruptError(
                f"Repository index fact cache is unreadable: {exc}"
            ) from exc
        return {
            (str(row[0]), str(row[1])): FileFacts.model_validate_json(row[2])
            for row in rows
            if (str(row[0]), str(row[1])) in keys
        }

    def graph_edges(self, generation_id: int) -> tuple[GraphEdge, ...]:
        try:
            with closing(self._connect(read_only=True)) as connection:
                rows = connection.execute(
                    """
                    SELECT source, target, kind, line, evidence, unresolved
                    FROM graph_edges WHERE generation_id = ?
                    ORDER BY source, kind, target, line
                    """,
                    (generation_id,),
                ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise RepositoryIndexCorruptError(
                f"Repository graph is unreadable: {exc}"
            ) from exc
        return tuple(
            GraphEdge(
                source=row[0],
                target=row[1],
                kind=GraphEdgeKind(row[2]),
                line=row[3],
                evidence=row[4],
                unresolved=bool(row[5]),
            )
            for row in rows
        )

    def file_scores(self, generation_id: int) -> tuple[FileGraphScore, ...]:
        try:
            with closing(self._connect(read_only=True)) as connection:
                rows = connection.execute(
                    """
                    SELECT path, importance, component FROM file_scores
                    WHERE generation_id = ? ORDER BY path
                    """,
                    (generation_id,),
                ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise RepositoryIndexCorruptError(
                f"Repository graph scores are unreadable: {exc}"
            ) from exc
        return tuple(
            FileGraphScore(path=row[0], importance=row[1], component=row[2])
            for row in rows
        )

    def compact_map(
        self, generation: IndexGeneration, *, max_files: int = 50
    ) -> RepositoryMap:
        if max_files < 1 or max_files > _MAX_MAP_FILES:
            raise ValueError("max_files must be between 1 and 200.")
        try:
            with closing(self._connect(read_only=True)) as connection:
                score_rows = connection.execute(
                    "SELECT path, importance, component FROM file_scores "
                    "WHERE generation_id = ? "
                    "ORDER BY importance DESC, path LIMIT ?",
                    (generation.id, max_files),
                ).fetchall()
                paths = [str(row[0]) for row in score_rows]
                symbols: dict[str, list[str]] = defaultdict(list)
                if paths:
                    placeholders = ",".join("?" for _ in paths)
                    symbol_rows = connection.execute(
                        "SELECT f.path, s.qualified_name FROM files AS f "
                        "JOIN symbol_cache AS s ON s.content_hash = f.content_hash "
                        "AND s.language = f.language "
                        f"WHERE f.generation_id = ? AND f.path IN ({placeholders}) "
                        "ORDER BY f.path, s.line_start LIMIT 1000",
                        (generation.id, *paths),
                    ).fetchall()
                    for path, symbol in symbol_rows:
                        if len(symbols[str(path)]) < _MAX_MAP_SYMBOLS_PER_FILE:
                            symbols[str(path)].append(str(symbol))
        except sqlite3.DatabaseError as exc:
            raise RepositoryIndexCorruptError(
                f"Repository map is unreadable: {exc}"
            ) from exc
        return RepositoryMap(
            generation=generation.id,
            entries=tuple(
                RepositoryMapEntry(
                    path=str(row[0]),
                    importance=float(row[1]),
                    component=int(row[2]),
                    symbols=tuple(symbols[str(row[0])]),
                )
                for row in score_rows
            ),
        )

    def search(
        self,
        generation: IndexGeneration,
        query: str,
        *,
        mode: RepositorySearchMode = RepositorySearchMode.AUTO,
        direction: RepositoryDependencyDirection = (
            RepositoryDependencyDirection.DEPENDENCIES
        ),
        path: str | None = None,
        max_results: int,
    ) -> RepositorySearchResult:
        if max_results < 1 or max_results > _MAX_SEARCH_RESULTS:
            raise ValueError("max_results must be between 1 and 100.")
        path_prefix = _normalize_path_filter(path)
        match_query = _fts_match_query(query, mode=mode)
        lexical_rows: list[tuple[Any, ...]] = []
        try:
            with closing(self._connect(read_only=True)) as connection:
                if mode in {
                    RepositorySearchMode.AUTO,
                    RepositorySearchMode.TEXT,
                    RepositorySearchMode.DEPENDENCY,
                }:
                    lexical_rows = connection.execute(
                        """
                        SELECT path, line_start, line_end, content,
                               bm25(chunks_fts)
                        FROM chunks_fts
                        WHERE chunks_fts MATCH ? AND generation_id = ?
                        ORDER BY bm25(chunks_fts), path, line_start
                        LIMIT ?
                        """,
                        (match_query, generation.id, _MAX_SEARCH_RESULTS),
                    ).fetchall()
                structural_rows = _structural_rows(
                    connection, generation.id, query, mode, _MAX_SEARCH_RESULTS
                )
                if mode is RepositorySearchMode.SYMBOL and not structural_rows:
                    lexical_rows = connection.execute(
                        """
                        SELECT path, line_start, line_end, content,
                               bm25(chunks_fts)
                        FROM chunks_fts
                        WHERE chunks_fts MATCH ? AND generation_id = ?
                        ORDER BY bm25(chunks_fts), path, line_start
                        LIMIT ?
                        """,
                        (match_query, generation.id, _MAX_SEARCH_RESULTS),
                    ).fetchall()
                seed_paths = _search_seed_paths(
                    mode=mode,
                    lexical_rows=lexical_rows,
                    structural_rows=structural_rows,
                )
                seed_paths.update(
                    str(row[0])
                    for row in connection.execute(
                        "SELECT path FROM files WHERE generation_id = ? "
                        "AND path LIKE ? LIMIT 20",
                        (generation.id, f"%{query}%"),
                    ).fetchall()
                )
                dependency_rows = (
                    _dependency_rows(
                        connection,
                        generation.id,
                        seed_paths,
                        direction=(
                            RepositoryDependencyDirection.BOTH
                            if mode is RepositorySearchMode.AUTO
                            else direction
                        ),
                        max_hops=2,
                        max_results=_MAX_SEARCH_RESULTS,
                    )
                    if mode
                    in {RepositorySearchMode.AUTO, RepositorySearchMode.DEPENDENCY}
                    else []
                )
                importance = {
                    str(row[0]): float(row[1])
                    for row in connection.execute(
                        "SELECT path, importance FROM file_scores "
                        "WHERE generation_id = ?",
                        (generation.id,),
                    ).fetchall()
                }
                search_context = _search_context(
                    connection,
                    generation.id,
                    mode=mode,
                    direction=(
                        RepositoryDependencyDirection.BOTH
                        if mode is RepositorySearchMode.AUTO
                        else direction
                    ),
                    query=query,
                    path_prefix=path_prefix,
                    importance=importance,
                    seed_paths=seed_paths,
                    root_rows=(*lexical_rows, *structural_rows),
                    rows=(*lexical_rows, *structural_rows, *dependency_rows),
                )
        except sqlite3.DatabaseError as exc:
            raise RepositoryIndexCorruptError(
                f"Repository index search failed: {exc}"
            ) from exc

        lexical_matches = tuple(
            RepositorySearchMatch(
                root=generation.root,
                path=str(row[0]),
                line_start=int(row[1]),
                line_end=int(row[2]),
                snippet=str(row[3]),
                score_reason=_score_reason(
                    path=str(row[0]), content=str(row[3]), query=query.casefold()
                ),
                generation=generation.id,
            )
            for row in lexical_rows
            if mode is not RepositorySearchMode.DEPENDENCY
        )
        structural_matches = tuple(
            RepositorySearchMatch(
                root=generation.root,
                path=str(row[0]),
                line_start=int(row[1]),
                line_end=int(row[2]),
                symbol=str(row[3]),
                snippet=str(row[3]),
                relationship=str(row[4]),
                score_reason=str(row[5]),
                generation=generation.id,
            )
            for row in structural_rows
        )
        dependency_matches = tuple(
            RepositorySearchMatch(
                root=generation.root,
                path=str(row[0]),
                line_start=int(row[1]),
                line_end=int(row[2]),
                snippet=str(row[3]),
                relationship=str(row[4]),
                score_reason=str(row[5]),
                dependency_direction=RepositoryDependencyDirection(str(row[6])),
                generation=generation.id,
            )
            for row in dependency_rows
        )
        candidates = tuple(
            match
            for match in _search_candidates(
                mode=mode,
                structural_matches=structural_matches,
                lexical_matches=lexical_matches,
                dependency_matches=dependency_matches,
            )
            if _matches_path_filter(match.path, path_prefix)
        )
        matches, groups = group_repository_matches(
            rank_repository_matches(
                add_module_boundaries(
                    candidates, incoming_components=search_context.incoming_components
                ),
                query=query,
                importance=importance,
                max_results=max_results,
            )
        )
        return RepositorySearchResult(
            generation=generation,
            query=query,
            mode=mode,
            dependency_direction=(
                RepositoryDependencyDirection.BOTH
                if mode is RepositorySearchMode.AUTO
                else direction
                if mode is RepositorySearchMode.DEPENDENCY
                else None
            ),
            total_matches=len({
                (match.path, match.line_start, match.relationship)
                for match in candidates
            }),
            matches=matches,
            groups=groups,
            suggestions=suggest_repository_queries(
                query=query, mode=mode, matches=matches, groups=groups
            ),
            dependency_trees=search_context.dependency_trees,
            structural_coverage=generation.structural_file_count > 0,
        )

    def prune_generations(
        self, root: Path, *, keep: int, protected: frozenset[int] = frozenset()
    ) -> tuple[int, ...]:
        if keep < 1:
            raise ValueError("At least one complete generation must be retained.")
        try:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                rows = connection.execute(
                    """
                    SELECT g.id
                    FROM generations AS g
                    JOIN roots AS r ON r.id = g.root_id
                    WHERE r.path = ? AND g.status = ?
                    ORDER BY g.id DESC
                    """,
                    (str(root.resolve()), IndexStatus.COMPLETE.value),
                ).fetchall()
                retained = {int(row[0]) for row in rows[:keep]} | set(protected)
                removed = tuple(
                    int(row[0]) for row in rows if int(row[0]) not in retained
                )
                connection.executemany(
                    "DELETE FROM imports_index WHERE generation_id = ?",
                    ((generation_id,) for generation_id in removed),
                )
                connection.executemany(
                    "DELETE FROM references_index WHERE generation_id = ?",
                    ((generation_id,) for generation_id in removed),
                )
                connection.executemany(
                    "DELETE FROM symbols WHERE generation_id = ?",
                    ((generation_id,) for generation_id in removed),
                )
                connection.executemany(
                    "DELETE FROM file_scores WHERE generation_id = ?",
                    ((generation_id,) for generation_id in removed),
                )
                connection.executemany(
                    "DELETE FROM graph_edges WHERE generation_id = ?",
                    ((generation_id,) for generation_id in removed),
                )
                connection.executemany(
                    "DELETE FROM generation_facts WHERE generation_id = ?",
                    ((generation_id,) for generation_id in removed),
                )
                connection.executemany(
                    "DELETE FROM chunks_fts WHERE generation_id = ?",
                    ((generation_id,) for generation_id in removed),
                )
                connection.executemany(
                    "DELETE FROM files WHERE generation_id = ?",
                    ((generation_id,) for generation_id in removed),
                )
                connection.executemany(
                    "DELETE FROM generations WHERE id = ?",
                    ((generation_id,) for generation_id in removed),
                )
                connection.commit()
        except sqlite3.DatabaseError as e:
            raise RepositoryIndexCorruptError(
                f"Failed to prune repository index generations: {e}"
            ) from e
        return removed

    def clear(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            target = Path(f"{self.path}{suffix}")
            if target.exists():
                target.unlink()

    def _connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        if read_only:
            connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        else:
            connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        if not read_only:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
        return connection


def _copy_unchanged_analysis(
    connection: sqlite3.Connection, base_generation_id: int, generation_id: int
) -> None:
    connection.execute(
        """
        INSERT INTO chunks_fts(
            generation_id, path, line_start, line_end, content
        )
        SELECT ?, source.path, source.line_start, source.line_end, source.content
        FROM chunks_fts AS source
        JOIN files AS old_file
          ON old_file.generation_id = source.generation_id
         AND old_file.path = source.path
        JOIN files AS new_file
          ON new_file.generation_id = ? AND new_file.path = source.path
         AND new_file.content_hash = old_file.content_hash
        WHERE source.generation_id = ?
        """,
        (generation_id, generation_id, base_generation_id),
    )


def _fts_match_query(query: str, *, mode: RepositorySearchMode) -> str:
    terms = re.findall(r"[^\W]+", query, flags=re.UNICODE)
    if not terms:
        raise ValueError("Repository search query must contain searchable text.")
    if mode is not RepositorySearchMode.TEXT:
        expanded = tuple(dict.fromkeys((*terms, *repository_query_concepts(query))))
        return " OR ".join(f'"{term}"' for term in expanded)

    clauses: list[str] = []
    for match in re.finditer(r'"([^"]+)"|([^\s"]+)', query):
        segment = match.group(1) or match.group(2)
        segment_terms = re.findall(r"[^\W]+", segment, flags=re.UNICODE)
        if not segment_terms:
            continue
        if match.group(1) is not None:
            clauses.append(f'"{" ".join(segment_terms)}"')
        else:
            clauses.extend(f'"{term}"' for term in segment_terms)
    return " AND ".join(clauses)


def _normalize_path_filter(path: str | None) -> str | None:
    if path is None or not path.strip():
        return None
    raw = path.strip()
    if raw.startswith("/") or PureWindowsPath(raw).is_absolute():
        raise ValueError("Repository search path must be repository-relative.")
    normalized = raw.replace("\\", "/").strip("/")
    candidate = PurePosixPath(normalized)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Repository search path must be repository-relative.")
    return candidate.as_posix()


def _matches_path_filter(path: str, prefix: str | None) -> bool:
    return prefix is None or path == prefix or path.startswith(f"{prefix}/")


def _search_seed_paths(
    *,
    mode: RepositorySearchMode,
    lexical_rows: Sequence[tuple[Any, ...]],
    structural_rows: Sequence[tuple[Any, ...]],
) -> set[str]:
    if mode is RepositorySearchMode.DEPENDENCY and structural_rows:
        definitions = {
            str(row[0]) for row in structural_rows if str(row[4]) == "defines"
        }
        return definitions or {str(row[0]) for row in structural_rows}
    return {str(row[0]) for row in (*lexical_rows, *structural_rows)}


def _search_candidates(
    *,
    mode: RepositorySearchMode,
    structural_matches: Sequence[RepositorySearchMatch],
    lexical_matches: Sequence[RepositorySearchMatch],
    dependency_matches: Sequence[RepositorySearchMatch],
) -> tuple[RepositorySearchMatch, ...]:
    if mode is RepositorySearchMode.DEPENDENCY:
        definitions = tuple(
            match for match in structural_matches if match.relationship == "defines"
        )
        return (*definitions, *dependency_matches)
    return (*structural_matches, *lexical_matches, *dependency_matches)


def _incoming_components(
    connection: sqlite3.Connection, generation_id: int, paths: set[str]
) -> dict[str, frozenset[str]]:
    if not paths:
        return {}
    placeholders = ",".join("?" for _ in paths)
    rows = connection.execute(
        "SELECT source, target FROM graph_edges "
        "WHERE generation_id = ? AND kind = 'imports' "
        f"AND target IN ({placeholders})",
        (generation_id, *sorted(paths)),
    ).fetchall()
    return incoming_component_map([
        (str(source), str(target)) for source, target in rows
    ])


def _search_context(
    connection: sqlite3.Connection,
    generation_id: int,
    *,
    mode: RepositorySearchMode,
    direction: RepositoryDependencyDirection,
    query: str,
    path_prefix: str | None,
    importance: dict[str, float],
    seed_paths: set[str],
    root_rows: Sequence[tuple[Any, ...]],
    rows: Sequence[tuple[Any, ...]],
) -> _SearchContext:
    paths = {str(row[0]) for row in rows}
    incoming = _incoming_components(connection, generation_id, paths)
    if mode not in {RepositorySearchMode.AUTO, RepositorySearchMode.DEPENDENCY}:
        return _SearchContext(incoming_components=incoming, dependency_trees=())
    if mode is RepositorySearchMode.DEPENDENCY:
        roots = tuple(
            sorted(seed_paths, key=lambda path: _tree_root_key(path, importance))
        )[:_MAX_DEPENDENCY_TREE_ROOTS]
    else:
        roots = _auto_tree_roots(
            root_rows, query=query, path_prefix=path_prefix, importance=importance
        )
    return _SearchContext(
        incoming_components=incoming,
        dependency_trees=_dependency_trees(
            connection, generation_id, roots, direction=direction
        ),
    )


def _dependency_trees(
    connection: sqlite3.Connection,
    generation_id: int,
    roots: Sequence[str],
    *,
    direction: RepositoryDependencyDirection,
) -> tuple[RepositoryDependencyTree, ...]:
    adjacency: dict[str, list[tuple[str, str, RepositoryDependencyDirection]]] = (
        defaultdict(list)
    )
    frontier = set(roots)
    for _ in range(_MAX_DEPENDENCY_TREE_DEPTH):
        if not frontier:
            break
        next_frontier: set[str] = set()
        for current, neighbor, kind, edge_direction in _tree_edges(
            connection, generation_id, frontier, direction=direction
        ):
            if current == neighbor:
                continue
            edge = (neighbor, kind, edge_direction)
            if edge not in adjacency[current]:
                adjacency[current].append(edge)
            next_frontier.add(neighbor)
        frontier = next_frontier
    return tuple(
        _render_dependency_tree(root, adjacency, direction=direction) for root in roots
    )


def _tree_edges(
    connection: sqlite3.Connection,
    generation_id: int,
    frontier: set[str],
    *,
    direction: RepositoryDependencyDirection,
) -> tuple[tuple[str, str, str, RepositoryDependencyDirection], ...]:
    paths = sorted(frontier)
    placeholders = ",".join("?" for _ in paths)
    edges: set[tuple[str, str, str, RepositoryDependencyDirection]] = set()
    if direction in {
        RepositoryDependencyDirection.DEPENDENCIES,
        RepositoryDependencyDirection.BOTH,
    }:
        rows = connection.execute(
            "SELECT source, target, kind FROM graph_edges "
            "WHERE generation_id = ? AND target IS NOT NULL "
            "AND kind IN ('imports', 'inherits') "
            f"AND source IN ({placeholders}) ORDER BY source, target, kind",
            (generation_id, *paths),
        ).fetchall()
        edges.update(
            (
                str(source),
                _graph_target_path(str(target)),
                str(kind),
                RepositoryDependencyDirection.DEPENDENCIES,
            )
            for source, target, kind in rows
        )
    if direction in {
        RepositoryDependencyDirection.DEPENDENTS,
        RepositoryDependencyDirection.BOTH,
    }:
        symbol_conditions = " OR ".join("target LIKE ? ESCAPE '\\'" for _ in paths)
        rows = connection.execute(
            "SELECT source, target, kind FROM graph_edges "
            "WHERE generation_id = ? AND target IS NOT NULL "
            "AND kind IN ('imports', 'inherits') AND "
            f"(target IN ({placeholders}) OR {symbol_conditions}) "
            "ORDER BY source, target, kind",
            (
                generation_id,
                *paths,
                *(f"symbol:{_escape_like(path)}#%" for path in paths),
            ),
        ).fetchall()
        edges.update(
            (
                _graph_target_path(str(target)),
                str(source),
                str(kind),
                RepositoryDependencyDirection.DEPENDENTS,
            )
            for source, target, kind in rows
        )
    return tuple(sorted(edges))


def _graph_target_path(target: str) -> str:
    if target.startswith("symbol:"):
        return target.removeprefix("symbol:").partition("#")[0]
    return target


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _tree_root_key(path: str, importance: dict[str, float]) -> tuple[bool, float, str]:
    return _is_test_path(path), -importance.get(path, 0.0), path


def _auto_tree_roots(
    rows: Sequence[tuple[Any, ...]],
    *,
    query: str,
    path_prefix: str | None,
    importance: dict[str, float],
) -> tuple[str, ...]:
    evidence: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        path = str(row[0])
        if _matches_path_filter(path, path_prefix):
            evidence[path].append(str(row[3]))
    concepts = repository_query_concepts(query)

    def root_key(path: str) -> tuple[int, bool, float, str]:
        combined = " ".join((path, *evidence[path])).casefold()
        coverage = sum(concept in combined for concept in concepts)
        return -coverage, _is_test_path(path), -importance.get(path, 0.0), path

    return tuple(sorted(evidence, key=root_key))[:_MAX_DEPENDENCY_TREE_ROOTS]


def _is_test_path(path: str) -> bool:
    return any(
        part == "tests" or part.startswith("test_")
        for part in PurePosixPath(path).parts
    )


def _render_dependency_tree(
    root: str,
    adjacency: dict[str, list[tuple[str, str, RepositoryDependencyDirection]]],
    *,
    direction: RepositoryDependencyDirection,
) -> RepositoryDependencyTree:
    lines = [root]
    visited = {root}
    node_count = 1
    truncated = False

    def append_children(path: str, depth: int, prefix: str) -> None:
        nonlocal node_count, truncated
        children = adjacency.get(path, ())
        for position, (child, kind, edge_direction) in enumerate(children):
            if node_count >= _MAX_DEPENDENCY_TREE_NODES:
                lines.append(f"{prefix}└── … truncated")
                truncated = True
                return
            last = position == len(children) - 1
            connector = "└──" if last else "├──"
            cycle = child in visited
            suffix = " [cycle]" if cycle else ""
            edge = (
                f"{kind} →"
                if edge_direction is RepositoryDependencyDirection.DEPENDENCIES
                else f"← {kind}"
            )
            lines.append(f"{prefix}{connector} [{edge}] {child}{suffix}")
            node_count += 1
            if cycle or depth == _MAX_DEPENDENCY_TREE_DEPTH:
                continue
            visited.add(child)
            continuation = "    " if last else "│   "
            append_children(child, depth + 1, prefix + continuation)
            if truncated:
                return

    append_children(root, 1, "")
    return RepositoryDependencyTree(
        root=root,
        direction=direction,
        lines=tuple(lines),
        node_count=node_count,
        truncated=truncated,
    )


def _score_reason(*, path: str, content: str, query: str) -> str:
    reasons = ["lexical chunk match"]
    if query in path.casefold():
        reasons.append("path match")
    if query in content.casefold():
        reasons.append("exact phrase")
    return ", ".join(reasons)


def _dependency_rows(  # noqa: PLR0914
    connection: sqlite3.Connection,
    generation_id: int,
    seed_paths: set[str],
    *,
    direction: RepositoryDependencyDirection,
    max_hops: int,
    max_results: int,
) -> list[tuple[Any, ...]]:
    file_paths = {
        str(row[0])
        for row in connection.execute(
            "SELECT path FROM files WHERE generation_id = ?", (generation_id,)
        ).fetchall()
    }
    seeds = seed_paths & file_paths
    if not seeds:
        return []

    adjacency: dict[str, set[tuple[str, str, RepositoryDependencyDirection]]] = (
        defaultdict(set)
    )
    rows = connection.execute(
        "SELECT source, target, kind FROM graph_edges "
        "WHERE generation_id = ? AND target IS NOT NULL "
        "AND kind IN ('imports', 'references', 'calls', 'inherits')",
        (generation_id,),
    ).fetchall()
    for source_value, target_value, kind_value in rows:
        source = str(source_value)
        target = str(target_value)
        if target.startswith("symbol:"):
            target = target.removeprefix("symbol:").partition("#")[0]
        if source not in file_paths or target not in file_paths or source == target:
            continue
        kind = str(kind_value)
        if direction in {
            RepositoryDependencyDirection.DEPENDENCIES,
            RepositoryDependencyDirection.BOTH,
        }:
            adjacency[source].add((
                target,
                kind,
                RepositoryDependencyDirection.DEPENDENCIES,
            ))
        if direction in {
            RepositoryDependencyDirection.DEPENDENTS,
            RepositoryDependencyDirection.BOTH,
        }:
            adjacency[target].add((
                source,
                kind,
                RepositoryDependencyDirection.DEPENDENTS,
            ))

    queue = deque((seed, 0) for seed in sorted(seeds))
    visited = set(seeds)
    neighbors: list[tuple[str, int, str, RepositoryDependencyDirection]] = []
    while queue and len(neighbors) < max_results:
        current, depth = queue.popleft()
        if depth == max_hops:
            continue
        for neighbor, kind, edge_direction in sorted(adjacency.get(current, ())):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            distance = depth + 1
            neighbors.append((neighbor, distance, kind, edge_direction))
            queue.append((neighbor, distance))
            if len(neighbors) == max_results:
                break

    paths = [path for path, _, _, _ in neighbors]
    chunks_by_path: dict[str, tuple[Any, ...]] = {}
    if paths:
        placeholders = ",".join("?" for _ in paths)
        chunk_rows = connection.execute(
            "SELECT path, line_start, line_end, content FROM chunks_fts "
            f"WHERE generation_id = ? AND path IN ({placeholders}) "
            "ORDER BY path, line_start",
            (generation_id, *paths),
        ).fetchall()
        for row in chunk_rows:
            chunks_by_path.setdefault(str(row[0]), tuple(row[1:]))

    result: list[tuple[Any, ...]] = []
    for path, distance, kind, edge_direction in neighbors:
        line_start, line_end, content = chunks_by_path.get(path, (1, 1, path))
        result.append((
            path,
            line_start,
            line_end,
            content,
            f"dependency_distance_{distance}_{edge_direction.value}_via_{kind}",
            (
                f"{edge_direction.value} via {kind} "
                f"({distance} hop{'s' if distance > 1 else ''})"
            ),
            edge_direction.value,
        ))
    return result


def _structural_rows(
    connection: sqlite3.Connection,
    generation_id: int,
    query: str,
    mode: RepositorySearchMode,
    max_results: int,
) -> list[tuple[Any, ...]]:
    if mode is RepositorySearchMode.TEXT:
        return []
    patterns = _structural_patterns(query)
    symbol_predicate = " OR ".join(
        "(s.name LIKE ? OR s.qualified_name LIKE ?)" for _ in patterns
    )
    symbol_params = tuple(value for pattern in patterns for value in (pattern, pattern))
    definition_rows = connection.execute(
        f"""
        SELECT f.path, s.line_start, s.line_end,
               s.qualified_name,
               'defines', 'Python AST symbol definition'
        FROM files AS f
        JOIN symbol_cache AS s
          ON s.content_hash = f.content_hash AND s.language = f.language
        WHERE f.generation_id = ?
          AND ({symbol_predicate})
        ORDER BY CASE WHEN s.name = ? THEN 0 ELSE 1 END, f.path, s.line_start
        LIMIT ?
        """,
        (generation_id, *symbol_params, query, max_results),
    ).fetchall()
    if mode is not RepositorySearchMode.IMPACT:
        reference_predicate = " OR ".join("r.name LIKE ?" for _ in patterns)
        reference_rows = connection.execute(
            f"""
            SELECT f.path, r.line, r.line,
                   r.name,
                   r.kind, 'Python AST reference'
            FROM files AS f
            JOIN reference_cache AS r
              ON r.content_hash = f.content_hash AND r.language = f.language
            WHERE f.generation_id = ? AND ({reference_predicate})
            ORDER BY f.path, r.line
            LIMIT ?
            """,
            (generation_id, *patterns, max_results),
        ).fetchall()
        return [*definition_rows, *reference_rows][:max_results]

    pattern = f"%{query}%"
    definition_rows = connection.execute(
        """
        SELECT f.path, s.line_start, s.line_end,
               s.qualified_name,
               'defines', 'exact Python AST symbol definition'
        FROM files AS f
        JOIN symbol_cache AS s
          ON s.content_hash = f.content_hash AND s.language = f.language
        WHERE f.generation_id = ? AND (s.name = ? OR s.qualified_name = ?)
        ORDER BY f.path, s.line_start
        LIMIT ?
        """,
        (generation_id, query, query, _MAX_IMPACT_DEFINITIONS),
    ).fetchall()

    definition_targets: dict[str, tuple[str, ...]] = {}
    for row in definition_rows:
        target_path = str(row[0])
        target = f"symbol:{target_path}#{row[3]}"
        definition_targets[target_path] = (
            *definition_targets.get(target_path, ()),
            target,
        )
    target_paths = set(definition_targets)
    target_paths.update(
        str(row[0])
        for row in connection.execute(
            "SELECT path FROM files WHERE generation_id = ? AND path LIKE ?",
            (generation_id, pattern),
        ).fetchall()
    )
    if not target_paths:
        return definition_rows[:max_results]
    impact_rows: list[tuple[Any, ...]] = list(definition_rows)
    for target_path in sorted(target_paths):
        symbol_targets = definition_targets.get(target_path, ())
        if symbol_targets:
            placeholders = ",".join("?" for _ in symbol_targets)
            dependency_predicate = (
                "((e.kind = 'imports' AND e.target = ? "
                "AND e.evidence LIKE ? ESCAPE '\\') "
                "OR (e.kind IN ('references', 'calls', 'inherits') "
                f"AND e.target IN ({placeholders})))"
            )
            dependency_params = (
                target_path,
                f"%{_escape_like(query.rsplit('.', maxsplit=1)[-1])}%",
                *symbol_targets,
            )
        else:
            dependency_predicate = (
                "(e.target = ? OR e.target LIKE ?) "
                "AND e.kind IN ('imports', 'references', 'calls', 'inherits')"
            )
            dependency_params = (target_path, f"symbol:{target_path}#%")
        impact_rows.extend(
            connection.execute(
                f"""
                SELECT e.source, COALESCE(e.line, 1), COALESCE(e.line, 1),
                       COALESCE(e.evidence, e.source),
                       'dependent_via_' || e.kind,
                       'direct dependency graph edge'
                FROM graph_edges AS e
                WHERE e.generation_id = ? AND e.source != ?
                  AND {dependency_predicate}
                ORDER BY e.source, e.line
                LIMIT ?
                """,
                (generation_id, target_path, *dependency_params, max_results),
            ).fetchall()
        )
        impact_rows.extend(
            connection.execute(
                """
                SELECT e.target, 1, 1, e.target,
                       'tested_by', 'test relationship from dependency graph'
                FROM graph_edges AS e
                WHERE e.generation_id = ? AND e.source = ?
                  AND e.kind = 'tested_by' AND e.target IS NOT NULL
                ORDER BY e.target
                LIMIT ?
                """,
                (generation_id, target_path, max_results),
            ).fetchall()
        )
    return impact_rows[:max_results]


def _structural_patterns(query: str) -> tuple[str, ...]:
    terms = re.findall(r"[^\W]+", query, flags=re.UNICODE)
    return tuple(f"%{term}%" for term in dict.fromkeys(terms))
