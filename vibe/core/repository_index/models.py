from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class IndexStatus(StrEnum):
    BUILDING = auto()
    COMPLETE = auto()
    CANCELLED = auto()
    FAILED = auto()


class IndexPhase(StrEnum):
    IDLE = auto()
    DISCOVERING = auto()
    CHUNKING = auto()
    PARSING = auto()
    GRAPH = auto()
    PUBLISHING = auto()
    CLEARING = auto()


class DiscoveredFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    content_hash: str
    language: str
    size: int
    modified_ns: int


class DiscoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    root: Path
    files: tuple[DiscoveredFile, ...]
    skipped_binary: int = 0
    skipped_oversized: int = 0
    skipped_sensitive: int = 0
    skipped_symlink: int = 0


class IndexChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    ordinal: int
    line_start: int
    line_end: int
    content: str


class SymbolKind(StrEnum):
    CLASS = auto()
    FUNCTION = auto()
    METHOD = auto()
    VARIABLE = auto()


class ReferenceKind(StrEnum):
    REFERENCE = auto()
    CALL = auto()
    INHERITS = auto()


class GraphEdgeKind(StrEnum):
    CONTAINS = auto()
    IMPORTS = auto()
    DEFINES = auto()
    REFERENCES = auto()
    CALLS = auto()
    INHERITS = auto()
    TESTED_BY = auto()


class RepositorySearchMode(StrEnum):
    AUTO = auto()
    TEXT = auto()
    SYMBOL = auto()
    IMPACT = auto()


class SymbolFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    qualified_name: str
    kind: SymbolKind
    line_start: int
    line_end: int


class ImportFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    module: str
    imported_name: str | None = None
    alias: str | None = None
    level: int = 0
    line: int


class ReferenceFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: ReferenceKind
    line: int


class FileFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    content_hash: str
    language: str
    parser_version: int
    symbols: tuple[SymbolFact, ...] = ()
    imports: tuple[ImportFact, ...] = ()
    references: tuple[ReferenceFact, ...] = ()
    parse_error: str | None = None


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    target: str | None
    kind: GraphEdgeKind
    line: int | None = None
    evidence: str | None = None
    unresolved: bool = False


class FileGraphScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    importance: float
    component: int


class HydratedGraph(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    edges: tuple[GraphEdge, ...]
    scores: tuple[FileGraphScore, ...]
    topology_fingerprint: str


class IndexGeneration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    id: int
    root: Path
    status: IndexStatus
    file_count: int
    created_at: str
    completed_at: str | None = None
    topology_fingerprint: str | None = None


class RepositoryIndexState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    root: Path | None = None
    generation: IndexGeneration | None = None
    status: IndexStatus = IndexStatus.BUILDING
    phase: IndexPhase = IndexPhase.IDLE
    dirty: bool = True
    files_processed: int = 0
    files_total: int = 0
    error: str | None = None


class RepositorySearchMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    root: Path
    path: str
    line_start: int
    line_end: int
    snippet: str
    relationship: str = "text_match"
    score_reason: str
    generation: int


class RepositorySearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    generation: IndexGeneration
    query: str
    mode: RepositorySearchMode = RepositorySearchMode.AUTO
    total_matches: int
    matches: tuple[RepositorySearchMatch, ...]
    structural_coverage: bool = False
