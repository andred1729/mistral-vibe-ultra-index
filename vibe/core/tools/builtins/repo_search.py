from __future__ import annotations

from collections.abc import AsyncGenerator

from pydantic import BaseModel, Field

from vibe.core.repository_index.models import (
    RepositorySearchMode,
    RepositorySearchResult,
)
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.types import ToolStreamEvent


class RepoSearchArgs(BaseModel):
    query: str = Field(
        min_length=1,
        description="Concept, symbol, behavior, dependency, or code text to find.",
    )
    mode: RepositorySearchMode = Field(
        default=RepositorySearchMode.AUTO,
        description=(
            "auto for general retrieval, text for lexical evidence, symbol for "
            "definitions/references, or impact for dependents and related tests."
        ),
    )
    max_results: int = Field(default=20, ge=1, le=100)


class RepoSearchConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class RepoSearch(
    BaseTool[RepoSearchArgs, RepositorySearchResult, RepoSearchConfig, BaseToolState]
):
    async def run(
        self, args: RepoSearchArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | RepositorySearchResult, None]:
        if self.repository_index is None:
            raise ToolError("The repository index is unavailable for this session.")
        result = await self.repository_index.search(
            args.query, mode=args.mode, max_results=args.max_results
        )
        yield result
