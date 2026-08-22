from __future__ import annotations

from collections.abc import AsyncGenerator

from pydantic import BaseModel, Field

from vibe.core.repository_index.models import (
    RepositoryDependencyDirection,
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
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolStreamEvent


class RepoSearchArgs(BaseModel):
    query: str = Field(
        min_length=1,
        description=(
            "Concept, behavior, symbol, or code text to locate. In auto mode, use "
            "a compact task or symbol query to find a likely locus."
        ),
    )
    mode: RepositorySearchMode = Field(
        default=RepositorySearchMode.AUTO,
        description=(
            "auto is a compact locator for likely files and symbols; text finds "
            "lexical evidence; symbol finds definitions/references; impact finds "
            "direct callers or consumers, related tests, and what could break; "
            "dependency traverses the graph around a known likely locus."
        ),
    )
    direction: RepositoryDependencyDirection = Field(
        default=RepositoryDependencyDirection.DEPENDENCIES,
        description=(
            "Direction used by dependency mode: dependencies shows what the locus "
            "calls or requires; dependents shows incoming callers and consumers; "
            "both shows a bounded two-way neighborhood."
        ),
    )
    path: str | None = Field(
        default=None,
        description="Optional repository-relative path or directory prefix.",
    )
    limit: int = Field(default=20, ge=1, le=100)


class RepoSearchConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class RepoSearch(
    BaseTool[RepoSearchArgs, RepositorySearchResult, RepoSearchConfig, BaseToolState],
    ToolUIData[RepoSearchArgs, RepositorySearchResult],
):
    @classmethod
    def format_call_display(cls, args: RepoSearchArgs) -> ToolCallDisplay:
        return ToolCallDisplay(
            summary="Repo_Search",
            verb="Searching",
            message="Repo_Search",
            settled_verb="Searched",
            settled_message="Repo_Search",
        )

    @classmethod
    def format_result_display(cls, result: RepositorySearchResult) -> ToolResultDisplay:
        returned = len(result.matches)
        return ToolResultDisplay(
            success=True,
            message="Repo_Search",
            suffix=f"{returned} of {result.total_matches} matches",
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Searching repository"

    async def run(
        self, args: RepoSearchArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | RepositorySearchResult, None]:
        if self.repository_index is None:
            raise ToolError("The repository index is unavailable for this session.")
        result = await self.repository_index.search(
            args.query,
            mode=args.mode,
            direction=args.direction,
            path=args.path,
            max_results=args.limit,
        )
        yield result
