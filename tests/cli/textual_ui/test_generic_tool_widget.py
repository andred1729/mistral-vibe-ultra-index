from __future__ import annotations

from textual.content import Content
from textual.widgets import Static

from vibe.app_server.models import (
    CompletedEffectState,
    EffectCallDisplay,
    EffectResultDisplay,
    GenericEffectDetail,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
)
from vibe.cli.textual_ui.widgets.tool_widgets import (
    GenericToolResultWidget,
    get_result_widget,
)
from vibe.cli.textual_ui.widgets.tools import ToolCallMessage, ToolResultMessage


def test_generic_tool_result_preserves_multiline_fields() -> None:
    detail = GenericEffectDetail(
        tool_name="skill",
        display=EffectCallDisplay(summary="Loading skill", status_text="Loading skill"),
    )
    widget = get_result_widget(
        detail,
        {
            "name": "review",
            "content": "first instruction\nsecond instruction\nthird instruction",
            "skillDir": "/tmp/review",
            "metadata": {"source": "local"},
        },
        True,
        "Loaded skill: review",
    )

    assert isinstance(widget, GenericToolResultWidget)
    detail_widget = next(
        child for child in widget.compose() if isinstance(child, Static)
    )
    rendered = detail_widget.render()
    assert isinstance(rendered, Content)
    assert rendered.plain == (
        "name: review\n"
        "content: first instruction\n"
        "second instruction\n"
        "third instruction\n"
        "skillDir: /tmp/review\n"
        "metadata: {'source': 'local'}"
    )


def test_repo_search_tool_messages_receive_purple_style_class() -> None:
    entry = PublicEffectEntry(
        id="call-1",
        session_id="session-1",
        turn_id="turn-1",
        created_at=1,
        updated_at=1,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        title="repo_search",
        detail=GenericEffectDetail(
            tool_name="repo_search",
            display=EffectCallDisplay(
                summary="Repo_Search",
                message="Repo_Search",
                status_text="Searching repository",
            ),
        ),
        state=CompletedEffectState(
            display=EffectResultDisplay(
                success=True, verb="Searched", message="Repo_Search"
            )
        ),
    )

    assert "repo-search" in ToolCallMessage(entry).classes
    assert "repo-search" in ToolResultMessage(entry).classes
