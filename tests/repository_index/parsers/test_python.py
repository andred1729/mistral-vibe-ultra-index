from __future__ import annotations

from vibe.core.repository_index.models import ReferenceKind, SymbolKind
from vibe.core.repository_index.parsers.python import parse_python_facts


def test_extracts_python_symbols_imports_calls_and_inheritance() -> None:
    facts = parse_python_facts(
        "pkg/service.py",
        "hash",
        """from .base import BaseService
from pkg.helpers import build as make

SETTING = 1

class Service(BaseService):
    async def run(self):
        return make()
""",
    )

    assert [(symbol.qualified_name, symbol.kind) for symbol in facts.symbols] == [
        ("SETTING", SymbolKind.VARIABLE),
        ("Service", SymbolKind.CLASS),
        ("Service.run", SymbolKind.METHOD),
    ]
    assert [
        (item.module, item.imported_name, item.level) for item in facts.imports
    ] == [("base", "BaseService", 1), ("pkg.helpers", "build", 0)]
    assert any(
        reference.name == "BaseService" and reference.kind is ReferenceKind.INHERITS
        for reference in facts.references
    )
    assert any(
        reference.name == "make" and reference.kind is ReferenceKind.CALL
        for reference in facts.references
    )


def test_syntax_error_is_degraded_file_evidence_instead_of_build_failure() -> None:
    facts = parse_python_facts("broken.py", "hash", "def broken(:\n")

    assert facts.parse_error is not None
    assert facts.symbols == ()
