from __future__ import annotations

import ast

from vibe.core.repository_index.models import (
    FileFacts,
    ImportFact,
    ReferenceFact,
    ReferenceKind,
    SymbolFact,
    SymbolKind,
)

PYTHON_PARSER_VERSION = 1


class _FactVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: list[str] = []
        self.symbols: list[SymbolFact] = []
        self.imports: list[ImportFact] = []
        self.references: list[ReferenceFact] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add_symbol(node, SymbolKind.CLASS)
        for base in node.bases:
            if name := _expression_name(base):
                self.references.append(
                    ReferenceFact(
                        name=name, kind=ReferenceKind.INHERITS, line=node.lineno
                    )
                )
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        kind = SymbolKind.METHOD if self.scope else SymbolKind.FUNCTION
        self._add_symbol(node, kind)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        kind = SymbolKind.METHOD if self.scope else SymbolKind.FUNCTION
        self._add_symbol(node, kind)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        if not self.scope:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.symbols.append(
                        SymbolFact(
                            name=target.id,
                            qualified_name=target.id,
                            kind=SymbolKind.VARIABLE,
                            line_start=node.lineno,
                            line_end=node.end_lineno or node.lineno,
                        )
                    )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                ImportFact(module=alias.name, alias=alias.asname, line=node.lineno)
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.imports.append(
                ImportFact(
                    module=node.module or "",
                    imported_name=alias.name,
                    alias=alias.asname,
                    level=node.level,
                    line=node.lineno,
                )
            )

    def visit_Call(self, node: ast.Call) -> None:
        if name := _expression_name(node.func):
            self.references.append(
                ReferenceFact(name=name, kind=ReferenceKind.CALL, line=node.lineno)
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.references.append(
                ReferenceFact(
                    name=node.id, kind=ReferenceKind.REFERENCE, line=node.lineno
                )
            )

    def _add_symbol(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        kind: SymbolKind,
    ) -> None:
        qualified = ".".join([*self.scope, node.name])
        self.symbols.append(
            SymbolFact(
                name=node.name,
                qualified_name=qualified,
                kind=kind,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
            )
        )


def parse_python_facts(path: str, content_hash: str, source: str) -> FileFacts:
    visitor = _FactVisitor()
    try:
        visitor.visit(ast.parse(source, filename=path))
        error = None
    except (SyntaxError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return FileFacts(
        path=path,
        content_hash=content_hash,
        language="python",
        parser_version=PYTHON_PARSER_VERSION,
        symbols=tuple(visitor.symbols),
        imports=tuple(visitor.imports),
        references=tuple(visitor.references),
        parse_error=error,
    )


def _expression_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expression_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Subscript):
        return _expression_name(node.value)
    return None
