Search the complete repository index pinned to the current model inference.

Use `auto` for general code discovery, `text` for lexical evidence, `symbol` for definitions and references, `dependency` for the local dependency neighborhood, and `impact` before dependency-relevant edits to find dependents and related tests. Use `path` to restrict results to a repository-relative file or directory. Results cite the repository root, relative path, line range, matching snippet, score reason, relationship, and exact index generation.

Structural modes transparently fall back to language-neutral evidence when a parser or relationship is unavailable. Treat missing structural results as uncertainty, not proof that no dependency exists. Use `grep` for exact regex matching and `read_file` before editing a result.
