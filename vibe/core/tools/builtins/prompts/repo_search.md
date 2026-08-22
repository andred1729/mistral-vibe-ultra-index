Search the complete repository index pinned to the current model inference.

Use `auto` as a compact locator for likely files and symbols, not as an exhaustive repository map. Once you find a plausible function or symbol, use `impact` to find direct callers or consumers, related tests, and what an edit could break. Use `dependency` only around a likely locus: `direction=dependencies` shows what it calls or requires, `dependents` shows incoming callers and consumers, and `both` gives a bounded two-way neighborhood when both directions are needed. Do not request broad dependency trees before you know the likely locus.

Use `text` for lexical evidence and `symbol` for definitions and references. In `text` mode, all unquoted terms are required; wrap adjacent terms in double quotes to require an exact phrase. Use `path` to restrict results to a repository-relative file or directory. Results cite the repository root, relative path, line range, matching snippet, score reason, relationship, direction, and exact index generation.

Structural modes transparently fall back to language-neutral evidence when a parser or relationship is unavailable. Treat missing structural results as uncertainty, not proof that no dependency exists. Use `grep` for exact regex matching and `read_file` before editing a result.
