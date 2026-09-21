"""Call-edge extraction for the JavaScript/TypeScript family.

WHY THIS EXISTS
---------------
The Python side of the engine resolves call edges from a real AST, so
``callers`` and ``calls`` answer honestly for ``.py``. Every other language went
through the symbol adapter, which returns symbol SPANS and no resolved call
graph — so a ``.ts`` or ``.tsx`` symbol was indexed, retrievable, and carried
zero edges. On a repository whose front end is TypeScript that makes the one
question the tool exists to answer ("who calls this?") return nothing, with an
exit code and a message that read as "this symbol has no callers" rather than
"this language was never wired up".

A full TypeScript type-resolver is not the answer here: it would mean shipping a
compiler, and the edges it bought would still be approximate across a monorepo
of loosely-coupled packages. What this module does instead is deliberately
narrow and deliberately stated:

* strip comments, string literals and regex literals, so a name mentioned in
  prose or inside a message is never mistaken for a call;
* inside a template literal, only the ``${ ... }`` substitutions are code;
* match ``name(`` in what is left, keeping the final identifier of a member
  expression (``a.b.c(x)`` yields ``c``);
* drop reserved words that are legally followed by ``(`` (``if``, ``for``,
  ``catch``, ``await`` …), and drop a name introduced by ``function``, which is
  a definition rather than a call;
* drop the enclosing symbol's own name, so a definition is never reported as
  its own caller.

The result is a CANDIDATE set, in the same spirit as the Python extractor's own
docstring: a name that is called somewhere in this body. It does not resolve
which module the name came from, and it does not see calls made through a
value (``handlers[k]()``, ``obj["m"]()``). Reporting candidates is the correct
trade here — the alternative on offer was no edges at all.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set

#: Extensions this module knows how to read. Kept as a frozen set because both
#: the parser and the symbol-store builder must agree on "is this JS family?";
#: two copies of the list would drift.
JS_EXTENSIONS: frozenset = frozenset({
    ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts",
})

#: Words that may legally be followed by ``(`` without being a call. Missing one
#: costs a bogus edge from every file that uses it, which is why control flow,
#: the operators that take a parenthesised operand, and the declaration keywords
#: are all listed rather than only the obvious three.
_RESERVED: Set[str] = {
    "if", "for", "while", "switch", "catch", "return", "with", "do", "else",
    "function", "class", "new", "delete", "typeof", "void", "instanceof",
    "in", "of", "await", "yield", "throw", "case", "var", "let", "const",
    "import", "export", "default", "extends", "implements", "interface",
    "type", "enum", "namespace", "declare", "as", "is", "keyof", "infer",
    "readonly", "public", "private", "protected", "static", "abstract",
    "async", "get", "set", "this", "super", "constructor",
}

#: A name shorter than this is not distinctive enough to resolve across files.
#: Minified bundles are full of one- and two-character function names, and a
#: single ``g`` would otherwise collect a four-figure fan-in that buries every
#: real answer. Same-file edges are exempt (see the store builder).
MIN_CROSS_FILE_NAME = 3

#: A derived name that answers to more definitions than this across the tree is
#: not resolvable by name alone, so no cross-file edge is drawn for it. Same-file
#: resolution is always allowed and is never subject to this.
MAX_DEFINITION_CANDIDATES = 12

#: A body whose longest line exceeds this is treated as machine-generated and
#: skipped entirely. Minified and bundled output produces thousands of
#: single-letter "symbols" whose edges are noise in both directions.
MAX_LINE_FOR_EDGES = 400

_CALL = re.compile(r"(?:([A-Za-z_$][\w$]*)\s*\.\s*)?([A-Za-z_$][\w$]*)\s*\(")
_IDENT_TAIL = re.compile(r"[A-Za-z_$][\w$]*\s*$")
_WORD_BEFORE = re.compile(r"([A-Za-z_$][\w$]*)\s*$")


def is_js_source(path: str) -> bool:
    """True when ``path`` is a file this module can extract edges from."""
    lowered = str(path or "").lower()
    if lowered.endswith((".min.js", ".min.mjs", ".min.cjs", ".bundle.js")):
        return False
    for ext in JS_EXTENSIONS:
        if lowered.endswith(ext):
            return True
    return False


def _regex_allowed_here(out: List[str], index: int) -> bool:
    """Can a ``/`` at this position start a regex literal rather than divide?

    The lexical rule JavaScript itself uses: a regex may begin where a *value*
    may begin. After an identifier, a number, or a closing bracket, ``/`` is
    division; after an operator, a comma, or an opening brace it is a regex.
    Getting this wrong in the permissive direction would swallow the rest of a
    line as a "regex", so the check errs toward division.
    """
    j = index - 1
    while j >= 0 and out[j] in " \t\r\n":
        j -= 1
    if j < 0:
        return True
    prev = out[j]
    if prev in ")]}":
        return False
    if prev.isalnum() or prev in "_$":
        # `return /re/` and `typeof /re/` are values; `x / y` is division.
        word = _WORD_BEFORE.search("".join(out[max(0, j - 16):j + 1]))
        return bool(word and word.group(1) in _RESERVED)
    return True


def strip_noise(source: str) -> str:
    """Blank out comments, strings and regex literals, preserving offsets.

    Every removed character is replaced by a space (newlines are kept) so that
    line and column positions in the result still match the input. Code inside
    a template literal's ``${ ... }`` is preserved, because calls genuinely
    live there.
    """
    out: List[str] = []
    i = 0
    n = len(source)
    # Stack of open template literals; each entry is the ``${`` brace depth at
    # which the template resumes being a string.
    tmpl_depth: List[int] = []
    brace_depth = 0
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            while i < n and source[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if ch == "/" and nxt == "*":
            while i < n and not (source[i] == "*" and i + 1 < n and source[i + 1] == "/"):
                out.append("\n" if source[i] == "\n" else " ")
                i += 1
            for _ in range(min(2, n - i)):
                out.append(" ")
                i += 1
            continue
        if ch in "'\"":
            quote = ch
            out.append(" ")
            i += 1
            while i < n and source[i] != quote:
                if source[i] == "\\":
                    out.append(" ")
                    i += 1
                    if i < n:
                        out.append("\n" if source[i] == "\n" else " ")
                        i += 1
                    continue
                out.append("\n" if source[i] == "\n" else " ")
                i += 1
            if i < n:
                out.append(" ")
                i += 1
            continue
        if ch == "`":
            out.append(" ")
            i += 1
            tmpl_depth.append(brace_depth)
            # Consume literal text until `${` (code resumes) or the closing tick.
            while i < n:
                if source[i] == "\\":
                    out.append(" ")
                    i += 1
                    if i < n:
                        out.append("\n" if source[i] == "\n" else " ")
                        i += 1
                    continue
                if source[i] == "`":
                    out.append(" ")
                    i += 1
                    tmpl_depth.pop()
                    break
                if source[i] == "$" and i + 1 < n and source[i + 1] == "{":
                    out.append(" ")
                    out.append(" ")
                    i += 2
                    brace_depth += 1
                    break
                out.append("\n" if source[i] == "\n" else " ")
                i += 1
            else:
                if tmpl_depth:
                    tmpl_depth.pop()
            continue
        if ch == "{":
            brace_depth += 1
            out.append(ch)
            i += 1
            continue
        if ch == "}":
            brace_depth -= 1
            out.append(" " if tmpl_depth and brace_depth == tmpl_depth[-1] else ch)
            i += 1
            if tmpl_depth and brace_depth == tmpl_depth[-1]:
                # Back inside the template's literal text.
                while i < n:
                    if source[i] == "\\":
                        out.append(" ")
                        i += 1
                        if i < n:
                            out.append("\n" if source[i] == "\n" else " ")
                            i += 1
                        continue
                    if source[i] == "`":
                        out.append(" ")
                        i += 1
                        tmpl_depth.pop()
                        break
                    if source[i] == "$" and i + 1 < n and source[i + 1] == "{":
                        out.append(" ")
                        out.append(" ")
                        i += 2
                        brace_depth += 1
                        break
                    out.append("\n" if source[i] == "\n" else " ")
                    i += 1
            continue
        if ch == "/" and _regex_allowed_here(out, len(out)):
            j = i + 1
            closed = False
            while j < n and source[j] != "\n":
                if source[j] == "\\":
                    j += 2
                    continue
                if source[j] == "[":
                    while j < n and source[j] not in "]\n":
                        j += 2 if source[j] == "\\" else 1
                if source[j] == "/":
                    closed = True
                    break
                j += 1
            if closed:
                for _ in range(j - i + 1):
                    out.append(" ")
                i = j + 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def extract_calls(source: str, self_name: Optional[str] = None) -> List[str]:
    """Names called somewhere in ``source``, as a sorted candidate list.

    ``self_name`` is the enclosing symbol: it is excluded so the chunk that
    DEFINES a function is never reported as one of its own callers, which is
    the single most misleading edge this extractor could produce.
    """
    if not source:
        return []
    if max((len(ln) for ln in source.split("\n")), default=0) > MAX_LINE_FOR_EDGES:
        return []
    cleaned = strip_noise(source)
    own = (self_name or "").split(".")[-1]
    found: Set[str] = set()
    for match in _CALL.finditer(cleaned):
        name = match.group(2)
        if name in _RESERVED or name == own:
            continue
        if match.group(1) is None:
            # A bare `name(` preceded by `function` is a declaration, not a call.
            # `new Foo(` deliberately IS kept: construction is a real edge, and
            # it is how most of a TypeScript codebase reaches a class.
            before = cleaned[max(0, match.start() - 24):match.start()]
            word = _WORD_BEFORE.search(before)
            if word and word.group(1) in ("function", "class"):
                continue
        found.add(name)
    return sorted(found)
