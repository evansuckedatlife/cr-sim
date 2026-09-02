#!/usr/bin/env python3
"""Validate this map against the tree it cites. ``python map/_meta/check.py``.

This lives beside the map rather than in a scratch directory so a later agent
re-runs it instead of re-deriving it. Four checks, and the first two exist
because a cold walk failed on them:

1.  **Citation root.** Every ``path:line`` citation resolves against the *repo
    root* and nothing else. Suffix matching is refused explicitly: it makes
    ``train/run.py:125`` look valid when no ``train/`` directory exists at the
    root, and it silently resolves into ``.claude/worktrees/agent-*/``, which
    is three other checkouts and not the subject tree.
2.  **Symbol extent.** A citation written as ``` `Symbol` (`path:line`) ``` or
    ``` `path:line-line` `` beside a symbol name is parsed against the target
    file's AST: the line must fall *inside* that symbol. A checker that only
    tests ``line <= len(file)`` passes a range that overshoots its function by
    twenty-six lines into two unrelated module constants, which is how
    ``build_parser`` came to be cited as ``82-347``.
3.  **Links.** Every relative markdown link from a map file lands on a file
    that exists.
4.  **Frontmatter.** ``status: verified`` requires ``verified:`` and
    ``commit:``. The gate used to be a prose convention in a body section,
    which is how six of six process cards came to claim ``verified`` with no
    date and no commit anywhere in the file. ``universe:`` and ``cluster:``
    are checked against their closed sets, in the spelling the map actually
    uses.
5.  **Twins.** ``AGENTS.md`` and ``routing.md`` must be byte-identical to
    ``CLAUDE.md``. They are generated, never hand-edited; two entry files that
    drift is the failure mode the form warns about.
6.  **Budget, in characters.** The documented walks are measured end to end.
    Characters, never lines: in a one-line-per-noun index a line count is not
    a proxy for cost, and measuring lines is what let a 68-line section that
    is larger than an 82-line one be reported as the worst case.

Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

MAP = Path(__file__).resolve().parent.parent
ROOT = MAP.parent

#: A citation is a backticked path with at least one ``:line`` or ``:a-b``.
CITE = re.compile(
    r"`([A-Za-z0-9_./*-]+\.(?:py|md|json|csv|toml|html|txt))((?::\d+(?:-\d+)?)+)`"
)
#: ``Symbol`` immediately followed by a parenthesised citation of one file.
SYMBOL_CITE = re.compile(
    r"`([A-Za-z_][A-Za-z0-9_.]*)`\s*(?:\(|in\s+|at\s+)"
    r"`([A-Za-z0-9_./-]+\.py):(\d+)(?:-(\d+))?`"
)
LINK = re.compile(r"\[[^\]]*\]\(([^)#][^)]*)\)")
#: Two backticked names joined before a citation pair -- see below.
JOINER = re.compile(r"`\s*(?:/|and|or|,|→|->)\s*$")

#: Prefixes a non-rooted citation would have been resolved under by suffix
#: matching. Naming them is how the checker reports *why* a citation fails.
GUESSES = (
    "cr_sim/", "cr_sim/train/", "cr_sim/engine/", "cr_sim/api/",
    "cr_sim/data/", "cr_sim/play/", "cr_sim/mumu/", "cr_sim/render/",
    "scripts/", "tests/",
)

#: The walks the map documents, as lists of map-relative files. Each must land
#: inside the 2k-8k token band of the method's walk test. Measured in
#: characters; the band is stated in characters here so the check needs no
#: tokenizer, which this machine does not have.
CHARS_PER_TOKEN = 4
BAND = (2_000, 8_000)  # tokens
WALKS = {
    "noun walk (worst cluster)": ["CLAUDE.md", "objects/measurement/_index.md",
                                  "objects/measurement/lift.md"],
    "conventions lookup": ["CLAUDE.md", "CONTEXT.md",
                           "objects/measurement/lift.md"],
    "effects walk (worst entry)": ["CLAUDE.md", "effects/CONTEXT.md",
                                   "objects/interface/reward-variants.md"],
    "points-in walk": ["CLAUDE.md", "effects/points-in.md",
                       "objects/surfaces/play-server.md"],
    "quoting walk": ["CLAUDE.md", "effects/quoting-a-result.md",
                     "objects/measurement/lift.md"],
    "process walk": ["CLAUDE.md", "processes/CONTEXT.md",
                     "processes/fine-tune.md"],
}
#: effects/CONTEXT.md and CONTEXT.md are read one section at a time, as their
#: own headers instruct. The walk cost is the largest section plus the
#: preamble, not the file, so these are measured per ``##`` section.
BY_SECTION = {"effects/CONTEXT.md", "CONTEXT.md"}
#: A cluster index is read whole -- it is one table and already trimmed to
#: shelf lines, so there is no section to seek within.


def md_files() -> list[Path]:
    return sorted(p for p in MAP.rglob("*.md"))


def largest_section(path: Path) -> int:
    """Characters in the largest ``##`` section, plus the preamble."""
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^## ", text, flags=re.M)
    pre = len(parts[0])
    return pre + max((len(p) for p in parts[1:]), default=0)


def check_citations() -> list[str]:
    bad: list[str] = []
    total = 0
    for f in md_files():
        for m in CITE.finditer(f.read_text(encoding="utf-8")):
            total += 1
            p = m.group(1)
            if (ROOT / p).exists():
                continue
            guess = [g for g in GUESSES if (ROOT / (g + p)).exists()]
            hint = f" -- did you mean `{guess[0]}{p}`?" if guess else ""
            bad.append(f"{f.relative_to(MAP)}: `{p}` does not exist at the "
                       f"repo root{hint}")
    print(f"citations: {total} checked, {len(bad)} unresolvable")
    return bad


def check_symbol_extents() -> list[str]:
    """A cited line must fall inside the symbol it is written beside."""
    bad: list[str] = []
    trees: dict[str, dict[str, list[tuple[int, int]]]] = {}
    checked = 0
    for f in md_files():
        text = f.read_text(encoding="utf-8")
        for m in SYMBOL_CITE.finditer(text):
            sym, path, start, end = m.group(1), m.group(2), m.group(3), m.group(4)
            # "`A` / `B` (`x`, `y`)" and "`A` and `B` (`x`, `y`)" pair two
            # symbols with two citations, and the regex only sees the second
            # name. That form is not a claim about `B` alone, so it is skipped
            # rather than reported: a checker that cries wolf on prose gets
            # switched off, and this check exists to be believed.
            before = text[max(0, m.start() - 24):m.start()]
            if JOINER.search(before):
                continue
            target = ROOT / path
            if not target.exists():
                continue  # check_citations already owns this failure
            if path not in trees:
                spans: dict[str, list[tuple[int, int]]] = {}
                try:
                    tree = ast.parse(target.read_text(encoding="utf-8"))
                except SyntaxError:
                    trees[path] = {}
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                         ast.ClassDef)):
                        spans.setdefault(node.name, []).append(
                            (node.lineno, node.end_lineno or node.lineno))
                trees[path] = spans
            name = sym.rsplit(".", 1)[-1]
            spans = trees[path].get(name)
            if not spans:
                continue  # not a def in that file; nothing to assert
            checked += 1
            lo, hi = int(start), int(end or start)
            # The decorator line sits one or two above ``node.lineno``.
            if not any(a - 3 <= lo and hi <= b for a, b in spans):
                where = ", ".join(f"{a}-{b}" for a, b in spans)
                bad.append(f"{f.relative_to(MAP)}: `{sym}` cited at "
                           f"`{path}:{start}{'-' + end if end else ''}` but "
                           f"{name} spans {where}")
    print(f"symbol extents: {checked} checked, {len(bad)} outside their symbol")
    return bad


def check_links() -> list[str]:
    bad: list[str] = []
    total = 0
    for f in md_files():
        for m in LINK.finditer(f.read_text(encoding="utf-8")):
            target = m.group(1).split("#")[0]
            if not target or target.startswith(("http://", "https://")):
                continue
            total += 1
            if not (f.parent / target).exists():
                bad.append(f"{f.relative_to(MAP)}: broken link -> {target}")
    print(f"links: {total} checked, {len(bad)} broken")
    return bad


UNIVERSES = {"live", "leftover", "ghost", "deliberate ghost"}
CLUSTERS = {"build", "battle", "interface", "measurement", "surfaces"}


def frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        return {}
    head = text.split("\n---\n", 1)[0]
    out = {}
    for line in head.split("\n")[1:]:
        if ":" in line and not line.startswith(" "):
            k, _, v = line.partition(":")
            out[k.strip()] = v.split("#")[0].strip()
    return out


def check_frontmatter() -> list[str]:
    bad: list[str] = []
    n = 0
    for f in md_files():
        if f.name in ("_index.md", "CONTEXT.md", "CLAUDE.md", "AGENTS.md",
                      "routing.md") or "_templates" in f.parts:
            continue
        fm = frontmatter(f.read_text(encoding="utf-8"))
        if not fm:
            continue
        n += 1
        rel = f.relative_to(MAP)
        if fm.get("status") == "verified" and not (fm.get("verified")
                                                   and fm.get("commit")):
            bad.append(f"{rel}: status: verified without verified:/commit:")
        u = fm.get("universe")
        if u is not None and u not in UNIVERSES:
            bad.append(f"{rel}: universe: {u!r} is not one of "
                       f"{sorted(UNIVERSES)}")
        c = fm.get("cluster")
        if c is not None and c not in CLUSTERS:
            bad.append(f"{rel}: cluster: {c!r} is not one of {sorted(CLUSTERS)}")
        if fm.get("type") == "object" and c and f.parent.name != c:
            bad.append(f"{rel}: cluster: {c!r} but the file is in "
                       f"{f.parent.name}/")
    print(f"frontmatter: {n} cards checked, {len(bad)} problems")
    return bad


def check_twins() -> list[str]:
    """`AGENTS.md` and `routing.md` are generated from `CLAUDE.md`, never edited."""
    bad: list[str] = []
    entry = (MAP / "CLAUDE.md").read_bytes()
    for name in ("AGENTS.md", "routing.md"):
        twin = MAP / name
        if not twin.exists():
            bad.append(f"{name}: missing")
        elif twin.read_bytes() != entry:
            bad.append(f"{name}: has drifted from CLAUDE.md "
                       f"-- regenerate with `cp map/CLAUDE.md map/{name}`")
    print(f"twins: 2 checked, {len(bad)} drifted")
    return bad


def check_budget() -> list[str]:
    bad: list[str] = []
    for name, files in WALKS.items():
        chars = 0
        parts = []
        for rel in files:
            p = MAP / rel
            if not p.exists():
                bad.append(f"{name}: {rel} does not exist")
                continue
            n = largest_section(p) if rel in BY_SECTION else len(
                p.read_text(encoding="utf-8"))
            chars += n
            parts.append(f"{rel}={n // CHARS_PER_TOKEN}")
        tok = chars // CHARS_PER_TOKEN
        flag = "" if BAND[0] <= tok <= BAND[1] else "  <-- OUT OF BAND"
        print(f"  {name}: ~{tok} tok ({' + '.join(parts)}){flag}")
        if flag:
            bad.append(f"{name}: ~{tok} tokens, band is {BAND[0]}-{BAND[1]}")
    return bad


def main() -> int:
    print(f"map: {MAP}\nrepo root: {ROOT}\n")
    failures = (check_citations() + check_symbol_extents() + check_links()
                + check_frontmatter() + check_twins())
    print("walk budget (chars/4, the floor estimator):")
    failures += check_budget()
    print()
    if failures:
        print(f"FAIL -- {len(failures)} problem(s):")
        for line in failures:
            print(f"  {line}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
