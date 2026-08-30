# Schema — the rules of this map

The closed set of node types, the labels they carry, and the naming they
follow. When practice and this file disagree, reconcile the same day. Schema
drift is how a map rots into a second spec.

`python map/_meta/check.py` enforces the mechanical half: citation roots, symbol
extents, links, and the walk budget in characters. What it cannot check is
below.

## Node types

This map may contain these and nothing else.

| `type:` | Lives at | Carries |
|---|---|---|
| `object` | `objects/<cluster>/<slug>.md` | the seven sections of `_templates/object.md` |
| `process` | `processes/<slug>.md` | Input → Movement → Output, numbered cited steps, `consumes` / `produces`, a `Citations:` block with a date and a commit |
| — (catalog) | `CLAUDE.md`, `AGENTS.md`, `routing.md` | routing only, under ~60 lines, no payload |
| — (contract) | `CONTEXT.md`, `objects/CONTEXT.md`, `processes/CONTEXT.md`, `effects/CONTEXT.md` | how to walk the folder it sits in |
| — (index) | `objects/<cluster>/_index.md` | one line per noun in that cluster; `objects/_index.md` is a router with no rows |
| — (effects entry) | `effects/<slug>.md` | one arrival that is not an edit — `points-in.md`, `quoting-a-result.md` |
| — (registry) | `_meta/overrides.md` | every place a card knowingly departs from a comment |
| — (checker) | `_meta/check.py` | the mechanical rules, runnable |

`processes/` holds **six** verb cards and nothing else — the movements that
actually run. A seventh is a structural claim and is argued in
`processes/CONTEXT.md`.

`effects/CONTEXT.md` is a catalog — "if you are changing X, open these cards" —
never a copied waterfall. It is read one `##` section at a time and says so in
its own header. **An arrival that is not an edit gets its own file** rather than
a twelfth section: `points-in.md` answers *what outside the tree points in*, and
`quoting-a-result.md` answers *may these two numbers share an axis*. Neither is
needed by any entry in the hub, and both were unreachable while they were
sections of it.

Where `effects/CONTEXT.md` and a card disagree, **fix the card, in the card**.
Recording the correction on the hub instead leaves the wrong sentence standing
in the file the documented walk lands on — which is how `NetConfig`'s field
count came to have three values in three files.

## Frontmatter on an object card

```yaml
---
type: object
cluster: build | battle | interface | measurement | surfaces
universe: live | leftover | ghost | deliberate ghost
status: stub | verified | stale
verified: YYYY-MM-DD        # required by status: verified
commit: <short sha or branch>   # required by status: verified
entity: <path to the owning file, from the repo root>
---
```

- `cluster` is closed at five. Adding a sixth is a structural change, argued in
  `objects/CONTEXT.md` before any card moves.
- `universe` is closed at four, and **`deliberate ghost` is spelled with a
  space**, here and everywhere. It means the thing exists in the game build and
  this engine declines it on purpose, with the reason at the declining site.
- **A noun with two universes gets two cards.** `universe:` has to be true of
  the whole file it sits on or the label is not queryable, and a paragraph
  inside a `live` card is not a filing. See `objects/battle/action-select.md`
  and `objects/battle/action-select-rand.md`.
- `status: verified` requires **all three** of `verified:`, `commit:` and
  `path:line` citations in the Shape section. They are frontmatter fields rather
  than a prose convention in a body section, because a prose convention is what
  six of six process cards omitted. Absent any one, it is `stub` or `stale`. A
  confident wrong date is worse than an honest `stale`.

Process cards carry `status`, `verified`, `commit`, `consumes` and `produces`,
and the same gate applies.

## Labels that make it queryable

`type`, `cluster`, `universe`, `status`, `verified`, `commit`, `entity`. Object
cards link each other by relative path under `Connected to`; those links draw
the graph on their own. No wikilinks — this map lives in a git repo, not a
vault, and a relative path resolves in a code host.

## Naming

- **Citation root.** Every `path:line` resolves against the **repo root** and
  nothing else: `cr_sim/train/run.py:125`, never the shortened
  *train/run.py:125*. A shortened citation resolves only by suffix matching, and
  suffix matching also resolves into `.claude/worktrees/agent-*/` — three other
  checkouts of this repo. A bare `:NNN` continues the file last named in the
  same paragraph, and that is the only abbreviation. `check.py` refuses the
  fallback by name.
- **A cited line must be inside the symbol it is named beside.** `check.py`
  parses the target's AST and asserts it. A checker that only tests
  `line <= len(file)` passes a range that overshoots a function by twenty-six
  lines into two unrelated constants.
- Slugs are kebab-case and name the noun as the code names it: `unit-spec.md`,
  `vec-env-config.md`. Not the product word — that goes in the card's first
  sentence and in the collision table in `CONTEXT.md`.
- One card per noun. A noun that appears under two *names* gets one card and one
  row in the collision table. A noun that appears in two *universes* gets two
  cards; those are different situations.
- `_meta/` holds the rules. A cluster `_index.md` is rebuilt as cards land and
  **is never a place to store a fact that belongs on a card** — a field count, a
  call-site list, a default. Its Card column is a link so an unwritten card is a
  failure and not a name that reads as coverage.

## What may not go in a card

- As-built behaviour copied out of the source. Point at the file that owns it.
- An audit report. A card is seven short sections, not a findings list.
- A second spec. If a card and the code disagree, the code wins and the card is
  what gets fixed — and the card says which comment it is overriding **and adds
  a row to `_meta/overrides.md`**.
- A restatement of a unit convention or a name collision. Those live once, in
  `CONTEXT.md`, and a card cites them.
- **A count of generated data.** How many directories are under `runs/`, how
  many checkpoints are on disk: these drift between one reading and the next and
  a checked-in number is wrong by the afternoon. Own the *shape* — the two-file
  contract, the writers, the search depth — and say "every run directory".
