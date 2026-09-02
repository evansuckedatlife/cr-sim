# cr-sim system map

An edit map for this repo. The tree is the source of truth; this map cites it
`path:line` **from the repo root** and never restates it. Two hops, then source.

## Route by what you are doing

| You are | Go to | Then |
|---|---|---|
| **changing a line and want the blast radius** | [`effects/CONTEXT.md`](effects/CONTEXT.md) — eleven entries, one per change someone actually makes. `grep -n '^## '` and read **your entry**, not the file | the cards it names |
| **quoting or comparing a number**, no code changing | [`effects/quoting-a-result.md`](effects/quoting-a-result.md) — bug 6's other half, the one an edit-keyed index cannot catch | the six-input checklist |
| **asking what a noun is** | `objects/<cluster>/_index.md` — one line per noun: universe, owning `path:line`, card | that noun's card |
| **asking what points *into* the tree** | [`effects/points-in.md`](effects/points-in.md) — `tests/`, `scripts/`, the watcher, the root `CLAUDE.md`, `runs/`, the worktrees | the card each row names |
| **following a movement end to end** | [`processes/CONTEXT.md`](processes/CONTEXT.md) — six verbs and why there is no seventh | one verb card |

Pick a cluster by the question, not by where the file lives:

| Changing | Cluster |
|---|---|
| a stat, a unit, a level ladder, the decoded build | [`objects/build/_index.md`](objects/build/_index.md) |
| a tick phase, an entity, pathing, targeting, buffs, the state hash | [`objects/battle/_index.md`](objects/battle/_index.md) |
| an observation channel, the action mask, a reward term, a net shape | [`objects/interface/_index.md`](objects/interface/_index.md) |
| a lift, a verdict, a rating, a checkpoint, a demo shard, a random stream | [`objects/measurement/_index.md`](objects/measurement/_index.md) |
| the play server, the progress page, the CLI, MuMu, a one-off script | [`objects/surfaces/_index.md`](objects/surfaces/_index.md) |

`objects/CONTEXT.md` argues those five seams. `CONTEXT.md` holds what no card
may restate: the universes, the name collisions, the unit conventions, and the
six-bug charter this map exists for. Neither is a hop on a normal walk.

## Where the rest lives

| File | What it holds |
|---|---|
| [`CONTEXT.md`](CONTEXT.md) | universes; name collisions; unit conventions; the six bugs, numbered — the **only** place a bug is attributed to a cluster |
| [`_meta/schema.md`](_meta/schema.md) | the closed set of node types and labels this map may contain |
| [`_meta/overrides.md`](_meta/overrides.md) | every place a card knowingly departs from a comment — the list a stale-comment sweep reads |
| [`_meta/check.py`](_meta/check.py) | `python map/_meta/check.py` — citation roots, symbol extents, links, walk budget |
| [`_templates/`](_templates/) | blank `object.md` / `process.md`; a new card is a copy |

## Four universes

Every noun is marked **live** (implement and cite against it), **leftover**
(present, off the main path), **ghost** (named, not wired — do not implement
against it) or **deliberate ghost** (present in the game build, declined here on
purpose, with the reason at the declining site — implementing one without
reading its reason removes a tripwire). `CONTEXT.md` names the ones the brief
got wrong.

## The one rule

This map is not a second spec. Where a card and the code disagree, the code wins
and **the card is what gets fixed**, in the same change, with a row in
`_meta/overrides.md`. Correcting a card in a *different* file is how three files
came to carry three answers. `status: verified` requires a date, a commit and
citations.
