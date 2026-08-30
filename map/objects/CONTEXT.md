# objects/ — the noun library

One card per noun. Clusters are **the questions an editor arrives with**, not
the folder layout of `cr_sim/`. Three of the five cut across the package tree
on purpose; if you look for a noun where its file lives rather than where its
question lives, you will not find it.

Read your cluster's `_index.md` first — `objects/<cluster>/_index.md`. It has a
line for every noun in that cluster, including the ones no card will ever be
written for, so a cold agent reaches source in one hop whether or not a card
exists.

**One index per cluster, five files.** There is no whole-tree index to open by
accident; `_index.md` at this level is a router and holds no rows. A row whose
Card column names a card is a **link**, so a promised-and-unwritten card is a
broken link that `map/_meta/check.py` fails on, rather than a name that reads as
coverage.

## The five clusters

| Cluster | The question it answers | Which files it draws from |
|---|---|---|
| **build** | *Where does this number come from, what unit is it in, and which ladder scales it?* | `cr_sim/data/{source,cards,csv_loader,decode,leveling,validate,interactions,engagement}.py`, `cr_sim/engine/{fixed,constants,specs}.py`, `reference/` |
| **battle** | *What happens inside a match, and what makes it reproducible?* | everything else in `cr_sim/engine/`, plus `cr_sim/replay.py` |
| **interface** | *What does the agent see, what may it do, and what is it paid?* | `cr_sim/api/`, `cr_sim/train/nets.py`, `cr_sim/data/card_features.py` |
| **measurement** | *What does this number mean, and what was it measured against?* | `cr_sim/train/{run,ppo,clone,selfplay,evaluate,ladder,schedule,scripted,proposal}.py`, the measurement `scripts/` |
| **surfaces** | *Who outside the training loop reads these nouns, and what breaks silently when one moves?* | `cr_sim/{cli,soak}.py`, `cr_sim/play/`, `cr_sim/render/`, `cr_sim/mumu/`, `cr_sim/train/{watch,report,notify,bot}.py`, the one-off `scripts/` |

## Why these five, and not the package tree

Three of the boundaries are real seams a change crosses. **Which bug lives in
which cluster is settled once, in the charter table at
[`../CONTEXT.md`](../CONTEXT.md), and cited by number here.** This file argues
the seams; it does not re-derive the attributions, because deriving them twice
is exactly how bug 1 acquired two incompatible roots on two hub files.

- **build ends, battle begins, at `UnitSpec`.** That is the one place
  milliseconds, milli-tiles and tiles-per-minute stop existing
  (`cr_sim/engine/specs.py`). The card ladder and the tower ladder are *not the
  same ladder* and both are cut here — which is what makes level 11 a bad
  *arena* ([`measurement/tower-level.md`](measurement/tower-level.md)). It is
  not why charter row 1's field went missing; that was two construction paths in
  `cr_sim/train/run.py`, and row 1 files it under `interface` accordingly.
- **battle ends, interface begins, at the encoder.** A battle is a state; an
  observation is a promise about a tensor's shape. Charter rows 3 and 4.
- **interface ends, measurement begins, at the artefact.** A checkpoint, a demo
  shard, a verdict. Charter rows 2, 5 and 6.

`surfaces` is not a leftovers drawer. It is the only cluster whose members are
*not* covered by the change-impact edges the other four draw: nothing in the
training loop imports `cr_sim/play/server.py`, so no card in `interface` would
ever name it — yet it holds a second, independent `DEFAULT_DECK`
(`cr_sim/play/server.py:46`) whose divergence from `cr_sim/train/run.py:54`
repermutes 80 observation columns with no shape change and no guard. That is
exactly the failure this form warns about: what points *into* the tree is
invisible from inside it. Its five cards are the landing sites for
[`../effects/points-in.md`](../effects/points-in.md).

`cr_sim/data/card_features.py` lives in the data package and is filed under
**interface**, because the only question anyone asks it is "what does the agent
see about a card." It is also where bug 4 is written down and measured.

## Rules for a card in here

- Copy `../_templates/object.md`. Seven sections, all of them.
- Cite `path:line`, **rooted at the repo root** (`../CONTEXT.md`, Citation
  root). Where a comment and the code disagree, the code wins, the card says
  which comment it is overriding, and the override gets a row in
  [`../_meta/overrides.md`](../_meta/overrides.md) — one file, because a
  stale-comment sweep reads the list and not sixty cards.
- **Fix the card, in the card.** Recording a correction on a different file
  leaves the wrong sentence standing where the documented walk lands.
- **If you change this** is Hits / Does not hit, first-order only. "Does not
  hit" names the obvious next noun that is the *wrong* one.
- Never restate a unit convention or a name collision. Those live once, in
  `../CONTEXT.md`. Cite it.
- Never paste an audit report. Point at the file that owns the behaviour.
- Ghosts get cards only when someone would otherwise implement against them.
  Everything else ghostly stays an index line.
- **A noun with two universes gets two cards, not a paragraph.** `universe:` is
  a frontmatter label and has to be true of the whole file it sits on, or it is
  not queryable. `ActionSelect` is the worked example: live as an interpreter
  node ([`battle/action-select.md`](battle/action-select.md)), a deliberate
  ghost in its `rand(n)` spelling
  ([`battle/action-select-rand.md`](battle/action-select-rand.md)).

## Cluster folders

`build/`, `battle/`, `interface/`, `measurement/` and `surfaces/` each hold
their own `_index.md` and their cards. An empty cluster folder is worse than no
folder — it reads as coverage — and so is a Card column naming a file that does
not exist, which is why that column is a link the checker resolves.
