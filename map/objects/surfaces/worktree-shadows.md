---
type: object
cluster: surfaces
universe: leftover
status: verified
verified: 2026-08-30
commit: dc47f51
entity: .claude/worktrees
---

# Worktree shadows

`.claude/worktrees/agent-*/` — three other checkouts of this repo, each a
complete `cr_sim/` from a different era, living inside the subject tree. Not a
noun the code owns; a noun **every search you run here owns**.

## Why this shape

Agents work in git worktrees, so the experiments actually moving are usually
not in this checkout. `cr_sim/train/watch.py:4427-4441` reads their `runs/`
directories on purpose — a page showing only `main`'s runs showed nothing live
while four sweeps ran a directory away. That is a deliberate, useful edge.

The cost is that the shadows are indistinguishable from the subject tree to
anything that walks the filesystem. **Any `grep -rn` from the repo root returns
them first**, alphabetically, before `cr_sim/`. Three of the hits for
`GRID_FEATURE_CHANNELS` are 9- and 13-channel encoders from eras this map does
not describe.

This is why the map's citation rule is absolute. A citation written
*train/run.py:125* resolves against the repo root under **no** interpretation —
there is no `train/` directory there — so a validator that falls back to suffix
matching finds it under *.claude/worktrees/agent-∗/cr_sim/* and reports it
valid. `map/_meta/check.py` refuses the fallback for exactly that reason and
names the prefix it would have guessed instead.

## Shape

- `.claude/worktrees/agent-a0511dcd4e2a118fb/`,
  `agent-a84fa956d179068ac/`, `agent-acff606c02b4824d0/` — three checkouts.
  Their `cr_sim/api/encoding.py` files are 655, 594 and 822 lines against this
  tree's; none is the current encoder.
- A fourth partial copy sits under `runs/_diag/`.
- `agent-acff606c02b4824d0/runs` holds **194 MB** of experiment results that
  exist nowhere else — `runs/`, `data_cache/` and `checkpoints/` are gitignored,
  so git does not carry them (root `CLAUDE.md:114-116`).
- **Never `git worktree remove --force`.** It follows Windows directory
  junctions and has already destroyed the real `data_cache` once
  (root `CLAUDE.md:110-111`).

Citations: `cr_sim/train/watch.py:4427-4441`; root `CLAUDE.md:110-111`,
`:114-116`; `map/_meta/check.py`.
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** nothing in the subject tree.
- **owned-by:** nothing. It is outside every import graph and inside every
  filesystem walk.
- **joins:** [`progress-page.md`](progress-page.md) (`_run_roots` reads them on
  purpose), [`../measurement/run-directory.md`](../measurement/run-directory.md).
- **looks-like-but-is-not:** `cr_sim/`. That is the entire point of this card.
  A file under `.claude/worktrees/` that reads like the module you are editing
  is a different commit of it.

## If you change this

- **Hits:** nothing, because nothing here should be changed. The card exists so
  a search result from one of these directories is recognised before it is
  edited or cited.
- **Does not hit:** the suite. `pytest` collects from `tests/` and never
  descends here — so the obvious next assumption, that a green run means these
  copies are consistent with the tree, is wrong. They are three different trees
  and nothing compares them.

## Surfaces

| Surface | Role |
|---|---|
| `grep -rn` from the repo root | reads them **first** |
| `cr_sim/train/watch.py:4427-4441` | reads their `runs/`, deliberately |
| `map/_meta/check.py` | refuses to resolve a citation into them |
| `pytest` | none |

## See

- Source: `.claude/worktrees/`, root `CLAUDE.md`
