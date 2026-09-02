---
type: process
status: verified
verified: 2026-08-30
commit: dc47f51
consumes: [objects/measurement/checkpoint.md, objects/measurement/demonstrations.md]
produces: [objects/measurement/demonstrations.md, objects/measurement/checkpoint.md, objects/measurement/ladder.md]
---

# expert-iterate

Run collect → clone → rate in order, then make this round's clone the next
round's proposer.

`scripts/expert_iterate.py:156`. It is a **driver**, not a measurement: it
shells out to the three existing entry points with the round's own directories
and stops on the first non-zero exit (`:21-26`, `:113-126`). The rating comes
from `run_ladder.py` and the merge gate from `make_demos.py`, because a loop
that scores itself is the one thing this project must not build.

## Input → Movement → Output

In: a `--seed-policy` checkpoint and a round count. Movement: per round, collect
`--shards` shards with the current proposer, clone them, rate the clone against
fixed anchors, then set `proposer = runs/iter-N/cloned.pt`. Out: per round
`data_cache/demos-iterN/`, `runs/iter-N/`, `runs/iter-N-ladder/`, and one
`runs/expert-iteration.json` recording the chain (`:187-194`).

## Why this shape

**The missing arrow.** AlphaZero's improvement operator is three: the policy
proposes, the search refines the proposal, the refined distribution trains the
policy. `SearchBot` was the second and `clone.collect` the third; the first
never existed here — the search drew about fourteen stratified-random placements
out of a mean of 104 legal ones, 13.5% coverage, and the network's opinion about
which fourteen deserved an exact engine branch was never consulted (`:5-11`).

**It closes because the expert is cheap.** A decision at `candidates=14,
horizon_seconds=15` is 375 ms and an episode about 17 s, so 360 episodes across
six shards is 17-20 minutes; cloning is minutes and rating about fifteen. One
turn is under an hour (`:13-19`).

**Every round is rated against the same anchors or the rounds are not comparable
to each other** (`:95-99`; default `["random"]` at `:160`).

**`--dry-run` prints the commands and runs none of them** (`:103`, `:116-117`) —
the honest way to see what a round costs before spending it.

## Steps

1. Refuse a missing `--seed-policy` unless dry (`:158-159`).
2. Per round, name the directories: `data_cache/demos-iter<N>` and
   `runs/iter-<N>` (`:166-167`).
3. `--shards` calls of `demo_command` (`:170-171`; built at `:129-145`) — see
   [`collect-demonstrations.md`](collect-demonstrations.md).
4. One `clone_command` (`:173`; built at `:148-153`) — see [`clone.md`](clone.md).
5. Unless `--skip-ladder`, one `run_ladder.py` invocation naming this round's
   clone as the sole entrant (`:175-185`) — see
   [`rate-on-the-ladder.md`](rate-on-the-ladder.md). `--skip-ladder` leaves the
   round with no number attached and the next round built on an unmeasured
   teacher, which its own help says is how three invalid comparisons happened
   here (`:104-109`).
6. `proposer = out / "cloned.pt"` (`:190`). That line is the loop.

## What the driver passes, and what it does not

`demo_command` and `clone_command` are functions rather than inline lists
precisely so a test can parse them back through the target's own
`build_parser()` (`:132-136`; `tests/test_train.py:940-951`) — "a flag this
driver forgets is a knob the round silently takes the default of, which is how
the search's own value distribution came to be computed and discarded in the
same round."

| Reaches | make_demos (10 flags) | clone_policy (6 flags) | run_ladder (6 + anchors) |
|---|---|---|---|
| `--tower-level` | yes (`:141`) | **no** | yes (`:180`) |
| `--observation` | no (it takes `--observations`) | yes (`:152`) | derived from the checkpoints |
| `--episodes` | yes, per shard (`:138`) | no (defaults to 120) | `--ladder-episodes` (`:179`) |
| `--targets` | — | yes (`:153`) | — |
| the reward flags | **no** | has none | — |

## The live instance of bug 1's shape

`--tower-level` (`:85`, default 5) reaches `make_demos.py` and `run_ladder.py`
and **does not reach `scripts/clone_policy.py`**, whose own default is 5
(`scripts/clone_policy.py:188`). So `expert_iterate.py --tower-level 11` runs a
round whose demonstrations and rating are at level 11 while the clone's 120
evaluation battles, its `verdict.json`, its `metrics.jsonl` row and the
`tower_level` written into its `config.json` (`scripts/clone_policy.py:382`) are all at
level 5. At 11 the towers outlast the match and about 90% of battles draw, so
those are not the same game. Nothing fails; the defaults agree, so it is silent
until the flag is used. This is bug 1 one layer up — a value that reaches one
construction path and not another — and the driver is where it lives now.

The parallel gap is the reward: no reward flag reaches `make_demos.py`, so every
round harvests its value column under that script's defaults
(`projected:elixir=0,tower=1,horizon_seconds=3`,
`scripts/make_demos.py:99-120`), and `clone_policy.py` has no reward flag at all.
Recorded rather than reconciled: the shard's `meta` carries the weight tuple read
off a real environment (`scripts/make_demos.py:330`).

## If you change this

- **Hits:** all three child processes' parsers. A renamed or newly-required flag
  in `make_demos.py`, `clone_policy.py` or `run_ladder.py` breaks a round at
  `subprocess.run` with a non-zero exit and no other symptom (`:119-125`).
  `tests/test_train.py:929-951`, which parses these command lines back.
  `runs/expert-iteration.json` (`:192-194`), the only record of which clone
  proposed which round.
  [`objects/measurement/search-bot.md`](../objects/measurement/search-bot.md) —
  `--policy-candidates` is clamped by `SearchBot` to leave the random floor
  intact, so the number this driver asks for is not necessarily the number the
  round took (`:60-65`; the clamp is reported at `scripts/make_demos.py:310-316`).
- **Does not hit:** a mismatched `--observation`. It is the obvious next worry
  and it is already guarded loudly: the driver sends `--observation` to
  `clone_policy.py` only, so the shards stay `v1`, and
  `scripts/clone_policy.py:233-239` refuses the merge by name rather than training
  quietly. Nor does it hit the collapse gate — that stays in
  `make_demos.collapse_refusal` (`:404-443`), which prints and does not exit, so
  a collapsed shard does **not** stop the round. The operator has to read the
  output.

## Universe note

**live code, no surviving round.** Its own `--targets` help records that a real
round wrote `runs/iter-1/cloned.pt` with `targets: 'hard'` over a shard whose
fallback rate was 0.0 (`:72-82`) — so this has run at least once. Nothing from
it is on disk now: checked 2026-08-30, there is no `runs/iter-*`, no
`runs/expert-iteration.json` and no `data_cache/demos-iter*` in this checkout or
in any of the three worktrees under `.claude/worktrees/`. `runs/` and
`data_cache/` are gitignored (root `CLAUDE.md:114-116`), so absence here is not
evidence the loop does not work — only that no round's numbers can be checked.

## Surfaces

| Surface | Role |
|---|---|
| a human at a shell | the only caller |
| the progress page | sees `runs/iter-N` and `runs/iter-N-ladder`, because the children register themselves. **The driver does not** — a multi-round run is several hours with no entry of its own until a child writes one |
| `tests/test_train.py` | parses both command builders back through the targets' parsers |

## Citations

Every `path:line` in the steps above is rooted at the repo root and was
resolved against the tree, together with the AST extent of the symbol it
is named beside (`../_meta/check.py`).

Verified 2026-08-30 against `main` @ `dc47f51`, **working tree**: `scripts/expert_iterate.py`, `scripts/make_demos.py`
are among the nine files uncommitted at that commit, so their line
numbers are working-tree numbers and were re-checked here rather than
inherited (`../CONTEXT.md`, Verification basis).

## See

- Objects: [`demonstrations.md`](../objects/measurement/demonstrations.md),
  [`checkpoint.md`](../objects/measurement/checkpoint.md),
  [`search-bot.md`](../objects/measurement/search-bot.md),
  [`ladder.md`](../objects/measurement/ladder.md)
- Source: `scripts/expert_iterate.py`
