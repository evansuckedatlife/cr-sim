---
type: process
status: verified
verified: 2026-08-30
commit: dc47f51
consumes: [objects/measurement/demonstrations.md, objects/interface/net-config.md, objects/interface/policy-heads.md]
produces: [objects/measurement/checkpoint.md, objects/measurement/run-directory.md, objects/measurement/verdict.md]
---

# clone

Fit a network to the expert's recorded decisions, then play it against the
random control so the result is a claim about winning rather than about
agreement.

## Input → Movement → Output

In: a directory of `shard-*.npz`, a head, an observation name, and a tower
level. Movement: merge the shards under three refusals, split a holdout, fit
policy and value together for `--epochs`, then play three arms — control,
sampled, greedy — over one shared seed list. Out: `runs/<name>/cloned.pt`
beside the full four-file run directory, so the clone lands on the same page and
the same axes as the training runs it will be compared to.

## Why this shape

**Agreement is not the claim.** A policy agreeing with the expert 60% of the
time can still lose every match, because the 40% it gets wrong are the decisions
that mattered — and this repo has measured two heads matching the expert's tile
4.8% and 5.1% while winning 85% and 96% (root `CLAUDE.md`, "Agreement with the
expert does not track winning"). So the script ends in battles, against the same
control every other number here sits on (`scripts/clone_policy.py:8-16`,
`:308-311`).

**Passes are down-weighted or the argmax is always "do nothing".** Nearly half
the expert's decisions are passes; the rest are spread over some seven hundred
placements. At `pass_weight=1.0` agreement froze at exactly the pass fraction
for fourteen epochs and the greedy policy lost 100% of its matches
(`cr_sim/train/clone.py:550-561`).

**Three fields must agree across shards or the merge is refused.**
`observation`, `reward` and `proposer` (`scripts/clone_policy.py:82-90`,
`_agree` at `:99-119`). A set whose encoding varies row to row is undetectable
downstream: the channel count matches, training converges, and the checkpoint
carries whichever name was declared.

## Steps

1. Glob `shard-*.npz` and `merge` (`:222-223`; `merge` at `:52-96`). `target` is
   all-or-nothing across shards (`:78-79`) — a half-filled target array trains
   some rows on the search's beliefs and the rest on zeros.
2. Optionally `subset` for a sample-efficiency curve (`:224-225`; `:123-149`).
   The slice carries `observation`, `reward` and `proposer` through, because
   blanking them switched off the very guard in step 4 (`:131-138`).
3. `--targets hard` sets `data.target = None` (`:226-227`), which throws away
   the search's distribution *after* collecting it. `soft` is the third arrow of
   the loop; `hard` is for the shards on disk that cannot support it
   (`:174-187`).
4. Refuse a declared `--observation` that disagrees with the recorded one
   (`:233-239`); warn, do not fail, when the shards predate provenance
   (`:240-244`). This is the check that turned bug 3 from an unverifiable
   declaration into a comparison against the file.
5. Build the environment (`make_env`, `:256-262`): 20 TPS, frame skip 30,
   120 s, `tower_level=args.tower_level` (default **5**, `:188`), a seeded random
   opponent at `60_000 + offset`, and **no `reward_weights`** — so this
   environment scores `simple:shaping=0.01`.
6. Build the network: `ActorCritic(net_config_for(probe, head=args.head))`
   (`:267`). See [`objects/interface/net-config.md`](../objects/interface/net-config.md)
   — `vocab_size` and `vector_size` come off the probe environment, so a
   same-size deck swap passes a strict load.
7. `clone(net, data, CloneConfig(...))` (`:281-285`; `cr_sim/train/clone.py:568`).
   It seeds torch (`:575`), splits a 10% holdout with its own generator
   (`:592-594`), and each epoch shuffles the training indices off the **global**
   stream (`:598`) — reproducible only because `:575` reset it. Loss is
   `policy + 0.5 * value` (`:612-614`).
8. Save `cloned.pt` (`:288-302`): `state_dict`, `observation`, `proposer`,
   `demo_meta`, `targets`, `pass_weight`, `head`, and the last epoch's stats. See
   [`objects/measurement/checkpoint.md`](../objects/measurement/checkpoint.md).
9. Play three arms over one seed list drawn from `default_rng(777)`
   (`:313-319`): control (`net=None`), sampled, greedy. Each gets a fresh
   environment.
10. Compute the lift by hand — `difference.std(ddof=1) / sqrt(n)`, over the
    control's own `std(ddof=1)` (`:321-338`). This is `paired_lift`'s arithmetic
    respelled, not `paired_lift` itself.
11. Write `verdict.json` through `write_verdict` (`:372`;
    `cr_sim/train/evaluate.py:483`), carrying `eval_opponent` and `eval_reward`
    read off a real environment (`:355`, `:361`). The headline is whichever arm
    scored better (`:346`).
12. Write `config.json` (`:378-389`) and a two-line `metrics.jsonl` through
    `check_lift_is_named` (`:391-416`). Two identical rows at `updates` 1 and 2
    (`:415-416`), because a clone does not learn over time and the page needs
    two points to draw anything.

## If you change this

- **Hits:**
  [`objects/measurement/checkpoint.md`](../objects/measurement/checkpoint.md) —
  `cloned.pt`'s payload is what `head_for_parameters` and `player_from_checkpoint`
  read (`cr_sim/train/ladder.py:98`, `:240`).
  [`fine-tune.md`](fine-tune.md) — `--init-from` refuses a head mismatch
  (`cr_sim/train/run.py:684-689`) and takes **weights only**, never the
  optimiser (`:690-694`).
  [`rate-on-the-ladder.md`](rate-on-the-ladder.md) — a clone is the usual
  `--anchor`.
  [`objects/measurement/run-directory.md`](../objects/measurement/run-directory.md)
  and its four files; the progress page reads all four.
  `scripts/expert_iterate.py:148-153`, which builds this command line.
- **Does not hit:** the demonstrations. Nothing here rewrites a shard, and
  `--observation` cannot repair one — step 4 refuses rather than converts. And
  it does **not** hit the training reward: this script has no `--reward` flag at
  all, so its verdict is always denominated in `simple:shaping=0.01` no matter
  what the shards' value column was harvested under. That gap is deliberate and
  shared by every offline script here (root `CLAUDE.md`, "A lift also needs the
  reward it was counted in"), which is exactly why `eval_reward` is written into
  both the verdict and the row.

## The unowned draw this process still has

`--seed` (`:213`) reaches `CloneConfig.seed` and `subset`, and nothing else.
The network is constructed at `:267`, **before** `clone()` calls
`torch.manual_seed` at `cr_sim/train/clone.py:575`, and nothing in
`scripts/clone_policy.py` seeds torch. So the clone's **initial weights** are
drawn from torch's global generator, which is seeded from OS entropy per
process — measured here: two fresh interpreters reported `torch.initial_seed()`
29251125811100 and 29253538301300, and the same `nn.Linear(4,4)` came out
-0.2588 and -0.1449. `cr_sim/train/ppo.py` does not have this shape: it seeds at
`:211` and builds the network at `:243`. Two runs of one `clone_policy.py`
command line therefore start from different weights and cannot be compared
below the sampled noise floor. Filed on
[`objects/measurement/random-streams.md`](../objects/measurement/random-streams.md).

## Surfaces

| Surface | Role |
|---|---|
| a human at a shell | the primary caller |
| `scripts/expert_iterate.py` | writes this command line, and **omits `--tower-level`** — see [`expert-iterate.md`](expert-iterate.md) |
| `cr_sim/train/watch.py` | reads all four output files; `runs/cloned/verdict.json`'s flat mirror is the case its own docstring calls dangerous (`cr_sim/train/watch.py:725`) |
| `cr_sim/train/report.py` | reads `verdict.json`'s flat keys (`cr_sim/train/report.py:45`) |
| `tests/test_clone.py`, `tests/test_measurement.py` | hold the merge refusals and the verdict shape |

## Citations

Every `path:line` in the steps above is rooted at the repo root and was
resolved against the tree, together with the AST extent of the symbol it
is named beside (`../_meta/check.py`).

Verified 2026-08-30 against `main` @ `dc47f51`, **working tree**: `scripts/clone_policy.py`, `cr_sim/train/clone.py`
are among the nine files uncommitted at that commit, so their line
numbers are working-tree numbers and were re-checked here rather than
inherited (`../CONTEXT.md`, Verification basis).

## See

- Objects: [`demonstrations.md`](../objects/measurement/demonstrations.md),
  [`checkpoint.md`](../objects/measurement/checkpoint.md),
  [`verdict.md`](../objects/measurement/verdict.md),
  [`metrics-row.md`](../objects/measurement/metrics-row.md),
  [`config-json.md`](../objects/measurement/config-json.md),
  [`policy-heads.md`](../objects/interface/policy-heads.md)
- Source: `scripts/clone_policy.py`, `cr_sim/train/clone.py:537-660`
