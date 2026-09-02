# effects/ — if you are changing X, open these cards

A catalog, not a waterfall. Each entry is a change phrased the way someone
about to make it would phrase it, the cards to open before making it, and the
**specific thing that breaks without failing**. Nothing here restates a card. If
this file and a card disagree, the card is wrong and gets fixed — but check the
code first, because the code wins over both.

Every entry below is on a path where a real bug shipped with a green suite. The
six are numbered in `../CONTEXT.md`; this file is the index they argue for.

**Read your entry, not the file.** One `##` section per change; find yours
first and read only that range:

```bash
grep -n '^## ' map/effects/CONTEXT.md
```

Two entries are not here, because they answer a different question and no entry
below needs them loaded:

| If you are | Go to |
|---|---|
| about to **quote or compare a number** — no code changing | [`quoting-a-result.md`](quoting-a-result.md) |
| asking **what outside the tree points in** — `tests/`, `scripts/`, the watcher, the root `CLAUDE.md`, `runs/`, the worktrees | [`points-in.md`](points-in.md) |

---

## 1. Changing a flag that configures an environment — bug 1

You are editing `build_parser` in `cr_sim/train/run.py:82-321`, or a `CRSimEnv`
keyword anywhere.

**Open:** [`../objects/interface/crsim-env.md`](../objects/interface/crsim-env.md)
→ [`../objects/interface/vec-env-config.md`](../objects/interface/vec-env-config.md)
→ [`../processes/fine-tune.md`](../processes/fine-tune.md)
→ [`../objects/measurement/config-json.md`](../objects/measurement/config-json.md).

**The silent break.** There are **two** construction paths and they are written
out separately: `_env()` at `cr_sim/train/run.py:533-546` and the `VecEnvConfig`
literal at `:994-1020`. `VecEnvConfig` has a default for every field
(`cr_sim/api/vec.py:69-113`), so a field set in one and not the other does not
raise — it silently takes the dataclass default in the worker processes only.
That is bug 1 exactly: `tower_level` defaults to 11 there, so `--tower-level 5
--workers 8` trained every rollout at 11 while `config.json` recorded 5 and the
probe evaluated at 5.

**Still open, same shape.** `skip_forced` (`cr_sim/api/env.py:318`) is a live
`CRSimEnv` parameter with **no `VecEnvConfig` field at all**. No entry point
sets it today. Wire a `--skip-forced` flag through `_env()` alone and bug 1
reproduces with no test touching it.

**Also check:** does the flag reach `config.json`? A new key is spent out of the
progress page's four-key A/B budget (`_AB_MAX_DIFF`,
`cr_sim/train/watch.py:1799-1803`). And `config.json` still has **no `workers`
key** — nothing in the artifact says which construction path built the
environments.

---

## 2. Changing what a demonstration records — bugs 2 and 3

You are editing `Demonstrations` (`cr_sim/train/clone.py:41-156`), `collect`
(`cr_sim/train/clone.py:241-422`), or `scripts/make_demos.py`.

**Open:** [`../processes/collect-demonstrations.md`](../processes/collect-demonstrations.md)
→ [`../objects/measurement/demonstrations.md`](../objects/measurement/demonstrations.md)
→ [`../processes/clone.md`](../processes/clone.md).

**The silent break.** Three fields exist only to say what the file *is*:
`observation` (`cr_sim/train/clone.py:78`), `reward` (`:86`) and `proposer`
(`:103`) — three separate lines, not one block. Adding a fourth thing a
shard depends on without adding a field for it recreates bug 2 — the shipped
demonstrations carried value targets from a reward no fine-tune optimised, and
the inherited critic predicted +1.48 where returns averaged +0.47. Adding a
field without teaching `merge._agree` about it
(`scripts/clone_policy.py:99-119`) recreates it one layer up: shards that
disagree merge quietly, the channel count matches, and training converges.

**The part that is mitigated, not closed** (bug 3): `Demonstrations.grid` stores
the **already-encoded** grid (`cr_sim/train/clone.py:45`). Nothing re-derives it. The
`observation` field turned `--observation` from an unverifiable declaration into
a comparison against the file (`scripts/clone_policy.py:233-239`), but a shard is still
a tensor whose meaning is a name.

**Also check:** `subset` and `merge` must carry every new field through, or a
`--fraction` run silently switches off the guard that reads it
(`scripts/clone_policy.py:131-138`). And `scripts/expert_iterate.py:129-145` builds
`make_demos`' command line from ten flags — a new required flag there breaks a
round at `subprocess.run` and nowhere else.

---

## 3. Changing the observation layout, or the deck — bug 4

You are editing `cr_sim/api/encoding.py`, `cr_sim/data/card_features.py`, or a
`DEFAULT_DECK` literal.

**Open:** [`../objects/interface/observation-vector.md`](../objects/interface/observation-vector.md)
→ [`../objects/interface/encoding-config.md`](../objects/interface/encoding-config.md)
→ [`../objects/interface/card-features.md`](../objects/interface/card-features.md)
→ [`../objects/interface/observation-grid.md`](../objects/interface/observation-grid.md)
→ [`../objects/interface/check-observation.md`](../objects/interface/check-observation.md).

**The silent break.** 80 of the trunk's 102 vector columns are card-identity
one-hots keyed on **vocabulary position**, and the vocabulary is the deck union
rebuilt per environment. Swap decks and column *i* means a different card, with
the same `vocab_size`, the same `vector_size`, a passing strict load and a
passing `check_observation` — because that check compares a shape. **No model
artefact records the deck.** `runs/<name>/config.json` records it
(`cr_sim/train/run.py:606`) and nothing reads it back.

**The second literal.** `cr_sim/play/server.py:46` is an independent
`DEFAULT_DECK`, byte-identical to `cr_sim/train/run.py:54` today. Change one and
the browser opponent's vocabulary repermutes against the checkpoint's, and no
test imports both —
[`../objects/surfaces/play-server.md`](../objects/surfaces/play-server.md).

**Also check:** channel *order* is unpinned. Swapping the `swarm` and `spells`
entries in `GRID_FEATURE_CHANNELS` leaves `tests/test_observation_v2.py` and
`tests/test_api_encoding.py` fully green while repermuting channels 8-11 under
every v2 checkpoint. And terrain must stay the **last** channel, outside the
normalisation slices (`../CONTEXT.md`, unit conventions).

---

## 4. Changing the reward, or its shaping — bugs 2 and 6

You are editing `cr_sim/api/reward.py`, `_reward_weights`
(`cr_sim/train/run.py:403`), or a `--shaping` / `--tower-weight` /
`--elixir-weight` default.

**Open:** [`../objects/interface/reward-variants.md`](../objects/interface/reward-variants.md)
→ [`../objects/measurement/reward-schedule.md`](../objects/measurement/reward-schedule.md)
→ [`../objects/measurement/lift.md`](../objects/measurement/lift.md)
→ [`../processes/collect-demonstrations.md`](../processes/collect-demonstrations.md).

**The silent break.** A lift is a difference of *returns* over the control's own
spread, so the reward is in the numerator **and** the denominator. Change a
`ProjectionWeights` field default and you move: every future demonstration set's
value column; every future clone's critic; **and the pinned evaluation scale**,
because `EVAL_REWARD = ProjectionWeights()` is built from field defaults
(`cr_sim/train/run.py:342`) — the one thing pinning was introduced to make
impossible.

**Already-open mismatch.** `scripts/make_demos.py:115` defaults
`--elixir-weight` to **0.0** while `cr_sim/train/run.py:182` defaults it to
**0.3**, under a comment at `scripts/make_demos.py:98` claiming the defaults match.

**Do not "fix" the gamma-correct potential.** `docs/training.md:483-505`
describes `r = γΦ(s') − Φ(s)`, which was written and reverted; half its stated
reason is now false. Ghost. Do not re-derive the plan from that paragraph.

**Do not aim an anneal at `--shaping` under `projected` or `five-term`.** Every
`_shaped_value` call site is in the branch those rewards do not take, and 0.01
against 5.00 is bit-identical under both (`cr_sim/api/env.py:263-281`). The knob
that is actually the shaping is chosen by `knob_for_reward`
(`cr_sim/train/schedule.py:97`).

---

## 5. Changing anything that produces a number people compare — bug 6

You are editing `evaluate`, `paired_lift`, `evaluate_paired`,
`evaluation_seeds`, a probe, or `fit_ratings`.

**Open:** [`../processes/evaluate-against-a-control.md`](../processes/evaluate-against-a-control.md)
→ [`../objects/measurement/lift.md`](../objects/measurement/lift.md)
→ [`../objects/measurement/verdict.md`](../objects/measurement/verdict.md)
→ [`../objects/measurement/metrics-row.md`](../objects/measurement/metrics-row.md)
→ [`../processes/rate-on-the-ladder.md`](../processes/rate-on-the-ladder.md).

**The silent break.** A number is only comparable to another when they share
**both** the control and the reward. Three things already differ across live
callers and none of them raises:

- **Denominator.** Three conventions across ten sites —
  [`../objects/measurement/lift-callers.md`](../objects/measurement/lift-callers.md)
  owns the list, including the one script whose output is the 0.062 noise floor.
- **Arena.** `--tower-level` defaults differently at twelve entry points and
  almost nothing records which one a number came from —
  [`../objects/measurement/tower-level.md`](../objects/measurement/tower-level.md).
  **Nothing refuses a verdict that omits it.**
- **Seeds.** `evaluation_seeds(block=0)` is byte-identical to the fixed list
  every historical number was measured on (`cr_sim/train/evaluate.py:310-315`). Changing the
  master seed or the draw order re-bases every number in `runs/` at once.

**Two guards you must not route around.** `check_lift_is_named`
(`cr_sim/train/selfplay.py:55`, five raises) for a metrics row and `write_verdict`
(`cr_sim/train/evaluate.py:483`, three raises) for the file. If a new writer
cannot satisfy them, that is the guard working — `scripts/register_job.py:50` is
the one metrics writer that skips them, and `cr_sim/train/watch.py:526-528` names
it as the exception in source.

**A rating is exempt from the reward, not from the rest.** Elo is fitted on
crowns (`cr_sim/train/ladder.py:550-559`), which no reward touches — which is why
`_LIFT_KEYS` (`cr_sim/train/evaluate.py:477-480`) keys the reward clause to the lift and not
to the file. Never put an Elo and a lift on one axis, or in one field.

---

## 6. Adding a random draw anywhere — bug 5

You are about to write `np.random.*`, `torch.rand*`, `random.*`, or to sample
from a policy.

**Open:** [`../objects/measurement/random-streams.md`](../objects/measurement/random-streams.md)
first, then the card for whatever you are drawing inside.

**The rule this repo learned the hard way.** Every draw is owned by a named
generator, derived arithmetically from things a reader can see — a seed, a
battle seed, a block index, a mode name. Never from the global stream, and never
from `hash()`, which is salted per process and so is reproducible within one run
and not between two (`cr_sim/train/ladder.py:504-506`).

**The four ways it has already gone wrong here**, each fixed in place with the
measurement written beside it:

| Where | What it cost |
|---|---|
| torch's global stream inside `--workers` processes | three fresh spawns reported different `torch.initial_seed()` and shared none of their first twenty opponent actions (`cr_sim/api/vec.py:87-100`) |
| `np.random.shuffle` in the PPO minibatch loop | identical rollouts, different updates, from one `--seed` (`cr_sim/train/ppo.py:213-222`) |
| `evaluation_probe` sampling off the global stream | +0.905 / +1.228 / +0.970 on identical inputs, against a 0.062 sd noise floor (`cr_sim/train/selfplay.py:457-467`) |
| `evaluate_paired` keying its stream on an arm's **index** in `modes` | +1.320 vs +1.197 for one checkpoint, decided by whether the caller also wanted greedy (`cr_sim/train/evaluate.py:437-459`) |

**Still unowned, and worth knowing before you compare two clones.**
`scripts/clone_policy.py:267` builds `ActorCritic` **before** `clone()` seeds
torch at `cr_sim/train/clone.py:575`, and nothing in that script seeds it. So a
clone's initial weights come off a per-process OS-entropy seed —
`torch.initial_seed()` measured at 29251125811100 and 29253538301300 in two
fresh interpreters. `cr_sim/train/ppo.py` does not have this shape: it seeds at
`:211` and builds at `:243`.

---

## 7. Changing a card stat, a level ladder, or a unit

**Open:** [`../objects/build/unit-spec.md`](../objects/build/unit-spec.md)
→ [`../objects/build/card-ladder.md`](../objects/build/card-ladder.md)
→ [`../objects/build/tower-ladder.md`](../objects/build/tower-ladder.md)
→ [`../objects/build/validation-gates.md`](../objects/build/validation-gates.md).

Conversion from milliseconds, milli-tiles and tiles-per-minute happens **once**,
at spec-build time. The rules and the eight hard-coded `18`s live in
`../CONTEXT.md`; do not restate them, and do not add a ninth. A stat change also
moves `reference/anchors.json`, which ~33 test files reach through one import —
see [`points-in.md`](points-in.md).

## 8. Changing a tick phase, an entity, or anything the state hash sees

**Open:** [`../objects/battle/battle.md`](../objects/battle/battle.md)
→ [`../objects/battle/entity-ids.md`](../objects/battle/entity-ids.md)
→ [`../objects/battle/state-hash.md`](../objects/battle/state-hash.md)
→ [`../objects/battle/rng.md`](../objects/battle/rng.md)
→ [`../objects/battle/lookahead.md`](../objects/battle/lookahead.md).

Entity ids and list order are hashed, so a branch that burns ids without giving
them back desyncs a replay that never diverged. `SearchBot` branches the battle
per candidate on every decision — so a phase-order change is a change to the
**expert**, and therefore to every demonstration set collected after it.

## 9. Changing a checkpoint's payload, or `ActorCritic`'s parameter count

**Open:** [`../objects/measurement/checkpoint.md`](../objects/measurement/checkpoint.md)
→ [`../objects/interface/policy-heads.md`](../objects/interface/policy-heads.md)
→ [`../objects/interface/net-config.md`](../objects/interface/net-config.md).

`HEAD_BY_PARAMETERS` (`cr_sim/train/ladder.py:98`) identifies the head of the
22-of-42 checkpoints that record neither `head` nor `observation` by **counting
parameters**. Any change to `ActorCritic`'s parameter count silently invalidates
that table, and `player_from_checkpoint` then builds the wrong network for a
real file. No checkpoint of any of the four kinds records the deck, the reward
or the tower level.

## 10. Adding a key to `config.json` or a metrics row

**Open:** [`../objects/measurement/config-json.md`](../objects/measurement/config-json.md)
→ [`../objects/measurement/metrics-row.md`](../objects/measurement/metrics-row.md)
→ [`../objects/measurement/run-directory.md`](../objects/measurement/run-directory.md).

A config key costs a slot in the four-key A/B budget; **deleting** one is worse
than adding one, because it changes the key set and makes every new run
unpairable with every old one. A metrics row is an open dict with exactly one
guard (`check_lift_is_named`), which is why the guard has five raises rather than
a schema.

## 11. Changing `cr_sim/train/watch.py`

**Open:** [`../objects/surfaces/progress-page.md`](../objects/surfaces/progress-page.md)
→ [`../objects/measurement/run-directory.md`](../objects/measurement/run-directory.md)
→ [`../objects/measurement/config-json.md`](../objects/measurement/config-json.md).

**The rule is in the root `CLAUDE.md` and nowhere in the source:** Python does
not reload an edited module in a running process, so after any change to
`watch.py` the served page silently keeps rendering the old code. A fresh
timestamp proves nothing — the stale process rewrites `progress.html` every 15
seconds. Verify by grepping the generated page for a string only the new code
emits. `read_ladder` (`cr_sim/train/watch.py:775`) is landed and dark for exactly this reason.
