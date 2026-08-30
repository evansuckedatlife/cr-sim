---
type: object
cluster: interface
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/api/reward.py
---

# Reward variants

What the agent is paid, in three spellings picked by the **type** of one
argument. The CLI word and the class name do not line up: `--reward simple` has
no class at all (an inline `_shaped_value`), `--reward five-term` is
`RewardWeights` / `RewardTracker`, `--reward projected` is `ProjectionWeights`
/ `ProjectedReward`.

## Why this shape

Crown difference is the objective and is almost always zero — most matches at
this length end level, so most episodes would pay nothing, and the measurement
that says so is at the top of the file: random play wins 29% *against an
opponent that never plays a card*, because the other 71% end level
(`cr_sim/api/reward.py:1-8`). All three variants answer that
the same way: define a running score, pay its **change**. Summed over an
episode the shaping cancels, so it is dense enough to guide the search and
cannot change which policy is optimal; paying a term directly is farmable, and
the named farm is elixir (`cr_sim/api/reward.py:33-40`).

**Two of the three make that a guarantee and one does not.** That is what
`telescopes` names, and it is the most consequential fact on this card.
`_shaped_value` and `ProjectedReward.score` are pure functions of the battle.
`RewardTracker.score` is not: `_observe` accumulates kite ticks and death
attribution, so the score depends on the path to the state as well as on the
state (`cr_sim/api/reward.py:129-132`, `:230-248`). The episode return still
telescopes for a tracker that saw every state — pinned at
`tests/test_reward.py:222-252` — but no state may be skipped, and policy
invariance is not guaranteed the way it is for a true potential
(`cr_sim/api/reward.py:333-335`).

**What telescoping does to the return.** Under `projected` the return is
`phi(s_T) - phi(s_0)` exactly, so the forced-decision run-out scores only its
endpoints instead of every state it passes through. That is arithmetic, not an
approximation, and the measured cost of not doing it is written at the site
(`cr_sim/api/env.py:466-478`). The same identity is why the critic has a ceiling —
see [../measurement/explained-variance.md](../measurement/explained-variance.md).

Selection is by type rather than by a separate flag so the variant and its
coefficients cannot be set inconsistently (`cr_sim/api/env.py:337-341`), and so
`reset()` can rebuild the reward when a schedule moves the weights, through one
dispatch, without changing the variant by accident (`cr_sim/api/env.py:214-221`).

## Shape

- `_build_reward(team, registry, reward_weights)` — the whole dispatch:
  `ProjectionWeights` first, any other non-`None` second, `None` last. `None`
  is a meaningful value here, not "unset".
- **simple** — `_shaped_value` is `crown_diff + w * tower_hp_frac_diff`. Both
  sides start at full health so the opening value is 0, and `w = 0` recovers
  the pure sparse objective through this same path rather than a second one.
- **five-term** — the argparse default (`cr_sim/train/run.py:279`). Six weights, five
  of them shaping; `crowns` lives here so the whole reward is described in one
  place and is deliberately not reduced. `score` is literally
  `sum(self._terms.values())`.
- **projected** — three knobs (`tower`, `elixir`, `horizon_seconds`); the board
  is played out and priced instead of weighted. `score` returns a three-term
  sum while `terms` carries a fourth key, `projected_ticks`, that is **not** in
  the score. Rebuilding a score by summing `terms` is right for the tracker and
  silently wrong here.
- `telescopes` is read at exactly two places, both `getattr(..., False)`, both
  in `CRSimEnv`: the skip in `step` and the settle at the end of
  `_run_out_forced_decisions`.
- `reward_shaping_weight` reaches `VecEnvConfig` and the worker faithfully and
  is then never read under either reward anyone trains with — every
  `_shaped_value` call site in `CRSimEnv` sits inside the `else` of
  `if self._reward is not None`. Measured rather than asserted, 0.01 against
  5.00.
- `as_dict()` on both weight classes is what a run directory and a lift label
  record. A coefficient missing from it is a coefficient no artefact carries —
  the hole `ProjectionWeights.tower` sat in (`cr_sim/api/reward.py:307-316`).

Citations: `cr_sim/api/reward.py:61-81`, `:114`, `:129-132`, `:182-211`,
`:230-248`, `:283-316`, `:319`, `:340-347`, `:369-392`;
`cr_sim/api/env.py:199-209`, `:212-226`, `:314`, `:337-341`, `:409-415`,
`:479-484`, `:555`, `:568-574`; `cr_sim/api/vec.py:77`, `:81`, `:138-139`;
`cr_sim/train/run.py:403-413`; inertness measured at
`tests/test_reward_schedule.py:235-260`.
Verified 2026-08-30 against `main` @ `dc47f51`. Every file cited above is clean
in that tree. `scripts/make_demos.py`, pointed at from Surfaces, is not — its
line numbers are working-tree numbers.

**Code wins over one comment.** `cr_sim/train/schedule.py:18-20` says every
`_shaped_value` call site sits inside that `else`. Three of the five do
(`cr_sim/api/env.py:413`, `:482`, `:572`); the other two are in `CRSimSelfPlayEnv`
(`:711`, `:729`), which has no reward object and therefore no such `if`. The
conclusion holds — the weight is real only where no reward object exists — but
an editor grepping the name finds five sites and only three match the sentence ([`../../_meta/overrides.md`](../../_meta/overrides.md), row 13).

## Connected to

- **owns:** `RewardWeights`, `RewardTracker`, `ProjectionWeights`,
  `ProjectedReward`, `unit_elixir_values`, and the `terms` dict logged beside
  every step.
- **owned-by:** [crsim-env.md](crsim-env.md) — the env constructs it, holds the
  `_previous` potential, and is the only caller of `step`.
- **joins:**
  [../measurement/reward-schedule.md](../measurement/reward-schedule.md) (moves
  these weights over a run),
  [../measurement/explained-variance.md](../measurement/explained-variance.md)
  (the ceiling this shape imposes),
  [../measurement/verdict.md](../measurement/verdict.md) (`reward_name` reads
  the weight tuple off the env that played),
  [../measurement/demonstrations.md](../measurement/demonstrations.md) (the
  value targets a clone's critic inherits),
  [vec-env-config.md](vec-env-config.md) (the field the workers build from).
- **looks-like-but-is-not:** `reward_shaping_weight` / `--shaping` looks like
  the shaping knob and is one only under `--reward simple`; the real one is
  `ProjectionWeights.tower` / `.elixir`, or the five non-crown `RewardWeights`
  fields. `EVAL_REWARD` (`cr_sim/train/run.py:342`) is a `ProjectionWeights` and is
  **not** the training reward. `reward.describe` (`:395`) has zero callers.
  `reward.__all__` (`:58`) omits `ProjectionWeights`, `ProjectedReward` and
  `describe`, all three imported by name elsewhere, so it is not the export
  list it looks like.

## If you change this

- **Hits:** the critic's explained-variance ceiling — it is a property of this
  object, not of the net or the observation
  ([../measurement/explained-variance.md](../measurement/explained-variance.md)).
  Every scale label: `reward_name` (`cr_sim/train/selfplay.py:163-185`) builds its
  string from `as_dict()`, so an added or renamed field changes the label on
  every verdict row and every demo shard's `meta`. `RewardSchedule`:
  `SHAPING_FIELDS` names fields by string, per knob
  (`cr_sim/train/schedule.py:80-85`), so a new numeric field is carried unchanged
  unless named there and a non-numeric one makes `anneal_to_zero` raise
  (`:155-165`). The run-out shortcut: put a path-dependent term in
  `ProjectedReward.score` and `telescopes = True` becomes a lie that changes
  rewards silently — the guard is `tests/test_lookahead.py:301-341`. And the
  worker path: a weights object is pickled into `VecEnvConfig` and again as an
  RPC payload (`cr_sim/api/vec.py:81`, `:437`), so anything unpicklable fails only
  under `--workers`. And — the sharp one — **`ProjectionWeights`' field
  defaults are the pinned probe.** `EVAL_REWARD = ProjectionWeights()` is
  constructed with no arguments (`cr_sim/train/run.py:342`), so changing a default
  moves the scale every in-run lift is measured on, for every future run, with
  nothing in `config.json` reading differently except the value it copies out.
- **Does not hit:** the engine. Nothing in `cr_sim/engine/` knows the reward
  exists — the tracker reads the battle and never instruments it
  (`cr_sim/api/reward.py:117-121`) — so `engine/battle.py` is the obvious next stop
  and the wrong one. It does not hit the observation either: reward terms are
  signed, observation channels are `[0,1]` by construction, and adding a term
  adds no channel (see `../../CONTEXT.md`, unit conventions). And **switching
  which variant a run trains under** does not move the in-run lift's scale:
  `_eval_env` passes `EVAL_REWARD` explicitly, precisely so the promotion
  criterion stops being a function of the training reward
  (`cr_sim/train/run.py:324-342`, `:561`). Assuming the lift series follows the
  training reward is the obvious next inference and the wrong one — but note
  the exception in Hits, which is a different change to the same file.

## Surfaces

| Surface | Role |
|---|---|
| `python -m cr_sim.train.run` (`--reward`, `--tower-weight`, `--elixir-weight`, `--horizon-seconds`, `--shaping`) | writes — builds the weights object at `cr_sim/train/run.py:403-413` |
| `cr_sim.train.schedule`, through `set_reward_weights` | writes — rebuilds it at each env's next reset |
| `scripts/make_demos.py:190-211` | writes — through its own shim, because `horizon_seconds` names the *search* horizon in that script |
| `train/selfplay.reward_name` | reads — the label on every lift and every shard |
| `cr_sim/engine/` | none |
| a human reading `runs/<name>/config.json` | reads — `reward`, `reward_schedule`, `eval_reward` |

## See

- Source: `cr_sim/api/reward.py`; the dispatch and the potential's bookkeeping
  at `cr_sim/api/env.py:199-226` and `:444-585`
- As-built: `docs/training.md`, "The reward pays for waiting" — working-tree
  line numbers; that file is modified in the tree this card was verified against
