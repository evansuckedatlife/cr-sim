---
type: process
status: verified
verified: 2026-08-30
commit: dc47f51
consumes: [objects/battle/battle.md, objects/interface/observation-features.md, objects/interface/reward-variants.md, objects/measurement/search-bot.md]
produces: [objects/measurement/demonstrations.md]
---

# collect-demonstrations

Watch the one-ply search expert play, and write down every decision at which
more than one action was legal.

## Input → Movement → Output

In: a build, a deck, a tower level, a reward, and optionally a checkpoint that
proposes which placements are worth an engine branch. Movement: one shard plays
`--episodes` battles; at every decision the bot branches the battle per
candidate placement and plays each branch forward, and the *scores it assigned*
— not the move it made — become the training target. Out: one `shard-NN.npz`
per observation variant, carrying the encoded grid, the vector, the mask, the
chosen index, the discounted return, the search's distribution, and a `meta`
block stating what produced it.

## Why this shape

**The chosen move is not a function of the state.** Candidates are sampled, so
one board can produce two different moves, and fitting the move fits the
sampling (`cr_sim/train/clone.py:53-60`). The search's *values over the
candidates it looked at* are a function of the state, so those are what the
shard carries.

**Sharded, and each shard must play different battles.** A match costs about
seventeen seconds. `--shard N` sets `offset = N * 10_000`
(`scripts/make_demos.py:273`), and that offset reaches both the opponent and
the reset seed (`cr_sim/train/clone.py:292-302`, `:339`). Until `seed_offset` existed the
offset reached only the opponent, so six shards of sixty episodes were six runs
of the *same* sixty battles against six different opponents.

**One playthrough produces every observation variant.** The expert reads the
`Battle`, not the observation (`cr_sim/train/clone.py:350` passes `env.battle`), so every
variant sees the identical trajectory and the identical decisions — which is
the only thing that makes an encoding ablation a paired comparison rather than
two experiments (`cr_sim/train/clone.py:426`, `scripts/make_demos.py:72-82`).

## Steps

1. Parse. The flag set is written separately from `run.py`'s and has drifted
   before — `scripts/make_demos.py:60-66` says so, and `_flag_names`
   (`:446-455`) exists so a test can hold the script's own error messages to
   the flags that exist. Entry `:266`.
2. Build the world once: `LogicData.load(DEFAULT_BUILD)`, `build_level_table`,
   `build_card_registry` (`:269-270`). See
   [`objects/build/logic-data.md`](../objects/build/logic-data.md).
3. Build the environment factory (`env_factory`, `:165-213`). The opponent is a
   seeded random policy at `50_000 + seed` (`:181`) or a second `SearchBot`
   (`:183-189`). `CRSimEnv` is constructed at `:190-211` with
   `tower_level=args.tower_level` (`:194`, default **5** at `:71`).
4. Build the reward through a shim, not from `args` (`:201-210`). In this
   script `horizon_seconds` is the **search's** horizon (15 s) and
   `--reward-horizon-seconds` is the **reward's** (3 s); passing `args` straight
   to `_reward_weights` would build the projection with a five-times-too-long
   lookahead. The shim names every field explicitly so a new knob in
   `_reward_weights` (`cr_sim/train/run.py:403`) fails loudly here.
5. If `--proposer` is a checkpoint, load it against a probe environment and
   build a per-battle proposer (`:278-292`; `proposer_factory`,
   `cr_sim/train/proposal.py:154`). `--proposer none` is byte-for-byte the old
   bot (`:123-128`).
6. Build the search config (`search_config`, `:216-238`) and the per-battle
   expert (`expert_factory`, `:241-263`). The bot's candidate stream is derived
   per battle by `battle_stream_seed(shard, battle_seed)` (`:251`), not carried
   across the shard — the docstring at `:223-232` records what the old
   behaviour cost.
7. `collect` runs the loop (`cr_sim/train/clone.py:241`). Per episode:
   `env.reset(seed=seed_offset + episode)` (`:339`); per decision, keep the row
   only when `int(flat.sum()) > 1` (`:352`) — a state with one legal action
   teaches no preference and would let the pass action dominate (`:305-310`).
8. Build the target row from the search's own scores (`_target_row`,
   `cr_sim/train/clone.py:172`; called at `:376-380`), scaled by that position's own
   candidate spread and floored at `min_spread=1e-3` (`:267`). Below the floor
   the search had no preference and the row collapses to the move it made;
   `collapsed` counts those (`:383`).
9. Walk the episode's rewards backwards to a discounted return, and give each
   kept state **its own step's** return, not an evenly-strided one
   (`:391-404`). Kept decisions are sparse and unevenly spaced.
10. Stamp the meta (`scripts/make_demos.py:321-355`). `reward_weights` is read
    off a real environment via `reward_name(make_env(0))` (`:330`), not
    asserted from the flags; `policy_candidates` is the **effective** count
    after `SearchBot`'s clamp, not the number requested (`:310-316`).
11. Write one `shard-NN.npz` per variant (`:373-377`; `Demonstrations.save`,
    `cr_sim/train/clone.py:123`) and print the collapse refusal if the guided fallback rate
    is more than ten points above the unguided baseline (`collapse_refusal`,
    `:404-443`). **The shard is written either way** — the refusal says do not
    merge it, and says what it measured.

## If you change this

- **Hits:**
  [`objects/measurement/demonstrations.md`](../objects/measurement/demonstrations.md)
  — every field of the file, including `observation`, `reward` and `proposer`,
  which `clone_policy.merge._agree` refuses to mix
  (`scripts/clone_policy.py:99-119`).
  [`clone.md`](clone.md) — the value column *is* the clone's critic, and the
  critic is what PPO inherits.
  [`objects/interface/reward-variants.md`](../objects/interface/reward-variants.md)
  — changing a `ProjectionWeights` default moves the value column of every
  future shard.
  `scripts/expert_iterate.py:129-145`, which builds this script's command line
  and passes ten of its flags; a flag that driver forgets is a knob the round
  silently takes the default of. `tests/test_train.py:940-951` parses that
  command back through `build_parser()`.
- **Does not hit:** the **search's** own weights.
  `SearchBotConfig.tower_weight` / `.elixir_weight`
  (`cr_sim/train/scripted.py:117`, `:121`) are how the bot *chooses*, and the
  source says in place that they are **not** the reward's
  (`cr_sim/train/scripted.py:108-116`). `search_config` (`:216-238`) never sets either, so
  `--elixir-weight` here moves the recorded value column and **not one decision
  the bot makes**. See the collision row in `../CONTEXT.md`.
  It also does not hit `Demonstrations.grid`'s meaning: the grid is stored
  **already encoded** (`cr_sim/train/clone.py:45`), so a change to the encoder does not
  migrate an existing shard — it makes it a file whose declared `observation` is
  now the wrong name for its contents.

## Code wins over the comment

Filed at [`../_meta/overrides.md`](../_meta/overrides.md), row 18.

`scripts/make_demos.py:98` says "The defaults here match `cr_sim.train.run`'s,
which is the point." **They do not match on elixir.** `--elixir-weight` here
defaults to **0.0** (`:115`); `cr_sim/train/run.py:182` defaults to **0.3**. So
an unflagged collection harvests its value column under
`projected:elixir=0,tower=1,horizon_seconds=3` while an unflagged fine-tune
optimises `projected:elixir=0.3,...` — bug 2's exact shape, now *recorded* in
the shard's meta (`:330`) rather than closed. `docs/training.md` already records
that the shipped shards were collected at elixir 0.

## Surfaces

| Surface | Role |
|---|---|
| a human at a shell | runs six shards in parallel; reads the per-shard progress line (`:299-303`) |
| `scripts/expert_iterate.py` | writes this script's command line (`:129-145`) |
| `tests/test_train.py`, `tests/test_clone.py` | parse the parser, and hold the refusal text to flags that exist |
| the progress page | **nothing.** A collection is not a run and registers no `runs/` entry — see [`../effects/points-in.md`](../effects/points-in.md) |

## Citations

Every `path:line` in the steps above is rooted at the repo root and was
resolved against the tree, together with the AST extent of the symbol it
is named beside (`../_meta/check.py`).

Verified 2026-08-30 against `main` @ `dc47f51`, **working tree**: `scripts/make_demos.py`, `cr_sim/train/clone.py`, `cr_sim/train/scripted.py`
are among the nine files uncommitted at that commit, so their line
numbers are working-tree numbers and were re-checked here rather than
inherited (`../CONTEXT.md`, Verification basis).

## See

- Objects: [`demonstrations.md`](../objects/measurement/demonstrations.md),
  [`search-bot.md`](../objects/measurement/search-bot.md),
  [`reward-variants.md`](../objects/interface/reward-variants.md),
  [`observation-features.md`](../objects/interface/observation-features.md),
  [`random-streams.md`](../objects/measurement/random-streams.md)
- Source: `scripts/make_demos.py`, `cr_sim/train/clone.py:241-424`
