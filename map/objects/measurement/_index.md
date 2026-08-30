# measurement — what a number means, and what it was measured against

One line per noun in this cluster, including the ones no card will be written
for. This is the shelf list: it exists so a cold agent reaches source in one
hop without opening the cluster in bulk.

**A row with a card is a pointer, not an entry.** The gloss lives on the card;
this line carries the noun, its universe, the owning `path:line` and the link.
Where the Card column reads `—` the index line *is* the whole entry, and
that is deliberate — those rows keep their gloss because nothing else holds
it. Adding a fact here that belongs on a card is how the two came to disagree
about `NetConfig`'s field count; put it on the card
([`../../_meta/schema.md`](../../_meta/schema.md), Naming).

Paths are relative to the **repo root** — always, with no exception, because
a suffix match resolves into `.claude/worktrees/agent-*/`, which is three other
checkouts. `python map/_meta/check.py` enforces it. Universes, name collisions
and unit conventions live once in [`../../CONTEXT.md`](../../CONTEXT.md) and no
row may restate them.

| Noun | Universe | Owner | Status | Card |
|---|---|---|---|---|
| **`Lift`** | live | `cr_sim/train/evaluate.py:330`, arithmetic `:356-372` | verified | [`lift.md`](lift.md) |
| `Result` / `evaluate` / `evaluate_paired` | live | `cr_sim/train/evaluate.py:77`, `:137`, `:377`, `:437-473` | verified | [`lift.md`](lift.md) |
| `evaluation_seeds` | live | `cr_sim/train/evaluate.py:296` | verified | [`lift.md`](lift.md) |
| `paired_lift`'s call sites — six, in four files, and three denominator conventions across ten | live | `cr_sim/train/evaluate.py:330` | verified | [`lift-callers.md`](lift-callers.md) |
| `tower_level` as a **measurement** input — the arena a number was taken in; twelve entry points, two conventions, and nothing that refuses a verdict omitting it | live | `cr_sim/engine/battle.py:223`, `:470` | verified | [`tower-level.md`](tower-level.md) |
| the sampled noise floor is **0.062 sd**, not 0.04 | live | `cr_sim/train/evaluate.py:157-167`; `runs/sampled-noise-floor/noise.json` | verified | [`lift.md`](lift.md) |
| `write_verdict` | live | `cr_sim/train/evaluate.py:483`, `:507`, `:514`, `:526-545` | verified | [`verdict.md`](verdict.md) |
| `check_lift_is_named` / `SCORED_FAMILIES` | live | `cr_sim/train/selfplay.py:55`; raises `:111`, `:118`, `:133`, `:142`, `:148`; `:52` | verified | [`metrics-row.md`](metrics-row.md) |
| `opponent_name` / `reward_name` | live | `cr_sim/train/selfplay.py:188`, `:163` | verified | [`verdict.md`](verdict.md) |
| `EVAL_REWARD` | live | `cr_sim/train/run.py:342` | verified | [`verdict.md`](verdict.md) |
| **nothing refuses a verdict that omits `tower_level`** | live | `cr_sim/train/evaluate.py:483` (the three raises, none of them this); recorded in a verdict only by `scripts/measure_expert.py:190` | verified | [`tower-level.md`](tower-level.md) |
| `rotating_probe` | live | `cr_sim/train/evaluate.py:555`, `:592-611`; wired `cr_sim/train/run.py:979-984` | verified | [`verdict.md`](verdict.md) |
| `Demonstrations` | live | `cr_sim/train/clone.py:42`, `:66-118` | verified | [`demonstrations.md`](demonstrations.md) |
| `_collect_variants` | live | `cr_sim/train/clone.py:426`, configs `:454-458`, re-encode `:473-477` | verified | [`demonstrations.md`](demonstrations.md) |
| `collect` / `clone` / `CloneConfig` | live | `cr_sim/train/clone.py:241`, `:568`, `:538`, `:592` | verified | [`demonstrations.md`](demonstrations.md) |
| `merge` / `_agree` / `subset` | live | `scripts/clone_policy.py:52`, `:99`, `:123` | verified | [`demonstrations.md`](demonstrations.md) |
| `Checkpoint` (training) | live | `cr_sim/train/run.py:868`, `:891`, `:1053` | verified | [`checkpoint.md`](checkpoint.md) |
| `Checkpoint` (clone) | live | `scripts/clone_policy.py:288` | verified | [`checkpoint.md`](checkpoint.md) |
| `HEAD_BY_PARAMETERS` / `head_for_parameters` | live | `cr_sim/train/ladder.py:98`, `:106` | verified | [`checkpoint.md`](checkpoint.md) |
| `player_from_checkpoint` / `default_player_name` / `_GENERIC_STEMS` | live | `cr_sim/train/ladder.py:240`, `:226`, `:223`; `head_source` `:150` | verified | [`checkpoint.md`](checkpoint.md) |
| `RunDirectory` | live | `cr_sim/train/run.py:515-517`; `scripts/register_job.py:35` | verified | [`run-directory.md`](run-directory.md) |
| `MetricsRow` (`metrics.jsonl`) | live | `cr_sim/train/run.py:733`, guard `:913`; also `scripts/clone_policy.py:413`, `scripts/run_ladder.py:310`, `:337`, `scripts/evaluate_vs_expert.py:224`, `scripts/measure_expert.py:233`, `cr_sim/train/selfplay.py:548`, `cr_sim/train/ladder.py:837`; unguarded `scripts/register_job.py:50` | verified | [`metrics-row.md`](metrics-row.md) |
| `config.json` | live | `cr_sim/train/run.py:605-654`, `:648`; budget `cr_sim/train/watch.py:1803`, applied `:1974` | verified | [`config-json.md`](config-json.md) |
| `Verdict` (`verdict.json`) | live | `cr_sim/train/evaluate.py:483` | verified | [`verdict.md`](verdict.md) |
| `LadderTable` (`ladder.json`) / `Arms` (`arms.json`) | live | `scripts/run_ladder.py:272`, `:401`, `:391-394`, `:414-425` | verified | [`ladder.md`](ladder.md) |
| `_ladder_ratings` | live | `cr_sim/train/run.py:445`, mode `:465`, observation `:474`, tower level `:481` | verified | [`ladder.md`](ladder.md) |
| `Player` (ladder) / `Direction` / `Pairing` / `Rating` | live | `cr_sim/train/ladder.py:129`, `:303`, `:332`, `:372` | verified | [`ladder.md`](ladder.md) |
| `play_pairing` | live | `cr_sim/train/ladder.py:562` | verified | [`ladder.md`](ladder.md) |
| `fit_ratings` | live | `cr_sim/train/ladder.py:604`, `:636-651` | verified | [`ladder.md`](ladder.md) |
| `ladder_probe` | live | `cr_sim/train/ladder.py:737`, `:772-783`; wired `cr_sim/train/run.py:971-978` | verified | [`ladder.md`](ladder.md) |
| `check_equal_branch_budget` | live | **`cr_sim/train/proposal.py:204`**, not `train/ladder.py`; called `scripts/run_ladder.py:239`, and only where both sides are search players and one is guided (`:236-238`) | verified | [`ladder.md`](ladder.md) |
| `PPOConfig` / `Rollout` / `ppo.train` / `_update` | live | `cr_sim/train/ppo.py:54`, `:98`, `:166`, `:464` | verified | [`ppo.md`](ppo.md) |
| `PPOConfig.gamma` and `clone.collect(gamma=)` | ghost | `cr_sim/train/ppo.py:69`; `cr_sim/train/clone.py:245` | verified | [`ppo.md`](ppo.md) |
| `OpponentPool` | live | `cr_sim/train/selfplay.py:337`, `add` `:369`, `sample` `:394`, `oldest` `:399`, eviction `:381-392` | verified | [`self-play.md`](self-play.md) |
| `FrozenOpponent` / `PooledOpponent` | live | `cr_sim/train/selfplay.py:212`, `:403` | verified | [`self-play.md`](self-play.md) |
| `evaluation_probe` / `ancestor_probe` | live | `cr_sim/train/selfplay.py:426`, `:488`; `cr_sim/train/run.py:922-932`, `:953-961` | verified | [`self-play.md`](self-play.md) |
| `RewardSchedule` / `KNOBS` / `SHAPING_FIELDS` / `constant_schedule` / `anneal_to_zero` / `knob_for_reward` | live | `cr_sim/train/schedule.py:110`, `:87`, `:80`, `:252`, `:257`, `:97` | verified | [`reward-schedule.md`](reward-schedule.md) |
| the push | live | `cr_sim/train/run.py:743-767`; `cr_sim/api/env.py:418-442`, `:400-407`; `cr_sim/api/vec.py:419-439`, `:244-255` | verified | [`reward-schedule.md`](reward-schedule.md) |
| `reward_schedule` in `config.json` | live | `cr_sim/train/run.py:646-649`; budget `cr_sim/train/watch.py:1803`, applied `:1974` | verified | [`reward-schedule.md`](reward-schedule.md) |
| **`explained_variance`** | live | `cr_sim/train/ppo.py:416-420`, `:140`; ceiling `cr_sim/train/schedule.py:1-14` | verified | [`explained-variance.md`](explained-variance.md) |
| `ret_std` and the per-row `reward_weights` | live | `cr_sim/train/ppo.py:408`; `cr_sim/train/run.py:769-779` | verified | [`explained-variance.md`](explained-variance.md) |
| `clone`'s own `explained_variance` | live | `cr_sim/train/clone.py:628-631`, `:657` | verified | [`explained-variance.md`](explained-variance.md) |
| `SearchBotConfig` / `SearchBot` | live | `cr_sim/train/scripted.py:76`, `:178` | verified | [`search-bot.md`](search-bot.md) |
| `battle_stream_seed` | live | `cr_sim/train/scripted.py:52` | verified | [`search-bot.md`](search-bot.md) |
| `policy_proposer` / `proposer_identity` | live | `cr_sim/train/proposal.py:80`, `:173`, stream `:144-146`, T=0 argsort `:133-137` | verified | [`search-bot.md`](search-bot.md) |
| `policy_proposer(battle_seed_of=...)` | ghost | `cr_sim/train/proposal.py:81`, doc `:101-105`, default `:144` | verified | [`search-bot.md`](search-bot.md) |
| **`RandomStreamOwnership`** | live | this map; sources on the four rows below | verified | [`random-streams.md`](random-streams.md) |
| the five closed by `8fbe4a5`: worker self-play generator, PPO minibatch shuffle (required, not defaulted), `evaluation_probe`, `ancestor_probe` both sides, and `evaluate_paired` keyed on **mode** rather than index in `modes` | live | `cr_sim/api/vec.py:234-242`; `cr_sim/train/ppo.py:222`, `:471-483`; `cr_sim/train/selfplay.py:467`, `:532-534`; `cr_sim/train/evaluate.py:457-459` | verified | [`random-streams.md`](random-streams.md) |
| also owned: `ladder._stream_seed`, `rotating_probe`, `evaluation_seeds`, `battle_stream_seed`, `policy_proposer`, the clone holdout split, and the random control arm's `default_rng(0)` | live | `cr_sim/train/evaluate.py:555`, `:592-611`; wired `cr_sim/train/run.py:979-984` | verified | [`random-streams.md`](random-streams.md) |
| **R1** | live | `scripts/clone_policy.py:316-317`; falls to `cr_sim/train/evaluate.py:205-206`; advanced by `cr_sim/train/clone.py:575`, `:598` | verified | [`random-streams.md`](random-streams.md) |
| **R2** | live | `cr_sim/train/selfplay.py:403-411`, `:256-262`, `:329-330` | verified | [`random-streams.md`](random-streams.md) |
| **R3** | live | `cr_sim/api/vec.py:198`, `:223`, `:330-333`; `cr_sim/train/run.py:1013` | verified | [`random-streams.md`](random-streams.md) |
| **R4** | live | `cr_sim/api/env.py:385`, `:695` | verified | [`random-streams.md`](random-streams.md) |
| `_resolve_device` | live | `cr_sim/train/run.py:350` | verified | [`ghost-knobs.md`](ghost-knobs.md) |
| `--device xpu` | ghost | `cr_sim/train/run.py:363-385` | verified | [`ghost-knobs.md`](ghost-knobs.md) |
| the gamma-correct potential | ghost | `docs/training.md:483-505` | verified | [`ghost-knobs.md`](ghost-knobs.md) |
| `--probe rotating` | live | `cr_sim/train/run.py:96-107`, `:979-984` | verified | [`ghost-knobs.md`](ghost-knobs.md) |
| `--reward simple` / `--shaping` | leftover | `cr_sim/train/run.py:229-237` | verified | [`ghost-knobs.md`](ghost-knobs.md) |
| `--opponent idle` | leftover | `cr_sim/train/run.py:316-321`; `cr_sim/train/evaluate.py:701-709` | verified | [`ghost-knobs.md`](ghost-knobs.md) |
| `scripts/make_demos.py` | live | `scripts/make_demos.py:190-210`, `:330`; default observation `cr_sim/api/env.py:319` | verified | [`demonstrations.md`](demonstrations.md) |
| `collapse_refusal` | live | `scripts/make_demos.py:404`, `:436-443`, `:446` | verified | [`demonstrations.md`](demonstrations.md) |
| entry points: `python -m cr_sim.train.run`, `python -m cr_sim.train.evaluate`, `scripts/clone_policy.py`, `scripts/run_ladder.py`, `scripts/evaluate_vs_expert.py`, `scripts/evaluate_checkpoints.py`, `scripts/measure_expert.py`, `scripts/make_demos.py`, `scripts/expert_iterate.py`. Every one prepends `parents[1]` to `sys.path`, so `cr_sim` is imported **by repo position** and a script moved one directory deeper imports a different package or none | live | `cr_sim/train/run.py:510`; `cr_sim/train/evaluate.py:692`; `scripts/clone_policy.py:219`; `scripts/run_ladder.py:143`; `scripts/evaluate_vs_expert.py:60`; `scripts/evaluate_checkpoints.py:25` (module level, no `main`); `scripts/measure_expert.py:54`; `scripts/make_demos.py:266`; `scripts/expert_iterate.py:156`; the path insert e.g. `scripts/make_demos.py:27` | verified | [`../../processes/CONTEXT.md`](../../processes/CONTEXT.md) |
| `scripts/expert_iterate.py` | live | `scripts/expert_iterate.py:156`, `:171-186`, `:193` | verified | [`ghost-knobs.md`](ghost-knobs.md) |

---
