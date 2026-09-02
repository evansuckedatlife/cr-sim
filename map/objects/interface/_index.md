# interface — what the agent sees, may do, and is paid

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
| `ObservationFeatures` | live | `cr_sim/api/encoding.py:177` | verified | [`observation-features.md`](observation-features.md) |
| `OBSERVATION_V1` | live | `cr_sim/api/encoding.py:215` | verified | [`observation-features.md`](observation-features.md) |
| `OBSERVATION_V2` | live | `cr_sim/api/encoding.py:230`, rationale `:219-229` | verified | [`observation-features.md`](observation-features.md) |
| `OBSERVATION_V3` | live | `cr_sim/api/encoding.py:244` | verified | [`observation-features.md`](observation-features.md) |
| `parse_observation` | live | `cr_sim/api/encoding.py:251`, `:275` | verified | [`observation-features.md`](observation-features.md) |
| `GRID_FEATURE_CHANNELS` | live | `cr_sim/api/encoding.py:288`; iterated `:302-307` | verified | [`observation-grid.md`](observation-grid.md) |
| `grid_channels()` | live | `cr_sim/api/encoding.py:295` | verified | [`observation-grid.md`](observation-grid.md) |
| `_encode_grid` / `_terrain_channel` | live | `cr_sim/api/encoding.py:735`, `:693`, `:818-827` | verified | [`observation-grid.md`](observation-grid.md) |
| `HP_NORM` 6000 / `COUNT_NORM` 4 / `SPELL_NORM` 1000 / `DPS_NORM` 800 / `REACH_NORM` 12 | live | `cr_sim/api/encoding.py:122`, `:127`, `:133`, `:139`, `:144` | verified | [`observation-grid.md`](observation-grid.md) |
| `GRID_CHANNELS` / `N_GRID_CHANNELS` — v1-only aliases with test-only consumers | leftover | `cr_sim/api/encoding.py:312`, `:313` | stub | — |
| `EncodingConfig` | live | `cr_sim/api/encoding.py:320` | verified | [`encoding-config.md`](encoding-config.md) |
| `EncodingConfig.vocab` | live | `cr_sim/api/encoding.py:337`; built `:373` | verified | [`encoding-config.md`](encoding-config.md) |
| `build_encoding_config` / `observation_shapes` | live | `cr_sim/api/encoding.py:365`, `:389` | verified | [`encoding-config.md`](encoding-config.md) |
| `_encode_vector` | live | `cr_sim/api/encoding.py:853` | verified | [`observation-vector.md`](observation-vector.md) |
| `hand_onehot_layout` | live | `cr_sim/api/encoding.py:401`, `:415`; guard `tests/test_action_head.py:83-104` | verified | [`observation-vector.md`](observation-vector.md) |
| `_vector_length` | live | `cr_sim/api/encoding.py:419` | verified | [`observation-vector.md`](observation-vector.md) |
| `_card_features` / `encode_observation` | live | `cr_sim/api/encoding.py:835`, `:886` | verified | [`observation-vector.md`](observation-vector.md) |
| `_team_towers` / `total_tower_hitpoints` | live | `cr_sim/api/encoding.py:467`, `:506` | verified | [`observation-vector.md`](observation-vector.md) |
| `NUM_CARD_SLOTS` = 5 / `NOOP_SLOT` = 4 | live | `cr_sim/api/encoding.py:107`, `:108` | verified | [`action-mask.md`](action-mask.md) |
| `OBS_TILE_SPAN` = 1 / `PLACEMENT_TILE_SPAN` = 2 | live | `cr_sim/api/encoding.py:112`, `:114` | verified | [`action-mask.md`](action-mask.md) |
| `legal_action_mask` | live | `cr_sim/api/encoding.py:586`, `:616`, `:624` | verified | [`action-mask.md`](action-mask.md) |
| `decode_action` | live | `cr_sim/api/encoding.py:523` | verified | [`action-mask.md`](action-mask.md) |
| `_placement_grid` — `lru_cache(128)` keyed including `fallen_enemy_towers`; returns a read-only array | live | `cr_sim/api/encoding.py:643` | stub | — |
| `cell_to_world` — `y` mirrored for RED, `x` never | live | `cr_sim/api/encoding.py:429` | stub | — |
| `action_grid_shape` — test-only consumer | leftover | `cr_sim/api/encoding.py:384` | stub | — |
| `_can_deploy_cached` — 36 lines of docstring and an `lru_cache(4096)` with **zero callers anywhere in the tree**; `_placement_grid` calls `arena.can_deploy` directly | ghost | `cr_sim/api/encoding.py:549` | stub | — |
| the three reward variants, selected **by the type of `reward_weights`**: `None` gives the inline shaped reward, `RewardWeights` the five-term tracker, `ProjectionWeights` the projected one. `None` is a value, not "unset" | live | `cr_sim/api/env.py:212-226` | verified | [`reward-variants.md`](reward-variants.md) |
| `RewardWeights` / `RewardTracker` | live | `cr_sim/api/reward.py:62`, `:114`, `:129-132`, `:230-248`, `:333-335` | verified | [`reward-variants.md`](reward-variants.md) |
| `ProjectionWeights` / `ProjectedReward` | live | `cr_sim/api/reward.py:284`, `:319`, `:340-347`; read `cr_sim/api/env.py:479`, `:555` | verified | [`reward-variants.md`](reward-variants.md) |
| `RewardTracker.score` is `sum(terms.values())`; `ProjectedReward.score` is **not** | live | `cr_sim/api/reward.py:211`, `:386-392` | verified | [`reward-variants.md`](reward-variants.md) |
| `reward_shaping_weight` | live | `cr_sim/api/env.py:263-281`; `cr_sim/api/vec.py:77` | verified | [`reward-variants.md`](reward-variants.md) |
| `as_dict` on both weight classes | live | `cr_sim/api/reward.py:80`, `:307-316`; read `cr_sim/train/selfplay.py:178-185` | verified | [`reward-variants.md`](reward-variants.md) |
| `reward.describe` — zero callers; the name collides with two unrelated `describe` functions | ghost | `cr_sim/api/reward.py:395` | stub | — |
| `reward.__all__` — omits `ProjectionWeights`, `ProjectedReward` and `describe`, all three imported by name elsewhere | leftover | `cr_sim/api/reward.py:58` | stub | — |
| `CRSimEnv` | live | `cr_sim/api/env.py:245` | verified | [`crsim-env.md`](crsim-env.md) |
| `CRSimEnv.encoding` | live | `cr_sim/api/env.py:372` | verified | [`crsim-env.md`](crsim-env.md) |
| `skip_forced` | live | `cr_sim/api/env.py:318`, `:349`, `:462`; absent from `cr_sim/api/vec.py:60-113` | verified | [`crsim-env.md`](crsim-env.md) |
| `_run_out_forced_decisions` / `_MAX_FORCED_RUN_OUT` | live | `cr_sim/api/env.py:528`, `:151` | verified | [`crsim-env.md`](crsim-env.md) |
| `_apply_action` | live | `cr_sim/api/env.py:168-184`, `:514-516`, `:562` | verified | [`crsim-env.md`](crsim-env.md) |
| `set_reward_weights` (pending-at-reset) / `idle_opponent_policy` / `wants_battle` duck-flag | live | `cr_sim/api/env.py:418`, `:154`, `:522` | stub | — |
| `HAS_GYMNASIUM` and the `Box` / `MultiDiscrete` / `DictSpace` shims | live | `cr_sim/api/env.py:51-114` | stub | — |
| `CRSimEnv.reset(seed=None)` | live | `cr_sim/api/env.py:385`, `:695` | verified | [`random-streams.md`](../measurement/random-streams.md) |
| `CRSimSelfPlayEnv` | leftover | `cr_sim/api/env.py:610`, `:639-646`, `:677` | verified | [`crsim-env.md`](crsim-env.md) |
| `VecEnvConfig` | live | `cr_sim/api/vec.py:61` | verified | [`vec-env-config.md`](vec-env-config.md) |
| `VecEnvConfig.tower_level` / `.observation` / `.seed` | live | `cr_sim/api/vec.py:76`, `:113`, `:101` | verified | [`vec-env-config.md`](vec-env-config.md) |
| `VecEnvConfig.level` (card level) | ghost | `cr_sim/api/vec.py:75`; also `cr_sim/api/env.py:312`, `:659`, `cr_sim/data/card_features.py:164` | verified | [`vec-env-config.md`](vec-env-config.md) |
| divergent defaults for one knob: `frame_skip` is 30 in `CRSimEnv`, 6 in `VecEnvConfig`, 10 in argparse; `ticks_per_second` 60/60/20; `tower_level` 11 everywhere in `run.py` but 5 in all eight `scripts/` | live | `cr_sim/api/env.py:145`, `cr_sim/api/vec.py:74`, `cr_sim/train/run.py:122` | verified | [`vec-env-config.md`](vec-env-config.md) |
| the "field for field" parity test is not field-for-field: it asserts six hand-written fields, does not iterate `dataclasses.fields`, and **does not assert `observation`**. **Code wins over its own docstring** | live | `tests/test_train.py:563-628` | verified | [`vec-env-config.md`](vec-env-config.md) |
| `CRSimVecEnv` / `_worker` / `_build_env` / the RPC verbs / per-worker seed derivation | live | `cr_sim/api/vec.py:281`, `:147`, `:116`, `:327-332` | verified | [`vec-env-config.md`](vec-env-config.md) |
| `WorkerDied` / `_WORKER_TIMEOUT` | live | `cr_sim/api/vec.py:265`, `:278` | stub | — |
| `NetConfig` | live | `cr_sim/train/nets.py:57` | verified | [`net-config.md`](net-config.md) |
| `net_config_for` | live | `cr_sim/train/nets.py:191` | verified | [`net-config.md`](net-config.md) |
| the seven `NetConfig` fields no entry point can set | leftover | `train/nets.py:67,68,81,137,139,144,163` | verified | [`net-config.md`](net-config.md) |
| `NetConfig.card_stats` | live | `cr_sim/train/nets.py:149`, `:225`, `:418` | verified | [`net-config.md`](net-config.md) |
| `POLICY_HEADS` / flat head / `ActorCritic` / trunk / `_SPATIAL_DEPTH` | live | `cr_sim/train/nets.py:53`, `:561`, `:521`, `:529-548`, `:574` | verified | [`policy-heads.md`](policy-heads.md) |
| `FactoredHead` / `card_embedding` | live | `cr_sim/train/nets.py:244`, `:283`, `:298-316` | verified | [`policy-heads.md`](policy-heads.md) |
| `FactoredStatsHead` | live | `cr_sim/train/nets.py:344` | verified | [`policy-heads.md`](policy-heads.md) |
| `ConvPlacementHead` | live | `cr_sim/train/nets.py:470`, `:494-502`, `:517` | verified | [`policy-heads.md`](policy-heads.md) |
| `MASKED_LOGIT` / `_apply_mask` | live | `cr_sim/train/nets.py:675`, `:678` | verified | [`policy-heads.md`](policy-heads.md) |
| `masked_categorical` | ghost | `cr_sim/train/nets.py:684` | verified | [`policy-heads.md`](policy-heads.md) |
| `CARD_FEATURE_NAMES` / `card_feature_vector` / `card_feature_table` | live | `cr_sim/data/card_features.py:228`, `:442`, `:589` | verified | [`card-features.md`](card-features.md) |
| `CARD_FEATURE_LEVEL` = 11 and a pinned `TickClock(60)` | live | `cr_sim/data/card_features.py:164` | verified | [`card-features.md`](card-features.md) |
| the mirrored `_HP_NORM` / `_DPS_NORM` / `_REACH_NORM` / `_COUNT_NORM` | live | `cr_sim/data/card_features.py:175-178`; pin `tests/test_card_features.py:92-101` | verified | [`card-features.md`](card-features.md) |
| the vocab-position hazard, written down and measured at source: permuting the vocab leaves the head invariant to 4e-07 and moves the trunk by 56% relative L2, because 80 of the 102 columns of `vector.0.weight` are per-vocab-index one-hots. The fix named there is a change to the observation, not to that file | live | `cr_sim/data/card_features.py:20-45` | verified | [`card-features.md`](card-features.md) |
| `check_observation` | live | `cr_sim/train/evaluate.py:89` | verified | [`check-observation.md`](check-observation.md) |
| no **model** artefact records the deck. All four checkpoint payloads are `state_dict`, `head`, `observation` and metrics; a demo shard records `observation`, `reward`, `proposer`. `runs/*/config.json` does record `"deck"` | live | `cr_sim/train/run.py:868-885`, `:891-895`, `:1053`; `scripts/clone_policy.py:288-302`; `cr_sim/train/run.py:606` | verified | [`check-observation.md`](check-observation.md) |

---
