# battle — what happens inside a match, and what makes it reproducible

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
| `Battle` | live | `cr_sim/engine/battle.py:239` | verified | [`battle.md`](battle.md) |
| `Battle.PHASES` | live | `cr_sim/engine/battle.py:243-263` | verified | [`battle.md`](battle.md) |
| `Battle.step` | live | `cr_sim/engine/battle.py:754` | verified | [`battle.md`](battle.md) |
| `BattleConfig` | live | `cr_sim/engine/battle.py:210` | verified | [`battle-config.md`](battle-config.md) |
| `BattleConfig.tower_level` | live | `cr_sim/engine/battle.py:222`; used `:470` | verified | [`battle-config.md`](battle-config.md) |
| `BattleConfig.level` | live | `cr_sim/engine/battle.py:221`; used `:526` | verified | [`battle-config.md`](battle-config.md) |
| `BattleResult` — winner, crowns, ticks, `reason` | live | `cr_sim/engine/battle.py:231` | stub | — |
| `Player` | live | `cr_sim/engine/battle.py:144` | verified | [`battle.md`](battle.md) |
| `Battle.play_card` | live | `cr_sim/engine/battle.py:538` | verified | [`battle.md`](battle.md) |
| `Battle.clone` / `_SHARED` / `_HISTORIES` | live | `cr_sim/engine/battle.py:689`, `:680`, `:687` | verified | [`battle-clone.md`](battle-clone.md) |
| `Battle.graveyard` / `Battle.damage_log` | live | `cr_sim/engine/battle.py:372`, `:358` | verified | [`battle-clone.md`](battle-clone.md) |
| `Battle.frames` / `_capture_frame` — never hashed, cosmetic by construction, so recording cannot change an outcome | live | `cr_sim/engine/battle.py:410`, `:789` | stub | — |
| `Battle._towers` | live | `cr_sim/engine/battle.py:356`, `:486` | verified | [`battle.md`](battle.md) |
| `_refresh_occupancy` / `_occupancy_signature` — rebuilds path-grid building costs only when the standing-structure signature changes; was 40% of throughput, and keys on ids | live | `cr_sim/engine/battle.py:1957`, `:395` | stub | — |
| the module docstring at `cr_sim/engine/battle.py:16` still calls targeting, combat, projectiles and collision "stubs". All four are complete phases (`:1078`, `:1133`, `:1539`, `:2119`). It is M1-era and wrong | live | `cr_sim/engine/battle.py:16` | verified | [`battle.md`](battle.md) |
| `Entity` | live | `cr_sim/engine/entity.py:134` | verified | [`entity.md`](entity.md) |
| `Entity.__deepcopy__` / `_slots_for` | live | `cr_sim/engine/entity.py:275`, `:123` | verified | [`entity-copy.md`](entity-copy.md) |
| `Entity.clone` — zero callers; the Clone spell builds a fresh `Entity` instead. **Its docstring says otherwise; the code wins** | ghost | `cr_sim/engine/entity.py:256`; `cr_sim/engine/battle.py:2610` | verified | [`entity-copy.md`](entity-copy.md) |
| `Team` / `EntityKind` / `EntityState` | live | `cr_sim/engine/entity.py:36`, `:47`, `:55` | verified | [`entity.md`](entity.md) |
| `next_entity_id` | live | `cr_sim/engine/entity.py:74` | verified | [`entity-ids.md`](entity-ids.md) |
| `entity_id_cursor` / `restore_entity_ids` | live | `cr_sim/engine/entity.py:86`, `:99` | verified | [`entity-ids.md`](entity-ids.md) |
| `reset_entity_ids` | live | `cr_sim/engine/entity.py:109`; called `cr_sim/engine/battle.py:349` | verified | [`entity-ids.md`](entity-ids.md) |
| `Rng` (PCG-XSH-RR 32) / `Rng.stream(label)` | live | `cr_sim/engine/rng.py:29`, `:86` | verified | [`rng.md`](rng.md) |
| the engine draws from exactly two streams: `deck:{team.name}` and `aeospawn:{effect.id}`. Nothing else in `engine/` draws | live | `cr_sim/engine/battle.py:436`, `:1643` | verified | [`rng.md`](rng.md) |
| `Rng.state` / `.restore` / `.chance` / `.between` — defined, documented, exercised only by `tests/test_engine_core.py:162`; no caller in `cr_sim/` | ghost | `cr_sim/engine/rng.py:70`, `:76`, `:98`, `:102` | stub | — |
| `state_hash` | live | `cr_sim/replay.py:38` | verified | [`state-hash.md`](state-hash.md) |
| `Battle.hash` | live | `cr_sim/engine/battle.py:841` | verified | [`state-hash.md`](state-hash.md) |
| `Command` | live | `cr_sim/replay.py:73` | verified | [`replay.md`](replay.md) |
| `Replay` / `DivergenceError` / `compare_hashes` | leftover | `cr_sim/replay.py:97`, `:34`, `:165`; claim `:3`, omission `:127-140` | verified | [`replay.md`](replay.md) |
| `Projection` / `project` / `elixir_advantage` / `committed_value` | live | `cr_sim/engine/lookahead.py:31`, `:109`, `:155`, `:169` | verified | [`lookahead.md`](lookahead.md) |
| `Arena` / `load_arena` | live | `cr_sim/engine/arena.py:109`, `:473` | verified | [`arena.md`](arena.md) |
| `Tile` (IntFlag) | live | `cr_sim/engine/arena.py:57` | verified | [`arena.md`](arena.md) |
| `Tile.MARKER` / `Tile.CENTRE` | leftover | `cr_sim/engine/arena.py:72`, `:76`; `cr_sim/engine/pathgrid.py:147` | verified | [`arena.md`](arena.md) |
| `_WATER` / `_BLOCKED` / `_LANE_MASK` — plain-int mirrors of the flags, because `int & IntFlag` dispatches to enum machinery and cost microseconds per `is_walkable` call | live | `cr_sim/engine/arena.py:90-95` | stub | — |
| `TowerPlacement` | live | `cr_sim/engine/arena.py:99`; mirror `:440-468` | verified | [`arena.md`](arena.md) |
| `Arena.is_road` — never called | ghost | `cr_sim/engine/arena.py:212` | stub | — |
| `PathGrid` / `PATH_COSTS` / `load_path_costs` | live | `cr_sim/engine/pathgrid.py:82`, `:43`, `:72` | verified | [`pathing.md`](pathing.md) |
| `flow_field` / `GOAL_SNAP` / `MAX_FIELDS` | live | `cr_sim/engine/pathgrid.py:329`, `:321`, `:326` | verified | [`pathing.md`](pathing.md) |
| `find_path` (A*) / `_DIAGONAL_HALVES` / `costs["heuristic"]` | leftover | `cr_sim/engine/pathgrid.py:216`, `:69` | verified | [`pathing.md`](pathing.md) |
| `next_cell` / `field_path` / `simplify` | live | `cr_sim/engine/pathgrid.py:390`, `:415`, `:290` | stub | — |
| `Route` / `route_to` | live | `cr_sim/engine/pathing.py:39`, `:105` | verified | [`pathing.md`](pathing.md) |
| `crosses_river` / `line_blocked` / `step_towards` | live | `cr_sim/engine/pathing.py:239`, `:162`, `:222` | stub | — |
| `SpatialIndex` / `DEFAULT_CELL` | live | `cr_sim/engine/spatial.py:36`, `:33` | verified | [`pathing.md`](pathing.md) |
| `AttackState` | live | `cr_sim/engine/combat.py:54` | verified | [`attack-state.md`](attack-state.md) |
| `ramp_damage` | live | `cr_sim/engine/combat.py:163` | verified | [`attack-state.md`](attack-state.md) |
| `PendingHit` / `DamageEvent` / `apply_hit` / `apply_area_damage` / `advance_attack` / `damage_for` | live | `cr_sim/engine/combat.py:199`, `:129`, `:265`, `:281`, `:216`, `:139` | verified | [`attack-state.md`](attack-state.md) |
| `apply_multiplier` / `as_delta` / `apply_delta` | live | `cr_sim/engine/buffs.py:145`, `:191`, `:173` | verified | [`buff-percent.md`](buff-percent.md) |
| `BuffSpec` / `BuffState` / `ActiveBuff` / `BuffTick` / `build_buff_spec` | live | `cr_sim/engine/buffs.py:221`, `:462`, `:439`, `:425`, `:328` | verified | [`buff-percent.md`](buff-percent.md) |
| `ProjectileSpec` / `Projectile` / `RollingProjectile` / `flight_ticks` — flight in whole ticks; the Log rolls with its own axis-aligned push | live | `cr_sim/engine/projectiles.py:49`, `:188`, `:273`, `:108` | stub | — |
| `AreaEffectSpec` / `AreaEffect` / `build_area_effect_spec` — persistent ground effects; `on_hit_action` fires per touched entity | live | `cr_sim/engine/areaeffects.py:43`, `:226`, `:180` | stub | — |
| `AreaEffectSpec.is_instant` — never called | ghost | `cr_sim/engine/areaeffects.py:94` | stub | — |
| `SpellPlan` / `plan_spell` — waves times projectiles per cast; Arrows is 3 waves of 10 | live | `cr_sim/engine/spells.py:41`, `:91` | stub | — |
| `SpellPlan.is_waved` / `.does_nothing` — documented as "reported rather than silently ignored"; nothing reports them. `is_scattered` *is* used | ghost | `cr_sim/engine/spells.py:65`, `:73` | stub | — |
| `separate` / `resolve_collisions` / `IMMOVABLE_MASS` — mass-weighted push-apart, structures immovable | live | `cr_sim/engine/movement.py:68`, `:134`, `:40` | stub | — |
| `acquire_target` / `should_keep_target` / `can_target` / `in_attack_range` | live | `cr_sim/engine/targeting.py:147`, `:206`, `:96`, `:131` | verified | [`targeting.md`](targeting.md) |
| `gap_between` / `within_gap` | live | `cr_sim/engine/targeting.py:55`, `:72` | verified | [`targeting.md`](targeting.md) |
| `in_sight_range` — in `__all__`, called nowhere; the logic is inlined at `cr_sim/engine/targeting.py:180-190` and `cr_sim/engine/battle.py:1119` | ghost | `cr_sim/engine/targeting.py:143` | stub | — |
| `nearest_structure` — not in `__all__`, zero references anywhere | ghost | `cr_sim/engine/targeting.py:229` | stub | — |
| `point_along` — movement is a position derived from a running travel total, not a per-tick delta, which bounds truncation error at one subtile instead of accumulating | live | `cr_sim/engine/fixed.py:108` | stub | — |
| `push_away` — every knockback, and deliberately **not** `point_along`, which clamps and made every push a silent no-op | live | `cr_sim/engine/fixed.py:132` | stub | — |
| `pack_offsets` / `ring_offsets` — `ring_offsets` uses the card's `SummonRadius`; `pack_offsets` is a derived default for the four cards that ship none | live | `cr_sim/engine/fixed.py:172`, `:217` | stub | — |
| `distance` / `distance_squared` / `within_range` / `clamp` — `distance` uses `math.isqrt`, exact, no float | live | `cr_sim/engine/fixed.py:80`, `:74`, `:96`, `:168` | stub | — |
| `circles_overlap` — exported in `__all__`, called by nothing in `cr_sim/`, `tests/` or `scripts/` | ghost | `cr_sim/engine/fixed.py:101` | stub | — |
| `BattleTimeline` / `build_timeline` / `ElixirSegment` | live | `cr_sim/engine/elixir.py:52`, `:143`, `:38` | verified | [`elixir.md`](elixir.md) |
| `ElixirBar` | live | `cr_sim/engine/elixir.py:92` | verified | [`elixir.md`](elixir.md) |
| `BattleTimeline.elixir_gain_per_tick` | ghost | `cr_sim/engine/elixir.py:75` | verified | [`elixir.md`](elixir.md) |
| `ActionInterpreter` | live | `cr_sim/engine/actions.py:482` | verified | [`action-interpreter.md`](action-interpreter.md) |
| `_HANDLERS` | live | `cr_sim/engine/actions.py:1548` | verified | [`action-interpreter.md`](action-interpreter.md) |
| `ActionInterpreter.unsupported` | live | `cr_sim/engine/actions.py:537` | verified | [`action-interpreter.md`](action-interpreter.md) |
| `evaluate_expression` / `ExpressionError` | live | `cr_sim/engine/actions.py:197`, `:115` | verified | [`action-interpreter.md`](action-interpreter.md) |
| `ActionContext` / `expression_scope` | live | `cr_sim/engine/actions.py:306`, `:322-342` | verified | [`action-interpreter.md`](action-interpreter.md) |
| `_handle_select` (`ActionSelect`) | live | `cr_sim/engine/actions.py:871`, `:874` | verified | [`action-select.md`](action-select.md) |
| `ActionSelect` with `Condition = "rand(n)"` | deliberate ghost | `cr_sim/engine/actions.py:73-78`, `:902-907` | verified | [`action-select-rand.md`](action-select-rand.md) |
| `ActionTaunt` | deliberate ghost | `cr_sim/engine/actions.py:59-64` | verified | [`action-taunt.md`](action-taunt.md) |
| `ActionGroundToAir` | deliberate ghost | `cr_sim/engine/actions.py:79-82` | verified | [`action-ground-to-air.md`](action-ground-to-air.md) |
| `ActionChangeGameObjectData` + `NewProjectileData` | deliberate ghost | `cr_sim/engine/actions.py:66-72`, `:1408-1412` | verified | [`action-change-data-projectile.md`](action-change-data-projectile.md) |
| `matches_filter` / `_Pending` / the per-handler functions | live | `cr_sim/engine/actions.py:368`, `:476`, `:789-1340` | stub | — |

---
