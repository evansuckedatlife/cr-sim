# How to walk this map

The repo is the source of truth. This map is a catalog of its **nouns**, marked
by universe and cited `path:line`. It exists for one job: to answer *what else
moves if I change this* before the change is made.

## The walk

**This file is not a hop.** It is the shelf every card cites and none may
restate: the universes, the collisions, the units, the charter. Read the section
you were sent to.

The walk itself is two hops, and `CLAUDE.md` routes it:

1. `CLAUDE.md` → a cluster index, an `effects/` entry, or a verb card.
2. `objects/<cluster>/_index.md` → one line per noun: universe, owning
   `path:line`, card. Enough to reach source in one hop even where no card
   exists — and where the Card column reads `—`, that line **is** the entry,
   deliberately.
3. The card.

If you find yourself opening a third map file to answer one question about the
repo, the map has failed and the fix is to move a file, not to add a fourth.

**Never read a cluster index in bulk from outside its cluster**, and never read
`objects/` whole. The indexes are shelf lists on purpose: a row with a card
carries no gloss, because the gloss lives on the card.

## Citation root — one root, no exceptions

Every `path:line` in this map resolves against the **repo root**:
`cr_sim/train/run.py:125`, never the shortened *train/run.py:125*. There is
no `train/`,
`engine/`, `api/` or `data/` directory at the root, so a shortened citation
resolves only by suffix matching — and suffix matching also resolves into
`.claude/worktrees/agent-*/`, which is three other checkouts of this repo from
older eras (`objects/surfaces/worktree-shadows.md`). A bare `:NNN` continues the
file last named in the same paragraph; that is the only abbreviation allowed.

`python map/_meta/check.py` enforces it, refuses the suffix fallback by name,
and additionally asserts that a line cited beside a symbol falls **inside** that
symbol.

## Verification basis

Everything cited here was checked against `main` @ `dc47f51` on 2026-08-30 with
an **uncommitted working tree**. Ten files are modified at that commit, and the
root `CLAUDE.md` — carrying only this map's routing row — is one of them. Line
numbers in these **six** are working-tree numbers, not `dc47f51` numbers:

`cr_sim/train/clone.py`, `cr_sim/train/evaluate.py`, `cr_sim/train/ladder.py`,
`cr_sim/train/scripted.py`, `scripts/make_demos.py`, `docs/training.md`
(plus `tests/test_clone.py`, `tests/test_proposal.py`, `tests/test_train.py`).

A card citing any of them must re-verify before it may claim `verified`, and
must say in its own `Citations` block that it did. `python map/_meta/check.py`
resolves every citation in this map against the working tree and asserts each
falls inside the symbol it is named beside; it does not know the difference
between the tree and the commit, so this paragraph is the part a person owns.

## The three universes

| Universe | Meaning | What you may do |
|---|---|---|
| **live** | In force. | Implement and cite against it. |
| **leftover** | Still present, no longer the main path. | Touch only if that path is in scope. Do not delete on sight: a leftover here is usually kept alive by a test that is the only thing checking a real invariant. |
| **ghost** | Named or filed, not wired — dead functions, unreachable branches, knobs no entry point can set, documented plans that were reverted. | **Do not implement against it.** Do not "fix" it into existence because a docstring describes it working. |

A fourth universe, spelled with a space: **deliberate ghost** — something
present in the game build that this engine declines to implement, with the
reason written at the declining site. **Implementing one without reading its
reason removes a tripwire.** There are exactly four, the reasons are written
together at `cr_sim/engine/actions.py:56-82`, and each has its own card because
a `universe:` label has to be true of the whole file it sits on:

| Declined | Because | Card |
|---|---|---|
| `ActionTaunt` | this engine has no target lock to clear; left absent so a future taunt mechanic trips the coverage gate instead of inheriting a silent no-op | [`objects/battle/action-taunt.md`](objects/battle/action-taunt.md) |
| `ActionChangeGameObjectData` with `NewProjectileData` | damage is resolved *from* a projectile at spec-build time, so swapping one means rebuilding the damage, and the two users give no way to check it | [`objects/battle/action-change-data-projectile.md`](objects/battle/action-change-data-projectile.md) |
| `ActionSelect` with `Condition = "rand(n)"` | the interpreter is given no random stream on purpose; always taking branch zero would make the Spell Cauldron a Lightning dispenser | [`objects/battle/action-select-rand.md`](objects/battle/action-select-rand.md) |
| `ActionGroundToAir` | its only user is a hero form whose ability graph is not wired up at all | [`objects/battle/action-ground-to-air.md`](objects/battle/action-ground-to-air.md) |

### The three ghosts the brief names, re-checked against source

- **`--device xpu` — ghost, confirmed.** Reports available, runs a gradient step
  6.6x faster than eight CPU threads, then fails a real training loop three
  ways. Deliberately never chosen by `auto`. Refusal at
  `cr_sim/train/run.py:363-371`; why `auto` skips it at `:373-385`.
- **The gamma-correct potential `r = yPhi(s') - Phi(s)` — ghost, confirmed.**
  Written and reverted once. Half the stated reason is now false: `projected`
  measures 1.038 score calls per decision, not 2. `docs/training.md:483-505`.
  Do not re-derive the plan from that paragraph.
- **`ActionSelect` — NOT a ghost any more. Code wins over the brief.** The
  brief's "22 definitions and does nothing anywhere" was true, and was fixed in
  `5bda200`. It now reads `SubActions` + `PerActionConditions`, first-true-wins
  with a fall-off-the-end default: `cr_sim/engine/actions.py:871-925`, with its
  own gate exception at `:646-656`. Filed **live, formerly ghost**
  ([`objects/battle/action-select.md`](objects/battle/action-select.md)). What
  remains a deliberate ghost is one spelling of it, and that spelling has its
  own card — see the table above.

Every other ghost in the index was checked the same way.

## Name collisions — stated once, here

A card may cite these. It may not restate them.

### Product word vs code name

At least six cards ship under a name the build does not use. The dictionary is
`cr_sim/data/interactions.py:114-252`, built from the build's own
`HighresImageFilename` art rather than from name similarity.

| Product | In the build |
|---|---|
| Bandit | `Assassin` |
| Executioner | `AxeMan` |
| Mother Witch | `WitchMother` |
| Night Witch | `DarkWitch` |
| Sparky | `ZapMachine` |
| Flying Machine | `DartBarrell` |

### Same word, different noun

| Word | Means | And also means |
|---|---|---|
| **Card** | the deck entry: `Rarity`, `ManaCost` — `cr_sim/data/cards.py:55` | *not* the character it summons, which owns `Hitpoints` and `Damage`. `Goblins` is one card and three `Goblin_Stab` characters. |
| **level** | `BattleConfig.level`, a **displayed** 1-15 level, converted per rarity by `RarityScale.internal_level` (`cr_sim/data/leveling.py:92`) at `cr_sim/engine/battle.py:526` | `BattleConfig.tower_level`, a **raw** tower level fed straight to `TowerScale` at `cr_sim/engine/battle.py:470`. Different ladder, no display conversion, both default to 11. |
| **AEO** | a build namespace, "Area Effect Object" — `cr_sim/data/source.py:35` | `AreaEffect`, the live entity (`cr_sim/engine/areaeffects.py:226`), and `AreaEffectSpec` (`:43`). Three nouns, one abbreviation. |
| **speed** | `UnitSpec.speed`, tiles/minute, **reporting only** — `cr_sim/engine/specs.py:112` | `UnitSpec.speed_per_tick`, subtiles/tick, the only one the loop reads — `:113`. |
| **describe** | `cr_sim/api/reward.py:395`, a ghost with no callers | `cr_sim/train/bot.py:85` and `scripts/bench_engine.py:173`, both unrelated. |
| **DEFAULT_DECK** | `cr_sim/train/run.py:54` — what every training and evaluation path uses | `cr_sim/play/server.py:46` — a second, independent literal. Byte-identical today. Change one and the browser opponent's vocabulary repermutes against the checkpoint's, with no shape change and no guard. |
| **v2** | `OBSERVATION_V2`, a **frozen literal field list** — `cr_sim/api/encoding.py:230-236` | what `parse_observation` stamps on *any* flag list, always `version=2` (`:275`). A 17-channel flag list is therefore `version=2` and is refused against a v3 environment. |
| **terms** | `RewardTracker.score` **is** `sum(terms.values())` — `cr_sim/api/reward.py:211` | `ProjectedReward.score` is **not**: `terms` also carries `projected_ticks`, which is not in the score — `:386-392`. |
| **elixir weight** | the **reward's**: `ProjectionWeights.elixir` (`cr_sim/api/reward.py:300`), reached by `--elixir-weight` in `cr_sim/train/run.py:182` (default 0.3) and in `scripts/make_demos.py:115` (default **0.0**) | the **search's**: `SearchBotConfig.elixir_weight` (`cr_sim/train/scripted.py:121`, default 0.0), which decides what the bot plays and which the source says in place is *not* the reward's (`cr_sim/train/scripted.py:108-116`). `make_demos.search_config` (`:216-238`) never sets it, so that flag moves the recorded value column and not one decision. Same for `tower_weight`. |

### The axis order that reverses across one boundary

The action mask is `mask[slot, x, y]`, shape `(5, action_width, action_height)`
(`cr_sim/api/encoding.py:616`). The observation grid is `(channels, height, width)` —
`(C, 32, 18)` (`cr_sim/api/encoding.py:735`). The convolutional placement head builds
`(batch, slot, y, x)` and is transposed once, at `cr_sim/train/nets.py:517`.
Reading both files in one sitting and carrying one convention into the other is
the mistake this row exists to prevent.

## Unit conventions — stated once, here

**Three input units, one output unit.** The game files speak **milliseconds**,
**milli-tiles** (1/1000 tile) and **tiles-per-minute**. The engine speaks
**ticks** and **subtiles** (1/18000 tile), and nothing else. Conversion happens
exactly once, at spec-build time (`cr_sim/engine/specs.py`). No millisecond and
no milli-tile value may reach the tick loop.

| Rule | Where |
|---|---|
| 1 tile = 18 000 subtiles; 1 milli-tile = 18 subtiles, exact | `cr_sim/engine/fixed.py:48`, `:51` |
| ms to ticks rounds **half-up**, never truncates | `cr_sim/engine/constants.py:49-58` |
| `Speed x 5` at 60 TPS and `x 15` at 20 TPS are both exact — that is *why* a 20 TPS run is comparable to a 60 TPS one rather than an approximation of it | `cr_sim/engine/constants.py:65-79`, `cr_sim/engine/fixed.py:11-18` |
| `ChargeRange` is **hundredths of a tile**, the only distance in `characters.csv` that is not milli-tiles | `cr_sim/engine/specs.py:343` |
| `spawn_groups.toml` is **half-tiles**, and its values are grid *lines*, not cell indices: King Tower `x=18` is tile 9.0, dead centre | `cr_sim/engine/arena.py:8-15`, `cr_sim/engine/fixed.py:64` |
| `RelativeX/Y` and `MirroredX/Y` are **whole tiles**; `XPositionExpression` / `YPositionExpression` in the *same node* are milli-tiles | `cr_sim/engine/actions.py:760-780` |
| Buff percentages carry **two conventions in one column**: `<= 0` is a delta off 100, `>= 100` is the whole multiplier. Three functions, not interchangeable — `apply_multiplier(base, raw)`, `as_delta(raw)`, `apply_delta(base, summed)`. Routing a summed delta through `apply_multiplier` cancels two stacked Rages | `cr_sim/engine/buffs.py:145`, `:191`, `:173` |
| `CrownTowerDamagePercent` is a **negative delta**, applied `damage*(100+p)//100` | `cr_sim/engine/specs.py:293-301`, `cr_sim/engine/combat.py:46-51` |
| Elixir is fixed-point **thousandths**; `ElixirBar.exact` is a float and is display only, never a spend decision | `cr_sim/engine/constants.py:38`, `cr_sim/engine/elixir.py:106` |
| Every integer division **truncates toward zero**, matching the game. `distance()` uses `math.isqrt` — exact, no float | `cr_sim/data/leveling.py:110`, `cr_sim/engine/fixed.py:80-93` |
| Observation channels are all `[0,1]` by construction; signedness belongs in the reward, not the encoding. Terrain is pre-scaled and must stay the **last** channel, outside the normalisation slices | `cr_sim/api/encoding.py:18-22`, `:296-301`, `:818-827` |
| The encoder's four normalisation constants are **duplicated** into `cr_sim/data/card_features.py:175-178` rather than imported — and the copy **is** pinned, by `tests/test_card_features.py:92-101`, which asserts all four equal. This corrects an earlier audit that called it an unpinned hand-copy. | as cited |

**`18` is hard-coded at eight sites** instead of going through `milli_tiles()`
or `SUBTILES_PER_MILLI_TILE`: `engine/actions.py:324,325,362,752,999` and
`engine/battle.py:649,2606,2607`. Any change to the subtile base silently
misses all eight.

## What this map is for

Six bugs shipped here with a green test suite. Every one was a change-impact
failure: someone changed or added a thing and did not know what else it
touched. The map's charter is to make each visible **before** the change.

**This table is the only place in the map where a bug is attributed to a
cluster.** `objects/CONTEXT.md` argues the seams and cites these rows by number;
it does not re-derive them. Two derivations is how bug 1 came to have two
incompatible roots on two hub files.

| # | The failure | Cluster that must carry it | The shape to look for |
|---|---|---|---|
| 1 | `--tower-level` reached `_env()` and not `VecEnvConfig`, which defaults to 11. Every `--workers` run trained at 11 while `config.json` recorded 5 and the probe evaluated at 5. About 90% of battles drew; the agent learned from shaping alone. | interface | a config field that reaches one construction path and not another. Fixed, with the reason written in place at `cr_sim/train/run.py:996-1009` — but the *shape* is still live: `skip_forced` has no `VecEnvConfig` field at all. |
| 2 | `scripts/make_demos.py` built its env with no `reward_weights`, so demonstrations carried value targets from one reward while every fine-tune ran another. The inherited critic predicted +1.48 where returns averaged +0.47. | measurement | an artefact whose reward is not recorded inside it. Closed — the weights are now stamped from the env that actually played. |
| 3 | Demonstrations stored the already-encoded grid, so `--observation` was an unverifiable declaration about a file, written into the checkpoint and trusted by every later run. | interface + measurement | a stored tensor whose encoding is a *name* rather than something re-derivable. |
| 4 | 80 of the trunk's 102 vector columns are card-identity one-hots keyed on **vocab position**, and the vocab is the deck union rebuilt per environment. Swap decks and column *i* silently means a different card. | interface | a layout keyed on a per-environment vocabulary. Measured and written down at `cr_sim/data/card_features.py:20-45`. Still live. |
| 5 | Five random streams were unowned. Multi-worker training was never reproducible; the promotion probe returned +0.905 / +1.228 / +0.970 on identical inputs. | measurement | a draw with no named owner. Five were closed in `8fbe4a5`; the index marks what remains. |
| 6 | A lift is meaningless without **both** the control it was measured against and the reward it was counted in — the reward sits in the numerator and the denominator. Three rounds of invalid comparisons and one retracted headline. | measurement | two numbers put on one axis without establishing they share a scale. |

A card that would not have caught its bug is not finished.
