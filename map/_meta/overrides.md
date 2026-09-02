# Overrides — every place a card knowingly departs from a comment

The map's one rule is that the code wins and the card is what gets fixed. This
file is the other half of that rule: **the list of comments and docstrings the
cards have already ruled against.**

It is here, in one file, rather than distributed across 60 cards, because it is
the list a **stale-comment sweep** reads. Someone fixing docstrings needs to
know which ones are already known-wrong and what the correct reading is; that
question is asked of the whole tree at once, and an answer split across sixty
files cannot be asked at all. It is also what stops each card inventing its own
phrasing for the same act — "Code wins", "code wins over that docstring", "the
docstring is not wrong, it is narrower than it reads".

**A card that overrides a comment adds a row here and links to this file.** A
card that does not override anything says nothing about it. Where a row is
fixed in source, delete the row and the card's paragraph in the same change.

## Rows

| # | The comment | What it claims | What the code does | Card |
|---|---|---|---|---|
| 1 | `cr_sim/engine/actions.py:13` | "**Three** things had to be worked out from the files" | Four, and the docstring itself then bolds four paragraphs (`:15`, `:23`, `:33`, `:41`) | [`../objects/battle/action-interpreter.md`](../objects/battle/action-interpreter.md) |
| 2 | `cr_sim/engine/actions.py:876-879` | "every one of the **twenty-two** `ActionSelect` nodes"; "**not one** carries the `NextAction`" | 20 nodes, and exactly one carries a `NextAction` — `ACTION.InfernoDragon_EV1_UpdateAttackSequence`. The handler's own closing comment (`:925-926`) names it | [`../objects/battle/action-select.md`](../objects/battle/action-select.md) |
| 3 | `cr_sim/engine/entity.py:288-291` | `Entity.clone` "only ever runs on troops and buildings (the Clone spell's targets)" | Nothing calls `Entity.clone` at all; the Clone spell constructs a fresh `Entity` at `cr_sim/engine/battle.py:2610` | [`../objects/battle/entity-copy.md`](../objects/battle/entity-copy.md) |
| 4 | `cr_sim/engine/battle.py:16-18` | targeting, combat, projectiles and collision are "stubs that name what they will do" | All four are complete phases (`:1078`, `:1133`, `:1539`, `:2119`). M1-era paragraph | [`../objects/battle/battle.md`](../objects/battle/battle.md) |
| 5 | `cr_sim/engine/pathing.py:8-12` | weighted pathfinding is future work; the module is "the skeleton" | The grid path is live and is the default; `route_to` is the entry point | [`../objects/battle/pathing.md`](../objects/battle/pathing.md) |
| 6 | `cr_sim/engine/buffs.py:39-61` | the module stores the raw value and applies one `(100 + percent)` formula everywhere | `apply_multiplier` (`:145`) implements a **two-convention** split. The docstring is the earlier decision, kept | [`../objects/battle/buff-percent.md`](../objects/battle/buff-percent.md) |
| 7 | `cr_sim/replay.py:3` | what a `Replay` persists | It persists six things and **not** `tower_level`, so a "reproduced" battle can run its towers at the default 11 | [`../objects/battle/replay.md`](../objects/battle/replay.md) |
| 8 | `ConvPlacementHead`'s docstring | credits `decode_action` with reading the flat index | `decode_action` takes a `(slot, x, y)` sequence and never sees a flat index. `tests/test_action_head.py` is what holds the convention | [`../objects/interface/action-mask.md`](../objects/interface/action-mask.md) |
| 9 | `hand_onehot_layout`'s docstring | the offset is "derived here … so the two cannot drift apart" | `start = 2 + 1` is a literal (`cr_sim/api/encoding.py:415`). The promise holds by a different mechanism — `tests/test_action_head.py:83-104` | [`../objects/interface/observation-vector.md`](../objects/interface/observation-vector.md) |
| 10 | `_team_towers`' docstring | `env.py`'s reward reads through it "rather than maintaining their own notion of the towers" | True only of the simple shaped reward; `RewardTracker._observe` reads `battle._towers[...]` directly (`cr_sim/api/reward.py:217-222`) | [`../objects/interface/observation-vector.md`](../objects/interface/observation-vector.md) |
| 11 | `tests/test_observation_v2.py:387` | a newly added set means "`v2` contains it" | Its assertion at `:404` checks `OBSERVATION_V3`, correctly — v2 is frozen | [`../objects/interface/observation-features.md`](../objects/interface/observation-features.md) |
| 12 | the masked-logit assert's comment | an assert is what prevents the NaN | Five call sites construct a `Categorical` directly, without it. `MASKED_LOGIT = -1e8` plus the always-legal noop cell is what prevents it | [`../objects/interface/policy-heads.md`](../objects/interface/policy-heads.md) |
| 13 | `cr_sim/train/schedule.py:18-20` | every `_shaped_value` call site sits inside that `else` | Three of five do; two are in `CRSimSelfPlayEnv`, which has no reward object and no such `if`. The conclusion holds, the census does not | [`../objects/interface/reward-variants.md`](../objects/interface/reward-variants.md) |
| 14 | the "field for field" parity test's name | it compares `VecEnvConfig` to `_env()` field for field | Six fields by hand, no `dataclasses.fields`, and **no `observation`** | [`../objects/interface/vec-env-config.md`](../objects/interface/vec-env-config.md) |
| 15 | `net_config_for`'s `card_encoder_hidden` docstring | "a field rather than a literal so it can be swept without a source edit" | Nothing reaches it; there is no flag | [`../objects/interface/net-config.md`](../objects/interface/net-config.md) |
| 16 | `docs/training.md:343` | cites `explained_variance` at `cr_sim/train/ppo.py:405-409` | It is at `:416-420`. The formula it describes is unchanged | [`../objects/measurement/explained-variance.md`](../objects/measurement/explained-variance.md) |
| 17 | `rotating_probe`'s docstring (`cr_sim/train/evaluate.py:592-611`) | describes two edits to `run.py` "rather than making them" | `cr_sim/train/run.py:979-984` already made them; the probe is wired behind `--probe rotating` | [`../objects/measurement/ghost-knobs.md`](../objects/measurement/ghost-knobs.md), [`../objects/measurement/verdict.md`](../objects/measurement/verdict.md) |
| 18 | `scripts/make_demos.py:98` | "The defaults here match `cr_sim.train.run`'s, which is the point" | They do not match on elixir: 0.0 here (`:115`) against 0.3 (`cr_sim/train/run.py:182`) | [`../processes/collect-demonstrations.md`](../processes/collect-demonstrations.md) |
| 19 | `docs/training.md:483-505` | the gamma-correct potential `r = γΦ(s′) − Φ(s)`, and its stated reason | Written and reverted; half the reason is now false — `projected` measures 1.038 score calls per decision, not 2 | [`../CONTEXT.md`](../CONTEXT.md), [`../objects/measurement/ghost-knobs.md`](../objects/measurement/ghost-knobs.md) |

## What is not a row here

A comment that is **narrower than it reads** but not wrong, and a comment the
code has simply outgrown in line numbers. Those get a sentence on the card and
no row, because a sweep reading this file would otherwise spend its time on
prose that is fine.
