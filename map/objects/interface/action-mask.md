---
type: object
cluster: interface
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/api/encoding.py
---

# Action space and legality mask

What the agent may do: `MultiDiscrete([5, 9, 16])` — a hand slot (four cards plus
a pass) and a cell on the 9x16 placement grid, 720 actions — and the boolean
`(5, 9, 16)` mask of which of them are legal right now. Built by
`legal_action_mask`; turned back into world coordinates by `decode_action`.

## Why this shape

**Two tile spans, not one.** The observation grid is one cell per tile; the
action grid is one cell per **two** tiles. Full tile resolution would be 2,880
discrete actions, and most of that space is redundant — a Giant dropped half a
tile left is the same tactical decision. Two tiles still separates the two
bridges, both Princess Towers and the two lanes, and most spell radii here are at
least a tile wide. A 4x cut for no resolution the interaction suite would notice.

**The mask is why training works at all.** Without it, most of a 720-way space is
unaffordable or on the wrong half of the board, and a policy spends nearly every
sample learning the rules instead of the game.

**Pass is a first-class action, marked at exactly one cell.** Saving elixir is a
legitimate choice, not an absence of choices — so it gets a slot rather than
being faked by leaving everything unmarked. Marking all 144 of its cells would
hand "do nothing" a fifth of the probability mass before the network has learned
anything, and passing is the only action this environment never punishes.

## Shape

- `NUM_CARD_SLOTS = 5`, `NOOP_SLOT = 4`; `OBS_TILE_SPAN = 1`,
  `PLACEMENT_TILE_SPAN = 2`.
- `mask[slot, x, y]`, `dtype=bool` — **x before y**, the order an action tuple is
  written. The observation grid reverses this; see the axis-order row in
  [`../../CONTEXT.md`](../../CONTEXT.md).
- Legal iff: the slot holds a real card, the player can afford it, and
  `Arena.can_deploy` accepts the cell's centre for that card's own flags
  (`can_deploy_on_enemy_side`, `can_place_on_water`) given
  `battle.fallen_enemy_towers(team)`. Pass is unconditional.
- `_placement_grid` is `lru_cache(128)` keyed **including** `fallen_enemy_towers`
  — a Princess Tower kill expands the deploy zone mid-episode, and a key without
  it hands out the pre-kill zone for the rest of the match. Returned read-only.
- `decode_action` **raises** on an out-of-range slot or cell rather than clamping,
  so a head with wrong `MultiDiscrete` bounds errors instead of placing cards
  somewhere plausible. `cell_to_world` mirrors **y only** — both lanes and both
  bridges sit at the same x for either side, which is what lets one set of
  weights play both colours.

Citations: `cr_sim/api/encoding.py:107-114` (the four constants),
`:586-641` (`legal_action_mask`), `:616` (the mask allocation and its axis
order), `:624` (the single noop cell), `:523-546` (`decode_action`),
`:429-447` (`cell_to_world`), `:643-691` (`_placement_grid`),
`cr_sim/api/env.py:365-367` (the `MultiDiscrete`).

## Connected to

- **owns:** the flat 720-way categorical every head emits —
  [`policy-heads.md`](./policy-heads.md).
- **owned-by:** [`encoding-config.md`](./encoding-config.md) supplies
  `action_width` / `action_height`.
- **joins:** `CRSimEnv._apply_action` and `CRSimEnv.legal_action_mask`
  (`cr_sim/api/env.py:168-184`, `:587-592`) — index rows, card stub;
  [`net-config.md`](./net-config.md) (`num_actions`, `num_slots`, `num_cells`).
- **looks-like-but-is-not:** `_can_deploy_cached` (`cr_sim/api/encoding.py:549-583`) is a
  **ghost** — 36 lines of docstring, an `lru_cache(4096)`, and zero callers
  anywhere in the tree; `_placement_grid` calls `arena.can_deploy` directly.
  `action_grid_shape` (`:384`) is **leftover**, test-only.

## If you change this

- **Hits:** `NetConfig.num_actions` and `num_cells` (`cr_sim/train/nets.py:166-172`,
  `:235`); `ConvPlacementHead`, which hardcodes the 2:1 span ratio as
  `(grid_height + 1) // 2` and raises if it stops holding
  (`cr_sim/train/nets.py:493-502`); the flat-index inverse, re-derived at **four**
  production sites — `cr_sim/train/ppo.py:153-163`, `cr_sim/train/selfplay.py:332-333`,
  `cr_sim/play/policy.py:170-171`, `cr_sim/train/scripted.py:355-362`; the two places that
  take `np.argwhere(mask)[0]` and feed the triple straight to `_apply_action`
  without re-validating (`cr_sim/api/env.py:514-516`, `:562-564`).
- **Does not hit:** the **observation grid**. The obvious next stop — "the board
  got coarser, so the encoder must change" — is wrong: the two grids are separate
  `_grid_shape` calls at different spans (`cr_sim/api/encoding.py:371-372`), and no
  observation channel moves. It also does not hit `check_observation`, which
  compares a feature set and knows nothing about action shape, so a checkpoint
  trained on a different action grid loads past it and fails later on
  `policy_head` weights.

**The flat-index convention has no owner.** `_flat_mask` C-order-flattens
`(slot, x, y)` (`cr_sim/train/ppo.py:149-150`) and every consumer re-derives the inverse
by hand. `ConvPlacementHead`'s docstring credits `cr_sim.api.encoding.decode_action`
with reading the flat index — `decode_action` takes a `(slot, x, y)` sequence and
never sees one. **Code wins over that docstring** ([`../../_meta/overrides.md`](../../_meta/overrides.md), row 8). What actually holds the
convention is `tests/test_action_head.py`, which round-trips a known cell through
the conv head and `decode_action`.

## Surfaces

| Surface | Role |
|---|---|
| `cr_sim/train/ppo.py:241`, `:345`; `train/selfplay.py`; `cr_sim/train/evaluate.py:183`; `cr_sim/train/ladder.py:544`; `cr_sim/train/proposal.py:251` | read, once per decision |
| `cr_sim/api/vec.py:210`, `:224` | read — the mask crosses the worker pipe with every observation |
| `cr_sim/play/policy.py:155` | read, browser opponent |
| `scripts/bench_engine.py:283`, `scripts/measure_expert.py:101` | read |
| `runs/` artefacts | **none** — no mask, and no action-grid shape, is ever stored |

## See

- Source: `cr_sim/api/encoding.py`
- As-built: `docs/training.md`, section "Which action head"

*Verified 2026-08-30 against `main` @ `dc47f51`. `cr_sim/train/scripted.py:355-362`,
`cr_sim/train/evaluate.py:183` and `cr_sim/train/ladder.py:544` are in the uncommitted working
tree; `evaluate.py` and `ladder.py` carry no line shift at those points,
`scripted.py` does. See [`../../CONTEXT.md`](../../CONTEXT.md).*
