---
type: object
cluster: interface
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/api/encoding.py
---

# Observation grid

The spatial half of what the agent sees: a `(channels, 32, 18)` float32 map of
the arena, one cell per tile, in the acting team's own perspective. Built by
`_encode_grid`; its channel list is built by `grid_channels()`.

## Why this shape

"Where is the enemy's army relative to my towers" is a spatial question, and a
flat vector throws the geometry away — the network would have to relearn
distance out of a list of coordinates. One cell per tile because coarser blurs
adjacent lanes' hitpoint mass together and finer buys nothing: no collision
footprint in this build is anywhere near sub-tile.

The channel *order* is the load-bearing part, and it is not written down as a
list. It is the **iteration order of a dict literal**. Channels are 8 fixed
hitpoint channels, then whichever flagged sets are on **in
`GRID_FEATURE_CHANNELS` insertion order**, then terrain. Terrain is last on
purpose: it is pre-scaled where it is written, so every other channel's
normalisation stays expressible as a leading slice rather than a scatter of
indices.

## Shape

- `(len(channels), grid_height=32, grid_width=18)` — height first. The action
  mask reverses this; see the axis-order row in
  [`../../CONTEXT.md`](../../CONTEXT.md) rather than carrying one convention
  into the other.
- Channels: `_HP_CHANNELS` (8, fixed tuple) → flagged sets → `"terrain"`.
  `GRID_FEATURE_CHANNELS` is `{swarm: 2, spells: 2, threat: 4}`, so v2 is
  `…hp × 8, own/enemy_body_count, own/enemy_spell_damage, terrain` and the
  count block sits at indices 8-9, the spell block at 10-11.
- Normalisations, each set just above the largest real value in this build:
  `HP_NORM` 6000, `COUNT_NORM` 4, `SPELL_NORM` 1000, `DPS_NORM` 800,
  `REACH_NORM` 12. Damage **sums** over a stack; reach takes the **max** —
  two Musketeers put out twice the damage and do not shoot any further.
- Towers are excluded from the body-count channels (they would put a constant
  three where a Skeleton army has to be read) and included in the threat
  channels (a Princess Tower's reach is the least safe ground on the map).
- Spell damage is painted over a disc of the spell's own radius, not one cell.
- `_terrain_channel` and `_placement_grid` are both `lru_cache`d on a frozen
  `Arena`; the grid returned by `_placement_grid` is read-only.

Citations: `cr_sim/api/encoding.py:735` (`_encode_grid`), `:759` (allocation),
`:761-763` (channel offsets resolved by name), `:288-292`
(`GRID_FEATURE_CHANNELS`), `:295-307` (`grid_channels`), `:147-173` (the four
channel tuples), `:122`, `:127`, `:133`, `:139`, `:144` (the norms), `:782-802`
(sum vs max, towers), `:803-816` (the spell disc), `:818-827` (the
normalisation slices and terrain last), `:694` (`_terrain_channel`).

## Connected to

- **owns:** nothing below it — this is the leaf the trunk's first convolution
  reads.
- **owned-by:** [`observation-features.md`](./observation-features.md) decides
  which sets are on; [`encoding-config.md`](./encoding-config.md) fixes the
  dimensions for an env's lifetime.
- **joins:** [`net-config.md`](./net-config.md) (`grid_channels` → `conv.0`);
  [`policy-heads.md`](./policy-heads.md) (`ConvPlacementHead` reads the trunk's
  own feature map, not this array); the mirrored norms in
  [`card-features.md`](./card-features.md).
- **looks-like-but-is-not:** `GRID_CHANNELS` / `N_GRID_CHANNELS`
  (`cr_sim/api/encoding.py:312-313`) are **v1-only frozen aliases with test-only
  consumers** — leftover. Reading them instead of `config.channels` gives nine
  names whatever the env encodes.

## If you change this

- **Hits:** `NetConfig.grid_channels` and `conv.0.weight` in both the actor and
  the separate critic trunk (`cr_sim/train/nets.py:231`, `:530`, `:550-551`); the *meaning*
  of channels 8-11 in every v2 checkpoint on disk, if the change is a reorder
  rather than an append; `config.json`'s `"observation_channels"`
  (`cr_sim/train/run.py:629`), which records names and so makes an append visible and
  a reorder visible only to a reader who compares two runs; `_HP_CHANNEL_COUNT`
  and the normalisation slices, which assume the hitpoint block is leading and
  terrain is last.
- **Does not hit:** the observation **vector**. The obvious next stop —
  `hand_onehot_layout`, `NetConfig.hand_offset`, `_vector_length` — is the wrong
  one: the grid and the vector are two independent arrays and no grid change has
  ever moved a vector column. It also does not hit the **legality mask**: the
  mask is built from elixir, hand and terrain-deploy rules
  (`cr_sim/api/encoding.py:586-641`), never from this array.

**No test pins the relative order of the swarm and spell blocks.** Every test
resolves a channel by `.index(name)` (`tests/test_api_encoding.py:204`,
`tests/test_observation_v2.py:171`, `:205`), and the pinned tuple is v1's only
(`tests/test_observation_v2.py:104-108`). Swapping the `swarm` and `spells`
entries in the dict literal leaves the suite green and repermutes channels 8-11
under every v2 checkpoint that exists.

## Surfaces

| Surface | Role |
|---|---|
| `ActorCritic.conv` / `critic_conv` (`cr_sim/train/nets.py:529-548`) | reads, every forward |
| `runs/*/config.json` `"observation_channels"` | written once per run; read by humans |
| `Demonstrations.grid` (`cr_sim/train/clone.py:355`, `:476`) | written already-encoded — see [`check-observation.md`](./check-observation.md) |
| `cr_sim/render/`, `cr_sim/play/` | none — the viewers read the battle, not this |
| `tests/test_api_encoding.py`, `tests/test_observation_v2.py` | read, by name |

## See

- Source: `cr_sim/api/encoding.py`
- As-built: `docs/training.md`, section "Which observation changes helped"

*Verified 2026-08-30 against `main` @ `dc47f51`.*
