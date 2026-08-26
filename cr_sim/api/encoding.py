"""Observation and action encoding between the engine's native state and the
fixed-size numpy arrays a learned policy actually consumes.

The engine's state is a list of :class:`~cr_sim.engine.entity.Entity` of
varying length, plus two :class:`~cr_sim.engine.battle.Player` records. Neither
shape is fixed, and a policy network needs one that is. Two representations do
the two jobs the state actually has:

*   A **multi-channel grid** over the arena, because "where is the enemy's
    army relative to my towers" is a spatial question and a flat vector throws
    the geometry away -- a network would have to relearn distance from
    scratch out of a list of (x, y) pairs.
*   A **flat vector** for everything that has no position at all: elixir,
    hand contents, tower health, match clock. Folding these into the grid
    (e.g. broadcasting elixir over every cell) would waste channels on a
    scalar and make the grid's per-cell semantics inconsistent.

Every field in both is scaled into ``[0, 1]``. Nothing here is naturally
negative -- elixir, hitpoints, counts, fractions of a timer -- so ``[-1, 1]``
would only halve the usable range for no benefit. Anything genuinely signed
(who is ahead) belongs in the *reward*, computed in :mod:`cr_sim.api.env`, not
smuggled into the observation as a sign bit the network has to discover.

**Full information.** Both sides' hands are encoded, not just the acting
team's. Real Clash Royale hides the opponent's hand; this environment does
not, the same way AlphaStar-style training pipelines feed a learner privileged
state that a deployed policy would not see live. The task instructions call
for "per team: elixir, the 4-card hand + next card, tower hitpoints" without
qualification, and self-play with full information converges on interesting
policies faster than one where both sides are guessing at eleven unknowns
before they can plan around them. A partially-observable variant that masks
the enemy hand is a straightforward filter over this same vector, left for
whoever needs it.

**Placement resolution.** Two different grids exist for two different jobs.
The *observation* grid is one cell per tile (18 x 32 = 576 cells): coarser
would blur adjacent lanes' hitpoint mass together, and finer buys nothing
since no unit's collision footprint is anywhere near sub-tile. The *action*
grid is one cell per **two** tiles (9 x 16 = 144 cells). Full tile resolution
for actions was considered and rejected: 576 cells x 5 card slots is 2880
discrete actions, and the overwhelming majority of that space is redundant --
a Giant dropped half a tile to the left is the same tactical decision as one
dropped on the tile line, and a policy has to spend samples discovering that
before it can even start learning card selection. A 2-tile cell still keeps
the two bridges (each 2 tiles wide) and both Princess Towers in their own
cells, keeps the two lanes clearly separated in x, and most spell radii in
this build are at least a tile wide, so a 2-tile aim error rarely changes
what a spell catches. That is a 4x cut in the action count (144 x 5 = 720)
for no loss of tactical resolution the interaction suite would notice.

**Card identity.** A hand slot is encoded as its mana cost plus a one-hot over
this *episode's own deck vocabulary* (the union of both decks' card names,
built once when the encoding config is constructed), not a one-hot over every
card in the game. A fixed training run plays a fixed pair of decks, so the
vocabulary a policy actually needs is at most 16 cards, not the full pool --
encoding against the full card list would make almost every one-hot entry a
permanent zero the network has to learn to ignore.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from ..data.cards import CardRegistry
from ..engine.arena import Arena
from ..engine.battle import Battle
from ..engine.constants import MAX_ELIXIR
from ..engine.entity import Entity, EntityKind, Team
from ..engine.fixed import tiles, to_tiles

__all__ = [
    "NUM_CARD_SLOTS",
    "NOOP_SLOT",
    "OBS_TILE_SPAN",
    "PLACEMENT_TILE_SPAN",
    "HP_NORM",
    "GRID_CHANNELS",
    "N_GRID_CHANNELS",
    "EncodingConfig",
    "build_encoding_config",
    "cell_to_world",
    "action_grid_shape",
    "observation_shapes",
    "decode_action",
    "legal_action_mask",
    "encode_observation",
    "total_tower_hitpoints",
]

#: Four hand slots plus a no-op. A card that cannot currently be played is not
#: "illegal" in the same sense as an out-of-bounds cell -- the agent may
#: rightly prefer to save elixir -- so passing is a first-class action rather
#: than something the legality mask has to fake by leaving everything else
#: unmarked.
NUM_CARD_SLOTS = 5
NOOP_SLOT = NUM_CARD_SLOTS - 1

#: Observation grid: one cell per tile. See the module docstring for why this
#: differs from the action grid's resolution.
OBS_TILE_SPAN = 1
#: Action grid: one cell per two tiles. See the module docstring.
PLACEMENT_TILE_SPAN = 2

#: Hitpoint mass that saturates a grid cell to 1.0. Set a bit above the
#: highest single-entity hitpoints in this build (Golem, 5120 at tournament
#: standard; King Tower, 4224) so one full-health max-value entity alone reads
#: below saturation, and only a real stack -- two overlapping tanks, or a
#: swarm bunched into one cell -- pushes a cell to 1.0. Verified via
#: reference/card_stats.json and build_tower_spec at level 11.
HP_NORM = 6000.0

#: Order is fixed and index-addressed by _encode_grid; do not reorder without
#: updating the slicing below.
GRID_CHANNELS = (
    "own_ground_hp",
    "own_air_hp",
    "own_building_hp",
    "own_tower_hp",
    "enemy_ground_hp",
    "enemy_air_hp",
    "enemy_building_hp",
    "enemy_tower_hp",
    "terrain",
)
N_GRID_CHANNELS = len(GRID_CHANNELS)
#: All channels except the last (terrain) carry hitpoint mass and share one
#: normalisation; terrain is already scaled at the point it is written.
_HP_CHANNEL_COUNT = N_GRID_CHANNELS - 1


@dataclass(frozen=True, slots=True)
class EncodingConfig:
    """Everything the encoder needs to fix array shapes for one environment's
    lifetime: arena geometry and the two decks in play.

    Built once, when an environment is constructed, and reused for every
    ``reset()``. Gymnasium's contract is that ``observation_space`` and
    ``action_space`` hold for the life of the env, so nothing that changes
    shape between episodes -- deck composition above all -- may leak into the
    per-episode encoding path; it is fixed here, once, instead.
    """

    grid_width: int
    grid_height: int
    action_width: int
    action_height: int
    #: This episode's card vocabulary: the union of both decks, sorted for a
    #: stable index assignment independent of dict/set iteration order.
    vocab: tuple[str, ...]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def _grid_shape(arena: Arena, span: int) -> tuple[int, int]:
    """(width, height) in cells of ``span``-tile squares covering the arena.

    Ceiling division so a non-exact fit still covers the whole board rather
    than clipping a partial row or column off the far edge. The standard
    18x32 arena divides evenly at both spans used here (18/32 by 1, and by 2),
    so this only matters if a future build's tilemap does not.
    """
    width_tiles = round(arena.width_tiles)
    height_tiles = round(arena.height_tiles)
    width = max(1, -(-width_tiles // span))
    height = max(1, -(-height_tiles // span))
    return width, height


def build_encoding_config(
    arena: Arena, blue_deck: Sequence[str], red_deck: Sequence[str]
) -> EncodingConfig:
    grid_width, grid_height = _grid_shape(arena, OBS_TILE_SPAN)
    action_width, action_height = _grid_shape(arena, PLACEMENT_TILE_SPAN)
    vocab = tuple(sorted(set(blue_deck) | set(red_deck)))
    return EncodingConfig(
        grid_width=grid_width,
        grid_height=grid_height,
        action_width=action_width,
        action_height=action_height,
        vocab=vocab,
    )


def action_grid_shape(arena: Arena) -> tuple[int, int]:
    """(width, height) of the placement grid, without needing a full config."""
    return _grid_shape(arena, PLACEMENT_TILE_SPAN)


def observation_shapes(config: EncodingConfig) -> dict[str, tuple[int, ...]]:
    """Array shapes :func:`encode_observation` returns, for building a gym
    space without running a battle.

    The vector length is derived the same way :func:`_encode_vector` builds
    it, term for term, rather than a memorised constant -- see the comment
    there for what each term is.
    """
    grid_shape = (N_GRID_CHANNELS, config.grid_height, config.grid_width)
    return {"grid": grid_shape, "vector": (_vector_length(config),)}


def _vector_length(config: EncodingConfig) -> int:
    hand_slot = 1 + config.vocab_size  # cost + one-hot identity
    per_side = 4 * hand_slot + hand_slot  # four hand slots + the next card
    return 2 + 2 * per_side + 6 + 1 + 2 + 1
    # elixir(2) + hands+next(2 sides) + tower fracs(6) + tick(1) + crowns(2) + overtime(1)


# ------------------------------------------------------------------ geometry


def cell_to_world(grid_x: int, grid_y: int, team: Team, arena: Arena, *, span: int) -> tuple[int, int]:
    """A grid cell's centre, in world subtiles, mirrored to ``team``'s own view.

    Only the y axis is mirrored. Blue defends low y and Red defends high y, so
    a Blue and a Red observation of "the same" tactical position are related
    by ``y -> height - y``; x needs no such transform because both bridges
    (and both lanes) sit at the same x for either side. Canonicalising y this
    way is what lets one set of policy weights play both sides in self-play --
    without it, the network would have to learn two mirror-image strategies
    that happen to require identical actions, doubling what it has to
    discover for no tactical difference.
    """
    cx = (grid_x + 0.5) * span
    cy_own = (grid_y + 0.5) * span
    cy = cy_own if team is Team.BLUE else arena.height_tiles - cy_own
    x = min(max(tiles(cx), 0), arena.width - 1)
    y = min(max(tiles(cy), 0), arena.height - 1)
    return x, y


def _world_to_cell(
    x: int, y: int, team: Team, arena: Arena, width: int, height: int, span: int
) -> tuple[int, int] | None:
    """Inverse of :func:`cell_to_world`: a world point -> its grid cell, or
    ``None`` if it falls outside the grid this config declared."""
    x_tiles = to_tiles(x)
    y_tiles = to_tiles(y)
    own_y_tiles = y_tiles if team is Team.BLUE else arena.height_tiles - y_tiles
    gx = int(x_tiles // span)
    gy = int(own_y_tiles // span)
    if 0 <= gx < width and 0 <= gy < height:
        return gx, gy
    return None


# --------------------------------------------------------------------- towers


def _team_towers(battle: Battle, team: Team) -> dict[str, Entity]:
    """``team``'s King and both Princess towers, alive or destroyed.

    A destroyed tower is removed from ``battle.entities`` the tick it dies
    (see ``Battle._phase_resolve_deaths``) and moved to ``battle.graveyard``,
    so a lookup that only scans ``entities`` would make a lost tower vanish
    from the encoding instead of reading as zero health. Concatenating the two
    lists cannot double-count a tower: Battle moves an entity from one to the
    other, it never appears in both.

    Princess towers are distinguished by x rather than by name, since nothing
    in the data labels one "left" and the other "right"; sorting by x gives a
    stable, arbitrary-but-consistent order that does not depend on spawn
    order or dict iteration.
    """
    candidates = [
        e
        for e in (*battle.entities, *battle.graveyard)
        if e.kind is EntityKind.TOWER and e.team is team
    ]
    king = next((e for e in candidates if "King" in getattr(e.spec, "name", "")), None)
    princesses = sorted((e for e in candidates if e is not king), key=lambda e: e.x)
    out: dict[str, Entity] = {}
    if king is not None:
        out["king"] = king
    if len(princesses) >= 1:
        out["princess_low_x"] = princesses[0]
    if len(princesses) >= 2:
        out["princess_high_x"] = princesses[1]
    return out


def _tower_frac(entity: Entity | None) -> float:
    """Fraction of max hitpoints remaining -- 0.0 for a missing or dead tower."""
    if entity is None or entity.max_hitpoints <= 0:
        return 0.0
    return max(0, entity.hitpoints) / entity.max_hitpoints


def total_tower_hitpoints(battle: Battle, team: Team) -> tuple[int, int]:
    """Summed current/max hitpoints across ``team``'s three towers.

    Exposed publicly because :mod:`cr_sim.api.env` needs the same number for
    its dense reward term, and it has to agree with what the observation
    shows the agent -- both read through :func:`_team_towers` rather than each
    maintaining their own notion of "the towers".
    """
    towers = _team_towers(battle, team)
    current = sum(max(0, t.hitpoints) for t in towers.values())
    maximum = sum(t.max_hitpoints for t in towers.values())
    return current, maximum


# ---------------------------------------------------------------- action side


def decode_action(
    action: Sequence[int], team: Team, arena: Arena, config: EncodingConfig
) -> tuple[int, int, int] | None:
    """``(card_slot, grid_x, grid_y)`` -> ``(card_slot, world_x, world_y)`` in
    subtiles, or ``None`` for the no-op slot.

    Raises on an out-of-range slot or cell rather than clamping, because a
    silently clamped action would let a badly-configured policy head (wrong
    ``MultiDiscrete`` bounds) place cards at the wrong points without ever
    producing an error to notice.
    """
    slot, grid_x, grid_y = (int(v) for v in action)
    if slot == NOOP_SLOT:
        return None
    if not (0 <= slot < NOOP_SLOT):
        raise ValueError(f"card slot {slot} outside [0, {NOOP_SLOT})")
    if not (0 <= grid_x < config.action_width and 0 <= grid_y < config.action_height):
        raise ValueError(
            f"placement cell ({grid_x}, {grid_y}) outside the "
            f"{config.action_width}x{config.action_height} action grid"
        )
    x, y = cell_to_world(grid_x, grid_y, team, arena, span=PLACEMENT_TILE_SPAN)
    return slot, x, y


@lru_cache(maxsize=4096)
def _can_deploy_cached(
    arena: Arena, team: Team, x: int, y: int, anywhere: bool, on_water: bool
) -> bool:
    """Memoised ``Arena.can_deploy``, for callers outside the mask builder.

    ``can_deploy`` calls ``own_half``, which calls ``river_band`` /
    ``river_rows``, which rescans the whole 36x64 tilemap for the river band
    on *every* call rather than caching it -- profiling
    ``legal_action_mask`` showed that scan alone accounts for essentially
    all of a mask build's time, since a mask calls ``can_deploy`` up to
    ``4 * action_width * action_height`` times (576 for the standard board)
    and every one of those re-triggers it. The action grid only ever asks
    about a fixed set of cell-centre points per ``(arena, team)``, and at
    most a handful of ``(anywhere, on_water)`` combinations exist across the
    card pool, so the whole result space is small and unchanging within an
    episode -- caching it here, without touching ``Arena`` itself, turns a
    per-mask O(board area) rescan into a one-time cost.
    """
    return arena.can_deploy(team, x, y, anywhere=anywhere, on_water=on_water)


def legal_action_mask(
    battle: Battle, team: Team, registry: CardRegistry, config: EncodingConfig
) -> np.ndarray:
    """Boolean ``(5, action_width, action_height)`` mask of currently legal
    actions for ``team``.

    Indexed as ``mask[slot, x, y]``, in exactly the order an action tuple is
    written. That matters more than it looks: shaping this like an image
    ``(slot, y, x)`` while the action space is ``(slot, x, y)`` reads fine in
    both places and silently transposes every placement an agent picks from
    the mask, which on a 9x16 grid is a legal-looking cell in the wrong half of
    the board rather than an error.

    An action is legal exactly when three things all hold: the slot holds a
    real card, the player can afford it, and ``Arena.can_deploy`` accepts the
    cell's centre point for that card's own placement rules (enemy-side and
    water flags included). The no-op slot is unconditionally legal -- it is
    always a valid choice to spend nothing this decision.

    This mask is the reason an RL agent trained against this environment does
    not waste most of its samples: without it, the overwhelming majority of a
    5 x 9 x 16 action space is either unaffordable or off the legal half of
    the board, and a policy has to learn that by trial and error before it can
    learn anything about which of the *legal* actions is good.
    """
    width, height = config.action_width, config.action_height
    mask = np.zeros((NUM_CARD_SLOTS, width, height), dtype=bool)
    # Exactly one cell, not the whole slot. Passing has no position, so every
    # (NOOP, x, y) decodes to the same thing -- marking all 144 legal would
    # spend a fifth of the policy's output on duplicates of one action and
    # hand "do nothing" a fifth of the probability mass before the network has
    # learned anything. That matters here more than it usually would: passing
    # is the only action that is never punished, so it is already the
    # comfortable local optimum for this task.
    mask[NOOP_SLOT, 0, 0] = True

    player = battle.players[team]
    hand = player.hand
    arena = battle.arena
    for slot, card_name in enumerate(hand):
        card = registry.get(card_name)
        if card is None or not player.elixir.can_afford(card.mana_cost):
            continue
        mask[slot] = _placement_grid(
            arena, team,
            card.can_deploy_on_enemy_side, card.can_place_on_water,
            width, height,
        )
    return mask


@lru_cache(maxsize=32)
def _placement_grid(
    arena: Arena,
    team: Team,
    anywhere: bool,
    on_water: bool,
    width: int,
    height: int,
) -> np.ndarray:
    """Which cells a card with these placement flags may be put on.

    Where a card *may* go depends only on the terrain and on the card's own two
    flags -- never on elixir, the hand, or anything that changes during a
    match. So it is computed once per combination and looked up thereafter, and
    there are only eight combinations: two flags, two teams.

    Building it per mask instead meant 576 ``can_deploy`` calls every time the
    mask was asked for, which profiling put at a third of a training step. The
    grid it produced was identical every time.

    Returned read-only, because a cached array handed out by reference is one
    careless ``mask[slot] |= ...`` away from corrupting every future lookup.
    """
    grid = np.zeros((width, height), dtype=bool)
    for gy in range(height):
        for gx in range(width):
            x, y = cell_to_world(gx, gy, team, arena, span=PLACEMENT_TILE_SPAN)
            if arena.can_deploy(team, x, y, anywhere=anywhere, on_water=on_water):
                grid[gx, gy] = True
    grid.flags.writeable = False
    return grid


# ----------------------------------------------------------- observation side


@lru_cache(maxsize=8)
def _terrain_channel(arena: Arena, team: Team, width: int, height: int) -> np.ndarray:
    """Static per-(arena, team) terrain map, in own-perspective grid cells.

    Cached because it never changes within an episode -- the tilemap is fixed
    by the build -- and recomputing 576 tile lookups on every single
    observation would spend real time re-deriving something constant for the
    whole run. ``Arena`` is a frozen dataclass of hashable fields, so it works
    as an lru_cache key directly.
    """
    out = np.zeros((height, width), dtype=np.float32)
    for gy in range(height):
        for gx in range(width):
            x, y = cell_to_world(gx, gy, team, arena, span=OBS_TILE_SPAN)
            if arena.is_blocked(x, y):
                out[gy, gx] = 1.0
            elif arena.is_water(x, y):
                out[gy, gx] = 0.5
    return out


def _encode_grid(battle: Battle, team: Team, config: EncodingConfig) -> np.ndarray:
    """Multi-channel hitpoint-mass map. See the module docstring for the
    channel list and why projectiles and area effects are left out of it:
    neither carries hitpoints that mean "board presence" the way a troop's
    or a building's do, and where a spell is about to land is already known
    from the action that just cast it, not from parsing entities after the
    fact.
    """
    width, height = config.grid_width, config.grid_height
    grid = np.zeros((N_GRID_CHANNELS, height, width), dtype=np.float32)
    for entity in battle.entities:
        if entity.dead or entity.kind not in (
            EntityKind.TROOP, EntityKind.BUILDING, EntityKind.TOWER
        ):
            continue
        cell = _world_to_cell(entity.x, entity.y, team, battle.arena, width, height, OBS_TILE_SPAN)
        if cell is None:
            continue
        gx, gy = cell
        side = "own" if entity.team is team else "enemy"
        if entity.kind is EntityKind.TROOP:
            role = "air" if entity.flying else "ground"
        elif entity.kind is EntityKind.BUILDING:
            role = "building"
        else:
            role = "tower"
        channel = GRID_CHANNELS.index(f"{side}_{role}_hp")
        grid[channel, gy, gx] += entity.hitpoints

    grid[:_HP_CHANNEL_COUNT] = np.minimum(1.0, grid[:_HP_CHANNEL_COUNT] / HP_NORM)
    grid[_HP_CHANNEL_COUNT] = _terrain_channel(battle.arena, team, width, height)
    return grid


def _scalar(value: float) -> np.ndarray:
    return np.array([value], dtype=np.float32)


def _card_features(card_name: str | None, registry: CardRegistry, config: EncodingConfig) -> np.ndarray:
    """One hand slot: its cost in ``[0, 1]`` followed by a one-hot over this
    episode's deck vocabulary (see the module docstring). An empty slot --
    only ``next_card`` past the end of a short deck can be one -- is all
    zeros, which is a legitimate "nothing here" encoding rather than a value
    that collides with a real card.
    """
    out = np.zeros(1 + config.vocab_size, dtype=np.float32)
    if card_name is None:
        return out
    card = registry.get(card_name)
    if card is not None:
        out[0] = min(1.0, card.mana_cost / MAX_ELIXIR)
    if card_name in config.vocab:
        out[1 + config.vocab.index(card_name)] = 1.0
    return out


def _encode_vector(
    battle: Battle, team: Team, registry: CardRegistry, config: EncodingConfig
) -> np.ndarray:
    opponent = team.opponent
    parts: list[np.ndarray] = [
        _scalar(battle.players[team].elixir.units / MAX_ELIXIR),
        _scalar(battle.players[opponent].elixir.units / MAX_ELIXIR),
    ]

    for side in (team, opponent):
        player = battle.players[side]
        for card_name in player.hand:
            parts.append(_card_features(card_name, registry, config))
        parts.append(_card_features(player.next_card, registry, config))

    for side in (team, opponent):
        towers = _team_towers(battle, side)
        for key in ("king", "princess_low_x", "princess_high_x"):
            parts.append(_scalar(_tower_frac(towers.get(key))))

    parts.append(_scalar(min(1.0, battle.tick / max(1, battle.timeline.total_ticks))))
    parts.append(_scalar(battle.players[team].crowns / 3.0))
    parts.append(_scalar(battle.players[opponent].crowns / 3.0))
    parts.append(_scalar(1.0 if battle.in_overtime else 0.0))

    return np.concatenate(parts).astype(np.float32)


def encode_observation(
    battle: Battle, team: Team, registry: CardRegistry, config: EncodingConfig
) -> dict[str, np.ndarray]:
    """``team``'s full observation of ``battle``: a spatial grid plus a flat
    feature vector. See the module docstring for the design of each."""
    return {
        "grid": _encode_grid(battle, team, config),
        "vector": _encode_vector(battle, team, registry, config),
    }
