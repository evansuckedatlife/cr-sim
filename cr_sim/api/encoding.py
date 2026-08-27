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
from ..engine.arena import Arena, TowerPlacement
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
    "ObservationFeatures",
    "OBSERVATION_V1",
    "OBSERVATION_V2",
    "grid_channels",
    "parse_observation",
    "EncodingConfig",
    "build_encoding_config",
    "cell_to_world",
    "action_grid_shape",
    "observation_shapes",
    "hand_onehot_layout",
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

#: Bodies in one cell that saturate a count channel. Four is the Skeletons
#: card's own summon count, so a single cheap swarm card reads at 1.0 and
#: anything larger is a genuine pile-up rather than one card.
COUNT_NORM = 4.0

#: Damage in flight or sitting in a cloud that saturates a spell channel.
#: Above the deck's own worst single hit (Fireball, 688 at tournament
#: standard) and below a Rocket, so an ordinary spell reads high without
#: everything saturating.
SPELL_NORM = 1000.0

#: The hitpoint-mass channels, in fixed order.
_HP_CHANNELS = (
    "own_ground_hp",
    "own_air_hp",
    "own_building_hp",
    "own_tower_hp",
    "enemy_ground_hp",
    "enemy_air_hp",
    "enemy_building_hp",
    "enemy_tower_hp",
)
#: Bodies per cell, which hitpoint mass alone cannot express: three Skeletons
#: at 81 each and one Knight at 243 produce an identical hitpoint cell, and
#: swarm-versus-tank is the read that decides whether a Log or a Knight is the
#: answer.
_COUNT_CHANNELS = ("own_body_count", "enemy_body_count")
#: Damage in flight and damage sitting on the board. Projectiles and area
#: effects were excluded from the grid entirely, which left the agent unable
#: to see an incoming Fireball or a Poison cloud standing on its own troops --
#: both of which are things a player reacts to rather than remembers.
_SPELL_CHANNELS = ("own_spell_damage", "enemy_spell_damage")


@dataclass(frozen=True, slots=True)
class ObservationFeatures:
    """Which observation an environment encodes.

    Versioned rather than switched, because changing the observation
    invalidates every checkpoint trained on the old one: the network's first
    convolution has one filter bank per input channel, and weights for nine
    channels cannot be loaded into a network expecting thirteen. A run records
    its ``version`` so a checkpoint can be refused loudly instead of failing
    on a shape mismatch nobody can place.

    The two vector flags deliberately do *not* change the vector's length --
    they zero a span that stays where it was -- so hiding information is not
    also a shape change, and an ablation over them compares networks of
    identical size.
    """

    version: int = 1
    #: Projectile and area-effect damage as two more grid channels.
    spells: bool = False
    #: Bodies per cell alongside hitpoint mass.
    swarm: bool = False
    #: Zero the opponent's four hand slots and next card.
    #:
    #: This is the genuinely private information in Clash Royale. Enemy
    #: *elixir* is not: the regeneration schedule is public and every card
    #: played is visible, so a player who counts can reconstruct the bar
    #: exactly -- which ``tests/test_api_encoding.py`` demonstrates rather
    #: than asserts. The hand cannot be reconstructed at all.
    hide_enemy_hand: bool = False
    #: Zero the opponent's elixir scalar. Kept separate from the hand because
    #: it is a different claim: not "the agent should not know this" but "the
    #: agent should have to keep count of it".
    hide_enemy_elixir: bool = False


#: What every checkpoint before this existed was trained on.
OBSERVATION_V1 = ObservationFeatures(version=1)
#: Everything the reference spec called for at once. Each flag is separately
#: switchable so the ablation can say which of them paid.
OBSERVATION_V2 = ObservationFeatures(
    version=2, spells=True, swarm=True,
    hide_enemy_hand=True, hide_enemy_elixir=True,
)


def parse_observation(spec: str) -> ObservationFeatures:
    """``"v1"``, ``"v2"``, or a comma-separated list of individual flags.

    Individual flags are what an ablation needs: "v2" turns on four changes at
    once and a single lift number over it cannot say which of them paid, or
    whether one of them is quietly costing.
    """
    text = (spec or "v1").strip().lower()
    if text in ("v1", "1", ""):
        return OBSERVATION_V1
    if text in ("v2", "2", "all"):
        return OBSERVATION_V2
    known = {"spells", "swarm", "hide_enemy_hand", "hide_enemy_elixir"}
    flags = {name.strip() for name in text.split(",") if name.strip()}
    unknown = flags - known
    if unknown:
        raise ValueError(
            f"unknown observation flag(s) {sorted(unknown)}; "
            f"expected 'v1', 'v2', or any of {sorted(known)}")
    return ObservationFeatures(version=2, **{name: True for name in flags})


def grid_channels(features: ObservationFeatures = OBSERVATION_V1) -> tuple[str, ...]:
    """The grid's channels, in the order :func:`_encode_grid` writes them.

    Terrain stays last whatever is switched on, so the normalisation slices
    below stay expressible as leading ranges rather than as a scatter of
    indices.
    """
    channels = list(_HP_CHANNELS)
    if features.swarm:
        channels.extend(_COUNT_CHANNELS)
    if features.spells:
        channels.extend(_SPELL_CHANNELS)
    channels.append("terrain")
    return tuple(channels)


#: The v1 channel list, kept under its old name because callers outside this
#: module read it.
GRID_CHANNELS = grid_channels(OBSERVATION_V1)
N_GRID_CHANNELS = len(GRID_CHANNELS)
#: All channels except the last (terrain) carry hitpoint mass and share one
#: normalisation; terrain is already scaled at the point it is written.
_HP_CHANNEL_COUNT = len(_HP_CHANNELS)


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
    #: Which observation this config encodes; see :class:`ObservationFeatures`.
    features: ObservationFeatures = OBSERVATION_V1

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def channels(self) -> tuple[str, ...]:
        return grid_channels(self.features)


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
    arena: Arena,
    blue_deck: Sequence[str],
    red_deck: Sequence[str],
    features: ObservationFeatures = OBSERVATION_V1,
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
        features=features,
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
    grid_shape = (len(config.channels), config.grid_height, config.grid_width)
    return {"grid": grid_shape, "vector": (_vector_length(config),)}


def hand_onehot_layout(config: EncodingConfig) -> tuple[int, int, int, int]:
    """Where the acting team's hand identities sit in the observation vector.

    Returns ``(start, stride, count, width)``: the acting team's slot ``i``
    one-hot is ``vector[start + i * stride : start + i * stride + width]``.

    Exposed because a card-conditioned policy head has to read *which card* a
    slot holds, and the alternative -- a network hard-coding the offsets it
    believes the encoder uses -- silently reads the wrong span the moment a
    field is added ahead of the hand. Derived here from the same terms
    :func:`_encode_vector` builds, so the two cannot drift apart without a
    test noticing.
    """
    stride = 1 + config.vocab_size   # cost, then the identity one-hot
    start = 2 + 1                    # two elixir scalars, then past slot 0's cost
    return start, stride, NUM_CARD_SLOTS - 1, config.vocab_size


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
    arena: Arena,
    team: Team,
    x: int,
    y: int,
    anywhere: bool,
    on_water: bool,
    fallen_enemy_towers: frozenset[TowerPlacement] = frozenset(),
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

    ``fallen_enemy_towers`` is part of the cache key on purpose: unlike
    everything else this function is keyed on, it *does* change mid-episode
    -- a Princess Tower kill expands the deploy zone -- and a cache that did
    not account for that would keep answering with the pre-kill zone for the
    rest of the match.
    """
    return arena.can_deploy(
        team, x, y,
        anywhere=anywhere, on_water=on_water,
        fallen_enemy_towers=fallen_enemy_towers,
    )


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
    water flags included, and any Princess Towers already destroyed -- see
    ``Battle.fallen_enemy_towers``). The no-op slot is unconditionally legal
    -- it is always a valid choice to spend nothing this decision.

    This mask is the reason an RL agent trained against this environment does
    not waste most of its samples: without it, the overwhelming majority of a
    5 x 9 x 16 action space is either unaffordable or off the legal half of
    the board, and a policy has to learn that by trial and error before it can
    learn anything about which of the *legal* actions is good. The same logic
    cuts the other way for the expanded zone past a fallen tower: an agent
    that never sees those cells marked legal here will never sample them,
    however good the resulting push would be.
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
    fallen = battle.fallen_enemy_towers(team)
    for slot, card_name in enumerate(hand):
        card = registry.get(card_name)
        if card is None or not player.elixir.can_afford(card.mana_cost):
            continue
        mask[slot] = _placement_grid(
            arena, team,
            card.can_deploy_on_enemy_side, card.can_place_on_water,
            width, height,
            fallen,
        )
    return mask


@lru_cache(maxsize=128)
def _placement_grid(
    arena: Arena,
    team: Team,
    anywhere: bool,
    on_water: bool,
    width: int,
    height: int,
    fallen_enemy_towers: frozenset[TowerPlacement] = frozenset(),
) -> np.ndarray:
    """Which cells a card with these placement flags may be put on.

    Where a card *may* go depends only on the terrain, the card's own two
    flags, and which of the opponent's Princess Towers are already down --
    never on elixir, the hand, or anything else that changes during a match.
    So it is computed once per combination and looked up thereafter. That used
    to be only eight combinations (two flags, two teams) before
    ``fallen_enemy_towers`` joined the key; now it is that times the number of
    distinct fallen-tower sets actually seen, which is at most four per team
    (each of the opponent's two Princess Towers, up or down) -- still small,
    hence the larger but still bounded ``maxsize``.

    Building it per mask instead meant 576 ``can_deploy`` calls every time the
    mask was asked for, which profiling put at a third of a training step. The
    grid it produced was identical every time *for a fixed set of standing
    towers* -- leaving ``fallen_enemy_towers`` out of the key entirely was the
    bug: the first grid built each episode, from before any tower died, would
    keep being handed out unchanged for the rest of the match, and the mask
    would never show the expanded zone even after a tower actually fell.

    Returned read-only, because a cached array handed out by reference is one
    careless ``mask[slot] |= ...`` away from corrupting every future lookup.
    """
    grid = np.zeros((width, height), dtype=bool)
    for gy in range(height):
        for gx in range(width):
            x, y = cell_to_world(gx, gy, team, arena, span=PLACEMENT_TILE_SPAN)
            if arena.can_deploy(
                team, x, y,
                anywhere=anywhere, on_water=on_water,
                fallen_enemy_towers=fallen_enemy_towers,
            ):
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


def _spell_damage(entity: Entity) -> tuple[int, int]:
    """``(damage, radius)`` a projectile or area effect threatens with.

    Read off the spec rather than recomputed: a projectile's damage is what it
    will deal when it lands, and an area effect's is what it deals per
    application, which is the right scale for "how much is about to happen
    here" in both cases. A shot with no damage of its own -- Goblin Barrel,
    which is pure delivery -- contributes nothing here and shows up as troops
    when it lands, which is when it becomes board presence.
    """
    # ``pspec`` on a Projectile, ``aspec`` on an AreaEffect. Neither is
    # ``Entity.spec``, which stays None on both -- reading that instead is how
    # a spell channel ends up permanently zero while everything else about it
    # looks right.
    spec = (getattr(entity, "pspec", None) or getattr(entity, "aspec", None)
            or entity.spec)
    if spec is None:
        return 0, 0
    return int(getattr(spec, "damage", 0) or 0), int(getattr(spec, "radius", 0) or 0)


def _encode_grid(battle: Battle, team: Team, config: EncodingConfig) -> np.ndarray:
    """Multi-channel map of the board, from ``team``'s point of view.

    Hitpoint mass per cell is the base layer. Two optional layers sit beside
    it, both added because the base layer cannot express what they carry:

    *   **Bodies per cell.** Three Skeletons at 81 hitpoints each and one
        Knight at 243 write an identical hitpoint cell, and which of the two
        it is decides whether the answer is a Log or a Knight. Swarm-versus-
        tank is a core read and hitpoint mass alone is blind to it.
    *   **Spell damage.** Projectiles and area effects used to be excluded
        from the grid entirely, on the reasoning that neither carries
        hitpoints and that a spell's landing point is known from the action
        that cast it. That reasoning only holds for the agent's *own* spells.
        It cannot see an incoming Fireball, or a Poison cloud standing on its
        own troops, and both are things a player reacts to.
    """
    width, height = config.grid_width, config.grid_height
    channels = config.channels
    grid = np.zeros((len(channels), height, width), dtype=np.float32)
    features = config.features
    counts = channels.index("own_body_count") if features.swarm else -1
    spells = channels.index("own_spell_damage") if features.spells else -1

    for entity in battle.entities:
        if entity.dead:
            continue
        cell = _world_to_cell(entity.x, entity.y, team, battle.arena, width, height, OBS_TILE_SPAN)
        if cell is None:
            continue
        gx, gy = cell
        own = entity.team is team
        side = "own" if own else "enemy"
        if entity.kind in (EntityKind.TROOP, EntityKind.BUILDING, EntityKind.TOWER):
            if entity.kind is EntityKind.TROOP:
                role = "air" if entity.flying else "ground"
            elif entity.kind is EntityKind.BUILDING:
                role = "building"
            else:
                role = "tower"
            grid[channels.index(f"{side}_{role}_hp"), gy, gx] += entity.hitpoints
            # Towers are permanent scenery, not a swarm; counting them would
            # put a constant three in the same channel a Skeleton army has to
            # be read out of.
            if counts >= 0 and entity.kind is not EntityKind.TOWER:
                grid[counts + (0 if own else 1), gy, gx] += 1.0
        elif spells >= 0 and entity.kind in (EntityKind.PROJECTILE, EntityKind.AREA_EFFECT):
            damage, radius = _spell_damage(entity)
            if damage <= 0:
                continue
            channel = spells + (0 if own else 1)
            # Painted over the footprint, because a spell's threat is an area
            # and a single cell would tell the agent a Fireball is somewhere
            # without saying what it covers.
            span = max(0, int(to_tiles(radius)))
            for dy in range(-span, span + 1):
                for dx in range(-span, span + 1):
                    x, y = gx + dx, gy + dy
                    if 0 <= x < width and 0 <= y < height and dx * dx + dy * dy <= span * span:
                        grid[channel, y, x] += damage

    grid[:_HP_CHANNEL_COUNT] = np.minimum(1.0, grid[:_HP_CHANNEL_COUNT] / HP_NORM)
    if counts >= 0:
        grid[counts:counts + 2] = np.minimum(1.0, grid[counts:counts + 2] / COUNT_NORM)
    if spells >= 0:
        grid[spells:spells + 2] = np.minimum(1.0, grid[spells:spells + 2] / SPELL_NORM)
    grid[len(channels) - 1] = _terrain_channel(battle.arena, team, width, height)
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
    features = config.features
    parts: list[np.ndarray] = [
        _scalar(battle.players[team].elixir.units / MAX_ELIXIR),
        # Zeroed rather than dropped, so hiding it is not also a shape change
        # and an ablation compares networks of identical size.
        _scalar(0.0 if features.hide_enemy_elixir
                else battle.players[opponent].elixir.units / MAX_ELIXIR),
    ]

    for side in (team, opponent):
        player = battle.players[side]
        hidden = features.hide_enemy_hand and side is opponent
        for card_name in player.hand:
            parts.append(_card_features(None if hidden else card_name, registry, config))
        parts.append(_card_features(None if hidden else player.next_card, registry, config))

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
