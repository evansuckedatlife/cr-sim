"""The battlefield.

Terrain is not guessed -- it is read from the game's own ``tilemaps/tilemap.csv``,
a 36x64 grid of **half-tiles** covering the 18x32 tile arena. Each cell is a
bitfield, and every value in the shipped file decomposes cleanly into the flags
in :class:`Tile` with no leftover bits, which is what confirms the reading.

Two coordinate systems meet here, and confusing them is the easy mistake:

*   **Cells** are indexed ``0..35`` by ``0..63``. Cell ``i`` covers the span
    between half-tile grid *lines* ``i`` and ``i+1``.
*   **Positions** -- both ``spawn_groups.toml`` and everything the engine does --
    are grid *lines*. The King Tower's ``x=18`` is the line at 18 half-tiles,
    i.e. tile 9.0, dead centre of an 18-tile board. Read as a cell index it
    would be 9.25 and the tower would sit slightly off-centre forever.

The geometry that falls out, all confirmed against the file:

*   River spans cell rows 30-33, i.e. tiles y 15 -> 17: two tiles tall, centred
    on y=16, exactly half of 32.
*   Two bridges at cells x 5-8 and x 27-30, each two tiles wide, centred on
    tiles x 3.5 and x 14.5 -- which are precisely the Princess Tower x
    positions. Towers sit in line with their bridge.
*   The map holds **both** sides; ``spawn_groups.toml`` lists only one, and the
    opponent's structures mirror through ``y -> 64 - y``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import IntFlag
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..data.source import LogicData
from .entity import Team
from .fixed import SUBTILES_PER_TILE, half_tiles

__all__ = [
    "Tile",
    "Arena",
    "TowerPlacement",
    "load_arena",
    "HALF_TILES_WIDE",
    "HALF_TILES_TALL",
]

#: The standard arena, in half-tiles. 141 of the 158 arena rows in the build
#: point at the same tilemap, so arena identity is cosmetic, not geometric.
HALF_TILES_WIDE = 36
HALF_TILES_TALL = 64

_SUBTILES_PER_HALF_TILE = SUBTILES_PER_TILE // 2


class Tile(IntFlag):
    """Terrain bits. Verified exhaustive against the shipped tilemap."""

    NONE = 0
    #: Road markings that steer pathfinding; the two lanes are tagged
    #: separately, and the King Tower's footprint is split between them.
    LANE_LEFT = 1
    LANE_RIGHT = 2
    #: Impassable. Used for the out-of-play margins and tower footprints.
    BLOCKED = 16
    #: The river. Ground units cannot enter; the bridges are simply the cells
    #: in the river band that do *not* carry this bit.
    WATER = 32
    #: Decoration/spawn anchors, symmetric about both axes. No gameplay effect
    #: that is visible in the data.
    MARKER = 128
    #: Marks the centre line of a bridge.
    BRIDGE = 256
    #: The arena centre point.
    CENTRE = 512


@dataclass(frozen=True, slots=True)
class TowerPlacement:
    """A structure's position, in subtiles, for one team."""

    name: str
    team: Team
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class Arena:
    """Terrain plus structure placement."""

    cells: tuple[int, ...]
    half_width: int = HALF_TILES_WIDE
    half_height: int = HALF_TILES_TALL
    towers: tuple[TowerPlacement, ...] = ()
    source: str = "tilemap.csv"

    # --------------------------------------------------------------- extents

    @property
    def width_tiles(self) -> float:
        return self.half_width / 2

    @property
    def height_tiles(self) -> float:
        return self.half_height / 2

    @property
    def width(self) -> int:
        """Arena width in subtiles."""
        return self.half_width * _SUBTILES_PER_HALF_TILE

    @property
    def height(self) -> int:
        return self.half_height * _SUBTILES_PER_HALF_TILE

    # ----------------------------------------------------------- cell access

    def cell(self, cx: int, cy: int) -> int:
        """Terrain bits at a cell index. Out of bounds reads as BLOCKED."""
        if not (0 <= cx < self.half_width and 0 <= cy < self.half_height):
            return int(Tile.BLOCKED)
        return self.cells[cy * self.half_width + cx]

    def cell_at(self, x: int, y: int) -> tuple[int, int]:
        """Subtile position -> the cell index containing it."""
        return x // _SUBTILES_PER_HALF_TILE, y // _SUBTILES_PER_HALF_TILE

    def flags_at(self, x: int, y: int) -> int:
        cx, cy = self.cell_at(x, y)
        return self.cell(cx, cy)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    # ------------------------------------------------------------ walkability

    def is_water(self, x: int, y: int) -> bool:
        return bool(self.flags_at(x, y) & Tile.WATER)

    def is_blocked(self, x: int, y: int) -> bool:
        return bool(self.flags_at(x, y) & Tile.BLOCKED)

    def is_walkable(self, x: int, y: int, *, flying: bool = False) -> bool:
        """Can a unit occupy this point?

        Flying units ignore terrain entirely -- that is the whole point of
        flight in this game, and it is why the river is a real strategic
        boundary only for ground troops.
        """
        if not self.in_bounds(x, y):
            return False
        if flying:
            return True
        return not (self.flags_at(x, y) & (Tile.WATER | Tile.BLOCKED))

    def is_road(self, x: int, y: int) -> bool:
        return bool(self.flags_at(x, y) & (Tile.LANE_LEFT | Tile.LANE_RIGHT))

    # --------------------------------------------------------------- geometry

    def river_rows(self) -> tuple[int, int]:
        """Inclusive cell-row range of the river band."""
        rows = [
            cy
            for cy in range(self.half_height)
            if any(self.cell(cx, cy) & Tile.WATER for cx in range(self.half_width))
        ]
        return (rows[0], rows[-1]) if rows else (0, -1)

    def river_band(self) -> tuple[int, int]:
        """The river's ``(top, bottom)`` edges in subtiles."""
        first, last = self.river_rows()
        if last < first:
            return (0, 0)
        return first * _SUBTILES_PER_HALF_TILE, (last + 1) * _SUBTILES_PER_HALF_TILE

    def bridges(self) -> tuple[tuple[int, int, int], ...]:
        """Each bridge as ``(left, right, centre)`` in subtiles.

        A bridge is a run of cells inside the river band that is *not* water --
        the data does not mark bridges positively, it marks the river and leaves
        the gaps.
        """
        first, last = self.river_rows()
        if last < first:
            return ()
        crossable = [
            cx
            for cx in range(self.half_width)
            if not (self.cell(cx, first) & Tile.WATER)
        ]
        spans: list[tuple[int, int]] = []
        for cx in crossable:
            if spans and cx == spans[-1][1] + 1:
                spans[-1] = (spans[-1][0], cx)
            else:
                spans.append((cx, cx))
        return tuple(
            (
                lo * _SUBTILES_PER_HALF_TILE,
                (hi + 1) * _SUBTILES_PER_HALF_TILE,
                (lo + hi + 1) * _SUBTILES_PER_HALF_TILE // 2,
            )
            for lo, hi in spans
        )

    def midline(self) -> int:
        """The halfway line in subtiles -- the boundary between the two sides."""
        return self.height // 2

    def nearest_bridge(self, x: int) -> tuple[int, int, int]:
        """The bridge whose centre is closest to ``x``."""
        bridges = self.bridges()
        return min(bridges, key=lambda b: abs(b[2] - x))

    # ------------------------------------------------------------- deployment

    def own_half(self, team: Team) -> tuple[int, int]:
        """The ``(low, high)`` subtile y-range a team may deploy in.

        Deployment stops at the near bank of the river, not at the midline --
        the river itself is never a legal placement.
        """
        top, bottom = self.river_band()
        if team is Team.BLUE:
            return 0, top
        return bottom, self.height

    def can_deploy(
        self,
        team: Team,
        x: int,
        y: int,
        *,
        anywhere: bool = False,
        on_water: bool = False,
    ) -> bool:
        """Whether ``team`` may place a card at this point.

        ``anywhere`` covers the cards flagged ``CanDeployOnEnemySide`` -- every
        spell, plus Miner and Goblin Drill. ``on_water`` covers spells, which
        may be cast over the river where troops may not stand.
        """
        if not self.in_bounds(x, y):
            return False
        flags = self.flags_at(x, y)
        if flags & Tile.BLOCKED:
            return False
        if (flags & Tile.WATER) and not on_water:
            return False
        if anywhere:
            return True
        low, high = self.own_half(team)
        return low <= y < high

    # ---------------------------------------------------------------- towers

    def towers_for(self, team: Team) -> tuple[TowerPlacement, ...]:
        return tuple(t for t in self.towers if t.team is team)

    def king_tower(self, team: Team) -> TowerPlacement | None:
        for tower in self.towers_for(team):
            if "King" in tower.name:
                return tower
        return None

    def princess_towers(self, team: Team) -> tuple[TowerPlacement, ...]:
        return tuple(t for t in self.towers_for(team) if "King" not in t.name)

    # ------------------------------------------------------------------ debug

    def render(self) -> str:  # pragma: no cover - debugging aid
        glyphs = []
        for cy in range(self.half_height):
            row = []
            for cx in range(self.half_width):
                flags = self.cell(cx, cy)
                if flags & Tile.WATER:
                    row.append("~")
                elif flags & Tile.BLOCKED:
                    row.append("#")
                elif flags & (Tile.LANE_LEFT | Tile.LANE_RIGHT):
                    row.append(":")
                else:
                    row.append(".")
            glyphs.append("".join(row))
        return "\n".join(glyphs)


def _read_tilemap(path: str | Path) -> tuple[tuple[int, ...], int, int]:
    """Parse a tilemap CSV into a flat cell array.

    The file uses the same header/type-row preamble as the logic tables, and its
    first column is an unused row label. A few event tilemaps put entity *names*
    in cells rather than integers; those are treated as ordinary ground, since
    structure placement comes from ``spawn_groups`` rather than the terrain.
    """
    rows = list(csv.reader(Path(path).open(encoding="utf-8-sig")))
    if len(rows) < 3:
        raise ValueError(f"{path} is not a tilemap")
    body = rows[3:]

    grid: list[list[int]] = []
    for row in body:
        if len(row) < 2:
            continue
        parsed: list[int] = []
        for text in row[1:]:
            text = text.strip()
            if not text:
                parsed.append(0)
                continue
            try:
                parsed.append(int(text))
            except ValueError:
                parsed.append(0)

        grid.append(parsed)

    width = max(len(r) for r in grid)
    # Trailing all-zero rows are padding, not playable board.
    while grid and not any(grid[-1]):
        grid.pop()
    height = len(grid)
    cells: list[int] = []
    for row in grid:
        padded = row + [0] * (width - len(row))
        cells.extend(padded)
    return tuple(cells), width, height


def _tower_placements(
    data: LogicData, group: str, half_height: int
) -> tuple[TowerPlacement, ...]:
    """Read one side's structures and mirror them for the opponent.

    ``spawn_groups.toml`` records a single side; the other is the reflection
    ``y -> half_height - y``, which is why the file only ever lists three
    objects for a two-player match.
    """
    body: Mapping[str, object] = data.namespace("SPAWN_GROUP").get(group, {})
    objects = body.get("Objects") if isinstance(body, dict) else None
    if not isinstance(objects, Sequence):
        return ()

    placements: list[TowerPlacement] = []
    for entry in objects:
        if not isinstance(entry, dict):
            continue
        name = entry.get("Data")
        hx, hy = entry.get("x"), entry.get("y")
        if not isinstance(name, str) or not isinstance(hx, int) or not isinstance(hy, int):
            continue
        placements.append(
            TowerPlacement(name=name, team=Team.BLUE, x=half_tiles(hx), y=half_tiles(hy))
        )
        placements.append(
            TowerPlacement(
                name=name, team=Team.RED, x=half_tiles(hx), y=half_tiles(half_height - hy)
            )
        )
    return tuple(placements)


def load_arena(
    data: LogicData,
    tilemap: str | Path | None = None,
    *,
    spawn_group: str = "King_PrincessTowers",
) -> Arena:
    """Build the standard arena from the extracted tilemap and spawn group."""
    if tilemap is None:
        tilemap = Path(data.root).parent / "tilemaps" / "tilemap.csv"
    cells, width, height = _read_tilemap(tilemap)
    return Arena(
        cells=cells,
        half_width=width,
        half_height=height,
        towers=_tower_placements(data, spawn_group, height),
        source=str(Path(tilemap).name),
    )
