"""Screen pixel <-> arena tile <-> engine subtile mapping.

This is the piece that actually closes accuracy questions. A recording from
:mod:`cr_sim.mumu.capture` is only useful for settling something like
``bridge-width`` or ``building-collision-shape`` (see ``reference/anchors.json``)
if a pixel in a captured frame can be turned into the same tile coordinate the
engine itself reasons in, and back. Connection and capture get you the frame;
this module is what makes the frame comparable to a simulated tick.

The arena's own dimensions are not repeated here as separate magic numbers --
they are read out of :mod:`cr_sim.engine.arena` (``HALF_TILES_WIDE`` /
``HALF_TILES_TALL``, 36 x 64 half-tiles = 18 x 32 tiles) and the tile <-> subtile
conversion is delegated to :mod:`cr_sim.engine.fixed` (``tiles`` / ``to_tiles``),
so the board's size and the definition of a subtile each have exactly one
source of truth in the codebase.

**Why a perspective transform and not a simple scale-and-offset.** Clash
Royale's arena camera is tilted, not top-down: the near edge of the playfield
(the bottom of a portrait screen, closest to the camera) spans more pixels for
the same 18-tile width than the far edge does. An affine map -- 6 degrees of
freedom, no cross terms between x and y -- cannot represent that convergence;
two parallel tile-space lines would stay parallel in pixel space, and they
visibly do not in a screenshot of the game. A homography (8 DOF, a 3x3 matrix
up to scale) is the standard fix and is exactly determined by 4 point
correspondences, which is why calibration only ever needs the four playfield
corners.

**Why numpy and not OpenCV for the homography.** A homography from 4 exact
correspondences is a linear algebra problem, not a computer-vision one: stack
two equations per correspondence into an 8x9 matrix and the homography (up to
scale) is that matrix's null vector, found by SVD. ``numpy.linalg.svd`` gives
this directly with no approximation beyond floating point, and the same
routine degrades gracefully into a least-squares fit if more than 4
correspondences are ever supplied. Pulling in OpenCV for a 9-line closed-form
solve would be a dependency this project does not otherwise need.

**The default calibration is a placeholder, not a measurement.** There is no
MuMu install on the machine this module was written on (checked: absent from
PATH and every known install location -- see ``adb.py``'s module docstring),
so ``DEFAULT_CALIBRATION`` below could not be checked against a real
screenshot. Its numbers are a starting guess from the general shape of the
game's HUD (a timer bar eats the top of the screen, the card hand eats the
bottom) and its label says so. Treat it the way ``reference/anchors.json``
treats an unconfirmed figure: usable to keep the code running, not evidence of
anything, until :func:`calibrate_from_corners` replaces it with corners read
off an actual capture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..engine.arena import HALF_TILES_TALL, HALF_TILES_WIDE
from ..engine.fixed import tiles as _tile_length_to_subtiles
from ..engine.fixed import to_tiles as _subtiles_to_tile_length

__all__ = [
    "Point",
    "ARENA_WIDTH_TILES",
    "ARENA_HEIGHT_TILES",
    "ArenaCalibration",
    "DEFAULT_CALIBRATION",
    "CoordinateMapper",
    "calibrate_from_corners",
    "TileOutOfBoundsError",
    "tile_in_bounds",
    "require_tile_in_bounds",
]

#: A screen or tile coordinate pair. Screen points are pixels (origin top-left,
#: y down, matching a captured frame's own (row, col) indexing); tile points
#: are arena tiles (origin at the corner reached by tile (0, 0), independent of
#: which team's side that happens to be for a given recording -- this module
#: does not know or care which account is "you").
Point = tuple[float, float]

#: The arena is 18 tiles wide and 32 tall. Read from cr_sim.engine.arena's own
#: half-tile constants rather than restated, so a future change to the arena
#: module cannot silently desync this mapping from the engine it validates.
ARENA_WIDTH_TILES: float = HALF_TILES_WIDE / 2
ARENA_HEIGHT_TILES: float = HALF_TILES_TALL / 2

#: Placeholder playfield corners for a 1080x1920 portrait MuMu window, in
#: screen pixels. UNVERIFIED -- see the module docstring. Ordered so each one
#: lines up with the tile corner of the same name: top_left <-> tile (0, 0),
#: top_right <-> tile (ARENA_WIDTH_TILES, 0), and so on down the playfield.
_DEFAULT_TOP_LEFT_PX: Point = (220.0, 180.0)
_DEFAULT_TOP_RIGHT_PX: Point = (860.0, 180.0)
_DEFAULT_BOTTOM_LEFT_PX: Point = (40.0, 1550.0)
_DEFAULT_BOTTOM_RIGHT_PX: Point = (1040.0, 1550.0)
_DEFAULT_SCREEN_WIDTH = 1080
_DEFAULT_SCREEN_HEIGHT = 1920


@dataclass(frozen=True, slots=True)
class ArenaCalibration:
    """The screen-pixel quadrilateral one playfield occupies, plus the
    resolution it was measured on.

    Pure data -- JSON round-trippable via :meth:`save` / :meth:`load` -- kept
    separate from :class:`CoordinateMapper` so a calibration can be recorded,
    diffed, or shipped alongside a frame recording without also carrying
    numpy arrays.
    """

    top_left: Point
    top_right: Point
    bottom_left: Point
    bottom_right: Point
    screen_width: int = _DEFAULT_SCREEN_WIDTH
    screen_height: int = _DEFAULT_SCREEN_HEIGHT
    #: Free-text provenance, e.g. "measured-2026-08-26" versus the shipped
    #: "placeholder-...-unverified" default. Not used for anything but keeping
    #: honest track of what a given calibration actually is.
    label: str = "unlabelled"

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_left": list(self.top_left),
            "top_right": list(self.top_right),
            "bottom_left": list(self.bottom_left),
            "bottom_right": list(self.bottom_right),
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArenaCalibration":
        return cls(
            top_left=tuple(data["top_left"]),
            top_right=tuple(data["top_right"]),
            bottom_left=tuple(data["bottom_left"]),
            bottom_right=tuple(data["bottom_right"]),
            screen_width=int(data["screen_width"]),
            screen_height=int(data["screen_height"]),
            label=str(data.get("label", "unlabelled")),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ArenaCalibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


DEFAULT_CALIBRATION = ArenaCalibration(
    top_left=_DEFAULT_TOP_LEFT_PX,
    top_right=_DEFAULT_TOP_RIGHT_PX,
    bottom_left=_DEFAULT_BOTTOM_LEFT_PX,
    bottom_right=_DEFAULT_BOTTOM_RIGHT_PX,
    screen_width=_DEFAULT_SCREEN_WIDTH,
    screen_height=_DEFAULT_SCREEN_HEIGHT,
    label="placeholder-1080x1920-portrait-unverified",
)


def _homography_from_points(src: Sequence[Point], dst: Sequence[Point]) -> np.ndarray:
    """3x3 homography mapping each ``src`` point to the matching ``dst`` point.

    Direct Linear Transform: every correspondence contributes two rows to an
    (2n x 9) matrix, and the homography -- up to the arbitrary scale a
    homogeneous matrix always has -- is that matrix's right null vector. SVD's
    smallest singular vector gives the null vector exactly when n == 4 (an
    8x9 matrix has an exact 1-dimensional null space) and the corresponding
    least-squares fit when n > 4, so this one routine covers both the exact
    4-corner calibration this module actually uses and any future
    over-determined one without a separate code path.
    """
    if len(src) != len(dst):
        raise ValueError("src and dst must have the same number of points")
    if len(src) < 4:
        raise ValueError("need at least 4 point correspondences to fit a homography")

    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    a = np.array(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(a)
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def _apply_homography(h: np.ndarray, point: Point) -> Point:
    x, y = point
    vec = h @ np.array([x, y, 1.0])
    return float(vec[0] / vec[2]), float(vec[1] / vec[2])


class TileOutOfBoundsError(ValueError):
    """A tile coordinate fell outside the 18x32 arena.

    Raised rather than silently clamped: a caller mapping a screen tap to a
    tile that lands off the board almost always means the calibration is
    wrong or the tap missed the playfield entirely, and clamping would hide
    exactly the bug calibration work is trying to surface.
    """


def tile_in_bounds(tile_x: float, tile_y: float) -> bool:
    return 0.0 <= tile_x <= ARENA_WIDTH_TILES and 0.0 <= tile_y <= ARENA_HEIGHT_TILES


def require_tile_in_bounds(tile_x: float, tile_y: float) -> None:
    if not tile_in_bounds(tile_x, tile_y):
        raise TileOutOfBoundsError(
            f"({tile_x}, {tile_y}) is outside the {ARENA_WIDTH_TILES}x{ARENA_HEIGHT_TILES} arena"
        )


class CoordinateMapper:
    """Bidirectional screen-pixel <-> arena-tile <-> engine-subtile mapping,
    built once from an :class:`ArenaCalibration`.

    Two homographies are fit independently -- tile -> pixel and pixel -> tile
    -- rather than inverting one 3x3 matrix, because both directions are used
    often enough (placing a card from a tile, reading a detected unit's tile
    from its pixel position) that keeping the code path identical for both
    is worth the second SVD, which costs nothing measurable at 4 points.
    """

    def __init__(self, calibration: ArenaCalibration) -> None:
        self.calibration = calibration
        tile_corners: tuple[Point, Point, Point, Point] = (
            (0.0, 0.0),
            (ARENA_WIDTH_TILES, 0.0),
            (0.0, ARENA_HEIGHT_TILES),
            (ARENA_WIDTH_TILES, ARENA_HEIGHT_TILES),
        )
        px_corners: tuple[Point, Point, Point, Point] = (
            calibration.top_left,
            calibration.top_right,
            calibration.bottom_left,
            calibration.bottom_right,
        )
        self._tile_to_px = _homography_from_points(tile_corners, px_corners)
        self._px_to_tile = _homography_from_points(px_corners, tile_corners)

    def tile_to_screen(self, tile_x: float, tile_y: float) -> Point:
        return _apply_homography(self._tile_to_px, (tile_x, tile_y))

    def screen_to_tile(self, px: float, py: float) -> Point:
        return _apply_homography(self._px_to_tile, (px, py))

    def tile_to_subtile(self, tile_x: float, tile_y: float) -> tuple[int, int]:
        return _tile_length_to_subtiles(tile_x), _tile_length_to_subtiles(tile_y)

    def subtile_to_tile(self, x: int, y: int) -> Point:
        return _subtiles_to_tile_length(x), _subtiles_to_tile_length(y)

    def screen_to_subtile(self, px: float, py: float) -> tuple[int, int]:
        tile_x, tile_y = self.screen_to_tile(px, py)
        return self.tile_to_subtile(tile_x, tile_y)

    def subtile_to_screen(self, x: int, y: int) -> Point:
        tile_x, tile_y = self.subtile_to_tile(x, y)
        return self.tile_to_screen(tile_x, tile_y)


def calibrate_from_corners(
    top_left: Point,
    top_right: Point,
    bottom_left: Point,
    bottom_right: Point,
    *,
    screen_width: int,
    screen_height: int,
    label: str = "custom",
) -> CoordinateMapper:
    """Build a mapper directly from four measured playfield corners.

    This is the actual calibration path: screenshot the emulator (see
    :func:`cr_sim.mumu.capture.screencap`), find the four corners of the
    playing field in pixel coordinates -- by eye is enough, this only needs
    to be done once per resolution -- and hand them to this function in the
    same top-left/top-right/bottom-left/bottom-right order
    :class:`ArenaCalibration` uses.
    """
    calibration = ArenaCalibration(
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        screen_width=screen_width,
        screen_height=screen_height,
        label=label,
    )
    return CoordinateMapper(calibration)
