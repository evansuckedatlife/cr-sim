"""ADB bridge to a MuMu Android emulator instance.

This exists to validate cr-sim against the real game, not to play it. Several
entries in ``reference/anchors.json``'s ``open_questions`` -- ``bridge-width``,
``lightning-bolt-count``, ``heal-spirit-total``, ``building-collision-shape``
-- cannot be settled from the extracted game files alone; they need a
frame-by-frame recording of an actual match compared against the simulator.
Screen capture (:mod:`cr_sim.mumu.capture`) and coordinate calibration
(:mod:`cr_sim.mumu.geometry`) are the load-bearing pieces of this package for
that reason, in that priority order. :mod:`cr_sim.mumu.adb` is the connection
underneath both. :mod:`cr_sim.mumu.input` exists mainly so a recording session
can be driven without a human holding the phone, and is gated behind an
explicit opt-in -- see its own module docstring for why.

:mod:`cr_sim.mumu.geometry` reads the arena's width and height and the
subtile conversion out of ``cr_sim.engine.arena`` / ``cr_sim.engine.fixed``
rather than restating those numbers, so the board's dimensions have exactly
one source of truth across the engine and this package.
"""

from __future__ import annotations

from .adb import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    AdbBridge,
    AdbCommandError,
    AdbDevice,
    AdbError,
    AdbNotFoundError,
    AdbTimeoutError,
    find_adb_binary,
    instance_port,
)
from .capture import FrameRecorder, MissingImagingLibraryError, RecordingSummary, screencap
from .geometry import (
    ARENA_HEIGHT_TILES,
    ARENA_WIDTH_TILES,
    DEFAULT_CALIBRATION,
    ArenaCalibration,
    CoordinateMapper,
    Point,
    TileOutOfBoundsError,
    calibrate_from_corners,
    require_tile_in_bounds,
    tile_in_bounds,
)
from .input import DEFAULT_CARD_SLOTS, InputController, InputNotAllowedError

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "AdbBridge",
    "AdbCommandError",
    "AdbDevice",
    "AdbError",
    "AdbNotFoundError",
    "AdbTimeoutError",
    "find_adb_binary",
    "instance_port",
    "FrameRecorder",
    "MissingImagingLibraryError",
    "RecordingSummary",
    "screencap",
    "ARENA_HEIGHT_TILES",
    "ARENA_WIDTH_TILES",
    "DEFAULT_CALIBRATION",
    "ArenaCalibration",
    "CoordinateMapper",
    "Point",
    "TileOutOfBoundsError",
    "calibrate_from_corners",
    "require_tile_in_bounds",
    "tile_in_bounds",
    "DEFAULT_CARD_SLOTS",
    "InputController",
    "InputNotAllowedError",
]
