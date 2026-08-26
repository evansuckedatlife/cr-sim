"""Synthetic input into the emulator: taps, swipes, and card placement.

Sending input to a live Clash Royale client through automation, rather than a
human touching the screen, is against Supercell's Terms of Service and can
get the account actioned. Whether to accept that risk for a given account is
the caller's decision, not this module's -- so nothing here refuses to run.
Instead, every method that actually injects input requires the caller to say
so explicitly by constructing :class:`InputController` with ``allow_input=True``.
"""

from __future__ import annotations

import time
from typing import Sequence

from .adb import AdbBridge
from .geometry import CoordinateMapper, Point, require_tile_in_bounds

__all__ = [
    "InputNotAllowedError",
    "DEFAULT_CARD_SLOTS",
    "InputController",
]


class InputNotAllowedError(RuntimeError):
    """Raised by every injecting method unless ``allow_input=True`` was
    passed to the controller's constructor."""


#: Screen positions of the four hand slots, matching
#: :data:`cr_sim.mumu.geometry.DEFAULT_CALIBRATION`'s 1080x1920 layout.
#: Unverified for the same reason that calibration's corners are (see
#: geometry.py's module docstring) -- an even spread across the bottom card
#: tray, not a measurement.
DEFAULT_CARD_SLOTS: tuple[Point, Point, Point, Point] = (
    (170.0, 1800.0),
    (410.0, 1800.0),
    (650.0, 1800.0),
    (890.0, 1800.0),
)


class InputController:
    """Taps, swipes, and the two-step card-placement gesture.

    :meth:`play_card` mirrors how the real client reads a placement: select a
    card in the hand (tap its slot), then tap the arena position the unit
    should appear at. A single tap directly on the arena does nothing without
    a card already selected, so this is two device taps for one logical
    action, with a short pause between them for the client's own selection
    animation to register -- untested against a real client on this machine,
    since none is available; treat ``select_delay_s`` as a starting guess to
    tune once one is.
    """

    def __init__(
        self,
        bridge: AdbBridge,
        mapper: CoordinateMapper | None = None,
        *,
        allow_input: bool = False,
        card_slots: Sequence[Point] = DEFAULT_CARD_SLOTS,
    ) -> None:
        self.bridge = bridge
        self.mapper = mapper
        self.allow_input = allow_input
        self.card_slots = tuple(card_slots)

    def _require_allowed(self) -> None:
        if not self.allow_input:
            raise InputNotAllowedError(
                "input injection is disabled by default; construct InputController with "
                "allow_input=True to enable it (see this module's docstring for why)"
            )

    def tap(self, x: int, y: int) -> None:
        self._require_allowed()
        self.bridge.shell(f"input tap {x} {y}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> None:
        self._require_allowed()
        self.bridge.shell(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")

    def play_card(
        self,
        slot: int,
        tile_x: float,
        tile_y: float,
        *,
        select_delay_s: float = 0.15,
    ) -> None:
        """Select the card in ``slot`` (0-indexed into ``card_slots``) and
        place it at arena tile ``(tile_x, tile_y)``."""
        self._require_allowed()
        if self.mapper is None:
            raise ValueError("play_card needs a CoordinateMapper -- construct InputController with one")
        if not 0 <= slot < len(self.card_slots):
            raise ValueError(f"slot {slot} is outside the {len(self.card_slots)} configured card slots")
        require_tile_in_bounds(tile_x, tile_y)

        slot_x, slot_y = self.card_slots[slot]
        self.tap(round(slot_x), round(slot_y))
        time.sleep(select_delay_s)
        px, py = self.mapper.tile_to_screen(tile_x, tile_y)
        self.tap(round(px), round(py))
