"""Engine-wide constants and the tick <-> millisecond conversion.

The game files express every duration in milliseconds, but the engine must run
on whole ticks: a unit either attacks on tick N or it does not. All conversion
happens here, once, at spec-build time -- no millisecond value should reach the
hot loop.

**Why 60 ticks per second.** Every duration in the files is a multiple of 50ms,
and 50ms is exactly 3 ticks at 60Hz, so 60 TPS can represent the game's own
timing grid without rounding. It is also an exact multiple of 20 TPS, which
means the same battle can be simulated at 20 TPS -- a third of the work -- for
bulk training, and any divergence between the two is a real signal about which
mechanics are tick-sensitive rather than an artefact of rounding.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "TICKS_PER_SECOND",
    "VERIFICATION_TPS",
    "TRAINING_TPS",
    "TickClock",
    "MAX_ELIXIR",
    "ELIXIR_PRECISION",
]

#: Default engine rate. See the module docstring.
TICKS_PER_SECOND = 60

#: The rate correctness is verified at, and the cheaper rate used for training.
VERIFICATION_TPS = 60
TRAINING_TPS = 20

#: Elixir is tracked in fixed-point thousandths of a unit so the 2800/1400/930ms
#: regeneration rates divide evenly without floats.
ELIXIR_PRECISION = 1000
MAX_ELIXIR = 10


@dataclass(frozen=True, slots=True)
class TickClock:
    """Converts the files' milliseconds into this run's ticks."""

    ticks_per_second: int = TICKS_PER_SECOND

    def ticks(self, milliseconds: int | None, default: int = 0) -> int:
        """Milliseconds -> whole ticks, rounded half-up.

        Rounding rather than truncating matters: a 700ms load time at 20 TPS is
        14 ticks either way, but an odd value like 350ms is 7 ticks rounded and
        6 truncated, and a whole tick of windup decides close interactions.
        """
        if milliseconds is None:
            return default
        return (milliseconds * self.ticks_per_second + 500) // 1000

    def milliseconds(self, ticks: int) -> int:
        return ticks * 1000 // self.ticks_per_second

    def seconds_to_ticks(self, seconds: int) -> int:
        return seconds * self.ticks_per_second

    def subtiles_per_tick(self, speed: int) -> int:
        """A ``Speed`` in tiles/minute -> subtiles travelled per tick.

        Exact at both 60 and 20 TPS; see :mod:`cr_sim.engine.fixed`.
        """
        from .fixed import SUBTILES_PER_TILE

        return speed * SUBTILES_PER_TILE // (60 * self.ticks_per_second)

    def is_exact_for(self, speed: int) -> bool:
        """Whether ``speed`` converts to whole subtiles per tick with no remainder."""
        from .fixed import SUBTILES_PER_TILE

        return (speed * SUBTILES_PER_TILE) % (60 * self.ticks_per_second) == 0
