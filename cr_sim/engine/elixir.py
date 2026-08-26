"""Elixir regeneration and the match timeline.

Both come out of ``battle_timelines.csv`` rather than being hard-coded, because
they are one schedule: the rate changes are defined as a sequence of segments
that runs across regulation *and* overtime without resetting.

The ``Default`` timeline reads:

===============  ==========  ==================
segment          length      per elixir
===============  ==========  ==================
0:00 - 2:00      120s        2800ms  (1x)
2:00 - 4:00      120s        1400ms  (2x)
4:00 - 5:00      60s          930ms  (3x)
===============  ==========  ==================

with 180s of regulation and 120s of overtime and 6 starting elixir. Note the
segments do not line up with the period boundaries: the 2x segment starts one
minute before regulation ends and continues a minute into overtime. Hard-coding
"double elixir in overtime" would get that wrong in both directions.

Elixir is held in fixed-point thousandths so the three rates divide evenly and
no float ever touches the bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..data.csv_loader import Table
from ..data.source import LogicData
from .constants import ELIXIR_PRECISION, MAX_ELIXIR, TickClock

__all__ = ["ElixirSegment", "BattleTimeline", "ElixirBar", "build_timeline"]


@dataclass(frozen=True, slots=True)
class ElixirSegment:
    """One constant-rate stretch of the match."""

    start_tick: int
    end_tick: int
    ms_per_elixir: int
    multiplier_tenths: int

    @property
    def multiplier(self) -> float:
        return self.multiplier_tenths / 10


@dataclass(frozen=True, slots=True)
class BattleTimeline:
    """The match's shape: how long it runs and how fast elixir arrives."""

    regulation_ticks: int
    overtime_ticks: int
    starting_elixir: int
    segments: tuple[ElixirSegment, ...]
    clock: TickClock

    @property
    def total_ticks(self) -> int:
        return self.regulation_ticks + self.overtime_ticks

    def segment_at(self, tick: int) -> ElixirSegment:
        for segment in self.segments:
            if tick < segment.end_tick:
                return segment
        return self.segments[-1]

    def ticks_per_elixir(self, tick: int) -> int:
        """Whole ticks to regenerate one elixir at ``tick``'s rate."""
        return self.clock.ticks(self.segment_at(tick).ms_per_elixir)

    def elixir_gain_per_tick(self, tick: int) -> int:
        """Fixed-point elixir added each tick at the current rate.

        Computed as a per-tick increment rather than "one elixir every N ticks"
        so that a rate change mid-regeneration carries the partial progress
        across, which is what the in-game bar does.
        """
        ticks = self.ticks_per_elixir(tick)
        if ticks <= 0:
            return ELIXIR_PRECISION
        return ELIXIR_PRECISION // ticks

    def is_overtime(self, tick: int) -> bool:
        return tick >= self.regulation_ticks


@dataclass(slots=True)
class ElixirBar:
    """One player's elixir, in fixed-point thousandths."""

    timeline: BattleTimeline
    amount: int = 0
    #: Fractional remainder carried between ticks so integer division does not
    #: quietly lose elixir over a three-minute match.
    _remainder: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.amount == 0:
            self.amount = self.timeline.starting_elixir * ELIXIR_PRECISION

    @property
    def units(self) -> int:
        """Whole elixir available to spend."""
        return self.amount // ELIXIR_PRECISION

    @property
    def exact(self) -> float:
        """Elixir as a float. Display only -- never used for a spend decision."""
        return self.amount / ELIXIR_PRECISION

    def regenerate(self, tick: int) -> None:
        ticks = self.timeline.ticks_per_elixir(tick)
        if ticks <= 0:
            return
        # Distribute one elixir over `ticks` ticks exactly, carrying the
        # remainder so no fraction is lost to truncation.
        self._remainder += ELIXIR_PRECISION
        gain, self._remainder = divmod(self._remainder, ticks)
        cap = MAX_ELIXIR * ELIXIR_PRECISION
        self.amount = min(cap, self.amount + gain)
        if self.amount >= cap:
            self._remainder = 0

    def can_afford(self, cost: int) -> bool:
        return self.units >= cost

    def spend(self, cost: int) -> bool:
        if not self.can_afford(cost):
            return False
        self.amount -= cost * ELIXIR_PRECISION
        return True

    def add(self, units: int) -> None:
        """Grant elixir directly (Elixir Collector, Mirror refunds)."""
        cap = MAX_ELIXIR * ELIXIR_PRECISION
        self.amount = min(cap, self.amount + units * ELIXIR_PRECISION)


def build_timeline(
    data: LogicData, name: str = "Default", clock: TickClock | None = None
) -> BattleTimeline:
    """Read a named timeline out of ``battle_timelines.csv``."""
    clock = clock or TickClock()
    table: Table | None = data.tables.get("battle_timelines")
    if table is None:
        raise KeyError("battle_timelines.csv is missing from this build")
    record = table.get(name)
    if record is None:
        raise KeyError(f"no battle timeline named {name!r}")

    section_lengths = record.array("SectionLength")
    section_types = record.array("SectionType")
    regulation = overtime = 0
    for length, kind in zip(section_lengths, section_types):
        if kind == "Overtime":
            overtime += length
        else:
            regulation += length

    rate_lengths = record.array("ElixirRateLength")
    full_bar_ms = record.array("ElixirFullBarMS")
    visible = record.array("ElixirRateVisible")

    segments: list[ElixirSegment] = []
    at = 0
    for index, seconds in enumerate(rate_lengths):
        # ElixirFullBarMS is the time to fill all ten; one elixir is a tenth.
        ms_each = full_bar_ms[index] // MAX_ELIXIR
        start = clock.seconds_to_ticks(at)
        at += seconds
        segments.append(
            ElixirSegment(
                start_tick=start,
                end_tick=clock.seconds_to_ticks(at),
                ms_per_elixir=ms_each,
                multiplier_tenths=visible[index] if index < len(visible) else 10,
            )
        )

    return BattleTimeline(
        regulation_ticks=clock.seconds_to_ticks(regulation),
        overtime_ticks=clock.seconds_to_ticks(overtime),
        starting_elixir=record.scalar("StartingElixir", 0),
        segments=tuple(segments),
        clock=clock,
    )
