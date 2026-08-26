"""The M0 stat gate.

The ingestion pipeline turns raw game files into scaled per-level numbers via a
chain of assumptions -- the milli-tile unit, the shared power ladder, the rarity
offset, truncation toward zero, damage living on the projectile for ranged
units.  Every one of those is a place to be quietly wrong, and a simulator that
is quietly wrong about Knight's hitpoints is worthless.

So the pipeline is pinned against a frozen table of values checked against the
live game (``reference/card_stats.json``).  A mismatch is reported per field
rather than raised, because a *new* game build legitimately changes balance
numbers: the report tells you which cards moved so you can re-check and re-freeze
deliberately, instead of silently trusting stale data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .cards import TOURNAMENT_DISPLAY_LEVEL, CardRegistry, card_stat_summary
from .leveling import LevelTable
from .source import LogicData

__all__ = ["Mismatch", "ValidationReport", "validate_cards", "load_reference", "write_reference"]

DEFAULT_REFERENCE = Path(__file__).resolve().parents[2] / "reference" / "card_stats.json"

#: Fields compared against the reference, and whether a difference is fatal.
CHECKED_FIELDS = (
    "elixir",
    "rarity",
    "hitpoints",
    "damage",
    "hit_speed",
    "range",
    "speed",
    "count",
    "dps",
    "radius",
    "damage_per_second",
    "damage_source",
)


@dataclass(frozen=True, slots=True)
class Mismatch:
    card: str
    field: str
    expected: Any
    actual: Any

    def __str__(self) -> str:
        return f"{self.card}.{self.field}: expected {self.expected!r}, got {self.actual!r}"


@dataclass(slots=True)
class ValidationReport:
    checked: int = 0
    matched: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)
    missing_from_build: list[str] = field(default_factory=list)
    missing_from_reference: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches and not self.missing_from_build

    def summary(self) -> str:
        lines = [
            f"checked {self.checked} field(s) across the reference set: "
            f"{self.matched} matched, {len(self.mismatches)} mismatched"
        ]
        if self.missing_from_build:
            lines.append(
                f"  {len(self.missing_from_build)} reference card(s) absent from this build: "
                + ", ".join(self.missing_from_build[:10])
            )
        if self.missing_from_reference:
            lines.append(
                f"  {len(self.missing_from_reference)} card(s) in this build have no reference entry"
            )
        for mismatch in self.mismatches[:40]:
            lines.append(f"  ! {mismatch}")
        if len(self.mismatches) > 40:
            lines.append(f"  ... and {len(self.mismatches) - 40} more")
        return "\n".join(lines)


def load_reference(path: str | Path = DEFAULT_REFERENCE) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["cards"]


def validate_cards(
    data: LogicData,
    levels: LevelTable,
    registry: CardRegistry,
    reference: Mapping[str, Mapping[str, Any]],
    *,
    display_level: int = TOURNAMENT_DISPLAY_LEVEL,
    fields: Iterable[str] = CHECKED_FIELDS,
) -> ValidationReport:
    """Compare computed stats against the frozen reference."""
    report = ValidationReport()
    fields = tuple(fields)

    for name, expected in reference.items():
        card = registry.get(name)
        if card is None:
            report.missing_from_build.append(name)
            continue
        actual = card_stat_summary(data, levels, card, display_level=display_level)
        for key in fields:
            if key not in expected:
                continue
            report.checked += 1
            if actual.get(key) == expected[key]:
                report.matched += 1
            else:
                report.mismatches.append(
                    Mismatch(card=name, field=key, expected=expected[key], actual=actual.get(key))
                )

    for card in registry.standard():
        if card.name not in reference:
            report.missing_from_reference.append(card.name)
    return report


def write_reference(
    data: LogicData,
    levels: LevelTable,
    registry: CardRegistry,
    path: str | Path = DEFAULT_REFERENCE,
    *,
    display_level: int = TOURNAMENT_DISPLAY_LEVEL,
    build: str = "unknown",
    only: Iterable[str] | None = None,
) -> int:
    """Freeze the current build's computed stats as the reference table.

    Only run this deliberately, after spot-checking against the live game -- it
    is what the gate compares against, so regenerating it blindly defeats it.
    """
    names = set(only) if only is not None else {c.name for c in registry.standard()}
    cards: dict[str, dict[str, Any]] = {}
    for card in registry.standard():
        if card.name not in names:
            continue
        summary = card_stat_summary(data, levels, card, display_level=display_level)
        cards[card.name] = {k: summary[k] for k in CHECKED_FIELDS if k in summary}

    payload = {
        "_comment": (
            "Frozen expected card stats at the displayed tournament level. "
            "Regenerate only after checking against the live game."
        ),
        "build": build,
        "display_level": display_level,
        "cards": dict(sorted(cards.items())),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return len(cards)
