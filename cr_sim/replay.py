"""Deterministic state hashing and replay.

A battle is fully described by ``(seed, configuration, command list)``. Nothing
else may influence the outcome -- no wall clock, no dict iteration order, no
floating point. This module is how that claim gets *enforced* rather than
merely intended.

:func:`state_hash` folds the entire battle state into one integer each tick. Two
runs that agree on every tick's hash are identical simulations; two that diverge
report the exact tick where they first differed, which turns "the replay
desyncs" from a hunt into a single-step diff. It is also what lets the 20 TPS
training runs be compared against the 60 TPS verification runs.

The hash deliberately covers only *simulation* state. Anything cosmetic or
derived is excluded, so adding a debug counter or a render hint cannot
invalidate a stored replay.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .engine.entity import Entity

__all__ = ["state_hash", "Command", "Replay", "DivergenceError", "compare_hashes"]

_HASH_BYTES = 8


class DivergenceError(RuntimeError):
    """Raised when two runs of the same seed disagree."""


def state_hash(tick: int, entities: Iterable[Entity], extra: Sequence[int] = ()) -> int:
    """Fold the whole simulation state into one integer.

    Entities are hashed in list order, which is spawn order and never mutated
    mid-tick, so the digest is stable without needing to sort. Only fields that
    can affect the future are included -- position, health, lifecycle, target --
    because hashing a cosmetic field would make harmless changes look like
    desyncs.
    """
    digest = hashlib.blake2b(digest_size=_HASH_BYTES)
    digest.update(tick.to_bytes(4, "little"))
    for entity in entities:
        digest.update(
            b"".join(
                (
                    entity.id.to_bytes(4, "little", signed=False),
                    int(entity.team).to_bytes(1, "little"),
                    int(entity.kind).to_bytes(1, "little"),
                    int(entity.state).to_bytes(1, "little"),
                    (1 if entity.dead else 0).to_bytes(1, "little"),
                    (entity.x & 0xFFFFFFFF).to_bytes(4, "little"),
                    (entity.y & 0xFFFFFFFF).to_bytes(4, "little"),
                    max(0, entity.hitpoints).to_bytes(4, "little"),
                    max(0, entity.shield).to_bytes(4, "little"),
                    max(0, entity.deploy_ticks_left).to_bytes(2, "little"),
                    (entity.target_id & 0xFFFFFFFF).to_bytes(4, "little"),
                )
            )
        )
    for value in extra:
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return int.from_bytes(digest.digest(), "little")


@dataclass(frozen=True, slots=True)
class Command:
    """A player action: play ``card`` at a board position on a given tick."""

    tick: int
    team: int
    card: str
    x: int
    y: int

    def as_json(self) -> dict[str, Any]:
        return {"tick": self.tick, "team": self.team, "card": self.card, "x": self.x, "y": self.y}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "Command":
        return cls(
            tick=payload["tick"],
            team=payload["team"],
            card=payload["card"],
            x=payload["x"],
            y=payload["y"],
        )


@dataclass(slots=True)
class Replay:
    """Everything needed to reproduce a battle exactly."""

    seed: int
    ticks_per_second: int
    decks: dict[str, list[str]] = field(default_factory=dict)
    levels: dict[str, int] = field(default_factory=dict)
    commands: list[Command] = field(default_factory=list)
    #: Per-tick state hashes, when recorded. Optional because storing one per
    #: tick costs ~85KB per battle and is only needed while verifying.
    hashes: list[int] = field(default_factory=list)
    #: Optional per-tick entity snapshots for the viewer. Never hashed.
    frames: list[dict[str, Any]] = field(default_factory=list)
    build: str = "unknown"

    def add(self, command: Command) -> None:
        self.commands.append(command)

    def commands_for_tick(self, tick: int) -> list[Command]:
        return [c for c in self.commands if c.tick == tick]

    def by_tick(self) -> dict[int, list[Command]]:
        """Commands indexed by tick, so playback is O(1) per tick."""
        grouped: dict[int, list[Command]] = {}
        for command in self.commands:
            grouped.setdefault(command.tick, []).append(command)
        return grouped

    # ------------------------------------------------------------ persistence

    def to_json(self, *, include_frames: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "seed": self.seed,
            "ticks_per_second": self.ticks_per_second,
            "build": self.build,
            "decks": self.decks,
            "levels": self.levels,
            "commands": [c.as_json() for c in self.commands],
        }
        if self.hashes:
            payload["hashes"] = [str(h) for h in self.hashes]
        if include_frames and self.frames:
            payload["frames"] = self.frames
        return payload

    def save(self, path: str | Path, *, include_frames: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_json(include_frames=include_frames)), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Replay":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            seed=payload["seed"],
            ticks_per_second=payload["ticks_per_second"],
            build=payload.get("build", "unknown"),
            decks=payload.get("decks", {}),
            levels=payload.get("levels", {}),
            commands=[Command.from_json(c) for c in payload.get("commands", [])],
            hashes=[int(h) for h in payload.get("hashes", [])],
            frames=payload.get("frames", []),
        )


def compare_hashes(left: Sequence[int], right: Sequence[int]) -> int | None:
    """First tick at which two hash streams disagree, or ``None`` if identical.

    Returning the tick rather than a bare boolean is the whole point: it says
    where to look.
    """
    for tick, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return tick
    if len(left) != len(right):
        return min(len(left), len(right))
    return None
