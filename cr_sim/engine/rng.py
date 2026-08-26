"""Deterministic random numbers.

Python's :mod:`random` is not usable here: it is a shared global, its stream can
be perturbed by anything else in the process, and its internals are free to
change between releases. A simulator whose replays must reproduce exactly needs
a generator it owns.

This is PCG-XSH-RR 32-bit -- small, fast, statistically sound, and specified
precisely enough that the same seed yields the same stream forever.

Every source of randomness in the battle draws from one instance owned by the
battle, so a battle is fully described by ``(seed, command list)``. Subsystems
that need independent streams (deck shuffling versus spawn jitter) take a
:meth:`Rng.stream` child rather than interleaving draws on the shared one --
otherwise adding a single draw in one subsystem would shift every later draw in
all the others and silently invalidate saved replays.
"""

from __future__ import annotations

__all__ = ["Rng"]

_MASK64 = (1 << 64) - 1
_MASK32 = (1 << 32) - 1
_MULTIPLIER = 6364136223846793005
_DEFAULT_INCREMENT = 1442695040888963407


class Rng:
    """A seeded PCG32 stream."""

    __slots__ = ("_state", "_increment")

    def __init__(self, seed: int = 0, increment: int = _DEFAULT_INCREMENT) -> None:
        # The increment must be odd for the LCG to have full period.
        self._increment = ((increment << 1) | 1) & _MASK64
        self._state = 0
        self._step()
        self._state = (self._state + (seed & _MASK64)) & _MASK64
        self._step()

    def _step(self) -> None:
        self._state = (self._state * _MULTIPLIER + self._increment) & _MASK64

    def next_u32(self) -> int:
        state = self._state
        self._step()
        # XSH-RR output permutation.
        xorshifted = (((state >> 18) ^ state) >> 27) & _MASK32
        rotation = (state >> 59) & 31
        return ((xorshifted >> rotation) | (xorshifted << ((-rotation) & 31))) & _MASK32

    def below(self, bound: int) -> int:
        """Uniform integer in ``[0, bound)``, rejection-sampled to avoid bias.

        The naive ``next_u32() % bound`` skews toward low values whenever bound
        does not divide 2**32. Over millions of simulated battles that bias is
        the kind of thing that quietly teaches an agent something untrue.
        """
        if bound <= 0:
            raise ValueError("bound must be positive")
        if bound == 1:
            return 0
        threshold = (-bound) % bound  # == 2**32 % bound
        while True:
            value = self.next_u32()
            if value >= threshold:
                return value % bound

    def between(self, low: int, high: int) -> int:
        """Uniform integer in the inclusive range ``[low, high]``."""
        if high < low:
            raise ValueError(f"empty range [{low}, {high}]")
        return low + self.below(high - low + 1)

    def chance(self, numerator: int, denominator: int) -> bool:
        """True with probability ``numerator / denominator``."""
        return self.below(denominator) < numerator

    def shuffle(self, items: list) -> None:
        """In-place Fisher-Yates using this stream."""
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]

    def stream(self, label: str) -> "Rng":
        """An independent child stream, derived from this one and ``label``.

        Deriving by label rather than by draw order means a subsystem can add or
        remove draws without shifting anybody else's stream, so existing replays
        stay valid.
        """
        mixed = self._state
        for char in label.encode("utf-8"):
            mixed = ((mixed ^ char) * 0x100000001B3) & _MASK64
        return Rng(seed=mixed, increment=mixed | 1)

    def state(self) -> tuple[int, int]:
        """Full internal state, for snapshotting into a replay."""
        return self._state, self._increment

    def restore(self, state: tuple[int, int]) -> None:
        self._state, self._increment = state

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Rng(state=0x{self._state:016x})"
