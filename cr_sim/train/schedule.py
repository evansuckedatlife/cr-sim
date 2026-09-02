"""Moving the shaping weight over training, and only the weight that is real.

The shaping is eating the learning signal. Under ``--reward projected`` the
reward is an exact potential -- ``r = phi(s') - phi(s)``, verified to 2.2e-16
-- so return-to-go telescopes to ``phi(s_T) - phi(s_t)``, and phi is close to
a martingale: regressing phi_T on phi_t gives a slope of 1.03-1.07 and
R-squared(return-to-go | phi_t) is 0.0027. The consequence is that the
explained-variance ceiling of 0.29 is a property of the *reward*, not of the
critic or of the observation, and GAE's advantages are therefore mostly
noise. That predicts exactly what a million steps of PPO measured: the
sampled arm moved +0.74 sd and greedy +0.024 -- distributional sharpening,
not credit assignment. Annealing the shaping toward zero ends on the sparse
crown objective, where episode-return variance actually lives: 77.4% crowns
against 8.4% tower health.

**The knob is not the one named "shaping".** ``reward_shaping_weight``
(``--shaping``) and the projected potential's shaping are two different
things sharing a word. Every ``_shaped_value`` call site sits inside the
``else`` of ``if self._reward is not None``, so ``--shaping`` does nothing
whatsoever unless ``--reward simple``. Measured on identical seeds and an
identical action stream, 0.01 against 5.00::

    projected: IDENTICAL      five-term: IDENTICAL      simple: DIFFERS

A five hundred fold change is bit-identical under the two rewards anyone
trains with, so a schedule aimed there is a run that reports an anneal and
performs none. What this module moves instead:

- ``--reward projected``: ``tower`` and ``elixir`` to zero, ``horizon_seconds``
  held.
- ``--reward five-term``: the five non-crown ``RewardWeights`` fields to zero,
  ``crowns`` held.
- ``--reward simple``: ``shaping`` to zero -- the one case where that flag is
  the whole shaping term.

``crowns`` is never annealed under any knob. That is the objective, not
shaping. At the zero endpoint the episode return equals the final crown
difference *exactly* -- verified, returns 2.0/3.0/3.0/2.0 against crown
differences 2/3/3/2 -- so the schedule terminates on the sparse objective
through the same code path rather than through a special case.

**Linear, not cosine or exponential.** The thing being annealed is a
coefficient in a potential, and the only consumer whose behaviour depends on
the *shape* is the critic, which carries its scale across updates -- PPO's
actor normalises advantages per minibatch and is scale-free, the value loss
fits raw returns and is not. A linear ramp is a constant drift the critic can
track; an exponential dumps most of the change into a short window where the
critic is furthest behind. The midpoint of a linear schedule is also exactly
recoverable from config.json, which matters more than the shape does.

**The axis is steps, not updates**, because ``--resume`` keeps the step count
and replays update indices. Steps is the one axis a fresh run and a resumed
one agree on.

**What this module does not do, and cannot.** Nothing here shows that
annealing produces a better policy. It shows that a schedule exists, is
applied at the correct boundary, reaches the workers, and terminates exactly
on the sparse crown objective. The claim that it helps needs a paired A/B of
two full 1M-step runs at the measured 28.0 steps/s -- about 9.9 hours each,
20 hours sequential, and they cannot overlap because one run already occupies
eight workers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ..api.reward import ProjectionWeights, RewardWeights

__all__ = [
    "RewardSchedule", "KNOBS", "SHAPING_FIELDS",
    "knob_for_reward", "constant_schedule", "anneal_to_zero",
]


#: Which fields of each knob are *shaping* -- the ones an anneal drives to
#: zero. Everything else in a knob's weight tuple is carried unchanged, which
#: is how ``crowns`` and ``horizon_seconds`` stay put.
SHAPING_FIELDS: dict[str, tuple[str, ...]] = {
    "projection": ("tower", "elixir"),
    "five_term": ("tower_damage", "own_tower_hp", "elixir_trade",
                  "counterpush", "kite"),
    "shaping": ("shaping",),
}

KNOBS = tuple(SHAPING_FIELDS)

#: Which knob is the real shaping under each ``--reward``.
_KNOB_FOR_REWARD = {
    "projected": "projection",
    "five-term": "five_term",
    "simple": "shaping",
}


def knob_for_reward(reward: str) -> str:
    """The knob that is actually the shaping under ``--reward <reward>``."""
    try:
        return _KNOB_FOR_REWARD[reward]
    except KeyError:  # pragma: no cover - argparse constrains this
        raise ValueError(f"no shaping knob known for reward {reward!r}") from None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class RewardSchedule:
    """A weight tuple as a function of the training step.

    ``start`` and ``end`` are the full weight tuple written out literally at
    both endpoints, not a flag plus a delta, so no reader has to re-derive
    either one from a default that may since have changed.
    """

    knob: str = "projection"
    start: dict[str, Any] = field(default_factory=dict)
    end: dict[str, Any] = field(default_factory=dict)
    shape: str = "linear"
    axis: str = "steps"
    start_step: int = 0
    #: 0 means "not resolved yet"; :meth:`resolved` fills it from the run's
    #: total step count at 80%. See there for why not 100%.
    end_step: int = 0

    def __post_init__(self) -> None:
        if self.knob not in SHAPING_FIELDS:
            raise ValueError(
                f"unknown knob {self.knob!r}; expected one of {KNOBS}")
        if self.shape != "linear":
            raise ValueError(
                f"only a linear schedule is supported, not {self.shape!r}. A "
                "coefficient in a potential is annealed linearly so the "
                "critic sees a constant drift it can track, and so the "
                "midpoint is exactly recoverable from config.json.")
        if self.axis != "steps":
            raise ValueError(
                f"only a steps axis is supported, not {self.axis!r}. --resume "
                "keeps the step count and replays update indices, so steps is "
                "the one axis a fresh run and a resumed one agree on.")
        if set(self.start) != set(self.end):
            raise ValueError(
                "a schedule's endpoints must describe the same weight tuple; "
                f"start has {sorted(self.start)} and end has {sorted(self.end)}")
        # 0 is the unset sentinel, not a step. :meth:`resolved` fills it in
        # from the run's total, and rejecting it here made that path
        # unreachable for every nonzero --anneal-start: `--anneal
        # --anneal-start 500` died in the constructor before run.py could
        # call resolved(), with an unhandled ValueError at startup.
        if self.end_step and self.end_step < self.start_step:
            raise ValueError(
                f"end_step {self.end_step} is before start_step {self.start_step}")
        for key, low in self.start.items():
            high = self.end[key]
            if _is_number(low) and _is_number(high):
                continue
            if low != high:
                raise ValueError(
                    f"{key!r} is not a number and cannot be interpolated, but "
                    f"the endpoints differ: {low!r} -> {high!r}. "
                    "horizon_seconds=None means 'play to the end of the "
                    "match', which is roughly forty times the cost and is not "
                    "a point on any ramp.")

    # -- the schedule ------------------------------------------------------

    @property
    def is_constant(self) -> bool:
        """Whether this schedule ever moves.

        A constant schedule is the default and must stay bit-identical to a
        run with no schedule at all, so callers short-circuit on this rather
        than pushing the same weights over and over.
        """
        return all(self.start[k] == self.end[k] for k in self.start)

    def resolved(self, total_steps: int) -> "RewardSchedule":
        """Fill in an unset ``end_step`` at 80% of the run.

        Held at the end value for the last 20% -- roughly 200k steps, about
        two hours at the measured 28 steps/s -- deliberately. Otherwise the
        final checkpoint is measured under a weight that was still moving, and
        the sparse objective, which is the entire reason the schedule exists,
        never gets a stationary stretch to be measured on.
        """
        if self.end_step:
            return self
        return replace(self, end_step=max(self.start_step, int(0.8 * total_steps)))

    def at(self, step: int) -> dict[str, Any]:
        """The weight tuple in force at ``step``.

        Clamped at both ends, never extrapolated: past ``end_step`` a linear
        ramp continued would drive a coefficient negative, which is not a
        smaller version of the shaping but a reward pointing the other way.
        """
        if step <= self.start_step:
            return dict(self.start)
        if step >= self.end_step:
            return dict(self.end)
        span = self.end_step - self.start_step
        t = (step - self.start_step) / span
        out: dict[str, Any] = {}
        for key, low in self.start.items():
            high = self.end[key]
            out[key] = low + t * (high - low) if _is_number(low) else low
        return out

    # -- turning the tuple into the objects the env takes -------------------

    def weights_at(self, step: int) -> tuple[Any, float | None]:
        """``(reward_weights, shaping_weight)`` for ``step``.

        The pair :meth:`cr_sim.api.env.CRSimEnv.set_reward_weights` takes: an
        object whose *type* selects the reward -- ``None`` selects the simple
        one -- and the simple reward's own scalar knob.
        """
        values = self.at(step)
        if self.knob == "projection":
            return ProjectionWeights(**values), None
        if self.knob == "five_term":
            return RewardWeights(**values), None
        return None, float(values["shaping"])

    def as_dict(self) -> dict[str, Any]:
        """What config.json records.

        One nested key rather than nine flat ones: watch.py pairs two runs for
        A/B only while their config key *sets* differ by at most four, so a
        schedule spread across separate top-level fields would make every new
        run unpairable with every old one.
        """
        return {
            "knob": self.knob,
            "fields": list(SHAPING_FIELDS[self.knob]),
            "shape": self.shape,
            "axis": self.axis,
            "start_step": self.start_step,
            "end_step": self.end_step,
            # Each env adopts at its own next reset, so an update straddling a
            # schedule step carries two weights. Stated here so a reader knows
            # a metrics row's weight is a target, not a per-battle fact.
            "boundary": "episode_reset",
            "constant": self.is_constant,
            "start": dict(self.start),
            "end": dict(self.end),
        }


def constant_schedule(knob: str, values: dict[str, Any]) -> RewardSchedule:
    """A schedule that never moves -- the default, and today's behaviour."""
    return RewardSchedule(knob=knob, start=dict(values), end=dict(values))


def anneal_to_zero(knob: str, values: dict[str, Any], *,
                   start_step: int = 0, end_step: int = 0) -> RewardSchedule:
    """``values`` at the start, its shaping fields at zero by ``end_step``.

    Only the fields named in :data:`SHAPING_FIELDS` move. ``crowns`` and
    ``horizon_seconds`` are carried across unchanged -- the first because it
    is the objective rather than shaping, the second because ``None`` there
    means "play the match out" and is not a point on a ramp.
    """
    shaped = set(SHAPING_FIELDS[knob])
    end = {k: (0.0 if k in shaped else v) for k, v in values.items()}
    return RewardSchedule(knob=knob, start=dict(values), end=end,
                          start_step=start_step, end_step=end_step)
