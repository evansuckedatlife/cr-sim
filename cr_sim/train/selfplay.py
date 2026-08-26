"""Self-play, and an honest progress signal to watch it with.

Two things a long run needs that the trainer did not have.

**An opponent worth beating.** Training against random play teaches a policy to
beat random play, and that is a low ceiling: the measured gap between a random
agent and a trained one has been inside noise on every run so far, partly
because there was nothing to learn *from*. The opponent here is a frozen copy
of the policy itself, refreshed every so often, so the thing it is trying to
beat improves as it does.

Frozen rather than live. Both sides sharing one set of weights would make the
opponent change underneath the rollout that is scoring it, and the advantage
estimates would be measuring a moving target. A snapshot is a fixed opponent
for as long as it is in place, which is what the algorithm assumes.

**A number that means what it says.** The trainer's own return is measured
while the policy is still exploring, averaged over a sliding window, on
whatever seeds the rollout happened to draw. Measured against a paired-seed
control it has run about eighteen points optimistic -- reporting 55% for a
policy that evaluated at 37% -- and the cause is still unexplained. So a long
run should not be steered by it. :func:`evaluation_probe` plays the current
policy against a random control on fixed seeds, which is the same measurement
the final evaluation makes, and is the one worth watching.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Sequence

import numpy as np

from ..api.env import CRSimEnv
from .nets import ActorCritic

__all__ = ["FrozenOpponent", "evaluation_probe"]


class FrozenOpponent:
    """A snapshot of the policy, playing the other side.

    Holds its own copy of the weights rather than a reference to the live
    network, so refreshing is an explicit act and the opponent cannot drift
    mid-rollout.
    """

    __slots__ = ("_net", "_nvec", "_torch", "_rng", "refreshes")

    def __init__(self, net: ActorCritic, nvec: Sequence[int], seed: int = 0) -> None:
        import torch

        self._torch = torch
        self._nvec = [int(v) for v in nvec]
        self._rng = np.random.default_rng(seed)
        self.refreshes = 0
        self._net = self._snapshot(net)

    def _snapshot(self, net: ActorCritic) -> ActorCritic:
        clone = copy.deepcopy(net)
        clone.eval()
        for parameter in clone.parameters():
            parameter.requires_grad_(False)
        return clone

    def refresh(self, net: ActorCritic) -> None:
        """Adopt the current policy as the new opponent."""
        self._net = self._snapshot(net)
        self.refreshes += 1

    def __call__(self, observation: dict, mask: np.ndarray) -> tuple[int, int, int]:
        flat = mask.reshape(-1)
        if not flat.any():
            return (0, 0, 0)
        torch = self._torch
        with torch.no_grad():
            logits, _ = self._net(
                torch.from_numpy(observation["grid"]).unsqueeze(0),
                torch.from_numpy(observation["vector"]).unsqueeze(0),
                torch.from_numpy(flat).unsqueeze(0),
            )
            # Sampled, not greedy. A greedy opponent plays one fixed line and
            # the policy learns to beat that line rather than the game -- and
            # a near-uniform policy's argmax is an arbitrary fixed placement,
            # which is worse than random to practise against.
            index = int(torch.distributions.Categorical(logits=logits).sample())
        slots, width, height = self._nvec
        slot, remainder = divmod(index, width * height)
        gx, gy = divmod(remainder, height)
        return (min(slot, slots - 1), gx, gy)


def evaluation_probe(
    make_env: Callable[[], CRSimEnv],
    episodes: int = 40,
    seed: int = 12345,
) -> Callable[[ActorCritic], dict[str, Any]]:
    """Build a probe that scores a policy against a random control.

    Paired seeds: both arms play the same battles rather than the same
    *number* of battles. The per-episode spread here is several times larger
    than any effect worth seeing, so unpaired sampling would need far more
    episodes to say anything.
    """
    from .evaluate import evaluate

    seeds = [int(s) for s in np.random.default_rng(seed).integers(0, 2**31 - 1, episodes)]
    control_env = make_env()
    control = evaluate(control_env, None, episodes=episodes, seeds=seeds)
    control_wins = float(np.mean([c > 0 for c in control["crowns"]]))
    control_return = float(np.mean(control["returns"]))

    def probe(net: ActorCritic) -> dict[str, Any]:
        result = evaluate(make_env(), net, episodes=episodes, seeds=seeds, greedy=False)
        wins = float(np.mean([c > 0 for c in result["crowns"]]))
        spread = float(np.std(control["returns"])) or 1.0
        return {
            "eval_return": float(np.mean(result["returns"])),
            "eval_win": wins,
            "control_return": control_return,
            "control_win": control_wins,
            # In control standard deviations, because the raw gap means
            # nothing without knowing how noisy the control is.
            "eval_lift_sd": (float(np.mean(result["returns"])) - control_return) / spread,
        }

    return probe
