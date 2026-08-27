"""The policy and value network.

The observation is deliberately two things rather than one: a stack of spatial
channels over the 32x18 board, and a flat vector of everything that is not
spatial (elixir, the hand, tower hitpoints, the clock). They want different
treatment, and flattening the board into the vector would throw away the one
structure this game is entirely about -- that what matters is *where* things
are relative to each other.

So the board goes through a small convolutional trunk and the vector through a
short MLP, and the two are concatenated before the heads. The convolution is
what lets the network learn "a tank with support behind it" as a pattern rather
than as a particular pair of coordinates it has to memorise separately for
every tile.

**The policy head is masked.** Actions are ``(slot, x, y)`` flattened to a
single categorical over 720. The overwhelming majority are illegal at any given
moment -- unaffordable, no such card in hand, or the wrong half of the board --
and a policy that has to discover that by trial and error spends nearly all of
its samples learning the rules instead of the game. Illegal logits are set to
``-inf`` before the softmax, so they receive exactly zero probability and, more
importantly, zero gradient.

The masking has one failure mode worth naming: if every action were masked the
softmax would be all-``-inf`` and produce NaN. That cannot happen here because
the no-op slot is unconditionally legal, and :func:`masked_categorical` asserts
it rather than trusting it, because a NaN that reaches the optimiser destroys
the whole network silently and several thousand steps later.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ActorCritic", "NetConfig", "masked_categorical"]


@dataclass(frozen=True, slots=True)
class NetConfig:
    grid_channels: int
    grid_height: int
    grid_width: int
    vector_size: int
    num_actions: int
    #: Channel width of the convolutional trunk. Small on purpose: the board is
    #: 32x18 and the features that matter (a clump, a lane, a lone tank) are
    #: local and cheap. A wide trunk here mostly buys slower rollouts, and
    #: rollout speed is what a CPU-bound self-play run is short of.
    channels: int = 64
    hidden: int = 256
    #: Give the critic its own encoder rather than sharing the actor's.
    #:
    #: A shared trunk makes both losses compete for the same parameters, and
    #: the policy gradient wins: the features end up chosen for acting, and
    #: the critic predicts returns from a representation built for someone
    #: else's job. Measured here, explained variance sat under 0.1 while
    #: sharing -- close enough to zero that PPO's advantages were mostly
    #: noise, which is the whole reason the previous run committed to a
    #: strategy that never got stronger.
    #:
    #: Costs roughly twice the forward pass. Nearly free in wall-clock, since
    #: this environment is bound by simulating battles, not by the network.
    separate_critic: bool = True


def _orthogonal(module: nn.Module, gain: float) -> nn.Module:
    """Orthogonal init, the PPO default.

    Not decoration: with the near-zero gain used on the policy head below, the
    initial distribution over 720 actions is close to uniform, so early
    exploration is broad instead of committing hard to whatever the random
    initialisation happened to favour.
    """
    for name, param in module.named_parameters():
        if "weight" in name and param.dim() > 1:
            nn.init.orthogonal_(param, gain)
        elif "bias" in name:
            nn.init.constant_(param, 0.0)
    return module


class ActorCritic(nn.Module):
    """Shared trunk, separate policy and value heads."""

    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.config = config

        def _encoder():
            conv = nn.Sequential(
                _orthogonal(nn.Conv2d(config.grid_channels, config.channels, 3, padding=1), 2**0.5),
                nn.ReLU(),
                _orthogonal(nn.Conv2d(config.channels, config.channels, 3, stride=2, padding=1), 2**0.5),
                nn.ReLU(),
                _orthogonal(nn.Conv2d(config.channels, config.channels, 3, stride=2, padding=1), 2**0.5),
                nn.ReLU(),
                nn.Flatten(),
            )
            vector = nn.Sequential(
                _orthogonal(nn.Linear(config.vector_size, config.hidden), 2**0.5),
                nn.ReLU(),
            )
            trunk = nn.Sequential(
                _orthogonal(nn.Linear(conv_out + config.hidden, config.hidden), 2**0.5),
                nn.ReLU(),
            )
            return conv, vector, trunk

        conv_out = config.channels * ((config.grid_height + 3) // 4) * ((config.grid_width + 3) // 4)
        self.conv, self.vector, self.trunk = _encoder()
        if config.separate_critic:
            self.critic_conv, self.critic_vector, self.critic_trunk = _encoder()
        # Near-zero gain on the policy head keeps the opening distribution flat
        # across all 720 actions; gain 1 on the value head because its output is
        # a return estimate, not a logit.
        self.policy_head = _orthogonal(nn.Linear(config.hidden, config.num_actions), 0.01)
        self.value_head = _orthogonal(nn.Linear(config.hidden, 1), 1.0)

    def encode(self, grid: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        return self.trunk(torch.cat([self.conv(grid), self.vector(vector)], dim=-1))

    def encode_value(self, grid: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """Features for the critic, which are its own when it has its own."""
        if not self.config.separate_critic:
            return self.encode(grid, vector)
        return self.critic_trunk(
            torch.cat([self.critic_conv(grid), self.critic_vector(vector)], dim=-1)
        )

    def critic_parameters(self):
        """Just the value side, so it can be given its own learning rate."""
        if not self.config.separate_critic:
            return list(self.value_head.parameters())
        return (
            list(self.critic_conv.parameters())
            + list(self.critic_vector.parameters())
            + list(self.critic_trunk.parameters())
            + list(self.value_head.parameters())
        )

    def forward(
        self, grid: torch.Tensor, vector: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked logits and the value estimate."""
        logits = self.policy_head(self.encode(grid, vector))
        value = self.value_head(self.encode_value(grid, vector)).squeeze(-1)
        return _apply_mask(logits, mask), value

    def policy_logits(
        self, grid: torch.Tensor, vector: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Masked logits alone, for callers with no use for the value.

        ``forward`` runs both encoders. With ``separate_critic`` -- the
        default -- that is two full convolutional trunks, and the second one
        exists only to produce a number that an opponent, an evaluation or the
        play server immediately discards. Measured at batch 1 on one thread,
        the whole forward is 2393us and this is 1334us, so the discarded half
        was 44% of every inference the rollout workers do.

        The returned tensor is the same one ``forward`` returns first, element
        for element: this computes a strict subset of the same graph.
        """
        return _apply_mask(self.policy_head(self.encode(grid, vector)), mask)

    @torch.no_grad()
    def act(
        self, grid: torch.Tensor, vector: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions for a rollout. Returns ``(action, log_prob, value)``."""
        logits, value = self(grid, vector, mask)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        return action, distribution.log_prob(action), value

    def evaluate(
        self,
        grid: torch.Tensor,
        vector: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score stored actions during an update. ``(log_prob, entropy, value)``."""
        logits, value = self(grid, vector, mask)
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution.log_prob(action), distribution.entropy(), value


#: What an illegal logit is set to. Chosen rather than ``-inf`` because
#: ``-inf`` produces NaN gradients wherever a whole row is masked, and a NaN
#: reaching the optimiser corrupts every weight in the network silently. This
#: is large enough that a masked action's probability underflows to zero in
#: float32 anyway.
MASKED_LOGIT = -1e8


def _apply_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return logits
    return logits.masked_fill(~mask, MASKED_LOGIT)


def masked_categorical(
    logits: torch.Tensor, mask: torch.Tensor
) -> torch.distributions.Categorical:
    """A categorical over legal actions only.

    Asserts that every row has at least one legal action. The no-op slot makes
    that true by construction, but an all-masked row produces a uniform
    distribution over impossible actions rather than an error, and the failure
    would show up thousands of steps later as a policy that has learned
    nonsense.
    """
    assert mask.any(dim=-1).all(), "an observation had no legal action at all"
    return torch.distributions.Categorical(logits=_apply_mask(logits, mask))
