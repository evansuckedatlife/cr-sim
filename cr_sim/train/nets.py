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

__all__ = [
    "ActorCritic", "NetConfig", "masked_categorical", "FactoredHead",
    "ConvPlacementHead", "net_config_for",
]


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
    #: ``"flat"`` -- one linear layer over all ``num_actions`` outputs -- or
    #: ``"factored"``: pick the card, then the tile, with the tile head
    #: conditioned on an embedding of the card.
    #:
    #: A flat *masked* categorical is strictly more expressive than an
    #: autoregressive factorisation of the same space; both can represent any
    #: distribution over the legal set, so this is not a correctness argument.
    #: It is a sample-efficiency one. Cloning the search bot produced 6,094
    #: play examples spread over 443 distinct (card, tile) pairs -- fourteen
    #: apiece -- and agreement on the expert's exact tile reached 5.4%. A flat
    #: head has one independent weight vector per (card, tile) and so learns
    #: nothing about "in front of my own tower" from a Knight that it can
    #: apply to a Musketeer. The factored head shares the tile weights across
    #: cards and passes the card in as an input, so every placement example
    #: trains the same map.
    #:
    #: ``"conv"`` is a third option and the one this board's geometry asks
    #: for. The observation grid is 32x18 at one cell per tile; the placement
    #: grid is 16x9 at one cell per two tiles; and the trunk's second
    #: convolution is stride 2, so the feature map it produces is *already*
    #: 16x9 -- the placement grid, computed and then thrown away by the layer
    #: after it. A 1x1 convolution over that map emits the placement logits
    #: directly, five output channels by 16 by 9, which is the action space.
    #:
    #: The reason to want it is not the parameter count, though that falls by
    #: a factor of twenty on the head. It is translation equivariance. A
    #: ``Linear(hidden, 720)`` learns every cell as an independent output, so
    #: "a Knight two tiles left of the bridge" is memorised separately at
    #: every location it was ever seen; a convolution learns the relative
    #: pattern once and applies it everywhere. That is exactly the sparsity
    #: the demonstrations have -- 443 distinct (card, tile) pairs at about
    #: fourteen examples apiece -- and it is the same argument that made the
    #: observation a grid rather than a flat vector, applied to the output.
    #: It also stays one flat masked categorical, so nothing downstream
    #: changes.
    head: str = "flat"
    #: Card-identity vocabulary, from the encoding config. Zero means the
    #: factored head falls back to conditioning on the hand *slot* index
    #: rather than on which card is in it.
    vocab_size: int = 0
    #: Where the acting team's hand one-hots live in the observation vector;
    #: see :func:`cr_sim.api.encoding.hand_onehot_layout`.
    hand_offset: int = 0
    hand_stride: int = 0
    #: Width of the card embedding the placement head is conditioned on.
    card_embedding: int = 32
    #: Hidden width of the shared placement head.
    place_hidden: int = 128
    #: Channels of non-spatial context broadcast across the placement map for
    #: the convolutional head. A pure convolution over the board cannot know
    #: which cards are in hand or how much elixir there is, and without this
    #: it cannot tell a Knight from a Fireball.
    place_context: int = 32
    #: Card slots including the pass slot, which is what ``num_actions``
    #: factors by. Mirrors ``cr_sim.api.encoding.NUM_CARD_SLOTS``; kept as a
    #: field rather than imported so this module stays free of the encoder.
    num_slots: int = 5

    @property
    def num_cells(self) -> int:
        cells, remainder = divmod(self.num_actions, self.num_slots)
        if remainder:
            raise ValueError(
                f"{self.num_actions} actions do not factor into "
                f"{self.num_slots} card slots")
        return cells


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


def net_config_for(env, **overrides) -> NetConfig:
    """The shapes a network needs, read off an environment.

    One place, because there are five: the trainer, the evaluator, the cloner,
    the worker processes and the browser server all build a network for the
    same environment, and a field added in four of them is a shape mismatch
    that only shows up when the fifth loads a checkpoint. The card-conditioned
    head in particular needs the encoder's own hand layout, which is not
    something a caller should be restating.
    """
    from ..api.encoding import NUM_CARD_SLOTS, hand_onehot_layout, observation_shapes

    shapes = observation_shapes(env.encoding)
    offset, stride, _count, width = hand_onehot_layout(env.encoding)
    nvec = [int(v) for v in env.action_space.nvec]
    grid = shapes["grid"]
    return NetConfig(
        grid_channels=int(grid[0]),
        grid_height=int(grid[1]),
        grid_width=int(grid[2]),
        vector_size=int(shapes["vector"][0]),
        num_actions=int(nvec[0] * nvec[1] * nvec[2]),
        num_slots=NUM_CARD_SLOTS,
        vocab_size=width,
        hand_offset=offset,
        hand_stride=stride,
        **overrides,
    )


class FactoredHead(nn.Module):
    """Card first, then tile, emitted as ordinary joint logits.

    The output is ``log P(slot) + log P(cell | slot)`` laid out in exactly the
    same flat order the environment's action index uses, so everything
    downstream -- the legality mask, sampling, greedy argmax, PPO's ratio,
    the cloner's cross-entropy -- is unchanged. This is a *reparameterisation*
    of the 720 logits, not a different action space, which is what makes the
    two heads comparable on the same demonstrations.

    Two things it buys over a single ``Linear(hidden, 720)``:

    *   **Shared placement weights.** ``place_out`` maps one hidden vector to
        all 144 tiles and is the same matrix whatever card is being placed, so
        "in front of my own King Tower" is learned once rather than five
        times. A flat head's tile weights for the Knight and for the Musketeer
        share nothing at all.
    *   **Card identity, not slot index.** The hand rotates; slot 2 is a
        different card every cycle. The conditioning vector is an embedding of
        the card *in* the slot, read out of the observation's own one-hot, so
        what the head learns about a Knight follows the Knight around the
        hand.

    The pass slot has no card, so it carries a learned vector of its own.
    """

    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.config = config
        self.slots = config.num_slots
        self.cells = config.num_cells
        self.play_slots = self.slots - 1

        width = config.card_embedding
        if config.vocab_size > 0:
            # A linear map of the slot's one-hot *is* an embedding lookup, and
            # this way the head takes the observation exactly as it comes
            # rather than needing an integer card id threaded through the
            # rollout.
            self.card_embedding = _orthogonal(
                nn.Linear(config.vocab_size, width, bias=False), 1.0)
        else:
            self.card_embedding = None
            self.slot_embedding = nn.Parameter(torch.zeros(self.play_slots, width))
            nn.init.normal_(self.slot_embedding, std=0.1)
        #: The pass slot is not a card and gets its own vector.
        self.pass_embedding = nn.Parameter(torch.zeros(1, width))
        nn.init.normal_(self.pass_embedding, std=0.1)

        self.slot_head = _orthogonal(nn.Linear(config.hidden, self.slots), 0.01)
        self.place_in = _orthogonal(
            nn.Linear(config.hidden + width, config.place_hidden), 2**0.5)
        self.place_out = _orthogonal(nn.Linear(config.place_hidden, self.cells), 0.01)

    def _context(self, vector: torch.Tensor) -> torch.Tensor:
        """One conditioning vector per card slot: ``(batch, slots, width)``."""
        batch = vector.shape[0]
        config = self.config
        if self.card_embedding is not None:
            start, stride, width = config.hand_offset, config.hand_stride, config.vocab_size
            # Sliced, not indexed by an id: the observation already carries a
            # one-hot per slot, and reading it directly means the head cannot
            # disagree with the encoder about which card is where.
            onehots = torch.stack(
                [vector[:, start + i * stride: start + i * stride + width]
                 for i in range(self.play_slots)],
                dim=1,
            )
            cards = self.card_embedding(onehots)
        else:
            cards = self.slot_embedding.unsqueeze(0).expand(batch, -1, -1)
        return torch.cat([cards, self.pass_embedding.unsqueeze(0).expand(batch, -1, -1)],
                         dim=1)

    def forward(
        self, features: torch.Tensor, vector: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch = features.shape[0]
        context = self._context(vector)
        shared = features.unsqueeze(1).expand(-1, self.slots, -1)
        cell_logits = self.place_out(
            F.relu(self.place_in(torch.cat([shared, context], dim=-1)))
        )                                            # (batch, slots, cells)
        slot_logits = self.slot_head(features)       # (batch, slots)

        grid = mask.reshape(batch, self.slots, self.cells)
        # A slot is choosable exactly when at least one of its cells is, which
        # is the only definition consistent with the flat mask -- deriving it
        # from elixir separately would let the two disagree and hand the
        # policy mass on a card it cannot play.
        slot_ok = grid.any(dim=-1)
        cell_log = torch.log_softmax(_apply_mask(cell_logits, grid), dim=-1)
        slot_log = torch.log_softmax(_apply_mask(slot_logits, slot_ok), dim=-1)
        joint = slot_log.unsqueeze(-1) + cell_log
        # Already normalised over the legal set; re-masking is what keeps an
        # illegal cell inside a legal slot at exactly zero probability rather
        # than at the finite floor two log-softmaxes leave behind.
        return _apply_mask(joint.reshape(batch, -1), mask)



class ConvPlacementHead(nn.Module):
    """Placement logits as a 1x1 convolution over the trunk's own feature map.

    The map the second convolution produces is the placement grid exactly --
    16x9 for this arena, one cell per two tiles -- so the logits for "play
    slot s at cell (x, y)" are a per-cell readout rather than a projection
    through a flattened vector. Five output channels, one per hand slot plus
    the pass slot, and the mask does the rest.

    **Index order is the trap.** ``cr_sim.api.encoding.decode_action`` reads a
    flat index as ``slot``, then ``x``, then ``y``: ``divmod(index, width *
    height)`` and then ``divmod(remainder, height)``. A convolution's natural
    layout is ``(slot, y, x)``, and flattening that directly transposes every
    placement into a legal-looking cell in the wrong part of the board without
    erroring anywhere. The permute below is what stops that, and
    ``tests/test_action_head.py`` round-trips a known cell through this head
    and ``decode_action`` rather than trusting it.
    """

    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.config = config
        self.slots = config.num_slots
        # The placement grid, in the convolution's own (y, x) order.
        self.height = (config.grid_height + 1) // 2
        self.width = (config.grid_width + 1) // 2
        if self.height * self.width * self.slots != config.num_actions:
            raise ValueError(
                f"the trunk's feature map is {self.height}x{self.width}, which "
                f"is {self.height * self.width} cells per slot against the "
                f"action space's {config.num_actions // self.slots}. The "
                "convolutional head only works while the observation grid is "
                "one cell per tile and the placement grid is one per two.")
        self.context = _orthogonal(
            nn.Linear(config.hidden, config.place_context), 2**0.5)
        self.place = _orthogonal(
            nn.Conv2d(config.channels + config.place_context, self.slots, 1), 0.01)

    def forward(
        self, features: torch.Tensor, spatial: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch = spatial.shape[0]
        context = F.relu(self.context(features))
        broadcast = context[:, :, None, None].expand(-1, -1, self.height, self.width)
        logits = self.place(torch.cat([spatial, broadcast], dim=1))
        # (batch, slot, y, x) -> (batch, slot, x, y), which is the order the
        # flat action index is built in.
        logits = logits.permute(0, 1, 3, 2).reshape(batch, -1)
        return _apply_mask(logits, mask)


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
        if config.head == "factored":
            self.policy_head = FactoredHead(config)
        elif config.head == "conv":
            self.policy_head = ConvPlacementHead(config)
        elif config.head == "flat":
            self.policy_head = _orthogonal(
                nn.Linear(config.hidden, config.num_actions), 0.01)
        else:
            raise ValueError(
                f"unknown policy head {config.head!r}; "
                "expected 'flat', 'factored' or 'conv'")
        self.value_head = _orthogonal(nn.Linear(config.hidden, 1), 1.0)

    #: Index into ``self.conv`` just past the first stride-2 convolution and
    #: its activation. Sliced rather than split into two modules so the
    #: parameter names stay ``conv.0``..``conv.4`` and every checkpoint
    #: written before the convolutional head existed still loads.
    _SPATIAL_DEPTH = 4

    def encode(self, grid: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        return self.encode_spatial(grid, vector)[0]

    def encode_spatial(
        self, grid: torch.Tensor, vector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(trunk features, placement-resolution feature map)``.

        Both come out of one pass. The map is what the trunk's second
        convolution produces and its third throws away by striding over it,
        and it is the placement grid exactly -- see :class:`ConvPlacementHead`.
        """
        spatial = self.conv[:self._SPATIAL_DEPTH](grid)
        flat = self.conv[self._SPATIAL_DEPTH:](spatial)
        return self.trunk(torch.cat([flat, self.vector(vector)], dim=-1)), spatial

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
        """Return masked logits and the value estimate.

        Both heads return logits over the same flat action space in the same
        order, so which one is in use changes nothing for any caller.
        """
        logits = self.policy_logits(grid, vector, mask)
        value = self.value_head(self.encode_value(grid, vector)).squeeze(-1)
        return logits, value

    def policy_logits(
        self, grid: torch.Tensor, vector: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Masked action logits, without running the critic.

        Worth having separately: with a separate critic encoder, ``forward``
        spends roughly half its time computing a value nothing is going to
        read. Every place that only chooses an action -- a frozen self-play
        opponent, the evaluator, the browser server -- was paying that.
        """
        features, spatial = self.encode_spatial(grid, vector)
        head = self.policy_head
        if isinstance(head, ConvPlacementHead):
            return head(features, spatial, mask)
        if isinstance(head, FactoredHead):
            return head(features, vector, mask)
        return _apply_mask(head(features), mask)

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
