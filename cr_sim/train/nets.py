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
    "FactoredStatsHead", "ConvPlacementHead", "net_config_for", "POLICY_HEADS",
]

#: Every head :class:`ActorCritic` can build, and the ``--head`` choices of
#: both entry points that write a checkpoint's ``"head"`` field.
#:
#: One tuple because the two argparse lists were written out by hand and went
#: stale: ``"factored-stats"`` shipped complete -- config, head, worker
#: round-trip, all of it -- and was unreachable from
#: ``python -m cr_sim.train.run`` and ``scripts/clone_policy.py`` alike, since
#: neither would accept the name. A head no entry point can select is a head
#: no checkpoint can ever be written with.
POLICY_HEADS: tuple[str, ...] = ("flat", "factored", "factored-stats", "conv")


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
    #:
    #: ``"factored-stats"`` is the fourth, and it is the factored head with
    #: its card *lookup* replaced by an encoder over the card's own stats --
    #: see :class:`FactoredStatsHead`. A fourth name rather than a change to
    #: ``"factored"`` because every ``load_state_dict`` in this tree is
    #: strict, seven factored checkpoints exist on disk, and one of them is
    #: the resume target of a live run: renaming a parameter under
    #: ``policy_head`` kills that run at its next restart. A migration shim is
    #: not available anyway -- the old weights are a free 32x8 table with no
    #: encoder to project onto.
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
    #: One row of card statistics per vocabulary entry, **in ``vocab`` order**,
    #: for the ``"factored-stats"`` head. Built once by
    #: :func:`net_config_for` from :func:`cr_sim.data.card_features
    #: .card_feature_table`; empty for every other head.
    #:
    #: A tuple of tuples of plain floats and nothing else. This dataclass is
    #: ``frozen`` and ``slots``, so the field has to stay hashable, and
    #: ``asdict()`` pickles the whole config into ``VecEnvConfig`` for each of
    #: the spawned worker processes to rebuild. A numpy array would break the
    #: auto ``__hash__`` and ``__eq__``; a registry or a ``LogicData`` would
    #: not survive the pickle.
    card_stats: tuple[tuple[float, ...], ...] = ()
    #: Hidden width of the card-stat encoder. A field rather than a literal so
    #: it can be swept without a source edit.
    card_encoder_hidden: int = 32

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

    One place, because there are four: the trainer, the evaluator, the cloner
    and the worker processes all build a network for the same environment, and
    a field added in three of them is a shape mismatch that only shows up when
    the fourth loads a checkpoint. The card-conditioned head in particular
    needs the encoder's own hand layout, which is not something a caller
    should be restating.

    The browser server is the exception, and not a happy one:
    :meth:`cr_sim.play.policy.PolicyOpponent._ensure` has a battle rather than
    an environment and restates every field by hand. It restated eight of them
    and not ``card_stats``, which built a ``"factored-stats"`` head with no
    table -- a ``ValueError`` raised on the first move, swallowed by
    ``PlaySession._think``, and an opponent that played nothing for the rest
    of the match. Anything added here has to be added there too, until that
    path can be given a real environment.
    """
    from ..api.encoding import NUM_CARD_SLOTS, hand_onehot_layout, observation_shapes

    shapes = observation_shapes(env.encoding)
    offset, stride, _count, width = hand_onehot_layout(env.encoding)
    nvec = [int(v) for v in env.action_space.nvec]
    grid = shapes["grid"]
    # The stat table is the whole expensive part of the card-stat head --
    # resolving every card, following its projectile chain, scaling to level --
    # and it is a function of static data, so it is computed exactly once here
    # and carried on the config. Nothing on the per-decision path touches the
    # data layer. Built only for the head that reads it: every other head
    # would pay the resolve cost for a field it never looks at.
    #
    # Keyed on the *encoding's* vocab, in the encoding's order, because that
    # is the order the observation's one-hot bits are set in.
    if overrides.get("head") == "factored-stats" and "card_stats" not in overrides:
        from ..data.card_features import card_feature_table

        overrides["card_stats"] = card_feature_table(
            env.data, env.levels, env.registry, env.encoding.vocab)
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


class FactoredStatsHead(FactoredHead):
    """The same head, conditioned on what a card *does* rather than which it is.

    :class:`FactoredHead` has one free column of ``card_embedding.weight`` per
    vocabulary entry. The vocabulary is this episode's deck union, so column
    ``i`` means whatever ``vocab[i]`` happens to be -- and playing a different
    deck of the same size silently hands the Knight's learned column to
    whatever card sorts into position 4. No error, no shape mismatch, just a
    head conditioned on the wrong card.

    Here the columns are *computed* instead: each is a small MLP applied to
    that card's own statistics -- hitpoints, damage per second, reach, speed,
    what it targets, what it leaves behind -- so the head learns "slow, tanky,
    ground-only goes at the bridge" rather than "card 4 goes at the bridge",
    and a card that was never in a training deck gets a column for free. See
    :mod:`cr_sim.data.card_features` for the features and why each is
    there.

    **The mix order is the thing that must not be got wrong.**
    ``onehots @ encoder(table)``: encode the table, *then* mix. Never
    ``encoder(onehots @ table)``. An empty hand slot is an all-zero one-hot,
    and a ``bias=False`` matmul maps that to an exactly-zero conditioning
    vector, which is the semantics the base head has and every downstream
    thing assumes. ``encoder(0)`` instead returns the encoder's bias and the
    LayerNorm's shift, so an empty slot would quietly start meaning something.

    **The trap is invisible at initialisation**, which is what makes it worth
    this much prose. :func:`_orthogonal` zeroes every bias, ``nn.LayerNorm``
    starts at beta 0, and LayerNorm of an all-zero row is all-zero -- so on a
    freshly built head ``card_encoder(zeros(47))`` measures exactly 0.0 and
    the wrong mix order is indistinguishable from the right one. One Adam step
    ends that: measured on the test deck at seed 0, ``|encoder(0)|`` goes from
    0.0 to 5.61 against a real card's 5.68. So from step 1 of any real run an
    empty hand slot would condition on a vector the size of a card's, and a
    check made at initialisation cannot see it.
    ``tests/test_action_head_stats.py`` moves the encoder's biases off zero
    before asserting, for exactly that reason.

    **No runtime cache, deliberately.** The table is 8 rows through a 47-32-32
    MLP and is recomputed every forward. Measured on this machine while an
    8-worker training run had the cores: 272us at batch 1 against 4992us for
    the whole ``policy_logits``, which is 5.5% of it, and 544us at batch 256
    against 89.3ms, which is 0.6%. Both ratios move with whatever else the box
    is doing -- the same measurement with a test suite running as well read
    8.3% and 0.7% -- and stay a rounding error either way, against a decision
    that also has to encode an observation and build a legality mask. A cache
    would buy nothing and cost the failure this codebase specialises in: a
    table computed once and never invalidated leaves the head frozen from step
    1 while a "the parameters moved" test stays green, because the encoder's
    weights still move. ``tests/test_action_head_stats.py`` counts
    the encoder's forward passes per decision, so a cache added later has to
    turn that test red before it can be believed.

    Shapes: ``(47) -> Linear -> (32) -> Tanh -> Linear -> (32) -> LayerNorm``.
    The output width **must** stay ``config.card_embedding``: ``place_in`` is
    ``Linear(hidden + width, place_hidden)`` and the width is baked into it.
    Holding it there keeps ``place_in``, ``place_out``, ``slot_head`` and
    ``pass_embedding`` byte-identical to the factored head's, which leaves
    warm-starting from a factored checkpoint's tile weights possible later.

    Depth one and tanh rather than ReLU: the inputs are clipped to ``[-1, 1]``
    and tanh keeps the hidden activations bounded too, so an unseen
    combination of stats lands inside the region the second layer was trained
    over instead of arbitrarily far outside it. The ``LayerNorm`` makes the
    conditioning's scale independent of how extreme an unseen card is; it is
    applied per row, to the table, so it couples no card to any other.
    """

    def __init__(self, config: NetConfig) -> None:
        if config.vocab_size <= 0 or not config.card_stats:
            raise ValueError(
                "the 'factored-stats' head needs a card stat table; "
                "net_config_for builds it from the environment's vocabulary. "
                "A NetConfig built by hand has to pass card_stats itself.")
        if len(config.card_stats) != config.vocab_size:
            raise ValueError(
                f"card_stats has {len(config.card_stats)} rows against a "
                f"vocabulary of {config.vocab_size}. Row i is what slot i's "
                "one-hot bit selects; a table of a different length is a head "
                "conditioned on the wrong cards.")
        super().__init__(config)
        # The identity lookup the base class just built is exactly the thing
        # this head exists to remove. Assigning None drops it from
        # named_parameters() and from state_dict() -- and leaves the attribute
        # in place, so the inherited _context's `is not None` branch is the one
        # this class overrides rather than a fallback it might fall into.
        self.card_embedding = None

        width, hidden = config.card_embedding, config.card_encoder_hidden
        features = len(config.card_stats[0])
        self.card_encoder = nn.Sequential(
            _orthogonal(nn.Linear(features, hidden), 5 / 3),   # tanh's gain
            nn.Tanh(),
            _orthogonal(nn.Linear(hidden, width), 1.0),        # the lookup's
            nn.LayerNorm(width),
        )
        #: The stat table, non-persistent on purpose. A persistent buffer or a
        #: parameter would put it in ``state_dict()`` and turn strict loading
        #: of this head's own checkpoints into a versioning problem the day a
        #: feature is added or a normalisation constant changes.
        self.register_buffer(
            "card_stats",
            torch.tensor(config.card_stats, dtype=torch.float32),
            persistent=False,
        )

    def _context(self, vector: torch.Tensor) -> torch.Tensor:
        config = self.config
        start, stride, width = config.hand_offset, config.hand_stride, config.vocab_size
        onehots = torch.stack(
            [vector[:, start + i * stride: start + i * stride + width]
             for i in range(self.play_slots)],
            dim=1,
        )                                            # (batch, play_slots, vocab)
        table = self.card_encoder(self.card_stats)   # (vocab, width)
        cards = onehots @ table                      # (batch, play_slots, width)
        # The pass slot stays a learned parameter and is not run through the
        # encoder. It has no card, so there is nothing to encode -- and a zero
        # stat row put through the encoder would come out identical to an empty
        # hand slot's conditioning, which is wrong in the one way that matters:
        # pass is unconditionally legal and an empty slot is never legal.
        return torch.cat(
            [cards, self.pass_embedding.unsqueeze(0).expand(vector.shape[0], -1, -1)],
            dim=1)


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
        elif config.head == "factored-stats":
            self.policy_head = FactoredStatsHead(config)
        elif config.head == "conv":
            self.policy_head = ConvPlacementHead(config)
        elif config.head == "flat":
            self.policy_head = _orthogonal(
                nn.Linear(config.hidden, config.num_actions), 0.01)
        else:
            raise ValueError(
                f"unknown policy head {config.head!r}; expected one of "
                + ", ".join(repr(name) for name in POLICY_HEADS))
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

        Measured at batch 1 on one thread: 2393us for the whole forward
        against 1334us for this, so 44% of every inference those three do was
        a value nobody read. Waste rather than a bottleneck, though -- an
        interleaved A/B over whole self-play battles came back at ~1.0x,
        because a decision is ~118ms and this forward is ~1.1ms of it.
        """
        features, spatial = self.encode_spatial(grid, vector)
        head = self.policy_head
        if isinstance(head, ConvPlacementHead):
            return head(features, spatial, mask)
        if isinstance(head, FactoredHead):
            return head(features, vector, mask)
        return _apply_mask(head(features), mask)

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
