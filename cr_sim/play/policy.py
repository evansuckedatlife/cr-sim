"""Let a trained checkpoint be the opponent.

The bridge between a saved policy and a live match. The training environment
drives the engine itself, so a policy there never has to be asked "what would
you do with this battle" out of context -- here it does, which is the whole
difference between an environment and an opponent.

Sampled rather than argmax, and that is not a detail. A policy this early is
close to uniform over its legal actions, so its argmax is whichever tiny bias
the initialisation left behind: measured, the same checkpoint wins 8% of
matches greedily against 34% sampled, worse than random either way but
*visibly* broken greedily -- it plays the same card on the same tile every
cycle. A human playing against that learns nothing about the simulator, which
is what the page is for.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..api.encoding import (
    build_encoding_config,
    decode_action,
    encode_observation,
    legal_action_mask,
)
from ..engine.battle import Battle
from ..engine.entity import Team
from ..engine.fixed import to_tiles

if TYPE_CHECKING:  # pragma: no cover - import cost only paid when typing
    from .server import PlayServer

__all__ = ["policy_controller", "PolicyOpponent"]


class PolicyOpponent:
    """A trained network playing one side of a live match."""

    def __init__(self, checkpoint: Path, server: "PlayServer", seed: int = 0) -> None:
        import torch

        from ..train.nets import ActorCritic, NetConfig

        self.torch = torch
        self.server = server
        self.rng = np.random.default_rng(seed)
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            # Raised here, at construction, because that is where the caller's
            # fallback to random play can catch it. Deferring the file open to
            # the first move -- which is what this used to do, since the
            # network's shapes need a live match -- meant a bad path surfaced
            # inside a request handler instead, crashing every poll rather than
            # quietly playing a weaker opponent.
            raise FileNotFoundError(f"no checkpoint at {self.checkpoint}")
        # Loaded eagerly for the same reason. The weights are small; only the
        # network they go into has to wait for a match.
        self._payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self._config: Any = None
        self._net: ActorCritic | None = None
        self._net_cls = ActorCritic
        self._net_config_cls = NetConfig

    # -- setup -------------------------------------------------------------

    def _ensure(self, battle: Battle, team: Team) -> None:
        """Build the encoder and network on the first call.

        Deferred because the encoding depends on the decks in play, and those
        are only known once a match exists. Building it at construction would
        bake in whatever deck happened to be configured when the server
        started and quietly mis-encode every match after a deck change.
        """
        if self._net is not None:
            return
        blue = battle.players[Team.BLUE].deck
        red = battle.players[Team.RED].deck
        # Which observation the weights were trained on belongs to the
        # checkpoint. Building a v1 encoding for a v2 policy is a shape error
        # at best and, where the channel counts happen to agree, a policy
        # reading channels that mean something else.
        from ..api.encoding import parse_observation

        features = parse_observation(
            str(self._payload.get("observation", "v1"))
            if isinstance(self._payload, dict) else "v1")
        self._config = build_encoding_config(battle.arena, blue, red, features)

        observation = encode_observation(battle, team, self.server.registry, self._config)
        nvec = (5, self._config.action_width, self._config.action_height)
        from ..api.encoding import NUM_CARD_SLOTS, hand_onehot_layout

        offset, stride, _count, width = hand_onehot_layout(self._config)
        head = (self._payload.get("head", "flat")
                if isinstance(self._payload, dict) else "flat")
        # The stat-conditioned head reads a table of card statistics, one row
        # per vocabulary entry, and the config it is built from has to carry
        # it -- ``net_config_for`` does that for the trainer, the evaluator,
        # the cloner and the workers, and this is the one load path that
        # restates the config by hand instead. Without it the head raises on
        # the *first move* rather than at load, and ``PlaySession._think``
        # catches that by setting ``self.controller = None``: the opponent
        # stops playing for the rest of the match instead of falling back to
        # random, which reads as a policy that decided to pass 60 times.
        #
        # Keyed on ``self._config.vocab``, which is what the observation's
        # one-hot bits are indexed by; anything else conditions the head on
        # the wrong card.
        card_stats: tuple[tuple[float, ...], ...] = ()
        if head == "factored-stats":
            from ..data.card_features import card_feature_table

            card_stats = card_feature_table(
                self.server.data, self.server.levels, self.server.registry,
                self._config.vocab)
        net = self._net_cls(
            self._net_config_cls(
                grid_channels=observation["grid"].shape[0],
                grid_height=observation["grid"].shape[1],
                grid_width=observation["grid"].shape[2],
                vector_size=observation["vector"].shape[0],
                num_actions=int(np.prod(nvec)),
                num_slots=NUM_CARD_SLOTS,
                vocab_size=width,
                hand_offset=offset,
                hand_stride=stride,
                # Which head the weights were trained with belongs to the
                # checkpoint; a factored head's parameters do not fit a flat
                # one and load_state_dict would fail on a tensor name the
                # player has no way to interpret.
                head=head,
                card_stats=card_stats,
            )
        )
        state = self._payload.get("state_dict", self._payload)
        # Loaded strictly. A checkpoint whose shapes disagree with the current
        # encoding was trained on a different observation, and letting it load
        # partially would produce an opponent that looks trained and is not.
        net.load_state_dict(state)
        net.eval()
        self._net = net
        self._nvec = nvec

    # -- play --------------------------------------------------------------

    def __call__(self, battle: Battle, team: Team) -> tuple[str, int, int] | None:
        self._ensure(battle, team)
        assert self._net is not None

        observation = encode_observation(battle, team, self.server.registry, self._config)
        mask = legal_action_mask(battle, team, self.server.registry, self._config)
        flat = mask.reshape(-1)
        if not flat.any():
            return None

        torch = self.torch
        with torch.no_grad():
            logits = self._net.policy_logits(
                torch.from_numpy(observation["grid"]).unsqueeze(0),
                torch.from_numpy(observation["vector"]).unsqueeze(0),
                torch.from_numpy(flat).unsqueeze(0),
            )
            index = int(torch.distributions.Categorical(logits=logits).sample())

        slots, width, height = self._nvec
        slot, remainder = divmod(index, width * height)
        grid_x, grid_y = divmod(remainder, height)
        decoded = decode_action((slot, grid_x, grid_y), team, battle.arena, self._config)
        if decoded is None:
            return None  # the no-op slot: it chose to hold elixir

        card_slot, x, y = decoded
        hand = battle.players[team].hand
        if card_slot >= len(hand):
            return None
        return hand[card_slot], x, y


def policy_controller(checkpoint: Path, server: "PlayServer", seed: int = 0):
    """An :data:`~cr_sim.play.session.AiController` backed by a checkpoint."""
    return PolicyOpponent(checkpoint, server, seed)
