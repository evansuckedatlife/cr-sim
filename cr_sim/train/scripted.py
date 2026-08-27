"""A bot that searches instead of learning.

Everything that has beaten a real game did one of three things first: imitate
a competent player (AlphaStar trained on 971,000 human replays before any
reinforcement learning, and that supervised agent alone outranked 84% of
humans), spend overwhelming compute (OpenAI Five ran PPO on 128,000 cores with
batches of one to three million timesteps), or search (AlphaZero uses tree
search as a policy improvement operator). This project had none of them: raw
PPO, from random initialisation, on one laptop, with a batch of 1,536. The flat
results were the method, not a bug.

Of the three, search is the one available here, because the engine is
deterministic and clones in under a millisecond. Almost no reinforcement
learning environment can afford to actually roll out its own future; this one
can. So the bot below does not encode any Clash Royale knowledge at all. It
asks the simulator.

For each candidate placement it branches the battle, plays the card, runs the
board forward with neither side acting again, and keeps whichever placement
leaves the position best. That is one-ply search with a rollout evaluation --
the crudest thing in the AlphaZero family -- and it needs no training, no
gradients and no reward design.

It exists to serve three purposes at once, which is why it is worth more than
its sophistication suggests:

*   **A demonstration source.** The small-compute literature bootstraps from
    rule-based play rather than human data, sometimes from a couple of hundred
    episodes. This generates that corpus.
*   **An opponent worth facing.** Self-play from random initialisation gave the
    critic nothing predictable to fit; a searching opponent is consistent.
*   **A yardstick.** The random control's per-episode spread is wide enough
    that a +0.375 reading over 40 battles measured -0.033 over 300. "Beats the
    scripted bot" is a claim that means something.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..api.encoding import NOOP_SLOT
from ..engine.entity import Team
from ..engine.lookahead import committed_value

__all__ = ["SearchBotConfig", "SearchBot", "scripted_opponent"]


@dataclass(frozen=True, slots=True)
class SearchBotConfig:
    """How hard the bot thinks."""

    #: Placements evaluated per decision. Every legal action is far too many --
    #: there are up to 720 and each costs a branch -- and most are equivalent
    #: anyway, since moving a Knight one tile sideways rarely changes the
    #: outcome. Sampling covers the interesting variety at a fraction of the
    #: cost.
    candidates: int = 14
    #: How far each branch is played forward, in seconds. Long enough for a
    #: push to reach a tower and for a defender to intercept it; longer mostly
    #: buys the opponent's reply, which this bot cannot predict anyway.
    #: Measured, not guessed. Against a random opponent over 16 matches:
    #: 4 seconds wins 31%, 8 seconds wins 94%, 15 seconds wins 100% and never
    #: loses. A card takes several seconds to walk anywhere, so a short
    #: horizon scores it before it has done anything.
    horizon_seconds: float = 15.0
    #: Playing nothing is always considered. Without it the bot spends elixir
    #: the moment it has any, which is the single most common way to lose at
    #: Clash Royale, and no amount of placement quality makes up for it.
    #:
    #: The bonus is added to the do-nothing branch, so a play must be better
    #: than waiting by this margin before it is made. Small: waiting is
    #: already evaluated fairly, this only breaks ties toward patience.
    #: Measured at 0.05 this drops the bot from 94% wins to 19%: it waits for
    #: a play that is clearly better rather than merely better, and in a game
    #: where the opponent is also committing, clearly-better rarely comes.
    patience: float = 0.01
    #: Below this, the bot always waits. Committing the last of the bar leaves
    #: nothing to answer with, and the projection cannot see that because it
    #: assumes neither side plays again.
    reserve_elixir: float = 1.0
    #: How the projected board is scored. These are NOT the reward's weights.
    #:
    #: committed_value was built as a potential for reward shaping, where only
    #: its change over time matters and a constant offset is irrelevant. Using
    #: it to *choose* an action is a different job, and its elixir term makes
    #: search actively bad at it: spending is immediate and certain, while the
    #: card's effect on the board takes longer than the horizon to appear, so
    #: every placement looks worse than waiting. Measured, a bot scoring this
    #: way lost 70% of its matches to a random agent that lost 45%.
    tower_weight: float = 1.0
    #: Off, and this is the single most consequential number here. At 0.3 the
    #: bot draws 100% of its matches -- it never plays at all -- and at 0.0 it
    #: wins 100%.
    elixir_weight: float = 0.0
    seed: int = 0


class SearchBot:
    """Picks the placement whose projected board is best.

    Stateless between decisions on purpose. It re-reads the board every time
    rather than carrying a plan, because a plan made two seconds ago was made
    against a board that no longer exists.
    """

    __slots__ = ("config", "team", "_rng", "_evaluated", "_encoding",
                 "last_scores")

    def __init__(self, team: Team, config: SearchBotConfig | None = None) -> None:
        self.team = team
        self.config = config or SearchBotConfig()
        self._rng = np.random.default_rng(self.config.seed)
        #: Built once on first use. It depends only on the arena and the card
        #: pool, neither of which changes during a battle, and rebuilding it
        #: per branch would cost more than the branch does.
        self._encoding = None
        #: Branches evaluated, for reporting what a decision actually cost.
        self._evaluated = 0
        #: Every action the last decision looked at, with what it scored:
        #: ``[(flat_index, value), ...]``.
        #:
        #: The single chosen action is a poor training target. Candidates are
        #: sampled, so the same board yields a different choice depending on
        #: which placements happened to be drawn -- the expert is not a
        #: function of the state, and no policy can learn one. The scores are:
        #: they say what the search actually believed about each move, which
        #: is what AlphaZero trains on rather than the move it played.
        self.last_scores: list[tuple[int, float]] = []

    @property
    def evaluated(self) -> int:
        return self._evaluated

    def _encoding_for(self, battle):
        if self._encoding is None:
            from ..api.encoding import build_encoding_config

            self._encoding = build_encoding_config(
                battle.arena, battle.config.blue_deck, battle.config.red_deck)
        return self._encoding

    def _score(self, battle, horizon: int) -> float:
        return committed_value(
            battle, self.team, horizon,
            tower_weight=self.config.tower_weight,
            elixir_weight=self.config.elixir_weight,
        )

    def _sample_actions(self, mask: np.ndarray) -> list[tuple[int, int, int]]:
        """Candidate placements, spread across the cards in hand.

        Sampled per card rather than uniformly over legal cells: a uniform
        draw over the mask is dominated by whichever card has the most legal
        tiles, so an expensive card with a small deploy zone would rarely be
        considered at all.
        """
        config = self.config
        legal = np.argwhere(mask)
        if not len(legal):
            return []
        slots = sorted({int(a[0]) for a in legal if int(a[0]) != NOOP_SLOT})
        if not slots:
            return []
        per_slot = max(1, config.candidates // len(slots))
        chosen: list[tuple[int, int, int]] = []
        for slot in slots:
            cells = legal[legal[:, 0] == slot]
            if not len(cells):
                continue
            take = min(per_slot, len(cells))
            picked = self._rng.choice(len(cells), size=take, replace=False)
            chosen.extend(tuple(int(v) for v in cells[i]) for i in picked)
        return chosen

    def __call__(self, observation: dict, mask: np.ndarray, battle=None):
        """Choose an action. ``battle`` is required -- this bot reads the board.

        Falls back to passing when it cannot see the battle, rather than
        playing randomly. A bot that silently degrades to random play would be
        indistinguishable from the control it is meant to beat.
        """
        if battle is None:
            return (NOOP_SLOT, 0, 0)

        player = battle.players[self.team]
        if player.elixir.exact < self.config.reserve_elixir:
            return (NOOP_SLOT, 0, 0)

        horizon = int(self.config.horizon_seconds * battle.config.ticks_per_second)
        slots, width, height = mask.shape
        # What the board is worth if nothing more is played. Every candidate is
        # measured against this, so the bot is asking "does this card improve
        # the position" rather than "which card looks best".
        waiting = self._score(battle, horizon)
        best_value = waiting + self.config.patience
        best_action = (NOOP_SLOT, 0, 0)
        scores: list[tuple[int, float]] = [
            (NOOP_SLOT * width * height, waiting)
        ]

        for action in self._sample_actions(mask):
            branch = battle.clone()
            if not _play(branch, self.team, action, self._encoding_for(battle)):
                continue
            self._evaluated += 1
            value = self._score(branch, horizon)
            index = action[0] * width * height + action[1] * height + action[2]
            scores.append((int(index), float(value)))
            if value > best_value:
                best_value, best_action = value, action
        self.last_scores = scores
        return best_action


def _play(branch, team: Team, action: tuple[int, int, int], encoding) -> bool:
    """Apply a placement to a branch, in the coordinates the env uses.

    A rejected placement is ordinary rather than exceptional: the mask is
    computed against the live board and a branch is that same board, but a
    card can still be unaffordable by the time it is tried.
    """
    from ..api.env import _apply_action

    return bool(_apply_action(branch, team, action, encoding))


def scripted_opponent(team: Team, config: SearchBotConfig | None = None):
    """A bot in the shape the environment's ``opponent_policy`` expects.

    The environment hands an opponent an observation and a mask, not the
    battle -- deliberately, so a policy cannot read state it would not have in
    the real game. This bot genuinely needs the board, so the environment
    passes it when the callable declares it wants one.
    """
    bot = SearchBot(team, config)

    def policy(observation, mask, battle=None):
        return bot(observation, mask, battle)

    policy.wants_battle = True  # type: ignore[attr-defined]
    policy.bot = bot  # type: ignore[attr-defined]
    return policy
