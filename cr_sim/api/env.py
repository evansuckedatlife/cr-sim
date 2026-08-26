"""Gymnasium-style environments over one :class:`~cr_sim.engine.battle.Battle`.

Two environments live here because a Clash Royale match has two agents in it
and there are two legitimate ways to expose that to a training loop; see
:class:`CRSimEnv` and :class:`CRSimSelfPlayEnv` for which problem each one
solves.

``gymnasium`` is declared as an optional dependency in ``pyproject.toml``
(the ``rl`` extra), not a hard one -- this package has to import, and its own
tests have to pass, whether or not it happens to be installed in the
environment running them. Both classes below are written against the real
Gymnasium API (``reset(seed=...)`` -> ``(obs, info)``, ``step(action)`` ->
the 5-tuple ``(obs, reward, terminated, truncated, info)``,
``.observation_space``/``.action_space``) so that installing gymnasium and
having this module pick up the real ``gymnasium.Env`` base class and the
real ``gymnasium.spaces`` changes nothing about how these classes behave --
only what they inherit from and how their spaces are type-checked elsewhere.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ..data.cards import CardRegistry
from ..data.leveling import LevelTable
from ..data.source import LogicData
from ..engine.arena import Arena, load_arena
from ..engine.battle import Battle, BattleConfig
from ..engine.entity import Team
from ..render.web import render_ascii
from .encoding import (
    NUM_CARD_SLOTS,
    EncodingConfig,
    build_encoding_config,
    decode_action,
    encode_observation,
    legal_action_mask,
    observation_shapes,
    total_tower_hitpoints,
)

__all__ = ["CRSimEnv", "CRSimSelfPlayEnv", "idle_opponent_policy", "HAS_GYMNASIUM"]

try:  # pragma: no cover - exercised by whichever environment actually has it
    import gymnasium as _gymnasium
    from gymnasium import spaces as _spaces

    HAS_GYMNASIUM = True
except ImportError:  # pragma: no cover
    _gymnasium = None
    _spaces = None
    HAS_GYMNASIUM = False


if HAS_GYMNASIUM:
    Box = _spaces.Box
    MultiDiscrete = _spaces.MultiDiscrete
    DictSpace = _spaces.Dict
    _EnvBase = _gymnasium.Env
else:  # pragma: no cover - only exercised in an environment without gymnasium

    class Box:
        """Stand-in for ``gymnasium.spaces.Box``, used only when gymnasium is
        not installed. Carries just the pieces this package's own tests and
        callers need -- shape, dtype, bounds, a membership check -- not the
        full Gymnasium space machinery (sampling, seeding, flattening)."""

        def __init__(self, low: float, high: float, shape: Sequence[int], dtype=np.float32) -> None:
            self.low = low
            self.high = high
            self.shape = tuple(shape)
            self.dtype = dtype

        def contains(self, x: Any) -> bool:
            arr = np.asarray(x)
            return (
                arr.shape == self.shape
                and bool(np.all(arr >= self.low))
                and bool(np.all(arr <= self.high))
            )

    class MultiDiscrete:
        """Stand-in for ``gymnasium.spaces.MultiDiscrete``."""

        def __init__(self, nvec: Sequence[int]) -> None:
            self.nvec = np.asarray(nvec, dtype=np.int64)
            self.shape = self.nvec.shape

        def contains(self, x: Any) -> bool:
            arr = np.asarray(x)
            return arr.shape == self.shape and bool(np.all(arr >= 0)) and bool(np.all(arr < self.nvec))

    class DictSpace:
        """Stand-in for ``gymnasium.spaces.Dict``."""

        def __init__(self, spaces: dict) -> None:
            self.spaces = spaces

        def contains(self, x: Any) -> bool:
            return all(key in x and space.contains(x[key]) for key, space in self.spaces.items())

    class _EnvBase:
        """Stand-in for ``gymnasium.Env``. See the module docstring: this
        exists purely so ``class CRSimEnv(_EnvBase)`` resolves to something
        when gymnasium is absent, not to reproduce its behaviour."""

        metadata: dict = {}


#: Matches BattleConfig's own default, so a caller who never touches either
#: knob gets full-rate simulation. See DEFAULT_FRAME_SKIP for the paired knob.
DEFAULT_TICKS_PER_SECOND = 60
#: 6 ticks at 60 TPS is 100ms between decisions: fast enough that a policy can
#: still react within a sub-second attack windup, slow enough that most of
#: the 60 tick-offsets inside that window are not distinct decisions worth
#: exploring separately. cr_sim.engine.constants.TRAINING_TPS (20) is offered
#: for cheap bulk training via ``ticks_per_second``; halve this alongside it
#: to hold the same 100ms real-time decision cadence rather than silently
#: slowing decisions down by 3x.
DEFAULT_FRAME_SKIP = 6


def idle_opponent_policy(observation: dict, mask: np.ndarray) -> tuple[int, int, int]:
    """The default opponent for :class:`CRSimEnv`: always passes.

    A stand-in for bring-up and unit testing, not for producing a useful
    policy -- an agent trained against an opponent that never plays a card
    learns to punish an empty board, which is not the game it will be
    evaluated on. Pass a real ``opponent_policy`` (a scripted heuristic such
    as ``cr_sim.cli``'s demo script, or a frozen snapshot of the learner
    itself) for training runs that need to mean something.
    """
    del observation, mask
    return (NUM_CARD_SLOTS - 1, 0, 0)


def _apply_action(battle: Battle, team: Team, action: Sequence[int], config: EncodingConfig) -> bool:
    """Decode and attempt one action; returns whether it actually played.

    Does not consult the legality mask -- ``Battle.play_card`` already
    re-validates elixir and placement and simply returns ``False`` on
    failure, so every action, masked-legal or not, is routed through the same
    one check rather than through this and the mask separately, which could
    drift apart if either changed without the other.
    """
    decoded = decode_action(action, team, battle.arena, config)
    if decoded is None:
        return False
    slot, x, y = decoded
    hand = battle.players[team].hand
    if slot >= len(hand):
        return False
    return battle.play_card(team, hand[slot], x, y)


def _advance(battle: Battle, frame_skip: int) -> int:
    """Step the engine up to ``frame_skip`` ticks, stopping early if the
    match ends mid-skip. Returns the number of ticks actually run."""
    ran = 0
    for _ in range(frame_skip):
        if battle.finished:
            break
        battle.step()
        ran += 1
    return ran


def _shaped_value(battle: Battle, team: Team, shaping_weight: float) -> float:
    """One side's running score: crown difference plus a weighted tower
    health fraction difference. The reward each step is the *change* in this
    value; see :class:`CRSimEnv` for why."""
    opponent = team.opponent
    crown_diff = battle.players[team].crowns - battle.players[opponent].crowns
    own_hp, own_max = total_tower_hitpoints(battle, team)
    enemy_hp, enemy_max = total_tower_hitpoints(battle, opponent)
    own_frac = own_hp / own_max if own_max else 0.0
    enemy_frac = enemy_hp / enemy_max if enemy_max else 0.0
    return crown_diff + shaping_weight * (own_frac - enemy_frac)


def _info(battle: Battle) -> dict[str, Any]:
    """The hash, tick and crowns every step's ``info`` carries. The hash is
    what a determinism check compares across two runs of the same seed; the
    others are what a training loop typically logs without having to reach
    into ``battle`` itself.
    """
    return {
        "hash": battle.hash(),
        "tick": battle.tick,
        "blue_crowns": battle.players[Team.BLUE].crowns,
        "red_crowns": battle.players[Team.RED].crowns,
        "finished": battle.finished,
        "reason": battle.result.reason if battle.result is not None else None,
    }


class CRSimEnv(_EnvBase):
    """One battle, viewed from one team's side, as a single-agent environment.

    **Reward.** Crown difference is the true objective -- it is what actually
    wins a match -- but it is sparse: a battle can run thousands of ticks
    between crowns, and a learner gets no signal at all on most of them.
    Tower hitpoint difference is the dense shaping term underneath it: every
    trade of damage moves it, which is gradient on ticks where nothing
    decisive has happened yet. The reward returned each step is the *change*
    in ``crown_diff + reward_shaping_weight * tower_hp_frac_diff`` since the
    previous step, not a value recomputed from scratch -- summed across an
    episode this telescopes exactly to ``final_crown_diff +
    reward_shaping_weight * final_tower_hp_frac_diff`` (both sides start at
    full tower health, so the initial value is always 0 and cannot bias the
    sum). That means ``reward_shaping_weight=0`` recovers the pure sparse
    crown-difference objective through this same one code path rather than a
    second one to keep in sync, and annealing the weight toward zero over
    training is a one-line change wherever this constructor is called.

    **Opponent.** ``opponent_policy(observation, mask) -> action`` is called
    once per decision point for the team ``CRSimEnv`` does not control,
    defaulting to :func:`idle_opponent_policy`. This is the simpler of the
    two two-agent designs this package offers -- see :class:`CRSimSelfPlayEnv`
    for the other -- and is the right one for single-agent training against a
    fixed adversary (a scripted heuristic, or a frozen checkpoint for
    curriculum play) and for evaluation, where exactly one side is the policy
    under test and the other's behaviour needs to be reproducible.

    **Frame skip.** The agent decides every ``frame_skip`` engine ticks; the
    engine still simulates every one of them in between, at whatever
    ``ticks_per_second`` the battle runs. See :data:`DEFAULT_FRAME_SKIP` and
    :data:`DEFAULT_TICKS_PER_SECOND` for the default cadence and how to scale
    them together.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        data: LogicData,
        levels: LevelTable,
        registry: CardRegistry,
        blue_deck: Sequence[str],
        red_deck: Sequence[str],
        *,
        team: Team = Team.BLUE,
        opponent_policy: Callable[[dict, np.ndarray], Sequence[int]] | None = None,
        ticks_per_second: int = DEFAULT_TICKS_PER_SECOND,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        level: int = 11,
        tower_level: int = 11,
        reward_shaping_weight: float = 0.01,
        max_ticks: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.data = data
        self.levels = levels
        self.registry = registry
        self.blue_deck = tuple(blue_deck)
        self.red_deck = tuple(red_deck)
        self.team = team
        self.opponent_policy = opponent_policy or idle_opponent_policy
        self.ticks_per_second = ticks_per_second
        self.frame_skip = frame_skip
        self.level = level
        self.tower_level = tower_level
        self.reward_shaping_weight = reward_shaping_weight
        self.max_ticks = max_ticks
        self.render_mode = render_mode

        # Loaded once here rather than left to Battle's own default: the
        # tilemap never changes within an env's lifetime, and re-parsing its
        # CSV on every reset() would be pure waste across a long training run.
        self._arena: Arena = load_arena(data)
        self._config = build_encoding_config(self._arena, self.blue_deck, self.red_deck)
        shapes = observation_shapes(self._config)
        self.observation_space = DictSpace(
            {
                "grid": Box(0.0, 1.0, shapes["grid"], dtype=np.float32),
                "vector": Box(0.0, 1.0, shapes["vector"], dtype=np.float32),
            }
        )
        self.action_space = MultiDiscrete(
            [NUM_CARD_SLOTS, self._config.action_width, self._config.action_height]
        )

        self.battle: Battle | None = None
        self._prev_value = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options  # accepted for Gymnasium API compatibility; unused here
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        self.battle = Battle(
            self.data,
            self.levels,
            self.registry,
            BattleConfig(
                seed=seed,
                ticks_per_second=self.ticks_per_second,
                blue_deck=self.blue_deck,
                red_deck=self.red_deck,
                level=self.level,
                tower_level=self.tower_level,
            ),
            arena=self._arena,
        )
        self._prev_value = _shaped_value(self.battle, self.team, self.reward_shaping_weight)
        return self._observe(self.team), _info(self.battle)

    def step(self, action: Sequence[int]):
        if self.battle is None:
            raise RuntimeError("call reset() before step()")
        if self.battle.finished:
            raise RuntimeError("battle already finished; call reset() to start a new episode")

        _apply_action(self.battle, self.team, action, self._config)

        opponent = self.team.opponent
        opponent_obs = self._observe(opponent)
        opponent_mask = legal_action_mask(self.battle, opponent, self.registry, self._config)
        opponent_action = self.opponent_policy(opponent_obs, opponent_mask)
        _apply_action(self.battle, opponent, opponent_action, self._config)

        _advance(self.battle, self.frame_skip)

        value = _shaped_value(self.battle, self.team, self.reward_shaping_weight)
        reward = value - self._prev_value
        self._prev_value = value

        terminated = self.battle.finished
        truncated = (
            not terminated and self.max_ticks is not None and self.battle.tick >= self.max_ticks
        )
        return self._observe(self.team), reward, terminated, truncated, _info(self.battle)

    def legal_action_mask(self) -> np.ndarray:
        """The controlled team's current legal-action mask; see
        :func:`cr_sim.api.encoding.legal_action_mask`."""
        if self.battle is None:
            raise RuntimeError("call reset() before legal_action_mask()")
        return legal_action_mask(self.battle, self.team, self.registry, self._config)

    def render(self):
        if self.battle is None:
            return None
        text = render_ascii(self.battle.arena, self.battle.entities)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self) -> None:
        self.battle = None

    def _observe(self, team: Team) -> dict[str, np.ndarray]:
        return encode_observation(self.battle, team, self.registry, self._config)


class CRSimSelfPlayEnv:
    """Both sides of one battle, stepped together, for self-play training.

    ``reset``/``step`` deal in a dict keyed by :class:`~cr_sim.engine.entity.Team`
    rather than one team's view: ``{Team.BLUE: obs_blue, Team.RED: obs_red}``
    in from ``reset()``, ``{Team.BLUE: action_blue, Team.RED: action_red}``
    expected by ``step()``. This is the alternative to :class:`CRSimEnv`'s
    ``opponent_policy`` hook, and the one that matters for producing a
    genuinely strong policy: an opponent fixed at construction time -- a
    scripted heuristic, or one frozen checkpoint -- is a stationary target a
    learner can overfit to and stop improving against, where self-play keeps
    the opponent exactly as strong as the learner currently is. Both teams'
    actions are applied to the battle before either side's engine ticks run,
    so neither side has an ordering advantage within one decision: applying
    Blue's action and running ticks before Red has even chosen would let Red
    react to a placement Blue has not actually committed to yet from the
    engine's point of view.

    Deliberately not a ``gymnasium.Env`` subclass. Gymnasium's API is
    single-agent by design -- one observation, one action, one pair of done
    flags -- and forcing two agents through it would mean either wrapping a
    second env instance around this one or overloading the single-agent
    contract until callers can no longer rely on what Gymnasium's own
    documentation says about it. PettingZoo defines the standard multi-agent
    contract this class's shape is modelled after; adopting it as an actual
    base class would add a second hard dependency on top of gymnasium already
    being optional, for what is, underneath, the same dict-of-per-team
    reset/step this class already provides directly.
    """

    def __init__(
        self,
        data: LogicData,
        levels: LevelTable,
        registry: CardRegistry,
        blue_deck: Sequence[str],
        red_deck: Sequence[str],
        *,
        ticks_per_second: int = DEFAULT_TICKS_PER_SECOND,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        level: int = 11,
        tower_level: int = 11,
        reward_shaping_weight: float = 0.01,
        max_ticks: int | None = None,
    ) -> None:
        self.data = data
        self.levels = levels
        self.registry = registry
        self.blue_deck = tuple(blue_deck)
        self.red_deck = tuple(red_deck)
        self.ticks_per_second = ticks_per_second
        self.frame_skip = frame_skip
        self.level = level
        self.tower_level = tower_level
        self.reward_shaping_weight = reward_shaping_weight
        self.max_ticks = max_ticks

        self._arena: Arena = load_arena(data)
        self._config = build_encoding_config(self._arena, self.blue_deck, self.red_deck)
        shapes = observation_shapes(self._config)
        self.observation_space = DictSpace(
            {
                "grid": Box(0.0, 1.0, shapes["grid"], dtype=np.float32),
                "vector": Box(0.0, 1.0, shapes["vector"], dtype=np.float32),
            }
        )
        self.action_space = MultiDiscrete(
            [NUM_CARD_SLOTS, self._config.action_width, self._config.action_height]
        )

        self.battle: Battle | None = None
        self._prev_value: dict[Team, float] = {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        self.battle = Battle(
            self.data,
            self.levels,
            self.registry,
            BattleConfig(
                seed=seed,
                ticks_per_second=self.ticks_per_second,
                blue_deck=self.blue_deck,
                red_deck=self.red_deck,
                level=self.level,
                tower_level=self.tower_level,
            ),
            arena=self._arena,
        )
        self._prev_value = {
            t: _shaped_value(self.battle, t, self.reward_shaping_weight) for t in (Team.BLUE, Team.RED)
        }
        obs = {t: self._observe(t) for t in (Team.BLUE, Team.RED)}
        return obs, _info(self.battle)

    def step(self, actions: dict[Team, Sequence[int]]):
        if self.battle is None:
            raise RuntimeError("call reset() before step()")
        if self.battle.finished:
            raise RuntimeError("battle already finished; call reset() to start a new episode")

        for team in (Team.BLUE, Team.RED):
            _apply_action(self.battle, team, actions[team], self._config)

        _advance(self.battle, self.frame_skip)

        reward: dict[Team, float] = {}
        for team in (Team.BLUE, Team.RED):
            value = _shaped_value(self.battle, team, self.reward_shaping_weight)
            reward[team] = value - self._prev_value[team]
            self._prev_value[team] = value

        terminated = self.battle.finished
        truncated = (
            not terminated and self.max_ticks is not None and self.battle.tick >= self.max_ticks
        )
        obs = {t: self._observe(t) for t in (Team.BLUE, Team.RED)}
        return obs, reward, terminated, truncated, _info(self.battle)

    def legal_action_mask(self, team: Team) -> np.ndarray:
        if self.battle is None:
            raise RuntimeError("call reset() before legal_action_mask()")
        return legal_action_mask(self.battle, team, self.registry, self._config)

    def render(self) -> str | None:
        if self.battle is None:
            return None
        return render_ascii(self.battle.arena, self.battle.entities)

    def close(self) -> None:
        self.battle = None

    def _observe(self, team: Team) -> dict[str, np.ndarray]:
        return encode_observation(self.battle, team, self.registry, self._config)
