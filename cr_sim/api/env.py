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
from .reward import (
    ProjectedReward, ProjectionWeights, RewardTracker, RewardWeights,
)
from .encoding import (
    NUM_CARD_SLOTS,
    OBSERVATION_V1,
    EncodingConfig,
    ObservationFeatures,
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
#: 30 ticks at 60 TPS is 500ms between decisions.
#:
#: This was measured rather than guessed, and the first guess (100ms) was much
#: too fast. A decision is only worth sampling if there is more than one legal
#: action at it, and for most of a match there is not: elixir is spent about as
#: fast as it accrues, so a player is broke most of the time and passing is the
#: only thing available. Counting forced decisions across a full match:
#:
#: ==========  ==================  ===================
#: cadence     decisions forced    mean legal actions
#: ==========  ==================  ===================
#: 300ms                    93%                    7
#: 600ms                    87%                   13
#: 1.0s                     78%                   21
#: 2.0s                     56%                   41
#: ==========  ==================  ===================
#:
#: At 100ms nearly every sample a rollout collects has a single legal action,
#: contributes no gradient, and still costs a full network evaluation. 500ms
#: keeps enough reactivity to time a Log or snipe with a spell while roughly
#: halving the wasted samples.
#:
#: cr_sim.engine.constants.TRAINING_TPS (20) is offered for cheap bulk training
#: via ``ticks_per_second``; scale this alongside it to hold the same real-time
#: cadence rather than silently changing how often the agent acts.
DEFAULT_FRAME_SKIP = 30

#: Ceiling on how many no-choice decisions ``skip_forced`` will run through in
#: one step. A whole match is a few hundred decisions at the default cadence,
#: so this only ever binds if something has gone wrong -- and spinning forever
#: inside a step is a much worse failure than returning early.
_MAX_FORCED_RUN_OUT = 4096


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


def _build_reward(team, registry, reward_weights):
    """The reward object ``reward_weights`` selects, by its type.

    Pulled out of ``CRSimEnv.__init__`` so that :meth:`CRSimEnv.reset` can
    rebuild the reward when a schedule has moved its weights, through this one
    dispatch rather than a second copy of it. Rebuilding rather than mutating
    ``self._reward.weights`` is deliberate: a rebuild cannot change the
    reward's *type* by accident, and reset() re-baselines the potential
    anyway.
    """
    if isinstance(reward_weights, ProjectionWeights):
        return ProjectedReward(team, reward_weights)
    if reward_weights is not None:
        return RewardTracker(team, registry, reward_weights)
    return None


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
    second one to keep in sync.

    **``reward_shaping_weight`` is read only when ``reward_weights`` is
    None.** This paragraph used to end by saying that annealing it toward
    zero over training was "a one-line change wherever this constructor is
    called", and that sentence was a trap. Every call site of
    ``_shaped_value`` is inside the ``else`` of ``if self._reward is not
    None``, so under ``--reward projected`` or ``--reward five-term`` -- which
    is every fine-tune this project has run -- the weight is never read at
    all. Measured on identical seeds and an identical action stream, 0.01
    against 5.00, a five hundred fold change:

        projected: IDENTICAL     five-term: IDENTICAL     simple: DIFFERS

    So an anneal aimed here is a run that reports an anneal and performs
    none. The shaping actually in force under ``projected`` is
    ``ProjectionWeights.tower`` and ``.elixir``; under ``five-term`` it is the
    five non-crown ``RewardWeights`` fields. To move any of them over
    training, hand the new weights to :meth:`set_reward_weights`, which
    applies them at the next reset -- see there for why the boundary matters.

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
        reward_weights: "RewardWeights | ProjectionWeights | None" = None,
        max_ticks: int | None = None,
        render_mode: str | None = None,
        skip_forced: bool = True,
        observation: ObservationFeatures = OBSERVATION_V1,
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
        #: When given, the five-term reward replaces the single tower-health
        #: difference. Kept optional so the simpler signal stays available as a
        #: control -- an ablation needs something to ablate against.
        self.reward_weights = reward_weights
        # Which weights are passed picks the reward: hand-weighted terms, or
        # the change in what the board projects to. Dispatching on the type
        # rather than a separate flag keeps the two from being set
        # inconsistently.
        self._reward = _build_reward(team, registry, reward_weights)
        #: A weights object waiting for the next reset(), as a zero-or-one
        #: element list because ``None`` is itself a valid value -- it selects
        #: the simple shaped reward -- so it cannot double as "nothing
        #: pending". See :meth:`set_reward_weights`.
        self._pending_reward: list = []
        self.max_ticks = max_ticks
        self.render_mode = render_mode
        self.skip_forced = skip_forced

        # Loaded once here rather than left to Battle's own default: the
        # tilemap never changes within an env's lifetime, and re-parsing its
        # CSV on every reset() would be pure waste across a long training run.
        self._arena: Arena = load_arena(data)
        self.observation = observation
        self._config = build_encoding_config(
            self._arena, self.blue_deck, self.red_deck, observation)
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

    @property
    def encoding(self) -> EncodingConfig:
        """The encoder this environment's observations and actions are built
        with. Public because a policy network's shapes -- and, for a
        card-conditioned head, the layout of the hand inside the observation
        vector -- are properties of the encoding, and every caller that builds
        a network for this env needs them from one place rather than
        restating them."""
        return self._config

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
        # The one safe point to adopt a scheduled weight. See
        # set_reward_weights for why anywhere else is a correctness bug.
        if self._pending_reward:
            self.reward_weights, pending_shaping = self._pending_reward.pop()
            if pending_shaping is not None:
                self.reward_shaping_weight = pending_shaping
            self._reward = _build_reward(
                self.team, self.registry, self.reward_weights)

        if self._reward is not None:
            self._reward.reset(self.battle)
            self._prev_value = 0.0
        else:
            self._prev_value = _shaped_value(
                self.battle, self.team, self.reward_shaping_weight
            )
        return self._observe(self.team), _info(self.battle)

    def set_reward_weights(self, weights, *,
                           shaping_weight: float | None = None) -> None:
        """Take effect at the next :meth:`reset`, never mid-episode.

        Mid-episode the reward stops being potential-based. The tracker's
        ``_previous`` holds the potential under the *old* weight, so the very
        next step is paid ``phi_new(s_new) - phi_old(s_old)`` -- a genuine
        reward plus a fabricated one for the weight change, charged in full to
        whatever action happened to be taken there. Measured, switching
        ``ProjectionWeights`` from (tower=1, elixir=0.3) to (0, 0) at step 5
        of an episode:

            step 5 reward:  no-switch -0.007802   switched -0.159656

        Nineteen times the genuine reward, handed to one arbitrary action.
        And it is invisible in aggregate, which is what makes it dangerous:
        the episode return still telescopes correctly to its own endpoint
        weights, so the existing telescoping invariant stays green over it.

        ``weights`` is the object whose *type* selects the reward, exactly as
        the constructor's argument does, so ``None`` is meaningful -- it
        selects the simple shaped reward, whose own knob is
        ``shaping_weight``.
        """
        self._pending_reward = [(weights, shaping_weight)]

    def step(self, action: Sequence[int]):
        if self.battle is None:
            raise RuntimeError("call reset() before step()")
        if self.battle.finished:
            raise RuntimeError("battle already finished; call reset() to start a new episode")

        _apply_action(self.battle, self.team, action, self._config)

        self._opponent_move()

        _advance(self.battle, self.frame_skip)

        # Decided before the reward, because whether a run-out follows is what
        # says whether this state needs scoring at all.
        terminated = self.battle.finished
        truncated = (
            not terminated and self.max_ticks is not None and self.battle.tick >= self.max_ticks
        )
        running_out = self.skip_forced and not terminated and not truncated

        reward = 0.0
        if self._reward is not None:
            # A telescoping reward that is about to be scored again at the end
            # of the run-out is scored *there and only there*. score() is a
            # pure function of state, so (phi_mid - phi_prev) + (phi_end -
            # phi_mid) is phi_end - phi_prev and the intermediate term
            # cancels exactly -- measured bit-identical on two of three
            # episodes and 2.2e-16 on the third, which is the same floating
            # point floor the potential identity itself sits at.
            #
            # It was not free: under `projected` this state was scored on
            # every single non-terminal decision purely to have its
            # contribution cancelled a few lines later -- exactly 2.00 score
            # calls per decision, 48.7% of all projections, and 26.4% of all
            # environment wall time.
            if not (running_out and getattr(self._reward, "telescopes", False)):
                reward = self._reward.step(self.battle, self.frame_skip)
        else:
            value = _shaped_value(self.battle, self.team, self.reward_shaping_weight)
            reward = value - self._prev_value
            self._prev_value = value

        if running_out:
            reward += self._run_out_forced_decisions()
            terminated = self.battle.finished
            truncated = (
                not terminated
                and self.max_ticks is not None
                and self.battle.tick >= self.max_ticks
            )

        return self._observe(self.team), reward, terminated, truncated, _info(self.battle)

    def _opponent_move(self) -> None:
        """Let the other side act, without asking it when there is no choice.

        The same rule the agent gets, and for the same reason -- but it matters
        more here. A neural opponent costs a full observation encode and a
        forward pass every time it is consulted, and it was being consulted on
        every one of the agent's forced decisions too, roughly nine times per
        agent step. Since the opponent is broke on most of those ticks and its
        only legal move is to pass, nearly all of that work produced the action
        it would have taken anyway. Skipping it took self-play from 12 steps a
        second to something worth running overnight.
        """
        opponent = self.team.opponent
        mask = legal_action_mask(self.battle, opponent, self.registry, self._config)
        legal = int(mask.sum())
        if legal <= 0:
            return
        if legal == 1:
            only = tuple(int(v) for v in np.argwhere(mask)[0])
            _apply_action(self.battle, opponent, only, self._config)
            return
        # A searching opponent needs the board, not an encoded view of it --
        # it branches the battle to evaluate placements. Passed only when the
        # callable asks, so a network opponent keeps seeing exactly what a
        # player would and cannot quietly read state it should not have.
        if getattr(self.opponent_policy, "wants_battle", False):
            action = self.opponent_policy(self._observe(opponent), mask, self.battle)
        else:
            action = self.opponent_policy(self._observe(opponent), mask)
        _apply_action(self.battle, opponent, action, self._config)

    def _run_out_forced_decisions(self) -> float:
        """Advance past every state where there is nothing to decide.

        For most of a match the only legal action is to pass: elixir is spent
        about as fast as it accrues, so a player is broke most of the time.
        Measured across a full match, 89% of decisions at the default cadence
        have exactly one legal action. Handing those to the policy costs a full
        network evaluation each, produces a transition whose gradient is
        identically zero (a one-action softmax cannot be wrong), and dilutes
        every batch nine to one.

        Frame skip already does this for *time*; this does it for *choice*. The
        result is the same MDP -- a state with one action is not a decision
        point, and taking the only move available cannot change the policy --
        with roughly nine times the useful samples per step.

        Reward accrued while running out is returned so it is not lost: what
        happens during those ticks still counts, it just is not attributable to
        a choice.
        """
        gained = 0.0
        # A reward that is a pure potential sums to the difference between the
        # first and last state, so it is scored once at the end rather than at
        # every state passed through. Each score costs a board projection and
        # this loop runs about nine times per real decision, so the
        # intermediate ones were roughly two thirds of the run's compute --
        # for a number that cancels.
        telescoping = getattr(self._reward, "telescopes", False)
        # Bounded so a pathological state cannot spin here forever. A whole
        # match at the default cadence is a few hundred decisions.
        for _ in range(_MAX_FORCED_RUN_OUT):
            mask = self.legal_action_mask()
            if int(mask.sum()) > 1:
                break
            only = tuple(int(v) for v in np.argwhere(mask)[0])

            _apply_action(self.battle, self.team, only, self._config)
            self._opponent_move()
            _advance(self.battle, self.frame_skip)

            if self._reward is not None:
                if not telescoping:
                    gained += self._reward.step(self.battle, self.frame_skip)
            else:
                value = _shaped_value(self.battle, self.team, self.reward_shaping_weight)
                gained += value - self._prev_value
                self._prev_value = value

            if self.battle.finished or (
                self.max_ticks is not None and self.battle.tick >= self.max_ticks
            ):
                break
        # One settle point for every way out of the loop. A telescoping reward
        # is scored here and only here, so whichever exit was taken, the caller
        # is paid the change across the whole run-out.
        if telescoping and self._reward is not None:
            gained += self._reward.step(self.battle, self.frame_skip)
        return gained

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
    
    **No ``reward_weights``, so no schedule.** This class takes only
    ``reward_shaping_weight``, which means the only shaping it can anneal is
    the simple reward's -- the one case where that flag is real. Deliberately
    not extended: nothing but ``tests/test_api_env.py`` builds this today,
    ``cr_sim.train.run`` does self-play through :class:`CRSimEnv`'s
    ``opponent_policy`` instead, and a second reward-construction site to keep
    in sync is how two rewards drift apart. Noted so whoever adopts this knows
    what they are not getting.
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
