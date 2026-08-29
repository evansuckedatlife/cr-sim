"""Whether the policy actually steers the search, and whether it stays honest.

The loop this closes has three arrows -- policy proposes, search refines, the
refined distribution trains the policy -- and this project built the second and
the third and left the first open for its whole life. ``_sample_actions`` draws
about fourteen stratified-random placements out of a mean of 104 legal ones,
13.5% coverage, and the other 86.5% are never scored at all.

Four things have to hold before that is worth doing, and each of them is a way
this codebase has already been wrong at least once:

*   **The old bot is untouched.** Not "equivalent" -- untouched. The golden
    below was captured by running the *pre-change* source, so it is a
    reproduction check rather than a self-consistency check.
*   **The budget is equal.** A guided bot that appends its nominations to the
    random draw is a bot with a bigger budget, and it would beat the unguided
    one for the least interesting reason there is. Measured during
    development: filling the remainder at a reduced budget took 63 branches
    where the unguided bot took 32, because ``per_slot`` is ``max(1, budget //
    slots)`` and asking for fewer does not draw fewer.
*   **Nothing touches torch's global generator.** That defect is live in this
    repo -- ``evaluate.py:206`` samples off the unseeded global stream in two
    production scripts -- and it has already produced two arms of one A/B with
    28 and 29 decisions in what was meant to be the same battle.
*   **The target still spans something.** Changing the proposal changes the
    *labels*: the cloner trains against the search's distribution over the
    candidates that were actually scored. A proposer that nominates what the
    policy already likes narrows that support, and the clone then sharpens a
    preference instead of improving one.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from cr_sim.api.encoding import NOOP_SLOT
from cr_sim.api.env import CRSimEnv
from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.entity import Team
from cr_sim.train.clone import Demonstrations, collect
from cr_sim.train.proposal import (
    check_equal_branch_budget, measure_decision_cost, policy_proposer,
    proposer_factory, proposer_identity,
)
from cr_sim.train.scripted import SearchBot, SearchBotConfig

from .test_data_pipeline import BUILD

DECK = ("Knight", "Musketeer", "Cannon", "Skeletons",
        "IceSpirits", "Log", "Fireball", "Goblins")

#: The configuration every real-battle test here uses. Small on purpose: a
#: decision at the shipped 14 candidates and 15 seconds costs 375 ms, and a
#: test that costs six minutes is a test nobody runs.
SEARCH = SearchBotConfig(candidates=6, horizon_seconds=8.0, seed=0)

#: The shipped candidate count, for the one test that has to be run at it.
LEGACY = SearchBotConfig(candidates=14, horizon_seconds=8.0, seed=0)


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


@pytest.fixture(scope="module")
def make_env(world):
    data, levels, registry = world

    def factory(opponent=None) -> CRSimEnv:
        return CRSimEnv(data, levels, registry, DECK, DECK,
                        ticks_per_second=20, frame_skip=40,
                        max_ticks=20 * 40, tower_level=5,
                        opponent_policy=opponent)

    return factory


@pytest.fixture(scope="module")
def nvec(make_env):
    env = make_env(None)
    env.reset(seed=0)
    return [int(v) for v in env.action_space.nvec]


class _FixedNet:
    """A stand-in for ``ActorCritic`` that returns logits chosen by the test.

    The proposer reads exactly one thing off a network -- ``policy_logits`` --
    so a stub is enough to test the ranking, and it makes the tie-breaking and
    influence cases exact instead of statistical. The real network is used
    where the real network is the point: in the reproducibility and
    equal-budget runs below.
    """

    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def policy_logits(self, grid, vector, mask):
        flat = np.asarray(mask).reshape(-1)
        row = np.full(flat.shape, -1e8, dtype=np.float32)
        row[:len(self._values)] = self._values[:len(row)]
        return torch.from_numpy(row).unsqueeze(0)


def _favouring(index: int, size: int, edge: float = 40.0) -> _FixedNet:
    """A policy that wants one cell far more than any other."""
    values = np.zeros(size, dtype=np.float32)
    values[index] = edge
    return _FixedNet(values)


def _play(env, bot, limit: int):
    """Run a battle, recording what the bot considered at each decision."""
    observation, _ = env.reset(seed=0)
    seen = []
    while len(seen) < limit:
        mask = env.legal_action_mask()
        if not mask.reshape(-1).any():
            break
        choice = bot(observation, mask, env.battle)
        seen.append((tuple(int(v) for v in choice),
                     [(int(i), round(float(v), 6)) for i, v in bot.last_scores]))
        observation, _, terminated, truncated, _ = env.step(choice)
        if terminated or truncated:
            break
    return seen


# ----------------------------------------------------- the bot that exists

#: Ten decisions of ``SearchBot(LEGACY)`` on battle seed 0, captured by
#: checking out the source as it stood *before* the proposer hook was added,
#: running it, and putting the file back. A reproduction check rather than a
#: self-consistency check: every demonstration set, clone and verdict on this
#: machine came off this code path, and if any of it moved they would all have
#: to be recollected.
#:
#: At the shipped ``candidates=14`` deliberately, not at this file's cheaper
#: ``SEARCH``. With six candidates over four legal cards ``per_slot`` rounds
#: to one, and a one-element draw is the same draw with or without
#: replacement -- so the golden was blind to ``replace=True``, which is one of
#: the two ways the sampler can silently change. Fourteen draws three per
#: card and sees it.
GOLDEN = [
    ((4, 0, 0),
     [(576, 0.0), (84, 0.0), (68, 0.0), (115, 0.0), (145, 0.0), (148, 0.0), (146, 0.0), (372, 0.0), (356, 0.0), (416, 0.0), (530, 0.0), (502, 0.0), (516, 0.0)]),
    ((4, 0, 0),
     [(576, 0.0), (113, 0.0), (96, 0.0), (34, 0.0), (259, 0.0), (214, 0.0), (146, 0.0), (307, 0.0), (402, 0.0), (293, 0.0), (468, 0.0), (501, 0.0), (436, 0.0)]),
    ((4, 0, 0),
     [(576, 0.0), (1, 0.0), (0, 0.0), (51, 0.0), (278, 0.0), (213, 0.0), (230, 0.0), (388, 0.0), (338, 0.0), (352, 0.0), (563, 0.0), (482, 0.0), (529, 0.0)]),
    ((4, 0, 0),
     [(576, 0.0), (114, 0.0), (98, 0.0), (96, 0.0), (161, 0.0), (225, 0.0), (243, 0.0), (337, 0.0), (341, 0.0), (324, 0.0), (562, 0.0), (549, 0.0), (436, 0.0)]),
    ((4, 0, 0),
     [(576, 0.0), (33, 0.0), (80, 0.0), (86, 0.0), (211, 0.0), (225, 0.0), (182, 0.0), (322, 0.0), (406, 0.0), (325, 0.0), (515, 0.0), (437, 0.0), (434, 0.0)]),
    ((4, 0, 0),
     [(576, 0.0), (37, 0.0), (112, 0.0), (51, 0.0), (148, 0.0), (260, 0.0), (147, 0.0), (368, 0.0), (405, 0.0), (306, 0.0), (530, 0.0), (534, 0.0), (464, 0.0)]),
    ((0, 8, 6),
     [(576, 0.0), (134, 0.032063), (51, 0.0), (80, 0.0), (228, 0.0), (149, 0.0), (225, 0.0), (416, 0.0), (374, 0.0), (406, 0.0), (481, 0.0), (561, 0.0), (434, 0.0)]),
    ((4, 0, 0),
     [(576, 0.096188), (84, 0.096188), (67, 0.096188), (102, 0.096188), (208, 0.096188), (276, 0.096188), (197, 0.096188), (419, 0.096188), (290, 0.096188), (340, 0.096188), (514, 0.096188), (563, 0.096188), (564, 0.096188)]),
    ((0, 6, 5),
     [(576, 0.160313), (52, 0.160313), (114, 0.160313), (101, 0.176566), (176, 0.160313), (256, 0.160313), (212, 0.160313), (322, 0.160313), (387, 0.160313), (389, 0.160313), (448, 0.160313), (452, 0.160313), (560, 0.160313)]),
    ((4, 0, 0),
     [(576, 0.240691), (131, 0.240691), (129, 0.240691), (96, 0.240691), (261, 0.240691), (160, 0.240691), (144, 0.240691), (337, 0.240691), (419, 0.240691), (401, 0.240691), (569, 0.240691), (558, 0.240691), (484, 0.240691)]),
]

def test_the_random_proposer_is_unchanged(make_env):
    """``proposer=None`` must be the old bot, to the index and the value.

    Not "as good as": the same. Six demonstration shards, two clones and seven
    verdicts on this machine were produced by this path, and the whole
    argument for adding a proposer at all is that the thing it replaces stays
    available and stays reproducible.

    Killed by reversing the per-slot loop order, and by drawing the placements
    with replacement. **Not** killed by deleting the ``sorted()`` around the
    slot set, and that is a property of the game rather than a gap here: a
    hand has four card slots, so the set is a subset of ``{0, 1, 2, 3}``,
    small non-negative ints hash to themselves, and a CPython set of them
    always iterates in ascending order. Checked exhaustively over all 64
    insertion orders -- ``list(set) == sorted(set)`` in every one. The
    ``sorted`` is defensive, not live, and no test can distinguish it.
    """
    bot = SearchBot(Team.BLUE, LEGACY)
    assert _play(make_env(None), bot, len(GOLDEN)) == GOLDEN


# --------------------------------------------------- the proposal itself


def test_ties_break_by_flat_index(nvec):
    """Exact ties are the common case, not the corner case.

    The factored head shares its tile weights across cards, so two tiles can
    carry bit-identical logits routinely, and ``torch.topk`` promises no
    ordering among equal values. The ranking is a stable numpy argsort so the
    order is the flat index, which is a property of the action space rather
    than of whichever reduction order a kernel happened to use.
    """
    slots, width, height = nvec
    size = slots * width * height
    values = np.zeros(size, dtype=np.float32)
    for index in (300, 12, 145, 7):
        values[index] = 5.0
    propose = policy_proposer(_FixedNet(values), nvec)

    mask = np.ones(size, dtype=bool)
    order = propose({"grid": np.zeros(1, np.float32),
                     "vector": np.zeros(1, np.float32)}, mask, 0)
    assert order[:4] == [7, 12, 145, 300]


def test_the_policy_proposal_never_touches_the_global_stream(nvec):
    """Torch's global generator is not this module's to spend.

    ``Categorical.sample`` and ``multinomial`` both draw from it unless handed
    a generator, and that is live in production here: every sampled number in
    ``runs/_anchor/*.json`` is unreproducible because of it. A proposal that
    advanced the global stream would also shift every *other* consumer of it
    -- the trainer's own rollouts included -- by however many decisions the
    search happened to make.
    """
    slots, width, height = nvec
    size = slots * width * height
    values = np.linspace(0.0, 3.0, size, dtype=np.float32)
    mask = np.ones(size, dtype=bool)
    observation = {"grid": np.zeros(1, np.float32),
                   "vector": np.zeros(1, np.float32)}

    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    propose = policy_proposer(_FixedNet(values), nvec, temperature=0.7, seed=5)
    drawn = [propose(observation, mask, step) for step in range(20)]
    assert torch.equal(torch.random.get_rng_state(), before)

    # And the same proposer built twice from the same seed proposes the same
    # thing, which is what "seeded" has to mean.
    twin = policy_proposer(_FixedNet(values), nvec, temperature=0.7, seed=5)
    assert [twin(observation, mask, step) for step in range(20)] == drawn
    # Different decisions must not be the same draw over and over, or the
    # sampling temperature is decoration.
    assert len({tuple(row[:5]) for row in drawn}) > 1


def test_a_proposal_at_temperature_zero_reads_no_generator_at_all(nvec):
    """The default path is not merely seeded; it is not random.

    A stable argsort over a numpy copy. Worth its own case because "seeded"
    and "deterministic" are different claims, and only the second one survives
    a caller who reseeds between decisions.
    """
    slots, width, height = nvec
    size = slots * width * height
    # Nearly tied on purpose. A "greedy" implemented as temperature 1e-3 is
    # indistinguishable from the argsort on well-separated logits -- every
    # weight but the top one underflows -- and looks correct right up until it
    # meets a network whose logits are close together, which is what a trained
    # policy's are. This is the same objection that made `greedy` a real flag
    # on FrozenOpponent rather than temperature=1e-3.
    values = np.linspace(0.0, 0.002, size, dtype=np.float32)
    mask = np.ones(size, dtype=bool)
    observation = {"grid": np.zeros(1, np.float32),
                   "vector": np.zeros(1, np.float32)}
    propose = policy_proposer(_FixedNet(values), nvec)

    torch.manual_seed(7)
    first = propose(observation, mask, 0)
    torch.manual_seed(99999)
    assert propose(observation, mask, 0) == first
    assert propose(observation, mask, 3) == first
    # It really is the ranking, not an accident of the ordering.
    assert first[0] == size - 1


def test_a_proposal_never_nominates_the_no_op(nvec):
    """Waiting is seeded into the scores for free and is not a candidate.

    A proposer that nominated it would spend one of the search's branches
    re-scoring a position the bot has already scored, and would displace a
    placement to do it.
    """
    slots, width, height = nvec
    size = slots * width * height
    noop = NOOP_SLOT * width * height
    values = np.zeros(size, dtype=np.float32)
    values[noop] = 100.0
    propose = policy_proposer(_FixedNet(values), nvec)
    mask = np.ones(size, dtype=bool)
    assert noop not in propose({"grid": np.zeros(1, np.float32),
                                "vector": np.zeros(1, np.float32)}, mask, 0)


# ------------------------------------------------------- the merge rule


def test_an_empty_proposal_falls_back_to_the_full_random_draw(make_env, nvec):
    """A proposer with nothing to say must cost nothing, not cost branches.

    The fallback is the default rather than a degraded default. A network that
    is absent, or whose suggestions are all illegal, leaves the bot exactly as
    strong as it was -- otherwise every failure of the proposal machinery
    would show up as a quietly weaker expert.
    """
    def silent(observation, mask, decision):
        return []

    plain = _play(make_env(None), SearchBot(Team.BLUE, SEARCH), 6)
    empty = _play(make_env(None),
                  SearchBot(Team.BLUE,
                            replace(SEARCH, policy_candidates=4),
                            silent), 6)
    assert empty == plain


def test_a_floor_of_random_candidates_is_kept(make_env, nvec):
    """The policy may not have the whole candidate set, however hard it asks.

    This is the support-collapse guard, and it is the one that protects the
    *labels*. The target is a softmax over the candidates that were scored,
    scaled by their own spread: candidates the policy already likes score
    alike, the spread falls, rows collapse to a one-hot on the chosen action,
    and the clone trains on the policy's own preference wearing the search's
    clothes. The measured version of that failure -- 86% of wait-states
    carrying a uniform target and a clone that played a card at every single
    decision -- is what ``min_spread`` was added for.
    """
    slots, width, height = nvec
    greedy = SearchBotConfig(candidates=6, horizon_seconds=8.0, seed=0,
                             policy_candidates=6)
    bot = SearchBot(Team.BLUE, greedy,
                    policy_proposer(_FixedNet(np.linspace(
                        0.0, 3.0, slots * width * height, dtype=np.float32)),
                        nvec))
    # Clamped, and the clamp is recorded rather than applied silently.
    assert greedy.random_floor == 2
    assert bot.config.policy_candidates == 4
    assert bot.requested_policy_candidates == 6
    assert bot.clamped

    # And the floor survives into the candidate set the search actually
    # branches on, which is the claim that matters.
    unguided = SearchBot(Team.BLUE, SEARCH)
    plain = _play(make_env(None), unguided, 6)
    guided = _play(make_env(None), bot, 6)
    for (_, mine), (_, theirs) in zip(guided, plain):
        kept = {i for i, _ in mine} & {i for i, _ in theirs}
        # The no-op is in both by construction; the floor is the rest.
        assert len(kept) >= 3


def test_the_policy_decides_which_candidates_are_considered(make_env, nvec):
    """The point of the whole exercise, stated as a measurement.

    A policy that wants one particular cell must get that cell branched on,
    far more often than the random draw would reach it. The random draw covers
    13.5% of a mean 104 legal actions, so a specific cell arrives by chance
    roughly one decision in eight; if guiding the proposal did not change
    which placements are scored, this would come out at that rate.
    """
    slots, width, height = nvec
    # A tile in the middle of the deploy zone, on the first card slot, so it
    # is legal whenever that card is affordable.
    wanted = 0 * width * height + 6 * height + 6
    proposer = policy_proposer(_favouring(wanted, slots * width * height), nvec)
    guided = replace(SEARCH, policy_candidates=4)

    considered = 0
    legal_times = 0
    env = make_env(None)
    observation, _ = env.reset(seed=0)
    bot = SearchBot(Team.BLUE, guided, proposer)
    for _ in range(12):
        mask = env.legal_action_mask()
        if not mask.reshape(-1).any():
            break
        legal_times += int(bool(mask.reshape(-1)[wanted]))
        choice = bot(observation, mask, env.battle)
        considered += int(any(i == wanted for i, _ in bot.last_scores))
        observation, _, terminated, truncated, _ = env.step(choice)
        if terminated or truncated:
            break

    plain = SearchBot(Team.BLUE, SEARCH)
    by_chance = sum(int(any(i == wanted for i, _ in scores))
                    for _, scores in _play(make_env(None), plain, 12))

    assert legal_times >= 6, "the cell has to be legal for this to say anything"
    # Every decision where it was legal, against the random draw's near-never.
    assert considered == legal_times
    assert considered > by_chance + 3


def test_the_two_experts_are_compared_at_an_equal_branch_budget(
        make_env, nvec):
    """A win bought with more branches is not the win being claimed.

    Two halves. The harness refuses a mismatched configuration outright, and
    the bots' *measured* branch counts are checked against each other on a
    real battle -- because the configuration is what was asked for and the
    count is what was spent. Filling the remainder of the budget from
    ``_sample_actions`` at a reduced budget passes the first check and fails
    the second: ``per_slot`` is ``max(1, budget // slots)``, so asking for two
    across four legal cards still returns four, and the guided bot took 63
    branches where the unguided one took 32.
    """
    guided = replace(SEARCH, policy_candidates=4)
    with pytest.raises(ValueError, match="candidates"):
        check_equal_branch_budget(
            SEARCH, replace(SEARCH, candidates=14))
    with pytest.raises(ValueError, match="horizon_seconds"):
        check_equal_branch_budget(
            SEARCH, replace(SEARCH, horizon_seconds=15.0))
    check_equal_branch_budget(SEARCH, guided)

    slots, width, height = nvec
    proposer = policy_proposer(
        _FixedNet(np.linspace(0.0, 3.0, slots * width * height,
                              dtype=np.float32)), nvec)

    # Board for board, which is the exact form of the claim. The two bots
    # diverge into different games after their first different move, so a
    # whole-battle count compares two different sets of positions; asking both
    # of them about the *same* position removes that and leaves only the
    # merge arithmetic. The guided bot's answer is thrown away -- the unguided
    # one drives, so both walk the same board sequence.
    env = make_env(None)
    observation, _ = env.reset(seed=0)
    driver = SearchBot(Team.BLUE, SEARCH)
    shadow = SearchBot(Team.BLUE, guided, proposer)
    decisions = 0
    while decisions < 12:
        mask = env.legal_action_mask()
        if not mask.reshape(-1).any():
            break
        before = (driver.evaluated, shadow.evaluated)
        choice = driver(observation, mask, env.battle)
        shadow(observation, mask, env.battle)
        assert (driver.evaluated - before[0]
                == shadow.evaluated - before[1]), (
            f"decision {decisions}: the unguided bot branched "
            f"{driver.evaluated - before[0]} times and the guided one "
            f"{shadow.evaluated - before[1]}")
        decisions += 1
        observation, _, terminated, truncated, _ = env.step(choice)
        if terminated or truncated:
            break
    assert decisions >= 8 and driver.evaluated > 0

    # And across whole battles, which is the number the head-to-head harness
    # reports. Looser on purpose: the two bots play different games, so the
    # battles are different lengths -- measured here at 40 decisions against
    # 38, from an identical 160 branches each -- and that alone moves
    # branches-per-decision by 5%. The residual is episode length, not budget;
    # the exact form of the claim is the board-for-board check above. An
    # appending proposer would land near 2.0 either way.
    plain = measure_decision_cost(
        lambda seed: make_env(None),
        lambda seed: SearchBot(Team.BLUE, SEARCH), seeds=(0, 1))
    steered = measure_decision_cost(
        lambda seed: make_env(None),
        lambda seed: SearchBot(Team.BLUE, guided, proposer), seeds=(0, 1))
    ratio = steered["branches_per_decision"] / plain["branches_per_decision"]
    assert 0.90 <= ratio <= 1.10, (plain, steered)


# ------------------------------------------------------- reproducibility


def test_a_guided_battle_reproduces_its_whole_decision_sequence(
        make_env, nvec):
    """The same seed twice, compared decision by decision, not in aggregate.

    Two arms of one A/B on this machine once produced 28 and 29 decisions from
    what was meant to be the same battle, because a torch draw came off the
    unseeded global stream. An aggregate check -- same win rate, same mean
    return -- would not have caught it. This compares the chosen action, the
    full candidate set and every score, at both temperatures, and it reseeds
    the global generator between the two runs so that anything reading it
    would diverge.
    """
    slots, width, height = nvec
    values = np.linspace(0.0, 3.0, slots * width * height, dtype=np.float32)
    guided = replace(SEARCH, policy_candidates=4)

    for temperature in (0.0, 0.6):
        def build():
            return SearchBot(Team.BLUE, guided, policy_proposer(
                _FixedNet(values), nvec, temperature=temperature, seed=11))

        torch.manual_seed(0)
        first = _play(make_env(None), build(), 10)
        torch.manual_seed(4242)
        second = _play(make_env(None), build(), 10)
        assert first == second, f"diverged at temperature {temperature}"
        assert len(first) == 10


def test_a_proposer_is_rebuilt_per_battle_from_that_battles_seed(nvec):
    """A proposer carried across battles is a function of the wrong thing.

    ``search_opponent`` already rebuilds the search bot whenever the battle
    seed changes, for exactly this reason: the bot samples, so a bot carried
    across episodes depends on how many decisions came before rather than on
    the seed, and two arms of a paired evaluation stop playing the same
    battle from their first different move. The proposer inherits the
    property by being rebuilt with it.
    """
    slots, width, height = nvec
    values = np.linspace(0.0, 3.0, slots * width * height, dtype=np.float32)
    build = proposer_factory(_FixedNet(values), nvec, temperature=0.5, seed=3)
    mask = np.ones(slots * width * height, dtype=bool)
    observation = {"grid": np.zeros(1, np.float32),
                   "vector": np.zeros(1, np.float32)}

    assert build(17)(observation, mask, 0) == build(17)(observation, mask, 0)
    assert build(17)(observation, mask, 0) != build(18)(observation, mask, 0)


# ----------------------------------------------- what the cloner is handed


def test_the_target_still_covers_waiting(make_env, nvec):
    """Every recorded row is a distribution, and waiting keeps its mass.

    Two properties, and the second is the one a proposer can break. Each row
    sums to one over at most ``candidates + 1`` entries -- the placements that
    were branched plus the no-op -- and every row where the expert waited
    carries mass on the no-op index or is an honest one-hot on it. Drop the
    unconditional no-op entry and a wait-decision's target becomes a
    distribution over placements the expert declined to make.
    """
    slots, width, height = nvec

    def make_expert(env):
        bot = SearchBot(
            Team.BLUE, replace(SEARCH, policy_candidates=4),
            policy_proposer(_favouring(6 * height + 6, slots * width * height),
                            nvec))

        def expert(observation, mask, battle=None):
            return bot(observation, mask, battle)
        expert.bot = bot
        return expert

    data = collect(lambda index: make_env(None), make_expert, episodes=1,
                   proposer_name="policy:test")
    assert len(data) > 0
    assert data.proposer == "policy:test"
    noop = (slots - 1) * width * height
    assert (data.action == noop).any(), (
        "the expert never waited, so this says nothing about waiting")
    spread_rows = 0
    for row, action in zip(data.target, data.action):
        assert math.isclose(float(row.sum()), 1.0, abs_tol=1e-6)
        assert int(np.count_nonzero(row)) <= SEARCH.candidates + 1
        if int(np.count_nonzero(row)) == 1:
            # The min_spread fallback: the search could not separate its
            # candidates and the only label left is the move it made.
            continue
        spread_rows += 1
        # Every row the search *did* have an opinion about carries mass on
        # waiting, because waiting is seeded into the scores unconditionally.
        # Without that seeding a wait-decision's target becomes a distribution
        # over placements the expert declined to make -- which is exactly the
        # failure that put a uniform target on 86% of wait-states and produced
        # a clone that played a card at every single decision.
        assert row[noop] > 0.0, f"no mass on waiting in a row of {row.sum()}"
        if int(action) == noop:
            assert row[noop] == row.max()
    assert spread_rows > 0


def test_the_collection_records_how_far_its_targets_collapsed(make_env):
    """The collapse rate is measured, not worried about.

    ``make_demos`` gates a merge on this number, so it has to be in the file
    rather than in somebody's memory of the run.

    Asserted against ground truth rather than against a range. The version of
    this test that shipped checked only ``0 <= rate <= 1`` and
    ``spread_mean >= 0``, so hardcoding both call sites in
    ``cr_sim.train.clone`` to ``spreads.append(0.0); collapsed += 0`` -- which
    disables ``make_demos``' merge gate outright, since ``collapse_refusal``
    turns on exactly this field -- left forty-six tests green. A dead
    diagnostic reports 0.0, which is inside every range a range check can
    write down.
    """
    def make_expert(env):
        bot = SearchBot(Team.BLUE, SEARCH)

        def expert(observation, mask, battle=None):
            return bot(observation, mask, battle)
        expert.bot = bot
        return expert

    def one_hot_fraction(data) -> float:
        rows = np.asarray(data.target)
        return float(np.mean([int(np.count_nonzero(row) == 1) for row in rows]))

    data = collect(lambda index: make_env(None), make_expert, episodes=1,
                   meta={"proposer": "random"})
    meta = json.loads(data.meta)
    assert meta["decisions"] == len(data)
    assert meta["proposer"] == "random"

    # A row that fell back is a one-hot on the move the bot happened to make;
    # a row that did not spreads mass over the candidates the search scored.
    # So the recorded rate is countable off the targets themselves.
    assert meta["min_spread_fallback_rate"] == pytest.approx(
        one_hot_fraction(data))
    # And the mean spread is a real measurement of the candidate values, not
    # the 0.0 a dead diagnostic reports.
    assert meta["spread_mean"] > 0.0

    # The other end, forced: no candidate set on this board separates by 10.0,
    # so every row must fall back and the file must say so.
    collapsed = collect(lambda index: make_env(None), make_expert, episodes=1,
                        min_spread=10.0, meta={"proposer": "random"})
    assert one_hot_fraction(collapsed) == 1.0
    assert json.loads(collapsed.meta)["min_spread_fallback_rate"] == 1.0


def test_demonstrations_refuse_to_merge_across_proposers(tmp_path):
    """Two proposers are two label sets, and merging them is undetectable.

    The same guard ``observation`` and ``reward`` already have, for a sharper
    reason: the target is the search's distribution over the candidates it
    actually scored, so a shard proposed by one network and a shard proposed
    by another train different rows against different supervision. Shapes
    match, channel counts match, and the training curve converges either way.
    """
    from scripts.clone_policy import merge

    def _shard(name, proposer):
        path = tmp_path / name
        Demonstrations(
            grid=np.zeros((2, 1, 2, 2), np.float32),
            vector=np.zeros((2, 3), np.float32),
            mask=np.ones((2, 4), bool),
            action=np.zeros(2, np.int64),
            value=np.zeros(2, np.float32),
            target=np.full((2, 4), 0.25, np.float32),
            episodes=1, play_rate=0.5, observation="v1", reward="projected",
            proposer=proposer).save(path)
        return path.with_suffix(".npz")

    same = [_shard("a", "random"), _shard("b", "random")]
    assert merge(same).proposer == "random"

    mixed = [_shard("c", "random"), _shard("d", "policy:abc123@t0p9")]
    with pytest.raises(SystemExit) as raised:
        merge(mixed)
    message = str(raised.value)
    assert "proposer" in message
    assert "c.npz" in message and "d.npz" in message

    # A shard written before the field existed is refused against a stamped
    # one too, rather than assumed to be the random draw.
    with pytest.raises(SystemExit):
        merge([_shard("e", ""), _shard("f", "random")])


def test_a_shard_round_trips_its_proposer_and_its_metadata(tmp_path):
    """Saved and loaded, or the stamp is a comment.

    ``merge`` reads these off the file. A field the writer sets and the reader
    drops is exactly how ``target`` came to be silently discarded by every
    clone this project ever ran.
    """
    path = tmp_path / "shard.npz"
    Demonstrations(
        grid=np.zeros((1, 1, 2, 2), np.float32),
        vector=np.zeros((1, 3), np.float32),
        mask=np.ones((1, 4), bool),
        action=np.zeros(1, np.int64),
        value=np.zeros(1, np.float32),
        episodes=1, proposer="policy:deadbeef@t0p9",
        meta='{"min_spread_fallback_rate": 0.5}').save(path)
    loaded = Demonstrations.load(path.with_suffix(".npz"))
    assert loaded.proposer == "policy:deadbeef@t0p9"
    assert json.loads(loaded.meta)["min_spread_fallback_rate"] == 0.5


def test_a_collapsed_shard_is_written_but_refused_for_merging():
    """A diagnostic that throws the data away stops being run.

    Twenty minutes of collection is not discarded because a rate tripped; the
    run says plainly what it measured and that the shard must not be merged.
    """
    from scripts.make_demos import collapse_refusal

    def _demos(rate):
        return Demonstrations(
            grid=np.zeros((1, 1, 2, 2), np.float32),
            vector=np.zeros((1, 3), np.float32),
            mask=np.ones((1, 4), bool),
            action=np.zeros(1, np.int64),
            value=np.zeros(1, np.float32),
            meta=json.dumps({"min_spread_fallback_rate": rate}))

    assert collapse_refusal(_demos(0.05), 0.0) == ""
    assert collapse_refusal(_demos(0.10), 0.0) == ""
    refusal = collapse_refusal(_demos(0.42), 0.0)
    assert "REFUSED" in refusal and "42.0%" in refusal


def test_a_proposer_is_named_by_its_weights_and_not_by_its_path(tmp_path):
    """``runs/iter-2/cloned.pt`` is a different network week to week.

    Expert iteration overwrites its checkpoint every round, so a shard naming
    the path would merge cleanly with one collected from the file that used to
    be there -- and the merge guard would agree, because the strings match.
    """
    first, second = tmp_path / "a.pt", tmp_path / "b.pt"
    first.write_bytes(b"weights one")
    second.write_bytes(b"weights two")
    assert proposer_identity(None) == "random"
    assert proposer_identity(first, temperature=0.0, policy_candidates=9) \
        != proposer_identity(second, temperature=0.0, policy_candidates=9)
    assert proposer_identity(first, temperature=0.0, policy_candidates=9) \
        .endswith("@t0p9")
    # The same weights under a different name are the same proposer.
    third = tmp_path / "c.pt"
    third.write_bytes(b"weights one")
    assert proposer_identity(third) == proposer_identity(first)
