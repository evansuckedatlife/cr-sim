"""Learning to play by copying something that already can.

The step this project skipped. AlphaStar trained on 971,000 human replays
before any reinforcement learning, and that supervised agent alone outranked
84% of human players. Starting from random initialisation is what failed here:
a Clash Royale push is a conjunction -- tank, then support, timed, on one lane
-- that random placement essentially never produces, and no reward can
reinforce behaviour that never occurs.

What these tests protect is the part that is easy to get quietly wrong: which
states are worth learning from, whether the value head is trained at all, and
whether the policy learned the expert or memorised it.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cr_sim.train.clone import CloneConfig, Demonstrations, clone, collect
from cr_sim.train.nets import ActorCritic, NetConfig

NVEC = (5, 3, 4)
NUM_ACTIONS = NVEC[0] * NVEC[1] * NVEC[2]


class _Space:
    def __init__(self, nvec):
        self.nvec = np.asarray(nvec, dtype=np.int64)


class _Env:
    """A stand-in with a scripted mask pattern, so what gets kept is knowable.

    Real battles make the interesting case rare: most states have exactly one
    legal action, and a dataset that included them would be almost entirely
    the pass action.
    """

    def __init__(self, forced_steps=2, choice_steps=3):
        self.action_space = _Space(NVEC)
        self.forced_steps = forced_steps
        self.choice_steps = choice_steps
        self.battle = None
        self._step = 0

    def _obs(self):
        return {"grid": np.full((2, 4, 3), self._step, dtype=np.float32),
                "vector": np.full(6, self._step, dtype=np.float32)}

    def reset(self, seed=None):
        self._step = 0
        return self._obs(), {}

    def legal_action_mask(self):
        mask = np.zeros(NVEC, dtype=bool)
        if self._step < self.forced_steps:
            mask[4, 0, 0] = True          # nothing to decide
        else:
            mask[4, 0, 0] = True
            mask[0, 1, 2] = True
            mask[1, 0, 1] = True
        return mask

    def step(self, action):
        self._step += 1
        done = self._step >= self.forced_steps + self.choice_steps
        return self._obs(), 1.0, done, False, {}


def _expert(_env):
    def choose(observation, mask, battle=None):
        legal = np.argwhere(mask)
        playable = [a for a in legal if int(a[0]) != 4]
        return tuple(int(v) for v in (playable[0] if playable else legal[0]))
    return choose


def _net():
    return ActorCritic(NetConfig(
        grid_channels=2, grid_height=4, grid_width=3,
        vector_size=6, num_actions=NUM_ACTIONS))


# ------------------------------------------------------------------ collecting


class _UnevenEnv:
    """Choices arriving at irregular intervals, which is how a match delivers them.

    ``choices`` lists the env steps where more than one action is legal --
    everywhere else the only move is to pass. Real play looks like this: how
    long a player stays broke depends on what was just spent, so the decisions
    worth recording are sparse and the gaps between them are not equal.
    """

    def __init__(self, choices=(0, 1, 5, 7), rewards=(1., 2., 3., 4., 5., 6., 7., 8.)):
        self.action_space = _Space(NVEC)
        self.choices = set(choices)
        self.rewards = tuple(rewards)
        self.battle = None
        self._step = 0

    def _obs(self):
        return {"grid": np.full((2, 4, 3), self._step, dtype=np.float32),
                "vector": np.full(6, self._step, dtype=np.float32)}

    def reset(self, seed=None):
        self._step = 0
        return self._obs(), {}

    def legal_action_mask(self):
        mask = np.zeros(NVEC, dtype=bool)
        mask[4, 0, 0] = True
        if self._step in self.choices:
            mask[0, 1, 2] = True
        return mask

    def step(self, action):
        reward = self.rewards[self._step]
        self._step += 1
        return self._obs(), reward, self._step >= len(self.rewards), False, {}


def _returns(rewards, gamma):
    running = 0.0
    tail = []
    for reward in reversed(rewards):
        running = reward + gamma * running
        tail.append(running)
    tail.reverse()
    return tail


def test_only_states_with_a_real_choice_are_recorded():
    """A state with one legal action teaches nothing.

    The expert had no alternative there, so copying it conveys no preference
    -- and since a player is broke for most of a match, including those states
    would make the dataset almost entirely the pass action and teach the
    policy to pass.
    """
    data = collect(lambda i: _Env(forced_steps=2, choice_steps=3),
                   _expert, episodes=1)
    assert len(data) == 3, "forced states were recorded"
    assert data.episodes == 1


def test_the_recorded_action_is_the_one_the_expert_took():
    data = collect(lambda i: _Env(), _expert, episodes=1)
    slots, width, height = NVEC
    for index in data.action:
        slot, remainder = divmod(int(index), width * height)
        gx, gy = divmod(remainder, height)
        assert (slot, gx, gy) in {(0, 1, 2), (1, 0, 1)}


def test_the_mask_is_kept_with_each_state():
    """Without it the policy would be trained against actions that were never
    available, and cross-entropy would push probability onto illegal moves."""
    data = collect(lambda i: _Env(), _expert, episodes=1)
    assert data.mask.shape == (len(data), NUM_ACTIONS)
    for row, index in zip(data.mask, data.action):
        assert row[index], "the expert's own action was masked out"


def test_the_play_rate_is_recorded():
    """An expert that mostly passes teaches a policy to mostly pass, and this
    environment rewards that -- passing is the one action never punished."""
    data = collect(lambda i: _Env(), _expert, episodes=1)
    assert data.play_rate == 1.0


def test_values_are_discounted_returns_and_align_with_the_states():
    data = collect(lambda i: _Env(), _expert, episodes=1, gamma=0.5)
    assert len(data.value) == len(data.action)
    # Every reward is 1.0, so returns are positive and fall toward the end.
    assert (data.value > 0).all()
    assert data.value[0] > data.value[-1]


def test_each_value_is_the_return_at_the_step_it_was_recorded_at():
    """The value target belongs to the state it is stored beside.

    Kept decisions are sparse and unevenly spaced -- only states with more
    than one legal action are recorded -- so the i-th kept state is not the
    i-th step of the episode. Handing it the return of an evenly strided step
    instead, which is what this did, gives the value head a target from a
    nearby but different position, and a critic fitted to a smeared target is
    the one thing the clone exists to avoid producing.
    """
    rewards = (1., 2., 3., 4., 5., 6., 7., 8.)
    choices = (0, 1, 5, 7)
    data = collect(lambda i: _UnevenEnv(choices, rewards), _expert,
                   episodes=1, gamma=0.5)

    # gamma and the rewards are chosen so every return is exact in binary.
    tail = _returns(rewards, 0.5)
    assert tail == [3.921875, 5.84375, 7.6875, 9.375, 10.75, 11.5, 11.0, 8.0]

    assert len(data) == len(choices)
    assert data.value.tolist() == [3.921875, 5.84375, 11.5, 8.0]

    # What the even stride produced instead: eight steps over four kept
    # decisions strided by two, so three of the four targets came from a step
    # the expert was never asked about.
    strided = [tail[i * (len(rewards) // len(choices))] for i in range(len(choices))]
    assert strided == [3.921875, 7.6875, 10.75, 11.0]
    assert sum(a != b for a, b in zip(strided, data.value.tolist())) == 3


def test_the_decision_index_restarts_with_each_episode():
    """It is a position within an episode, not within the dataset -- and the
    returns it indexes are rebuilt per episode."""
    rewards = (1., 2., 3., 4., 5., 6., 7., 8.)
    choices = (0, 1, 5, 7)
    data = collect(lambda i: _UnevenEnv(choices, rewards), _expert,
                   episodes=3, gamma=0.5)
    assert data.value.tolist() == [3.921875, 5.84375, 11.5, 8.0] * 3


def test_variant_collection_indexes_the_returns_the_same_way(monkeypatch):
    """The second copy of the same loop. ``collect`` re-encodes one
    playthrough per observation variant rather than replaying it, and that
    copy carried the same striding -- so an encoding ablation would have
    compared two networks fitted to the same misplaced targets."""
    from cr_sim.api import encoding

    monkeypatch.setattr(encoding, "build_encoding_config",
                        lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(
        encoding, "encode_observation",
        lambda battle, team, registry, config: battle._obs(), raising=False)

    def make_env(index):
        env = _UnevenEnv((0, 1, 5, 7))
        # The variant path encodes off the battle, not the observation.
        env.battle = env
        env.arena = env.team = env.registry = None
        env.blue_deck = env.red_deck = None
        return env

    out = collect(make_env, _expert, episodes=1, gamma=0.5,
                  variants={"a": None, "b": None})
    assert set(out) == {"a", "b"}
    for demos in out.values():
        assert demos.value.tolist() == [3.921875, 5.84375, 11.5, 8.0]


def test_several_episodes_accumulate():
    one = collect(lambda i: _Env(), _expert, episodes=1)
    three = collect(lambda i: _Env(), _expert, episodes=3)
    assert len(three) == 3 * len(one)
    assert three.episodes == 3


def test_a_dataset_survives_a_round_trip(tmp_path):
    data = collect(lambda i: _Env(), _expert, episodes=2)
    path = tmp_path / "demo.npz"
    data.save(path)
    back = Demonstrations.load(path)
    assert len(back) == len(data)
    assert np.array_equal(back.action, data.action)
    assert back.play_rate == data.play_rate


# -------------------------------------------------------------------- cloning


def test_cloning_teaches_the_policy_the_expert_s_choice():
    """The whole point: afterwards the network should pick what the expert
    picked, on states it was not trained on."""
    data = collect(lambda i: _Env(), _expert, episodes=40)
    net = _net()
    seen = []
    clone(net, data, CloneConfig(epochs=12, batch_size=64, seed=0),
          on_epoch=seen.append)
    assert seen[-1]["agreement"] > 0.9, (
        f"agreement only reached {seen[-1]['agreement']:.2f}")


def test_cloning_trains_the_value_head_too():
    """Skipping it hands reinforcement learning a critic that predicts
    nothing, which is exactly what stalled every run on this project --
    explained variance sat at 0.00 and the advantages were noise."""
    data = collect(lambda i: _Env(), _expert, episodes=40)
    net = _net()
    seen = []
    clone(net, data, CloneConfig(epochs=12, batch_size=64, seed=0),
          on_epoch=seen.append)
    assert seen[-1]["value_loss"] < seen[0]["value_loss"], "the critic did not fit"


def test_agreement_is_measured_on_states_that_were_held_back():
    """Otherwise it reports memorisation. Training loss falls either way and
    says nothing about whether the expert was learned."""
    data = collect(lambda i: _Env(), _expert, episodes=20)
    net = _net()
    seen = []
    clone(net, data, CloneConfig(epochs=2, batch_size=32, holdout=0.5, seed=0),
          on_epoch=seen.append)
    assert 0.0 <= seen[-1]["agreement"] <= 1.0
    assert seen[-1]["epoch"] == 2


# ------------------------------------------------- what the target says


class _ScoringExpert:
    """An expert that publishes candidate values the way the search bot does."""

    def __init__(self, scores, choice, patience=0.01):
        self.last_scores = scores
        self._choice = choice

        class _Config:
            pass

        self.config = _Config()
        self.config.patience = patience

    def __call__(self, observation, mask, battle=None):
        return self._choice


def _row(scores, choice, **kwargs):
    from cr_sim.train.clone import _expert_patience, _target_row

    expert = _ScoringExpert(scores, choice, kwargs.pop("patience", 0.01))
    slots, width, height = NVEC
    index = (choice[0] * width * height + choice[1] * height + choice[2])
    return _target_row(scores, index, width, height, slots,
                       _expert_patience(expert),
                       kwargs.pop("temperature", 0.35),
                       kwargs.pop("min_spread", 1e-3), NUM_ACTIONS)


def test_a_target_the_search_could_not_separate_falls_back_to_what_it_did():
    """The bug this replaces, in one case.

    The softmax is scaled by the candidates' own spread, so it is scale-free:
    values equal to four decimal places produce a *confident* preference for
    whichever one rounded highest, and exactly equal values produce a uniform
    distribution over about fifteen candidates of which fourteen are
    placements. Measured on the recorded demonstrations, 86% of the states
    where the expert waited carried an exactly uniform target and the pass
    action was the argmax in none of 10,940 rows -- so a policy trained on
    them played a card at every decision, against the expert's 56%.
    """
    pass_index = (NVEC[0] - 1) * NVEC[1] * NVEC[2]
    flat = [(pass_index, 0.5), (7, 0.5), (19, 0.5), (23, 0.5)]

    # With no margin at all the values are exactly equal, there is nothing to
    # prefer, and the only signal left is the action the expert took.
    row = _row(flat, (4, 0, 0), patience=0.0)
    assert row[pass_index] == 1.0
    row = _row(flat, (0, 1, 2), patience=0.0)
    assert row[0 * NVEC[1] * NVEC[2] + 1 * NVEC[2] + 2] == 1.0

    # With the expert's real margin, waiting is genuinely ahead and the
    # distribution says so rather than spreading itself over the placements.
    row = _row(flat, (4, 0, 0))
    assert row.argmax() == pass_index
    assert row[pass_index] > 0.9


def test_waiting_is_scored_with_the_margin_the_expert_required():
    """A play is only taken if it beats waiting by ``patience``. If the
    recorded scores leave that margin out, a state where waiting won produces
    a target whose largest entry is the placement that lost."""
    pass_index = (NVEC[0] - 1) * NVEC[1] * NVEC[2]
    # A placement beats waiting by 0.005, which is inside the 0.01 margin, so
    # the expert waited. The target has to agree with it.
    scores = [(pass_index, 1.000), (7, 1.005), (19, 0.900), (23, 0.700)]
    row = _row(scores, (4, 0, 0))
    assert row.argmax() == pass_index, (
        "the target prefers a placement the expert declined")

    # Outside the margin, the placement wins and the target should say so.
    scores = [(pass_index, 1.000), (7, 1.200), (19, 0.900), (23, 0.700)]
    row = _row(scores, (0, 0, 7))
    assert row.argmax() == 7


def test_a_real_spread_still_produces_a_distribution_not_a_label():
    """The soft target's whole point: several placements were nearly as good,
    which a one-hot cannot say."""
    pass_index = (NVEC[0] - 1) * NVEC[1] * NVEC[2]
    scores = [(pass_index, 0.0), (7, 1.0), (19, 0.95), (23, 0.1)]
    row = _row(scores, (0, 0, 7))
    assert row.argmax() == 7
    assert 0.0 < row[19] < row[7], "the near-equivalent placement carries no mass"
    assert int((row > 0).sum()) == 4


def test_the_expert_clears_its_scores_when_it_declines_to_search():
    """``last_scores`` used to keep the previous decision's numbers through
    the early returns, so a reader would be handed the search's beliefs about
    a board that no longer exists."""
    from cr_sim.engine.entity import Team
    from cr_sim.train.scripted import SearchBot, SearchBotConfig

    bot = SearchBot(Team.BLUE, SearchBotConfig())
    bot.last_scores = [(1, 2.0)]
    assert bot(None, np.zeros(NVEC, dtype=bool), None) == (4, 0, 0)
    assert bot.last_scores == []


# ------------------------------------------ what the cloner is fitting


def test_merging_shards_keeps_the_search_s_own_beliefs(tmp_path):
    """The cloner branches on ``target``, the search's value distribution over
    every placement it evaluated, and falls back to the single move played
    when it is absent. ``merge`` used to drop it by omitting one keyword, so
    every clone this project ever ran took the fallback -- while the loader,
    the saver and the collector all handled it correctly and every test passed.
    """
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from scripts.clone_policy import merge

    def shard(index: int) -> Demonstrations:
        rows = 4
        target = np.zeros((rows, NUM_ACTIONS), dtype=np.float32)
        target[:, index] = 1.0
        return Demonstrations(
            grid=np.zeros((rows, 2, 4, 3), dtype=np.float32),
            vector=np.zeros((rows, 6), dtype=np.float32),
            mask=np.ones((rows, NUM_ACTIONS), dtype=bool),
            action=np.zeros(rows, dtype=np.int64),
            value=np.zeros(rows, dtype=np.float32),
            target=target, episodes=1, play_rate=0.5)

    paths = []
    for index in (3, 7):
        path = tmp_path / f"shard-{index:02d}.npz"
        shard(index).save(path)
        paths.append(path)

    merged = merge(paths)
    assert merged.target is not None, "the search's beliefs were dropped"
    assert merged.target.shape == (8, NUM_ACTIONS)
    assert merged.target[0].argmax() == 3
    assert merged.target[-1].argmax() == 7


def test_the_soft_target_is_what_the_cloner_actually_fits():
    """Not the recorded action. The two disagree here on purpose: if the
    target were being ignored the policy would learn the action instead, and
    nothing else would look different.
    """
    from cr_sim.train.clone import CloneConfig, clone

    rows = 64
    wanted = 5
    recorded = 9
    target = np.zeros((rows, NUM_ACTIONS), dtype=np.float32)
    target[:, wanted] = 1.0
    data = Demonstrations(
        grid=np.zeros((rows, 2, 4, 3), dtype=np.float32),
        vector=np.zeros((rows, 6), dtype=np.float32),
        mask=np.ones((rows, NUM_ACTIONS), dtype=bool),
        action=np.full(rows, recorded, dtype=np.int64),
        value=np.zeros(rows, dtype=np.float32),
        target=target, episodes=1, play_rate=1.0)

    net = _net()
    clone(net, data, CloneConfig(epochs=15, batch_size=32, learning_rate=5e-3,
                                holdout=0.25, seed=0))
    with torch.no_grad():
        logits, _ = net(torch.zeros(1, 2, 4, 3), torch.zeros(1, 6),
                        torch.ones(1, NUM_ACTIONS, dtype=torch.bool))
    assert int(logits.argmax()) == wanted, (
        "the cloner fitted the recorded action, not the search's distribution")
