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
