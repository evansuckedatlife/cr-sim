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
    # The row only; _target_row also returns the candidates' spread, which is
    # the collapse diagnostic a policy-proposed shard is gated on and which
    # these cases are not about.
    return _target_row(scores, index, width, height, slots,
                       _expert_patience(expert),
                       kwargs.pop("temperature", 0.35),
                       kwargs.pop("min_spread", 1e-3), NUM_ACTIONS)[0]


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


# ------------------------------------------- a demonstration set states what it is


def _demo(observation="v1", reward="projected", channels=9, rows=3):
    import numpy as np
    from cr_sim.train.clone import Demonstrations
    return Demonstrations(
        grid=np.zeros((rows, channels, 32, 18), dtype=np.float32),
        vector=np.zeros((rows, 102), dtype=np.float32),
        mask=np.ones((rows, 720), dtype=bool),
        action=np.zeros(rows, dtype=np.int64),
        value=np.zeros(rows, dtype=np.float32),
        episodes=1, play_rate=0.5,
        observation=observation, reward=reward)


def test_a_shard_records_the_encoding_and_reward_it_was_written_under(tmp_path):
    """--observation used to be a claim about a file, not a fact read from it.

    Most mismatches die on the channel count, but two variants of equal width
    and different meaning train quietly, stamp the wrong name onto the
    checkpoint, and then every run trusts it -- check_observation compares a
    shape, so it agrees.
    """
    from cr_sim.train.clone import Demonstrations

    path = tmp_path / "shard-00.npz"
    _demo(observation="v3", reward="projected").save(path)
    back = Demonstrations.load(path)
    assert back.observation == "v3"
    assert back.reward == "projected"


def test_a_shard_written_before_provenance_reports_empty_not_v1(tmp_path):
    """An unstamped shard is unknown, which is not the same as v1.

    Defaulting it to "v1" would make every legacy set assert something nobody
    recorded, and the check downstream would then pass on a guess.
    """
    import numpy as np
    from cr_sim.train.clone import Demonstrations

    path = tmp_path / "shard-00.npz"
    np.savez_compressed(
        path,
        grid=np.zeros((2, 9, 32, 18), dtype=np.float32),
        vector=np.zeros((2, 102), dtype=np.float32),
        mask=np.ones((2, 720), dtype=bool),
        action=np.zeros(2, dtype=np.int64),
        value=np.zeros(2, dtype=np.float32),
        episodes=1, play_rate=0.5)
    back = Demonstrations.load(path)
    assert back.observation == "", "an unstamped shard must not claim an encoding"
    assert back.reward == ""


def test_merging_shards_recorded_under_different_encodings_is_refused(tmp_path):
    """The row-to-row hazard nothing downstream could ever detect."""
    import pytest as _pytest
    from scripts.clone_policy import merge

    a, b = tmp_path / "shard-00.npz", tmp_path / "shard-01.npz"
    _demo(observation="v1").save(a)
    _demo(observation="spells").save(b)
    with _pytest.raises(SystemExit) as caught:
        merge([a, b])
    message = str(caught.value)
    assert "observation" in message
    assert "shard-00.npz" in message and "shard-01.npz" in message


def test_merging_a_stamped_shard_with_an_unstamped_one_is_refused(tmp_path):
    """Mixing a known encoding with an unknown one is the same hazard."""
    import numpy as np
    import pytest as _pytest
    from scripts.clone_policy import merge

    a, b = tmp_path / "shard-00.npz", tmp_path / "shard-01.npz"
    _demo(observation="v3").save(a)
    np.savez_compressed(
        b,
        grid=np.zeros((2, 9, 32, 18), dtype=np.float32),
        vector=np.zeros((2, 102), dtype=np.float32),
        mask=np.ones((2, 720), dtype=bool),
        action=np.zeros(2, dtype=np.int64),
        value=np.zeros(2, dtype=np.float32),
        episodes=1, play_rate=0.5)
    with _pytest.raises(SystemExit):
        merge([a, b])


def test_shards_that_agree_merge_and_carry_their_provenance(tmp_path):
    from scripts.clone_policy import merge

    a, b = tmp_path / "shard-00.npz", tmp_path / "shard-01.npz"
    _demo(observation="v3", reward="projected").save(a)
    _demo(observation="v3", reward="projected").save(b)
    merged = merge([a, b])
    assert merged.observation == "v3"
    assert merged.reward == "projected"
    assert len(merged) == 6


def test_taking_a_fraction_of_a_set_does_not_blank_its_provenance():
    """Selecting rows must not change what the rows are.

    ``subset`` rebuilt the dataclass field by field and omitted these two, so
    every ``--fraction`` run reported "these shards record no observation" and
    skipped the one check that stops a set recorded under one encoding being
    trained as another. The rows it returns came out of a file that knew;
    losing that on the way through is worse than never having had it, because
    the warning it triggers reads as a property of the corpus.
    """
    from scripts.clone_policy import subset

    data = _demo(observation="v3", reward="projected", rows=10)
    sliced = subset(data, 0.5, seed=0)
    assert len(sliced) == 5
    assert sliced.observation == "v3"
    assert sliced.reward == "projected"


def test_a_fraction_run_still_refuses_a_mismatched_encoding(tmp_path, capsys):
    """The guard the blanking switched off, exercised end to end.

    Asserting on ``subset``'s fields alone would stay green if ``main`` ever
    stopped consulting them, and it is ``main``'s refusal that is the thing
    worth having.
    """
    import pytest as _pytest
    from scripts.clone_policy import main

    _demo(observation="v3", reward="projected", rows=10).save(
        tmp_path / "shard-00.npz")
    with _pytest.raises(SystemExit) as caught:
        main(["--demos", str(tmp_path), "--out", str(tmp_path / "out"),
              "--observation", "v1", "--fraction", "0.5", "--epochs", "0",
              "--episodes", "0"])
    assert "recorded under 'v3'" in str(caught.value)


def test_collect_stamps_names_not_the_loop_variables_that_shadow_them(tmp_path):
    """collect's step loop rebinds both names, once per step.

    `observation, reward, terminated, truncated, _ = env.step(choice)` means a
    parameter called `reward` or `observation` is silently overwritten before
    the payload is built. The first version of this stamped the last step's
    float into every shard's reward, and an actual observation dict into its
    encoding name -- which numpy wrote as an object array that would not load
    back without allow_pickle. A provenance field that round-trips perfectly
    while carrying the wrong kind of thing is this codebase's signature bug.
    """
    import inspect

    from cr_sim.train import clone as clone_module

    signature = inspect.signature(clone_module.collect)
    for shadowed in ("reward", "observation"):
        assert shadowed not in signature.parameters, (
            f"`{shadowed}` is rebound by collect's step loop; a parameter of "
            "that name is overwritten before it is ever stamped")
        assert f"{shadowed}_name" in signature.parameters

    source = inspect.getsource(clone_module.collect)
    assert "reward=reward_name" in source
    assert "observation=observation_name" in source


def test_collect_actually_stamps_what_it_was_told(tmp_path):
    """The behavioural half: the source check above cannot see a wrong value.

    Runs the real collect over the module's toy environment and reads the
    stamps back off disk, which is what caught the object-array failure.
    """
    import numpy as np

    from cr_sim.train.clone import Demonstrations

    data = collect(lambda i: _Env(), _expert, episodes=2,
                   reward_name="projected", observation_name="v3")
    assert data.observation == "v3"
    assert data.reward == "projected"

    path = tmp_path / "stamped.npz"
    data.save(path)
    raw = np.load(path)  # allow_pickle=False: an object array fails here
    assert raw["observation"].dtype != object
    assert raw["reward"].dtype != object
    back = Demonstrations.load(path)
    assert back.observation == "v3"
    assert back.reward == "projected"


def test_both_collect_paths_stamp_provenance(tmp_path):
    """The single-variant and variants paths build their payloads separately.

    They have drifted before -- `merge` once dropped `target` by omitting one
    keyword -- so both are pinned rather than one standing in for both.
    """
    import inspect

    from cr_sim.train import clone as clone_module

    plain = inspect.getsource(clone_module.collect)
    variants = inspect.getsource(clone_module._collect_variants)
    assert "reward=reward_name" in plain, "single-variant payload"
    assert "reward=reward_name" in variants, "variants payload"
    assert "observation=observation_name," in plain, "single-variant path"
    assert "observation=name," in variants, "variants path uses the variant key"
    assert "reward_name" in inspect.signature(
        clone_module._collect_variants).parameters, (
        "the helper must receive it; collect's local is not in its scope")
