"""The training pipeline.

These are not convergence tests -- nothing here asserts that the agent gets
good, which would need far more compute than a test suite should take. They
assert the things that go wrong *silently* in an RL pipeline and are otherwise
discovered as "the run didn't learn anything" several hours later:

* a masked action being sampled anyway,
* an action index decoded to a different cell than the one it was scored at,
* the advantage bootstrapping across an episode boundary,
* NaN reaching the optimiser.

Each of those produces a training run that looks healthy on every metric and
learns nothing.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cr_sim.api.encoding import NOOP_SLOT  # noqa: E402
from cr_sim.api.env import CRSimEnv  # noqa: E402
from cr_sim.data.cards import build_card_registry  # noqa: E402
from cr_sim.data.leveling import build_level_table  # noqa: E402
from cr_sim.data.source import LogicData  # noqa: E402
from cr_sim.train.nets import ActorCritic, NetConfig, masked_categorical  # noqa: E402
from cr_sim.train.ppo import PPOConfig, _unflatten_action, compute_gae, train  # noqa: E402

from .test_data_pipeline import BUILD  # noqa: E402

DECK = ("Knight", "Musketeer", "Cannon", "Skeletons", "IceSpirits", "Log", "Fireball", "Goblins")


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


def _env(world, **kwargs):
    data, levels, registry = world
    kwargs.setdefault("ticks_per_second", 20)
    kwargs.setdefault("frame_skip", 20)
    kwargs.setdefault("max_ticks", 20 * 40)
    return CRSimEnv(data, levels, registry, DECK, DECK, **kwargs)


# ------------------------------------------------------------------ masking


def test_a_masked_action_is_never_sampled(world):
    """The mask has to bind at the distribution, not merely be advisory.

    A policy that can still sample illegal actions wastes most of its rollout
    on rejected placements, and the rejections look like ordinary bad play.
    """
    env = _env(world)
    obs, _ = env.reset(seed=1)
    mask = torch.from_numpy(env.legal_action_mask().reshape(1, -1))
    logits = torch.randn(1, mask.shape[1])

    distribution = masked_categorical(logits, mask)
    samples = distribution.sample((512,)).reshape(-1)
    assert mask[0, samples].all(), "sampled an action the mask forbade"
    assert torch.isfinite(distribution.entropy()).all()


def test_masked_logits_carry_no_probability(world):
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, 3] = True
    probabilities = masked_categorical(torch.zeros(1, 8), mask).probs
    assert probabilities[0, 3] == pytest.approx(1.0)
    assert probabilities[0, [0, 1, 2, 4, 5, 6, 7]].sum() == pytest.approx(0.0)


def test_an_all_masked_row_is_caught_rather_than_producing_nan():
    """A NaN here corrupts every weight in the network and does it quietly."""
    with pytest.raises(AssertionError):
        masked_categorical(torch.zeros(1, 4), torch.zeros(1, 4, dtype=torch.bool))


def test_passing_is_exactly_one_action(world):
    """Not 144 duplicates of it.

    Every (NOOP, x, y) decodes to the same no-op, so marking the whole slot
    legal spends a fifth of the policy's output on copies of one action, and
    hands "do nothing" a fifth of the probability mass before training starts.
    """
    env = _env(world)
    env.reset(seed=1)
    mask = env.legal_action_mask()
    assert mask[NOOP_SLOT].sum() == 1


# ----------------------------------------------------------------- decoding


def test_a_flat_action_decodes_to_the_cell_it_was_scored_at(world):
    """The categorical is flat; the environment wants ``(slot, x, y)``.

    The flatten and the unflatten have to agree axis for axis. A transposed
    decode is still a valid-looking action, so it never raises -- it just
    places every card somewhere other than where the policy chose.
    """
    env = _env(world)
    env.reset(seed=1)
    nvec = [int(v) for v in env.action_space.nvec]
    mask = env.legal_action_mask()

    flat = mask.reshape(-1)
    for index in np.flatnonzero(flat):
        slot, gx, gy = _unflatten_action(int(index), nvec)
        assert mask[slot, gx, gy], f"index {index} decoded to an illegal cell"


def test_every_flat_index_round_trips(world):
    env = _env(world)
    env.reset(seed=1)
    nvec = [int(v) for v in env.action_space.nvec]
    total = int(np.prod(nvec))
    seen = set()
    for index in range(total):
        seen.add(_unflatten_action(index, nvec))
    assert len(seen) == total, "the decode is not a bijection"


# --------------------------------------------------------------------- gae


def test_the_advantage_does_not_bootstrap_across_an_episode_boundary():
    """Otherwise the last step of a won match rewards the first step of the next.

    The policy would learn that whatever it did on an opening tick was worth a
    crown, which is both wrong and impossible to notice from the loss curves.
    """
    reward = torch.zeros(3, 1)
    value = torch.zeros(3, 1)
    done = torch.tensor([[0.0], [1.0], [0.0]])
    last_value = torch.tensor([100.0])

    advantage, _ = compute_gae(reward, value, done, last_value, gamma=0.99, lam=0.95)
    # Step 1 ends the episode, so nothing after it may reach step 0.
    assert advantage[1].item() == pytest.approx(0.0)
    assert advantage[0].item() == pytest.approx(0.0)
    # The final step still bootstraps, because it did not end an episode.
    assert advantage[2].item() > 0


def test_the_advantage_propagates_backwards_within_an_episode():
    reward = torch.zeros(3, 1)
    value = torch.zeros(3, 1)
    done = torch.zeros(3, 1)
    advantage, returns = compute_gae(
        reward, value, done, torch.tensor([1.0]), gamma=0.99, lam=0.95
    )
    assert advantage[2] > 0 and advantage[1] > 0 and advantage[0] > 0
    assert advantage[0] < advantage[2], "later steps are closer to the payoff"
    assert torch.allclose(returns, advantage + value)


# ---------------------------------------------------------------- the loop


def test_the_network_shapes_line_up_with_the_environment(world):
    env = _env(world)
    obs, _ = env.reset(seed=1)
    nvec = [int(v) for v in env.action_space.nvec]
    net = ActorCritic(
        NetConfig(
            grid_channels=obs["grid"].shape[0],
            grid_height=obs["grid"].shape[1],
            grid_width=obs["grid"].shape[2],
            vector_size=obs["vector"].shape[0],
            num_actions=int(np.prod(nvec)),
            channels=8,
            hidden=16,
        )
    )
    grid = torch.from_numpy(obs["grid"]).unsqueeze(0)
    vector = torch.from_numpy(obs["vector"]).unsqueeze(0)
    mask = torch.from_numpy(env.legal_action_mask().reshape(1, -1))

    logits, value = net(grid, vector, mask)
    assert logits.shape == (1, int(np.prod(nvec)))
    assert value.shape == (1,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_a_short_run_completes_and_produces_finite_weights(world):
    """The end-to-end guard: a run that finishes with NaN weights has failed
    even though every intermediate log line looked ordinary."""
    stats: list[dict] = []
    net = train(
        lambda index: _env(world),
        PPOConfig(total_steps=128, horizon=16, num_envs=2, epochs=1, minibatches=1, seed=0),
        on_update=stats.append,
    )
    assert stats, "no update ran"
    for parameter in net.parameters():
        assert torch.isfinite(parameter).all(), "training produced non-finite weights"
    for record in stats:
        for key in ("policy_loss", "value_loss", "entropy"):
            assert np.isfinite(record[key]), f"{key} was {record[key]}"
        assert 0.0 <= record["noop_fraction"] <= 1.0


# ------------------------------------------------------- forced decisions


def test_skipping_forced_decisions_removes_them_entirely(world):
    """A state with one legal action is not a decision.

    For most of a match the only thing available is to pass -- elixir is spent
    about as fast as it accrues. Handing those states to the policy costs a
    network evaluation each and produces a transition whose gradient is
    identically zero, because a one-action softmax cannot be wrong. It also
    dilutes every batch nine to one.
    """
    env = _env(world, skip_forced=True)
    env.reset(seed=1)
    rng = np.random.default_rng(0)

    steps = 0
    while True:
        mask = env.legal_action_mask()
        assert int(mask.sum()) > 1, "a forced decision reached the policy"
        legal = np.argwhere(mask)
        action = tuple(int(v) for v in legal[rng.integers(len(legal))])
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    assert steps > 0


def test_skipping_does_not_shorten_the_battle(world):
    """The same match is simulated either way; only the sampling changes.

    This is what makes it the same MDP rather than a different, easier game:
    the ticks all still happen, and taking the only available move cannot have
    gone differently.
    """
    rng = np.random.default_rng(0)
    ticks = {}
    for skip in (False, True):
        env = _env(world, skip_forced=skip)
        env.reset(seed=7)
        while True:
            legal = np.argwhere(env.legal_action_mask())
            action = tuple(int(v) for v in legal[rng.integers(len(legal))])
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        ticks[skip] = info["tick"]
    assert ticks[True] == ticks[False], f"{ticks} -- the match length changed"


def test_reward_during_a_run_out_is_not_lost(world):
    """What happens while running out still counts, it just is not a choice.

    Dropping it would make the agent blind to everything that happened between
    its decisions, which on a 500ms cadence is most of the match.
    """
    env = _env(world, skip_forced=True)
    env.reset(seed=2)
    rng = np.random.default_rng(1)
    total = 0.0
    while True:
        legal = np.argwhere(env.legal_action_mask())
        action = tuple(int(v) for v in legal[rng.integers(len(legal))])
        _, reward, terminated, truncated, _ = env.step(action)
        total += reward
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    assert total != 0.0, "a whole match produced exactly zero reward"


# --------------------------------------------------------------- metrics file


def test_each_update_writes_exactly_one_metrics_row(tmp_path):
    """One update, one line.

    The writer once emitted each update twice -- once on the way into the
    callback and again on the way out, after the evaluation had added its
    fields. Both rows carried the same update number, so a run read as two
    trainers racing on one file, and every average over the file counted each
    update twice.
    """
    from collections import Counter

    from cr_sim.train.run import main

    code = main([
        "--steps", "128", "--horizon", "16", "--envs", "2",
        "--match-seconds", "20", "--eval-every", "0", "--device", "cpu", "--save-every", "1000",
        "--opponent", "idle", "--out", str(tmp_path), "--name", "once",
    ])
    assert code == 0

    rows = [
        json.loads(line)
        for line in (tmp_path / "once" / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows, "the run wrote no metrics at all"
    repeated = {u: n for u, n in Counter(r["updates"] for r in rows).items() if n > 1}
    assert not repeated, f"updates written more than once: {repeated}"


# ------------------------------------------------------------------- resuming


def test_a_run_resumes_from_its_checkpoint_without_losing_progress(tmp_path):
    """A crash at hour four should not cost four hours.

    The machine this trains on has bugchecked twice in a day, so an
    unattended run needs to survive one. What makes that worth testing rather
    than assuming: a resume that silently restarts from zero steps, or that
    truncates the metrics file it was supposed to continue, looks exactly like
    a working resume until you read the numbers afterwards.
    """
    import json
    import torch

    from cr_sim.train.run import main

    common = [
        "--horizon", "16", "--envs", "2", "--match-seconds", "20",
        "--eval-every", "0", "--device", "cpu", "--save-every", "1", "--opponent", "idle",
        "--out", str(tmp_path), "--name", "resumed",
    ]
    assert main(["--steps", "128", *common]) == 0

    checkpoint = torch.load(
        tmp_path / "resumed" / "checkpoint.pt", map_location="cpu",
        weights_only=False)
    assert checkpoint["steps"] > 0, "checkpoint recorded no progress"
    assert checkpoint["optimiser"] is not None, "optimiser state was not saved"
    # Adam keeps per-parameter moments; an optimiser that has never stepped
    # saves an empty state, which would restart the adaptation from scratch.
    assert checkpoint["optimiser"]["state"], "optimiser state was empty"

    first_rows = [
        json.loads(line) for line in
        (tmp_path / "resumed" / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    steps_before = first_rows[-1]["steps"]

    assert main(["--steps", str(steps_before + 128), "--resume", *common]) == 0

    rows = [
        json.loads(line) for line in
        (tmp_path / "resumed" / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) > len(first_rows), "resuming truncated the earlier metrics"
    assert rows[len(first_rows)]["steps"] > steps_before, (
        "the resumed leg restarted the step count instead of continuing it")
    assert rows[len(first_rows)]["updates"] > first_rows[-1]["updates"], (
        "the resumed leg restarted the update count")


def test_resuming_without_a_checkpoint_fails_loudly(tmp_path):
    """Better than silently starting a fresh run under the same name and
    overwriting what someone was trying to recover."""
    from cr_sim.train.run import main

    assert main([
        "--steps", "64", "--horizon", "16", "--envs", "2",
        "--match-seconds", "20", "--eval-every", "0", "--device", "cpu", "--resume",
        "--out", str(tmp_path), "--name", "nothing-here",
    ]) == 1


def test_a_run_stops_when_the_gradients_go_non_finite(world, monkeypatch):
    """A non-finite gradient is caught at the update that produced it.

    The step after one leaves every weight NaN: rollouts then sample from a
    uniform distribution, checkpoints saved afterwards are worthless, and the
    metrics look like an ordinary bad run rather than a broken one.

    Checked on the norm that clip_grad_norm_ already returns, which is free.
    The obvious version -- sweeping every parameter with isfinite().all() --
    is thirty pairs of tiny kernels each followed by a blocking host readback,
    fired right after the optimiser queues a few hundred more, and that
    pattern exhausted the Level Zero driver on every GPU run attempted here.
    """
    import torch.nn as nn

    import cr_sim.train.ppo as ppo_module

    real_clip = nn.utils.clip_grad_norm_

    def poisoned(parameters, max_norm, *args, **kwargs):
        real_clip(parameters, max_norm, *args, **kwargs)
        return torch.tensor(float("nan"))

    monkeypatch.setattr(ppo_module.nn.utils, "clip_grad_norm_", poisoned)
    with pytest.raises(RuntimeError, match="non-finite"):
        ppo_module.train(
            lambda index: _env(world),
            PPOConfig(total_steps=256, horizon=16, num_envs=2, epochs=1,
                      minibatches=1, seed=0),
        )


def test_policy_only_inference_matches_the_full_forward(world):
    """The cheap inference path must be the same answer, not a similar one.

    ``policy_logits`` exists so that a caller with no use for the value does
    not pay for the critic encoder. That is only safe if it computes a strict
    subset of the same graph, so the two are compared element for element
    rather than approximately.
    """
    env = _env(world)
    obs, _ = env.reset(seed=1)
    nvec = [int(v) for v in env.action_space.nvec]
    net = ActorCritic(
        NetConfig(
            grid_channels=obs["grid"].shape[0],
            grid_height=obs["grid"].shape[1],
            grid_width=obs["grid"].shape[2],
            vector_size=obs["vector"].shape[0],
            num_actions=int(np.prod(nvec)),
            channels=8,
            hidden=16,
        )
    ).eval()
    grid = torch.from_numpy(obs["grid"]).unsqueeze(0)
    vector = torch.from_numpy(obs["vector"]).unsqueeze(0)
    mask = torch.from_numpy(env.legal_action_mask().reshape(1, -1))

    with torch.no_grad():
        full, _value = net(grid, vector, mask)
        cheap = net.policy_logits(grid, vector, mask)
    assert torch.equal(full, cheap)


def test_policy_only_inference_does_not_run_the_critic(world):
    """The saving is the point, so guard it rather than trusting the code.

    Without this, someone could implement ``policy_logits`` as ``forward()[0]``
    and every other test would still pass while the rollout workers went back
    to computing a value nobody reads -- 44% of every batch-of-one inference.
    It is waste rather than a bottleneck: an interleaved A/B over whole
    self-play battles came back at ~1.0x, because a decision is ~118ms and
    this forward is ~1.1ms of it.
    """
    env = _env(world)
    obs, _ = env.reset(seed=1)
    nvec = [int(v) for v in env.action_space.nvec]
    config = NetConfig(
        grid_channels=obs["grid"].shape[0],
        grid_height=obs["grid"].shape[1],
        grid_width=obs["grid"].shape[2],
        vector_size=obs["vector"].shape[0],
        num_actions=int(np.prod(nvec)),
        channels=8,
        hidden=16,
    )
    assert config.separate_critic, "the saving only exists with a separate critic"
    net = ActorCritic(config).eval()
    grid = torch.from_numpy(obs["grid"]).unsqueeze(0)
    vector = torch.from_numpy(obs["vector"]).unsqueeze(0)
    mask = torch.from_numpy(env.legal_action_mask().reshape(1, -1))

    calls: list[str] = []
    for name in ("critic_conv", "critic_vector", "critic_trunk", "value_head"):
        getattr(net, name).register_forward_hook(
            lambda _m, _i, _o, name=name: calls.append(name)
        )

    with torch.no_grad():
        net.policy_logits(grid, vector, mask)
    assert calls == [], f"policy_logits ran the critic: {calls}"

    with torch.no_grad():
        net(grid, vector, mask)
    assert calls, "forward should still produce a value"
# ------------------------------------------- naming the opponent a lift faced


def test_the_in_run_evaluation_faces_a_random_opponent(tmp_path):
    """The probe used to face an idle opponent -- one that never plays a card
    -- while every large verdict on this project faced a random one. Both were
    called "lift". This pins which one the run actually measures against, and
    that the run records it.
    """
    from cr_sim.train.run import main

    code = main([
        "--steps", "64", "--horizon", "8", "--envs", "2",
        "--match-seconds", "20", "--eval-every", "1", "--eval-episodes", "2",
        "--device", "cpu", "--save-every", "1000", "--opponent", "idle",
        "--out", str(tmp_path), "--name", "named",
    ])
    assert code == 0

    config = json.loads((tmp_path / "named" / "config.json").read_text())
    assert config["eval_opponent"] == "random"

    rows = [
        json.loads(line)
        for line in (tmp_path / "named" / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    lifts = [r for r in rows if "eval_lift_sd" in r]
    assert lifts, "the run evaluated but recorded no lift"
    assert all(r["eval_opponent"] == "random" for r in lifts)


# ------------------------------------- the workers play the game that was asked for


def test_the_workers_get_the_tower_level_the_run_was_launched_with(tmp_path,
                                                                   monkeypatch):
    """--tower-level reached the evaluation probe and not the workers.

    VecEnvConfig defaults tower_level to 11, and run.py built one without
    passing it, so any run with --workers trained every rollout at level 11
    while config.json recorded 5 and the probe measured at 5. At level 11 the
    towers outlast a 120-second match: ~90% of battles draw, crowns almost
    never fire, and the agent learns from shaping alone. Nothing in the suite
    mentioned tower_level, so nothing could have caught it.
    """
    from cr_sim.api import vec as vec_module

    captured = {}

    class _Stop(RuntimeError):
        pass

    def _capture(config, num_envs, workers):
        captured["config"] = config
        raise _Stop("captured")

    monkeypatch.setattr(vec_module, "CRSimVecEnv", _capture)

    from cr_sim.train.run import main

    with pytest.raises(_Stop):
        main([
            "--steps", "64", "--horizon", "8", "--envs", "2", "--workers", "1",
            "--match-seconds", "20", "--tower-level", "5", "--device", "cpu",
            "--out", str(tmp_path), "--name", "towers",
        ])

    config = captured["config"]
    assert config.tower_level == 5, (
        f"workers build tower_level={config.tower_level} for a run launched "
        "with --tower-level 5; they would train on a different game from the "
        "one the run records and evaluates on")


def test_the_worker_config_agrees_with_the_probe_env_field_for_field(tmp_path,
                                                                    monkeypatch):
    """The general form of the bug above, so the next dropped field is caught.

    run.py builds the same battle twice -- once as a local CRSimEnv it probes
    and evaluates with, once as a VecEnvConfig the workers run. Any field that
    disagrees means the run measures a different game from the one it trains
    on, silently.
    """
    from cr_sim.api import vec as vec_module

    captured = {}

    class _Stop(RuntimeError):
        pass

    def _capture(config, num_envs, workers):
        captured["config"] = config
        raise _Stop("captured")

    monkeypatch.setattr(vec_module, "CRSimVecEnv", _capture)

    from cr_sim.train.run import main

    argv = [
        "--steps", "64", "--horizon", "8", "--envs", "2", "--workers", "1",
        "--match-seconds", "20", "--tower-level", "5", "--tps", "20",
        "--frame-skip", "30", "--shaping", "0.02", "--device", "cpu",
        "--reward", "projected", "--tower-weight", "0.4",
        "--elixir-weight", "0.7", "--out", str(tmp_path), "--name", "agree",
    ]
    with pytest.raises(_Stop):
        main(argv)

    config = captured["config"]
    assert config.tower_level == 5
    assert config.ticks_per_second == 20
    assert config.frame_skip == 30
    assert config.max_ticks == 20 * 20
    assert config.reward_shaping_weight == pytest.approx(0.02)

    # The field that actually decides what the workers are paid, which is not
    # the one above. --shaping reaches VecEnvConfig faithfully and is then
    # never read: every _shaped_value call site sits inside the branch
    # `projected` does not take, and 0.01 against 5.00 is bit-identical under
    # it. Asserting only on that field's transit is a green test over a knob
    # the run ignores -- so assert on the weights the reward is actually built
    # from, and on the effect: the worker's env pays the reward these weights
    # define.
    from cr_sim.api.reward import ProjectionWeights
    from cr_sim.api.vec import _build_env
    from cr_sim.train.selfplay import reward_name

    assert config.reward_weights == ProjectionWeights(
        tower=0.4, elixir=0.7, horizon_seconds=3.0)

    from cr_sim.data.cards import build_card_registry
    from cr_sim.data.leveling import build_level_table
    from cr_sim.data.source import LogicData

    data = LogicData.load(config.build)
    worker_env = _build_env(config, data, build_level_table(data),
                            build_card_registry(data), 0)
    assert reward_name(worker_env) == (
        "projected:elixir=0.7,horizon_seconds=3,tower=0.4")


def _tiny_rollout(horizon=8, envs=4, actions=12, seed=0):
    """A batch big enough to have several distinct minibatch orders."""
    from cr_sim.train.ppo import Rollout, compute_gae

    generator = torch.Generator().manual_seed(seed)
    grid = torch.rand((horizon, envs, 2, 4, 3), generator=generator)
    vector = torch.rand((horizon, envs, 6), generator=generator)
    mask = torch.ones((horizon, envs, actions), dtype=torch.bool)
    action = torch.randint(0, actions, (horizon, envs), generator=generator)
    log_prob = torch.full((horizon, envs), -float(np.log(actions)))
    value = torch.zeros((horizon, envs))
    done = torch.zeros((horizon, envs))
    advantage = torch.rand((horizon, envs), generator=generator) - 0.5
    reward = advantage.clone()
    _, ret = compute_gae(reward, value, done, torch.zeros(envs), 0.99, 0.95)
    return Rollout(grid=grid, vector=vector, mask=mask, action=action,
                   log_prob=log_prob, value=value, reward=reward, done=done,
                   advantage=advantage, ret=ret)


class _RecordingShuffler:
    """A shuffler that keeps every permutation it produced."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self.orders: list[list[int]] = []

    def shuffle(self, indices) -> None:
        self._rng.shuffle(indices)
        self.orders.append([int(i) for i in indices])


def test_the_minibatch_order_comes_from_a_stream_the_update_owns():
    """``np.random.shuffle`` draws from numpy's *global legacy* RandomState.

    ``train`` seeds torch and builds a local ``default_rng``; nothing seeds
    that global RandomState, which is initialised from OS entropy at import.
    Three fresh processes doing exactly what ``train`` does first and then one
    ``np.random.shuffle(np.arange(12))`` produced [6,5,1,7,3,0,11,9,10,4,8,2],
    [10,2,1,11,3,7,6,0,9,4,5,8] and [3,11,7,9,1,2,4,10,5,6,0,8].

    So a run reproduced its rollout and not its update: at ``--workers 0``
    with byte-identical config.json, two runs of one command agreed on
    update-1 mean_return to the last digit and disagreed on policy_loss
    (-0.080 against -0.121) and value_loss (0.056 against 0.092), and by
    update 2 the entropy and the explained variance had parted company.
    """
    from cr_sim.train.ppo import PPOConfig, _update

    def two_updates(shuffler):
        torch.manual_seed(0)
        net = ActorCritic(NetConfig(
            grid_channels=2, grid_height=4, grid_width=3,
            vector_size=6, num_actions=12))
        optimiser = torch.optim.Adam(net.parameters(), lr=1e-3)
        rollout = _tiny_rollout()
        config = PPOConfig(epochs=2, minibatches=2, entropy_coefficient=0.0)
        stats = [_update(net, optimiser, rollout, config, "cpu", None,
                         shuffler=shuffler) for _ in range(2)]
        return stats

    first, second = _RecordingShuffler(0), _RecordingShuffler(0)
    stats_a, stats_b = two_updates(first), two_updates(second)

    # The update really drew its order from the object it was handed. Without
    # this the equality below holds vacuously over an update that shuffles
    # from somewhere else entirely.
    assert len(first.orders) == 4, "the update did not shuffle through its own stream"
    assert len({tuple(o) for o in first.orders}) > 1, (
        "every epoch saw the identical order, so the shuffle is inert")

    assert first.orders == second.orders
    assert stats_a == stats_b


def test_two_runs_of_one_seed_take_the_same_update_path(world):
    """The end-to-end form: same command, same seed, same numbers.

    Two ``train`` calls in one process, which is the case the global
    RandomState fails: the first advances it and the second starts wherever
    the first stopped, so the losses diverge while the rollout does not.
    """
    def run() -> list[dict]:
        stats: list[dict] = []
        train(
            lambda index: _env(world),
            PPOConfig(total_steps=64, horizon=16, num_envs=2, epochs=2,
                      minibatches=2, seed=0),
            on_update=stats.append,
        )
        return [{k: v for k, v in s.items()
                 if k in ("policy_loss", "value_loss", "entropy",
                          "explained_variance")} for s in stats]

    first, second = run(), run()
    assert first, "no update ran"
    assert first == second, (
        "two runs of one seed took different update paths, so no A/B on this "
        "machine measures the change it was launched for")


def test_two_identical_self_play_runs_face_the_same_opponent(world):
    """A ``--seed`` that does not reach the workers is not a seed at all.

    The self-play opponent samples from its own policy, and inside a spawned
    worker that draw came off torch's *global* stream, which ``_worker`` never
    seeded -- it sets the thread count and stops. Windows spawns rather than
    forks, so every worker imported torch fresh and seeded itself from OS
    entropy: three fresh processes running the identical construction on one
    fixed board reported ``torch.initial_seed()`` of 81036942797900,
    81144705125800 and 81234665151700 and shared none of their first twenty
    opponent actions. End to end, ``--steps 1600 --workers 4 --seed 0`` run
    twice from one tree gave update-1 mean_return +0.2556 and -0.1125, while
    the same command at ``--workers 0`` -- where the rollout runs in the
    seeded parent -- matched to the last digit.

    That is what made runs/learn-lvl5-kl01 unreplayable, and it is what a
    paired A/B of two full runs would have measured instead of the change it
    was testing.

    Two separately spawned workers, because one process reused within a test
    would share whatever stream the first play left behind and prove nothing.
    """
    from dataclasses import asdict

    from cr_sim.api.vec import CRSimVecEnv, VecEnvConfig
    from cr_sim.train.nets import net_config_for

    probe = _env(world, frame_skip=30, max_ticks=20 * 60, tower_level=5)
    probe.reset(seed=0)
    torch.manual_seed(7)
    shape = net_config_for(probe, head="flat")
    state = {k: v.clone() for k, v in ActorCritic(shape).state_dict().items()}

    config = VecEnvConfig(
        build=BUILD, blue_deck=DECK, red_deck=DECK, ticks_per_second=20,
        frame_skip=30, tower_level=5, max_ticks=20 * 60,
        net_config=asdict(shape), seed=4)

    def play() -> list[float]:
        vec = CRSimVecEnv(config, num_envs=1, workers=1)
        try:
            vec.set_opponent(state)
            vec.reset([11])
            rewards = []
            for _ in range(20):
                _, step_rewards, _, _, _ = vec.step([(NOOP_SLOT, 0, 0)])
                rewards.append(float(step_rewards[0]))
            return rewards
        finally:
            vec.close()

    first = play()
    # The agent passes every decision, so every reward on this list is the
    # opponent's doing. Without this the assertion below would hold over a
    # worker whose opponent never placed anything at all.
    assert len(set(first)) > 1, "the opponent did nothing worth reproducing"

    assert first == play(), (
        "two identically configured workers played different battles, so the "
        "run's seed does not reach the opponent and no --workers run can be "
        "replayed or paired against another")


# --------------------------------- the demonstrations are harvested under the
# --------------------------------- reward they will later be fine-tuned with


def test_make_demos_builds_the_reward_the_fine_tune_will_use(world):
    """Read off the environment make_demos actually builds.

    The test that used to carry this name never imported make_demos at all:
    it called ``run._reward_weights`` on a hand-built namespace, which is the
    function make_demos happens to call, and asserted on its return value. So
    make_demos could accept ``--tower-weight``, stamp it into the shard's
    meta, and hand 1.0 to the reward, with the whole of this file green.
    Measured under that mutation: the shard's own meta read
    ``projected:elixir=0,horizon_seconds=3,tower=1`` for a run launched with
    ``--tower-weight 0.5``, because the meta is read back off the same wrong
    environment.

    (The namespace-level assertions still exist, one test down, under a name
    that says what they test.)
    """
    import scripts.make_demos as make_demos
    from cr_sim.train.selfplay import reward_name

    args = make_demos.build_parser().parse_args([
        "--reward", "projected", "--tower-weight", "0.5",
        "--elixir-weight", "0.2", "--reward-horizon-seconds", "3",
        # The search's own lookahead, which is a different quantity that
        # shares a name and a unit. It must not reach the reward.
        "--horizon-seconds", "15"])
    env = make_demos.env_factory(args, 0, world)(0)

    assert reward_name(env) == (
        "projected:elixir=0.2,horizon_seconds=3,tower=0.5")


def test_the_collapse_refusal_names_flags_that_exist():
    """``--min-random-candidates`` is half the documented remedy for target
    collapse and no command line exposed it.

    ``make_demos`` printed "Lower --policy-candidates or raise
    --min-random-candidates and collect again", and
    ``make_demos.py --min-random-candidates 8`` exited 2 on an unrecognised
    argument: a repo-wide grep found ``min_random_candidates`` set only inside
    SearchBot's own clamp, so ``SearchBotConfig.random_floor`` was permanently
    ``max(2, candidates // 3)`` -- the floor the whole collapse defence rests
    on, untunable by the operator the message is addressed to.
    """
    import json as _json
    import re
    from types import SimpleNamespace

    import scripts.make_demos as make_demos

    refusal = make_demos.collapse_refusal(
        SimpleNamespace(meta=_json.dumps({"min_spread_fallback_rate": 1.0})),
        0.0)
    assert refusal, "a fully collapsed shard was not refused"
    named = set(re.findall(r"--[a-z][a-z-]+", refusal))
    assert named, "the refusal names no remedy at all"
    assert named <= make_demos._flag_names(), (
        f"the refusal tells the operator to use {sorted(named - make_demos._flag_names())}, "
        "which this script's parser does not accept")

    # And the flag reaches the floor rather than only the parser.
    parser = make_demos.build_parser()
    raised = make_demos.search_config(
        parser.parse_args(["--min-random-candidates", "8"]), guided=True)
    assert raised.random_floor == 8
    # Which is what it is for: the floor is taken out of the proposer's share.
    assert raised.effective_policy_candidates == 6

    default = make_demos.search_config(parser.parse_args([]), guided=True)
    assert default.random_floor == 4, "the default floor moved"


def test_the_iteration_driver_passes_the_knobs_its_round_depends_on():
    """``expert_iterate`` computed the search's own distribution and threw it
    away in the same round.

    It invoked clone_policy without ``--targets``, so that script's default of
    ``hard`` applied and ``data.target = None`` discarded the soft target --
    the third arrow of the loop this driver exists to close. Measured on a
    real round: ``--rounds 1 --shards 1 --episodes 2`` wrote
    runs/iter-1/cloned.pt recording ``targets: 'hard'`` over a shard whose
    ``min_spread_fallback_rate`` was 0.0, i.e. whose targets were healthy.

    Parsed back through the two scripts' own parsers rather than compared as
    strings, so a renamed or retyped flag fails here too.
    """
    import scripts.clone_policy as clone_policy
    import scripts.expert_iterate as expert_iterate
    import scripts.make_demos as make_demos

    args = expert_iterate.build_parser().parse_args([
        "--seed-policy", "runs/clone-v1-paired/cloned.pt",
        "--candidates", "6", "--min-random-candidates", "3"])

    collected = make_demos.build_parser().parse_args(
        [str(v) for v in expert_iterate.demo_command(
            args, 0, "data_cache/demos-iter1", "seed.pt")][2:])
    assert collected.candidates == 6
    assert collected.min_random_candidates == 3

    cloned = clone_policy.build_parser().parse_args(
        [str(v) for v in expert_iterate.clone_command(
            args, "data_cache/demos-iter1", "runs/iter-1")][2:])
    assert cloned.targets == "soft"

    # The default the driver was silently taking, pinned so this test says
    # why passing the flag matters rather than merely that it is passed.
    assert clone_policy.build_parser().parse_args([]).targets == "hard"


def test_run_reward_weights_builds_what_the_flags_ask_for():
    """The clone's critic is what reinforcement learning inherits.

    make_demos built its env with no reward_weights at all, so every
    demonstration set carried value targets from the simple shaped reward
    while every fine-tune ran `projected`. The inherited critic arrived
    predicting +1.48 where PPO's returns averaged +0.47 -- a quantity nobody
    was optimising.

    This tests ``run._reward_weights`` and nothing else; whether make_demos
    calls it with the right arguments is the test above.
    """
    from types import SimpleNamespace

    from cr_sim.api.reward import ProjectionWeights
    from cr_sim.train.run import _reward_weights

    weights = _reward_weights(SimpleNamespace(
        reward="projected", horizon_seconds=3.0, elixir_weight=0.0,
        tower_weight=0.5))
    assert isinstance(weights, ProjectionWeights)
    assert weights.horizon_seconds == 3.0
    assert weights.elixir == 0.0
    # The tower coefficient reaches the weights too. It had no flag anywhere
    # and was pinned at 1.0, so a demonstration set could not be harvested
    # under the shaping a fine-tune starts from even in principle.
    assert weights.tower == 0.5

    # And a caller that does not name the knob is told so, rather than being
    # handed a default. make_demos passes a hand-built namespace here -- the
    # search's horizon is not the reward's -- and a getattr default there
    # would let it build a reward it did not ask for and say nothing. Loud is
    # the whole point: this project has already trained a run at tower level
    # 11 while its config recorded 5.
    with pytest.raises(AttributeError):
        _reward_weights(SimpleNamespace(
            reward="projected", horizon_seconds=3.0, elixir_weight=0.0))


def test_every_head_the_network_can_build_is_reachable_from_both_entry_points():
    """A head no entry point accepts is a head no checkpoint can be written
    with, and that is not a hypothetical: ``"factored-stats"`` shipped
    complete -- config field, head class, worker round-trip -- while both
    ``--head`` choice tuples still said ``("flat", "factored", "conv")``, so
    ``python -m cr_sim.train.run --head factored-stats`` exited 2 and nothing
    downstream could ever be handed one.

    ``run.py`` and ``clone_policy.py`` are the only two writers of a
    checkpoint's ``"head"`` field, so they are the two that have to agree with
    the network.
    """
    from cr_sim.train.nets import POLICY_HEADS
    from cr_sim.train.run import build_parser

    import scripts.clone_policy as clone_policy

    # Pinned as a literal as well as compared, so shrinking the tuple to make
    # a test pass is itself a failure rather than a smaller matrix.
    assert set(POLICY_HEADS) == {"flat", "factored", "factored-stats", "conv"}

    for head in POLICY_HEADS:
        assert build_parser().parse_args(["--head", head]).head == head
        assert clone_policy.build_parser().parse_args(["--head", head]).head == head

    # And the refusal still works, in both directions: a name the network
    # cannot build must not be accepted by either parser.
    for parser in (build_parser(), clone_policy.build_parser()):
        with pytest.raises(SystemExit):
            parser.parse_args(["--head", "autoregressive-maybe"])


def test_every_head_the_command_line_offers_actually_builds_a_network():
    """The other direction. A choice the parser accepts and ``ActorCritic``
    refuses is a run that dies after loading the data and building the
    environment, which on this project is minutes in."""
    from cr_sim.api.env import CRSimEnv
    from cr_sim.data.cards import build_card_registry
    from cr_sim.data.leveling import build_level_table
    from cr_sim.data.source import LogicData
    from cr_sim.train.nets import POLICY_HEADS, ActorCritic, net_config_for

    from .test_data_pipeline import BUILD

    data = LogicData.load(BUILD)
    levels, registry = build_level_table(data), build_card_registry(data)
    deck = ("Knight", "Musketeer", "Cannon", "Skeletons",
            "IceSpirits", "Log", "Fireball", "Goblins")
    env = CRSimEnv(data, levels, registry, deck, deck, ticks_per_second=20,
                   frame_skip=20, max_ticks=400)
    env.reset(seed=0)
    for head in POLICY_HEADS:
        net = ActorCritic(net_config_for(env, head=head))
        assert sum(p.numel() for p in net.policy_head.parameters()) > 0, head


def test_every_reward_a_run_can_train_under_can_also_be_recorded():
    """The two parsers are written separately, and drifted once already.

    run.py defaults to five-term and every real run passes --reward projected
    explicitly, so pinning the *defaults* together would pin a fiction. What
    has to hold is that a reward the fine-tune can be launched with is one the
    demonstrations can be harvested under -- otherwise the value targets and
    the objective are different quantities and nothing says so.
    """
    from cr_sim.train.run import build_parser

    import scripts.make_demos as make_demos

    def choices_for(parser, flag):
        for action in parser._actions:
            if flag in action.option_strings:
                return set(action.choices or ())
        raise AssertionError(f"{flag} is not a flag on this parser")

    run_rewards = choices_for(build_parser(), "--reward")
    demo_rewards = choices_for(make_demos.build_parser(), "--reward")
    missing = run_rewards - demo_rewards
    assert not missing, (
        f"a run can train under {sorted(missing)} but demonstrations cannot "
        "be recorded under it, so its value targets would come from a "
        "different reward than the objective")


def test_the_demo_generator_exposes_the_reward_knobs_the_run_does():
    from cr_sim.train.run import build_parser

    import scripts.make_demos as make_demos

    demo_flags = {s for a in make_demos.build_parser()._actions
                  for s in a.option_strings}
    for flag in ("--reward", "--elixir-weight", "--tower-weight",
                 "--tower-level"):
        assert flag in demo_flags, (
            f"{flag} changes what the demonstrations mean and must be "
            "settable when recording them")
    assert "--reward-horizon-seconds" in demo_flags


def test_make_demos_does_not_confuse_the_search_horizon_with_the_rewards():
    """Two different quantities that share a name and a unit.

    --horizon-seconds is how far the SEARCH projects each candidate (15s);
    --reward-horizon-seconds is the projected reward's lookahead (3s). Passing
    the parser's namespace straight to _reward_weights, which reads
    `horizon_seconds`, would build the reward with a five-times-too-long
    lookahead.
    """
    import scripts.make_demos as make_demos

    source = pathlib.Path(make_demos.__file__).read_text(encoding="utf-8")
    assert "horizon_seconds=args.reward_horizon_seconds" in source, (
        "the reward must be built from --reward-horizon-seconds, not from "
        "--horizon-seconds, which is the search's own lookahead")
    assert "_reward_weights(args)" not in source, (
        "passing args directly picks up the search horizon as the reward's")


# ---------------------------------------- a measurement not taken says nothing


def test_the_ancestor_probe_reports_nothing_rather_than_nan():
    """--ancestor-episodes 0 wrote NaN into every row for a whole run.

    np.mean([]) is NaN, json.dumps writes it as a bare NaN token which is not
    valid JSON, and the page then drew an empty self-play ladder with no way to
    tell a broken ladder from one that was never measured. An absent key is the
    honest report of a measurement not taken.

    The pool must be non-empty for this to bite: with nothing in it the probe
    exits through the older `ancestor is None` guard and never reaches the
    arithmetic. A first version of this test made that mistake and passed
    against the unfixed source.
    """
    import math

    from cr_sim.train.selfplay import ancestor_probe

    class _Pool:
        """Non-empty, so the probe gets past its ancestor-is-None guard."""

        generations = 3

        def __len__(self):
            return 1

        def oldest(self):
            return object()

    def _make_env(opponent):
        raise AssertionError(
            "a zero-episode probe must not build an environment at all")

    probe = ancestor_probe(_make_env, _Pool(), (5, 9, 16), episodes=0)
    row = probe(None)
    assert row == {}, f"zero episodes must produce no keys, got {row}"
    for value in row.values():
        assert not (isinstance(value, float) and math.isnan(value))


def test_a_run_records_whether_it_measured_a_ladder_at_all(tmp_path,
                                                           monkeypatch):
    """ancestor_episodes decides whether the self-play ladder exists.

    It was not recorded, so a run directory with no ancestor rows was
    indistinguishable from one whose probe had failed.
    """
    from cr_sim.api import vec as vec_module

    captured = {}

    class _Stop(RuntimeError):
        pass

    def _capture(config, num_envs, workers):
        captured["config"] = config
        raise _Stop("captured")

    monkeypatch.setattr(vec_module, "CRSimVecEnv", _capture)

    from cr_sim.train.run import main

    with pytest.raises(_Stop):
        main([
            "--steps", "64", "--horizon", "8", "--envs", "2", "--workers", "1",
            "--match-seconds", "20", "--device", "cpu",
            "--ancestor-episodes", "7",
            "--out", str(tmp_path), "--name", "ladder",
        ])

    written = json.loads((tmp_path / "ladder" / "config.json").read_text())
    assert written["ancestor_episodes"] == 7


# ------------------------------------------ a run is a file, checked before it runs


def _recipe_args(**over):
    from cr_sim.train.run import build_parser, _parse_with_recipe
    argv = ["--steps", "64", "--envs", "4", "--workers", "2", "--head",
            "factored", "--tower-level", "5", "--seed", "3", "--name", "r"]
    for k, v in over.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return _parse_with_recipe(build_parser(), argv)


def test_a_runs_own_config_json_relaunches_it(tmp_path):
    """Nineteen of forty-one flags were recorded nowhere.

    steps, workers, envs, lr, entropy, device, the whole anneal and both
    ladder flags were absent from config.json, so the file this project
    treats as a run's record could describe the run and could not relaunch
    it. The recipe key closes that: what recipe_of writes, --config reads
    back to the same namespace.
    """
    from cr_sim.train.run import build_parser, recipe_of, _parse_with_recipe

    original = _recipe_args(ladder_anchor="random", lr="0.0001")
    # Shaped like the real file: the recipe beside twenty derived keys.
    record = {"recipe": recipe_of(original), "eval_opponent": "random",
              "observation_channels": ["own_ground_hp"]}
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(record), encoding="utf-8")

    back = _parse_with_recipe(build_parser(), ["--config", str(cfg)])
    for key, value in recipe_of(original).items():
        assert recipe_of(back)[key] == value, key
    # And the flags that were previously lost are among what came back.
    for key in ("steps", "workers", "envs", "lr", "ladder_anchor", "seed"):
        assert recipe_of(back)[key] == recipe_of(original)[key], key


def test_a_recipe_key_the_parser_does_not_know_is_refused(tmp_path):
    """A misspelt key that is silently skipped trains on the default."""
    from cr_sim.train.run import build_parser, _parse_with_recipe

    bad = tmp_path / "r.yaml"
    bad.write_text("tower_level: 5\ntower_levle: 11\n", encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        _parse_with_recipe(build_parser(), ["--config", str(bad)])
    assert "tower_levle" in str(caught.value)


def test_command_line_flags_beat_the_recipe(tmp_path):
    """Relaunching a run with one flag changed must be a one-flag command."""
    from cr_sim.train.run import build_parser, _parse_with_recipe

    r = tmp_path / "r.yaml"
    r.write_text("seed: 3\nhead: factored\ntower_level: 5\n", encoding="utf-8")
    args = _parse_with_recipe(build_parser(),
                              ["--config", str(r), "--seed", "7"])
    assert args.seed == 7, "the command line must override the file"
    assert args.head == "factored", "the file must still supply what was not typed"


def test_the_recipe_is_written_into_config_json(tmp_path):
    """The record must be able to relaunch the run it records."""
    from cr_sim.train.run import main

    code = main([
        "--steps", "64", "--horizon", "8", "--envs", "2", "--workers", "0",
        "--match-seconds", "20", "--eval-every", "1000", "--device", "cpu",
        "--save-every", "1000", "--opponent", "idle",
        "--out", str(tmp_path), "--name", "rec",
    ])
    assert code == 0
    written = json.loads((tmp_path / "rec" / "config.json").read_text())
    assert "recipe" in written, "config.json must carry the launch recipe"
    recipe = written["recipe"]
    assert recipe["steps"] == 64
    assert recipe["match_seconds"] == 20
    # The previously-unrecorded ones, spot-checked.
    for key in ("workers", "envs", "device", "eval_every", "save_every"):
        assert key in recipe, f"{key} was one of the nineteen flags nobody recorded"
    # Invocation-only flags must not be carried into the next launch.
    for key in ("config", "doctor", "resume", "replace"):
        assert key not in recipe, f"{key} describes this invocation, not the run"


def test_doctor_refuses_envs_that_do_not_divide_by_workers(tmp_path, capsys):
    from cr_sim.train.run import main

    code = main(["--doctor", "--envs", "8", "--workers", "3",
                 "--out", str(tmp_path), "--name", "d"])
    assert code == 2
    out = capsys.readouterr().out
    assert "FAIL" in out and "does not divide" in out


def test_doctor_refuses_a_head_the_borrowed_weights_do_not_have(tmp_path, capsys):
    """main() refuses this too, but only after loading the whole build."""
    import torch
    from cr_sim.train.run import main

    ck = tmp_path / "clone.pt"
    torch.save({"head": "factored", "observation": "v1", "state_dict": {}}, ck)
    code = main(["--doctor", "--head", "flat", "--init-from", str(ck),
                 "--out", str(tmp_path), "--name", "d"])
    assert code == 2
    assert "factored" in capsys.readouterr().out


def test_doctor_warns_when_the_search_expert_is_a_ladder_anchor(tmp_path, capsys):
    """Naming the expert as an anchor made one probe cost 33,055 seconds.

    A warning and not a failure: playing the expert is sometimes what you
    mean. But it must say the cost, because the flag looked harmless.
    """
    from cr_sim.train.run import main

    code = main(["--doctor", "--probe", "ladder",
                 "--ladder-anchor", "random", "--ladder-anchor", "search-c18h15",
                 "--out", str(tmp_path), "--name", "d"])
    out = capsys.readouterr().out
    assert "search-c18h15" in out and "33,055" in out
    assert code == 0, "a warning must not fail the preflight"


def test_doctor_refuses_an_anchor_the_ratings_table_does_not_hold(tmp_path, capsys):
    """An unrated anchor is pinned at 0 Elo and shifts every reading silently."""
    from cr_sim.train.run import main

    table = pathlib.Path("runs/agent-expert-rating/ladder.json")
    if not table.is_file():
        pytest.skip("the offline ratings table is not on this machine")
    code = main(["--doctor", "--probe", "ladder",
                 "--ladder-anchor", "no-such-player",
                 "--ladder-ratings", str(table),
                 "--observation", "v1", "--tower-level", "5",
                 "--out", str(tmp_path), "--name", "d"])
    assert code == 2
    assert "no-such-player" in capsys.readouterr().out


def test_doctor_writes_nothing(tmp_path):
    """A preflight that creates the run directory is not a preflight."""
    from cr_sim.train.run import main

    main(["--doctor", "--out", str(tmp_path), "--name", "never-made"])
    assert not (tmp_path / "never-made").exists()


def test_ship_gates_on_a_demonstrated_regression_only():
    """SHIP unless the candidate's greedy interval sits wholly below the baseline's.

    Overlap is the honest reading of most real comparisons here and must
    ship; only a separated, lower interval is a regression a gate can know.
    """
    import scripts.ship as ship

    def row(path, mode, lift, lo, hi):
        return {"checkpoint": path, "mode": mode, "lift": lift,
                "ci_low": lo, "ci_high": hi, "eval_opponent": "random"}
    c, b = pathlib.Path("c.pt"), pathlib.Path("b.pt")

    overlapping = [row("c.pt", "greedy", 2.10, 1.90, 2.30),
                   row("b.pt", "greedy", 2.17, 1.96, 2.37)]
    assert ship.verdict(overlapping, c, b)[0] is True

    regressed = [row("c.pt", "greedy", 1.40, 1.20, 1.60),
                 row("b.pt", "greedy", 2.17, 1.96, 2.37)]
    assert ship.verdict(regressed, c, b)[0] is False

    better = [row("c.pt", "greedy", 2.60, 2.40, 2.80),
              row("b.pt", "greedy", 2.17, 1.96, 2.37)]
    assert ship.verdict(better, c, b)[0] is True
