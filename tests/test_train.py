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
