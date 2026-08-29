"""The KL trust region, and the demonstrations it is anchored to.

Fine-tuning a behavioural clone with plain PPO walked it back toward playing
nothing: over 34 updates the pass rate climbed 8% -> 15% -> 36% while entropy
fell the whole time, so the collapse was the policy gradient's doing and not
the entropy bonus's. ``KL(reference || policy)`` is the standard remedy and
the one AlphaStar kept on for the whole of its league training.

These tests do not claim it wins. They claim it is not inert -- which on this
project is the failure mode with a track record: three mechanics have shipped
doing nothing while their tests passed.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cr_sim.train.clone import Demonstrations  # noqa: E402
from cr_sim.train.nets import ActorCritic, NetConfig  # noqa: E402
from cr_sim.train.ppo import PPOConfig, Rollout, _update, compute_gae  # noqa: E402

NVEC = (5, 3, 4)
NUM_ACTIONS = NVEC[0] * NVEC[1] * NVEC[2]
PASS = (NVEC[0] - 1) * NVEC[1] * NVEC[2]


def _net(seed: int = 0):
    torch.manual_seed(seed)
    return ActorCritic(NetConfig(
        grid_channels=2, grid_height=4, grid_width=3,
        vector_size=6, num_actions=NUM_ACTIONS))


def _rollout(horizon=16, envs=4, seed=0):
    """A batch that pays for passing and not for anything else.

    Half the transitions take the pass action and are followed by reward, the
    other half take a placement and are not. That mix matters: PPO normalises
    advantages within a minibatch, so a batch where *every* sample is a paid
    pass has mean-zero advantage and produces almost no gradient at all. The
    drift being reproduced is a preference between two things, which is what
    this is.
    """
    generator = torch.Generator().manual_seed(seed)
    grid = torch.rand((horizon, envs, 2, 4, 3), generator=generator)
    vector = torch.rand((horizon, envs, 6), generator=generator)
    mask = torch.ones((horizon, envs, NUM_ACTIONS), dtype=torch.bool)
    passing = torch.zeros((horizon, envs), dtype=torch.bool)
    passing[::2] = True
    action = torch.where(
        passing, torch.full((horizon, envs), PASS, dtype=torch.long),
        torch.randint(0, PASS, (horizon, envs), generator=generator))
    log_prob = torch.full((horizon, envs), -float(np.log(NUM_ACTIONS)))
    value = torch.zeros((horizon, envs))
    done = torch.zeros((horizon, envs))
    # The advantage is supplied directly rather than bootstrapped: what is
    # being tested is what the update does with a given advantage, and
    # deriving it through GAE only adds a discount factor to reason about.
    advantage = torch.where(passing, torch.ones((horizon, envs)),
                            -torch.ones((horizon, envs)))
    reward = advantage.clone()
    _, ret = compute_gae(reward, value, done, torch.zeros(envs), 0.99, 0.95)
    return Rollout(grid=grid, vector=vector, mask=mask, action=action,
                   log_prob=log_prob, value=value, reward=reward, done=done,
                   advantage=advantage, ret=ret)


def _pass_probability(net, rollout) -> float:
    with torch.no_grad():
        logits, _ = net(
            rollout.grid.reshape(-1, 2, 4, 3),
            rollout.vector.reshape(-1, 6),
            rollout.mask.reshape(-1, NUM_ACTIONS))
        return float(torch.softmax(logits, dim=-1)[:, PASS].mean())


def _drift(kl: float, updates: int = 4) -> float:
    net = _net()
    anchor = copy.deepcopy(net) if kl > 0 else None
    if anchor is not None:
        anchor.eval()
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)
    optimiser = torch.optim.Adam(net.parameters(), lr=3e-3, eps=1e-5)
    rollout = _rollout()
    config = PPOConfig(kl_coefficient=kl, entropy_coefficient=0.0,
                       epochs=2, minibatches=2)
    before = _pass_probability(net, rollout)
    shuffler = np.random.default_rng(0)
    for _ in range(updates):
        _update(net, optimiser, rollout, config, "cpu", anchor,
                shuffler=shuffler)
    return _pass_probability(net, rollout) - before


def test_without_a_trust_region_the_policy_walks_to_the_action_it_is_paid_for():
    """The control. Without this the next test could pass by the update doing
    nothing at all."""
    assert _drift(kl=0.0) > 0.05


def test_the_trust_region_slows_the_walk():
    free = _drift(kl=0.0)
    held = _drift(kl=5.0)
    assert held < free, (
        f"anchored drift {held:+.4f} was not smaller than free drift {free:+.4f}")


def test_the_divergence_is_reported_and_starts_at_zero():
    """A policy identical to its anchor has zero KL from it. If the number
    reported were not the divergence at all, this is where it shows."""
    net = _net()
    anchor = copy.deepcopy(net)
    anchor.eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    rollout = _rollout()
    config = PPOConfig(kl_coefficient=1.0, entropy_coefficient=0.0,
                       epochs=1, minibatches=1)
    # A zero learning rate so the first minibatch's divergence is measured
    # before anything has moved.
    optimiser = torch.optim.Adam(net.parameters(), lr=0.0)
    stats = _update(net, optimiser, rollout, config, "cpu", anchor,
                    shuffler=np.random.default_rng(0))
    assert "reference_kl" in stats
    assert stats["reference_kl"] == pytest.approx(0.0, abs=1e-5)

    stats = _update(net, optimiser, rollout, config, "cpu", None,
                    shuffler=np.random.default_rng(0))
    assert stats["reference_kl"] == 0.0, "no anchor should report no divergence"


def test_the_coefficient_at_zero_changes_nothing():
    """Plain PPO has to stay plain PPO. Two runs of the same update with the
    trust region switched off must be bit-identical, or every result recorded
    before it existed is no longer comparable."""
    rollout = _rollout()
    config_off = PPOConfig(kl_coefficient=0.0, entropy_coefficient=0.01,
                           epochs=2, minibatches=2)

    def weights_after(anchor):
        net = _net()
        optimiser = torch.optim.Adam(net.parameters(), lr=1e-3, eps=1e-5)
        # The minibatch permutation comes from a stream this call owns.
        # It used to come from numpy's global legacy RandomState, which this
        # line had to reseed by hand -- and which a real run never seeded at
        # all, so two runs of one command with one --seed took different
        # update paths.
        _update(net, optimiser, rollout, config_off, "cpu", anchor,
                shuffler=np.random.default_rng(0))
        return [p.detach().clone() for p in net.parameters()]

    without = weights_after(None)
    with_unused_anchor = weights_after(_net(seed=1))
    for a, b in zip(without, with_unused_anchor):
        assert torch.equal(a, b)


def test_a_run_refuses_a_trust_region_with_nothing_to_anchor_to(tmp_path):
    """``--kl`` without ``--init-from`` anchors to a random initialisation,
    which is not a trust region."""
    from cr_sim.train.run import main

    with pytest.raises(SystemExit, match="init-from"):
        main(["--steps", "16", "--horizon", "4", "--envs", "1", "--kl", "1.0",
              "--out", str(tmp_path), "--name", "anchorless"])
