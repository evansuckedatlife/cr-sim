"""Self-play against a spread of past selves, and the ladder that measures it.

A single frozen opponent lets the learner cycle: beat last week's strategy,
forget the one before, and go round in circles while the return says nothing
is wrong. A pool of ancestors has to be beaten all at once.

Everything here exists because this code shipped inert. The pool was created
and filled exactly once -- with the randomly initialised network -- so
self-play would have spent an entire run beating a policy that never improved,
and the ancestor probe was constructed and never called. Both looked wired.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cr_sim.train.nets import ActorCritic, NetConfig
from cr_sim.train.selfplay import OpponentPool, PooledOpponent

NVEC = (5, 9, 16)


def _net(seed: int) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(NetConfig(
        grid_channels=9, grid_height=32, grid_width=18,
        vector_size=102, num_actions=NVEC[0] * NVEC[1] * NVEC[2]))


def _first_weight(net) -> float:
    # detach first: pool members carry requires_grad=False, but the live
    # networks handed in do not, and float() on a grad-tracking tensor warns.
    return float(next(net.parameters()).detach().flatten()[0])


def test_the_pool_keeps_its_oldest_member(pool_capacity=3):
    """The oldest is the ladder's benchmark. Evicting oldest-first would turn
    the pool back into a sliding window of recent selves, which is the thing
    it exists to avoid."""
    pool = OpponentPool(capacity=pool_capacity, seed=0)
    first = _net(0)
    pool.add(first)
    for seed in range(1, 8):
        pool.add(_net(seed))
    assert len(pool) == pool_capacity
    assert _first_weight(pool.oldest()) == _first_weight(first)


def test_the_pool_keeps_the_newest_member_too():
    """The newest is the only ancestor near the learner's own strength."""
    pool = OpponentPool(capacity=3, seed=0)
    for seed in range(6):
        pool.add(_net(seed))
    newest = _net(5)
    weights = [_first_weight(pool.sample()) for _ in range(60)]
    assert _first_weight(newest) in weights, "the newest generation was evicted"


def test_a_snapshot_does_not_move_when_the_learner_does():
    """An opponent that drifted mid-rollout would make the advantage estimates
    measure a moving target, which the algorithm assumes they do not."""
    net = _net(0)
    pool = OpponentPool(capacity=4, seed=0)
    pool.add(net)
    before = _first_weight(pool.oldest())
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.add_(1.0)
    assert _first_weight(pool.oldest()) == before


def test_an_empty_pool_has_nothing_to_offer():
    pool = OpponentPool(capacity=4, seed=0)
    assert pool.sample() is None and pool.oldest() is None and len(pool) == 0


def test_refreshing_draws_an_ancestor_rather_than_the_current_policy():
    """The point of the pool: a refresh may hand back any generation, not
    only the most recent one."""
    pool = OpponentPool(capacity=4, seed=0)
    for seed in range(4):
        pool.add(_net(seed))
    opponent = PooledOpponent(pool, _net(99), NVEC, seed=0)
    drawn = set()
    for _ in range(80):
        opponent.refresh(_net(99))
        drawn.add(_first_weight(opponent._net))
    assert len(drawn) > 1, "every refresh drew the same ancestor"
    assert _first_weight(_net(99)) not in drawn, "it drew the live policy"


def test_the_pool_is_filled_as_training_refreshes():
    """The bug this file was written for.

    ``train`` calls ``on_refresh`` once per refresh cycle, and the runner uses
    it to add a generation. Without that call the pool holds only the network
    it was seeded with, and every opponent for the rest of the run is the
    randomly initialised policy.
    """
    import inspect

    from cr_sim.train import ppo, run

    assert "on_refresh" in inspect.signature(ppo.train).parameters
    source = inspect.getsource(run.main)
    assert "on_refresh=" in source, "the runner never fills the pool"
    assert "probe_holder.get(\"ancestor\")" in source, "the ladder is never run"
