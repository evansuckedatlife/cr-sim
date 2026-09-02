"""The card-stat action head: what it conditions on, and what it does not.

:class:`~cr_sim.train.nets.FactoredStatsHead` replaces the factored head's
32-slot identity lookup with a small encoder over the card's own statistics.
The point of the change is that a card outside the training decks is
representable at all, so the tests that matter here are the ones about *which*
card a slot is conditioned on -- not about the network converging.

Two traps this file exists for, both of which stayed green in earlier drafts:

* **The mix order.** ``onehots @ encoder(table)`` and
  ``encoder(onehots @ table)`` agree on every non-empty slot and differ only on
  an empty one, and at initialisation they do not even differ there: every bias
  in the head starts at zero and ``LayerNorm`` of an all-zero row is all-zero,
  so ``encoder(0)`` measures exactly 0.0 on a freshly built net. One optimiser
  step ends that. So the empty-slot assertion is made with the encoder's biases
  moved off zero, and the mutation ``cards = self.card_encoder(onehots @
  self.card_stats)`` turns it red.
* **Row order.** Reading row ``vocab_size - 1 - i`` instead of row ``i``
  conditions a Cannon in hand on the Skeletons' stat row and errors nowhere.

Both were applied to :meth:`FactoredStatsHead._context` and measured:
``tests/test_action_head.py`` reports 23 passed under either one, including
its own ``factored-stats`` parametrisations. The tests below go red.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cr_sim.api.encoding import NUM_CARD_SLOTS  # noqa: E402
from cr_sim.api.env import CRSimEnv  # noqa: E402
from cr_sim.data.card_features import (  # noqa: E402
    CARD_FEATURE_COUNT, card_feature_table,
)
from cr_sim.data.cards import build_card_registry  # noqa: E402
from cr_sim.data.leveling import build_level_table  # noqa: E402
from cr_sim.data.source import LogicData  # noqa: E402
from cr_sim.train.nets import (  # noqa: E402
    ActorCritic, FactoredStatsHead, NetConfig, net_config_for,
)

from .test_action_head import DECK  # noqa: E402
from .test_data_pipeline import BUILD  # noqa: E402

#: Eight cards, none of them in ``DECK``, so a net built for one has never
#: seen a single card of the other. Same *size*, because the observation's
#: width is a function of the vocabulary size and a different size is a shape
#: mismatch in the trunk before the head is reached.
UNSEEN_DECK = ("Pekka", "Wizard", "Bowler", "BabyDragon",
               "Zap", "Tornado", "Barbarians", "Minions")


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


def _env(world, deck):
    data, levels, registry = world
    env = CRSimEnv(data, levels, registry, deck, deck,
                   ticks_per_second=20, frame_skip=20, max_ticks=20 * 40)
    env.reset(seed=0)
    return env


@pytest.fixture(scope="module")
def env(world):
    return _env(world, DECK)


@pytest.fixture(scope="module")
def unseen_env(world):
    return _env(world, UNSEEN_DECK)


def _head(env) -> tuple[ActorCritic, FactoredStatsHead]:
    net = ActorCritic(net_config_for(env, head="factored-stats"))
    head = net.policy_head
    assert isinstance(head, FactoredStatsHead)
    return net, head


def _vector_with(config, slot_cards: dict[int, int]) -> torch.Tensor:
    """A zeroed observation vector with one card one-hot set per named slot.

    Every slot not named is left all-zero, which is how the encoder writes an
    empty hand slot.
    """
    vector = torch.zeros(1, config.vector_size)
    for slot, card_index in slot_cards.items():
        start = config.hand_offset + slot * config.hand_stride
        vector[0, start + card_index] = 1.0
    return vector


# ------------------------------------------------------------- construction


def test_the_head_refuses_a_config_with_no_table(env):
    """The browser server builds its ``NetConfig`` by hand, and the day it
    forgets this field the failure is a first move that raises, is swallowed,
    and leaves an opponent that passes for the rest of the match."""
    config = net_config_for(env, head="factored-stats", card_stats=())
    with pytest.raises(ValueError, match="needs a card stat table"):
        ActorCritic(config)


def test_the_head_refuses_a_table_of_the_wrong_length(env):
    """Row ``i`` is what slot ``i``'s one-hot selects. A shorter table is a
    head conditioned on the wrong cards, and a matmul that happens to work."""
    full = net_config_for(env, head="factored-stats")
    config = net_config_for(
        env, head="factored-stats", card_stats=full.card_stats[:-1])
    with pytest.raises(ValueError, match="rows against a vocabulary"):
        ActorCritic(config)


def test_the_identity_lookup_is_gone_from_the_parameters(env):
    """The whole point of the head. A ``card_embedding`` still in the
    state dict is a free column per vocabulary slot, which is the thing being
    replaced."""
    net, head = _head(env)
    names = set(net.state_dict())
    assert not any(name.startswith("policy_head.card_embedding") for name in names)
    assert "policy_head.card_encoder.0.weight" in names
    # And the table itself is not a checkpoint field: it is a function of
    # static card data, so a feature added later must not invalidate weights.
    assert "policy_head.card_stats" not in names
    assert head.card_stats.shape == (net.config.vocab_size, CARD_FEATURE_COUNT)


def test_net_config_for_keys_the_table_on_the_environment_s_vocab(env, world):
    """In the encoding's order, because that is the order the observation's
    one-hot bits are set in."""
    data, levels, registry = world
    config = net_config_for(env, head="factored-stats")
    assert config.card_stats == card_feature_table(
        data, levels, registry, env.encoding.vocab)
    assert len(config.card_stats) == len(env.encoding.vocab)
    # Every other head pays nothing for a field it never reads.
    for head in ("flat", "factored", "conv"):
        assert net_config_for(env, head=head).card_stats == ()


def test_the_config_survives_the_worker_round_trip(env):
    """Each spawned rollout worker rebuilds its network from
    ``asdict(net_config)`` sent down a pipe. A field that does not survive
    that is a worker whose opponent is a differently shaped network."""
    config = net_config_for(env, head="factored-stats")
    shape = asdict(config)
    rebuilt = NetConfig(**pickle.loads(pickle.dumps(shape)))
    assert rebuilt == config
    assert hash(rebuilt) == hash(config)
    ActorCritic(rebuilt)


# ------------------------------------------------- which card a slot reads


def test_slot_i_is_conditioned_on_the_row_its_one_hot_bit_selects(env):
    """The row-order trap. Reading ``vocab_size - 1 - i`` conditions a Cannon
    on the Skeletons' statistics: no error, no shape mismatch, and every test
    in ``test_action_head.py`` still green."""
    net, head = _head(env)
    config = net.config
    with torch.no_grad():
        rows = head.card_encoder(head.card_stats)
        for index in range(config.vocab_size):
            context = head._context(_vector_with(config, {0: index}))
            assert torch.allclose(context[0, 0], rows[index], atol=1e-6), index
            # And it is genuinely selective: no other row would do.
            others = [j for j in range(config.vocab_size) if j != index]
            assert not any(
                torch.allclose(context[0, 0], rows[j], atol=1e-6) for j in others)


def test_the_conditioning_follows_the_card_around_the_hand(env):
    """The hand rotates; slot 2 is a different card every cycle. Two slots
    holding the same card must read the same, and that is what makes what the
    head learns about a Knight follow the Knight."""
    net, head = _head(env)
    config = net.config
    with torch.no_grad():
        context = head._context(_vector_with(config, {0: 3, 2: 3, 1: 5}))
    assert torch.allclose(context[0, 0], context[0, 2], atol=1e-6)
    assert not torch.allclose(context[0, 0], context[0, 1], atol=1e-6)


def test_an_empty_hand_slot_conditions_on_exactly_zero(env):
    """The mix-order trap: encode the table, *then* mix. Never
    ``encoder(onehots @ table)``.

    An empty slot is an all-zero one-hot, and a ``bias=False`` matmul takes
    that to an exactly-zero conditioning vector -- the semantics the base head
    has. ``encoder(0)`` returns the encoder's bias and the LayerNorm's shift
    instead, so an empty slot quietly starts meaning something.

    **Asserted with the biases moved off zero, which is the whole point.**
    ``_orthogonal`` zeroes every bias and ``LayerNorm`` starts at beta 0, so
    on a freshly built head ``encoder(0)`` is exactly 0.0 and the broken order
    passes this test. One optimiser step ends that, so the check has to be
    made somewhere a real run actually lives.
    """
    net, head = _head(env)
    config = net.config

    with torch.no_grad():
        for parameter in head.card_encoder.parameters():
            if parameter.dim() == 1:
                parameter.add_(torch.randn_like(parameter) * 0.5)
        # The premise: with the biases moved, the encoder no longer maps zero
        # to zero. Without this the assertion below is vacuous.
        zeroed = head.card_encoder(torch.zeros(1, CARD_FEATURE_COUNT))
        assert float(zeroed.abs().max()) > 0.1

        context = head._context(_vector_with(config, {1: 0}))
    for slot in (0, 2, 3):
        assert int(torch.count_nonzero(context[0, slot])) == 0, (
            f"empty slot {slot} carries a conditioning vector of "
            f"norm {float(context[0, slot].norm()):.4f}")
    assert int(torch.count_nonzero(context[0, 1])) > 0


def test_the_pass_slot_is_not_run_through_the_encoder(env):
    """Pass has no card. A zero stat row through the encoder would come out
    identical to an empty hand slot's conditioning, and the two are opposites:
    pass is unconditionally legal and an empty slot is never legal."""
    net, head = _head(env)
    with torch.no_grad():
        context = head._context(_vector_with(net.config, {0: 1}))
    assert torch.equal(context[0, NUM_CARD_SLOTS - 1], head.pass_embedding[0])


def test_the_placement_head_reads_which_card_is_in_the_slot(env):
    """A head that ignores its conditioning is a slower flat head with a
    bottleneck, and every metric would look the same."""
    net, head = _head(env)
    config = net.config
    grid = torch.randn(1, config.grid_channels, config.grid_height, config.grid_width)
    mask = torch.ones(1, config.num_actions, dtype=torch.bool)
    cells = config.num_cells

    with torch.no_grad():
        first = net.policy_logits(grid, _vector_with(config, {0: 0}), mask)[0, :cells]
        second = net.policy_logits(grid, _vector_with(config, {0: 1}), mask)[0, :cells]
    assert not torch.allclose(first, second, atol=1e-6), (
        "swapping the card in slot 0 did not move the placement logits")


# ------------------------------------------------------- the decision path


def test_the_encoder_runs_exactly_once_per_decision(env):
    """The table is recomputed every forward on purpose -- see the class
    docstring's cost measurement. A cache added later would freeze the head
    from step 1 while a "the parameters moved" test stayed green, because the
    encoder's weights still move. Counted after a warm-up call, so a cache
    populated on the first forward is what this catches.
    """
    net, head = _head(env)
    config = net.config
    grid = torch.randn(2, config.grid_channels, config.grid_height, config.grid_width)
    vector = torch.rand(2, config.vector_size)
    mask = torch.ones(2, config.num_actions, dtype=torch.bool)

    with torch.no_grad():
        net.policy_logits(grid, vector, mask)      # warm-up

    ran: list[int] = []
    handle = head.card_encoder.register_forward_hook(lambda m, i, o: ran.append(1))
    try:
        with torch.no_grad():
            net.policy_logits(grid, vector, mask)
    finally:
        handle.remove()
    assert sum(ran) == 1, (
        f"the card encoder ran {sum(ran)} times on the decision path; a stale "
        "table is a head that stops learning without any test noticing")


def test_gradient_reaches_the_card_encoder(env):
    """A conditioning path that is detached would still reduce the loss
    through the trunk and look like it was learning."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    net, head = _head(env)
    config = net.config
    cells = config.num_cells
    grid = torch.randn(16, config.grid_channels, config.grid_height, config.grid_width)
    vector = torch.rand(16, config.vector_size)
    mask = torch.ones(16, config.num_actions, dtype=torch.bool)
    target = torch.full((16,), 3 * cells + 11, dtype=torch.long)

    before = {name: p.detach().clone() for name, p in head.named_parameters()}
    optimiser = torch.optim.Adam(net.parameters(), lr=1e-2)
    first = None
    for _ in range(20):
        logits, _ = net(grid, vector, mask)
        loss = F.cross_entropy(logits, target)
        if first is None:
            first = float(loss.detach())
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    assert float(loss.detach()) < first

    moved = {name for name, p in head.named_parameters()
             if not torch.allclose(before[name], p.detach())}
    assert "card_encoder.0.weight" in moved, "no gradient reached the stat encoder"
    assert "card_encoder.2.weight" in moved
    assert "place_out.weight" in moved, "no gradient reached the shared tile matrix"


def test_the_stat_table_does_not_move_under_training(env):
    """It is card data, not a parameter. A table that drifted would be an
    identity embedding again, learned one row at a time."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    net, head = _head(env)
    config = net.config
    before = head.card_stats.detach().clone()
    optimiser = torch.optim.Adam(net.parameters(), lr=1e-2)
    grid = torch.randn(8, config.grid_channels, config.grid_height, config.grid_width)
    vector = torch.rand(8, config.vector_size)
    mask = torch.ones(8, config.num_actions, dtype=torch.bool)
    for _ in range(5):
        logits, _ = net(grid, vector, mask)
        loss = F.cross_entropy(logits, torch.zeros(8, dtype=torch.long))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    assert torch.equal(head.card_stats, before)
    assert not head.card_stats.requires_grad


# ------------------------------------------------------------ the payoff


def test_weights_trained_on_one_deck_play_a_deck_they_have_never_seen(
    env, unseen_env, world
):
    """This is the feature. Eight cards the network was never trained on, in
    the same eight hand slots: the state dict loads strictly, the head
    conditions on the new cards' own statistics, and the logits over the legal
    set are finite.

    The factored head cannot do this. Its ``card_embedding`` column ``i`` means
    whatever ``vocab[i]`` was during training, so the same load silently hands
    the Knight's column to whichever card sorts into position 4.
    """
    data, levels, registry = world
    assert not set(DECK) & set(UNSEEN_DECK)

    trained = ActorCritic(net_config_for(env, head="factored-stats"))
    fresh = ActorCritic(net_config_for(unseen_env, head="factored-stats"))
    # Strict, which is the claim: no missing and no unexpected keys.
    fresh.load_state_dict(trained.state_dict())

    assert fresh.policy_head.card_stats.shape == trained.policy_head.card_stats.shape
    assert not torch.equal(
        fresh.policy_head.card_stats, trained.policy_head.card_stats), (
        "the two decks produced the same stat table; the test proves nothing")
    assert torch.equal(
        fresh.policy_head.card_stats,
        torch.tensor(
            card_feature_table(data, levels, registry, unseen_env.encoding.vocab),
            dtype=torch.float32))

    observation, _ = unseen_env.reset(seed=1)
    mask = torch.from_numpy(unseen_env.legal_action_mask().reshape(1, -1))
    with torch.no_grad():
        logits = fresh.policy_logits(
            torch.from_numpy(observation["grid"]).unsqueeze(0),
            torch.from_numpy(observation["vector"]).unsqueeze(0),
            mask)
    assert torch.isfinite(logits[mask]).all()
    assert bool(mask.any())

    # And the conditioning is the unseen cards', not the trained deck's: same
    # weights, same slot, different card, different vector.
    with torch.no_grad():
        vector = _vector_with(fresh.config, {0: 0})
        assert not torch.allclose(
            fresh.policy_head._context(vector)[0, 0],
            trained.policy_head._context(vector)[0, 0], atol=1e-5)


def test_relabelling_the_vocabulary_moves_the_trunk_and_not_the_head(env):
    """Where the zero-shot property stops, stated as a measurement.

    Permute the vocabulary index assignment consistently across all ten card
    one-hot blocks in the observation *and* across the stat table's rows. That
    is a pure relabelling: identical cards, identical hand slots, identical
    board. The head is built to be blind to it, and is. The trunk is not --
    ``ActorCritic.vector`` takes the whole observation vector, and 80 of its
    102 input columns are per-vocab-index one-hot columns that all receive
    gradient, so training memorises card identity there exactly as the head's
    old 32x8 table did.

    This is here so ``card_features``'s claim about the boundary is a checked
    one. Dropping the identity one-hots from the observation is the change
    that would move the second assertion, and it is a change to the encoder,
    not to the head.
    """
    from dataclasses import replace

    config = net_config_for(env, head="factored-stats")
    width = config.vocab_size
    permutation = [(index + 3) % width for index in range(width)]

    torch.manual_seed(0)
    net = ActorCritic(config)
    relabelled = replace(
        config,
        card_stats=tuple(config.card_stats[permutation[i]] for i in range(width)))
    torch.manual_seed(0)
    other = ActorCritic(relabelled)
    other.load_state_dict(net.state_dict())
    net.eval()
    other.eval()

    observation, _ = env.reset(seed=2)
    vector = torch.from_numpy(observation["vector"]).unsqueeze(0)
    grid = torch.from_numpy(observation["grid"]).unsqueeze(0)
    moved = vector.clone()
    # Ten blocks: four hand slots plus the next card, for each side. Each is a
    # cost scalar followed by the identity one-hot.
    for block in range(10):
        start = 2 + block * (1 + width) + 1
        row = vector[0, start:start + width].clone()
        for index in range(width):
            moved[0, start + index] = row[permutation[index]]
    assert not torch.equal(moved, vector), "the permutation was a no-op"

    with torch.no_grad():
        assert torch.allclose(
            net.policy_head._context(vector),
            other.policy_head._context(moved), atol=1e-5), (
            "the head's conditioning moved under a pure relabelling, which is "
            "the identity dependence this head exists to remove")
        before, _ = net.encode_spatial(grid, vector)
        after, _ = other.encode_spatial(grid, moved)
    drift = float((before - after).norm() / before.norm())
    assert drift > 0.1, (
        f"the trunk moved by {drift:.1%} under the relabelling. If this has "
        "become small the observation no longer carries card identity, and "
        "card_features' statement of the boundary needs rewriting -- not this "
        "assertion loosening")


def test_the_unseen_deck_is_a_real_deck_the_environment_can_play(unseen_env):
    """Guards the test above from being vacuous: an environment that refused
    the deck would make every assertion about it meaningless."""
    for _ in range(5):
        mask = unseen_env.legal_action_mask().reshape(-1)
        legal = np.flatnonzero(mask)
        assert legal.size
        unseen_env.step(np.array(
            np.unravel_index(int(legal[-1]), unseen_env.action_space.nvec)))
