"""The factored action head, and the properties that make it interchangeable.

The policy chooses one of 720 actions: five card slots (four cards plus a
pass) times a 9x16 placement grid. A flat head is one linear layer over all
720; the factored head picks the slot, then the tile, with the tile head
conditioned on an embedding of the card actually in that slot and its weights
shared across cards.

The argument for factoring is *not* expressiveness -- a flat masked
categorical can represent any distribution over the legal set, and so can the
factorisation, which is why the tests below pin equivalence rather than
superiority. It is sample efficiency: cloning the search bot produced 6,094
play examples over 443 distinct (card, tile) pairs, and a flat head learns
nothing about a tile from one card that it can apply to another.

What is easy to get quietly wrong, and is what these check:

* the joint logits landing in a different order than the environment's action
  index, which transposes every placement without erroring,
* mass leaking onto an illegal action because two log-softmaxes leave a finite
  floor rather than a true zero,
* the card conditioning reading the wrong span of the observation vector, so
  the head is conditioned on an elixir scalar or on the opponent's hand,
* the head silently doing nothing -- being a reparameterisation that ignores
  its conditioning input entirely.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cr_sim.api.encoding import (  # noqa: E402
    NUM_CARD_SLOTS, build_encoding_config, hand_onehot_layout,
)
from cr_sim.api.env import CRSimEnv  # noqa: E402
from cr_sim.data.cards import build_card_registry  # noqa: E402
from cr_sim.data.leveling import build_level_table  # noqa: E402
from cr_sim.data.source import LogicData  # noqa: E402
from cr_sim.train.nets import (  # noqa: E402
    ActorCritic, ConvPlacementHead, FactoredHead, NetConfig, net_config_for,
)

from .test_data_pipeline import BUILD  # noqa: E402

DECK = ("Knight", "Musketeer", "Cannon", "Skeletons",
        "IceSpirits", "Log", "Fireball", "Goblins")


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


@pytest.fixture(scope="module")
def env(world):
    data, levels, registry = world
    env = CRSimEnv(data, levels, registry, DECK, DECK,
                   ticks_per_second=20, frame_skip=20, max_ticks=20 * 40)
    env.reset(seed=0)
    return env


def _mask(batch: int, num_actions: int, cells: int, rng) -> torch.Tensor:
    """A mask shaped like a real one: the pass cell always, plus a couple of
    slots with some of their cells legal."""
    mask = np.zeros((batch, num_actions), dtype=bool)
    pass_index = (NUM_CARD_SLOTS - 1) * cells
    mask[:, pass_index] = True
    for row in range(batch):
        for slot in rng.choice(NUM_CARD_SLOTS - 1, size=2, replace=False):
            picked = rng.choice(cells, size=max(1, cells // 3), replace=False)
            mask[row, int(slot) * cells + picked] = True
    return torch.from_numpy(mask)


# ------------------------------------------------------ the encoding contract


def test_the_hand_layout_points_at_the_card_that_is_actually_in_the_slot(world):
    """A head conditioned on the wrong span of the vector is conditioned on
    the opponent's hand, or on an elixir scalar, and nothing errors."""
    from cr_sim.api.encoding import encode_observation
    from cr_sim.engine.entity import Team

    data, levels, registry = world
    env = CRSimEnv(data, levels, registry, DECK, DECK,
                   ticks_per_second=20, frame_skip=20, max_ticks=20 * 40)
    env.reset(seed=3)
    config = env.encoding
    start, stride, count, width = hand_onehot_layout(config)
    assert count == NUM_CARD_SLOTS - 1

    observation = encode_observation(env.battle, Team.BLUE, registry, config)
    vector = observation["vector"]
    hand = env.battle.players[Team.BLUE].hand
    for slot in range(count):
        onehot = vector[start + slot * stride: start + slot * stride + width]
        assert onehot.sum() == pytest.approx(1.0), (
            f"slot {slot}'s one-hot is not a one-hot: {onehot}")
        assert config.vocab[int(np.argmax(onehot))] == hand[slot]


# ---------------------------------------------- interchangeable with the flat


@pytest.mark.parametrize("head", ["flat", "factored", "conv"])
def test_no_probability_reaches_a_masked_action(env, head):
    """Two log-softmaxes composed leave a finite floor, not a true zero, so
    the joint has to be re-masked. Without that an illegal placement keeps a
    tiny but nonzero probability and the rollout samples it."""
    net = ActorCritic(net_config_for(env, head=head))
    rng = np.random.default_rng(0)
    config = net.config
    grid = torch.randn(8, config.grid_channels, config.grid_height, config.grid_width)
    vector = torch.rand(8, config.vector_size)
    mask = _mask(8, config.num_actions, config.num_cells, rng)

    logits, _ = net(grid, vector, mask)
    probs = torch.distributions.Categorical(logits=logits).probs.detach()
    assert float(probs[~mask].sum()) == 0.0
    assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)


@pytest.mark.parametrize("head", ["flat", "factored", "conv"])
def test_the_head_produces_one_logit_per_environment_action(env, head):
    net = ActorCritic(net_config_for(env, head=head))
    config = net.config
    assert config.num_actions == int(np.prod(env.action_space.nvec))
    assert config.num_cells == config.num_actions // NUM_CARD_SLOTS
    grid = torch.randn(2, config.grid_channels, config.grid_height, config.grid_width)
    vector = torch.rand(2, config.vector_size)
    mask = _mask(2, config.num_actions, config.num_cells, np.random.default_rng(1))
    logits, value = net(grid, vector, mask)
    assert logits.shape == (2, config.num_actions)
    assert value.shape == (2,)


def test_the_factored_head_lays_slots_out_in_the_environment_s_order(env):
    """``slot * cells + cell``, the same flatten the mask and the action
    decoder use. Getting this wrong is a legal-looking placement in the wrong
    half of the board rather than an error."""
    config = net_config_for(env, head="factored")
    net = ActorCritic(config)
    cells = config.num_cells

    # Exactly one slot legal, and within it exactly one cell. Whatever the
    # weights are, all the mass has to land on that one flat index.
    mask = torch.zeros(1, config.num_actions, dtype=torch.bool)
    mask[0, (NUM_CARD_SLOTS - 1) * cells] = True   # pass is always legal
    target = 2 * cells + 37
    mask[0, target] = True
    grid = torch.randn(1, config.grid_channels, config.grid_height, config.grid_width)
    vector = torch.rand(1, config.vector_size)
    logits, _ = net(grid, vector, mask)
    probs = torch.distributions.Categorical(logits=logits).probs.detach()
    assert float(probs[0, target] + probs[0, (NUM_CARD_SLOTS - 1) * cells]) == pytest.approx(1.0, abs=1e-5)


def test_a_slot_with_no_legal_cell_gets_no_probability(env):
    """The slot distribution is derived from the flat mask, not from a second
    reading of elixir. If the two could disagree the policy would put mass on
    a card it cannot play, and the placement inside it would be rejected by
    the engine and read as ordinary bad play."""
    config = net_config_for(env, head="factored")
    net = ActorCritic(config)
    cells = config.num_cells
    mask = torch.zeros(1, config.num_actions, dtype=torch.bool)
    mask[0, (NUM_CARD_SLOTS - 1) * cells] = True
    mask[0, 1 * cells + 5] = True
    logits, _ = net(
        torch.randn(1, config.grid_channels, config.grid_height, config.grid_width),
        torch.rand(1, config.vector_size), mask)
    probs = torch.distributions.Categorical(logits=logits).probs.detach()
    for slot in (0, 2, 3):
        assert float(probs[0, slot * cells:(slot + 1) * cells].sum()) == 0.0


# ------------------------------------------------- the conditioning is not inert


def test_the_placement_head_actually_reads_which_card_is_in_the_slot(env):
    """The whole argument for factoring is that the tile head sees the card.
    A head that ignores its conditioning input is a slower flat head with a
    bottleneck, and every metric would look the same.
    """
    config = net_config_for(env, head="factored")
    assert config.vocab_size > 0, "the deck vocabulary did not reach the head"
    net = ActorCritic(config)
    cells = config.num_cells
    start, stride, count, width = config.hand_offset, config.hand_stride, 4, config.vocab_size

    grid = torch.randn(1, config.grid_channels, config.grid_height, config.grid_width)
    mask = torch.ones(1, config.num_actions, dtype=torch.bool)

    def logits_for(card_index: int) -> torch.Tensor:
        vector = torch.zeros(1, config.vector_size)
        vector[0, start: start + width] = 0.0
        vector[0, start + card_index] = 1.0
        out, _ = net(grid, vector, mask)
        return out[0, 0:cells]

    first, second = logits_for(0), logits_for(1)
    assert not torch.allclose(first, second, atol=1e-6), (
        "swapping the card in slot 0 did not change where the head wants to "
        "place it -- the conditioning is inert")


def test_the_placement_weights_are_shared_across_cards(env):
    """The sample-efficiency claim in one assertion: there is one tile matrix,
    not one per card. A flat head has ``hidden x 720`` policy weights; the
    factored head's tile output is ``place_hidden x 144`` whatever the card.
    """
    net = ActorCritic(net_config_for(env, head="factored"))
    head = net.policy_head
    assert isinstance(head, FactoredHead)
    assert head.place_out.weight.shape == (head.cells, net.config.place_hidden)

    flat = ActorCritic(net_config_for(env, head="flat"))
    flat_policy = sum(p.numel() for p in flat.policy_head.parameters())
    factored_policy = sum(p.numel() for p in head.parameters())
    assert factored_policy < flat_policy, (
        f"factored head has {factored_policy} policy parameters against the "
        f"flat head's {flat_policy}")


def test_an_unknown_head_name_is_refused(env):
    with pytest.raises(ValueError, match="unknown policy head"):
        ActorCritic(net_config_for(env, head="autoregressive-maybe"))


# ------------------------------------------------------------------- training


def test_the_factored_head_can_be_trained_to_a_particular_placement(env):
    """Not a convergence claim about the game -- a check that gradients reach
    the card embedding and the shared tile matrix at all. A head whose
    conditioning is detached would still reduce the loss through the trunk and
    look like it was learning."""
    import torch.nn.functional as F

    config = net_config_for(env, head="factored")
    net = ActorCritic(config)
    cells = config.num_cells
    rng = np.random.default_rng(0)
    grid = torch.randn(16, config.grid_channels, config.grid_height, config.grid_width)
    vector = torch.rand(16, config.vector_size)
    mask = torch.ones(16, config.num_actions, dtype=torch.bool)
    target = torch.full((16,), 3 * cells + 11, dtype=torch.long)

    before = {name: p.detach().clone()
              for name, p in net.policy_head.named_parameters()}
    optimiser = torch.optim.Adam(net.parameters(), lr=1e-2)
    first = None
    for _ in range(40):
        logits, _ = net(grid, vector, mask)
        loss = F.cross_entropy(logits, target)
        if first is None:
            first = float(loss.detach())
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    assert float(loss.detach()) < first
    assert int(logits.argmax(dim=-1)[0]) == 3 * cells + 11

    moved = {name for name, p in net.policy_head.named_parameters()
             if not torch.allclose(before[name], p.detach())}
    assert "card_embedding.weight" in moved, "no gradient reached the card embedding"
    assert "place_out.weight" in moved, "no gradient reached the shared tile matrix"


# ------------------------------------------------- the convolutional head


def test_the_placement_grid_is_exactly_the_trunk_s_own_feature_map(env):
    """The whole premise. The observation is 32x18 at one cell per tile, the
    placement grid is 16x9 at one per two tiles, and the trunk's second
    convolution is stride 2 -- so the map it produces is the placement grid,
    already computed and thrown away by the layer after it."""
    config = net_config_for(env, head="conv")
    net = ActorCritic(config)
    grid = torch.randn(3, config.grid_channels, config.grid_height, config.grid_width)
    vector = torch.rand(3, config.vector_size)
    _features, spatial = net.encode_spatial(grid, vector)
    assert spatial.shape == (3, config.channels,
                             env.encoding.action_height, env.encoding.action_width)
    assert (spatial.shape[2] * spatial.shape[3] * NUM_CARD_SLOTS
            == config.num_actions)


def test_the_conv_head_lands_a_cell_where_decode_action_reads_it(env):
    """The transposition trap, pinned end to end.

    ``decode_action`` reads a flat index as slot, then x, then y. A
    convolution's natural layout is (slot, y, x), and flattening that directly
    puts every placement in a legal-looking cell somewhere else on the board
    without raising anything. So: drive one cell of the head's input, and
    check the flat index that comes out decodes to the world point that cell
    stands for.
    """
    from cr_sim.api.encoding import PLACEMENT_TILE_SPAN, cell_to_world, decode_action
    from cr_sim.engine.entity import Team

    config = net_config_for(env, head="conv")
    net = ActorCritic(config)
    head = net.policy_head
    assert isinstance(head, ConvPlacementHead)

    slot, gy, gx = 2, 3, 5
    with torch.no_grad():
        head.place.weight.zero_()
        head.place.bias.zero_()
        head.place.weight[slot, 0, 0, 0] = 1.0
        head.context.weight.zero_()
        head.context.bias.zero_()

    spatial = torch.zeros(1, config.channels, head.height, head.width)
    spatial[0, 0, gy, gx] = 10.0
    features = torch.zeros(1, config.hidden)
    mask = torch.ones(1, config.num_actions, dtype=torch.bool)
    index = int(head(features, spatial, mask).argmax())

    width, height = env.encoding.action_width, env.encoding.action_height
    assert index == slot * width * height + gx * height + gy

    decoded = decode_action((slot, gx, gy), Team.BLUE, env.battle.arena, env.encoding)
    assert decoded is not None
    assert decoded[1:] == cell_to_world(
        gx, gy, Team.BLUE, env.battle.arena, span=PLACEMENT_TILE_SPAN)


def test_the_conv_head_reads_the_non_spatial_state(env):
    """A convolution over the board alone cannot know which cards are in hand
    or how much elixir there is. If the context projection were inert the head
    would place a Fireball exactly where it places a Knight."""
    config = net_config_for(env, head="conv")
    net = ActorCritic(config)
    grid = torch.randn(1, config.grid_channels, config.grid_height, config.grid_width)
    mask = torch.ones(1, config.num_actions, dtype=torch.bool)
    first = net.policy_logits(grid, torch.zeros(1, config.vector_size), mask)
    second = net.policy_logits(grid, torch.ones(1, config.vector_size), mask)
    assert not torch.allclose(first, second, atol=1e-6), (
        "the board decided everything; the non-spatial context is inert")


def test_the_conv_head_is_the_smallest_of_the_three(env):
    counts = {}
    for head in ("flat", "factored", "conv"):
        net = ActorCritic(net_config_for(env, head=head))
        counts[head] = sum(p.numel() for p in net.policy_head.parameters())
    assert counts["conv"] < counts["factored"] < counts["flat"], counts


@pytest.mark.parametrize("head", ["flat", "factored", "conv"])
def test_policy_logits_matches_forward_and_skips_the_critic(env, head):
    """Choosing an action does not need a value. With a separate critic
    encoder, ``forward`` spends about half its time computing one that is
    thrown away, and every frozen opponent and evaluation was paying it."""
    config = net_config_for(env, head=head)
    net = ActorCritic(config)
    net.eval()
    grid = torch.randn(4, config.grid_channels, config.grid_height, config.grid_width)
    vector = torch.rand(4, config.vector_size)
    mask = _mask(4, config.num_actions, config.num_cells, np.random.default_rng(2))

    with torch.no_grad():
        both, _value = net(grid, vector, mask)
        only = net.policy_logits(grid, vector, mask)
    assert torch.equal(both, only)

    ran = []
    handles = [module.register_forward_hook(lambda m, i, o: ran.append(m))
               for module in (net.critic_conv, net.critic_vector,
                              net.critic_trunk, net.value_head)]
    try:
        with torch.no_grad():
            net.policy_logits(grid, vector, mask)
    finally:
        for handle in handles:
            handle.remove()
    assert not ran, f"the critic ran anyway: {[type(m).__name__ for m in ran]}"
