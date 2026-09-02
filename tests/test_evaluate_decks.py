"""The zero-shot deck evaluation: that it plays the deck it says it does.

``scripts/evaluate_decks.py`` exists to answer one question -- does a policy
whose head reads a card's *statistics* survive a deck it never trained on
better than one whose head reads a lookup table -- and the way that question
gets answered wrongly is not a crash. It is a run that quietly plays the
training deck eight more times, or that hands the stat head the training
deck's statistics while the board holds different cards. Either produces a
clean null: two arms that score the same, because nothing varied.

That is this codebase's signature failure applied to a measurement instead of
to a feature, so the tests here assert on what the environment and the network
actually hold, never on the script running to completion.

The mechanism test at the bottom is the one that pins the claim being
measured. On an unseen deck the lookup head's conditioning is *unchanged* --
same weights, same one-hot bit, so the Knight's learned column now describes
whichever card sorted into the Knight's position -- while the stat head's
conditioning moves, because row ``i`` of its table is that deck's card. The
lookup arm is not uninformed about a new deck; it is misinformed, and that is
a stronger claim than "it has no slot for the card", which is what the head
was originally described as fixing.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cr_sim.api.env import CRSimEnv  # noqa: E402
from cr_sim.data.card_features import card_feature_table  # noqa: E402
from cr_sim.data.cards import build_card_registry  # noqa: E402
from cr_sim.data.leveling import build_level_table  # noqa: E402
from cr_sim.data.source import LogicData  # noqa: E402
from cr_sim.train.nets import ActorCritic, net_config_for  # noqa: E402
from cr_sim.train.run import DEFAULT_DECK  # noqa: E402

from .test_data_pipeline import BUILD  # noqa: E402

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_decks import (  # noqa: E402
    DECK_SIZE, Bench, build_parser, cost_matched, decks_for, load_policy,
    sample_decks,
)

#: Eight cards, none of them in ``DEFAULT_DECK``. Same size, because an 8-card
#: mirror deck is what holds the observation width at the trained one.
UNSEEN_DECK = ("Pekka", "Wizard", "Bowler", "BabyDragon",
               "Zap", "Tornado", "Barbarians", "Minions")


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


@pytest.fixture(scope="module")
def bench():
    return Bench(build=BUILD, tower_level=5, observation="v1")


@pytest.fixture(scope="module")
def pool(bench):
    return [card.name for card in bench.registry.standard()]


def _env(world, deck):
    data, levels, registry = world
    env = CRSimEnv(data, levels, registry, deck, deck,
                   ticks_per_second=20, frame_skip=20, max_ticks=20 * 40)
    env.reset(seed=0)
    return env


def _checkpoint(tmp_path, env, head, name):
    """A real checkpoint of ``head``, written the way the cloner writes one."""
    net = ActorCritic(net_config_for(env, head=head))
    path = tmp_path / f"{name}.pt"
    torch.save({"state_dict": net.state_dict(), "observation": "v1",
                "head": head}, path)
    return path, net


# ------------------------------------------------------------ deck sampling


def test_sample_decks_never_returns_a_card_the_policy_trained_on(pool):
    """"Unseen" has to be a property of the returned decks, not of the intent
    behind the call. A single training card leaking into a deck makes that
    deck's result a mixture of two different measurements."""
    decks = sample_decks(pool, 12, 20260828, exclude=DEFAULT_DECK)
    assert len(decks) == 12
    for deck in decks:
        assert not set(deck) & set(DEFAULT_DECK), deck


def test_sample_decks_returns_distinct_decks_of_distinct_cards(pool):
    """Two identical decks would be reported as two independent measurements
    of the same thing, and a repeated card is not a legal deck."""
    decks = sample_decks(pool, 12, 20260828, exclude=DEFAULT_DECK)
    assert len(set(decks)) == len(decks)
    for deck in decks:
        assert len(deck) == DECK_SIZE
        assert len(set(deck)) == DECK_SIZE


def test_sample_decks_is_deterministic_in_its_seed(pool):
    """Both arms are handed the same decks by being handed the same seed. If
    the draw moved between processes, each policy would face its own
    opponents and the two lifts would not be comparable."""
    assert (sample_decks(pool, 6, 4242, exclude=DEFAULT_DECK)
            == sample_decks(pool, 6, 4242, exclude=DEFAULT_DECK))
    assert (sample_decks(pool, 6, 4242, exclude=DEFAULT_DECK)
            != sample_decks(pool, 6, 99, exclude=DEFAULT_DECK))


def test_sample_decks_refuses_a_pool_too_small_to_fill_a_deck(pool):
    with pytest.raises(SystemExit, match="cannot fill a deck"):
        sample_decks(pool[:20], 1, 0, exclude=pool[:15])


def test_sample_decks_rejects_a_repeat_when_a_repeat_is_actually_likely(pool):
    """The dedup, exercised where it can fire.

    Over 114 cards a collision never happens, so on the real pool the guard is
    unreachable and a test on that pool proves nothing about it. Nine
    candidates make exactly nine decks of eight, so drawing all nine forces
    the loop to reject repeats over and over -- and the result still has to be
    nine distinct decks.
    """
    small = pool[:9]
    decks = sample_decks(small, 9, 7, size=DECK_SIZE)
    assert len(decks) == 9
    assert len(set(decks)) == 9


def test_sample_decks_refuses_more_decks_than_the_pool_can_make(pool):
    """Without this the rejection loop spins forever looking for a tenth deck
    that does not exist, which in an hour-long evaluation is indistinguishable
    from slow progress."""
    with pytest.raises(SystemExit, match="make only 9 decks"):
        sample_decks(pool[:9], 10, 7, size=DECK_SIZE)


def test_slicing_the_sweep_does_not_change_which_decks_are_played(pool):
    """The reason ``--only`` slices after the draw instead of drawing less.

    Three processes covering indices 0-3, 4-6 and 7-9 have to between them
    play exactly the ten decks one process would have played. Draw
    ``len(only)`` decks per process instead and each gets its own first-three,
    all of them identical to each other -- the merged file then reports one
    deck measured three times as three decks, with no sign anything is wrong.
    """
    whole = decks_for(pool, 10, 20260828)
    sliced = (decks_for(pool, 10, 20260828, only=[0, 1, 2, 3])
              + decks_for(pool, 10, 20260828, only=[4, 5, 6])
              + decks_for(pool, 10, 20260828, only=[7, 8, 9]))
    assert sliced == whole
    # The labels travel with the decks, so a merged file can be read back.
    assert [label for label, _ in whole] == [f"unseen-{i:02d}" for i in range(10)]
    # And the shortcut this exists to rule out really does go wrong. Drawing
    # only what a process needs is *right* for the first slice -- the draws are
    # sequential from one stream -- and wrong for every slice after it, which
    # is the worst way for it to be wrong: it looks correct when checked on
    # process one.
    naive = [deck for _, deck in decks_for(pool, 3, 20260828)]
    assert naive == [deck for _, deck in whole[:3]]
    assert naive != [deck for _, deck in whole[4:7]]


def test_the_training_deck_anchor_is_the_deck_the_policies_trained_on(pool):
    """It is the row every unseen number is read against, so it has to be the
    training deck itself and not another draw."""
    labelled = decks_for(pool, 2, 20260828, include_training=True)
    assert labelled[0] == ("training-deck", tuple(DEFAULT_DECK))
    assert len(labelled) == 3


def test_the_training_deck_anchor_keeps_its_declared_card_order(pool):
    """A deck is a cycle, not a set.

    The order the eight cards are listed in sets the starting hand, so sorting
    them makes different battles out of the same cards -- measured, the random
    control takes 20% of 40 seeds with ``DEFAULT_DECK`` as written and 28%
    with it sorted. The vocabulary is sorted either way, so the observation's
    layout is byte-identical and nothing downstream can notice. Sorting here
    would leave the anchor row quietly incomparable to every number this
    project has already recorded on this deck.
    """
    labelled = decks_for(pool, 0, 20260828, include_training=True)
    assert labelled[0][1] == tuple(DEFAULT_DECK)
    assert labelled[0][1] != tuple(sorted(DEFAULT_DECK))


def test_slicing_past_the_end_of_the_deck_list_is_refused(pool):
    """Silently returning fewer decks would leave a gap in the merged sweep
    that looks exactly like a deck that was never interesting."""
    with pytest.raises(SystemExit, match="outside the 4 decks"):
        decks_for(pool, 4, 20260828, only=[4])


# ------------------------------------------------------ the cost confound


def test_a_uniform_draw_is_far_more_expensive_than_the_training_deck(bench, pool):
    """The confound the cost filter exists for, measured rather than assumed.

    ``DEFAULT_DECK`` is a deliberately cheap cycle deck. Eight cards drawn
    uniformly from the rest of the game are not, and a policy that scores
    nothing on them has met two changes at once.
    """
    costs = {card.name: card.mana_cost for card in bench.registry.standard()}
    training = np.mean([costs[name] for name in DEFAULT_DECK])
    drawn = [np.mean([costs[name] for name in deck])
             for _, deck in decks_for(pool, 10, 20260828)]
    assert training == pytest.approx(2.5)
    assert np.mean(drawn) > training + 1.0


def test_the_cost_filter_holds_the_elixir_economy_near_the_training_deck(
        bench, pool):
    """What isolates unfamiliar cards from an unfamiliar economy."""
    costs = {card.name: card.mana_cost for card in bench.registry.standard()}
    accept = cost_matched(costs, 0.25)
    assert accept.target == pytest.approx(2.5)
    decks = decks_for(pool, 6, 20260828, accept=accept)
    for _, deck in decks:
        assert abs(np.mean([costs[name] for name in deck]) - 2.5) <= 0.25
        # Still genuinely unseen: cheap is not an excuse to reuse a card.
        assert not set(deck) & set(DEFAULT_DECK)
    assert len({deck for _, deck in decks}) == 6


def test_the_cost_filter_uses_the_costs_it_is_given():
    """A closure over an explicit table, so this states the costs it means
    rather than depending on what the card data happens to say today."""
    costs = {"a": 1, "b": 1, "c": 9, "d": 9}
    accept = cost_matched(costs, 0.0, reference=("a", "c"))
    assert accept.target == pytest.approx(5.0)
    assert accept(("a", "c")) and accept(("b", "d"))
    assert not accept(("a", "b"))


def test_a_filter_no_deck_can_satisfy_is_refused_rather_than_looped_on(pool):
    """The budget's reachable branch. With a predicate in play the deck count
    cannot tell whether the request is satisfiable, so this is the only guard
    left -- and an evaluation that never starts must say so, not spin."""
    with pytest.raises(SystemExit, match="filter is too strict"):
        decks_for(pool, 3, 20260828, accept=lambda deck: False)


# ------------------------------------------------- the deck actually played


def test_the_bench_plays_the_deck_it_was_given(bench):
    """The failure that produces a clean null. If ``make_env`` ignored its
    argument every deck would be the training deck, both arms would score
    their training-deck number, and the report would read as a tidy
    "no difference on unseen decks"."""
    env = bench.make_env(UNSEEN_DECK)
    assert env.blue_deck == tuple(UNSEEN_DECK)
    assert env.red_deck == tuple(UNSEEN_DECK)
    # And the encoder agrees, which is what the one-hot bits are built from.
    assert env.encoding.vocab == tuple(sorted(UNSEEN_DECK))
    assert not set(env.encoding.vocab) & set(DEFAULT_DECK)


def test_an_unseen_mirror_deck_keeps_the_observation_the_trained_width(bench):
    """The premise the whole experiment rests on. A mirror deck of eight makes
    a vocabulary of eight whatever the cards are, so the observation width is
    unchanged and every checkpoint on disk loads against it. If this were
    false the measurement would be a shape error rather than a result."""
    trained = bench.make_env(DEFAULT_DECK)
    unseen = bench.make_env(UNSEEN_DECK)
    for env in (trained, unseen):
        observation, _ = env.reset(seed=0)
        assert observation["vector"].shape == (102,)
        assert env.encoding.vocab_size == DECK_SIZE


def test_the_bench_refuses_a_deck_with_a_repeated_card(bench):
    with pytest.raises(SystemExit, match="repeated card"):
        bench.make_env(("Knight",) * DECK_SIZE)


# --------------------------------------------- the network built to play it


@pytest.mark.parametrize("head", ["factored", "factored-stats"])
def test_a_checkpoint_loads_strictly_against_a_deck_it_never_saw(
        tmp_path, world, bench, head):
    """No missing keys, no unexpected keys, and a legal action out the far
    end. Nothing errors on an unseen deck -- which is exactly why the failure
    this measures is silent."""
    trained_env = _env(world, DEFAULT_DECK)
    path, _ = _checkpoint(tmp_path, trained_env, head, head)
    unseen = bench.make_env(UNSEEN_DECK)
    unseen.reset(seed=0)
    net, payload = load_policy(path, unseen)
    assert payload["head"] == head
    observation, _ = unseen.reset(seed=0)
    mask = unseen.legal_action_mask().reshape(-1)
    with torch.no_grad():
        logits = net.policy_logits(
            torch.as_tensor(observation["grid"], dtype=torch.float32)[None],
            torch.as_tensor(observation["vector"], dtype=torch.float32)[None],
            torch.as_tensor(mask, dtype=torch.bool)[None])
    assert bool(mask[int(logits.argmax())]), "picked an illegal action"


def test_load_policy_refuses_a_checkpoint_that_does_not_fit_the_head(
        tmp_path, world, bench):
    """Strictly, so a weight that silently stayed at its random
    initialisation cannot be scored as if it had been trained. The whole
    experiment is two checkpoints differenced against one control; a partial
    load would put an untrained layer into one of the arms and read as a
    regression in the head."""
    trained_env = _env(world, DEFAULT_DECK)
    path, _ = _checkpoint(tmp_path, trained_env, "factored-stats", "partial")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["state_dict"]["policy_head.card_encoder.0.weight"]
    torch.save(payload, path)
    unseen = bench.make_env(UNSEEN_DECK)
    unseen.reset(seed=0)
    with pytest.raises(RuntimeError, match="[Mm]issing key"):
        load_policy(path, unseen)


def test_load_policy_builds_the_stat_table_for_the_deck_being_played(
        tmp_path, world, bench):
    """The measurement's load-bearing line. ``net_config_for`` reads the stat
    table off the environment it is handed, so handing it the *training*
    environment would describe eight cards that are not on the board -- both
    arms then wrong in the same way, and a null that means nothing."""
    data, levels, registry = world
    trained_env = _env(world, DEFAULT_DECK)
    path, _ = _checkpoint(tmp_path, trained_env, "factored-stats", "stats")

    unseen = bench.make_env(UNSEEN_DECK)
    unseen.reset(seed=0)
    net, _ = load_policy(path, unseen)
    played = net.policy_head.card_stats

    want = torch.tensor(card_feature_table(data, levels, registry,
                                           tuple(sorted(UNSEEN_DECK))),
                        dtype=torch.float32)
    assert torch.allclose(played, want, atol=1e-6)
    # And it is genuinely not the training deck's table. Without this the
    # assertion above passes for a table that happens to be right by
    # coincidence of the two decks having similar cards.
    trained_table = torch.tensor(
        card_feature_table(data, levels, registry, tuple(sorted(DEFAULT_DECK))),
        dtype=torch.float32)
    assert played.shape == trained_table.shape
    assert not torch.allclose(played, trained_table, atol=1e-3)


# ------------------------------------------------------------ the mechanism


def test_the_lookup_head_is_misinformed_on_an_unseen_deck_and_the_encoder_is_not(
        tmp_path, world, bench):
    """What the experiment is actually testing, asserted directly.

    Hold the observation fixed -- slot 0 holding the card at vocabulary
    position 4 -- and change only which deck is on the board. The lookup head
    reads its own weight column 4 either way, so its conditioning does not
    move by a single float: it applies what it learned about the training
    deck's fifth card to a completely different card. The stat head recomputes
    row 4 from that deck's statistics, so its conditioning does move.

    A card the lookup head has "no slot for" was the original framing and it
    is too kind. There is always a slot. It holds the wrong card.
    """
    trained_env = _env(world, DEFAULT_DECK)
    trained_env_bench = bench.make_env(DEFAULT_DECK)
    trained_env_bench.reset(seed=0)
    unseen = bench.make_env(UNSEEN_DECK)
    unseen.reset(seed=0)

    position = 4
    assert (trained_env_bench.encoding.vocab[position]
            != unseen.encoding.vocab[position])

    moved = {}
    for head in ("factored", "factored-stats"):
        path, _ = _checkpoint(tmp_path, trained_env, head, head)
        contexts = []
        for env in (trained_env_bench, unseen):
            net, _ = load_policy(path, env)
            config = net.config
            vector = torch.zeros(1, config.vector_size)
            vector[0, config.hand_offset + position] = 1.0
            with torch.no_grad():
                contexts.append(net.policy_head._context(vector)[0, 0].clone())
        moved[head] = float((contexts[0] - contexts[1]).abs().max())

    # Byte-identical: the same weights, indexed by the same one-hot bit.
    assert moved["factored"] == 0.0
    # And the encoder's conditioning is a different vector entirely.
    assert moved["factored-stats"] > 0.1, moved


def test_the_parser_exposes_the_flags_the_experiment_is_run_with():
    """A flag nobody can inspect is a flag nobody checks -- the same reason
    ``scripts/clone_policy.py`` grew a ``build_parser``."""
    args = build_parser().parse_args(
        ["a.pt", "b.pt", "--decks", "3", "--episodes", "7",
         "--tower-level", "5", "--deck-seed", "11"])
    assert args.checkpoints == ["a.pt", "b.pt"]
    assert (args.decks, args.episodes, args.tower_level, args.deck_seed) == (
        3, 7, 5, 11)
