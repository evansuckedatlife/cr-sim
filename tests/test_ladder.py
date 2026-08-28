"""Whether a rating measures the players rather than the machinery around them.

The lift this project has always reported is saturated: everything it has
built lands in a band 0.024 sd wide, and the expert anchor that band is aimed
at resolves to +/-0.179 sd at n=150. A rating opens that band to roughly 300
Elo, but only if four things hold, and each of them is a way the old metric
already failed at least once:

*   **The mirror is real.** The controlled side acts before ``_opponent_move``
    inside every frame-skip window, all match, so it moves first every time.
    The same weights on both sides measured 0.550 and 0.600 for the controlled
    side where 0.500 is the truth.
*   **Greedy reproduces exactly.** That is the entire argument for a greedy
    ladder, and ``temperature=1e-3`` -- which is what a ladder built out of
    today's ``FrozenOpponent`` would have to use -- is only *nearly* argmax.
*   **A saturated pairing does not blow the fit up.** The expert is 100-0
    against the random control and an unregularised Bradley-Terry maximum is
    at infinity.
*   **Every row says who it played.** Read off the environment, never off the
    argument list.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from cr_sim.api.encoding import NOOP_SLOT
from cr_sim.api.env import CRSimEnv
from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.train.ladder import (
    Direction, Pairing, Player, fit_ratings, head_for_parameters,
    ladder_probe, play_pairing, player_from_checkpoint,
)
from cr_sim.train.nets import ActorCritic, net_config_for

from .test_data_pipeline import BUILD

DECK = ("Knight", "Musketeer", "Cannon", "Skeletons",
        "IceSpirits", "Log", "Fireball", "Goblins")


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


@pytest.fixture(scope="module")
def make_env(world):
    """A battle short enough to run a few dozen of.

    Forty seconds at two-second decisions. Nothing measured here is a rating
    -- these check that the machinery's guarantees hold, which is what a test
    can afford. A real pairing is 800 battles.
    """
    data, levels, registry = world

    def factory(opponent=None) -> CRSimEnv:
        return CRSimEnv(data, levels, registry, DECK, DECK,
                        ticks_per_second=20, frame_skip=40,
                        max_ticks=20 * 40, tower_level=5,
                        opponent_policy=opponent)

    return factory


@pytest.fixture(scope="module")
def net(make_env):
    """A fixed randomly-initialised policy.

    Seeded, because the weights decide which battles are decisive and an
    unseeded fixture would draw different ones depending on what else the
    test session imported first. A test whose discriminating power moves with
    the import order is a test that will one day pass over the broken case.
    """
    env = make_env(None)
    env.reset(seed=0)
    torch.manual_seed(0)
    built = ActorCritic(net_config_for(env, head="flat"))
    built.eval()
    return built


def _passive(name: str):
    """A named opponent that never plays a card."""
    def policy(observation, mask):
        return (NOOP_SLOT, 0, 0)

    policy.opponent_name = name  # type: ignore[attr-defined]
    return policy


# ------------------------------------------------------------- the mirror


def test_a_mirrored_pairing_of_identical_nets_scores_exactly_one_half(
        make_env, net):
    """A player is not stronger than itself, and one direction says it is.

    ``CRSimEnv.step`` applies the controlled side's action and only then calls
    ``_opponent_move()``, inside every frame-skip window, all match. So the
    controlled side moves first every time, and a single-direction score is
    that advantage plus the policy. Measured on two independent nets at n=40,
    the controlled side scored 0.550 and 0.600 against its own weights.

    The mirror swaps which player the environment controls, so both halves of
    the bias cancel exactly rather than approximately -- which is why this
    asserts equality against 0.5 and not a tolerance.
    """
    a = Player(name="twin-a", kind="net", net=net)
    b = Player(name="twin-b", kind="net", net=net)
    pairing = play_pairing(make_env, a, b, seeds=[5, 6, 7, 8], mode="greedy")

    # These seeds are decisive on purpose. A pairing that drew every battle
    # would score 0.5 in every direction and the assertion below would hold
    # over a mirror that does nothing at all -- which is this codebase's
    # signature failure, not a passing test. Here the controlled side takes
    # 0.625 of a match-up against its own weights, and the mirror puts it
    # back to 0.5.
    assert any(c != 0 for c in pairing.forward.crowns)

    assert pairing.score == 0.5

    # And the thing the mirror is correcting is really there: rating on the
    # controlled side alone would report this number instead.
    assert pairing.forward.score != 0.5
    assert pairing.forward.score == pairing.reverse.score


def test_a_greedy_pairing_reproduces_bit_identically(make_env, net):
    """Exact float equality, run to run, which is the whole point of greedy.

    The precedent is scripts/measure_sampled_noise.py, which asserts the same
    thing for the same reason: if a repeat does not land on the same number,
    the battles are not being held fixed and no difference measured through
    them means anything.

    The sampled arm is held to the same standard, which is stronger than it
    sounds -- ``evaluate.py``'s no-generator branch draws from torch's global
    stream, it is live in two production scripts, and every sampled number in
    ``runs/_anchor/*.json`` is unreproducible because of it.
    """
    a = Player(name="alpha", kind="net", net=net)
    b = Player(name="beta", kind="random")

    first = play_pairing(make_env, a, b, seeds=[21, 22, 23], mode="greedy")
    second = play_pairing(make_env, a, b, seeds=[21, 22, 23], mode="greedy")
    assert first.score == second.score
    assert first.forward.crowns == second.forward.crowns
    assert first.reverse.crowns == second.reverse.crowns

    # Sampled is a different policy and gets its own generator rather than
    # torch's global one. Two plays of the same pairing must still land on the
    # same battles; without the generator the second play continues wherever
    # the first left off.
    third = play_pairing(make_env, a, b, seeds=[21, 22, 23], mode="sampled")
    fourth = play_pairing(make_env, a, b, seeds=[21, 22, 23], mode="sampled")
    assert third.forward.crowns == fourth.forward.crowns
    assert third.reverse.crowns == fourth.reverse.crowns


# --------------------------------------------------------------- the fit


def _lopsided(a: str, b: str, battles: int = 100) -> Pairing:
    """A synthetic table where ``a`` wins every battle both ways round."""
    seeds = list(range(battles))
    forward = Direction(blue=a, red=b, seeds=seeds, wins=battles, losses=0,
                        draws=0, score=1.0, crowns=[1] * battles)
    reverse = Direction(blue=b, red=a, seeds=seeds, wins=0, losses=battles,
                        draws=0, score=0.0, crowns=[-1] * battles)
    return Pairing(a=a, b=b, mode="greedy", forward=forward, reverse=reverse,
                   score=1.0, seed_correlation=None, a_ref="a", b_ref="b")


def test_the_rating_stays_finite_when_a_player_wins_every_battle():
    """The expert is 100-0 against the random control. That is not rare here.

    An unregularised Bradley-Terry fit has its maximum at infinity on such an
    edge -- the likelihood is monotone in the rating difference and never
    turns over -- so the fit runs away and the Hessian goes to zero, which
    leaves no standard error either. A Gaussian prior on every rating makes
    the posterior mode finite and gives the curvature back.
    """
    ratings = fit_ratings([_lopsided("expert", "random")])

    elo = ratings["expert"].elo
    assert math.isfinite(elo)
    # With N(0, 400) on a 100-0 edge the mode sits around +820. Without the
    # prior the Newton iteration walks off by roughly 174 points a step and
    # ends five figures out.
    assert 300.0 < elo < 3000.0
    assert ratings["random"].elo == 0.0
    assert ratings["random"].pinned

    error = ratings["expert"].error
    assert math.isfinite(error) and error > 0.0
    assert ratings["expert"].games == 200


def test_a_rating_orders_two_players_the_way_the_pairings_do():
    """The fit is not only finite, it is right way up and roughly calibrated.

    A player scoring 0.75 against the anchor sits about 190 Elo above it
    before the prior pulls it in; one scoring 0.25 sits about the same amount
    below.
    """
    def pairing(name: str, wins: int, losses: int) -> Pairing:
        seeds = list(range(wins + losses))
        forward = Direction(blue=name, red="random", seeds=seeds, wins=wins,
                            losses=losses, draws=0,
                            score=wins / (wins + losses),
                            crowns=[1] * wins + [-1] * losses)
        reverse = Direction(blue="random", red=name, seeds=seeds, wins=losses,
                            losses=wins, draws=0,
                            score=losses / (wins + losses),
                            crowns=[1] * losses + [-1] * wins)
        return Pairing(a=name, b="random", mode="greedy", forward=forward,
                       reverse=reverse, score=wins / (wins + losses),
                       seed_correlation=None)

    ratings = fit_ratings([pairing("strong", 75, 25), pairing("weak", 25, 75)])
    assert ratings["strong"].elo > 100.0
    assert ratings["weak"].elo < -100.0
    assert ratings["strong"].elo > ratings["random"].elo > ratings["weak"].elo


# ----------------------------------------------------------- who was played


def test_the_probe_labels_each_anchor_from_the_env_it_actually_played(
        make_env, net):
    """The label follows the environment, not the argument list.

    Two anchors go in; the factory hands both pairings a third opponent
    instead. A probe that names its rows from what it was *handed* reports two
    scores against two opponents it never played, silently, and the rows still
    pass every guard -- which is exactly how "lift" came to mean two
    incompatible things on this project. A probe that reads the name off the
    environment reports one opponent, the one that was really there.
    """
    alpha = Player(name="alpha", kind="random")
    beta = Player(name="beta", kind="random")

    def scores(factory) -> set:
        probe = ladder_probe(factory, [alpha, beta], episodes=2, blocks=8,
                             ratings={}, mode="greedy")
        stats = probe(net)
        return {k for k in stats if k.startswith("ladder_score_")}

    # Straight through: the anchors are who they say they are.
    assert scores(make_env) == {"ladder_score_alpha", "ladder_score_beta"}

    def crossed(opponent=None):
        """Every anchor arrives as somebody else entirely."""
        if getattr(opponent, "opponent_name", None) in ("alpha", "beta"):
            opponent = _passive("gamma")
        return make_env(opponent)

    assert scores(crossed) == {"ladder_score_gamma"}


def test_a_probe_reading_names_both_sides_and_says_which_block_it_played(
        make_env, net):
    """A ladder row is a score, so ``check_lift_is_named``'s lift clause never
    fires on it. The probe therefore has to carry its own name, its opponent's
    reference, and which battles it played -- and it routes its own dict
    through the guard rather than trusting the caller to.
    """
    from cr_sim.train.selfplay import check_lift_is_named

    anchors = [Player(name="random", kind="random")]
    probe = ladder_probe(make_env, anchors, episodes=2, blocks=8,
                         ratings={"random": 0.0, "search-c18h15": 900.0},
                         mode="greedy")
    first, second = probe(net), probe(net)

    assert check_lift_is_named(first) is first
    assert first["eval_opponent"] == "ladder"
    assert first["ladder_opponent_ref"] == "random@random"
    assert math.isfinite(first["ladder_elo"])
    # Expert-relative, without ever having played the expert. That is the
    # whole design: the rating is transitive, so the probe pays for three
    # cheap anchors and reads the expensive gap off the fit.
    assert (first["ladder_elo_vs_expert"]
            == pytest.approx(first["ladder_elo"] - 900.0))
    # Rotating blocks, because the promotion window is three readings and
    # three readings of one fixed seed list share all of their seed luck.
    assert (first["eval_block"], second["eval_block"]) == (0, 1)


def test_a_checkpoint_with_an_unrecognised_parameter_count_is_refused(tmp_path):
    """22 of 42 checkpoints here record neither head nor observation.

    ``load_policy`` defaults those to flat/v1, which is a guess that happens
    to be right for some of them. A wrong head does not raise -- the weights
    load into a network of the right shape and the policy simply plays badly
    -- so a ladder built on a guess rates a network nobody trained.
    """
    assert head_for_parameters(1_838_545) == "flat"
    assert head_for_parameters(1_710_646) == "factored"
    assert head_for_parameters(1_662_214) == "conv"
    assert head_for_parameters(1_713_046) == "factored-stats"

    # The 1,011,921-parameter generation: nine files on this machine, none of
    # which load into the current ActorCritic at all.
    with pytest.raises(ValueError, match="1,011,921"):
        head_for_parameters(1_011_921)

    unstamped = tmp_path / "unstamped.pt"
    torch.save({"state_dict": {"w": torch.zeros(1_710_646)}}, unstamped)
    player = player_from_checkpoint(unstamped, name="unstamped")
    assert player.head == "factored"
    # And it says the head was inferred, so a reader can see which ratings
    # rest on an inference rather than on a recorded field.
    assert player.head_source == "inferred-from-parameter-count"

    strange = tmp_path / "strange.pt"
    torch.save({"state_dict": {"w": torch.zeros(7)}}, strange)
    with pytest.raises(ValueError, match="parameters"):
        player_from_checkpoint(strange)

    stamped = tmp_path / "stamped.pt"
    torch.save({"state_dict": {"w": torch.zeros(7)}, "head": "conv",
                "observation": "v2"}, stamped)
    recorded = player_from_checkpoint(stamped)
    assert (recorded.head, recorded.observation) == ("conv", "v2")
    assert recorded.head_source == "recorded"


def test_the_ladder_table_reader_keeps_the_rating_out_of_the_lift(tmp_path):
    """``watch.read_ladder`` lands dark and still has to be right.

    The progress page cannot render a ladder yet -- that needs a new shape
    beside "flat", "paired" and "arms", and the watcher on port 8899 holds the
    module it was started with, so an edit to watch.py changes nothing until
    somebody restarts it. The reader is landed anyway, and tested anyway,
    because untested code that is switched on later is how the page came to
    claim "no evaluations yet" over a real evaluation.

    What it must not do is hand a rating to anything expecting a lift. They
    are unrelated scales that happen to rank the same players the same way,
    which is exactly what lets the confusion survive long enough to matter.
    """
    from cr_sim.train.watch import read_ladder

    assert read_ladder(tmp_path) is None

    (tmp_path / "ladder.json").write_text(json.dumps({
        "mode": "greedy", "episodes": 400, "observation": "v1",
        "anchor": "random", "prior_sd": 400.0,
        "ratings": [
            {"name": "random", "elo": 0.0, "error": 0.0, "ci_low": 0.0,
             "ci_high": 0.0, "games": 800, "pinned": True},
            {"name": "clone-v1-paired:cloned", "elo": 604.0, "error": 13.0,
             "ci_low": 578.5, "ci_high": 629.5, "games": 800},
            {"name": "broken", "elo": None, "games": 0},
        ],
        "pairings": [{"a": "clone-v1-paired:cloned", "b": "random",
                      "score": 0.97, "games": 800,
                      "seed_correlation": None}],
    }), encoding="utf-8")

    table = read_ladder(tmp_path)
    assert [r["name"] for r in table["ratings"]] == [
        "clone-v1-paired:cloned", "random"], "not sorted by rating"
    # A rating with no number is dropped rather than plotted as zero, which is
    # where a broken measurement and a level-with-random policy look alike.
    assert all(r["name"] != "broken" for r in table["ratings"])
    # An anchor held at its offline value is not a measurement this run made.
    assert table["ratings"][1]["pinned"] is True
    assert table["ratings"][0]["pinned"] is False
    assert table["ratings"][0]["ci"] == [578.5, 629.5]
    # And nothing here is called "lift".
    assert not any("lift" in key for row in table["ratings"] for key in row)
    assert table["pairings"][0]["seed_correlation"] is None


def test_a_verdict_written_by_the_ladder_passes_the_guard(tmp_path):
    """The ladder's own verdict goes through ``write_verdict`` like every
    other measurement, and its rows go through ``check_lift_is_named``.

    Not one ``verdict.json`` on this machine carries ``eval_opponent``; all
    seven would be refused today, and ``report.py`` renders four of them as
    "beats an unnamed opponent". A new metric that repeated that would be
    worse than the old one, because it would look new.
    """
    from cr_sim.train.evaluate import write_verdict
    from cr_sim.train.selfplay import check_lift_is_named

    with pytest.raises(ValueError, match="eval_opponent"):
        write_verdict(tmp_path / "verdict.json",
                      {"ladder_elo": 604.0, "lift": 2.1})

    written = write_verdict(tmp_path / "verdict.json", {
        "ladder_elo": 604.0, "eval_opponent": "ladder",
        "ladder_opponent_ref": "random|c6h8"})
    assert json.loads((tmp_path / "verdict.json").read_text(
        encoding="utf-8")) == written

    with pytest.raises(ValueError, match="which weights"):
        check_lift_is_named({"ladder_elo": 604.0, "eval_opponent": "ladder"})
