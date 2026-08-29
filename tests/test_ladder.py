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
import pathlib

import numpy as np
import pytest
import torch

from cr_sim.api.encoding import NOOP_SLOT
from cr_sim.api.env import CRSimEnv
from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.entity import Team
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


def test_the_reverse_direction_puts_the_other_player_in_control(make_env, net):
    """Two *different* players, because a twin cannot tell a mirror from a
    replay.

    The version of this test that shipped paired one net object against
    itself and asserted the pairing came out at 0.5. That holds arithmetically
    for any implementation -- ``(s + (1 - s)) / 2`` is 0.5 whatever s is --
    so deleting the mirror entirely left the whole file green. Measured
    against the real defect: with ``_play_direction(make_env, b, a, ...)``
    changed to replay ``a`` as the controlled side, a bot that beats the
    random control 8-0 rates +40 Elo above it instead of +468.

    So the claim is made on an asymmetric pairing, where a replay and a mirror
    give different numbers: the reverse direction must hand control to the
    *other* player, the two directions must therefore be different battles,
    and the mirror-average of a lopsided pairing must not land on 0.5.
    """
    a = Player(name="alpha", kind="net", net=net)
    b = Player(name="beta", kind="random")
    pairing = play_pairing(make_env, a, b, seeds=[1, 2, 6, 8], mode="greedy")

    # Which player the environment controls, read off the direction rather
    # than off the argument list.
    assert (pairing.forward.blue, pairing.forward.red) == ("alpha", "beta")
    assert (pairing.reverse.blue, pairing.reverse.red) == ("beta", "alpha")

    # These seeds are decisive on purpose: a pairing that drew every battle
    # scores 0.5 in every direction and everything below holds over a mirror
    # that does nothing.
    assert any(c != 0 for c in pairing.forward.crowns)
    assert pairing.forward.score == 0.75 and pairing.reverse.score == 0.25
    assert pairing.score == (pairing.forward.score
                             + 1.0 - pairing.reverse.score) / 2.0
    # A lopsided pairing does not come out level. A reverse direction that
    # replayed the forward one would put every pairing here, whatever the
    # players did.
    assert pairing.score == 0.75


def test_a_mirrored_pairing_of_identical_nets_scores_exactly_one_half(
        make_env, net):
    """The corollary, and only the corollary.

    ``CRSimEnv.step`` applies the controlled side's action and only then calls
    ``_opponent_move()``, inside every frame-skip window, all match. So the
    controlled side moves first every time, and a single-direction score is
    that advantage plus the policy. Measured on two independent nets at n=40,
    the controlled side scored 0.550 and 0.600 against its own weights.

    The mirror swaps which player the environment controls, so both halves of
    the bias cancel exactly rather than approximately -- which is why this
    asserts equality against 0.5 and not a tolerance. It cannot stand alone:
    see the test above for why.
    """
    a = Player(name="twin-a", kind="net", net=net)
    b = Player(name="twin-b", kind="net", net=net)
    pairing = play_pairing(make_env, a, b, seeds=[5, 6, 7, 8], mode="greedy")

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

    from cr_sim.train.evaluate import evaluation_seeds

    anchors = [Player(name="random", kind="random")]
    probe = ladder_probe(make_env, anchors, episodes=2, blocks=8,
                         ratings={"random": 300.0, "search-c18h15": 900.0},
                         mode="greedy", ratings_source="runs/x/ladder.json")
    first, second = probe(net), probe(net)

    assert check_lift_is_named(first) is first
    assert first["eval_opponent"] == "ladder"
    assert first["ladder_opponent_ref"] == "random@random"

    # The rating is the fit's answer, not a constant that happens to be
    # finite. math.isfinite(elo) and vs_expert == elo - 900 are both satisfied
    # by zero, so hardcoding the probe's rating to 0.0 -- which throws away
    # the fit and the anchor's pinned value together -- left this file green.
    # Replayed here: the same battles, the same pinning, the same arithmetic.
    block_seeds = evaluation_seeds(2, block=0, seed=12345, blocks=8)
    replay = play_pairing(make_env, Player(name="policy", kind="net", net=net),
                          anchors[0], seeds=block_seeds, mode="greedy")
    fitted = fit_ratings([replay], pinned={"random": 300.0})
    assert first["ladder_elo"] == pytest.approx(fitted["policy"].elo)
    assert first["ladder_score_random"] == replay.score
    # And it sits on the anchor's scale rather than floating free: a policy
    # scoring above 0.5 against an anchor pinned at +300 rates above +300.
    assert replay.score > 0.5 and first["ladder_elo"] > 300.0
    assert math.isfinite(first["ladder_elo"])

    # What the scale rests on, on the row. Without it two ladder_elo values
    # fitted against differently-pinned anchors are indistinguishable -- the
    # same battles, 377 Elo apart, every other field identical.
    assert first["ladder_pinned"] == {"random": 300.0}
    assert first["ladder_ratings_source"] == "runs/x/ladder.json"

    # Expert-relative, without ever having played the expert. That is the
    # whole design: the rating is transitive, so the probe pays for three
    # cheap anchors and reads the expensive gap off the fit.
    assert (first["ladder_elo_vs_expert"]
            == pytest.approx(first["ladder_elo"] - 900.0))
    # Rotating blocks, because the promotion window is three readings and
    # three readings of one fixed seed list share all of their seed luck.
    assert (first["eval_block"], second["eval_block"]) == (0, 1)


def test_the_probe_refuses_an_anchor_its_rating_table_does_not_name():
    """An unrated anchor is pinned at 0 Elo -- level with a uniform random
    agent -- and every recorded field on the row is identical either way.

    Measured on one real edge from runs/agent-ladder-v1 (headablate-factored
    against the v1 clone, a-score 0.408 over 60 games), fitted exactly as the
    probe fits it: the clone anchor pinned at its offline +382 gives
    ladder_elo +313.6 and ladder_elo_vs_expert -586.4; the same battles with
    that anchor absent from the table give -63.6 and -963.6. Both rows carry
    the same ladder_opponent, ladder_opponent_ref, ladder_mode and
    eval_opponent, and both pass check_lift_is_named.

    It is one keystroke away: parse_player("runs/x/cloned.pt") is named
    "x:cloned" and parse_player("clone=runs/x/cloned.pt") is named "clone",
    and only one of those matches a ladder.json.

    Eleven lines below the pinning, the probe already reports the *expert* as
    absent rather than as zero for exactly this reason.
    """
    def unused_env(opponent=None):
        raise AssertionError("no battle should be played")

    anchors = [Player(name="clone", kind="random")]
    with pytest.raises(ValueError, match="no rating in the table"):
        ladder_probe(unused_env, anchors, episodes=2,
                     ratings={"clone-v1-paired:cloned": 382.0})

    # An empty table is the unrated case and stays allowed: the probe then
    # rates against its anchors alone, and says so by pinning them all at 0.
    assert ladder_probe(unused_env, anchors, episodes=2, ratings={}) is not None


def test_a_fit_that_cannot_reach_its_anchor_is_refused():
    """fit_ratings(anchor="random") pinned nothing when random was not in the
    graph, and every caller went on saying "anchored at random = 0".

    Measured: a roster of two checkpoints plus the clone, no random anchor,
    printed "headablate-conv -71 ... vs random 0.399" under run_ladder's own
    "vs random" header -- i.e. this checkpoint loses to a uniform random agent
    -- while the same checkpoint on the same thirty seeds scores 0.665 against
    random and rates +200 in a ladder that contains it. ladder.json recorded
    an anchor of "random" and pinned: false on all three, and verdict.json
    said "Elo, anchored at random = 0". Every rating in that file was 400+ Elo
    off the scale it claimed and nothing said so.
    """
    with pytest.raises(ValueError, match="not among the players"):
        fit_ratings([_lopsided("conv", "clone")], anchor="random")

    # The escape hatch is naming what is pinned, which is what the in-run
    # probe does: a fit with an explicit pinning is on a scale the caller can
    # describe.
    rated = fit_ratings([_lopsided("conv", "clone")], anchor="random",
                        pinned={"clone": 382.0})
    assert rated["clone"].pinned and rated["clone"].elo == 382.0
    assert rated["conv"].elo > 382.0


def test_a_search_players_reference_records_the_budget_it_actually_took(tmp_path):
    """Player.ref stamped the requested policy-candidate count, and SearchBot
    clamps it at construction to leave the random floor intact.

    So "search-c6h4-p5@ckpt" and "search-c6h4-p4@ckpt" build the identical bot
    and produced two different refs, and at fourteen candidates p10, p11, p12
    and p14 are one bot wearing four names -- which, since Player equality
    follows the SearchBotConfig, would enter one ladder as several entrants
    splitting the same bot's games. scripts/make_demos.py already stamps its
    shards with the clamped value, so the two disagreed about one run.
    """
    from cr_sim.train.ladder import parse_player
    from cr_sim.train.scripted import SearchBot

    checkpoint = tmp_path / "proposer.pt"
    checkpoint.write_bytes(b"not a real checkpoint, only bytes to hash")

    asked = parse_player(f"search-c6h4-p5@{checkpoint}")
    clamped = parse_player(f"search-c6h4-p4@{checkpoint}")
    assert asked.ref == clamped.ref

    # And the number in the ref is the one the bot takes, read off a built bot
    # rather than off the spec.
    bot = SearchBot(Team.BLUE, asked.search)
    assert bot.config.policy_candidates == 4
    assert asked.ref.endswith(f"p{bot.config.policy_candidates}")

    # A bot that really does take a different budget still gets its own name.
    assert parse_player(f"search-c14h15-p9@{checkpoint}").ref != asked.ref


def test_the_in_run_probe_loads_its_anchors_weights(make_env, net, tmp_path):
    """run.py handed ``ladder_probe`` unloaded Players, so ``--probe ladder``
    against any checkpoint anchor crashed at the first evaluation.

    ``Player.load`` is what puts a network behind a checkpoint anchor, and
    scripts/run_ladder.py calls it while the in-run path never did. Measured:
    ``--probe ladder --ladder-anchor checkpoints/headablate-flat.pt
    --eval-every 1`` trained to the first probe and died in
    ``FrozenOpponent._snapshot`` on ``'NoneType' object has no attribute
    'to'``; a ``search-c6h4-p3@ckpt`` anchor died one layer up in
    ``_proposer_of``. Only "random" and an unproposed "search-cXhY" ever
    worked, which is not the feature -- the anchors this exists for are the
    clone and the guided expert -- and every ladder_probe test in this file
    used a random anchor, so fifteen green tests covered the probe without
    once exercising its documented input.
    """
    from cr_sim.train.run import _ladder_anchors

    checkpoint = tmp_path / "anchor.pt"
    torch.save({"state_dict": net.state_dict(), "head": "flat",
                "observation": "v1"}, checkpoint)

    shape = make_env(None)
    shape.reset(seed=0)
    anchors = _ladder_anchors(
        [str(checkpoint), f"search-c4h4-p2@{checkpoint}", "random"], shape)

    assert [a.kind for a in anchors] == ["net", "search", "random"]
    # The checkpoint anchor's own weights, and the search anchor's proposer.
    assert anchors[0].net is not None
    assert anchors[1].net is not None
    assert anchors[2].net is None, "a random anchor has no weights to load"

    # And the probe can actually play the one that used to crash it.
    reading = ladder_probe(make_env, anchors[:1], episodes=1,
                           ratings={})(net)
    assert math.isfinite(reading["ladder_elo"])
    assert reading["ladder_opponent_ref"] == f"anchor@{checkpoint}"

    # The default is still a single random anchor.
    assert [a.kind for a in _ladder_anchors(None, shape)] == ["random"]


def test_a_rating_table_from_another_measurement_is_refused(tmp_path):
    """``--ladder-ratings`` read only the names and the Elo out of a
    ladder.json and ignored what the file says it measured.

    The probe is hardcoded to greedy, and a sampled table rates the same
    weights on a different scale: measured on the same players and the same
    seed block, the v1 clone is +393.0 greedy and +183.7 sampled -- 209 Elo
    apart, with the field reordered as well -- and feeding each into the
    probe's own arithmetic on one greedy edge gives ladder_elo +337 against
    +129, with both rows stamping ``ladder_mode: greedy``. run_ladder's own
    --mode help says it: "sampled is a different policy and needs its own
    ladder, its own ratings".
    """
    from cr_sim.train.run import _ladder_ratings

    path = tmp_path / "ladder.json"

    def write(**overrides):
        payload = {"mode": "greedy", "observation": "v1", "tower_level": 5,
                   "ratings": [{"name": "random", "elo": 0.0},
                               {"name": "clone-v1-paired:cloned",
                                "elo": 393.0}]}
        payload.update(overrides)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def load():
        return _ladder_ratings(path, mode="greedy", observation="v1",
                               tower_level=5)

    write()
    assert load() == {"random": 0.0, "clone-v1-paired:cloned": 393.0}

    write(mode="sampled")
    with pytest.raises(SystemExit, match="sampled"):
        load()

    write(observation="v2")
    with pytest.raises(SystemExit, match="observation"):
        load()

    write(tower_level=11)
    with pytest.raises(SystemExit, match="tower level"):
        load()

    # A table written before the level was recorded cannot answer the
    # question, and refusing it would make the flag unusable against every
    # table on this machine. Checked where it is present, and only there.
    write()
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["tower_level"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load() == {"random": 0.0, "clone-v1-paired:cloned": 393.0}


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


def _write_entrant(path, env, seed: int) -> None:
    import torch as _torch

    _torch.manual_seed(seed)
    built = ActorCritic(net_config_for(env, head="flat"))
    _torch.save({"state_dict": built.state_dict(), "head": "flat",
                 "observation": "v1"}, path)


def _run_ladder(tmp_path, name, entrants, world):
    """One tiny offline ladder, end to end through the script."""
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import scripts.run_ladder as run_ladder

    argv = ["--name", name, "--episodes", "1", "--workers", "0",
            "--tower-level", "5", "--tps", "20", "--frame-skip", "40",
            "--match-seconds", "40", "--anchor", "random",
            "--out", str(tmp_path)]
    for spec in entrants:
        argv += ["--entrant", spec]
    assert run_ladder.main(argv) == 0
    out = tmp_path / name
    return (json.loads((out / "verdict.json").read_text(encoding="utf-8")),
            json.loads((out / "ladder.json").read_text(encoding="utf-8")),
            [json.loads(line) for line
             in (out / "metrics.jsonl").read_text(
                 encoding="utf-8").splitlines()],
            json.loads((out / "arms.json").read_text(encoding="utf-8")))


def test_the_offline_ladder_never_writes_a_lift_it_cannot_name(
        tmp_path, make_env, world):
    """The verdict paired one player's Elo with another player's lift, under
    an eval_opponent that named neither measurement.

    Measured: runs/audit-ladder-greedy/verdict.json records
    ``eval_opponent: 'ladder', ladder_player: 'headablate-factored',
    ladder_elo: 419.41, lift: 0.78053`` -- and that lift is byte-identical to
    arms.json[0], which is ``headablate-conv`` with ``eval_opponent:
    'random'``, the worst-rated of the four players. report.py rendered it as
    "100 paired battles against ladder put the lift at +0.781 sd". The same
    defect is on disk in runs/agent-ladder-v1, whose ``ladder_player`` is the
    clone and whose ``lift`` is headablate-flat's.

    So: flatten a lift only where there is exactly one arm to flatten, and
    name the player and the opponent it belongs to.
    """
    probe = make_env(None)
    probe.reset(seed=0)
    alpha = tmp_path / "alpha.pt"
    beta = tmp_path / "beta.pt"
    _write_entrant(alpha, probe, seed=11)
    _write_entrant(beta, probe, seed=12)

    verdict, table, rows, arms = _run_ladder(
        tmp_path, "one-arm", [f"alpha={alpha}"], world)

    # One arm, so the lift is flattened -- and says whose it is and who it was
    # played against, neither of which the top-level fields can carry.
    assert verdict["lift"] == arms[0]["lift"]
    assert verdict["lift_player"] == arms[0]["name"] == "alpha"
    assert verdict["lift_opponent"] == arms[0]["eval_opponent"] == "random"
    assert verdict["eval_opponent"] == "ladder"
    # The arena the ratings were fitted in, so --ladder-ratings can check it.
    assert table["tower_level"] == 5

    # Two arms, and the file declines to represent them with one number.
    two, _, _, arms_two = _run_ladder(
        tmp_path, "two-arms", [f"alpha={alpha}", f"beta={beta}"], world)
    assert len(arms_two) == 2
    assert "lift" not in two and "lift_player" not in two
    assert "arms.json" in two["note"]

    # And the guard underneath: a verdict carrying both scales without saying
    # whose lift it is, is refused outright.
    from cr_sim.train.evaluate import write_verdict

    with pytest.raises(ValueError, match="lift_player"):
        write_verdict(tmp_path / "bad.json", {
            "eval_opponent": "ladder", "ladder_elo": 419.4,
            "ladder_player": "headablate-factored", "lift": 0.781})


def test_an_offline_ladders_rows_keep_the_rating_off_the_pairings(
        tmp_path, make_env, world):
    """A whole-graph rating rode along on every per-direction row.

    In runs/agent-ladder-v1, headablate-flat's +309.4 appears on a row naming
    ``random`` and on another naming ``clone-v1-paired:cloned`` -- neither of
    which produced it; it is a Bradley-Terry fit over four pairings. Four of
    the twelve rows read ``player=random elo=+0.0``, a pinned constant in the
    same field as a fitted rating, because ``Rating.pinned`` was dropped when
    the row was written.
    """
    probe = make_env(None)
    probe.reset(seed=0)
    alpha = tmp_path / "alpha.pt"
    _write_entrant(alpha, probe, seed=11)

    _, _, rows, _ = _run_ladder(tmp_path, "rows", [f"alpha={alpha}"], world)

    scored = [r for r in rows if "ladder_score" in r]
    rated = [r for r in rows if "ladder_elo" in r]
    assert scored and rated

    # A pairing row carries its own direction's score and no rating at all.
    assert all("ladder_elo" not in r for r in scored)
    assert {r["ladder_opponent"] for r in scored} == {"random", "alpha"}

    # A rating row names the whole roster, because that is what the fit was
    # over, and says whether the number was fitted or held fixed.
    assert {r["ladder_player"] for r in rated} == {"alpha", "random"}
    anchor = next(r for r in rated if r["ladder_player"] == "random")
    player = next(r for r in rated if r["ladder_player"] == "alpha")
    assert anchor["ladder_elo_pinned"] is True and anchor["ladder_elo"] == 0.0
    assert player["ladder_elo_pinned"] is False
    assert player["ladder_pinned"] == {"random": 0.0}
    assert player["ladder_opponent_ref"] == "random"

    # And the page counts them: a ladder run is an evaluated run.
    from cr_sim.train.watch import summarise

    assert summarise(rows)["evaluations"] == len(rated)


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

    # And what the rating was pinned to. An Elo is only a number relative to
    # whatever the fit held fixed: the same battles with one anchor pinned at
    # its offline +382 and with that anchor absent from the table -- pinned at
    # 0, level with a uniform random agent -- came out 377 points apart with
    # every other field on the row identical.
    with pytest.raises(ValueError, match="pinned"):
        check_lift_is_named({"ladder_elo": 604.0, "eval_opponent": "ladder",
                             "ladder_opponent_ref": "random@random"})
    named = {"ladder_elo": 604.0, "eval_opponent": "ladder",
             "ladder_opponent_ref": "random@random",
             "ladder_pinned": {"random": 0.0}}
    assert check_lift_is_named(named) is named
