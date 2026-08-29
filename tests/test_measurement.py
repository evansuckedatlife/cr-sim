"""Whether a measurement says what it was measured against.

"Lift" was reported on two incompatible scales for most of this project's
life. The in-run probe played the policy against an opponent that never plays
a card; the large paired verdicts played it against one that spends its elixir
on legal placements; both were written down as "lift" and compared to each
other. The random control wins 92% of the idle matches and 26% of the random
ones, so the two numbers never lived on the same scale and no comparison
between them meant anything.

A comment does not stop that happening again. Refusing to write the row does,
and reading the opponent off the environment rather than taking it as an
argument stops a caller labelling a measurement with an opponent it did not
actually face.

The other half is whether a measurement can still move. Lift against the
uniform random control is saturated -- the search expert beats that control
100-0, +2.716 sd over 40 battles -- so a better policy has nowhere left to
register and the metric fails silently, reporting a high number while
measuring nothing. The rest of this file covers the evaluation that faces the
expert instead, and the rotating seed blocks that stop three consecutive
readings from being three readings of the same forty battles' luck.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cr_sim.api.encoding import NOOP_SLOT
from cr_sim.api.env import CRSimEnv
from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData

from .test_data_pipeline import BUILD

DECK = ("Knight", "Musketeer", "Cannon", "Skeletons",
        "IceSpirits", "Log", "Fireball", "Goblins")


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


def test_a_lift_cannot_be_recorded_without_naming_its_opponent():
    """"Lift" was reported on two incompatible scales for most of this
    project's life: the in-run probe faced an opponent that never plays a card
    while the large paired verdicts faced a random one, and the two numbers
    were compared to each other. The control wins 92% of the idle matches and
    26% of the random ones, so they never lived on the same scale.

    A comment does not stop that happening again; refusing to write the row
    does.
    """
    from cr_sim.train.selfplay import check_lift_is_named

    with pytest.raises(ValueError, match="eval_opponent"):
        check_lift_is_named({"updates": 3, "eval_lift_sd": 0.42})
    with pytest.raises(ValueError, match="eval_opponent"):
        check_lift_is_named({"updates": 3, "eval_lift_sd": 0.42, "eval_opponent": ""})

    named = {"updates": 3, "eval_lift_sd": 0.42, "eval_opponent": "random",
             "eval_reward": "projected:elixir=0.3,horizon_seconds=3,tower=1"}
    assert check_lift_is_named(named) is named
    # A row with no lift on it is not a measurement and needs no label.
    plain = {"updates": 4, "entropy": 3.1}
    assert check_lift_is_named(plain) is plain


def test_the_opponent_a_probe_faced_is_read_off_the_environment(world):
    """Not taken as an argument. A caller cannot label a measurement with an
    opponent it did not actually play."""
    from cr_sim.train.run import _random_opponent
    from cr_sim.train.selfplay import opponent_name

    assert opponent_name(_env(world)) == "idle"
    assert opponent_name(_env(world, opponent_policy=_random_opponent(0))) == "random"

    def anonymous(observation, mask):
        return (NOOP_SLOT, 0, 0)

    # Unknown rather than guessed: a wrong label is worse than an absent one.
    assert opponent_name(_env(world, opponent_policy=anonymous)) == "unknown"


# ------------------------------------------------ a yardstick with room left


def _tiny(world, **kwargs):
    """A battle short enough to run several of in a test.

    Forty seconds at two-second decisions. Nothing here is a measurement --
    these check that the machinery runs end to end and that its guarantees
    hold, which is what a test can afford. A real evaluation against the
    expert is two orders of magnitude more simulation.

    Not shorter, and that is measured rather than cautious: at a twenty-second
    match the searching opponent placed nothing at all, and every assertion
    below then passes against a board where neither side ever played -- two
    seeds produce the same battle hash and the tests confirm nothing.
    """
    kwargs.setdefault("frame_skip", 40)
    kwargs.setdefault("max_ticks", 20 * 40)
    return _env(world, **kwargs)


def _cheap_expert():
    """A search opponent thinned to what a test can pay for.

    Four candidates eight seconds forward, against eighteen at fifteen. It
    plays much worse than the measured expert and that is fine: every test
    below is about the evaluation's structure, not about how strong the
    opponent is.

    The horizon is the one number that cannot be cut freely. A card takes
    several seconds to walk anywhere, so a short projection scores it before
    it has done anything and every placement looks worse than waiting: at two
    seconds this bot placed nothing in a whole match, which is the idle
    opponent wearing the label "search".
    """
    from cr_sim.train.evaluate import search_opponent
    from cr_sim.train.scripted import SearchBotConfig

    return search_opponent(SearchBotConfig(candidates=4, horizon_seconds=8.0))


def test_the_search_opponent_names_the_scale_it_measures_on(world):
    """A lift against the expert and a lift against random are different
    numbers, not a better and a worse one. The opponent has to be able to say
    so, or the two end up in the same column."""
    from cr_sim.train.selfplay import opponent_name

    assert opponent_name(_env(world, opponent_policy=_cheap_expert())) == "search"


def test_the_search_opponent_actually_plays(world):
    """The bot passes when it cannot see the board, and the environment only
    hands it the board when the callable declares ``wants_battle``. Lose that
    flag and the expert opponent silently becomes an opponent that never plays
    a card -- which is the idle scale this project already mistook for a real
    one, wearing the label "search".

    So the flag is not asserted directly; the opponent is watched through a
    spy that copies whatever the real one declares, and the test is that it
    plays.
    """
    inner = _cheap_expert()
    chosen: list[tuple[int, int, int]] = []

    def spy(observation, mask, battle=None):
        action = tuple(int(v) for v in inner(observation, mask, battle))
        chosen.append(action)
        return action

    spy.wants_battle = getattr(inner, "wants_battle", False)

    env = _tiny(world, opponent_policy=spy)
    env.reset(seed=4)
    for _ in range(8):
        _, _, terminated, truncated, _ = env.step((NOOP_SLOT, 0, 0))
        if terminated or truncated:
            break
    assert chosen, "the opponent was never consulted"
    assert any(slot != NOOP_SLOT for slot, _, _ in chosen), (
        "the search opponent never placed a card -- it is being asked without "
        "the battle and has fallen back to passing")


def test_the_search_opponent_is_a_function_of_the_battle_seed(world):
    """The subtle way a paired evaluation stops being paired.

    The bot samples its candidate placements, so what it plays depends on how
    far its generator has been advanced -- a count of every decision in every
    episode before this one. Two arms diverge on their first different move,
    so from the second episode onward they would face experts drawing
    different candidates on the same seed, and the two arms would no longer be
    playing the same battle at all. Rebuilt per battle, the opponent depends
    on the seed and nothing else.
    """
    def play(env, seed):
        env.reset(seed=seed)
        while True:
            _, _, terminated, truncated, info = env.step((NOOP_SLOT, 0, 0))
            if terminated or truncated:
                return info["hash"]

    fresh = play(_tiny(world, opponent_policy=_cheap_expert()), 5)

    warmed = _tiny(world, opponent_policy=_cheap_expert())
    other = play(warmed, 99)
    assert play(warmed, 5) == fresh, (
        "the same seed played out differently after another battle -- the "
        "opponent is carrying state across episodes, so two arms of a paired "
        "evaluation face different experts")
    # And the check is sensitive to anything at all: two seeds do differ.
    assert other != fresh


def test_a_verdict_cannot_be_written_without_naming_its_opponent(tmp_path):
    """The same guard check_lift_is_named puts on a metrics row, for the file
    that outlives the run. A bare lift on disk, read later beside one measured
    against a different opponent, is worse than no number."""
    from cr_sim.train.evaluate import write_verdict

    with pytest.raises(ValueError, match="eval_opponent"):
        write_verdict(tmp_path / "verdict.json", {"episodes": 40, "lift": 0.42})
    assert not (tmp_path / "verdict.json").exists()

    named = {"episodes": 40, "lift": 0.42, "eval_opponent": "search"}
    write_verdict(tmp_path / "verdict.json", named)
    assert json.loads((tmp_path / "verdict.json").read_text())["eval_opponent"] == "search"


def test_an_evaluation_against_the_expert_runs_and_reports_both_ways_of_playing(world):
    """End to end on two battles, which is a smoke test and not a result.

    What it pins is the shape: the control and both arms play the identical
    seeds, greedy and sampled each get their own interval, and the flattened
    headline the report reads is one of them rather than an average of the
    two. The clone is +1.623 greedy and +0.709 sampled, so a change can leave
    the argmax untouched while moving the distribution around it -- averaging
    them would hide exactly the thing worth seeing.
    """
    from cr_sim.train.evaluate import evaluate_paired, evaluation_seeds
    from cr_sim.train.nets import ActorCritic, net_config_for

    seeds = evaluation_seeds(2, block=0)
    probe = _tiny(world, opponent_policy=_cheap_expert())
    probe.reset(seed=0)
    net = ActorCritic(net_config_for(probe))
    net.eval()

    verdict = evaluate_paired(
        lambda: _tiny(world, opponent_policy=_cheap_expert()),
        net, episodes=2, seeds=seeds)

    assert verdict["eval_opponent"] == "search"
    assert verdict["episodes"] == 2 and verdict["seeds"] == seeds
    for mode in ("greedy", "sampled"):
        arm = verdict[mode]
        assert set(arm) >= {"win", "loss", "draw", "lift", "ci_low", "ci_high"}
        assert arm["ci_low"] <= arm["lift"] <= arm["ci_high"]
    # The headline is one of the arms, not a blend of them, and it says which.
    assert verdict["mode"] in ("greedy", "sampled")
    assert verdict["lift"] == verdict[verdict["mode"]]["lift"]
    assert verdict["lift"] == max(verdict["greedy"]["lift"], verdict["sampled"]["lift"])
    # The report and the runs page read these flat keys and nothing else.
    assert set(verdict) >= {"lift", "ci_low", "ci_high", "win", "loss", "episodes"}


def test_the_greedy_arm_is_reproducible(world):
    """Greedy is the arm a difference should be readable in: it has no
    sampling noise of its own, so on fixed seeds against a fixed opponent it
    is the same battle twice. If this ever stops holding, a greedy lift has
    picked up a source of variance nobody has accounted for."""
    from cr_sim.train.evaluate import evaluate, evaluation_seeds
    from cr_sim.train.nets import ActorCritic, net_config_for

    seeds = evaluation_seeds(2, block=0)
    probe = _tiny(world)
    probe.reset(seed=0)
    net = ActorCritic(net_config_for(probe))
    net.eval()

    first = evaluate(_tiny(world), net, episodes=2, seeds=seeds, greedy=True)
    second = evaluate(_tiny(world), net, episodes=2, seeds=seeds, greedy=True)
    assert first["returns"] == second["returns"]
    assert first["crowns"] == second["crowns"]


def test_the_sampled_arm_is_reproducible_when_it_owns_its_stream(world):
    """The sampled arm draws from torch's global generator, which nothing in
    an evaluation seeds. Two identical sampled evaluations of the same
    checkpoint on the same seeds measured +0.583 and +0.581 sd -- so a
    difference smaller than that between two runs was never a difference.

    ``generator`` is opt-in rather than a seed on the global stream: the
    in-run probe calls ``evaluate`` mid-training, and seeding globally there
    would reset the sampling of the run being measured.
    """
    import torch

    from cr_sim.train.evaluate import evaluate, evaluation_seeds
    from cr_sim.train.nets import ActorCritic, net_config_for

    seeds = evaluation_seeds(2, block=0)
    probe = _tiny(world)
    probe.reset(seed=0)
    net = ActorCritic(net_config_for(probe))
    net.eval()

    def sampled():
        return evaluate(_tiny(world), net, episodes=2, seeds=seeds, greedy=False,
                        generator=torch.Generator().manual_seed(11))["returns"]

    assert sampled() == sampled()


# ------------------------------------------------------- rotating the seeds


def test_block_zero_is_the_seed_list_every_existing_measurement_used():
    """Rotation must not quietly invalidate the numbers already recorded.

    The blocks are consecutive draws from one generator, so block 0 is
    byte-identical to the single draw the fixed probe made. Every lift on this
    project was measured on it.
    """
    from cr_sim.train.evaluate import evaluation_seeds

    existing = [int(s) for s in
                np.random.default_rng(12345).integers(0, 2**31 - 1, 40)]
    assert evaluation_seeds(40, block=0) == existing


def test_consecutive_blocks_share_no_battle():
    """The flaw the rotation exists for.

    The in-run probe replayed the same forty seeds every reading, and the
    runner promotes on the mean of the last three. A window over three
    readings of the *same* battles is forty battles, not a hundred and twenty:
    it averages away the reading-to-reading noise and re-selects the shared
    seed-level luck three times over. That is the failure the rolling mean was
    introduced to fix, one level down.
    """
    from cr_sim.train.evaluate import EVAL_BLOCKS, evaluation_seeds

    blocks = [evaluation_seeds(40, block=b) for b in range(EVAL_BLOCKS)]
    seen: set[int] = set()
    for block in blocks:
        assert len(block) == 40
        assert not seen & set(block), "two blocks replay the same battle"
        seen |= set(block)
    # Fixed, not random: two runs have to evaluate on the same battles or they
    # are not comparable, which is the whole reason the seeds were fixed.
    assert evaluation_seeds(40, block=3) == blocks[3]
    # And it wraps, so a run longer than the cycle keeps rotating rather than
    # running out of blocks.
    assert evaluation_seeds(40, block=EVAL_BLOCKS) == blocks[0]


def test_the_rotating_probe_changes_battles_between_readings(world):
    """A drop-in for the fixed probe: same keys, same meanings, different
    seeds each reading.

    ``eval_lift_sd`` stays the sampled arm, because that is what the old probe
    measured and what the runner's promotion reads -- changing which arm it
    names would move the scale without changing the name.
    """
    from cr_sim.train.evaluate import evaluation_seeds, rotating_probe

    probe = rotating_probe(lambda: _tiny(world), episodes=2, blocks=2)
    readings = [probe(None) for _ in range(3)]

    assert [r["eval_block"] for r in readings] == [0, 1, 0]
    assert evaluation_seeds(2, block=0, blocks=2) != evaluation_seeds(2, block=1, blocks=2)
    # The control is paired per block, so a reading is only ever differenced
    # against a control that played the same battles.
    assert readings[0]["control_return"] == readings[2]["control_return"]
    assert readings[0]["control_return"] != readings[1]["control_return"]
    # Every key the fixed probe emits, so cr_sim.train.run needs no other
    # change than the name at the call site.
    assert set(readings[0]) >= {
        "eval_return", "eval_win", "control_return", "control_win",
        "eval_lift_sd", "eval_opponent", "eval_episodes",
    }
    from cr_sim.train.selfplay import check_lift_is_named

    for reading in readings:
        check_lift_is_named(reading)


def test_the_rotating_probe_can_report_the_argmax_separately(world):
    """Off by default -- it doubles the evaluation cost, which is pure
    overhead in a training run. On, because a change can leave the argmax
    untouched and still move the distribution around it: the clone is +1.623
    greedy and +0.709 sampled, and reading one as the other is how a previous
    run was misread."""
    from cr_sim.train.evaluate import rotating_probe

    assert "eval_lift_sd_greedy" not in rotating_probe(
        lambda: _tiny(world), episodes=2, blocks=2)(None)
    both = rotating_probe(lambda: _tiny(world), episodes=2, blocks=2, greedy=True)(None)
    assert "eval_lift_sd_greedy" in both and "eval_lift_sd" in both


# ------------------------------------------- whose crowns, and whose name


def test_crowns_are_read_from_the_agents_own_side(world):
    """``Result["crowns"]`` used to be blue's, whoever the agent was.

    ``returns`` has always been team-relative -- it comes from
    ``_shaped_value(battle, self.team, ...)`` -- and this did not, so an
    environment built with ``team=RED`` reported the *opponent's* crown
    difference beside the agent's own return. Opposite signs, no error, no
    warning. Measured on twelve seeds before the fix: mean return -0.083
    against mean crowns +0.083, agreeing in sign on 0% of the decisive
    battles.

    Nothing colour-balanced can be built on top of that, which is why it is
    the first step of the ladder and not a footnote to it.
    """
    from cr_sim.engine.entity import Team
    from cr_sim.train.evaluate import evaluate, evaluation_seeds
    from cr_sim.train.run import _random_opponent

    # Ninety seconds at level five, not the forty-second toy the rest of this
    # file uses: at the default tower level a short match draws every battle,
    # and a battle with no crowns in it cannot show whose crowns were counted.
    seeds = evaluation_seeds(6)
    readings = {}
    for team in (Team.BLUE, Team.RED):
        env = _env(world, team=team, frame_skip=30, max_ticks=20 * 90,
                   tower_level=5, opponent_policy=_random_opponent(11))
        readings[team] = evaluate(env, None, episodes=len(seeds), seeds=seeds)

    # The bug is invisible from blue's side, which is why it survived: every
    # existing number on this project was measured on a team=BLUE env.
    for team, result in readings.items():
        decisive = [(r, c) for r, c in zip(result["returns"], result["crowns"])
                    if c != 0]
        assert decisive, f"team={team.name} drew every battle; nothing measured"
        for value, crowns in decisive:
            assert (value > 0) == (crowns > 0), (
                f"team={team.name}: a return of {value:+.3f} beside "
                f"{crowns:+d} crowns")

    red = readings[Team.RED]
    assert (sum(red["returns"]) > 0) == (sum(red["crowns"]) > 0)


def test_a_frozen_opponent_can_be_named_and_can_play_the_argmax(world, tmp_path):
    """Two blockers, both in one ``__slots__`` tuple.

    Without ``opponent_name`` the class had no ``__dict__`` either, so setting
    the attribute raised ``AttributeError: ... no __dict__ for setting new
    attributes`` and every frozen opponent reported as "unknown". A
    checkpoint-vs-checkpoint ladder could not name its own opponent, so no row
    it wrote could pass ``check_lift_is_named`` and no verdict could pass
    ``write_verdict``.

    And without a real ``greedy`` flag the closest thing available was
    ``temperature=1e-3``, which is *nearly* argmax. The whole argument for a
    greedy ladder is that it reproduces exactly, and near-argmax still samples
    between two logits within 1e-3 of each other -- which are precisely the
    decisions a close pairing turns on.
    """
    import numpy as np
    import torch

    from cr_sim.train.evaluate import write_verdict
    from cr_sim.train.selfplay import FrozenOpponent, opponent_name

    nvec = (5, 4, 5)
    width = nvec[1] * nvec[2]

    class Undecided(torch.nn.Module):
        """Two actions a thousandth of a logit apart, and nothing else close.

        A real network produces ties like this routinely; this one produces
        them on purpose, so the difference between "argmax" and "nearly
        argmax" is a fact about the test rather than about the weights it
        happened to draw.
        """

        def __init__(self) -> None:
            super().__init__()
            self.unused = torch.nn.Parameter(torch.zeros(1))
            row = torch.full((1, nvec[0] * width), -50.0)
            row[0, 5] = 0.0
            row[0, 9] = -1e-4
            self.register_buffer("row", row)

        def policy_logits(self, grid, vector, mask):
            return self.row

    observation = {"grid": np.zeros((1, 1, 1), dtype=np.float32),
                   "vector": np.zeros(1, dtype=np.float32)}
    mask = np.ones(nvec, dtype=bool)

    named = FrozenOpponent(Undecided(), nvec, name="pool:gen3", greedy=True)
    assert named.opponent_name == "pool:gen3"
    assert opponent_name(_env(world, opponent_policy=named)) == "pool:gen3"
    # And a verdict carrying that name is now writable at all, which it was
    # not: write_verdict refuses one with no eval_opponent.
    written = write_verdict(tmp_path / "verdict.json",
                            {"lift": 0.4, "eval_opponent": named.opponent_name})
    assert written["eval_opponent"] == "pool:gen3"

    torch.manual_seed(0)
    greedy = {named(observation, mask) for _ in range(50)}
    assert len(greedy) == 1, "a greedy opponent played two different moves"
    assert greedy == {(0, 1, 0)}, "and it did not play the argmax"

    torch.manual_seed(0)
    sampling = FrozenOpponent(Undecided(), nvec, name="pool:gen3", greedy=False)
    drawn = {sampling(observation, mask) for _ in range(50)}
    assert len(drawn) > 1, "a sampling opponent played one fixed line"


def test_a_ladder_row_cannot_be_recorded_without_naming_both_sides():
    """A ladder row carries a *score*, not a lift, so the lift clause never
    fires on it -- the accidental exemption ``ancestor_probe`` used to enjoy.

    It emitted ``ancestor_win`` and no ``eval_lift_sd``, so nothing checked
    it, and it recorded which ancestor it faced as an integer age against a
    pool that evicts from the middle. The existing self-play ladder was
    already an unnamed measurement; the new one does not get to copy it.

    Two separate demands. *Who* is the kind of opponent. *Which* is the
    weights, and a rating is transitive, so a row naming only the kind cannot
    be placed on the graph at all.
    """
    from cr_sim.train.selfplay import check_lift_is_named

    with pytest.raises(ValueError, match="eval_opponent"):
        check_lift_is_named({"ladder_score": 0.6})
    with pytest.raises(ValueError, match="which weights"):
        check_lift_is_named({"ladder_score": 0.6, "eval_opponent": "pool"})
    with pytest.raises(ValueError, match="which weights"):
        check_lift_is_named({"ladder_elo": 120.0, "eval_opponent": "ladder"})

    named = {"ladder_score": 0.6, "eval_opponent": "pool",
             "ladder_opponent_ref": "runs/clone-v3-paired/cloned.pt"}
    assert check_lift_is_named(named) is named

    # A self-play run's row holds a lift against the random control, an
    # ancestor score against a pool member, and -- under --probe ladder -- a
    # rating against named anchors. Three measurements, three opponents, one
    # dict. A shared eval_opponent means whichever writer runs last relabels
    # the others, which is what happened on a smoke run: the ancestor's score
    # arrived on a row claiming it was played against the rating ladder's
    # anchors. So each family names its own side.
    three = {"eval_lift_sd": 0.4, "eval_opponent": "random",
             "eval_reward": "projected:elixir=0.3,horizon_seconds=3,tower=1",
             "ancestor_score": 0.6, "ancestor_opponent": "pool",
             "ancestor_opponent_ref": "gen3",
             "ladder_elo": 120.0, "ladder_opponent": "ladder",
             "ladder_opponent_ref": "random@random",
             # And what the rating was pinned to: an Elo fitted against
             # differently-pinned anchors is on a different scale, and the
             # same battles came out 377 points apart over that alone.
             "ladder_pinned": {"random": 0.0}}
    assert check_lift_is_named(three) is three
    for missing in ("ancestor_opponent_ref", "ladder_opponent_ref"):
        with pytest.raises(ValueError, match="which weights"):
            check_lift_is_named({k: v for k, v in three.items()
                                 if k != missing})


def test_the_expert_verdict_is_measured_on_the_shared_seed_set(tmp_path):
    """The anchor everything is aimed at was not on the scale it was quoted on.

    +2.716 [+2.369, +3.063] came from ``range(40)`` -- seeds 0 to 39 -- at
    n=40 under ``ProjectionWeights(horizon_seconds=3.0)``. The clone's +2.167
    it is compared against came from 150 seeds under the ordinary evaluation
    reward. Different battles, a different n, and a different unit, because a
    lift is denominated in the control's spread and two rewards give the
    control two different spreads.

    And the file it wrote went around ``write_verdict`` with a bare
    ``write_text``, so the anchor for this whole project carried no
    ``eval_opponent`` at all -- ``report.py`` renders it as "beats an unnamed
    opponent" to this day.
    """
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "scripts"))
    import measure_expert

    from cr_sim.train.evaluate import evaluation_seeds

    out = tmp_path / "expert"
    assert measure_expert.main([
        "--episodes", "2", "--candidates", "3", "--horizon-seconds", "6",
        "--match-seconds", "30", "--out", str(out)]) == 0

    verdict = json.loads((out / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["seeds"] == evaluation_seeds(2)
    # The opponent both arms faced, not the arm being measured. Naming the arm
    # here is the exact confusion check_lift_is_named exists to refuse.
    assert verdict["eval_opponent"] == "random"
    assert verdict["arm"].startswith("search-c3")

    # The reward it was actually measured under, read off the numbers rather
    # than off the string beside them. Under the evaluation reward every
    # episode return telescopes to the crown difference plus at most one
    # hundredth of a tower-health fraction; the projected reward this script
    # used to build wanders an order of magnitude further than that, which is
    # why the old +2.716 and the clone's +2.167 were never in the same unit.
    returns = verdict["control"]["returns"]
    # Per battle, under its own key. It used to be "crowns", written twice
    # into the same dict literal: the mean this script computes lost to the
    # list, so the summary never reached the file and the field's type
    # disagreed with evaluate_paired's control block, which writes the float.
    crowns = verdict["control"]["crowns_per_battle"]
    assert len(returns) == len(crowns) == 2
    assert any(r != 0.0 for r in returns), "no reward accrued; nothing measured"
    assert max(abs(r - c) for r, c in zip(returns, crowns)) <= 0.0101
    # And the mean is there too, agreeing with the list beside it and with
    # what every other paired verdict on this machine means by "crowns".
    assert verdict["control"]["crowns"] == pytest.approx(
        sum(crowns) / len(crowns))

    # And the row it registers on the progress page goes through the guard.
    from cr_sim.train.selfplay import check_lift_is_named

    rows = [json.loads(line) for line
            in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows and all(check_lift_is_named(r) is r for r in rows)


def test_the_ancestor_ladder_names_which_ancestor_it_played(world):
    """It never did, and nothing made it.

    ``ancestor_probe`` emitted ``ancestor_win`` and no ``eval_lift_sd``, so
    ``check_lift_is_named`` never fired on it -- an accidental exemption it
    enjoyed for its whole life. It built its opponent anonymously, so the
    environment reported "unknown", and it recorded which ancestor that was as
    an integer age against a pool that evicts from the *middle*: two runs'
    "generation 1" are not the same weights, and neither is one run's own
    generation 1 before and after an eviction.
    """
    from cr_sim.train.nets import ActorCritic, net_config_for
    from cr_sim.train.selfplay import (
        OpponentPool, ancestor_probe, check_lift_is_named, opponent_name,
    )

    probe_env = _tiny(world)
    probe_env.reset(seed=0)
    net = ActorCritic(net_config_for(probe_env, head="flat"))
    net.eval()
    pool = OpponentPool(capacity=4, seed=0)
    pool.add(net)

    faced = []

    def make_env(opponent=None):
        env = _tiny(world, opponent_policy=opponent)
        faced.append(opponent_name(env))
        return env

    nvec = tuple(int(v) for v in probe_env.action_space.nvec)
    row = ancestor_probe(make_env, pool, nvec, episodes=2)(net)

    # Read off the environment the battles were actually played in.
    assert faced == ["pool:gen1"]
    assert row["ancestor_opponent"] == "pool"
    assert row["ancestor_opponent_ref"] == "gen1"
    assert 0.0 <= row["ancestor_score"] <= 1.0
    # Wins plus half the draws -- the same statistic the offline ladder fits
    # ratings from, so a self-play run's ladder and runs/<name>/ladder.json
    # speak one language.
    assert row["ancestor_score"] == pytest.approx(
        row["ancestor_win"] + 0.5 * (1.0 - row["ancestor_win"]
                                     - row["ancestor_loss"]))
    assert check_lift_is_named(row) is row


def test_the_default_probe_is_a_function_of_the_policy_and_the_seeds(world):
    """The number promotion is decided on has to be the same number twice.

    ``evaluation_probe`` is ``run.py``'s default ``--probe`` and the source of
    every ``eval_lift_sd`` in the record. Its sampled arm called ``evaluate``
    with no generator, so the draw came off torch's global stream: three
    consecutive readings of one checkpoint, one net object and one seed list
    -- after a single ``torch.manual_seed(0)`` -- measured +0.905, +1.228 and
    +0.970, a 0.32 sd spread. Re-running the same three calls each preceded by
    ``torch.manual_seed(0)`` returned +0.905 three times exactly, which pins
    the cause on stream position rather than on the battles. The repo's own
    sampled noise floor is 0.062 sd and the last full PPO run moved greedy by
    0.024; ``run.py`` averages three of these readings and promotes on the
    mean.
    """
    import torch

    from cr_sim.train.nets import ActorCritic, net_config_for
    from cr_sim.train.run import _random_opponent
    from cr_sim.train.selfplay import evaluation_probe

    def make_env():
        return _tiny(world, opponent_policy=_random_opponent(90_000))

    shape = make_env()
    shape.reset(seed=0)
    torch.manual_seed(0)
    net = ActorCritic(net_config_for(shape, head="flat"))
    net.eval()

    probe = evaluation_probe(make_env, episodes=3, seed=4)
    first, second = probe(net), probe(net)

    # Sampling really is what this arm does, so two readings differing would
    # be the generator and not the battles: the greedy argmax would agree
    # whatever stream it was on.
    assert first["eval_return"] != first["control_return"]
    assert first == second, (
        "two readings of one net over one seed list disagreed, so the number "
        "run.py promotes on is not a function of the policy")


def test_the_ancestor_probe_is_a_function_of_the_policy_and_the_seeds(world):
    """Both sides of it: the ancestor samples from its policy too.

    ``ancestor_probe`` built its ``FrozenOpponent`` with no generator and
    called ``evaluate`` with none either, so a self-play run's most readable
    progress signal was two draws off torch's global stream.
    """
    import torch

    from cr_sim.train.nets import ActorCritic, net_config_for
    from cr_sim.train.selfplay import OpponentPool, ancestor_probe

    probe_env = _tiny(world)
    probe_env.reset(seed=0)
    torch.manual_seed(1)
    ancestor = ActorCritic(net_config_for(probe_env, head="flat"))
    ancestor.eval()
    torch.manual_seed(2)
    net = ActorCritic(net_config_for(probe_env, head="flat"))
    net.eval()
    pool = OpponentPool(capacity=4, seed=0)
    pool.add(ancestor)

    def make_env(opponent=None):
        return _tiny(world, opponent_policy=opponent)

    nvec = tuple(int(v) for v in probe_env.action_space.nvec)
    probe = ancestor_probe(make_env, pool, nvec, episodes=3, seed=5)
    first, second = probe(net), probe(net)

    assert 0.0 < first["ancestor_score"] < 1.0 or first["ancestor_return"] != 0.0
    assert first == second


def test_a_paired_arm_measures_the_same_thing_whatever_else_was_asked_for(world):
    """The sampled arm's stream is keyed on the mode, not on its position.

    ``evaluate_paired`` derived each arm's generator from ``enumerate(modes)``,
    so the sampled arm drew one stream when greedy was also requested (index
    1) and a different one when it was asked for alone (index 0). Measured on
    checkpoints/headablate-factored.pt over 40 seeds against the same random
    control, the same weights and the same battles gave sampled lifts of
    +1.3198 and +1.1971 -- 0.123 sd apart, twice the sampled noise floor
    ``evaluate``'s own docstring documents, decided entirely by what else the
    caller wanted. Both single-arm callers are live: scripts/run_ladder.py
    passes ``modes=(args.mode,)`` and scripts/evaluate_vs_expert.py passes
    ``modes=tuple(args.modes)``.
    """
    import torch

    from cr_sim.train.evaluate import evaluate_paired, evaluation_seeds
    from cr_sim.train.nets import ActorCritic, net_config_for
    from cr_sim.train.run import _random_opponent

    def make_env():
        return _tiny(world, opponent_policy=_random_opponent(60_000))

    shape = make_env()
    shape.reset(seed=0)
    torch.manual_seed(0)
    net = ActorCritic(net_config_for(shape, head="flat"))
    net.eval()
    seeds = evaluation_seeds(3, block=0)

    both = evaluate_paired(make_env, net, episodes=3, seeds=seeds,
                           modes=("greedy", "sampled"))
    alone = evaluate_paired(make_env, net, episodes=3, seeds=seeds,
                            modes=("sampled",))

    # The two arms are genuinely different policies over these battles, so a
    # sampled arm that had quietly become the greedy one would show here.
    assert both["sampled"]["lift"] != both["greedy"]["lift"]
    assert alone["sampled"]["lift"] == both["sampled"]["lift"], (
        "asking for the sampled arm alone measured a different number from "
        "asking for it beside greedy, on the same battles")


def test_no_dict_literal_writes_the_same_key_twice():
    """A repeated key silently discards the earlier value, and one did.

    ``scripts/measure_expert.py`` wrote ``"crowns"`` twice into one verdict
    dict: the mean crown difference the code above it computes, and then the
    per-battle list. The list won, so the summary that file's own comment
    describes never reached disk -- and the field's *type* then disagreed with
    ``evaluate_paired``'s control block, which writes the float, leaving two
    verdicts on this machine that mean different things by ``control.crowns``.

    Checked over the whole tree rather than that one literal, because the
    consequence -- a value computed, written, and thrown away, with no error
    anywhere -- is the same wherever it happens, and because a test pinned to
    one line would go green the moment the same mistake moved.
    """
    import ast
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    sources = sorted(
        list((root / "cr_sim").rglob("*.py"))
        + list((root / "scripts").glob("*.py"))
        + list((root / "tests").glob("*.py")))
    assert len(sources) > 40, "the scan found almost nothing to scan"

    offenders = []
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            repeated = sorted({k for k in keys if keys.count(k) > 1})
            if repeated:
                offenders.append(
                    f"{path.relative_to(root)}:{node.lineno} {repeated}")

    assert not offenders, (
        "these dict literals write a key twice, so the first value is "
        f"computed and discarded: {offenders}")


def test_the_expert_evaluation_survives_a_caller_who_wants_one_arm(tmp_path):
    """``evaluate_paired``'s ``modes`` is optional and this script assumed it.

    ``verdict["greedy"]`` and ``verdict["sampled"]`` were indexed unguarded in
    three places, so ``--modes greedy`` raised a KeyError *after* paying for
    every battle -- and against the search expert a battle costs 16.4 seconds.
    ``cr_sim.train.evaluate``'s own CLI has always guarded this; this one had
    not.

    Played against the random opponent rather than the expert, because what
    is being tested is the writer and not the opponent, and the expert is two
    orders of magnitude more simulation.
    """
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "scripts"))
    import evaluate_vs_expert

    out = tmp_path / "one-arm"
    assert evaluate_vs_expert.main([
        "checkpoints/headablate-factored.pt", "--episodes", "2",
        "--opponent", "random", "--modes", "greedy",
        "--match-seconds", "20", "--out", str(out)]) == 0

    verdict = json.loads((out / "verdict.json").read_text(encoding="utf-8"))
    assert "greedy" in verdict and "sampled" not in verdict
    assert verdict["eval_opponent"] == "random"

    row = json.loads((out / "metrics.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    # The arm that was played is on the row; the one that was not is absent
    # rather than invented.
    assert row["eval_lift_sd_greedy"] == verdict["greedy"]["lift"]
    assert "eval_lift_sd_sampled" not in row

    # And the arena agrees with the other two evaluation entry points. This
    # defaulted to 5 while cr_sim.train.evaluate's CLI and cr_sim.train.run
    # both defaulted to 11 -- the same silent-disagreement shape that trained
    # a whole run at level 11 while its config.json recorded 5.
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    from cr_sim.train.run import build_parser

    assert config["tower_level"] == build_parser().get_default("tower_level")
