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

    named = {"updates": 3, "eval_lift_sd": 0.42, "eval_opponent": "random"}
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
