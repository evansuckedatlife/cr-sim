"""Annealing the shaping, and annealing the knob that is actually the shaping.

Two separate claims live here and they are worth keeping apart, because only
the first is settled by anything in this file.

**What is tested.** A schedule exists, is a function of steps, clamps at both
ends, applies only at an episode boundary, survives the trip into a worker
process, terminates exactly on the sparse crown objective, and does not move
the scale the in-run probe measures on. A constant schedule -- the default --
does nothing at all.

**What is not, and cannot be here.** Whether annealing produces a better
policy. That needs a paired A/B of two full 1M-step runs at the measured 28.0
steps/s: about 9.9 hours each, roughly 20 sequential, and they cannot overlap
because one run already occupies all eight workers. Nothing below should be
read as evidence that the anneal works.

The trap these tests exist to keep shut is that ``--shaping`` is *inert* under
both rewards anyone trains with. A 500x change in it is bit-identical under
``projected`` and ``five-term``, so a schedule aimed there is a run that
reports an anneal and performs none --
:func:`test_the_annealed_knob_is_the_one_the_reward_actually_reads` is the
test that catches that, and it fails against the knob the class docstring used
to recommend.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cr_sim.api.env import CRSimEnv
from cr_sim.api.reward import ProjectedReward, ProjectionWeights, RewardWeights
from cr_sim.train.schedule import (
    RewardSchedule, anneal_to_zero, constant_schedule, knob_for_reward)

from .test_data_pipeline import BUILD

DECK = ("Knight", "Musketeer", "Cannon", "Skeletons",
        "IceSpirits", "Log", "Fireball", "Goblins")

#: A short, decisive match. Level 5 towers halve the draw rate at no extra
#: compute, which is what makes "the return equals the crown difference" a
#: statement about a number that is usually not zero.
ARENA = dict(ticks_per_second=20, frame_skip=30, tower_level=5)
MATCH_TICKS = 20 * 120


@pytest.fixture(scope="module")
def world():
    from cr_sim.data.cards import build_card_registry
    from cr_sim.data.leveling import build_level_table
    from cr_sim.data.source import LogicData

    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


def _env(world, weights=None, *, shaping=0.01, seed=None):
    data, levels, registry = world
    env = CRSimEnv(data, levels, registry, DECK, DECK,
                   max_ticks=MATCH_TICKS, reward_weights=weights,
                   reward_shaping_weight=shaping, **ARENA)
    if seed is not None:
        env.reset(seed=seed)
    return env


def _play_out(env, *, seed, stop_after=None):
    """Drive one episode with a reproducible action stream.

    The stream is a function of ``seed`` alone -- not of the env -- so two
    environments handed the same seed take the identical actions and any
    difference in their rewards is the reward's doing.
    """
    rng = np.random.default_rng(seed)
    slots, width, height = env.action_space.nvec
    rewards = []
    while True:
        mask = env.legal_action_mask().reshape(-1)
        index = int(rng.choice(np.flatnonzero(mask))) if mask.any() else 0
        slot, remainder = divmod(index, width * height)
        gx, gy = divmod(remainder, height)
        _, reward, terminated, truncated, info = env.step(
            (min(slot, slots - 1), gx, gy))
        rewards.append(reward)
        if terminated or truncated:
            return rewards, info
        if stop_after is not None and len(rewards) >= stop_after:
            return rewards, info


def _crown_difference(env) -> int:
    return (env.battle.players[env.team].crowns
            - env.battle.players[env.team.opponent].crowns)


def _is_whole(value: float) -> bool:
    """Whether a return is a whole number of crowns.

    The exact endpoint property, and the one that holds on every episode
    rather than only on the ones that reach a real finish. With the shaping at
    zero the projected potential is the projected crown difference alone --
    an integer -- and both sides start level, so the return telescopes to an
    integer. With the shaping in force it carries a tower-health fraction and
    an elixir lead and is not.

    A *truncated* episode's return is the crown difference the board projects
    to three seconds out, which can legitimately differ by one from the crowns
    on the scoreboard when a tower is about to fall -- so it is integrality,
    not equality with ``info``, that is the invariant here.
    """
    return abs(value - round(value)) < 1e-9


# ------------------------------------------------------------------ the ramp


def test_the_schedule_is_a_function_of_steps_and_clamps_at_both_ends():
    """Start, exact midpoint, end, and well past the end.

    The midpoint is asserted exactly rather than approximately because a
    linear ramp on the steps axis has an exact midpoint and that is most of
    why it was chosen over a cosine: the value at any step is recoverable from
    config.json with a pencil.

    Past ``end_step`` the schedule is clamped, never extrapolated. A ramp
    continued past zero does not give a smaller shaping, it gives a reward
    pointing the other way -- an agent paid for losing tower health.
    """
    schedule = anneal_to_zero(
        "projection",
        {"tower": 1.0, "elixir": 0.3, "horizon_seconds": 3.0},
        start_step=100_000, end_step=900_000)

    assert schedule.at(0) == {"tower": 1.0, "elixir": 0.3, "horizon_seconds": 3.0}
    assert schedule.at(100_000) == {"tower": 1.0, "elixir": 0.3,
                                    "horizon_seconds": 3.0}
    middle = schedule.at(500_000)
    assert middle["tower"] == pytest.approx(0.5)
    assert middle["elixir"] == pytest.approx(0.15)
    assert schedule.at(900_000) == {"tower": 0.0, "elixir": 0.0,
                                    "horizon_seconds": 3.0}

    # The clamp. Three times past the end, and at a step far beyond any run.
    for beyond in (900_001, 2_700_000, 10_000_000):
        assert schedule.at(beyond) == {"tower": 0.0, "elixir": 0.0,
                                       "horizon_seconds": 3.0}
        assert schedule.at(beyond)["tower"] >= 0.0

    # And the axis is steps. An identical schedule read on an updates axis --
    # a few thousand rather than a million -- would already be finished here.
    assert schedule.at(1_000)["tower"] == 1.0


def test_an_unset_end_step_lands_at_eighty_percent_of_the_run():
    """The last fifth is held at zero on purpose.

    Otherwise the final checkpoint is measured under a weight that was still
    moving, and the sparse crown objective -- the thing the whole schedule
    exists to reach -- never gets a stationary stretch to be measured on. At
    the measured 28.0 steps/s that fifth is about two hours.
    """
    schedule = anneal_to_zero(
        "projection", {"tower": 1.0, "elixir": 0.3, "horizon_seconds": 3.0},
    ).resolved(1_000_000)
    assert schedule.end_step == 800_000
    assert schedule.at(800_000)["tower"] == 0.0
    assert schedule.at(1_000_000)["tower"] == 0.0
    # A resolved schedule is not re-resolved: an explicit end stays put.
    pinned = anneal_to_zero(
        "projection", {"tower": 1.0, "elixir": 0.3, "horizon_seconds": 3.0},
        end_step=123).resolved(1_000_000)
    assert pinned.end_step == 123


def test_a_late_start_does_not_have_to_name_an_end():
    """``--anneal-start`` was unusable on its own, and crashed at startup.

    ``RewardSchedule.__post_init__`` rejected ``end_step 0 is before
    start_step 500`` before ``resolved()`` -- whose whole job is filling that
    zero in from the run's total -- could ever be reached, so `--anneal
    --anneal-start 500` exited with an unhandled ValueError while
    ``--anneal-end``'s own help still said "0 means 80% of --steps". Zero is
    the unset sentinel, not a step.
    """
    from cr_sim.train.run import build_parser
    from cr_sim.train.run import _reward_schedule

    args = build_parser().parse_args(
        ["--steps", "1000", "--reward", "projected", "--anneal",
         "--anneal-start", "500"])
    schedule = _reward_schedule(args)
    assert (schedule.start_step, schedule.end_step) == (500, 800)
    # And the ramp really starts where it was told to rather than at zero.
    assert schedule.at(500)["tower"] == 1.0
    assert schedule.at(650)["tower"] == pytest.approx(0.5)
    assert schedule.at(800)["tower"] == 0.0

    # A real inversion is still refused: this is a sentinel, not a hole.
    with pytest.raises(ValueError, match="before start_step"):
        anneal_to_zero(
            "projection", {"tower": 1.0, "elixir": 0.3, "horizon_seconds": 3.0},
            start_step=500, end_step=200)


def test_a_schedule_never_anneals_the_objective():
    """Crowns are the objective, not shaping, and are never reduced.

    Neither is ``horizon_seconds``: ``None`` there means "play the match out",
    roughly forty times the cost, so it is not a point on any ramp and the
    schedule refuses endpoints that differ.
    """
    five = anneal_to_zero("five_term", RewardWeights().as_dict())
    assert five.end["crowns"] == 1.0
    assert all(five.end[f] == 0.0 for f in
               ("tower_damage", "own_tower_hp", "elixir_trade",
                "counterpush", "kite"))

    projection = anneal_to_zero("projection", ProjectionWeights().as_dict())
    assert projection.end["horizon_seconds"] == 3.0
    assert projection.end["tower"] == 0.0 and projection.end["elixir"] == 0.0

    with pytest.raises(ValueError, match="cannot be interpolated"):
        RewardSchedule(knob="projection",
                       start={"tower": 1.0, "elixir": 0.3,
                              "horizon_seconds": 3.0},
                       end={"tower": 0.0, "elixir": 0.0,
                            "horizon_seconds": None},
                       end_step=10)


def test_the_annealed_knob_is_the_one_the_reward_actually_reads(world):
    """``--shaping`` is inert under both rewards anyone trains with.

    Every ``_shaped_value`` call site sits inside the ``else`` of ``if
    self._reward is not None``, so under ``projected`` and ``five-term`` the
    weight is never read. Measured here rather than asserted: 0.01 against
    5.00, a five hundred fold change, on an identical action stream.

    This is the whole reason the schedule does not touch ``--shaping`` unless
    ``--reward simple``. A schedule aimed at it under the default reward is a
    run that reports an anneal and performs none, which is exactly what the
    environment's own class docstring used to recommend.
    """
    projected = ProjectionWeights(horizon_seconds=3.0)
    for weights in (projected, RewardWeights(), None):
        low = _play_out(_env(world, weights, shaping=0.01, seed=3), seed=3)[0]
        high = _play_out(_env(world, weights, shaping=5.00, seed=3), seed=3)[0]
        inert = low == high
        assert inert is (weights is not None), (
            f"--shaping should be inert exactly when a reward object is "
            f"present; weights={type(weights).__name__} inert={inert}")

    # And the schedule agrees with the measurement, per reward.
    assert knob_for_reward("projected") == "projection"
    assert knob_for_reward("five-term") == "five_term"
    assert knob_for_reward("simple") == "shaping"


# ------------------------------------------- the weight reaches the reward


def test_set_reward_weights_actually_reaches_the_reward_computation(world):
    """Not stored and dropped. This codebase has shipped inert features.

    The endpoint is exact and that is the point of the whole schedule: with
    the shaping at zero the projected potential is the projected crown
    difference alone, both sides start level, so the episode return telescopes
    to the final crown difference *as an integer*. Under the starting weights
    it is emphatically not an integer, which is what makes this sharp.
    """
    env = _env(world, ProjectionWeights(horizon_seconds=3.0))
    env.set_reward_weights(ProjectionWeights(tower=0.0, elixir=0.0,
                                             horizon_seconds=3.0))
    env.reset(seed=11)

    # It reached the reward object, not just the env's bookkeeping.
    assert isinstance(env._reward, ProjectedReward)
    assert env._reward.weights == ProjectionWeights(
        tower=0.0, elixir=0.0, horizon_seconds=3.0)

    rewards, info = _play_out(env, seed=11)
    total = sum(rewards)
    assert _is_whole(total), (
        f"return {total} is not a whole number of crowns; the shaping is "
        "still in the reward")
    if info["finished"]:
        assert total == pytest.approx(_crown_difference(env), abs=1e-9)

    # The same battle under the starting weights is not an integer, so the
    # assertion above is not something every weight satisfies.
    plain = _env(world, ProjectionWeights(horizon_seconds=3.0))
    plain.reset(seed=11)
    unannealed = sum(_play_out(plain, seed=11)[0])
    assert not _is_whole(unannealed)
    assert unannealed != pytest.approx(total, abs=1e-6)

    # And across a spread of seeds, so this is not one lucky battle -- with at
    # least one reaching a real finish, where the return is the scoreboard.
    finishes = 0
    for seed in range(4):
        arm = _env(world, ProjectionWeights(tower=0.0, elixir=0.0,
                                            horizon_seconds=3.0), seed=seed)
        played, ended = _play_out(arm, seed=seed)
        assert _is_whole(sum(played))
        if ended["finished"]:
            finishes += 1
            assert sum(played) == pytest.approx(
                _crown_difference(arm), abs=1e-9)
    assert finishes, "no battle finished; the crown identity went untested"


def test_the_weight_changes_only_at_a_reset(world):
    """Mid-episode the reward stops being potential-based.

    ``_previous`` holds the potential under the *old* weight, so the next step
    is paid ``phi_new(s_new) - phi_old(s_old)``: a genuine reward plus a
    fabricated one for the weight change, charged in full to whatever action
    happened to be there. Measured at 19x the genuine reward for that step --
    -0.159656 against -0.007802 -- and invisible in the episode return, which
    still telescopes correctly to its own endpoint weights. That is what makes
    it dangerous: the existing telescoping invariant stays green over it.
    """
    weights = ProjectionWeights(horizon_seconds=3.0)
    zeroed = ProjectionWeights(tower=0.0, elixir=0.0, horizon_seconds=3.0)

    twin = _env(world, weights, seed=5)
    switched = _env(world, weights, seed=5)
    before_twin, _ = _play_out(twin, seed=5, stop_after=5)
    before_switched, _ = _play_out(switched, seed=5, stop_after=5)
    assert before_twin == before_switched

    switched.set_reward_weights(zeroed)

    # The very next step, and every step to the end of this episode, is
    # bit-identical. The pending weight is pending, not applied.
    after_twin, _ = _play_out(twin, seed=99)
    after_switched, _ = _play_out(switched, seed=99)
    assert after_twin == after_switched, (
        "a weight set mid-episode changed a reward inside that episode")

    # At the next reset it takes effect, and the endpoint is exact.
    switched.reset(seed=12)
    twin.reset(seed=12)
    annealed = sum(_play_out(switched, seed=12)[0])
    held = sum(_play_out(twin, seed=12)[0])
    assert _is_whole(annealed) and not _is_whole(held)
    assert held != pytest.approx(annealed, abs=1e-6)


def test_a_constant_schedule_is_bit_identical_to_pushing_nothing(world):
    """The default, and it must cost nothing and change nothing.

    Two claims. The schedule itself never moves, so a caller can skip the push
    entirely; and pushing it anyway -- which rebuilds the reward object at
    reset -- is bit-identical to never touching it, so the two are the same
    run either way.
    """
    values = ProjectionWeights(horizon_seconds=3.0).as_dict()
    schedule = constant_schedule("projection", values).resolved(1_000_000)
    assert schedule.is_constant
    for step in (0, 1, 400_000, 800_000, 5_000_000):
        assert schedule.at(step) == values

    untouched = _env(world, ProjectionWeights(horizon_seconds=3.0), seed=8)
    pushed = _env(world, ProjectionWeights(horizon_seconds=3.0))
    weights, shaping = schedule.weights_at(0)
    pushed.set_reward_weights(weights, shaping_weight=shaping)
    pushed.reset(seed=8)

    a, _ = _play_out(untouched, seed=8)
    b, _ = _play_out(pushed, seed=8)
    assert a == b, "a constant schedule changed the reward"


def test_a_telescoping_reward_is_scored_once_per_step(world):
    """The board was projected twice per decision to cancel one of them.

    ``step`` scored the state and then the run-out scored it again, and
    ``score()`` is a pure function of state, so (phi_mid - phi_prev) +
    (phi_end - phi_mid) is phi_end - phi_prev exactly. Measured at 2.00 score
    calls per non-terminal decision under ``projected`` -- structural, not
    policy-dependent -- which was 48.7% of all projections and 26.4% of all
    environment wall time, for a term that cancels.

    Landed before any anneal measurement so the anneal is not credited with
    the speedup.
    """
    calls = {"n": 0}

    class Counting(ProjectedReward):
        def score(self, battle):
            calls["n"] += 1
            return ProjectedReward.score(self, battle)

    weights = ProjectionWeights(horizon_seconds=3.0)
    counted = _env(world, weights)
    counted._reward = Counting(counted.team, weights)
    counted.reset(seed=6)
    calls["n"] = 0                      # the reset's baseline score
    rewards, _ = _play_out(counted, seed=6)

    per_step = calls["n"] / len(rewards)
    assert per_step == pytest.approx(1.0, abs=1e-9), (
        f"{per_step:.2f} score calls per step; a telescoping reward needs one")

    # And the rewards are the same ones the two-score path produced. The
    # cancellation is exact in real arithmetic and lands at the float floor
    # here, which is the same 2.2e-16 the potential identity itself sits at.
    plain = _env(world, weights, seed=6)
    reference, _ = _play_out(plain, seed=6)
    assert len(reference) == len(rewards)
    assert max(abs(x - y) for x, y in zip(reference, rewards)) < 1e-12
    assert sum(reference) == pytest.approx(sum(rewards), abs=1e-12)


def test_the_episode_reward_still_telescopes_across_a_scheduled_change(world):
    """Extends test_lookahead's telescoping invariant across a weight change.

    That test is the guard a naive mid-episode anneal breaks, and it breaks
    *quietly*: switching the weights inside an episode still leaves the return
    telescoping to its own endpoints, so the original test stays green while
    one arbitrary action has been paid the difference between two different
    potentials. What it cannot survive is being asked to telescope under a
    *single* named weight tuple.

    Two halves, and both are needed. A weight set mid-episode must not take
    effect, so this episode telescopes under the weights it started with; and
    the next episode, which begins after the reset that adopts them,
    telescopes under the new ones.
    """
    start_weights = ProjectionWeights(horizon_seconds=2.0)
    end_weights = ProjectionWeights(tower=0.25, elixir=0.05,
                                    horizon_seconds=2.0)

    env = _env(world, start_weights, seed=4)
    held = ProjectedReward(env.team, start_weights)
    before = held.score(env.battle)

    rewards, _ = _play_out(env, seed=4, stop_after=3)
    # Set part-way through, and it must change nothing about this episode.
    env.set_reward_weights(end_weights)
    rest, _ = _play_out(env, seed=4)
    total = sum(rewards) + sum(rest)
    assert total == pytest.approx(held.score(env.battle) - before, abs=1e-9), (
        "the episode stopped telescoping under the weights it started with, "
        "so the change was applied inside the episode")

    # The next episode adopts them, and telescopes under those.
    env.reset(seed=7)
    adopted = ProjectedReward(env.team, end_weights)
    opening = adopted.score(env.battle)
    after = sum(_play_out(env, seed=7)[0])
    assert after == pytest.approx(
        adopted.score(env.battle) - opening, abs=1e-9)
    assert env._reward.weights == end_weights


# ------------------------------------------------------- and into a worker


def test_the_workers_adopt_the_annealed_weight():
    """The one that matters most: this is the --tower-level bug's exact shape.

    ``VecEnvConfig`` is frozen and pickled once per worker, so without an RPC
    there is no way to move a worker's reward at all -- and a field ``_env()``
    sets while the workers do not is how ``--tower-level 5 --workers 8``
    trained every rollout at level 11 while config.json recorded 5 and the
    probe ran at 5.

    Asserted on the reward the worker actually paid, not on a reply: a branch
    that answers ``True`` and does nothing is precisely the failure being
    guarded against. With the shaping at zero the summed reward is a whole
    number of crowns; with it in force it is not.
    """
    from cr_sim.api.vec import CRSimVecEnv, VecEnvConfig
    from cr_sim.api.encoding import NOOP_SLOT

    base = VecEnvConfig(
        build=BUILD, blue_deck=DECK, red_deck=DECK,
        ticks_per_second=20, frame_skip=30, tower_level=5,
        max_ticks=MATCH_TICKS, opponent_seed=17,
        reward_weights=ProjectionWeights(horizon_seconds=3.0))

    def one_episode(push):
        vec = CRSimVecEnv(base, num_envs=1, workers=1)
        try:
            if push is not None:
                vec.set_reward_weights(push)
            vec.reset([21])
            total = 0.0
            for _ in range(400):
                _, rewards, dones, crowns, _ = vec.step([(NOOP_SLOT, 0, 0)])
                total += float(rewards[0])
                if dones[0]:
                    return total, int(crowns[0])
            raise AssertionError("episode never ended")
        finally:
            vec.close()

    annealed, crowns = one_episode(
        ProjectionWeights(tower=0.0, elixir=0.0, horizon_seconds=3.0))
    held, held_crowns = one_episode(None)

    # Same seed, same opponent stream, same actions: the battles agree and
    # only the reward differs.
    assert crowns == held_crowns
    assert annealed == pytest.approx(crowns, abs=1e-9), (
        "the worker's reward did not adopt the pushed weights")
    assert annealed == pytest.approx(round(annealed), abs=1e-9)
    assert abs(held - round(held)) > 1e-6
    assert held != pytest.approx(annealed, abs=1e-6)


# ----------------------------------------------- what a run must record


def _capture_run(tmp_path, monkeypatch, argv):
    """Run ``main`` far enough to write config.json, then stop.

    ``on_net`` fires before the first rollout, so hijacking the probe gives
    the environment the probe would have played in without paying for a
    single update.
    """
    import cr_sim.train.run as run_module

    captured: dict = {}

    class _Stop(RuntimeError):
        pass

    def _fake_probe(make_env, **kwargs):
        captured["eval_env"] = make_env()
        raise _Stop("captured")

    monkeypatch.setattr(run_module, "evaluation_probe", _fake_probe)
    with pytest.raises(_Stop):
        run_module.main([
            "--steps", "64", "--horizon", "8", "--envs", "1", "--workers", "0",
            "--match-seconds", "20", "--tower-level", "5", "--tps", "20",
            "--frame-skip", "30", "--device", "cpu", "--opponent", "random",
            "--out", str(tmp_path), "--name", "sched", *argv])
    captured["config"] = json.loads(
        (tmp_path / "sched" / "config.json").read_text(encoding="utf-8"))
    return captured


def test_a_run_records_its_schedule_completely_enough_to_reproduce_it(
        tmp_path, monkeypatch):
    """Rebuilt from the file, not merely present in it.

    A schedule recorded as a flag plus a default is not reproducible: the
    default may since have moved. Both endpoints are written out literally, so
    this test reconstructs the schedule from config.json alone and checks it
    produces the same weights the run used at the start, the midpoint and the
    end.
    """
    captured = _capture_run(tmp_path, monkeypatch, [
        "--reward", "projected", "--anneal", "--anneal-end", "1000",
        "--tower-weight", "0.8", "--elixir-weight", "0.2"])
    recorded = captured["config"]["reward_schedule"]

    rebuilt = RewardSchedule(
        knob=recorded["knob"], start=recorded["start"], end=recorded["end"],
        shape=recorded["shape"], axis=recorded["axis"],
        start_step=recorded["start_step"], end_step=recorded["end_step"])

    assert rebuilt.at(0) == {"tower": 0.8, "elixir": 0.2,
                             "horizon_seconds": 3.0}
    middle = rebuilt.at(500)
    assert middle["tower"] == pytest.approx(0.4)
    assert middle["elixir"] == pytest.approx(0.1)
    assert rebuilt.at(1000) == {"tower": 0.0, "elixir": 0.0,
                                "horizon_seconds": 3.0}
    assert rebuilt.weights_at(1000)[0] == ProjectionWeights(
        tower=0.0, elixir=0.0, horizon_seconds=3.0)

    # And the trap is labelled. --shaping is written for A/B key-set
    # compatibility with every historical run, but it is inert here.
    assert recorded["shaping_is_inert"] is True
    assert recorded["boundary"] == "episode_reset"
    assert captured["config"]["shaping"] == 0.01


def test_a_run_without_the_flag_records_a_constant_schedule(
        tmp_path, monkeypatch):
    """Constant is the default and stays expressible.

    Every command that worked before this existed still means what it meant:
    the recorded endpoints are equal, ``constant`` says so, and the run pushes
    nothing.
    """
    captured = _capture_run(tmp_path, monkeypatch, ["--reward", "projected"])
    recorded = captured["config"]["reward_schedule"]
    assert recorded["constant"] is True
    # Read off the CLI rather than written down here. Nothing is passed to
    # this run, so what the schedule must record is precisely what argparse
    # resolved -- and pinning the number in the test instead meant that
    # moving --elixir-weight's default to 0.0 failed a test about whether a
    # constant schedule is constant.
    from cr_sim.train.run import build_parser

    defaults = build_parser().parse_args(["--reward", "projected"])
    assert recorded["start"] == recorded["end"] == {
        "tower": defaults.tower_weight,
        "elixir": defaults.elixir_weight,
        "horizon_seconds": 3.0}

    rebuilt = RewardSchedule(
        knob=recorded["knob"], start=recorded["start"], end=recorded["end"],
        start_step=recorded["start_step"], end_step=recorded["end_step"])
    assert rebuilt.is_constant
    assert rebuilt.at(0) == rebuilt.at(10**9)


def test_a_constant_schedule_pushes_nothing_at_all(tmp_path, monkeypatch):
    """Bit-identity with today, asserted at the run level.

    The cheapest possible proof that ``--anneal``-less runs are unchanged: no
    environment is ever handed new weights, so no reward object is ever
    rebuilt and no worker RPC is ever sent.

    Worth recording how this test earned its keep, because at first it did
    not. ``record`` originally guarded the push twice -- once on
    ``schedule.is_constant`` and once on whether the weights had actually
    moved -- and this test stayed green over *either* mutation, because the
    surviving guard covered for the broken one. Two independent conditions
    protecting one behaviour means neither can be held to account by any
    single-line mutation, which is precisely the shape of test this codebase
    keeps shipping. The redundant short-circuit was removed; the change
    detection is the one mechanism, and replacing its condition with ``True``
    turns this red.
    """
    import cr_sim.train.run as run_module

    pushes: list = []
    original = CRSimEnv.set_reward_weights

    def _spy(self, weights, **kwargs):
        pushes.append(weights)
        return original(self, weights, **kwargs)

    monkeypatch.setattr(CRSimEnv, "set_reward_weights", _spy)

    def _run(extra):
        pushes.clear()
        run_module.main([
            "--steps", "128", "--horizon", "16", "--envs", "1",
            "--workers", "0", "--match-seconds", "20", "--tower-level", "5",
            "--tps", "20", "--frame-skip", "30", "--device", "cpu",
            "--opponent", "random", "--reward", "projected",
            "--eval-every", "10000", "--save-every", "10000",
            "--out", str(tmp_path), "--name", "pushes", *extra])
        return list(pushes)

    assert _run([]) == [], "a run without --anneal pushed a reward weight"

    moved = _run(["--anneal", "--anneal-end", "64"])
    assert moved, "an annealed run never pushed anything"
    assert moved[-1] == ProjectionWeights(tower=0.0, elixir=0.0,
                                          horizon_seconds=3.0)


def test_the_annealed_weight_reaches_the_worker_processes(tmp_path, monkeypatch):
    """The push that actually moves a real run, and nothing covered it.

    Under ``--workers`` the rollout lives in other processes and
    ``local_envs`` holds only the shape probe, so the local push moves nothing
    that trains. Deleting ``parallel.set_reward_weights(...)`` from
    ``run.record`` left all seventeen reward-schedule tests plus the worker
    config test green while every rollout kept paying the un-annealed weight:
    measured, four pushes to the local list and zero to the workers, with
    config.json recording the schedule and every metrics row's reward_weights
    claiming tower=0.0.

    The two tests that look like they cover it do not.
    ``test_a_constant_schedule_pushes_nothing_at_all`` runs ``--workers 0``
    and spies on ``CRSimEnv``; ``test_the_workers_adopt_the_annealed_weight``
    drives ``CRSimVecEnv`` directly and never goes through run.py -- which is
    the --tower-level bug's exact shape, one layer up.
    """
    import cr_sim.train.run as run_module
    from cr_sim.api.vec import CRSimVecEnv

    to_workers: list = []
    to_local: list = []
    worker_push = CRSimVecEnv.set_reward_weights
    local_push = CRSimEnv.set_reward_weights

    def _worker_spy(self, weights, **kwargs):
        to_workers.append(weights)
        return worker_push(self, weights, **kwargs)

    def _local_spy(self, weights, **kwargs):
        to_local.append(weights)
        return local_push(self, weights, **kwargs)

    monkeypatch.setattr(CRSimVecEnv, "set_reward_weights", _worker_spy)
    monkeypatch.setattr(CRSimEnv, "set_reward_weights", _local_spy)

    run_module.main([
        "--steps", "128", "--horizon", "16", "--envs", "1", "--workers", "1",
        "--match-seconds", "20", "--tower-level", "5", "--tps", "20",
        "--frame-skip", "30", "--device", "cpu", "--opponent", "random",
        "--reward", "projected", "--eval-every", "0",
        "--save-every", "10000", "--out", str(tmp_path),
        "--name", "worker-anneal", "--anneal", "--anneal-end", "64"])

    assert to_workers, (
        "the annealed weight never left the parent process, so every rollout "
        "trained at the un-annealed weight while config.json and every "
        "metrics row recorded the schedule")
    # Both lists, field for field: a run whose local probe and whose workers
    # disagree about the reward is training on one game and measuring another.
    assert to_workers == to_local
    assert to_workers[-1] == ProjectionWeights(tower=0.0, elixir=0.0,
                                               horizon_seconds=3.0)


def test_a_resumed_run_moves_its_rollout_onto_the_resumed_steps_weight(
        tmp_path, monkeypatch):
    """Steps is the schedule's axis so that a resume lands where it left off.

    ``record`` fires at the *end* of an update, so a resumed run's opening
    rollout would be collected under whatever weights ``_env()`` was built
    with -- the schedule's step-zero values -- while the row it then writes
    claims the resumed step's. One update out of five hundred, and exactly the
    class of disagreement between what a run did and what it recorded that
    this project has already paid for twice.

    Asserted on the *order* of events, because that is the whole claim: the
    environments are moved before anything is collected, not after.
    """
    import cr_sim.train.run as run_module

    events: list = []
    original = CRSimEnv.set_reward_weights

    def _spy(self, weights, **kwargs):
        events.append(("push", weights))
        return original(self, weights, **kwargs)

    guard = run_module.check_lift_is_named

    def _row(stats):
        events.append(("row", int(stats["steps"])))
        return guard(stats)

    monkeypatch.setattr(CRSimEnv, "set_reward_weights", _spy)
    monkeypatch.setattr(run_module, "check_lift_is_named", _row)

    common = [
        "--horizon", "16", "--envs", "1", "--workers", "0",
        "--match-seconds", "20", "--tower-level", "5", "--tps", "20",
        "--frame-skip", "30", "--device", "cpu", "--opponent", "random",
        "--reward", "projected", "--anneal", "--anneal-end", "1000",
        # Passed rather than inherited. The elixir term is half of what this
        # asserts -- that the ramp scales *every* weight and not just the
        # tower -- and at the CLI's own default of 0.0 the expectation below
        # would be 0.0 == 0.0, which a schedule that never touched elixir
        # would satisfy just as well.
        "--elixir-weight", "0.3",
        "--eval-every", "10000", "--save-every", "1",
        "--out", str(tmp_path), "--name", "resumed",
    ]
    assert run_module.main(["--steps", "64", *common]) == 0
    rows = [json.loads(line) for line in
            (tmp_path / "resumed" / "metrics.jsonl").read_text(
                encoding="utf-8").splitlines()]
    assert rows[-1]["steps"] == 64

    events.clear()
    assert run_module.main(["--steps", "256", "--resume", *common]) == 0

    assert events, "the resumed run neither pushed nor recorded anything"
    kind, weights = events[0]
    assert kind == "push", (
        "the resumed run collected an update before moving its environments "
        f"onto the resumed step's weights; first event was {events[0]!r}")
    # at(64) on a ramp from 1.0 to 0.0 over 1000 steps.
    assert weights.tower == pytest.approx(1.0 - 64 / 1000)
    assert weights.elixir == pytest.approx(0.3 * (1.0 - 64 / 1000))


def test_the_probe_scale_does_not_move_with_the_training_schedule(
        tmp_path, monkeypatch):
    """The probe measures on a fixed scale, whatever the run is training on.

    ``eval_lift_sd`` is a difference of returns against a control that is
    evaluated once and cached, spread and all. Built from the training reward,
    an annealed run shrinks its own arm while the control keeps the scale it
    was measured on, so the lift drifts from nothing but the scale and
    promotion walks back toward the earliest, highest-shaping checkpoints.

    Read off the environment the probe was actually handed, not off a
    constructor argument.
    """
    from cr_sim.train.selfplay import reward_name

    scales = set()
    for extra in (["--reward", "projected", "--elixir-weight", "0.9"],
                  ["--reward", "five-term"],
                  ["--reward", "projected", "--anneal", "--tower-weight", "0.1"]):
        captured = _capture_run(tmp_path, monkeypatch, extra)
        scales.add(reward_name(captured["eval_env"]))
        assert captured["config"]["eval_reward"] == {
            "kind": "projected", "tower": 1.0, "elixir": 0.3,
            "horizon_seconds": 3.0}

    assert len(scales) == 1, (
        f"the probe measured on {len(scales)} different scales: {scales}")


def test_a_lift_row_must_record_the_reward_it_was_measured_under():
    """The identical mistake ``eval_opponent`` already refuses, one axis over.

    A lift names who was played and not what was counted, and the two policies
    produce a different lift under ``projected`` than under ``five-term``. A
    run that anneals produces both within itself.
    """
    from cr_sim.train.selfplay import check_lift_is_named

    with pytest.raises(ValueError, match="eval_reward"):
        check_lift_is_named({"eval_lift_sd": 0.4, "eval_opponent": "random"})
    with pytest.raises(ValueError, match="eval_reward"):
        check_lift_is_named({"eval_lift_sd": 0.4, "eval_opponent": "random",
                             "eval_reward": ""})

    named = {"eval_lift_sd": 0.4, "eval_opponent": "random",
             "eval_reward": "projected:elixir=0.3,horizon_seconds=3,tower=1"}
    assert check_lift_is_named(named) is named

    # A row with no lift is not a measurement and needs no scale.
    plain = {"updates": 2, "entropy": 1.0}
    assert check_lift_is_named(plain) is plain


def test_the_recorded_scale_names_the_weights_and_not_the_variant(world):
    """"projected" is not a scale.

    The same variant at tower=1.0 and at tower=0.0 produces returns an order
    of magnitude apart, and an annealed run produces both. That is exactly the
    gap ``Demonstrations.reward`` still has: the shipped shards say
    'projected' and docs/training.md records they were collected under
    ``--elixir-weight 0``, which is nowhere in the file.
    """
    from cr_sim.train.selfplay import reward_name

    full = reward_name(_env(world, ProjectionWeights(horizon_seconds=3.0)))
    zeroed = reward_name(_env(world, ProjectionWeights(
        tower=0.0, elixir=0.0, horizon_seconds=3.0)))
    assert full != zeroed
    assert "tower=1" in full and "tower=0" in zeroed

    assert reward_name(_env(world, RewardWeights())).startswith("five-term:")
    assert reward_name(_env(world, None, shaping=0.02)) == "simple:shaping=0.02"
