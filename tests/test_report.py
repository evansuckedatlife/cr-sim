"""The multi-run comparison.

The live page answers "how is this one going". This answers the question that
decides what to build next -- which of these worked, and what was different
about it -- and until it existed the only place six runs had ever been
compared was in conversation.

The care here is about not repeating readings this project has already got
wrong: counting duplicated rows twice, and presenting forty battles as if they
settled something.
"""

from __future__ import annotations

import json

import pytest

from cr_sim.train.report import collect, render_index


def _run(tmp_path, name, rows, config=None, verdict=None):
    run = tmp_path / name
    run.mkdir()
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if config is not None:
        (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if verdict is not None:
        (run / "verdict.json").write_text(json.dumps(verdict), encoding="utf-8")
    return run


def _row(update, steps, **extra):
    row = {"updates": update, "steps": steps, "episodes": steps // 20,
           "steps_per_second": 40.0, "entropy": 4.3, "value_loss": 0.2,
           "policy_loss": -0.01, "mean_return": 0.1, "win_rate": 0.3,
           "noop_fraction": 0.02}
    row.update(extra)
    return row


def test_a_run_written_twice_per_update_is_counted_once(tmp_path):
    """An early bug wrote every update twice and those files are still on disk.

    Averaging over them counts each update twice, which is the reading that
    made a stalled run look busy and cost a healthy trainer its life.
    """
    rows = []
    for update in (1, 2, 3):
        rows.append(_row(update, update * 1000, eval_lift_sd=0.2))
        rows.append(_row(update, update * 1000, eval_lift_sd=0.2))
    _run(tmp_path, "doubled", rows)
    record = collect(tmp_path)[0]
    assert len(record["rows"]) == 3
    assert len(record["evaluations"]) == 3


def test_a_ladder_run_is_reported_as_measured_rather_than_as_never_evaluated(
        tmp_path):
    """``--probe ladder`` writes a rating and no lift, and every selector in
    this module keyed on ``eval_lift_sd``.

    Over a real ladder run's rows, ``collect`` returned ``evaluations: []``,
    ``mean_lift: None`` and ``best_lift: None``, and ``_verdict_of`` rendered
    ("dim", "never evaluated", "This run recorded no evaluations, so nothing
    here says whether it learned anything") over the run's only measurement.
    """
    from cr_sim.train.report import _verdict_of

    rows = [_row(u, u * 1000, ladder_elo=elo, ladder_elo_error=90.0,
                 ladder_mode="greedy", ladder_opponent="ladder",
                 ladder_opponent_ref="random@random",
                 ladder_pinned={"random": 0.0, "clone": 382.0},
                 eval_opponent="ladder", eval_episodes=40)
            for u, elo in ((1, 300.0), (2, 420.0))]
    _run(tmp_path, "rated", rows)
    record = collect(tmp_path)[0]

    assert record["ladder_elo"] == [300.0, 420.0]
    assert record["latest_elo"] == 420.0 and record["best_elo"] == 420.0
    # Never in a field called lift: an Elo and a lift are unrelated scales.
    assert record["mean_lift"] is None and record["best_lift"] is None

    chip, label, sentence = _verdict_of(record)
    assert label != "never evaluated"
    assert "clone" in label and chip == "good", (chip, label)
    assert "+420" in sentence and "not a lift" in sentence

    # And a run rated below the best anchor is not sold as a win.
    rows[-1]["ladder_elo"] = 100.0
    _run(tmp_path, "under", rows)
    under = [r for r in collect(tmp_path) if r["name"] == "under"][0]
    chip, label, _ = _verdict_of(under)
    assert chip == "warn" and label == "rated below clone"


def test_a_rating_says_which_table_its_anchors_were_pinned_by(tmp_path):
    """``ladder_ratings_source`` is written by ``ladder_probe`` and was read by
    nothing on either page.

    An Elo is only a number relative to what the fit held fixed. Two probes
    pinned from two different offline ladders can carry ``ladder_pinned``
    dicts that agree in shape and name -- the same anchors at ratings fitted
    over different battles -- and the pins alone cannot tell them apart. The
    file the pins came out of is what identifies the scale.
    """
    from cr_sim.train.report import _verdict_of

    def rated(name, source=None):
        extra = {"ladder_ratings_source": source} if source else {}
        rows = [_row(u, u * 1000, ladder_elo=elo, ladder_elo_error=90.0,
                     ladder_opponent="ladder",
                     ladder_opponent_ref="random@random",
                     ladder_pinned={"random": 0.0, "expert": 604.0},
                     eval_opponent="ladder", eval_episodes=40, **extra)
                for u, elo in ((1, 300.0), (2, 420.0))]
        return _run(tmp_path, name, rows)

    rated("pinned-by-a-table", "runs/agent-expert-rating/ladder.json")
    rated("pinned-by-nothing")
    records = {r["name"]: r for r in collect(tmp_path)}

    known = records["pinned-by-a-table"]
    assert known["ladder_ratings_source"] ==         "runs/agent-expert-rating/ladder.json"
    assert "runs/agent-expert-rating/ladder.json" in _verdict_of(known)[2]

    # And a probe with no table behind it says so, rather than reading as
    # though it had been pinned by the same one.
    unknown = records["pinned-by-nothing"]
    assert unknown["ladder_ratings_source"] is None
    assert "no recorded ratings table" in _verdict_of(unknown)[2]


def test_a_directory_with_no_metrics_is_skipped(tmp_path):
    (tmp_path / "empty").mkdir()
    _run(tmp_path, "real", [_row(1, 1000)])
    assert [r["name"] for r in collect(tmp_path)] == ["real"]


def test_a_larger_evaluation_replaces_the_inline_readings(tmp_path):
    """Forty battles steer a run; they do not settle one.

    Where a run carries a proper paired evaluation, the index must lead with
    it rather than with the mean of the small ones -- and must say how many
    battles it rests on.
    """
    _run(tmp_path, "settled",
         [_row(i, i * 1000, eval_lift_sd=0.1) for i in range(1, 6)],
         verdict={"episodes": 300, "lift": 0.42, "ci_low": 0.18, "ci_high": 0.66,
                  "eval_opponent": "random"})
    page = render_index(collect(tmp_path))
    assert "beats random" in page
    assert "300 paired battles against random" in page
    assert "+0.420" in page


def test_a_verdict_is_labelled_with_the_opponent_it_was_measured_against(tmp_path):
    """The chip said "beats random" whatever the verdict actually faced.

    A run evaluated against the search expert would have published on this
    page under that label, beside genuine random-opponent lifts -- the same
    confusion that already cost two rounds of invalid comparisons, since the
    control wins 92% of idle matches and 26% of random ones. The number is
    only meaningful with the opponent attached.
    """
    _run(tmp_path, "vs-search",
         [_row(1, 1000, eval_lift_sd=0.1)],
         verdict={"episodes": 60, "lift": 0.30, "ci_low": 0.10, "ci_high": 0.50,
                  "eval_opponent": "search"})
    page = render_index(collect(tmp_path))
    assert "beats search" in page
    assert "60 paired battles against search" in page
    assert "beats random" not in page, (
        "a verdict measured against the search expert was labelled as though "
        "it had beaten the random control")


def test_an_interval_containing_zero_is_not_called_a_win(tmp_path):
    """A positive point estimate whose interval spans zero has not shown
    anything, and this project has already mistaken one for a signal."""
    _run(tmp_path, "unsettled",
         [_row(i, i * 1000, eval_lift_sd=0.2) for i in range(1, 6)],
         verdict={"episodes": 300, "lift": 0.12, "ci_low": -0.05, "ci_high": 0.29})
    page = render_index(collect(tmp_path))
    assert "not distinguishable" in page
    assert "beats random" not in page


def test_a_run_that_was_never_evaluated_says_so(tmp_path):
    _run(tmp_path, "unmeasured", [_row(i, i * 1000) for i in range(1, 4)])
    page = render_index(collect(tmp_path))
    assert "never evaluated" in page


def test_the_index_links_to_every_run(tmp_path):
    _run(tmp_path, "alpha", [_row(1, 1000)])
    _run(tmp_path, "beta", [_row(1, 1000)])
    page = render_index(collect(tmp_path))
    assert 'href="alpha.html"' in page and 'href="beta.html"' in page


def test_the_settings_that_differed_are_shown_beside_the_result(tmp_path):
    """A lift number is unreadable without knowing what it played against."""
    _run(tmp_path, "configured",
         [_row(i, i * 1000, eval_lift_sd=0.3) for i in range(1, 6)],
         config={"reward": "projected", "opponent": "random", "frame_skip": 30})
    page = render_index(collect(tmp_path))
    assert "projected" in page and "random" in page and "frame-skip 30" in page


def test_the_halves_are_compared_rather_than_a_slope_fitted(tmp_path):
    rows = [_row(i, i * 1000, eval_lift_sd=0.1) for i in range(1, 5)]
    rows += [_row(i, i * 1000, eval_lift_sd=0.5) for i in range(5, 9)]
    _run(tmp_path, "trend", rows)
    record = collect(tmp_path)[0]
    assert record["early_lift"] == pytest.approx(0.1)
    assert record["late_lift"] == pytest.approx(0.5)


def test_lifts_measured_under_two_rewards_are_not_averaged_into_one(tmp_path):
    """``max(evaluations)`` was taken over every reading a run wrote, whatever
    produced it.

    A lift is a difference of *returns* divided by the control's own spread,
    so the reward that scored those returns is in the numerator and in the
    denominator both. That is not hypothetical on this machine: every offline
    script here builds ``CRSimEnv`` with no ``reward_weights`` and measures
    under ``simple:shaping=0.01``, while ``cr_sim.train.run.EVAL_REWARD`` pins
    the in-run probe to ``projected:tower=1,elixir=0.3,horizon_seconds=3``.
    The two numbers are in different units, and the mean and the maximum of a
    set spanning both are comparisons between units rather than between
    policies.

    ``eval_opponent`` is identical on every row here on purpose: naming the
    opponent has been forced since ``check_lift_is_named`` landed, and a
    reading that names it correctly still says nothing about what was
    counted.
    """
    from cr_sim.train.report import _verdict_of

    rows = [_row(1, 1000, eval_lift_sd=0.20, eval_opponent="random",
                 eval_reward="simple:shaping=0.01"),
            _row(2, 2000, eval_lift_sd=0.40, eval_opponent="random",
                 eval_reward="simple:shaping=0.01"),
            _row(3, 3000, eval_lift_sd=2.10, eval_opponent="random",
                 eval_reward="projected:elixir=0.3,horizon_seconds=3,tower=1")]
    _run(tmp_path, "two-scales", rows,
         config={"reward": "projected", "opponent": "random"})
    record = collect(tmp_path)[0]

    # Withheld, not guessed. The old code returned 2.10 here, which is the
    # projected-scale reading winning a maximum taken across two scales.
    assert record["best_lift"] is None
    assert record["mean_lift"] is None
    assert record["early_lift"] is None and record["late_lift"] is None
    # And every scale is named, so a reader can see why.
    assert record["lift_scales"] == [
        "random / simple:shaping=0.01",
        "random / projected:elixir=0.3,horizon_seconds=3,tower=1"]

    # The readings themselves are kept. Dropping them would report a run that
    # measured itself three times as one that never measured anything -- the
    # identical bug the ladder branch of _verdict_of exists to fix, one scale
    # down.
    assert record["evaluations"] == [0.20, 0.40, 2.10]
    cls, label, sentence = _verdict_of(record)
    assert label != "never evaluated"
    assert "3 readings" in sentence and "simple:shaping=0.01" in sentence

    page = render_index(collect(tmp_path))
    assert "never evaluated" not in page
    assert "+2.100" not in page, (
        "the largest reading was published as the run's best lift, and it is "
        "on a different scale from the other two")


def test_one_scale_still_gets_its_mean_and_its_best(tmp_path):
    """The guard above must cost nothing where a run really is on one scale.

    Every existing run on this machine records no ``eval_reward`` at all, so
    its readings share the one (absent) scale and must summarise exactly as
    they did before.
    """
    _run(tmp_path, "recorded",
         [_row(i, i * 1000, eval_lift_sd=0.1 * i, eval_opponent="random",
               eval_reward="simple:shaping=0.01") for i in range(1, 5)])
    _run(tmp_path, "legacy",
         [_row(i, i * 1000, eval_lift_sd=0.1 * i) for i in range(1, 5)])
    recorded, legacy = sorted(collect(tmp_path), key=lambda r: r["name"])[::-1]

    assert recorded["name"] == "recorded" and legacy["name"] == "legacy"
    for record in (recorded, legacy):
        assert record["best_lift"] == pytest.approx(0.4)
        assert record["mean_lift"] == pytest.approx(0.25)
        assert record["one_scale"] is True
    assert legacy["lift_scales"] == ["an unnamed opponent / an unrecorded reward"]


def test_the_comparison_table_names_the_unit_a_lift_was_measured_in(tmp_path):
    """The Reward column showed ``config["reward"]`` -- the *training* variant.

    That is not the scale the two lift columns beside it are in. The trainer
    pins its probe to ``EVAL_REWARD`` whatever ``--reward`` was, so a run
    trained on ``projected`` and one trained on ``five-term`` can share a
    measurement scale, and two runs both trained on ``projected`` can be
    measured on different ones. Rendered from the training variant alone,
    those two are the same cell.
    """
    _run(tmp_path, "old-scale",
         [_row(1, 1000, eval_lift_sd=2.16, eval_opponent="random",
               eval_reward="simple:shaping=0.01")],
         config={"reward": "projected", "opponent": "random"})
    _run(tmp_path, "new-scale",
         [_row(1, 1000, eval_lift_sd=1.02, eval_opponent="random",
               eval_reward="projected:elixir=0.3,horizon_seconds=3,tower=1")],
         config={"reward": "projected", "opponent": "random"})
    page = render_index(collect(tmp_path))

    assert "simple:shaping=0.01" in page
    assert "projected:elixir=0.3,horizon_seconds=3,tower=1" in page
    # Both rows still say what they were trained on; the point is that the
    # cell no longer says only that.
    assert page.count("measured on") >= 2

def test_a_ladders_verdict_is_read_as_a_rating_and_never_as_a_lift(tmp_path):
    """``_verdict_of`` indexed ``verdict["ci_low"]`` on every verdict it found.

    A ladder writes a ``verdict.json`` too, and it is a rating. Five of the
    fifteen verdicts on this machine carry ``ladder_elo`` and no lift at all
    -- ``agent-ladder-expert-rung``, ``agent-proposal-headtohead``,
    ``agent-verify-metric``, ``audit-ladder-norandom``,
    ``audit-ladder-sampled`` -- so ``python -m cr_sim.train.report`` raised
    ``KeyError: 'ci_low'`` over the whole directory and **no comparison page
    could be generated at all** while any ladder run was present. The metric
    built to replace the lift is the one that broke the page that reads it.

    The other two ladder verdicts do carry a lift, and it is worse than
    absent. It is one arm's number against the *random control*, flattened in
    beside a whole-graph Elo under ``eval_opponent: "ladder"`` --
    ``audit-ladder-greedy`` pairs a ``ladder_player`` of
    ``headablate-factored`` with the worst-rated entrant's +0.781. Read
    through the lift branch that comes out as "beats ladder", on a number
    measured against random, for a player the top-level fields do not name.
    ``write_verdict`` refuses such a file today unless it says
    ``lift_player`` and ``lift_opponent``; these two predate that clause and
    say neither.
    """
    from cr_sim.train.report import _verdict_of

    rating_rows = [_row(u, u * 1000, ladder_elo=elo, ladder_elo_error=90.0,
                        ladder_opponent="ladder",
                        ladder_opponent_ref="random@random",
                        ladder_pinned={"random": 0.0},
                        eval_opponent="ladder", eval_episodes=100)
                   for u, elo in ((1, 300.0), (2, 419.4))]

    # A rating and nothing else. This is the shape that raised.
    _run(tmp_path, "rating-only", rating_rows,
         verdict={"episodes": 100, "eval_opponent": "ladder",
                  "ladder_elo": 419.4, "ladder_player": "clone",
                  "ladder_pinned": {"random": 0.0}})
    # A rating with one arm's lift flattened in, and nothing saying whose.
    _run(tmp_path, "rating-and-a-lift", rating_rows,
         verdict={"episodes": 100, "eval_opponent": "ladder",
                  "ladder_elo": 419.4, "ladder_player": "headablate-factored",
                  "lift": 0.781, "ci_low": 0.55, "ci_high": 1.01})

    records = {r["name"]: r for r in collect(tmp_path)}
    for name in ("rating-only", "rating-and-a-lift"):
        _cls, label, sentence = _verdict_of(records[name])
        assert "rated" in label, f"{name} was not reported as a rating"
        assert "Elo" in sentence
        # The lift is not published under the ladder's name. arms.json keeps
        # every arm with its own eval_opponent; nothing here has to guess.
        assert "beats" not in label
        assert "+0.781" not in sentence

    # And the page renders at all, which is the part that was broken.
    page = render_index(collect(tmp_path))
    assert "rating-only.html" in page and "rating-and-a-lift.html" in page

    # A verdict that really is a lift is untouched by any of this.
    _run(tmp_path, "a-real-lift", [_row(1, 1000, eval_lift_sd=0.4)],
         verdict={"episodes": 300, "lift": 0.42, "ci_low": 0.18,
                  "ci_high": 0.66, "eval_opponent": "random",
                  "eval_reward": "simple:shaping=0.01"})
    lift = next(r for r in collect(tmp_path) if r["name"] == "a-real-lift")
    assert _verdict_of(lift)[1] == "beats random"
