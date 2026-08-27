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
         verdict={"episodes": 300, "lift": 0.42, "ci_low": 0.18, "ci_high": 0.66})
    page = render_index(collect(tmp_path))
    assert "beats random" in page
    assert "300 paired battles" in page
    assert "+0.420" in page


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
