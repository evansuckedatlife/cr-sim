"""The wording the bot puts in front of someone who just opened Discord.

:mod:`cr_sim.train.bot` answers `/status`, `/run <name>` and `/compare <a>
<b>` with plain text built by `overview`, `describe` and `compare`. None of
that needs a live Discord connection to test -- only fake `runs/`
directories -- so this exercises the wording directly, which is where the
actual risk is: not that the bot fails to connect, but that it tells someone
an unproven result is a proven one, or shows a run's own dashboard when the
name was mistyped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from cr_sim.train.bot import compare, describe, overview


def _row(update, steps, **extra):
    row = {
        "updates": update, "steps": steps, "episodes": steps // 20,
        "steps_per_second": 25.0, "entropy": 4.0,
    }
    row.update(extra)
    return row


def _make_run(base: Path, name: str, rows: list[dict], start_offset: float = 0.0) -> Path:
    """A run directory with metrics and a datable start, like a real one."""
    run = base / name
    run.mkdir()
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    config = run / "config.json"
    config.write_text("{}", encoding="utf-8")
    # Backdated like test_watch.py's ordering fixture, so _started_at (which
    # reads config.json) has something other than "whenever the test ran"
    # to sort by.
    stamp = 1_000_000 + start_offset
    os.utime(config, (stamp, stamp))
    return run


# --------------------------------------------------------------- no runs at all


def test_overview_of_an_empty_runs_directory_says_so_rather_than_an_empty_table(tmp_path):
    """A blank table under a header reads as a bug, not as "nothing yet"."""
    assert overview(tmp_path) == "No runs have recorded anything yet."


# ------------------------------------------------------------- no evaluations


def test_overview_of_a_run_with_no_evaluations_reads_as_not_measured_not_zero(tmp_path):
    """A dash and "not measured" is the honest state before the first eval;
    a run that has not been judged must never render as a 0.000 lift, which
    would read as "evaluated and found equal to random"."""
    _make_run(tmp_path, "alpha", [_row(1, 1000)])
    text = overview(tmp_path)
    assert "--" in text
    assert "not measured" in text


def test_describe_of_a_run_with_no_evaluations_says_so_plainly(tmp_path):
    """/run on a run a minute old should show the run, not silence or a
    crash on the missing eval fields."""
    _make_run(tmp_path, "alpha", [_row(1, 1000)])
    text = describe(tmp_path, "alpha")
    assert "alpha" in text
    assert "no evaluations yet" in text


# ------------------------------------------------------------------ verdict.json


def test_describe_reports_an_interval_that_clears_zero_as_clearing_it(tmp_path):
    """The paired evaluation is the strongest evidence the project has, so
    its wording has to be exact: an interval entirely above zero is the one
    case allowed to sound confident."""
    run = _make_run(tmp_path, "alpha", [_row(1, 1000, eval_lift_sd=0.3,
                                              eval_win=0.4, control_win=0.3)])
    (run / "verdict.json").write_text(json.dumps(
        {"episodes": 300, "lift": 0.22, "ci_low": 0.05, "ci_high": 0.39}),
        encoding="utf-8")

    text = describe(tmp_path, "alpha")

    assert "clears zero" in text
    assert "unproven" not in text


def test_describe_reports_an_interval_that_contains_zero_as_unproven(tmp_path):
    """The failure mode this guards against is a confident-sounding claim
    for a result that has not actually been shown to beat random -- an
    interval that still contains zero must say "unproven", not "clears
    zero", even though the point estimate is positive."""
    run = _make_run(tmp_path, "alpha", [_row(1, 1000, eval_lift_sd=0.1,
                                              eval_win=0.35, control_win=0.3)])
    (run / "verdict.json").write_text(json.dumps(
        {"episodes": 300, "lift": 0.14, "ci_low": -0.02, "ci_high": 0.30}),
        encoding="utf-8")

    text = describe(tmp_path, "alpha")

    assert "contains zero" in text
    assert "unproven" in text
    assert "clears zero" not in text


# ------------------------------------------------------------------- unknown


def test_describe_of_an_unknown_run_name_says_so_instead_of_crashing(tmp_path):
    """A typo in a Discord command is routine, not exceptional, and must
    come back as a message rather than a stack trace in the bot's log."""
    _make_run(tmp_path, "alpha", [_row(1, 1000)])
    text = describe(tmp_path, "does-not-exist")
    assert "No run called" in text
    assert "does-not-exist" in text
    assert "/status" in text


# --------------------------------------------------------------------- order


def test_overview_lists_runs_newest_first_not_alphabetically(tmp_path):
    """/status is asked to see what is happening now, and what is happening
    now is whatever started most recently -- alphabetical order would bury
    a run started five minutes ago under one from last week just because
    its name sorts earlier."""
    # oldest to newest: zulu, alpha, mike -- chosen so neither an
    # alphabetical nor a reverse-alphabetical sort of the names would
    # coincidentally match sorting by start time. A run of names picked
    # without checking that would let this test pass even if overview()
    # sorted by name instead of by _started_at.
    for i, name in enumerate(["zulu", "alpha", "mike"]):
        _make_run(tmp_path, name, [_row(1, 1000)], start_offset=i * 3600)
    newest_first = ["mike", "alpha", "zulu"]
    assert newest_first != sorted(newest_first)
    assert newest_first != sorted(newest_first, reverse=True)

    text = overview(tmp_path)

    positions = [text.index(name) for name in newest_first]
    assert positions == sorted(positions), "newest run did not come first"


# ------------------------------------------------------------------- compare


def test_compare_stitches_two_runs_into_one_message(tmp_path):
    """/compare exists to be read side by side; both names and both
    summaries have to actually be present in one reply."""
    _make_run(tmp_path, "alpha", [_row(1, 1000, eval_lift_sd=0.2,
                                       eval_win=0.4, control_win=0.3)])
    _make_run(tmp_path, "beta", [_row(1, 1000, eval_lift_sd=0.6,
                                      eval_win=0.6, control_win=0.4)])

    text = compare(tmp_path, "alpha", "beta")

    assert "alpha" in text and "beta" in text
    assert text.index("alpha") < text.index("beta")
    assert "\n\n" in text, "the two runs are not visually separated"


def test_compare_with_an_unknown_second_run_names_the_mistake_not_a_crash(tmp_path):
    """Half a valid /compare should still fail readably instead of raising,
    since a mistyped second name is as routine as a mistyped first one."""
    _make_run(tmp_path, "alpha", [_row(1, 1000)])

    text = compare(tmp_path, "alpha", "nope")

    assert "alpha" in text
    assert "No run called" in text
