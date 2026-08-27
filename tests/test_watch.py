"""The training progress view.

Its job is to answer one question -- is this getting better -- from a file that
is being appended to while it reads. So the things worth testing are that it
survives a half-written line, that it distinguishes "no evaluations yet" from
"evaluated at zero", and that it puts the honest number in front rather than
the flattering one.

That last point is the reason this exists at all. The trainer's own return is
measured while the policy is exploring and has run about eighteen points
optimistic against a paired-seed control; a progress page that led with it
would show a run improving when it was not.
"""

from __future__ import annotations

import json

import pytest

from cr_sim.train.watch import read_metrics, render, summarise


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _row(update, steps, **extra):
    row = {
        "updates": update, "steps": steps, "episodes": steps // 20,
        "steps_per_second": 25.0, "entropy": 4.3, "value_loss": 1.0,
        "policy_loss": -0.01, "mean_return": 0.2, "win_rate": 0.3,
        "noop_fraction": 0.01,
    }
    row.update(extra)
    return row


# ------------------------------------------------------------------ reading


def test_a_half_written_final_line_is_dropped_not_fatal(tmp_path):
    """The run appends while this reads, so the last line can be incomplete.

    That is normal rather than an error, and a viewer that crashed on it would
    fail exactly when someone was checking on a long run.
    """
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps(_row(1, 1000)) + "\n" + '{"updates": 2, "steps": 20',
        encoding="utf-8",
    )
    rows = read_metrics(path)
    assert len(rows) == 1 and rows[0]["updates"] == 1


def test_a_missing_file_reads_as_empty(tmp_path):
    """Asking about a run that has not written anything yet is ordinary."""
    assert read_metrics(tmp_path / "nope.jsonl") == []


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps(_row(1, 100)) + "\n\n\n" + json.dumps(_row(2, 200)) + "\n",
        encoding="utf-8",
    )
    assert len(read_metrics(path)) == 2


# ---------------------------------------------------------------- summarising


def test_no_evaluations_is_distinct_from_evaluating_at_zero():
    """One means "not measured yet", the other means "no better than random",
    and showing a zero for the first would be a lie about a run that has not
    been judged."""
    none_yet = summarise([_row(1, 1000)])
    assert none_yet["latest_lift"] is None
    assert none_yet["evaluations"] == 0

    measured = summarise([_row(1, 1000, eval_lift_sd=0.0, eval_win=0.3, control_win=0.3)])
    assert measured["latest_lift"] == 0.0
    assert measured["evaluations"] == 1


def test_the_best_evaluation_is_kept_not_the_last(tmp_path):
    """A run that peaks and regresses should still report the peak, because
    that is the checkpoint that was saved."""
    rows = [
        _row(1, 1000, eval_lift_sd=0.1),
        _row(2, 2000, eval_lift_sd=0.8),
        _row(3, 3000, eval_lift_sd=-0.2),
    ]
    summary = summarise(rows)
    assert summary["best_lift"] == 0.8
    assert summary["best_at_steps"] == 2000
    assert summary["latest_lift"] == -0.2


def test_an_empty_run_summarises_without_raising():
    assert summarise([])["updates"] == 0


# ------------------------------------------------------------------ the page


def test_the_page_leads_with_the_lift_not_the_rollout_return():
    """The rollout return is the flattering number and the misleading one."""
    page = render([_row(1, 1000, eval_lift_sd=0.4, eval_win=0.5, control_win=0.3)], "run")
    assert page.index("lift vs control") < page.index("rollout win")
    assert "eighteen points optimistic" in page, "the caveat is not stated"


def test_the_page_embeds_the_series_it_draws():
    rows = [
        _row(1, 1000, eval_lift_sd=0.1, eval_win=0.3, control_win=0.3),
        _row(2, 2000, eval_lift_sd=0.4, eval_win=0.5, control_win=0.3),
    ]
    page = render(rows, "run")
    payload = json.loads(page.split("const DATA = ", 1)[1].split(";\n", 1)[0])
    run = payload["runs"]["run"]
    assert run["series"]["lift"] == [[1000, 0.1], [2000, 0.4]]
    assert run["summary"]["best_lift"] == 0.4
    assert payload["order"] == ["run"]


def test_the_page_renders_before_any_evaluation_exists():
    """Opening it a minute into a run should show the run, not an error."""
    page = render([_row(1, 1000)], "run")
    assert "no evaluations yet" in page
    assert "<svg" in page or "empty" in page


def test_the_page_carries_its_own_logic_and_data():
    """No build step and no fetch -- it is opened from disk while a run runs.

    Web fonts are the one exception, and they are allowed only because they
    degrade silently: the page must name a real fallback stack so it still
    reads correctly on a machine with no network, which is the state this is
    most likely to be opened in.
    """
    page = render([_row(1, 1000, eval_lift_sd=0.2)], "run")
    assert "<script" in page and "</html>" in page
    assert "const DATA = " in page, "the page fetches its data instead of carrying it"

    external = [
        line for line in page.splitlines()
        if ("http://" in line or "https://" in line)
        and "fonts.googleapis.com" not in line
        and "fonts.gstatic.com" not in line
    ]
    assert not external, f"page reaches outside for {external}"
    assert "ui-sans-serif" in page and "ui-monospace" in page, "no fallback stack"


# ------------------------------------------------------ explaining the numbers


def test_every_metric_on_the_page_is_explained_somewhere_on_it():
    """A dashboard of unexplained numbers is a dashboard nobody can act on.

    Two of these were actively misread on this project: value loss was called
    mis-calibrated when it was only measured against a different reward scale,
    and a pair of positive lift readings were called a trend when six of them
    averaged to noise. The glossary exists so the page cannot be read that way
    again.
    """
    page = render([_row(1, 1000, eval_lift_sd=0.3, eval_win=0.4, control_win=0.3)], "run")
    for term in ("Lift vs control", "Explained variance", "Entropy",
                 "Value loss and return spread", "Rollout win rate",
                 "Pass rate", "Throughput"):
        assert term in page, f"{term!r} is shown but never explained"


def test_the_page_says_what_zero_means_for_the_two_numbers_that_need_it():
    """Both have a meaningful zero that is not obvious from the value alone."""
    page = render([_row(1, 1000, eval_lift_sd=0.0)], "run")
    assert "no better than random" in page
    assert "no better than guessing the average" in page


def test_the_headline_reads_noise_as_noise():
    """A lift inside the control's own bounce must not be presented as a win.

    The failure this prevents is real: an early +0.23 was reported as the
    first positive signal, and six evaluations later the mean was +0.04.
    """
    page = render([_row(1, 1000, eval_lift_sd=0.2, eval_win=0.4, control_win=0.3)], "run")
    assert "Indistinguishable from random" in page


def test_the_page_carries_the_critic_series_it_now_leads_on():
    rows = [
        _row(1, 1000, explained_variance=-0.02, ret_std=0.5),
        _row(2, 2000, explained_variance=0.31, ret_std=0.6),
    ]
    payload = json.loads(
        render(rows, "run").split("const DATA = ", 1)[1].split(";\n", 1)[0])
    series = payload["runs"]["run"]["series"]
    assert series["explained_variance"] == [[1000, -0.02], [2000, 0.31]]
    assert series["ret_std"] == [[1000, 0.5], [2000, 0.6]]


# ---------------------------------------------------- charts with thin data
#
# The page renders in JavaScript, so asserting on the HTML string tests the
# template rather than the behaviour -- `no evaluations yet` appears in the
# script source whatever the data is, which is why the test that looked for it
# passed for a year without checking anything. These run the page's own
# functions under node and assert on what they actually return.

import shutil
import subprocess

node = shutil.which("node")
needs_node = pytest.mark.skipif(node is None, reason="needs node to run the page's JS")


def _call(rows, expression):
    """Evaluate `expression` against the page's own chart code."""
    page = render(rows, "run")
    script = page.split("<script>", 1)[1].split("</script>", 1)[0]
    # Everything up to the start-up block is pure functions; the block itself
    # touches the DOM and is not what is under test here.
    body = script.split("(function start()", 1)[0]
    result = subprocess.run(
        [node, "-e", body + chr(10) + "process.stdout.write(String(" + expression + "));"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@needs_node
def test_one_evaluation_is_reported_rather_than_called_none():
    """A single reading cannot be drawn as a line, but it is not nothing.

    The page printed "no evaluations yet" over the top of a real evaluation
    for the first hour of a run -- the one thing a progress view must never
    do, which is deny a measurement it is holding.
    """
    rows = [_row(1, 15360, eval_lift_sd=0.118, eval_win=0.32, control_win=0.30)]
    out = _call(rows, "chart('t','n',[{name:'lift',color:'#000',points:DATA.runs['run'].series.lift}],{zero:true})")
    assert "no evaluations yet" not in out, "denied an evaluation it was given"
    assert "0.118" in out, "the single reading is not shown"
    assert "one reading so far" in out


@needs_node
def test_no_evaluations_still_reads_as_none():
    out = _call([_row(1, 1000)], "chart('t','n',[{name:'lift',color:'#000',points:DATA.runs['run'].series.lift}],{})")
    assert "no evaluations yet" in out


@needs_node
def test_two_readings_at_the_same_step_do_not_produce_a_degenerate_axis():
    """Identical endpoints rendered as "15k -> 15k", which reads as a bug."""
    rows = [_row(1, 15360, eval_lift_sd=0.1), _row(2, 15360, eval_lift_sd=0.2)]
    out = _call(rows, "chart('t','n',[{name:'lift',color:'#000',points:DATA.runs['run'].series.lift}],{})")
    assert out.count("15k</text>") == 0
    assert "15k decisions" in out


@needs_node
def test_the_legend_is_not_clipped_by_a_long_series_name():
    """Fixed right padding cut "rollout win" down to "rollout wc"."""
    rows = [_row(i, i * 1000, win_rate=0.2 + i / 100) for i in range(1, 5)]
    out = _call(rows, "chart('t','n',[{name:'rollout win',color:'#000',points:DATA.runs['run'].series.rollout_win}],{})")
    assert "rollout win</text>" in out, "series label truncated"
    # the label starts inside the viewBox and its text has room to the edge
    import re
    x = float(re.search(r'x="([0-9.]+)"[^>]*>rollout win</text>', out).group(1))
    assert x + len("rollout win") * 6.3 <= 640, "label runs past the right edge"


# ------------------------------------------------------------ several at once


def test_several_runs_are_carried_on_one_page():
    """Comparing runs is the common case, and doing it across two files is
    how a difference gets missed."""
    from cr_sim.train.watch import render_multi

    page = render_multi([
        ("alpha", [_row(1, 1000, eval_lift_sd=0.4)], True),
        ("beta", [_row(1, 1000, eval_lift_sd=-0.1)], False),
    ])
    payload = json.loads(page.split("const DATA = ", 1)[1].split(";\n", 1)[0])
    assert payload["order"] == ["alpha", "beta"]
    assert payload["runs"]["alpha"]["summary"]["latest_lift"] == 0.4
    assert payload["runs"]["beta"]["summary"]["latest_lift"] == -0.1


def test_a_finished_run_is_not_presented_as_moving():
    """The pip beside a tab says whether that run is still writing. Showing a
    run that stopped hours ago as live is how a dead watcher went unnoticed
    for an hour on this project."""
    from cr_sim.train.watch import render_multi

    page = render_multi([("done", [_row(1, 1000)], False)])
    payload = json.loads(page.split("const DATA = ", 1)[1].split(";\n", 1)[0])
    assert payload["runs"]["done"]["live"] is False


def test_the_page_offers_a_split_view_only_when_there_is_something_to_compare():
    from cr_sim.train.watch import render_multi

    page = render_multi([("only", [_row(1, 1000)], True)])
    assert "split-toggle" in page
    assert "order.length < 2" in page, "the toggle is never hidden for one run"


def test_runs_are_ordered_by_when_they_started(tmp_path):
    """Chronological, not alphabetical.

    Sorting by name puts today's run between two from last week, and the
    question this page answers is almost always what changed since the
    previous one -- which only reads in the order they happened.
    """
    import os

    from cr_sim.train.watch import _started_at

    # Deliberately reverse-alphabetical, so a name sort would fail this.
    order = ["zulu", "mike", "alpha"]
    for i, name in enumerate(order):
        run = tmp_path / name
        run.mkdir()
        (run / "metrics.jsonl").write_text(
            json.dumps(_row(1, 1000)) + chr(10), encoding="utf-8")
        config = run / "config.json"
        config.write_text("{}", encoding="utf-8")
        # Backdated, so the real _started_at (which takes the earlier of
        # creation and modification) reads these as the start times.
        stamp = 1_000_000 + i * 3600
        os.utime(config, (stamp, stamp))

    found = sorted((d for d in tmp_path.iterdir() if d.is_dir()), key=_started_at)
    assert [d.name for d in found] == order
    assert found != sorted(tmp_path.iterdir()), "this would pass on a name sort"


def test_a_run_without_a_config_still_sorts(tmp_path):
    """Runs from before the config was written must not crash the ordering."""
    from cr_sim.train.watch import _started_at

    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "metrics.jsonl").write_text(
        json.dumps(_row(1, 1000)) + chr(10), encoding="utf-8")
    assert _started_at(bare) > 0
    assert _started_at(tmp_path / "missing") == 0.0


# ------------------------------------------------------------------ the ladder


def test_the_ladder_reaches_the_page():
    """Self-play measures itself against its own past, and that has to show.

    It is the most sensitive number available: the reference is fixed and
    roughly the right difficulty, where lift against a random control has a
    spread wide enough that a +0.375 reading on this project measured -0.033
    when replayed over 300 battles.
    """
    from cr_sim.train.watch import render_multi

    rows = [
        _row(1, 1000, ancestor_win=0.30, ancestor_loss=0.50, ancestor_age=1),
        _row(2, 2000, ancestor_win=0.62, ancestor_loss=0.21, ancestor_age=3),
    ]
    page = render_multi([("sp", rows, True)])
    payload = json.loads(page.split("const DATA = ", 1)[1].split(";\n", 1)[0])
    run = payload["runs"]["sp"]
    assert run["series"]["ancestor_win"] == [[1000, 0.30], [2000, 0.62]]
    assert run["summary"]["ancestor_win"] == 0.62
    assert run["summary"]["ancestor_age"] == 3


def test_a_run_without_a_ladder_says_so_rather_than_showing_zero():
    """Most runs were not self-play. Reporting 0% would read as 'it loses
    every one', which is a different claim from 'this was never measured'."""
    from cr_sim.train.watch import render_multi

    payload = json.loads(
        render_multi([("plain", [_row(1, 1000)], True)])
        .split("const DATA = ", 1)[1].split(";\n", 1)[0])
    assert payload["runs"]["plain"]["summary"]["ancestor_win"] is None
    assert payload["runs"]["plain"]["series"]["ancestor_win"] == []


# ------------------------------------------------------------- what is running


def test_the_page_lists_the_jobs_it_was_given():
    """Three copies of one script once ran at the same time, each holding
    1.3GB and most of a core, and nothing on this page said so -- it only knew
    about runs that write metrics, and a script that has not finished writes
    nothing. Task Manager was the only way to find out."""
    from cr_sim.train.watch import render_multi

    page = render_multi(
        [("a", [_row(1, 1000)], True)],
        jobs=[{"kind": "training", "name": "a", "pid": 42,
               "memory_mb": 900, "age_seconds": 600, "processes": 3}])
    payload = json.loads(page.split("const DATA = ", 1)[1].split(";\n", 1)[0])
    assert payload["jobs"][0]["kind"] == "training"
    assert payload["jobs"][0]["pid"] == 42


def test_no_jobs_is_a_state_the_page_can_render():
    from cr_sim.train.watch import render_multi

    payload = json.loads(
        render_multi([("a", [_row(1, 1000)], True)])
        .split("const DATA = ", 1)[1].split(";\n", 1)[0])
    assert payload["jobs"] == []


def test_processes_of_one_job_are_grouped():
    """Launching a module spawns a shell that spawns the worker, so every job
    appears at least twice. Listing all of them buries the case this exists to
    surface: one job running more than once by mistake."""
    from cr_sim.train.watch import _group

    grouped = _group([
        {"kind": "training", "name": "run-a", "pid": 1, "memory_mb": 20,
         "age_seconds": 900.0},
        {"kind": "training", "name": "run-a", "pid": 2, "memory_mb": 1000,
         "age_seconds": 890.0},
        {"kind": "cloning a policy", "name": None, "pid": 3, "memory_mb": 1200,
         "age_seconds": 60.0},
    ])
    assert len(grouped) == 2
    training = next(g for g in grouped if g["kind"] == "training")
    assert training["processes"] == 2
    assert training["memory_mb"] == 1020
    # The oldest process is the one that was launched; its children are newer.
    assert training["pid"] == 1


def test_one_job_started_three_times_is_flagged():
    """The case that actually happened, and that grouping nearly hid.

    Three copies of one script ran at once with no --name to tell them apart,
    so they collapse into a single entry. What gives them away is the process
    count: launching a module costs a shell and a worker, so more than two
    means the job was started more than once.
    """
    from cr_sim.train.watch import _group

    grouped = _group([
        {"kind": "cloning a policy", "name": None, "pid": pid,
         "memory_mb": 1200, "age_seconds": 60.0}
        for pid in range(1, 7)
    ])
    assert len(grouped) == 1
    assert grouped[0]["processes"] == 6
    assert grouped[0]["suspicious"], "three simultaneous clones went unflagged"


def test_a_training_run_with_worker_processes_is_not_flagged():
    """Environment workers are the point of --workers, not a mistake."""
    from cr_sim.train.watch import _group

    grouped = _group([
        {"kind": "training", "name": "run-a", "pid": pid, "memory_mb": 400,
         "age_seconds": 900.0}
        for pid in range(1, 8)
    ])
    assert grouped[0]["processes"] == 7
    assert not grouped[0]["suspicious"]


def test_a_single_job_is_not_flagged():
    from cr_sim.train.watch import _group

    grouped = _group([
        {"kind": "measuring the expert", "name": None, "pid": 1,
         "memory_mb": 20, "age_seconds": 30.0},
        {"kind": "measuring the expert", "name": None, "pid": 2,
         "memory_mb": 400, "age_seconds": 28.0},
    ])
    assert grouped[0]["processes"] == 2
    assert not grouped[0]["suspicious"]
