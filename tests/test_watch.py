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
    assert payload["series"]["lift"] == [[1000, 0.1], [2000, 0.4]]
    assert payload["summary"]["best_lift"] == 0.4
    assert payload["title"] == "run"


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
    assert payload["series"]["explained_variance"] == [[1000, -0.02], [2000, 0.31]]
    assert payload["series"]["ret_std"] == [[1000, 0.5], [2000, 0.6]]


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
    out = _call(rows, "chart('t','n',[{name:'lift',color:'#000',points:DATA.series.lift}],{zero:true})")
    assert "no evaluations yet" not in out, "denied an evaluation it was given"
    assert "0.118" in out, "the single reading is not shown"
    assert "one reading so far" in out


@needs_node
def test_no_evaluations_still_reads_as_none():
    out = _call([_row(1, 1000)], "chart('t','n',[{name:'lift',color:'#000',points:DATA.series.lift}],{})")
    assert "no evaluations yet" in out


@needs_node
def test_two_readings_at_the_same_step_do_not_produce_a_degenerate_axis():
    """Identical endpoints rendered as "15k -> 15k", which reads as a bug."""
    rows = [_row(1, 15360, eval_lift_sd=0.1), _row(2, 15360, eval_lift_sd=0.2)]
    out = _call(rows, "chart('t','n',[{name:'lift',color:'#000',points:DATA.series.lift}],{})")
    assert out.count("15k</text>") == 0
    assert "15k decisions" in out


@needs_node
def test_the_legend_is_not_clipped_by_a_long_series_name():
    """Fixed right padding cut "rollout win" down to "rollout wc"."""
    rows = [_row(i, i * 1000, win_rate=0.2 + i / 100) for i in range(1, 5)]
    out = _call(rows, "chart('t','n',[{name:'rollout win',color:'#000',points:DATA.series.rollout_win}],{})")
    assert "rollout win</text>" in out, "series label truncated"
    # the label starts inside the viewBox and its text has room to the edge
    import re
    x = float(re.search(r'x="([0-9.]+)"[^>]*>rollout win</text>', out).group(1))
    assert x + len("rollout win") * 6.3 <= 640, "label runs past the right edge"
