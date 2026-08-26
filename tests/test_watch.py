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


def test_the_page_is_self_contained():
    """No build step, no CDN -- it is opened from disk while a run continues."""
    page = render([_row(1, 1000, eval_lift_sd=0.2)], "run")
    assert "http://" not in page and "https://" not in page
    assert "<script" in page and "</html>" in page
