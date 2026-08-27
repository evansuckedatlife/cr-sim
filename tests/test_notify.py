"""Deciding what is worth interrupting someone for.

:mod:`cr_sim.train.notify` polls ``runs/`` on a timer and turns what changed
into Discord messages. The interesting bugs here are not in the network code
-- they are in the decision of *whether* to say something: a bot that
announces every run it finds on startup is a bot that gets muted within a
day, and a bot that calls noise a trend is a bot that gets ignored the day it
is right. ``changes()`` is built to be tested without a network specifically
so that decision can be checked directly, which is what this file does.
"""

from __future__ import annotations

import json
from pathlib import Path

from cr_sim.train.notify import (
    NOISE,
    STALE_SECONDS,
    changes,
    sparkline,
    _load_state,
    _save_state,
)


def _row(update, steps, **extra):
    row = {
        "updates": update, "steps": steps, "episodes": steps // 20,
        "steps_per_second": 25.0, "entropy": 4.0,
    }
    row.update(extra)
    return row


def _write(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _fields(event) -> dict[str, str]:
    return {name: value for name, value, _inline in event.fields}


# ------------------------------------------------------------ first sighting


def test_a_run_seen_for_the_first_time_produces_no_events(tmp_path):
    """A bot that announces every run it finds on startup gets muted.

    Pointing this at a `runs/` directory that already has weeks of finished
    runs in it must not replay all of them into the channel -- the first
    look at any run is recorded, not announced.
    """
    run = tmp_path / "alpha"
    _write(run, [_row(1, 1000),
                 _row(2, 2000, eval_lift_sd=0.4, eval_win=0.5, control_win=0.3)])

    events, state = changes(tmp_path, {})

    assert events == []
    assert state["alpha"]["evaluations"] == 1


# -------------------------------------------------------------- evaluations


def test_a_new_evaluation_produces_exactly_one_event_carrying_the_lift(tmp_path):
    """The lift is the number the whole system exists to report; it has to
    actually be on the event, not just implied by an evaluation count."""
    run = tmp_path / "alpha"
    _write(run, [_row(1, 1000)])
    _, state = changes(tmp_path, {})  # first sighting, seeded silently

    _write(run, [_row(1, 1000),
                 _row(2, 2000, eval_lift_sd=0.6, eval_win=0.6, control_win=0.4)])
    events, state = changes(tmp_path, state)

    assert len(events) == 1
    event = events[0]
    assert event.kind == "evaluation"
    assert event.run == "alpha"
    assert "+0.600" in _fields(event)["lift"]


def test_the_same_data_seen_twice_produces_no_second_event(tmp_path):
    """A watcher polls on a timer, so most polls see nothing new -- the
    common case is silence, and a duplicate ping every poll would be as bad
    as no ping at all."""
    run = tmp_path / "alpha"
    _write(run, [_row(1, 1000)])
    _, state = changes(tmp_path, {})  # first sighting, seeded silently

    _write(run, [_row(1, 1000),
                 _row(2, 2000, eval_lift_sd=0.6, eval_win=0.6, control_win=0.4)])
    first_events, state = changes(tmp_path, state)
    assert len(first_events) == 1

    second_events, state = changes(tmp_path, state)  # nothing changed
    assert second_events == []


def test_a_lift_inside_the_noise_band_is_not_described_as_an_improvement(tmp_path):
    """The failure this guards against already happened: an early +0.375 was
    reported here as the first positive signal, and it measured -0.033 on
    retest over more battles. A reading inside NOISE must read as noise, not
    as a win, or the next borderline number gets over-trusted the same way.
    """
    run = tmp_path / "alpha"
    _write(run, [_row(1, 1000)])
    _, state = changes(tmp_path, {})

    _write(run, [_row(1, 1000),
                 _row(2, 2000, eval_lift_sd=0.1, eval_win=0.35, control_win=0.3)])
    events, state = changes(tmp_path, state)

    assert len(events) == 1
    lift_field = _fields(events[0])["lift"]
    assert "inside the noise" in lift_field
    assert "better than random" not in lift_field
    assert 0.1 < NOISE, "the fixture must actually be inside the band being tested"


# ------------------------------------------------------------ quiet vs done


def test_a_stale_run_is_reported_as_stopped_only_once(tmp_path):
    """Silence past STALE_SECONDS means the process died, not that it is
    slow -- and once said, it must not be repeated on every subsequent poll
    of the same dead run, or a watcher left running overnight spams the
    channel with the same corpse."""
    run = tmp_path / "alpha"
    _write(run, [_row(1, 1000)])
    mtime = (run / "metrics.jsonl").stat().st_mtime

    _, state = changes(tmp_path, {}, now=mtime + 1)  # first sighting, fresh
    assert state["alpha"]["live"] is True

    events, state = changes(tmp_path, state, now=mtime + STALE_SECONDS + 10)
    assert len(events) == 1
    assert events[0].kind == "stopped"
    assert "stopped" in events[0].title
    assert "No update in seven minutes." in events[0].body

    events, state = changes(tmp_path, state, now=mtime + STALE_SECONDS + 20)
    assert events == [], "the same dead run was reported stopped a second time"


def test_a_run_that_reached_total_steps_is_reported_as_finished_not_stopped(tmp_path):
    """finished and stopped are different messages, and the distinction
    matters at 3am: one says the run succeeded and can be left alone, the
    other says something needs attention."""
    run = tmp_path / "alpha"
    _write(run, [_row(1, 1000, total_steps=1000)])
    mtime = (run / "metrics.jsonl").stat().st_mtime

    _, state = changes(tmp_path, {}, now=mtime + 1)
    events, state = changes(tmp_path, state, now=mtime + STALE_SECONDS + 10)

    assert len(events) == 1
    assert events[0].kind == "finished"
    assert "finished" in events[0].title
    assert "No update" not in events[0].body


# ---------------------------------------------------------------- sparkline


def test_sparkline_of_fewer_than_two_points_is_empty():
    """A single reading has no trend to draw, and drawing one anyway would
    claim a shape that is not there."""
    assert sparkline([]) == ""
    assert sparkline([0.5]) == ""


def test_sparkline_of_constant_values_is_flat_not_a_division_by_zero():
    """high - low is zero when every reading is identical; that has to fall
    back to the lowest block for all of them rather than raising."""
    assert sparkline([0.3, 0.3, 0.3]) == "▁▁▁"


def test_sparkline_of_a_rising_series_climbs_from_low_to_high():
    out = sparkline([0.0, 0.5, 1.0])
    assert out[0] == "▁"
    assert out[-1] == "█"
    assert len(out) == 3


# -------------------------------------------------------------- persistence


def test_state_round_trips_through_disk_so_a_restart_does_not_replay_history(tmp_path):
    """The watcher is meant to survive being restarted -- a crash, a reboot,
    a deploy -- without re-announcing every evaluation a run has ever
    produced. That only works if what gets saved to `.notified.json` and
    read back is the same state, not a lossy summary of it."""
    run = tmp_path / "alpha"
    _write(run, [_row(1, 1000),
                 _row(2, 2000, eval_lift_sd=0.5, eval_win=0.55, control_win=0.35)])

    events, state = changes(tmp_path, {})
    assert events == []  # first sighting; the evaluation is already in this data
    _save_state(tmp_path, state)

    reloaded = _load_state(tmp_path)
    events, state = changes(tmp_path, reloaded)
    assert events == [], "a restart replayed history that had already been reported"
