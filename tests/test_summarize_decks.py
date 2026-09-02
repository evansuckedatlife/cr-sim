"""The arithmetic the zero-shot verdict is read off.

``scripts/summarize_decks.py`` turns per-battle returns into the one number
worth quoting -- the encoder's return minus the lookup's, paired battle by
battle -- and a sign error or a wrong denominator there is a conclusion, not a
crash. None of it needs an environment or a network, so all of it can be
checked against returns whose answer is known by construction.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_decks import head_to_head, interval, main  # noqa: E402


def _row(name, mode, differences, *, spread=2.0, deck=("A",), label="unseen-00"):
    return {
        "deck": list(deck), "deck_label": label, "name": name, "mode": mode,
        "checkpoint": f"runs/{name}/cloned.pt", "head": "flat",
        "observation": "v1", "episodes": len(differences),
        "eval_opponent": "random", "control_win": 0.26,
        "control_spread": spread, "differences": list(differences),
        "win": 0.5, "loss": 0.1, "draw": 0.4,
        "lift": float(np.mean(differences) / (spread or 1.0)),
        "ci_low": 0.0, "ci_high": 0.0, "return": 0.0, "crowns": 0.0,
    }


def test_the_head_to_head_gap_is_the_two_policies_differenced_per_battle():
    """Known returns, known answer. The lookup beat the control by 1 and 2
    crowns; the encoder by 3 and 5. The gap is 2 and 3, over a spread of 2."""
    base = _row("lookup", "greedy", [1.0, 2.0])
    treat = _row("encoder", "greedy", [3.0, 5.0])
    assert np.allclose(head_to_head(base, treat), [1.0, 1.5])


def test_the_control_cancels_out_of_the_head_to_head_gap():
    """The reason this is quoted instead of two overlapping intervals.

    Shift the control by any amount and both arms' differences move together,
    so the gap between them must not move at all. If it did, the number would
    be measuring the control rather than the two policies.
    """
    policy_a, policy_b = np.array([2.0, -1.0, 4.0]), np.array([3.0, 0.5, 4.0])
    first, second = np.array([0.0, 0.0, 0.0]), np.array([5.0, -2.0, 1.0])
    gaps = []
    for control in (first, second):
        gaps.append(head_to_head(_row("lookup", "greedy", policy_a - control),
                                 _row("encoder", "greedy", policy_b - control)))
    assert np.allclose(gaps[0], gaps[1])
    assert np.allclose(gaps[0], (policy_b - policy_a) / 2.0)


def test_the_gap_is_signed_towards_the_treatment():
    """A sign error here reverses the verdict and nothing else changes."""
    worse = head_to_head(_row("lookup", "greedy", [4.0, 4.0]),
                         _row("encoder", "greedy", [1.0, 1.0]))
    assert worse.mean() < 0


def test_the_gap_is_quoted_in_control_standard_deviations():
    """The unit every other lift in this project is in. A gap divided by
    something else is not comparable to the parity numbers beside it."""
    base = _row("lookup", "greedy", [0.0, 0.0], spread=4.0)
    treat = _row("encoder", "greedy", [8.0, 8.0], spread=4.0)
    assert np.allclose(head_to_head(base, treat), [2.0, 2.0])


def test_a_zero_spread_control_does_not_divide_by_zero():
    """A control that drew every battle has no spread. That is a degenerate
    deck, not a division by zero, and it must not put inf into the summary."""
    base = _row("lookup", "greedy", [1.0], spread=0.0)
    treat = _row("encoder", "greedy", [2.0], spread=0.0)
    assert np.all(np.isfinite(head_to_head(base, treat)))


def test_an_interval_needs_more_than_one_sample():
    """One deck is not a sample of decks, and an interval drawn over it would
    read as a measurement."""
    assert interval([1.0]) is None
    mean, low, high = interval([1.0, 2.0, 3.0])
    assert low < mean < high
    assert mean == pytest.approx(2.0)


def test_two_files_disagreeing_about_a_deck_are_refused(tmp_path, capsys):
    """A sweep sliced wrongly puts two different decks under one label.
    Merging them silently would report one deck measured twice as two decks,
    with an interval half the width it earned."""
    first, second = tmp_path / "p0.json", tmp_path / "p1.json"
    first.write_text(json.dumps([
        _row("lookup", "greedy", [1.0, 2.0], deck=("Knight",)),
        _row("encoder", "greedy", [1.0, 2.0], deck=("Knight",))]))
    second.write_text(json.dumps([
        _row("lookup", "greedy", [1.0, 2.0], deck=("Pekka",)),
        _row("encoder", "greedy", [1.0, 2.0], deck=("Pekka",))]))
    with pytest.raises(SystemExit, match="appears twice with different cards"):
        main([str(first), str(second), "--baseline", "lookup",
              "--treatment", "encoder"])


def test_the_training_deck_is_kept_out_of_the_unseen_aggregate(tmp_path, capsys):
    """The anchor is printed and must not be pooled into the generalisation
    number: a policy scoring well on the deck it trained on is the opposite of
    the claim being made."""
    rows = []
    for label, gap in (("training-deck", 10.0), ("unseen-00", 1.0),
                       ("unseen-01", 1.0)):
        rows.append(_row("lookup", "greedy", [0.0, 0.0], label=label,
                         deck=(label,)))
        rows.append(_row("encoder", "greedy", [gap, gap], label=label,
                         deck=(label,)))
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows))
    main([str(path), "--baseline", "lookup", "--treatment", "encoder"])
    printed = capsys.readouterr().out
    # 1.0 / spread 2.0 == +0.500 pooled over the two unseen decks. Had the
    # training deck been included the mean would be +2.000.
    assert "+0.500" in printed
    assert "encoder ahead on 2 of 2 unseen decks" in printed
