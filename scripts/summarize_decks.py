"""Read ``evaluate_decks.py``'s rows and answer the question they were for.

Three summaries, because they are three different claims and this project has
already put two of them on one axis once:

*   **Per deck**, each arm against that deck's own random control. What a
    reader checks a headline against.
*   **Pooled over decks**, per-battle differences standardised by each deck's
    own control spread before pooling. Decks differ enormously in how noisy
    they are -- a Golem mirror at tower level 5 is a different variance
    regime from a Bats mirror -- so pooling raw returns would weight the
    loudest decks most.
*   **Head to head**, the encoder's return minus the lookup's, per battle.
    This is the sharpest of the three and the one worth quoting: both arms
    played the *same* seeds on the *same* decks against the *same* control, so
    the control cancels exactly and what is left is a paired difference
    between the two policies. Comparing their two confidence intervals by eye
    instead throws that pairing away and needs several times the battles to
    say the same thing.

The deck-level interval is reported beside the battle-level one and is the
conservative reading. A deck is the unit a generalisation claim is made over:
1,500 battles across ten decks is ten samples of "a deck this policy has never
seen", not 1,500, and the battle-level interval will always look tighter than
the claim deserves.

    python scripts/summarize_decks.py runs/cardstat-parity/decks.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def head_to_head(base, treat):
    """``treat`` minus ``base``, per battle, in control standard deviations.

    Both arms played the same seeds on the same deck against the same control,
    so the control cancels exactly:
    ``(treat - control) - (base - control) == treat - base``. That is what
    makes this worth quoting over two overlapping confidence intervals -- the
    control's variance, which is most of the variance, is differenced away
    rather than added twice.

    The denominator is the control's spread, the same unit every other lift in
    this project is quoted in, so a number here is comparable to a number
    there.
    """
    spread = base["control_spread"] or 1.0
    return ((np.asarray(treat["differences"], dtype=float)
             - np.asarray(base["differences"], dtype=float)) / spread)


def interval(values):
    """Mean and a 95% interval, or ``None`` when there is nothing to say."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    error = float(values.std(ddof=1)) / np.sqrt(len(values))
    mean = float(values.mean())
    return mean, mean - 1.96 * error, mean + 1.96 * error


def main(argv=None):
    parser = argparse.ArgumentParser(prog="summarize-decks")
    parser.add_argument("rows", type=Path, nargs="+",
                        help="one or more files from evaluate_decks.py --out. "
                             "A sweep split across processes writes one each; "
                             "they are read back as the single experiment they "
                             "were drawn as.")
    parser.add_argument("--baseline", default="clone-cardstat-lookup",
                        help="the arm the other is differenced against")
    parser.add_argument("--treatment", default="clone-cardstat-encoder")
    args = parser.parse_args(argv)

    rows = []
    for path in args.rows:
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    by_deck = defaultdict(dict)
    for row in rows:
        key = (row["name"], row["mode"])
        held = by_deck[row["deck_label"]].get(key)
        # Two files claiming the same deck-arm-mode is a sweep that was sliced
        # wrongly, or the same file passed twice. Averaging them would hide it
        # and halve the interval on nothing.
        if held is not None and held["deck"] != row["deck"]:
            raise SystemExit(
                f"{row['deck_label']} appears twice with different cards: "
                f"{held['deck']} and {row['deck']}. These files are not one "
                "experiment.")
        by_deck[row["deck_label"]][key] = row

    labels = sorted(by_deck, key=lambda name: (name != "training-deck", name))
    names = sorted({row["name"] for row in rows})
    print(f"{len(rows)} rows, {len(labels)} decks, arms: {', '.join(names)}")
    print(f"opponent: {rows[0]['eval_opponent']}, "
          f"{rows[0]['episodes']} battles per deck per arm\n")

    for mode in ("greedy", "sampled"):
        print(f"=== {mode} ===")
        print(f"{'deck':<16}{'control':>9}"
              f"{'lookup lift':>26}{'encoder lift':>26}"
              f"{'encoder - lookup':>26}")
        deck_gap, deck_lookup, deck_encoder = [], [], []
        pooled_gap, pooled_lookup, pooled_encoder = [], [], []
        for label in labels:
            arms = by_deck[label]
            base = arms.get((args.baseline, mode))
            treat = arms.get((args.treatment, mode))
            if base is None or treat is None:
                continue
            spread = base["control_spread"] or 1.0
            gap = head_to_head(base, treat)
            told = interval(gap)
            row = (f"{label:<16}{base['control_win']:>8.0%} "
                   f"{base['lift']:>+9.3f} [{base['ci_low']:+.2f},{base['ci_high']:+.2f}]"
                   f"{treat['lift']:>+10.3f} [{treat['ci_low']:+.2f},{treat['ci_high']:+.2f}]")
            if told:
                row += f"{told[0]:>+10.3f} [{told[1]:+.2f},{told[2]:+.2f}]"
            print(row)
            if label == "training-deck":
                continue
            deck_gap.append(float(gap.mean()))
            deck_lookup.append(base["lift"])
            deck_encoder.append(treat["lift"])
            pooled_gap.extend(gap.tolist())
            pooled_lookup.extend(
                (np.asarray(base["differences"]) / spread).tolist())
            pooled_encoder.extend(
                (np.asarray(treat["differences"]) / spread).tolist())

        if not deck_gap:
            print()
            continue
        print(f"\n  unseen decks only, {len(deck_gap)} decks "
              f"x {rows[0]['episodes']} battles")
        for title, battle, deck in (
                ("lookup vs control ", pooled_lookup, deck_lookup),
                ("encoder vs control", pooled_encoder, deck_encoder),
                ("encoder vs lookup ", pooled_gap, deck_gap)):
            per_battle = interval(battle)
            per_deck = interval(deck)
            line = f"  {title}  per battle "
            line += (f"{per_battle[0]:>+7.3f} "
                     f"[{per_battle[1]:+.3f}, {per_battle[2]:+.3f}]"
                     if per_battle else f"{'--':>7}")
            # One deck is not a sample of decks. Saying so beats printing an
            # interval of width zero, and beats crashing on it.
            line += "   per deck " + (
                f"{per_deck[0]:>+7.3f} [{per_deck[1]:+.3f}, {per_deck[2]:+.3f}]"
                if per_deck else
                f"{np.mean(deck):>+7.3f} (one deck, no interval)")
            print(line)
        wins = sum(1 for value in deck_gap if value > 0)
        print(f"  encoder ahead on {wins} of {len(deck_gap)} unseen decks\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
