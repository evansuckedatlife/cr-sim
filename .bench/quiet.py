"""How far into a projection does the board actually stay busy?

If a 300-tick projection goes quiet at tick 90, the remaining 210 ticks of
simulation cannot change anything it reads -- so the question is how often that
happens and how early.
"""
from __future__ import annotations
import os, sys, importlib.util
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("bench", os.path.join(HERE, "bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

from cr_sim.engine.lookahead import _is_quiet
from cr_sim.engine.entity import Team, entity_id_cursor, restore_entity_ids
from cr_sim.engine.fixed import tiles

w = bench.world()


def trial(horizon, samples_every=40, upto=1600):
    """Simulate a match; at intervals, branch and see when the branch goes quiet."""
    from cr_sim.engine.battle import Battle, BattleConfig
    data, levels, registry = w
    b = Battle(data, levels, registry, BattleConfig(
        seed=7, ticks_per_second=20, blue_deck=bench.DECK_A, red_deck=bench.DECK_B))
    plays = {}
    for tick, team, card, x, y in bench.SCRIPT:
        plays.setdefault(tick, []).append((team, card, x, y))

    rows = []
    for t in range(upto):
        for team, card, x, y in plays.get(t % bench.PERIOD, ()):
            b.play_card(Team.BLUE if team == 0 else Team.RED, card, tiles(x), tiles(y))
        b.step()
        if t % samples_every or t < 60:
            continue
        if _is_quiet(b):
            rows.append((t, 0, len(b.entities)))  # the free path already
            continue
        cursor = entity_id_cursor()
        branch = b.clone()
        n = 0
        try:
            while n < horizon and not branch.finished:
                branch.step()
                n += 1
                if _is_quiet(branch):
                    break
        finally:
            restore_entity_ids(cursor)
        rows.append((t, n, len([e for e in b.entities if not e.dead])))
    return rows


for horizon in (60, 300):
    rows = trial(horizon)
    busy = [r for r in rows if r[1] > 0]
    early = [r for r in busy if r[1] < horizon]
    print(f"horizon {horizon} ticks: {len(rows)} samples, "
          f"{len(rows) - len(busy)} already quiet at the root, "
          f"{len(early)}/{len(busy)} of the rest went quiet before the horizon")
    if busy:
        total = sum(r[1] for r in busy)
        print(f"  ticks actually simulated: {total} of {len(busy) * horizon} "
              f"({total / (len(busy) * horizon) * 100:.0f}%)")
    print("  sample (root tick, ticks to quiet, entities):",
          [(r[0], r[1], r[2]) for r in rows[:12]])
