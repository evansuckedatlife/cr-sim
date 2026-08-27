"""How much work each relaxation pass of the collision sweep actually does."""
from __future__ import annotations
import os, sys, importlib.util
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("bench", os.path.join(HERE, "bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

import cr_sim.engine.movement as M
from cr_sim.engine.entity import EntityKind

real_separate = M.separate
_INCORPOREAL = (EntityKind.PROJECTILE, EntityKind.AREA_EFFECT)

stats = Counter()
movers = Counter()


def instrumented(index, arena=None, *, max_radius, passes=3):
    buckets = index._buckets
    cell = index.cell
    columns, rows = index.columns, index.rows
    moved = 0
    for p in range(passes):
        touched = 0
        moved_ids = set()
        stats[f"pass{p}_run"] += 1
        for source in buckets:
            if not source:
                continue
            for a in source:
                if a.dead or a.deploy_ticks_left > 0 or a.kind in _INCORPOREAL:
                    continue
                radius_a = a.collision_radius
                identity = a.id
                flying = a.flying
                reach = radius_a + max_radius
                lx = max(0, (a.x - reach) // cell)
                hx = min(columns - 1, (a.x + reach) // cell)
                ly = max(0, (a.y - reach) // cell)
                hy = min(rows - 1, (a.y + reach) // cell)
                for cy in range(ly, hy + 1):
                    base = cy * columns
                    for cx in range(lx, hx + 1):
                        for b in buckets[base + cx]:
                            if identity >= b.id:
                                continue
                            stats[f"pass{p}_cand"] += 1
                            if b.flying != flying:
                                continue
                            total = radius_a + b.collision_radius
                            dx = b.x - a.x
                            dy = b.y - a.y
                            if total <= 0 or dx * dx + dy * dy >= total * total:
                                continue
                            if b.dead or b.deploy_ticks_left > 0 or b.kind in _INCORPOREAL:
                                continue
                            stats[f"pass{p}_overlap"] += 1
                            if real_separate(a, b, arena):
                                touched += 1
                                moved_ids.add(a.id)
                                moved_ids.add(b.id)
        movers[f"pass{p}_movers"] += len(moved_ids)
        moved += touched
        if not touched:
            break
    return moved


M.resolve_collisions = instrumented
import cr_sim.engine.battle as B
B.resolve_collisions = instrumented

w = bench.world()
b = bench.busy_battle(w, upto=1200)
print("ticks 1200")
for k in sorted(stats):
    print(f"  {k:<18} {stats[k]}")
for k in sorted(movers):
    print(f"  {k:<18} {movers[k]}")
