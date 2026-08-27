"""Single-process A/B of collision broad-phase variants on a real board."""
from __future__ import annotations
import os, sys, time, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("bench", os.path.join(HERE, "bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

from cr_sim.engine.entity import EntityKind

w = bench.world()
b = bench.busy_battle(w)
b._index.rebuild(b.entities)
idx = b._index
R = b._max_radius
print("entities", len(b.entities), "max_radius", R, "cell", idx.cell,
      "occupancy", idx.occupancy())

_INCORPOREAL = (EntityKind.PROJECTILE, EntityKind.AREA_EFFECT)


def _walk_flat():
    """One generator frame; the bucket walk inlined."""
    buckets = idx._buckets
    cell = idx.cell
    columns, rows = idx.columns, idx.rows
    for source in buckets:
        if not source:
            continue
        for entity in source:
            reach = entity.collision_radius + R
            eid = entity.id
            lx = (entity.x - reach) // cell
            hx = (entity.x + reach) // cell
            ly = (entity.y - reach) // cell
            hy = (entity.y + reach) // cell
            if lx < 0:
                lx = 0
            if ly < 0:
                ly = 0
            if hx >= columns:
                hx = columns - 1
            if hy >= rows:
                hy = rows - 1
            for cy in range(ly, hy + 1):
                base = cy * columns
                for cx in range(lx, hx + 1):
                    for other in buckets[base + cx]:
                        if eid < other.id:
                            yield entity, other


def _consume(walk):
    n = 0
    for a, bb in walk:
        n += 1
    return n


def v_current_walk():
    return _consume(idx.pairs(R))


def v_flat_walk():
    return _consume(_walk_flat())


def _filtered(walk):
    """Exactly the filter chain resolve_collisions runs today."""
    n = 0
    for a, bb in walk:
        if a.dead or bb.dead:
            continue
        if a.kind in _INCORPOREAL or bb.kind in _INCORPOREAL:
            continue
        if a.flying != bb.flying:
            continue
        if a.deploy_ticks_left > 0 or bb.deploy_ticks_left > 0:
            continue
        reach = a.collision_radius + bb.collision_radius
        if reach <= 0:
            continue
        dx = bb.x - a.x
        dy = bb.y - a.y
        if dx * dx + dy * dy >= reach * reach:
            continue
        n += 1
    return n


def v_current_filtered():
    return _filtered(idx.pairs(R))


def v_flatgen_filtered():
    return _filtered(_walk_flat())


def v_fused():
    """Flat walk with the per-a filters hoisted and the pair test inlined."""
    buckets = idx._buckets
    cell = idx.cell
    columns, rows = idx.columns, idx.rows
    n = 0
    for source in buckets:
        if not source:
            continue
        for entity in source:
            if entity.dead or entity.kind >= 3 or entity.deploy_ticks_left > 0:
                continue
            ra = entity.collision_radius
            reach = ra + R
            eid = entity.id
            ax, ay = entity.x, entity.y
            flying = entity.flying
            lx = (ax - reach) // cell
            hx = (ax + reach) // cell
            ly = (ay - reach) // cell
            hy = (ay + reach) // cell
            if lx < 0:
                lx = 0
            if ly < 0:
                ly = 0
            if hx >= columns:
                hx = columns - 1
            if hy >= rows:
                hy = rows - 1
            for cy in range(ly, hy + 1):
                base = cy * columns
                for cx in range(lx, hx + 1):
                    for other in buckets[base + cx]:
                        if eid >= other.id:
                            continue
                        if other.flying != flying:
                            continue
                        rr = ra + other.collision_radius
                        dx = other.x - ax
                        dy = other.y - ay
                        if rr <= 0 or dx * dx + dy * dy >= rr * rr:
                            continue
                        if other.dead or other.kind >= 3 or other.deploy_ticks_left > 0:
                            continue
                        n += 1
    return n


def cells_near(x, y, radius):
    """Candidate bucket lists for a query circle -- the shared geometry."""
    cell = idx.cell
    columns, rows = idx.columns, idx.rows
    lx = (x - radius) // cell
    hx = (x + radius) // cell
    ly = (y - radius) // cell
    hy = (y + radius) // cell
    if lx < 0:
        lx = 0
    if ly < 0:
        ly = 0
    if hx >= columns:
        hx = columns - 1
    if hy >= rows:
        hy = rows - 1
    buckets = idx._buckets
    return [buckets[cy * columns + cx]
            for cy in range(ly, hy + 1) for cx in range(lx, hx + 1)]


def v_fused_via_cells():
    n = 0
    for source in idx._buckets:
        if not source:
            continue
        for entity in source:
            if entity.dead or entity.kind >= 3 or entity.deploy_ticks_left > 0:
                continue
            ra = entity.collision_radius
            eid = entity.id
            ax, ay = entity.x, entity.y
            flying = entity.flying
            for bucket in cells_near(ax, ay, ra + R):
                for other in bucket:
                    if eid >= other.id:
                        continue
                    if other.flying != flying:
                        continue
                    rr = ra + other.collision_radius
                    dx = other.x - ax
                    dy = other.y - ay
                    if rr <= 0 or dx * dx + dy * dy >= rr * rr:
                        continue
                    if other.dead or other.kind >= 3 or other.deploy_ticks_left > 0:
                        continue
                    n += 1
    return n


def timeit(fn, reps=3, inner=100):
    out = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        dt = time.perf_counter() - t0
        if dt < out:
            out = dt
    return out / inner


print("counts:", v_current_walk(), v_flat_walk(), v_current_filtered(),
      v_flatgen_filtered(), v_fused(), v_fused_via_cells())

names = ["current_walk", "flat_walk", "current_filtered", "flatgen_filtered", "fused", "fused_via_cells"]
fns = {
    "current_walk": v_current_walk,
    "flat_walk": v_flat_walk,
    "current_filtered": v_current_filtered,
    "flatgen_filtered": v_flatgen_filtered,
    "fused": v_fused,
    "fused_via_cells": v_fused_via_cells,
}
best = {n: float("inf") for n in names}
for _ in range(9):
    for n in names:
        t = timeit(fns[n])
        if t < best[n]:
            best[n] = t
for n in names:
    print(f"{n:<18} {best[n]*1e6:8.1f} us   {best['current_filtered']/best[n]:6.2f}x")
