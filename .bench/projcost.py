"""Where a projection's time actually goes: the clone, the ticks, the read."""
from __future__ import annotations
import os, sys, time, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("bench", os.path.join(HERE, "bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

from cr_sim.engine.lookahead import _read, _is_quiet, project
from cr_sim.engine.entity import entity_id_cursor, restore_entity_ids

CLOCK = time.perf_counter
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080)
    except Exception:
        pass


def best(fn, reps=9, inner=1):
    out = float("inf")
    for _ in range(reps):
        t0 = CLOCK()
        for _ in range(inner):
            fn()
        dt = CLOCK() - t0
        if dt < out:
            out = dt
    return out / inner


w = bench.world()
for upto in (200, 1000):
    b = bench.busy_battle(w, upto)
    live = sum(1 for e in b.entities if not e.dead)
    print(f"root tick {upto}: {live} alive, {len(b.graveyard)} dead, "
          f"{len(b.damage_log)} damage events, frames={len(b.frames)}")
    print(f"  clone            {best(b.clone, 9, 100) * 1e6:9.1f} us")
    print(f"  _read            {best(lambda: _read(b, 0, False), 9, 200) * 1e6:9.1f} us")
    print(f"  _is_quiet        {best(lambda: _is_quiet(b), 9, 2000) * 1e6:9.1f} us")

    def run(n):
        def go():
            cursor = entity_id_cursor()
            branch = b.clone()
            try:
                for _ in range(n):
                    if branch.finished:
                        break
                    branch.step()
            finally:
                restore_entity_ids(cursor)
        return go

    for n, label in ((60, "3s"), (300, "15s")):
        total = best(lambda: project(b, n), 5, 3)
        ticks = best(run(n), 5, 3)
        print(f"  project({label:>3})     {total * 1e3:9.3f} ms   "
              f"(clone+ticks {ticks * 1e3:.3f} ms)")
