"""What Battle.clone's deepcopy actually walks, counted by type."""
from __future__ import annotations
import copy, os, sys, importlib.util
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("bench", os.path.join(HERE, "bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

w = bench.world()
b = bench.busy_battle(w)
print("entities", len(b.entities))

counts = Counter()
real = copy.deepcopy
depth = [0]


def counting(x, memo=None, _nil=[]):
    counts[type(x).__name__] += 1
    return real(x, memo) if memo is not None else real(x)


copy.deepcopy = counting
import cr_sim.engine.battle as B
B._deepcopy = counting
b.clone()
copy.deepcopy = real
B._deepcopy = real

total = sum(counts.values())
print("deepcopy calls per clone:", total)
for name, n in counts.most_common(24):
    print(f"  {name:<28} {n}")

print()
print("container sizes:")
for name in ("_routes", "_attacks", "_by_id_map", "_charge", "_last_attack",
             "_hit_counts", "_spawn_timers", "_spawn_children", "entities",
             "_occupancy_signature", "_phase_fns"):
    v = getattr(b, name)
    print(f"  {name:<24} {len(v)}")
print("  rng streams", len(getattr(b.rng, "__dict__", {}) or {}))
