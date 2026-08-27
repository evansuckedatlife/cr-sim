"""Profiling: cProfile hot list plus a non-profiler phase attribution."""
from __future__ import annotations

import cProfile, io, os, pstats, sys, time
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
_TREE = None
for _i, _a in enumerate(sys.argv):
    if _a == "--tree":
        _TREE = sys.argv[_i + 1]
        del sys.argv[_i:_i + 2]
        break
sys.path.insert(0, _TREE or os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("bench", os.path.join(HERE, "bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

w = bench.world()
MODE = sys.argv[1] if len(sys.argv) > 1 else "phases"


def phases():
    """Per-phase wall time, measured with perf_counter rather than cProfile.

    cProfile's per-call overhead is larger than several of these phases, so its
    attribution between them is not trustworthy; this is.
    """
    base = bench.busy_battle(w)
    b = base.clone()
    names = list(b.PHASES)
    fns = list(b._phase_fns)
    totals = [0.0] * len(fns)
    clock = time.perf_counter
    N = 400
    t_all = clock()
    for _ in range(N):
        b._index.rebuild(b.entities)
        from cr_sim.engine.entity import Team
        b._pre_tick_crowns = (b.players[Team.BLUE].crowns, b.players[Team.RED].crowns)
        for i, fn in enumerate(fns):
            t0 = clock()
            fn()
            totals[i] += clock() - t0
        b.tick += 1
    total = clock() - t_all
    print(f"{N} ticks, {total*1e3:.1f} ms total, {total/N*1e6:.1f} us/tick "
          f"(instrumented; adds ~{17*0.1:.1f} us/tick)")
    order = sorted(range(len(fns)), key=lambda i: -totals[i])
    for i in order:
        if totals[i] * 1e6 / N < 0.5:
            continue
        print(f"  {names[i]:<28} {totals[i]/N*1e6:8.1f} us/tick  {totals[i]/total*100:5.1f}%")


def cprof(what="tick"):
    if what == "tick":
        base = bench.busy_battle(w)
        b = base.clone()
        fn = lambda: [b.step() for _ in range(400)]
    elif what == "match":
        fn = lambda: bench.busy_battle(w, upto=2400)
    elif what == "encode":
        from cr_sim.api.encoding import build_encoding_config, encode_observation
        from cr_sim.engine.entity import Team
        b = bench.busy_battle(w)
        cfg = build_encoding_config(b.arena, bench.DECK_A, bench.DECK_B)
        registry = w[2]
        fn = lambda: [encode_observation(b, Team.BLUE, registry, cfg) for _ in range(2000)]
    elif what == "clone":
        b = bench.busy_battle(w)
        fn = lambda: [b.clone() for _ in range(2000)]
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(28)
    print(s.getvalue())


if MODE == "phases":
    phases()
else:
    cprof(MODE)
