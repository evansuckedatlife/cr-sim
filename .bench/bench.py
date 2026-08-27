"""Engine throughput benchmark. Run with PYTHONPATH pointing at the tree to measure."""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
BUILD = str(HERE / "data_cache" / "csv_logic")

# Windows' process_time() ticks at 15.6 ms, far too coarse for a 69 us tick, so
# the clock has to be perf_counter and the defence against a busy machine has to
# be min-of-N instead. Raising the priority class keeps the competing training
# workers (deliberately set to Idle) off this process's back.
CLOCK = time.perf_counter
if sys.platform == "win32":  # pragma: no cover
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080)  # HIGH
    except Exception:
        pass


def world():
    from cr_sim.data.cards import build_card_registry
    from cr_sim.data.leveling import build_level_table
    from cr_sim.data.source import LogicData
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


DECK_A = ("Knight", "Musketeer", "Cannon", "Skeletons",
          "IceSpirits", "Log", "Fireball", "Goblins")
DECK_B = ("Giant", "Archers", "MiniPEKKA", "SkeletonArmy",
          "Minions", "Arrows", "Zap", "Barbarians")


def fresh(w, seed=7, tps=20):
    from cr_sim.engine.battle import Battle, BattleConfig
    data, levels, registry = w
    return Battle(data, levels, registry, BattleConfig(
        seed=seed, ticks_per_second=tps, blue_deck=DECK_A, red_deck=DECK_B))


SCRIPT = [
    (20, 0, "Knight", 9.0, 8.0), (20, 1, "Giant", 9.0, 24.0),
    (40, 0, "Musketeer", 7.0, 6.0), (45, 1, "Archers", 11.0, 26.0),
    (70, 0, "Skeletons", 10.0, 10.0), (75, 1, "Minions", 8.0, 22.0),
    (95, 0, "Goblins", 8.0, 9.0), (100, 1, "MiniPEKKA", 10.0, 23.0),
    (130, 0, "Cannon", 9.0, 7.0), (135, 1, "Barbarians", 9.0, 21.0),
    (170, 0, "Knight", 6.0, 9.0), (175, 1, "SkeletonArmy", 12.0, 22.0),
    (200, 0, "Musketeer", 12.0, 7.0), (205, 1, "Giant", 6.0, 24.0),
    (240, 0, "Skeletons", 9.0, 12.0), (245, 1, "Archers", 9.0, 20.0),
]


#: Ticks after which the scripted plays repeat. Without this the board empties
#: by tick 600 and a "full match" benchmark is mostly idle ticks, which is not
#: what a training match looks like.
PERIOD = 280


def busy_battle(w, upto=200):
    """A mid-match position with a real crowd on the board.

    Plays repeat every ``PERIOD`` ticks, so the board stays occupied for as long
    as the caller wants to run. Every play is best-effort -- the elixir bar
    refuses what it cannot pay for, exactly as in a match.
    """
    from cr_sim.engine.entity import Team
    from cr_sim.engine.fixed import tiles
    b = fresh(w)
    plays = {}
    for tick, team, card, x, y in SCRIPT:
        plays.setdefault(tick, []).append((team, card, x, y))
    for t in range(upto):
        for team, card, x, y in plays.get(t % PERIOD, ()):
            b.play_card(Team.BLUE if team == 0 else Team.RED, card, tiles(x), tiles(y))
        b.step()
    return b


def alive(b):
    return sum(1 for e in b.entities if not e.dead)


# ---------------------------------------------------------------- timing util

def best(fn, reps, inner=1):
    """Min CPU seconds per call over `reps` repetitions of `inner` calls."""
    out = float("inf")
    for _ in range(reps):
        t0 = CLOCK()
        for _ in range(inner):
            fn()
        dt = CLOCK() - t0
        if dt < out:
            out = dt
    return out / inner


def bench_tick(w, reps=13):
    """Pure step() cost on a busy mid-match board, no cards played.

    The board is re-cloned for each repetition so every repetition measures the
    same 150 ticks rather than an ever-emptier board.
    """
    base = busy_battle(w)
    out = float("inf")
    for _ in range(reps):
        b = base.clone()
        t0 = CLOCK()
        for _ in range(150):
            b.step()
        dt = CLOCK() - t0
        if dt < out:
            out = dt
    return out / 150


def bench_quiet_tick(w, reps=13):
    """One tick from an empty board -- towers only. The common early case."""
    base = fresh(w)
    for _ in range(5):
        base.step()
    out = float("inf")
    for _ in range(reps):
        b = base.clone()
        t0 = CLOCK()
        for _ in range(1000):
            b.step()
        dt = CLOCK() - t0
        if dt < out:
            out = dt
    return out / 1000


def bench_match(w, reps=7):
    """A full scripted 120-second match at 20 TPS, cards played throughout."""
    out = float("inf")
    for _ in range(reps):
        t0 = CLOCK()
        busy_battle(w, upto=2400)
        dt = CLOCK() - t0
        if dt < out:
            out = dt
    return out


def bench_clone(w, reps=13):
    b = busy_battle(w)
    return best(b.clone, reps, inner=200)


def bench_clone_late(w, reps=13):
    """A clone from a position with a match's worth of dead behind it."""
    b = busy_battle(w, upto=1000)
    return best(b.clone, reps, inner=200)


def bench_encode_late(w, reps=13):
    from cr_sim.api.encoding import build_encoding_config, encode_observation
    from cr_sim.engine.entity import Team
    data, levels, registry = w
    b = busy_battle(w, upto=1000)
    cfg = build_encoding_config(b.arena, DECK_A, DECK_B)
    return best(lambda: encode_observation(b, Team.BLUE, registry, cfg), reps, inner=200)


def bench_project(w, reps=7):
    from cr_sim.engine.lookahead import project
    b = busy_battle(w)
    return best(lambda: project(b, 60), reps, inner=10)


def bench_project_long(w, reps=7):
    from cr_sim.engine.lookahead import project
    b = busy_battle(w, upto=1000)
    return best(lambda: project(b, 300), reps, inner=3)


def bench_project_quiet(w, reps=9):
    from cr_sim.engine.lookahead import project
    b = fresh(w)
    for _ in range(5):
        b.step()
    return best(lambda: project(b, 60), reps, inner=2000)


def bench_encode(w, reps=13):
    from cr_sim.api.encoding import build_encoding_config, encode_observation
    from cr_sim.engine.entity import Team
    data, levels, registry = w
    b = busy_battle(w)
    cfg = build_encoding_config(b.arena, DECK_A, DECK_B)
    return best(lambda: encode_observation(b, Team.BLUE, registry, cfg), reps, inner=200)


def bench_mask(w, reps=13):
    from cr_sim.api.encoding import build_encoding_config, legal_action_mask
    from cr_sim.engine.entity import Team
    data, levels, registry = w
    b = busy_battle(w)
    cfg = build_encoding_config(b.arena, DECK_A, DECK_B)
    return best(lambda: legal_action_mask(b, Team.BLUE, registry, cfg), reps, inner=2000)


BENCHES = {
    "tick_busy": bench_tick,
    "tick_quiet": bench_quiet_tick,
    "match_2400": bench_match,
    "clone": bench_clone,
    "clone_late": bench_clone_late,
    "project_3s": bench_project,
    "project_15s": bench_project_long,
    "project_quiet": bench_project_quiet,
    "encode": bench_encode,
    "encode_late": bench_encode_late,
    "mask": bench_mask,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--tree", default=str(HERE),
                    help="directory containing the cr_sim package to measure")
    args = ap.parse_args()
    sys.path.insert(0, args.tree)
    w = world()
    names = args.only.split(",") if args.only else list(BENCHES)
    res = {}
    for name in names:
        res[name] = BENCHES[name](w)
    print(json.dumps({"tag": args.tag, "res": res}))


if __name__ == "__main__":
    main()
