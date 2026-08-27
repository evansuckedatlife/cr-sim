"""Single-process interleaved A/B.

The baseline tree is imported under a second top-level package name -- every
import inside cr_sim is relative, so the package works fine under any name --
which puts both versions in one process. Each benchmark then runs
A/B/A/B/... back to back with the minimum kept per side, so a load spike hits
both sides of the comparison rather than only whichever one happened to be
running when it arrived. Process-level interleaving cannot promise that: two
subprocesses never overlap in time.
"""
from __future__ import annotations

import argparse, importlib, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = str(ROOT / "data_cache" / "csv_logic")

CLOCK = time.perf_counter
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080)
    except Exception:
        pass

DECK_A = ("Knight", "Musketeer", "Cannon", "Skeletons",
          "IceSpirits", "Log", "Fireball", "Goblins")
DECK_B = ("Giant", "Archers", "MiniPEKKA", "SkeletonArmy",
          "Minions", "Arrows", "Zap", "Barbarians")
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
PERIOD = 280
BUSY_TICK, LATE_TICK = 200, 1000


class Side:
    """One version of the engine, with its own module namespace."""

    def __init__(self, package: str) -> None:
        self.pkg = package
        self.battle_mod = importlib.import_module(f"{package}.engine.battle")
        self.entity = importlib.import_module(f"{package}.engine.entity")
        self.fixed = importlib.import_module(f"{package}.engine.fixed")
        self.lookahead = importlib.import_module(f"{package}.engine.lookahead")
        self.encoding = importlib.import_module(f"{package}.api.encoding")
        cards = importlib.import_module(f"{package}.data.cards")
        leveling = importlib.import_module(f"{package}.data.leveling")
        source = importlib.import_module(f"{package}.data.source")
        data = source.LogicData.load(BUILD)
        self.world = (data, leveling.build_level_table(data),
                      cards.build_card_registry(data))

    def fresh(self, seed=7, tps=20):
        data, levels, registry = self.world
        Battle = self.battle_mod.Battle
        Config = self.battle_mod.BattleConfig
        return Battle(data, levels, registry, Config(
            seed=seed, ticks_per_second=tps, blue_deck=DECK_A, red_deck=DECK_B))

    def scripted(self, upto=BUSY_TICK):
        Team = self.entity.Team
        tiles = self.fixed.tiles
        b = self.fresh()
        plays = {}
        for tick, team, card, x, y in SCRIPT:
            plays.setdefault(tick, []).append((team, card, x, y))
        for t in range(upto):
            for team, card, x, y in plays.get(t % PERIOD, ()):
                b.play_card(Team.BLUE if team == 0 else Team.RED,
                            card, tiles(x), tiles(y))
            b.step()
        return b


def best(fn, reps, inner):
    lowest = float("inf")
    for _ in range(reps):
        t0 = CLOCK()
        for _ in range(inner):
            fn()
        dt = CLOCK() - t0
        if dt < lowest:
            lowest = dt
    return lowest / inner


def steps(base, count):
    b = base.clone()
    t0 = CLOCK()
    for _ in range(count):
        b.step()
    return (CLOCK() - t0) / count


#: Per benchmark: how many rounds to interleave. Short blocks and many rounds,
#: because the defence against a loaded machine is a paired comparison repeated
#: often, not one long timed stretch that is certain to be interrupted.
ROUNDS = {
    "tick_busy": 200, "tick_quiet": 200, "match_2400": 7,
    "clone": 500, "clone_late": 500,
    "project_3s": 40, "project_15s": 20, "project_quiet": 500,
    "encode": 500, "encode_late": 500, "mask": 500,
}


# Each entry: name -> (setup(side) -> callable, reps, inner). The callable is
# what gets timed; setup work (building the fixture) happens once per side.
def make(name, side):
    if name == "tick_busy":
        base = side.scripted()
        return (lambda: steps(base, 10)), 1, True
    if name == "tick_quiet":
        base = side.fresh()
        for _ in range(5):
            base.step()
        return (lambda: steps(base, 100)), 1, True
    if name == "match_2400":
        return (lambda: side.scripted(2400)), 1, False
    if name == "clone":
        b = side.scripted()
        return b.clone, 2, False
    if name == "clone_late":
        b = side.scripted(LATE_TICK)
        return b.clone, 2, False
    if name == "project_3s":
        b = side.scripted()
        return (lambda: side.lookahead.project(b, 60)), 1, False
    if name == "project_15s":
        b = side.scripted(LATE_TICK)
        return (lambda: side.lookahead.project(b, 300)), 1, False
    if name == "project_quiet":
        b = side.fresh()
        for _ in range(5):
            b.step()
        return (lambda: side.lookahead.project(b, 60)), 50, False
    if name in ("encode", "encode_late"):
        b = side.scripted(BUSY_TICK if name == "encode" else LATE_TICK)
        cfg = side.encoding.build_encoding_config(b.arena, DECK_A, DECK_B)
        registry = side.world[2]
        Team = side.entity.Team
        enc = side.encoding.encode_observation
        return (lambda: enc(b, Team.BLUE, registry, cfg)), 12, False
    if name == "mask":
        b = side.scripted()
        cfg = side.encoding.build_encoding_config(b.arena, DECK_A, DECK_B)
        registry = side.world[2]
        Team = side.entity.Team
        mask = side.encoding.legal_action_mask
        return (lambda: mask(b, Team.BLUE, registry, cfg)), 300, False
    raise KeyError(name)


NAMES = ["tick_busy", "tick_quiet", "match_2400", "clone", "clone_late",
         "project_3s", "project_15s", "project_quiet", "encode",
         "encode_late", "mask"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-path", required=True)
    ap.add_argument("--baseline-pkg", default="cr_sim_base")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    # Baseline path last: it also holds a `cr_sim`, and if that shadowed the
    # worktree's both sides would silently measure the same code.
    sys.path.insert(0, args.baseline_path)
    sys.path.insert(0, str(ROOT))
    new = Side("cr_sim")
    old = Side(args.baseline_pkg)
    print("after :", new.battle_mod.__file__)
    print("before:", old.battle_mod.__file__)
    assert str(ROOT) in new.battle_mod.__file__, "the 'after' side is not the worktree"
    assert args.baseline_path.replace("/", os.sep) in old.battle_mod.__file__.replace("/", os.sep)

    b = new.scripted()
    late = new.scripted(LATE_TICK)
    print(f"busy fixture  (tick {BUSY_TICK}): "
          f"{sum(1 for e in b.entities if not e.dead)} entities alive, "
          f"{len(b.graveyard)} dead")
    print(f"late fixture  (tick {LATE_TICK}): "
          f"{sum(1 for e in late.entities if not e.dead)} entities alive, "
          f"{len(late.graveyard)} dead")
    print()

    names = args.only.split(",") if args.only else NAMES
    rows = []
    for name in names:
        new_fn, new_inner, new_self = make(name, new)
        old_fn, old_inner, old_self = make(name, old)
        low_new = low_old = float("inf")
        ratios = []
        rounds = ROUNDS.get(name, args.rounds)
        for round_index in range(rounds):
            paired = {}
            # Alternate which side goes first, so neither gets a systematic
            # advantage from cache state left by the other.
            order = (("old", old_fn, old_inner, old_self),
                     ("new", new_fn, new_inner, new_self))
            if round_index % 2:
                order = order[::-1]
            for label, fn, inner, self_timed in order:
                # A self-timed benchmark returns its own per-unit figure, so
                # that setup it must do afresh each round (re-cloning a board
                # before stepping it) stays outside what is measured.
                t = fn() if self_timed else best(fn, 1, inner)
                paired[label] = t
                if label == "old":
                    low_old = min(low_old, t)
                else:
                    low_new = min(low_new, t)
            # The paired ratio is the robust statistic here: both sides were
            # measured seconds apart under the same competing load, so a spike
            # that hits one round inflates numerator and denominator together
            # and leaves the ratio alone. The median across rounds then throws
            # away the rounds where it did not.
            ratios.append(paired["old"] / paired["new"])
        ratios.sort()
        median = ratios[len(ratios) // 2]
        rows.append((name, low_old, low_new, median, len(ratios)))
        print(f"  {name} done", file=sys.stderr)

    width = max(len(n) for n, *_ in rows)
    print(f"{'benchmark'.ljust(width)}   {'before':>12}   {'after':>12}   "
          f"{'min-ratio':>9}   {'median':>7}   rounds")
    for name, a, bb, median, n in rows:
        unit, scale = ("ms", 1e3) if a >= 1e-3 else ("us", 1e6)
        print(f"{name.ljust(width)}   {a * scale:9.2f} {unit}   "
              f"{bb * scale:9.2f} {unit}   {a / bb:8.2f}x   {median:6.2f}x   {n}")


if __name__ == "__main__":
    main()
