"""Engine throughput benchmark.

Answers, reproducibly, the questions that decide how fast training can run:
what a tick costs, what a whole match costs, what branching a position costs,
what a three- and a fifteen-second lookahead cost, and what turning a position
into an observation costs.

    python scripts/bench_engine.py                       # one tree, a table
    python scripts/bench_engine.py --only tick_busy,clone
    python scripts/bench_engine.py --json                # machine-readable
    python scripts/bench_engine.py --against ../cr-sim-main   # A/B two trees

**Board size is part of the measurement.** A tick on an empty board and a tick
in the middle of a fight differ by more than an order of magnitude, so a figure
quoted without an entity count means nothing. Every fixture here reports how
many entities were alive and how many were in the graveyard, and both matter:
the living drive the tick, and the dead drive what a clone has to copy.

**Measurement, on a machine that is doing other things.** ``--against`` loads
the other tree as a second top-level package in *this* process and interleaves
the two, benchmark by benchmark, round by round -- so a load spike lands on
both sides rather than on whichever subprocess happened to be running. Two
statistics come out of that and they fail in opposite directions, which is why
both are printed:

*min-ratio*
    The minimum per side over all rounds. Interference only ever adds time, so
    the minimum is the round that was least disturbed and the best estimate of
    the real cost -- provided some round got through cleanly.
*median*
    The median of the per-round ratios. Robust when no round is clean, but
    biased *towards 1.00* under heavy load, because interference adds roughly
    the same absolute time to numerator and denominator.

Under load the two bracket the truth. When they disagree widely, the machine
was too busy and the run is worth repeating. ``mask`` is deliberately included
as a control: nothing in this optimisation pass touched it, so it should read
1.00x, and how far it strays is the run's noise floor.

``time.process_time`` would be the more honest clock for CPU-bound work, but on
Windows it advances in 15.6 ms steps, which cannot resolve a tick at all.
``perf_counter`` with short timed blocks and many rounds is the workable
combination; the process also asks for a high priority class so that background
work at normal or idle priority stays out of the way.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.abc
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD = ROOT / "data_cache" / "csv_logic"

CLOCK = time.perf_counter
if sys.platform == "win32":  # pragma: no cover - platform specific
    try:
        import ctypes

        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080  # HIGH
        )
    except Exception:
        pass


#: Two decks chosen to exercise the parts of a tick that cost anything: a tank
#: to hold a push together, swarms to fill the collision sweep, air units so the
#: flying/ground split in collision is actually taken, spells so projectiles and
#: area effects run.
DECK_A = ("Knight", "Musketeer", "Cannon", "Skeletons",
          "IceSpirits", "Log", "Fireball", "Goblins")
DECK_B = ("Giant", "Archers", "MiniPEKKA", "SkeletonArmy",
          "Minions", "Arrows", "Zap", "Barbarians")

#: ``(tick, team, card, x, y)``, repeating every ``PERIOD`` ticks. Scripted so
#: one run is comparable with any other, and repeating because a one-shot script
#: leaves the board empty by tick 600 -- a "full match" benchmark over an empty
#: board measures nothing a training match does.
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

#: A crowded board with no history behind it: a fight at its peak.
BUSY_TICK = 200
#: A thinner board with a match's worth of dead behind it, which is what most
#: training decisions actually look like -- and the only fixture where the cost
#: of copying a graveyard shows up at all.
LATE_TICK = 1000


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Import another checkout's ``cr_sim`` under a second top-level name.

    Every import inside the package is relative, so it does not care what it is
    called -- which is what makes two versions loadable side by side in one
    process, and that in turn is what makes an interleaved A/B possible at all.
    """

    def __init__(self, alias: str, tree: Path) -> None:
        self.alias = alias
        self.package = Path(tree) / "cr_sim"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.alias and not fullname.startswith(self.alias + "."):
            return None
        rest = fullname[len(self.alias):].lstrip(".")
        base = self.package
        if rest:
            base = base.joinpath(*rest.split("."))
        if base.is_dir():
            return importlib.util.spec_from_file_location(
                fullname, base / "__init__.py",
                submodule_search_locations=[str(base)],
            )
        return importlib.util.spec_from_file_location(fullname, str(base) + ".py")


class Side:
    """One version of the engine, with its own module namespace and fixtures."""

    def __init__(self, package: str, build: str) -> None:
        self.package = package
        self.battle = importlib.import_module(f"{package}.engine.battle")
        self.entity = importlib.import_module(f"{package}.engine.entity")
        self.fixed = importlib.import_module(f"{package}.engine.fixed")
        self.lookahead = importlib.import_module(f"{package}.engine.lookahead")
        self.encoding = importlib.import_module(f"{package}.api.encoding")
        cards = importlib.import_module(f"{package}.data.cards")
        leveling = importlib.import_module(f"{package}.data.leveling")
        source = importlib.import_module(f"{package}.data.source")
        data = source.LogicData.load(build)
        self.world = (data, leveling.build_level_table(data),
                      cards.build_card_registry(data))

    def fresh(self, seed: int = 7, tps: int = 20):
        data, levels, registry = self.world
        return self.battle.Battle(data, levels, registry, self.battle.BattleConfig(
            seed=seed, ticks_per_second=tps, blue_deck=DECK_A, red_deck=DECK_B))

    def scripted(self, upto: int = BUSY_TICK):
        """Run the scripted match to ``upto`` and hand back the position."""
        Team = self.entity.Team
        tiles = self.fixed.tiles
        battle = self.fresh()
        plays: dict[int, list] = {}
        for tick, team, card, x, y in SCRIPT:
            plays.setdefault(tick, []).append((team, card, x, y))
        for t in range(upto):
            for team, card, x, y in plays.get(t % PERIOD, ()):
                # Best-effort: the elixir bar refuses what it cannot pay for,
                # exactly as in a real match.
                battle.play_card(Team.BLUE if team == 0 else Team.RED,
                                 card, tiles(x), tiles(y))
            battle.step()
        return battle


def describe(battle) -> str:
    living = sum(1 for e in battle.entities if not e.dead)
    return f"{living} alive, {len(battle.graveyard)} in the graveyard"


# ------------------------------------------------------------------- timing

def _block(fn, inner: int) -> float:
    start = CLOCK()
    for _ in range(inner):
        fn()
    return (CLOCK() - start) / inner


def _steps(base, count: int) -> float:
    """Seconds per ``step()`` over ``count`` ticks from a fresh copy of ``base``.

    Re-cloned every time so every round measures the same ticks from the same
    board rather than an ever-emptier one, and the clone stays outside the timed
    region.
    """
    battle = base.clone()
    start = CLOCK()
    for _ in range(count):
        battle.step()
    return (CLOCK() - start) / count


#: ``name -> (rounds, description)``. Rounds are set so each timed block stays
#: short: a long block on a busy machine is certain to be interrupted, and
#: min-of-N only works if some round gets through cleanly.
BENCHES = {
    "tick_busy": (200, "one tick, crowded board, no cards played"),
    "tick_quiet": (200, "one tick, towers only"),
    "match_2400": (7, "a full 120s match at 20 TPS, cards played throughout"),
    "clone": (500, "Battle.clone from the crowded board"),
    "clone_late": (500, "Battle.clone from the late board"),
    "project_3s": (40, "3s lookahead -- what the shaped reward pays per step"),
    "project_15s": (20, "15s lookahead -- what SearchBot pays, 14x a move"),
    "project_quiet": (500, "lookahead over an inert board; nothing simulated"),
    "encode": (500, "one observation from the crowded board"),
    "encode_late": (500, "one observation from the late board"),
    "mask": (500, "the legal-action mask -- untouched, so a noise control"),
    "decision": (20, "one env.step with the projected reward: what training buys"),
}


def make(name: str, side: Side):
    """``(callable, inner, self_timed)`` for one benchmark on one side.

    ``self_timed`` means the callable returns its own per-unit figure, so that
    setup it has to redo each round stays outside what gets measured.
    """
    if name == "tick_busy":
        base = side.scripted()
        return (lambda: _steps(base, 10)), 1, True
    if name == "tick_quiet":
        base = side.fresh()
        for _ in range(5):
            base.step()
        return (lambda: _steps(base, 100)), 1, True
    if name == "match_2400":
        return (lambda: side.scripted(2400)), 1, False
    if name == "clone":
        return side.scripted().clone, 2, False
    if name == "clone_late":
        return side.scripted(LATE_TICK).clone, 2, False
    if name == "project_3s":
        battle = side.scripted()
        return (lambda: side.lookahead.project(battle, 60)), 1, False
    if name == "project_15s":
        battle = side.scripted(LATE_TICK)
        return (lambda: side.lookahead.project(battle, 300)), 1, False
    if name == "project_quiet":
        battle = side.fresh()
        for _ in range(5):
            battle.step()
        return (lambda: side.lookahead.project(battle, 60)), 50, False
    if name in ("encode", "encode_late"):
        battle = side.scripted(BUSY_TICK if name == "encode" else LATE_TICK)
        config = side.encoding.build_encoding_config(battle.arena, DECK_A, DECK_B)
        registry = side.world[2]
        team = side.entity.Team.BLUE
        encode = side.encoding.encode_observation
        return (lambda: encode(battle, team, registry, config)), 12, False
    if name == "decision":
        # A whole env.step with the projection reward: encode, mask, act,
        # advance, run out the forced decisions, and score the projection at
        # each end of the run-out. This is the unit training actually buys.
        env_mod = importlib.import_module(f"{side.package}.api.env")
        reward_mod = importlib.import_module(f"{side.package}.api.reward")
        data, levels, registry = side.world
        env = env_mod.CRSimEnv(
            data, levels, registry, DECK_A, DECK_B,
            ticks_per_second=20,
            reward_weights=reward_mod.ProjectionWeights(),
        )

        def decisions(count=25):
            """Mean cost of ``count`` consecutive decisions from a fixed seed.

            Averaged rather than minimised over individual decisions: their
            costs differ by an order of magnitude depending on how long a
            run-out of forced moves follows, and training pays the mean. The
            env is reset every round so each round -- and each side of an A/B
            -- measures exactly the same decisions.
            """
            env.reset(seed=7)
            start = CLOCK()
            for _ in range(count):
                mask = env.legal_action_mask()
                legal = mask.nonzero()
                action = [int(v[0]) for v in legal] if legal[0].size else [4, 0, 0]
                if env.step(action)[2]:
                    env.reset(seed=7)
            return (CLOCK() - start) / count

        return decisions, 1, True
    if name == "mask":
        battle = side.scripted()
        config = side.encoding.build_encoding_config(battle.arena, DECK_A, DECK_B)
        registry = side.world[2]
        team = side.entity.Team.BLUE
        mask = side.encoding.legal_action_mask
        return (lambda: mask(battle, team, registry, config)), 300, False
    raise KeyError(name)


def measure(side: Side, name: str, rounds: int) -> float:
    fn, inner, self_timed = make(name, side)
    lowest = float("inf")
    for _ in range(rounds):
        seconds = fn() if self_timed else _block(fn, inner)
        if seconds < lowest:
            lowest = seconds
    return lowest


def compare(new: Side, old: Side, name: str, rounds: int):
    """Interleaved A/B for one benchmark: ``(before, after, median ratio)``."""
    new_fn, new_inner, new_self = make(name, new)
    old_fn, old_inner, old_self = make(name, old)
    low_new = low_old = float("inf")
    ratios = []
    for index in range(rounds):
        order = [("old", old_fn, old_inner, old_self),
                 ("new", new_fn, new_inner, new_self)]
        # Alternate which side goes first so neither gets a systematic
        # advantage from cache state the other left behind.
        if index % 2:
            order.reverse()
        paired = {}
        for label, fn, inner, self_timed in order:
            seconds = fn() if self_timed else _block(fn, inner)
            paired[label] = seconds
            if label == "old":
                low_old = min(low_old, seconds)
            else:
                low_new = min(low_new, seconds)
        ratios.append(paired["old"] / paired["new"])
    ratios.sort()
    return low_old, low_new, ratios[len(ratios) // 2]


def _format(seconds: float) -> str:
    if seconds >= 1e-3:
        return f"{seconds * 1e3:9.2f} ms"
    return f"{seconds * 1e6:9.2f} us"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", default=str(DEFAULT_BUILD),
                        help="csv_logic directory to load")
    parser.add_argument("--against", default=None, metavar="TREE",
                        help="another checkout to A/B against, interleaved in "
                             "this process")
    parser.add_argument("--only", default=None,
                        help="comma-separated subset of: " + ", ".join(BENCHES))
    parser.add_argument("--rounds", type=int, default=None,
                        help="override the per-benchmark round count")
    parser.add_argument("--json", action="store_true",
                        help="print one JSON object instead of a table")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    names = args.only.split(",") if args.only else list(BENCHES)
    for name in names:
        if name not in BENCHES:
            parser.error(f"unknown benchmark {name!r}; choose from {', '.join(BENCHES)}")

    new = Side("cr_sim", args.build)
    old = None
    if args.against:
        sys.meta_path.insert(0, _AliasFinder("cr_sim_before", Path(args.against)))
        old = Side("cr_sim_before", args.build)
        if Path(new.battle.__file__).samefile(old.battle.__file__):
            parser.error("--against resolved to this same tree")

    if not args.json:
        print(f"busy fixture (tick {BUSY_TICK}): {describe(new.scripted())}")
        print(f"late fixture (tick {LATE_TICK}): {describe(new.scripted(LATE_TICK))}")
        if old is not None:
            print(f"before: {old.battle.__file__}")
            print(f"after : {new.battle.__file__}")
        print()

    rows = []
    for name in names:
        rounds = args.rounds or BENCHES[name][0]
        if old is None:
            rows.append((name, None, measure(new, name, rounds), None, rounds))
        else:
            before, after, median = compare(new, old, name, rounds)
            rows.append((name, before, after, median, rounds))
        print(f"  {name} done", file=sys.stderr)

    if args.json:
        print(json.dumps({
            name: {"before": before, "after": after,
                   "median_ratio": median, "rounds": rounds}
            for name, before, after, median, rounds in rows
        }))
        return

    width = max(len(n) for n, *_ in rows)
    if old is None:
        for name, _, seconds, _, _ in rows:
            print(f"{name.ljust(width)}   {_format(seconds)}   {BENCHES[name][1]}")
        return
    print(f"{'benchmark'.ljust(width)}   {'before':>12}   {'after':>12}   "
          f"{'min-ratio':>9}   {'median':>7}   rounds")
    for name, before, after, median, rounds in rows:
        print(f"{name.ljust(width)}   {_format(before)}   {_format(after)}   "
              f"{before / after:8.2f}x   {median:6.2f}x   {rounds}")


if __name__ == "__main__":
    main()
