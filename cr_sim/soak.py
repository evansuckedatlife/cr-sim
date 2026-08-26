"""Run a great many matches and report what looked wrong.

A test suite asks whether the cases someone thought of behave correctly. This
asks a different question: across a hundred thousand matches nobody designed,
does anything happen that should not? The two find different bugs. A spell that
is subtly mis-scaled passes every test written about the spells that were
checked, and shows up here as a card whose damage-per-cast sits an order of
magnitude off its neighbours.

What it looks for, and why each is a real signal rather than noise:

*exceptions*
    Any at all. The engine runs 10,000 ticks a match; a path taken once in
    50,000 matches is one no test will ever reach on purpose.
*matches that never resolve*
    A match that hits its tick ceiling with both kings alive is legal -- most
    do -- but one that hits it with nothing alive and no damage being dealt is
    a stalled simulation.
*cards that deploy nothing*
    A card played with the elixir to pay for it should put something on the
    board or change something. One that reliably does neither is inert, and
    inert is exactly how every ACTION-driven card failed before the
    interpreter existed.
*damage that cannot happen*
    Negative hitpoints, damage exceeding a target's maximum in one hit from a
    unit that cannot do that, healing past full.

Workers are deliberately fewer than the machine's cores. This is a background
job; taking every core makes the machine unusable and the job is not more
urgent than whatever else is running on it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD = ROOT / "data_cache" / "csv_logic"

#: Cores left for the rest of the machine.
SPARE_CORES = 3


@dataclass
class MatchReport:
    """What one match produced."""

    index: int
    seed: int
    blue: list[str]
    red: list[str]
    ticks: int = 0
    reason: str = ""
    blue_crowns: int = 0
    red_crowns: int = 0
    damage_events: int = 0
    total_damage: int = 0
    peak_entities: int = 0
    cards_played: int = 0
    #: Cards that were played and produced neither a unit nor any damage.
    inert_cards: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    error: str | None = None
    seconds: float = 0.0


_WORLD: dict[str, Any] = {}


def _world(build: str) -> tuple:
    """Load the build once per worker process.

    Parsing the whole card set takes about a second; doing it per match would
    dominate the run and measure the loader instead of the engine.
    """
    if "data" not in _WORLD:
        from .data.cards import build_card_registry
        from .data.leveling import build_level_table
        from .data.source import LogicData

        data = LogicData.load(build)
        _WORLD["data"] = data
        _WORLD["levels"] = build_level_table(data)
        _WORLD["registry"] = build_card_registry(data)
        _WORLD["pool"] = [c.name for c in _WORLD["registry"].standard()]
    return _WORLD["data"], _WORLD["levels"], _WORLD["registry"], _WORLD["pool"]


def run_match(job: tuple[int, int, str, int, int, bool]) -> dict:
    """Play one match. Never raises: a crash is the finding."""
    index, seed, build, tps, match_seconds, spells_only = job
    report = MatchReport(index=index, seed=seed, blue=[], red=[])
    started = time.perf_counter()
    try:
        import numpy as np

        from .engine.battle import Battle, BattleConfig
        from .engine.entity import EntityKind, Team
        from .engine.fixed import tiles

        data, levels, registry, pool = _world(build)
        rng = np.random.default_rng(seed)

        if spells_only:
            # Weighted toward spells, which is where the accuracy doubt is.
            spells = [c.name for c in registry.standard() if c.kind.value == "spell"]
            others = [n for n in pool if n not in spells]
            def deck():
                picked = list(rng.choice(spells, size=min(5, len(spells)), replace=False))
                picked += list(rng.choice(others, size=8 - len(picked), replace=False))
                return picked
        else:
            def deck():
                return list(rng.choice(pool, size=8, replace=False))

        blue, red = deck(), deck()
        report.blue, report.red = blue, red

        battle = Battle(
            data, levels, registry,
            BattleConfig(seed=seed, ticks_per_second=tps, blue_deck=blue, red_deck=red),
        )
        limit = tps * match_seconds
        played: list[tuple[str, int, int]] = []

        while battle.result is None and battle.tick < limit:
            if battle.tick % max(1, tps // 2) == 0:
                for team in (Team.BLUE, Team.RED):
                    player = battle.players[team]
                    hand = [
                        n for n in player.hand
                        if (card := registry.get(n)) is not None
                        and player.elixir.can_afford(card.mana_cost)
                    ]
                    if not hand:
                        continue
                    name = hand[int(rng.integers(len(hand)))]
                    x = tiles(float(rng.uniform(1.5, 16.5)))
                    y = tiles(float(rng.uniform(1.5, 14.5) if team is Team.BLUE
                                    else rng.uniform(17.5, 30.5)))
                    before = len(battle.entities)
                    if battle.play_card(team, name, x, y):
                        played.append((name, battle.tick, len(battle.entities) - before))
            battle.step()
            report.peak_entities = max(report.peak_entities, len(battle.entities))

        report.ticks = battle.tick
        result = battle.result
        report.reason = result.reason if result else "tick limit"
        report.blue_crowns = battle.players[Team.BLUE].crowns
        report.red_crowns = battle.players[Team.RED].crowns
        report.damage_events = len(battle.damage_log)
        report.total_damage = sum(e.amount for e in battle.damage_log)
        report.cards_played = len(played)

        _check(battle, report, played, registry, limit)
    except Exception:  # noqa: BLE001 - the whole point is to catch these
        report.error = traceback.format_exc(limit=6)
    report.seconds = round(time.perf_counter() - started, 3)
    return asdict(report)


def _check(battle, report: MatchReport, played, registry, limit: int) -> None:
    """Everything that should not have happened."""
    from .engine.entity import EntityKind

    for entity in battle.entities + battle.graveyard:
        if entity.hitpoints < 0:
            report.anomalies.append(f"negative hitpoints on {_name(entity)}")
            break
    for entity in battle.entities:
        if entity.max_hitpoints and entity.hitpoints > entity.max_hitpoints:
            report.anomalies.append(f"{_name(entity)} above maximum hitpoints")
            break
        if not entity.flying and entity.kind is EntityKind.TROOP:
            if not battle.arena.is_walkable(entity.x, entity.y, flying=False):
                report.anomalies.append(f"{_name(entity)} standing on impassable ground")
                break

    if report.ticks >= limit and report.damage_events == 0:
        report.anomalies.append("ran to the tick limit with no damage dealt at all")

    # A card played with the elixir to pay for it that put nothing on the board
    # is the signature of an unimplemented mechanic.
    inert = Counter()
    for name, _tick, delta in played:
        card = registry.get(name)
        if card is None:
            continue
        if delta <= 0 and card.kind.value != "spell":
            inert[name] += 1
    report.inert_cards = sorted(n for n, count in inert.items() if count >= 2)


def _name(entity) -> str:
    return entity.spec.name if entity.spec is not None else entity.kind.name


def summarise(reports: Sequence[dict]) -> dict:
    """Aggregate the findings, worst first."""
    errors = Counter()
    anomalies = Counter()
    inert = Counter()
    reasons = Counter()
    ticks = []
    for report in reports:
        if report["error"]:
            errors[report["error"].strip().splitlines()[-1][:160]] += 1
        for note in report["anomalies"]:
            anomalies[note] += 1
        for name in report["inert_cards"]:
            inert[name] += 1
        reasons[report["reason"]] += 1
        ticks.append(report["ticks"])
    return {
        "matches": len(reports),
        "errors": errors.most_common(10),
        "anomalies": anomalies.most_common(10),
        "inert_cards": inert.most_common(20),
        "reasons": reasons.most_common(),
        "mean_ticks": sum(ticks) / max(1, len(ticks)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cr-sim-soak")
    parser.add_argument("--matches", type=int, default=100_000)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 4) - SPARE_CORES),
        help=f"defaults to cores minus {SPARE_CORES}; this is a background job",
    )
    parser.add_argument("--tps", type=int, default=20)
    parser.add_argument("--match-seconds", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "soak")
    parser.add_argument("--spells", action="store_true",
                        help="weight decks toward spells")
    parser.add_argument("--report-every", type=int, default=2000)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    detail = args.out / "matches.jsonl"
    started = time.perf_counter()

    jobs = (
        (i, args.seed + i, str(args.build), args.tps, args.match_seconds, args.spells)
        for i in range(args.matches)
    )
    done = 0
    keep: list[dict] = []
    print(f"soak: {args.matches} matches on {args.workers} workers "
          f"({os.cpu_count()} cores, leaving {(os.cpu_count() or 0) - args.workers})",
          flush=True)

    with detail.open("w", encoding="utf-8") as stream, Pool(args.workers) as pool:
        # imap_unordered so a slow match does not hold up the report, and
        # chunked so the queue is not the bottleneck at this count.
        for report in pool.imap_unordered(run_match, jobs, chunksize=16):
            done += 1
            # Only the interesting ones are written. A hundred thousand clean
            # matches is a hundred thousand lines nobody will read, and the
            # summary counts them anyway.
            if report["error"] or report["anomalies"] or report["inert_cards"]:
                stream.write(json.dumps(report) + "\n")
                stream.flush()
            keep.append({k: report[k] for k in
                         ("error", "anomalies", "inert_cards", "reason", "ticks")})
            if done % args.report_every == 0:
                elapsed = time.perf_counter() - started
                summary = summarise(keep)
                print(
                    f"  {done:>7}/{args.matches}  {done / elapsed:6.1f} match/s  "
                    f"errors {sum(c for _, c in summary['errors'])}  "
                    f"anomalies {sum(c for _, c in summary['anomalies'])}  "
                    f"inert {len(summary['inert_cards'])}",
                    flush=True,
                )
                (args.out / "summary.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )

    summary = summarise(keep)
    summary["seconds"] = round(time.perf_counter() - started, 1)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\ndone in {summary['seconds'] / 60:.1f} min -> {args.out}")
    for label in ("errors", "anomalies", "inert_cards"):
        if summary[label]:
            print(f"\n{label}:")
            for name, count in summary[label]:
                print(f"  {count:>6}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
