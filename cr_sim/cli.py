"""``cr-sim`` command line.

    python -m cr_sim.cli ingest              summarise the loaded build
    python -m cr_sim.cli cards [--kind troop] list the playable card pool
    python -m cr_sim.cli card Knight          full resolved stats for one card
    python -m cr_sim.cli validate             run the stat gate
    python -m cr_sim.cli freeze               re-freeze the regression baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data.cards import (
    TOURNAMENT_DISPLAY_LEVEL,
    CardKind,
    build_card_registry,
    card_stat_summary,
)
from .data.leveling import build_level_table
from .data.source import LogicData, UnknownEntity
from .data.validate import (
    DEFAULT_REFERENCE,
    load_reference,
    validate_cards,
    write_reference,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD = ROOT / "data_cache" / "csv_logic"


def _load(build: Path):
    data = LogicData.load(build)
    return data, build_level_table(data), build_card_registry(data)


def cmd_ingest(args) -> int:
    data, levels, registry = _load(args.build)
    counts = data.summary()
    csv_tables = {k: v for k, v in counts.items() if k.startswith("csv:")}
    entities = {
        ns: len(data.names(ns)) for ns in ("CHARACTER", "BUILDING", "PROJECTILE", "AEO", "EXT")
    }
    print(f"build:      {args.build}")
    provenance = args.build / "_PROVENANCE.txt"
    if provenance.exists():
        for line in provenance.read_text(encoding="utf-8").splitlines():
            print(f"            {line}")
    print(f"csv tables: {len(csv_tables)}")
    print(f"entities:   {entities}")
    print(f"rarities:   {{{', '.join(f'{r.name}: L1-L{r.level_count}' for r in levels.rarities.values())}}}")
    print(f"cards:      {len(registry)} rows, {len(registry.standard())} standard playable, "
          f"{len(registry.evolutions())} evolutions")
    by_kind = {k.value: len(registry.of_kind(k)) for k in CardKind}
    print(f"            {by_kind}")
    errors = data.sections.get("_ERRORS", {})
    if errors:
        print(f"parse errors: {errors}")
        return 2
    return 0


def cmd_cards(args) -> int:
    data, levels, registry = _load(args.build)
    cards = registry.standard()
    if args.kind:
        cards = tuple(c for c in cards if c.kind.value == args.kind)
    cards = sorted(cards, key=lambda c: (c.mana_cost, c.name))
    print(f"{'card':24} {'kind':9} {'rarity':10} {'e':>2} {'hp':>6} {'dmg':>6} {'dps':>5} {'n':>2}")
    for card in cards:
        s = card_stat_summary(data, levels, card, display_level=args.level)
        print(
            f"{card.name:24} {card.kind.value:9} {card.rarity:10} {card.mana_cost:>2} "
            f"{str(s.get('hitpoints', '-')):>6} {str(s.get('damage', '-')):>6} "
            f"{str(s.get('dps', '-')):>5} {s.get('count', 1):>2}"
        )
    print(f"\n{len(cards)} cards at displayed level {args.level}")
    return 0


def cmd_card(args) -> int:
    data, levels, registry = _load(args.build)
    card = registry.get(args.name)
    if card is not None:
        print(json.dumps(card_stat_summary(data, levels, card, display_level=args.level), indent=2))
    try:
        entity = data.resolve(args.name if card is None else (card.summons() or [(args.name, 1)])[0][0])
    except UnknownEntity:
        if card is None:
            print(f"no card or entity named {args.name!r}", file=sys.stderr)
            return 1
        return 0
    print("\nresolved entity attributes:")
    noise = ("Export", "Effect", "SWF", "Asset", "Shadow", "Anim", "Prefab", "Shader", "Mesh",
             "FileName", "Skin", "Icon")
    for key, value in sorted(entity.items()):
        if args.all or not any(n in key for n in noise):
            print(f"  {key:34} {value}")
    return 0


def cmd_validate(args) -> int:
    data, levels, registry = _load(args.build)

    anchors = json.loads((ROOT / "reference" / "anchors.json").read_text(encoding="utf-8"))
    verified = {k: v for k, v in anchors["verified_cards"].items()}
    report = validate_cards(data, levels, registry, verified, display_level=anchors["display_level"])
    print("== anchors (hand-verified live values) ==")
    print(report.summary())

    if args.reference.exists():
        baseline = load_reference(args.reference)
        drift = validate_cards(data, levels, registry, baseline)
        print("\n== regression baseline ==")
        print(drift.summary())
        ok = report.ok and drift.ok
    else:
        print(f"\nno regression baseline at {args.reference}; run `freeze` to create one")
        ok = report.ok

    open_items = [q for q in anchors.get("open_questions", []) if q.get("status") == "open"]
    if open_items:
        print(f"\n== {len(open_items)} open question(s) ==")
        for item in open_items:
            print(f"  [{item['id']}] {item['question']}")
            print(f"      resolve: {item['how_to_resolve']}")
    return 0 if ok else 1


#: A couple of recognisable decks so `battle` works with no arguments.
DECKS = {
    "hog_cycle": ("HogRider", "Musketeer", "Cannon", "Skeletons", "IceSpirits", "Log", "Fireball", "Goblins"),
    "giant_beatdown": ("Giant", "Musketeer", "MiniPekka", "Archer", "Zap", "Fireball", "Knight", "Bomber"),
    "golem_beatdown": ("Golem", "BabyDragon", "Wizard", "MegaMinion", "Barbarians", "Zap", "Arrows", "Knight"),
}


def _resolve_deck(name: str) -> tuple[str, ...]:
    if name in DECKS:
        return DECKS[name]
    return tuple(part.strip() for part in name.split(",") if part.strip())


def cmd_battle(args) -> int:
    from .engine.battle import Battle, BattleConfig
    from .engine.entity import Team
    from .engine.fixed import SUBTILES_PER_TILE, to_tiles
    from .render.web import build_icon_map, render_ascii, render_replay

    data, levels, registry = _load(args.build)
    blue, red = _resolve_deck(args.blue), _resolve_deck(args.red)
    for deck, label in ((blue, "blue"), (red, "red")):
        missing = [c for c in deck if registry.get(c) is None]
        if missing:
            print(f"unknown {label} card(s): {missing}", file=sys.stderr)
            return 1

    battle = Battle(
        data,
        levels,
        registry,
        BattleConfig(
            seed=args.seed,
            ticks_per_second=args.tps,
            blue_deck=blue,
            red_deck=red,
            level=args.level,
            record_frames=args.html is not None,
            frame_interval=args.frame_interval,
        ),
    )
    print(f"seed={args.seed} tps={args.tps} level={args.level}")
    print(f"blue: {', '.join(blue)}")
    print(f"red:  {', '.join(red)}")
    print(f"blue opening hand: {battle.players[Team.BLUE].hand}")

    # A scripted opponent so there is something to watch. This is not an agent
    # and makes no tactical decisions -- it spends elixir on whatever is in hand
    # as soon as it can afford it, which is enough to exercise deployment, lane
    # routing, building lifetimes and the tick loop. M2 replaces it.
    tile = SUBTILES_PER_TILE
    lanes = (3.5, 9.0, 14.5)
    played: list[str] = []
    rng = battle.rng.stream("demo-script")
    next_play = {Team.BLUE: int(1.5 * args.tps), Team.RED: int(3.0 * args.tps)}
    interval = max(1, int(args.interval * args.tps))

    limit = args.ticks if args.ticks else battle.timeline.total_ticks
    while battle.tick < limit and not battle.finished:
        for team in (Team.BLUE, Team.RED):
            if battle.tick < next_play[team]:
                continue
            player = battle.players[team]
            affordable = []
            for name in player.hand:
                card = registry.get(name)
                if card is not None and player.elixir.can_afford(card.mana_cost):
                    affordable.append(name)
            if not affordable:
                continue
            choice = affordable[rng.below(len(affordable))]
            lane = lanes[rng.below(len(lanes))]
            y = 11.0 + rng.below(4) if team is Team.BLUE else 21.0 - rng.below(4)
            if battle.play_card(team, choice, int(lane * tile), int(y * tile)):
                played.append(
                    f"t={battle.tick / args.tps:>5.1f}s {team.name:5} {choice:12} @({lane}, {y})"
                )
                next_play[team] = battle.tick + interval
        battle.step()

    result = battle.result or battle._decide("time")
    print("\ndeployments:")
    for line in played:
        print(f"  {line}")
    alive = [e for e in battle.entities if not e.dead and int(e.kind) != 2]
    print(f"\nafter {battle.tick} ticks ({battle.tick / args.tps:.1f}s): {len(alive)} unit(s) alive")
    for e in alive:
        print(
            f"  {getattr(e.spec, 'name', '?'):16} {e.team.name:5} "
            f"({to_tiles(e.x):>6.2f},{to_tiles(e.y):>6.2f}) hp={e.hitpoints}"
        )
    print(f"\nresult: {result.reason}, crowns {result.blue_crowns}-{result.red_crowns}")
    print(f"state hash: {battle.hash():016x}")

    if args.ascii:
        print("\n" + render_ascii(battle.arena, battle.entities))
    if args.html:
        out = render_replay(
            battle.arena,
            battle.frames,
            args.html,
            # playback advances one frame at a time; the clock still counts real ticks
            ticks_per_second=args.tps / max(1, args.frame_interval),
            real_tps=args.tps,
            icons=build_icon_map(registry),
            costs={c.name: c.mana_cost for c in registry.standard()},
            meta=f"seed {args.seed} &middot; {args.tps} TPS &middot; level {args.level}",
        )
        print(f"\nwrote {out}  ({len(battle.frames)} frames) - open it in a browser")
    return 0


def cmd_arena(args) -> int:
    from .engine.arena import load_arena
    from .engine.entity import Team
    from .engine.fixed import to_tiles

    data, _levels, _registry = _load(args.build)
    arena = load_arena(data)
    top, bottom = arena.river_band()
    print(f"{arena.width_tiles} x {arena.height_tiles} tiles  (source: {arena.source})")
    print(f"river:   y {to_tiles(top)} -> {to_tiles(bottom)}   midline y={to_tiles(arena.midline())}")
    for i, (lo, hi, centre) in enumerate(arena.bridges()):
        print(
            f"bridge {i}: x {to_tiles(lo)}..{to_tiles(hi)} "
            f"({to_tiles(hi - lo)} tiles wide), centre x={to_tiles(centre)}"
        )
    print("towers:")
    for tower in arena.towers:
        print(f"  {tower.name:14} {tower.team.name:5} ({to_tiles(tower.x):>5}, {to_tiles(tower.y):>5})")
    for team in (Team.BLUE, Team.RED):
        low, high = arena.own_half(team)
        print(f"{team.name} deploy zone: y {to_tiles(low)} .. {to_tiles(high)}")
    if args.map:
        print()
        print(arena.render())
    return 0


def cmd_freeze(args) -> int:
    data, levels, registry = _load(args.build)
    provenance = args.build / "_PROVENANCE.txt"
    build_id = provenance.read_text(encoding="utf-8").splitlines()[0] if provenance.exists() else "unknown"
    count = write_reference(data, levels, registry, args.reference, build=build_id)
    print(f"froze {count} cards to {args.reference}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cr-sim")
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD, help="decoded csv_logic dir")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="summarise the loaded build").set_defaults(func=cmd_ingest)

    p = sub.add_parser("cards", help="list the playable card pool")
    p.add_argument("--kind", choices=[k.value for k in CardKind])
    p.add_argument("--level", type=int, default=TOURNAMENT_DISPLAY_LEVEL)
    p.set_defaults(func=cmd_cards)

    p = sub.add_parser("card", help="full stats for one card or entity")
    p.add_argument("name")
    p.add_argument("--level", type=int, default=TOURNAMENT_DISPLAY_LEVEL)
    p.add_argument("--all", action="store_true", help="include cosmetic fields")
    p.set_defaults(func=cmd_card)

    p = sub.add_parser("validate", help="run the stat gate")
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("battle", help="run a battle and optionally write an HTML replay")
    p.add_argument("--blue", default="hog_cycle", help="deck name or comma-separated cards")
    p.add_argument("--red", default="giant_beatdown")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tps", type=int, default=60)
    p.add_argument("--level", type=int, default=TOURNAMENT_DISPLAY_LEVEL)
    p.add_argument("--ticks", type=int, default=0, help="stop early (0 = full match)")
    p.add_argument("--html", type=Path, help="write a standalone replay viewer here")
    p.add_argument("--ascii", action="store_true", help="print a terminal snapshot")
    p.add_argument("--frame-interval", type=int, default=3, help="record 1 viewer frame every N ticks")
    p.add_argument("--interval", type=float, default=2.5, help="seconds between scripted deployments")
    p.set_defaults(func=cmd_battle)

    p = sub.add_parser("arena", help="print the arena geometry")
    p.add_argument("--map", action="store_true", help="draw the terrain grid")
    p.set_defaults(func=cmd_arena)

    p = sub.add_parser("freeze", help="re-freeze the regression baseline")
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    p.set_defaults(func=cmd_freeze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
