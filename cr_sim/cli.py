"""``cr-sim`` command line.

    python -m cr_sim.cli ingest              summarise the loaded build
    python -m cr_sim.cli cards [--kind troop] list the playable card pool
    python -m cr_sim.cli card Knight          full resolved stats for one card
    python -m cr_sim.cli validate             run the stat gate
    python -m cr_sim.cli interactions         the interaction-matrix gate (verification gate #2)
    python -m cr_sim.cli engagement          reach + Princess Tower support
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
from .data.engagement import (
    duel_matrix,
    tower_matrix,
    write_duel_csv,
    write_tower_csv,
)
from .data.interactions import (
    DEFAULT_SHEET,
    KNOWN_UNMAPPED,
    NAME_MAP,
    SIM_CARDS,
    build_profiles,
    categorise_sheet_comparison,
    compute_matrix,
    load_sheet,
    predicted_winner,
    simulate_matrix,
    write_generated_csv,
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


DEFAULT_GENERATED_MATRIX = ROOT / "reference" / "interactions_generated.csv"
DEFAULT_DUELS = ROOT / "reference" / "duels_generated.csv"
DEFAULT_TOWER_ASSIST = ROOT / "reference" / "tower_assist_generated.csv"


def cmd_engagement(args) -> int:
    """Who wins once reach and the Princess Tower are accounted for.

    The hit-count matrix next door answers "how many hits"; this answers
    "and does that win the fight", which is what a hit count is read for.
    """
    data, levels, registry = _load(args.build)
    defenses, attacks, _labels = build_profiles(data, levels, registry)

    duels = duel_matrix(defenses, attacks)
    decided = [d for d in duels.values() if d.winner in ("a", "b")]
    clean = [d for d in decided if d.clean]
    print(f"{len(duels)} fightable pair(s): {len(decided)} decided, "
          f"{len(clean)} ({len(clean) / max(1, len(decided)):.1%}) won without "
          f"the winner being hit once")

    print(f"\nlargest head starts -- hits landed before the other side could answer:")
    for (a, b), fight in sorted(duels.items(), key=lambda kv: -kv[1].head_start)[:args.top]:
        if fight.winner not in ("a", "b"):
            continue
        won = a if fight.winner == "a" else b
        print(f"  {won:<18} +{fight.head_start:>3} free hits vs "
              f"{b if fight.winner == 'a' else a}")

    assists = tower_matrix(defenses, attacks)
    changed = [v for v in assists.values() if v.saved]
    free = [v for v in assists.values() if v.with_tower == 0]
    saved = [v.saved for v in changed]
    print(f"\n{len(assists)} (defender, attacker) pair(s) with a Princess Tower firing:")
    print(f"  changes the hit count on {len(changed)} "
          f"({len(changed) / max(1, len(assists)):.1%}), mean {sum(saved) / max(1, len(saved)):.1f} "
          f"hits saved, max {max(saved) if saved else 0}")
    print(f"  finishes the job before the troop lands a hit on {len(free)} pair(s)")

    biggest = sorted(((k, v) for k, v in assists.items() if v.saved),
                     key=lambda kv: -kv[1].saved)[:args.top]
    print(f"\nwhere the tower does the most work:")
    for (defender, attacker), v in biggest:
        print(f"  {attacker:<18} vs {defender:<16} {v.alone:>3} hits alone -> "
              f"{v.with_tower:<3} with tower")

    if args.write:
        write_duel_csv(duels, args.duels_out)
        write_tower_csv(assists, args.tower_out)
        print(f"\nwrote {args.duels_out}\n      {args.tower_out}")
    return 0


def cmd_interactions(args) -> int:
    import datetime

    data, levels, registry = _load(args.build)
    defenses, attacks, labels = build_profiles(data, levels, registry)
    computed = compute_matrix(defenses, attacks)
    print(
        f"computed matrix: {len(defenses)} defenders x {len(attacks)} attackers "
        f"-> {len(computed)} applicable pair(s), from the whole standard pool + towers"
    )

    simulated = {}
    if args.simulate:
        sim_cards = tuple(c.strip() for c in args.sim_cards.split(",")) if args.sim_cards else SIM_CARDS
        print(f"simulating {len(sim_cards)} card(s) pairwise ({len(sim_cards) * (len(sim_cards) - 1)} duels)...")
        simulated = simulate_matrix(data, levels, registry, sim_cards)
        # The signal worth reporting is not raw agreement -- arithmetic and a
        # real duel should usually agree on the stronger card -- but the
        # pairs where they *don't*: those name a mechanic (deploy time,
        # closing distance, retargeting, splash) that only the simulation
        # can see.
        comparable = 0
        agree = 0
        surprises = []
        for (d, a), sim in simulated.items():
            if sim.winner not in ("attacker", "defender"):
                continue  # a draw or a timeout has no clean predicted counterpart
            guess = predicted_winner(computed, d, a)
            if guess is None or guess == "draw":
                continue
            comparable += 1
            if guess == sim.winner:
                agree += 1
            else:
                surprises.append((d, a, guess, sim))
        print(f"  {len(simulated)} duel(s) run; {comparable} have a clean arithmetic prediction to compare")
        if comparable:
            print(f"  arithmetic and simulation agree on the winner in {agree}/{comparable} ({100 * agree / comparable:.1f}%)")
        if surprises:
            print(f"  {len(surprises)} mechanic-driven surprise(s) -- arithmetic predicted one winner, the duel gave the other:")
            for d, a, guess, sim in surprises[:15]:
                print(f"    {a:16} vs {d:16} arithmetic picks {guess:9} sim says {sim.winner:9} in {sim.ticks}t, {sim.hits_landed} hit(s) landed")

    if args.write:
        provenance = args.build / "_PROVENANCE.txt"
        build_id = provenance.read_text(encoding="utf-8").splitlines()[0] if provenance.exists() else "unknown"
        write_generated_csv(
            args.out, defenses, attacks, computed, simulated=simulated,
            build=build_id, generated_at=datetime.date.today().isoformat(),
        )
        print(f"wrote {args.out}")

    if args.sheet.exists():
        sheet = load_sheet(args.sheet)
        comparisons, unmapped, not_applicable = categorise_sheet_comparison(sheet, defenses, attacks)
        agree = sum(1 for c in comparisons if c.category == "agree")
        explained = sum(1 for c in comparisons if c.category == "explained")
        defect = sum(1 for c in comparisons if c.category == "defect")
        total = len(comparisons)
        print(f"\n== cross-check against the (stale, ~1yr old) community sheet {args.sheet} ==")
        print(f"sheet cells: {len(sheet)}; mapped names: {len(NAME_MAP)}; unmapped names: {len(KNOWN_UNMAPPED)}")
        print(f"comparable pairs: {total}  (unmapped: {len(unmapped)}, not applicable: {len(not_applicable)})")
        if total:
            print(
                f"  agree:            {agree:5} ({100 * agree / total:5.1f}%)  -- engine reproduces the sheet's number"
            )
            print(
                f"  explained:        {explained:5} ({100 * explained / total:5.1f}%)  -- bare hp/damage don't match the "
                f"sheet either; a stat moved since it was written"
            )
            print(
                f"  defect:           {defect:5} ({100 * defect / total:5.1f}%)  -- bare hp/damage DO match the sheet; "
                f"only the engine's shield/tower/ramp adjustment disagrees"
            )
        if defect:
            print(f"\n== all {defect} 'defect' pair(s), worst first ==")
            defects = sorted((c for c in comparisons if c.category == "defect"), key=lambda c: -c.delta)
            for c in defects[:60]:
                print(
                    f"  {c.attacker_csv:18} kills {c.defender_csv:18} sheet={c.sheet_hits:>3} engine={c.engine_hits:>3} "
                    f"naive={c.naive_hits!s:>3}  hp={c.hitpoints} shield={c.shield_hitpoints} dmg={c.damage}"
                )
            if len(defects) > 60:
                print(f"  ... and {len(defects) - 60} more")
    else:
        print(f"\nno sheet at {args.sheet}; skipping the cross-check")
    return 0


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


def _aim(battle, team, card, lanes, rng):
    """Pick somewhere sensible to put a card.

    Troops go down in their owner's half, as they must. A spell is aimed at the
    enemy instead -- at the biggest clump of them if there is one, otherwise at
    a tower. Dropping spells on your own side, which is what the deploy
    positions do, means every one of them lands on empty grass.

    This is placement, not tactics: it exists so a replay shows spells doing
    something. A real agent arrives with M8.
    """
    from .engine.entity import EntityKind, Team
    from .engine.fixed import to_tiles

    is_spell = card is not None and card.kind is CardKind.SPELL
    if not is_spell:
        lane = lanes[rng.below(len(lanes))]
        return lane, (11.0 + rng.below(4) if team is Team.BLUE else 21.0 - rng.below(4))

    enemies = [
        e
        for e in battle.entities
        if not e.dead
        and e.team is not team
        and e.kind is EntityKind.TROOP
        and not e.is_deploying
    ]
    if enemies:
        # The unit with the most company within a couple of tiles.
        def crowd(target):
            return sum(
                1
                for other in enemies
                if abs(other.x - target.x) < 2 * 18000 and abs(other.y - target.y) < 2 * 18000
            )

        best = max(enemies, key=lambda e: (crowd(e), e.id))
        return to_tiles(best.x), to_tiles(best.y)

    towers = [t for t in battle._towers[team.opponent] if not t.dead and "King" not in t.spec.name]
    if towers:
        target = towers[rng.below(len(towers))]
        return to_tiles(target.x), to_tiles(target.y)
    return None


def cmd_battle(args) -> int:
    from .engine.battle import Battle, BattleConfig
    from .engine.entity import EntityKind, Team
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
            card = registry.get(choice)
            spot = _aim(battle, team, card, lanes, rng)
            if spot is None:
                continue
            lane, y = spot
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
    # Towers are excluded because they are always there, and projectiles and
    # area effects because they are not units -- a Poison cloud still drifting
    # when the whistle goes was being counted and printed as an anonymous "?".
    alive = [
        e
        for e in battle.entities
        if not e.dead and e.kind in (EntityKind.TROOP, EntityKind.BUILDING)
    ]
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
            scale=args.scale,
            art_scale=args.icon_scale,
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

    p = sub.add_parser("interactions", help="the interaction-matrix gate: compute, simulate, cross-check")
    p.add_argument("--sheet", type=Path, default=DEFAULT_SHEET, help="community hits-to-kill CSV to cross-check")
    p.add_argument("--write", action="store_true", help="write the computed matrix to --out")
    p.add_argument("--out", type=Path, default=DEFAULT_GENERATED_MATRIX)
    p.add_argument("--simulate", action="store_true", help="also run real duels for a curated card subset")
    p.add_argument("--sim-cards", help="comma-separated card names, overriding the default subset")
    p.set_defaults(func=cmd_interactions)

    p = sub.add_parser("engagement", help="who wins once reach and tower support are counted")
    p.add_argument("--write", action="store_true", help="write both CSVs")
    p.add_argument("--duels-out", type=Path, default=DEFAULT_DUELS)
    p.add_argument("--tower-out", type=Path, default=DEFAULT_TOWER_ASSIST)
    p.add_argument("--top", type=int, default=12, help="how many examples to print")
    p.set_defaults(func=cmd_engagement)

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
    p.add_argument("--scale", type=int, default=15, help="viewer pixels per half-tile")
    p.add_argument("--icon-scale", type=float, default=2.2, help="art size relative to the real hitbox")
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
