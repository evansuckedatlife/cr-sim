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

    p = sub.add_parser("freeze", help="re-freeze the regression baseline")
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    p.set_defaults(func=cmd_freeze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
