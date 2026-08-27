"""The interaction-matrix gate: verification gate #2 from the project plan.

The stat gate (:mod:`cr_sim.data.validate`) pins individual numbers -- a
Knight's hitpoints, a Fireball's damage. It says nothing about whether those
numbers *combine* correctly: whether a shield really absorbs one whole hit,
whether a spell's reduced tower damage is applied with the right sign, whether
an Inferno ramp is timed right. This module is the next layer up.

Two matrices are built, deliberately by different means:

``computed``
    Pure arithmetic over every playable card's resolved stats --
    ``hits = ceil(effective_hitpoints / damage)``, honouring shields, crown
    tower damage percentage, air/ground targeting and Inferno-style ramps.
    Cheap enough to run over the whole standard pool both ways.

``simulated``
    An actual duel through :class:`~cr_sim.engine.battle.Battle` for a smaller
    set of cards. This is the only one of the two that can see deploy time,
    first-hit timing, retargeting, or a ranged unit that simply never gets hit
    -- the things arithmetic cannot express. Where the two disagree, that
    disagreement *names a mechanic*, which is the most useful output either
    matrix produces.

A third, smaller piece cross-checks both against ``reference/hits_to_kill.csv``,
a community-maintained sheet that is roughly a year old (see
``reference/hits_to_kill.md``). Because the sheet predates an unknown number of
balance patches, a mismatch against it is not automatically a bug -- see
:func:`categorise_sheet_comparison` for how the three possible readings
(agreement, a stat that plausibly moved, or a defect the stats do not explain)
are told apart.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..engine.constants import TickClock
from ..engine.specs import SpecError, UnitSpec, build_tower_spec, build_unit_spec
from .cards import CardRegistry, card_stat_summary
from .leveling import LevelTable, build_tower_scales, tower_class_for
from .source import LogicData

__all__ = [
    "DEFAULT_SHEET",
    "Ref",
    "NAME_MAP",
    "DefenseProfile",
    "AttackProfile",
    "HitResult",
    "SimResult",
    "SheetRow",
    "SheetComparison",
    "build_profiles",
    "compute_hits",
    "compute_hits_naive",
    "compute_matrix",
    "predicted_winner",
    "simulate_duel",
    "simulate_matrix",
    "load_sheet",
    "categorise_sheet_comparison",
    "write_generated_csv",
    "SIM_CARDS",
]

DEFAULT_SHEET = Path(__file__).resolve().parents[2] / "reference" / "hits_to_kill.csv"

TOURNAMENT_DISPLAY_LEVEL = 11
TOWER_KEYS = ("KingTower", "PrincessTower")


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# --------------------------------------------------------------- name mapping


@dataclass(frozen=True, slots=True)
class Ref:
    """One community-sheet name resolved to something the engine can build.

    ``card`` supplies the rarity/level context (every scalable stat in this
    game rides that card's ladder, even for a sub-character it merely spawns).
    ``character`` overrides which entity to resolve when it is not the card's
    primary summon -- Golem's card summons nothing named "Golemite", it is a
    death spawn, but the death spawn still scales on Golem's own rarity.
    ``shield_only`` asks for just the shield component (Guards' shield,
    breakable independently of the body it protects) rather than the whole
    unit. ``tower`` overrides everything and points straight at a Crown Tower.
    """

    card: str = ""
    character: str | None = None
    shield_only: bool = False
    tower: str | None = None


#: Community-sheet name (lower-cased, as it appears in the CSV) -> Ref.
#:
#: Built from the game's own asset filenames (``HighresImageFilename`` on each
#: card, e.g. ``Assassin`` ships art named ``bandit_dl.png``), not from
#: guessing at spelling -- several cards were renamed between their internal
#: dev name and their public one (Assassin/Bandit, AxeMan/Executioner,
#: WitchMother/Mother Witch, DarkWitch/Night Witch, ZapMachine/Sparky,
#: DartBarrell/Flying Machine...) and a name-shaped guess would silently pair
#: the wrong two cards. Sheet names with no confident match are simply absent
#: here rather than fuzzy-matched to the nearest string; see the module
#: docstring in ``reference/hits_to_kill.md`` for what was left out and why.
NAME_MAP: dict[str, Ref] = {
    # -- direct card names -------------------------------------------------
    "knight": Ref("Knight"),
    "musketeer": Ref("Musketeer"),
    "mini pekka": Ref("MiniPekka"),
    "pekka": Ref("Pekka"),
    "wizard": Ref("Wizard"),
    "valkyrie": Ref("Valkyrie"),
    "giant": Ref("Giant"),
    "golem": Ref("Golem"),
    "baby dragon": Ref("BabyDragon"),
    "hog rider": Ref("HogRider"),
    "archers": Ref("Archer"),
    "bomber": Ref("Bomber"),
    "royal giant": Ref("RoyalGiant"),
    "balloon": Ref("Balloon"),
    "mega knight": Ref("MegaKnight"),
    "mega minion": Ref("MegaMinion"),
    "prince": Ref("Prince"),
    "dark prince": Ref("DarkPrince"),
    "bowler": Ref("Bowler"),
    "witch": Ref("Witch"),
    "fireball": Ref("Fireball"),
    "zap": Ref("Zap"),
    "log": Ref("Log"),
    "rocket": Ref("Rocket"),
    "lightning": Ref("Lightning"),
    "arrows": Ref("Arrows"),
    "snowball": Ref("Snowball"),
    "freeze": Ref("Freeze"),
    "mortar": Ref("Mortar"),
    "cannon": Ref("Cannon"),
    "tesla": Ref("Tesla"),
    "bomb tower": Ref("BombTower"),
    "inferno tower": Ref("InfernoTower"),
    "xbow": Ref("Xbow"),
    "furnace": Ref("FirespiritHut"),
    "goblin hut": Ref("GoblinHut"),
    "barb hut": Ref("BarbarianHut"),
    "tombstone": Ref("Tombstone"),
    "goblin cage": Ref("GoblinCage"),
    "goblin drill": Ref("GoblinDrill"),
    "elixir collector": Ref("Elixir Collector"),
    "goblins": Ref("Goblins"),
    "spear goblin": Ref("SpearGoblins"),
    "skeletons": Ref("Skeletons"),
    "bats": Ref("Bats"),
    "minions": Ref("Minions"),
    "ice spirit": Ref("IceSpirits"),
    "fire spirit": Ref("FireSpirits"),
    "electro spirit": Ref("ElectroSpirit"),
    "heal spirit": Ref("Heal"),
    "guards": Ref("SkeletonWarriors"),
    "royal recruits": Ref("RoyalRecruits"),
    "royal hogs": Ref("RoyalHogs"),
    "electro wizard": Ref("ElectroWizard"),
    "electro dragon": Ref("ElectroDragon"),
    "electro giant": Ref("ElectroGiant"),
    "ice wizard": Ref("IceWizard"),
    "inferno dragon": Ref("InfernoDragon"),
    "lava hound": Ref("LavaHound"),
    "giant skeleton": Ref("GiantSkeleton"),
    "goblin giant": Ref("GoblinGiant"),
    "goblin demolisher": Ref("GoblinDemolisher"),
    "goblin machine": Ref("GoblinMachine"),
    "elixir golem": Ref("ElixirGolem"),
    "elite barbarian": Ref("AngryBarbarians"),
    "barbarian": Ref("Barbarians"),
    "firecracker": Ref("Firecracker"),
    "fisherman": Ref("Fisherman"),
    "hunter": Ref("Hunter"),
    "miner": Ref("Miner"),
    "mighty miner": Ref("MightyMiner"),
    "little prince": Ref("LittlePrince"),
    "golden knight": Ref("GoldenKnight"),
    "monk": Ref("Monk"),
    "skeleton king": Ref("SkeletonKing"),
    "boss bandit": Ref("BossBandit"),
    "skeleton dragons": Ref("SkeletonDragons"),
    "ram rider": Ref("RamRider"),
    "battle ram": Ref("BattleRam"),
    "battle healer": Ref("BattleHealer"),
    "cannon cart": Ref("MovingCannon"),
    "lumberjack": Ref("RageBarbarian"),
    "skeleton barrel": Ref("SkeletonBalloon"),
    "sparky": Ref("ZapMachine"),
    "zappies": Ref("MiniSparkys"),
    "royal ghost": Ref("Ghost"),
    "rune giant": Ref("GiantBuffer"),
    "barb barrel": Ref("BarbLog"),
    "royal delivery": Ref("RoyalDelivery"),
    "wall breaker": Ref("Wallbreakers"),
    "three musketeers": Ref("ThreeMusketeers"),
    "queen": Ref("ArcherQueen"),
    "berserker": Ref("Berserker"),
    "phoenix": Ref("Phoenix"),
    # Defender-only: Princess's shot bursts into shrapnel with no single
    # correct "damage per hit" (see ``_MULTI_PROJECTILE_DAMAGE_UNRELIABLE``),
    # so she never gets an attacker role, but her own hitpoints are ordinary.
    "princess": Ref("Princess"),
    # Rascals summons RascalBoy (primary) and 2x RascalGirl (secondary); "rascal
    # boy" is that primary summon, same as the plain "Rascals" card would give.
    "rascal boy": Ref("Rascals"),
    "rascal girl": Ref("Rascals", character="RascalGirl"),
    "sus bush": Ref("SuspiciousBush"),
    "ice golem": Ref("IceGolemite"),
    # -- renamed since dev-name extraction, confirmed via asset filename ----
    "bandit": Ref("Assassin"),
    "executioner": Ref("AxeMan"),
    "dart goblin": Ref("BlowdartGoblin"),
    "flying machine": Ref("DartBarrell"),
    "magic archer": Ref("EliteArcher"),
    "night witch": Ref("DarkWitch"),
    "mother witch": Ref("WitchMother"),
    # -- evolutions (defender-only unless noted) -----------------------------
    "evo barbs": Ref("Barbarians_EV1"),
    "evo archers": Ref("Archer_EV1"),
    "evo musketeer": Ref("Musketeer_EV1"),
    "evo skelly barrel": Ref("SkeletonBalloon_EV1"),
    "evo wizard shield": Ref("Wizard_EV1", shield_only=True),
    "guard shield": Ref("SkeletonWarriors", shield_only=True),
    # -- sub-characters: death spawns and reveal-forms, scaled on the card
    #    that owns them rather than any ladder of their own ------------------
    "golemite": Ref("Golem", character="Golemite"),
    "elixir golemite": Ref("ElixirGolem", character="ElixirGolem2"),
    "elixir blob": Ref("ElixirGolem", character="ElixirGolem4"),
    "lava pup": Ref("LavaHound", character="LavaPups"),
    "bush goblin": Ref("SuspiciousBush", character="BushGoblin"),
    "phoenix egg": Ref("Phoenix", character="PhoenixEgg"),
    "goblinstein doctor": Ref("Goblinstein", character="goblinstein_doctor"),
    # -- towers --------------------------------------------------------------
    "tower princess": Ref(tower="PrincessTower"),
    "king tower": Ref(tower="KingTower"),
}

#: Sheet names deliberately left unmapped, and why. Not consulted by the code
#: -- it exists so the CLI report can explain a gap instead of just showing
#: one, and so the next person to extend this table knows what was already
#: considered and rejected rather than re-litigating it.
KNOWN_UNMAPPED = {
    "clone": "not a card with its own hitpoints -- describes any cloned unit, which the "
    "Clone spell reduces to ~1 HP; every attacker kills one in a single hit, so the "
    "row is trivially true and carries no discriminating information anyway.",
    "most shields": "not a specific unit -- a generic reference to 'most shielded units', "
    "not resolvable to one set of stats.",
    "half demolisher": "situational (mid-fight state), not a stat this pipeline models.",
    "fully healed evo bats": "situational buffed state, not the card's base stats.",
    "fully healed evo witch": "situational buffed state, not the card's base stats.",
    "decoy barrel": "unidentified: no confirmed engine entity, left unmapped rather than guessed.",
    "devoy barrel": "unidentified (likely 'decoy barrel' misspelled): same as above.",
    "cannoneer": "unidentified: no card in this build's asset filenames confirms this name.",
    "dagger duchess": "unidentified: no confirmed engine entity, left unmapped rather than guessed.",
    "royal chef": "unidentified: no confirmed engine entity, left unmapped rather than guessed.",
    "lp guardian": "Little Prince's companion rabbit is not a separately resolvable summon "
    "in this registry (LittlePrince's card summons only LittlePrince itself).",
    "lp ability": "Little Prince's special ability is action-graph driven, not a plain stat.",
    "goblinstein monster": "Goblinstein's transformed 'monster' form is action-graph driven.",
    "goblinstein ability": "Goblinstein's ability is action-graph driven, not a plain stat.",
    "golden knight ability": "Golden Knight's dash is action-graph driven, not a plain stat.",
    "mighty bomb": "Mighty Miner's bomb throw is a secondary ability, not modelled separately.",
    "egiant reflection": "Electro Giant's reflected-damage buff is a defensive mechanic on the "
    "attacker who hits it, not a hits-to-kill number.",
    "three musketeers bayonet": "the reworked Musketeers' bayonet charge is a distinct attack "
    "mode from their ordinary shot, not modelled separately.",
    "goblin machine rocket": "Goblin Machine's rocket is a secondary special attack, not its "
    "ordinary hit.",
    "void 1st": "Void (dev name DarkMagic) deals three different DPS tiers depending on how "
    "many targets it catches (ActionLaserBall in the raw data) -- action-graph driven "
    "and not something the plain-stat pipeline resolves correctly at all; its naive "
    "'damage' field is a decoy value from an unrelated part of the payload.",
    "void 2nd": "see 'void 1st'.",
    "void 3rd": "see 'void 1st'.",
    "evo rg recoil": "Royal Ghost evolution's recoil mechanic, not its ordinary attack.",
    "evo ghost soldier": "a spawned sub-unit of the Royal Ghost evolution, not separately "
    "resolvable in this registry.",
    "evo wall breaker runner": "Wall Breakers evolution's charge-in mechanic, not its detonation.",
    "evo wall breaker death": "Wall Breakers evolution's detonation, a secondary mechanic.",
    "evo tesla shockwave": "Tesla evolution's shockwave, a secondary mechanic on top of its shots.",
    "evo cannon barrage": "Cannon evolution's barrage, a secondary mechanic on top of its shots.",
    "evo exe": "Executioner (dev name AxeMan) evolution's boomerang-return hit is a distinct "
    "second hit, not modelled separately from its ordinary throw.",
    "evo battle ram pushback": "Battle Ram evolution's pushback is a status effect, not damage.",
    "ram rider bola": "Ram Rider's bola stun is a status effect, not a hits-to-kill number.",
    "golemite death": "the burst a dying Golemite leaves is a death-damage field, not its "
    "ordinary attack -- excluded to keep 'golemite' meaning one thing.",
    "loon death": "Balloon's death bomb, a secondary mechanic distinct from its ordinary attack.",
    "giant skelly death": "Giant Skeleton's death bomb, a secondary mechanic.",
    "demolisher death": "Goblin Demolisher's death explosion, a secondary mechanic.",
    "mega knight spawn": "Mega Knight's landing shockwave, a secondary mechanic.",
    "ewiz spawn": "Electro Wizard's on-spawn zap, a secondary mechanic.",
    "ice wiz spawn": "Ice Wizard's on-spawn freeze pulse, a secondary mechanic.",
    "goblin drill spawn": "Goblin Drill's emergence damage, a secondary mechanic.",
    "goblinstein": "only Goblinstein's sub-forms (doctor/ability/monster) are named on the "
    "sheet -- there is no bare row/column for the champion's base form to map.",
    "goblin curse": "the card mixes a plain 'Damage' field with an OnStartingAction that "
    "actually drives its curse effect, the same unreliable-top-level-number pattern "
    "confirmed on Void/DarkMagic -- excluded from the attacker role rather than trusted.",
    "evo bats": "the evolution's Hitpoints field is stored as a relative modifier "
    "(['%', 150], meaning 150% of the base Bat) rather than an absolute number -- this "
    "pipeline has no code path that resolves that encoding, so build_unit_spec fails. A "
    "genuine pre-existing gap in the ingestion pipeline, not something to paper over here.",
    "earthquake": "deals damage_per_second over its duration, not a discrete per-hit amount "
    "-- there is no 'hits to kill' for a continuous-damage spell in this model.",
    "poison": "same as 'earthquake': damage-over-time, no discrete hit to count.",
    "tornado": "pulls units together; whatever damage it deals is damage_per_second, same "
    "as 'earthquake'.",
    "rage": "a buff card -- it deals no damage of its own.",
    "vines": "defined entirely in its ACTION graph (OnStartingAction), not stat fields; "
    "card_stat_summary reports no usable 'damage' for it at all.",
    "golem death": "Golem's death explosion is a secondary DeathDamage field, not his "
    "ordinary (nonexistent -- he is melee-only and untargetable by ranged logic here "
    "the same as any troop) attack; kept out to keep 'golem' meaning one thing.",
    "phoenix death": "Phoenix's revive-from-egg-on-death is a life-cycle mechanic, not a "
    "hits-to-kill number.",
    "cursed hog": "unidentified: reads like Goblin Curse's effect applied to a Hog Rider "
    "(a combo, not a card), not a resolvable single entity.",
    "goblin brawler": "the character exists in the game files (CHARACTER.GoblinBrawler) "
    "but is not summoned by any registered card in this build, so there is no owning "
    "card to scale it against -- left unmapped rather than guessing a rarity/level.",
    "spirit empress": "unidentified: no card or character in this build matches the name.",
    "spirit empress air": "unidentified: no card or character in this build matches the name.",
    "spirit empress ground": "unidentified: no card or character in this build matches the name.",
    "rascals": "ambiguous on its own -- the card summons two different units (RascalBoy, "
    "RascalGirl) with different stats, and the sheet separately names both ('rascal "
    "boy', 'rascal girl') as attacker columns, so a bare 'rascals' row does not pick one.",
}


# ----------------------------------------------------------------- profiles


@dataclass(frozen=True, slots=True)
class DefenseProfile:
    """What a defender can take, at tournament standard."""

    key: str
    hitpoints: int
    shield_hitpoints: int
    flying: bool
    is_tower: bool = False
    is_building: bool = False
    #: Hitbox radius. Ranges are measured surface-to-surface, so turning a
    #: reach advantage into a distance one unit must walk needs both bodies'
    #: radii as well as the range difference.
    collision_radius: int = 0


@dataclass(frozen=True, slots=True)
class AttackProfile:
    """What an attacker can deal, at tournament standard."""

    key: str
    damage: int
    crown_tower_damage_percent: int
    attacks_ground: bool
    attacks_air: bool
    hit_speed_ticks: int = 0
    load_time_ticks: int = 0
    #: Hitbox-to-hitbox reach in subtiles, and how fast this unit closes a
    #: gap (zero for a building, which never closes one). Neither changes a
    #: hit count, which is why the matrix above ignores them -- but together
    #: they decide whether the fight that hit count describes happens on even
    #: terms or not at all. See :mod:`cr_sim.data.engagement`.
    attack_range: int = 0
    speed_per_tick: int = 0
    #: How far this unit can *acquire* a target. Only two units in the build
    #: have a sight range shorter than their reach and neither attacks, so
    #: this never shortens a real attacker -- but it is what decides whether a
    #: unit being shot from outside its sight ever reacts at all.
    sight_range: int = 0
    variable_damage: tuple[int, ...] = ()
    variable_damage_ticks: tuple[int, ...] = ()
    is_spell: bool = False
    #: Dies on its own first hit -- Ice Spirit, Balloon's death bomb, the
    #: Skeleton Barrel. Treating one as a repeating attacker is how a model
    #: concludes an Ice Spirit grinds down a Giant.
    kamikaze: bool = False
    #: Giant, Golem, Hog Rider, Royal Giant, Balloon and their kin cannot
    #: target a troop at all -- not "deals reduced damage to one", genuinely
    #: cannot select one as a target. A pure hp/damage division does not know
    #: that, and will confidently report a hit count for a fight that can
    #: never happen; :func:`compute_hits` excludes it instead.
    target_only_buildings: bool = False


def _spec_profiles(spec: UnitSpec, key: str) -> tuple[DefenseProfile, AttackProfile | None]:
    defense = DefenseProfile(
        key=key,
        hitpoints=spec.hitpoints,
        shield_hitpoints=spec.shield_hitpoints,
        flying=spec.flying,
        is_tower=spec.kind.name == "TOWER",
        is_building=spec.kind.name in ("BUILDING", "TOWER"),
        collision_radius=spec.collision_radius,
    )
    attack = None
    if spec.damage > 0:
        attack = AttackProfile(
            key=key,
            damage=spec.damage,
            crown_tower_damage_percent=spec.crown_tower_damage_percent,
            attacks_ground=spec.attacks_ground,
            attacks_air=spec.attacks_air,
            hit_speed_ticks=spec.hit_speed_ticks,
            load_time_ticks=spec.load_time_ticks,
            attack_range=spec.attack_range,
            speed_per_tick=spec.speed_per_tick,
            sight_range=spec.sight_range,
            variable_damage=spec.variable_damage,
            variable_damage_ticks=spec.variable_damage_ticks,
            target_only_buildings=spec.target_only_buildings,
            kamikaze=spec.kamikaze,
        )
    return defense, attack


#: Extra characters needed beyond each card's primary summon: sheet key ->
#: (owning card, entity name to resolve). Scaled on the owning card's rarity,
#: since Clash Royale has no separate level ladder for a death spawn.
_EXTRA_CHARACTERS: dict[str, tuple[str, str]] = {
    "Golemite": ("Golem", "Golemite"),
    "ElixirGolem2": ("ElixirGolem", "ElixirGolem2"),
    "ElixirGolem4": ("ElixirGolem", "ElixirGolem4"),
    "LavaPups": ("LavaHound", "LavaPups"),
    "BushGoblin": ("SuspiciousBush", "BushGoblin"),
    "PhoenixEgg": ("Phoenix", "PhoenixEgg"),
    "goblinstein_doctor": ("Goblinstein", "goblinstein_doctor"),
    "RascalGirl": ("Rascals", "RascalGirl"),
}

#: Shield-only defender keys -> (owning card or evolution card, is a distinct
#: DefenseProfile of just the shield component). Guards' shield breaks
#: independently of the skeleton behind it; asking "hits to kill Guards" and
#: "hits to break Guards' shield" are two different, both legitimate,
#: questions and the sheet asks both.
_SHIELD_ONLY: dict[str, str] = {
    "SkeletonWarriors#shield": "SkeletonWarriors",
    "Wizard_EV1#shield": "Wizard_EV1",
}

#: Cards whose payload mixes a plain ``damage`` number with an action-graph
#: (``OnStartingAction``/``OnHitAction``) that actually drives the real
#: behaviour -- Void (dev name DarkMagic) deals three different DPS tiers by
#: how many units it catches, and its top-level ``Damage`` field is a decoy
#: from deep in that action graph, not what the card does. Trusting it would
#: be exactly the "confident nonsense" a fuzzy match produces, just from a
#: different source, so these are excluded from the attacker role entirely.
_ACTION_GRAPH_DAMAGE_UNRELIABLE = {"DarkMagic", "GoblinCurse"}

#: Troops whose listed ``damage`` is *per projectile*, not per attack --
#: Hunter fires 10 pellets in one shotgun blast (all ten only connect
#: point-blank; fewer at range), Princess's shot bursts into 5 shrapnel
#: pieces that spread across an area rather than stacking onto one target.
#: There is no single "damage per hit" number that is right for both the
#: close-range and long-range case, so -- same principle as the action-graph
#: exclusion above -- these are left out of the attacker role rather than
#: quietly treated as a single, much-too-low hit. (Arrows has the same flag
#: but is a spell whose three-wave *total* is already well-established
#: elsewhere in this codebase -- see ``test_interactions.py`` -- so it is
#: handled correctly by the spell path below instead of excluded.)
_MULTI_PROJECTILE_DAMAGE_UNRELIABLE = {"Hunter", "Princess"}


def build_profiles(
    data: LogicData,
    levels: LevelTable,
    registry: CardRegistry,
    *,
    display_level: int = TOURNAMENT_DISPLAY_LEVEL,
    clock: TickClock | None = None,
) -> tuple[dict[str, DefenseProfile], dict[str, AttackProfile], dict[str, str]]:
    """Resolve every standard card (plus the sub-characters and towers the
    sheet separately names) into engine-ready combat profiles.

    Returns ``(defenses, attacks, labels)``. A card that fails to resolve (a
    handful of spells with no summon, or a malformed entity) is silently
    absent from both rather than raising -- the gate is about what *can* be
    checked, and a missing entry is visible as "excluded", not a crash.
    """
    clock = clock or TickClock()
    defenses: dict[str, DefenseProfile] = {}
    attacks: dict[str, AttackProfile] = {}
    labels: dict[str, str] = {}

    def _resolve(character: str, rarity: str) -> UnitSpec | None:
        scale = levels.get(rarity)
        level = scale.internal_level(display_level)
        try:
            return build_unit_spec(data, levels, character, level=level, rarity=rarity, clock=clock)
        except SpecError:
            return None

    # -- standard cards + explicitly-needed evolutions -----------------------
    evolutions_needed = (
        "Barbarians_EV1", "Archer_EV1", "Musketeer_EV1",
        "SkeletonBalloon_EV1", "Wizard_EV1",
    )
    cards = list(registry.standard()) + [
        registry[name] for name in evolutions_needed if registry.get(name) is not None
    ]
    for card in cards:
        labels[card.name] = card.name
        summons = card.summons()
        if summons:
            character_name, _count = summons[0]
            spec = _resolve(character_name, card.rarity)
            if spec is not None:
                defense, attack = _spec_profiles(spec, card.name)
                defenses[card.name] = defense
                excluded = card.name in _ACTION_GRAPH_DAMAGE_UNRELIABLE or card.name in _MULTI_PROJECTILE_DAMAGE_UNRELIABLE
                if attack is not None and not excluded:
                    attacks[card.name] = attack
            continue
        # A pure spell: no UnitSpec, damage lives on its payload.
        if card.name in _ACTION_GRAPH_DAMAGE_UNRELIABLE:
            continue
        summary = card_stat_summary(data, levels, card, display_level=display_level)
        damage = summary.get("damage")
        if not isinstance(damage, int) or damage <= 0:
            continue
        waves = summary.get("projectile_waves")
        total = damage * waves if isinstance(waves, int) and waves > 1 else damage
        attacks[card.name] = AttackProfile(
            key=card.name,
            damage=total,
            crown_tower_damage_percent=summary.get("crown_tower_damage_percent", 0),
            attacks_ground=True,
            attacks_air=True,
            is_spell=True,
        )

    # -- sub-characters --------------------------------------------------
    for key, (owner_card, character_name) in _EXTRA_CHARACTERS.items():
        card = registry.get(owner_card)
        if card is None:
            continue
        spec = _resolve(character_name, card.rarity)
        if spec is None:
            continue
        defense, attack = _spec_profiles(spec, key)
        defenses[key] = defense
        if attack is not None:
            attacks[key] = attack
        labels[key] = f"{key} ({owner_card})"

    # -- shield-only defenders --------------------------------------------
    for key, owner_card in _SHIELD_ONLY.items():
        card = registry.get(owner_card)
        if card is None or not card.summons():
            continue
        character_name = card.summons()[0][0]
        spec = _resolve(character_name, card.rarity)
        if spec is None or spec.shield_hitpoints <= 0:
            continue
        defenses[key] = DefenseProfile(
            key=key, hitpoints=spec.shield_hitpoints, shield_hitpoints=0, flying=spec.flying,
        )
        labels[key] = f"{owner_card} shield"

    # -- towers, on their own progression, not the card ladder ------------
    scales = build_tower_scales(data.globals_map())
    for tower_name in TOWER_KEYS:
        try:
            spec = build_tower_spec(
                data, tower_name, scales[tower_class_for(tower_name)], level=display_level, clock=clock,
            )
        except SpecError:
            continue
        defense, attack = _spec_profiles(spec, tower_name)
        defenses[tower_name] = defense
        if attack is not None:
            attacks[tower_name] = attack
        labels[tower_name] = tower_name

    return defenses, attacks, labels


# ------------------------------------------------------------ hit arithmetic


@dataclass(frozen=True, slots=True)
class HitResult:
    hits: int
    ticks: int | None = None


def _ramped_hits(
    hitpoints: int, stages: tuple[int, ...], stage_ticks: tuple[int, ...], hit_speed_ticks: int,
) -> int | None:
    """Hits to kill against a damage ramp (Inferno Tower/Dragon).

    ``stage_ticks`` are each stage's *duration*, not a cumulative threshold
    (that is how :func:`cr_sim.engine.specs.build_unit_spec` stores them), so
    the cumulative switchover points are built here before walking hit by hit.
    """
    if hit_speed_ticks <= 0 or not stages:
        return None
    thresholds: list[int] = []
    total = 0
    for t in stage_ticks:
        total += t
        thresholds.append(total)
    elapsed = 0
    remaining = hitpoints
    hits = 0
    while remaining > 0:
        stage_index = sum(1 for t in thresholds if elapsed >= t)
        stage_index = min(stage_index, len(stages) - 1)
        dmg = stages[stage_index]
        if dmg <= 0:
            return None
        remaining -= dmg
        hits += 1
        elapsed += hit_speed_ticks
        if hits > 5000:
            return None
    return hits


def compute_hits(defense: DefenseProfile, attack: AttackProfile) -> HitResult | None:
    """The engine's real answer: shield first, crown-tower percent, ramps.

    Returns ``None`` when the pair does not apply at all -- a ground-only
    attacker against a flyer, a building-only attacker against a troop it can
    never select as a target, or an attacker with no usable damage -- rather
    than forcing a number for something the engine cannot express.
    """
    if defense.flying and not attack.attacks_air:
        return None
    if not defense.flying and not attack.attacks_ground:
        return None
    if attack.target_only_buildings and not defense.is_building:
        return None

    pct = attack.crown_tower_damage_percent if defense.is_tower else 0

    def reduce(x: int) -> int:
        return x * (100 + pct) // 100 if pct else x

    dmg = reduce(attack.damage)
    if dmg <= 0:
        return None

    shield_hits = _ceil_div(defense.shield_hitpoints, dmg) if defense.shield_hitpoints > 0 else 0

    if attack.variable_damage:
        stages = tuple(reduce(s) for s in attack.variable_damage)
        body_hits = _ramped_hits(defense.hitpoints, stages, attack.variable_damage_ticks, attack.hit_speed_ticks)
        if body_hits is None:
            return None
    else:
        body_hits = _ceil_div(defense.hitpoints, dmg)

    hits = shield_hits + body_hits
    ticks = None
    if attack.hit_speed_ticks or attack.load_time_ticks:
        ticks = attack.load_time_ticks + max(0, hits - 1) * attack.hit_speed_ticks
    return HitResult(hits=hits, ticks=ticks)


def compute_hits_naive(defense: DefenseProfile, attack: AttackProfile) -> int | None:
    """Bare ``ceil(hitpoints / damage)`` -- no shield, no crown-tower percent,
    no ramp. What a spreadsheet author would get from the two headline
    numbers alone.

    This exists to tell apart *why* the engine disagrees with a source: if
    this naive number matches where the adjusted one does not, the raw stats
    line up and the adjustment logic (shield/crown-tower/ramp handling) is
    the thing worth doubting. See :func:`categorise_sheet_comparison`.
    """
    if defense.flying and not attack.attacks_air:
        return None
    if not defense.flying and not attack.attacks_ground:
        return None
    if attack.target_only_buildings and not defense.is_building:
        return None
    dmg = attack.variable_damage[0] if attack.variable_damage else attack.damage
    if dmg <= 0:
        return None
    return _ceil_div(defense.hitpoints, dmg)


def predicted_winner(
    computed: Mapping[tuple[str, str], HitResult], defender: str, attacker: str,
) -> str | None:
    """Arithmetic's guess at a 1v1 duel: whoever's time-to-kill is shorter.

    Deliberately ignorant of everything only a simulation can see -- deploy
    time, the distance still to close, retargeting, splash catching more than
    one body. That is the point: comparing this guess against an actual
    simulated duel (:func:`simulate_duel`) tells apart the pairs where those
    things do not matter (arithmetic and simulation agree) from the ones
    where they decide the fight (they do not), and the second group is what a
    pure stat gate can never surface.

    Returns ``"attacker"``, ``"defender"``, ``"draw"`` (equal time-to-kill),
    or ``None`` when neither side's timing is computable (a spell, a building
    that cannot be attacked back, ...).
    """
    to_kill_defender = computed.get((defender, attacker))
    to_kill_attacker = computed.get((attacker, defender))
    if to_kill_defender is None:
        return "defender" if to_kill_attacker is not None else None
    if to_kill_attacker is None:
        return "attacker"
    if to_kill_defender.ticks is None or to_kill_attacker.ticks is None:
        return None
    if to_kill_defender.ticks < to_kill_attacker.ticks:
        return "attacker"
    if to_kill_attacker.ticks < to_kill_defender.ticks:
        return "defender"
    return "draw"


def compute_matrix(
    defenses: Mapping[str, DefenseProfile], attacks: Mapping[str, AttackProfile],
) -> dict[tuple[str, str], HitResult]:
    """Every (defender, attacker) pair's computed hit count."""
    out: dict[tuple[str, str], HitResult] = {}
    for def_key, defense in defenses.items():
        for atk_key, attack in attacks.items():
            result = compute_hits(defense, attack)
            if result is not None:
                out[(def_key, atk_key)] = result
    return out


# -------------------------------------------------------------- simulation


@dataclass(frozen=True, slots=True)
class SimResult:
    defender: str
    attacker: str
    winner: str  # "attacker" | "defender" | "draw" | "timeout"
    ticks: int
    hits_landed: int  # hits the attacker landed on the defender


#: The RL agent's own training deck (see ``cr_sim/train/run.py:DEFAULT_DECK``)
#: plus a curated set of common, recognisable ladder troops -- enough to make
#: the simulated cross-check meaningful without paying for the full pool.
#: Buildings and spells are excluded: a 1v1 duel needs two units that can walk
#: toward each other. See the CLI report for what this leaves uncovered.
SIM_CARDS: tuple[str, ...] = (
    "Knight", "Musketeer", "Skeletons", "IceSpirits", "Goblins",
    "MiniPekka", "Pekka", "Wizard", "Valkyrie", "Giant", "Golem",
    "BabyDragon", "HogRider", "Barbarians", "MegaMinion", "Prince",
    "DarkPrince", "Bowler", "Witch", "Archer", "Bomber", "RoyalGiant",
    "Balloon", "MegaKnight", "GiantSkeleton",
)


def simulate_duel(
    data: LogicData,
    levels: LevelTable,
    registry: CardRegistry,
    defender_name: str,
    attacker_name: str,
    *,
    display_level: int = TOURNAMENT_DISPLAY_LEVEL,
    gap: float = 2.0,
    limit: int = 6000,
) -> SimResult | None:
    """Run one 1v1 duel, towers removed, and report who won and how.

    Mirrors ``tests/test_combat.py``'s ``duel`` helper: both units spawn
    ``gap`` tiles apart with nothing else on the field, so the result isolates
    the two cards rather than anything about deployment or lanes.
    """
    from ..engine.battle import Battle, BattleConfig
    from ..engine.entity import Entity, EntityKind, Team
    from ..engine.fixed import tiles

    def_card = registry.get(defender_name)
    atk_card = registry.get(attacker_name)
    if def_card is None or atk_card is None or not def_card.summons() or not atk_card.summons():
        return None

    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=1, blue_deck=(defender_name,), red_deck=(defender_name,), level=display_level),
    )
    battle.entities = [e for e in battle.entities if e.kind is not EntityKind.TOWER]
    battle._towers = {Team.BLUE: [], Team.RED: []}

    def spawn(card, team, y):
        character_name = card.summons()[0][0]
        scale = levels.get(card.rarity)
        spec = build_unit_spec(
            data, levels, character_name,
            level=scale.internal_level(display_level), rarity=card.rarity, clock=battle.clock,
        )
        entity = Entity(
            kind=spec.kind, team=team, x=tiles(9), y=tiles(y),
            hitpoints=spec.hitpoints, spec=spec,
            collision_radius=spec.collision_radius, mass=spec.mass,
            flying=spec.flying, shield=spec.shield_hitpoints,
        )
        battle._register(entity)
        return entity

    try:
        blue = spawn(def_card, Team.BLUE, 10.0)
        red = spawn(atk_card, Team.RED, 10.0 + gap)
    except SpecError:
        return None

    for _ in range(limit):
        battle.step()
        if blue.dead and red.dead:
            break
        # A kamikaze unit (Ice Spirit, Wall Breakers, Balloon's bomb) is
        # consumed by its own attack the instant it fires -- see
        # engine/battle.py's "the attack *is* the death" -- which can kill it
        # several ticks before the projectile it just launched actually
        # lands. Stopping on its death alone would end the duel before that
        # projectile's damage is applied and misreport a kill that is already
        # in flight as a stalemate. Wait for no projectile/area-effect left
        # in play once someone is down, not just for one side's death.
        if (blue.dead or red.dead) and not any(
            e.kind in (EntityKind.PROJECTILE, EntityKind.AREA_EFFECT) for e in battle.entities
        ):
            break

    hits_landed = sum(1 for e in battle.damage_log if e.target_id == blue.id)
    if blue.dead and red.dead:
        winner = "draw"
    elif blue.dead:
        winner = "attacker"
    elif red.dead:
        winner = "defender"
    else:
        winner = "timeout"
    return SimResult(
        defender=defender_name, attacker=attacker_name, winner=winner,
        ticks=battle.tick, hits_landed=hits_landed,
    )


def simulate_matrix(
    data: LogicData,
    levels: LevelTable,
    registry: CardRegistry,
    card_names: tuple[str, ...] = SIM_CARDS,
    *,
    display_level: int = TOURNAMENT_DISPLAY_LEVEL,
) -> dict[tuple[str, str], SimResult]:
    """Every ordered pair in ``card_names``, simulated as a duel."""
    out: dict[tuple[str, str], SimResult] = {}
    for defender in card_names:
        for attacker in card_names:
            if defender == attacker:
                continue
            result = simulate_duel(data, levels, registry, defender, attacker, display_level=display_level)
            if result is not None:
                out[(defender, attacker)] = result
    return out


# --------------------------------------------------------- the stale sheet


@dataclass(frozen=True, slots=True)
class SheetRow:
    defender_csv: str
    attacker_csv: str
    hits: int


def load_sheet(path: str | Path = DEFAULT_SHEET) -> list[SheetRow]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header = [h.strip().lower() for h in rows[0][1:]]
    out: list[SheetRow] = []
    for row in rows[1:]:
        defender = row[0].strip().lower()
        for attacker, cell in zip(header, row[1:]):
            cell = cell.strip()
            if not cell:
                continue
            try:
                hits = int(cell)
            except ValueError:
                continue
            out.append(SheetRow(defender_csv=defender, attacker_csv=attacker, hits=hits))
    return out


@dataclass(frozen=True, slots=True)
class SheetComparison:
    defender_csv: str
    attacker_csv: str
    sheet_hits: int
    engine_hits: int
    naive_hits: int | None
    hitpoints: int
    shield_hitpoints: int
    damage: int
    crown_tower_damage_percent: int
    category: str  # "agree" | "explained" | "defect"

    @property
    def delta(self) -> int:
        return abs(self.engine_hits - self.sheet_hits)


def categorise_sheet_comparison(
    sheet: list[SheetRow], defenses: Mapping[str, DefenseProfile], attacks: Mapping[str, AttackProfile],
) -> tuple[list[SheetComparison], list[SheetRow], list[SheetRow]]:
    """Compare the stale sheet against the engine's current stats, three ways.

    The sheet is roughly a year old (see ``reference/hits_to_kill.md``), and
    Clash Royale rebalances constantly, so "the sheet disagrees" is
    ambiguous by itself. A cell's hit count implies a hitpoints/damage
    relationship, and that lets the two readings be told apart:

    * **agree** -- the engine's current stats reproduce the sheet's number.
      Independent corroboration.
    * **explained** -- they disagree, and the *bare* arithmetic (ignoring
      shield/crown-tower/ramp handling) does not match the sheet either. The
      raw hitpoints or damage simply differ from what they were when the
      sheet was written -- a balance change, not a defect.
    * **defect** -- they disagree, but the bare arithmetic *does* match the
      sheet. The raw numbers line up; it is specifically the engine's
      shield/crown-tower/ramp adjustment logic that produces a different
      answer. That is the one category worth treating as a bug.

    Returns ``(comparisons, unmapped, not_applicable)`` where ``unmapped``
    are sheet rows naming something outside :data:`NAME_MAP` and
    ``not_applicable`` are mapped rows the engine has no opinion on (air/ground
    mismatch, an excluded attacker, a missing profile).
    """
    comparisons: list[SheetComparison] = []
    unmapped: list[SheetRow] = []
    not_applicable: list[SheetRow] = []

    for row in sheet:
        def_ref = NAME_MAP.get(row.defender_csv)
        atk_ref = NAME_MAP.get(row.attacker_csv)
        if def_ref is None or atk_ref is None:
            unmapped.append(row)
            continue

        def_key = _defense_key(def_ref)
        atk_key = _attack_key(atk_ref)
        defense = defenses.get(def_key)
        attack = attacks.get(atk_key)
        if defense is None or attack is None:
            not_applicable.append(row)
            continue

        adjusted = compute_hits(defense, attack)
        if adjusted is None:
            not_applicable.append(row)
            continue
        naive = compute_hits_naive(defense, attack)

        if adjusted.hits == row.hits:
            category = "agree"
        elif naive is not None and naive == row.hits:
            category = "defect"
        else:
            category = "explained"

        comparisons.append(
            SheetComparison(
                defender_csv=row.defender_csv,
                attacker_csv=row.attacker_csv,
                sheet_hits=row.hits,
                engine_hits=adjusted.hits,
                naive_hits=naive,
                hitpoints=defense.hitpoints,
                shield_hitpoints=defense.shield_hitpoints,
                damage=attack.damage,
                crown_tower_damage_percent=attack.crown_tower_damage_percent if defense.is_tower else 0,
                category=category,
            )
        )
    return comparisons, unmapped, not_applicable


def _extra_character_key(ref: Ref) -> str | None:
    """The ``_EXTRA_CHARACTERS`` key a sub-character override resolves to."""
    for key, (owner, char) in _EXTRA_CHARACTERS.items():
        if owner == ref.card and char == ref.character:
            return key
    return None


def _defense_key(ref: Ref) -> str:
    if ref.tower:
        return ref.tower
    if ref.shield_only:
        return f"{ref.card}#shield"
    if ref.character:
        return _extra_character_key(ref) or ref.card
    return ref.card


def _attack_key(ref: Ref) -> str:
    if ref.tower:
        return ref.tower
    if ref.shield_only:
        return ""  # shields never attack
    if ref.character:
        return _extra_character_key(ref) or ""
    return ref.card


# --------------------------------------------------------------- CSV output


def write_generated_csv(
    path: str | Path,
    defenses: Mapping[str, DefenseProfile],
    attacks: Mapping[str, AttackProfile],
    computed: Mapping[tuple[str, str], HitResult],
    *,
    simulated: Mapping[tuple[str, str], SimResult] | None = None,
    build: str = "unknown",
    generated_at: str = "",
) -> None:
    """Write the computed (and, where available, simulated) matrix.

    Same orientation as ``reference/hits_to_kill.csv`` -- row is the unit
    being killed, column is the unit doing the killing -- so the two can be
    diffed directly. Regenerated output, not hand-maintained: the same split
    ``card_stats.json`` has against ``anchors.json``.
    """
    path = Path(path)
    defender_keys = sorted(defenses)
    attacker_keys = sorted(attacks)
    simulated = simulated or {}

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"# generated by cr_sim.data.interactions -- build {build}, {generated_at}\n")
        f.write(
            "# computed = ceil(effective_hitpoints / damage), honouring shields, crown-tower "
            "percent, air/ground and damage ramps. 'sim:hits/win' cells (where present) come "
            "from an actual Battle duel instead of arithmetic. Regenerate; do not hand-edit.\n"
        )
        writer = csv.writer(f)
        writer.writerow(["card", *attacker_keys])
        for def_key in defender_keys:
            row = [def_key]
            for atk_key in attacker_keys:
                result = computed.get((def_key, atk_key))
                cell = str(result.hits) if result else ""
                sim = simulated.get((def_key, atk_key))
                if sim is not None:
                    cell = f"{cell}|sim:{sim.hits_landed}/{sim.winner}"
                row.append(cell)
            writer.writerow(row)
