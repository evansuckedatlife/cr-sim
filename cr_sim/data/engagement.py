"""Who actually wins, once reach and the Princess Tower are in the room.

:mod:`cr_sim.data.interactions` answers one question exactly -- how many hits
does A need to kill B -- and then guesses at duels by comparing time-to-kill.
Its own ``predicted_winner`` docstring is honest that this guess is
"deliberately ignorant" of everything a simulation would see. Two of the
things it ignores are not details.

**Reach.** Ranges here are hitbox-to-hitbox: ``gap_between`` subtracts both
collision radii before comparing against ``attack_range``, so two units'
ranges are directly comparable, and the difference between them is a real
distance the shorter-ranged one has to walk while being shot. A Musketeer
out-reaches a Knight by several tiles and the Knight closes at one tile per
second, so the Musketeer lands free hits before the Knight can answer.
Comparing time-to-kill scores that fight as though both sides start swinging
together, which is not a rounding error -- it is the entire reason ranged
troops exist.

**The Princess Tower.** Almost no defensive fight happens alone. The tower is
firing throughout, so the hits *your* troop needs is not
``ceil(hitpoints / damage)`` but however many it still needs once the tower
has taken its share. That is the difference between a card that trades and a
card that trades up, and no 1v1 matrix can show it.

Neither effect changes a hit count, so neither belongs in the matrix next
door. Both change who wins, which is what a hit count is read *for*.

What this still assumes, stated so the numbers are read for what they are:
the two units engage head-on and the shorter-ranged one walks straight in --
no pathing around a third body, no retargeting, no splash catching a second
unit, and the tower already has its target in range at tick zero. Those
assumptions are what make this arithmetic rather than a simulation.
``simulate_duel`` in :mod:`cr_sim.data.interactions` is the check on them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

from .interactions import AttackProfile, DefenseProfile

__all__ = [
    "Engagement", "TowerAssist", "can_hit", "free_hits", "ticks_to_close",
    "opening_gap", "engagement_delays",
    "resolve_duel", "hits_with_tower", "duel_matrix", "tower_matrix",
    "write_duel_csv", "write_tower_csv",
]

#: A fight unresolved at five minutes has not resolved -- real matches are
#: three. The cap only ever catches pairs that cannot hurt each other fast
#: enough to matter, and those are reported as stalemates rather than
#: silently awarded to whoever happened to be ahead.
MAX_TICKS = 60 * 300

#: Hits after which a duel is abandoned, so a pathological profile turns into
#: a reported stalemate instead of a hang.
MAX_HITS = 4000


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def can_hit(defense: DefenseProfile, attack: AttackProfile) -> bool:
    """Whether this attacker can select this defender at all.

    Not "deals reduced damage to" -- genuinely cannot target. A ground-only
    attacker has no answer to a flyer, and a building-only attacker (Giant,
    Hog Rider, Balloon) cannot choose a troop even while that troop kills it.
    """
    if defense.flying and not attack.attacks_air:
        return False
    if not defense.flying and not attack.attacks_ground:
        return False
    if attack.target_only_buildings and not defense.is_building:
        return False
    return attack.damage > 0 or bool(attack.variable_damage)


@dataclass(slots=True)
class _Body:
    """Hitpoints behind a shield, with the engine's overflow rule.

    A shield soaks whole hits and the excess is discarded rather than carried
    into the body -- see ``Entity.apply_damage``. That is why a big hit into a
    small shield is wasted, and why modelling a shield as extra hitpoints
    would quietly overstate every heavy hitter on the board.
    """

    hitpoints: int
    shield: int = 0

    def take(self, damage: int) -> None:
        if damage <= 0:
            return
        if self.shield > 0:
            self.shield = max(0, self.shield - damage)
            return
        self.hitpoints -= damage

    @property
    def dead(self) -> bool:
        return self.hitpoints <= 0


def _scaled(damage: int, percent: int) -> int:
    return damage * (100 + percent) // 100 if percent else damage


def _damage_at(attack: AttackProfile, elapsed: int, percent: int) -> int:
    """Damage of a single hit, honouring an Inferno-style ramp.

    ``variable_damage_ticks`` are stage *durations* rather than cumulative
    thresholds -- that is how ``build_unit_spec`` stores them -- so the
    running totals are rebuilt here. ``elapsed`` is measured from the start of
    this engagement, because a ramp resets when the target changes.
    """
    stages = attack.variable_damage
    if not stages:
        return _scaled(attack.damage, percent)
    total = 0
    index = 0
    for duration in attack.variable_damage_ticks:
        total += duration
        if elapsed >= total:
            index += 1
        else:
            break
    return _scaled(stages[min(index, len(stages) - 1)], percent)


def _hit_times(attack: AttackProfile, start: int) -> Iterator[int]:
    """When this attacker's hits land, given it engages at ``start``.

    The first lands after ``LoadTime`` -- the windup before a first swing --
    and the rest on the ``HitSpeed`` cycle. A profile with no hit speed (a
    spell) lands exactly one.
    """
    tick = start + attack.load_time_ticks
    # A kamikaze unit destroys itself delivering its one hit, so a hit speed
    # is meaningless for it -- and reading one anyway is what makes naive
    # arithmetic believe an Ice Spirit can grind down a Giant.
    if attack.kamikaze or attack.hit_speed_ticks <= 0:
        yield tick
        return
    for _ in range(MAX_HITS):
        yield tick
        tick += attack.hit_speed_ticks


def opening_gap(a: AttackProfile, b: AttackProfile) -> int:
    """The default distance a fight starts from: the longer unit's reach.

    A lane engagement opens the moment the longer-ranged unit can fire, so
    that is the natural zero point. Callers with a different scenario -- a
    unit dropped on top of another, or a harness that spawns them a fixed
    distance apart -- pass their own ``engage_gap`` instead.
    """
    return max(a.attack_range, b.attack_range)


def engagement_delays(
    a: AttackProfile, b: AttackProfile, engage_gap: int,
) -> tuple[int | None, int | None]:
    """When each side starts firing, given they begin ``engage_gap`` apart.

    Two phases, because both units walk until one of them has something to
    shoot at. First they approach together at their combined speed until the
    longer-ranged one is in range and stops; then only the shorter-ranged one
    keeps closing. Collapsing that into a single closing speed is wrong in
    both directions -- it under-credits reach when the fight starts far out
    and over-credits it when the fight starts close.

    ``None`` for a side means it never fires: out-reached and unable to move.
    """
    reaches = {"a": a.attack_range, "b": b.attack_range}
    speeds = {"a": a.speed_per_tick, "b": b.speed_per_tick}
    longer = "a" if reaches["a"] >= reaches["b"] else "b"
    shorter = "b" if longer == "a" else "a"

    elapsed = 0
    gap = engage_gap
    if gap > reaches[longer]:
        closing = speeds["a"] + speeds["b"]
        if closing <= 0:
            return None, None  # neither can close the distance to anything
        elapsed = _ceil_div(gap - reaches[longer], closing)
        gap = reaches[longer]

    delays = {longer: elapsed}
    if gap <= reaches[shorter]:
        delays[shorter] = elapsed
    elif speeds[shorter] <= 0:
        delays[shorter] = None
    else:
        delays[shorter] = elapsed + _ceil_div(gap - reaches[shorter], speeds[shorter])
    return delays["a"], delays["b"]


def ticks_to_close(
    shooter: AttackProfile, mover: AttackProfile, engage_gap: int | None = None,
) -> int | None:
    """How long ``mover`` spends walking before it can answer ``shooter``.

    Zero means it is not out-reached at all. ``None`` means never: it is
    out-reached *and* cannot move, so it is a building being shot from
    outside its own range and the fight is already over.
    """
    gap = opening_gap(shooter, mover) if engage_gap is None else engage_gap
    shooter_delay, mover_delay = engagement_delays(shooter, mover, gap)
    if mover_delay is None:
        return None
    return max(0, mover_delay - (shooter_delay or 0))


def free_hits(
    shooter: AttackProfile, mover: AttackProfile, engage_gap: int | None = None,
) -> int | None:
    """Hits the longer-ranged unit lands before the other can answer.

    ``None`` is unlimited -- the mover can never reach it.
    """
    window = ticks_to_close(shooter, mover, engage_gap)
    if window is None:
        return None
    if window <= 0:
        return 0
    count = 0
    for tick in _hit_times(shooter, 0):
        if tick >= window:
            break
        count += 1
    return count


@dataclass(frozen=True, slots=True)
class Engagement:
    """One head-on fight, resolved hit by hit."""

    #: ``"a"``, ``"b"``, ``"both"`` (they kill each other on the same tick),
    #: ``"stalemate"`` (neither dies inside :data:`MAX_TICKS`), or
    #: ``"neither"`` (neither can target the other at all).
    winner: str
    a_hits: int
    b_hits: int
    ticks: int
    #: Hits the winner landed before the loser could answer -- the number
    #: that makes a ranged troop worth its elixir.
    head_start: int
    note: str = ""

    @property
    def clean(self) -> bool:
        """Won without ever being hit back."""
        return ((self.winner == "a" and self.b_hits == 0)
                or (self.winner == "b" and self.a_hits == 0))


def resolve_duel(
    a_defense: DefenseProfile, a_attack: AttackProfile,
    b_defense: DefenseProfile, b_attack: AttackProfile,
    *, engage_gap: int | None = None,
) -> Engagement:
    """Fight A against B head-on, with reach deciding who shoots first.

    ``engage_gap`` is the hitbox-to-hitbox distance the fight opens at, and
    defaults to :func:`opening_gap` -- the longer-ranged unit firing the
    moment it can. Pass a smaller one to model a unit dropped on top of
    another, which is the whole point of placing a Mini P.E.K.K.A *on* a
    Musketeer rather than in front of her.
    """
    a_can = can_hit(b_defense, a_attack)
    b_can = can_hit(a_defense, b_attack)
    if not a_can and not b_can:
        return Engagement("neither", 0, 0, 0, 0, "neither can target the other")

    gap = opening_gap(a_attack, b_attack) if engage_gap is None else engage_gap
    a_start, b_start = engagement_delays(a_attack, b_attack, gap)
    # A side that cannot target the other, or that is out-reached and cannot
    # move to fix it, simply never fires. Both are the same thing to the
    # timeline below, and running them through it instead of short-circuiting
    # is what keeps the hit counts real: an early return would report a
    # Musketeer beating a Balloon in zero hits.
    if not a_can:
        a_start = None
    if not b_can:
        b_start = None
    if a_start is None and b_start is None:
        return Engagement("stalemate", 0, 0, 0, 0,
                          "neither side can bring its weapon to bear")

    notes: list[str] = []
    if not a_can:
        notes.append("a cannot target b")
    elif a_start is None:
        notes.append("a is out-reached and cannot close")
    if not b_can:
        notes.append("b cannot target a")
    elif b_start is None:
        notes.append("b is out-reached and cannot close")

    a_body = _Body(a_defense.hitpoints, a_defense.shield_hitpoints)
    b_body = _Body(b_defense.hitpoints, b_defense.shield_hitpoints)
    a_pct = a_attack.crown_tower_damage_percent if b_defense.is_tower else 0
    b_pct = b_attack.crown_tower_damage_percent if a_defense.is_tower else 0

    a_times = _hit_times(a_attack, a_start) if a_start is not None else iter(())
    b_times = _hit_times(b_attack, b_start) if b_start is not None else iter(())
    a_next, b_next = next(a_times, None), next(b_times, None)
    a_hits = b_hits = a_head = b_head = 0

    def finish(winner: str, tick: int) -> Engagement:
        # The head start belongs to whoever won it, so it is read off the
        # winner rather than always off A.
        head = a_head if winner == "a" else b_head if winner == "b" else 0
        return Engagement(winner, a_hits, b_hits, tick, head, "; ".join(notes))

    while a_next is not None or b_next is not None:
        tick = min(t for t in (a_next, b_next) if t is not None)
        if tick > MAX_TICKS:
            break
        # Both sides firing on this tick land together, so a mutual kill is a
        # real outcome rather than an artefact of iteration order.
        if a_next == tick:
            b_body.take(_damage_at(a_attack, tick - a_start, a_pct))
            a_hits += 1
            if b_hits == 0:
                a_head += 1
            a_next = next(a_times, None)
        if b_next == tick:
            a_body.take(_damage_at(b_attack, tick - b_start, b_pct))
            b_hits += 1
            if a_hits == 0:
                b_head += 1
            b_next = next(b_times, None)
        if a_body.dead and b_body.dead:
            notes.append("they kill each other on the same tick")
            return finish("both", tick)
        if b_body.dead:
            return finish("a", tick)
        if a_body.dead:
            return finish("b", tick)

    notes.append("neither dies inside five minutes")
    return Engagement("stalemate", a_hits, b_hits, MAX_TICKS, 0, "; ".join(notes))


@dataclass(frozen=True, slots=True)
class TowerAssist:
    """What a Princess Tower is worth against one defender."""

    #: Hits the troop needs alone. ``None`` if it cannot target at all.
    alone: int | None
    #: Hits the troop still needs while the tower fires too.
    with_tower: int | None
    #: Hits the tower needs on its own.
    tower_alone: int | None
    #: Tower hits spent alongside ``with_tower``. Without this the cell cannot
    #: distinguish "Fireball and one tower hit" from "Zap and eight of them",
    #: which are the same number in the troop column and nothing alike at the
    #: table: the second needs the target to stand in range for six seconds.
    tower_hits: int | None = None

    @property
    def saved(self) -> int | None:
        """Hits the tower took off the troop's bill."""
        if self.alone is None or self.with_tower is None:
            return None
        return self.alone - self.with_tower


def hits_with_tower(
    defense: DefenseProfile, attack: AttackProfile, tower: AttackProfile,
    *, tower_delay: int = 0,
) -> TowerAssist:
    """How many hits the troop still needs with a Princess Tower helping.

    Both are assumed already engaged, which is the defensive case worth
    asking about: something is walking at your tower, the tower is shooting
    it, and the question is what you have to spend to finish the job.
    """
    troop_can = can_hit(defense, attack)
    tower_can = can_hit(defense, tower)

    def solo(profile: AttackProfile) -> int | None:
        """Hits this attacker needs on its own.

        Counted arithmetically rather than off the hit timeline, because "how
        many hits does it take" is a different question from "does this unit
        live long enough to land them". Reading it off the timeline made every
        single-shot profile -- every spell, every kamikaze unit -- report
        ``None`` the moment one cast was not enough, which quietly deleted
        Fireball-needs-two-on-a-Musketeer, and with it the most-quoted tower
        interaction in the game.
        """
        if not can_hit(defense, profile):
            return None
        body = _Body(defense.hitpoints, defense.shield_hitpoints)
        step = max(1, profile.hit_speed_ticks)
        hits = 0
        while not body.dead and hits < MAX_HITS:
            body.take(_damage_at(profile, hits * step, 0))
            hits += 1
        return hits if body.dead else None

    alone = solo(attack) if troop_can else None
    tower_alone = solo(tower) if tower_can else None
    if not troop_can:
        return TowerAssist(None, None, tower_alone, None)
    if not tower_can:
        return TowerAssist(alone, alone, None, 0)

    body = _Body(defense.hitpoints, defense.shield_hitpoints)
    troop_times = _hit_times(attack, 0)
    tower_times = _hit_times(tower, tower_delay)
    troop_next, tower_next = next(troop_times, None), next(tower_times, None)
    troop_hits = tower_hits = 0
    while troop_next is not None or tower_next is not None:
        tick = min(t for t in (troop_next, tower_next) if t is not None)
        if tick > MAX_TICKS:
            break
        if troop_next == tick:
            body.take(_damage_at(attack, tick, 0))
            troop_hits += 1
            troop_next = next(troop_times, None)
        if tower_next == tick:
            body.take(_damage_at(tower, tick - tower_delay, 0))
            tower_hits += 1
            tower_next = next(tower_times, None)
        if body.dead:
            return TowerAssist(alone, troop_hits, tower_alone, tower_hits)
    return TowerAssist(alone, None, tower_alone, None)


def duel_matrix(
    defenses: Mapping[str, DefenseProfile], attacks: Mapping[str, AttackProfile],
) -> dict[tuple[str, str], Engagement]:
    """Every unordered pair that can actually fight, resolved.

    Spells are excluded: a spell has no hitpoints and no reach, so "who wins"
    is not a question about it.
    """
    fighters = sorted(k for k in attacks if k in defenses and not attacks[k].is_spell)
    out: dict[tuple[str, str], Engagement] = {}
    for i, a in enumerate(fighters):
        for b in fighters[i + 1:]:
            out[(a, b)] = resolve_duel(defenses[a], attacks[a], defenses[b], attacks[b])
    return out


def tower_matrix(
    defenses: Mapping[str, DefenseProfile], attacks: Mapping[str, AttackProfile],
    *, tower_key: str = "PrincessTower",
) -> dict[tuple[str, str], TowerAssist]:
    """Hits needed against every defender, alone and with the tower firing."""
    tower = attacks.get(tower_key)
    if tower is None:
        raise KeyError(f"no {tower_key} attack profile; cannot measure its support")
    out: dict[tuple[str, str], TowerAssist] = {}
    for def_key, defense in defenses.items():
        if defense.is_tower:
            continue  # a tower does not defend against its own support
        for atk_key, attack in attacks.items():
            if atk_key == tower_key:
                continue
            result = hits_with_tower(defense, attack, tower)
            if result.alone is not None:
                out[(def_key, atk_key)] = result
    return out


def write_tower_csv(
    results: Mapping[tuple[str, str], TowerAssist], path: str | Path,
) -> Path:
    """``hits-alone/hits-with-tower`` for every (defender, attacker)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({atk for _, atk in results})
    rows: dict[str, dict[str, str]] = {}
    for (defender, attacker), assist in results.items():
        cell = (f"{assist.alone}/{assist.with_tower}+{assist.tower_hits}"
                if assist.with_tower is not None else f"{assist.alone}/-")
        rows.setdefault(defender, {})[attacker] = cell
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            "# generated by cr_sim.data.engagement -- regenerate, do not hand-edit\n"
            "# rows are defenders, columns are attackers, tournament standard.\n"
            "# cell is 'hits alone / troop hits + tower hits when a Princess "
            "Tower fires too', both engaged from tick zero.\n")
        writer = csv.writer(handle)
        writer.writerow(["card"] + columns)
        for name in sorted(rows):
            writer.writerow([name] + [rows[name].get(c, "") for c in columns])
    return path


def write_duel_csv(
    results: Mapping[tuple[str, str], Engagement], path: str | Path,
) -> Path:
    """One row per pair, with the head start reach bought."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            "# generated by cr_sim.data.engagement -- regenerate, do not hand-edit\n"
            "# head-on 1v1 at tournament standard. free_hits is what the "
            "longer-ranged unit lands before the other closes.\n")
        writer = csv.writer(handle)
        writer.writerow(["a", "b", "winner", "a_hits", "b_hits", "seconds",
                         "free_hits", "untouched", "note"])
        for (a, b), fight in sorted(results.items()):
            writer.writerow([a, b, fight.winner, fight.a_hits, fight.b_hits,
                             f"{fight.ticks / 60:.2f}", fight.head_start,
                             "yes" if fight.clean else "", fight.note])
    return path
