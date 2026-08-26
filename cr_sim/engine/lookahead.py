"""What the board is worth if nobody plays another card.

The engine is deterministic, so from any position there is exactly one answer
to "what happens if both players stop playing" -- not a distribution, not an
estimate, a fact. Playing a branch forward and reading the result is therefore
an *exact* evaluation of the current board, and it costs a clone plus some
ticks rather than a trained network.

That matters because the learned critic has been the weak link: measured on
its own training distribution it explained six per cent of the variance in
returns, so PPO's advantages were mostly noise. A projection needs no training
and cannot be miscalibrated.

It is not a full evaluation of the *position* -- elixir in hand, cards in
cycle and everything either player might do next are all invisible to it. It
answers a narrower question, exactly: what is already committed to the board?
That is the question "do I need to respond right now" turns on.
"""

from __future__ import annotations

from dataclasses import dataclass

from .entity import EntityKind, Team, entity_id_cursor, restore_entity_ids

__all__ = ["Projection", "project", "committed_value"]


@dataclass(frozen=True, slots=True)
class Projection:
    """Where the board ends up if neither side plays again."""

    blue_crowns: int
    red_crowns: int
    #: Total remaining hitpoints across each side's standing towers.
    blue_tower_hitpoints: int
    red_tower_hitpoints: int
    #: Fraction of starting tower hitpoints still standing, per side.
    blue_tower_fraction: float
    red_tower_fraction: float
    #: Ticks simulated to reach this. Zero means the board was already quiet.
    ticks: int
    #: Whether the projection reached a real end rather than the horizon.
    decided: bool

    def crowns(self, team: Team) -> int:
        return self.blue_crowns if team is Team.BLUE else self.red_crowns

    def tower_fraction(self, team: Team) -> float:
        return (
            self.blue_tower_fraction if team is Team.BLUE else self.red_tower_fraction
        )


def _is_quiet(battle) -> bool:
    """True when nothing on the board can change without a card being played.

    Towers with nothing to shoot at and no buff ticking on them are inert, so
    a position holding only untouched towers projects to itself. This is the
    common case between pushes, and skipping the clone entirely is worth more
    than any amount of tuning the simulation that follows.
    """
    if battle._pending_waves or battle._pending:
        return False
    for entity in battle.entities:
        if entity.kind is not EntityKind.TOWER:
            return False
        if entity.buffs is not None and bool(entity.buffs):
            return False
    # A King that is mid-wake will act on its own; that is a change.
    return not any(battle._king_waking.values())


def _read(battle, ticks: int, decided: bool) -> Projection:
    totals: dict[Team, int] = {Team.BLUE: 0, Team.RED: 0}
    starting: dict[Team, int] = {Team.BLUE: 0, Team.RED: 0}
    for entity in (*battle.entities, *battle.graveyard):
        if entity.kind is not EntityKind.TOWER:
            continue
        starting[entity.team] += entity.max_hitpoints
        if not entity.dead:
            totals[entity.team] += max(entity.hitpoints, 0)

    def fraction(team: Team) -> float:
        return totals[team] / starting[team] if starting[team] else 0.0

    blue = battle.players[Team.BLUE].crowns
    red = battle.players[Team.RED].crowns
    return Projection(
        blue_crowns=blue,
        red_crowns=red,
        blue_tower_hitpoints=totals[Team.BLUE],
        red_tower_hitpoints=totals[Team.RED],
        blue_tower_fraction=fraction(Team.BLUE),
        red_tower_fraction=fraction(Team.RED),
        ticks=ticks,
        decided=decided,
    )


def project(battle, horizon_ticks: int | None = None) -> Projection:
    """Play ``battle`` forward with neither side playing a card.

    ``horizon_ticks`` bounds how far to look; ``None`` runs to the end of the
    match, which is the exact answer and costs about two hundred milliseconds
    from a mid-match position. A few seconds is usually enough to see whether
    a push connects, and costs about a tenth of that.

    The live battle is untouched -- everything happens on a clone.
    """
    if _is_quiet(battle):
        return _read(battle, ticks=0, decided=False)

    branch = battle.clone()
    start = branch.tick
    limit = branch.tick + horizon_ticks if horizon_ticks is not None else None
    # Entity ids come from a module-level counter, so anything the branch
    # spawns takes ids the live battle was going to use. Handing them back
    # keeps a projection invisible: asking what happens next must not change
    # what happens next. The branch is discarded here, so nothing survives to
    # collide with the ids being reissued.
    cursor = entity_id_cursor()
    try:
        while not branch.finished:
            if limit is not None and branch.tick >= limit:
                break
            branch.step()
        return _read(branch, ticks=branch.tick - start, decided=branch.finished)
    finally:
        restore_entity_ids(cursor)


def committed_value(
    battle,
    team: Team,
    horizon_ticks: int | None = None,
    tower_weight: float = 1.0,
) -> float:
    """``project`` reduced to one number, from ``team``'s point of view.

    Positive means the board as it stands resolves in ``team``'s favour. A
    crown is worth one and the tower-health difference is worth
    ``tower_weight`` at most, so crowns dominate, which is the actual win
    condition -- tower health only separates equal crowns.
    """
    projection = project(battle, horizon_ticks)
    other = team.opponent
    crowns = projection.crowns(team) - projection.crowns(other)
    towers = projection.tower_fraction(team) - projection.tower_fraction(other)
    return float(crowns) + tower_weight * towers
