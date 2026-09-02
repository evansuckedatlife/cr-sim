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

from dataclasses import dataclass, replace

from .constants import MAX_ELIXIR
from .entity import EntityKind, Team, entity_id_cursor, restore_entity_ids

__all__ = ["Projection", "project", "committed_value", "elixir_advantage"]


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
    # The dead are walked as a second list rather than spread into one tuple
    # with the living. A destroyed tower has to be counted -- that is the whole
    # point of looking at the graveyard -- but building a tuple of every corpse
    # in the match to find it is work proportional to how long the match has
    # run, on a function that is the entire cost of a projection whenever the
    # board is quiet enough to skip the simulation.
    for source in (battle.entities, battle.graveyard):
        for entity in source:
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
    # A branch has no viewer, so it has no reason to build viewer frames.
    # ``clone`` empties the frame list for exactly this reason, but the *switch*
    # lives on the config, which a branch shares with the battle it came from --
    # so a projection taken while a replay was being recorded went on capturing
    # a frame every few ticks and then threw the lot away. Replacing the config
    # with a copy is what makes the intent actually hold; nothing else on it
    # differs, and nothing the projection reports depends on frames.
    if branch.config.record_frames:
        branch.config = replace(branch.config, record_frames=False)
    start = branch.tick
    limit = branch.tick + horizon_ticks if horizon_ticks is not None else None
    # Entity ids come from a module-level counter, so anything the branch
    # spawns takes ids the live battle was going to use. Handing them back
    # keeps a projection invisible: asking what happens next must not change
    # what happens next. The branch is discarded here, so nothing survives to
    # collide with the ids being reissued.
    cursor = entity_id_cursor()
    try:
        # The horizon test is hoisted out of the loop rather than re-asked on
        # every one of up to three hundred ticks.
        if limit is None:
            while not branch.finished:
                branch.step()
        else:
            step = branch.step
            while branch.tick < limit and not branch.finished:
                step()
        return _read(branch, ticks=branch.tick - start, decided=branch.finished)
    finally:
        restore_entity_ids(cursor)


def elixir_advantage(battle, team: Team) -> float:
    """``team``'s elixir lead, as a fraction of a full bar.

    Read from the board as it stands rather than from the projection. A
    projection plays forward without either side spending, so both bars fill
    to the cap and the projected difference is always zero -- the number is
    only meaningful now.
    """
    other = team.opponent
    mine = battle.players[team].elixir.exact
    theirs = battle.players[other].elixir.exact
    return (mine - theirs) / MAX_ELIXIR


def committed_value(
    battle,
    team: Team,
    horizon_ticks: int | None = None,
    tower_weight: float = 1.0,
    elixir_weight: float = 0.3,
) -> float:
    """``project`` reduced to one number, from ``team``'s point of view.

    Positive means the position resolves in ``team``'s favour. A crown is
    worth one and the tower-health difference is worth ``tower_weight`` at
    most, so crowns dominate -- which is the actual win condition, with tower
    health only separating equal crowns.

    Elixir is the one term not read from the projection, and the one that
    makes this usable as a reward potential. Spending elixir lowers the value
    immediately, so a card has to buy back at least what it cost in board
    terms before it counts as a good play. That is an elixir trade, falling
    out of the arithmetic rather than being scored separately.
    """
    projection = project(battle, horizon_ticks)
    other = team.opponent
    crowns = projection.crowns(team) - projection.crowns(other)
    towers = projection.tower_fraction(team) - projection.tower_fraction(other)
    return (
        float(crowns)
        + tower_weight * towers
        + elixir_weight * elixir_advantage(battle, team)
    )
