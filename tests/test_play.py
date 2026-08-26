"""The interactive match server.

What can break here is different from the rest of the project. The engine's
tests ask whether the simulation is right; these ask whether a person can
drive it -- that a card they can see is a card they can play, that a rejection
says why, that changing a deck changes the match, and that the clock advances
by real time rather than by however often the browser happened to ask.

The last one is the subtle one. A session driven by frames rather than by a
clock runs at whatever speed the client renders, which makes the same inputs
produce different battles on different machines.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cr_sim.engine.entity import Team
from cr_sim.play.server import DEFAULT_DECK, PlayServer
from cr_sim.play.session import MAX_CATCHUP_SECONDS, PlaySession, SessionConfig, random_controller

from .test_data_pipeline import BUILD


@pytest.fixture(scope="module")
def server():
    return PlayServer(Path(BUILD))


@pytest.fixture
def session(server):
    return PlaySession(
        data=server.data,
        levels=server.levels,
        registry=server.registry,
        config=SessionConfig(human_deck=DEFAULT_DECK, ai_deck=DEFAULT_DECK, seed=1),
        controller=random_controller(1),
    )


# ------------------------------------------------------------------- clock


def test_time_advances_by_the_clock_not_by_the_caller(session):
    """Ten small polls and one large one must land on the same tick.

    A session that stepped once per request would run at the browser's frame
    rate, so the same match would play out differently on different machines
    and a replay of its commands would not reproduce it.
    """
    start = 1000.0
    session._last_wall = start
    for i in range(1, 11):
        session.advance(start + i * 0.1)
    stepwise = session.battle.tick

    other = PlaySession(
        data=session.data, levels=session.levels, registry=session.registry,
        config=session.config, controller=random_controller(1),
    )
    other._last_wall = start
    other.advance(start + 1.0)
    assert other.battle.tick == stepwise


def test_a_long_gap_is_capped(session):
    """A backgrounded tab stops asking for frames.

    Without a ceiling the first request after it returns would simulate every
    missed tick at once and block for as long as the tab was hidden.
    """
    session._last_wall = 0.0
    ticks = session.advance(600.0)
    assert ticks <= MAX_CATCHUP_SECONDS * session.config.ticks_per_second + 1


def test_a_paused_session_does_not_advance(session):
    session._last_wall = 0.0
    session.advance(1.0)
    held = session.battle.tick
    session.paused = True
    session.advance(5.0)
    assert session.battle.tick == held


def test_speed_scales_the_clock(session):
    session._last_wall = 0.0
    session.advance(1.0)
    normal = session.battle.tick

    fast = PlaySession(
        data=session.data, levels=session.levels, registry=session.registry,
        config=session.config, controller=None,
    )
    fast.speed = 2.0
    fast._last_wall = 0.0
    fast.advance(1.0)
    assert fast.battle.tick > normal


# -------------------------------------------------------------------- play


def test_a_card_in_hand_can_be_played(session):
    session._last_wall = 0.0
    session.advance(1.0)
    hand = session.snapshot()["you"]["hand"]
    affordable = [c for c in hand if c["cost"] <= session.snapshot()["you"]["elixir"]]
    if not affordable:
        pytest.skip("no affordable card in the opening hand for this seed")
    assert session.play(affordable[0]["name"], 3.5, 11.0)["ok"]


def test_a_rejection_says_why(session):
    """Four different reasons a card does not appear, and a silent failure
    reads as a bug to whoever pressed the button."""
    assert "not in hand" in session.play("Golem", 3.5, 11.0)["reason"]

    session._last_wall = 0.0
    session.advance(2.0)
    hand = session.snapshot()["you"]["hand"]
    expensive = max(hand, key=lambda c: c["cost"])
    if expensive["cost"] > session.snapshot()["you"]["elixir"]:
        assert "elixir" in session.play(expensive["name"], 3.5, 11.0)["reason"]

    cheap = min(hand, key=lambda c: c["cost"])
    session.battle.players[Team.BLUE].elixir.add(10)
    if cheap["kind"] != "spell":
        assert "placed" in session.play(cheap["name"], 9.0, 30.0)["reason"]


def test_playing_a_card_cycles_it_out_of_hand(session):
    session._last_wall = 0.0
    session.advance(3.0)
    session.battle.players[Team.BLUE].elixir.add(10)
    before = session.snapshot()["you"]["hand"]
    played = before[0]["name"]
    assert session.play(played, 3.5, 11.0)["ok"]
    after = [c["name"] for c in session.snapshot()["you"]["hand"]]
    assert played not in after, "a played card stayed in hand"


def test_every_play_is_logged_for_replay(session):
    session._last_wall = 0.0
    session.advance(3.0)
    session.battle.players[Team.BLUE].elixir.add(10)
    card = session.snapshot()["you"]["hand"][0]["name"]
    session.play(card, 3.5, 11.0)
    mine = [c for c in session.commands if c["team"] == "BLUE"]
    assert mine and mine[-1]["card"] == card
    assert "tick" in mine[-1], "a command without its tick cannot be replayed"


# --------------------------------------------------------------- opponent


def test_the_opponent_actually_plays_cards(session):
    """An idle opponent shows a person nothing.

    Most of what there is to look at -- units meeting in the lane, a tower
    being defended -- does not happen at all against a side that never deploys.
    """
    now = 0.0
    session._last_wall = now
    for _ in range(300):
        now += 0.1
        session.advance(now)
    assert [c for c in session.commands if c["team"] == "RED"], "the opponent never played"


# ------------------------------------------------------------------ decks


def test_a_deck_must_be_eight_different_cards(server):
    assert not server.handle("/api/new", {"human": ["Knight"] * 8, "ai": list(DEFAULT_DECK)})["ok"]
    assert not server.handle("/api/new", {"human": list(DEFAULT_DECK)[:4], "ai": list(DEFAULT_DECK)})["ok"]


def test_an_unknown_card_is_named_in_the_rejection(server):
    result = server.handle("/api/new", {"human": ["Nonesuch"] + list(DEFAULT_DECK)[1:],
                                        "ai": list(DEFAULT_DECK)})
    assert not result["ok"] and "Nonesuch" in result["reason"]


def test_changing_a_deck_changes_the_match(server):
    other = ("Golem", "BabyDragon", "Wizard", "MegaMinion",
             "Barbarians", "Zap", "Arrows", "Knight")
    assert server.handle("/api/new", {"human": list(other), "ai": list(DEFAULT_DECK)})["ok"]
    hand = {c["name"] for c in server.handle("/api/state", {})["you"]["hand"]}
    assert hand <= set(other), f"hand {hand} is not from the new deck"


# ------------------------------------------------------------------ setup


def test_setup_carries_the_real_arena(server):
    """The page draws the shipped tilemap, so what you click is the geometry
    the engine tests against rather than a picture of it."""
    setup = server.setup()
    assert setup["widthTiles"] == 18.0 and setup["heightTiles"] == 32.0
    assert len(setup["terrain"]) == setup["halfHeight"]
    assert len(setup["terrain"][0]) == setup["halfWidth"]
    assert any(2 in row for row in setup["terrain"]), "no water in the tilemap"


def test_setup_offers_the_whole_playable_pool(server):
    setup = server.setup()
    assert len(setup["cards"]) == len(server.registry.standard())
    assert all(c["cost"] > 0 for c in setup["cards"])
    # Sorted by cost so the picker opens on the cheap cards, which is where
    # a deck edit almost always starts.
    costs = [c["cost"] for c in setup["cards"]]
    assert costs == sorted(costs)


def test_a_snapshot_is_json_serialisable(server):
    """It is sent several times a second; anything unserialisable is a hard
    failure at exactly the wrong moment."""
    payload = json.dumps(server.handle("/api/state", {}))
    assert len(payload) > 100


def test_an_unknown_route_is_reported_rather_than_raising(server):
    assert not server.handle("/api/nope", {})["ok"]


# ------------------------------------------------------- opponent failures


def test_a_missing_checkpoint_falls_back_instead_of_crashing():
    """The page is worth having before any policy is.

    The failure has to happen where the fallback can catch it. Deferring the
    file open to the opponent's first move -- which is tempting, because the
    network's shapes need a live match -- surfaces it inside a request handler
    instead, and every poll then returns a traceback rather than a match.
    """
    server = PlayServer(Path(BUILD), Path("runs/definitely-not-here/final.pt"))
    label = getattr(server.session, "opponent_label", "")
    assert label.startswith("random"), label
    assert "policy failed" in label
    assert server.handle("/api/state", {})["tick"] >= 0, "the server stopped serving"


def test_a_controller_that_raises_retires_rather_than_killing_the_match(session):
    """An opponent that fails costs you an opponent, not the match.

    Raising reaches the caller through whatever drives the clock, and for the
    web server that is every poll -- one bad move would become a stream of
    tracebacks.
    """
    def broken(battle, team):
        raise RuntimeError("no")

    session.controller = broken
    session._next_ai_tick = 0
    session._last_wall = 0.0
    session.advance(1.0)

    assert session.controller is None, "the broken controller was kept"
    assert "RuntimeError" in (session.controller_error or "")
    session.advance(2.0)  # and the match keeps running
    assert session.battle.tick > 0


def test_a_checkpoint_path_resolves_against_the_project_root():
    """``runs/...`` is how the trainer prints it, and only resolves from there.

    Started from anywhere else a correct-looking command would silently play a
    random opponent.
    """
    from cr_sim.play.server import ROOT, _resolve

    assert _resolve(None) is None
    existing = Path("pyproject.toml")
    assert _resolve(existing) in (existing, ROOT / existing)
    assert _resolve(Path("nope/nope.pt")) == Path("nope/nope.pt")


# ------------------------------------------------------------------ icons


def test_icons_are_urls_the_browser_can_fetch(server):
    """The icon map gives filesystem paths, which an <img> will not load.

    The replay viewer inlines them as data URIs instead; doing that here would
    put megabytes into every page load for a server that is already sitting
    there able to serve the files.
    """
    icons = server.setup()["icons"]
    assert icons, "no icons at all"
    assert all(url.startswith("/icon/") for url in icons.values()), "a raw path leaked"
    assert "Knight" in icons


def test_every_icon_url_resolves_to_a_real_file(server):
    for name, url in server.setup()["icons"].items():
        path = server._icon_files.get(url[len("/icon/"):])
        assert path is not None and path.is_file(), f"{name} -> {url}"


def test_icons_are_looked_up_by_name_not_by_joining_a_path(server):
    """So a crafted name cannot reach outside the icon pack."""
    for hostile in ("../pyproject.toml", r"..\pyproject.toml", "/etc/passwd"):
        assert server._icon_files.get(hostile) is None


def test_a_hand_card_carries_its_placement_rules(server):
    """The page previews legality while you aim; the server still decides."""
    hand = server.handle("/api/state", {})["you"]["hand"]
    assert hand
    for card in hand:
        assert set(card) >= {"name", "cost", "kind", "evo", "hasEvo", "anywhere", "water"}


# ------------------------------------------------------------- evolutions


def test_evolution_slots_are_capped_at_two(server):
    """Matching the game. Silently dropping a third would leave the page
    showing an evolution that never fires."""
    deck = list(DEFAULT_DECK)
    result = server.handle("/api/new", {
        "human": deck, "ai": deck, "humanEvolutions": deck[:3],
    })
    assert not result["ok"] and "2 evolution slots" in result["reason"]


def test_a_slot_must_name_a_card_that_has_an_evolution(server):
    deck = list(DEFAULT_DECK)
    result = server.handle("/api/new", {
        "human": deck, "ai": deck, "humanEvolutions": ["Fireball"],
    })
    assert not result["ok"] and "Fireball" in result["reason"]


def test_a_slot_must_name_a_card_in_the_deck(server):
    deck = list(DEFAULT_DECK)
    result = server.handle("/api/new", {
        "human": deck, "ai": deck, "humanEvolutions": ["Barbarians"],
    })
    assert not result["ok"] and "not in the deck" in result["reason"]


def test_a_slotted_card_reaches_the_battle(server):
    deck = list(DEFAULT_DECK)
    assert server.handle("/api/new", {
        "human": deck, "ai": deck, "humanEvolutions": ["Knight"],
    })["ok"]
    assert "Knight" in server.session.battle.players[Team.BLUE].evolutions
    # And the opponent gets none, because slots are a deck-building choice
    # per side rather than a property of the card.
    assert not server.session.battle.players[Team.RED].evolutions


def test_the_card_pool_says_which_cards_have_an_evolution(server):
    cards = server.setup()["cards"]
    assert any(c["hasEvo"] for c in cards), "no card reports an evolution"
    assert not all(c["hasEvo"] for c in cards), "every card reports one"
