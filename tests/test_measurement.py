"""Whether a measurement says what it was measured against.

"Lift" was reported on two incompatible scales for most of this project's
life. The in-run probe played the policy against an opponent that never plays
a card; the large paired verdicts played it against one that spends its elixir
on legal placements; both were written down as "lift" and compared to each
other. The random control wins 92% of the idle matches and 26% of the random
ones, so the two numbers never lived on the same scale and no comparison
between them meant anything.

A comment does not stop that happening again. Refusing to write the row does,
and reading the opponent off the environment rather than taking it as an
argument stops a caller labelling a measurement with an opponent it did not
actually face.
"""

from __future__ import annotations

import numpy as np
import pytest

from cr_sim.api.encoding import NOOP_SLOT
from cr_sim.api.env import CRSimEnv
from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData

from .test_data_pipeline import BUILD

DECK = ("Knight", "Musketeer", "Cannon", "Skeletons",
        "IceSpirits", "Log", "Fireball", "Goblins")


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


def _env(world, **kwargs):
    data, levels, registry = world
    kwargs.setdefault("ticks_per_second", 20)
    kwargs.setdefault("frame_skip", 20)
    kwargs.setdefault("max_ticks", 20 * 40)
    return CRSimEnv(data, levels, registry, DECK, DECK, **kwargs)


def test_a_lift_cannot_be_recorded_without_naming_its_opponent():
    """"Lift" was reported on two incompatible scales for most of this
    project's life: the in-run probe faced an opponent that never plays a card
    while the large paired verdicts faced a random one, and the two numbers
    were compared to each other. The control wins 92% of the idle matches and
    26% of the random ones, so they never lived on the same scale.

    A comment does not stop that happening again; refusing to write the row
    does.
    """
    from cr_sim.train.selfplay import check_lift_is_named

    with pytest.raises(ValueError, match="eval_opponent"):
        check_lift_is_named({"updates": 3, "eval_lift_sd": 0.42})
    with pytest.raises(ValueError, match="eval_opponent"):
        check_lift_is_named({"updates": 3, "eval_lift_sd": 0.42, "eval_opponent": ""})

    named = {"updates": 3, "eval_lift_sd": 0.42, "eval_opponent": "random"}
    assert check_lift_is_named(named) is named
    # A row with no lift on it is not a measurement and needs no label.
    plain = {"updates": 4, "entropy": 3.1}
    assert check_lift_is_named(plain) is plain


def test_the_opponent_a_probe_faced_is_read_off_the_environment(world):
    """Not taken as an argument. A caller cannot label a measurement with an
    opponent it did not actually play."""
    from cr_sim.train.run import _random_opponent
    from cr_sim.train.selfplay import opponent_name

    assert opponent_name(_env(world)) == "idle"
    assert opponent_name(_env(world, opponent_policy=_random_opponent(0))) == "random"

    def anonymous(observation, mask):
        return (NOOP_SLOT, 0, 0)

    # Unknown rather than guessed: a wrong label is worse than an absent one.
    assert opponent_name(_env(world, opponent_policy=anonymous)) == "unknown"
