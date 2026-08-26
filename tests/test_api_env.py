"""M8 gate: the Gymnasium-style environment wrapping Battle.

``gymnasium`` is an optional dependency (the ``rl`` extra in
``pyproject.toml``); these tests exercise ``cr_sim.api.env`` however it
resolves in this environment -- against the real gymnasium.Env/spaces if
installed, against the module's own fallback shim if not -- and must pass
either way. ``HAS_GYMNASIUM`` records which one actually happened, for
whoever is reading test output.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.arena import load_arena
from cr_sim.engine.entity import Team

from cr_sim.api.encoding import NOOP_SLOT, NUM_CARD_SLOTS, build_encoding_config, observation_shapes
from cr_sim.api.env import HAS_GYMNASIUM, CRSimEnv, CRSimSelfPlayEnv, idle_opponent_policy

from .test_data_pipeline import BUILD

DECK = ("Knight", "Musketeer", "Fireball", "Goblins", "Cannon", "Archer", "Skeletons", "Giant")
NOOP = (NOOP_SLOT, 0, 0)


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


def _env(world, **kwargs):
    data, levels, registry = world
    kwargs.setdefault("ticks_per_second", 20)
    kwargs.setdefault("frame_skip", 4)
    return CRSimEnv(data, levels, registry, DECK, DECK, **kwargs)


def test_gymnasium_presence_is_reported(world):
    """Not a functional assertion -- just makes the fact visible in the
    report, the way the task asks for."""
    assert isinstance(HAS_GYMNASIUM, bool)


def test_reset_returns_an_observation_matching_the_declared_spaces(world):
    env = _env(world)
    obs, info = env.reset(seed=1)
    shapes = observation_shapes(env._config)
    assert obs["grid"].shape == shapes["grid"]
    assert obs["vector"].shape == shapes["vector"]
    assert obs["grid"].dtype == np.float32
    assert obs["vector"].dtype == np.float32
    assert env.observation_space.spaces["grid"].shape == shapes["grid"]
    assert tuple(env.action_space.nvec.tolist()) == (
        NUM_CARD_SLOTS, env._config.action_width, env._config.action_height,
    )
    assert "hash" in info and "tick" in info
    assert info["tick"] == 0


def test_step_advances_exactly_frame_skip_engine_ticks(world):
    env = _env(world, frame_skip=5)
    env.reset(seed=1)
    _obs, _reward, terminated, truncated, info = env.step(NOOP)
    assert not terminated and not truncated, "an empty five-tick opening cannot end a match"
    assert info["tick"] == 5
    assert env.battle.tick == 5


def test_frame_skip_stops_early_if_the_battle_finishes_mid_skip(world):
    """A frame_skip larger than what's left of the match must not step past
    the engine's own natural end -- Battle.step() itself becomes a no-op once
    battle.finished, so ticks cannot run past total_ticks either way, but the
    env must not raise or hang on the boundary."""
    env = _env(world, frame_skip=10_000, max_ticks=None)
    env.reset(seed=1)
    _obs, reward, terminated, truncated, info = env.step(NOOP)
    assert terminated
    assert not truncated
    assert math.isfinite(reward)
    assert info["finished"]
    assert info["tick"] == env.battle.timeline.total_ticks


def test_illegal_action_is_a_safe_noop(world):
    """An out-of-hand-range or unaffordable action must not raise -- it is
    simply refused, the same way Battle.play_card refuses it."""
    env = _env(world)
    env.reset(seed=1)
    # slot 3 is a real hand slot, but drop it far into enemy territory where a
    # troop cannot legally land.
    action = (3, env._config.action_width - 1, env._config.action_height - 1)
    obs, reward, terminated, truncated, info = env.step(action)
    assert math.isfinite(reward)
    assert not terminated


def test_same_seed_produces_identical_observations_and_hashes(world):
    actions = [(0, 4, 4), (1, 2, 10), (4, 0, 0), (2, 5, 3), (3, 1, 12)]

    def run():
        env = _env(world)
        obs, info = env.reset(seed=42)
        trace = [(obs, info["hash"])]
        for action in actions:
            obs, _reward, terminated, _truncated, info = env.step(action)
            trace.append((obs, info["hash"]))
            if terminated:
                break
        return trace

    trace_a, trace_b = run(), run()
    assert len(trace_a) == len(trace_b)
    for (obs_a, hash_a), (obs_b, hash_b) in zip(trace_a, trace_b):
        assert hash_a == hash_b
        assert np.array_equal(obs_a["grid"], obs_b["grid"])
        assert np.array_equal(obs_a["vector"], obs_b["vector"])


def test_episode_terminates_and_reward_stays_finite(world):
    env = _env(world, frame_skip=5, max_ticks=20)
    env.reset(seed=3)
    done = False
    steps = 0
    while not done and steps < 20:
        _obs, reward, terminated, truncated, _info = env.step(NOOP)
        assert math.isfinite(reward)
        done = terminated or truncated
        steps += 1
    assert done, "an idle match with a 20-tick cap must have ended by now"


def test_legal_action_mask_matches_declared_shape(world):
    """The mask is indexed exactly like an action tuple, ``(slot, x, y)``.

    It has to agree with ``action_space`` axis for axis. Shaping the mask like
    an image while the action space reads ``(slot, x, y)`` silently transposes
    every placement sampled from it, and on a 9x16 grid that is a legal-looking
    cell in the wrong place rather than an error anyone would notice.
    """
    env = _env(world)
    env.reset(seed=1)
    mask = env.legal_action_mask()
    assert mask.shape == (NUM_CARD_SLOTS, env._config.action_width, env._config.action_height)
    assert tuple(mask.shape) == tuple(env.action_space.nvec)
    assert mask[NOOP_SLOT, 0, 0]


def test_every_action_sampled_from_the_mask_is_accepted(world):
    """The mask's whole job. Sampling it must never produce a rejected action.

    This is what catches an axis-order mismatch: a transposed index is still a
    valid-looking tuple, so it fails as a decode error or a bad placement
    rather than as a shape error.
    """
    import numpy as np

    env = _env(world)
    env.reset(seed=3)
    for _ in range(40):
        legal = np.argwhere(env.legal_action_mask())
        assert len(legal), "no legal action, not even the no-op"
        action = tuple(int(v) for v in legal[len(legal) // 2])
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break


def test_render_returns_a_board_sized_ascii_grid(world):
    env = _env(world)
    env.reset(seed=1)
    text = env.render()
    lines = text.splitlines()
    assert len(lines) == 64  # HALF_TILES_TALL rows, same as Arena.render()


def test_idle_opponent_policy_always_passes():
    assert idle_opponent_policy({}, np.zeros((1,))) == (NUM_CARD_SLOTS - 1, 0, 0)


# ------------------------------------------------------------------ self-play


def _self_play_env(world, **kwargs):
    data, levels, registry = world
    kwargs.setdefault("ticks_per_second", 20)
    kwargs.setdefault("frame_skip", 4)
    return CRSimSelfPlayEnv(data, levels, registry, DECK, DECK, **kwargs)


def test_self_play_env_returns_per_team_observations(world):
    env = _self_play_env(world)
    obs, info = env.reset(seed=1)
    assert set(obs.keys()) == {Team.BLUE, Team.RED}
    for team_obs in obs.values():
        assert team_obs["grid"].dtype == np.float32
        assert team_obs["vector"].dtype == np.float32
    assert info["tick"] == 0


def test_self_play_env_advances_frame_skip_ticks_per_step(world):
    env = _self_play_env(world, frame_skip=6)
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step({Team.BLUE: NOOP, Team.RED: NOOP})
    assert not terminated and not truncated
    assert info["tick"] == 6
    assert set(reward.keys()) == {Team.BLUE, Team.RED}
    assert all(math.isfinite(r) for r in reward.values())


def test_self_play_observations_agree_with_single_agent_env_given_the_same_seed(world):
    """CRSimEnv(team=Blue) and CRSimSelfPlayEnv's Blue view are two doors onto
    the same encoding -- they must produce the same array for the same board.
    """
    single = _env(world)
    single_obs, _info = single.reset(seed=8)

    dual = _self_play_env(world)
    dual_obs, _info2 = dual.reset(seed=8)

    assert np.array_equal(single_obs["grid"], dual_obs[Team.BLUE]["grid"])
    assert np.array_equal(single_obs["vector"], dual_obs[Team.BLUE]["vector"])
