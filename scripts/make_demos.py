"""Record the searching bot playing, so a policy can learn from it.

The bootstrap step. Over 16 matches against a random opponent the bot wins
100% and never loses, so its decisions are worth copying -- which is the whole
premise of behavioural cloning, and the step this project skipped when it
started reinforcement learning from random initialisation.

Sharded because a match costs about seventeen seconds: the bot branches the
battle for every candidate placement and plays each one fifteen seconds
forward. Run several shards at once and merge them.

    python scripts/make_demos.py --episodes 60 --shard 0 --out data_cache/demos
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cr_sim.api.encoding import parse_observation
from cr_sim.api.env import CRSimEnv
from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.entity import Team
from cr_sim.train.clone import Demonstrations, collect
from cr_sim.train.run import (
    DEFAULT_BUILD, DEFAULT_DECK, _random_opponent, _reward_weights)
from cr_sim.train.scripted import SearchBot, SearchBotConfig


def build_parser() -> argparse.ArgumentParser:
    """Exposed so a test can compare these flags against run.py's.

    The two parsers are written separately and drifted once already: this one
    built its environment with no reward at all, so every demonstration set
    carried value targets from a reward no fine-tune ever optimised.
    """
    parser = argparse.ArgumentParser(prog="make-demos")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data_cache/demos"))
    parser.add_argument("--tower-level", type=int, default=5)
    parser.add_argument(
        "--observations", default="v1",
        help="comma-separated observation variants to record, each written to "
             "its own subdirectory of --out. One playthrough produces all of "
             "them: the expert reads the battle rather than the observation, "
             "so every variant sees the same trajectory and the same "
             "decisions, which is what makes an encoding ablation a paired "
             "comparison instead of two experiments. Names are those "
             "cr_sim.api.encoding.parse_observation accepts, plus a bare "
             "flag name for a single-change variant.",
    )
    parser.add_argument("--match-seconds", type=int, default=120)
    parser.add_argument("--frame-skip", type=int, default=30)
    parser.add_argument("--tps", type=int, default=20)
    parser.add_argument(
        "--horizon-seconds", type=float, default=15.0,
        help="how far the SEARCH projects each candidate. Not the reward's "
             "projection horizon -- that is --reward-horizon-seconds, and the "
             "two are different quantities that happen to share a unit.",
    )
    # The reward the value column is harvested under. It had none: this script
    # built its env with no reward_weights at all, so every demonstration set
    # ever collected carried value targets from the simple shaped reward while
    # every fine-tune ran `projected`. The clone's critic is the part
    # reinforcement learning inherits, so it arrived predicting a quantity
    # nobody was optimising -- +1.48 against returns averaging +0.47. The
    # defaults here match cr_sim.train.run's, which is the point.
    parser.add_argument(
        "--reward", choices=("simple", "five-term", "projected"),
        default="projected",
        help="which reward the recorded value targets are computed under. "
             "Must match the reward the clone is later fine-tuned with, or "
             "the inherited critic predicts the wrong quantity.",
    )
    parser.add_argument("--elixir-weight", type=float, default=0.0)
    parser.add_argument(
        "--reward-horizon-seconds", type=float, default=3.0,
        help="the projected reward's lookahead, matching run.py's "
             "--horizon-seconds. 0 disables the projection.",
    )
    parser.add_argument("--candidates", type=int, default=14)
    parser.add_argument(
        "--opponent", choices=("random", "bot"), default="random",
        help="who the expert plays. 'random' matches how the result is "
             "measured; 'bot' produces harder positions but takes twice as "
             "long, since both sides then search.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    data = LogicData.load(DEFAULT_BUILD)
    levels, registry = build_level_table(data), build_card_registry(data)
    # Shards must not replay each other's battles, or the merged set is the
    # same few games repeated and the policy learns those rather than the game.
    offset = args.shard * 10_000

    def make_env(index: int) -> CRSimEnv:
        seed = offset + index
        opponent = None
        if args.opponent == "random":
            opponent = _random_opponent(50_000 + seed)
        else:
            foe = SearchBot(Team.RED, SearchBotConfig(
                horizon_seconds=args.horizon_seconds,
                candidates=args.candidates, seed=seed))

            def opponent(observation, mask, battle=None, _foe=foe):
                return _foe(observation, mask, battle)
            opponent.wants_battle = True
        return CRSimEnv(
            data, levels, registry, DEFAULT_DECK, DEFAULT_DECK,
            ticks_per_second=args.tps, frame_skip=args.frame_skip,
            max_ticks=args.tps * args.match_seconds,
            tower_level=args.tower_level,
            # Through a shim, because _reward_weights reads
            # `horizon_seconds` and in this script that name is the *search*
            # horizon (15s), not the reward's projection horizon (3s). Passing
            # args directly would silently build the projection with a
            # five-times-too-long lookahead -- the same shape of bug as the
            # one this flag exists to fix.
            reward_weights=_reward_weights(SimpleNamespace(
                reward=args.reward,
                horizon_seconds=args.reward_horizon_seconds,
                elixir_weight=args.elixir_weight)),
            opponent_policy=opponent)

    def make_expert(env):
        bot = SearchBot(Team.BLUE, SearchBotConfig(
            horizon_seconds=args.horizon_seconds,
            candidates=args.candidates, seed=args.shard))

        def expert(observation, mask, battle=None):
            return bot(observation, mask, battle)
        # collect reads the search's scores off this, and they are the real
        # training target -- the chosen move is not a function of the state.
        expert.bot = bot
        return expert

    started = time.perf_counter()

    def progress(done: int, samples: int) -> None:
        rate = (time.perf_counter() - started) / done
        print(f"shard {args.shard}: {done}/{args.episodes} episodes, "
              f"{samples} decisions, {rate:.0f}s/episode, "
              f"{(args.episodes - done) * rate / 60:.0f} min left", flush=True)

    names = [n.strip() for n in args.observations.split(",") if n.strip()]
    if names == ["v1"]:
        demos = collect(make_env, make_expert, episodes=args.episodes,
                        on_episode=progress,
                        reward_name=args.reward,
                        observation_name="v1")
        results = {"": demos}
    else:
        results = collect(
            make_env, make_expert, episodes=args.episodes, on_episode=progress,
            reward_name=args.reward,
            variants={name: parse_observation(name) for name in names})

    for name, demos in results.items():
        directory = args.out / name if name else args.out
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"shard-{args.shard:02d}.npz"
        demos.save(path)
        print(f"shard {args.shard}: wrote {len(demos)} decisions from "
              f"{demos.episodes} episodes to {path} "
              f"(play rate {demos.play_rate:.0%}, "
              f"observation {demos.observation!r}, reward {demos.reward!r}, "
              f"{(time.perf_counter() - started) / 60:.0f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
