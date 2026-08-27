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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cr_sim.api.env import CRSimEnv
from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.entity import Team
from cr_sim.train.clone import Demonstrations, collect
from cr_sim.train.run import DEFAULT_BUILD, DEFAULT_DECK, _random_opponent
from cr_sim.train.scripted import SearchBot, SearchBotConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make-demos")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data_cache/demos"))
    parser.add_argument("--tower-level", type=int, default=5)
    parser.add_argument("--match-seconds", type=int, default=120)
    parser.add_argument("--frame-skip", type=int, default=30)
    parser.add_argument("--horizon-seconds", type=float, default=15.0)
    parser.add_argument("--candidates", type=int, default=14)
    parser.add_argument(
        "--opponent", choices=("random", "bot"), default="random",
        help="who the expert plays. 'random' matches how the result is "
             "measured; 'bot' produces harder positions but takes twice as "
             "long, since both sides then search.",
    )
    args = parser.parse_args(argv)

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
            ticks_per_second=20, frame_skip=args.frame_skip,
            max_ticks=20 * args.match_seconds, tower_level=args.tower_level,
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

    demos = collect(make_env, make_expert, episodes=args.episodes,
                    on_episode=progress)

    # Only worth keeping if the expert actually won them. A bot having a bad
    # run teaches a policy to have a bad run, and cloning cannot tell the
    # difference.
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"shard-{args.shard:02d}.npz"
    demos.save(path)
    print(f"shard {args.shard}: wrote {len(demos)} decisions from "
          f"{demos.episodes} episodes to {path} "
          f"(play rate {demos.play_rate:.0%}, "
          f"{(time.perf_counter() - started) / 60:.0f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
