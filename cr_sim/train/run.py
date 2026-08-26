"""``python -m cr_sim.train.run`` -- launch a training job.

Kept separate from :mod:`cr_sim.train.ppo` so the algorithm stays importable
and testable without dragging in argument parsing, file layout or checkpoint
policy. This module owns the things a *run* needs and the algorithm does not:
where results go, how often to save, and what to record.

Metrics go to a JSONL file rather than to stdout alone. A run that takes hours
and prints to a terminal loses everything the moment that terminal closes, and
the questions worth asking afterwards -- did entropy collapse before or after
the return moved, was the value loss already diverging -- need the whole series,
not the last line.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..data.cards import build_card_registry
from ..data.leveling import build_level_table
from ..data.source import LogicData
from ..api.encoding import NOOP_SLOT
from ..api.env import CRSimEnv
from ..api.reward import RewardWeights
from .ppo import PPOConfig, train

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD = ROOT / "data_cache" / "csv_logic"

#: A recognisable, cheap deck. Cycle rather than beatdown on purpose: more
#: decisions per match means more gradient per second of simulation, and the
#: cards are individually simple, so early learning is about placement and
#: timing rather than about a Golem's twenty-second commitment.
DEFAULT_DECK = (
    "Knight", "Musketeer", "Cannon", "Skeletons",
    "IceSpirits", "Log", "Fireball", "Goblins",
)


def _random_opponent(seed: int):
    """An opponent that spends its elixir on legal placements.

    Weak, but not passive, and the difference matters: against an opponent that
    never plays a card there is nothing to kite and almost nothing to destroy,
    so two of the five reward terms measure a board that never exists.
    """
    rng = np.random.default_rng(seed)

    def policy(observation, mask):
        legal = np.argwhere(mask)
        if not len(legal):
            return (NOOP_SLOT, 0, 0)
        return tuple(int(v) for v in legal[rng.integers(len(legal))])

    return policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cr-sim-train")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--envs", type=int, default=8, help="parallel battles per rollout")
    parser.add_argument("--horizon", type=int, default=256)
    parser.add_argument("--tps", type=int, default=20, help="engine tick rate")
    parser.add_argument("--frame-skip", type=int, default=10, help="ticks per decision")
    parser.add_argument("--match-seconds", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy", type=float, default=0.02)
    parser.add_argument(
        "--shaping", type=float, default=0.01,
        help="weight on tower-health difference. At 0.01 a whole match's tower "
             "damage is worth about 0.02 against 1.0 per crown, so the reward is "
             "effectively sparse; raise it to give credit between crowns.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "runs")
    parser.add_argument("--name", default="ppo")
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--save-every", type=int, default=10, help="updates between checkpoints")
    parser.add_argument(
        "--reward", choices=("simple", "five-term"), default="five-term",
        help="'simple' is crowns plus a tower-health difference, kept as a control; "
             "'five-term' adds tower damage, elixir trade, counterpush and kites.",
    )
    parser.add_argument(
        "--opponent", choices=("idle", "random"), default="random",
        help="'idle' never plays a card, which leaves the kite and trade terms with "
             "nothing to measure. 'random' spends its elixir on legal placements.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"

    data = LogicData.load(args.build)
    levels = build_level_table(data)
    registry = build_card_registry(data)

    def make_env(index: int) -> CRSimEnv:
        # Each environment gets its own opponent stream, so eight parallel
        # battles do not face the identical sequence of placements and report
        # a smoother result than the policy has earned.
        opponent = _random_opponent(args.seed * 1000 + index) if args.opponent == "random" else None
        return CRSimEnv(
            data, levels, registry, DEFAULT_DECK, DEFAULT_DECK,
            ticks_per_second=args.tps,
            frame_skip=args.frame_skip,
            max_ticks=args.tps * args.match_seconds,
            reward_shaping_weight=args.shaping,
            reward_weights=RewardWeights() if args.reward == "five-term" else None,
            opponent_policy=opponent,
        )

    config = PPOConfig(
        total_steps=args.steps,
        horizon=args.horizon,
        num_envs=args.envs,
        learning_rate=args.lr,
        entropy_coefficient=args.entropy,
        seed=args.seed,
    )
    (out / "config.json").write_text(
        json.dumps({**asdict(config), "deck": list(DEFAULT_DECK), "tps": args.tps,
                    "frame_skip": args.frame_skip, "match_seconds": args.match_seconds,
                    "shaping": args.shaping, "reward": args.reward,
                    "opponent": args.opponent}, indent=2),
        encoding="utf-8",
    )

    started = time.perf_counter()
    net_holder: dict[str, torch.nn.Module] = {}

    with metrics_path.open("w", encoding="utf-8") as stream:
        def record(stats: dict) -> None:
            stream.write(json.dumps(stats) + "\n")
            stream.flush()  # a run that dies at hour three should keep hour two
            print(
                f"update {stats['updates']:4d}  steps {stats['steps']:>9d}  "
                f"{stats['steps_per_second']:6.0f}/s  "
                f"return {stats['mean_return']:+8.4f}  "
                f"win {stats['win_rate']:4.0%}  "
                f"entropy {stats['entropy']:6.3f}  "
                f"pass {stats['noop_fraction']:4.0%}  "
                f"loss {stats['policy_loss']:+.4f}/{stats['value_loss']:.4f}",
                flush=True,
            )
            net = net_holder.get("net")
            if net is not None and stats["updates"] % args.save_every == 0:
                torch.save(
                    {"state_dict": net.state_dict(), "stats": stats},
                    out / "checkpoint.pt",
                )

        # The network's shapes come from the first observation, so it does not
        # exist until the trainer has one. It hands it back here, which is what
        # lets this checkpoint mid-run instead of only at the end.
        net = train(
            make_env, config,
            on_update=record,
            on_net=lambda built: net_holder.__setitem__("net", built),
        )

    torch.save({"state_dict": net.state_dict()}, out / "final.pt")
    elapsed = time.perf_counter() - started
    print(f"\ndone in {elapsed / 60:.1f} min -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
