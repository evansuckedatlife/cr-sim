"""How far apart do two sampled evaluations of the *same* checkpoint land?

`docs/training.md` tells every reader that "any sampled comparison closer than
about 0.04 sd is measuring the random number generator", and that figure comes
from one pair of runs that measured +0.583 and +0.581. A difference of 0.002
between two runs is not evidence that the spread is 0.002 -- it is one draw,
and if those two runs happened to share a sampling stream it is not even that.

The number matters more than it looks. It is the threshold this project uses to
decide whether a sampled result means anything, so if it is too small, every
sampled comparison in the docs reads as more conclusive than it is.

Greedy is not measured here because greedy has no spread at all: it reproduces
bit-identically, which this script asserts rather than assumes, since a greedy
arm that moved would mean the two arms are not playing the same battles and the
sampled spread below would be measuring that instead.

    python scripts/measure_sampled_noise.py runs/clone-cardstat-lookup/cloned.pt \
        --streams 4 --episodes 150
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cr_sim.train.evaluate import evaluate, paired_lift
from cr_sim.train.run import DEFAULT_DECK
from cr_sim.train.selfplay import opponent_name

from scripts.evaluate_decks import Bench, load_policy


def build_parser():
    parser = argparse.ArgumentParser(prog="measure-sampled-noise")
    parser.add_argument("checkpoint")
    parser.add_argument("--streams", type=int, default=4,
                        help="how many independent sampling streams to play")
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--tower-level", type=int, default=5)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--out", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    bench = Bench(tower_level=args.tower_level, observation="v1")
    deck = tuple(DEFAULT_DECK)
    seeds = [int(s) for s in
             np.random.default_rng(args.seed).integers(0, 2**31 - 1, args.episodes)]

    env = bench.make_env(deck)
    env.reset(seed=0)
    net, payload = load_policy(args.checkpoint, env)

    started = time.perf_counter()
    control_env = bench.make_env(deck)
    control = evaluate(control_env, None, episodes=args.episodes, seeds=seeds)
    faced = opponent_name(control_env)
    print(f"{Path(args.checkpoint).parent.name}, head {payload.get('head')}, "
          f"{args.episodes} battles vs the {faced} control", flush=True)

    # Greedy twice. It must land in exactly the same place both times; if it
    # does not, the battles are not being held fixed and nothing below is
    # about the sampling stream.
    first = paired_lift(evaluate(bench.make_env(deck), net, episodes=args.episodes,
                                 seeds=seeds, greedy=True), control)
    second = paired_lift(evaluate(bench.make_env(deck), net, episodes=args.episodes,
                                  seeds=seeds, greedy=True), control)
    print(f"  greedy, twice: {first['lift']:+.4f} and {second['lift']:+.4f}",
          flush=True)
    assert first["lift"] == second["lift"], (
        "greedy moved between two runs on identical seeds, so the sampled "
        "spread below would be measuring that rather than the sampling stream")

    lifts = []
    for index in range(args.streams):
        stream = torch.Generator().manual_seed(1_000_003 + 7919 * index)
        stats = paired_lift(
            evaluate(bench.make_env(deck), net, episodes=args.episodes,
                     seeds=seeds, greedy=False, generator=stream), control)
        lifts.append(stats["lift"])
        print(f"  sampled, stream {index}: {stats['lift']:+.4f} "
              f"[{stats['ci_low']:+.3f}, {stats['ci_high']:+.3f}]  "
              f"win {stats['win']:.0%}", flush=True)

    lifts = np.asarray(lifts)
    spread = float(lifts.std(ddof=1)) if len(lifts) > 1 else 0.0
    print(f"\n  {len(lifts)} streams: mean {lifts.mean():+.4f}, "
          f"sd {spread:.4f}, range {lifts.max() - lifts.min():.4f}")
    print(f"  Two runs of this checkpoint can therefore differ by about "
          f"{1.96 * spread * 2**0.5:.3f} sd at 95%, on identical battles "
          f"against an identical control.")
    print(f"\n{(time.perf_counter() - started) / 60:.1f} min", flush=True)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "checkpoint": str(args.checkpoint),
            "head": payload.get("head"), "episodes": args.episodes,
            "eval_opponent": faced, "greedy": first["lift"],
            "greedy_repeat": second["lift"],
            "sampled_lifts": [float(v) for v in lifts],
            "sampled_sd": spread,
            "sampled_range": float(lifts.max() - lifts.min()),
        }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
