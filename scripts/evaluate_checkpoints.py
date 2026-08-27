"""Play a saved policy against the random control over paired seeds.

Separate from training so a checkpoint can be scored without being retrained,
and so several checkpoints can be scored against the *same* control arm --
which is what makes their intervals comparable rather than merely similar.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cr_sim.api.encoding import parse_observation
from cr_sim.api.env import CRSimEnv
from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.train.evaluate import evaluate
from cr_sim.train.nets import ActorCritic, net_config_for
from cr_sim.train.run import DEFAULT_BUILD, DEFAULT_DECK, _random_opponent
from cr_sim.train.selfplay import opponent_name

p = argparse.ArgumentParser()
p.add_argument("checkpoints", nargs="+")
p.add_argument("--episodes", type=int, default=150)
p.add_argument("--tower-level", type=int, default=5)
p.add_argument("--seed", type=int, default=777)
p.add_argument("--out", default=None)
args = p.parse_args()

build = LogicData.load(DEFAULT_BUILD)
levels, registry = build_level_table(build), build_card_registry(build)
seeds = [int(s) for s in
         np.random.default_rng(args.seed).integers(0, 2**31 - 1, args.episodes)]


def make_env(observation):
    return CRSimEnv(build, levels, registry, DEFAULT_DECK, DEFAULT_DECK,
                    ticks_per_second=20, frame_skip=30, max_ticks=20 * 120,
                    tower_level=args.tower_level,
                    observation=parse_observation(observation),
                    opponent_policy=_random_opponent(60_000))


started = time.perf_counter()
# One control arm, shared. Every checkpoint is differenced against the same
# battles, so two lifts differ by the policies and not by which seeds each
# happened to draw.
control_env = make_env("v1")
control = evaluate(control_env, None, episodes=args.episodes, seeds=seeds)
control_returns = np.asarray(control["returns"])
control_crowns = np.asarray(control["crowns"])
spread = control_returns.std(ddof=1) or 1.0
print(f"control ({opponent_name(control_env)} opponent, {args.episodes} battles): "
      f"win {np.mean(control_crowns > 0):.0%} loss {np.mean(control_crowns < 0):.0%} "
      f"draw {np.mean(control_crowns == 0):.0%}   [{time.perf_counter() - started:.0f}s]",
      flush=True)
print(f"{'checkpoint':<28}{'mode':>8}{'win':>7}{'loss':>7}{'draw':>7}"
      f"{'lift sd':>10}{'95% CI':>22}")

rows = []
for path in args.checkpoints:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    observation = str(payload.get("observation", "v1"))
    env = make_env(observation)
    env.reset(seed=0)
    net = ActorCritic(net_config_for(env, head=payload.get("head", "flat")))
    net.load_state_dict(payload["state_dict"])
    net.eval()
    for mode in ("greedy", "sampled"):
        result = evaluate(make_env(observation), net, episodes=args.episodes,
                          seeds=seeds, greedy=(mode == "greedy"))
        crowns = np.asarray(result["crowns"])
        difference = np.asarray(result["returns"]) - control_returns
        error = difference.std(ddof=1) / np.sqrt(len(difference))
        lift = difference.mean() / spread
        low, high = ((difference.mean() - 1.96 * error) / spread,
                     (difference.mean() + 1.96 * error) / spread)
        name = Path(path).parent.name
        print(f"{name:<28}{mode:>8}{np.mean(crowns > 0):>7.0%}"
              f"{np.mean(crowns < 0):>7.0%}{np.mean(crowns == 0):>7.0%}"
              f"{lift:>+10.3f}   [{low:+.3f}, {high:+.3f}]", flush=True)
        rows.append({"checkpoint": str(path), "name": name, "mode": mode,
                     "observation": observation,
                     "head": payload.get("head", "flat"),
                     "win": float(np.mean(crowns > 0)),
                     "loss": float(np.mean(crowns < 0)),
                     "lift": float(lift), "ci_low": float(low),
                     "ci_high": float(high), "episodes": args.episodes,
                     "eval_opponent": opponent_name(env)})

if args.out:
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"{(time.perf_counter() - started) / 60:.1f} min")
