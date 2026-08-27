"""Measure the searching bot and record it where the training page can see it.

The expert is not a training run -- it does not learn, and its numbers do not
move -- but it is the only thing on this project that reliably beats the
random control, so it belongs on the same page as the runs trying to reach it.
Written as a flat line on purpose: a constant is what it is, and a reader
should be able to see how far a learning curve is from it.

    python scripts/measure_expert.py --episodes 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cr_sim.api.env import CRSimEnv
from cr_sim.api.reward import ProjectionWeights
from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.entity import Team
from cr_sim.train.run import DEFAULT_BUILD, DEFAULT_DECK, _random_opponent
from cr_sim.train.scripted import SearchBot, SearchBotConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="measure-expert")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--tower-level", type=int, default=5)
    parser.add_argument("--horizon-seconds", type=float, default=15.0)
    parser.add_argument("--candidates", type=int, default=18)
    parser.add_argument("--out", type=Path, default=Path("runs/search-expert"))
    args = parser.parse_args(argv)

    data = LogicData.load(DEFAULT_BUILD)
    levels, registry = build_level_table(data), build_card_registry(data)

    def make_env(seed: int) -> CRSimEnv:
        return CRSimEnv(
            data, levels, registry, DEFAULT_DECK, DEFAULT_DECK,
            ticks_per_second=20, frame_skip=30, max_ticks=20 * 120,
            tower_level=args.tower_level, reward_shaping_weight=0.01,
            reward_weights=ProjectionWeights(horizon_seconds=3.0),
            opponent_policy=_random_opponent(70_000 + seed))

    def run(agent, seed: int) -> tuple[int, float]:
        env = make_env(seed)
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        slots, width, height = (int(v) for v in env.action_space.nvec)
        total = 0.0
        while True:
            mask = env.legal_action_mask()
            flat = mask.reshape(-1)
            if agent is None:
                index = int(rng.choice(np.flatnonzero(flat))) if flat.any() else 0
                slot, remainder = divmod(index, width * height)
                gx, gy = divmod(remainder, height)
                action = (min(slot, slots - 1), gx, gy)
            else:
                action = agent(None, mask, env.battle)
            _, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        battle = env.battle
        crowns = (battle.players[Team.BLUE].crowns
                  - battle.players[Team.RED].crowns)
        return crowns, total

    started = time.perf_counter()
    print(f"{args.episodes} paired battles, tower level {args.tower_level}",
          flush=True)
    control = [run(None, seed) for seed in range(args.episodes)]
    bot = SearchBot(Team.BLUE, SearchBotConfig(
        horizon_seconds=args.horizon_seconds, candidates=args.candidates, seed=3))
    expert = [run(bot, seed) for seed in range(args.episodes)]

    control_crowns = np.array([c for c, _ in control])
    control_returns = np.array([r for _, r in control])
    crowns = np.array([c for c, _ in expert])
    returns = np.array([r for _, r in expert])
    spread = control_returns.std(ddof=1) or 1.0
    difference = returns - control_returns
    error = difference.std(ddof=1) / np.sqrt(len(difference))
    lift = float(difference.mean() / spread)
    low = float((difference.mean() - 1.96 * error) / spread)
    high = float((difference.mean() + 1.96 * error) / spread)

    print(f"{'arm':<18}{'win':>8}{'loss':>8}{'draw':>8}{'lift sd':>10}{'95% CI':>20}")
    print(f"{'random control':<18}{np.mean(control_crowns > 0):>8.0%}"
          f"{np.mean(control_crowns < 0):>8.0%}{np.mean(control_crowns == 0):>8.0%}"
          f"{'--':>10}{'--':>20}")
    print(f"{'search expert':<18}{np.mean(crowns > 0):>8.0%}"
          f"{np.mean(crowns < 0):>8.0%}{np.mean(crowns == 0):>8.0%}"
          f"{lift:>+10.3f}   [{low:+.3f}, {high:+.3f}]")

    # Written as a flat series so the live page can draw it as the line every
    # learning curve is trying to reach. It does not learn, so it does not
    # move, and showing it as a constant is the honest shape.
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({
        "reward": "none (search)", "opponent": "random",
        "tower_level": args.tower_level, "frame_skip": 30, "tps": 20,
        "match_seconds": 120, "num_envs": 0, "horizon": 0,
        "note": (f"One-ply search over the simulator, {args.candidates} "
                 f"candidate placements each played {args.horizon_seconds:.0f}s "
                 "forward. No training, no gradients, no reward design -- it "
                 "asks the engine what happens and keeps the best answer."),
    }, indent=2), encoding="utf-8")

    row = {
        "updates": 1, "steps": 0, "episodes": args.episodes,
        "steps_per_second": 0.0, "entropy": 0.0, "value_loss": 0.0,
        "policy_loss": 0.0, "mean_return": float(returns.mean()),
        "win_rate": float(np.mean(crowns > 0)), "noop_fraction": 0.0,
        "eval_lift_sd": lift, "eval_win": float(np.mean(crowns > 0)),
        "control_win": float(np.mean(control_crowns > 0)),
        "eval_return": float(returns.mean()),
        "control_return": float(control_returns.mean()),
    }
    with (args.out / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for update in (1, 2):
            stream.write(json.dumps({**row, "updates": update}) + "\n")
    (args.out / "verdict.json").write_text(json.dumps({
        "episodes": args.episodes, "lift": lift,
        "ci_low": low, "ci_high": high,
        "win": float(np.mean(crowns > 0)), "loss": float(np.mean(crowns < 0)),
        "note": "One-ply search. Not a learned policy -- the bar, not a result.",
    }, indent=2), encoding="utf-8")
    print(f"\n{(time.perf_counter() - started) / 60:.1f} min -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
