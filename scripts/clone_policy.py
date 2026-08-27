"""Train a policy to copy the searching bot, then measure whether it worked.

The bootstrap AlphaStar did with 971,000 human replays and this project did
not do at all. Its supervised agent alone outranked 84% of human players
before any reinforcement learning; the reinforcement learning that followed
refined a competent policy rather than creating one from noise.

The measurement that matters is at the bottom and is deliberately harsh. Two
numbers can look like success and mean nothing:

*   **Agreement** on held-out states says the policy learned the expert, not
    that it can play. A policy agreeing 60% of the time may still lose every
    match, because the 40% it gets wrong are the decisions that mattered.
*   **Winning** is the claim worth making, so the cloned policy is played
    against the same random control the whole project has been measured
    against, over enough battles to mean something.

    python scripts/clone_policy.py --demos data_cache/demos --out runs/cloned
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

from cr_sim.api.env import CRSimEnv
from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.entity import Team
from cr_sim.train.clone import CloneConfig, Demonstrations, clone
from cr_sim.train.evaluate import evaluate
from cr_sim.train.nets import ActorCritic, NetConfig
from cr_sim.train.run import DEFAULT_BUILD, DEFAULT_DECK, _random_opponent


def merge(paths: list[Path]) -> Demonstrations:
    parts = [Demonstrations.load(p) for p in paths]
    if not parts:
        raise SystemExit("no shards found")
    return Demonstrations(
        grid=np.concatenate([p.grid for p in parts]),
        vector=np.concatenate([p.vector for p in parts]),
        mask=np.concatenate([p.mask for p in parts]),
        action=np.concatenate([p.action for p in parts]),
        value=np.concatenate([p.value for p in parts]),
        episodes=sum(p.episodes for p in parts),
        play_rate=float(np.mean([p.play_rate for p in parts])),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clone-policy")
    parser.add_argument("--demos", type=Path, default=Path("data_cache/demos"))
    parser.add_argument("--out", type=Path, default=Path("runs/cloned"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--pass-weight", type=float, default=0.1,
        help="how much a 'play nothing' decision counts against a placement. "
             "At 1.0 the policy learns to always pass -- nearly half the "
             "expert's decisions are passes while the rest are spread over "
             "seven hundred placements -- and greedy play then loses every "
             "match.",
    )
    parser.add_argument("--tower-level", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=120,
                        help="battles in the final evaluation")
    args = parser.parse_args(argv)

    shards = sorted(args.demos.glob("shard-*.npz"))
    data = merge(shards)
    print(f"{len(shards)} shards, {len(data):,} decisions from "
          f"{data.episodes} episodes, expert played on "
          f"{data.play_rate:.0%} of them\n", flush=True)

    build = LogicData.load(DEFAULT_BUILD)
    levels, registry = build_level_table(build), build_card_registry(build)

    def make_env(seed_offset: int = 0) -> CRSimEnv:
        return CRSimEnv(
            build, levels, registry, DEFAULT_DECK, DEFAULT_DECK,
            ticks_per_second=20, frame_skip=30, max_ticks=20 * 120,
            tower_level=args.tower_level,
            opponent_policy=_random_opponent(60_000 + seed_offset))

    probe = make_env()
    observation, _ = probe.reset(seed=0)
    slots, width, height = (int(v) for v in probe.action_space.nvec)
    net = ActorCritic(NetConfig(
        grid_channels=observation["grid"].shape[0],
        grid_height=observation["grid"].shape[1],
        grid_width=observation["grid"].shape[2],
        vector_size=observation["vector"].shape[0],
        num_actions=slots * width * height))

    started = time.perf_counter()
    history: list[dict] = []

    def report(stats: dict) -> None:
        history.append(stats)
        print(f"epoch {stats['epoch']:>3}  policy {stats['policy_loss']:.4f}  "
              f"value {stats['value_loss']:.4f}  "
              f"agree {stats['agreement']:.1%}  "
              f"on plays {stats['play_agreement']:.1%}  "
              f"plays {stats['play_rate']:.1%}  "
              f"expl var {stats['explained_variance']:+.3f}", flush=True)

    clone(net, data, CloneConfig(
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, pass_weight=args.pass_weight,
        pass_action=(slots - 1) * width * height, seed=0), on_epoch=report)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(),
                "clone": history[-1] if history else {}},
               args.out / "cloned.pt")

    # The claim worth making. Agreement says the expert was learned; this says
    # whether the result can play, against the same control every other number
    # on this project was measured against.
    print(f"\n{args.episodes} paired battles against the random control:",
          flush=True)
    seeds = [int(s) for s in
             np.random.default_rng(777).integers(0, 2**31 - 1, args.episodes)]
    control = evaluate(make_env(), None, episodes=args.episodes, seeds=seeds)
    trained = evaluate(make_env(), net, episodes=args.episodes, seeds=seeds,
                       greedy=False)
    greedy = evaluate(make_env(), net, episodes=args.episodes, seeds=seeds,
                      greedy=True)

    control_returns = np.asarray(control["returns"])
    spread = control_returns.std(ddof=1) or 1.0
    print(f"{'arm':<20}{'win':>8}{'loss':>8}{'draw':>8}{'lift sd':>10}"
          f"{'95% CI':>20}")

    def line(label: str, result) -> dict:
        crowns = np.asarray(result["crowns"])
        difference = np.asarray(result["returns"]) - control_returns
        error = difference.std(ddof=1) / np.sqrt(len(difference))
        lift = difference.mean() / spread
        low, high = ((difference.mean() - 1.96 * error) / spread,
                     (difference.mean() + 1.96 * error) / spread)
        print(f"{label:<20}{np.mean(crowns > 0):>8.0%}{np.mean(crowns < 0):>8.0%}"
              f"{np.mean(crowns == 0):>8.0%}{lift:>+10.3f}"
              f"   [{low:+.3f}, {high:+.3f}]", flush=True)
        return {"win": float(np.mean(crowns > 0)),
                "loss": float(np.mean(crowns < 0)),
                "lift": float(lift), "ci_low": float(low), "ci_high": float(high)}

    control_crowns = np.asarray(control["crowns"])
    print(f"{'random control':<20}{np.mean(control_crowns > 0):>8.0%}"
          f"{np.mean(control_crowns < 0):>8.0%}"
          f"{np.mean(control_crowns == 0):>8.0%}{'--':>10}{'--':>20}")
    sampled_stats = line("cloned, sampled", trained)
    greedy_stats = line("cloned, greedy", greedy)
    best = greedy_stats if greedy_stats["lift"] >= sampled_stats["lift"] else sampled_stats
    verdict = {"episodes": args.episodes,
               "sampled": sampled_stats, "greedy": greedy_stats,
               # Flattened as well as nested: the report reads these keys, and
               # the honest headline is whichever way of playing scored better.
               "lift": best["lift"], "ci_low": best["ci_low"],
               "ci_high": best["ci_high"], "win": best["win"],
               "loss": best["loss"],
               "clone": history[-1] if history else {}}
    (args.out / "verdict.json").write_text(json.dumps(verdict, indent=2),
                                           encoding="utf-8")

    # Registered as a run so it appears beside the training runs it is meant
    # to be compared against. A flat series, because a clone does not learn
    # over time -- it is a single result, and drawing it as a curve would
    # imply a trajectory it does not have.
    (args.out / "config.json").write_text(json.dumps({
        "reward": "behavioural cloning", "opponent": "random",
        "tower_level": args.tower_level, "frame_skip": 30, "tps": 20,
        "match_seconds": 120, "num_envs": 0, "horizon": 0,
        "note": (f"Cloned from {data.episodes} expert episodes "
                 f"({len(data):,} decisions). Trained on the search's value "
                 "distribution over candidate placements, not the move it "
                 "played -- the bot samples its candidates, so its choice is "
                 "not a function of the state."),
    }, indent=2), encoding="utf-8")
    last = history[-1] if history else {}
    row = {
        "updates": 1, "steps": 0, "episodes": args.episodes,
        "steps_per_second": 0.0,
        "entropy": 0.0, "value_loss": last.get("value_loss", 0.0),
        "policy_loss": last.get("policy_loss", 0.0),
        "explained_variance": last.get("explained_variance"),
        "mean_return": 0.0, "win_rate": best["win"],
        "noop_fraction": 1.0 - last.get("play_rate", 0.0),
        "eval_lift_sd": best["lift"], "eval_win": best["win"],
        "control_win": float(np.mean(control_crowns > 0)),
    }
    with (args.out / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for update in (1, 2):
            stream.write(json.dumps({**row, "updates": update}) + chr(10))
    print(f"\n{(time.perf_counter() - started) / 60:.1f} min -> {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
