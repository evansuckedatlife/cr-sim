"""Measure the searching bot on the scale everything else is quoted against.

The expert is not a training run -- it does not learn, and its numbers do not
move -- but it is the only thing on this project that reliably beats the
random control, so it belongs on the same page as the runs trying to reach it.
Written as a flat line on purpose: a constant is what it is, and a reader
should be able to see how far a learning curve is from it.

    python scripts/measure_expert.py --episodes 150

**The number this replaces was not on the scale it was quoted beside.** The
+2.716 [+2.369, +3.063] anchor was measured on ``range(40)`` -- literally
seeds 0 to 39 -- at n=40, under ``ProjectionWeights(horizon_seconds=3.0)``.
The clone's +2.167 it is compared against was measured on 150 seeds drawn from
``default_rng(777)`` under the ordinary crown-plus-tower-health evaluation
reward. Different battles, a different n, and a different numerator *and*
denominator: the lift's unit is the control's spread, and two different
rewards give the control two different spreads. HANDOFF's claim that all the
arms sit on "the same 150 paired seeds" is true for seven of them and false
for this one.

So: ``evaluation_seeds()``, the evaluation reward, ``paired_lift`` for the
arithmetic and ``write_verdict`` for the file -- the same four things every
other measurement here goes through. The number will move, and it should: the
old one was answering a different question.
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
from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.entity import Team
from cr_sim.train.evaluate import (
    Result, evaluation_seeds, paired_lift, write_verdict,
)
from cr_sim.train.run import DEFAULT_BUILD, DEFAULT_DECK, _random_opponent
from cr_sim.train.scripted import SearchBot, SearchBotConfig
from cr_sim.train.selfplay import (check_lift_is_named, opponent_name,
                                   reward_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="measure-expert")
    parser.add_argument(
        "--episodes", type=int, default=150,
        help="paired battles per arm. 150 is what every other arm on this "
             "machine was measured over; 40 -- the old default -- resolves "
             "to about +/-0.35 sd, which is fourteen times wider than the "
             "band of results it was being used to separate.")
    parser.add_argument("--tower-level", type=int, default=5)
    parser.add_argument("--tps", type=int, default=20)
    parser.add_argument("--frame-skip", type=int, default=30)
    parser.add_argument("--match-seconds", type=int, default=120)
    parser.add_argument("--horizon-seconds", type=float, default=15.0)
    parser.add_argument("--candidates", type=int, default=18)
    parser.add_argument(
        "--block", type=int, default=0,
        help="which rotating seed block to play. Block 0 is the list every "
             "existing measurement used.")
    parser.add_argument("--out", type=Path, default=Path("runs/search-expert"))
    args = parser.parse_args(argv)

    data = LogicData.load(DEFAULT_BUILD)
    levels, registry = build_level_table(data), build_card_registry(data)

    def make_env(seed: int) -> CRSimEnv:
        # The evaluation reward -- crown difference plus a hundredth of the
        # tower-health difference -- and not the projected one this script
        # used to build. A lift is measured in control standard deviations,
        # so changing the reward changes the unit, and a number in a
        # different unit cannot be read beside the others whatever its sign.
        return CRSimEnv(
            data, levels, registry, DEFAULT_DECK, DEFAULT_DECK,
            ticks_per_second=args.tps, frame_skip=args.frame_skip,
            max_ticks=args.tps * args.match_seconds,
            tower_level=args.tower_level,
            opponent_policy=_random_opponent(70_000 + seed))

    config = SearchBotConfig(horizon_seconds=args.horizon_seconds,
                             candidates=args.candidates)

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

    def expert(seed: int) -> tuple[int, float]:
        # A fresh bot per battle, reseeded from the battle's own seed. The bot
        # *samples* its candidate placements, so a single instance plays a
        # function of how far its generator has been advanced -- which is a
        # count of every decision in every episode before this one. Rebuilt,
        # it is a function of the seed alone, which is what makes this
        # reproducible and what lets a second run of the same block be
        # compared to the first.
        derived = (3 * 1_000_003 + seed) % (2 ** 31 - 1)
        from dataclasses import replace

        return run(SearchBot(Team.BLUE, replace(config, seed=derived)), seed)

    seeds = evaluation_seeds(args.episodes, block=args.block)
    started = time.perf_counter()
    print(f"{args.episodes} paired battles on seed block {args.block}, "
          f"tower level {args.tower_level}", flush=True)
    control_pairs = [run(None, seed) for seed in seeds]
    expert_pairs = [expert(seed) for seed in seeds]

    control = Result(returns=[r for _, r in control_pairs],
                     crowns=[c for c, _ in control_pairs])
    measured = Result(returns=[r for _, r in expert_pairs],
                      crowns=[c for c, _ in expert_pairs])
    # The one arithmetic, not a fourth spelling of it. paired_lift takes the
    # difference per battle and only then averages, which is the whole point
    # of handing both arms the same seeds.
    arm = paired_lift(measured, control)
    control_crowns = np.asarray(control["crowns"], dtype=float)
    faced = opponent_name(make_env(0))
    # The unit the lift is denominated in, read off the same env.
    scale = reward_name(make_env(0))

    print(f"{'arm':<18}{'win':>8}{'loss':>8}{'draw':>8}{'lift sd':>10}"
          f"{'95% CI':>20}")
    print(f"{'random control':<18}{np.mean(control_crowns > 0):>8.0%}"
          f"{np.mean(control_crowns < 0):>8.0%}"
          f"{np.mean(control_crowns == 0):>8.0%}{'--':>10}{'--':>20}")
    print(f"{'search expert':<18}{arm['win']:>8.0%}{arm['loss']:>8.0%}"
          f"{arm['draw']:>8.0%}{arm['lift']:>+10.3f}"
          f"   [{arm['ci_low']:+.3f}, {arm['ci_high']:+.3f}]")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({
        "reward": "evaluation only", "opponent": faced,
        "eval_opponent": faced, "eval_episodes": args.episodes,
        "tower_level": args.tower_level, "frame_skip": args.frame_skip,
        "tps": args.tps, "match_seconds": args.match_seconds,
        "num_envs": 0, "horizon": 0,
        "note": (f"One-ply search over the simulator, {args.candidates} "
                 f"candidate placements each played {args.horizon_seconds:.0f}s "
                 "forward. No training, no gradients, no reward design -- it "
                 "asks the engine what happens and keeps the best answer. "
                 f"Measured over {args.episodes} paired battles on "
                 f"evaluation_seeds block {args.block} under the evaluation "
                 "reward, which is the seed set and the unit every other arm "
                 "here was measured in. The +2.716 this replaces was 40 "
                 "battles on seeds 0-39 under a projected reward, and was "
                 "never on the same scale as the clone it was quoted beside."),
    }, indent=2), encoding="utf-8")

    verdict = {
        "episodes": args.episodes,
        # Who the two arms played, read off a real environment. Not the arm:
        # the expert is what is being *measured* here, the random opponent is
        # what it was measured against, and putting the arm in this field is
        # the exact confusion the guard exists to refuse.
        "eval_opponent": faced,
        "arm": f"search-c{args.candidates}h{args.horizon_seconds:g}",
        "seeds": seeds,
        "block": int(args.block),
        "tower_level": args.tower_level,
        "reward": "evaluation (crowns + 0.01 * tower health)",
        "expert": {"candidates": args.candidates,
                   "horizon_seconds": args.horizon_seconds},
        "control": {
            "win": float(np.mean(control_crowns > 0)),
            "loss": float(np.mean(control_crowns < 0)),
            "draw": float(np.mean(control_crowns == 0)),
            "return": float(np.mean(control["returns"])),
            "crowns": float(control_crowns.mean()),
            "spread": (float(np.asarray(control["returns"], dtype=float)
                             .std(ddof=1)) if args.episodes > 1 else 0.0),
            # Per battle, so the arithmetic above is re-checkable from the
            # file and so the reward this was measured under is readable off
            # it rather than taken on trust from the "reward" string beside
            # it. Under the evaluation reward a return telescopes to the
            # crown difference plus at most a hundredth; under the projected
            # one it does not, and that is a fact about the numbers rather
            # than about the label.
            "returns": [float(v) for v in control["returns"]],
            "crowns": [int(v) for v in control["crowns"]],
        },
        "greedy": dict(arm),
        "mode": "greedy",
        "note": "One-ply search. Not a learned policy -- the bar, not a result.",
        **{k: arm[k] for k in ("lift", "ci_low", "ci_high", "win", "loss")},
    }
    write_verdict(args.out / "verdict.json", verdict)

    row = check_lift_is_named({
        "updates": 1, "steps": 0, "episodes": args.episodes,
        "steps_per_second": 0.0, "entropy": 0.0, "value_loss": 0.0,
        "policy_loss": 0.0, "mean_return": float(np.mean(measured["returns"])),
        "win_rate": arm["win"], "noop_fraction": 0.0,
        "eval_lift_sd": arm["lift"], "eval_win": arm["win"],
        "control_win": float(np.mean(control_crowns > 0)),
        "eval_return": arm["return"],
        "control_return": float(np.mean(control["returns"])),
        "eval_opponent": faced, "eval_reward": scale,
        "eval_episodes": args.episodes,
        "eval_block": int(args.block),
    })
    with (args.out / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for update in (1, 2):
            stream.write(json.dumps({**row, "updates": update}) + "\n")
    print(f"\n{(time.perf_counter() - started) / 60:.1f} min -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
