"""Score a checkpoint against the search expert instead of against random.

The yardstick this project has used from the beginning is lift over a uniform
random control, and it is spent. The one-ply search expert already beats that
control 100-0 over 40 paired battles, +2.716 sd, so there is essentially no
headroom left in the number: a better expert, or a better clone of one, has
nowhere to register. A saturated metric is worse than a noisy one, because a
noisy one at least looks wrong -- a saturated one reports a high figure
forever and stops responding to the thing it is supposed to be measuring.

So put the expert on the other side of the net. The structure is unchanged:
both arms play the same fixed seeds, the lift is still the paired per-battle
difference against the uniform random control in control standard deviations,
and ``verdict.json`` keeps its shape, so a result from here reads like a
result from scripts/clone_policy.py. What changes is that the control now
loses every battle instead of winning a quarter of them, which is where a
control belongs and where the scale has room above it.

Greedy and sampled are reported separately and never merged. They are not the
same policy -- the clone measures +1.623 greedy and +0.709 sampled against the
same control -- so a change can leave the argmax untouched while moving the
whole distribution around it, and a single headline number has already caused
one run to be read as a regression it was not.

    python scripts/evaluate_vs_expert.py runs/cloned/cloned.pt --episodes 60

Slow, and honestly so: every opponent decision branches the simulator
``--candidates`` times and plays each branch fifteen seconds forward. Budget
roughly an order of magnitude more wall clock than the same evaluation against
the random control.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cr_sim.api.encoding import parse_observation
from cr_sim.api.env import CRSimEnv
from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.train.evaluate import (
    evaluate_paired, evaluation_seeds, load_policy, search_opponent,
    write_verdict,
)
from cr_sim.train.run import DEFAULT_BUILD, DEFAULT_DECK, _random_opponent
from cr_sim.train.scripted import SearchBotConfig
from cr_sim.train.selfplay import (check_lift_is_named, opponent_name,
                                   reward_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evaluate-vs-expert")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument(
        "--tower-level", type=int, default=11,
        help="Crown Tower level. 11, agreeing with cr_sim.train.evaluate's "
             "CLI and with cr_sim.train.run -- this defaulted to 5 while both "
             "of those defaulted to 11, which is two evaluation entry points "
             "quietly playing in different arenas. That class of mismatch "
             "already trained a whole run at level 11 while config.json "
             "recorded 5. Pass --tower-level 5 explicitly for the arena the "
             "clones were measured in.")
    parser.add_argument("--tps", type=int, default=20)
    parser.add_argument("--frame-skip", type=int, default=30)
    parser.add_argument("--match-seconds", type=int, default=120)
    parser.add_argument(
        "--opponent", choices=("search", "random"), default="search",
        help="'search' is the point of this script; 'random' is here so the "
             "same command can produce the old measurement on the same seeds, "
             "which is the only way to say how much of a difference between "
             "two verdicts is the opponent and how much is the policy.")
    parser.add_argument("--candidates", type=int, default=18,
                        help="placements the expert evaluates per decision. "
                             "Matched to scripts/measure_expert.py's default, "
                             "so the opponent here is the bot that was measured "
                             "at +2.716 sd and not a weaker relative of it.")
    parser.add_argument("--horizon-seconds", type=float, default=15.0,
                        help="how far the expert plays each branch forward. "
                             "Measured against a random opponent: 4s wins 31%%, "
                             "8s wins 94%%, 15s wins 100%%.")
    parser.add_argument(
        "--block", type=int, default=0,
        help="which rotating seed block to play. Block 0 is the seed list "
             "every existing measurement on this project used; another block "
             "is a disjoint set of battles, and running two is how to tell a "
             "result from one seed set's luck.")
    parser.add_argument(
        "--observation", default=None,
        help="which observation to build the environment with. Defaults to "
             "whatever the checkpoint says it was trained on, which is the "
             "only choice that can be right.")
    parser.add_argument(
        "--modes", nargs="+", default=["greedy", "sampled"],
        choices=("greedy", "sampled"),
        help="which arms to play. Both by default: they are two different "
             "policies and neither substitutes for the other.")
    parser.add_argument("--out", type=Path, default=None,
                        help="run directory to write. Defaults to "
                             "runs/vs-<opponent>-<checkpoint's run name>.")
    args = parser.parse_args(argv)

    build = LogicData.load(DEFAULT_BUILD)
    levels, registry = build_level_table(build), build_card_registry(build)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    observation = parse_observation(args.observation
                                    or str(payload.get("observation", "v1")))
    expert = SearchBotConfig(horizon_seconds=args.horizon_seconds,
                             candidates=args.candidates)

    def make_env() -> CRSimEnv:
        # A fresh opponent per environment. The two arms each hold their own,
        # so neither advances the other's generator; and the search opponent
        # rebuilds itself per battle from the battle's seed, so both arms face
        # the same expert on the same seed rather than merely the same kind of
        # expert. See cr_sim.train.evaluate.search_opponent.
        opponent = (search_opponent(expert) if args.opponent == "search"
                    else _random_opponent(60_000))
        return CRSimEnv(
            build, levels, registry, DEFAULT_DECK, DEFAULT_DECK,
            ticks_per_second=args.tps, frame_skip=args.frame_skip,
            max_ticks=args.tps * args.match_seconds,
            tower_level=args.tower_level, observation=observation,
            opponent_policy=opponent)

    seeds = evaluation_seeds(args.episodes, block=args.block)
    net = load_policy(args.checkpoint, make_env())

    started = time.perf_counter()
    faced = opponent_name(make_env())
    # The scale the lift is denominated in, read off the same
    # environment. See check_lift_is_named.
    scale = reward_name(make_env())
    print(f"{args.episodes} paired battles against the {faced} opponent, "
          f"seed block {args.block}", flush=True)
    verdict = evaluate_paired(make_env, net, episodes=args.episodes,
                              seeds=seeds, modes=tuple(args.modes))
    verdict["block"] = int(args.block)
    verdict["checkpoint"] = str(args.checkpoint)
    verdict["head"] = str(payload.get("head", "flat"))
    verdict["observation"] = str(payload.get("observation", "v1"))
    if args.opponent == "search":
        verdict["expert"] = {"candidates": args.candidates,
                             "horizon_seconds": args.horizon_seconds}

    print(f"\n{'arm':<20}{'win':>8}{'loss':>8}{'draw':>8}{'lift sd':>10}"
          f"{'95% CI':>22}")
    control = verdict["control"]
    print(f"{'random control':<20}{control['win']:>8.0%}{control['loss']:>8.0%}"
          f"{control['draw']:>8.0%}{'--':>10}{'--':>22}")
    # Guarded, the way cr_sim.train.evaluate's own CLI guards it. `modes` is
    # optional in evaluate_paired, so a --modes greedy caller used to get a
    # KeyError here -- *after* paying for every battle.
    for mode in ("greedy", "sampled"):
        arm = verdict.get(mode)
        if arm is None:
            continue
        print(f"{'trained, ' + mode:<20}{arm['win']:>8.0%}{arm['loss']:>8.0%}"
              f"{arm['draw']:>8.0%}{arm['lift']:>+10.3f}"
              f"   [{arm['ci_low']:+.3f}, {arm['ci_high']:+.3f}]", flush=True)

    out = args.out or Path("runs") / f"vs-{faced}-{args.checkpoint.parent.name}"
    out.mkdir(parents=True, exist_ok=True)
    # Refuses to write a lift that does not say who it was measured against.
    # A verdict against the expert and one against random are no more
    # comparable than the idle and random scales were, and that confusion has
    # already cost this project two rounds of invalid comparisons.
    write_verdict(out / "verdict.json", verdict)

    # Registered as a run so it lands beside the runs it is meant to be read
    # against. Flat, because a checkpoint does not learn while being scored --
    # drawing it as a curve would imply a trajectory it does not have.
    (out / "config.json").write_text(json.dumps({
        "reward": "evaluation only", "opponent": faced,
        "eval_opponent": faced, "eval_episodes": args.episodes,
        "tower_level": args.tower_level, "frame_skip": args.frame_skip,
        "tps": args.tps, "match_seconds": args.match_seconds,
        "num_envs": 0, "horizon": 0,
        "note": (f"{args.checkpoint} scored over {args.episodes} paired "
                 f"battles against the {faced} opponent on seed block "
                 f"{args.block}. "
                 + ", ".join(f"{m} {verdict[m]['lift']:+.3f} sd"
                             for m in ("greedy", "sampled") if m in verdict)
                 + ". The random "
                 "control faces the same opponent on the same seeds, so the "
                 "lift is on a different scale from every lift measured "
                 "against a random opponent -- not a better or worse number, "
                 "a different one."),
    }, indent=2), encoding="utf-8")

    row = {
        "updates": 1, "steps": 0, "episodes": args.episodes,
        "steps_per_second": 0.0, "entropy": 0.0, "value_loss": 0.0,
        "policy_loss": 0.0, "mean_return": verdict[verdict["mode"]]["return"],
        "win_rate": verdict["win"], "noop_fraction": 0.0,
        "eval_lift_sd": verdict["lift"], "eval_win": verdict["win"],
        "control_win": control["win"],
        "eval_return": verdict[verdict["mode"]]["return"],
        "control_return": control["return"],
        # Both arms on the row where both were played, because the headline
        # is one of them and a reader cannot tell which way a change moved
        # without the other. Absent rather than invented where an arm was not
        # played at all.
        **{f"eval_lift_sd_{m}": verdict[m]["lift"]
           for m in ("greedy", "sampled") if m in verdict},
        "eval_mode": verdict["mode"],
        "eval_block": int(args.block),
        # Named, because a lift compared against one measured on a different
        # opponent is not a comparison. See
        # cr_sim.train.selfplay.check_lift_is_named.
        "eval_opponent": faced, "eval_reward": scale,
        "eval_episodes": args.episodes,
        "observation": verdict["observation"], "head": verdict["head"],
    }
    check_lift_is_named(row)
    with (out / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for update in (1, 2):
            stream.write(json.dumps({**row, "updates": update}) + "\n")

    # The denominator, printed rather than left implicit. Against an opponent
    # this strong the control loses every battle in much the same way, and a
    # collapsed spread inflates every lift divided by it -- the raw returns
    # beside it are what a suspicious reading gets checked against.
    print(f"\ncontrol return {control['return']:+.4f} +/- {control['spread']:.4f}"
          f"  <- the unit the lift is measured in")
    print(f"headline: {verdict['lift']:+.3f} sd ({verdict['mode']}) against "
          f"the {faced} opponent")
    print(f"{(time.perf_counter() - started) / 60:.1f} min -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
