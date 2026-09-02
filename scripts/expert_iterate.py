"""One turn of expert iteration: propose, search, clone, rate, repeat.

    python scripts/expert_iterate.py --rounds 2 --seed-policy runs/clone-v1-paired/cloned.pt

**The loop this project had two thirds of.** AlphaZero's improvement operator
is three arrows -- the policy proposes, the search refines the proposal, and
the refined distribution trains the policy. ``SearchBot`` is the second and
:func:`cr_sim.train.clone.collect` is the third. The first has never existed
here: the search drew about fourteen stratified-random placements out of a mean
of 104 legal ones, 13.5% coverage, and the network's opinion about which
fourteen were worth an exact engine branch was never consulted.

**Why it closes here and did not close in the papers this imitates.** Expert
iteration is normally unaffordable because the expert is expensive. Measured on
this machine: a decision at ``candidates=14, horizon_seconds=15`` is 375 ms,
27 decisions an episode, so about 17 s of wall per episode including the
environment. 360 episodes across six shards is **17-20 minutes**. Cloning is
minutes and rating is about fifteen. **One turn is under an hour**, which is
what makes iterating worth building rather than describing.

**What this script is and is not.** It is a driver: it runs the three existing
entry points in order, with the round's own directories, and stops on the first
failure rather than carrying a broken round forward. It does not invent any
measurement of its own -- the rating comes from ``run_ladder.py`` and the
merge gate from ``make_demos.py`` -- because a loop that scores itself is the
one thing this project must not build.

Nothing here is skippable in a hurry: ``--dry-run`` prints the commands and
runs none of them, which is the honest way to see what a round costs before
spending it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="expert-iterate")
    parser.add_argument(
        "--seed-policy", type=Path, required=True,
        help="the policy that proposes candidates in round 1. Round n uses "
             "round n-1's clone, which is the loop.")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--first-round", type=int, default=1)
    parser.add_argument(
        "--episodes", type=int, default=60,
        help="episodes per shard. Six shards of 60 is 360 battles, about "
             "17-20 minutes of wall on eight cores.")
    parser.add_argument("--shards", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=14)
    parser.add_argument(
        "--policy-candidates", type=int, default=9,
        help="of --candidates. Clamped by SearchBot to leave "
             "max(2, candidates // 3) from the random draw, whatever is asked "
             "for here: without that floor the target's support collapses "
             "onto what the policy already believed.")
    parser.add_argument(
        "--min-random-candidates", type=int, default=0,
        help="candidates the random draw keeps whatever the proposer wants, "
             "passed straight to make_demos.py. 0 is its default floor, "
             "max(2, candidates // 3). This is the other half of the remedy "
             "the collapse refusal names.")
    parser.add_argument(
        "--targets", choices=("soft", "hard"), default="soft",
        help="what the clone fits. 'soft' is the search's own distribution "
             "over the placements it evaluated -- the third arrow of the loop "
             "this script exists to close, and the only reason the round "
             "computes a target at all. It was not passed, so clone_policy's "
             "own default of 'hard' applied and `data.target = None` threw "
             "the distribution away after collecting it: a real round wrote "
             "runs/iter-1/cloned.pt recording targets='hard' over a shard "
             "whose min_spread_fallback_rate was 0.0. 'hard' is for the "
             "shards already on disk, which cannot support soft targets.")
    parser.add_argument("--proposer-temperature", type=float, default=0.0)
    parser.add_argument("--horizon-seconds", type=float, default=15.0)
    parser.add_argument("--tower-level", type=int, default=5)
    parser.add_argument("--head", default="flat")
    parser.add_argument("--observation", default="v1")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--ladder-episodes", type=int, default=150,
        help="seeds per direction for the round's rating. 150 resolves to "
             "about +/-43 Elo at 95%%; a real proposal improvement should be "
             "worth 100 or more, and anything smaller than 43 must be "
             "reported as unresolved rather than as a win.")
    parser.add_argument(
        "--anchor", action="append", default=[],
        help="repeatable, passed straight to run_ladder.py. Every round is "
             "rated against the same anchors or the rounds are not "
             "comparable to each other.")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--demos", type=Path, default=ROOT / "data_cache")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-ladder", action="store_true",
        help="collect and clone without rating. The round then has no number "
             "attached to it, and the next round is built on an unmeasured "
             "teacher -- which is how three invalid comparisons happened "
             "here already. For debugging the plumbing only.")
    return parser


def _run(command: list[str], *, dry: bool, log: Path | None = None) -> None:
    printable = " ".join(str(c) for c in command)
    print(f"  $ {printable}", flush=True)
    if dry:
        return
    started = time.perf_counter()
    result = subprocess.run([str(c) for c in command], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(
            f"expert-iterate: `{printable}` exited {result.returncode}. "
            "Stopping rather than carrying a broken round forward -- a clone "
            "of a half-collected demonstration set is a teacher nobody can "
            "describe.")
    print(f"    {(time.perf_counter() - started) / 60:.1f} min", flush=True)


def demo_command(args, shard: int, demos, proposer) -> list:
    """One shard's collection, as the command that runs it.

    Built here rather than inline so a test can parse it back through
    ``make_demos.build_parser()``: a flag this driver forgets is a knob the
    round silently takes the default of, which is how the search's own value
    distribution came to be computed and discarded in the same round.
    """
    return [sys.executable, "scripts/make_demos.py",
            "--episodes", args.episodes, "--shard", shard,
            "--out", demos, "--candidates", args.candidates,
            "--horizon-seconds", args.horizon_seconds,
            "--tower-level", args.tower_level,
            "--proposer", proposer,
            "--proposer-temperature", args.proposer_temperature,
            "--policy-candidates", args.policy_candidates,
            "--min-random-candidates", args.min_random_candidates]


def clone_command(args, demos, out) -> list:
    """The round's clone, as the command that runs it. See ``demo_command``."""
    return [sys.executable, "scripts/clone_policy.py",
            "--demos", demos, "--out", out, "--head", args.head,
            "--observation", args.observation, "--epochs", args.epochs,
            "--targets", args.targets]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dry_run and not args.seed_policy.exists():
        raise SystemExit(f"no such policy: {args.seed_policy}")
    anchors = args.anchor or ["random"]

    proposer = args.seed_policy
    history = []
    for offset in range(args.rounds):
        index = args.first_round + offset
        demos = args.demos / f"demos-iter{index}"
        out = args.runs / f"iter-{index}"
        print(f"\n=== round {index}: proposals from {proposer} ===", flush=True)

        for shard in range(args.shards):
            _run(demo_command(args, shard, demos, proposer), dry=args.dry_run)

        _run(clone_command(args, demos, out), dry=args.dry_run)

        if not args.skip_ladder:
            ladder = [sys.executable, "scripts/run_ladder.py",
                      "--name", f"iter-{index}-ladder",
                      "--entrant", f"iter{index}={out / 'cloned.pt'}",
                      "--episodes", args.ladder_episodes,
                      "--tower-level", args.tower_level,
                      "--workers", args.workers,
                      "--out", args.runs]
            for anchor in anchors:
                ladder += ["--anchor", anchor]
            _run(ladder, dry=args.dry_run)

        history.append({"round": index, "proposer": str(proposer),
                        "demos": str(demos), "clone": str(out / "cloned.pt")})
        # The loop: this round's student is next round's proposer.
        proposer = out / "cloned.pt"

    if not args.dry_run:
        (args.runs / "expert-iteration.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")
    print("\nrounds:", json.dumps(history, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
