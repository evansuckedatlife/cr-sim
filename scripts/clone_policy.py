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

from cr_sim.api.encoding import parse_observation
from cr_sim.api.env import CRSimEnv
from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.entity import Team
from cr_sim.train.clone import CloneConfig, Demonstrations, clone
from cr_sim.train.evaluate import evaluate, write_verdict
from cr_sim.train.nets import POLICY_HEADS, ActorCritic, net_config_for
from cr_sim.train.run import DEFAULT_BUILD, DEFAULT_DECK, _random_opponent
from cr_sim.train.selfplay import (check_lift_is_named, opponent_name,
                                   reward_name)


UNRECORDED = "(unrecorded)"


def merge(paths: list[Path]) -> Demonstrations:
    """Concatenate shards into one training set.

    ``target`` is carried through with everything else, and that is not
    incidental. The shards on disk hold the search's own value distribution
    over every placement it evaluated, which is what the cloner is written to
    train against -- and this function used to drop it, silently, by omitting
    one keyword. The loader read it, the saver wrote it, ``clone`` branched on
    it, and every clone actually run fell down the ``target is None`` path and
    fitted the single move the search happened to play. A feature that is
    inert everywhere it is used still passes every test that only checks it
    round-trips through a file.
    """
    parts = [Demonstrations.load(p) for p in paths]
    if not parts:
        raise SystemExit("no shards found")
    targets = [p.target for p in parts]
    return Demonstrations(
        grid=np.concatenate([p.grid for p in parts]),
        vector=np.concatenate([p.vector for p in parts]),
        mask=np.concatenate([p.mask for p in parts]),
        action=np.concatenate([p.action for p in parts]),
        value=np.concatenate([p.value for p in parts]),
        # All or nothing: a half-filled target array would train some rows
        # against the search's beliefs and the rest against zeros, which is a
        # worse label than either.
        target=(np.concatenate(targets)
                if all(t is not None for t in targets) else None),
        episodes=sum(p.episodes for p in parts),
        play_rate=float(np.mean([p.play_rate for p in parts])),
        observation=_agree(parts, "observation", paths),
        reward=_agree(parts, "reward", paths),
        # The third field, and the one whose mismatch is hardest to see. A
        # different proposer is a different *label*: the target is the
        # search's distribution over the candidates it scored, so shards from
        # two proposers train some rows against one supervision signal and the
        # rest against another, with the same shapes, the same channel count
        # and a training curve that converges either way.
        proposer=_agree(parts, "proposer", paths),
        # One entry per shard rather than one merged number: the collapse rate
        # is a property of a collection run, and averaging six of them would
        # hide the one shard that ran away.
        meta=json.dumps({"shards": [
            {"path": str(q), "meta": p.meta} for p, q in zip(parts, paths)]}),
    )


def _agree(parts: "list[Demonstrations]", field: str, paths: list[Path]) -> str:
    """The value every shard agrees on, or a refusal.

    Merging shards recorded under different encodings makes a set whose grids
    mean different things row to row, and nothing downstream can detect it:
    the channel count matches, the training converges, and the checkpoint
    carries whichever name was declared. Shards written before provenance
    existed report an empty string, and mixing those with stamped ones is the
    same hazard, so that is refused too -- naming the files, because the fix
    is to re-record the odd one out.
    """
    seen = {getattr(p, field) for p in parts}
    if len(seen) == 1:
        return seen.pop()
    where = {getattr(p, field): q.name for p, q in zip(parts, paths)}
    raise SystemExit(
        f"shards disagree on {field}: "
        + ", ".join(f"{v!r} (e.g. {n})" for v, n in sorted(where.items()))
        + f". They cannot be merged: a set whose {field} varies row to row is"
        " undetectable downstream. Re-record the odd shards, or point --demos"
        " at one consistent directory.")



def subset(data: Demonstrations, fraction: float, seed: int) -> Demonstrations:
    """A random slice of the demonstrations, for a sample-efficiency curve.

    Sampled over the whole set rather than truncated to the first N episodes:
    the shards are ordered, so a prefix is a handful of battles seen in full
    while a random draw is the same distribution of positions at a smaller
    count -- which is what "learns more per example" has to be measured on.

    ``observation`` and ``reward`` are carried through. They were not, and a
    slice is exactly where dropping them does damage: ``main`` refuses a set
    whose recorded encoding disagrees with ``--observation``, and that check
    reads the field. Taking a fraction blanked it, so the guard fell through
    to the "these shards predate provenance" warning and every
    sample-efficiency run trained with the one check on the encoding switched
    off -- the check being the point of the field. Selecting rows must not
    change what the rows are.
    """
    keep = max(1, int(len(data) * max(0.0, min(1.0, fraction))))
    order = np.random.default_rng(seed).permutation(len(data))[:keep]
    order.sort()
    return Demonstrations(
        grid=data.grid[order], vector=data.vector[order], mask=data.mask[order],
        action=data.action[order], value=data.value[order],
        target=None if data.target is None else data.target[order],
        episodes=data.episodes, play_rate=data.play_rate,
        observation=data.observation, reward=data.reward,
        proposer=data.proposer, meta=data.meta)


def build_parser() -> argparse.ArgumentParser:
    """Its own function, the way ``cr_sim.train.run`` has one.

    So that what this script will accept can be asked without running a
    clone. ``--head`` in particular went stale here while the network grew a
    fourth head, and a flag nobody can inspect is a flag nobody checks.
    """
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
    parser.add_argument(
        "--targets", choices=("soft", "hard"), default="hard",
        help="'soft' fits the search's own distribution over the placements "
             "it evaluated; 'hard' fits the single move it played. Soft is "
             "the better idea and the shards on disk cannot support it: the "
             "target was built from candidate values scaled by their own "
             "spread, and on the states where the search chose to wait those "
             "values are equal to four decimal places, so 86%% of them carry "
             "an exactly uniform distribution over fifteen-odd candidates of "
             "which fourteen are placements. The pass action is the target's "
             "argmax in none of 10,940 recorded decisions. cr_sim.train.clone "
             "now floors the spread and gives waiting the margin the search "
             "required of a play, so demonstrations collected from here on "
             "can use 'soft'.",
    )
    parser.add_argument("--tower-level", type=int, default=5)
    parser.add_argument(
        "--observation", default="v1",
        help="which observation the demonstrations were recorded with. It has "
             "to match, and a mismatch is a silent one: the shapes line up "
             "whenever the channel counts do.")
    parser.add_argument(
        "--head", choices=POLICY_HEADS, default="flat",
        help="'flat' is one linear layer over all 720 actions; 'factored' "
             "picks the card, then the tile, with the tile head conditioned "
             "on an embedding of the card and its placement weights shared "
             "across cards. Both parameterise the same masked categorical, so "
             "the comparison is about sample efficiency, not expressiveness. "
             "'factored-stats' is 'factored' with the card lookup replaced by "
             "an encoder over the card's own statistics, so a clone trained on "
             "one deck's demonstrations conditions correctly on a card it "
             "never saw. 'conv' emits the placements as a 1x1 convolution "
             "over the trunk's own feature map.",
    )
    parser.add_argument(
        "--fraction", type=float, default=1.0,
        help="train on this fraction of the demonstrations. The point of a "
             "factored head is learning more per example, which only shows up "
             "as a curve against how many examples there are.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=120,
                        help="battles in the final evaluation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    shards = sorted(args.demos.glob("shard-*.npz"))
    data = merge(shards)
    if args.fraction < 1.0:
        data = subset(data, args.fraction, args.seed)
    if args.targets == "hard":
        data.target = None
    # --observation names the encoding the net is built for. Until now that
    # was an unverified claim about a file: most mismatches happen to die on
    # the channel count, but two variants of equal width and different meaning
    # train quietly and stamp the wrong name onto the checkpoint, after which
    # a run's check_observation agrees with it because it compares a shape.
    if data.observation and data.observation != args.observation:
        raise SystemExit(
            f"--observation says {args.observation!r} but these shards were "
            f"recorded under {data.observation!r}. This flag is written into "
            "the checkpoint and every later run trusts it, so it is not a "
            f"preference. Pass --observation {data.observation}, or point "
            f"--demos at a set recorded under {args.observation}.")
    if not data.observation:
        print(f"WARNING: these shards record no observation, so --observation "
              f"{args.observation!r} cannot be checked against them. They "
              "predate provenance; re-record to make this verifiable.",
              flush=True)
    print(f"{len(shards)} shards, {len(data):,} decisions from "
          f"{data.episodes} episodes, expert played on "
          f"{data.play_rate:.0%} of them, "
          f"observation {data.observation or UNRECORDED}, "
          f"reward {data.reward or UNRECORDED}, "
          f"proposer {data.proposer or UNRECORDED}", flush=True)
    print(flush=True)

    build = LogicData.load(DEFAULT_BUILD)
    levels, registry = build_level_table(build), build_card_registry(build)

    def make_env(seed_offset: int = 0) -> CRSimEnv:
        return CRSimEnv(
            build, levels, registry, DEFAULT_DECK, DEFAULT_DECK,
            ticks_per_second=20, frame_skip=30, max_ticks=20 * 120,
            tower_level=args.tower_level,
            observation=parse_observation(args.observation),
            opponent_policy=_random_opponent(60_000 + seed_offset))

    probe = make_env()
    probe.reset(seed=0)
    slots, width, height = (int(v) for v in probe.action_space.nvec)
    net = ActorCritic(net_config_for(probe, head=args.head))

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
        pass_action=(slots - 1) * width * height, seed=args.seed),
        on_epoch=report)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(),
                "observation": args.observation,
                # Which expert taught this network. A clone of a
                # policy-guided search is a clone of a different teacher, and
                # the whole point of expert iteration is that the teacher
                # moves between rounds.
                "proposer": data.proposer,
                "demo_meta": data.meta,
                "targets": args.targets, "pass_weight": args.pass_weight,
                # Which head these weights are. A factored head's parameters
                # do not fit a flat one, and whatever loads this needs to
                # build the network they belong to.
                "head": args.head,
                "clone": history[-1] if history else {}},
               args.out / "cloned.pt")
    if args.episodes <= 0:
        print(f"\n{(time.perf_counter() - started) / 60:.1f} min -> {args.out} "
              "(no battles played; --episodes 0)", flush=True)
        return 0

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
               # Read off the environment the control actually played in.
               # This file used to go around write_verdict with a bare
               # write_text and no eval_opponent at all, so the clone's
               # +2.167 -- the number half this project is quoted against --
               # renders in report.py as "beats an unnamed opponent". Not one
               # verdict.json on this machine carries the field; all seven
               # would be refused today, which is the guard working.
               "eval_opponent": opponent_name(make_env()),
               # And the scale, off the same environment. The metrics row
               # below has always carried it; this file did not, so the
               # clone's +2.167 sat on disk with no record of the reward its
               # returns were denominated in -- beside an expert anchor that
               # was measured under a different one.
               "eval_reward": reward_name(make_env()),
               "eval_episodes": args.episodes,
               "seeds": seeds,
               "mode": "greedy" if best is greedy_stats else "sampled",
               "sampled": sampled_stats, "greedy": greedy_stats,
               # Flattened as well as nested: the report reads these keys, and
               # the honest headline is whichever way of playing scored better.
               "lift": best["lift"], "ci_low": best["ci_low"],
               "ci_high": best["ci_high"], "win": best["win"],
               "loss": best["loss"],
               "clone": history[-1] if history else {}}
    write_verdict(args.out / "verdict.json", verdict)

    # Registered as a run so it appears beside the training runs it is meant
    # to be compared against. A flat series, because a clone does not learn
    # over time -- it is a single result, and drawing it as a curve would
    # imply a trajectory it does not have.
    (args.out / "config.json").write_text(json.dumps({
        "reward": "behavioural cloning", "opponent": "random",
        "eval_opponent": opponent_name(make_env()),
        "eval_episodes": args.episodes,
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
        # Named, because a lift compared against one measured on a different
        # opponent is not a comparison. See cr_sim.train.selfplay.check_lift_is_named.
        "eval_opponent": opponent_name(make_env()),
        # And the scale, for the same reason: a lift is a difference of
        # returns, and a return is denominated in the reward that scored it.
        "eval_reward": reward_name(make_env()),
        "eval_episodes": args.episodes,
        "observation": args.observation, "head": args.head,
        "proposer": data.proposer,
        "targets": args.targets, "pass_weight": args.pass_weight,
        "fraction": args.fraction,
    }
    check_lift_is_named(row)
    with (args.out / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for update in (1, 2):
            stream.write(json.dumps({**row, "updates": update}) + chr(10))
    print(f"\n{(time.perf_counter() - started) / 60:.1f} min -> {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
