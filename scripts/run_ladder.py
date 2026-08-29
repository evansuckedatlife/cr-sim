"""Play a ladder and fit ratings, offline, and write it where it can be read.

    python scripts/run_ladder.py --name ladder-v1 --episodes 40 \
        --entrant runs/learn-lvl5-kl01/final.pt \
        --anchor random --anchor runs/clone-v1-paired/cloned.pt \
        --anchor search-c6h8 --workers 8

**A star, not a round robin.** 33 loadable checkpoints is 528 pairings, and at
4.0 minutes a mirrored pairing that is 35 hours. Against a fixed set of four
anchors it is 132 pairings -- 8.8 hours single-threaded, about 1.1 across
eight processes -- and the rating is transitive, so the star answers the same
question the round robin does provided the graph stays connected.

**One ladder per observation.** ``check_observation`` refuses a v2 checkpoint
in a v1 environment for a real reason, and the environment encodes one
observation for both sides of a battle, so there is no arrangement in which a
v1 and a v3 policy play each other honestly. A cross-observation Elo is not a
number. This script refuses the mix rather than producing one.

**What it writes, and who reads it.** ``metrics.jsonl`` one row per pairing
direction, through ``check_lift_is_named``; ``verdict.json`` through
``write_verdict``; ``arms.json`` in the shape the live page already reads,
where ``lift`` is a genuine lift against the shared random control on the same
seeds and never the Elo; and ``ladder.json``, the native table.

``ladder.json`` is **invisible on the progress page** and that is deliberate.
Rendering it needs a new shape in ``cr_sim/train/watch.py``, and the running
watcher must not be restarted, so a watch.py edit is inert until somebody
does. ``cr_sim.train.watch.read_ladder`` is the reader, landed and dark. The
run's own note says so, rather than letting a reader assume the page is
showing everything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cr_sim.api.encoding import parse_observation
from cr_sim.api.env import CRSimEnv
from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.train.evaluate import (
    evaluate_paired, evaluation_seeds, load_policy, write_verdict,
)
from cr_sim.train.ladder import (
    expected_score, fit_ratings, parse_player, play_pairing,
)
from cr_sim.train.proposal import check_equal_branch_budget
from cr_sim.train.run import DEFAULT_BUILD, DEFAULT_DECK, _random_opponent
from cr_sim.train.selfplay import check_lift_is_named

#: Rebuilt per worker process rather than shipped across the pickle boundary.
_WORLD: dict = {}


def world(build: Path):
    if "data" not in _WORLD:
        data = LogicData.load(build)
        _WORLD["data"] = data
        _WORLD["levels"] = build_level_table(data)
        _WORLD["registry"] = build_card_registry(data)
    return _WORLD["data"], _WORLD["levels"], _WORLD["registry"]


def env_factory(settings: dict):
    data, levels, registry = world(Path(settings["build"]))
    observation = parse_observation(settings["observation"])

    def make_env(opponent=None) -> CRSimEnv:
        return CRSimEnv(
            data, levels, registry, DEFAULT_DECK, DEFAULT_DECK,
            ticks_per_second=settings["tps"],
            frame_skip=settings["frame_skip"],
            max_ticks=settings["tps"] * settings["match_seconds"],
            tower_level=settings["tower_level"],
            observation=observation,
            opponent_policy=opponent)

    return make_env


def play(task: dict) -> dict:
    """One mirrored pairing, in whatever process picks it up.

    Takes and returns plain data. Checkpoints are re-read from disk here
    rather than shipped as weights, so a worker's task is a few hundred bytes
    and the parent never holds eight copies of a network.
    """
    import torch

    # One thread per process. Eight workers each spinning up torch's default
    # thread pool oversubscribe the machine several times over, and a
    # batch-of-one forward -- which is all an evaluation does -- gets nothing
    # from the extra threads anyway. The binding cost here is the engine.
    torch.set_num_threads(1)
    settings = task["settings"]
    make_env = env_factory(settings)
    probe = make_env(None)
    probe.reset(seed=0)
    a = parse_player(task["a"]).load(probe)
    b = parse_player(task["b"]).load(probe)
    started = time.perf_counter()
    pairing = play_pairing(make_env, a, b, seeds=task["seeds"],
                           mode=task["mode"])
    return {
        "a": pairing.a, "b": pairing.b, "mode": pairing.mode,
        "a_ref": pairing.a_ref, "b_ref": pairing.b_ref,
        "score": pairing.score, "games": pairing.games,
        "seed_correlation": pairing.seed_correlation,
        "seconds": time.perf_counter() - started,
        "directions": [
            {"blue": d.blue, "red": d.red, "wins": d.wins, "losses": d.losses,
             "draws": d.draws, "score": d.score, "crowns": d.crowns,
             "seeds": d.seeds}
            for d in (pairing.forward, pairing.reverse)],
    }


def _rebuild(record: dict):
    """A results dict back into a Pairing, for the fit."""
    from cr_sim.train.ladder import Direction, Pairing

    forward, reverse = (
        Direction(blue=d["blue"], red=d["red"], seeds=d["seeds"],
                  wins=d["wins"], losses=d["losses"], draws=d["draws"],
                  score=d["score"], crowns=d["crowns"])
        for d in record["directions"])
    return Pairing(a=record["a"], b=record["b"], mode=record["mode"],
                   forward=forward, reverse=reverse, score=record["score"],
                   seed_correlation=record["seed_correlation"],
                   a_ref=record["a_ref"], b_ref=record["b_ref"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run-ladder")
    parser.add_argument("--name", default="ladder")
    parser.add_argument("--entrant", action="append", default=[],
                        help="a checkpoint to rate. Repeatable.")
    parser.add_argument("--anchor", action="append", default=[],
                        help="'random', 'search-c6h8', or a checkpoint. "
                             "Repeatable. Every entrant plays every anchor.")
    parser.add_argument(
        "--pairing", nargs=2, action="append", default=[],
        metavar=("A", "B"),
        help="an explicit head-to-head, on top of the star. Repeatable.")
    parser.add_argument(
        "--episodes", type=int, default=40,
        help="seeds per direction; a mirrored pairing plays twice this many "
             "battles. 40 resolves to about +/-83 Elo at 95%%, 150 to +/-43, "
             "400 to +/-26. Greedy has no sampling component at all, so the "
             "only uncertainty here is the finite seed sample.")
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--mode", choices=("greedy", "sampled"),
                        default="greedy",
                        help="never both in one table. Greedy reproduces "
                             "bit-identically and is what a rating wants; "
                             "sampled is a different policy and needs its own "
                             "ladder, its own ratings and twice the battles.")
    parser.add_argument("--tower-level", type=int, default=5)
    parser.add_argument("--tps", type=int, default=20)
    parser.add_argument("--frame-skip", type=int, default=30)
    parser.add_argument("--match-seconds", type=int, default=120)
    parser.add_argument("--prior-sd", type=float, default=400.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument(
        "--no-arms", action="store_true",
        help="skip the paired lift against the random control. That lift is "
             "what puts these players back on the scale every historical "
             "number sits on, and it is what the live page can render.")
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(1)
    entrants = [parse_player(s) for s in args.entrant]
    anchors = [parse_player(s) for s in args.anchor]
    if not entrants or not anchors:
        raise SystemExit("a ladder needs at least one --entrant and one "
                         "--anchor")

    names = [p.name for p in entrants + anchors]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise SystemExit(
            f"two players are both called {duplicates}. A rating is keyed by "
            "name, so a collision does not fail -- it merges two entrants "
            "into one row and rates a policy that does not exist. Pass "
            "name=path to tell them apart.")

    observations = {p.observation for p in entrants + anchors if p.kind == "net"}
    if len(observations) > 1:
        raise SystemExit(
            f"these players were trained on {sorted(observations)}. The "
            "environment encodes one observation for both sides of a battle, "
            "so they cannot play each other, and a rating fitted across them "
            "is not a number. Run one ladder per observation.")
    observation = observations.pop() if observations else "v1"

    settings = {"build": str(args.build), "tps": args.tps,
                "frame_skip": args.frame_skip,
                "match_seconds": args.match_seconds,
                "tower_level": args.tower_level, "observation": observation}
    seeds = evaluation_seeds(args.episodes, block=args.block)

    by_name = {p.name: p for p in entrants + anchors}
    tasks = [{"a": args.entrant[i], "b": args.anchor[j], "seeds": seeds,
              "mode": args.mode, "settings": settings}
             for i in range(len(entrants)) for j in range(len(anchors))]
    spec_of = {p.name: s for p, s in
               zip(entrants + anchors, args.entrant + args.anchor)}
    for a, b in args.pairing:
        tasks.append({"a": spec_of.get(a, a), "b": spec_of.get(b, b),
                      "seeds": seeds, "mode": args.mode, "settings": settings})

    # An equal branch budget, enforced rather than assumed, and enforced over
    # every pairing rather than only the explicit ones -- a guided expert
    # usually enters as an --anchor, which is the star and not a --pairing.
    #
    # Only where the claim needs it, though. A guided search against an
    # unguided one is a claim about *which* fourteen placements the branches
    # were spent on, and a bot that quietly took sixteen would win on the
    # budget alone; two unguided searches at different budgets are a
    # legitimate rung on the ladder and are left alone.
    for task in tasks:
        left, right = parse_player(task["a"]), parse_player(task["b"])
        if left.kind == "search" and right.kind == "search" and (
                left.proposer is not None or right.proposer is not None):
            check_equal_branch_budget(left, right)

    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"{len(tasks)} mirrored pairings x {2 * args.episodes} battles, "
          f"{args.mode}, observation {observation}", flush=True)

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            records = list(pool.map(play, tasks))
    else:
        records = []
        for task in tasks:
            record = play(task)
            print(f"  {record['a']:>28} vs {record['b']:<28} "
                  f"{record['score']:.3f}  ({record['seconds'] / 60:.1f} min)",
                  flush=True)
            records.append(record)

    pairings = [_rebuild(r) for r in records]
    ratings = fit_ratings(pairings, prior_sd=args.prior_sd, anchor="random")

    print(f"\n{'player':<32}{'elo':>8}{'+/-':>7}{'games':>7}  vs random")
    for rating in sorted(ratings.values(), key=lambda r: -r.elo):
        print(f"{rating.name:<32}{rating.elo:>+8.0f}"
              f"{1.96 * rating.error:>7.0f}{rating.games:>7}"
              f"  {expected_score(rating.elo, 0.0):>8.3f}", flush=True)

    # ---------------------------------------------------------- the files

    (out / "ladder.json").write_text(json.dumps({
        "mode": args.mode, "episodes": args.episodes, "block": args.block,
        "observation": observation, "prior_sd": args.prior_sd,
        "anchor": "random",
        "players": [{"name": p.name, "kind": p.kind,
                     "checkpoint": str(p.checkpoint) if p.checkpoint else None,
                     "head": p.head, "head_source": p.head_source,
                     "observation": p.observation, "ref": p.ref}
                    for p in by_name.values()],
        "pairings": records,
        "ratings": [{"name": r.name, "elo": r.elo, "error": r.error,
                     "ci_low": r.elo - 1.96 * r.error,
                     "ci_high": r.elo + 1.96 * r.error,
                     "games": r.games, "pinned": r.pinned}
                    for r in sorted(ratings.values(), key=lambda r: -r.elo)],
        # Reported rather than assumed. The battle-count table treats the two
        # directions on one seed as perfectly correlated, which is the
        # conservative reading; nobody has measured whether they are.
        "seed_correlations": {f"{r['a']} vs {r['b']}": r["seed_correlation"]
                              for r in records},
    }, indent=2), encoding="utf-8")

    rows = []
    for record in records:
        for direction, index in zip(record["directions"], (0, 1)):
            rows.append(check_lift_is_named({
                "updates": len(rows) + 1, "steps": 0,
                "episodes": args.episodes, "steps_per_second": 0.0,
                "entropy": 0.0, "value_loss": 0.0, "policy_loss": 0.0,
                "mean_return": 0.0, "noop_fraction": 0.0,
                "win_rate": direction["wins"] / max(1, args.episodes),
                "ladder_score": direction["score"],
                "ladder_elo": ratings[direction["blue"]].elo
                if direction["blue"] in ratings else 0.0,
                "ladder_player": direction["blue"],
                # Who, and which weights. A rating is transitive, so a row
                # naming only the kind cannot be placed on the graph at all.
                "eval_opponent": direction["red"],
                "ladder_opponent": direction["red"],
                "ladder_opponent_ref": (record["b_ref"] if index == 0
                                        else record["a_ref"]),
                "eval_episodes": args.episodes,
                "mode": record["mode"], "direction": index,
                "seeds": direction["seeds"],
            }))

    arms = []
    if not args.no_arms:
        print("\nand the same players against the shared random control, on "
              "the same seeds, so these land on the existing scale:",
              flush=True)
        make_env = env_factory(settings)

        def control_env() -> CRSimEnv:
            return make_env(_random_opponent(60_000))

        for player in entrants + [p for p in anchors if p.kind == "net"]:
            probe = control_env()
            probe.reset(seed=0)
            net = load_policy(player.checkpoint, probe)
            verdict = evaluate_paired(control_env, net, episodes=args.episodes,
                                      seeds=seeds, modes=(args.mode,))
            arm = verdict[args.mode]
            print(f"  {player.name:<32}{arm['lift']:>+8.3f}"
                  f"   [{arm['ci_low']:+.3f}, {arm['ci_high']:+.3f}]",
                  flush=True)
            arms.append({
                "name": player.name, "checkpoint": str(player.checkpoint),
                "mode": args.mode, "observation": player.observation,
                "head": player.head,
                # A lift, and only ever a lift. The rating lives in
                # ladder.json under "elo"; putting an Elo in a field called
                # "lift" is the scale conflation this project has already
                # paid for three times.
                "lift": arm["lift"], "ci_low": arm["ci_low"],
                "ci_high": arm["ci_high"], "win": arm["win"],
                "loss": arm["loss"], "episodes": args.episodes,
                "eval_opponent": verdict["eval_opponent"],
                "elo": ratings[player.name].elo if player.name in ratings else None,
            })
        (out / "arms.json").write_text(json.dumps(arms, indent=2),
                                       encoding="utf-8")

    top = max(ratings.values(), key=lambda r: r.elo)
    write_verdict(out / "verdict.json", {
        "episodes": args.episodes,
        "eval_opponent": "ladder",
        "mode": args.mode,
        "seeds": seeds,
        "ladder_opponent_ref": "|".join(sorted(p.ref for p in anchors)),
        "ladder_elo": top.elo, "ladder_player": top.name,
        "ratings": [{"name": r.name, "elo": r.elo, "error": r.error,
                     "games": r.games} for r in
                    sorted(ratings.values(), key=lambda r: -r.elo)],
        **({"lift": arms[0]["lift"], "ci_low": arms[0]["ci_low"],
            "ci_high": arms[0]["ci_high"], "win": arms[0]["win"],
            "loss": arms[0]["loss"]} if arms else {}),
        "note": (f"Elo, anchored at random = 0, fitted from "
                 f"{len(records)} mirrored pairings. Not a lift: the two "
                 "scales are unrelated and must not be plotted on one axis."),
    })

    with (out / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")

    correlations = [r["seed_correlation"] for r in records
                    if r["seed_correlation"] is not None]
    (out / "config.json").write_text(json.dumps({
        "reward": "evaluation only", "opponent": "ladder",
        "eval_opponent": "ladder", "eval_episodes": args.episodes,
        "tower_level": args.tower_level, "frame_skip": args.frame_skip,
        "tps": args.tps, "match_seconds": args.match_seconds,
        "num_envs": 0, "horizon": 0, "kind": "job",
        "mode": args.mode, "observation": observation,
        "note": (
            f"An Elo ladder over {len(by_name)} players, {args.mode}, "
            f"{args.episodes} seeds a direction. Every arm here met the SAME "
            f"random opponent on the SAME {args.episodes} paired seeds, and "
            "every pairing was played both directions and mirror-averaged. "
            "Ratings are anchored at random = 0 with a Gaussian prior of "
            f"{args.prior_sd:g}, which is what keeps a 100-0 pairing finite. "
            "The lifts in arms.json are lifts and the ratings in ladder.json "
            "are Elo; they are different scales and belong on different axes. "
            "ladder.json is invisible on this page -- its reader, "
            "cr_sim.train.watch.read_ladder, is landed but the running "
            "watcher holds the older module, so the table is in the file and "
            "the page is showing the arms instead."
            + (f" Seed correlation between directions: "
               f"{np.mean(correlations):+.3f} mean over "
               f"{len(correlations)} pairings." if correlations else "")),
    }, indent=2), encoding="utf-8")

    print(f"\n{(time.perf_counter() - started) / 60:.1f} min -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
