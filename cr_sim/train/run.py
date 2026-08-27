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
from typing import Any
from pathlib import Path

import numpy as np
import torch

from ..data.cards import build_card_registry
from ..data.leveling import build_level_table
from ..data.source import LogicData
from ..api.encoding import NOOP_SLOT
from ..api.env import CRSimEnv
from ..api.reward import ProjectionWeights, RewardWeights
from .ppo import PPOConfig, train
from .selfplay import (
    FrozenOpponent, OpponentPool, PooledOpponent,
    ancestor_probe, evaluation_probe,
)

ROOT = Path(__file__).resolve().parents[2]

#: Evaluations averaged before a checkpoint may be promoted. Three at the
#: default cadence is 120 battles, which is still not enough to conclude with
#: but is enough that one lucky draw cannot carry it.
_BEST_WINDOW = 3
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
    parser.add_argument(
        "--workers", type=int, default=0,
        help="processes to spread the environments over. 0 runs them in this "
             "process, one after another. About 90%% of a decision is "
             "simulating the battle, so this is most of the throughput "
             "available -- but it cannot carry a self-play opponent, whose "
             "weights would have to be shipped to every worker on each "
             "refresh, so it applies to --opponent random or idle.",
    )
    parser.add_argument("--horizon", type=int, default=256)
    parser.add_argument("--tps", type=int, default=20, help="engine tick rate")
    parser.add_argument("--frame-skip", type=int, default=10, help="ticks per decision")
    parser.add_argument("--match-seconds", type=int, default=120)
    parser.add_argument(
        "--tower-level", type=int, default=11,
        help="Crown Tower level. At 11 a 120-second match ends with 92%% of "
             "tower health untouched and 92%% of matches drawn, so crowns -- "
             "the only real objective -- almost never fire and the agent "
             "learns from shaping alone. Level 5 halves the draw rate at no "
             "extra compute. A training-environment choice, not a change to "
             "the simulator: evaluate at 11 to see what transfers.",
    )
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
    parser.add_argument(
        "--save-every", type=int, default=10, help="updates between checkpoints")
    parser.add_argument(
        "--resume", action="store_true",
        help="continue from checkpoint.pt in the run directory, keeping the "
             "optimiser state and step count. Metrics are appended rather "
             "than overwritten.",
    )
    parser.add_argument(
        "--reward", choices=("simple", "five-term", "projected"), default="five-term",
        help="'simple' is crowns plus a tower-health difference, kept as a control; "
             "'five-term' adds tower damage, elixir trade, counterpush and kites; "
             "'projected' plays the position out with neither side playing "
             "again and pays the change in that outcome, which prices a board "
             "exactly instead of weighting proxies for it.",
    )
    parser.add_argument(
        "--eval-every", type=int, default=10,
        help="updates between honest evaluations against a random control. The "
             "trainer's own return is measured while exploring and has run "
             "eighteen points optimistic; this is the number to watch.",
    )
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument(
        "--horizon-seconds", type=float, default=3.0,
        help="how far --reward projected looks ahead; 0 plays to the end of "
             "the match, which is exact but costs about forty times more",
    )
    parser.add_argument(
        "--refresh-every", type=int, default=20,
        help="updates between drawing a new self-play opponent from the pool",
    )
    parser.add_argument(
        "--pool-size", type=int, default=8,
        help="how many past versions of the policy to keep as opponents. One "
             "lets the learner cycle -- beat last week's strategy, forget the "
             "one before, and go round in circles while the return says "
             "nothing is wrong. A spread of ancestors has to be beaten at "
             "once. Set 1 for the old single-snapshot behaviour.",
    )
    parser.add_argument(
        "--ancestor-episodes", type=int, default=30,
        help="battles per ladder measurement against the oldest kept version",
    )
    parser.add_argument(
        "--opponent", choices=("idle", "random", "self"), default="self",
        help="'idle' never plays a card, which leaves the kite and trade terms with "
             "nothing to measure. 'random' spends its elixir on legal placements.",
    )
    return parser


def _reward_weights(args):
    """The weights object whose type selects the reward."""
    if args.reward == "five-term":
        return RewardWeights()
    if args.reward == "projected":
        return ProjectionWeights(
            horizon_seconds=args.horizon_seconds if args.horizon_seconds > 0 else None
        )
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"

    data = LogicData.load(args.build)
    levels = build_level_table(data)
    registry = build_card_registry(data)

    # Built after the first environment, because the network's shapes come
    # from an observation and the opponents hold a copy of the network.
    opponents: list = []

    def _env(opponent=None) -> CRSimEnv:
        return CRSimEnv(
            data, levels, registry, DEFAULT_DECK, DEFAULT_DECK,
            ticks_per_second=args.tps,
            frame_skip=args.frame_skip,
            max_ticks=args.tps * args.match_seconds,
            tower_level=args.tower_level,
            reward_shaping_weight=args.shaping,
            reward_weights=_reward_weights(args),
            opponent_policy=opponent,
        )

    def make_env(index: int) -> CRSimEnv:
        # Each environment gets its own opponent, so eight parallel battles do
        # not face an identical sequence of placements and report a smoother
        # result than the policy has earned.
        if args.opponent == "random":
            return _env(_random_opponent(args.seed * 1000 + index))
        if args.opponent == "self":
            # Filled in once the network exists; until then the environment
            # faces an idle side, which only affects the first rollout.
            holder: list = []
            opponents.append(holder)
            return _env(lambda obs, mask, h=holder: h[0](obs, mask) if h else (NOOP_SLOT, 0, 0))
        return _env(None)

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
                    "tower_level": args.tower_level,
                    "horizon_seconds": args.horizon_seconds,
                    "opponent": args.opponent}, indent=2),
        encoding="utf-8",
    )

    started = time.perf_counter()
    resume_state = None
    if args.resume:
        checkpoint_path = out / "checkpoint.pt"
        if not checkpoint_path.exists():
            print(f"--resume given but {checkpoint_path} does not exist", file=sys.stderr)
            return 1
        resume_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print(f"resuming from {resume_state.get('steps', 0):,} steps "
              f"({resume_state.get('updates', 0)} updates)", flush=True)

    optimiser_holder: dict[str, Any] = {}
    net_holder: dict[str, torch.nn.Module] = {}
    probe_holder: dict[str, Any] = {}
    best = {"lift": float("-inf")}
    #: Recent lift readings, for promoting on their mean.
    recent: list[float] = []

    probe_env = _env(None)
    probe_env.reset(seed=0)
    config_nvec = (
        int(probe_env.action_space.nvec[1]), int(probe_env.action_space.nvec[2])
    )

    # Appended when resuming: the point of a restart is to keep what the
    # run had already recorded, and "w" would delete the hours being
    # recovered.
    with metrics_path.open("a" if args.resume else "w", encoding="utf-8") as stream:
        def record(stats: dict) -> None:
            # No write here. Every exit from this function ends at _write, and
            # writing on the way in as well emitted each update twice -- once
            # without the eval fields and once with, which read as two trainers
            # racing on one file.
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
            if net is None:
                _write(stats)
                return
            probe = probe_holder.get("probe")
            if probe is not None and args.eval_every and stats["updates"] % args.eval_every == 0:
                stats.update(probe(net))
                print(
                    f"          eval: return {stats['eval_return']:+.4f} vs control "
                    f"{stats['control_return']:+.4f}  win {stats['eval_win']:.0%} vs "
                    f"{stats['control_win']:.0%}  lift {stats['eval_lift_sd']:+.2f} sd",
                    flush=True,
                )
                # Promoted on a rolling mean, never on a single reading.
                #
                # This used to keep whichever checkpoint scored the highest
                # lift, which sounds like keeping the best and is really
                # keeping the luckiest: each reading is 40 battles, and the
                # maximum of nineteen noisy readings is selected for its
                # noise. Measured -- the checkpoint chosen that way scored
                # +0.375 on its 40 battles and -0.033 on 300, while the final
                # weights, chosen by nothing at all, scored +0.141.
                #
                # A mean over several consecutive evaluations cannot be
                # carried by one lucky draw, and the window is what makes the
                # comparison worth anything.
                recent.append(stats["eval_lift_sd"])
                del recent[:-_BEST_WINDOW]
                if len(recent) >= _BEST_WINDOW:
                    rolling = sum(recent) / len(recent)
                    stats["rolling_lift"] = rolling
                    if rolling > best["lift"]:
                        best["lift"] = rolling
                        torch.save(
                            {
                                "state_dict": net.state_dict(),
                                "stats": stats,
                                "rolling_lift": rolling,
                                "window": _BEST_WINDOW,
                            },
                            out / "best.pt",
                        )
            if stats["updates"] % args.save_every == 0:
                # Optimiser state included deliberately. Adam's moment
                # estimates are most of what a long run has learned about its
                # own gradients; restarting without them throws that away and
                # the updates just after a restart look like a bad checkpoint.
                torch.save(
                    {
                        "state_dict": net.state_dict(),
                        "optimiser": (
                            optimiser_holder["optimiser"].state_dict()
                            if "optimiser" in optimiser_holder else None
                        ),
                        "steps": stats["steps"],
                        "updates": stats["updates"],
                        "stats": stats,
                    },
                    out / "checkpoint.pt",
                )
            _write(stats)

        def _write(stats: dict) -> None:
            # Written *after* the evaluation, not before. The probe adds the
            # eval fields to this same dict, so writing first recorded every
            # row without the one number worth keeping -- the honest lift
            # against the control lived only in the console.
            print(json.dumps(stats), file=stream)
            stream.flush()  # a run that dies at hour three should keep hour two

        # The network's shapes come from the first observation, so it does not
        # exist until the trainer has one. It hands it back here, which is what
        # lets this checkpoint mid-run instead of only at the end.
        snapshots: list = []
        pool = OpponentPool(capacity=args.pool_size, seed=args.seed)

        def _on_net(built) -> None:
            net_holder["net"] = built
            if args.opponent == "self":
                nvec = (5, config_nvec[0], config_nvec[1])
                # The starting policy is the pool's first member, so the ladder
                # has a benchmark from the very first update rather than only
                # once a refresh has happened.
                pool.add(built)
                for holder in opponents:
                    snapshot = PooledOpponent(pool, built, nvec, seed=args.seed)
                    holder.append(snapshot)
                    snapshots.append(snapshot)
                probe_holder["ancestor"] = ancestor_probe(
                    _env, pool, nvec, episodes=args.ancestor_episodes
                )
            probe_holder["probe"] = evaluation_probe(
                lambda: _env(None), episodes=args.eval_episodes
            )

        parallel = None
        if args.workers:
            if args.opponent == "self":
                print("--workers cannot carry a self-play opponent; "
                      "use --opponent random, or drop --workers", file=sys.stderr)
                return 1
            from ..api.vec import CRSimVecEnv, VecEnvConfig

            parallel = CRSimVecEnv(
                VecEnvConfig(
                    build=args.build, blue_deck=DEFAULT_DECK, red_deck=DEFAULT_DECK,
                    ticks_per_second=args.tps, frame_skip=args.frame_skip,
                    max_ticks=args.tps * args.match_seconds,
                    reward_shaping_weight=args.shaping,
                    reward_weights=_reward_weights(args),
                    opponent_seed=(args.seed * 1000 if args.opponent == "random" else None),
                ),
                num_envs=args.envs,
                workers=args.workers,
            )

        try:
            net = train(
                make_env, config,
                on_update=record,
                on_net=_on_net,
                opponents=snapshots,
                refresh_every=args.refresh_every,
                # The optimiser itself, not its state at startup: a checkpoint
                # needs the moment estimates as they are when it is written.
                on_optimiser=lambda o: optimiser_holder.__setitem__("optimiser", o),
                resume=resume_state,
                parallel=parallel,
            )
        finally:
            # Worker processes are daemons, so they would die with the parent
            # anyway -- but not before a crash leaves eight of them holding
            # CPU until the interpreter finally exits.
            if parallel is not None:
                parallel.close()


    torch.save({"state_dict": net.state_dict()}, out / "final.pt")
    elapsed = time.perf_counter() - started
    print(f"\ndone in {elapsed / 60:.1f} min -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
