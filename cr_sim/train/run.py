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
from ..api.encoding import NOOP_SLOT, grid_channels, parse_observation
from ..api.env import CRSimEnv
from ..api.reward import ProjectionWeights, RewardWeights
from .nets import net_config_for
from .ppo import PPOConfig, train
from .selfplay import (
    FrozenOpponent, OpponentPool, PooledOpponent,
    ancestor_probe, check_lift_is_named, evaluation_probe, opponent_name,
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

    # Carried on the callable so a measurement can say which opponent it
    # faced; see cr_sim.train.selfplay.opponent_name for why that is not
    # optional here.
    policy.opponent_name = "random"
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
             "available. Self-play works here too: the opponent's shapes "
             "travel in the worker config and its weights are sent on each "
             "refresh, so a worker runs its own opponent rather than the "
             "parent doing every forward pass.",
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
    parser.add_argument(
        "--observation", default="v1",
        help="which observation to encode: 'v1' (the original nine grid "
             "channels and both hands in full), 'v2' (spell and area-effect "
             "channels, per-cell body counts, and the opponent's hand and "
             "elixir hidden), or a comma-separated subset of spells, swarm, "
             "hide_enemy_hand, hide_enemy_elixir. Changing this invalidates "
             "every checkpoint trained on the other one -- the first "
             "convolution has a filter bank per input channel -- so it is "
             "recorded in the run and in every checkpoint it writes.",
    )
    parser.add_argument(
        "--head", choices=("flat", "factored", "conv"), default="flat",
        help="'flat' is one linear layer over all 720 actions; 'factored' "
             "picks the card, then the tile, with the tile head conditioned "
             "on an embedding of the card and its weights shared across "
             "cards. Not a correctness difference -- a flat masked "
             "categorical can represent anything the factorisation can -- but "
             "a sample-efficiency one, and placements are the sparse part.",
    )
    parser.add_argument("--entropy", type=float, default=0.02)
    parser.add_argument(
        "--kl", type=float, default=0.0,
        help="weight on KL(reference || policy), a trust region around the "
             "weights --init-from supplied. The standard remedy for a "
             "fine-tune that walks a competent policy back to nothing, and "
             "measured here: plain PPO from the behavioural clone raised its "
             "pass rate from 8%% to 36%% over 34 updates while entropy fell "
             "the whole time, so the collapse is the policy gradient's doing "
             "and an anchor is what holds it. Requires --init-from.",
    )
    parser.add_argument(
        "--kl-reference", type=Path, default=None,
        help="weights the trust region anchors to. Defaults to --init-from, "
             "which is right for a fresh fine-tune and wrong for --resume: on "
             "a restart the policy has already moved, and anchoring to where "
             "it currently is holds it nowhere. Give the clone's own "
             "checkpoint to continue a run against the same anchor it "
             "started with.",
    )
    parser.add_argument(
        "--elixir-weight", type=float, default=0.3,
        help="weight on the elixir lead inside --reward projected's "
             "potential. This is what makes a card cost something, and it is "
             "also why passing pays: spending drops the potential now while "
             "the card's effect on the board takes longer than the "
             "projection's horizon to appear. Measured on the clone's own "
             "rollouts, a pass earns +0.071 more reward than a placement at "
             "0.3 and -0.010 less at 0.0. The searching bot needed it at 0.0 "
             "for the same reason: at 0.3 it never played a card at all.",
    )
    parser.add_argument(
        "--shaping", type=float, default=0.01,
        help="weight on tower-health difference. At 0.01 a whole match's tower "
             "damage is worth about 0.02 against 1.0 per crown, so the reward is "
             "effectively sparse; raise it to give credit between crowns.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="where the network runs: cpu, cuda, xpu (Intel), or auto to take "
             "the best available. Worth more than it looks -- with the "
             "environments spread over worker processes, the parent spends "
             "roughly 40-50%% of every update in the network, nearly all of "
             "it in the PPO gradient step. Requires a torch build with that "
             "backend compiled in; the plain wheel is CPU-only.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "runs")
    parser.add_argument("--name", default="ppo")
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument(
        "--save-every", type=int, default=10, help="updates between checkpoints")
    parser.add_argument(
        "--init-from", type=Path, default=None,
        help="start from these weights instead of from random. The order "
             "every successful game agent used: AlphaStar's supervised agent "
             "outranked 84%% of human players before any reinforcement "
             "learning, and the learning refined a competent policy rather "
             "than creating one. Unlike --resume this takes only the weights, "
             "so the optimiser starts clean and the step count starts at zero.",
    )
    parser.add_argument(
        "--opponent-temperature", type=float, default=1.0,
        help="how sharply a self-play opponent plays its own policy. 1.0 "
             "samples it as-is, which sounds neutral and is not: a policy "
             "with entropy near the uniform maximum is still nearly random, "
             "leaving the outcome as unpredictable as it was against a random "
             "agent and the critic with nothing to fit. Below 1.0 sharpens "
             "toward its own preferences.",
    )
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
        "--eval-every", type=int, default=20,
        help="updates between honest evaluations against a random control. "
             "Pure overhead: an evaluation plays 40 paired battles and a "
             "ladder 30 more, so at every 10 updates a long run spends about "
             "a quarter of itself measuring. Every 20 still gives dozens of "
             "readings, which is more than enough to see a trend.",
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


def _resolve_device(name: str) -> str:
    """Pick a device, and say plainly when the asked-for one is unavailable.

    Silently falling back to CPU would leave a run quietly three times slower
    than intended, with nothing on the page to say so -- which is the failure
    mode this project keeps producing.
    """
    if name != "auto":
        if name.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit(
                "cuda requested but this torch has no CUDA "
                f"({torch.__version__}). Install a CUDA build, or use --device cpu."
            )
        if name.startswith("xpu") and not getattr(
            getattr(torch, "xpu", None), "is_available", lambda: False
        )():
            raise SystemExit(
                "xpu requested but this torch has no XPU "
                f"({torch.__version__}). Install the Intel build with "
                "pip install torch --index-url "
                "https://download.pytorch.org/whl/xpu, or use --device cpu."
            )
        return name
    if torch.cuda.is_available():
        return "cuda"
    # XPU is deliberately not chosen automatically, even when present. On the
    # machine this was developed on it reports available, runs a gradient step
    # 6.6x faster than eight CPU threads, and then fails a real training loop
    # three different ways: an unimplemented convolution, out of device
    # memory, and out of Level Zero resources during the optimiser's own state
    # allocation. The rollout's several hundred small forward passes, each
    # with a blocking host readback, exhaust the driver's handles before the
    # first update. A default that picks a backend which cannot finish an
    # update is worse than no default at all -- ask for it explicitly.
    return "cpu"


def _load_reference(path, args, env):
    """The frozen policy a trust region pulls back toward.

    Built from the checkpoint rather than from the live network, because on a
    resume the live network is already several thousand updates from the thing
    it was supposed to stay near, and anchoring to it would hold it nowhere.
    """
    from .nets import ActorCritic, net_config_for

    payload = torch.load(path, map_location="cpu", weights_only=False)
    net = ActorCritic(net_config_for(env, head=payload.get("head", args.head)))
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net


def _reward_weights(args):
    """The weights object whose type selects the reward."""
    if args.reward == "five-term":
        return RewardWeights()
    if args.reward == "projected":
        return ProjectionWeights(
            horizon_seconds=args.horizon_seconds if args.horizon_seconds > 0 else None,
            elixir=args.elixir_weight,
        )
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Resolved before anything is written, because config.json records it.
    anchor_path = args.kl_reference or args.init_from

    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"

    device = _resolve_device(args.device)
    if device != "cpu":
        print(f"network on {device}", flush=True)

    data = LogicData.load(args.build)
    levels = build_level_table(data)
    registry = build_card_registry(data)

    # Built after the first environment, because the network's shapes come
    # from an observation and the opponents hold a copy of the network.
    opponents: list = []

    observation = parse_observation(args.observation)

    def _env(opponent=None) -> CRSimEnv:
        return CRSimEnv(
            data, levels, registry, DEFAULT_DECK, DEFAULT_DECK,
            observation=observation,
            ticks_per_second=args.tps,
            frame_skip=args.frame_skip,
            max_ticks=args.tps * args.match_seconds,
            tower_level=args.tower_level,
            reward_shaping_weight=args.shaping,
            reward_weights=_reward_weights(args),
            opponent_policy=opponent,
        )

    def _eval_env() -> CRSimEnv:
        """The environment the honest evaluation is played in.

        A *random* opponent, not an idle one. This used to be ``_env(None)``,
        which never plays a card, while the large paired verdicts faced a
        random agent -- and both were reported as "lift" and compared to each
        other. They were never comparable: the control wins 92% of the idle
        matches and 26% of the random ones.
        """
        return _env(_random_opponent(90_000))

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
        head=args.head,
        kl_coefficient=args.kl,
    )
    (out / "config.json").write_text(
        json.dumps({**asdict(config), "deck": list(DEFAULT_DECK), "tps": args.tps,
                    "frame_skip": args.frame_skip, "match_seconds": args.match_seconds,
                    "shaping": args.shaping, "reward": args.reward,
                    "tower_level": args.tower_level,
                    "horizon_seconds": args.horizon_seconds,
                    "opponent": args.opponent, "head": args.head,
                    "kl": args.kl, "elixir_weight": args.elixir_weight,
                    "kl_reference": str(anchor_path) if anchor_path else None,
                    "observation": args.observation,
                    "observation_channels": list(grid_channels(observation)),
                    # Which opponent the in-run lift is measured against, read
                    # off a real evaluation environment rather than asserted.
                    # A run's own lift series is only comparable to another
                    # run's when these agree.
                    "eval_opponent": opponent_name(_eval_env()),
                    "eval_episodes": args.eval_episodes}, indent=2),
        encoding="utf-8",
    )

    started = time.perf_counter()
    resume_state = None
    if args.kl > 0.0 and not anchor_path:
        raise SystemExit(
            "--kl anchors the policy to a fixed reference. Without "
            "--init-from or --kl-reference that reference is a random "
            "initialisation, and anchoring to noise is not a trust region.")
    if args.init_from:
        if args.resume:
            raise SystemExit(
                "--init-from and --resume do different things and cannot be "
                "combined: one starts a new run from borrowed weights, the "
                "other continues a run that stopped.")
        if not args.init_from.is_file():
            raise SystemExit(f"no weights at {args.init_from}")
        borrowed = torch.load(args.init_from, map_location="cpu",
                              weights_only=False)
        if borrowed.get("head", args.head) != args.head:
            raise SystemExit(
                f"--init-from holds a {borrowed.get('head')!r} head but "
                f"--head is {args.head!r}. The two do not share a parameter "
                "shape, and loading one into the other would fail on a tensor "
                "name with no obvious owner.")
        # Weights only. The optimiser state belongs to whatever produced these
        # -- supervised cloning, in the case this exists for -- and its moment
        # estimates describe a different objective entirely.
        resume_state = {"state_dict": borrowed["state_dict"],
                        "steps": 0, "updates": 0}
        print(f"starting from {args.init_from}", flush=True)
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
    probe_obs, _ = probe_env.reset(seed=0)
    config_nvec = (
        int(probe_env.action_space.nvec[1]), int(probe_env.action_space.nvec[2])
    )
    # Enough for a worker to rebuild the opponent network. Shapes travel in
    # the config; weights arrive per refresh, which is what lets self-play run
    # across processes instead of costing three and a half times the
    # throughput to stay in one.
    # Built through the same helper the trainer uses, so a worker's opponent
    # network cannot be a different shape -- or a different head -- from the
    # learner whose weights it is about to be handed.
    net_shape = asdict(net_config_for(probe_env, head=args.head))

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
            # The ladder: how the policy fares against the oldest version of
            # itself still in the pool. More readable than lift against a
            # random control, whose per-episode spread is wide enough that a
            # +0.23 reading on this project turned out to be noise.
            ancestor = probe_holder.get("ancestor")
            if ancestor is not None and args.eval_every and stats["updates"] % args.eval_every == 0:
                stats.update(ancestor(net))
                if "ancestor_win" in stats:
                    print(f"          ladder: {stats['ancestor_win']:.0%} vs its own "
                          f"generation {stats['ancestor_age']}", flush=True)

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
                                # Which head these weights are, so whatever
                                # loads them builds the network they fit.
                                "head": args.head,
                                "observation": args.observation,
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
                        "head": args.head,
                        "observation": args.observation,
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
            print(json.dumps(check_lift_is_named(stats)), file=stream)
            stream.flush()  # a run that dies at hour three should keep hour two

        # The network's shapes come from the first observation, so it does not
        # exist until the trainer has one. It hands it back here, which is what
        # lets this checkpoint mid-run instead of only at the end.
        snapshots: list = []
        pool = OpponentPool(capacity=args.pool_size, seed=args.seed)

        def _on_refresh(net, update: int) -> None:
            """Snapshot this generation, then hand an ancestor to the workers.

            Added before the draw, so the pool contains this generation when
            the opponents pick from it -- the other order leaves the pool a
            generation behind for ever.
            """
            pool.add(net)
            if parallel is not None:
                drawn = pool.sample()
                if drawn is not None:
                    parallel.set_opponent(drawn.state_dict())

        def _on_net(built) -> None:
            net_holder["net"] = built
            if args.opponent == "self":
                nvec = (5, config_nvec[0], config_nvec[1])
                # The starting policy is the pool's first member, so the ladder
                # has a benchmark from the very first update rather than only
                # once a refresh has happened.
                pool.add(built)
                for holder in opponents:
                    snapshot = PooledOpponent(
                        pool, built, nvec, seed=args.seed,
                        temperature=args.opponent_temperature)
                    holder.append(snapshot)
                    snapshots.append(snapshot)
                probe_holder["ancestor"] = ancestor_probe(
                    _env, pool, nvec, episodes=args.ancestor_episodes
                )
                # Seeded now rather than at the first refresh: without this the
                # workers face an idle opponent for the opening stretch of the
                # run, which is not self-play and not what the metrics claim.
                if parallel is not None:
                    parallel.set_opponent(built.state_dict())
            # See _eval_env: a random opponent, not an idle one, and the
            # probe records which it was on every row it produces.
            probe_holder["probe"] = evaluation_probe(
                _eval_env, episodes=args.eval_episodes,
            )

        parallel = None
        if args.workers:
            from ..api.vec import CRSimVecEnv, VecEnvConfig

            parallel = CRSimVecEnv(
                VecEnvConfig(
                    build=args.build, blue_deck=DEFAULT_DECK, red_deck=DEFAULT_DECK,
                    ticks_per_second=args.tps, frame_skip=args.frame_skip,
                    max_ticks=args.tps * args.match_seconds,
                    reward_shaping_weight=args.shaping,
                    reward_weights=_reward_weights(args),
                    observation=observation,
                    opponent_seed=(args.seed * 1000 if args.opponent == "random" else None),
                    net_config=(net_shape if args.opponent == "self" else None),
                ),
                num_envs=args.envs,
                workers=args.workers,
            )

        try:
            net = train(
                make_env, config,
                reference=_load_reference(anchor_path, args, probe_env)
                if args.kl > 0.0 else None,
                device=device,
                on_update=record,
                on_net=_on_net,
                opponents=snapshots,
                refresh_every=args.refresh_every,
                # Without this the pool holds only the network it was seeded
                # with -- the randomly initialised one -- and self-play would
                # spend the entire run beating a policy that never improved.
                on_refresh=_on_refresh,
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


    torch.save({"state_dict": net.state_dict(), "head": args.head,
                "observation": args.observation}, out / "final.pt")
    elapsed = time.perf_counter() - started
    print(f"\ndone in {elapsed / 60:.1f} min -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
