"""Record the searching bot playing, so a policy can learn from it.

The bootstrap step. Over 16 matches against a random opponent the bot wins
100% and never loses, so its decisions are worth copying -- which is the whole
premise of behavioural cloning, and the step this project skipped when it
started reinforcement learning from random initialisation.

Sharded because a match costs about seventeen seconds: the bot branches the
battle for every candidate placement and plays each one fifteen seconds
forward. Run several shards at once and merge them.

    python scripts/make_demos.py --episodes 60 --shard 0 --out data_cache/demos
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cr_sim.api.encoding import parse_observation
from cr_sim.api.env import CRSimEnv
from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.entity import Team
from cr_sim.train.clone import Demonstrations, collect
from cr_sim.train.run import (
    DEFAULT_BUILD, DEFAULT_DECK, _random_opponent, _reward_weights)
from cr_sim.train.selfplay import reward_name
from cr_sim.train.proposal import proposer_factory, proposer_identity
from cr_sim.train.scripted import SearchBot, SearchBotConfig

#: How often the *unguided* search finds its candidates inseparable and the
#: target collapses onto the move it happened to make.
#:
#: Measured on this machine rather than guessed. Zero, on two samples: 0 of
#: 4,494 rows across the three shards in ``data_cache/demos_fixed`` are
#: one-hot, and a fresh four-episode collection at the shipped defaults
#: (candidates 14, horizon 15 s) fell back on 0 of 109 decisions, with a mean
#: candidate spread of 0.0585 against a floor of 1e-3. The unguided draw
#: spreads its candidates across the cards in hand, so the search almost
#: always has something to prefer.
#:
#: It is the reference :func:`collapse_refusal` compares a policy-proposed
#: shard against, because the number that matters is not the rate itself but
#: how far guiding the proposal moved it.
BASELINE_FALLBACK_RATE = 0.0


def build_parser() -> argparse.ArgumentParser:
    """Exposed so a test can compare these flags against run.py's.

    The two parsers are written separately and drifted once already: this one
    built its environment with no reward at all, so every demonstration set
    carried value targets from a reward no fine-tune ever optimised.
    """
    parser = argparse.ArgumentParser(prog="make-demos")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data_cache/demos"))
    parser.add_argument("--tower-level", type=int, default=5)
    parser.add_argument(
        "--observations", default="v1",
        help="comma-separated observation variants to record, each written to "
             "its own subdirectory of --out. One playthrough produces all of "
             "them: the expert reads the battle rather than the observation, "
             "so every variant sees the same trajectory and the same "
             "decisions, which is what makes an encoding ablation a paired "
             "comparison instead of two experiments. Names are those "
             "cr_sim.api.encoding.parse_observation accepts, plus a bare "
             "flag name for a single-change variant.",
    )
    parser.add_argument("--match-seconds", type=int, default=120)
    parser.add_argument("--frame-skip", type=int, default=30)
    parser.add_argument("--tps", type=int, default=20)
    parser.add_argument(
        "--horizon-seconds", type=float, default=15.0,
        help="how far the SEARCH projects each candidate. Not the reward's "
             "projection horizon -- that is --reward-horizon-seconds, and the "
             "two are different quantities that happen to share a unit.",
    )
    # The reward the value column is harvested under. It had none: this script
    # built its env with no reward_weights at all, so every demonstration set
    # ever collected carried value targets from the simple shaped reward while
    # every fine-tune ran `projected`. The clone's critic is the part
    # reinforcement learning inherits, so it arrived predicting a quantity
    # nobody was optimising -- +1.48 against returns averaging +0.47. The
    # defaults here match cr_sim.train.run's, which is the point.
    parser.add_argument(
        "--reward", choices=("simple", "five-term", "projected"),
        default="projected",
        help="which reward the recorded value targets are computed under. "
             "Must match the reward the clone is later fine-tuned with, or "
             "the inherited critic predicts the wrong quantity.",
    )
    parser.add_argument(
        "--tower-weight", type=float, default=1.0,
        help="weight on the surviving-tower-health difference inside "
             "--reward projected's potential. Mirrors cr_sim.train.run's flag "
             "of the same name so a demonstration set can be harvested under "
             "the weights a fine-tune will start from -- the value column is a "
             "return under this reward, and the clone's critic is what PPO "
             "inherits.",
    )
    parser.add_argument("--elixir-weight", type=float, default=0.0)
    parser.add_argument(
        "--reward-horizon-seconds", type=float, default=3.0,
        help="the projected reward's lookahead, matching run.py's "
             "--horizon-seconds. 0 disables the projection.",
    )
    parser.add_argument("--candidates", type=int, default=14)
    parser.add_argument(
        "--proposer", default="none",
        help="a checkpoint whose policy proposes which placements the search "
             "spends its branches on, or 'none' for the stratified random "
             "draw that produced every shard on this machine. 'none' is the "
             "default and the fallback: it is byte-for-byte the old bot, so "
             "an unflagged run reproduces exactly.")
    parser.add_argument(
        "--proposer-temperature", type=float, default=0.0,
        help="0 ranks by the policy's logits with a stable argsort and "
             "touches no random number generator at all. Above zero it "
             "samples without replacement from a generator this proposer "
             "owns -- never torch's global stream, which is unseeded and is "
             "why every sampled number in runs/_anchor is unreproducible.")
    parser.add_argument(
        "--policy-candidates", type=int, default=9,
        help="how many of --candidates the proposer supplies. Clamped to "
             "candidates - max(2, candidates // 3), so the target's support "
             "always spans placements the policy did not choose. Ignored "
             "without --proposer.")
    parser.add_argument(
        "--baseline-fallback", type=float, default=BASELINE_FALLBACK_RATE,
        help="the unguided bot's min_spread fallback rate. A guided shard "
             "more than ten points above it has collapsed its own target and "
             "the run says so rather than letting it be merged.")
    parser.add_argument(
        "--opponent", choices=("random", "bot"), default="random",
        help="who the expert plays. 'random' matches how the result is "
             "measured; 'bot' produces harder positions but takes twice as "
             "long, since both sides then search.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    data = LogicData.load(DEFAULT_BUILD)
    levels, registry = build_level_table(data), build_card_registry(data)
    # Shards must not replay each other's battles, or the merged set is the
    # same few games repeated and the policy learns those rather than the game.
    offset = args.shard * 10_000

    def make_env(index: int) -> CRSimEnv:
        seed = offset + index
        opponent = None
        if args.opponent == "random":
            opponent = _random_opponent(50_000 + seed)
        else:
            foe = SearchBot(Team.RED, SearchBotConfig(
                horizon_seconds=args.horizon_seconds,
                candidates=args.candidates, seed=seed))

            def opponent(observation, mask, battle=None, _foe=foe):
                return _foe(observation, mask, battle)
            opponent.wants_battle = True
        return CRSimEnv(
            data, levels, registry, DEFAULT_DECK, DEFAULT_DECK,
            ticks_per_second=args.tps, frame_skip=args.frame_skip,
            max_ticks=args.tps * args.match_seconds,
            tower_level=args.tower_level,
            # Through a shim, because _reward_weights reads
            # `horizon_seconds` and in this script that name is the *search*
            # horizon (15s), not the reward's projection horizon (3s). Passing
            # args directly would silently build the projection with a
            # five-times-too-long lookahead -- the same shape of bug as the
            # one this flag exists to fix.
            reward_weights=_reward_weights(SimpleNamespace(
                reward=args.reward,
                horizon_seconds=args.reward_horizon_seconds,
                elixir_weight=args.elixir_weight,
                # Named explicitly rather than defaulted with getattr. A shim
                # that quietly tolerates a missing field is how a caller ends
                # up building a different reward from the one it thinks it is;
                # if _reward_weights grows a knob, this must fail loudly and
                # be told what to do with it.
                tower_weight=args.tower_weight)),
            opponent_policy=opponent)

    # --------------------------------------------------- who proposes what

    guided = args.proposer not in ("", "none", "None")
    build_proposer = None
    if guided:
        from cr_sim.train.evaluate import load_policy

        probe = make_env(0)
        probe.reset(seed=offset)
        proposer_net = load_policy(Path(args.proposer), probe)
        build_proposer = proposer_factory(
            proposer_net, probe.action_space.nvec,
            temperature=args.proposer_temperature, seed=args.shard)
        print(f"shard {args.shard}: candidates proposed by {args.proposer} "
              f"at temperature {args.proposer_temperature:g}, "
              f"{args.policy_candidates} of {args.candidates} placements",
              flush=True)

    search = SearchBotConfig(
        horizon_seconds=args.horizon_seconds,
        candidates=args.candidates, seed=args.shard,
        policy_candidates=args.policy_candidates if guided else 0)

    def make_expert(env):
        # A proposer per battle, keyed by the battle's own seed, so its stream
        # is a function of (shard, battle seed, decision) and never of how many
        # episodes came before -- the flaw _random_opponent still has.
        proposer = (None if build_proposer is None
                    else build_proposer(int(env.battle.config.seed)))
        bot = SearchBot(Team.BLUE, search, proposer)

        def expert(observation, mask, battle=None):
            return bot(observation, mask, battle)
        # collect reads the search's scores off this, and they are the real
        # training target -- the chosen move is not a function of the state.
        expert.bot = bot
        return expert

    started = time.perf_counter()

    def progress(done: int, samples: int) -> None:
        rate = (time.perf_counter() - started) / done
        print(f"shard {args.shard}: {done}/{args.episodes} episodes, "
              f"{samples} decisions, {rate:.0f}s/episode, "
              f"{(args.episodes - done) * rate / 60:.0f} min left", flush=True)

    # The *effective* count, not the flag's. SearchBot clamps the proposal to
    # leave the random floor intact, and a shard stamped with the number that
    # was asked for over a bot that took fewer is a file declaring something
    # about itself rather than recording it -- the exact failure
    # Demonstrations.observation exists to prevent.
    effective = search.effective_policy_candidates if guided else 0
    if guided and effective != args.policy_candidates:
        print(f"shard {args.shard}: --policy-candidates "
              f"{args.policy_candidates} clamped to {effective}, keeping "
              f"{search.random_floor} random candidates so the target's "
              "support still spans placements the policy did not choose",
              flush=True)
    stamp = proposer_identity(
        Path(args.proposer) if guided else None,
        temperature=args.proposer_temperature,
        policy_candidates=effective)
    meta = {
        # The full weight tuple these value targets were harvested under,
        # read off a real environment rather than asserted. Demonstrations
        # records only the variant *name*, and the shipped shards say
        # reward='projected' while docs/training.md records that they were
        # collected under --elixir-weight 0 -- which is not in the file. The
        # clone's critic is what PPO inherits, and that exact mismatch has
        # already cost this project once: an inherited critic predicting +1.48
        # against returns averaging +0.47.
        "reward_weights": reward_name(make_env(0)),
        "proposer": stamp,
        "proposer_checkpoint": args.proposer if guided else "",
        "proposer_temperature": float(args.proposer_temperature),
        "candidates": int(args.candidates),
        "policy_candidates": int(effective),
        "policy_candidates_requested": int(args.policy_candidates if guided else 0),
        "min_random_candidates": int(search.random_floor),
        "search_horizon_seconds": float(args.horizon_seconds),
        # A forward is not bit-stable across thread counts, so a different
        # reduction order can flip a near-tie and change which placements were
        # proposed. Recorded, so a divergence is detectable rather than silent.
        "torch_threads": _torch_threads(),
    }

    names = [n.strip() for n in args.observations.split(",") if n.strip()]
    if names == ["v1"]:
        demos = collect(make_env, make_expert, episodes=args.episodes,
                        on_episode=progress,
                        reward_name=args.reward,
                        observation_name="v1",
                        proposer_name=stamp, meta=meta)
        results = {"": demos}
    else:
        results = collect(
            make_env, make_expert, episodes=args.episodes, on_episode=progress,
            reward_name=args.reward, proposer_name=stamp, meta=meta,
            variants={name: parse_observation(name) for name in names})

    for name, demos in results.items():
        directory = args.out / name if name else args.out
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"shard-{args.shard:02d}.npz"
        demos.save(path)
        measured = json.loads(demos.meta) if demos.meta else {}
        print(f"shard {args.shard}: wrote {len(demos)} decisions from "
              f"{demos.episodes} episodes to {path} "
              f"(play rate {demos.play_rate:.0%}, "
              f"observation {demos.observation!r}, reward {demos.reward!r}, "
              f"proposer {demos.proposer!r}, "
              f"{(time.perf_counter() - started) / 60:.0f} min)", flush=True)
        print(f"shard {args.shard}: candidate spread "
              f"{measured.get('spread_mean', 0.0):.4f} mean, min_spread "
              f"fallback {measured.get('min_spread_fallback_rate', 0.0):.1%}",
              flush=True)
        refusal = collapse_refusal(demos, args.baseline_fallback)
        if refusal:
            print(refusal, flush=True)
    return 0


def _torch_threads() -> int:
    try:
        import torch

        return int(torch.get_num_threads())
    except Exception:                                   # pragma: no cover
        return 0


def collapse_refusal(demos: Demonstrations, baseline: float,
                     margin: float = 0.10) -> str:
    """A refusal to merge a shard whose target has collapsed, or "".

    The shard is written either way -- throwing away a twenty-minute
    collection because a diagnostic tripped is how a number stops being
    measured at all -- but the run says plainly that it must not be merged,
    and says what it measured.

    The failure this catches is self-reinforcing and would otherwise be
    invisible. A policy proposer nominates placements the policy already
    rates highly, so their engine-scored values sit closer together, so more
    rows fall below ``min_spread`` and collapse to a one-hot on the chosen
    action, so the clone trains on the policy's own preference dressed as the
    search's belief -- and the next round's proposer is more confident still.
    That is the shape of the measured disaster the ``min_spread`` floor was
    added for: 86% of wait-states carrying a uniform target, the pass action
    never the argmax in 10,940 decisions, and a clone that played a card at
    every single decision.

    It has not happened yet. The first guided collection on this machine came
    back at a *higher* mean spread than the unguided one -- 0.0936 against
    0.0585, both at a 0% fallback rate -- so the proposer is nominating
    placements that differ in value rather than placements that are alike.
    That is four episodes and one checkpoint. The gate stays.
    """
    if not demos.meta:
        return ""
    rate = float(json.loads(demos.meta).get("min_spread_fallback_rate", 0.0))
    if rate <= baseline + margin:
        return ""
    return (
        f"REFUSED FOR MERGING: {rate:.1%} of this shard's targets fell back "
        f"to the single chosen action, against {baseline:.1%} for the "
        f"unguided draw -- {rate - baseline:+.1%}, past the {margin:.0%} "
        "margin. The proposal has collapsed the target's support onto what "
        "the policy already believed, and a clone trained on it sharpens a "
        "preference rather than improving one. The shard is on disk; do not "
        "merge it. Lower --policy-candidates or raise "
        "--min-random-candidates and collect again.")


if __name__ == "__main__":
    sys.exit(main())
