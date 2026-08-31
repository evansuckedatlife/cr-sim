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
import os
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
from .nets import POLICY_HEADS, net_config_for
from .ppo import PPOConfig, train
from .schedule import anneal_to_zero, constant_schedule, knob_for_reward
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


def save_checkpoint(payload: dict, path: Path) -> Path:
    """Write a checkpoint so that a crash mid-write cannot destroy the old one.

    ``torch.save`` straight onto the destination truncates it first, so the
    file spends the length of a 20 MB serialisation being neither the old
    checkpoint nor the new one. This machine bugchecks, ``supervise.ps1``
    rewrites ``checkpoint.pt`` every three updates, and ``--resume`` from that
    one file is the entire crash-resilience strategy -- so the window where
    the strategy has nothing to resume from was being opened several times an
    hour, on purpose, by the thing meant to protect it.

    A sibling temporary file and ``os.replace`` instead. ``os.replace`` is
    atomic within one volume on Windows as well as POSIX, and the temporary
    sits beside the destination precisely so the two are always on the same
    volume. A crash before the replace leaves the previous checkpoint whole
    and a ``.tmp`` file the next save overwrites; a crash after it leaves the
    new one whole. There is no third outcome.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except BaseException:
        # A half-written temporary is not evidence of anything and would be
        # loaded by nothing, but leaving 20 MB of it behind on every crash
        # fills a disk that has 3.4 GB free.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return path


#: The settings that decide which arena a metrics row was measured in. Two
#: readings that disagree on any of these are not one series: at tower level 11
#: a 120-second match ends with 92% of matches drawn and at 5 it does not, and a
#: lift is a difference of returns, so the reward is in the numerator and the
#: denominator both. `_ladder_ratings` already refuses a rating table that
#: disagrees about `tower_level`; this is the same refusal for a resume.
_ARENA_KEYS = ("tower_level", "elixir_weight", "reward", "match_seconds",
               "frame_skip")

#: The keys a `register_job.py` placeholder row is made of. A metrics file
#: holding nothing outside this set has measured nothing, so a name registered
#: as a job and then trained is not a name that loses anything. Kept identical
#: to `register_job._PLACEHOLDER_KEYS` on purpose: two guards over the same
#: directory that disagree about what "empty" means are worse than one.
_PLACEHOLDER_KEYS = {"updates", "steps", "episodes"}


def _stamp_aside(path: Path) -> "Path | None":
    """Move ``path`` out of the way under a timestamp, and say where it went.

    The same move ``scripts/register_job.py`` makes, for the same reason: a
    name that already holds a run is refused, and the escape hatch has to keep
    the bytes rather than truncate them. ``metrics.<stamp>.jsonl`` because
    ``watch.discover`` globs the exact string ``metrics.jsonl``, so what is
    moved aside does not reappear on the page as a second copy of the entry.
    """
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for attempt in range(100):
        suffix = stamp if not attempt else f"{stamp}-{attempt}"
        target = path.with_name(f"{path.stem}.{suffix}{path.suffix}")
        if not target.exists():
            path.rename(target)
            return target
    raise SystemExit(f"cannot find a free name to move {path} aside")


def _measured_rows(metrics: Path) -> int:
    """How many rows in ``metrics`` carry something that was measured.

    A line this cannot parse counts: the question is "is there anything here
    to lose", and answering "no" because a file is malformed is exactly the
    wrong way round.
    """
    if not metrics.is_file():
        return 0
    count = 0
    for line in metrics.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            count += 1
            continue
        if not isinstance(row, dict) or set(row) - _PLACEHOLDER_KEYS:
            count += 1
    return count


def _guard_run_directory(out: Path, args: Any) -> None:
    """Refuse to start a fresh run on top of one that already exists.

    ``register_job.py`` grew this guard because one mistyped ``--name``
    replaced a training run's rows and its config. ``cr_sim.train.run`` is the
    thing that *produces* those rows and had no guard at all: it opens
    ``metrics.jsonl`` with mode ``"w"`` whenever ``--resume`` is absent,
    rewrites ``config.json`` unconditionally, and overwrites
    ``checkpoint.pt``, ``best.pt`` and ``final.pt`` as it goes -- with exit
    code 0 and a "done" line. ``runs/`` is gitignored and there is no backup.

    Measured before this existed: a directory holding updates 1-4 out to step
    128, re-run under the same name without ``--resume``, came back holding
    update 1 alone, a config claiming ``total_steps`` 32 and ``seed`` 7, and a
    checkpoint at 32 steps. Nothing said so.

    ``--resume`` is the sanctioned way to continue a run and is let through
    here; ``--replace`` is the escape hatch, and it stamps the old files aside
    rather than deleting them.
    """
    if args.resume:
        return
    metrics = out / "metrics.jsonl"
    rows = _measured_rows(metrics)
    weights = [name for name in ("checkpoint.pt", "best.pt", "final.pt")
               if (out / name).is_file()]
    if not rows and not weights:
        return

    if args.replace:
        for name in ("metrics.jsonl", "config.json",
                     "checkpoint.pt", "best.pt", "final.pt"):
            _stamp_aside(out / name)
        return

    held = []
    if rows:
        held.append(f"{metrics} holds {rows} row(s) carrying measurements")
    if weights:
        held.append("it holds " + ", ".join(weights))
    raise SystemExit(
        f"{out} is not an empty slot: {'; and '.join(held)}. Starting a fresh "
        f"run here would truncate the metrics, rewrite config.json and "
        f"overwrite every checkpoint, and runs/ is gitignored with no backup. "
        f"Pass --resume to continue this run from its checkpoint, or --name "
        f"something else to start a new one, or --replace to move the "
        f"existing files aside under a timestamp first.")


def _check_resume_arena(out: Path, record: dict) -> dict:
    """Keep a resumed run's config honest about the rows it already holds.

    ``config.json`` is rewritten from the current CLI on every start, resume
    included -- so resuming a run trained at ``--tower-level 11`` with today's
    launcher, which passes 5, trained the remainder in a different arena *and*
    rewrote the run's own config to claim it had always been 5. The earlier
    rows record no level of their own; that config was the only thing that
    knew what they meant.

    Two rules, and between them nothing already on disk is relabelled:

    * A key the recorded config carries and the CLI disagrees with is refused,
      the way ``_ladder_ratings`` refuses a rating table fitted at another
      tower level.
    * A key the recorded config does not carry is left out rather than
      invented. Seven runs here predate these being recorded at all, and
      stamping today's value onto them would put a number where the file's
      silence is the only honest answer.
    """
    config = out / "config.json"
    if not config.is_file():
        return record
    try:
        recorded = json.loads(config.read_text(encoding="utf-8"))
    except (ValueError, OSError, UnicodeDecodeError):
        # Unreadable, so nothing can be checked against it -- and overwriting
        # it is what destroys the evidence. Leave it where it is.
        raise SystemExit(
            f"--resume, but {config} cannot be read, so what this run's "
            f"existing rows were measured under cannot be established. Move it "
            f"aside by hand if the rows are genuinely worth continuing.")
    if not isinstance(recorded, dict):
        return record

    out_record = dict(record)
    for key in _ARENA_KEYS:
        if key not in recorded:
            # Never invent a claim about rows that predate the key.
            out_record.pop(key, None)
            continue
        if recorded[key] != record.get(key):
            raise SystemExit(
                f"--resume, but {config} records {key}={recorded[key]!r} and "
                f"this command asks for {key}={record.get(key)!r}. The rows "
                f"already in this run were measured at {recorded[key]!r}; "
                f"training the rest at {record.get(key)!r} puts two arenas in "
                f"one series and rewrites the only record of what the first "
                f"half meant. Pass --{key.replace('_', '-')} "
                f"{recorded[key]!r} to continue this run, or --name something "
                f"else to start a new one.")
    return out_record


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
    parser.add_argument(
        "--probe", choices=("fixed", "rotating", "ladder"), default="fixed",
        help="what the in-run measurement is. 'fixed' -- the default, and "
             "unchanged -- is lift over a random control on one fixed seed "
             "list, which every existing run was steered by. 'rotating' is "
             "the same lift on a different seed block each reading, so the "
             "three-reading promotion window stops re-selecting one seed "
             "set's luck. 'ladder' rates the policy against named anchors "
             "and promotes on Elo instead: the lift is saturated -- every "
             "result this project has lands in a band 0.024 sd wide -- while "
             "the rating spreads the same battles over about 300 points. "
             "Defaulted to 'fixed' so an old command still means what it "
             "meant.")
    parser.add_argument(
        "--ladder-anchor", action="append", default=[],
        help="who the ladder probe rates against: 'random', "
             "'search-c6h8', or a checkpoint. Repeatable. Defaults to "
             "random alone, which is one rung and not a scale.")
    parser.add_argument(
        "--ladder-ratings", type=Path, default=None,
        help="a ladder.json whose fitted ratings pin this run's anchors. "
             "Without it the anchors sit at 0 and ladder_elo is only "
             "relative to them; with it the probe reports "
             "ladder_elo_vs_expert without ever playing the expert.")
    parser.add_argument("--horizon", type=int, default=256)
    parser.add_argument("--tps", type=int, default=20, help="engine tick rate")
    parser.add_argument("--frame-skip", type=int, default=10, help="ticks per decision")
    parser.add_argument("--match-seconds", type=int, default=120)
    parser.add_argument(
        "--tower-level", type=int, default=5,
        help="Crown Tower level. Defaults to 5, and that default is the "
             "measurement: at 11 a 120-second match ends with 92%% of tower "
             "health untouched and 92%% of matches drawn, so crowns -- the "
             "only real objective -- almost never fire and the agent learns "
             "from shaping alone. Level 5 halves the draw rate at no extra "
             "compute. It defaulted to 11 for the first year here and "
             "runs/learn-1m-factored-lvl11 is 557,056 steps of what that "
             "buys. A training-environment choice, not a change to the "
             "simulator: pass 11 explicitly to evaluate what transfers. "
             "Every run records its own level in config.json, so nothing "
             "already measured is reinterpreted by this default moving.",
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
        "--head", choices=POLICY_HEADS, default="flat",
        help="'flat' is one linear layer over all 720 actions; 'factored' "
             "picks the card, then the tile, with the tile head conditioned "
             "on an embedding of the card and its weights shared across "
             "cards. Not a correctness difference -- a flat masked "
             "categorical can represent anything the factorisation can -- but "
             "a sample-efficiency one, and placements are the sparse part. "
             "'factored-stats' is 'factored' with the card lookup replaced by "
             "an encoder over the card's own statistics -- hitpoints, damage, "
             "reach, speed, what it targets -- so the head conditions on what "
             "a card does rather than on which vocabulary slot it landed in, "
             "and a card that was never in a training deck gets a "
             "conditioning vector for free. 'conv' emits the placement logits "
             "as a 1x1 convolution over the trunk's own feature map.",
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
        "--elixir-weight", type=float, default=0.0,
        help="weight on the elixir lead inside --reward projected's "
             "potential. This is what makes a card cost something, and it is "
             "also why passing pays: spending drops the potential now while "
             "the card's effect on the board takes longer than the "
             "projection's horizon to appear. Measured on the clone's own "
             "rollouts, a pass earns +0.071 more reward than a placement at "
             "0.3 and -0.010 less at 0.0, which is why this defaults to 0.0. "
             "The searching bot needed it at 0.0 for the same reason: at 0.3 "
             "it never played a card at all. Pass 0.3 explicitly to train an "
             "agent that is charged for its elixir.",
    )
    parser.add_argument(
        "--tower-weight", type=float, default=1.0,
        help="weight on the surviving-tower-health difference inside "
             "--reward projected's potential, against 1.0 for a crown. This "
             "had no flag at all and was pinned at 1.0, so it was neither "
             "reachable nor recorded -- a run's config.json wrote down "
             "--shaping, which that reward never reads, and not the "
             "coefficient actually scaling its tower term.",
    )
    parser.add_argument(
        "--anneal", action="store_true",
        help="drive the shaping to zero over training, ending on the sparse "
             "crown objective. Which coefficient that is depends on --reward "
             "and is NOT --shaping: 'projected' anneals tower and elixir, "
             "'five-term' the five non-crown weights, 'simple' the shaping "
             "weight itself -- the only reward that reads it. crowns is never "
             "annealed; it is the objective, not shaping. The reason: the "
             "projected reward is an exact potential, so return-to-go "
             "telescopes to a near-martingale and GAE's advantages are mostly "
             "noise -- which is why a million steps moved greedy by +0.024. "
             "Off by default, and off is bit-identical to a run without it.",
    )
    parser.add_argument(
        "--anneal-start", type=int, default=0,
        help="step the anneal begins at; before it the weights are held at "
             "their starting values.",
    )
    parser.add_argument(
        "--anneal-end", type=int, default=0,
        help="step the anneal reaches zero at. 0 means 80%% of --steps, "
             "leaving the last fifth of the run stationary on the sparse "
             "objective -- otherwise the final checkpoint is measured under a "
             "weight that was still moving, and the objective the whole "
             "schedule exists to reach never gets a stretch to be measured "
             "on.",
    )
    parser.add_argument(
        "--shaping", type=float, default=0.01,
        help="weight on tower-health difference, under --reward simple ONLY. "
             "Inert under 'projected' and 'five-term': every call site sits "
             "inside the branch those two rewards do not take, and 0.01 "
             "against 5.00 -- a five hundred fold change -- is bit-identical "
             "under both. Use --tower-weight and --elixir-weight for "
             "'projected'. At 0.01 a whole match's tower damage is worth "
             "about 0.02 against 1.0 per crown, so the simple reward is "
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
        "--replace", action="store_true",
        help="start fresh in a --name that already holds a run, moving its "
             "metrics.jsonl, config.json and checkpoints aside under a "
             "timestamp first. Not a delete: the old files stay in the "
             "directory under names the watcher does not enumerate. Without "
             "this, a name already holding measurements or a checkpoint is "
             "refused rather than overwritten.",
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


#: The scale the in-run probe measures returns on, fixed for every run and
#: independent of whatever the training reward is doing.
#:
#: ``_eval_env`` used to build from ``_reward_weights(args)`` -- the *training*
#: reward -- and that quietly makes the promotion criterion a function of the
#: training schedule. ``eval_lift_sd`` is a difference of returns against a
#: control that is evaluated once and cached, spread and all. Anneal the
#: training shaping and the policy arm's returns shrink while the cached
#: control keeps the scale it was measured on, so the lift series drifts from
#: nothing but the reward scale, and the run promotes toward its own earliest,
#: highest-shaping checkpoints. watch.py's run-wide ``best_lift = max`` would
#: then be systematically the first evaluation.
#:
#: Pinning removes that whole class of drift, and buys something the lift
#: never had: two runs trained under different rewards now measure their lift
#: on the same scale. It does mean a run whose *training* reward is not this
#: one reports a lift on a different scale from the one it would have reported
#: before -- which is why every row now carries ``eval_reward`` saying so.
EVAL_REWARD = ProjectionWeights()

#: Distinguishes 'the reward --reward selected' from an explicit None,
#: which is itself a valid weights argument -- it selects the simple
#: reward. A default of None could not tell the two apart.
_TRAINING_REWARD = object()


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
            tower=args.tower_weight,
        )
    return None


def _reward_schedule(args):
    """How the shaping moves over this run -- constant unless asked.

    The default is a constant schedule, which is today's behaviour exactly:
    nothing is ever pushed, so no reward object is ever rebuilt and no RPC is
    ever sent. ``--anneal`` drives the shaping fields of whichever knob
    ``--reward`` actually selects to zero, leaving ``crowns`` and
    ``horizon_seconds`` alone. See cr_sim.train.schedule for why the knob is
    not ``--shaping``.
    """
    knob = knob_for_reward(args.reward)
    weights = _reward_weights(args)
    values = (weights.as_dict() if weights is not None
              else {"shaping": args.shaping})
    if not args.anneal:
        return constant_schedule(knob, values).resolved(args.steps)
    return anneal_to_zero(
        knob, values,
        start_step=args.anneal_start, end_step=args.anneal_end,
    ).resolved(args.steps)


#: The mode the in-run ladder probe plays. Greedy, because a rating wants an
#: opponent that plays the same line every time, and because greedy reproduces
#: bit-identically. Named once so the ratings table this run is checked
#: against and the probe that consumes it cannot drift apart.
_LADDER_PROBE_MODE = "greedy"


def _ladder_ratings(path, *, mode: str, observation: str, tower_level: int):
    """The anchors' fitted ratings, refusing a table fitted on another game.

    A ladder.json records the mode, observation and tower level it was fitted
    under, and the loader read none of them. Measured on the same players and
    the same seed block, a ``sampled`` table rates the v1 clone +183.7 where
    the ``greedy`` table rates the same weights +393.0 -- 209 Elo apart, with
    the field reordered as well -- and feeding each into the probe's own
    arithmetic on one greedy edge gives ``ladder_elo`` +337 against +129.
    Every row stamps ``ladder_mode: greedy`` either way. run_ladder's own
    ``--mode`` help says it plainly: "sampled is a different policy and needs
    its own ladder, its own ratings".

    ``tower_level`` is checked only where the file records it. Tables written
    before that field existed cannot answer the question, and refusing them
    outright would make ``--ladder-ratings`` unusable against every table on
    this machine; a table that does record it and disagrees is refused.
    """
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded = str(loaded.get("mode", ""))
    if recorded != mode:
        raise SystemExit(
            f"{path} holds a {recorded!r} rating table and the in-run probe "
            f"plays {mode!r}. Greedy and sampled are two different policies "
            "rated on two different scales -- measured 209 Elo apart on the "
            "same weights and the same seeds -- and pinning one against the "
            f"other moves every ladder_elo this run writes. Rate the anchors "
            f"with `scripts/run_ladder.py --mode {mode}`.")
    recorded = str(loaded.get("observation", ""))
    if recorded != observation:
        raise SystemExit(
            f"{path} rated its players under observation {recorded!r} and "
            f"this run encodes {observation!r}. The environment encodes one "
            "observation for both sides, so the two cannot share a ladder at "
            "all.")
    level = loaded.get("tower_level")
    if level is not None and int(level) != int(tower_level):
        raise SystemExit(
            f"{path} rated its players at tower level {int(level)} and this "
            f"run trains at {int(tower_level)}. At level 11 the towers "
            "outlast the match and 90% of battles draw; a rating from one "
            "level is not on the other's scale.")
    return {str(r["name"]): float(r["elo"]) for r in loaded.get("ratings", [])}


def _ladder_anchors(specs, env):
    """The probe's anchors, with their weights loaded off ``env``'s shapes.

    ``Player.load`` is what puts a network behind a checkpoint anchor, and
    the in-run path never called it: ``--ladder-anchor
    checkpoints/headablate-flat.pt`` trained happily to the first evaluation
    and then died in ``FrozenOpponent._snapshot`` on ``'NoneType' object has
    no attribute 'to'``, twenty updates in with no evaluation ever written. A
    policy-guided search anchor failed the same way one layer up. Only
    ``random`` and an unproposed ``search-cXhY`` ever worked, which is not the
    feature -- the anchors this exists for are the clone and the expert.

    Loading here also routes every anchor through ``check_observation``, which
    is the only thing that would catch a v2 checkpoint entering a v1 run.
    """
    from .ladder import parse_player

    return [parse_player(spec).load(env) for spec in (specs or ["random"])]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Resolved before anything is written, because config.json records it.
    anchor_path = args.kl_reference or args.init_from

    out = args.out / args.name
    # Both guards before the mkdir, and before anything at all is loaded or
    # written. A fresh run on top of an existing one truncates its metrics and
    # overwrites its checkpoints; a resume into a different arena relabels the
    # rows it is continuing. A refusal that only arrives after twenty minutes
    # of rollouts is not a guard, and the config write is a hundred lines
    # further down -- so the arena is settled here and the result carried to
    # it, rather than checked where it happens to be written.
    _guard_run_directory(out, args)
    arena = _check_resume_arena(out, {key: getattr(args, key)
                                      for key in _ARENA_KEYS}) \
        if args.resume else None
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

    def _env(opponent=None, *, reward_weights=_TRAINING_REWARD) -> CRSimEnv:
        return CRSimEnv(
            data, levels, registry, DEFAULT_DECK, DEFAULT_DECK,
            observation=observation,
            ticks_per_second=args.tps,
            frame_skip=args.frame_skip,
            max_ticks=args.tps * args.match_seconds,
            tower_level=args.tower_level,
            reward_shaping_weight=args.shaping,
            reward_weights=(_reward_weights(args)
                            if reward_weights is _TRAINING_REWARD
                            else reward_weights),
            opponent_policy=opponent,
        )

    def _eval_env() -> CRSimEnv:
        """The environment the honest evaluation is played in.

        A *random* opponent, not an idle one. This used to be ``_env(None)``,
        which never plays a card, while the large paired verdicts faced a
        random agent -- and both were reported as "lift" and compared to each
        other. They were never comparable: the control wins 92% of the idle
        matches and 26% of the random ones.

        And a *pinned* reward, not the training one. See EVAL_REWARD: a probe
        whose scale follows the training schedule turns the promotion
        criterion into a function of that schedule.
        """
        return _env(_random_opponent(90_000), reward_weights=EVAL_REWARD)

    #: The environments the rollout actually steps, so a scheduled weight can
    #: be pushed to them. Under --workers they live in other processes and
    #: this holds only the shape probe, which is why CRSimVecEnv needs the
    #: same push over its own pipe.
    local_envs: list[CRSimEnv] = []

    def make_env(index: int) -> CRSimEnv:
        # Each environment gets its own opponent, so eight parallel battles do
        # not face an identical sequence of placements and report a smoother
        # result than the policy has earned.
        if args.opponent == "random":
            built = _env(_random_opponent(args.seed * 1000 + index))
            local_envs.append(built)
            return built
        if args.opponent == "self":
            # Filled in once the network exists; until then the environment
            # faces an idle side, which only affects the first rollout.
            holder: list = []
            opponents.append(holder)
            built = _env(
                lambda obs, mask, h=holder: h[0](obs, mask) if h else (NOOP_SLOT, 0, 0))
            local_envs.append(built)
            return built
        built = _env(None)
        local_envs.append(built)
        return built

    # Resolved before config.json, because config.json records it -- and
    # resolved once, so the endpoints a reader sees are the endpoints the run
    # used rather than a flag they have to re-derive from.
    schedule = _reward_schedule(args)

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
    record = {**asdict(config), "deck": list(DEFAULT_DECK), "tps": args.tps,
              "frame_skip": args.frame_skip, "match_seconds": args.match_seconds,
              "shaping": args.shaping, "reward": args.reward,
              "tower_level": args.tower_level,
              "horizon_seconds": args.horizon_seconds,
              "opponent": args.opponent, "head": args.head,
              # Self-play's cadence, and the weights the run started
              # from. None of these were recorded, so a run directory
              # could not answer "was this actually self-play, against
              # what, drawn how often, from which clone?" -- which is
              # exactly the question its flat metrics provoke.
              "pool_size": args.pool_size,
              "refresh_every": args.refresh_every,
              # 0 disables the ladder entirely, which is worth stating:
              # a run with no ancestor rows is not a run whose ladder
              # broke, it is a run that never measured one.
              "ancestor_episodes": args.ancestor_episodes,
              "opponent_temperature": args.opponent_temperature,
              "init_from": str(args.init_from) if args.init_from else None,
              "resumed": bool(args.resume),
              "kl": args.kl, "elixir_weight": args.elixir_weight,
              "kl_reference": str(anchor_path) if anchor_path else None,
              "observation": args.observation,
              "observation_channels": list(grid_channels(observation)),
              # Which opponent the in-run lift is measured against, read
              # off a real evaluation environment rather than asserted.
              # A run's own lift series is only comparable to another
              # run's when these agree.
              "eval_opponent": opponent_name(_eval_env()),
              # One new key, not four: watch.py pairs two runs for A/B
              # only while their config key sets differ by at most four,
              # so every field added here is spent out of that budget.
              "probe": args.probe,
              # Two new keys, nested, and that is the whole budget:
              # watch.py pairs two runs for A/B only while their config
              # key sets differ by at most four. "shaping" above stays
              # where it is for the same reason -- deleting a key
              # changes the key set and would make every new run
              # unpairable with every old one -- but shaping_is_inert
              # inside here finally says what it is.
              "reward_schedule": {
                  **schedule.as_dict(),
                  "shaping_is_inert": args.reward != "simple",
              },
              # The scale the in-run lift is measured on, which is now a
              # constant rather than whatever --reward happened to be.
              "eval_reward": {"kind": "projected", **EVAL_REWARD.as_dict()},
              "eval_episodes": args.eval_episodes}
    # Written from the CLI every start, resume included -- so on a resume the
    # arena-defining half of it is whatever `_check_resume_arena` settled at
    # the top of this function, which is a key short wherever the recorded
    # config predates that key. Silence about rows nobody can characterise is
    # the honest answer; a number is not.
    if arena is not None:
        for key in _ARENA_KEYS:
            if key not in arena:
                record.pop(key, None)
    (out / "config.json").write_text(json.dumps(record, indent=2),
                                     encoding="utf-8")

    # The anchors' fitted ratings, read once. Without them a ladder probe
    # still rates the policy against its anchors, but every anchor sits at 0
    # and the rating is only relative to them -- ladder_elo_vs_expert needs a
    # rating for the expert, which is what the offline ladder produced.
    offline_ratings: dict[str, float] = {}
    if args.ladder_ratings is not None:
        offline_ratings = _ladder_ratings(
            args.ladder_ratings, mode=_LADDER_PROBE_MODE,
            observation=args.observation, tower_level=args.tower_level)

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
    #: The best rolling value of whatever statistic is steering promotion --
    #: a lift, or the ladder's rating. Not called "lift": under --probe
    #: ladder it holds an Elo, and one name over two scales is what this
    #: project has already paid three rounds of invalid comparisons for.
    best = {"score": float("-inf")}
    #: Recent readings of that statistic, for promoting on their mean.
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
        #: The weight tuple most recently handed to the environments. Compared
        #: rather than pushed blindly, so a constant schedule -- the default --
        #: sends nothing at all and a run without --anneal is bit-identical to
        #: one from before this existed.
        #:
        #: Seeded from step 0 because that is what ``_env()`` built the
        #: environments with, whatever step the run is starting at. On a
        #: resume that is a *stale* value, which is the point -- see
        #: ``_push_reward_weights``.
        pushed = {"weights": schedule.at(0)}

        def _push_reward_weights(step: int) -> None:
            """Hand the schedule's weights at ``step`` to every environment.

            One guard, not two. A schedule that has not moved is not pushed,
            and that single condition is what makes a constant schedule -- the
            default -- cost nothing and change nothing. An ``if not
            schedule.is_constant`` short-circuit as well read as defensive and
            was worse than that: with two independent conditions guarding one
            behaviour, neither can be broken on its own, so no test could hold
            either of them to account.
            """
            values = schedule.at(step)
            if values == pushed["weights"]:
                return
            weights, shaping = schedule.weights_at(step)
            # Both, always. A field _env() sets and the workers do not is the
            # exact shape of the --tower-level bug, which trained every
            # rollout at level 11 while config.json recorded 5.
            for env in local_envs:
                env.set_reward_weights(weights, shaping_weight=shaping)
            if parallel is not None:
                parallel.set_reward_weights(weights, shaping_weight=shaping)
            pushed["weights"] = values

        def record(stats: dict) -> None:
            # The schedule, before anything else this update does. Steps, not
            # updates, because --resume keeps the step count and replays
            # update indices.
            _push_reward_weights(int(stats["steps"]))
            # On every row, not only the ones that moved. A schedule in
            # config.json plus a --resume that began at a different step does
            # not reconstruct what each row was measured under. It is the
            # weight *pushed* at this update: each env adopts at its own next
            # reset, so a row is a target rather than a per-battle fact.
            stats["reward_weights"] = dict(pushed["weights"])

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
                if "eval_lift_sd" in stats:
                    print(
                        f"          eval: return {stats['eval_return']:+.4f} vs control "
                        f"{stats['control_return']:+.4f}  win {stats['eval_win']:.0%} vs "
                        f"{stats['control_win']:.0%}  lift {stats['eval_lift_sd']:+.2f} sd",
                        flush=True,
                    )
                if "ladder_elo" in stats:
                    scores = "  ".join(
                        f"{key[len('ladder_score_'):]} {value:.3f}"
                        for key, value in sorted(stats.items())
                        if key.startswith("ladder_score_"))
                    versus = (f"  ({stats['ladder_elo_vs_expert']:+.0f} vs expert)"
                              if "ladder_elo_vs_expert" in stats else "")
                    print(f"          ladder: {stats['ladder_elo']:+.0f} Elo"
                          f"{versus}  {scores}", flush=True)
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
                #
                # And what it promotes *on* is the ladder's rating where
                # there is one. eval_lift_sd is the sampled arm at forty
                # battles, where a single reading's standard error is about
                # 0.12 sd -- the noisiest number in the run -- and it is then
                # selected on by a maximum. ladder_elo is greedy, which
                # reproduces bit-identically, and it is named. The lift keeps
                # being written either way, because it is the bridge to every
                # historical number; it just stops steering.
                promoted_on = ("ladder_elo" if "ladder_elo" in stats
                               else "eval_lift_sd")
                if promoted_on in stats:
                    recent.append(float(stats[promoted_on]))
                del recent[:-_BEST_WINDOW]
                if len(recent) >= _BEST_WINDOW:
                    rolling = sum(recent) / len(recent)
                    # Never in a field called "lift". An Elo and a lift are
                    # unrelated scales, and this project has already paid
                    # three rounds of invalid comparisons for putting two
                    # scales under one name.
                    rolling_key = ("rolling_ladder_elo"
                                   if promoted_on == "ladder_elo"
                                   else "rolling_lift")
                    stats[rolling_key] = rolling
                    if rolling > best["score"]:
                        best["score"] = rolling
                        save_checkpoint(
                            {
                                "state_dict": net.state_dict(),
                                # Which head these weights are, so whatever
                                # loads them builds the network they fit.
                                "head": args.head,
                                "observation": args.observation,
                                "stats": stats,
                                rolling_key: rolling,
                                # Which statistic chose these weights. A
                                # checkpoint promoted on Elo and one promoted
                                # on lift are not comparable, and the file
                                # used to say "rolling_lift" whichever it was.
                                "promoted_on": promoted_on,
                                "window": _BEST_WINDOW,
                            },
                            out / "best.pt",
                        )
            if stats["updates"] % args.save_every == 0:
                # Optimiser state included deliberately. Adam's moment
                # estimates are most of what a long run has learned about its
                # own gradients; restarting without them throws that away and
                # the updates just after a restart look like a bad checkpoint.
                save_checkpoint(
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
            # Before the first rollout, not after it. `record` runs at the end
            # of an update, so on a --resume the opening update would collect
            # under the weights _env() was constructed with -- the schedule's
            # step-zero values -- while every row and every later rollout used
            # the resumed step's. Steps was chosen as the axis precisely so a
            # resumed run agrees with a fresh one about where it is; an
            # opening update at the wrong weight would give that away for the
            # sake of one hook.
            _push_reward_weights(
                int(resume_state.get("steps", 0)) if resume_state else 0)
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
            if args.probe == "ladder":
                from .ladder import ladder_probe

                probe_holder["probe"] = ladder_probe(
                    _env, _ladder_anchors(args.ladder_anchor, probe_env),
                    episodes=args.eval_episodes,
                    ratings=offline_ratings, mode=_LADDER_PROBE_MODE,
                    ratings_source=(str(args.ladder_ratings)
                                    if args.ladder_ratings else ""),
                )
            elif args.probe == "rotating":
                from .evaluate import rotating_probe

                probe_holder["probe"] = rotating_probe(
                    _eval_env, episodes=args.eval_episodes,
                )
            else:
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
                    # Every field _env() sets, the workers must set too. This
                    # one was missing, and VecEnvConfig defaults it to 11, so
                    # `--tower-level 5 --workers 8` trained every rollout at
                    # level 11 while config.json recorded 5 and the evaluation
                    # probe ran at 5. That is not a smaller effect than it
                    # sounds: at level 11 the towers outlast the match, 90% of
                    # battles end in a draw and crowns almost never fire, so
                    # the agent learned from shaping alone -- which
                    # docs/training.md already identified and which
                    # --tower-level was added to fix.
                    tower_level=args.tower_level,
                    reward_shaping_weight=args.shaping,
                    reward_weights=_reward_weights(args),
                    observation=observation,
                    opponent_seed=(args.seed * 1000 if args.opponent == "random" else None),
                    # The run's seed, which CRSimVecEnv makes distinct per
                    # worker. Without it a self-play opponent sampled from
                    # torch's global stream in a freshly spawned process, so
                    # --seed 0 played different battles every time it was run.
                    seed=args.seed,
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


    save_checkpoint({"state_dict": net.state_dict(), "head": args.head,
                     "observation": args.observation}, out / "final.pt")
    elapsed = time.perf_counter() - started
    print(f"\ndone in {elapsed / 60:.1f} min -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
