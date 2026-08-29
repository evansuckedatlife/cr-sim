"""Score a trained policy against a control.

Training-time return is not evidence on its own. It is measured while the
policy is still sampling from its own distribution, averaged over a sliding
window, and computed on whatever seeds the rollout happened to draw -- so it
mixes learning with exploration noise and with luck. A number that went up
during training can still be a policy that does nothing useful.

What settles it is playing the finished policy on fixed seeds against a control
that gets the identical seeds, and reporting both. The control here is a
uniform random choice over *legal* actions, which is a much stronger baseline
than it sounds: the legality mask already encodes the rules, so random play
spends every elixir it has on real placements and beats a passive opponent
about a third of the time.

**Who the two arms play matters as much as the control does.** Every number on
this project has been measured against an opponent that is either idle or
random, and the random one is now used up: the one-ply search expert beats the
random control 100-0 over 40 paired battles, +2.716 sd. A yardstick that a
policy has already run off the end of cannot show that the next policy is
better, and it fails silently while doing it -- the reading stays high and
stops moving. :func:`search_opponent` puts the expert on the other side of the
net instead, where the random control wins none of the battles and there is a
whole game's worth of headroom above it.

The lift is still defined exactly as it was -- the paired per-battle difference
against the uniform random control, in control standard deviations -- so the
*shape* of the result is unchanged and ``verdict.json`` keeps its keys. The
*scale* is not: a lift against the search opponent and a lift against the
random one are no more comparable than the idle and random scales were, which
is why nothing here writes a lift without ``eval_opponent`` beside it. See
:func:`cr_sim.train.selfplay.check_lift_is_named` for what that cost already.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from ..api.env import CRSimEnv
from ..data.cards import build_card_registry
from ..data.leveling import build_level_table
from ..data.source import LogicData
from ..engine.entity import Team
from .nets import ActorCritic, NetConfig, net_config_for
from .ppo import _unflatten_action
from .run import DEFAULT_BUILD, DEFAULT_DECK
from .scripted import SearchBot, SearchBotConfig

__all__ = [
    "evaluate", "load_policy", "check_observation", "Result",
    "search_opponent", "evaluation_seeds", "paired_lift", "evaluate_paired",
    "rotating_probe", "write_verdict",
]

#: How many distinct seed blocks a rotating evaluation cycles through.
#:
#: The in-run probe drew forty seeds once and replayed those same forty
#: battles on every reading. That is right for comparing two policies and
#: wrong for comparing a policy to itself over time: consecutive readings then
#: share all of their seed-level luck, so the rolling mean of three that
#: decides promotion averages three readings of one shared component rather
#: than a hundred and twenty battles. Eight blocks makes a window of three
#: cover a hundred and twenty distinct battles, and makes a run's nineteen
#: readings span three hundred and twenty rather than forty.
EVAL_BLOCKS = 8


class Result(dict):
    """Per-episode outcomes, summarised."""

    def summary(self, label: str) -> str:
        returns, crowns = self["returns"], self["crowns"]
        wins = sum(1 for c in crowns if c > 0) / max(1, len(crowns))
        return (
            f"{label:>10}: return {st.mean(returns):+.4f} +/- {st.pstdev(returns):.4f}  "
            f"crowns {st.mean(crowns):+.3f}  win {wins:.0%}  ({len(returns)} episodes)"
        )


def check_observation(payload: dict, env: CRSimEnv) -> None:
    """Refuse a checkpoint trained on a different observation.

    Changing the observation invalidates every checkpoint that predates the
    change: the first convolution has one filter bank per input channel, so
    nine channels of weights do not fit a thirteen-channel network. That
    failure surfaces as a size-mismatch error on ``conv.0.weight``, which says
    nothing about the actual cause -- and where the channel *count* happens to
    match while the channels mean different things, it does not fail at all
    and the policy simply plays badly.

    Checkpoints written before the field existed carry no observation and are
    assumed to be v1, which is what they are.
    """
    from ..api.encoding import parse_observation

    recorded = parse_observation(str(payload.get("observation", "v1")))
    current = env.encoding.features
    if recorded != current:
        raise ValueError(
            f"this checkpoint was trained on observation {recorded} and the "
            f"environment encodes {current}. They are different inputs; the "
            "weights do not mean the same thing. Build the environment with "
            "the matching observation, or retrain."
        )


def load_policy(checkpoint: Path, env: CRSimEnv) -> ActorCritic:
    """Rebuild the network from a checkpoint, using the env for its shapes.

    The shapes are not stored in the checkpoint because they are a property of
    the environment, not of the weights -- and taking them from the env is what
    makes a shape mismatch fail loudly here rather than silently score a policy
    against an observation it was never trained on.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    check_observation(payload, env)
    # Which head the weights were trained with is a property of the
    # checkpoint, not of the environment -- a factored head's parameters do
    # not fit a flat one, and loading them into it fails with a shape error
    # about a tensor nobody can place. Recorded when written; assumed flat
    # only for checkpoints that predate the field.
    net = ActorCritic(net_config_for(env, head=payload.get("head", "flat")))
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net


def evaluate(
    env: CRSimEnv,
    net: ActorCritic | None,
    *,
    episodes: int,
    seeds: list[int],
    greedy: bool = True,
    generator: "torch.Generator | None" = None,
) -> Result:
    """Play ``episodes`` matches. ``net=None`` plays uniformly at random.

    Both arms take the same seed list, so the comparison is over the same
    battles rather than over the same *number* of battles. With a per-episode
    spread wider than the effect being measured, paired seeds are what make a
    difference of this size readable at all.

    ``generator`` is the sampled arm's own random stream. Without it the
    sampling draws from torch's global one, which nothing here seeds, and two
    runs of the same checkpoint then land in different places.

    **How far apart is worth knowing, because it is much further than this
    docstring used to say.** It cited one pair measuring +0.583 and +0.581 and
    read a spread of 0.002 off it. Measured properly by
    ``scripts/measure_sampled_noise.py`` -- one checkpoint, four independent
    sampling streams, the same 150 battles against the same control -- the
    lifts were +0.833, +0.923, +0.964 and +0.849: sd 0.062, range 0.132, so
    two runs can differ by about 0.17 sd at 95%. The greedy arm over the same
    battles reproduced +2.1257 twice, exactly, which is what says the spread
    is the sampling and not the battles.

    So a sampled difference under roughly 0.17 sd is not a difference, and the
    0.04 sd figure quoted elsewhere in this project is about four times too
    small. Anything relying on it deserves a second look.

    Optional and off by default because the alternative -- seeding the global
    stream -- would reset the sampling of any training run that called this
    mid-flight, and the in-run probe calls it every twenty updates.
    """
    nvec = [int(v) for v in env.action_space.nvec]
    rng = np.random.default_rng(0)
    returns, crowns = [], []

    for index in range(episodes):
        observation, _ = env.reset(seed=seeds[index % len(seeds)])
        total = 0.0
        while True:
            mask = env.legal_action_mask()
            flat = mask.reshape(-1)
            if net is None:
                legal = np.flatnonzero(flat)
                choice = int(legal[rng.integers(len(legal))])
            else:
                with torch.no_grad():
                    # The actor only; an evaluation never reads the value.
                    logits = net.policy_logits(
                        torch.from_numpy(observation["grid"]).unsqueeze(0),
                        torch.from_numpy(observation["vector"]).unsqueeze(0),
                        torch.from_numpy(flat).unsqueeze(0),
                    )
                if greedy:
                    choice = int(logits.argmax(dim=-1))
                elif generator is not None:
                    # The same draw Categorical.sample makes, from a stream
                    # this evaluation owns. Masked actions carry a large
                    # negative logit rather than -inf, so the softmax is
                    # finite and multinomial never sees a degenerate row.
                    choice = int(torch.multinomial(
                        torch.softmax(logits, dim=-1), 1, generator=generator))
                else:
                    choice = int(torch.distributions.Categorical(logits=logits).sample())
            observation, reward, terminated, truncated, info = env.step(
                _unflatten_action(choice, nvec)
            )
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
        # The agent's own crown difference, not blue's. ``returns`` is
        # already team-relative -- it comes from _shaped_value(battle,
        # self.team, ...) -- and this line was not, so an env built with
        # team=RED reported the *opponent's* crowns beside the agent's own
        # return, with opposite signs and no error at all. Measured on twelve
        # seeds before the fix: mean return -0.083 against mean crowns +0.083,
        # and the two agreed in sign on 0% of the battles. Every existing
        # number is unaffected: they were all measured on team=BLUE envs,
        # where this is the same subtraction.
        mine = env.team
        theirs = mine.opponent
        crowns.append(info[f"{mine.name.lower()}_crowns"]
                      - info[f"{theirs.name.lower()}_crowns"])

    return Result(returns=returns, crowns=crowns)



# ------------------------------------------------- who the two arms play


def search_opponent(
    config: SearchBotConfig | None = None,
    *,
    team: Team = Team.RED,
    seed: int = 3,
    proposer: "Callable[[int], Any] | None" = None,
) -> Callable[..., Sequence[int]]:
    """The one-ply search expert, on the other side of the net.

    The random control is a used-up yardstick. The expert beats it 100-0 over
    40 paired battles (+2.716 sd), so a better expert -- or a better clone of
    one -- has nowhere left to register, and the metric goes on reporting a
    high number while measuring nothing. Facing the expert instead puts the
    control back at the bottom of the scale where a control belongs.

    A fresh bot per battle, reseeded from the battle's own seed. This is not
    tidiness: the bot *samples* its candidate placements, so what it plays
    depends on how far its generator has been advanced, and that is a count of
    every decision in every episode before this one. Two arms of a paired
    evaluation diverge on their first different move, so from episode two
    onward they would face experts drawing different candidates on the same
    seed -- the arms would no longer be playing the same battle, which is the
    only reason paired seeds buy anything. Rebuilt per battle, the opponent is
    a function of the seed alone.

    (The random opponent every existing number was measured against still has
    this flaw: ``_random_opponent`` is built once per environment and its
    generator carries across episodes. It is left alone deliberately -- fixing
    it would silently move the scale that every recorded verdict sits on.)
    """
    base = config or SearchBotConfig()
    state: dict[str, Any] = {"seed": object(), "bot": None}

    def policy(observation, mask, battle=None):
        key = None if battle is None else int(battle.config.seed)
        if state["bot"] is None or key != state["seed"]:
            state["seed"] = key
            derived = seed if key is None else (seed * 1_000_003 + key) % (2 ** 31 - 1)
            # The proposer is rebuilt with the bot and from the same derived
            # seed. Its stream is then a function of the battle and the
            # decision number, which is the same property the bot's own
            # candidate draw has and for the same reason.
            state["bot"] = SearchBot(
                team, replace(base, seed=derived),
                None if proposer is None else proposer(derived))
        return state["bot"](observation, mask, battle)

    # The bot branches the battle to score placements, so it needs the board
    # rather than an encoded view of it; see CRSimEnv._opponent_move.
    policy.wants_battle = True  # type: ignore[attr-defined]
    # Named, so a lift measured here cannot be written down without saying
    # what it was measured against. The scale is nothing like the random one:
    # the random control wins none of these matches, where it wins 26% against
    # a random opponent. See cr_sim.train.selfplay.check_lift_is_named.
    policy.opponent_name = "search"  # type: ignore[attr-defined]
    return policy


# --------------------------------------------------------- rotating seeds


def evaluation_seeds(
    episodes: int,
    *,
    block: int = 0,
    seed: int = 12345,
    blocks: int = EVAL_BLOCKS,
) -> list[int]:
    """The seeds for one evaluation block.

    Rotation, not randomisation. Each block is a fixed reproducible set cut
    from one master stream, so two runs still evaluate on the same battles and
    stay comparable; consecutive blocks are disjoint, so two consecutive
    readings share no battle at all.

    ``block=0`` returns exactly the list the fixed probe used to draw --
    ``default_rng(seed).integers(0, 2**31 - 1, episodes)`` -- because the
    blocks are consecutive draws from that same generator. Every lift already
    recorded on this project was measured on block 0, and keeping it identical
    is what lets the new readings be read against the old ones rather than
    merely resembling them.
    """
    if episodes <= 0:
        return []
    blocks = max(1, int(blocks))
    rng = np.random.default_rng(seed)
    chunk = rng.integers(0, 2 ** 31 - 1, episodes)
    for _ in range(int(block) % blocks):
        chunk = rng.integers(0, 2 ** 31 - 1, episodes)
    return [int(s) for s in chunk]


# --------------------------------------------------------- paired scoring


def paired_lift(result: Result, control: Result) -> dict[str, float]:
    """One arm's score against the control arm it shares seeds with.

    The difference is taken per battle and only then averaged, which is the
    whole point of handing both arms the same seeds: the per-episode spread
    here is several times larger than any effect worth seeing, and an unpaired
    difference of means would need far more episodes to say the same thing.

    Reported in control standard deviations because the raw gap means nothing
    without knowing how noisy the control is -- the same arithmetic
    scripts/measure_expert.py, scripts/clone_policy.py and
    scripts/evaluate_checkpoints.py each spell out, kept in one place so a
    fourth caller cannot spell it out slightly differently.

    That unit needs watching against a strong opponent. The control's spread
    is the denominator, and a control that loses every battle *the same way*
    has very little of it, so the same raw gap reads as a much larger lift
    than it would against a random opponent. The raw returns are reported
    beside the lift, and ``evaluate_paired`` records the spread itself, so a
    reading inflated by a collapsed denominator is visible rather than
    inferred.
    """
    control_returns = np.asarray(control["returns"], dtype=float)
    returns = np.asarray(result["returns"], dtype=float)
    crowns = np.asarray(result["crowns"], dtype=float)
    # ddof=1 is undefined on a single episode. A tiny run is a smoke test of
    # the machinery, not a measurement, so it degrades to a spread of one
    # rather than to nan -- a nan here propagates into verdict.json and reads
    # as a result.
    spread = float(control_returns.std(ddof=1)) if len(control_returns) > 1 else 1.0
    spread = spread if spread > 0.0 else 1.0
    difference = returns - control_returns
    mean = float(difference.mean()) if len(difference) else 0.0
    error = (float(difference.std(ddof=1)) / np.sqrt(len(difference))
             if len(difference) > 1 else 0.0)
    return {
        "win": float(np.mean(crowns > 0)),
        "loss": float(np.mean(crowns < 0)),
        "draw": float(np.mean(crowns == 0)),
        "lift": mean / spread,
        "ci_low": (mean - 1.96 * error) / spread,
        "ci_high": (mean + 1.96 * error) / spread,
        "return": float(returns.mean()),
        "crowns": float(crowns.mean()),
    }


def evaluate_paired(
    make_env: Callable[[], CRSimEnv],
    net: ActorCritic | None,
    *,
    episodes: int,
    seeds: Sequence[int],
    modes: Sequence[str] = ("greedy", "sampled"),
) -> dict[str, Any]:
    """Play the control and each mode of the policy over the same seeds.

    Greedy and sampled are reported separately and neither is folded into the
    other. They are not the same policy: the clone measures +1.623 greedy and
    +0.709 sampled against the same control, so a change can leave the argmax
    untouched and still move the whole distribution around it. A single
    headline number hid exactly that once already, and the run was read as a
    regression it was not.

    A fresh environment per arm. They face the same *kind* of opponent by
    construction and, for the search opponent, the same opponent battle for
    battle -- but one shared environment would also share the opponent's
    generator state between arms, which is the thing pairing exists to stop.
    """
    from .selfplay import opponent_name

    seeds = [int(s) for s in seeds]
    control_env = make_env()
    control = evaluate(control_env, None, episodes=episodes, seeds=seeds)
    # Read off the environment the control actually played in, never taken as
    # an argument: a caller cannot then label a measurement with an opponent
    # it did not face.
    faced = opponent_name(control_env)

    control_crowns = np.asarray(control["crowns"], dtype=float)
    verdict: dict[str, Any] = {
        "episodes": int(episodes),
        "eval_opponent": faced,
        "seeds": seeds,
        "control": {
            "win": float(np.mean(control_crowns > 0)),
            "loss": float(np.mean(control_crowns < 0)),
            "draw": float(np.mean(control_crowns == 0)),
            "return": float(np.mean(control["returns"])),
            "crowns": float(control_crowns.mean()),
            # The lift's unit, recorded rather than left implicit. Against a
            # strong opponent the control loses every battle in much the same
            # way, and a small denominator inflates every lift measured
            # through it -- a reader has to be able to see that happening.
            "spread": (float(np.asarray(control["returns"], dtype=float).std(ddof=1))
                       if len(control["returns"]) > 1 else 0.0),
        },
    }
    for index, mode in enumerate(modes):
        # A stream per arm, derived from the battles being played, so the same
        # checkpoint on the same block gives the same answer twice. The greedy
        # arm ignores it; the sampled arm is otherwise unreproducible, and an
        # unreproducible yardstick cannot settle whether a change moved
        # anything. Derived arithmetically, not from hash() -- string hashing
        # is salted per process, so that would be reproducible within a run
        # and not between two.
        stream = torch.Generator().manual_seed(
            ((seeds[0] if seeds else 0) + 7919 * index) % (2 ** 31 - 1))
        result = evaluate(make_env(), net, episodes=episodes, seeds=seeds,
                          greedy=(mode == "greedy"), generator=stream)
        verdict[mode] = paired_lift(result, control)

    # Flattened as well as nested, and the headline is whichever way of
    # playing scored better -- the same rule scripts/clone_policy.py uses, so
    # a verdict written here means the same thing as one written there.
    # cr_sim.train.report reads these flat keys and nothing else.
    scored = [(verdict[m]["lift"], m) for m in modes if m in verdict]
    if scored:
        best = max(scored)[1]
        verdict.update({k: verdict[best][k]
                        for k in ("lift", "ci_low", "ci_high", "win", "loss")})
        verdict["mode"] = best
    return verdict


def write_verdict(path: Path, verdict: dict[str, Any]) -> dict[str, Any]:
    """Write ``verdict.json``, refusing one that does not name its opponent.

    The same guard :func:`cr_sim.train.selfplay.check_lift_is_named` puts on a
    metrics row, for the file that outlives the run. A verdict is a bare lift
    on disk; read later beside one measured against a different opponent it is
    worse than no number, and this project has already spent two rounds of
    comparisons finding that out.
    """
    if not verdict.get("eval_opponent"):
        raise ValueError(
            "a verdict carries a lift but no eval_opponent. A lift is "
            "meaningless without the opponent it was measured against -- the "
            "same policy scores wildly differently against an idle, a random "
            "and a searching one. See cr_sim.train.selfplay.opponent_name."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return verdict


# ------------------------------------------------------- the in-run probe


def rotating_probe(
    make_env: Callable[[], CRSimEnv],
    episodes: int = 40,
    seed: int = 12345,
    *,
    blocks: int = EVAL_BLOCKS,
    greedy: bool = False,
) -> Callable[[ActorCritic], dict[str, Any]]:
    """A drop-in for :func:`cr_sim.train.selfplay.evaluation_probe` that rotates seeds.

    The probe it replaces draws its seeds once and replays those same battles
    on every reading. Consecutive readings therefore share all of their
    seed-level luck, and cr_sim.train.run promotes on the mean of the last
    three -- which averages three readings of one shared component instead of
    a hundred and twenty battles. That is the failure the rolling mean was
    introduced to fix, one level down: the window stops a single lucky
    *reading* being selected for and does nothing about a lucky *seed set*
    being selected for, because every reading in the window contains it.

    Here reading *n* plays block ``n % blocks``, so a window of three spans
    three disjoint sets of battles and the shared component is gone.

    The control arm is per block and cached, because a lift is only paired
    against a control that played the same seeds. That costs ``blocks``
    control arms over a run instead of one -- 320 battles rather than 40 at
    the defaults -- paid once each, in the first ``blocks`` evaluations, and
    free afterwards.

    Emits every key :func:`evaluation_probe` emits, with the same meanings, so
    the runner's printout and its promotion rule keep working untouched.
    ``eval_lift_sd`` stays the *sampled* arm: that is what the old probe
    measured and what a policy still being trained actually plays, and
    changing which arm the promotion reads would move the scale without
    changing the name. ``greedy=True`` adds a second arm under
    ``eval_lift_sd_greedy``, at double the evaluation cost, for runs that want
    to watch the argmax move separately from the distribution around it.

    **To adopt it, cr_sim/train/run.py needs two mechanical one-line edits.**
    That file belongs to another author, so they are described rather than
    made:

    1. In the ``from .selfplay import (...)`` block near the top, drop
       ``evaluation_probe`` from the list and add one line beside the other
       relative imports::

           from .evaluate import rotating_probe

    2. At the single call site -- the ``probe_holder["probe"] = ...``
       assignment just below the ``# See _eval_env`` comment -- change the
       function name and nothing else::

           probe_holder["probe"] = rotating_probe(
               _eval_env, episodes=args.eval_episodes,
           )

    ``_BEST_WINDOW``, the promotion block and the printed ``eval:`` line all
    read the same keys they read now.
    """
    from .selfplay import opponent_name, reward_name

    cache: dict[int, tuple[Result, str, str]] = {}
    readings = {"n": 0}

    def probe(net: ActorCritic) -> dict[str, Any]:
        block = readings["n"] % max(1, blocks)
        readings["n"] += 1
        block_seeds = evaluation_seeds(episodes, block=block, seed=seed,
                                       blocks=blocks)
        if block not in cache:
            control_env = make_env()
            cache[block] = (
                evaluate(control_env, None, episodes=episodes, seeds=block_seeds),
                opponent_name(control_env),
                # The scale, from the environment the control actually played
                # in. Cached with the control because the control's returns
                # and the spread underneath them are denominated in it.
                reward_name(control_env),
            )
        control, faced, scale = cache[block]
        control_return = float(np.mean(control["returns"]))
        spread = float(np.std(control["returns"])) or 1.0

        # Seeded from the block, so the sampling stream is a property of the
        # battles being played rather than of how many readings came before.
        # Two readings of the same block then differ only by the policy, which
        # is the only thing a probe is trying to show.
        stream = torch.Generator().manual_seed((seed * 8191 + block) % (2 ** 31 - 1))
        sampled = evaluate(make_env(), net, episodes=episodes,
                           seeds=block_seeds, greedy=False, generator=stream)
        stats = {
            "eval_return": float(np.mean(sampled["returns"])),
            "eval_win": float(np.mean([c > 0 for c in sampled["crowns"]])),
            "control_return": control_return,
            "control_win": float(np.mean([c > 0 for c in control["crowns"]])),
            # In control standard deviations, because the raw gap means
            # nothing without knowing how noisy the control is.
            "eval_lift_sd": (float(np.mean(sampled["returns"])) - control_return) / spread,
            "eval_opponent": faced,
            "eval_reward": scale,
            "eval_episodes": int(episodes),
            # Which battles this reading was played on. Without it a series of
            # rotating readings is indistinguishable from a noisy fixed one,
            # and a promotion cannot be traced back to the seeds that made it.
            "eval_block": int(block),
            "eval_blocks": int(max(1, blocks)),
        }
        if greedy:
            argmax = evaluate(make_env(), net, episodes=episodes,
                              seeds=block_seeds, greedy=True)
            stats["eval_return_greedy"] = float(np.mean(argmax["returns"]))
            stats["eval_win_greedy"] = float(np.mean([c > 0 for c in argmax["crowns"]]))
            stats["eval_lift_sd_greedy"] = (
                (float(np.mean(argmax["returns"])) - control_return) / spread)
        return stats

    return probe


# -------------------------------------------------------------------- CLI


def _opponent_for(name: str, config: SearchBotConfig | None = None):
    """Build the opponent one evaluation environment faces.

    Built per environment rather than shared: each arm of a paired evaluation
    holds its own, and a single instance would have its generator advanced by
    whichever arm played first.
    """
    if name == "random":
        from .run import _random_opponent

        return _random_opponent(90_000)
    if name == "search":
        return search_opponent(config)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cr-sim-eval")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--tps", type=int, default=20)
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--match-seconds", type=int, default=120)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--tower-level", type=int, default=11)
    parser.add_argument(
        "--opponent", choices=("idle", "random", "search"), default="idle",
        help="who both arms play. 'idle' never plays a card, 'random' spends "
             "its elixir on legal placements -- the two scales every number on "
             "this project sits on, and they are not the same scale. 'search' "
             "is the one-ply expert, which the random control loses to 100-0; "
             "it is the only one of the three with headroom left in it. The "
             "default is unchanged so an old command still means what it meant.")
    parser.add_argument("--candidates", type=int, default=18,
                        help="placements the search opponent evaluates per move")
    parser.add_argument("--horizon-seconds", type=float, default=15.0,
                        help="how far the search opponent plays each branch")
    parser.add_argument(
        "--block", type=int, default=0,
        help="which rotating seed block to play. Block 0 is the seed list "
             "every existing measurement used, so it is the one to quote "
             "against them; another block is a disjoint set of battles, and "
             "is how to check a result was not one seed set's luck.")
    parser.add_argument("--verdict", type=Path, default=None,
                        help="write the result as a verdict.json here")
    parser.add_argument(
        "--sample", action="store_true",
        help="headline the sampled arm rather than the better one. Both are "
             "always reported: the clone is +1.623 greedy and +0.709 sampled, "
             "so a change can leave the argmax untouched and still move the "
             "distribution around it, and one number hides that.")
    parser.add_argument(
        "--observation", default=None,
        help="which observation to build the environment with. Defaults to "
             "whatever the checkpoint says it was trained on, which is the "
             "only choice that can be right.")
    args = parser.parse_args(argv)

    data = LogicData.load(args.build)
    levels = build_level_table(data)
    registry = build_card_registry(data)

    from ..api.encoding import parse_observation

    recorded = torch.load(args.checkpoint, map_location="cpu",
                          weights_only=False).get("observation", "v1")
    observation = parse_observation(args.observation or str(recorded))
    search = SearchBotConfig(horizon_seconds=args.horizon_seconds,
                             candidates=args.candidates)

    def make_env() -> CRSimEnv:
        return CRSimEnv(
            data, levels, registry, DEFAULT_DECK, DEFAULT_DECK,
            ticks_per_second=args.tps,
            frame_skip=args.frame_skip,
            max_ticks=args.tps * args.match_seconds,
            tower_level=args.tower_level,
            observation=observation,
            opponent_policy=_opponent_for(args.opponent, search),
        )

    seeds = evaluation_seeds(args.episodes, block=args.block)
    net = load_policy(args.checkpoint, make_env())
    verdict = evaluate_paired(make_env, net, episodes=args.episodes, seeds=seeds)
    verdict["block"] = int(args.block)
    if args.sample and "sampled" in verdict:
        verdict.update({k: verdict["sampled"][k]
                        for k in ("lift", "ci_low", "ci_high", "win", "loss")})
        verdict["mode"] = "sampled"

    print(f"{args.episodes} paired battles against the "
          f"{verdict['eval_opponent']} opponent, seed block {args.block}\n")
    print(f"{'arm':<20}{'win':>8}{'loss':>8}{'draw':>8}{'lift sd':>10}"
          f"{'95% CI':>22}")
    control = verdict["control"]
    print(f"{'random control':<20}{control['win']:>8.0%}{control['loss']:>8.0%}"
          f"{control['draw']:>8.0%}{'--':>10}{'--':>22}")
    for mode in ("greedy", "sampled"):
        arm = verdict.get(mode)
        if arm is None:
            continue
        print(f"{'trained, ' + mode:<20}{arm['win']:>8.0%}{arm['loss']:>8.0%}"
              f"{arm['draw']:>8.0%}{arm['lift']:>+10.3f}"
              f"   [{arm['ci_low']:+.3f}, {arm['ci_high']:+.3f}]")
    # The denominator, printed rather than left implicit: a control that loses
    # every battle in much the same way has little spread, and every lift
    # divided by it is inflated in proportion.
    print(f"\ncontrol return {control['return']:+.4f} +/- {control['spread']:.4f}"
          f"  <- the unit the lift is measured in")
    print(f"headline: {verdict['lift']:+.3f} sd ({verdict['mode']}) against "
          f"the {verdict['eval_opponent']} opponent")

    if args.verdict is not None:
        write_verdict(args.verdict, verdict)
        print(f"-> {args.verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
