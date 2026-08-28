"""Score saved policies on decks they were never trained on.

The point of :class:`cr_sim.train.nets.FactoredStatsHead` is a capability, not
a better number on the training deck: a head conditioned on a card's own
statistics has a conditioning vector for a card it has never seen, and a head
conditioned on a lookup table does not. This script is how that claim gets
measured rather than asserted.

**What "never seen" means here, exactly.** The observation's card vocabulary is
the episode's *deck union* -- ``sorted(set(blue) | set(red))`` -- so a mirror
deck of eight cards always produces a vocabulary of eight and an observation of
the same width whatever the cards are. Every checkpoint therefore loads against
any 8-card mirror deck with no missing and no unexpected keys. Nothing errors;
column ``i`` of the lookup head's table simply now means whatever card sorted
into position ``i``. The lookup arm is not merely uninformed about the new
deck, it is *misinformed* -- it applies the Knight's learned placement vector
to whichever card took the Knight's position -- and that is the failure being
measured.

Non-mirror decks are refused by construction. They change the vocabulary size,
which changes the observation width, which no existing checkpoint can load;
allowing them here would turn a capability test into a shape error.

**The caveat, stated in the script and not only in the write-up.** The head is
not the only thing that reads a card's identity. The trunk's vector MLP takes
the same one-hots: of ``vector.0.weight``'s 102 input columns, 80 are card
identity bits -- both sides, four hand slots and the next card each, eight
wide -- of which 32 are the acting hand the head also reads. Nothing here
re-keys those. Swapping the deck therefore leaves the trunk interpreting every
one of them as a different card, and it does so identically for both arms.

Measured on the two trained checkpoints, holding the observation fixed and
changing only the deck: the lookup head's conditioning moves by 0.000 -- the
same weights indexed by the same bit, so a completely different deck produces a
byte-identical conditioning vector -- while the encoder's moves by 1.40 of its
own norm on average. What this script measures is the whole agent, of which the
head is one part, so a null here is evidence about the agent and not about the
head, and any lift is a lower bound on what the encoder is worth once the trunk
stops keying on vocabulary position too.

    python scripts/evaluate_decks.py runs/a/cloned.pt runs/b/cloned.pt \
        --decks 10 --episodes 150 --out runs/x/decks.json
"""

from __future__ import annotations

import argparse
import json
import math
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
from cr_sim.train.evaluate import evaluate, paired_lift
from cr_sim.train.nets import ActorCritic, net_config_for
from cr_sim.train.run import DEFAULT_BUILD, DEFAULT_DECK, _random_opponent
from cr_sim.train.selfplay import opponent_name

#: Cards in a deck. Eight is the game's own deck size and also what holds the
#: observation width at the trained one; see the module docstring.
DECK_SIZE = 8


def sample_decks(pool, count, seed, *, exclude=(), size=DECK_SIZE, accept=None):
    """``count`` distinct mirror decks drawn from ``pool``, minus ``exclude``.

    ``accept``, when given, is a predicate a deck must satisfy to be kept. It
    exists for one specific confound. ``DEFAULT_DECK`` averages 2.50 elixir --
    chosen that way on purpose, see its docstring -- while eight cards drawn
    uniformly from the 114 outside it average 4.09. A policy scoring nothing on
    those decks has met two changes at once, unfamiliar cards and an elixir
    economy 64% more expensive than anything it ever played, and the result
    cannot separate them. A cost-matched draw holds the second fixed.

    Deterministic in ``seed``, because both arms have to be handed the *same*
    decks: a comparison in which each policy drew its own opponents is not a
    comparison. Every returned deck is ``size`` distinct cards, none of them in
    ``exclude``, so "unseen" is a property of the returned value rather than of
    the intent behind the call.

    Asking for more decks than the pool can distinctly make is refused up
    front, by counting them. The rejection loop below would otherwise look
    forever for a deck that does not exist, and for a script whose whole job is
    to be left running for an hour that is the worst available failure: an
    infinite loop is indistinguishable from slow progress.

    The loop carries its own attempt budget as well, which is what stops an
    ``accept`` too strict for the pool from spinning forever. The count below
    cannot see ``accept`` -- it counts decks, not acceptable ones -- so with a
    predicate in play the budget is the only guard, and it names how far it
    got rather than reporting an empty result as success.
    """
    banned = {str(name) for name in exclude}
    candidates = sorted({str(name) for name in pool} - banned)
    if len(candidates) < size:
        raise SystemExit(
            f"{len(candidates)} cards available outside the excluded set, "
            f"which cannot fill a deck of {size}.")
    possible = math.comb(len(candidates), size)
    if count > possible:
        raise SystemExit(
            f"{count} distinct decks asked for, but {len(candidates)} cards "
            f"make only {possible} decks of {size}.")
    rng = np.random.default_rng(seed)
    decks: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    # Rejection rather than one fixed draw: two identical decks in the list
    # would be reported as two independent measurements of the same thing.
    budget = 1000 + 100 * count
    for _ in range(budget):
        if len(decks) == count:
            return tuple(decks)
        # str(), not numpy's str_: these end up in a JSON file and in an
        # environment's deck, and a numpy scalar there is a surprise waiting.
        pick = tuple(sorted(
            str(name) for name in rng.choice(candidates, size=size, replace=False)))
        if pick in seen:
            continue
        # Recorded before the predicate, so a deck this predicate rejects is
        # not drawn and rejected again for the rest of the budget.
        seen.add(pick)
        if accept is not None and not accept(pick):
            continue
        decks.append(pick)
    raise SystemExit(
        f"gave up after {budget} draws with {len(decks)} of {count} decks "
        f"filled. Either the filter is too strict for this pool, or the "
        f"count above should have refused this first.")


def cost_matched(costs, window, reference=DEFAULT_DECK):
    """A predicate keeping decks whose mean elixir is within ``window``.

    Returned as a closure over the cost table rather than read from a global,
    so a test can state the costs it means instead of depending on what the
    card data happens to say today.
    """
    target = float(np.mean([costs[name] for name in reference]))

    def accept(deck):
        return abs(float(np.mean([costs[name] for name in deck])) - target) <= window

    accept.target = target
    return accept


def decks_for(pool, count, seed, *, only=None, include_training=False,
              accept=None):
    """The labelled decks a run plays, as ``[(label, deck), ...]``.

    ``only`` slices the work across processes so a sweep that would take three
    hours in one takes one hour in three. It is applied *after* the full
    ``count`` is drawn, which is the whole point: drawing ``len(only)`` decks
    instead would give each process a different deck list, and the merged
    result would be a set of decks that no single run would ever have played
    -- reported as though it were one experiment. Same shape as the shard
    merge in ``scripts/clone_policy.py``, and the same silent failure.
    """
    decks = sample_decks(pool, count, seed, exclude=DEFAULT_DECK, accept=accept)
    labelled = [(f"unseen-{index:02d}", deck)
                for index, deck in enumerate(decks)]
    if only is not None:
        for index in only:
            if not 0 <= index < len(labelled):
                raise SystemExit(
                    f"--only {index} is outside the {len(labelled)} decks "
                    f"--decks {count} draws.")
        labelled = [labelled[index] for index in only]
    if include_training:
        # In its declared order, not sorted. A deck is a *cycle*: the order
        # the eight cards are listed in sets the starting hand and what
        # follows it, so the same eight cards in a different order are
        # different battles. Measured on the same 40 seeds at tower level 5,
        # the random control takes 20% of them with DEFAULT_DECK as written
        # and 28% with the same cards sorted. The vocabulary is sorted either
        # way, so nothing about the observation's layout moves and there is
        # no shape to catch it.
        #
        # This row is the anchor every unseen deck is read against, and it is
        # only an anchor if it is the deck the policies were trained on and
        # the deck the recorded baselines were measured on.
        labelled = [("training-deck", tuple(DEFAULT_DECK))] + labelled
    return labelled


def load_policy(path, env):
    """A checkpoint's network, built for the environment it will be *played* in.

    Not for the one it was trained in, and that distinction is the experiment.
    ``net_config_for`` reads the card-stat table off ``env``'s vocabulary, so a
    ``"factored-stats"`` checkpoint played on a new deck gets that deck's
    statistics and conditions on them correctly. Build the config from the
    training deck instead and the stats head silently describes eight cards
    that are not on the board -- which makes both arms wrong in the same way
    and produces a clean, meaningless null.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    net = ActorCritic(net_config_for(env, head=payload.get("head", "flat")))
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net, payload


class Bench:
    """Everything the environments share, loaded once."""

    def __init__(self, build=DEFAULT_BUILD, tower_level=5, observation="v1"):
        self.data = LogicData.load(build)
        self.levels = build_level_table(self.data)
        self.registry = build_card_registry(self.data)
        self.tower_level = tower_level
        self.observation = observation

    def make_env(self, deck):
        """A mirror battle on ``deck``.

        Both sides play it, which is what keeps the vocabulary at eight and the
        observation at the width every checkpoint on disk was trained for.
        """
        deck = tuple(deck)
        if len(set(deck)) != len(deck):
            raise SystemExit(f"deck has a repeated card: {deck}")
        return CRSimEnv(
            self.data, self.levels, self.registry, deck, deck,
            ticks_per_second=20, frame_skip=30, max_ticks=20 * 120,
            tower_level=self.tower_level,
            observation=parse_observation(self.observation),
            opponent_policy=_random_opponent(60_000))


def build_parser():
    parser = argparse.ArgumentParser(prog="evaluate-decks")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--decks", type=int, default=10,
                        help="how many unseen mirror decks to draw")
    parser.add_argument("--deck-seed", type=int, default=20260828)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--tower-level", type=int, default=5)
    parser.add_argument("--seed", type=int, default=777,
                        help="battle seeds, shared by every arm and every deck")
    parser.add_argument("--observation", default="v1")
    parser.add_argument("--also-training-deck", action="store_true",
                        help="score the training deck too, as the anchor the "
                             "unseen decks are read against")
    parser.add_argument("--cost-window", type=float, default=None,
                        help="keep only decks whose mean elixir is within this "
                             "of the training deck's 2.50. Unfiltered, a "
                             "uniform draw averages 4.09, so an unseen-deck "
                             "result confounds unfamiliar cards with a much "
                             "more expensive economy.")
    parser.add_argument("--only", type=int, nargs="*", default=None,
                        help="indices into the deck list, to split one sweep "
                             "across processes. The full --decks count is "
                             "drawn first and then sliced, so which decks get "
                             "played does not depend on how the work was "
                             "divided.")
    parser.add_argument("--out", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    bench = Bench(tower_level=args.tower_level, observation=args.observation)
    pool = [c.name for c in bench.registry.standard()]
    accept = None
    if args.cost_window is not None:
        costs = {c.name: c.mana_cost for c in bench.registry.standard()}
        accept = cost_matched(costs, args.cost_window)
        print(f"cost-matched draw: mean elixir within {args.cost_window} of "
              f"the training deck's {accept.target:.2f}", flush=True)
    labelled = decks_for(pool, args.decks, args.deck_seed, only=args.only,
                         include_training=args.also_training_deck,
                         accept=accept)
    labels = [label for label, _ in labelled]
    decks = [deck for _, deck in labelled]

    seeds = [int(s) for s in
             np.random.default_rng(args.seed).integers(0, 2**31 - 1, args.episodes)]
    started = time.perf_counter()
    rows: list[dict] = []
    print(f"{len(decks)} decks x {len(args.checkpoints)} checkpoints x "
          f"{args.episodes} battles, tower level {args.tower_level}", flush=True)

    for label, deck in zip(labels, decks):
        # One control per deck, played once and differenced against by every
        # checkpoint. Two controls built the same way come out identical here,
        # but "identical because nothing varies" is a property of today's code
        # and this has to stay a property of the measurement.
        control_env = bench.make_env(deck)
        control = evaluate(control_env, None, episodes=args.episodes, seeds=seeds)
        control_returns = np.asarray(control["returns"], dtype=float)
        control_crowns = np.asarray(control["crowns"])
        faced = opponent_name(control_env)
        print(f"\n{label}: {', '.join(deck)}", flush=True)
        print(f"  {'arm':<34}{'win':>7}{'loss':>7}{'draw':>7}{'lift sd':>10}"
              f"{'95% CI':>22}", flush=True)
        print(f"  {'random control':<34}{np.mean(control_crowns > 0):>7.0%}"
              f"{np.mean(control_crowns < 0):>7.0%}"
              f"{np.mean(control_crowns == 0):>7.0%}{'--':>10}{'--':>22}",
              flush=True)
        for path in args.checkpoints:
            env = bench.make_env(deck)
            env.reset(seed=0)
            net, payload = load_policy(path, env)
            name = Path(path).parent.name
            for index, mode in enumerate(("greedy", "sampled")):
                # A stream per arm, the same arithmetic evaluate_paired uses,
                # so the sampled arm is reproducible between processes.
                stream = torch.Generator().manual_seed(
                    (seeds[0] + 7919 * index) % (2 ** 31 - 1))
                result = evaluate(bench.make_env(deck), net,
                                  episodes=args.episodes, seeds=seeds,
                                  greedy=(mode == "greedy"), generator=stream)
                stats = paired_lift(result, control)
                print(f"  {name + ', ' + mode:<34}{stats['win']:>7.0%}"
                      f"{stats['loss']:>7.0%}{stats['draw']:>7.0%}"
                      f"{stats['lift']:>+10.3f}"
                      f"   [{stats['ci_low']:+.3f}, {stats['ci_high']:+.3f}]",
                      flush=True)
                rows.append({
                    "deck": list(deck), "deck_label": label,
                    "checkpoint": str(path), "name": name,
                    "head": payload.get("head", "flat"), "mode": mode,
                    "observation": str(payload.get("observation", "v1")),
                    "episodes": args.episodes,
                    "eval_opponent": faced,
                    "control_win": float(np.mean(control_crowns > 0)),
                    "control_spread": float(control_returns.std(ddof=1)),
                    # Per-battle differences, kept so decks can be pooled
                    # afterwards without replaying anything.
                    "differences": [
                        float(d) for d in
                        np.asarray(result["returns"], dtype=float) - control_returns],
                    **{k: float(v) for k, v in stats.items()},
                })
            if args.out:
                # Written after every checkpoint, not once at the end. A
                # multi-hour evaluation that dies on deck nine should not take
                # decks one through eight with it.
                out = Path(args.out)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n{(time.perf_counter() - started) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
