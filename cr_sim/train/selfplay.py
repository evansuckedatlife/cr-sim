"""Self-play, and an honest progress signal to watch it with.

Two things a long run needs that the trainer did not have.

**An opponent worth beating.** Training against random play teaches a policy to
beat random play, and that is a low ceiling: the measured gap between a random
agent and a trained one has been inside noise on every run so far, partly
because there was nothing to learn *from*. The opponent here is a frozen copy
of the policy itself, refreshed every so often, so the thing it is trying to
beat improves as it does.

Frozen rather than live. Both sides sharing one set of weights would make the
opponent change underneath the rollout that is scoring it, and the advantage
estimates would be measuring a moving target. A snapshot is a fixed opponent
for as long as it is in place, which is what the algorithm assumes.

**A number that means what it says.** The trainer's own return is measured
while the policy is still exploring, averaged over a sliding window, on
whatever seeds the rollout happened to draw. Measured against a paired-seed
control it has run about eighteen points optimistic -- reporting 55% for a
policy that evaluated at 37% -- and the cause is still unexplained. So a long
run should not be steered by it. :func:`evaluation_probe` plays the current
policy against a random control on fixed seeds, which is the same measurement
the final evaluation makes, and is the one worth watching.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Sequence

import numpy as np

from ..api.env import CRSimEnv
from .nets import ActorCritic

__all__ = [
    "FrozenOpponent", "OpponentPool", "PooledOpponent",
    "evaluation_probe", "ancestor_probe", "opponent_name", "reward_name",
    "check_lift_is_named", "SCORED_FAMILIES",
]


#: Statistic families that report a *score* rather than a lift, each with its
#: own opponent fields.
#:
#: A score is not a lift, so :func:`check_lift_is_named`'s original clause
#: never fired on one -- the accidental exemption ``ancestor_probe`` enjoyed
#: for its whole life, emitting a win rate against "the oldest one" and
#: recording which ancestor that was as an integer age against a pool that
#: evicts from the middle.
SCORED_FAMILIES = ("ladder_", "ancestor_")


def check_lift_is_named(stats: dict) -> dict:
    """Refuse to record a lift that does not say who it was measured against.

    A comment is not a guardrail, and this class of error has already cost
    this project two rounds of invalid comparisons: the in-run probe faced an
    opponent that never plays a card while the large paired verdicts faced a
    random agent, both were called "lift", and they were compared to each
    other. The control wins 92% of the idle matches and 26% of the random
    ones, so the two numbers never lived on the same scale.

    Every writer of a metrics row goes through here, so a lift without its
    opponent cannot reach the file at all. Rows written before the field
    existed have no ``eval_opponent`` and are the idle ones.

    **And what was counted, not only who was played.** A lift is a difference
    of *returns*, and a return is denominated in whatever reward the
    environment was scoring, so the same two policies produce a different lift
    under ``projected`` than under ``five-term`` -- and a run that anneals its
    shaping produces both within itself, its own arm shrinking against a
    control that was evaluated once and cached with the scale it had then.
    That is the identical mistake one axis over, so it gets the identical
    refusal: ``eval_reward``, written by :func:`reward_name` off the
    environment that actually played. Pinning the in-run probe's reward
    (``cr_sim.train.run.EVAL_REWARD``) is what removes the drift; this is what
    makes the omission unrecordable.

    **A ladder row needs a second door, because it would walk past the first
    one.** A ladder reports a *score* -- the fraction of a mirrored pairing a
    player took -- not a lift, so ``eval_lift_sd`` is absent and the clause
    above never fires. That is exactly the accidental exemption
    :func:`ancestor_probe` used to enjoy: it emitted ``ancestor_win`` and no
    ``eval_lift_sd``, so nothing checked it, and it recorded which ancestor it
    faced as an integer age against a pool that evicts. The existing self-play
    ladder was already an unnamed measurement; the new one does not get to
    copy it.

    Two things are demanded of a ladder row and they are different questions.
    *Who* is the kind of opponent -- "pool", "ladder", "random". *Which* is
    the weights: "pool" is not an opponent, ``runs/clone-v3-paired/cloned.pt``
    is. A rating is transitive, so a row naming only the kind cannot be placed
    on the graph at all.

    **Each family of scores names its own side, and that is not a loophole.**
    One metrics row genuinely carries several measurements against several
    different opponents: cr_sim.train.run merges the random-control probe,
    the self-play ancestor ladder and the rating ladder into one dict. A
    single ``eval_opponent`` cannot name all of them, and whichever writer
    happened to run last would relabel the others -- measured on a smoke run,
    where the ancestor's score arrived on a row saying it had been played
    against the rating ladder's anchors. So ``ladder_score`` looks for
    ``ladder_opponent`` and ``ancestor_score`` for ``ancestor_opponent``,
    each falling back to ``eval_opponent`` only where the row carries a single
    measurement and that field is unambiguous. A row carrying neither is
    refused exactly as before.
    """
    if "eval_lift_sd" in stats and not stats.get("eval_opponent"):
        raise ValueError(
            "a metrics row carries eval_lift_sd but no eval_opponent. A lift "
            "is meaningless without the opponent it was measured against -- "
            "the same policy scores wildly differently against an idle and a "
            "random one. See cr_sim.train.selfplay.opponent_name."
        )
    if "eval_lift_sd" in stats and not stats.get("eval_reward"):
        raise ValueError(
            "a metrics row carries eval_lift_sd but no eval_reward. A lift is "
            "a difference of *returns*, and a return is denominated in "
            "whatever reward the environment was scoring -- so the same two "
            "policies produce a different lift under `projected` than under "
            "`five-term`, and a run that anneals its shaping shrinks its own "
            "arm's returns while a cached control keeps the scale it was "
            "measured on. This is the identical mistake eval_opponent already "
            "refuses, one axis over: naming who was played and not what was "
            "counted. See cr_sim.train.selfplay.reward_name."
        )
    for prefix in SCORED_FAMILIES:
        if f"{prefix}score" not in stats and f"{prefix}elo" not in stats:
            continue
        if not (stats.get(f"{prefix}opponent") or stats.get("eval_opponent")):
            raise ValueError(
                f"a metrics row carries {prefix}score but no "
                f"{prefix}opponent and no eval_opponent. A score against an "
                "unnamed player is not a rating -- the whole point of a "
                "ladder is that the pairing graph is connected, and an edge "
                "with one end missing connects nothing. See "
                "cr_sim.train.selfplay.opponent_name."
            )
        if not stats.get(f"{prefix}opponent_ref"):
            raise ValueError(
                f"a {prefix.rstrip('_')} row names its opponent's kind but "
                "not which weights they were. 'pool' is not an opponent; "
                "'runs/clone-v3-paired/cloned.pt' is."
            )
    return stats


def reward_name(env: CRSimEnv) -> str:
    """The scale ``env`` measures returns on, as a label to record beside a lift.

    Read off the environment rather than taken as an argument, for exactly the
    reason :func:`opponent_name` is: a caller cannot then label a measurement
    with a scale it was not taken on.

    The full weight tuple, not the variant name. "projected" alone does not
    identify a scale -- the same variant at ``tower=1.0`` and at ``tower=0.0``
    produces returns an order of magnitude apart, and under an anneal it
    produces both within one run. The variant name is what
    ``Demonstrations.reward`` recorded, and the set collected under
    ``--elixir-weight 0`` is indistinguishable in that file from one collected
    at 0.3.
    """
    weights = getattr(env, "reward_weights", None)
    if weights is None:
        return f"simple:shaping={getattr(env, 'reward_shaping_weight', 0.0):g}"
    kind = ("projected" if type(weights).__name__ == "ProjectionWeights"
            else "five-term")
    terms = ",".join(f"{k}={v!r}" if v is None else f"{k}={v:g}"
                     for k, v in sorted(weights.as_dict().items()))
    return f"{kind}:{terms}"


def opponent_name(env: CRSimEnv) -> str:
    """What ``env``'s opponent is, as a short label to record beside a lift.

    Read off the environment rather than taken as an argument, so a caller
    cannot label a measurement with an opponent it did not actually face.
    Policies carry their own ``opponent_name``; anything that does not is
    reported as ``"unknown"`` rather than guessed at, because a wrong label is
    worse than an absent one.

    This exists because "lift" was reported on two incompatible scales for
    most of this project's life: the in-run probe faced an opponent that never
    plays a card, the large paired verdicts faced a random agent, and the two
    numbers were compared to each other. The random control wins 92% of the
    idle matches and 26% of the random ones, so a lift means nothing at all
    without this string next to it.
    """
    from ..api.env import idle_opponent_policy

    policy = getattr(env, "opponent_policy", None)
    if policy is None or policy is idle_opponent_policy:
        return "idle"
    return str(getattr(policy, "opponent_name", None) or "unknown")


class FrozenOpponent:
    """A snapshot of the policy, playing the other side.

    Holds its own copy of the weights rather than a reference to the live
    network, so refreshing is an explicit act and the opponent cannot drift
    mid-rollout.
    """

    __slots__ = ("_net", "_nvec", "_torch", "_rng", "refreshes", "temperature",
                 "opponent_name", "greedy", "generator")

    def __init__(self, net: ActorCritic, nvec: Sequence[int], seed: int = 0,
                 temperature: float = 1.0, *, name: str = "unknown",
                 greedy: bool = False,
                 generator: "Any | None" = None) -> None:
        import torch

        self._torch = torch
        self._nvec = [int(v) for v in nvec]
        self._rng = np.random.default_rng(seed)
        self.refreshes = 0
        #: What a measurement records as the opponent it faced.
        #:
        #: Without this slot the class had no ``__dict__`` either, so
        #: ``opponent.opponent_name = "..."`` raised AttributeError and
        #: :func:`opponent_name` reported every frozen opponent as "unknown".
        #: A checkpoint-vs-checkpoint ladder could not name its own opponent,
        #: which means no row it wrote could pass
        #: :func:`check_lift_is_named`. It is a string rather than a
        #: reference to the weights because a name is what a metrics row can
        #: hold; the weights themselves are named by
        #: ``ladder_opponent_ref``.
        self.opponent_name = str(name)
        #: Play the argmax, exactly, rather than sampling.
        #:
        #: A real flag and not ``temperature=1e-3``. The entire argument for
        #: a greedy ladder is that it reproduces bit-identically, and 1e-3 is
        #: *nearly* argmax: two logits within 1e-3 of each other still get
        #: sampled between, so a run can differ from its own repeat on the
        #: battles where the policy is undecided -- which are exactly the
        #: battles a close pairing turns on. "Nearly reproducible" is how a
        #: reproducibility claim quietly becomes false.
        self.greedy = bool(greedy)
        #: The sampled draw's own stream, when it has one.
        #:
        #: ``None`` keeps the original ``Categorical(...).sample()`` call on
        #: torch's global stream, which is what every self-play run so far
        #: has used and what its numbers reproduce from. A ladder cannot
        #: afford that -- an unreproducible yardstick cannot settle whether a
        #: change moved anything -- so it passes a generator derived
        #: arithmetically from the battles being played.
        self.generator = generator
        #: How sharply the opponent plays its own policy. 1.0 samples from it
        #: as-is, which sounds neutral and is not: an early policy has an
        #: entropy near the uniform maximum, so an opponent sampling from it
        #: is *still nearly random*. That leaves the match outcome as
        #: unpredictable as it was against a random agent, the critic with
        #: nothing it can fit, and the advantages as noise -- which is the
        #: thing self-play was supposed to fix. Below 1.0 sharpens toward the
        #: policy's own preferences and makes the environment learnable.
        self.temperature = max(1e-3, float(temperature))
        self._net = self._snapshot(net)

    def _snapshot(self, net: ActorCritic) -> ActorCritic:
        # On the CPU whatever device the learner is on. An opponent does
        # batch-of-one inference, which an accelerator is bad at anyway, and
        # it is fed observations built on the CPU -- mixing the two dispatches
        # convolution to a kernel the XPU backend does not implement, which
        # surfaces as "aten::_slow_conv2d_forward is not implemented" rather
        # than as anything mentioning the opponent.
        clone = copy.deepcopy(net).to("cpu")
        clone.eval()
        for parameter in clone.parameters():
            parameter.requires_grad_(False)
        return clone

    def refresh(self, net: ActorCritic) -> None:
        """Adopt the current policy as the new opponent."""
        self._net = self._snapshot(net)
        self.refreshes += 1

    def __call__(self, observation: dict, mask: np.ndarray) -> tuple[int, int, int]:
        flat = mask.reshape(-1)
        if not flat.any():
            return (0, 0, 0)
        torch = self._torch
        with torch.no_grad():
            # The actor only. An opponent chooses an action and never reads
            # a value, and with a separate critic encoder ``forward`` spends
            # about half its time computing one to throw away -- 2393us
            # against 1334us on the batch of one an opponent does.
            #
            # It is waste rather than a bottleneck, which is worth stating
            # because the microbenchmark invites the opposite reading: an
            # interleaved A/B over whole self-play battles came back at ~1.0x.
            # A decision is ~118ms and this forward is ~1.1ms of it.
            logits = self._net.policy_logits(
                torch.from_numpy(observation["grid"]).unsqueeze(0),
                torch.from_numpy(observation["vector"]).unsqueeze(0),
                torch.from_numpy(flat).unsqueeze(0),
            )
            # Sampled rather than greedy, but at a temperature. Greedy plays
            # one fixed line, and early on that line is an arbitrary
            # placement; sampling at 1.0 leaves an opponent barely different
            # from random. A temperature below one keeps the opponent varied
            # while making it predictable enough to learn against.
            #
            # Training wants that variety. A ladder wants the opposite: an
            # opponent that plays the same line every time, so a pairing
            # measures the policies and not the two generators behind them.
            if self.greedy:
                index = int(logits.argmax(dim=-1))
            elif self.generator is not None:
                index = int(torch.multinomial(
                    torch.softmax(logits / self.temperature, dim=-1), 1,
                    generator=self.generator))
            else:
                index = int(torch.distributions.Categorical(
                    logits=logits / self.temperature).sample())
        slots, width, height = self._nvec
        slot, remainder = divmod(index, width * height)
        gx, gy = divmod(remainder, height)
        return (min(slot, slots - 1), gx, gy)


class OpponentPool:
    """Past versions of the policy, kept rather than replaced.

    A single frozen opponent lets the learner cycle: it beats last week's
    strategy, forgets the one before, and goes round in circles looking busy
    the whole time. Nothing in the return says that is happening, because
    against its immediate past self a cycling policy wins about half the time
    forever.

    Keeping a spread of ancestors closes that off -- to do well the policy has
    to beat versions of itself from several stages back at once, and a
    counter-strategy that loses to a great-grandparent shows up immediately.

    The pool also supplies the measurement that matters. Progress against a
    *fixed* old self is far more readable than lift against a random control,
    whose per-episode spread is wide enough that a +0.23 reading on this
    project turned out to be noise.
    """

    __slots__ = ("_members", "_capacity", "_rng", "generations")

    def __init__(self, capacity: int = 8, seed: int = 0) -> None:
        self._members: list[ActorCritic] = []
        self._capacity = max(1, capacity)
        self._rng = np.random.default_rng(seed)
        #: How many snapshots have ever been added, which is the age scale the
        #: oldest surviving member is measured against.
        self.generations = 0

    def __len__(self) -> int:
        return len(self._members)

    def add(self, net: ActorCritic) -> None:
        """Take a snapshot of the current policy.

        Kept on the CPU whatever device the learner is on. These are only
        ever read -- to score the ladder, or to have their weights shipped to
        a worker -- and eight copies of a network sitting on an accelerator
        exhausted its resources partway through a run, which surfaced as an
        optimiser step failing rather than as anything pointing here.
        """
        import copy

        clone = copy.deepcopy(net).to("cpu")
        clone.eval()
        for parameter in clone.parameters():
            parameter.requires_grad_(False)
        self._members.append(clone)
        self.generations += 1
        while len(self._members) > self._capacity:
            # Drop from the middle, never the ends. The oldest member is the
            # benchmark the ladder is measured against and the newest is the
            # only one near the learner's own strength; evicting oldest-first
            # would turn the pool back into a sliding window of recent selves,
            # which is the thing it exists to avoid.
            self._members.pop(len(self._members) // 2)

    def sample(self) -> "ActorCritic | None":
        if not self._members:
            return None
        return self._members[int(self._rng.integers(len(self._members)))]

    def oldest(self) -> "ActorCritic | None":
        return self._members[0] if self._members else None


class PooledOpponent(FrozenOpponent):
    """A frozen opponent that draws a new ancestor from the pool on refresh."""

    __slots__ = ("_pool",)

    def __init__(self, pool: OpponentPool, net: ActorCritic, nvec, seed: int = 0,
                 temperature: float = 1.0) -> None:
        super().__init__(net, nvec, seed=seed, temperature=temperature)
        self._pool = pool

    def refresh(self, net: ActorCritic) -> None:
        """Adopt a random ancestor, not necessarily the newest one.

        ``net`` is ignored beyond keeping the signature the trainer expects --
        the pool is filled by the caller, so that adding a generation and
        choosing who plays stay separate decisions.
        """
        drawn = self._pool.sample()
        if drawn is not None:
            self._net = drawn
        self.refreshes += 1


def evaluation_probe(
    make_env: Callable[[], CRSimEnv],
    episodes: int = 40,
    seed: int = 12345,
) -> Callable[[ActorCritic], dict[str, Any]]:
    """Build a probe that scores a policy against a random control.

    Paired seeds: both arms play the same battles rather than the same
    *number* of battles. The per-episode spread here is several times larger
    than any effect worth seeing, so unpaired sampling would need far more
    episodes to say anything.
    """
    from .evaluate import evaluate

    seeds = [int(s) for s in np.random.default_rng(seed).integers(0, 2**31 - 1, episodes)]
    control_env = make_env()
    control = evaluate(control_env, None, episodes=episodes, seeds=seeds)
    control_wins = float(np.mean([c > 0 for c in control["crowns"]]))
    control_return = float(np.mean(control["returns"]))
    # Recorded here, from the environment the control actually played in, so
    # every lift this probe produces arrives with the scale it was measured
    # on attached. See :func:`opponent_name`.
    faced = opponent_name(control_env)
    # And the scale it was measured on, from the same environment. A lift is
    # a difference of returns, so it is denominated in whatever reward the
    # env was scoring -- see check_lift_is_named.
    scale = reward_name(control_env)

    def probe(net: ActorCritic) -> dict[str, Any]:
        result = evaluate(make_env(), net, episodes=episodes, seeds=seeds, greedy=False)
        wins = float(np.mean([c > 0 for c in result["crowns"]]))
        spread = float(np.std(control["returns"])) or 1.0
        return {
            "eval_return": float(np.mean(result["returns"])),
            "eval_win": wins,
            "control_return": control_return,
            "control_win": control_wins,
            # In control standard deviations, because the raw gap means
            # nothing without knowing how noisy the control is.
            "eval_lift_sd": (float(np.mean(result["returns"])) - control_return) / spread,
            "eval_opponent": faced,
            "eval_reward": scale,
            "eval_episodes": int(episodes),
        }

    return probe


def ancestor_probe(
    make_env,
    pool: OpponentPool,
    nvec,
    episodes: int = 30,
    seed: int = 999,
):
    """Score the current policy against the oldest version of itself.

    The most readable progress signal available here. The random control the
    other probe uses has a per-episode spread several times larger than any
    effect worth seeing, which is how six evaluations averaging +0.04 produced
    an individual reading of +0.23. An ancestor is fixed, deterministic to
    play against on fixed seeds, and roughly the right difficulty, so the same
    number of episodes says much more.

    Read it for what it is: beating your past self is evidence of movement,
    not of skill. Two policies can trade wins while both stay hopeless, which
    is why the random control stays as an anchor alongside this.
    """
    from .evaluate import evaluate

    seeds = [int(s) for s in np.random.default_rng(seed).integers(0, 2**31 - 1, episodes)]

    def probe(net: ActorCritic) -> dict:
        # No battles means no measurement, and no keys. np.mean([]) is NaN,
        # which json.dumps writes as a bare NaN token that is not valid JSON,
        # and which every consumer downstream then has to recognise: the page
        # drew an empty self-play ladder for a whole run because --ancestor
        # -episodes was 0 and the rows said NaN rather than saying nothing.
        # An absent key is the honest report of a measurement not taken.
        if episodes <= 0:
            return {}
        ancestor = pool.oldest()
        if ancestor is None:
            return {}
        age = pool.generations - len(pool) + 1
        # Named, at last. This probe used to build its opponent anonymously
        # and report a win rate against "the oldest one", which is a
        # measurement whose scale moves every time the pool evicts.
        env = make_env(FrozenOpponent(ancestor, nvec, seed=seed,
                                      name=f"pool:gen{age}"))
        result = evaluate(env, net, episodes=episodes, seeds=seeds, greedy=False)
        crowns = result["crowns"]
        if len(crowns) == 0:
            return {}
        wins = float(np.mean([c > 0 for c in crowns]))
        losses = float(np.mean([c < 0 for c in crowns]))
        return check_lift_is_named({
            "ancestor_win": wins,
            "ancestor_loss": losses,
            "ancestor_return": float(np.mean(result["returns"])),
            # How far back the benchmark sits, so a rising win rate can be
            # read against a benchmark that is itself getting older.
            "ancestor_age": age,
            # The same statistic the offline ladder fits ratings from --
            # wins plus half the draws -- so a self-play run's own ladder and
            # runs/<name>/ladder.json speak one language.
            "ancestor_score": wins + 0.5 * (1.0 - wins - losses),
            # Its own keys, not eval_opponent and not the rating ladder's.
            # cr_sim.train.run merges this dict into one row with the
            # random-control probe's and, under --probe ladder, the rating
            # probe's too. Sharing a field means whichever writer runs last
            # relabels the others: measured, the ancestor's score arrived on
            # a row claiming it had been played against the rating ladder's
            # anchors. See check_lift_is_named.
            "ancestor_opponent": "pool",
            # Which weights, not just which kind. An integer age against a
            # pool that evicts from the middle is not a reference anything
            # can be scored relative to later.
            "ancestor_opponent_ref": f"gen{age}",
            "ancestor_episodes": int(episodes),
        })

    return probe
