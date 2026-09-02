"""Let the policy say which placements are worth an exact engine branch.

**The arrow that was missing.** AlphaZero's loop has three: the policy
proposes, the search refines the proposal, and the refined distribution trains
the policy. This project built the second and the third -- ``SearchBot``
branches the battle for each candidate and :func:`cr_sim.train.clone.collect`
trains against the search's own value distribution -- and never built the
first. ``SearchBot._sample_actions`` draws about fourteen stratified-random
placements from a mean of **104 legal actions**, which is 13.5% coverage,
measured. The other 86.5% are never scored. The engine that makes each branch
exact is doing exact work on an arbitrary sample, and the only object in the
system holding an opinion about *which* fourteen are worth scoring has never
been consulted.

**Why it is affordable here and nowhere else.** One search decision at
``candidates=14, horizon_seconds=15`` measures 375 ms on this machine, 27.7 ms
of it per branch. One ``policy_logits`` forward at batch 1 measures 1.45 ms.
The forward is **0.386% of a decision**, so at an equal candidate budget
guiding the proposal is free -- and the budget is held equal in code, because
a win bought with more branches is not the win being claimed.

**Determinism, which is the part that has already gone wrong here.** Torch's
``Categorical.sample`` draws from the global generator unless it is handed one,
and that defect is live in production: every sampled number in
``runs/_anchor/*.json`` is unreproducible because of it, and two arms of one
A/B once produced 28 and 29 decisions from what was meant to be the same
battle. So:

*   ``temperature == 0.0`` -- the default -- **touches no random number
    generator at all.** It is a stable argsort over a numpy copy, so ties
    break by ascending flat index. ``torch.topk`` promises no ordering among
    exact ties and factored heads produce those routinely.
*   ``temperature > 0`` draws from a ``torch.Generator`` this proposer owns,
    seeded arithmetically from (proposer seed, battle seed, decision index).
    Never from ``hash()``: string hashing is salted per process, so a
    hash-derived stream is reproducible within a run and not between two.
*   The global stream is never read and never written. No ``torch.manual_seed``
    anywhere in this file.

**One honest limit.** A network forward is not bit-stable across thread
counts, so a different reduction order can flip a near-tie and change which
placements are proposed. Two things bound the damage: the search **rescores
every proposal with the exact engine**, so a difference in proposal *order*
changes nothing and only a difference in the candidate *set* can change the
decision; and :func:`proposer_identity` stamps the checkpoint's SHA-256, the
temperature, the candidate split and the thread count into the demonstration
file, so a divergence is detectable rather than silent. Bit-determinism across
machines is claimed only for ``proposer=None``, where it is true.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..api.encoding import NOOP_SLOT
from .scripted import SearchBotConfig

__all__ = [
    "policy_proposer", "proposer_factory", "proposer_identity",
    "check_equal_branch_budget", "measure_decision_cost",
]


def _stream_seed(seed: int, battle_seed: int, decision: int) -> int:
    """The generator seed for one proposal, derived by arithmetic.

    Arithmetic and not ``hash()``. Python salts string hashing per process, so
    a hash-derived stream reproduces inside one run and not between two, which
    is the worst possible failure: it looks deterministic for exactly as long
    as anybody is watching.
    """
    return (int(seed) * 1_000_003 + int(battle_seed) * 7919
            + int(decision)) % (2 ** 31 - 1)


def policy_proposer(net, nvec, *, temperature: float = 0.0, seed: int = 0,
                    top: int = 64, battle_seed_of: "Callable[[], int] | None" = None
                    ) -> Callable[[Any, np.ndarray, int], Sequence[int]]:
    """Rank the legal actions by the policy's own logits.

    ``net`` is an :class:`~cr_sim.train.nets.ActorCritic`; only its actor is
    run, because a proposal never reads a value. ``nvec`` is the action
    space's ``(slots, width, height)``.

    ``temperature == 0.0`` is a stable argsort and touches no generator at
    all. Above zero it is a draw without replacement from a
    ``torch.Generator`` this proposer owns -- a Plackett-Luce ordering, so
    taking the first *k* of what comes back is exactly "sample k without
    replacement", and the caller does not have to know k in advance.

    ``top`` is how many indices come back, and it has to stay comfortably
    above the ``policy_candidates`` any caller will ask for: the search takes
    the first distinct legal ones it is handed, so a short list is a silent
    cap on how much of the budget the policy actually gets. 64 against a
    shipped candidate count of 14.

    ``battle_seed_of`` is a zero-argument callable returning the seed of the
    battle in progress, for a proposer that outlives one battle. Everywhere
    in this package the bot is rebuilt per battle instead, so this is normally
    ``None`` and the battle seed is folded into ``seed`` by
    :func:`proposer_factory`.
    """
    import torch

    _slots, width, height = (int(v) for v in nvec)
    noop = NOOP_SLOT * width * height
    temperature = float(temperature)

    def propose(observation, mask, decision: int) -> list[int]:
        flat = np.ascontiguousarray(np.asarray(mask).reshape(-1))
        legal = np.flatnonzero(flat)
        # The no-op is seeded into the search's scores unconditionally and is
        # not a candidate; proposing it would spend a slot on a branch the
        # bot already has for free.
        legal = legal[legal != noop]
        if not len(legal):
            return []
        with torch.no_grad():
            logits = net.policy_logits(
                torch.from_numpy(np.asarray(observation["grid"], dtype=np.float32)).unsqueeze(0),
                torch.from_numpy(np.asarray(observation["vector"], dtype=np.float32)).unsqueeze(0),
                torch.from_numpy(flat).unsqueeze(0),
            )
        # A numpy copy in float64. The ordering is decided here rather than by
        # torch.topk, which promises nothing about exact ties -- and the
        # factored head produces exact ties across tiles routinely, because
        # the tile logits are shared and only the card term differs.
        values = np.asarray(logits[0].detach().to(torch.float64).numpy())[legal]
        if temperature <= 0.0:
            # Stable, so equal logits come back in ascending flat-index order
            # rather than in whatever order a kernel's reduction produced.
            order = np.argsort(-values, kind="stable")
            return [int(legal[int(i)]) for i in order[:max(0, int(top))]]

        weights = np.exp((values - values.max()) / temperature)
        positive = int(np.count_nonzero(weights))
        draws = min(int(top), positive, len(legal))
        if draws <= 0:
            return []
        battle_seed = 0 if battle_seed_of is None else int(battle_seed_of())
        generator = torch.Generator().manual_seed(
            _stream_seed(seed, battle_seed, decision))
        picked = torch.multinomial(torch.from_numpy(weights), draws,
                                   replacement=False, generator=generator)
        return [int(legal[int(i)]) for i in picked]

    return propose


def proposer_factory(net, nvec, *, temperature: float = 0.0, top: int = 64,
                     seed: int = 0):
    """A per-battle proposer builder, for a bot rebuilt from each battle's seed.

    ``search_opponent`` and ``ladder._search_driver`` already rebuild the
    ``SearchBot`` whenever the battle seed changes -- the bot samples its
    candidates, so a bot carried across episodes is a function of how many
    decisions came before rather than of the seed. The proposer is rebuilt
    with it, from the same derived seed, so the same property holds for its
    stream.
    """
    def build(battle_seed: int):
        return policy_proposer(net, nvec, temperature=temperature,
                               seed=(int(seed) * 1_000_003 + int(battle_seed))
                               % (2 ** 31 - 1), top=top)

    return build


def proposer_identity(checkpoint: "str | Path | None", *,
                      temperature: float = 0.0,
                      policy_candidates: int = 0) -> str:
    """What a demonstration shard records about who proposed its candidates.

    The weights by content, not by path: ``runs/iter-2/cloned.pt`` is a
    different network on Tuesday than it was on Monday, and a shard naming the
    path would merge cleanly with one collected from the earlier file. A
    SHA-256 prefix cannot.

    ``None`` is ``"random"`` -- the unguided bot, which is the default and the
    thing every existing shard on disk was collected with.
    """
    if checkpoint is None:
        return "random"
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()[:12]
    return f"policy:{digest}@t{float(temperature):g}p{int(policy_candidates)}"


def _config_of(player) -> SearchBotConfig:
    if isinstance(player, SearchBotConfig):
        return player
    config = getattr(player, "search", None) or getattr(player, "config", None)
    if not isinstance(config, SearchBotConfig):
        raise ValueError(
            f"{player!r} is not a search player: an equal-budget comparison "
            "is a claim about two searches, and there is nothing here to "
            "compare budgets between.")
    return config


def check_equal_branch_budget(a, b) -> None:
    """Refuse to compare two searches that are not paying the same price.

    The claim policy-guided proposal makes is "the same fourteen branches,
    spent better". A guided bot that quietly took sixteen would win, and it
    would win for the least interesting reason there is -- which is exactly
    what a comparison against ``proposer=None`` is supposed to rule out.

    Accepts a :class:`~cr_sim.train.scripted.SearchBotConfig`, a
    :class:`~cr_sim.train.ladder.Player`, or a built ``SearchBot``.
    """
    left, right = _config_of(a), _config_of(b)
    for field in ("candidates", "horizon_seconds"):
        mine, theirs = getattr(left, field), getattr(right, field)
        if mine != theirs:
            raise ValueError(
                f"these two searches do not have the same {field}: "
                f"{mine} against {theirs}. Policy-guided proposal claims the "
                "same branch budget spent better, and a budget difference "
                "would win the comparison on its own -- 14 candidates against "
                "6 is a different bot, and 15 seconds of horizon against 8 "
                "moves the win rate from 100% to 94% by itself. Equalise the "
                "budget or stop calling it a proposal comparison.")


def measure_decision_cost(make_env, make_bot, *, seeds: Sequence[int]) -> dict:
    """What a decision actually costs, and how many branches it actually took.

    The second half of the equal-budget claim, verified rather than asserted:
    ``check_equal_branch_budget`` checks what the two bots were *configured*
    to spend, and this counts what they *spent*. A proposer that appends its
    nominations to the random draw instead of replacing part of it passes the
    first check and fails this one.

    ``make_env(seed) -> env`` and ``make_bot(seed) -> SearchBot``. Returns the
    decision count, the branch count, the wall clock inside the bot, and the
    two ratios worth quoting.
    """
    import time

    decisions = branches = 0
    elapsed = 0.0
    for seed in seeds:
        env = make_env(int(seed))
        observation, _ = env.reset(seed=int(seed))
        bot = make_bot(int(seed))
        while True:
            mask = env.legal_action_mask()
            if not mask.reshape(-1).any():
                break
            started = time.perf_counter()
            choice = bot(observation, mask, env.battle)
            elapsed += time.perf_counter() - started
            decisions += 1
            observation, _, terminated, truncated, _ = env.step(choice)
            if terminated or truncated:
                break
        branches += int(bot.evaluated)
    return {
        "decisions": decisions,
        "evaluated": branches,
        "seconds": elapsed,
        "branches_per_decision": branches / max(1, decisions),
        "seconds_per_decision": elapsed / max(1, decisions),
    }
