"""Rate the policies against each other instead of lifting them off a control.

**Why the old metric had to go.** Every number this project has ever recorded
is a lift over a uniform random control, and that control is used up. The
behavioural clone beats it 96/2/2; the one-ply search expert beats it 100-0.
Everything the project has built -- the head ablation, the observation
ablation, the card encoder, a million steps of PPO -- lands between +2.167 and
+2.191 sd, a band **0.024 sd wide**. The expert anchor those results are aimed
at is quoted as +2.716 [+2.369, +3.063], a 95% half-width of +/-0.347 at n=40,
i.e. +/-0.179 at n=150. The metric's own resolution at the n anybody actually
runs is about seven times wider than the entire range of results it is being
asked to separate. That is not a metric with headroom left; it is a metric
that cannot distinguish any two arms on this machine.

**What replaces it.** A rating, which is *transitive*: a policy never has to
play the expert to be scored relative to it, provided the pairing graph is
connected. That one property is what makes expert-relative scoring affordable
as an in-run probe -- the obvious answer, making the probe face the expert
itself, costs 20.1 s per battle and would spend 73% of a 1M-step run on
evaluation. Rating through cheap anchors costs 4.9%.

The scale is pinned so the new numbers can be read against the old ones:
``random = 0 Elo``, by construction, because random is what every existing
measurement was taken against. The clone's 96/2/2 is a score of 0.97, which is
``400*log10(0.97/0.03)`` = **+604 Elo**. The band that is 0.024 sd wide today
is about 300 points wide here, against a +/-26 Elo floor at 400 seeds a
direction.

**Every pairing is played both directions, always, and this is not optional.**
:meth:`cr_sim.api.env.CRSimEnv.step` applies the controlled side's action and
*then* calls ``_opponent_move()``, inside every frame-skip window, all match:
the controlled side moves first, every time. Measured, the same weights on
both sides of a pairing scored 0.550 and 0.600 for the controlled side on two
independent nets where 0.500 is the truth. At n=40 neither is significant on
its own, but the mechanism is in the source, the sign is consistent, and the
effect is the same size as the A-vs-B disagreement between the two directions.
So the mirror here swaps *which player is controlled*, not merely which colour
it wears -- that cancels the colour and the move order together, and it is
what makes a self-pairing come out at exactly 0.5 rather than approximately.

**Greedy and sampled are separate arms throughout.** Separate pairings,
separate ratings, separate tables, never averaged into one headline. The clone
measures +1.623 greedy and +0.709 sampled against the same control; one number
for both hid exactly that once already.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..api.env import CRSimEnv
from ..engine.entity import Team
from .evaluate import EVAL_BLOCKS, check_observation, evaluation_seeds
from .nets import ActorCritic, net_config_for
from .ppo import _unflatten_action
from .scripted import SearchBot, SearchBotConfig
from .selfplay import FrozenOpponent, check_lift_is_named, opponent_name

__all__ = [
    "Player", "Direction", "Pairing", "Rating",
    "HEAD_BY_PARAMETERS", "head_for_parameters", "player_from_checkpoint",
    "default_player_name", "parse_player",
    "play_pairing", "fit_ratings", "expected_score", "ladder_probe",
]

#: ``search-c6h8`` -- six candidates, eight seconds a branch.
_SEARCH_SPEC = re.compile(r"^search-c(\d+)h([\d.]+)$")

#: Elo's natural-scale constant: a 400-point gap is 10:1 odds.
_C = math.log(10.0) / 400.0

#: Which head a checkpoint's parameter count identifies, for the 22 of 42
#: checkpoints on this machine that record neither ``head`` nor
#: ``observation``.
#:
#: ``load_policy`` defaults those to flat/v1, which is a *guess* -- right for
#: some of them and unverifiable from the file. A ladder must not guess
#: silently, because a checkpoint loaded into the wrong head does not fail, it
#: plays badly, and a rating built on it is a rating of the wrong network.
#:
#: These are the v1 counts. A v2 or v3 checkpoint has different ones (flat/v2
#: is 1,843,153 and factored/v3 is 1,719,862), so a count that matches nothing
#: here is either a different observation or the 1,011,921-parameter
#: generation that predates the current ActorCritic -- nine files on this
#: machine, none of which load at all. Either way it is refused rather than
#: assumed.
HEAD_BY_PARAMETERS = {
    1_838_545: "flat",
    1_710_646: "factored",
    1_662_214: "conv",
    1_713_046: "factored-stats",
}


def head_for_parameters(parameters: int) -> str:
    """Which head has ``parameters`` weights, or a refusal.

    Raises rather than falling back to "flat". An unrecognised count means the
    file is not a v1 checkpoint of a current head, and entering it into a
    ladder under a guessed head produces a rating for a network nobody built.
    """
    head = HEAD_BY_PARAMETERS.get(int(parameters))
    if head is None:
        raise ValueError(
            f"no head on this machine has {int(parameters):,} parameters. "
            f"The v1 counts are {HEAD_BY_PARAMETERS}. This checkpoint is "
            "either a different observation -- which cannot share a ladder "
            "with v1 anyway, since the environment encodes one observation "
            "for both sides -- or the 1,011,921-parameter generation that "
            "predates the current ActorCritic. Refused rather than guessed: "
            "a checkpoint loaded into the wrong head does not raise, it "
            "plays badly."
        )
    return head


@dataclass(frozen=True, slots=True)
class Player:
    """One entrant, and everything needed to say *which* entrant it was.

    ``checkpoint`` is the ``ladder_opponent_ref`` a metrics row carries.
    "pool" is not an opponent; ``runs/clone-v3-paired/cloned.pt`` is.
    """

    name: str
    #: "net" | "search" | "random".
    kind: str
    checkpoint: Path | None = None
    head: str = "flat"
    observation: str = "v1"
    search: SearchBotConfig | None = None
    #: Whether ``head`` came out of the file or out of the parameter count.
    #: Recorded on every player, so a reader can see which ratings rest on an
    #: inference.
    head_source: str = "recorded"
    #: A live network, for the in-run probe, which rates weights that are not
    #: on disk yet. Excluded from equality: two Players are the same entrant
    #: when they name the same weights.
    net: Any = field(default=None, compare=False, repr=False)
    seed: int = 0

    @property
    def ref(self) -> str:
        """What a row records as ``ladder_opponent_ref``."""
        if self.checkpoint is not None:
            return str(self.checkpoint)
        if self.kind == "search" and self.search is not None:
            return (f"c{self.search.candidates}"
                    f"h{self.search.horizon_seconds:g}")
        if self.kind == "random":
            return "random"
        return "live"

    def load(self, env: CRSimEnv) -> "Player":
        """Return a copy holding the weights, ready to play.

        The shapes come from the environment, never from the file, which is
        what makes a mismatch fail loudly here instead of quietly scoring a
        policy against an observation it was never trained on.
        """
        if self.kind != "net" or self.net is not None:
            return self
        import torch

        payload = torch.load(self.checkpoint, map_location="cpu",
                             weights_only=False)
        check_observation(payload, env)
        net = ActorCritic(net_config_for(env, head=self.head))
        net.load_state_dict(payload["state_dict"])
        net.eval()
        return replace(self, net=net)


#: Filenames that say *when* a checkpoint was written rather than *what* it
#: is. Every run directory holds several of them.
_GENERIC_STEMS = frozenset({"cloned", "best", "final", "checkpoint"})


def default_player_name(path: Path) -> str:
    """A name that identifies the weights rather than the directory.

    ``checkpoints/headablate-flat.pt`` and ``checkpoints/headablate-factored.pt``
    are both in ``checkpoints/``, so naming a player after its parent silently
    merges two entrants into one row -- measured on a smoke ladder, where both
    head ablations arrived as "checkpoints" and the fit rated a player that
    does not exist. And ``runs/learn-lvl5-kl01/{best,final}.pt`` are two
    different policies whose filenames are the only thing telling them apart.
    """
    stem = path.stem
    return f"{path.parent.name}:{stem}" if stem in _GENERIC_STEMS else stem


def player_from_checkpoint(path: "str | Path", *, name: str | None = None,
                           seed: int = 0) -> Player:
    """Read a checkpoint's identity off the file, inferring only what it must.

    ``head`` and ``observation`` are taken from the payload where they are
    recorded. Where they are not -- 22 of the 42 checkpoints here -- the head
    is inferred from the parameter count and ``head_source`` says so, and a
    count matching no head is refused outright.
    """
    import torch

    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") or {}
    recorded = payload.get("head")
    if recorded is None:
        head = head_for_parameters(sum(int(v.numel()) for v in state.values()))
        source = "inferred-from-parameter-count"
    else:
        head, source = str(recorded), "recorded"
    return Player(
        name=name or default_player_name(path),
        kind="net",
        checkpoint=path,
        head=head,
        observation=str(payload.get("observation", "v1")),
        head_source=source,
        seed=seed,
    )


def parse_player(spec: str) -> Player:
    """One entrant, from a command-line word or a config string.

    ``random``; ``search-c<candidates>h<horizon>``; or a path to a
    checkpoint, optionally written ``name=path``.

    The search bot is named for its *configuration*, never bare "search". A
    thinned expert is a different opponent -- 4s of horizon wins 31% of its
    matches, 8s wins 94%, 15s wins 100%, and at 2s the bot places nothing at
    all and is an idle opponent still wearing the label. Inheriting the name
    would move the scale without saying so.
    """
    name, _, rest = spec.partition("=")
    if not rest:
        name, rest = "", spec
    if rest == "random":
        return Player(name=name or "random", kind="random")
    found = _SEARCH_SPEC.match(rest)
    if found:
        return Player(
            name=name or rest, kind="search", seed=3,
            search=SearchBotConfig(candidates=int(found.group(1)),
                                   horizon_seconds=float(found.group(2))))
    return player_from_checkpoint(Path(rest), name=name or None)


@dataclass(frozen=True, slots=True)
class Direction:
    """One colour assignment of a pairing, played to the end.

    ``blue`` is the controlled side -- the one the environment plays and the
    one that moves first. ``red`` is read off the environment by
    :func:`cr_sim.train.selfplay.opponent_name`, never taken from the argument
    list, so a direction cannot be labelled with an opponent it did not face.
    """

    blue: str
    red: str
    seeds: list[int]
    wins: int
    losses: int
    draws: int
    #: (wins + 0.5*draws) / n, from BLUE's -- the controlled side's -- point
    #: of view.
    score: float
    #: Per battle, team-relative. See cr_sim.train.evaluate.evaluate: this was
    #: hardcoded to blue and is now the agent's own side.
    crowns: list[int]

    @property
    def results(self) -> list[float]:
        """Per battle, 1.0 / 0.5 / 0.0 for the controlled side."""
        return [1.0 if c > 0 else (0.0 if c < 0 else 0.5) for c in self.crowns]


@dataclass(frozen=True, slots=True)
class Pairing:
    """Both directions of one match-up, mirror-averaged."""

    a: str
    b: str
    #: "greedy" | "sampled". Never merged.
    mode: str
    forward: Direction
    reverse: Direction
    #: a's score, mirror-averaged: ``(forward + (1 - reverse)) / 2``.
    score: float
    #: Per-seed correlation between the two directions, ``None`` where one of
    #: them is constant and the correlation is undefined.
    #:
    #: A first-class output, not a diagnostic. The battle-count table turns on
    #: whether the two directions on one seed are independent, and nobody has
    #: ever measured it. Until it is measured, the conservative reading holds:
    #: treat them as perfectly correlated, effective n = seeds per direction,
    #: and quote +/-26 Elo at 400 rather than the +/-19 that independence
    #: would buy.
    seed_correlation: float | None
    a_ref: str = ""
    b_ref: str = ""

    @property
    def games(self) -> int:
        return (self.forward.wins + self.forward.losses + self.forward.draws
                + self.reverse.wins + self.reverse.losses + self.reverse.draws)

    @property
    def a_wins(self) -> float:
        """a's wins across both directions, draws counted separately."""
        return float(self.forward.wins + self.reverse.losses)

    @property
    def a_draws(self) -> float:
        return float(self.forward.draws + self.reverse.draws)


@dataclass(frozen=True, slots=True)
class Rating:
    """A fitted rating, its standard error, and what it was fitted from."""

    name: str
    elo: float
    error: float
    games: int
    #: True where the rating was held fixed rather than fitted -- the anchor
    #: at 0, or an offline rating the in-run probe rates against.
    pinned: bool = False


# --------------------------------------------------------- playing a pairing


def _random_driver(rng: np.random.Generator):
    def act(observation, mask, battle):
        legal = np.flatnonzero(mask.reshape(-1))
        return int(legal[rng.integers(len(legal))])
    return act


def _net_driver(net, *, greedy: bool, generator):
    import torch

    def act(observation, mask, battle):
        flat = mask.reshape(-1)
        with torch.no_grad():
            logits = net.policy_logits(
                torch.from_numpy(observation["grid"]).unsqueeze(0),
                torch.from_numpy(observation["vector"]).unsqueeze(0),
                torch.from_numpy(flat).unsqueeze(0),
            )
        if greedy:
            return int(logits.argmax(dim=-1))
        # A stream this pairing owns. Without one the draw comes off torch's
        # global generator, which nothing seeds -- the defect that makes every
        # sampled number in runs/_anchor/*.json unreproducible. The ladder
        # does not get to be the fourth writer with it.
        return int(torch.multinomial(torch.softmax(logits, dim=-1), 1,
                                     generator=generator))
    return act


def _search_driver(player: Player, team: Team, width: int, height: int):
    """The search bot on the *controlled* side.

    Rebuilt per battle from the battle's own seed, the way
    :func:`cr_sim.train.evaluate.search_opponent` rebuilds its own. The bot
    samples its candidates, so what it plays depends on how far its generator
    has been advanced -- which, if it were built once, would be a count of
    every decision in every episode before this one rather than a function of
    the seed.
    """
    state: dict[str, Any] = {"bot": None, "seed": object()}

    def act(observation, mask, battle):
        key = int(battle.config.seed)
        if state["bot"] is None or key != state["seed"]:
            base = player.search or SearchBotConfig()
            derived = (player.seed * 1_000_003 + key) % (2 ** 31 - 1)
            state["seed"] = key
            state["bot"] = SearchBot(team, replace(base, seed=derived))
        slot, gx, gy = state["bot"](observation, mask, battle)
        return int(slot) * width * height + int(gx) * height + int(gy)

    return act


def _opponent_policy(player: Player, nvec, *, mode: str, stream_seed: int):
    """``player`` in the shape ``CRSimEnv``'s ``opponent_policy`` expects.

    Named in every branch. A frozen opponent could not be named at all until
    ``opponent_name`` joined its ``__slots__``, and a ladder whose opponent
    reports "unknown" writes rows that ``check_lift_is_named`` refuses.
    """
    if player.kind == "random":
        from .run import _random_opponent

        policy = _random_opponent(stream_seed)
        policy.opponent_name = player.name  # type: ignore[attr-defined]
        return policy
    if player.kind == "search":
        from .evaluate import search_opponent

        policy = search_opponent(player.search or SearchBotConfig(),
                                 seed=player.seed)
        # The player's own name, not the bare "search". A thinned expert is a
        # *different opponent* -- 4s wins 31%, 8s 94%, 15s 100%, and a 2s
        # horizon is an idle opponent still wearing the label -- so inheriting
        # the name would silently move the scale.
        policy.opponent_name = player.name  # type: ignore[attr-defined]
        return policy
    import torch

    generator = None
    if mode != "greedy":
        generator = torch.Generator().manual_seed(stream_seed)
    return FrozenOpponent(player.net, nvec, seed=stream_seed,
                          name=player.name, greedy=(mode == "greedy"),
                          generator=generator)


def _stream_seed(seeds: Sequence[int], direction_index: int, mode: str) -> int:
    """One pairing arm's stream, derived arithmetically from the battles.

    Never from ``hash()``: string hashing is salted per process, so a
    hash-derived stream is reproducible within a run and not between two. The
    same discipline ``evaluate_paired`` already uses.
    """
    mode_index = 0 if mode == "greedy" else 1
    first = int(seeds[0]) if len(seeds) else 0
    return (first + 7919 * int(direction_index)
            + 104729 * mode_index) % (2 ** 31 - 1)


def _play_direction(make_env, blue: Player, red: Player, *, seeds,
                    mode: str, direction_index: int, nvec) -> Direction:
    stream = _stream_seed(seeds, direction_index, mode)
    slots, width, height = (int(v) for v in nvec)
    # Handed to the factory rather than assigned onto a built environment.
    # ``make_env`` is the caller's, and a caller that wraps or substitutes its
    # opponent -- which is exactly how the label is checked -- has to be the
    # one holding it.
    env = make_env(_opponent_policy(red, nvec, mode=mode, stream_seed=stream))
    # Off the environment, never off the argument. A caller cannot label a
    # direction with an opponent it did not actually play.
    faced = opponent_name(env)

    if blue.kind == "random":
        act = _random_driver(np.random.default_rng(stream))
    elif blue.kind == "search":
        act = _search_driver(blue, env.team, width, height)
    else:
        import torch

        generator = (None if mode == "greedy"
                     else torch.Generator().manual_seed(stream + 1))
        act = _net_driver(blue.net, greedy=(mode == "greedy"),
                          generator=generator)

    mine, theirs = env.team, env.team.opponent
    crowns: list[int] = []
    for seed in seeds:
        observation, _ = env.reset(seed=int(seed))
        while True:
            mask = env.legal_action_mask()
            choice = act(observation, mask, env.battle)
            observation, _, terminated, truncated, info = env.step(
                _unflatten_action(int(choice), nvec))
            if terminated or truncated:
                break
        crowns.append(int(info[f"{mine.name.lower()}_crowns"]
                          - info[f"{theirs.name.lower()}_crowns"]))

    wins = sum(1 for c in crowns if c > 0)
    losses = sum(1 for c in crowns if c < 0)
    draws = len(crowns) - wins - losses
    total = max(1, len(crowns))
    return Direction(blue=blue.name, red=faced, seeds=[int(s) for s in seeds],
                     wins=wins, losses=losses, draws=draws,
                     score=(wins + 0.5 * draws) / total, crowns=crowns)


def play_pairing(make_env, a: Player, b: Player, *, seeds: Sequence[int],
                 mode: str = "greedy") -> Pairing:
    """Play ``a`` against ``b`` both ways round and mirror-average the score.

    ``make_env(opponent_policy) -> CRSimEnv`` -- the same factory
    ``cr_sim.train.run`` already builds its environments through.

    The mirror swaps which player the environment *controls*, not merely which
    colour it wears. The controlled side acts before ``_opponent_move()``
    inside every frame-skip window, so colour and move order are one
    advantage here, and swapping the controlled side cancels both at once. It
    is also what makes a self-pairing score exactly 0.5: both directions are
    then literally the same battles, and ``(s + (1 - s)) / 2`` is 0.5 for any
    s, without an assumption about how large the bias was.
    """
    # One environment built to read the action shape off, because an opponent
    # has to know it before the environment that will hold it exists.
    nvec = [int(v) for v in make_env(None).action_space.nvec]
    forward = _play_direction(make_env, a, b, seeds=seeds, mode=mode,
                              direction_index=0, nvec=nvec)
    reverse = _play_direction(make_env, b, a, seeds=seeds, mode=mode,
                              direction_index=1, nvec=nvec)
    mine = np.asarray(forward.results, dtype=float)
    theirs = 1.0 - np.asarray(reverse.results, dtype=float)
    correlation: float | None = None
    if len(mine) > 1 and mine.std() > 0.0 and theirs.std() > 0.0:
        correlation = float(np.corrcoef(mine, theirs)[0, 1])
    return Pairing(
        a=a.name, b=b.name, mode=mode, forward=forward, reverse=reverse,
        score=(forward.score + (1.0 - reverse.score)) / 2.0,
        seed_correlation=correlation, a_ref=a.ref, b_ref=b.ref,
    )


# ---------------------------------------------------------------- the rating


def expected_score(elo_a: float, elo_b: float) -> float:
    """The logistic Elo expectation -- a 400-point gap is 10:1 odds."""
    return 1.0 / (1.0 + math.exp(-_C * (float(elo_a) - float(elo_b))))


def fit_ratings(pairings: Sequence[Pairing], *, prior_sd: float = 400.0,
                anchor: str = "random",
                pinned: "dict[str, float] | None" = None,
                iterations: int = 200,
                tolerance: float = 1e-10) -> dict[str, Rating]:
    """Bradley-Terry MAP over the pairing graph, draws as half a win each way.

    **The prior is not decoration.** The expert beats the random control 100-0
    and an unregularised fit diverges on that edge: the likelihood is
    monotone in the rating difference and its maximum is at infinity. A
    Gaussian ``N(0, prior_sd)`` on every rating makes the posterior mode
    finite -- the rating then grows logarithmically in the number of battles
    rather than without bound -- and its Hessian gives every player a standard
    error, which a saturated maximum-likelihood fit cannot.

    ``anchor`` is held at 0 by construction, because ``random = 0`` is the
    scale every number this project has recorded was measured against.
    ``pinned`` holds any other player at a rating fitted elsewhere, which is
    how the in-run probe rates one policy against anchors whose ratings came
    from the offline ladder.
    """
    pinned = dict(pinned or {})
    names: list[str] = []
    for pairing in pairings:
        for name in (pairing.a, pairing.b):
            if name not in names:
                names.append(name)
    for name in pinned:
        if name not in names:
            names.append(name)
    if anchor in names:
        pinned.setdefault(anchor, 0.0)
    if not names:
        return {}
    index = {name: i for i, name in enumerate(names)}
    size = len(names)

    # One edge per unordered pair, both directions folded in: a rating is a
    # property of the match-up, not of the colour.
    edges: dict[tuple[int, int], list[float]] = {}
    games: dict[str, int] = {name: 0 for name in names}
    for pairing in pairings:
        i, j = index[pairing.a], index[pairing.b]
        key = (i, j) if i <= j else (j, i)
        wins = pairing.a_wins + 0.5 * pairing.a_draws
        total = float(pairing.games)
        if i > j:
            wins = total - wins
        slot = edges.setdefault(key, [0.0, 0.0])
        slot[0] += wins
        slot[1] += total
        games[pairing.a] += int(total)
        games[pairing.b] += int(total)

    rating = np.zeros(size, dtype=float)
    for name, value in pinned.items():
        rating[index[name]] = float(value)
    free = [i for i, name in enumerate(names) if name not in pinned]
    has_prior = math.isfinite(prior_sd) and prior_sd > 0.0
    precision = 1.0 / (prior_sd ** 2) if has_prior else 0.0

    hessian = np.zeros((size, size), dtype=float)
    for _ in range(int(iterations)):
        gradient = np.zeros(size, dtype=float)
        hessian = np.zeros((size, size), dtype=float)
        for (i, j), (wins, total) in edges.items():
            if i == j or total <= 0.0:
                continue
            p = 1.0 / (1.0 + math.exp(
                -_C * float(np.clip(rating[i] - rating[j], -50_000, 50_000))))
            residual = _C * (wins - total * p)
            gradient[i] += residual
            gradient[j] -= residual
            curvature = _C * _C * total * p * (1.0 - p)
            hessian[i, i] -= curvature
            hessian[j, j] -= curvature
            hessian[i, j] += curvature
            hessian[j, i] += curvature
        gradient -= rating * precision
        hessian[np.diag_indices(size)] -= precision
        if not free:
            break
        block = -hessian[np.ix_(free, free)]
        try:
            step = np.linalg.solve(block, gradient[free])
        except np.linalg.LinAlgError:
            # Singular only where there is no prior and an edge is saturated,
            # which is the case the prior exists to stop. Fall through rather
            # than raise, so a fit run without one reports an absurd rating
            # instead of a traceback that hides why.
            step = np.linalg.pinv(block) @ gradient[free]
        rating[free] += step
        if float(np.max(np.abs(step))) < tolerance:
            break

    block = -hessian[np.ix_(free, free)] if free else np.zeros((0, 0))
    try:
        covariance = np.linalg.inv(block) if len(block) else block
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(block) if len(block) else block
    errors = np.zeros(size, dtype=float)
    for position, i in enumerate(free):
        variance = float(covariance[position, position]) if len(block) else 0.0
        errors[i] = math.sqrt(variance) if variance > 0.0 else float("inf")

    return {
        name: Rating(name=name, elo=float(rating[index[name]]),
                     error=float(errors[index[name]]),
                     games=int(games.get(name, 0)),
                     pinned=name in pinned)
        for name in names
    }


# ------------------------------------------------------------ the in-run probe


def ladder_probe(make_env, anchors: Sequence[Player], *, episodes: int = 40,
                 blocks: int = EVAL_BLOCKS,
                 ratings: "dict[str, float] | None" = None,
                 mode: str = "greedy", seed: int = 12345,
                 expert: str = "search-c18h15"
                 ) -> Callable[[ActorCritic], dict[str, Any]]:
    """Score the live policy against a few cheap anchors and rate it.

    **This is the affordability answer.** Three anchors at 40 seeds a
    direction is 72.5 s a reading: 29 minutes over a 1M-step run, 4.9% of it.
    Making the probe face the full search expert instead is 13.4 minutes a
    reading -- 5.4 hours of battles plus 1.8 hours of rotating control blocks,
    73% of the run. The rating is transitive, so ``ladder_elo_vs_expert``
    comes out of anchors the expert has already been rated against, without
    the probe ever playing it.

    Seed blocks rotate, the way ``rotating_probe`` does, because the
    promotion window is three readings and three readings of one fixed seed
    list share all of their seed-level luck -- the failure the rolling mean
    was introduced to fix, one level down. Three readings over disjoint
    blocks is 120 seeds a direction, +/-48 Elo.
    """
    ratings = dict(ratings or {})
    readings = {"n": 0}

    def probe(net: ActorCritic) -> dict[str, Any]:
        block = readings["n"] % max(1, blocks)
        readings["n"] += 1
        block_seeds = evaluation_seeds(episodes, block=block, seed=seed,
                                       blocks=blocks)
        entrant = Player(name="policy", kind="net", net=net)
        pairings, stats = [], {}
        refs = []
        for anchor in anchors:
            pairing = play_pairing(make_env, entrant, anchor,
                                   seeds=block_seeds, mode=mode)
            pairings.append(pairing)
            # Keyed by the name the environment reported, not by the name in
            # the argument list. The two are the same only when the caller
            # built what it said it built, and a label that cannot be wrong
            # is worth more than one that usually is not.
            label = pairing.forward.red
            stats[f"ladder_score_{label}"] = pairing.score
            refs.append(f"{label}@{anchor.ref}")

        fitted = fit_ratings(
            pairings, pinned={a.name: float(ratings.get(a.name, 0.0))
                              for a in anchors})
        elo = fitted["policy"].elo if "policy" in fitted else 0.0
        stats.update({
            "ladder_elo": float(elo),
            "ladder_elo_error": float(fitted["policy"].error)
            if "policy" in fitted else float("inf"),
            # The whole design in one key: an expert-relative number produced
            # without ever playing the expert. Absent, rather than zero, when
            # the expert has no offline rating -- a missing anchor reported as
            # 0 reads as "level with the expert".
            **({"ladder_elo_vs_expert": float(elo) - float(ratings[expert])}
               if expert in ratings else {}),
            "ladder_mode": mode,
            "ladder_opponent": "ladder",
            "eval_opponent": "ladder",
            "ladder_opponent_ref": "|".join(refs),
            "ladder_episodes": int(episodes),
            "eval_episodes": int(episodes),
            "eval_block": int(block),
            "eval_blocks": int(max(1, blocks)),
        })
        return check_lift_is_named(stats)

    return probe
