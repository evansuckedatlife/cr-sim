"""Learn to play by copying something that already can.

The order every successful game agent used, and the one this project skipped.
AlphaStar trained on 971,000 human replays by supervised learning *before* any
reinforcement learning, and that supervised agent alone outranked 84% of human
players; the reinforcement learning that followed was refinement on top of a
competent policy rather than the thing that created one. The small-compute
literature does the same with rule-based play instead of human data, sometimes
from only a couple of hundred episodes.

Starting from random initialisation is what failed here. A policy that has
never seen a good move has to stumble onto one, and a Clash Royale push is a
*conjunction* -- tank first, support behind, timed, on one lane -- that random
placement essentially never produces. No reward can reinforce behaviour that
never occurs, which is why four runs improved nothing and a 300-battle
evaluation could not tell a trained network from an untrained one.

There are no human replays for this game, so the expert is
:mod:`cr_sim.train.scripted`: a bot that searches the simulator rather than
knowing anything. It is not good. It only has to be better than noise, because
its job is to put the policy somewhere reinforcement learning can improve from.

Two heads are trained, not one. The policy learns which action the expert
took; the value head learns what the position was worth. Skipping the second
would hand reinforcement learning a critic that predicts nothing, and a critic
that predicts nothing is exactly what stalled every run so far -- explained
variance sat at 0.00 while the advantages it produced were noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

__all__ = ["Demonstrations", "collect", "clone", "CloneConfig"]


@dataclass(slots=True)
class Demonstrations:
    """What an expert did, and what it was worth."""

    grid: np.ndarray
    vector: np.ndarray
    mask: np.ndarray
    action: np.ndarray
    #: Discounted return from each state to the end of its episode. The value
    #: head's target, so reinforcement learning inherits a critic rather than
    #: starting one from scratch.
    value: np.ndarray
    #: What the search believed about every action it evaluated, as a
    #: distribution over the action space. Mostly zeros: only the handful of
    #: placements actually looked at carry mass.
    #:
    #: This rather than the chosen action, because the choice is not a
    #: function of the state -- candidates are sampled, so an identical board
    #: can produce a different move. The scores are, and they say which
    #: placements the search rated highly rather than merely which one won.
    target: np.ndarray | None = None
    episodes: int = 0
    #: How often the expert chose to play rather than pass. Worth recording:
    #: an expert that mostly passes teaches a policy to mostly pass, and that
    #: is a comfortable local optimum this environment rewards.
    play_rate: float = 0.0
    #: The encoding these grids are in, by the name
    #: :func:`~cr_sim.api.encoding.parse_observation` accepts.
    #:
    #: Recorded because ``clone_policy --observation`` was a *declaration*
    #: about a file rather than a fact read from it. Most mismatches happen to
    #: crash on the channel count, but two variants with the same width and
    #: different meaning train quietly and stamp the wrong name onto the
    #: checkpoint -- and from there the run's ``check_observation`` agrees with
    #: it, because it is comparing a shape. Empty means a shard written before
    #: this field existed, which is not the same as "v1" and must not be
    #: silently treated as it.
    observation: str = ""
    #: Which reward the ``value`` column was harvested under.
    #:
    #: The value head is the part of the clone reinforcement learning
    #: inherits, so a critic trained on one reward and fine-tuned under
    #: another arrives predicting the wrong quantity: this set was collected
    #: under the simple shaped reward while every fine-tune ran ``projected``,
    #: and the arriving critic predicted +1.48 where returns averaged +0.47.
    reward: str = ""

    def __len__(self) -> int:
        return len(self.action)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            grid=self.grid, vector=self.vector, mask=self.mask,
            action=self.action, value=self.value,
            episodes=self.episodes, play_rate=self.play_rate,
            observation=self.observation, reward=self.reward)
        if self.target is not None:
            payload["target"] = self.target
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path) -> "Demonstrations":
        raw = np.load(path)
        return cls(
            grid=raw["grid"], vector=raw["vector"], mask=raw["mask"],
            action=raw["action"], value=raw["value"],
            episodes=int(raw["episodes"]), play_rate=float(raw["play_rate"]),
            target=raw["target"] if "target" in raw.files else None,
            # str() through numpy's 0-d unicode array. Absent on any shard
            # written before provenance existed, and left empty there rather
            # than guessed at.
            observation=(str(raw["observation"])
                         if "observation" in raw.files else ""),
            reward=(str(raw["reward"]) if "reward" in raw.files else ""))



def _expert_patience(expert) -> float:
    """The margin a play had to beat waiting by for the expert to take it.

    Recorded into the target because the target has to be over the same
    numbers the decision was made from. Without it a state where waiting was
    chosen -- because nothing beat it by the margin -- produces a target whose
    largest entry is a placement, which is the opposite of what happened.
    """
    bot = getattr(expert, "bot", expert)
    return float(getattr(getattr(bot, "config", None), "patience", 0.0) or 0.0)


def _target_row(scores, index, width, height, slots, patience, temperature,
                min_spread, size):
    """The search's beliefs about one decision, as a distribution.

    Falls back to the action actually taken in the two cases where the scores
    say nothing: when there are none, and when they are all within
    ``min_spread`` of each other. See :func:`collect`'s ``min_spread`` for the
    measurement behind the second.
    """
    row = np.zeros(size, dtype=np.float32)
    if not scores:
        row[index] = 1.0
        return row
    indices = np.array([i for i, _ in scores])
    raw = np.array([v for _, v in scores], dtype=np.float64)
    # Waiting is scored the way the bot scored it: with its patience margin.
    raw = raw + patience * (indices == (slots - 1) * width * height)
    spread = float(raw.std())
    if spread < min_spread:
        row[index] = 1.0
        return row
    scale = max(1e-6, temperature * spread)
    weights = np.exp((raw - raw.max()) / scale)
    row[indices] = (weights / weights.sum()).astype(np.float32)
    return row


def collect(
    make_env: Callable[[int], Any],
    make_expert: Callable[[Any], Callable],
    episodes: int = 200,
    gamma: float = 0.997,
    on_episode: Callable[[int, int], None] | None = None,
    #: As a multiple of the candidate values' own standard deviation.
    #: At 0.35 the best action takes roughly half the mass while the
    #: near-equivalent ones keep enough to say they were nearly as good.
    target_temperature: float = 0.35,
    #: Below this spread in the search's candidate values, the search had no
    #: preference at all and the only signal left is which action it took.
    #:
    #: Not a guess. Measured over 120 real decisions: on the ones where the
    #: bot chose to wait the candidate values had a standard deviation of
    #: 0.00014, and on the ones where it played, 0.10 -- three orders of
    #: magnitude apart, with nothing in between. The softmax below is scaled
    #: by that same spread, so it is scale-free and turns a set of values that
    #: are equal to four decimal places into a *confident* preference for
    #: whichever arbitrary placement happened to round highest. Where they are
    #: exactly equal it produces a uniform distribution over the fifteen-odd
    #: candidates, of which fourteen are placements and one is waiting.
    #:
    #: The damage that did: 86% of the states where the expert waited carried
    #: an exactly uniform target, and the pass action was the target's argmax
    #: in *none* of 10,940 recorded decisions. A policy trained on it played a
    #: card at every single decision, against the expert's 56%.
    min_spread: float = 1e-3,
    variants: "dict[str, Any] | None" = None,
    #: Stamped onto every set this returns, so the file on disk states what it
    #: is instead of relying on whoever loads it to declare correctly. With
    #: ``variants`` the observation name comes from the variant's own key.
    #:
    #: Both are ``_name`` on purpose. This function's own loop binds
    #: ``observation`` and ``reward`` once per step -- ``observation, reward,
    #: terminated, truncated, _ = env.step(choice)`` -- so a parameter by
    #: either name is silently overwritten before it is ever read. The first
    #: version of this stamped 0.0 into every shard's reward and the last
    #: observation *dict* into its encoding name, which numpy then wrote as an
    #: object array that would not load back without allow_pickle.
    reward_name: str = "",
    observation_name: str = "v1",
):
    """Watch an expert play and write down every decision it faced.

    Only states where a real choice existed are kept. A state with one legal
    action teaches nothing -- the expert had no alternative, so copying it
    conveys no preference -- and including them would let the pass action
    dominate the dataset, since a player is broke for most of a match.

    ``variants`` maps a name to an
    :class:`~cr_sim.api.encoding.ObservationFeatures`, and makes this return a
    dict of one :class:`Demonstrations` per name instead of a single set. The
    expert reads the *battle*, not the observation, so every variant sees the
    identical trajectory and the identical decisions -- which is the only way
    an observation ablation is a comparison rather than two experiments. It is
    also the only affordable way: a demonstration set costs seventeen seconds
    a battle to produce, and collecting one per encoding would multiply that
    by the number of things being compared.
    """
    if variants:
        return _collect_variants(make_env, make_expert, episodes, gamma,
                                 on_episode, target_temperature, min_spread,
                                 variants, reward_name)
    grids: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    values: list[float] = []
    targets: list[np.ndarray] = []
    played = total = 0

    for episode in range(episodes):
        env = make_env(episode)
        observation, _ = env.reset(seed=episode)
        expert = make_expert(env)
        rewards: list[float] = []
        # The env step each kept decision was made at; see the returns below.
        decisions: list[int] = []

        while True:
            mask = env.legal_action_mask()
            flat = mask.reshape(-1)
            if not flat.any():
                break
            choice = expert(observation, mask, env.battle)
            if int(flat.sum()) > 1:
                slots, width, height = (int(v) for v in env.action_space.nvec)
                index = (int(choice[0]) * width * height
                         + int(choice[1]) * height + int(choice[2]))
                grids.append(observation["grid"])
                vectors.append(observation["vector"])
                masks.append(flat.copy())
                actions.append(index)
                # rewards holds one entry per step already taken, so the next
                # one appended -- the reward for leaving this state -- is at
                # this index, and so is this state's own return.
                decisions.append(len(rewards))
                total += 1
                played += int(choice[0] != slots - 1)

                # What the search thought of everything it looked at. A low
                # temperature keeps this close to "the best few", while still
                # telling the policy that several placements were nearly as
                # good -- which a single label cannot say, and which is most
                # of the signal when tiles are near-equivalent. Scaled by this
                # position's own spread, because how much placements differ
                # varies enormously; floored by min_spread, because below that
                # they do not differ at all.
                scores = list(getattr(expert, "last_scores", None)
                              or getattr(getattr(expert, "bot", None),
                                         "last_scores", []) or [])
                targets.append(_target_row(
                    scores, index, width, height, slots,
                    _expert_patience(expert), target_temperature, min_spread,
                    len(flat)))
            observation, reward, terminated, truncated, _ = env.step(choice)
            rewards.append(float(reward))
            if terminated or truncated:
                break

        # Discounted return to the end of the episode, walked backwards.
        running = 0.0
        tail: list[float] = []
        for reward in reversed(rewards):
            running = reward + gamma * running
            tail.append(running)
        tail.reverse()
        # Each kept state takes the return from its own step. Kept decisions
        # are sparse and unevenly spaced -- only states with more than one
        # legal action are recorded, and how long a player stays broke varies
        # -- so walking them at an even stride, which is what this did, hands
        # the value head the return of a nearby but different position.
        #
        # Every recorded decision is followed by exactly one env.step, so the
        # index is always inside tail.
        values.extend(tail[t] for t in decisions)
        if on_episode is not None:
            on_episode(episode + 1, len(actions))

    return Demonstrations(
        grid=np.asarray(grids, dtype=np.float32),
        vector=np.asarray(vectors, dtype=np.float32),
        mask=np.asarray(masks, dtype=bool),
        action=np.asarray(actions, dtype=np.int64),
        value=np.asarray(values, dtype=np.float32),
        episodes=episodes,
        play_rate=(played / total) if total else 0.0,
        target=np.asarray(targets, dtype=np.float32) if targets else None,
        observation=observation_name,
        reward=reward_name,
    )



def _collect_variants(make_env, make_expert, episodes, gamma, on_episode,
                      target_temperature, min_spread, variants,
                      reward_name=""):
    """:func:`collect`, recording every observation variant off one playthrough.

    Implemented by re-encoding the live battle once per variant at each
    decision the plain path would have kept, rather than by replaying the
    episode. The engine is deterministic and a replay would produce the same
    states, but it would also cost the same seventeen seconds a battle again
    for every variant compared.
    """
    from ..api.encoding import build_encoding_config, encode_observation

    out: dict[str, dict[str, list]] = {
        name: {"grid": [], "vector": []} for name in variants}
    masks: list = []
    actions: list[int] = []
    values: list[float] = []
    targets: list = []
    played = total = 0

    for episode in range(episodes):
        env = make_env(episode)
        observation, _ = env.reset(seed=episode)
        expert = make_expert(env)
        configs = {
            name: build_encoding_config(
                env.battle.arena, env.blue_deck, env.red_deck, features)
            for name, features in variants.items()
        }
        rewards: list[float] = []
        # The env step each kept decision was made at; see the returns below.
        decisions: list[int] = []

        while True:
            mask = env.legal_action_mask()
            flat = mask.reshape(-1)
            if not flat.any():
                break
            choice = expert(observation, mask, env.battle)
            if int(flat.sum()) > 1:
                slots, width, height = (int(v) for v in env.action_space.nvec)
                index = (int(choice[0]) * width * height
                         + int(choice[1]) * height + int(choice[2]))
                for name, config in configs.items():
                    encoded = encode_observation(
                        env.battle, env.team, env.registry, config)
                    out[name]["grid"].append(encoded["grid"])
                    out[name]["vector"].append(encoded["vector"])
                masks.append(flat.copy())
                actions.append(index)
                # rewards holds one entry per step already taken, so the next
                # one appended -- the reward for leaving this state -- is at
                # this index, and so is this state's own return.
                decisions.append(len(rewards))
                total += 1
                played += int(choice[0] != slots - 1)
                scores = list(getattr(expert, "last_scores", None)
                              or getattr(getattr(expert, "bot", None),
                                         "last_scores", []) or [])
                targets.append(_target_row(
                    scores, index, width, height, slots,
                    _expert_patience(expert), target_temperature, min_spread,
                    len(flat)))
            observation, reward, terminated, truncated, _ = env.step(choice)
            rewards.append(float(reward))
            if terminated or truncated:
                break

        running = 0.0
        tail: list[float] = []
        for reward in reversed(rewards):
            running = reward + gamma * running
            tail.append(running)
        tail.reverse()
        # By each decision's own step, not an even stride; see collect.
        values.extend(tail[t] for t in decisions)
        if on_episode is not None:
            on_episode(episode + 1, len(actions))

    shared = dict(
        mask=np.asarray(masks, dtype=bool),
        action=np.asarray(actions, dtype=np.int64),
        value=np.asarray(values, dtype=np.float32),
        episodes=episodes,
        play_rate=(played / total) if total else 0.0,
        target=np.asarray(targets, dtype=np.float32) if targets else None,
        reward=reward_name,
    )
    return {
        name: Demonstrations(
            grid=np.asarray(parts["grid"], dtype=np.float32),
            vector=np.asarray(parts["vector"], dtype=np.float32),
            # The variant's own key, not the caller's default: this is the one
            # place that knows which encoding produced these grids.
            observation=name,
            **shared,
        )
        for name, parts in out.items()
    }


@dataclass(slots=True)
class CloneConfig:
    epochs: int = 8
    batch_size: int = 256
    learning_rate: float = 3e-4
    #: Weight on the value head's regression against the policy's
    #: cross-entropy. The critic matters as much as the policy here: handing
    #: reinforcement learning a value function that predicts nothing is what
    #: made every previous run's advantages noise.
    value_coefficient: float = 0.5
    #: Held back to measure whether the policy learned the expert or memorised
    #: it. Without this the training loss falls either way and says nothing.
    holdout: float = 0.1
    #: How much a "pass" decision counts against a placement.
    #:
    #: Nearly half the expert's decisions are to play nothing, while the other
    #: half are spread over some seven hundred distinct placements -- so each
    #: individual placement carries under a tenth of a percent of the mass and
    #: cross-entropy parks its mode on passing. Measured: agreement froze at
    #: exactly the pass fraction for fourteen epochs, and the resulting policy
    #: lost 100% of its matches when played greedily, because its argmax was
    #: always "do nothing" and the opponent walked in unopposed.
    #:
    #: Down-weighting passes forces the policy to learn *where* to play. When
    #: to play is a far easier thing to recover, and sampling handles it.
    pass_weight: float = 0.1
    #: Which action index means "play nothing". Set from the environment.
    pass_action: int = -1
    seed: int = 0


def clone(net, data: Demonstrations, config: CloneConfig | None = None,
          on_epoch: Callable[[dict], None] | None = None):
    """Train ``net`` to reproduce the expert's choices, in place."""
    import torch
    import torch.nn as nn

    config = config or CloneConfig()
    torch.manual_seed(config.seed)
    optimiser = torch.optim.Adam(
        net.parameters(), lr=config.learning_rate, foreach=True)

    grid = torch.from_numpy(data.grid)
    vector = torch.from_numpy(data.vector)
    mask = torch.from_numpy(data.mask)
    action = torch.from_numpy(data.action)
    value = torch.from_numpy(data.value)
    soft = torch.from_numpy(data.target) if data.target is not None else None

    # A pass counts for less than a placement; see CloneConfig.pass_weight.
    weights = torch.ones(len(data))
    if config.pass_action >= 0:
        weights[action == config.pass_action] = config.pass_weight

    count = len(data)
    order = torch.randperm(count, generator=torch.Generator().manual_seed(config.seed))
    cut = int(count * (1.0 - config.holdout))
    train_idx, test_idx = order[:cut], order[cut:]

    for epoch in range(config.epochs):
        net.train()
        shuffled = train_idx[torch.randperm(len(train_idx))]
        losses = []
        for start in range(0, len(shuffled), config.batch_size):
            batch = shuffled[start:start + config.batch_size]
            logits, predicted = net(grid[batch], vector[batch], mask[batch])
            if soft is not None:
                # Cross-entropy against the search's own distribution. The
                # single chosen move is not a function of the state; these
                # scores are.
                log_probs = nn.functional.log_softmax(logits, dim=-1)
                per_sample = -(soft[batch] * log_probs).sum(dim=-1)
            else:
                per_sample = nn.functional.cross_entropy(
                    logits, action[batch], reduction="none")
            policy_loss = (per_sample * weights[batch]).sum() / weights[batch].sum()
            value_loss = nn.functional.mse_loss(predicted, value[batch])
            loss = policy_loss + config.value_coefficient * value_loss
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimiser.step()
            losses.append((policy_loss.item(), value_loss.item()))

        net.eval()
        with torch.no_grad():
            logits, predicted = net(grid[test_idx], vector[test_idx], mask[test_idx])
            picked = logits.argmax(dim=-1)
            truth = action[test_idx]
            agreement = (picked == truth).float().mean().item()
            spread = value[test_idx].var().item()
            explained = (
                1.0 - (value[test_idx] - predicted).var().item() / spread
                if spread > 1e-9 else float("nan")
            )
            # Overall agreement is dominated by the pass class and can sit at
            # exactly the pass fraction while the policy has learned nothing
            # about placement -- which it did, for fourteen epochs. These two
            # separate "knows when to play" from "knows where to".
            if config.pass_action >= 0:
                plays = truth != config.pass_action
                on_plays = ((picked == truth) & plays).float().sum().item()
                play_agreement = on_plays / max(1, int(plays.sum()))
                play_rate = (picked != config.pass_action).float().mean().item()
            else:
                play_agreement = float("nan")
                play_rate = float("nan")
        stats = {
            "epoch": epoch + 1,
            "policy_loss": float(np.mean([p for p, _ in losses])),
            "value_loss": float(np.mean([v for _, v in losses])),
            # On held-out states, so it says whether the expert was learned
            # rather than memorised.
            "agreement": agreement,
            # The number that matters: of the states where the expert played a
            # card, how often the policy picks the same one.
            "play_agreement": play_agreement,
            # How often the policy's own argmax is a placement rather than a
            # pass. At zero the greedy policy never plays and loses everything.
            "play_rate": play_rate,
            "explained_variance": explained,
        }
        if on_epoch is not None:
            on_epoch(stats)
    return net
