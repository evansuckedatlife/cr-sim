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

    def __len__(self) -> int:
        return len(self.action)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            grid=self.grid, vector=self.vector, mask=self.mask,
            action=self.action, value=self.value,
            episodes=self.episodes, play_rate=self.play_rate)
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
            target=raw["target"] if "target" in raw.files else None)


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
) -> Demonstrations:
    """Watch an expert play and write down every decision it faced.

    Only states where a real choice existed are kept. A state with one legal
    action teaches nothing -- the expert had no alternative, so copying it
    conveys no preference -- and including them would let the pass action
    dominate the dataset, since a player is broke for most of a match.
    """
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
        start = len(actions)

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
                total += 1
                played += int(choice[0] != slots - 1)

                # What the search thought of everything it looked at. A low
                # temperature keeps this close to "the best few", while still
                # telling the policy that several placements were nearly as
                # good -- which a single label cannot say, and which is most
                # of the signal when tiles are near-equivalent.
                scores = list(getattr(expert, "last_scores", None)
                              or getattr(getattr(expert, "bot", None),
                                         "last_scores", []) or [])
                row = np.zeros(len(flat), dtype=np.float32)
                if scores:
                    indices = np.array([i for i, _ in scores])
                    raw = np.array([v for _, v in scores], dtype=np.float64)
                    # Scaled by this position's own spread, not by a fixed
                    # constant. How much placements differ varies enormously:
                    # with a quiet board they are all worth about the same,
                    # and mid-push they are not. A fixed temperature produced
                    # a target with 8% of its mass on the best action out of
                    # fifteen -- barely distinguishable from uniform, and so
                    # carrying almost no signal.
                    spread = float(raw.std())
                    scale = max(1e-6, target_temperature * spread) if spread > 1e-9                         else 1e-6
                    weights = np.exp((raw - raw.max()) / scale)
                    row[indices] = (weights / weights.sum()).astype(np.float32)
                else:
                    row[index] = 1.0
                targets.append(row)
            observation, reward, terminated, truncated, _ = env.step(choice)
            rewards.append(float(reward))
            if terminated or truncated:
                break

        # Discounted return to the end of the episode, walked backwards. Only
        # the states that were kept get one, so the two arrays stay aligned.
        running = 0.0
        tail: list[float] = []
        for reward in reversed(rewards):
            running = reward + gamma * running
            tail.append(running)
        tail.reverse()
        kept = len(actions) - start
        step = max(1, len(tail) // max(1, kept))
        values.extend(tail[i * step] if i * step < len(tail) else 0.0
                      for i in range(kept))
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
    )


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
