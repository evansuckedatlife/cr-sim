"""PPO against the battle simulator.

Chosen over a value-based method for one reason that is specific to this game:
the action space is 720 wide and the *legal* subset changes every decision, as
elixir accrues and the hand cycles. A policy-gradient method takes that mask
natively -- illegal actions get zero probability and zero gradient -- whereas
Q-learning would have to either learn values for actions it can never take or
carry the same masking through a target network as well.

Two things here are not stock PPO and are worth saying why.

**Rollouts are collected per environment with the mask stored alongside.** The
mask at decision time is part of the transition, not something recoverable
later: by the time an update runs, that battle has moved on and the elixir and
hand that made an action legal are gone. Recomputing it would score the stored
action against a different action set than the one it was sampled from, which
quietly corrupts the ratio PPO is built on.

**Reward is already a difference.** The environment returns the change in
(crowns + shaped tower health) each step, so an episode's rewards telescope to
the final crown difference. That means the discount is doing much less work
than it usually does, and a value function that is merely adequate still gives
usable advantages.

This module is deliberately dependency-light: torch and numpy, no RL framework.
The environment is not a standard Gymnasium install here (there is a shim), and
wiring a framework around a shimmed env is more fragile than writing the ~200
lines of PPO that are actually needed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..api.encoding import NOOP_SLOT
from ..api.env import CRSimEnv
from .nets import ActorCritic, NetConfig, net_config_for

__all__ = ["PPOConfig", "Rollout", "train", "compute_gae"]

#: Episodes averaged into the reported return and win rate. Wide because the
#: per-episode spread is larger than any effect worth seeing, and a narrow
#: window turns that spread into a curve that looks like learning.
_RETURN_WINDOW = 200


@dataclass(frozen=True, slots=True)
class PPOConfig:
    total_steps: int = 100_000
    #: Decisions collected per environment before each update.
    horizon: int = 256
    num_envs: int = 4
    epochs: int = 4
    minibatches: int = 4
    learning_rate: float = 3e-4
    #: Learning rate for the value side. ``None`` uses ``learning_rate``.
    #:
    #: Worth raising above the actor's: the critic is solving a plain
    #: regression with a stationary-ish target, while the actor is walking a
    #: trust region and is the reason the rate is small in the first place.
    #: Tying them holds the critic back for the actor's benefit.
    value_learning_rate: "float | None" = 1e-3
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_coefficient: float = 0.2
    value_coefficient: float = 0.5
    #: Entropy bonus. Higher than the usual 0.01 because the legal action set
    #: is large and a placement policy collapses onto one tile early if it is
    #: not pushed to keep exploring where to put things.
    entropy_coefficient: float = 0.02
    max_grad_norm: float = 0.5
    #: Which policy head to build; see :class:`~cr_sim.train.nets.NetConfig`.
    head: str = "flat"
    #: Weight on ``KL(reference || policy)``, the standard trust region for
    #: fine-tuning a policy that already plays.
    #:
    #: Zero is plain PPO. Above zero the loss carries a pull back toward a
    #: frozen anchor -- in practice the behavioural clone the run started
    #: from. The direction is deliberate: ``KL(reference || policy)`` is the
    #: one that punishes *dropping* what the reference does, because its
    #: expectation is taken under the reference. The reverse direction is
    #: mode-seeking and happily lets the policy collapse onto one action, and
    #: collapsing onto one action -- passing -- is precisely this
    #: environment's failure mode. It is also the direction AlphaStar used
    #: against its supervised agent for the whole of its league training.
    kl_coefficient: float = 0.0
    seed: int = 0
    log_every: int = 1


@dataclass(slots=True)
class Rollout:
    """One batch of transitions, with the masks that produced them."""

    grid: torch.Tensor
    vector: torch.Tensor
    mask: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    advantage: torch.Tensor = field(default=None)  # type: ignore[assignment]
    ret: torch.Tensor = field(default=None)  # type: ignore[assignment]


def compute_gae(
    reward: torch.Tensor,
    value: torch.Tensor,
    done: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over ``(horizon, num_envs)`` tensors.

    ``done`` masks the bootstrap across an episode boundary. Without it the
    advantage at the last step of a won match would be carried into the first
    step of the next one, teaching the policy that whatever it happened to do
    on the opening tick was worth a crown.
    """
    horizon = reward.shape[0]
    advantage = torch.zeros_like(reward)
    running = torch.zeros_like(last_value)
    for step in reversed(range(horizon)):
        if step == horizon - 1:
            next_value = last_value
        else:
            next_value = value[step + 1]
        not_done = 1.0 - done[step]
        delta = reward[step] + gamma * next_value * not_done - value[step]
        running = delta + gamma * lam * not_done * running
        advantage[step] = running
    return advantage, advantage + value


def _stack_obs(observations: Sequence[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    grid = torch.from_numpy(np.stack([o["grid"] for o in observations]))
    vector = torch.from_numpy(np.stack([o["vector"] for o in observations]))
    return grid, vector


def _flat_mask(masks: Sequence[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack([m.reshape(-1) for m in masks]))


def _unflatten_action(index: int, nvec: Sequence[int]) -> tuple[int, int, int]:
    """Turn a flat categorical sample back into ``(slot, x, y)``.

    The inverse of the C-order flatten used on the mask, so the two agree axis
    for axis. Getting this wrong transposes placements without erroring.
    """
    slots, width, height = int(nvec[0]), int(nvec[1]), int(nvec[2])
    slot, remainder = divmod(int(index), width * height)
    gx, gy = divmod(remainder, height)
    assert slot < slots
    return slot, gx, gy


def train(
    make_env: Callable[[int], CRSimEnv],
    config: PPOConfig = PPOConfig(),
    *,
    device: str = "cpu",
    on_update: Callable[[dict], None] | None = None,
    on_net: Callable[[ActorCritic], None] | None = None,
    on_optimiser: "Callable[[Any], None] | None" = None,
    opponents: "Sequence[Any] | None" = None,
    refresh_every: int = 0,
    on_refresh: "Callable[[ActorCritic, int], None] | None" = None,
    resume: "dict | None" = None,
    parallel: "Any | None" = None,
    reference: "ActorCritic | None" = None,
) -> ActorCritic:
    """Run PPO and return the trained network.

    ``make_env`` takes an index so each environment can be seeded differently;
    identical seeds across the batch would collect the same battle several
    times over and report a misleadingly smooth return.

    ``opponents`` are the frozen policy snapshots facing the learner, one per
    environment. Refreshed every ``refresh_every`` updates so the thing being
    beaten improves alongside the thing beating it; at zero they never change,
    which makes the opponent a fixed sparring partner instead of self-play.

    ``parallel`` is a :class:`~cr_sim.api.vec.CRSimVecEnv` to collect rollouts
    through instead of stepping local environments one at a time. About 90% of
    a decision here is simulating the battle, so spreading that over processes
    is most of the throughput available; the network still runs once per batch
    in this process. ``make_env`` is still called once, for the observation
    and action shapes, and is not stepped.

    ``resume`` restarts a run that stopped: a checkpoint holding the network
    weights, the optimiser state and how many steps had been taken. The
    optimiser state is not optional -- Adam's moment estimates are most of what
    a long run has learned about its own gradients, and restarting without them
    throws away that adaptation and destabilises the first updates after the
    restart, which looks exactly like a bad checkpoint.

    ``on_net`` is handed the network as soon as it is built. Its shapes come
    from the first observation, so a caller that wants to checkpoint mid-run
    cannot construct it in advance and would otherwise have nothing to save
    until the whole job returns.
    """
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    # One local environment either way: its spaces define the network's
    # shapes, and building it is cheap next to a rollout.
    probe = make_env(0)
    nvec = [int(v) for v in probe.action_space.nvec]
    num_actions = int(np.prod(nvec))

    if parallel is not None:
        envs = []
        observations, masks = parallel.reset(
            [config.seed * 1000 + i for i in range(config.num_envs)]
        )
    else:
        envs = [probe] + [make_env(i) for i in range(1, config.num_envs)]
        observations = []
        for index, env in enumerate(envs):
            obs, _ = env.reset(seed=config.seed * 1000 + index)
            observations.append(obs)
        masks = [env.legal_action_mask() for env in envs]

    net = ActorCritic(net_config_for(probe, head=config.head)).to(device)
    if config.value_learning_rate is None:
        # foreach batches the per-parameter updates into a handful of fused
        # kernels rather than four to six per tensor. On CPU it is a modest
        # win; on an accelerator it is the difference between about a hundred
        # and fifty launches per step and about six.
        optimiser = torch.optim.Adam(
            net.parameters(), lr=config.learning_rate, eps=1e-5, foreach=True)
    else:
        # Two groups by identity, not by name: with a separate critic the two
        # sets are genuinely disjoint, and with a shared trunk this correctly
        # leaves only the value head on the faster rate.
        critic = net.critic_parameters()
        critic_ids = {id(p) for p in critic}
        actor = [p for p in net.parameters() if id(p) not in critic_ids]
        optimiser = torch.optim.Adam(
            [
                {"params": actor, "lr": config.learning_rate},
                {"params": critic, "lr": config.value_learning_rate},
            ],
            eps=1e-5,
            foreach=True,
        )

    resumed_steps = resumed_updates = 0
    if resume is not None:
        net.load_state_dict(resume["state_dict"])
        if resume.get("optimiser") is not None:
            optimiser.load_state_dict(resume["optimiser"])
        resumed_steps = int(resume.get("steps", 0))
        resumed_updates = int(resume.get("updates", 0))

    # The anchor for the trust region, taken *after* the resumed weights are
    # loaded so it is the policy the run actually starts from rather than the
    # random initialisation that briefly preceded it. Frozen and on the CPU:
    # it is only ever read.
    anchor = None
    if config.kl_coefficient > 0.0:
        anchor = reference
        if anchor is None:
            import copy as _copy

            anchor = _copy.deepcopy(net)
        anchor = anchor.to(device)
        anchor.eval()
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)

    if on_net is not None:
        on_net(net)
    # Handed out for the same reason as the network: a checkpoint that a run
    # can actually restart from needs the optimiser state, and only this
    # function owns it.
    if on_optimiser is not None:
        on_optimiser(optimiser)

    steps_done = resumed_steps
    episode_returns: list[float] = []
    episode_crowns: list[int] = []
    running_return = np.zeros(config.num_envs, dtype=np.float64)
    started = time.perf_counter()
    update_index = resumed_updates

    while steps_done < config.total_steps:
        buffers: dict[str, list[torch.Tensor]] = {
            key: [] for key in ("grid", "vector", "mask", "action", "log_prob", "value", "reward", "done")
        }

        for _ in range(config.horizon):
            grid, vector = _stack_obs(observations)
            mask = _flat_mask(masks)
            action, log_prob, value = net.act(grid.to(device), vector.to(device), mask.to(device))

            decoded = [_unflatten_action(int(a), nvec) for a in action]
            if parallel is not None:
                # Terminal episodes were reset inside their worker, so the
                # observations coming back already belong to the next one and
                # the crown difference of the finished episode arrives with
                # them.
                observations, rewards, dones, crowns, masks = parallel.step(decoded)
                running_return += rewards
                for index in np.flatnonzero(dones):
                    episode_returns.append(float(running_return[index]))
                    episode_crowns.append(int(crowns[index]))
                    running_return[index] = 0.0
            else:
                rewards = np.zeros(config.num_envs, dtype=np.float32)
                dones = np.zeros(config.num_envs, dtype=np.float32)
                for index, env in enumerate(envs):
                    obs, reward, terminated, truncated, _ = env.step(decoded[index])
                    rewards[index] = reward
                    running_return[index] += reward
                    if terminated or truncated:
                        dones[index] = 1.0
                        episode_returns.append(float(running_return[index]))
                        episode_crowns.append(
                            env.battle.players[env.team].crowns
                            - env.battle.players[env.team.opponent].crowns
                        )
                        running_return[index] = 0.0
                        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                    observations[index] = obs
                    masks[index] = env.legal_action_mask()

            buffers["grid"].append(grid)
            buffers["vector"].append(vector)
            buffers["mask"].append(mask)
            buffers["action"].append(action.cpu())
            buffers["log_prob"].append(log_prob.cpu())
            buffers["value"].append(value.cpu())
            buffers["reward"].append(torch.from_numpy(rewards))
            buffers["done"].append(torch.from_numpy(dones))
            steps_done += config.num_envs

        rollout = Rollout(**{key: torch.stack(value) for key, value in buffers.items()})
        with torch.no_grad():
            grid, vector = _stack_obs(observations)
            _, last_value = net(grid.to(device), vector.to(device), _flat_mask(masks).to(device))
        rollout.advantage, rollout.ret = compute_gae(
            rollout.reward, rollout.value, rollout.done,
            last_value.cpu(), config.gamma, config.gae_lambda,
        )

        stats = _update(net, optimiser, rollout, config, device, anchor)
        update_index += 1

        # The non-finite check lives in _update, on the gradient norm that
        # clip_grad_norm_ already computes. Sweeping every parameter here
        # instead was thirty pairs of tiny kernels each followed by a
        # blocking host readback, fired straight after the optimiser queued
        # a few hundred more -- which is precisely what exhausts a Level Zero
        # driver's handles, and it took down every GPU run attempted.

        if opponents and refresh_every and update_index % refresh_every == 0:
            # Fired before the opponents refresh, so a pool gains this
            # generation and then the opponents draw from a pool that
            # contains it. The other order leaves the pool one generation
            # behind for ever.
            if on_refresh is not None:
                on_refresh(net, update_index)
            # Every snapshot moves at once. Staggering them would have the
            # learner facing several generations in one batch, and the
            # advantage estimates would be averaging across opponents of
            # different strength.
            for opponent in opponents:
                opponent.refresh(net)

        # How often the policy chose to spend nothing. Worth watching rather
        # than inferring from the return: "always pass" is the comfortable
        # local optimum here, because passing is the only action that is never
        # punished, and a run that has collapsed into it looks stable on every
        # other metric.
        slot_size = nvec[1] * nvec[2]
        chosen_slots = rollout.action.reshape(-1) // slot_size
        stats["noop_fraction"] = float((chosen_slots == NOOP_SLOT).float().mean())

        # The scale of what the critic is asked to predict. Value loss is a
        # plain MSE against these, so it cannot be read at all without them:
        # a loss of 5 is a diverging critic if the targets have spread 0.7 and
        # an ordinary one if they have spread 3. Logged because that exact
        # ambiguity stalled a diagnosis -- a run showing value loss sixteen
        # times higher than its predecessors turned out to be unreadable
        # without knowing whether the targets had grown with it.
        stats["ret_mean"] = float(rollout.ret.mean())
        stats["ret_std"] = float(rollout.ret.std())

        # The scale-free version, and the one to actually watch. 0 means the
        # critic is no better than predicting the mean return, 1 means it
        # predicts perfectly, and negative means it is worse than a constant.
        # PPO's advantages are only as good as this: at 0 the advantage is the
        # Monte-Carlo return minus a constant, which is unbiased but so noisy
        # that the policy gradient is mostly variance.
        variance = float(rollout.ret.var())
        residual = float((rollout.ret - rollout.value).var())
        stats["explained_variance"] = (
            1.0 - residual / variance if variance > 1e-9 else float("nan")
        )

        elapsed = time.perf_counter() - started
        stats.update(
            steps=steps_done,
            updates=update_index,
            # Logged so a progress view can say how much is left. Without it
            # the only honest thing a page can show is a step count with no
            # denominator.
            total_steps=config.total_steps,
            elapsed_seconds=elapsed,
            # Measured over this leg only. Dividing a resumed total by the
            # time since restart reports a throughput no run ever achieved.
            steps_per_second=(steps_done - resumed_steps) / max(elapsed, 1e-9),
            episodes=len(episode_returns),
            # Averaged over a wide window on purpose. Episode returns here have
            # a standard deviation around 0.5, so a 20-episode mean carries a
            # standard error of 0.11 and swings by half a crown between updates
            # for reasons that have nothing to do with the policy. A window
            # that noisy reads as progress when nothing is happening -- it did,
            # reporting 0.78 for a policy that evaluated at 0.29.
            mean_return=float(np.mean(episode_returns[-_RETURN_WINDOW:]))
            if episode_returns else float("nan"),
            win_rate=float(np.mean([c > 0 for c in episode_crowns[-_RETURN_WINDOW:]]))
            if episode_crowns else float("nan"),
        )
        if on_update is not None:
            on_update(stats)
        elif update_index % config.log_every == 0:
            print(
                f"update {update_index:4d}  steps {steps_done:>8d}  "
                f"{stats['steps_per_second']:6.0f}/s  "
                f"return {stats['mean_return']:+7.3f}  "
                f"win {stats['win_rate']:4.0%}  "
                f"loss {stats['policy_loss']:+.4f}/{stats['value_loss']:.4f}  "
                f"entropy {stats['entropy']:.3f}  "
                f"pass {stats['noop_fraction']:.0%}"
            )

    for env in envs:
        env.close()
    return net


def _update(
    net: ActorCritic,
    optimiser: torch.optim.Optimizer,
    rollout: Rollout,
    config: PPOConfig,
    device: str,
    anchor: "ActorCritic | None" = None,
) -> dict:
    """One PPO update over the collected rollout.

    ``anchor`` is the frozen reference the trust region pulls back toward; see
    :attr:`PPOConfig.kl_coefficient`. ``None`` is plain PPO.
    """
    flat = {
        "grid": rollout.grid.reshape(-1, *rollout.grid.shape[2:]),
        "vector": rollout.vector.reshape(-1, rollout.vector.shape[-1]),
        "mask": rollout.mask.reshape(-1, rollout.mask.shape[-1]),
        "action": rollout.action.reshape(-1),
        "log_prob": rollout.log_prob.reshape(-1),
        "advantage": rollout.advantage.reshape(-1),
        "ret": rollout.ret.reshape(-1),
    }
    total = flat["action"].shape[0]
    batch_size = max(1, total // config.minibatches)
    indices = np.arange(total)

    policy_loss = value_loss = entropy_value = divergence_value = 0.0
    for _ in range(config.epochs):
        np.random.shuffle(indices)
        for start in range(0, total, batch_size):
            batch = torch.from_numpy(indices[start : start + batch_size])
            advantage = flat["advantage"][batch].to(device)
            # Normalised per minibatch: the shaped reward's scale drifts as
            # tower health does, and an unnormalised advantage makes the
            # effective learning rate drift with it.
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            log_prob, entropy, value = net.evaluate(
                flat["grid"][batch].to(device),
                flat["vector"][batch].to(device),
                flat["mask"][batch].to(device),
                flat["action"][batch].to(device),
            )
            ratio = (log_prob - flat["log_prob"][batch].to(device)).exp()
            unclipped = ratio * advantage
            clipped = ratio.clamp(1 - config.clip_coefficient, 1 + config.clip_coefficient) * advantage
            p_loss = -torch.min(unclipped, clipped).mean()
            v_loss = nn.functional.mse_loss(value, flat["ret"][batch].to(device))
            loss = (
                p_loss
                + config.value_coefficient * v_loss
                - config.entropy_coefficient * entropy.mean()
            )
            divergence = torch.zeros((), device=loss.device)
            if anchor is not None:
                grid_b = flat["grid"][batch].to(device)
                vector_b = flat["vector"][batch].to(device)
                mask_b = flat["mask"][batch].to(device)
                with torch.no_grad():
                    reference_logits, _ = anchor(grid_b, vector_b, mask_b)
                    reference_log = nn.functional.log_softmax(reference_logits, dim=-1)
                    reference_prob = reference_log.exp()
                current_logits, _ = net(grid_b, vector_b, mask_b)
                current_log = nn.functional.log_softmax(current_logits, dim=-1)
                # KL(reference || policy). Illegal actions carry exactly zero
                # reference probability, so they contribute nothing however
                # large the logit difference on them is.
                divergence = (reference_prob * (reference_log - current_log)).sum(-1).mean()
                loss = loss + config.kl_coefficient * divergence

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            # The norm is returned, not just applied. Checking it is free and
            # catches a non-finite gradient at the update that produced it --
            # which is the cause of non-finite weights, one step earlier.
            total_norm = nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm)
            if not torch.isfinite(total_norm):
                raise RuntimeError(
                    "gradients became non-finite. Weights after this step "
                    "would be NaN, every later rollout would sample from a "
                    "uniform distribution, and every checkpoint saved after "
                    "it would be worthless -- so the run stops here, with the "
                    "last good checkpoint still on disk."
                )
            optimiser.step()

            policy_loss, value_loss, entropy_value, divergence_value = (
                p_loss.item(), v_loss.item(), entropy.mean().item(),
                float(divergence.detach()),
            )

    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_value,
        # How far the policy has walked from the thing it started as. Worth
        # logging even at coefficient zero: a fine-tune that has drifted a
        # long way from a competent clone has usually not improved on it.
        "reference_kl": divergence_value,
    }
