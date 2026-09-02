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
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

from ..api.env import CRSimEnv
from ..data.cards import build_card_registry
from ..data.leveling import build_level_table
from ..data.source import LogicData
from .nets import ActorCritic, NetConfig
from .ppo import _unflatten_action
from .run import DEFAULT_BUILD, DEFAULT_DECK

__all__ = ["evaluate", "load_policy", "Result"]


class Result(dict):
    """Per-episode outcomes, summarised."""

    def summary(self, label: str) -> str:
        returns, crowns = self["returns"], self["crowns"]
        wins = sum(1 for c in crowns if c > 0) / max(1, len(crowns))
        return (
            f"{label:>10}: return {st.mean(returns):+.4f} +/- {st.pstdev(returns):.4f}  "
            f"crowns {st.mean(crowns):+.3f}  win {wins:.0%}  ({len(returns)} episodes)"
        )


def load_policy(checkpoint: Path, env: CRSimEnv) -> ActorCritic:
    """Rebuild the network from a checkpoint, using the env for its shapes.

    The shapes are not stored in the checkpoint because they are a property of
    the environment, not of the weights -- and taking them from the env is what
    makes a shape mismatch fail loudly here rather than silently score a policy
    against an observation it was never trained on.
    """
    observation, _ = env.reset(seed=0)
    net = ActorCritic(
        NetConfig(
            grid_channels=observation["grid"].shape[0],
            grid_height=observation["grid"].shape[1],
            grid_width=observation["grid"].shape[2],
            vector_size=observation["vector"].shape[0],
            num_actions=int(np.prod([int(v) for v in env.action_space.nvec])),
        )
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
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
) -> Result:
    """Play ``episodes`` matches. ``net=None`` plays uniformly at random.

    Both arms take the same seed list, so the comparison is over the same
    battles rather than over the same *number* of battles. With a per-episode
    spread wider than the effect being measured, paired seeds are what make a
    difference of this size readable at all.
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
                    device = next(net.parameters()).device
                    logits, _ = net(
                        torch.from_numpy(observation["grid"]).unsqueeze(0).to(device),
                        torch.from_numpy(observation["vector"]).unsqueeze(0).to(device),
                        torch.from_numpy(flat).unsqueeze(0).to(device),
                    )
                if greedy:
                    choice = int(logits.argmax(dim=-1))
                else:
                    choice = int(torch.distributions.Categorical(logits=logits).sample())
            observation, reward, terminated, truncated, info = env.step(
                _unflatten_action(choice, nvec)
            )
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
        crowns.append(info["blue_crowns"] - info["red_crowns"])

    return Result(returns=returns, crowns=crowns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cr-sim-eval")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--tps", type=int, default=20)
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--match-seconds", type=int, default=120)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--sample", action="store_true", help="sample instead of argmax")
    args = parser.parse_args(argv)

    data = LogicData.load(args.build)
    levels = build_level_table(data)
    registry = build_card_registry(data)

    def make_env() -> CRSimEnv:
        return CRSimEnv(
            data, levels, registry, DEFAULT_DECK, DEFAULT_DECK,
            ticks_per_second=args.tps,
            frame_skip=args.frame_skip,
            max_ticks=args.tps * args.match_seconds,
        )

    seeds = [int(s) for s in np.random.default_rng(12345).integers(0, 2**31 - 1, args.episodes)]
    env = make_env()
    net = load_policy(args.checkpoint, env)

    trained = evaluate(env, net, episodes=args.episodes, seeds=seeds, greedy=not args.sample)
    control = evaluate(make_env(), None, episodes=args.episodes, seeds=seeds)

    print(control.summary("random"))
    print(trained.summary("trained"))

    lift = st.mean(trained["returns"]) - st.mean(control["returns"])
    spread = st.pstdev(control["returns"]) or 1.0
    print(f"\n     lift: {lift:+.4f} return ({lift / spread:+.2f} control sd)")
    print(f"           {st.mean(trained['crowns']) - st.mean(control['crowns']):+.3f} crowns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
