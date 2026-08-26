"""A multiprocess vectorised environment, fanning independent battles across
worker processes so a training loop can collect many episodes' worth of
transitions per wall-clock second on this machine's 8 cores.

**Why Battle is rebuilt inside each worker rather than sent to it.** This
machine is Windows, where the ``multiprocessing`` default start method is
``spawn``, not ``fork`` -- there is no copy-on-write parent process to inherit
from, so *everything* a worker touches has to survive being pickled across a
pipe from the parent. A :class:`~cr_sim.engine.battle.Battle` holds a
:class:`~cr_sim.data.source.LogicData` with the entire decoded card/character/
projectile table set in it; whether that pickles at all is not something
worth finding out, and even if it does, shipping megabytes of parsed CSV/TOML
down a pipe on every worker start is wasted work when the worker can load its
own copy from disk directly. So a worker receives a :class:`VecEnvConfig` --
a build path, two decks, and a handful of scalars, all of them trivially
picklable -- and reloads ``LogicData`` itself the first thing it does. The
same reasoning is why actions and observations crossing the pipe are plain
tuples/ints and numpy arrays rather than anything holding a reference back
into engine state.

**Windows spawn semantics, concretely.** ``spawn`` re-imports this process's
entry module in every child, so:

*   the worker entry point must be a plain module-level function (a bound
    method, a closure or a lambda cannot be pickled as a ``Process`` target);
*   a script that constructs :class:`CRSimVecEnv` at import time rather than
    under ``if __name__ == "__main__":`` will re-run that construction in
    every spawned child before the child even reaches its own worker loop,
    which either duplicates work or recurses. This is a standard Python
    multiprocessing requirement, not specific to this module, but it is easy
    to forget on a platform where ``fork`` usually papers over it.

**Determinism.** Each worker is handed its own integer seed at ``reset()``
and builds its ``Battle`` from exactly that seed, the same way
:class:`cr_sim.api.env.CRSimEnv` does -- so two runs of this vec env with the
same list of seeds produce the same per-worker tick sequences and the same
state hashes, worker process identity aside.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..data.cards import build_card_registry
from ..data.leveling import build_level_table
from ..data.source import LogicData
from ..engine.entity import Team
from .env import CRSimEnv

__all__ = ["VecEnvConfig", "CRSimVecEnv"]


@dataclass(frozen=True, slots=True)
class VecEnvConfig:
    """The recipe a worker needs to build its own :class:`CRSimEnv`.

    Every field is a plain, picklable value on purpose -- see the module
    docstring for why nothing heavier (a loaded ``LogicData``, a ``Battle``)
    belongs here.
    """

    build: Path
    blue_deck: tuple[str, ...]
    red_deck: tuple[str, ...]
    team: Team = Team.BLUE
    ticks_per_second: int = 60
    frame_skip: int = 6
    level: int = 11
    tower_level: int = 11
    reward_shaping_weight: float = 0.01
    max_ticks: int | None = None


def _worker(config: VecEnvConfig, conn) -> None:
    """Entry point run inside a spawned worker process.

    Everything is rebuilt from ``config`` rather than received pre-built; see
    the module docstring. The loop below is a minimal RPC server over the
    pipe: ``("reset", seed)``, ``("step", action)`` and ``("close", None)``
    are the only messages :class:`CRSimVecEnv` ever sends.
    """
    data = LogicData.load(config.build)
    levels = build_level_table(data)
    registry = build_card_registry(data)
    env = CRSimEnv(
        data,
        levels,
        registry,
        config.blue_deck,
        config.red_deck,
        team=config.team,
        ticks_per_second=config.ticks_per_second,
        frame_skip=config.frame_skip,
        level=config.level,
        tower_level=config.tower_level,
        reward_shaping_weight=config.reward_shaping_weight,
        max_ticks=config.max_ticks,
    )
    try:
        while True:
            try:
                command, payload = conn.recv()
            except EOFError:
                return
            if command == "reset":
                conn.send(env.reset(seed=payload))
            elif command == "step":
                conn.send(env.step(payload))
            elif command == "close":
                return
            else:  # pragma: no cover - defensive; the parent never sends this
                raise ValueError(f"unknown vec env command {command!r}")
    finally:
        conn.close()


class CRSimVecEnv:
    """``num_envs`` independent :class:`~cr_sim.api.env.CRSimEnv` instances,
    each in its own worker process, stepped in lockstep.

    Not a drop-in ``gymnasium.vector.VectorEnv`` (that base class is only
    available when gymnasium itself is installed, which this package does
    not require -- see ``cr_sim.api.env``'s module docstring); it exposes the
    same ``reset()``/``step()`` batching shape -- a list of per-worker results
    stacked into arrays -- so adapting to that interface later, if gymnasium
    is present, is a thin wrapper rather than a rewrite.

    Construct this only under ``if __name__ == "__main__":`` (or equivalent)
    in the calling script; see the module docstring's note on Windows spawn
    semantics for why that guard is load-bearing here and not just tidiness.
    """

    def __init__(
        self,
        config: VecEnvConfig,
        num_envs: int = 8,
        *,
        mp_context: str = "spawn",
    ) -> None:
        self.num_envs = num_envs
        self._ctx = mp.get_context(mp_context)
        self._conns = []
        self._procs = []
        for _ in range(num_envs):
            parent_conn, child_conn = self._ctx.Pipe()
            proc = self._ctx.Process(target=_worker, args=(config, child_conn), daemon=True)
            proc.start()
            # Only the child needs its end; holding it open here too would
            # leave the pipe's write side alive in this process after the
            # child exits, and a recv() would then hang forever instead of
            # raising EOFError.
            child_conn.close()
            self._conns.append(parent_conn)
            self._procs.append(proc)
        self._closed = False

    def reset(self, seeds: Sequence[int | None] | None = None):
        seeds = list(seeds) if seeds is not None else [None] * self.num_envs
        if len(seeds) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} seeds, got {len(seeds)}")
        for conn, seed in zip(self._conns, seeds):
            conn.send(("reset", seed))
        results = [conn.recv() for conn in self._conns]
        obs = [r[0] for r in results]
        infos = [r[1] for r in results]
        return obs, infos

    def step(self, actions: Sequence[Sequence[int]]):
        if len(actions) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} actions, got {len(actions)}")
        for conn, action in zip(self._conns, actions):
            conn.send(("step", action))
        results = [conn.recv() for conn in self._conns]
        obs = [r[0] for r in results]
        reward = np.asarray([r[1] for r in results], dtype=np.float32)
        terminated = np.asarray([r[2] for r in results], dtype=bool)
        truncated = np.asarray([r[3] for r in results], dtype=bool)
        infos: list[dict[str, Any]] = [r[4] for r in results]
        return obs, reward, terminated, truncated, infos

    def close(self) -> None:
        if self._closed:
            return
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():  # pragma: no cover - only on a wedged worker
                proc.terminate()
        for conn in self._conns:
            conn.close()
        self._closed = True

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
