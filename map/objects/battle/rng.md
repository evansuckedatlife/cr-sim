---
type: object
cluster: battle
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/rng.py
---

# Rng

PCG-XSH-RR 32, owned by the battle. The engine's only source of randomness, and
it is drawn from in exactly two places.

## Why this shape

Python's `random` is unusable here: a shared global whose stream anything in
the process can perturb and whose internals may change between releases. A
simulator whose replays must reproduce needs a generator it owns and can pin
forever, which PCG-XSH-RR is specified precisely enough to be.

**Children are derived by label, not by draw order.** `Rng.stream("deck:BLUE")`
mixes the label into the state with an FNV-style walk. That means adding or
removing a draw inside one subsystem cannot shift another subsystem's stream,
so existing replays stay valid across an engine change. Interleaving draws on
one shared generator would make every saved replay a hostage to the next
feature.

**`below()` rejection-samples.** `next_u32() % bound` skews low whenever the
bound does not divide 2^32, and over millions of simulated battles that bias
is exactly the kind of thing that quietly teaches an agent something untrue.

The engine draws from **two** streams and no others:
`deck:{team.name}` for the opening shuffle (`cr_sim/engine/battle.py:436`) and
`aeospawn:{effect.id}` for area-effect spawn jitter (`:1643`). Everything else
in `engine/` is deterministic by construction. The interpreter is deliberately
given no stream at all — see [action-select.md](action-select.md).

## Shape

- `Rng(seed, increment)` — the increment is forced odd for full period;
  `__slots__` of two ints.
- `next_u32`, `below(bound)`, `shuffle(list)` (in-place Fisher-Yates).
- `stream(label)` — the child derivation.
- `state()` / `restore()` / `chance()` / `between()` — **ghosts.** Defined,
  documented, exercised only by `tests/test_engine_core.py:162`; no caller in
  `cr_sim/`. `state()`'s docstring says "for snapshotting into a replay", and
  no replay snapshots it.

Citations: `cr_sim/engine/rng.py:29`, `:53-67`, `:80-84`, `:86`, `:98`, `:102`;
`cr_sim/engine/battle.py:332`, `:436`, `:1643`.
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** nothing.
- **owned-by:** [battle-config.md](battle-config.md) — `Rng(config.seed)`,
  constructed once at `cr_sim/engine/battle.py:332`.
- **joins:** [battle.md](battle.md), [battle-clone.md](battle-clone.md) (the
  `Rng` is **not** in `_SHARED`, so a branch gets its own copy of the state and
  its draws do not disturb the original).
- **looks-like-but-is-not:** every random stream outside the engine. Bug 5's
  five unowned streams were `torch`'s global generator inside `--workers`
  processes, numpy's legacy `RandomState` in the PPO minibatch shuffle,
  `evaluation_probe`, `ancestor_probe`, and `evaluate_paired` keying its
  generator on an arm's index in `modes`. **None of them is this class** — they are
  [../measurement/random-streams.md](../measurement/random-streams.md). Also
  not entity ids ([entity-ids.md](entity-ids.md)), which are not random.

## If you change this

- **Hits:** every stored hash stream — a change to `below`, `shuffle` or
  `stream`'s mixing function changes the opening hand for a given seed, so
  every replay, every frozen evaluation and every "same seed" comparison moves
  at once. Adding a *third* engine stream is safe by construction as long as it
  takes a labelled child; drawing off `self.rng` directly is what breaks the
  guarantee.
- **Does not hit:** training reproducibility. A run under `--workers` is
  reproducible only if the *measurement* cluster's streams are owned; making
  this class more deterministic cannot help, because it is already exact. The
  obvious wrong response to "the promotion probe returned +0.905/+1.228/+0.970
  on identical inputs" is to reseed the battle — the battle was never the
  variable.

## Surfaces

| Surface | Role |
|---|---|
| `--seed` on `cr-sim battle` (`cr_sim/cli.py:544`) | writes |
| `BattleConfig.seed` from `env.py` / `VecEnvConfig` | writes |
| `tests/test_engine_core.py:148-204` | pins the stream and the ghosts |
| nothing serialises `Rng` state | — |

## See

- Source: `cr_sim/engine/rng.py`
