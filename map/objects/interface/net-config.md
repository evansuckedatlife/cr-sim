---
type: object
cluster: interface
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/train/nets.py
---

# NetConfig

The shapes a network needs, as a frozen picklable dataclass, plus `net_config_for`
— the one function that reads them off an environment. It is the seam where an
observation promise becomes a weight matrix.

## Why this shape

Four things build a network for the same environment: the trainer, the evaluator,
the cloner and the spawned workers. A field added in three of them is a shape
mismatch that only surfaces when the fourth loads a checkpoint. So the shapes are
read from `env.encoding` in exactly one place, and nothing else may restate them.

The dataclass is `frozen` **and** `slots` because it has to stay hashable and it
has to survive `asdict()` into `VecEnvConfig.net_config` for every spawned worker.
That is why `card_stats` is a tuple of tuples of plain floats: a numpy array would
break the auto `__hash__`, and a registry or a `LogicData` would not pickle.

## Shape

- Five shape fields — `grid_channels`, `grid_height`, `grid_width`,
  `vector_size`, `num_actions` — all derived, never passed.
- Four encoding-derived fields: `num_slots`, `vocab_size`, `hand_offset`,
  `hand_stride`.
- `head` (see [`policy-heads.md`](./policy-heads.md)) and `card_stats`.
- `num_cells` is a property: `num_actions // num_slots`, raising if it does not
  factor.
- `net_config_for(env, **overrides)` is called with **`head=` and nothing else**
  at all **eight** production sites (`train/{run,ppo,evaluate,ladder}.py`,
  `scripts/{clone_policy,evaluate_checkpoints,evaluate_decks}.py`). Only tests
  pass anything more (`tests/test_action_head_stats.py:111`, `:120`).
- `card_stats` is built **only** when `head == "factored-stats"` arrives in
  `overrides`, keyed on `env.encoding.vocab` in the encoding's order, because
  that is the order the observation's one-hot bits are set in.

Citations: `cr_sim/train/nets.py:56-163` (the dataclass), `:165-173`
(`num_cells`), `:191-242` (`net_config_for`), `:212-213` (shapes and hand layout
read off the env), `:225-229` (the stat table), `:231-241` (the construction),
`cr_sim/api/vec.py:102-105` (`net_config` crosses the pipe),
`cr_sim/train/run.py:728` (`asdict`), `:1019` (handed to the workers).

## Connected to

- **owns:** every head in [`policy-heads.md`](./policy-heads.md) — each reads
  only this object.
- **owned-by:** [`encoding-config.md`](./encoding-config.md) via
  `observation_shapes` and `hand_onehot_layout`.
- **joins:** [`observation-grid.md`](./observation-grid.md) (`grid_channels`);
  [`observation-vector.md`](./observation-vector.md) (`vector_size`,
  `hand_offset`, `hand_stride`); [`action-mask.md`](./action-mask.md)
  (`num_actions`); [`card-features.md`](./card-features.md) (`card_stats`);
  `VecEnvConfig` (`cr_sim/api/vec.py:105`) — index row, card stub.
- **looks-like-but-is-not:** `net_config_for`'s docstring names the exception,
  and it is a real one. `cr_sim/play/policy.py:110-138` has a battle rather than
  an environment and **restates all eleven fields by hand**. It once restated
  `card_stats` as nothing, which built a `"factored-stats"` head with no table —
  a `ValueError` on the first move, swallowed by `PlaySession._think`, and a
  browser opponent that played nothing for the rest of the match
  (`cr_sim/train/nets.py:200-208`).

## If you change this

- **Hits:** all eight `net_config_for` call sites at once, which is the point;
  the hand-restated copy at `cr_sim/play/policy.py:110-138`, which is **not** reached
  from any of them; `VecEnvConfig.net_config`, the pickled `asdict()` every
  worker rebuilds its opponent network from (`cr_sim/train/run.py:728`, `:1019`);
  `ConvPlacementHead.__init__`, whose `ValueError` reads `grid_height`,
  `grid_width`, `num_slots` and `num_actions` together (`cr_sim/train/nets.py:494-502`).
- **Does not hit:** any checkpoint on disk. The obvious next stop — "the network
  changed, so the saved weights record it" — is wrong. No checkpoint stores a
  `NetConfig`; the shapes are re-read from the environment at load
  (`cr_sim/train/evaluate.py:116-135`), which is precisely what makes a mismatch fail
  loudly on a tensor name rather than silently score a policy against an
  observation it was never trained on. `runs/*/config.json` carries an
  `asdict()` of the **PPO** config, not this one.

**Seven fields no entry point can set.** `channels`, `hidden`, `separate_critic`,
`card_embedding`, `place_hidden`, `place_context` and `card_encoder_hidden` have
no CLI flag anywhere in `cr_sim/` or `scripts/` — verified by search. The last
claims in its own docstring to be "a field rather than a literal so it can be
swept without a source edit"; nothing reaches it. **Code wins** ([`../../_meta/overrides.md`](../../_meta/overrides.md), row 15). Filed
**leftover**: changing a default here changes every future run and no past one,
and no artefact records which default a checkpoint was built under.

**`card_stats` is guarded by length only.** `FactoredStatsHead` raises when
`len(card_stats) != vocab_size` (`cr_sim/train/nets.py:418-424`). A **permutation** of the
same length passes, trains to a lower loss, and conditions the head on the wrong
cards — the same failure as bug 4, one layer up. See
[`encoding-config.md`](./encoding-config.md).

## Surfaces

| Surface | Role |
|---|---|
| `cr_sim/train/run.py:397`, `cr_sim/train/ppo.py:243`, `cr_sim/train/evaluate.py:131`, `cr_sim/train/ladder.py:215` | build |
| `scripts/clone_policy.py:267`, `scripts/evaluate_checkpoints.py:69`, `scripts/evaluate_decks.py:209` | build |
| `VecEnvConfig.net_config` → `api/vec.py:_build_env` | reads the pickled dict, per worker |
| `cr_sim/play/policy.py:110-138` | **restates by hand** — the one path `net_config_for` does not cover |
| `runs/*/checkpoint.pt` etc. | none |

## See

- Source: `cr_sim/train/nets.py`

*Verified 2026-08-30 against `main` @ `dc47f51`. `cr_sim/train/evaluate.py:116-135`,
`:131` and `cr_sim/train/ladder.py:215` are in the uncommitted working tree and carry no
line shift at those points; see [`../../CONTEXT.md`](../../CONTEXT.md).*
