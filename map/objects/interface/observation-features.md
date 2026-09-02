---
type: object
cluster: interface
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/api/encoding.py
---

# ObservationFeatures

Which observation an environment encodes: a frozen five-flag dataclass plus a
`version` integer. It is the whole of what a checkpoint records under the key
`"observation"`, and the whole of what `check_observation` compares.

## Why this shape

A version number is a **promise about a tensor shape that outlives the run that
made it**. The first convolution has one filter bank per input channel, so
weights for nine channels do not load into a network expecting thirteen —
versioned rather than switched, so a stale checkpoint is refused by name instead
of dying on a `conv.0.weight` size error nobody can place.

`OBSERVATION_V2` is a hand-frozen literal for one reason, written at
`cr_sim/api/encoding.py:219-229`: it used to be "every flag on", `threat` landed, and v2
silently went from 13 channels to 17. A checkpoint recording `"v2"` then parsed
*equal* to the new v2, sailed past the guard, and died on the raw size mismatch
the guard exists to replace. A named version has to mean the same thing forever.
"All of it" is what moves, and that is `OBSERVATION_V3` — which becomes v4, and
freezes, the next time a flag lands.

## Shape

- `ObservationFeatures` — `frozen=True, slots=True`; `version` participates in
  `__eq__` alongside the flags, and is the only thing separating a flag-list
  spelling from a named version.
- Fields: `version` (1), `spells`, `swarm`, `threat` (grid flags), and
  `hide_enemy_hand`, `hide_enemy_elixir` (vector flags, which zero a span and
  never resize it).
- `OBSERVATION_V1` 9 channels · `OBSERVATION_V2` 13 · `OBSERVATION_V3` 17.
  Counts are derived, not stored: `grid_channels()` builds them.
- `parse_observation` accepts `v1`/`v2`/`v3`/`all` **or** a comma-separated flag
  list — and the flag-list branch always stamps `version=2`. So
  `--observation spells,swarm,threat,hide_enemy_hand,hide_enemy_elixir` is
  17 channels carrying `version=2`, compares unequal to `OBSERVATION_V3`, and is
  refused against a v3 environment. Same tensor, different promise.

Citations: `cr_sim/api/encoding.py:176` (class), `:193`, `:195`, `:197`, `:199`,
`:207`, `:211` (fields), `:215`, `:230-236`, `:244-249` (the three versions),
`:219-229` (why v2 is frozen), `:251`, `:275` (parse), `:295-307`
(`grid_channels`), `:862-863`, `:868-871` (the vector flags zero, never resize).

## Connected to

- **owns:** the flag set that [`observation-grid.md`](./observation-grid.md)
  turns into channels.
- **owned-by:** `EncodingConfig.features` —
  [`encoding-config.md`](./encoding-config.md) (`cr_sim/api/encoding.py:337`).
- **joins:** [`check-observation.md`](./check-observation.md) (the only guard
  that reads it); [`net-config.md`](./net-config.md) via `grid_channels`;
  `VecEnvConfig.observation` (`cr_sim/api/vec.py:113`) — index row, card stub.
- **looks-like-but-is-not:** the *string* `"v2"` and the *integer* `version=2`.
  Every flag-list observation is `version=2`. See the `v2` row in
  [`../../CONTEXT.md`](../../CONTEXT.md).

## If you change this

- **Hits:** `grid_channels()` output width, and through it `NetConfig.grid_channels`
  and `conv.0.weight` (`cr_sim/train/nets.py:231`, `:530`); every checkpoint on disk
  recording the old name; `check_observation` (`cr_sim/train/evaluate.py:89`);
  `config.json`'s `"observation"` and `"observation_channels"`
  (`cr_sim/train/run.py:628-629`); `VecEnvConfig.observation`, which is the only reason
  a worker building v1 while the parent expects v2 is loud (`cr_sim/api/vec.py:110-113`);
  the browser opponent, which rebuilds its encoding from the checkpoint's
  recorded name (`cr_sim/play/policy.py:86-91`).
- **Does not hit:** the observation **vector's length**. The obvious next move —
  "a new flag changes the observation, so it changes the vector" — is wrong.
  `_vector_length` is a function of `vocab_size` alone (`cr_sim/api/encoding.py:419-423`),
  and both vector flags zero a span that stays where it was. Nor does it hit the
  **vocabulary**: two runs on the same version with different decks compare
  equal here and always will.

## Surfaces

| Surface | Role |
|---|---|
| `--observation` on `train/run.py`, `scripts/clone_policy.py`, `evaluate_*`, `run_ladder.py`, `make_demos.py` | writes (a string, parsed) |
| `runs/*/config.json` — `"observation"`, `"observation_channels"` | written once per run, read by humans and `train/watch.py` |
| checkpoint `"observation"` key (`cr_sim/train/run.py:1053`) | written; read by `check_observation` and `play/policy.py` |
| `Demonstrations.observation` (`cr_sim/train/clone.py:78`) | written per shard |
| `tests/test_observation_v2.py` | pins v1's exact tuple (`:104-108`) and the registry contract (`:376-410`) |

`tests/test_observation_v2.py:387`'s own docstring says a newly added set means
"`v2` contains it". Its assertion at `:404` checks `OBSERVATION_V3`, not v2 —
correctly, since v2 is frozen. **Code wins over that docstring** ([`../../_meta/overrides.md`](../../_meta/overrides.md), row 11).

## See

- Source: `cr_sim/api/encoding.py`
- As-built: `docs/training.md`, section "Which observation changes helped"

*Verified 2026-08-30 against `main` @ `dc47f51`.*
