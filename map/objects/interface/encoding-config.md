---
type: object
cluster: interface
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/api/encoding.py
---

# EncodingConfig

Everything the encoder needs to fix array shapes for one environment's
lifetime: arena geometry, the observation feature set, and **`vocab`** — this
episode's card vocabulary. `vocab` is where bug 4 lives.

## Why this shape

Gymnasium's contract is that `observation_space` and `action_space` hold for the
life of an env, so nothing that changes shape between episodes may leak into the
per-episode encoding path. It is fixed here once, at construction, and reused by
every `reset()`.

`vocab` is `tuple(sorted(set(blue_deck) | set(red_deck)))` — **the deck union,
rebuilt per environment**. A fixed run plays a fixed pair of decks, so a one-hot
over 16 cards beats a one-hot over the whole pool where almost every entry is a
permanent zero. The cost is the thing to hold on to:

> **Column *i* of every card one-hot means `vocab[i]`, and `vocab` is a property
> of the decks, not of the checkpoint.** Two same-size decks produce the same
> array shapes, load strictly, run, and mean different things column for column.
> Nothing errors. Measured at `cr_sim/data/card_features.py:20-45`: a pure
> relabelling leaves the *head* invariant to 4e-07 and moves the *trunk* by 56%
> relative L2.

`sorted()` is load-bearing for a different reason: it makes the index assignment
independent of dict/set iteration order, so two processes building the same
config agree. It does not make the vocabulary stable across decks — sorting is
what guarantees a *different* deck permutes rather than randomises.

## Shape

- `EncodingConfig` — `frozen=True, slots=True`: `grid_width`, `grid_height`,
  `action_width`, `action_height`, `vocab: tuple[str, ...]`, `features`.
  Properties `vocab_size` and `channels`.
- `build_encoding_config(arena, blue_deck, red_deck, features)` is the only
  constructor in production. `_grid_shape` ceiling-divides the arena by the span,
  giving 18x32 observation cells and 9x16 action cells on the standard board.
- `observation_shapes(config)` → `{"grid": (C, 32, 18), "vector": (102,)}` at
  `vocab_size = 8`. This is what `net_config_for` reads shapes from.
- Six independent construction sites, each passing its own pair of decks:
  `cr_sim/api/env.py:356` (`CRSimEnv`), `cr_sim/api/env.py:677` (`CRSimSelfPlayEnv`, which
  passes **no** `features` and is therefore pinned to v1),
  `cr_sim/play/policy.py:91`, `cr_sim/train/clone.py:455` (per observation variant; **working-tree line** — see [`../../CONTEXT.md`](../../CONTEXT.md)),
  `cr_sim/train/scripted.py:263`, `scripts/bench_engine.py:253`.

Citations: `cr_sim/api/encoding.py:319-347` (the dataclass and its properties),
`:337` (`vocab`), `:365-381` (`build_encoding_config`), `:373` (the union),
`:350-362` (`_grid_shape`), `:389-398` (`observation_shapes`),
`cr_sim/train/run.py:54` and `cr_sim/play/server.py:46` (the two independent
`DEFAULT_DECK` literals — see [`../../CONTEXT.md`](../../CONTEXT.md)).

## Connected to

- **owns:** the index space every card one-hot in
  [`observation-vector.md`](./observation-vector.md) is keyed on, and the row
  order of `card_stats` in [`card-features.md`](./card-features.md).
- **owned-by:** `CRSimEnv._config`, exposed as `CRSimEnv.encoding`
  (`cr_sim/api/env.py:372`) — index row, card stub.
- **joins:** [`observation-features.md`](./observation-features.md) (`features`);
  [`net-config.md`](./net-config.md) (`vocab_size`, `hand_offset`,
  `hand_stride`, `card_stats` all derive from here);
  [`action-mask.md`](./action-mask.md) (`action_width` × `action_height`).
- **looks-like-but-is-not:** `CardRegistry`. The registry holds every card in the
  build and is stable; `vocab` holds this env's eight and is not. A head keyed on
  the registry would be portable; nothing here is keyed on the registry.

## If you change this

- **Hits:** all 80 vocabulary-keyed columns of the observation vector
  (`cr_sim/api/encoding.py:835-851`); `_vector_length`, and through it `vector.0.weight`
  (`cr_sim/api/encoding.py:419-423`, `cr_sim/train/nets.py:538-540`) — so a **size** change fails
  strict loading and is loud; `NetConfig.vocab_size`/`hand_offset`/`hand_stride`
  (`cr_sim/train/nets.py:213`, `:236-239`); `FactoredHead.card_embedding.weight`, one
  column per entry (`cr_sim/train/nets.py:283-284`); the row order of `card_stats`,
  which `FactoredStatsHead` checks by **length only** (`cr_sim/train/nets.py:418-424`);
  the hand-restated config in `cr_sim/play/policy.py:110-138`.
- **Does not hit:** `check_observation`. The obvious next stop — "the guard that
  refuses a mismatched checkpoint" — is the wrong one. It compares
  `ObservationFeatures` and never touches `vocab`
  (`cr_sim/train/evaluate.py:89-113`), so a deck swap passes it silently. See
  [`check-observation.md`](./check-observation.md). It also does not hit the
  **grid**: no channel is keyed on a card identity, so a deck swap leaves every
  grid channel meaning exactly what it meant.

## Surfaces

| Surface | Role |
|---|---|
| `runs/*/config.json` `"deck"` (`cr_sim/train/run.py:606`) | writes — the module constant `DEFAULT_DECK`, not the deck read off the env |
| `best.pt` / `checkpoint.pt` / `final.pt` / `cloned.pt` | **none.** All four record `state_dict`, `head`, `observation` and metrics only (`cr_sim/train/run.py:868-885`, `:891`, `:1053`; `scripts/clone_policy.py:288-302`) |
| `Demonstrations` (`cr_sim/train/clone.py:40-100`) | **none** — records `observation`, `reward`, `proposer`, not the deck |
| no CLI flag anywhere | there is no `--deck`; every training path uses the literal |

A checkpoint moved out of its run directory therefore carries **no** record of
what its 80 vocabulary columns meant. This corrects the blanket phrasing in
[`../_index.md`](../_index.md): `config.json` does record a deck, at the run
level; no *model* artefact does, and nothing reads the `config.json` field back.

## See

- Source: `cr_sim/api/encoding.py`
- As-built: `cr_sim/data/card_features.py:20-45` — bug 4 measured, at source

*Verified 2026-08-30 against `main` @ `dc47f51`.*
