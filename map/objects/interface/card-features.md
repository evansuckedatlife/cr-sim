---
type: object
cluster: interface
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/data/card_features.py
---

# Card features

A card described by **what it does**, not by which card it is: 47 named,
identity-free numbers in `[-1, 1]`. `card_feature_table` builds one row per
vocabulary entry for `FactoredStatsHead`. This file lives in `cr_sim/data/` and
is filed under `interface` because the only question anyone asks it is what the
agent sees about a card — and because bug 4 is measured here, at source.

## Why this shape

`FactoredHead` has one free column of `card_embedding.weight` per vocabulary
entry, so column *i* means whatever `vocab[i]` happens to be. This module is the
alternative: compute the column from the card's own statistics, so the head
learns "slow, tanky, ground-only goes at the bridge" rather than "card 4 goes at
the bridge", and an unseen card gets a column for free.

**The boundary is narrower than it looks, and the file says so.** What this buys
is an identity-free *head*. The **trunk still sees identity**: the observation
carries ten card one-hots at `vocab`-indexed offsets and every one of them feeds
`ActorCritic.vector`. Measured on an 8-card mirror deck, a pure relabelling —
consistent across all ten one-hot blocks and the stat table, identical cards in
identical slots — leaves the head's conditioning invariant to **4e-07** and moves
the trunk's features by **56% relative L2**, because 80 of the 102 columns of
`vector.0.weight` are per-vocab-index one-hot columns and all of them receive
gradient. Dropping those one-hots is the change that would fix it, and **it is a
change to the observation, not to this file**.

**Card-local, always.** Every constant this divides by is fixed and named; nothing
is normalised against the other cards in a deck. A z-score over a vocabulary would
give the same card a different vector depending on what it was drawn alongside,
destroying the generalisation claim while looking like an improvement.

## Shape

- `CARD_FEATURE_NAMES` — **47** names in three blocks: A the card itself (valid
  for every card), B the deployed unit (gated by `has_unit`), C the spell or area
  payload (gated by `has_payload`). The encoder's first layer is indexed by this
  order, so inserting a name in the middle silently redefines every weight after
  it. `CARD_FEATURE_COUNT = len(...)`.
- `card_feature_table(data, levels, registry, names)` — one row per name **in the
  order given**, and `names` must be `config.vocab`. Raises `KeyError` for an
  unknown name rather than emitting a zero row, which the head would read as a
  card that is nothing.
- `CARD_FEATURE_LEVEL = 11` and `CARD_FEATURE_CLOCK = TickClock(60)` are pinned
  constants, deliberately **not** `env.level` / `env.ticks_per_second`, to avoid
  train/serve skew.
- Numbers come from `engine.specs.spec_for_card`, not the raw rows and not
  `cards.card_stat_summary`: `Damage` sits on only 51 of the 102 summoning cards
  because a ranged unit keeps its damage on its projectile, and `AttacksAir`
  exists nowhere but `UnitSpec`.
- Four normalisation constants are **mirrored** from the encoder rather than
  imported, to keep the data layer free of it — `_HP_NORM` 6000, `_DPS_NORM` 800,
  `_REACH_NORM` 12, `_COUNT_NORM` 4 — **and the copy is pinned**, by a test that
  asserts all four equal the encoder's. The rest are this module's own, several
  chosen against the mirrored ones on purpose (`_TOTAL_DPS_NORM` 1200 because
  Skeleton Army's fifteen bodies come to 1104.5; `_SIGHT_NORM` 20 because Goblin
  Cage sees nearly twice the longest attack range).

Citations: `cr_sim/data/card_features.py:228-242` (`CARD_FEATURE_NAMES`), `:244`
(`CARD_FEATURE_COUNT`), `:164`, `:169` (the pinned level and clock),
`:175-178` (the mirrored norms), `:442-456` (`card_feature_vector`),
`:589-610` (`card_feature_table`), `:20-45` (bug 4, measured),
`:1-19` (why the head needed it); pin at
`tests/test_card_features.py:94-101`.

## Connected to

- **owns:** the row space of `NetConfig.card_stats` —
  [`net-config.md`](./net-config.md).
- **owned-by:** [`encoding-config.md`](./encoding-config.md). `vocab` is the row
  order and the only correct one.
- **joins:** `FactoredStatsHead` — [`policy-heads.md`](./policy-heads.md);
  [`observation-grid.md`](./observation-grid.md) via the four mirrored norms;
  `UnitSpec` in the **build** cluster, which is where these numbers stop being
  milliseconds and milli-tiles ([`../../CONTEXT.md`](../../CONTEXT.md)).
- **looks-like-but-is-not:** this is **not** the fix for bug 4. It removes card
  identity from the *head* only. Treating a `factored-stats` run as
  deck-portable is the mistake the module's own docstring is written to prevent.

## If you change this

- **Hits:** `FactoredStatsHead.card_encoder`'s input width, which is
  `len(config.card_stats[0])` and so follows `CARD_FEATURE_COUNT` automatically —
  and therefore every `factored-stats` checkpoint on disk, whose
  `card_encoder.0.weight` is `(32, 47)` and is strict-loaded; the meaning of
  every weight after an inserted name, with no shape change and no error;
  `cr_sim/play/policy.py:114-119`, which builds the table itself for the browser
  opponent.
- **Does not hit:** the **observation**. The obvious next stop — "the network's
  view of a card changed, so the encoder must have" — is wrong in both
  directions: no column of the observation vector and no grid channel is built
  from this file, and `check_observation` cannot see a change here at all.
  It also does not hit the other three heads: `card_stats` is built **only** when
  `head == "factored-stats"` (`cr_sim/train/nets.py:225-229`), and is `()` otherwise.

The four mirrored norms are the one place where this file and the encoder must
agree, and the agreement is enforced — `tests/test_card_features.py:94-101`
asserts all four equal `cr_sim.api.encoding`'s. This corrects an earlier audit
that called them an unpinned hand-copy.

## Surfaces

| Surface | Role |
|---|---|
| `net_config_for` (`cr_sim/train/nets.py:225-229`) | reads, once per network build |
| `cr_sim/play/policy.py:114-119` | reads — builds the table by hand for the browser opponent |
| `FactoredStatsHead` buffer (`cr_sim/train/nets.py:440-448`) | **non-persistent** on purpose: keeping it out of `state_dict()` is what stops a feature addition from becoming a checkpoint-versioning problem |
| `runs/` artefacts | none — no table, and no feature name list, is ever stored |
| `tests/test_card_features.py`, `tests/test_action_head_stats.py` | read |

## See

- Source: `cr_sim/data/card_features.py`

*Verified 2026-08-30 against `main` @ `dc47f51`.*
