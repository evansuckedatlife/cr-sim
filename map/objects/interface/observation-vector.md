---
type: object
cluster: interface
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/api/encoding.py
---

# Observation vector

The flat half of what the agent sees: everything with no position — elixir, both
hands, tower health, the clock. 102 float32 columns at the only vocabulary any
training path has used. **80 of them are card-identity one-hots keyed on
vocabulary position.** Built by `_encode_vector`.

## Why this shape

Folding these into the grid — broadcasting elixir over 576 cells — would spend
channels on a scalar and make the grid's per-cell semantics inconsistent. So they
are a separate array, and the network gives them a separate MLP.

The width is a **function of `vocab_size`**, and that is deliberate: it is the
only thing that makes a vocabulary change of a different *size* loud. It does
nothing at all for a vocabulary change of the same size, which is bug 4.

## Shape

Concatenated in this order, with `V = config.vocab_size`, one **slot block** of
width `1 + V` per card slot — cost, then identity:

| Columns | What | Vocab-keyed |
|---|---|---|
| `0`, `1` | acting team's elixir, opponent's elixir | no |
| 10 blocks at `2 + s*(1+V)`, `s = 0..9` | `s = 0..3` acting hand, `s = 4` acting next card, `s = 5..8` opponent hand, `s = 9` opponent next card | the block's **first** column is mana cost; **the remaining `V` are the one-hot** |
| next 6 | tower fractions: acting king, low-x princess, high-x princess, then the opponent's three | no |
| next 4 | tick fraction, acting crowns/3, opponent crowns/3, overtime flag | no |

At `V = 8` — every training, evaluation, cloning and demo path on this machine —
the array is **102** wide and the vocabulary-keyed columns are exactly:

```
3-10  12-19  21-28  30-37  39-46  48-55  57-64  66-73  75-82  84-91   (80 columns)
```

The ten mana-cost columns are `2, 11, 20, 29, 38, 47, 56, 65, 74, 83`; the
12-column non-card tail begins at `92`.

`hand_onehot_layout(config)` returns `(start=3, stride=1+V, count=4, width=V)` —
it addresses **only the acting team's four hand slots**, 32 of the 80. That is
what the card-conditioned heads read. The trunk reads all 102.

An empty slot (only `next_card` past the end of a short deck) is all zeros — a
legitimate "nothing here" that collides with no real card. The two hiding flags
zero a span and never resize it.

Citations: `cr_sim/api/encoding.py:853-883` (`_encode_vector`, term by term),
`:835-851` (`_card_features` — the one-hot is set at `1 + vocab.index(name)`),
`:401-417` (`hand_onehot_layout`), `:419-423` (`_vector_length`),
`:886-894` (`encode_observation`), `:467-497` (`_team_towers`),
`:506-519` (`total_tower_hitpoints`), `:862-863`, `:868-871` (the hiding flags).

## Connected to

- **owns:** nothing below it.
- **owned-by:** [`encoding-config.md`](./encoding-config.md) — `vocab` is the
  index space, `vocab_size` is the width.
- **joins:** [`net-config.md`](./net-config.md) (`vector_size`, `hand_offset`,
  `hand_stride`); [`policy-heads.md`](./policy-heads.md) (`FactoredHead._context`
  and `FactoredStatsHead._context` slice this array directly);
  [`observation-features.md`](./observation-features.md) (the two hiding flags).
- **looks-like-but-is-not:** `_vector_length` is **not** a derivation of
  `_encode_vector` — it is a second hand-maintained copy of the same layout
  (`:419-423`). And `total_tower_hitpoints`' docstring says the observation and
  `env.py`'s reward both read through `_team_towers`, which scans
  `battle.entities` **plus** `battle.graveyard` (`cr_sim/api/encoding.py:484`)
  because a dead tower is moved to the graveyard rather than dropped,
  "rather than each
  maintaining their own notion of the towers" — true only of the simple shaped
  reward (`cr_sim/api/env.py:205-206`). `RewardTracker._observe` reads
  `battle._towers[...]` directly (`cr_sim/api/reward.py:217-222`). **Code wins** ([`../../_meta/overrides.md`](../../_meta/overrides.md), rows 9 and 10).

## If you change this

- **Hits:** `_vector_length`, which is the declared `observation_space` Box and
  `NetConfig.vector_size` and `vector.0.weight`; `hand_onehot_layout`, and
  through it `NetConfig.hand_offset`/`hand_stride` and the span both factored
  heads slice (`cr_sim/train/nets.py:303-310`, `:452-456`); `cr_sim/play/policy.py:97`, which
  restates the same layout by hand for the browser opponent; every checkpoint on
  disk, whose `vector.0.weight` has one column per column here.
- **Does not hit:** the **grid**, the **action space** or the **legality mask**.
  The obvious next stop after "I added something the agent can see" is the mask —
  wrong: the mask is built from elixir, hand contents and terrain
  (`cr_sim/api/encoding.py:586-641`) and reads no part of this array. Nor does it hit
  `check_observation`, which compares a feature set and never a width.

**Where the guard actually is.** `hand_onehot_layout`'s docstring claims the
offset is "derived here from the same terms `_encode_vector` builds, so the two
cannot drift apart." `start = 2 + 1` is a literal (`:415`) — **code wins over
that docstring**. But the promise holds anyway, and by a different mechanism:
`tests/test_action_head.py:83-104` re-encodes a live battle, slices with the
returned `(start, stride)`, and asserts the span is a one-hot naming
`hand[slot]`. Insert a scalar ahead of the hand and that test fails on the first
slot. This corrects the "silently wrong, no error" phrasing carried in
[`../_index.md`](../_index.md): the failure mode is real, the guard is a test in
`tests/`, not the function's own arithmetic. **The opponent's five blocks and the
12-column tail have no such round-trip.**

## Surfaces

| Surface | Role |
|---|---|
| `ActorCritic.vector` / `critic_vector` (`cr_sim/train/nets.py:538-540`) | reads all 102, every forward |
| `FactoredHead` / `FactoredStatsHead` (`cr_sim/train/nets.py:303-310`, `:452-456`) | read 32 of the 80 one-hot columns |
| `play/policy.py:97, 128-130` | restates the layout for the browser opponent |
| `Demonstrations.vector` (`cr_sim/train/clone.py:356`, `:477`) | written already-encoded |
| `tests/test_action_head.py:83`, `tests/test_observation_v2.py:259-280` | read; the only round-trip on the layout |

## See

- Source: `cr_sim/api/encoding.py`
- As-built: `cr_sim/data/card_features.py:20-45` — bug 4 measured, at source

*Verified 2026-08-30 against `main` @ `dc47f51`. `cr_sim/train/clone.py:356` and `:477`
are working-tree lines; see [`../../CONTEXT.md`](../../CONTEXT.md).*
