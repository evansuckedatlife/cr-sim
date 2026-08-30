---
type: object
cluster: interface
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/train/nets.py
---

# ActorCritic and the policy heads

The network. One convolutional-plus-MLP trunk, a value head, and a policy head
chosen by name from `POLICY_HEADS`. All four heads emit the **same** flat masked
categorical over 720 actions, so nothing downstream changes when one is swapped.

**A correction to the usual shorthand: `POLICY_HEADS` is four names carried by
three `nn.Module` classes.** `"flat"` is a bare `nn.Linear(hidden, num_actions)`
built inline, not a class; `"factored"`, `"factored-stats"` and `"conv"` are
`FactoredHead`, `FactoredStatsHead` (its subclass) and `ConvPlacementHead`.

## Why this shape

Every head is a **reparameterisation of the same 720 logits**, not a different
action space. That is what makes them comparable on one set of demonstrations —
the legality mask, sampling, greedy argmax, PPO's ratio and the cloner's
cross-entropy are all unchanged.

Why more than the flat head: cloning the search bot produced 6,094 examples over
443 distinct (card, tile) pairs — fourteen apiece. A flat head has one
independent weight vector per pair and learns nothing about "in front of my own
tower" from a Knight that it can apply to a Musketeer. `FactoredHead` shares the
tile weights across cards and passes the card in as input; `ConvPlacementHead`
goes further and makes placement **translation-equivariant**, because the trunk's
second convolution already produces the 16x9 placement grid and the layer after
it throws that away.

`"factored-stats"` is a fourth *name* rather than a change to `"factored"`
because every `load_state_dict` in this tree is strict, seven factored
checkpoints exist on disk, and one is a live run's resume target.

## Shape

- `POLICY_HEADS = ("flat", "factored", "factored-stats", "conv")` — one tuple,
  because two hand-written argparse lists went stale and `"factored-stats"`
  shipped complete and unreachable from both entry points.
- Trunk (`_encoder`, built twice when `separate_critic`): `Conv(3x3) → ReLU →
  Conv(3x3, stride 2) → ReLU → Conv(3x3, stride 2) → ReLU → Flatten`, in
  parallel with `Linear(vector_size, hidden) → ReLU`, concatenated into
  `Linear → ReLU`.
- `_SPATIAL_DEPTH = 4` slices `self.conv` just past the first stride-2
  convolution; `encode_spatial` returns `(trunk features, that 16x9 map)`. It is
  a literal index, kept so parameter names stay `conv.0`..`conv.4` and every
  pre-conv-head checkpoint still loads.
- `FactoredHead`: `log P(slot) + log P(cell | slot)`, remasked after the two
  log-softmaxes because two composed log-softmaxes leave a finite floor, not a
  true zero. A slot is choosable iff one of its cells is — derived from the mask,
  never from elixir separately.
- `FactoredStatsHead`: `onehots @ encoder(table)`, **never**
  `encoder(onehots @ table)`. An empty slot is an all-zero one-hot and must map
  to an exactly-zero conditioning vector; `encoder(0)` returns the encoder's bias
  instead. The trap is invisible at initialisation — every bias starts at zero —
  and one Adam step ends that.
- `ConvPlacementHead`: 1x1 convolution over the trunk's own map, then
  `permute(0, 1, 3, 2)` from `(batch, slot, y, x)` to `(batch, slot, x, y)`.
- `MASKED_LOGIT = -1e8`, not `-inf`: `-inf` gives NaN gradients wherever a whole
  row is masked, and a NaN reaching the optimiser corrupts every weight silently.

Citations: `cr_sim/train/nets.py:53` (`POLICY_HEADS`), `:521-568` (`ActorCritic`,
head dispatch at `:555-567`), `:528-546` (the trunk), `:574`, `:588-589`
(`_SPATIAL_DEPTH`), `:244-343` (`FactoredHead`, mask logic `:329-341`),
`:344-468` (`FactoredStatsHead`, mix order `:458-459`, guards `:413-424`),
`:470-519` (`ConvPlacementHead`, geometry guard `:494-502`, permute `:517`),
`:675-681` (`MASKED_LOGIT`, `_apply_mask`), `:623-646` (`policy_logits`).

## Connected to

- **owns:** nothing — the heads are leaves.
- **owned-by:** [`net-config.md`](./net-config.md). Every head reads only a
  `NetConfig`.
- **joins:** [`action-mask.md`](./action-mask.md) (the flat mask is the head's
  only legality input); [`observation-vector.md`](./observation-vector.md) (both
  factored heads slice it at `hand_offset`);
  [`card-features.md`](./card-features.md) (`FactoredStatsHead`'s table);
  [`observation-grid.md`](./observation-grid.md) (`conv.0`).
- **looks-like-but-is-not:** `masked_categorical` (`cr_sim/train/nets.py:684-696`) is a
  **ghost**. The module docstring credits it with the NaN guard
  (`:24-28`). It has zero production callers; the five sites that actually build
  a `Categorical` — `cr_sim/train/nets.py:653`, `:666`, `cr_sim/train/evaluate.py:206`,
  `cr_sim/train/selfplay.py:329`, `cr_sim/play/policy.py:167` — construct it directly, without
  the assert. **Code wins** ([`../../_meta/overrides.md`](../../_meta/overrides.md), row 12). What actually prevents the NaN is
  `MASKED_LOGIT = -1e8` plus the unconditionally legal noop cell.

## If you change this

- **Hits:** every checkpoint written with the head you changed — parameter names
  under `policy_head` are strict-loaded and there is no migration shim; the
  `"head"` string recorded in all four checkpoint payloads and read back at load
  (`cr_sim/train/run.py:868-885`, `:891`, `:1053`; `cr_sim/train/evaluate.py:131`;
  `scripts/clone_policy.py:288-302`); `POLICY_HEADS`, which is both argparse
  choice lists; `cr_sim/play/policy.py:120-138`, the one load path outside
  `net_config_for`; `VecEnvConfig.net_config`, since a worker's frozen opponent
  must be the same head as the learner.
- **Does not hit:** the **observation** or the **action space**. The obvious next
  stop after a head change is `check_observation` — wrong twice over: it compares
  an `ObservationFeatures` and never looks at `head`, and the head guard is a
  separate string comparison in `--init-from` (`cr_sim/train/run.py:684-689`) with no
  equivalent on `--resume`. A head change also does not move `num_actions`: all
  four heads emit the same 720.

**`_SPATIAL_DEPTH` is a literal index into a module list, and nothing validates
it against that list.** `ConvPlacementHead`'s `ValueError` checks the **arena**
— `grid_height`, `grid_width`, `num_slots`, `num_actions` — not the trunk. Insert
a layer into `_encoder()` and the check still passes while `conv[:4]` returns
something else. `encode` is unaffected (any split of the same `Sequential`
concatenates back), so the damage is confined to the `conv` head and to whatever
reads `encode_spatial`'s second return.

## Surfaces

| Surface | Role |
|---|---|
| `--head` on `train/run.py` and `scripts/clone_policy.py` | writes (a string from `POLICY_HEADS`) |
| checkpoint `"head"` key, all four payloads | written; read at every load |
| `runs/*/config.json` `"head"` (`cr_sim/train/run.py:602`, `:611`) | written twice, same value |
| `VecEnvConfig.net_config` → each worker | reads |
| `train/watch.py`, `train/report.py` | read `config.json`, not the module — see the `surfaces` cluster |
| `tests/test_action_head.py`, `tests/test_action_head_stats.py` | the only round-trip on index order and mix order |

## See

- Source: `cr_sim/train/nets.py`
- As-built: `docs/training.md`, section "Which action head"

*Verified 2026-08-30 against `main` @ `dc47f51`. `cr_sim/train/evaluate.py:131`, `:206`
are in the uncommitted working tree and carry no line shift at those points; see
[`../../CONTEXT.md`](../../CONTEXT.md).*
