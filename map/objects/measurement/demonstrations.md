---
type: object
cluster: measurement
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/train/clone.py
---

# Demonstrations

One shard of what the search expert did and what each position was worth —
`Demonstrations`, saved as `data_cache/demos*/shard-NN.npz`, produced by
`scripts/make_demos.py`, merged and consumed by `scripts/clone_policy.py`.

Verified 2026-08-30 against the working tree at `dc47f51` (`clone.py` and
`make_demos.py` are among the nine uncommitted files — `../../CONTEXT.md`).

## Why this shape

A demonstration set is an artefact that outlives the environment that produced
it, and the value column is the part reinforcement learning **inherits**. Bug 2
was exactly that: shards collected under one reward, fine-tuned under another,
and the arriving critic predicted a number nothing was going to return. So four
provenance fields are stamped by the collector rather than declared by the
loader, and the reason for each is different:

- `observation` — because `clone_policy --observation` was a declaration about
  a file. Most mismatches crash on the channel count; two variants of equal
  width and different meaning train quietly and stamp the wrong name onto the
  checkpoint (`cr_sim/train/clone.py:67-78`).
- `reward` — bug 2 (`:79-86`).
- `proposer` — changes the **labels**, not the inputs: the target is a
  distribution over the candidates that were actually scored (`:87-103`).
- `meta` — carries the fallback rate, which is how support collapse under a
  policy proposer is a measured quantity rather than an argument (`:104-118`).

`""` in any of these means *written before this field existed* and is not the
same as `"v1"` or `"random"`. Nothing may silently read it as one.

## Shape

- The record: `cr_sim/train/clone.py:42`, provenance fields `:67-118`, `save`
  `:123`, `load` `:136`. Stores the **already-encoded** grid and vector — bug
  3's shape.
- Collection: `collect` `:241` (`seed_offset` `:291-301` is why six shards are
  not six runs of the same sixty battles); `_collect_variants` `:426`,
  which re-encodes off the live battle once per variant at each kept decision
  (`:454-458`, `:473-477`) rather than replaying — that closes the declaration
  half of bug 3 for the stored arrays in the variants path.
- Training: `clone` `:568`, `CloneConfig` `:538`. The holdout split takes an
  explicit generator `:592`; the **per-epoch** shuffle does not `:598` — see
  R1 in [`random-streams.md`](random-streams.md).
- The collector: `scripts/make_demos.py`. It passes `reward_weights` and
  stamps the measured tuple off a real environment (`:190-210`, `:330`), so
  **bug 2 is closed**. It builds that `CRSimEnv` with **no `observation=`**
  (`:190-210`), so the env, `env.encoding`, `env.legal_action_mask()` and the
  stored `mask` column are always whatever `cr_sim/api/env.py:319` defaults to
  — v1 today, which is what makes the hard-coded `observation_name="v1"` at
  `:362` true by coincidence rather than by derivation.
- Merge refusals: `merge` `scripts/clone_policy.py:52` (it once dropped
  `target` by omitting one keyword, and every clone fell down the
  `target is None` path); `_agree` `:99` refuses shards disagreeing on
  observation, reward or proposer, and refuses mixing a stamped shard with a
  blank one; `subset` `:123` carries the fields through, because blanking them
  switched off the one check that reads them.
- `collapse_refusal` `scripts/make_demos.py:404` prints `REFUSED FOR MERGING`
  (`:436`) **and writes the shard anyway** — a person decides. `_flag_names`
  `:446` exists because the refusal once named a flag that did not exist.

## Connected to

- **owns:** the `value` column, which is the critic
  [`checkpoint.md`](checkpoint.md) inherits.
- **owned-by:** [`search-bot.md`](search-bot.md) — the expert every shard comes
  from, and the proposer that names the labels.
- **joins:** [`checkpoint.md`](checkpoint.md) (the clone checkpoint copies
  `observation`, `proposer`, `demo_meta` straight out of here);
  [`random-streams.md`](random-streams.md) (R1);
  [`../interface/`](../interface/) for what the stored grid actually *is*.
- **looks-like-but-is-not:** a replay. A shard is decisions with encoded
  observations, not a battle; nothing here can be re-simulated, which is why
  the provenance has to be stamped rather than re-derived.

## If you change this

- **Hits:** every shard already on disk becomes a shard written before your
  field existed, and `_agree` will then refuse to merge old with new
  (`scripts/clone_policy.py:99-121`) — which is the guard working, not a bug. Adding a
  field also means `merge` and `subset` must carry it, and both have already
  been the site of a silent drop.
- **Does not hit:** the checkpoint's recorded `observation`. That is copied
  from `--observation` at `scripts/clone_policy.py:289`, not from
  `data.observation` — the obvious next assumption, that stamping the shard
  makes the checkpoint honest, is wrong. `main` compares the two and warns;
  the checkpoint still records the flag. Nor does it hit `make_demos`' env: a
  new `Demonstrations` field does not give that env an `observation=`.

## Surfaces

| Surface | Role |
|---|---|
| `scripts/make_demos.py` | writes shards |
| `scripts/clone_policy.py` | reads, merges, subsets, trains |
| `scripts/expert_iterate.py` | writes and reads them as subprocess steps (`:171-173`) |
| a human reading `REFUSED FOR MERGING` | the only thing that stops a collapsed shard entering a corpus |

## See

- Source: `cr_sim/train/clone.py`
- As-built: `docs/training.md`
