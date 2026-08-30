---
type: object
cluster: measurement
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/train/evaluate.py
---

# Verdict

The file a measurement outlives its run in — `runs/<name>/verdict.json`,
written only through `write_verdict`, which refuses one that does not say what
it was measured against. Its per-row twin is `check_lift_is_named`, the guard
every writer of a `metrics.jsonl` row passes through.

Verified 2026-08-30 against the working tree at `dc47f51` (nine files
uncommitted — `../../CONTEXT.md`, Verification basis).

## Why this shape

A comment is not a guardrail. "Lift" was reported on two incompatible scales
for most of this project's life and the two numbers were compared to each
other; the cost was two rounds of invalid comparisons and one retracted
headline. So the provenance is not documented, it is **demanded at the write**,
and it is read **off the environment that actually played** rather than taken
as an argument — a caller cannot then label a measurement with an opponent or
a reward it did not face (`cr_sim/train/selfplay.py:163`, `:188`).

Two axes, not one. *Who* was played and *what was counted* fail the same way,
and the second was exempt for longer: the row guard demanded `eval_reward`
before the file did.

## Shape

- Three refusals in `write_verdict` (`cr_sim/train/evaluate.py:483`): no
  `eval_opponent` at all `:507`; a lift key without `eval_reward` `:514`,
  keyed on `_LIFT_KEYS` `:480` so a rating alone is exempt; and a file
  carrying both `lift` and `ladder_elo` without `lift_player` /
  `lift_opponent` `:526-545`.
- Five refusals in `check_lift_is_named` (`cr_sim/train/selfplay.py:55`):
  `eval_lift_sd` without `eval_opponent` `:110`; without `eval_reward` `:117`;
  a `SCORED_FAMILIES` prefix (`:52`) without its own `<prefix>opponent`
  `:129-140` and without `<prefix>opponent_ref` `:141-146`; and `ladder_elo`
  without `ladder_pinned` `:147-159`.
- `SCORED_FAMILIES` exists because a **score is not a lift** and walked past
  the original clause — the exemption `ancestor_probe` enjoyed for its whole
  life. Each family names its own side because one metrics row genuinely
  carries several measurements against several opponents and a single
  `eval_opponent` cannot name them all (`cr_sim/train/selfplay.py:44-51`, `:99-110`).
- `reward_name` records the **full weight tuple**, not the variant name: the
  same variant at two weights produces returns an order of magnitude apart,
  and an anneal produces both inside one run (`cr_sim/train/selfplay.py:172-179`).
  `opponent_name` reports `"unknown"` rather than guessing (`:198-201`).
- `EVAL_REWARD` pins the in-run probe to a constant, independent of `--reward`,
  so the promotion criterion stops being a function of the training schedule
  (`cr_sim/train/run.py:342`, with the whole argument at `:324-341`). It is
  **not** the training reward, which is what makes an in-run lift and an
  offline script's lift two different units.
- `rotating_probe` (`cr_sim/train/evaluate.py:555`) emits every key
  `evaluation_probe` emits, plus `eval_block` / `eval_blocks`. Its docstring
  still *describes* two edits to `run.py` "rather than making them"
  (`:592-611`); `cr_sim/train/run.py:979-984` already made them. **Code wins — the probe is
  wired and reachable behind `--probe rotating`** ([`../../_meta/overrides.md`](../../_meta/overrides.md), row 17).

## Connected to

- **owns:** the `eval_opponent` / `eval_reward` / `ladder_pinned` /
  `<family>_opponent_ref` fields wherever they appear.
- **owned-by:** [`run-directory.md`](run-directory.md) — the directory the
  file lands in, beside `config.json` and `metrics.jsonl`.
- **joins:** [`lift.md`](lift.md) (the number being guarded);
  [`ladder.md`](ladder.md) (the rating that is exempt from the reward clause
  and subject to the pinning clause); [`self-play.md`](self-play.md) (the two
  probes that produce rows).
- **looks-like-but-is-not:** `runs/<name>/config.json`. It also records
  `eval_opponent` and `eval_reward` (`cr_sim/train/run.py:634`, `:652`), but it records
  what the run was *configured* to measure against; the verdict records what
  an environment actually reported. Only the second can be wrong in a way that
  is detectable.

## If you change this

- **Hits:** every writer of a verdict, because they all go through this one
  function — `cr_sim/train/evaluate.py:789`, `scripts/clone_policy.py:372`,
  `scripts/run_ladder.py:426`, `scripts/measure_expert.py:231`,
  `scripts/evaluate_vs_expert.py:177`. Adding a refusal makes older files on
  disk unwritable, not unreadable: nothing re-validates what is already there.
- **Does not hit:** `report.py` and `watch.py`, which read these files and
  enforce nothing. The obvious next assumption — that tightening
  `write_verdict` makes the progress page refuse a bare number — is wrong;
  `report.py` withholds aggregates on its own separate rule, by comparing the
  `(eval_opponent, eval_reward)` pairs it finds in the rows
  (`cr_sim/train/report.py:82-99`). Nor does it touch `config.json`: a run can
  still record one `eval_opponent` there and write a verdict naming another.

## Surfaces

| Surface | Role |
|---|---|
| the five `write_verdict` callers | write |
| `cr_sim/train/report.py` | reads the flat keys for its chips (`:126-129`) |
| `cr_sim/train/watch.py` | reads for the progress page |
| a human quoting a result | the reader the guard exists for |

## See

- Source: `cr_sim/train/evaluate.py`, `cr_sim/train/selfplay.py`
