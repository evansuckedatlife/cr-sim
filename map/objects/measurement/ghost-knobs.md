---
type: object
cluster: measurement
universe: ghost
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/train/run.py
---

# Ghost and leftover knobs

The flags and paths in the measurement cluster that are reachable but not the
main path, or documented but not wired. One card so that a reader who arrives
at a docstring describing something working can find out, in one hop, whether
it exists.

Verified 2026-08-30 against the working tree at `dc47f51` (`docs/training.md`
is among the nine uncommitted files — `../../CONTEXT.md`).

## Why this shape

The failure this card exists to stop is not "a dead function is present". It is
**implementing against a description**. Three of the entries below are
described in prose that reads exactly like a specification of working code —
one of them is a paragraph explaining a design that was written and reverted,
another is a docstring listing two edits to another file that have since been
made. Reading either as a plan produces work that is already done, already
declined, or already known not to run.

`_resolve_device` is the counterweight and belongs here for that reason: it
**says plainly** when the asked-for device is unavailable, because a silent CPU
fallback leaves a run three times slower with nothing on the page to say so —
which is the failure mode this project keeps producing
(`cr_sim/train/run.py:350-355`).

## Shape

- **`--device xpu` — ghost.** Reports available, runs a gradient step 6.6x
  faster than eight CPU threads, then fails a real training loop three ways:
  an unimplemented convolution, out of device memory, and out of Level Zero
  resources during the optimiser's own state allocation. Explicit refusal path
  `cr_sim/train/run.py:363-371`; **deliberately never chosen by `auto`**, with
  the reason written at the declining site `:375-384`. A default that picks a
  backend which cannot finish an update is worse than no default.
- **The γ-correct potential `r = γΦ(s′) − Φ(s)` — ghost.** Written and
  reverted once. `docs/training.md:483-505`. Half the stated reason is **now
  false**: `projected` measures 1.038 score calls per decision, not 2. What
  survives is that γ is charged per *score* and a score is not a PPO timestep.
  It is explicitly **not** a prerequisite for the anneal. Do not re-derive the
  plan from that paragraph.
- **`rotating_probe`'s adoption note — stale, code wins.** Its docstring
  describes two edits to `run.py` "rather than making them"
  (`cr_sim/train/evaluate.py:592-611`); `cr_sim/train/run.py:979-984` already made them.
  The probe is wired and reachable behind `--probe rotating`
  (`cr_sim/train/run.py:97-107`). [`../../_meta/overrides.md`](../../_meta/overrides.md), row 17.
- **`--probe rotating` — live, unused.** No run on disk has used it: exactly
  one `config.json` in `runs/` records `"probe"` at all, and it records
  `"ladder"`.
- **`--reward simple` / `--shaping` — leftover.** Kept as a control, and
  explicitly inert under both rewards anyone trains with: every `_shaped_value`
  call site sits inside the branch `projected` and `five-term` do not take
  (`cr_sim/train/run.py:229-237`, and see
  [`reward-schedule.md`](reward-schedule.md)). `config.json` records
  `shaping_is_inert` `:648` rather than removing the key, because deleting a
  key would make every new run unpairable with every old one.
- **`--opponent idle` — leftover.** The scale two rounds of invalid
  comparisons were made on. The control wins 92% of idle matches and 26% of
  random ones. Kept only so an old command still means what it meant
  (`cr_sim/train/run.py:316-321`, `cr_sim/train/evaluate.py:701-709` — where `idle` is
  still the CLI **default**).
- **`PPOConfig.gamma` and `clone.collect(gamma=)` — ghosts of a different
  kind.** Unreachable knobs, not dead code. See [`ppo.md`](ppo.md).
- **`policy_proposer(battle_seed_of=...)` — ghost.** No caller. See
  [`search-bot.md`](search-bot.md).
- **`scripts/expert_iterate.py` — live code, no round ever completed.** It
  drives collect, clone and rate as subprocesses (`:156`, `:171-186`) and
  writes `runs/expert-iteration.json` at `:192-194`. There is no `runs/iter-*`
  on disk, no `runs/expert-iteration.json`, and no `data_cache/demos-iter*`.

## Connected to

- **owns:** nothing. Every entry is a pointer.
- **owned-by:** the cards that own the live half of each knob —
  [`ppo.md`](ppo.md), [`reward-schedule.md`](reward-schedule.md),
  [`search-bot.md`](search-bot.md), [`verdict.md`](verdict.md).
- **joins:** [`lift.md`](lift.md) — `--opponent idle` is a *scale*, and the
  reason `write_verdict` demands `eval_opponent` at all.
- **looks-like-but-is-not:** a **deliberate ghost**. Nothing here is one.
  A deliberate ghost is a thing the game build has and this engine declines on
  purpose, as a tripwire; these are things this project built, tried, and
  either reverted or left as a control. The universe column distinguishes
  them (`../../_meta/schema.md`).

## If you change this

- **Hits:** nothing downstream, which is the point of the card — every entry
  here is reachable only from a flag nobody passes or a paragraph nobody
  executes. Implementing one is a new feature, not a fix.
- **Does not hit:** the scale of any recorded number, with **one exception**.
  Removing `--opponent idle` or `--reward simple` would make an old command
  fail rather than silently mean something else, which is the safe direction;
  *changing* what either one does is the unsafe one, because every historical
  verdict measured under them stays on disk with its scale named and nothing
  re-validates it. The obvious next move — deleting a leftover because no
  current run uses it — is what `cr_sim/train/run.py:316-321` and `cr_sim/train/evaluate.py:701-709`
  refuse in prose.

## Surfaces

| Surface | Role |
|---|---|
| `docs/training.md` | describes the reverted γ plan; the only place it exists |
| `cr_sim.train.run --help` | the only place a leftover knob is visible |
| `tests/` | keep the leftovers alive; a leftover here is usually the only thing checking a real invariant |
| a later agent reading a docstring | the reader this card exists for |

## See

- Source: `cr_sim/train/run.py`
- As-built: `docs/training.md`
