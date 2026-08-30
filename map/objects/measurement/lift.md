---
type: object
cluster: measurement
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/train/evaluate.py
---

# Lift

One arm's paired per-battle advantage over the control that played the same
seeds, divided by the control's own spread. `paired_lift` in code; `lift` in a
`verdict.json`, `eval_lift_sd` on a metrics row, `mean_lift` / `best_lift`
after `report.py` has aggregated it.

Verified 2026-08-30 against the working tree at `dc47f51` (nine files
uncommitted — `../../CONTEXT.md`, Verification basis).

## Why this shape

The per-episode spread is several times larger than any effect worth seeing,
so the difference is taken **per battle and only then averaged**; unpaired
means would need far more episodes to say the same thing
(`cr_sim/train/evaluate.py:332-337`). The denominator is the control's spread
because a raw gap is unreadable without knowing how noisy the control is — and
that choice is the trap. Against a strong opponent the control loses every
battle the same way, the spread collapses, and every lift measured through it
inflates in proportion. That is why the spread is *recorded*, not inferred
(`:344-350`, `:427-435`).

This is bug 6's home. A lift is a function of six things and is meaningless
without them: **the control**, **the reward**, **the arm**, the episode count,
the tower level and the seed block. The reward is in the numerator *and* the
denominator, because both halves are returns.

Bug 6 is committed twice over: once by an editor changing the arithmetic, and
once by a reader putting two existing numbers on one axis. Only the first
arrives at this card by editing a line. The second arrives at
[`../../effects/quoting-a-result.md`](../../effects/quoting-a-result.md), which
holds these six as a refusal checklist.

## Shape

- Arithmetic: pairing, spread, CI. `cr_sim/train/evaluate.py:330`, spread
  `:359-360`, keys `:365-373`.
- Six inputs. Control: `evaluate(env, None, ...)`, uniform over the *legal*
  mask, `:176-181`. Opponent both arms face: `_opponent_for`, `:676`. Reward:
  read off the env, `:409-411`. Arm: greedy and sampled are separate and never
  merged, `:377-473`. Seeds: `evaluation_seeds`, `:296`, blocks drawn from
  `default_rng(12345)`; block 0 is byte-identical to the fixed list every
  historical number sits on, `:311-317`. Tower level: **not** an input to the
  arithmetic and not recorded — see the gap below.
- **Three denominator conventions and ten sites that compute one.** Enumerating
  them in prose here is what drifted — they live on
  [`lift-callers.md`](lift-callers.md), which is built from a caller search over
  `cr_sim/`, `scripts/` and `tests/` rather than from memory.
- The sampled arm's noise floor is **0.062 sd**, not the 0.04 quoted
  elsewhere: one checkpoint, four streams, the same battles and control.
  `cr_sim/train/evaluate.py:157-167`; the measurement is
  `runs/sampled-noise-floor/noise.json`, the script `scripts/measure_sampled_noise.py`.
  Greedy reproduced exactly over the same battles, which is what says the
  spread is the sampling and not the seeds.
- **Gap, no guard behind it: nothing refuses a verdict that omits
  `tower_level`.** At level 11 the towers outlast the match and the battles
  draw, so two lifts from two levels are not one scale. Which entry points
  choose which arena, what writes it down and what checks it are on
  [`tower-level.md`](tower-level.md) — one home, because this card and
  `../../effects/CONTEXT.md` each used to carry a different count of them.

## Connected to

- **owns:** the `control` block of a verdict, including `spread`
  (`cr_sim/train/evaluate.py:427-435`).
- **owned-by:** [`verdict.md`](verdict.md) — the file that has to name the
  opponent and the reward before this number may be written down.
- **joins:** [`random-streams.md`](random-streams.md) (the sampled arm is only
  reproducible if it owns a generator); [`self-play.md`](self-play.md) and
  [`ladder.md`](ladder.md) (the two in-run probes that emit `eval_lift_sd`);
  [`run-directory.md`](run-directory.md) (where it lands).
- **looks-like-but-is-not:** `ladder_elo`. Elo is fitted on crowns, which no
  reward touches; a lift is denominated in a reward. `write_verdict` refuses a
  file carrying both without saying whose lift it is
  (`cr_sim/train/evaluate.py:526-545`), and `cr_sim/train/run.py:858-865` refuses to call a rolling Elo
  `rolling_lift`. Also not `ancestor_score` or `ladder_score` — those are
  *scores*, which is exactly why `SCORED_FAMILIES` exists.

## If you change this

- **Hits:** [`lift-callers.md`](lift-callers.md) — **six call sites in
  four files**, one of which is the script that produced the 0.062 noise floor
  this card quotes. That list is a link and not a sentence on purpose: written
  out here it named two of the six and was wrong the next time a script landed.
  Also every `arms.json` row (`scripts/run_ladder.py:394`) and the flattened
  headline keys a verdict carries; `report.py` reads the flat keys and nothing
  else (`cr_sim/train/evaluate.py:465-473`, `cr_sim/train/report.py:126-129`).
- **Does not hit:** the two in-run probes. `evaluation_probe` and
  `rotating_probe` compute their own `eval_lift_sd` inline at `ddof=0` and do
  not import `paired_lift` (`cr_sim/train/selfplay.py:466-479`, `cr_sim/train/evaluate.py:633-654`) —
  the obvious next assumption, that fixing the formula in one place fixes the
  number a run promotes on, is wrong. `scripts/clone_policy.py` and
  `scripts/evaluate_checkpoints.py` also keep their own copies; all four are on
  [`lift-callers.md`](lift-callers.md). And it does
  **not** touch any number already on disk: every recorded lift stays on the
  scale it was measured on, which is why nothing here is retro-fitted.

## Surfaces

| Surface | Role |
|---|---|
| `cr_sim.train.run`'s promotion window | reads `eval_lift_sd`, writes `rolling_lift` (`cr_sim/train/run.py:851-866`) |
| `cr_sim/train/report.py` | reads the flat keys; withholds `mean_lift`/`best_lift` entirely when a run mixed scales (`:82-99`, `:126-129`) |
| `cr_sim/train/watch.py` | the progress page, run-wide `best_lift = max` |
| a human quoting a number | the only consumer that can conflate two controls, and the one bug 6 was about — routed at [`../../effects/quoting-a-result.md`](../../effects/quoting-a-result.md) |

## See

- Source: `cr_sim/train/evaluate.py`
- As-built: `docs/training.md`
