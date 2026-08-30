---
type: object
cluster: measurement
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/train/ppo.py
---

# explained_variance

The metrics column that says whether the critic beats predicting the mean
return — and the one number in this repo whose **ceiling is set by the reward**
rather than by the network or the observation. No product word; it is
`stats["explained_variance"]`, written once per update and read by four
surfaces.

## Why this shape

PPO's advantages are only as good as this. At 0 the advantage is the
Monte-Carlo return minus a constant: unbiased, and so noisy that the policy
gradient is mostly variance (`cr_sim/train/ppo.py:410-415`). So the number is not a
diagnostic anyone may skip past — it is the readout on whether an update is
credit assignment or distributional sharpening.

It sat near zero for the whole life of this project, and three explanations
were tested and refuted: the metric is correct, the critic fits, and the critic
arrives trained (`docs/training.md:338-356`). What is true instead is the
reason this card is in the reward's neighbourhood rather than the network's.
`projected` is an exact potential, so return-to-go telescopes to
`phi(s_T) - phi(s_t)` and `phi` is close to a martingale, which puts a **ceiling
on how much of the return variance is knowable from the state at all**. Handing
the critic the exact reward potential scores below zero. The ceiling was
measured by branching the engine at on-policy states and playing continuations
out: 0.29 at tower level 5, 0.135 at level 11 — so it moves with the tower
level as well as with the reward (`docs/training.md:359-382`).

That is the whole argument for
[reward-schedule.md](reward-schedule.md), and it is also the falsification: at
the anneal's zero endpoint the return is the sparse crown outcome and is no
longer a telescoped martingale, so this number should **rise above 0.29** — and
if it does not, the ceiling was the critic or the observation after all. Every
row carries the three fields that reading needs, so it is taken off the data
rather than argued about.

## Shape

- `1.0 - Var(ret - value) / Var(ret)` over the whole flattened batch, on raw
  returns, with `nan` when the return variance is under `1e-9`. There is no
  value clipping and advantage normalisation touches `advantage` only.
- `value` is the rollout-time prediction, and `compute_gae` returns
  `ret = advantage + value` **exactly** (`cr_sim/train/ppo.py:140`). So a constant
  critic scores exactly 0.0, and "EV about zero" reads literally as "the critic
  is a constant" — not as "the critic is bad by some margin".
- The three fields that make an anneal falsifiable sit on the same metrics row:
  `explained_variance`, `ret_std` (`cr_sim/train/ppo.py:408`), and `reward_weights`
  (`cr_sim/train/run.py:769-779`), which is written on every row and not only the ones
  that moved.
- **Code wins over one comment.** `docs/training.md:343` cites the computation
  at `cr_sim/train/ppo.py:405-409`; it is at `:416-420` in the tree this was verified
  against. The formula it describes is unchanged ([`../../_meta/overrides.md`](../../_meta/overrides.md), row 16).

Citations: `cr_sim/train/ppo.py:408`, `:410-420`, `:113-140`, `:361-364`;
`cr_sim/train/run.py:769-779`; the ceiling argument
`cr_sim/train/schedule.py:1-14`; the separate clone number
`cr_sim/train/clone.py:23-27`, `:628-631`, `:657`; readers
`cr_sim/train/watch.py:347`, `:3056`, `cr_sim/train/report.py:113`,
`scripts/clone_policy.py:279`, `:396`.
Verified 2026-08-30 against `main` @ `dc47f51`. `clone.py` and
`docs/training.md` are modified in that working tree, so their line numbers are
working-tree numbers.

## Connected to

- **owns:** nothing. It is a derived reading, and the whole point of this card
  is that it owns none of what sets it.
- **owned-by:** [ppo.md](ppo.md) computes it;
  [run-directory.md](run-directory.md) stores it as a `metrics.jsonl` column.
- **joins:**
  [../interface/reward-variants.md](../interface/reward-variants.md) (the
  potential that sets the ceiling), [reward-schedule.md](reward-schedule.md)
  (the intervention aimed at it),
  [../interface/observation-features.md](../interface/observation-features.md)
  (v1 carries hitpoint mass only, which is why the remaining signal is not
  linearly accessible from it), [demonstrations.md](demonstrations.md) (the
  value targets the critic inherits).
- **looks-like-but-is-not:** `clone.py`'s `explained_variance`
  (`cr_sim/train/clone.py:657`) shares the name and is a **different measurement** —
  it is fit against held-out *demonstration value targets*, not against
  on-policy returns, so its denominator is a different variance entirely.
  Putting the two on one axis is bug 6's shape one scale over. `value_loss` is
  the scale-dependent version of the same thing and moves when the reward's
  magnitude moves, which is exactly what this metric exists not to do.

## If you change this

- **Hits:** nothing downstream — nothing branches on it. The direction of
  travel runs the other way, and that is the fact an editor needs: **change the
  reward and this number's ceiling moves with it**, without a line of this file
  or of `nets.py` changing. Raising the tower level moves it too, by roughly
  halving the variance that is knowable at all
  (`docs/training.md:381-382`). If the formula itself is edited, the four
  readers break silently rather than loudly: `cr_sim/train/watch.py:347` and `:3056`,
  `cr_sim/train/report.py:113`, `scripts/clone_policy.py:396`, plus the leftover
  `notify.py` and `bot.py` paths.
- **Does not hit:** the critic. Reaching for `value_learning_rate`,
  `value_coefficient` or a bigger `hidden` is the obvious next move and the one
  the three refuted hypotheses already rule out — driving the real `_update` on
  a learnable target moves this from -0.167 to +0.945 in eight updates, so the
  machinery is not what is holding it down (`docs/training.md:347-351`). It
  also does not hit the lift: `eval_lift_sd` is a difference of returns against
  a control and is measured under a pinned reward, not under the critic.

## Surfaces

| Surface | Role |
|---|---|
| `cr_sim/train/ppo.py:418` | writes — once per update |
| `runs/<name>/metrics.jsonl` | stores — beside `ret_std` and `reward_weights` |
| `cr_sim/train/watch.py:347`, `:3056` | reads — the "expl var" panel, zero-anchored |
| `cr_sim/train/report.py:113` | reads |
| `scripts/clone_policy.py:279`, `:396` | writes — the *clone's* different number, under the same key |
| `cr_sim/train/notify.py`, `bot.py` | reads — leftover surfaces, still tested |
| humans | reads — this is the number the whole anneal is a bet on |

## See

- Source: `cr_sim/train/ppo.py:406-420`; the potential that sets the ceiling,
  `cr_sim/api/reward.py:319-347`
- As-built: `docs/training.md`, "Why explained variance is zero, and what it is
  not" — working-tree line numbers; that file is modified in the tree this card
  was verified against
