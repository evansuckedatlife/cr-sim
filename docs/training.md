# Training notes

The record of what was measured after behavioural cloning first produced a
policy that could play, and what to do next. Numbers here are 150 paired
battles against a **random** opponent at tower level 5 unless said otherwise,
with **sampled** and **greedy** reported separately, because on this task they
are two different policies and have been confused for each other more than
once.

## Say which opponent, and say which arm

Two measurement traps, both of which have already cost this project a round of
invalid comparisons.

**Which opponent.** The in-run evaluation probe played against an opponent that
never plays a card, while every large paired verdict played against one that
spends its elixir on legal placements. Both were written down as "lift" and
compared to each other. The random control wins 92% of the idle matches and 26%
of the random ones, so the two numbers never lived on the same scale. The probe
now faces a random opponent, `opponent_name` reads the opponent off the
environment rather than taking it on trust, and `check_lift_is_named` refuses to
write a metrics row that carries a lift without one.

**Which arm.** A run's `noop_fraction` is the *sampled* pass rate during
rollouts. A clone's reported `play_rate` is the *argmax* play rate on held-out
demonstration states. They are numbers about two different distributions, and
comparing one to the other is how "PPO is drifting toward passing" and "the
clone passes about as often as the expert" ended up looking like a
contradiction. Measured on one fixed set of 1,094 held-out states, with the
search expert passing on **44.2%** of them:

| policy | pass, greedy | pass, sampled | entropy |
|---|---|---|---|
| `runs/cloned` | 48.9% | 11.4% | 3.795 |
| `runs/ppo-from-clone` (checkpoint) | 61.0% | 31.7% | 2.712 |
| `runs/ppo-from-clone` (best) | 63.6% | 36.4% | 2.453 |

Most of the alarming 8% → 36% rise in `noop_fraction` is the sampled
distribution concentrating, not the policy changing its mind: entropy fell from
3.795 to 2.453 while the sampled rate chased a greedy rate that already stood
at 48.9%. But the greedy rate moved too, by fifteen points, and ended nineteen
points past the expert's. Both readings are true and only the pair of them says
what happened.

## pass_weight is not the lever it looked like

`CloneConfig.pass_weight` down-weights a "play nothing" example against a
placement. It was set to 0.1 because at 1.0 cross-entropy parks its mode on the
pass class and the greedy policy never plays. The hypothesis worth testing was
the opposite one: that 0.1 biases the clone away from the expert's real
behaviour, and that the *sampled* arm's weakness (+0.734 against greedy's
+1.623) is that bias showing.

It does move the sampled pass rate onto the expert's. It costs more than it
buys. All at 20 epochs, hard labels, 150 paired battles against the random
control:

| pass_weight | pass greedy | pass sampled | greedy lift | sampled lift |
|---|---|---|---|---|
| 0.1 | 48.9% | 11.4% | **+1.623** [+1.391, +1.855] | **+0.734** [+0.501, +0.966] |
| 0.5 | 91.0% | 35.5% | −2.356 [−2.617, −2.096] | +0.421 [+0.150, +0.691] |
| 1.0 | 94.4% | 44.5% | −2.456 [−2.693, −2.218] | +0.058 [−0.208, +0.324] |

At 1.0 the sampled pass rate is 44.5% against the expert's 44.2% — matched
almost exactly — and it is the *worst* sampled policy of the three, with an
interval touching zero. Both arms fall monotonically as the weight rises, and
neither interval overlaps 0.1's.

The reason is that `pass_weight` does not move "when to pass" on its own; it
moves the pass logit and drags the whole distribution with it. Matching a
summary statistic of the expert is not the same as behaving like it.

**Epochs are a confound, and a large one.** An earlier version of this sweep ran
12 epochs and every cell came out degenerate — greedy pass rates of 76%, 95%,
100% and lifts of −1.318, −2.802, −2.728. The 0.1 cell differs from
`runs/cloned` only in running 12 epochs instead of 20, and scores −1.318 against
+1.623. The pass class is learned in the first few epochs and placement takes
the rest, so any comparison of `pass_weight` made before convergence is a
comparison of undertrained policies.

## What the demonstrations were actually teaching

Two faults, neither of which any training run could have reported.

**The soft target was inert.** `merge()` joined shards without passing `target`
through, so every clone this project ran took the `target is None` fallback and
fitted the single move played. The loader read it, the saver wrote it, `clone()`
branched on it; the one function in the middle dropped it. That is the third
mechanic here to ship doing nothing while its tests passed.

**And wrong where it was not inert.** The softmax over candidate values is
scaled by their own standard deviation, which makes it scale-free: values equal
to four decimal places produce a *confident* preference for whichever one
rounded highest. Measured over 120 real decisions, the candidate spread is 0.10
where the bot plays and 0.00014 where it waits — three orders of magnitude
apart, nothing in between — and where they tie exactly the target is uniform
over the fifteen-odd candidates, fourteen of which are placements. Across the
420 recorded episodes, **86% of the states where the expert waited carry an
exactly uniform target**, and the pass action is the target's argmax in **none
of 10,940 rows**. A policy trained on it played a card at every single decision.

Fixed by scoring waiting with the same patience margin the decision rule
required of a play, and by falling back to the chosen action below a spread
floor. The demonstrations on disk predate the fix, so `clone_policy.py`
defaults to `--targets hard`; sets collected from here on can use `soft`.

## The reward pays for waiting

`--reward projected` scores a board as `crowns + tower_fraction + 0.3 ×
elixir_lead`. The elixir term is what makes a card cost something, and it is
also the term that makes passing pay: spending drops the potential immediately,
while the card's effect on the board takes longer than the three-second
projection to appear. Measured on the clone's own self-play rollouts over 2,520
decisions where a real choice existed, the immediate reward for passing exceeds
that for placing by:

- **+0.0714 ± 0.0364** at elixir weight 0.3 (cloned policy)
- **+0.0367 ± 0.0181** at elixir weight 0.3 (`ppo-from-clone`)
- **−0.0100** at elixir weight 0.0 — placing pays more

The searching bot needed the same number at 0.0 for the same reason: at 0.3 it
drew 100% of its matches because it never played a card at all. `--elixir-weight`
now exposes it. Potential-based shaping leaves the optimal policy unchanged only
in the limit; PPO sees it through GAE at λ=0.95 with a critic explaining 3% of
the return variance, and at that point the immediate term is most of the signal.

Two things that are **not** the cause, both checked rather than assumed:

- **The entropy bonus.** Over 8 chunks of one 1,536-sample rollout, one update
  each: the full loss moves mean P(pass) by **+0.0043** [−0.0017, +0.0103], the
  policy gradient alone by **+0.0048** [−0.0022, +0.0118], and the entropy term
  in isolation by **−0.0504** [−0.0549, −0.0459]. Removing the entropy bonus
  makes the drift slightly *larger*. Uniform-over-legal entropy is 4.47 nats
  against the policy's 3.65, so the bonus pushes toward spreading mass over ~99
  legal actions, of which passing is one.
- **A trust region being necessary.** `--kl` exists and is tested, but see the
  pass-rate table above: the fine-tune was moving toward the expert's rate, and
  an anchor is only worth having if the reference is worth being near.

## Which action head

Three parameterisations of the same masked categorical over 720 actions, so
the mask, sampling, argmax and losses are identical and only the head differs.
Cloned from the same 420 episodes, 20 epochs, `pass_weight` 0.1, hard labels;
150 paired battles against the random control at tower level 5.

| head | policy params | greedy lift | sampled lift | greedy W/L |
|---|---|---|---|---|
| flat | 185,040 | +1.705 [+1.446, +1.965] | +0.758 [+0.502, +1.015] | 85% / 5% |
| **factored** | 57,141 | **+2.167** [+1.962, +2.372] | **+0.964** [+0.734, +1.194] | 96% / 1% |
| conv | 8,709 | +0.840 [+0.530, +1.150] | +0.404 [+0.156, +0.652] | 57% / 18% |

**The factored head wins**, on greedy with intervals that do not overlap the
flat head's, and on sampled with intervals that do. Card first, then tile, with
one tile matrix shared across cards and conditioned on an embedding of the card
actually in the slot.

The interesting part is that agreement did not see it. On held-out states the
two heads reproduce the expert's exact tile 4.8% and 5.1% of the time — a
difference of nothing — while their greedy win rates are 85% and 96%. Imitation
accuracy is a poor proxy for play here, and any future comparison run on
agreement alone will miss effects this large.

**The convolutional head lost, and it should not have.** The trunk's second
convolution already produces a 16x9 feature map, which is the placement grid
exactly; a 1x1 convolution over it emits the placement logits with a
twenty-first of the parameters and full translation equivariance. It ends with
a policy loss of 4.24 against the flat head's 3.67 — it underfits. The head is
almost certainly too small (5 output channels straight off a 64-channel map,
with a 32-wide context vector), not wrong in principle, and it is the one thing
here worth another attempt before the idea is discarded: a hidden 1x1 layer, or
more context width, at a fraction of the flat head's parameters.

### The card-stat head: parity yes, generalisation no

`factored-stats` is the factored head with its card *lookup* replaced by an
encoder over the card's own statistics — hitpoints, damage, reach, speed, what
it targets, what it leaves behind — so the conditioning vector is computed from
what a card does rather than read out of a free column per vocabulary slot. The
argument for it was never a better number on the training deck. It was a
capability the lookup cannot have: a card outside the training decks gets a
conditioning vector for free.

Both arms are cloned from the same corpus (`data_cache/demos_v1v3/v1`, 11,940
decisions from 420 episodes), same recipe — soft targets, 20 epochs,
`pass_weight` 0.1, seed 0 — and scored in one `scripts/evaluate_checkpoints.py`
pass so they share a single control arm. Deliberately not a row in the table
above: that ablation used hard labels and these use soft, so the lookup arm was
re-run rather than differenced against a transcribed number. It came back
+2.126 against the +2.118 on record for this corpus and recipe, which is how
far the protocol reproduces.

| head | policy params | greedy lift | sampled lift | greedy W/L |
|---|---|---|---|---|
| factored (lookup) | 57,141 | +2.126 [+1.933, +2.319] | +0.799 [+0.553, +1.046] | 95% / 1% |
| factored-stats | 59,541 | +2.216 [+2.011, +2.422] | +0.750 [+0.495, +1.005] | 96% / 1% |

**Parity holds.** The intervals overlap heavily in both modes, which is the
pass condition. The encoder costs 2,400 parameters and no measurable training
time — 18.4 minutes against 18.5 for the lookup, run concurrently.

**The capability it was built for does not show up in play.** Both checkpoints
were scored on ten 8-card mirror decks drawn at random from the 114 cards
outside `DEFAULT_DECK`, 150 paired battles each against that deck's own random
control (`scripts/evaluate_decks.py`). Because both arms play the same seeds on
the same decks against the same control, the control cancels exactly and the two
policies can be differenced battle by battle instead of compared through two
overlapping intervals.

| ten unseen decks, 1,500 paired battles | lookup | encoder | encoder − lookup |
|---|---|---|---|
| greedy | +0.726 [+0.648, +0.804] | +0.654 [+0.578, +0.730] | **−0.072** [−0.140, −0.004] |
| sampled | +0.229 [+0.153, +0.305] | +0.216 [+0.139, +0.294] | −0.013 [−0.088, +0.063] |

The encoder is not better on decks it has never seen. It is ahead on 3 of the
10 (sign test *p* = 0.34), and the same head-to-head on the training deck is
+0.091 [−0.07, +0.25] — so if anything it is slightly better where it trained
and slightly worse where it did not, which is the opposite of the hypothesis.

**The mechanism works. It is the mechanism that is not worth anything.** The
head does exactly what it was built to do: holding the observation fixed and
changing only the deck, the lookup head's conditioning vector moves by 0.000 —
it is not merely uninformed about a new deck, it is *misinformed*, applying what
it learned about the Knight to whichever card sorted into the Knight's position
— while the encoder's moves by 1.40 of its own norm.

That change is not swallowed downstream either. Asked about the same states, the
two arms pick the same greedy action 72% of the time on the training deck but
only 47–64% on unseen decks, and the same *card* 75–91% against 95%. They
genuinely play differently on an unseen deck, on roughly half the decisions —
and none of that difference is an improvement. Knowing what a card does is not
the binding constraint. The rest of the network is still out of distribution: 80
of `vector.0.weight`'s 102 input columns are card-identity one-hot bits that
neither head re-keys, and nothing in the observation tells the trunk what the
units now on the board actually do.

**The obvious confound was controlled, and it was not the story.** Eight cards
drawn uniformly from the 114 average 4.09 elixir against `DEFAULT_DECK`'s 2.50,
so an unseen-deck result otherwise mixes unfamiliar cards with an economy 64%
more expensive than either policy ever played — and within the ten decks, lift
did correlate with cost (*r* = −0.61 between the lookup's lift and mean elixir,
*n* = 9, which is not significant). Repeating the whole sweep on six decks drawn
to within 0.4 elixir of the training deck (`--cost-window`) settles it the way a
correlation cannot:

| six cost-matched decks, 900 paired battles | lookup | encoder | encoder − lookup |
|---|---|---|---|
| greedy | +0.641 [+0.543, +0.738] | +0.492 [+0.400, +0.584] | −0.149 [−0.238, −0.060] |
| sampled | +0.245 [+0.150, +0.340] | +0.246 [+0.148, +0.344] | +0.001 [−0.098, +0.100] |

Matching the economy rescues nothing. The lookup arm scores +0.641 on
cost-matched decks against +0.726 on the uniform draw and +2.126 on the deck it
trained on, so roughly two thirds of the policy's edge is lost to the *cards*
being different, not to their being expensive. The encoder's greedy deficit is
if anything larger here, and its sampled gap is exactly zero.

**What this does not settle.** One seed per arm. Parity is robust to that — the
intervals overlap far too much for a seed to be the story — but a head-to-head
gap this small is within what a different initialisation could produce, so "not
better" is established and "slightly worse" is not. Note also that neither
sampled row above is a result at all: two runs of one checkpoint over identical
battles differ by up to 0.17 sd from the sampling stream alone (see
`scripts/measure_sampled_noise.py`), which is wider than any sampled difference
here. The honest next step is not more battles, it is the trunk: until the
observation stops encoding card identity as a position in a per-episode
vocabulary, no change to the head alone can make an unseen deck work.

## Which observation changes helped

Recorded off one playthrough with `collect(variants=...)`, so all four
encodings see identical states, identical expert decisions and identical
labels — the comparison is paired, not four experiments.

| variant | greedy lift | sampled lift | expl. var | on-plays agreement |
|---|---|---|---|---|
| v1 | −1.598 [−1.869, −1.328] | +0.124 [−0.087, +0.335] | 0.538 | 2.9% |
| swarm | **−0.898** [−1.169, −0.627] | +0.083 [−0.145, +0.310] | **0.571** | 0.8% |
| spells | −1.598 [−1.862, −1.335] | +0.000 [−0.232, +0.232] | 0.509 | 2.9% |
| v2 | −1.664 [−1.945, −1.383] | +0.033 [−0.198, +0.264] | 0.556 | 0.8% |

**Read this as underpowered, not as a verdict.** The paired set is 150 episodes
and 4,494 decisions, a third of the full corpus, and at that size 20 epochs is
not enough for a greedy policy to converge — every arm here loses, exactly as
the 12-epoch `pass_weight` cells did on the full corpus. The sampled arms are
all indistinguishable from each other and from zero.

What survives that caveat: the **body-count channels are the only change that
moved anything**. They lift the greedy arm from −1.598 to −0.898 with
non-overlapping intervals and raise the critic's explained variance from 0.538
to 0.571, which is the direction that matters most given a critic that explains
nothing in every fine-tune. The spell channels are indistinguishable from
nothing, and v2 — all four flags at once — is no better than v1, so the
hiding flags are at best neutral at this corpus size.

The honest next step is to re-collect at full scale rather than to conclude.

## The workers never got the tower level

`run.py` built its `VecEnvConfig` without passing `tower_level`, and
`VecEnvConfig` defaults it to 11. `_env()`, which builds the local probe, passed
it correctly. So the flag was honoured on the path that *measures* and dropped
on the path that does 100% of the training: any run launched
`--tower-level 5 --workers N` trained every rollout at level 11 while its
`config.json` recorded 5 and its evaluation probe ran at 5.

This document already said what that costs, three sections up: at level 11 the
towers outlast a 120-second match, ~90% of battles draw, crowns almost never
fire, and everything learned comes from shaping alone. `--tower-level` was added
to fix exactly that, and then never reached a worker.

Every logged statistic matches a level-11 probe and none matches level-5:

| | logged | L11 probe | L5 probe |
|---|---|---|---|
| `ret_std` | 0.201 | 0.217 | 0.654 |
| `value_loss` | 0.010–0.054 | 0.047 | 0.426 |
| rollout win rate | 0.025–0.11 | 0.063 | 0.291 |
| steps/episode | 27.7 | 28.0 | 25.3 |

Worker towers measure [4224, 2576, 2576] HP against the parent's
[3072, 1848, 1848]. `learn-1m-factored`, `learn-1m-flat` and `learn-1m-aborted`
all record `tower_level: 5` and all sit in the level-11 band, so **no number any
of them produced describes the configuration it claims**. Nothing in `tests/`
mentioned `tower_level`, and `test_train.py`'s `main()` call runs without
`--workers`, so it took the in-process path and could never have caught it.

Fixed, with a test that pins `tower_level` and a second that asserts the worker
config agrees with the probe env field for field, so the next dropped field is
caught by construction. After the fix, rollout win rate went **4% → 28%**: the
crowns fire now, and there is a gradient to follow.

A second scale error hides inside the first. The in-run probe evaluates at
level 5 against a random opponent while the rollout metrics come from level-11
self-play, so `win_rate` and `eval_lift_sd` on the same row describe different
games. `check_lift_is_named` guards the opponent half of that; the level half
was unguarded.

## Why explained variance is zero, and what it is not

Three hypotheses were tested independently. All three were refuted, and the
machinery is sound:

- **The metric is correct.** `ppo.py:405-409` computes `1 - Var(ret-value)/Var(ret)`
  over the whole flattened batch on raw returns, matched against an independent
  implementation to 3e-7 across four synthetic cases. GAE matches a brute-force
  reference to 1.4e-7 including done masking.
- **The critic fits.** Driving the real `_update` with the live hyperparameters
  on a learnable target moves explained variance −0.167 → +0.442 in one update
  and +0.945 in eight. There is no value clipping; advantage normalisation
  touches only `advantage`; the KL block runs under `no_grad` and contributes
  nothing to the critic; critic gradient norms are 0.015–0.061, nonzero.
- **The critic arrives trained.** `clone.py:444` trains a value head at
  `value_coefficient` 0.5 on γ-discounted returns with γ matching PPO's. All 12
  critic tensors are in the checkpoint, `strict=True` load matches every key,
  and on demo states it scores +0.685 against its own targets.

What is true instead, in two parts.

**The ceiling is low by construction.** `projected` is an exact potential
reward — `r = Φ(s') − Φ(s)`, verified to 2.2e-16, with every episode's summed
reward equal to Φ(s_T) to 4.4e-16. Returns therefore telescope to
Φ(s_T) − Φ(s_t), and Φ is close to a martingale: regressing Φ_T on Φ_t gives
slope 1.03–1.07 with intercept ±0.02, so E[G|s] is nearly constant and
R²(return-to-go | Φ_t) is +0.0027. Handing the critic the *exact* reward
potential scores −0.0011. `reward.py` says the same thing in its own comment:
it "explained six per cent of the variance in returns". Note also that
`ret = adv + value`, so a constant critic scores exactly 0.0 — "EV ≈ 0" reads
literally as "the critic is a constant".

**But the critic is below even that ceiling.** Branching the engine at
on-policy states and playing K continuations to the end puts Var(E[G|s])/Var(G)
at 0.135 [0.034, 0.257] at level 11 and **0.290 [0.153, 0.418] at level 5**,
against a measured explained variance of +0.006. A six-parameter linear read of
Φ and its parts scores +0.026 out-of-fold — better than the trained 64-channel
critic. Ridge over all 754 live v1 observation features gives out-of-fold R² ≤ 0
at every λ from 1 to 1e5, so **the remaining signal is not linearly accessible
from what the network can see**: v1 carries hitpoint mass only, and the
body-count, spell and threat channels that distinguish a Musketeer from a
Knight are off.

That is the case for `v3`. It is also why the level-5 fix matters twice: it
roughly doubles the variance that is knowable at all.

## Annealing the shaping, and the knob that is not `--shaping`

Straight out of the section above. If the explained-variance ceiling is a
property of the *reward* — Φ near-martingale, R²(return-to-go | Φ_t) = 0.0027
— then no critic and no observation fixes it, and the way out is to stop
paying the shaping at all by the end of the run. Episode-return variance
decomposes 77.4% crowns / 8.4% tower health, so the sparse objective is where
the signal already is.

**`--shaping` is not that knob and never was.** It is read at five places,
every one inside the `else` of `if self._reward is not None:`, so it does
nothing unless `--reward simple`. Measured on identical seeds and an identical
action stream, 0.01 against 5.00:

```
projected: 0.01 vs 5.00 -> IDENTICAL
five-term: 0.01 vs 5.00 -> IDENTICAL
simple:    0.01 vs 5.00 -> DIFFERS
```

A 500× change is bit-identical under both rewards anyone trains with, and
`CRSimEnv`'s own class docstring used to recommend annealing it — "a one-line
change wherever this constructor is called". Following that under `--reward
projected` gives a run that reports an anneal and performs none. The docstring
is corrected; `config.json` now records `reward_schedule.shaping_is_inert`.

What `--anneal` actually moves, per reward:

| `--reward` | annealed to zero | held |
|---|---|---|
| `projected` | `ProjectionWeights.tower`, `.elixir` | `horizon_seconds` |
| `five-term` | the five non-crown `RewardWeights` | `crowns` |
| `simple` | `--shaping` — the one case it is real | — |

`crowns` is never annealed: that is the objective, not shaping. At the zero
endpoint the episode return equals the final crown difference *exactly* —
returns 2.0/3.0/3.0/2.0 against crown differences 2/3/3/2 — so the schedule
terminates on the sparse objective through the same code path, not a special
case. `ProjectionWeights.tower` had no flag at all and was pinned at 1.0; it
is now `--tower-weight` and is recorded.

Linear, on the steps axis, ending at 80% of `--steps`. Linear because the only
consumer that cares about the *shape* is the critic, which carries its scale
across updates while the actor is scale-free (advantages are normalised per
minibatch); a constant drift is trackable, an exponential dumps the change
where the critic is furthest behind. Steps because `--resume` keeps the step
count and replays update indices. Ending early because otherwise the final
checkpoint is measured under a weight that was still moving.

**The weight changes only at `reset()`.** Mid-episode the reward stops being
potential-based: `_previous` holds Φ under the old weight, so one arbitrary
action is paid Φ_new(s') − Φ_old(s). Measured switching to (0, 0) at step 5:

```
step 5 reward:  no-switch -0.007802   switched -0.159656   spurious -0.151854
```

19× the genuine reward, and invisible in aggregate — the episode return still
telescopes correctly to its own endpoint weights, so the existing telescoping
test stays green over it. `CRSimVecEnv.set_reward_weights` is a fifth worker
RPC for the same reason `set_opponent` is one: `VecEnvConfig` is frozen and
pickled once per worker, and a field `_env()` sets while the workers do not is
exactly the `--tower-level` bug. Each env adopts at its own next reset, so an
update straddling a schedule step carries two weights — a row records the
weight *pushed*, which is a target, not a per-battle fact.

**The probe's reward is now pinned** (`run.EVAL_REWARD`), not built from the
training reward. Otherwise the policy arm's returns shrink with the schedule
while the cached control keeps its own scale, and promotion walks back toward
the earliest, highest-shaping checkpoints. As a side effect `eval_lift_sd` is
comparable across runs with different training rewards for the first time —
and `check_lift_is_named` now refuses a lift with no `eval_reward`, the same
guard one axis over.

**Free win, landed first so the anneal is not credited with it.** `env.step`
scored the board and then the run-out scored it again: exactly **2.00 score
calls per non-terminal decision** under `projected`. `score()` is a pure
function of state, so (Φ_mid − Φ_prev) + (Φ_end − Φ_mid) = Φ_end − Φ_prev and
the first is waste. Skipping it is bit-identical — measured 2.09/2.04/2.06/2.05
score calls per step before, 1.09/1.04/1.06/1.05 after, episode sums equal to
twelve decimal places — and removes 48.7% of all projections, about 26% of
environment wall time.

**What is not shown.** That annealing produces a better policy. That needs a
paired A/B of two full 1M-step runs at 28.0 steps/s: ~9.9 h each, ~20 h
sequential, and they cannot overlap because one run already occupies eight
workers. The tests prove application, boundary, worker delivery, endpoint
exactness and probe pinning, and nothing more.

The cheap falsification is available at 80% of a single run, not at the end of
two. With the shaping gone, return-to-go is the sparse crown outcome and is no
longer a telescoped martingale, so `explained_variance` should *rise above
0.29*; early in the anneal it should *fall*, the critic chasing a moving
scale. If EV is still pinned near 0.29 with the weight at exactly zero, the
anneal did nothing and the ceiling was the critic or the observation after
all. Every row carries `explained_variance`, `ret_std` and the active
`reward_weights`, so that reading is taken off the data rather than argued
about.

**Not done, deliberately: the γ-correct potential.** `r = γΦ(s′) − Φ(s)` is
the policy-invariant form for γ<1, and it was written and reverted here once
already. The revert was right, but **half of the reason it was given for is no
longer true and should not be re-derived from this paragraph.** It used to say
"2 scores per decision under `projected` and ~9 under `five-term`". The
`projected` half died with the double-score fix: `CRSimEnv.step` now skips
scoring a state that a telescoping reward is about to score again at the end of
the run-out, and the count is measured at **1.038 score calls per decision**
under `projected` — three episodes, random play, tower level 5, frame skip 30,
78 decisions, 81 calls, the extra three being one reset apiece. `five-term`
does not telescope and still scores more than once a decision: **3.115** in
that same configuration. Do not read that as the 9× the old sentence claimed;
the multiplier is a function of how many forced decisions the run-out
compresses, which moves with `frame_skip` and with the opponent, so it is a
number for whichever regime is being argued about and not a constant.

What survives, and it is enough on its own: applied inside
`RewardTracker.step` / `ProjectedReward.step`, γ is charged per *score*, and a
score is not a PPO timestep. The run-out compresses several engine decisions
into one `env.step`, so a γ per score prices time the agent never chose either
way. If it is wanted it belongs at the single point the reward crosses into the
trainer, and it is a separate change, not a prerequisite for this one.

## What to do next

Specific to one machine at roughly 46 decisions a second, not a literature
summary.

**Re-collect the demonstrations, at full scale, with the fixed targets.**
*Done, and in flight as `data_cache/demos_v1v3`:* 420 episodes over four
shards, recording v1 (9 channels) and v3 (17) off the same playthroughs, under
`--reward projected --elixir-weight 0 --tower-level 5`. Three things rode on
it. The soft target had never once been used; the observation ablation was
bounded by a corpus a third the size of the one the heads were compared on, at
which size nothing converges; and the value targets came from the wrong reward
entirely (below). `collect(variants=...)` records every encoding off one
playthrough, so v1 and v3 see identical states, identical labels and identical
decisions -- which makes the encoding comparison a paired experiment rather
than two.

**Then re-run the observation ablation** on that corpus, which is now the
*only* thing standing between the critic's measured +0.006 and the 0.290 of
return variance that is knowable at level 5.

**DAgger, not more episodes.** Behavioural cloning fails on states the expert
never visits, and the cloned policy visits different ones. The standard fix is
to roll out the *learner*, label those states with the expert, and aggregate.
This project can afford it in a way most cannot: the expert is a search over a
deterministic engine that clones in 0.69 ms, so it can be queried on any state,
at any time, with no human in the loop. Two or three rounds of a few hundred
episodes each, weighted toward states the learner reaches, is a better use of
two hours than doubling the corpus.

**Keep the fine-tune, drop the framing.** PPO from the clone did improve the
sampled policy. The thing to watch is not `noop_fraction` but the greedy pass
rate against the expert's 44.2%, measured on held-out states — a number that
costs one forward pass and that no run currently logs.

**Fix the critic before adding a league.** *Half done.* The reward mismatch is
fixed: `make_demos.py` now takes `--reward`, `--elixir-weight` and
`--reward-horizon-seconds`, defaults to the fine-tune's shape, and stamps what
it used onto every shard, which `clone_policy` verifies rather than trusts. The
cloned critic no longer arrives predicting +1.48 against returns averaging
+0.47.

That was never the reason explained variance is zero, though -- see the section
above. It is offset-invariant by construction, so a constant +1.5 error is
invisible to it, and PPO re-fits the scale within five updates anyway. The
remaining half is the **observation**: the signal is not linearly accessible
from v1's hitpoint-mass channels at any ridge penalty, so the next measurement
is the v3 clone against the v1 clone on the paired corpus. Population-based
training and AlphaStar's league both spend their compute on opponent diversity,
which is the wrong bottleneck while the critic explains nothing.

**Self-play needs a sharp opponent.** A frozen snapshot sampled at temperature
1.0 is nearly random when the policy's entropy is near uniform, which leaves the
outcome as unpredictable as it was against a random agent. `--opponent-temperature`
exists; it has never been swept.

## The policy proposes, and the search still decides

`SearchBot` drew about fourteen stratified-random placements per decision out
of a mean of **104 legal actions** -- 13.5% coverage, measured -- and the other
86.5% were never scored. The one object in the system holding an opinion about
*which* fourteen deserve an exact engine branch is the policy, and until now it
was never asked. AlphaZero's improvement operator is three arrows; this project
had the second and the third.

`SearchBot(team, config, proposer)` takes the missing one.
`cr_sim.train.proposal.policy_proposer` ranks the legal actions by the policy's
own logits, and `SearchBotConfig.policy_candidates` says how many of the
budget's placements come from that ranking. Everything is off by default:
`policy_candidates=0` with `proposer=None` is byte-for-byte the old bot, which
is checked against a golden captured by running the pre-change source.

**Measured, at the shipped `candidates=14, horizon_seconds=15`, one thread.**

| | random proposal | policy proposal |
|---|---|---|
| branches per decision | 13.47 | 13.50 |
| seconds per decision | 0.511 | 0.402 |
| one `policy_logits` forward | 1.34 ms | 1.34 ms |
| forward as a share of a decision | -- | **0.26%** |
| candidate overlap between the two | -- | 0.34 |

Board for board -- both bots asked about the same 43 positions -- the branch
counts matched on **43 of 43 decisions with no mismatches**. That is the equal
budget claim, and it is arithmetic rather than a promise: the random draw is
taken first and in full, and the proposal *replaces* the front of it. Filling
the remainder at a reduced budget instead does not work, and looks like it
does: `per_slot` is `max(1, budget // slots)`, so asking for two placements
across four legal cards still returns four, and a bot built that way took **63
branches where the unguided one took 32**.

**The labels move with the proposal, which is the part that can quietly poison
a clone.** The cloner trains against the search's distribution over the
candidates it actually scored, so changing which placements are scored changes
the supervision. Three defences, all on: a floor of `max(2, candidates // 3)`
random candidates that no proposer can displace; the candidate spread and the
`min_spread` fallback rate recorded into every shard's `meta`; and a refusal
printed by `make_demos.py` when a shard's fallback rate runs more than ten
points past the unguided baseline. `Demonstrations.proposer` joins `observation`
and `reward` in `clone_policy`'s merge guard, because a set whose proposer
varies row to row is undetectable downstream.

The feared failure is support collapse -- the proposer nominates what the
policy already likes, alike placements score alike, the spread falls, the row
collapses to a one-hot on the chosen action, and the clone sharpens a
preference instead of improving one. **It has not happened yet.** Four episodes
at the shipped settings: mean candidate spread **0.0585 unguided against 0.0936
guided**, both at a 0% fallback rate, and the guided expert played on 64% of
its decisions against 50%. One checkpoint, four episodes; the gate stays.

**Determinism.** `temperature=0.0` is a stable numpy argsort and touches no
generator at all, so ties break by ascending flat index --- `torch.topk`
promises no ordering among equal values and the factored head produces exact
ties routinely. Above zero the draw comes from a `torch.Generator` the proposer
owns, seeded arithmetically from (proposer seed, battle seed, decision index),
never from `hash()` and never from torch's global stream. The proposer is
rebuilt per battle alongside the bot, so it is a function of the battle and the
decision number rather than of how many episodes came before.

`scripts/expert_iterate.py` is the loop: collect with round *n-1*'s clone
proposing, clone, rate, repeat. Measured cost of one round at 360 episodes
across six shards is 17-20 minutes of collection, minutes of cloning and about
fifteen of rating --- **under an hour a turn**, which is what makes expert
iteration worth building here rather than describing.
