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

## What to do next

Specific to one machine at roughly 46 decisions a second, not a literature
summary.

**Re-collect the demonstrations, at full scale, with the fixed targets.**
Two things ride on it. The soft target has never once been used, and the
observation ablation above is bounded by a corpus a third the size of the one
the heads were compared on -- at which size nothing converges and only the
largest effect is visible. Recollecting 420 episodes costs about two hours over
four processes, `collect(variants=...)` records every encoding off one
playthrough, and it unblocks both.

**Then re-run the observation ablation.** Every downstream comparison
here is bounded by a corpus whose soft targets contradict the expert on 44% of
its own decisions and which has never been used with a working soft target at
all. Recollecting 420 episodes costs about two hours over four processes and is
the cheapest large improvement available. `collect(variants=...)` records every
observation encoding off one playthrough, so this is also the moment to record
whatever channels are wanted.

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

**Fix the critic before adding a league.** Explained variance sits at 0.00–0.03
in every fine-tune, and the cloned critic is worse than useless under a
different reward: it predicts +1.48 where returns average +0.47, because
`make_demos.py` records value targets under the *simple* reward and the
fine-tune trains under `projected`. Recording value targets under the reward
that will be trained on is a one-line change and has to come before anything
that assumes advantages carry signal. Population-based training and AlphaStar's
league both spend their compute on opponent diversity, which is the wrong
bottleneck while the critic explains nothing.

**Self-play needs a sharp opponent.** A frozen snapshot sampled at temperature
1.0 is nearly random when the policy's entropy is near uniform, which leaves the
outcome as unpredictable as it was against a random agent. `--opponent-temperature`
exists; it has never been swept.
