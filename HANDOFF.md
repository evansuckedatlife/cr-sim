# Handoff

Where this is, what worked, and what to distrust. Written for someone picking it
up cold — including me, later.

**This supersedes an earlier handoff whose headline was wrong.** It said
reinforcement learning erodes the clone. It does not. That claim came from
comparing two numbers measured on different arms against different opponents,
which is this project's signature failure and is documented below so the next
person recognises it faster than we did.

## The state in one paragraph

The **simulator** is in good shape: ~700 tests, a deterministic integer
fixed-point engine built from the game's own shipped tables, and a battle that
clones in 0.69 ms — which is the one real advantage here, because it lets the
engine roll out its own future. The **agent** works, in this order: a searching
expert beats random 100–0, a supervised clone of that expert reaches +1.623,
and PPO fine-tuning on top of the clone improves the sampled policy from ≈+0.7
to +1.239 — though only the clone's own figures have been re-measured;
see the note under the table. What does *not* work is reinforcement learning
from random initialisation, which four runs and 868,000 steps established.

## Results worth trusting

All eight arms below were measured against the **same random opponent** on the
**same 150 paired seeds** at tower level 5. This is the only table on the
project whose rows are comparable to each other, and producing it is what
overturned the previous handoff.

| arm | win | loss | lift | 95% interval |
|---|---|---|---|---|
| search expert (no learning) | 100% | 0% | **+2.716** | [+2.369, +3.063] (n=40) |
| cloned, greedy | 83% | 5% | **+1.623** | [+1.391, +1.855] ✔ |
| ppo-from-clone, greedy | 81% | 3% | +1.804 | [+1.552, +2.057] |
| **ppo-from-clone, sampled** | 63% | 7% | **+1.239** | [+0.999, +1.478] |
| selfplay-1m latest, greedy | 53% | 13% | +0.775 | [+0.554, +0.995] |
| selfplay-1m best, greedy | 48% | 13% | +0.731 | [+0.502, +0.961] |
| cloned, sampled | 45% | 10% | +0.684 | [+0.440, +0.928] ✔ |
| selfplay-1m best, sampled | 36% | 15% | +0.422 | [+0.197, +0.647] |
| random control | 26% | 27% | — | — |

✔ = re-measured since, and the figure shown is that re-measurement. Both
ticked rows were originally published here with different numbers. **The
unticked rows have not been checked at all.** This table was written as "the
only table on the project whose rows are comparable", which is exactly what
made an error in it expensive:

- **`cloned, greedy` was published here as +1.813 and is +1.623.** That figure
  appeared nowhere on disk — not in a verdict, a metric or a source file — and
  three separate readings (`runs/cloned/verdict.json`, an independent re-run by
  the ML session, and a 150-battle re-measurement into
  `runs/_anchor/cloned-recheck.json`) all agree on +1.623 [+1.391, +1.855]. The
  win and loss rates in the row were right, so the experiment was fine and the
  lift computation was not. A difference of 0.19 sd is not a rounding slip: it
  is most of the gap between this clone and a PPO run built on top of it.
- **`cloned, sampled` was not wrong, and re-running it found the noise floor.**
  Three 150-battle readings of the *same weights* against the *same* control
  gave +0.709, +0.725 and +0.684. Greedy, over the same two runs, reproduced
  bit-identically — 1.6230076626442986 both times. So on this setup **greedy is
  exactly repeatable and sampled carries roughly ±0.02 sd of run-to-run spread
  from the policy's own sampling.** Any sampled comparison closer than about
  0.04 sd is measuring the random number generator.
- **The `ppo-from-clone` rows are not on disk either.** The only saved verdict
  for it, `runs/poc-vs-random/verdict.json`, records `control_win: 0.04` with
  89% draws — a different, draw-saturated regime entirely, not this control.
  Re-score it before quoting +1.804 or +1.239 again.

The lesson is the one this file already teaches, turned on itself: a number
that is not written by the thing that measured it is a number nobody can check.
Point `--out` at a file for every evaluation worth citing.

Three readings, and they are the whole strategy:

1. **Imitation beats scale on one laptop.** 868,000 steps of self-play from
   random initialisation reached +0.775. Twelve epochs of supervised cloning
   from 420 expert episodes reached +1.623, in minutes.
2. **PPO fine-tuning looks like it works, but the greedy half is now unclear.**
   Sampled improved +0.72 → +1.239, and that is the load-bearing claim. The
   greedy comparison was stated as flat on the strength of +1.804 against
   +1.813; with the clone's true +1.623 the two are 1.623 against 1.804, which
   is not flat — but their intervals overlap heavily and the PPO figure has not
   been re-measured, so **"not distinguishable" is the only honest reading
   until `ppo-from-clone` is re-scored with a recorded opponent.** Falling
   entropy was mass concentrating on good actions, not collapse. **Always
   report greedy and sampled separately.**
3. **The pass rate was moving the right way.** The search expert passes on
   **44.3%** of its decisions. `ppo-from-clone` climbed 8% → 27% and never
   reached it, so it was moving toward the expert, not away from playing.

## The order that works

Search expert → demonstrations → behavioural clone → PPO with a KL anchor. Not
reinforcement learning from noise. AlphaStar trained on 971,000 human replays
before any RL and that supervised agent alone outranked 84% of humans; OpenAI
Five ran the same algorithm on 128,000 CPU cores. We have neither, so the third
option in the literature — search, which needs a fast exact simulator — is the
one available, and the engine's 0.69 ms clone is what makes it affordable.

## Numbers that fooled us

Read this before believing any figure in `runs/`.

- **The in-run probe faced an *idle* opponent; the paired evaluations faced a
  *random* one.** Both were called "lift" and compared to each other. The
  control wins **92%** of idle matches and **26%** of random ones. Every
  metrics row written before the fix is against an idle opponent, including all
  of `selfplay-1m` and all of `ppo-from-clone`. `write_verdict` now refuses a
  verdict with no `eval_opponent`, and `report.py`/`bot.py` name the opponent
  instead of labelling everything "beats random".
- **`eval_lift_sd` is the SAMPLED arm** (`selfplay.py` calls
  `evaluate(greedy=False)`). Comparing it against a greedy verdict is comparing
  two different policies. That, plus the idle/random error, is the entire
  content of the "PPO degrades the clone" claim that this handoff replaces.
- **A +0.375 reading over 40 battles measured −0.033 over 300.** Forty battles
  cannot separate a weak effect from zero here. The in-run probe is a trend
  line, not a result.
- **Checkpoint promotion kept the maximum of ~19 noisy readings**, which
  selects for noise. Now a rolling mean of 3 — but it still replays the same 40
  seeds, so consecutive readings share seed-level luck. Rotate seed blocks.
- **92% of matches were draws at tower level 11**, so crowns almost never fired
  and everything learned from shaping alone. Train at `--tower-level 5`.

## Two defects worth knowing about

**The reward pays for passing.** At the default `--elixir-weight 0.3` a pass
earns **+0.071 more reward than a placement** — measured twice, once in
`run.py`'s own help text and again on the clone's rollouts
(`pass − play immediate reward = +0.071 ± 0.036`). The searching bot needed
`elixir_weight=0.0` for the same reason: at 0.3 it never played a card at all.
**Use `--elixir-weight 0` for any RL run.**

**The demonstrations could not teach passing.** The soft target was built from
candidate values scaled by their own spread, and on states where the search
chose to wait those values are equal to four decimal places — so 86% of them
carried an exactly uniform distribution over fifteen-odd candidates, of which
fourteen were placements. **The pass action was the target's argmax in none of
10,940 recorded decisions.** `pass_weight=0.1` was compensating for that.

The recording is fixed and demonstrations were regenerated, but the fixed set
is small (**4,494 decisions** against the original's ~10,900) and clones trained
on it are much worse — greedy +0.445 at `pass_weight=0.1` and +0.240 at 1.0,
against the original clone's +1.623. **The limiting factor is demonstration
volume, not the pass weight.** Regenerating a full-size fixed set is the
cheapest known win available.

## Bugs that shipped green

Each had passing tests while doing nothing. This is the failure mode this
codebase generates, and why so many tests here run the real thing.

- `push_away` was a **no-op for every knockback in the game**.
- **Projectiles never applied the buff they carry.** Ice Spirits dealt damage
  and never froze — the entire card, and one of the eight the agent trains on.
- **The deploy zone never expanded** when a Princess Tower fell.
- **`RelativeX`/`RelativeY` were read as milli-tiles**, a factor of a thousand.
- **The progress page claimed "no evaluations yet"** over a real evaluation,
  and its test passed because that string is in the page source regardless.
- **`UnitSpec.damage_per_second` returned damage per *thousand ticks*** —
  16.67× too large at 60 TPS and a different wrong number at 20. Dead code, so
  nothing broke, but the observation encoder was about to consume it.

## Running it

```bash
python -m pytest                          # ~700 tests
python -m cr_sim.cli validate             # stat gate + open questions
python -m cr_sim.cli engagement --write   # reach + tower-support matrices

# the order that works. --elixir-weight 0 and --tower-level 5 are not optional.
python scripts/make_demos.py --episodes 70 --shard 0     # several shards
python scripts/clone_policy.py --demos data_cache/demos --out runs/cloned
python -m cr_sim.train.run --steps 1000000 --envs 8 --workers 4 \
    --tower-level 5 --reward projected --elixir-weight 0 \
    --init-from runs/cloned/cloned.pt \
    --kl 0.5 --kl-reference runs/cloned/cloned.pt \
    --opponent self --pool-size 8 --eval-every 3 --name my-run

python -m cr_sim.train.watch --every 20 --serve 8899     # phone-friendly page
python scripts/evaluate_checkpoints.py a.pt b.pt --episodes 150   # comparable arms
python scripts/evaluate_vs_expert.py                     # the unsaturated yardstick
```

`--envs` must divide evenly by `--workers`. Data is not in the repo: supply a
Clash Royale APK and run `python scripts/extract_apk.py <apk>`.

## What is running now

- **`runs/learn-1m-flat`** and **`runs/learn-1m-factored`** — a matched pair,
  1M steps each, clone-initialised, KL-anchored at 0.5, `--elixir-weight 0`,
  tower level 5, observation v1. Roughly 17½ hours each while they share the
  eight cores.
- **`runs/learn-1m-aborted`** — dead, kept for its metrics. It ran ten updates
  from `runs/cloned/cloned.pt`, then a restart passed `--resume` together with
  `--init-from` and it refused to start. **Do not resume it** — see below.

The pair differ **only** in the action head. Getting there took two corrections
to how the A/B was first set up, and both matter to anyone rebuilding it.

- The aborted flat run initialised from `runs/cloned/cloned.pt`, which stores
  **no `observation`, `targets`, `pass_weight` or `head` at all** — it predates
  those fields. Pairing that against a documented head-ablation clone would
  have made the head one of two variables. Both arms now start from the matched
  ablation pair (v1, hard targets, `pass_weight` 0.1, same demonstrations, same
  recipe), copied into `checkpoints/` so a worktree cleanup cannot take them.
- The head is **not** "the one lever nobody has measured". It is measured at
  clone scale, in `docs/training.md`: factored **+2.167** [+1.962, +2.372]
  against flat **+1.705** [+1.446, +1.965] greedy, intervals not overlapping,
  at 57,141 head parameters against 185,040. These runs ask something narrower
  and more interesting — whether that advantage survives PPO.

That clone-scale comparison also carries the most useful methodological finding
on this project: flat and factored reproduce the expert's exact tile on
held-out states **4.8% and 5.1%** of the time — a difference of nothing — while
their greedy win rates are 85% and 96%. **Held-out agreement cannot rank these
policies.** Any comparison scored on it calls this a tie.

## Recently landed

All of this was uncommitted when the previous handoff was written. It is now on
`main` — it had been sitting in the working tree with no session left to commit
it, which is how work went invisible after the last restart.

- **`threat` observation set** (`aae0532`) — reach and damage-per-second grid
  channels, as a named variant. **v2 was frozen at 13 channels and v3 added at
  17**, because a self-updating "v2" silently redefined what old checkpoints
  meant and let them sail past `check_observation` into a raw shape mismatch. A
  named version must mean the same thing forever.
- **Measurement fixes** (`e19e41b`) — `write_verdict` now refuses a lift with
  no `eval_opponent`, and `report.py`/`bot.py` name the opponent instead of
  labelling every positive result "beats random". `rotating_probe` cycles eight
  seed blocks, so a rolling window of three covers 120 distinct battles rather
  than three readings of the same forty. `clone.py` value targets are indexed
  by their own decision, not by even striding. Both scripts are tracked.
- **The all-time page** (`852b030`, `dce8338`) — lifetime model performance and
  games played, with every number carrying the control it was read against.

Suite is **852 passed, 1 skipped** on that tree.

A `reward-gamma` fix was written and **reverted**: the shaping really is
`φ(s') − φ(s)` where policy invariance needs `γφ(s') − φ(s)`, but `env.py`
scores the potential 2× per decision (9.23× for five-term), so charging the
discount per score over-corrects — measured as a net regression on 6/6 seeds,
with sign flips on 2. Re-land it *with* the `env.py` change that charges once
per recorded transition, or not at all.

## Open threads, in the order the numbers point

1. **Regenerate a full-size fixed demonstration set.** The pass-target defect
   is fixed but the corrected set is 40% the size and clones badly. Everything
   downstream inherits the clone's quality.
2. **Finish the flat-vs-factored A/B**, now running as `learn-1m-flat` and
   `learn-1m-factored` from the matched ablation clones.
3. **Re-score `ppo-from-clone` with a recorded opponent**, so the table above
   can be trusted end to end rather than in the four rows carrying a ✔.
4. **Measure the `threat` channels** — but not for the reason previously given
   here. The argument was that the clone matches the expert's placement on only
   5.4% of the expert's own states, so the grid must be too coarse. That
   argument is dead: the head ablation put two policies at **4.8% and 5.1%**
   agreement with **85% and 96%** greedy win rates. Agreement with the expert
   does not track winning, and cannot be used to indict the observation — or to
   compare anything else. Measure the channels by win rate, on demonstrations
   recorded under the encoding, or not at all.
5. **Let the policy propose the search's candidates.** The expert samples ~14
   placements uniformly at random and never benefits from anything the policy
   learns. Sampling from the clone closes that loop and denoises the labels at
   the same time. Cheapest change with the largest expected effect.
6. **Evaluate against the expert, not random.** The control is beaten 100–0, so
   the metric is saturated and a better policy cannot register.

## Traps

- **The watcher runs stale code.** Python does not reload an edited module in a
  running process — restart it after every `watch.py` change. This has already
  bitten once for real: two watchers from before the all-time page landed kept
  rewriting `progress.html` every 15 seconds with pre-all-time-page code, so
  the served dashboard silently had none of it. Check the page for a string
  only the new code emits, not just that the file's timestamp is fresh.
- **194 MB of results live only inside a worktree.** Every checkpoint behind
  the head and observation comparisons is under
  `.claude/worktrees/agent-acff606c02b4824d0/runs/` and nowhere else. The three
  head clones are copied to `checkpoints/`; the rest is not. Get it out before
  cleaning up worktrees, and see the `--force` trap below.
- **Interrupted commands leave processes running.** Every `python` call goes
  through a 0-byte Microsoft Store alias that spawns the real interpreter and
  lingers, so the process count is double what the work justifies. Disable the
  aliases in Settings → Apps → Advanced app settings.
- **Never `git worktree remove --force`.** It followed directory junctions once
  and deleted the real `data_cache`.
- **Two sessions work in this repo.** Check `Get-CimInstance Win32_Process`
  before starting anything long, and do not kill another session's run. A
  background agent started a 1M-step run here without telling anyone, and a
  restart of it passed `--resume` together with `--init-from`, which the CLI
  correctly refuses — so the run was simply gone, with the reason sitting only
  in `runs/<name>/train.log`. Read that file before assuming a run is alive.
- **The GPU does not work here.** `--device xpu` on an Intel Arc reports
  available, runs a gradient step 6.6× faster, then fails three ways. `--device
  auto` deliberately will not choose it.

## Layout

```
cr_sim/
  data/     Supercell decoder, csv_logic dialect, EXT inheritance, level scaling
  engine/   arena, entities, targeting, combat, spells, buffs, the ACTION
            interpreter, the 17-phase tick loop, and lookahead.py
  api/      Gymnasium env, observation/action encoding, rewards, vec env
  train/    PPO, self-play pool, the search expert, behavioural cloning,
            the live page, the multi-run report, Discord reporting
  play/     browser game against a checkpoint
reference/  anchors.json, hits_to_kill.csv and engagement.md are external truth
            or generated analysis; card_stats.json is a generated baseline whose
            only job is to make a new APK's balance changes visible
```
