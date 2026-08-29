# cr-sim

A mechanically exact Clash Royale battle engine, built as a training
environment for a machine learning agent. No ladder, shop or monetization.

## Every living python process belongs on the progress page

**If a `python` process is running against this repo, it must have an entry on
the progress page. No exceptions, including short ones.**

The page is the only place where the state of this project is visible in one
view. A process that is not on it is work nobody can see, and this project has
already lost work twice to exactly that: a training run that died with the
reason sitting only in a log file nobody opened, and an evaluation that ran 150
battles and threw the result away because it wrote to a directory that did not
exist. A number that exists only in a chat log is a number that does not exist.

Training runs put themselves on the page. **Everything else has to be
registered**, and that is the part that gets skipped:

```bash
python scripts/register_job.py --name bench-tick-loop \
    --note "Profiling the tick loop. Running." --status running
```

That writes `runs/<name>/{metrics.jsonl,config.json}`, which is all the watcher
enumerates. Update the note when the job finishes — an entry that still says
"Running." hours later is worse than none, because it is believed.

This applies to evaluations, benchmarks, sweeps, demo generation, profiling,
long test runs, and any investigation an agent is part-way through. Register it
when it starts, not when it produces an answer: the entry is most useful while
you are waiting for it.

Before starting anything long, see what is already running:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Select-Object ProcessId, CommandLine | Format-List
```

Two sessions work in this repo. **Never kill another session's run.** Worker
processes appear as `multiprocessing.spawn` children — attribute them to their
parent pid before concluding anything is orphaned.

## The watcher runs stale code

Python does not reload an edited module in a running process. After any change
to `cr_sim/train/watch.py`, restart the watcher or the served page silently
keeps rendering the old one. Verify by grepping the generated `progress.html`
for a string only the new code emits — a fresh timestamp proves nothing, since
the stale process rewrites the file every 15 seconds.

## A lift means nothing without the control it was measured against

The same policy scores wildly differently against an idle, a random and a
searching opponent: the control wins 92% of idle matches and 26% of random
ones. Conflating them has cost this project three rounds of invalid
comparisons, including a headline claim that reinforcement learning eroded the
clone. `write_verdict` now refuses a lift with no `eval_opponent`. Never put
two lifts on one axis without establishing they share a control.

Related, and equally load-bearing:

- **Greedy and sampled are two different policies.** Report them separately.
  Greedy reproduces bit-identically run to run — exact float equality, and
  `runs/sampled-noise-floor/noise.json` has the same lift twice to sixteen
  digits. Sampled carries **0.062 sd** of spread from its own sampling: one
  checkpoint, four independent streams, the same 150 battles and the same
  control gave +0.8327 / +0.9232 / +0.9642 / +0.8488. So two sampled runs can
  differ by **0.17 sd** at 95%, and any sampled comparison closer than that is
  measuring the random number generator. The ±0.02–0.04 figure this line used
  to carry came from three uncontrolled readings and is about four times too
  small; `cr_sim/train/evaluate.py` says so in the source. Anything sized off
  it — battle counts especially — is off by 4x.
- **A lift also needs the reward it was counted in.** It is a difference of
  *returns* over the control’s own spread, so the reward is in the numerator
  and the denominator both. The offline scripts here measure under
  `simple:shaping=0.01`; `run.EVAL_REWARD` pins the in-run probe to
  `projected:tower=1,elixir=0.3,horizon_seconds=3`. `check_lift_is_named` and
  `write_verdict` both refuse a lift with no `eval_reward`. A rating does not
  need one — Elo is fitted on crowns, which no reward touches.
- **Agreement with the expert does not track winning.** Two heads matched the
  expert's tile 4.8% and 5.1% of the time while winning 85% and 96% of their
  games. Never rank policies by held-out agreement.
- **Point `--out` at a file for every evaluation worth citing.** A number
  transcribed by hand is a number nobody can check; that is how `+1.813`
  entered the record for a figure that is `+1.623`.

## Tests must fail when the thing breaks

The failure mode this codebase generates is a test that passes over broken
behaviour. Several shipped green while doing nothing: `push_away` was a no-op
for every knockback in the game, projectiles never applied the buff they carry,
and the progress page claimed "no evaluations yet" over a real evaluation
because the test asserted on a string present in the page source regardless.

Assert on behaviour and on data, not on the presence of a literal. When adding
a test to this repo, break the source deliberately and confirm the test goes
red.

## Conventions

- Commit messages explain *why*, in prose, and record what was measured —
  including results that refuted the plan. **No `Co-Authored-By` trailer and no
  AI attribution.**
- Never run `git push` unless asked.
- **Never `git worktree remove --force`.** It follows Windows directory
  junctions and has already destroyed the real `data_cache` once.
- Secrets — the Discord webhook URL and bot token — come from environment
  variables or an explicit flag. Never a file, a test or the README.
- `runs/`, `data_cache/` and `checkpoints/` hold generated data and weights.
  They are gitignored, and 194 MB of experiment results currently exist only
  inside a worktree. Copy anything you care about out before cleaning up.
