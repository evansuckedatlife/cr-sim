# Handoff

Where this is, what worked, and what to distrust. Written for someone picking it
up cold — including me, later.

## The state in one paragraph

The **simulator** is in good shape: 663 tests, a deterministic integer
fixed-point engine built from the game's own shipped tables, and the two open
questions that blocked milestone M3 are closed. The **agent** took much longer
and only started working at the very end. What finally moved it was not a better
reward or a bigger network — it was doing the step every successful game agent
does first, which this project had skipped: learn from a competent
demonstrator before trying reinforcement learning.

## Results worth trusting

All measured as *lift* against a control over **paired battles** — both arms play
the same fixed seeds — reported in standard deviations of the control's own
spread, with a 95% interval. Anything without an interval is a 40-battle reading
and should be treated as a rumour; see *Numbers that fooled us* below.

| | lift | wins | losses | battles |
|---|---|---|---|---|
| search expert | +2.716 [+2.37, +3.06] | 100% | 0% | 40 |
| **cloned policy, greedy** | **+1.623 [+1.39, +1.86]** | **83%** | **5%** | 150 |
| cloned policy, sampled | +0.709 [+0.46, +0.96] | 55% | 16% | 150 |
| random control | — | 26% | 27% | — |

The **search expert** does no learning. For each decision it branches the live
battle for ~18 candidate placements, plays each one 15 seconds forward, and keeps
whichever leaves the board best. This is only possible because the engine clones
in 0.69 ms — almost no reinforcement learning environment can afford to roll out
its own future, and it turned out to be this project's one real advantage.

The **cloned policy** is a network trained to imitate that search. It captures
about 60% of the expert's advantage in a single forward pass.

## The thing that actually mattered

Four training runs from random initialisation produced nothing. A 300-battle
evaluation could not distinguish trained checkpoints from freshly initialised
networks — every encouraging number had been measuring a random network's
placement prior.

That is the expected outcome of the method at this scale, not a bug. AlphaStar
trained on 971,000 human replays *before* any reinforcement learning, and that
supervised agent alone outranked 84% of human players. OpenAI Five ran the same
PPO algorithm on 128,000 CPU cores with batches of one to three million
timesteps — a single OpenAI Five batch is roughly twenty times this project's
entire first run. We had neither, so the answer was the third option available in
the literature: **search**, which needs a fast exact simulator, which we have.

Order that works: search expert → demonstrations → behavioural clone →
reinforcement learning. Not reinforcement learning from noise.

## Numbers that fooled us

Read this before believing any figure in `runs/`.

- **The inline evaluation faced an *idle* opponent; the large paired evaluations
  faced a *random* one.** Both were reported as "lift" and compared to each
  other for an entire session. They are not comparable: the control wins 92% of
  idle matches and 26% of random ones. Fixed, but **every metric written before
  that fix is against an idle opponent** — including all of `selfplay-1m`.
- **A +0.375 reading over 40 battles measured −0.033 over 300.** Forty battles
  cannot separate a weak effect from zero here.
- **Checkpoint selection was picking the luckiest reading, not the best policy.**
  Keeping the maximum of nineteen noisy evaluations selects for noise. Promotion
  now needs a rolling mean of three.
- **92% of matches were draws** at tower level 11, so crowns — the only real
  objective — almost never fired and everything learned from shaping alone.
  Training runs use `--tower-level 5`, which halves the draw rate for free.
  Evaluate at 11 to see what transfers.

## Bugs that shipped green

Each of these had passing tests while doing nothing. This is the failure mode
this codebase generates, and the reason so many tests here run the real thing
rather than inspecting source.

- `push_away` was a **no-op for every knockback in the game**. It called
  `point_along` with `travelled = span + amount`, which returns the endpoint
  whenever `travelled >= segment_length` — always true. The death-pushback test
  passed the whole time, on retargeting movement rather than the push.
- **Projectiles never applied the buff they carry.** Ice Spirits dealt damage and
  never froze, which is the entire card, and it is one of the eight the agent
  trains on. Lightning did not stun; Snowball did not slow.
- **The deploy zone never expanded** when a Princess Tower fell, and the
  placement cache meant *no battle in the process* would ever see it.
- **`RelativeX`/`RelativeY` were read as milli-tiles**, a factor of a thousand,
  so Furnace's Fire Spirits spawned inside the building.
- **The progress page claimed "no evaluations yet"** over a real evaluation, and
  its test passed because that string is in the page's source regardless of data.

## Running it

```bash
python -m pytest                          # 663 tests
python -m cr_sim.cli validate             # stat gate + open questions
python -m cr_sim.cli battle --html r.html # a match you can watch

# training. --tower-level 5 is what makes matches resolve; --workers is most
# of the throughput; --init-from is the order that works.
python -m cr_sim.train.run --steps 400000 --envs 8 --workers 4 \
    --tower-level 5 --reward projected --opponent self \
    --init-from runs/cloned/cloned.pt --lr 1e-4 --entropy 0.005 --name my-run

python -m cr_sim.train.watch --every 15 --serve 8899   # live page, phone-friendly
python -m cr_sim.train.report                          # one page per run
python -m cr_sim.play.server --policy runs/cloned/cloned.pt --tower-level 5
```

Rebuilding the expert and the clone from scratch, about an hour:

```bash
python scripts/make_demos.py --episodes 70 --shard 0   # run several shards
python scripts/clone_policy.py --demos data_cache/demos --out runs/cloned
python scripts/measure_expert.py --episodes 40
```

Data is not in the repo. Supply a Clash Royale APK and run
`python scripts/extract_apk.py <apk-or-directory>`.

## Open threads

**Reinforcement learning erodes the clone.** `ppo-from-clone` started from the
+0.709 policy and fell to +0.115 within 33 updates, with explained variance
collapsing 0.551 → 0.011 and entropy 3.65 → 2.45. Even at `lr 1e-4`. AlphaStar
hit the same problem and solved it with a **KL penalty toward the supervised
policy** — that is the obvious next fix and it is not implemented here.

**Let the policy propose the search's candidates.** The expert currently samples
18 placements at random. Sampling from the cloned policy instead would evaluate
plausible moves rather than arbitrary ones — better search, better targets,
better policy, better search. This closes a loop that does not exist today: the
expert never benefits from anything the policy learns. Cheapest change with the
largest expected effect.

**Correct the policy on its own states (DAgger).** Cloning trained on states the
*expert* visits; the deployed policy visits its own, including ones the expert
never reaches. Label those.

**Put the expert in the opponent pool.** The ladder currently measures a policy
against its own past, which is movement rather than skill — two weak policies
trade wins forever. A fixed opponent that beats random 100–0 is a real wall.

**Search inside self-play (full AlphaZero).** The real answer and the expensive
one: ~20× slower per step, far more sample-efficient per step.

**The GPU does not work here.** `--device xpu` on an Intel Arc reports available,
runs a gradient step 6.6× faster than eight CPU threads, and then fails a real
training loop three ways — unimplemented convolution, out of device memory, and
out of Level Zero resources inside Adam's state allocation. The rollout's
hundreds of small forward passes, each with a blocking host readback, exhaust the
driver before the first update. `--device auto` deliberately will not choose it.

**Colab is probably slower, not faster.** `notebooks/cr_sim_colab.ipynb` measures
it rather than assuming: this workload is ~90% Python simulation, and free Colab
gives 2 vCPUs against a laptop's 8. Its real use is parallel sessions.

## Layout

```
cr_sim/
  data/     Supercell decoder, csv_logic dialect, EXT inheritance, level scaling
  engine/   arena, entities, targeting, combat, spells, buffs, the ACTION
            interpreter, the 17-phase tick loop, and lookahead.py — which
            branches a battle to ask what the board is already worth
  api/      Gymnasium-style env, observation/action encoding, rewards, vec env
  train/    PPO, self-play pool, the search expert, behavioural cloning,
            the live page, the multi-run report, Discord reporting
  play/     browser game against a checkpoint
reference/  anchors.json + hits_to_kill.csv are external truth, never generated;
            card_stats.json is a generated baseline whose only job is to make a
            new APK's balance changes visible instead of silent
```

The README has the simulator's own story — what the data pipeline established,
what the arena turned out to be, and which questions are still open.
