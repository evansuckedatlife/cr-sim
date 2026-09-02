---
type: object
cluster: measurement
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/battle.py
---

# Tower level — the arena a number was measured in

`BattleConfig.tower_level`: a **raw** tower level fed straight to `TowerScale`,
with no display conversion. Not a stat and not a knob — it is the *arena*, and
this card is its one home because three map files used to carry three different
counts of where it comes from.

## Why this shape

At level 11 the towers outlast a three-minute match and about **90% of battles
draw**. A drawn battle is a battle nobody learned from and nobody can be ranked
by, so a number measured at 11 and a number measured at 5 are not two readings
of one quantity — they are readings of two games. The source says so in two
places, in the two guards that exist (`cr_sim/train/run.py:483-486`,
`scripts/run_ladder.py:275-279`).

It is a *measurement* noun rather than an `interface` one for that reason. The
interface question is "does the field reach both construction paths" — that is
bug 1, and it lives on
[`../interface/vec-env-config.md`](../interface/vec-env-config.md). The
measurement question is "may these two numbers share an axis", which is bug 6,
and that is this card.

Not to be confused with `BattleConfig.level`, the **displayed** card level on a
different ladder — [`../../CONTEXT.md`](../../CONTEXT.md), collision table.

## Shape

**Where it is set.** `BattleConfig.tower_level` (`cr_sim/engine/battle.py:223`,
consumed `:470`); `CRSimEnv` (`cr_sim/api/env.py:313`, passed `:396`) and
`CRSimSelfPlayEnv` (`:660`, `:706`); `VecEnvConfig.tower_level`
(`cr_sim/api/vec.py:76`). Every one of those five defaults to **11**.

**Twelve entry points, two conventions, nothing that reconciles them.**

| Default | Where |
|---|---|
| **11**, four | `cr_sim/train/run.py:125`, `cr_sim/train/evaluate.py:700`, `scripts/evaluate_vs_expert.py:65`, `cr_sim/play/server.py:306` |
| **5**, eight | `scripts/clone_policy.py:188`, `scripts/evaluate_checkpoints.py:28`, `scripts/evaluate_decks.py:249`, `scripts/expert_iterate.py:85`, `scripts/make_demos.py:71`, `scripts/measure_expert.py:62`, `scripts/measure_sampled_noise.py:49`, `scripts/run_ladder.py:168` |

`scripts/evaluate_vs_expert.py:65-71` is the only one of the twelve whose help
text warns which arena it is choosing.

**Written into an artefact: twice.** `scripts/measure_expert.py:190` puts it in
a verdict; `scripts/run_ladder.py:279` puts it in a `ladder.json`, with the
reason written beside it. **No checkpoint of any kind records it**
([`checkpoint.md`](checkpoint.md)), and neither does a metrics row.

**Checked: once.** `cr_sim/train/run.py:480-486` refuses a `ladder.json` whose
recorded level disagrees with the run's — and only where the file happens to
carry one, which is only ever a `run_ladder` file. **Nothing refuses a verdict
that omits it**, and `write_verdict` has no clause for it
([`verdict.md`](verdict.md), three raises, none of them this).

Citations: as cited above; the bug-1 fix, written in place, at
`cr_sim/train/run.py:996-1009`.
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** the arena half of a comparison — one of the six inputs a lift is a
  function of ([`lift.md`](lift.md)).
- **owned-by:** [`../battle/battle-config.md`](../battle/battle-config.md).
- **joins:** [`../interface/vec-env-config.md`](../interface/vec-env-config.md)
  (bug 1's construction paths), [`verdict.md`](verdict.md),
  [`ladder.md`](ladder.md), [`config-json.md`](config-json.md),
  [`../surfaces/play-server.md`](../surfaces/play-server.md) (the fourth 11).
- **looks-like-but-is-not:** `BattleConfig.level` and `VecEnvConfig.level`. Same
  word, different ladder, different conversion, no shared flag —
  [`../surfaces/cli.md`](../surfaces/cli.md) holds the `--level` that does
  exist.

## If you change this

- **Hits:** every default listed above is independent, so changing one changes
  one program. Changing the *convention* — making the twelve agree — re-bases
  every future number against every recorded one, because the recorded ones do
  not say which arena they came from. It also hits the two guards, which compare
  a recorded level against a run's and would start firing on old files.
- **Does not hit:** any number already on disk, and none of the twelve
  defaults. Bug 1's own shape *is* pinned now —
  `tests/test_train.py:524-560` asserts the workers build the level the run was
  launched with, and `tests/test_ladder.py:571-600` drives all three clauses of
  the ladder guard. Neither reaches an argparse default: no test asserts that
  any of the twelve is 5 or 11, and nothing anywhere requires a verdict to name
  its arena. The obvious wrong next step is to "fix" the eight fives up to
  eleven for consistency: eleven is the arena that draws.

## Surfaces

| Surface | Role |
|---|---|
| `runs/*/config.json` | writes — the run's own level, from `_env()`'s argument |
| `verdict.json` | writes, from `scripts/measure_expert.py` only |
| `ladder.json` | writes (`scripts/run_ladder.py:279`) and reads (`cr_sim/train/run.py:480`) |
| a human comparing two lifts | the consumer this card exists for — see [`../../effects/quoting-a-result.md`](../../effects/quoting-a-result.md) |

## See

- Source: `cr_sim/engine/battle.py`, `cr_sim/api/vec.py`, `cr_sim/train/run.py`
