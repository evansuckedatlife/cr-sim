---
type: object
cluster: surfaces
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/cli.py
---

# cr-sim CLI

`cr_sim/cli.py` — nine subcommands over a decoded build, and the only human
view of it. `cr_sim/soak.py` sits beside it: a hundred thousand matches with no
policy in them.

## Why this shape

Every subcommand here is a **gate or a look**, never a movement — nothing it
writes is consumed by another verb, which is why none of it earns a card in
[`../../processes/CONTEXT.md`](../../processes/CONTEXT.md). It is also the only
consumer of `cr_sim/data/interactions.py` and `cr_sim/data/engagement.py`: two
modules that exist for `validate`, `interactions` and `engagement` and are
reachable from nothing in the training loop.

`soak` finds what a test suite structurally cannot. A spell subtly mis-scaled
passes every test written about the spells someone thought to check; a hundred
thousand unattended matches trip on the one nobody did
(`cr_sim/soak.py:1-14`).

## Shape

Nine subcommands, all wired in `main` (`cr_sim/cli.py:504-565`): `ingest`,
`cards`, `card`, `validate`, `interactions`, `engagement`, `battle`, `arena`,
`freeze`.

**This is where `--level` lives, and it is the answer to "is there a CLI flag
that sets the card level?"** Three subcommands take one, each defaulting to
`TOURNAMENT_DISPLAY_LEVEL`: `cards` (`cr_sim/cli.py:513`), `card` (`:518`) and
`battle` (`:546`). `cmd_battle` passes it straight into a `BattleConfig` at
`cr_sim/cli.py:380`, so `cr-sim battle --level 9` sets exactly the field
`BattleConfig.level` names — a **displayed** level, converted per rarity
([`../../CONTEXT.md`](../../CONTEXT.md), collision table, `level`).

What has no flag is the *training* path: `VecEnvConfig.level` and
`CRSimEnv(level=)` are pinned at 11 by four literals and reached by no argparse
anywhere in `cr_sim/train/` or `scripts/` — see
[`../interface/vec-env-config.md`](../interface/vec-env-config.md). The two
claims are about different construction paths and only the second one is a
ghost. `TOURNAMENT_DISPLAY_LEVEL = 11` is itself defined **twice**, at
`cr_sim/data/cards.py:343` and `cr_sim/data/interactions.py:72`, so a change to
the tournament standard is a two-site edit plus the four training literals.

Other entry-point facts worth one line each:

- `--build` is global and defaults to `data_cache/csv_logic`
  (`cr_sim/cli.py:506`); every subcommand loads through `_load` (`:57`).
- `cmd_validate` (`:127`) reads `reference/anchors.json` directly and compares
  against the frozen baseline; `freeze` (`:495`) rewrites that baseline.
- `cmd_interactions` (`:208`) is the interaction-matrix gate, cross-checked
  against a community sheet with **no agreement floor** on purpose — see
  [`../build/validation-gates.md`](../build/validation-gates.md).

Citations: `cr_sim/cli.py:57`, `:127`, `:208`, `:357`, `:380`, `:495`,
`:504-565`, `:513`, `:518`, `:546`; `cr_sim/data/cards.py:343`;
`cr_sim/data/interactions.py:72`; `cr_sim/soak.py:1-14`.
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** nothing. Every subcommand is a caller.
- **owned-by:** [`../build/logic-data.md`](../build/logic-data.md) — everything
  here starts by loading one.
- **joins:** [`../build/validation-gates.md`](../build/validation-gates.md),
  [`../build/anchors.md`](../build/anchors.md),
  [`../battle/battle-config.md`](../battle/battle-config.md),
  [`../battle/replay.md`](../battle/replay.md) (`--html`).
- **looks-like-but-is-not:** `python -m cr_sim.train.run`. Both are argparse
  entry points over the same package and they share no flag names, no defaults
  and no environment construction. A `--level` here and a `--tower-level` there
  are two different ladders ([`../../CONTEXT.md`](../../CONTEXT.md)).

## If you change this

- **Hits:** the regression baseline under `reference/` that `freeze` writes,
  and nothing in `tests/`. **No test file imports `cr_sim.cli` at all** —
  the gates it drives are tested through their own modules
  (`tests/test_interaction_matrix.py`, `tests/test_engagement.py`,
  `tests/test_data_pipeline.py`), never through the entry point that composes
  them. `cr_sim/data/engagement.py` and `cr_sim/data/interactions.py` have no
  caller but this file, so a signature change here is their whole blast
  radius.
- **Does not hit:** any training run, any checkpoint, any run directory.
  Nothing here writes under `runs/`, and — the obvious wrong next step —
  `cr-sim battle --level` cannot reproduce a training arena, because training
  never sets that field at all.

## Surfaces

| Surface | Role |
|---|---|
| a person at a terminal | the only reader |
| `reference/*.csv`, `reference/anchors.json` | reads and writes (`freeze`) |
| `tests/` | **none** — nothing imports `cr_sim.cli` |
| the progress page | none — register a `soak` run by hand (root `CLAUDE.md`) |

## See

- Source: `cr_sim/cli.py`, `cr_sim/soak.py`
