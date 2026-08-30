---
type: object
cluster: battle
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/actions.py
---

# ActionSelect

Run **one** of `SubActions`, chosen by condition. **Live, and formerly a
ghost** — the brief that called it "22 definitions doing nothing anywhere" was
true and was fixed in `5bda200`.

## Why this shape

It was previously routed through `NextAction`, and almost no `ActionSelect`
node in the build carries one, so the node did nothing. That silently removed
Vines' snare (its buff is chosen here), the Hunter evolution's net, the
P.E.K.K.A evolution's heal-on-kill, the Inferno Dragon evolution's attack-stage
ladder and the Mega Knight evolution's uppercut cadence — five cards that
loaded, played and did nothing.

The node needs its **own gate exception**, which is the subtle part.
`Condition` means two different things depending on the node: everywhere else
it is a gate, but on `ActionSelect` it is the **chooser**. Reading it as a gate
would let a card through on an unevaluable expression while still running
nothing. So `_passes` checks only `ExecuteIfTrue` when the class type is
`ActionSelect`, and leaves `Condition` to the handler.

`PerActionConditions` is one shorter than `SubActions` wherever a default
exists: first true wins, and falling off the end picks the last entry. Vines is
6 conditions to 7 branches; the Mini P.E.K.K.A hero form is 4 to 4 and
genuinely runs nothing when none match.

The other spelling — `Condition = "rand(n)"` — is a **deliberate
ghost** and has its own card, [`action-select-rand.md`](action-select-rand.md).
A noun with two universes gets two cards: this one is `live`, that one is not,
and a `universe:` label has to be true of the whole file it sits on.

## Shape

- `_handle_select` (`:871`) — `SubActions` (a bare string or Mapping is
  wrapped), then `PerActionConditions`, then `schedule` the chosen branch, then
  `_handle_next` because a select may still chain onward.
- Gate exception: `ActionInterpreter._passes` (`:641-656`).
- `_HANDLERS["ActionSelect"] = _handle_select` (`:1554`).

**The count, settled.** The handler docstring says "every one of the
twenty-two `ActionSelect` nodes in this build". An exhaustive walk of every
loaded namespace — including inline dicts nested inside other nodes, which is
where such a node could hide — finds **20**, all of them top-level in `ACTION`,
none in `EXT`. Of the 20: all 20 carry `SubActions`, 13 carry
`PerActionConditions`, 7 carry `Condition = "rand(n)"`.

**And the docstring's second claim is wrong too.** It says "not one carries the
`NextAction` this was previously routed through". Exactly one does —
`ACTION.InfernoDragon_EV1_UpdateAttackSequence`, whose `NextAction` is an
inline `ActionWithDuration` that sets a game tag. The handler's own closing
comment (`:925-926`) names that card, so **the code and its trailing comment
are right and the docstring's blanket claim is not.** Code wins; do not repeat
"twenty-two" or "not one". Both overrides are filed at
[`../../_meta/overrides.md`](../../_meta/overrides.md), row 2.

Citations: `cr_sim/engine/actions.py:871`, `:874-895` (the docstring being
overridden), `:896-927`, `:922-923`, `:641-656`, `:1554`, `:74-77`, `:905-909`.
Verified 2026-08-30 against `main` @ `dc47f51`; counts re-derived by walking
`LogicData.sections` over `data_cache/csv_logic`.

## Connected to

- **owns:** nothing; it is one handler.
- **owned-by:** [action-interpreter.md](action-interpreter.md).
- **joins:** [buff-percent.md](buff-percent.md) (Vines and the Ice Golemite
  hero pick a *buff size* here), [rng.md](rng.md) (the declined stream).
- **looks-like-but-is-not:** `ActionFilter` and `ActionRunIfGameObjectExists`,
  which both route through `_handle_next` and are gates, not choosers. Only
  this node reinterprets `Condition`.

## If you change this

- **Hits:** the five evolution/champion behaviours listed above, and the
  `<condition:...>` counter in `ActionInterpreter.unsupported`, which is the
  only signal that a branch was reached and declined. Adding a random stream
  belongs to [`action-select-rand.md`](action-select-rand.md), which is a
  mechanics change and not a fix.
- **Does not hit:** the coverage total. `ActionSelect` is already in
  `_HANDLERS`, so nothing about implementing more of it changes which class
  types are counted as unsupported. The obvious wrong reading of a non-zero
  `<select:rand(n)>` count is that the node is broken; it is declined, and the
  reason is written at `cr_sim/engine/actions.py:74-77`.

## Surfaces

| Surface | Role |
|---|---|
| `tests/test_actions.py`, `tests/test_evolutions.py` | pin the branch choice |
| `ActionInterpreter.unsupported` | records the declined spelling |
| commit `5bda200` | the fix that moved this from ghost to live |

## See

- Source: `cr_sim/engine/actions.py`
