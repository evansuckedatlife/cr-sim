---
type: object
cluster: battle
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/combat.py
---

# AttackState

One unit's place in its attack cycle, and the two-step hit that comes out of
it. An attack here is a small state machine, not "damage every `HitSpeed`
ticks".

## Why this shape

**`LoadTime` is not `HitSpeed`.** `LoadTime` is the windup before the *first*
hit after engaging; `HitSpeed` is the interval between later ones. A Knight
waits 700ms for its first swing and 1200ms between the rest. That is why a unit
forced to re-engage repeatedly never actually deals damage, and why resetting an
Inferno is worth a whole card.

**Damage lands at the END of a windup.** A unit killed during its load deals
nothing. That single choice is the entire basis of counter-pushing.

**Deciding and applying are deliberately separate.** If damage landed inline,
the outcome of a fight would depend on the order entities happen to sit in the
entity list: whichever unit iterated first would land the killing blow and the
other — already at zero — would be skipped before it could swing. Two identical
units placed symmetrically would produce a winner, which is plainly wrong.
Collecting `PendingHit`s and applying them together makes a tick simultaneous.

**`ramp_damage` is two ladders sharing one column pair**, and which one a unit
is on is decided by whether it also carries `VariableDamageTime`:

- *Timed* — Inferno Tower reads 17 / 62 / 331 with two 2000ms steps. The
  escalation is the card.
- *Per swing* — Monk reads 55 / 55 / 165 with **no** time steps and an
  `AttackSequence` of `[0, 1, 2]`. Walked as a timed ladder those steps never
  advance, he swings for 55 forever, and the card's whole mechanic is deleted.
  The sequence index cycles, so the ladder repeats rather than topping out.

The attacker's buffs scale what it *deals*; the victim's scale what it *takes*
(in `Entity.apply_damage`). Keeping them on opposite sides of the hit is what
lets a raged Monk hit a fortified Knight with both modifiers counting exactly
once.

## Shape

- `AttackState` — `cooldown`, `loaded`, `locked_ticks`, `stop_ticks`,
  `swings`, plus `engage` / `disengage`; `can_move` is `stop_ticks <= 0`.
- `advance_attack(state, spec, attacker, target) -> PendingHit | None` — decides
  only. Sets the next cooldown to `max(1, hit_speed_ticks)` and the
  post-swing stop from `StopTimeAfterAttack`.
- `PendingHit` — carries a per-swing `damage` override (the ramp, a charge
  connecting) and a `sequence_index`, so both mechanics land through one path.
- `damage_for` / `_reduce_for_tower` — crown-tower reduction applied as a
  negative delta, once.
- `apply_hit(hit, tick) -> DamageEvent | None`; `apply_area_damage`.
- `DamageEvent(tick, attacker_id, target_id, amount, lethal)` — appended to
  `Battle.damage_log`, an append-only history read outside the engine.

Citations: `cr_sim/engine/combat.py:54`, `:163`, `:199`, `:216`, `:265`,
`:281`, `:129`, `:139`, `:46-50`, `:1-23`, `:222-232` (decide/apply separation).
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** `PendingHit`, `DamageEvent`, `ramp_damage`, `damage_for`.
- **owned-by:** [../build/unit-spec.md](../build/unit-spec.md) — every timing
  and every ladder is a pre-converted spec field.
- **joins:** [targeting.md](targeting.md) (a target is the input),
  [buff-percent.md](buff-percent.md) (`apply_delta` on both sides of the hit),
  [battle.md](battle.md) (`_attacks` is keyed by entity id;
  `_phase_resolve_attacks` collects then applies).
- **looks-like-but-is-not:** `AttackFinishTime` / `OverrideAttackFinishTime`
  and the 250ms `ATTACK_FINISH_TIME_MS` global — **not implementable from the
  data** and deliberately unread; the reason is in the register at
  `cr_sim/engine/specs.py:66-78`. `StopTimeAfterAttack` is the one that *is* a
  movement stop, and it is zero on every entity that ships.

## If you change this

- **Hits:** every duel, so the interaction gate's `simulated` matrix and every
  hits-to-kill number; `Battle.damage_log`, which the reward
  (`cr_sim/api/reward.py`) and the interaction harness both read; and the
  Inferno/Monk/charge families specifically, which share this one path.
- **Does not hit:** the damage *numbers*. Base damage, the projectile chain and
  the crown-tower percentage are all resolved at spec-build time
  ([../build/unit-spec.md](../build/unit-spec.md)) — if a hit lands for the
  wrong amount at the right time, this is the wrong file. And it does not hit
  splash: `apply_area_damage` is a separate path that tests `is_targetable`,
  not the attack cycle.

## Surfaces

| Surface | Role |
|---|---|
| `cr_sim/data/interactions.py` | reads — hits-to-kill is this cycle, counted |
| `cr_sim/api/reward.py` | reads `damage_log` |
| `tests/test_combat.py`, `test_charge_and_ramp.py` | pin both ladders |

## See

- Source: `cr_sim/engine/combat.py`
