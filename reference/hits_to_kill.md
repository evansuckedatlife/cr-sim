# hits_to_kill.csv

A community-maintained interaction matrix: how many hits each attacker needs
to kill each defender, at tournament standard (level 11).

Source: <https://docs.google.com/spreadsheets/d/1A2agNsW8AFC1727tq4hmPlX46JEADxMGxBqamW7RuPw/htmlview>
("cr interactions - Hits/Kill"), exported 2026-08-27.

**Orientation.** `row` is the unit being killed, `column` is the unit doing the
killing, and the cell is the number of hits. Confirmed against the engine's own
resolved stats before use:

| defender | attacker | sheet | engine |
|---|---|---|---|
| musketeer | knight | 4 | 720 hp / 202 dmg = 3.6 |
| knight | musketeer | 9 | 1766 hp / 218 dmg = 8.1 |
| musketeer | fireball | 2 | 871 hp survives one 688 |

**Why it is here rather than generated.** `reference/` holds external truth, and
the whole point of a gate is that it is not produced by the thing it checks. The
same split applies to `anchors.json`, which is hand-written, against
`card_stats.json`, which is regenerated from the build and only exists to make a
new APK's balance changes visible rather than silent.

**It is not infallible.** It is a community spreadsheet: names are informal
("sus bush", "void 3rd"), some rows are situational ("fully healed evo witch",
"egiant reflection"), and a disagreement with the engine is a question to
investigate rather than an automatic engine bug. Several disagreements on this
project have gone the other way -- the public P.E.K.K.A damage figure was wrong,
and Crown Towers turned out to scale on their own progression.

**It is also roughly a year stale, and that changes how to read a disagreement.**
The "exported" date above is when a copy of the sheet was pulled, not when its
community maintainers last updated its numbers -- and Clash Royale rebalances
cards continuously, so a meaningful fraction of this sheet describes a version
of the game that no longer exists. `cr_sim/data/cards.py` and the rest of the
data layer are built from a *current* APK extract, so a mismatch against this
sheet is now genuinely ambiguous: it can mean the engine is wrong, or it can
mean nothing more than "Supercell changed a number since this row was last
edited." A concrete example the cross-check below turned up: the sheet
implies Miner deals full damage to a Crown Tower, but the current build's own
data gives him `CrownTowerDamagePercent: -80` (only 20% damage against a
tower) -- plausibly a nerf Miner picked up sometime in the last year, not an
extraction bug, since every other tunnelling troop checked (Hog Rider, Ram
Rider, Battle Ram) still shows no such reduction.

Because of that, this sheet is no longer treated as a pass/fail gate. It is
one input to `cr_sim.data.interactions`, which builds its own matrix from the
current build's resolved stats (arithmetic over the whole standard card pool,
plus real simulated duels for a smaller subset) and only uses this sheet as a
secondary cross-check, sorting each mapped disagreement into one of three
readings rather than asserting an agreement floor against it:

1. **agree** -- the engine's current stats reproduce the sheet's number.
   Independent corroboration.
2. **explained** -- they disagree, and even the *bare* hitpoints/damage
   arithmetic (ignoring shields, crown-tower percent, damage ramps) does not
   match the sheet either. A raw stat moved since the sheet was written --
   read this as "what changed", not as a defect.
3. **defect** -- they disagree, but the bare arithmetic *does* match the
   sheet. The raw numbers line up; only the engine's shield/tower/ramp
   handling produces a different answer. This is the one category worth
   treating as a possible bug -- see `python -m cr_sim.cli interactions` for
   the current list (as of this writing, all of it is one well-understood,
   non-bug pattern: for every attacker, the "royal recruits" and "dark
   prince" rows consistently give the *body-only* hit count -- the sheet
   simply never counts the hit that breaks their shield, while a separate
   "guard shield" row elsewhere on the sheet gets that arithmetic right for
   a third shielded card. That reads as a sheet-row convention -- these two
   rows implicitly assume the shield is already down -- not an engine bug;
   the engine's own shield rule (`Entity.apply_damage`) is independently
   documented and unrelated to this sheet).

Run `python -m cr_sim.cli interactions` for the current numbers; nothing here
is meant to be re-derived by hand from this file.
