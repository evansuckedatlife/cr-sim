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
