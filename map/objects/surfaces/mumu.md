---
type: object
cluster: surfaces
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/mumu/adb.py
---

# MuMu bridge

`cr_sim/mumu/{adb,capture,geometry,input}.py` — the emulator bridge. The one
path by which a claim about this engine can be settled against a real Clash
Royale client instead of against another file in this repo.

## Why this shape

It exists for the **open questions** in `reference/anchors.json`. A stat can be
checked against a community sheet; a *geometry* — bridge width, a building's
collision shape — cannot, and the only authority is a running client. `geometry`
is the piece that closes those: it turns a pixel in a captured frame into the
same tile coordinate the engine reasons in, and back, which is what makes a
frame comparable to a simulated tick
(`cr_sim/mumu/geometry.py:1-14`). It reads the board's dimensions out of
`cr_sim/engine/arena.py` and delegates the subtile conversion to
`cr_sim/engine/fixed.py` rather than repeating either, so the board size and the
definition of a subtile keep exactly one source of truth
([`../build/subtile.md`](../build/subtile.md)).

Two decisions are load-bearing and both are written at the declining site:

- **`exec-out`, never `adb shell screencap -p`.** `adb shell` allocates a
  pty-like channel, and on Windows that channel turns every `0x0A` into
  `0x0D 0x0A` — fine for text, silent corruption for a PNG, where roughly one
  byte in 256 is `0x0A` by chance. The guarantee only holds while the pipe is
  read as raw bytes end to end, which is why `AdbBridge.exec_out` always invokes
  its runner with `binary=True`. There is deliberately **no CRLF-stripping
  fallback** to keep untested (`cr_sim/mumu/capture.py:4-16`).
- **Input requires `allow_input=True`.** Injecting synthetic input into a live
  client is against Supercell's Terms of Service and can get an account
  actioned. Nothing here refuses to run — whether to accept that risk is the
  caller's decision — but every method that injects requires the caller to say
  so explicitly at construction (`cr_sim/mumu/input.py:1-9`).

## Shape

- `adb.py` is the only module that knows where the `adb` binary lives, and
  `AdbBridge` (`cr_sim/mumu/adb.py:141-262`) is the wrapper everything else
  goes through (the reason is at `cr_sim/mumu/adb.py:1-16`). PATH first, then MuMu's own shipped layouts, because
  a system-wide `adb` is likelier to be current than an emulator installer's.
- `capture` (`cr_sim/mumu/capture.py`) — screenshots and session recording for
  offline comparison.
- `geometry` (`cr_sim/mumu/geometry.py`) — pixel ↔ tile ↔ subtile.
- `input` (`cr_sim/mumu/input.py`) — taps, swipes, card placement, gated.

Citations: `cr_sim/mumu/adb.py:1-16`; `cr_sim/mumu/capture.py:1-16`;
`cr_sim/mumu/geometry.py:1-15`; `cr_sim/mumu/input.py:1-9`;
`cr_sim/mumu/__init__.py:1`.
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** `AdbBridge`, `InputController`, the geometry conversions.
- **owned-by:** [`../build/anchors.md`](../build/anchors.md) — the open
  questions are what this exists to close.
- **joins:** [`../battle/arena.md`](../battle/arena.md),
  [`../build/subtile.md`](../build/subtile.md).
- **looks-like-but-is-not:** [`play-server.md`](play-server.md). Both put a
  battle on a screen; only this one reads a screen it did not draw.

## If you change this

- **Hits:** `tests/test_mumu_adb.py` and `tests/test_mumu_geometry.py`. The
  second names `reference/anchors.json` in a comment and then reads the
  dimensions from `cr_sim.engine.arena` rather than duplicating them
  (`tests/test_mumu_geometry.py:24-29`) — the same discipline this package
  follows, and the reason a geometry change is an anchors question. Lands on
  [`../build/anchors.md`](../build/anchors.md).
- **Does not hit:** the engine, the encoder or any run. Nothing in `cr_sim/`
  imports this package. The obvious wrong assumption — that a geometry fix here
  corrects the simulator — is backwards: this measures the client so somebody
  can decide whether the simulator needs correcting.

## Surfaces

| Surface | Role |
|---|---|
| a MuMu Player 12 install, over `adb` | reads (and writes, only with `allow_input=True`) |
| `reference/anchors.json` | the open questions this settles |
| `tests/test_mumu_adb.py`, `tests/test_mumu_geometry.py` | pin the conversions |
| the training loop | none — nothing imports this |

## See

- Source: `cr_sim/mumu/adb.py`, `cr_sim/mumu/capture.py`,
  `cr_sim/mumu/geometry.py`, `cr_sim/mumu/input.py`
