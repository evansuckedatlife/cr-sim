---
type: object
cluster: surfaces
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/play/server.py
---

# Play server

`python -m cr_sim.play` — a local HTTP server that runs a live match against a
checkpoint in a browser. Four modules: `server.py` (routes and state),
`session.py` (the clock), `policy.py` (the opponent), `page.py` (one string of
HTML).

## Why this shape

**The battle runs on the server and the browser only draws.** A second
implementation in JavaScript would be a second set of answers to every question
the first one settled, so the page gets positions and hitpoints and decides
nothing — it asks, and the same `play_card` the training environment calls
answers (`cr_sim/play/session.py:1-15`). Time advances in whole engine ticks
driven by a real clock, so a dropped frame costs smoothness and never
determinism.

That is also why this card exists in `surfaces` and not in `interface`.
**Nothing in the training loop imports any of it**, so no `interface` or
`measurement` card would ever name it — and it holds four independent restatements
of things those clusters own. This is
[`../../effects/points-in.md`](../../effects/points-in.md) in miniature: what
points *into* the tree is invisible from inside it.

## Shape

Four restatements, each of something owned elsewhere, each with no guard:

- **A second `DEFAULT_DECK`** (`cr_sim/play/server.py:46`), byte-identical to
  `cr_sim/train/run.py:54` today and independent of it. Change one and the
  browser opponent's vocabulary repermutes against the checkpoint's: same
  `vocab_size`, same `vector_size`, strict load succeeds, `check_observation`
  passes, and 80 columns silently mean different cards. See
  [`../../CONTEXT.md`](../../CONTEXT.md) (collision table, `DEFAULT_DECK`) and
  [`../interface/observation-vector.md`](../interface/observation-vector.md).
- **A fourth `--tower-level` defaulting to 11** (`cr_sim/play/server.py:306`),
  carried through `PlayServer.__init__` (`:71`), `serve` (`:281`) and
  `SessionConfig.tower_level` (`cr_sim/play/session.py:57`). One home for the
  divergence: [`../measurement/tower-level.md`](../measurement/tower-level.md).
- **`NetConfig` restated by hand** in `PolicyOpponent._ensure`
  (`cr_sim/play/policy.py:121-137`) — the one load path `net_config_for` does
  not cover, because it has a battle rather than an environment. The field
  count and what it costs live on
  [`../interface/net-config.md`](../interface/net-config.md); this card does not
  repeat it.
- **`nvec` restated as a literal.** `cr_sim/play/policy.py:94` hardcodes
  `(5, action_width, action_height)` and then imports `NUM_CARD_SLOTS` on the
  next line (`:95`), which is the constant it just spelled out.

And one stream with a named owner that is never used:

- `PolicyOpponent.__init__` takes a `seed` and builds
  `self.rng = np.random.default_rng(seed)` (`cr_sim/play/policy.py:50`).
  **Nothing reads `self.rng`.** The move is sampled at
  `cr_sim/play/policy.py:167` through `torch.distributions.Categorical(...).sample()`
  with no generator, so the browser opponent draws off torch's global stream —
  bug 5's shape, on the one surface
  [`../measurement/random-streams.md`](../measurement/random-streams.md) does
  not reach.

**Failure is swallowed by design, and that is what hides the above.**
`PlaySession._think` catches every exception from the controller, sets
`self.controller = None` and records the string
(`cr_sim/play/session.py:211-223`). An opponent that raises on its first move
therefore reads as a policy that decided to pass for the rest of the match. The
reason is written in place: raising would reach the caller through every poll.
Both that behaviour and the `NetConfig` restatement above **are** pinned, and
by tests that say why in their own docstrings
(`tests/test_play.py:175-245`, parametrised over all four heads;
`tests/test_play.py:317-335`). What is pinned by nothing is the other three
restatements — the deck, the tower level and `nvec`.

Citations: `cr_sim/play/server.py:46`, `:52-64`, `:67`, `:71`, `:99-113`,
`:281`, `:306`; `cr_sim/play/session.py:47-63`, `:102`, `:211-223`;
`cr_sim/play/policy.py:40`, `:50`, `:70`, `:94-95`, `:121-137`, `:167`;
`cr_sim/play/page.py:23`.
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** `PlayServer`, `PlaySession`, `SessionConfig`, `PolicyOpponent`,
  `PAGE`.
- **owned-by:** nothing in the tree. It is a leaf consumer of
  [`../battle/battle.md`](../battle/battle.md) and
  [`../interface/encoding-config.md`](../interface/encoding-config.md).
- **joins:** [`../interface/net-config.md`](../interface/net-config.md),
  [`../measurement/tower-level.md`](../measurement/tower-level.md),
  [`../measurement/random-streams.md`](../measurement/random-streams.md),
  [`../interface/observation-vector.md`](../interface/observation-vector.md).
- **looks-like-but-is-not:** `cr_sim/render/web.py`, which also draws a battle
  in a browser. That one is a *replay* viewer — one self-contained HTML file,
  no server, frames cosmetic and never in the state hash
  (`cr_sim/render/web.py:1-11`). It reads nothing this file reads.

## If you change this

- **Hits:** nothing else in the tree. That is the whole hazard — it is the
  downstream end of four one-way edges, so a change *here* is contained and a
  change *there* arrives silently.
- **Does not hit:** the deck, the tower level or `nvec`, as far as the suite
  is concerned. `tests/test_play.py` is thorough about the *session* — the
  clock, rejections, evolution slots, a raising controller, and every head
  through this load path — so the obvious next assumption, that a green
  `test_play.py` means the restatements are covered, is wrong. **No test
  imports both `DEFAULT_DECK` literals**, and none asserts a tower level here.

## Surfaces

| Surface | Role |
|---|---|
| a person, in a browser | the only interactive read of the engine |
| `runs/*/final.pt` | reads — any checkpoint, by `_resolve` (`cr_sim/play/server.py:52-64`) |
| `tests/test_play.py` | 29 tests over the session, the heads and the fallback |
| the progress page | none — this process registers no run directory |

## See

- Source: `cr_sim/play/server.py`, `cr_sim/play/session.py`,
  `cr_sim/play/policy.py`, `cr_sim/play/page.py`
