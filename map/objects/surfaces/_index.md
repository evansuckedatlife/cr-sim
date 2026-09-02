# surfaces — who outside the training loop reads these nouns

One line per noun in this cluster, including the ones no card will be written
for. This is the shelf list: it exists so a cold agent reaches source in one
hop without opening the cluster in bulk.

**A row with a card is a pointer, not an entry.** The gloss lives on the card;
this line carries the noun, its universe, the owning `path:line` and the link.
Where the Card column reads `—` the index line *is* the whole entry, and
that is deliberate — those rows keep their gloss because nothing else holds
it. Adding a fact here that belongs on a card is how the two came to disagree
about `NetConfig`'s field count; put it on the card
([`../../_meta/schema.md`](../../_meta/schema.md), Naming).

Paths are relative to the **repo root** — always, with no exception, because
a suffix match resolves into `.claude/worktrees/agent-*/`, which is three other
checkouts. `python map/_meta/check.py` enforces it. Universes, name collisions
and unit conventions live once in [`../../CONTEXT.md`](../../CONTEXT.md) and no
row may restate them.

| Noun | Universe | Owner | Status | Card |
|---|---|---|---|---|
| `cr_sim/cli.py` | live | `cr_sim/cli.py:1-10` | verified | [`cli.md`](cli.md) |
| `cr_sim/soak.py` | live | `cr_sim/soak.py:1-14` | verified | [`cli.md`](cli.md) |
| `cr_sim/render/web.py` — one self-contained HTML file, no server and no external assets. Frames are cosmetic and never enter the state hash | live | `cr_sim/render/web.py:1-11` | stub | — |
| `cr_sim/play/server.py` / `session.py` / `page.py` | live | `cr_sim/play/server.py:1-14`, `cr_sim/play/session.py:1-15`, `cr_sim/play/page.py:1-14` | verified | [`play-server.md`](play-server.md) |
| **`play/server.DEFAULT_DECK`** | live | `cr_sim/play/server.py:46` vs `cr_sim/train/run.py:54` | verified | [`play-server.md`](play-server.md) |
| `play/policy.PolicyOpponent._ensure` | live | `cr_sim/play/policy.py:121-137` | verified | [`play-server.md`](play-server.md) |
| `cr_sim/play/policy.py:94` hardcodes `nvec = (5, …)` while importing `NUM_CARD_SLOTS` two lines later | live | `cr_sim/play/policy.py:94` vs `cr_sim/api/encoding.py:107` | verified | [`play-server.md`](play-server.md) |
| `cr_sim/play/policy.PolicyOpponent` — builds `np.random.default_rng(seed)` and never reads it; the move is sampled off torch's global stream | live | `cr_sim/play/policy.py:50` vs `:167` | verified | [`play-server.md`](play-server.md) |
| `cr_sim/train/watch.py` | live | `cr_sim/train/watch.py:150`; root `CLAUDE.md` "The watcher runs stale code" | verified | [`progress-page.md`](progress-page.md) |
| `watch.read_ladder` | ghost | `cr_sim/train/watch.py:775` | verified | [`progress-page.md`](progress-page.md) |
| `cr_sim/train/report.py` | live | `cr_sim/train/report.py:54`, `:82-99`, `:318` | verified | [`progress-page.md`](progress-page.md) |
| `scripts/register_job.py` | live | `scripts/register_job.py:35`; root `CLAUDE.md` | verified | [`progress-page.md`](progress-page.md) |
| `cr_sim/train/notify.py` — the webhook path, superseded in practice by the progress page. Still tested | leftover | `cr_sim/train/notify.py:1-23` | stub | — |
| `cr_sim/train/bot.py` — a Discord slash-command bot; needs a token and an invited bot, and nothing in the training loop imports it. Still tested | leftover | `cr_sim/train/bot.py:1-22` | stub | — |
| `cr_sim/mumu/{adb,capture,geometry,input}.py` | live | `cr_sim/mumu/adb.py:1`, `cr_sim/mumu/capture.py:1-15`, `cr_sim/mumu/geometry.py:1-14`, `cr_sim/mumu/input.py:1-9` | verified | [`mumu.md`](mumu.md) |
| `scripts/extract_apk.py` / `scripts/extract_icons.py` — one-time pipeline; ran once to build `data_cache/csv_logic` and `icons` | leftover | `scripts/extract_apk.py:1-7`, `scripts/extract_icons.py:1-10` | stub | — |
| `scripts/bench_engine.py` — a profiling harness, not a measurement of a policy; keeps `--against` for a two-tree A/B | leftover | `scripts/bench_engine.py:1-27` | stub | — |
| `scripts/evaluate_decks.py` / `scripts/summarize_decks.py` — live and correct, but they answer one closed question (zero-shot `FactoredStatsHead`); the answer is in `runs/agent-card-stat-encoder` — parity on the training deck, nothing zero-shot | leftover | `scripts/evaluate_decks.py:1-27` | stub | — |
| `scripts/measure_sampled_noise.py` — ran once and produced the 0.062 sd figure now quoted in the root `CLAUDE.md` and in `cr_sim/train/evaluate.py:159-169`. A constant, not a repeated movement | leftover | `scripts/measure_sampled_noise.py:1-20` | stub | — |
| `.claude/worktrees/agent-*/cr_sim/api/encoding.py` | leftover | `.claude/worktrees/**` | verified | [`worktree-shadows.md`](worktree-shadows.md) |

---
