---
type: object
cluster: measurement
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/train/scripted.py
---

# Search bot

The one-ply expert every demonstration comes from and every hard anchor is —
`SearchBotConfig`, `SearchBot`, `battle_stream_seed`, plus the proposer that
decides *which* placements it spends its branches on (`policy_proposer`,
`proposer_identity`).

Verified 2026-08-30 against the working tree at `dc47f51` (`scripted.py` is
among the nine uncommitted files — `../../CONTEXT.md`).

## Why this shape

It knows nothing. It asks the engine what happens and keeps the best answer —
which is why it is a usable teacher without any reward design, and why it is
**stateless between decisions**: a plan made two seconds ago was made against a
board that no longer exists (`cr_sim/train/scripted.py:180-182`).

Two properties are what make it *measurable* rather than merely good:

- **It is rebuilt per battle, from a seed derived from the battle's own.** The
  bot samples its candidates, so a bot carried across episodes draws as a
  function of how many decisions came before rather than of the seed — and two
  arms of a paired evaluation then stop playing the same battle from their
  first different move (`:54-60`). `battle_stream_seed` `:52` is the one
  derivation the package uses; it replaced three spellings, one of which
  (`scripts/make_demos.py`) keyed on the shard index instead, so every episode
  in a shard drew from one stream while the shard's proposer was correctly
  keyed per battle `:62-67`.
- **The clamp is recorded, not applied silently.** `policy_candidates` is cut
  to leave the random floor intact, and `requested_policy_candidates` keeps
  what was asked for beside what was taken `:208-218` — a bot that quietly took
  fewer would make an equal-budget comparison a comparison of something else.
  `effective_policy_candidates` `:164` is the one implementation; stamping the
  requested value instead once entered one ladder as two entrants splitting one
  bot's games (`cr_sim/train/ladder.py:176-188`).

## Shape

- `SearchBotConfig` `:76`, `random_floor` `:157`,
  `effective_policy_candidates` `:164`. `SearchBot` `:178`, `__init__` `:189`,
  `clamped` `:255`, `_candidates` `:308`, `__call__` `:397`.
- `last_scores` `:237-244` — every action the last decision looked at, with
  what it scored. **The single chosen action is a poor training target**:
  candidates are sampled, so the same board yields a different choice depending
  on what was drawn; the expert is not a function of the state and no policy
  can learn one. The scores are. That is what
  [`demonstrations.md`](demonstrations.md) trains against.
- `policy_proposer` `cr_sim/train/proposal.py:80`. `temperature == 0.0` — the
  default — **touches no generator at all**: a stable argsort over a numpy
  float64 copy, so ties break by ascending flat index, because `torch.topk`
  promises no ordering among exact ties and the factored head produces them
  routinely `:133-137`. Above zero it draws from a generator this proposer
  owns, keyed on `(proposer seed, battle seed, decision)` `:144-146`.
- `proposer_identity` `:173` names the weights by **content** (a SHA-256
  prefix), not by path: `runs/iter-2/cloned.pt` is a different network on
  Tuesday than on Monday, and a shard naming the path would merge cleanly with
  one collected from the earlier file.
- `check_equal_branch_budget` `:204` refuses two searches at different
  `candidates` or `horizon_seconds`. It lives here, not in `ladder.py`.
- **Ghost:** `policy_proposer(battle_seed_of=...)` `:81`, documented as
  normally `None` because the bot is rebuilt per battle everywhere in this
  package `:101-105`. **No caller passes it** — the only three occurrences of
  the name are the parameter, its docstring and its own default `:144`.

## Connected to

- **owns:** the candidate set and the value distribution over it.
- **owned-by:** [`demonstrations.md`](demonstrations.md) — the shards are what
  this exists to produce.
- **joins:** [`ladder.md`](ladder.md) (`search-cXhY` is a `Player`, and
  `Player.ref` stamps the proposer into the rating);
  [`lift.md`](lift.md) (`search_opponent` is the control's opponent when the
  random one is used up); [`random-streams.md`](random-streams.md).
- **looks-like-but-is-not:** `_random_opponent` (`cr_sim/train/run.py:60`). It
  is also an opponent built per environment, but its generator **carries across
  episodes** and is left that way deliberately — fixing it would silently move
  the scale every recorded verdict sits on
  (`cr_sim/train/evaluate.py:259-262`).

## If you change this

- **Hits:** every demonstration shard's labels, and therefore every clone
  fine-tuned from one. A different proposer is a different supervision signal
  over the same inputs, which is why `_agree` refuses to merge across it
  (`scripts/clone_policy.py:99`). Changing `candidates` or `horizon_seconds`
  changes which *opponent* a `search-cXhY` anchor is: a thinned expert is a
  different player, and `ladder._opponent_policy` stamps the player's own name
  rather than the bare "search" for exactly that reason
  (`cr_sim/train/ladder.py:485-489`).
- **Does not hit:** the engine's determinism, and it does not need to. The
  search **rescores every proposal with the exact engine**, so a difference in
  proposal *order* changes nothing and only a difference in the candidate *set*
  can change a decision (`cr_sim/train/proposal.py:40-48`). The obvious next worry — that a
  network forward is not bit-stable across thread counts, so guided
  demonstrations are irreproducible — is bounded by that, and by
  `proposer_identity` stamping the thread count into the shard. Bit-determinism
  across machines is claimed only for `proposer=None`, where it is true.

## Surfaces

| Surface | Role |
|---|---|
| `scripts/make_demos.py` | runs it to produce shards |
| `cr_sim/train/evaluate.py` `search_opponent` | puts it on the other side of the net |
| `cr_sim/train/ladder.py` | enters it as a rated `Player` |
| `scripts/measure_expert.py` | measures its anchor lift |
| `scripts/bench_engine.py` | profiles a decision; not a measurement of a policy |

## See

- Source: `cr_sim/train/scripted.py`, `cr_sim/train/proposal.py`
