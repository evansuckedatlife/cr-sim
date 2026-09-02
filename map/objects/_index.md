# Noun index — router

There is no whole-tree index. **The rows live one file down, per cluster**, so
no question loads four clusters it does not need:

| Cluster | Index | The question it answers |
|---|---|---|
| build | [`build/_index.md`](build/_index.md) | where a number comes from, what unit it is in, which ladder scales it |
| battle | [`battle/_index.md`](battle/_index.md) | what happens inside a match, and what makes it reproducible |
| interface | [`interface/_index.md`](interface/_index.md) | what the agent sees, may do, and is paid |
| measurement | [`measurement/_index.md`](measurement/_index.md) | what a number means, and what it was measured against |
| surfaces | [`surfaces/_index.md`](surfaces/_index.md) | who outside the training loop reads these nouns |

[`CONTEXT.md`](CONTEXT.md) argues why those five and not the package tree.

## How to read a row

`Noun | Universe | Owner | Status | Card`. **Owner** is a `path:line` rooted at
the repo root and is enough to reach source in one hop. **Card** is a link where
one exists and `—` where the index line *is* the whole entry, deliberately.

A row with a card carries **no gloss** — the gloss is on the card. That is not
brevity, it is one-home-per-fact: a count kept on an index row and on a card is
a count that will disagree, and did.

## Counts

Rebuilt from the tables, not asserted. Nothing here counts anything on disk —
`runs/`, `data_cache/` and `checkpoints/` change between one reading and the
next, and a checked-in count of generated data is wrong by the afternoon.

| Cluster | Index rows | Cards on disk |
|---|---|---|
| **build** | 47 | 11 |
| **battle** | 77 | 22 |
| **interface** | 66 | 12 |
| **measurement** | 63 | 17 |
| **surfaces** | 20 | 5 |
| **total** | **273** | **67** |

Most nouns share a card; that is what the Card column is for. The `surfaces`
cluster was the map's largest hole — nineteen nouns, five card names promised
and none written — and its change-impact lived in a hub instead. It has its five
cards now, and [`../effects/points-in.md`](../effects/points-in.md) lands on
them.

`../processes/` holds six verb cards, one per movement that actually runs. They
are not nouns and take no rows here.
