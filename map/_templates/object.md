---
type: object
cluster: {cluster}
universe: live | leftover | ghost | deliberate ghost
status: stub    # verified requires ALL of: verified:, commit:, and citations
verified:       # YYYY-MM-DD
commit:         # short sha or branch
entity: {path to the owning file, from the repo root}
---

# {Name}

{One sentence. If the product word and the file/type name differ, say both.}

## Why this shape

{The load-bearing why, not a field tour.}

## Shape

- {keys, constraints, or owning files}

Citations: `{path}:{line}` — rooted at the repo root, always
(`../../_meta/schema.md`, Naming). Verified {YYYY-MM-DD} against `{branch}` @
`{commit}`.

## Connected to

- **owns:**
- **owned-by:**
- **joins:**
- **looks-like-but-is-not:**

## If you change this

- **Hits:**
- **Does not hit:**

## Surfaces

| Surface | Role |
|---|---|
| {who} | {reads / writes / none} |

## See

- Source: `{path}`
- If this card overrides a comment, add a row to `../../_meta/overrides.md`.
