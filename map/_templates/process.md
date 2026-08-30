---
type: process   # System map verb card. The Context map process node is node.md.
status: stub    # verified requires ALL of: verified:, commit:, and a Citations block
verified:       # YYYY-MM-DD, filled in when status becomes verified
commit:         # short sha or branch the citations were checked against
consumes: []
produces: []
---

# {process-name}

{One sentence: the movement, not the nouns.}

## Input → Movement → Output

{Three sentences.}

## Why this shape

{What would break if the obvious shortcut existed.}

## Steps

1. {Cite `{path}:{line}`, rooted at the repo root.}

## If you change this

- **Hits:**
- **Does not hit:**

## Surfaces

| Surface | Role |
|---|---|
| {who} | {role} |

## Citations

{Every `path:line` this card rests on, and the basis.}

Verified {YYYY-MM-DD} against `{branch}` @ `{commit}`.

*This block is not optional. `status: verified` without a date, a commit and
citations is `stub` — `../_meta/schema.md`. It has a slot here because six of
six cards omitted it while it was only a convention.*

## See

- Objects: {links}
- Source: `{path}`
