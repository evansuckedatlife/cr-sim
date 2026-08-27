"""Put a job on the progress page, so nothing worth knowing lives only in a chat log.

The page enumerates ``runs/*/metrics.jsonl``, which means it shows training
runs and nothing else. Everything else this project spends its time on --
benchmarks, head-to-head comparisons between checkpoints, a long piece of work
an agent is part-way through -- was invisible on it, and the only record of
what a number meant was whatever conversation produced it.

So those become entries too. A job writes the same two files a run does:
``metrics.jsonl`` for anything numeric, and ``config.json`` carrying a ``note``
that says what the entry is, which the page now renders above the charts.

    python scripts/register_job.py --name bench-network \
        --note "..." --rows rows.json

    python scripts/register_job.py --name agent-engine-speed \
        --note "Profiling the tick loop. Running." --status running

A job with nothing measured yet still gets one row, because the page skips a
directory whose metrics file is empty and an entry that appears only once it
has an answer is exactly the entry you needed while waiting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def register(name: str, note: str, rows: list[dict] | None = None,
             runs_dir: Path | None = None, status: str = "") -> Path:
    """Write ``runs/<name>/`` so the watcher picks it up on its next rescan."""
    runs_dir = runs_dir or (ROOT / "runs")
    out = runs_dir / name
    out.mkdir(parents=True, exist_ok=True)

    body = list(rows or [])
    if not body:
        body = [{"updates": 1, "steps": 0, "episodes": 0}]
    # Numbered from one if the caller did not number them, so the page's
    # x-axis is the order they were produced in rather than all zero.
    for index, row in enumerate(body, start=1):
        row.setdefault("updates", index)

    (out / "metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in body), encoding="utf-8")

    stamped = note if not status else f"[{status}] {note}"
    (out / "config.json").write_text(json.dumps({
        "note": stamped,
        "kind": "job",
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="register-job")
    parser.add_argument("--name", required=True,
                        help="directory name under runs/, and the index label")
    parser.add_argument("--note", required=True,
                        help="what this entry is, in prose. Shown above the "
                             "charts. Say what was measured and against what, "
                             "because a lift number with no opponent named is "
                             "not a number anyone can use.")
    parser.add_argument("--status", default="",
                        help="running / done / stopped, prefixed to the note")
    parser.add_argument("--rows", type=Path, default=None,
                        help="JSON file holding a list of metrics rows, or - "
                             "for stdin. Omit for a job with nothing measured "
                             "yet.")
    parser.add_argument("--runs", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = None
    if args.rows is not None:
        text = sys.stdin.read() if str(args.rows) == "-" else args.rows.read_text(encoding="utf-8")
        rows = json.loads(text)
        if not isinstance(rows, list):
            parser.error("--rows must hold a JSON list of objects")

    out = register(args.name, args.note, rows, runs_dir=args.runs,
                   status=args.status)
    print(f"{out}  ({len(rows or [1])} row(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
