"""Put a job on the progress page, so nothing worth knowing lives only in a chat log.

The page enumerates ``runs/*/metrics.jsonl``, which means it shows training
runs and nothing else. Everything else this project spends its time on --
benchmarks, head-to-head comparisons between checkpoints, a long piece of work
an agent is part-way through -- was invisible on it, and the only record of
what a number meant was whatever conversation produced it.

So those become entries too. A job writes the same two files a run does:
``metrics.jsonl`` for anything numeric, and ``config.json`` carrying a ``note``
that says what the entry is, which the page now renders above the charts.

    python scripts/register_job.py --name bench-network --note "..." --rows rows.json

    python scripts/register_job.py --name agent-engine-speed \
        --note "Profiling the tick loop. Running." --status running

A job with nothing measured yet still gets one row, because the page skips a
directory whose metrics file is empty and an entry that appears only once it
has an answer is exactly the entry you needed while waiting.

**It will not write over an entry that already holds readings.** Both files
used to be written unconditionally, so one mistyped ``--name`` would replace a
training run's several hundred rows -- and the ``tower_level``,
``eval_opponent``, ``init_from`` and ``kl_reference`` in its config, which are
the only record of what those rows mean -- with a single placeholder row, exit
code 0 and a cheerful success line. ``runs/`` is gitignored and there is no
backup, so that is unrecoverable. A name already carrying real data is now
refused, and the two ways past it are explicit:

``--append`` keeps every existing row and adds the new ones after them,
continuing the ``updates`` numbering, and keeps every key in the existing
config except the note. ``--replace`` moves the old files aside to
``metrics.<stamp>.jsonl`` and ``config.<stamp>.json`` -- which the watcher does
not enumerate, since it globs the exact name -- rather than truncating them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The keys the placeholder row below is made of. A metrics file holding one
#: row and nothing outside this set has measured nothing, which is what a
#: ``--status running`` entry looks like before it has an answer -- so
#: re-registering it to say "done" destroys nothing and stays allowed.
_PLACEHOLDER_KEYS = {"updates", "steps", "episodes"}


class RunHasData(Exception):
    """Refusing to overwrite an entry that already holds readings."""


def _existing_rows(path: Path) -> list[dict]:
    """Every line already in a metrics file, damaged ones included.

    A line this cannot parse is still a line somebody wrote, and counting it
    is the whole point: the question being asked is "is there anything here
    to lose", and answering "no" because the file is malformed is exactly the
    wrong way round.
    """
    if not path.is_file():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            out.append({"unparsed": line})
            continue
        out.append(row if isinstance(row, dict) else {"unparsed": line})
    return out


def _why_protected(out: Path) -> str:
    """Why this directory must not be written over, or "" if it may be."""
    metrics = out / "metrics.jsonl"
    rows = _existing_rows(metrics)
    measured = [r for r in rows if set(r) - _PLACEHOLDER_KEYS]
    if len(rows) > 1 or measured:
        return (f"{metrics} already holds {len(rows)} row(s), "
                f"{len(measured)} of them carrying measurements")
    config = out / "config.json"
    if config.is_file():
        try:
            raw = json.loads(config.read_text(encoding="utf-8"))
        except (ValueError, OSError, UnicodeDecodeError):
            return f"{config} is unreadable, so what it records cannot be checked"
        if not isinstance(raw, dict) or raw.get("kind") != "job":
            # A training run's config is the only record of the tower level,
            # the opponent, the weights it started from and the anchor it was
            # held to. None of that exists anywhere else.
            kind = raw.get("kind") if isinstance(raw, dict) else raw
            return f"{config} was not written by this script (kind={kind!r})"
    return ""


def _stamped_aside(path: Path) -> "Path | None":
    """Move ``path`` out of the way under a timestamp, and say where it went.

    Named ``metrics.<stamp>.jsonl`` deliberately: ``watch.discover`` globs the
    exact string ``metrics.jsonl``, so the moved file keeps its contents
    without reappearing on the page as a second copy of the entry.
    """
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for attempt in range(100):
        suffix = stamp if not attempt else f"{stamp}-{attempt}"
        target = path.with_name(f"{path.stem}.{suffix}{path.suffix}")
        if not target.exists():
            path.rename(target)
            return target
    raise RunHasData(f"cannot find a free name to move {path} aside")


def register(name: str, note: str, rows: "list[dict] | None" = None,
             runs_dir: "Path | None" = None, status: str = "",
             append: bool = False, replace: bool = False) -> Path:
    """Write ``runs/<name>/`` so the watcher picks it up on its next rescan.

    Raises ``RunHasData`` rather than overwriting an entry that already holds
    readings, unless ``append`` or ``replace`` says which of the two things to
    do with them.
    """
    runs_dir = runs_dir or (ROOT / "runs")
    out = runs_dir / name
    out.mkdir(parents=True, exist_ok=True)
    metrics = out / "metrics.jsonl"
    config = out / "config.json"

    kept: list[dict] = []
    existing_config: dict = {}
    had_config = False
    if append:
        kept = _existing_rows(metrics)
        raw = None
        had_config = config.is_file()
        if config.is_file():
            try:
                raw = json.loads(config.read_text(encoding="utf-8"))
            except (ValueError, OSError, UnicodeDecodeError):
                raw = None
        # Every key that was there survives. A run's tower_level and
        # eval_opponent are the only record of what its rows mean, and adding
        # a row to a file is not a reason to forget them.
        if isinstance(raw, dict):
            existing_config = dict(raw)
    elif replace:
        _stamped_aside(metrics)
        _stamped_aside(config)
    else:
        reason = _why_protected(out)
        if reason:
            raise RunHasData(
                f"{out} is not an empty slot: {reason}. Pick another --name, "
                "or pass --append to add rows to it, or --replace to move the "
                "existing files aside first.")

    body = list(rows or [])
    if not body and not kept:
        body = [{"updates": 1, "steps": 0, "episodes": 0}]
    # Numbered from one if the caller did not number them, so the page's
    # x-axis is the order they were produced in rather than all zero --
    # continuing past whatever is already in the file when appending, because
    # restarting at one draws the new rows back underneath the old ones.
    start = 1
    for row in kept:
        try:
            start = max(start, int(row.get("updates", 0)) + 1)
        except (TypeError, ValueError):
            pass
    for index, row in enumerate(body, start=start):
        row.setdefault("updates", index)

    metrics.write_text(
        "".join(json.dumps(r) + "\n" for r in kept + body), encoding="utf-8")

    stamped = note if not status else f"[{status}] {note}"
    payload = dict(existing_config)
    payload["note"] = stamped
    payload["registered_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    # Only a config this script created says "job". Appending a row to a
    # training run does not turn it into one, and the page reads this key to
    # decide what the entry is.
    if not had_config:
        payload["kind"] = "job"

    config.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def main(argv: "list[str] | None" = None) -> int:
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
    parser.add_argument("--append", action="store_true",
                        help="add these rows after the ones already there, "
                             "keeping every existing config key but the note")
    parser.add_argument("--replace", action="store_true",
                        help="move the existing metrics.jsonl and config.json "
                             "aside under a timestamp and start fresh. Not a "
                             "delete: the old files stay in the directory "
                             "under names the watcher does not enumerate.")
    parser.add_argument("--runs", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.append and args.replace:
        parser.error("--append and --replace ask for opposite things")

    rows = None
    if args.rows is not None:
        text = sys.stdin.read() if str(args.rows) == "-" else args.rows.read_text(encoding="utf-8")
        rows = json.loads(text)
        if not isinstance(rows, list):
            parser.error("--rows must hold a JSON list of objects")

    try:
        out = register(args.name, args.note, rows, runs_dir=args.runs,
                       status=args.status, append=args.append,
                       replace=args.replace)
    except RunHasData as exc:
        print(f"register-job: refusing to write over data: {exc}",
              file=sys.stderr)
        return 2

    print(f"{out}  ({len(_existing_rows(out / 'metrics.jsonl'))} row(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
