"""Report training to Discord, so a run can be watched from anywhere.

The page it replaces has two problems that no amount of design fixes. It only
works on the local network, and iOS only delivers web push to an installed app
over HTTPS -- so a phone could watch a run while the tab was open and learn
nothing once it was not. Discord already solves both: it is reachable from
anywhere, it is already installed, and its notifications already work.

Two ways in, and the difference matters when setting it up:

*   **A webhook** is a URL. Nothing to authorise, nothing to keep logged in,
    and it can only post. Right for "tell me when something happens".
*   **A bot token** additionally lets it answer questions -- how is the run
    doing, what have all the runs done. Needs a bot created and invited.

The webhook path is the one to start with, and this module works with only
that.

What it says is chosen to be worth interrupting someone for. A new evaluation
that moved the verdict, a run that finished, a run that died -- not every
update, of which there are thousands, and not a number that has not changed.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .watch import read_metrics, summarise

__all__ = ["Event", "changes", "sparkline", "post", "watch_forever",
           "use_utf8_console"]

ROOT = Path(__file__).resolve().parents[2]


def use_utf8_console() -> None:
    """Make the em dash, the ellipsis, and the sparkline blocks survive print.

    Windows does not open a console in UTF-8: Python's ``sys.stdout`` inherits
    whatever code page the console started with, commonly cp1252 or cp437,
    and neither can encode U+2581..U+2588. The result was not a crash but
    silent corruption -- an em dash printed as U+FFFD -- which is worse,
    because nothing signals that the output is wrong. Reconfiguring the
    stream to UTF-8 is what a Windows console actually renders correctly, and
    costs nothing on a terminal that was UTF-8 already. Discord itself never
    sees this: embeds go over HTTP as UTF-8 JSON regardless.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            # A stream with no reconfigure (e.g. a test harness's capture
            # buffer) is already producing correct text; nothing to fix.
            pass


#: Under this, a lift reading is inside the control's own noise and means
#: nothing. Six evaluations on this project averaged +0.04 while individual
#: ones reached +0.23, so a message announcing one would be announcing noise.
NOISE = 0.25

#: Runs quiet for longer than this are reported as stopped. Generous: an
#: update at the slowest measured cadence takes a couple of minutes, and
#: crying wolf about a healthy run is worse than being slow to notice a dead
#: one.
STALE_SECONDS = 420.0

_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float], width: int = 24) -> str:
    """A trend small enough to fit in a chat message.

    A number on its own cannot say whether it is the top of a climb or the
    bottom of a slide, and that is usually the whole question.
    """
    points = [v for v in values if v is not None]
    if len(points) < 2:
        return ""
    if len(points) > width:
        step = len(points) / width
        points = [points[int(i * step)] for i in range(width)]
    low, high = min(points), max(points)
    if high - low < 1e-9:
        return _BLOCKS[0] * len(points)
    return "".join(
        _BLOCKS[min(len(_BLOCKS) - 1,
                    int((v - low) / (high - low) * (len(_BLOCKS) - 1)))]
        for v in points
    )


@dataclass(slots=True)
class Event:
    """Something worth saying out loud."""

    run: str
    kind: str
    title: str
    body: str
    colour: int = 0x58B4D0
    fields: list[tuple[str, str, bool]] = field(default_factory=list)


def _verdict(lift: float | None) -> tuple[str, int]:
    if lift is None:
        return "not measured", 0x8695A4
    if lift >= 0.5:
        return "clearly better than random", 0x5FB68C
    if lift >= NOISE:
        return "probably better than random", 0x5FB68C
    if lift > -NOISE:
        return "inside the noise", 0xDCAB57
    return "worse than random", 0xE58177


def _state_path(runs_dir: Path) -> Path:
    return runs_dir / ".notified.json"


def _load_state(runs_dir: Path) -> dict[str, Any]:
    path = _state_path(runs_dir)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(runs_dir: Path, state: dict[str, Any]) -> None:
    try:
        _state_path(runs_dir).write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        pass


def changes(runs_dir: Path, state: dict[str, Any] | None = None,
            now: float | None = None) -> tuple[list[Event], dict[str, Any]]:
    """What has happened since the last look.

    Returns the events and the state to remember. Separated from posting so
    the decision of what is worth saying can be tested without a network, and
    so a restart does not re-announce a fortnight of finished runs.
    """
    now = time.time() if now is None else now
    state = dict(state or {})
    events: list[Event] = []

    if not runs_dir.is_dir():
        return events, state

    for run in sorted(runs_dir.iterdir()):
        metrics = run / "metrics.jsonl"
        if not run.is_dir() or not metrics.is_file():
            continue
        rows = read_metrics(metrics)
        if not rows:
            continue
        by_update = {r.get("updates", i): r for i, r in enumerate(rows)}
        rows = [by_update[k] for k in sorted(by_update)]
        summary = summarise(rows)
        evaluations = [r for r in rows if "eval_lift_sd" in r]
        seen = state.get(run.name, {})
        fresh = (now - metrics.stat().st_mtime) < STALE_SECONDS

        if not seen:
            # First sighting. Recorded silently rather than announced: a bot
            # that shouts about every run it finds on startup gets muted.
            state[run.name] = {"evaluations": len(evaluations),
                               "live": fresh, "steps": summary.get("steps", 0)}
            continue

        if len(evaluations) > seen.get("evaluations", 0):
            latest = evaluations[-1]
            lift = latest["eval_lift_sd"]
            label, colour = _verdict(lift)
            trend = sparkline([e["eval_lift_sd"] for e in evaluations])
            variance = [r.get("explained_variance") for r in rows
                        if r.get("explained_variance") is not None]
            body = f"`{trend}`" if trend else ""
            fields = [
                ("lift", f"{lift:+.3f} sd — {label}", True),
                ("steps", f"{summary.get('steps', 0):,}"
                          + (f" / {summary['total_steps']:,}"
                             if summary.get("total_steps") else ""), True),
            ]
            if variance:
                fields.append(("explained var", f"{variance[-1]:+.3f}", True))
            if latest.get("ancestor_win") is not None:
                fields.append((
                    "vs its past self",
                    f"{latest['ancestor_win']:.0%} w / {latest['ancestor_loss']:.0%} l",
                    True))
            events.append(Event(
                run=run.name, kind="evaluation",
                title=f"{run.name} — evaluation {len(evaluations)}",
                body=body, colour=colour, fields=fields))

        if seen.get("live") and not fresh:
            done = (summary.get("total_steps")
                    and summary.get("steps", 0) >= summary["total_steps"])
            events.append(Event(
                run=run.name, kind="finished" if done else "stopped",
                title=f"{run.name} — {'finished' if done else 'stopped'}",
                body=(f"{summary.get('steps', 0):,} steps, "
                      f"{summary.get('episodes', 0):,} matches."
                      + ("" if done else " No update in seven minutes.")),
                colour=0x5FB68C if done else 0xDCAB57))

        state[run.name] = {"evaluations": len(evaluations), "live": fresh,
                           "steps": summary.get("steps", 0)}
    return events, state


def post(webhook: str, event: Event, timeout: float = 10.0) -> bool:
    """Send one event. Returns whether it landed.

    Failures are reported rather than raised: a monitor that dies because the
    network hiccuped is worse than one that misses a message.
    """
    embed: dict[str, Any] = {
        "title": event.title,
        "color": event.colour,
    }
    if event.body:
        embed["description"] = event.body
    if event.fields:
        embed["fields"] = [{"name": n, "value": v, "inline": i}
                           for n, v, i in event.fields]
    payload = json.dumps({"embeds": [embed]}).encode()
    request = urllib.request.Request(
        webhook, data=payload,
        headers={"Content-Type": "application/json",
                 "User-Agent": "cr-sim-notify/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _print_event(event: Event) -> None:
    """Show what would be posted -- the shape of the Discord embed, in text.

    Used by ``--dry-run`` so the wording and the numbers can be checked
    without a webhook, and incidentally by anyone testing by eye: this is
    also where the em dash and the sparkline blocks get exercised, so it
    doubles as the manual check for the console-encoding fix above.
    """
    print(f"[{event.kind}] {event.title}")
    if event.body:
        print(f"  {event.body}")
    for name, value, _inline in event.fields:
        print(f"  {name}: {value}")


def watch_forever(webhook: str | None, runs_dir: Path | None = None,
                  every: float = 60.0, announce: bool = True,
                  dry_run: bool = False) -> None:
    """Poll the runs directory and report what changes.

    With ``dry_run``, events are printed instead of posted and no webhook is
    required -- the same loop, checkable without a Discord server to send to.
    """
    runs_dir = runs_dir or (ROOT / "runs")
    sink = _print_event if dry_run else (lambda event: post(webhook, event))
    state = _load_state(runs_dir)
    if announce:
        sink(Event(
            run="", kind="hello", title="cr-sim is watching",
            body=f"Reporting on `{runs_dir}` every {every:.0f}s.",
            colour=0x58B4D0))
    # Seed silently on the first pass so a restart does not replay history.
    if not state:
        _, state = changes(runs_dir, state)
        _save_state(runs_dir, state)
    try:
        while True:
            time.sleep(max(5.0, every))
            events, state = changes(runs_dir, state)
            for event in events:
                sink(event)
            _save_state(runs_dir, state)
    except KeyboardInterrupt:
        return


def main(argv: list[str] | None = None) -> int:
    import argparse

    use_utf8_console()

    parser = argparse.ArgumentParser(prog="cr-sim-notify")
    parser.add_argument(
        "--webhook", default=os.environ.get("CR_SIM_DISCORD_WEBHOOK"),
        help="Discord webhook URL. Defaults to $CR_SIM_DISCORD_WEBHOOK, which "
             "is the better place for it -- a URL on a command line ends up "
             "in shell history and in screenshots.",
    )
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--every", type=float, default=60.0)
    parser.add_argument("--once", action="store_true",
                        help="report what changed and exit")
    parser.add_argument("--quiet-start", action="store_true",
                        help="do not announce that it started watching")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the events instead of posting them, so wording and "
             "numbers can be checked without a webhook",
    )
    args = parser.parse_args(argv)

    if not args.webhook and not args.dry_run:
        parser.error(
            "no webhook. In Discord: Server Settings > Integrations > "
            "Webhooks > New Webhook > Copy Webhook URL, then set "
            "CR_SIM_DISCORD_WEBHOOK to it. Or pass --dry-run to check the "
            "output without one.")

    if args.once:
        state = _load_state(args.runs)
        events, state = changes(args.runs, state)
        for event in events:
            if args.dry_run:
                _print_event(event)
            else:
                print(f"{event.kind}: {event.title}")
                post(args.webhook, event)
        _save_state(args.runs, state)
        print(f"{len(events)} event(s)")
        return 0

    watch_forever(args.webhook, args.runs, args.every,
                  announce=not args.quiet_start, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
