"""A Discord bot that watches training and answers questions about it.

:mod:`cr_sim.train.notify` only pushes -- it says when something happened and
nothing else. That covers "tell me if it breaks overnight", but not "how is it
going", which is the question actually asked, usually while away from the
machine.

So this does both: the same announcements, plus commands.

    /status              every run, newest first
    /run <name>          one run in detail, with trends
    /compare <a> <b>     two runs side by side
    /expert              what the search bot scores, as the bar to beat

Needs a bot token, which needs a bot created and invited to the server -- more
setup than a webhook, and worth it only for the commands. If all you want is
to be told when a run finishes, use notify.py instead.

    setx CR_SIM_DISCORD_TOKEN "..."
    setx CR_SIM_DISCORD_CHANNEL "123456789"
    python -m cr_sim.train.bot
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .notify import (NOISE, Event, changes, sparkline, use_utf8_console,
                     _load_state, _save_state)
from .watch import _started_at, read_metrics, summarise

__all__ = ["run_bot", "describe", "overview"]

ROOT = Path(__file__).resolve().parents[2]


def _runs(runs_dir: Path) -> list[tuple[str, list[dict[str, Any]], dict[str, Any]]]:
    """Every run with anything recorded, newest first."""
    found = []
    for run in sorted((d for d in runs_dir.iterdir() if d.is_dir()),
                      key=_started_at, reverse=True):
        rows = read_metrics(run / "metrics.jsonl")
        if not rows:
            continue
        by_update = {r.get("updates", i): r for i, r in enumerate(rows)}
        rows = [by_update[k] for k in sorted(by_update)]
        found.append((run.name, rows, summarise(rows)))
    return found


def _verdict(lift: float | None) -> str:
    if lift is None:
        return "not measured"
    if lift >= 0.5:
        return "clearly better"
    if lift >= NOISE:
        return "probably better"
    if lift > -NOISE:
        return "inside the noise"
    return "worse than random"


def overview(runs_dir: Path, limit: int = 10) -> str:
    """Every run in one message, newest first."""
    found = _runs(runs_dir)
    if not found:
        return "No runs have recorded anything yet."
    lines = ["**runs** — newest first", "```"]
    lines.append(f"{'run':<20}{'steps':>10}{'lift':>9}  verdict")
    for name, rows, summary in found[:limit]:
        lift = summary.get("latest_lift")
        lines.append(
            f"{name[:19]:<20}{summary.get('steps', 0):>10,}"
            f"{(f'{lift:+.3f}' if lift is not None else '--'):>9}"
            f"  {_verdict(lift)}")
    lines.append("```")
    if len(found) > limit:
        lines.append(f"…and {len(found) - limit} more.")
    return "\n".join(lines)


def describe(runs_dir: Path, name: str) -> str:
    """One run, with enough trend to see where it is heading."""
    for run_name, rows, summary in _runs(runs_dir):
        if run_name.lower() != name.lower():
            continue
        evaluations = [r["eval_lift_sd"] for r in rows if "eval_lift_sd" in r]
        variance = [r["explained_variance"] for r in rows
                    if r.get("explained_variance") is not None]
        entropy = [r["entropy"] for r in rows if "entropy" in r]
        ladder = [r for r in rows if "ancestor_win" in r]
        out = [f"**{run_name}**",
               f"{summary.get('steps', 0):,} steps"
               + (f" of {summary['total_steps']:,}" if summary.get("total_steps") else "")
               + f", {summary.get('episodes', 0):,} matches, "
                 f"{summary.get('steps_per_second', 0):.0f}/s"]
        if evaluations:
            half = len(evaluations) // 2 or 1
            early = sum(evaluations[:half]) / half
            late = sum(evaluations[half:]) / max(1, len(evaluations) - half)
            out.append(
                f"lift `{sparkline(evaluations)}` {evaluations[-1]:+.3f} "
                f"— {_verdict(evaluations[-1])}")
            out.append(f"  first half {early:+.3f} → second half {late:+.3f}")
        else:
            out.append("no evaluations yet")
        if variance:
            out.append(f"explained variance `{sparkline(variance)}` {variance[-1]:+.3f}"
                       + ("  (a critic at 0 makes the advantages noise)"
                          if variance[-1] < 0.1 else ""))
        if entropy:
            out.append(f"entropy `{sparkline(entropy)}` {entropy[0]:.2f} → {entropy[-1]:.2f}")
        if ladder:
            latest = ladder[-1]
            out.append(f"vs its own generation {latest['ancestor_age']}: "
                       f"{latest['ancestor_win']:.0%} w / {latest['ancestor_loss']:.0%} l")
        verdict_file = runs_dir / run_name / "verdict.json"
        if verdict_file.is_file():
            try:
                verdict = json.loads(verdict_file.read_text(encoding="utf-8"))
                out.append(
                    f"**{verdict['episodes']} paired battles**: "
                    f"{verdict['lift']:+.3f} sd "
                    f"[{verdict['ci_low']:+.3f}, {verdict['ci_high']:+.3f}]"
                    + ("  — the interval clears zero"
                       if verdict["ci_low"] > 0 else
                       "  — the interval contains zero, so this is unproven"))
            except (json.JSONDecodeError, KeyError, OSError):
                pass
        return "\n".join(out)
    return f"No run called `{name}`. Try /status."


def compare(runs_dir: Path, first: str, second: str) -> str:
    return describe(runs_dir, first) + "\n\n" + describe(runs_dir, second)


def run_bot(token: str, channel_id: int, runs_dir: Path | None = None,
            every: float = 60.0) -> None:
    """Connect, announce changes, and answer commands until stopped."""
    import discord
    from discord.ext import commands, tasks

    runs_dir = runs_dir or (ROOT / "runs")
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix="/", intents=intents, help_command=None)
    state: dict[str, Any] = _load_state(runs_dir)

    @tasks.loop(seconds=max(15.0, every))
    async def announce() -> None:
        nonlocal state
        channel = bot.get_channel(channel_id)
        if channel is None:
            return
        events, state = changes(runs_dir, state)
        _save_state(runs_dir, state)
        for event in events:
            embed = discord.Embed(title=event.title, colour=event.colour,
                                  description=event.body or None)
            for name, value, inline in event.fields:
                embed.add_field(name=name, value=value, inline=inline)
            await channel.send(embed=embed)

    @announce.before_loop
    async def _ready() -> None:
        await bot.wait_until_ready()
        # Seeded silently, so reconnecting does not replay every evaluation
        # a run has ever produced into the channel.
        nonlocal state
        if not state:
            _, state = changes(runs_dir, state)
            _save_state(runs_dir, state)

    @bot.event
    async def on_ready() -> None:
        print(f"connected as {bot.user}", flush=True)
        if not announce.is_running():
            announce.start()

    @bot.command(name="status")
    async def status(ctx) -> None:
        await ctx.send(overview(runs_dir))

    @bot.command(name="run")
    async def run(ctx, name: str) -> None:
        await ctx.send(describe(runs_dir, name))

    @bot.command(name="compare")
    async def compare_cmd(ctx, first: str, second: str) -> None:
        await ctx.send(compare(runs_dir, first, second))

    @bot.command(name="expert")
    async def expert(ctx) -> None:
        await ctx.send(describe(runs_dir, "search-expert"))

    @bot.command(name="help")
    async def help_cmd(ctx) -> None:
        await ctx.send(
            "`/status` every run · `/run <name>` one in detail · "
            "`/compare <a> <b>` two side by side · `/expert` the bar to beat")

    bot.run(token)


def main(argv: list[str] | None = None) -> int:
    import argparse

    use_utf8_console()

    parser = argparse.ArgumentParser(prog="cr-sim-bot")
    parser.add_argument("--token", default=os.environ.get("CR_SIM_DISCORD_TOKEN"))
    parser.add_argument("--channel", type=int,
                        default=int(os.environ.get("CR_SIM_DISCORD_CHANNEL", 0)))
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--every", type=float, default=60.0)
    args = parser.parse_args(argv)

    if not args.token or not args.channel:
        parser.error(
            "needs CR_SIM_DISCORD_TOKEN and CR_SIM_DISCORD_CHANNEL. For "
            "announcements only, cr_sim.train.notify needs just a webhook URL "
            "and no bot at all.")
    run_bot(args.token, args.channel, args.runs, args.every)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
