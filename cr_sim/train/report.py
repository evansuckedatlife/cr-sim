"""``python -m cr_sim.train.report`` -- every run, one page each, plus an index.

The single live page answers "how is this one going". This answers the other
question, which is the one that actually decides what to build next: *which of
these worked, and what was different about it*. Six runs have been done on this
project and the only place they had ever been compared was in conversation.

Each run gets its own page, rendered by :mod:`cr_sim.train.watch`, so the
detail view is the same one used to watch a run live. The index puts them side
by side with the settings that differed, because a lift number means nothing
without knowing what the opponent was.

Runs are read from disk and nothing is recomputed, with one exception: a run
may carry a ``verdict.json`` written by a larger paired evaluation than the
trainer does inline. Where one exists it is shown in place of the run's own
40-battle readings, because that is the number worth believing.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

from .watch import read_metrics, render, summarise

__all__ = ["collect", "render_index", "main"]

ROOT = Path(__file__).resolve().parents[2]


def _config(run: Path) -> dict[str, Any]:
    path = run / "config.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _verdict(run: Path) -> dict[str, Any] | None:
    path = run / "verdict.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def collect(runs_dir: Path) -> list[dict[str, Any]]:
    """One record per run that has recorded anything.

    Deduplicated by update number: an early bug wrote every update twice, and
    those files are still on disk. Averaging over them would count each update
    twice, which is exactly the reading that made a stalled run look busy.
    """
    from .watch import _started_at

    found: list[dict[str, Any]] = []
    # Chronological, matching the live page. A comparison table sorted by name
    # puts this week's run between two from last week, and the question being
    # asked is what changed since the previous one.
    for run in sorted((p for p in runs_dir.iterdir() if p.is_dir()), key=_started_at):
        rows = read_metrics(run / "metrics.jsonl")
        if not rows:
            continue
        by_update = {r.get("updates", i): r for i, r in enumerate(rows)}
        rows = [by_update[k] for k in sorted(by_update)]
        evaluations = [r["eval_lift_sd"] for r in rows if "eval_lift_sd" in r]
        variance = [r["explained_variance"] for r in rows if "explained_variance" in r]
        config = _config(run)
        found.append(
            {
                "name": run.name,
                "rows": rows,
                "summary": summarise(rows),
                "config": config,
                "verdict": _verdict(run),
                "evaluations": evaluations,
                "mean_lift": statistics.fmean(evaluations) if evaluations else None,
                "best_lift": max(evaluations) if evaluations else None,
                # Split in half rather than fitted: with fewer than twenty
                # readings a slope is mostly noise, and "did the second half
                # beat the first" is the honest version of the same question.
                "early_lift": (
                    statistics.fmean(evaluations[: len(evaluations) // 2])
                    if len(evaluations) >= 4 else None
                ),
                "late_lift": (
                    statistics.fmean(evaluations[len(evaluations) // 2:])
                    if len(evaluations) >= 4 else None
                ),
                "final_variance": (
                    statistics.fmean(variance[-max(1, len(variance) // 4):])
                    if variance else None
                ),
                "entropy_from": rows[0].get("entropy"),
                "entropy_to": rows[-1].get("entropy"),
            }
        )
    return found


def _verdict_of(record: dict[str, Any]) -> tuple[str, str, str]:
    """A chip class, a label, and the sentence under it."""
    verdict = record["verdict"]
    if verdict is not None:
        lo, hi = verdict["ci_low"], verdict["ci_high"]
        # Named, never assumed. These chips used to read "beats random"
        # whatever the verdict was measured against, and a run evaluated
        # against the search expert would have published on this page under
        # that label, beside genuine random-opponent lifts. The same policy
        # scores wildly differently against an idle, a random and a searching
        # opponent -- 92% of idle matches go to the control and 26% of random
        # ones -- and that confusion has already cost this project two rounds
        # of invalid comparisons. write_verdict refuses to record a lift
        # without eval_opponent; this is the reading half of that guard.
        foe = verdict.get("eval_opponent") or "an unnamed opponent"
        where = f"{verdict['episodes']} paired battles against {foe}"
        if lo > 0:
            return ("good", f"beats {foe}",
                    f"{where} put the lift at "
                    f"{verdict['lift']:+.3f} sd, and the whole 95% interval "
                    f"[{lo:+.3f}, {hi:+.3f}] sits above zero.")
        if hi < 0:
            return ("crit", f"worse than {foe}",
                    f"{where} put the lift at "
                    f"{verdict['lift']:+.3f} sd, entirely below zero.")
        return ("warn", "not distinguishable",
                f"{where} put the lift at "
                f"{verdict['lift']:+.3f} sd, but the 95% interval "
                f"[{lo:+.3f}, {hi:+.3f}] contains zero.")
    mean = record["mean_lift"]
    if mean is None:
        return ("dim", "never evaluated",
                "This run recorded no evaluations, so nothing here says "
                "whether it learned anything.")
    if mean >= 0.25:
        return ("good", "promising",
                f"Mean lift {mean:+.3f} sd over {len(record['evaluations'])} "
                "readings of 40 battles each. Worth a larger evaluation.")
    return ("warn", "inside the noise",
            f"Mean lift {mean:+.3f} sd over {len(record['evaluations'])} "
            "readings of 40 battles each, which is not enough to separate a "
            "weak effect from zero.")


def _fmt(value: Any, spec: str = ".3f", plus: bool = False) -> str:
    if value is None:
        return "&mdash;"
    if isinstance(value, (int, float)):
        text = format(value, ("+" if plus else "") + spec)
        return text
    return str(value)


def render_index(records: Sequence[dict[str, Any]]) -> str:
    """The comparison page."""
    rows = []
    for r in records:
        cls, label, _ = _verdict_of(r)
        s, c = r["summary"], r["config"]
        rows.append(
            "<tr>"
            f'<td><a href="{r["name"]}.html">{r["name"]}</a></td>'
            f'<td><span class="chip {cls}">{label}</span></td>'
            f'<td class="n">{s.get("steps", 0):,}</td>'
            f'<td class="n">{s.get("episodes", 0):,}</td>'
            f'<td>{c.get("reward", "&mdash;")}</td>'
            f'<td>{c.get("opponent", "&mdash;")}</td>'
            f'<td class="n">{c.get("frame_skip", "&mdash;")}</td>'
            f'<td class="n">{_fmt(r["mean_lift"], plus=True)}</td>'
            f'<td class="n">{_fmt(r["best_lift"], plus=True)}</td>'
            f'<td class="n">{_fmt(r["final_variance"], plus=True)}</td>'
            "</tr>"
        )

    cards = []
    for r in records:
        cls, label, sentence = _verdict_of(r)
        s, c = r["summary"], r["config"]
        trend = ""
        if r["early_lift"] is not None:
            direction = "rose" if r["late_lift"] > r["early_lift"] else "fell"
            trend = (
                f"<p>Lift {direction} from {r['early_lift']:+.3f} sd over the "
                f"first half of the run to {r['late_lift']:+.3f} over the "
                "second.</p>"
            )
        entropy = ""
        if r["entropy_from"] is not None and r["entropy_to"] is not None:
            entropy = (
                f"<p>Entropy went {r['entropy_from']:.3f} &rarr; "
                f"{r['entropy_to']:.3f}. Falling entropy is the policy "
                "committing, which is only good news alongside rising lift.</p>"
            )
        cards.append(
            f'<div class="card {cls}">'
            f'<div class="card-top"><h3><a href="{r["name"]}.html">{r["name"]}</a></h3>'
            f'<span class="chip {cls}">{label}</span></div>'
            f"<p>{sentence}</p>{trend}{entropy}"
            f'<div class="settings">'
            f'<span>{s.get("steps", 0):,} decisions</span>'
            f'<span>{s.get("episodes", 0):,} matches</span>'
            f'<span>{c.get("reward", "unknown")} reward</span>'
            f'<span>{c.get("opponent", "unknown")} opponent</span>'
            f'<span>frame-skip {c.get("frame_skip", "?")}</span>'
            "</div></div>"
        )

    return _INDEX.replace("__ROWS__", "\n".join(rows)).replace(
        "__CARDS__", "\n".join(cards)
    ).replace("__COUNT__", str(len(records)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cr-sim-report")
    parser.add_argument("runs", nargs="?", type=Path, default=ROOT / "runs")
    parser.add_argument("--out", type=Path, default=ROOT / "report")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args(argv)

    records = collect(args.runs)
    if not records:
        print(f"no runs with metrics under {args.runs}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.html").write_text(render_index(records), encoding="utf-8")
    for record in records:
        page = render(record["rows"], record["name"], back="index.html")
        (args.out / f"{record['name']}.html").write_text(page, encoding="utf-8")

    print(f"{len(records)} runs -> {args.out / 'index.html'}")
    if args.open:
        import webbrowser

        webbrowser.open((args.out / "index.html").resolve().as_uri())
    return 0


_INDEX = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cr-sim runs</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root{
  --ground:#EFF2F5;--panel:#FFFFFF;--panel2:#F6F8FA;
  --ink:#121A24;--soft:#3C4855;--muted:#626F7E;
  --rule:#D8DFE6;--hair:#E7ECF1;
  --accent:#14657F;--accentw:#E3EFF4;
  --good:#26704F;--goodw:#E2F0E9;
  --warn:#8A6516;--warnw:#F7EFDD;
  --crit:#A2352C;--critw:#F7E7E5;
  --shadow:0 1px 2px rgba(18,26,36,.06),0 8px 24px -12px rgba(18,26,36,.18);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0D1218;--panel:#141C25;--panel2:#1A232D;
  --ink:#E6ECF2;--soft:#C2CCD7;--muted:#8695A4;
  --rule:#26313D;--hair:#1E2833;
  --accent:#58B4D0;--accentw:#12303C;
  --good:#5FB68C;--goodw:#163529;
  --warn:#DCAB57;--warnw:#362B15;
  --crit:#E58177;--critw:#3A201D;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
}}
:root[data-theme="dark"]{
  --ground:#0D1218;--panel:#141C25;--panel2:#1A232D;
  --ink:#E6ECF2;--soft:#C2CCD7;--muted:#8695A4;
  --rule:#26313D;--hair:#1E2833;
  --accent:#58B4D0;--accentw:#12303C;
  --good:#5FB68C;--goodw:#163529;
  --warn:#DCAB57;--warnw:#362B15;
  --crit:#E58177;--critw:#3A201D;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);margin:0;padding:0 20px 80px;
  font-family:"Source Sans 3",ui-sans-serif,system-ui,sans-serif;font-size:16.5px;line-height:1.6;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto}
h1,h2,h3,.lbl{font-family:Archivo,ui-sans-serif,system-ui,sans-serif}
.n{font-family:"JetBrains Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
.lbl{font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
header{padding:52px 0 26px;border-bottom:1px solid var(--rule)}
h1{font-size:clamp(32px,5vw,46px);font-weight:700;letter-spacing:-.022em;margin:10px 0 0;line-height:1.05}
.dek{margin:14px 0 0;max-width:66ch;font-size:17.5px;color:var(--soft)}
h2{font-size:12px;font-weight:700;letter-spacing:.11em;text-transform:uppercase;color:var(--accent);margin:52px 0 5px}
.lede{margin:0 0 20px;max-width:72ch;color:var(--soft);font-size:15.5px}
.scroll{overflow-x:auto;border:1px solid var(--rule);border-radius:3px;background:var(--panel);box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;font-size:14.5px}
th,td{padding:11px 14px;text-align:left;border-bottom:1px solid var(--hair);white-space:nowrap}
thead th{font-family:Archivo,sans-serif;font-size:10.5px;font-weight:700;letter-spacing:.09em;
  text-transform:uppercase;color:var(--muted);background:var(--panel2)}
tbody tr:last-child td{border-bottom:0}
td.n,th.n{text-align:right}
a{color:var(--accent)}
.chip{font-family:Archivo,sans-serif;font-size:10.5px;font-weight:700;letter-spacing:.09em;
  text-transform:uppercase;padding:3px 8px;border-radius:2px;white-space:nowrap}
.chip.good{background:var(--goodw);color:var(--good)}
.chip.warn{background:var(--warnw);color:var(--warn)}
.chip.crit{background:var(--critw);color:var(--crit)}
.chip.dim{background:var(--hair);color:var(--muted)}
.cards{display:flex;flex-direction:column;gap:13px}
.card{background:var(--panel);border:1px solid var(--rule);border-left:3px solid var(--sev,var(--rule));
  border-radius:3px;padding:20px 22px;box-shadow:var(--shadow)}
.card.good{--sev:var(--good)}.card.warn{--sev:var(--warn)}
.card.crit{--sev:var(--crit)}.card.dim{--sev:var(--muted)}
.card-top{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:8px}
.card h3{font-size:18px;font-weight:600;margin:0;letter-spacing:-.012em}
.card h3 a{text-decoration:none}
.card p{margin:0 0 9px;max-width:74ch;color:var(--soft);font-size:14.5px}
.settings{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:12px;padding-top:11px;
  border-top:1px solid var(--hair);font-size:13px;color:var(--muted);
  font-family:"JetBrains Mono",ui-monospace,monospace}
footer{margin-top:56px;padding-top:18px;border-top:1px solid var(--rule);font-size:13px;color:var(--muted)}
</style></head><body>
<div class="wrap">
<header>
  <div class="lbl">cr-sim &middot; reinforcement learning</div>
  <h1>Every run, side by side</h1>
  <p class="dek">__COUNT__ training runs on the Clash Royale simulator. A lift number means nothing without knowing what the agent was playing against, so the settings that differed are next to the results that followed from them.</p>
</header>

<h2>Comparison</h2>
<p class="lede">Lift is measured in standard deviations of the random control&rsquo;s own score, so zero means indistinguishable from random however good the win rate looks. Explained variance is how much of the outcome the critic could predict; zero means it never learned to.</p>
<div class="scroll">
<table>
<thead><tr>
<th>Run</th><th>Verdict</th><th class="n">Decisions</th><th class="n">Matches</th>
<th>Reward</th><th>Opponent</th><th class="n">Skip</th>
<th class="n">Mean lift</th><th class="n">Best lift</th><th class="n">Final EV</th>
</tr></thead>
<tbody>
__ROWS__
</tbody>
</table>
</div>

<h2>Run by run</h2>
<p class="lede">Each links to its own page, with the full series and an explanation of every metric on it.</p>
<div class="cards">
__CARDS__
</div>

<footer>Read from each run&rsquo;s own metrics file. Where a run carries a larger paired evaluation, that is shown in place of the trainer&rsquo;s inline 40-battle readings.</footer>
</div>
</body></html>
"""


if __name__ == "__main__":
    import sys

    sys.exit(main())
