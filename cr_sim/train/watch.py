"""``python -m cr_sim.train.watch`` -- see how a run is going, while it goes.

A training run writes one JSON line per update and prints a scrolling wall of
numbers. Neither answers the question you actually have, which is whether the
thing is getting better -- that is a shape over time, and a column of figures
is the worst way to show a shape.

So this reads the metrics file and draws it, refreshing while the run
continues. It is deliberately a separate process: it must not be able to slow
the run down, crash it, or hold a lock on the file it is reading.

**The chart that matters is the lift, not the return.** The trainer's own
return is measured while the policy is exploring, averaged over a sliding
window, on whatever seeds the rollout drew -- and against a paired-seed control
it has run about eighteen points optimistic. So the lift against that control,
in units of the control's own standard deviation, is drawn first and largest,
with a zero line, because "indistinguishable from random" is the null result
this project keeps landing on and it should be impossible to miss.

Everything is plotted against *steps* rather than updates, so runs with
different batch sizes can be compared honestly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "runs" / "overnight-selfplay"


def read_metrics(path: Path) -> list[dict[str, Any]]:
    """Load the metrics file, tolerating a half-written final line.

    The run appends while this reads, so the last line can be incomplete. That
    is normal rather than an error, and dropping it is the whole handling
    required.
    """
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The handful of facts worth putting at the top of the page."""
    if not rows:
        return {"updates": 0}
    last = rows[-1]
    evals = [r for r in rows if "eval_lift_sd" in r]
    best = max(evals, key=lambda r: r["eval_lift_sd"]) if evals else None
    return {
        "updates": last.get("updates", 0),
        "steps": last.get("steps", 0),
        "episodes": last.get("episodes", 0),
        "steps_per_second": last.get("steps_per_second", 0.0),
        "entropy": last.get("entropy", 0.0),
        "evaluations": len(evals),
        "latest_lift": evals[-1]["eval_lift_sd"] if evals else None,
        "latest_win": evals[-1].get("eval_win") if evals else None,
        "control_win": evals[-1].get("control_win") if evals else None,
        "best_lift": best["eval_lift_sd"] if best else None,
        "best_at_steps": best.get("steps") if best else None,
        # Hours left at the current rate. Wrong the moment the rate changes,
        # which is why it is labelled as an estimate rather than a countdown.
        "eta_hours": (
            (last.get("total_steps", 0) - last.get("steps", 0))
            / max(1e-9, last.get("steps_per_second", 0.0)) / 3600
            if last.get("total_steps") else None
        ),
    }


def render(rows: Sequence[dict[str, Any]], title: str) -> str:
    """The page. One file, no dependencies, no build step."""
    series = {
        "steps": [r.get("steps", 0) for r in rows],
        "lift": [(r.get("steps", 0), r["eval_lift_sd"]) for r in rows if "eval_lift_sd" in r],
        "win": [(r.get("steps", 0), r["eval_win"]) for r in rows if "eval_win" in r],
        "control": [(r.get("steps", 0), r["control_win"]) for r in rows if "control_win" in r],
        "entropy": [(r.get("steps", 0), r.get("entropy", 0)) for r in rows],
        "value_loss": [(r.get("steps", 0), r.get("value_loss", 0)) for r in rows],
        "rollout_win": [(r.get("steps", 0), r.get("win_rate", 0)) for r in rows],
    }
    payload = json.dumps({"series": series, "summary": summarise(rows), "title": title})
    return _PAGE.replace("__DATA__", payload)


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<title>cr-sim training</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --line:#2b3444; --ink:#e6edf3;
          --dim:#8b949e; --good:#3fb950; --bad:#f85149; --accent:#58a6ff;
          --warn:#d29922; }
  * { box-sizing:border-box }
  body { margin:0; background:var(--bg); color:var(--ink); padding:20px;
         font:14px/1.5 "Segoe UI",ui-sans-serif,system-ui,sans-serif; }
  h1 { margin:0 0 2px; font-size:19px }
  .sub { color:var(--dim); font-size:12px; margin-bottom:16px }
  .tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
           gap:10px; margin-bottom:18px; max-width:1100px }
  .tile { background:var(--panel); border:1px solid var(--line); border-radius:8px;
          padding:10px 12px }
  .tile .k { font-size:10px; text-transform:uppercase; letter-spacing:.07em; color:var(--dim) }
  .tile .v { font-size:21px; font-weight:700; font-variant-numeric:tabular-nums }
  .tile .n { font-size:11px; color:var(--dim) }
  .good{color:var(--good)} .bad{color:var(--bad)} .warn{color:var(--warn)}
  .chart { background:var(--panel); border:1px solid var(--line); border-radius:8px;
           padding:12px 14px 6px; margin-bottom:12px; max-width:1100px }
  .chart h2 { margin:0 0 2px; font-size:13px }
  .chart p { margin:0 0 8px; color:var(--dim); font-size:11px }
  svg { width:100%; height:190px; display:block }
  .axis { stroke:var(--line); stroke-width:1 }
  .zero { stroke:var(--dim); stroke-width:1; stroke-dasharray:4 3 }
  .lbl { fill:var(--dim); font-size:10px }
  .empty { color:var(--dim); font-size:12px; padding:24px 0; text-align:center }
</style></head><body>
<h1 id="title">cr-sim training</h1>
<div class="sub">refreshes every 30s &middot; <span id="stamp"></span></div>
<div class="tiles" id="tiles"></div>
<div id="charts"></div>
<script>
const DATA = __DATA__;
const $ = (id) => document.getElementById(id);

function tile(k, v, note, cls) {
  return `<div class="tile"><div class="k">${k}</div>`
       + `<div class="v ${cls||''}">${v}</div>`
       + `<div class="n">${note||''}</div></div>`;
}

function pct(x) { return x === null || x === undefined ? '--' : (x*100).toFixed(0)+'%'; }
function num(x, d) { return x === null || x === undefined ? '--' : Number(x).toFixed(d===undefined?2:d); }

function renderTiles(s) {
  // The lift is the honest measure, so it leads and it is coloured. Anything
  // inside a quarter of a control standard deviation is noise, and saying so
  // on the tile stops a lucky evaluation reading as progress.
  let cls = 'warn', verdict = 'inside noise';
  if (s.latest_lift !== null && s.latest_lift >= 0.25) { cls = 'good'; verdict = 'better than random'; }
  else if (s.latest_lift !== null && s.latest_lift <= -0.25) { cls = 'bad'; verdict = 'worse than random'; }

  $('tiles').innerHTML = [
    tile('lift vs control', s.latest_lift === null ? '--' : num(s.latest_lift)+' sd', verdict, cls),
    tile('win rate', pct(s.latest_win), 'control ' + pct(s.control_win)),
    tile('best lift', s.best_lift === null ? '--' : num(s.best_lift)+' sd',
         s.best_at_steps ? 'at ' + (s.best_at_steps/1000).toFixed(0) + 'k steps' : ''),
    tile('steps', (s.steps/1000).toFixed(0) + 'k', s.updates + ' updates'),
    tile('episodes', s.episodes || '--', 'matches played'),
    tile('rate', num(s.steps_per_second, 0) + '/s',
         s.eta_hours ? num(s.eta_hours,1) + 'h remaining' : ''),
    tile('entropy', num(s.entropy), 'ln(90) = 4.5 is uniform'),
  ].join('');
}

function chart(title, note, series, opts) {
  opts = opts || {};
  const w = 1040, h = 190, padL = 46, padR = 14, padT = 12, padB = 24;
  if (!series.length || !series.some(s => s.points.length > 1)) {
    return `<div class="chart"><h2>${title}</h2><p>${note}</p>`
         + `<div class="empty">no evaluations yet</div></div>`;
  }
  const all = series.flatMap(s => s.points);
  const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (opts.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  if (hi - lo < 1e-9) { hi = lo + 1; }
  const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
  const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1;
  const X = (v) => padL + (v - x0) / Math.max(1, x1 - x0) * (w - padL - padR);
  const Y = (v) => padT + (hi - v) / (hi - lo) * (h - padT - padB);

  let body = '';
  if (opts.zero) body += `<line class="zero" x1="${padL}" x2="${w-padR}" y1="${Y(0)}" y2="${Y(0)}"/>`;
  for (const s of series) {
    if (s.points.length < 2) continue;
    const d = s.points.map((p,i) => (i?'L':'M') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1)).join(' ');
    body += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2"/>`;
    const last = s.points[s.points.length-1];
    body += `<circle cx="${X(last[0])}" cy="${Y(last[1])}" r="3" fill="${s.color}"/>`;
    body += `<text class="lbl" x="${w-padR}" y="${Y(last[1])-6}" text-anchor="end" fill="${s.color}">${s.name}</text>`;
  }
  body += `<line class="axis" x1="${padL}" x2="${w-padR}" y1="${h-padB}" y2="${h-padB}"/>`;
  body += `<text class="lbl" x="4" y="${Y(hi)+9}">${hi.toFixed(2)}</text>`;
  body += `<text class="lbl" x="4" y="${Y(lo)}">${lo.toFixed(2)}</text>`;
  body += `<text class="lbl" x="${padL}" y="${h-6}">${(x0/1000).toFixed(0)}k</text>`;
  body += `<text class="lbl" x="${w-padR}" y="${h-6}" text-anchor="end">${(x1/1000).toFixed(0)}k steps</text>`;
  return `<div class="chart"><h2>${title}</h2><p>${note}</p>`
       + `<svg viewBox="0 0 ${w} ${h}">${body}</svg></div>`;
}

(function start() {
  const s = DATA.summary, S = DATA.series;
  $('title').textContent = DATA.title;
  $('stamp').textContent = new Date().toLocaleTimeString();
  renderTiles(s);
  $('charts').innerHTML = [
    chart('Lift against a random control',
          'The honest measure: the policy and a random agent play the same seeds. '
          + 'Zero means indistinguishable. The rollout return below is measured while '
          + 'exploring and has run about eighteen points optimistic.',
          [{name:'lift (sd)', color:'#58a6ff', points:S.lift}], {zero:true}),
    chart('Win rate, evaluated',
          'Both arms on identical seeds, so the gap is the whole story.',
          [{name:'policy', color:'#3fb950', points:S.win},
           {name:'control', color:'#8b949e', points:S.control}]),
    chart('Policy entropy',
          'ln(90) is about 4.5 for this legal action set. Sitting there means the '
          + 'policy is still near-uniform and has not committed to anything.',
          [{name:'entropy', color:'#d29922', points:S.entropy}]),
    chart('Rollout win rate and value loss',
          'The trainer\'s own numbers. Useful for spotting divergence, not for '
          + 'judging strength.',
          [{name:'rollout win', color:'#a371f7', points:S.rollout_win}]),
  ].join('');
})();
</script></body></html>"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cr-sim-watch")
    parser.add_argument("run", nargs="?", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the page (default: <run>/progress.html)")
    parser.add_argument("--open", action="store_true", help="open it in a browser")
    parser.add_argument("--once", action="store_true",
                        help="write once and exit rather than refreshing")
    parser.add_argument("--every", type=float, default=30.0)
    args = parser.parse_args(argv)

    metrics = args.run / "metrics.jsonl"
    out = args.out or (args.run / "progress.html")

    def write() -> int:
        rows = read_metrics(metrics)
        out.write_text(render(rows, args.run.name), encoding="utf-8")
        return len(rows)

    count = write()
    print(f"{count} updates -> {out}")
    if args.open:
        import webbrowser

        webbrowser.open(out.resolve().as_uri())
    if args.once:
        return 0

    # The page reloads itself, so this only has to keep it current. Polling a
    # file rather than watching it: the run appends every minute or so, and a
    # watcher would be more machinery than the interval deserves.
    import time

    try:
        while True:
            time.sleep(args.every)
            count = write()
            print(f"  {count} updates", flush=True)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
