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
import hashlib
import json
import time
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



def _next_evaluation(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How long until this run evaluates again.

    Both numbers are inferred from the run's own history rather than read
    from its config, so this works for a run whose config was lost and for
    one whose cadence was changed by a resume.

    The cadence comes from the spacing of past evaluations; the pace from how
    long recent updates actually took, which is what a countdown should
    follow rather than an average over a run that has changed speed.
    """
    updates = [r["updates"] for r in rows if "eval_lift_sd" in r]
    if len(updates) < 2 or len(rows) < 2:
        return {"eval_every": None, "next_eval_seconds": None}
    gaps = [b - a for a, b in zip(updates, updates[1:]) if b > a]
    if not gaps:
        return {"eval_every": None, "next_eval_seconds": None}
    cadence = min(gaps)

    # Paced on the recent stretch. A run that was throttled partway through
    # would otherwise be timed by an average it no longer runs at.
    window = rows[-min(len(rows), 12):]
    span = (window[-1].get("elapsed_seconds"), window[0].get("elapsed_seconds"))
    if None in span or window[-1]["updates"] <= window[0]["updates"]:
        return {"eval_every": cadence, "next_eval_seconds": None}
    per_update = (span[0] - span[1]) / (window[-1]["updates"] - window[0]["updates"])
    if per_update <= 0:
        return {"eval_every": cadence, "next_eval_seconds": None}

    done = rows[-1]["updates"]
    remaining = cadence - (done - updates[-1]) % cadence
    return {"eval_every": cadence,
            "next_eval_seconds": max(0.0, remaining * per_update)}


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
        "total_steps": last.get("total_steps"),
        "elapsed_seconds": last.get("elapsed_seconds"),
        **_next_evaluation(rows),
        "ancestor_win": (
            [r["ancestor_win"] for r in rows if "ancestor_win" in r] or [None]
        )[-1],
        "ancestor_loss": (
            [r["ancestor_loss"] for r in rows if "ancestor_loss" in r] or [None]
        )[-1],
        "ancestor_age": (
            [r["ancestor_age"] for r in rows if "ancestor_age" in r] or [None]
        )[-1],
        # Hours left at the current rate. Wrong the moment the rate changes,
        # which is why it is labelled as an estimate rather than a countdown.
        "eta_hours": (
            (last.get("total_steps", 0) - last.get("steps", 0))
            / max(1e-9, last.get("steps_per_second", 0.0)) / 3600
            if last.get("total_steps") else None
        ),
    }




def _started_at(run: Path) -> float:
    """When a run began, as a timestamp.

    Taken from config.json, which is written once before the first update, so
    it dates the start rather than the most recent write. Falls back to the
    metrics file for runs that predate the config being written at all.
    """
    for name in ("config.json", "metrics.jsonl"):
        path = run / name
        if path.is_file():
            stat = path.stat()
            # st_ctime is creation time on Windows and metadata-change time
            # elsewhere; either dates the start closely enough to order by,
            # and st_mtime would sort by last write, which is a different
            # question.
            return min(stat.st_ctime, stat.st_mtime)
    return 0.0


def _note_of(run: Path) -> str:
    """One line of prose from config.json saying what this entry is.

    ``config.json`` has always been written and never read except for its
    timestamp. Everything that is not a training run -- the search expert, a
    cloned policy, a benchmark, a head-to-head between checkpoints -- looks
    on the index like a run that produced two flat points, and what it
    actually measured lived only in whatever conversation produced it.
    """
    path = run / "config.json"
    if not path.is_file():
        return ""
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("note", ""))
    except (json.JSONDecodeError, OSError):
        return ""


def _series_of(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def pair(key):
        return [(r.get("steps", 0), r[key]) for r in rows if key in r]

    return {
        "steps": [r.get("steps", 0) for r in rows],
        "lift": [(r.get("steps", 0), r["eval_lift_sd"]) for r in rows if "eval_lift_sd" in r],
        "win": pair("eval_win"),
        "control": pair("control_win"),
        "entropy": pair("entropy"),
        "value_loss": pair("value_loss"),
        "rollout_win": pair("win_rate"),
        # Added once the critic turned out to be the bottleneck. Value loss
        # cannot be read without knowing the spread of what it is fitting;
        # explained variance is that comparison done properly, and is what
        # decides whether PPO has a usable signal at all.
        "explained_variance": pair("explained_variance"),
        "ret_std": pair("ret_std"),
        "noop": pair("noop_fraction"),
        # The ladder: how the policy fares against the oldest version of
        # itself still in the pool. Only self-play runs record it.
        "ancestor_win": pair("ancestor_win"),
        "ancestor_loss": pair("ancestor_loss"),
    }


def render_multi(
    runs: "Sequence[tuple[str, Sequence[dict[str, Any]], bool]]",
    back: str | None = None,
    notes: "dict[str, str] | None" = None,
) -> "tuple[str, dict[str, Any]]":
    """One page holding several runs, as ``(name, rows, live)`` triples.

    Runs are compared far more often than they are watched alone -- the
    question is almost always "is this one doing better than that one", and
    answering it by opening two files and scrolling both is how a difference
    gets missed. Tabs switch between them and a split view puts two
    side by side against shared axes.

    ``live`` marks a run whose metrics file was written recently, so a
    finished run is not presented as though it were still moving.
    """
    body = {
        "runs": {
            name: {
                "series": _series_of(rows),
                "summary": summarise(rows),
                "live": bool(live),
                # What this entry *is*. A training run is legible from its
                # curves; a benchmark, a head-to-head or a piece of work an
                # agent is doing is not, and an index entry whose meaning
                # lives only in a chat log is an index entry nobody can read.
                "note": (notes or {}).get(name, ""),
            }
            for name, rows, live in runs
        },
        # Newest first. The question is nearly always what the run started
        # most recently is doing, and burying it under a fortnight of finished
        # runs makes that a scroll.
        "order": [name for name, _, _ in reversed(list(runs))],
        "back": back,
    }
    # A fingerprint of the data, so the page can re-render only when something
    # actually moved. Reloading on a timer throws away scroll position, an
    # open glossary and any chart mid-scrub, several times an hour, to redraw
    # numbers that had not changed.
    # When this was built, so a countdown keeps ticking between polls rather
    # than freezing at whatever it said when the page last loaded.
    body["generated_at"] = time.time()
    body["version"] = hashlib.sha1(
        json.dumps({k: v for k, v in body.items() if k != "generated_at"},
                   sort_keys=True, default=str).encode()).hexdigest()[:16]
    return _PAGE.replace("__DATA__", json.dumps(body)), body


def render(rows: Sequence[dict[str, Any]], title: str,
           back: str | None = None) -> str:
    """The page for a single run. One file, no dependencies, no build step."""
    return render_multi([(title, rows, True)], back=back)[0]


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>cr-sim training</title>
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="cr-sim">
<meta name="theme-color" content="#0D1218">
<link rel="apple-touch-icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 180 180'%3E%3Crect width='180' height='180' rx='40' fill='%230D1218'/%3E%3Cpath d='M40 128V52h18l22 34 22-34h18v76h-18V82l-22 33-22-33v46z' fill='%2358B4D0'/%3E%3C/svg%3E">
<link rel="manifest" href="data:application/json,%7B%22name%22%3A%22cr-sim%20training%22%2C%22short_name%22%3A%22cr-sim%22%2C%22display%22%3A%22standalone%22%2C%22background_color%22%3A%22%230D1218%22%2C%22theme_color%22%3A%220D1218%22%2C%22start_url%22%3A%22.%22%7D">
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
  --grey:#8695A4;
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
  --grey:#7C8B9A;
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
  --grey:#7C8B9A;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);margin:0;
  padding:env(safe-area-inset-top) 20px 60px;
  font-family:"Source Sans 3",ui-sans-serif,system-ui,-apple-system,sans-serif;
  font-size:16.5px;line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto}
h1,h2,h3,.lbl,.v,.kpi-n{font-family:Archivo,ui-sans-serif,system-ui,sans-serif}
.mono,.v,.kpi-n,td.n,.lbl-s{font-family:"JetBrains Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
.lbl{font-size:10.5px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}

header{padding:26px 0 14px;display:flex;align-items:flex-end;gap:14px;flex-wrap:wrap}
h1{font-size:clamp(22px,4vw,30px);font-weight:700;letter-spacing:-.02em;margin:0;flex:1}
.stamp{font-size:12px;color:var(--muted);display:flex;align-items:center;gap:7px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--good);box-shadow:0 0 0 3px var(--goodw)}
.dot.stale{background:var(--muted);box-shadow:0 0 0 3px var(--hair)}
.bell{appearance:none;background:var(--panel);border:1px solid var(--rule);border-radius:3px;
  color:var(--soft);font-family:Archivo,sans-serif;font-size:11px;font-weight:600;
  letter-spacing:.05em;padding:5px 10px;cursor:pointer}
.bell[data-on="1"]{background:var(--accentw);color:var(--accent);border-color:var(--accent)}

.tabbar{display:flex;align-items:flex-end;gap:10px;border-bottom:1px solid var(--rule)}
.tabs{display:flex;gap:2px;flex:1;overflow-x:auto;scrollbar-width:none;padding-bottom:1px}
.tabs::-webkit-scrollbar{display:none}
.tab{appearance:none;border:1px solid transparent;border-bottom:0;background:none;
  font-family:Archivo,sans-serif;font-size:13px;font-weight:600;color:var(--muted);
  padding:8px 12px;border-radius:3px 3px 0 0;cursor:pointer;display:flex;align-items:center;
  gap:7px;white-space:nowrap}
.tab[aria-selected="true"]{background:var(--panel);border-color:var(--rule);color:var(--ink);margin-bottom:-1px;padding-bottom:9px}
.tab .pip{width:6px;height:6px;border-radius:50%;background:var(--muted);flex:none}
.tab .pip.on{background:var(--good);box-shadow:0 0 0 2.5px var(--goodw)}
.tab .val{font-family:"JetBrains Mono",monospace;font-size:11.5px;font-variant-numeric:tabular-nums}
.split-toggle{appearance:none;background:var(--panel);border:1px solid var(--rule);border-radius:3px;
  font-family:Archivo,sans-serif;font-size:11px;font-weight:600;color:var(--soft);
  padding:6px 11px;cursor:pointer;margin-bottom:7px}
.split-toggle[aria-pressed="true"]{background:var(--accentw);color:var(--accent);border-color:var(--accent)}
@media (max-width:700px){.split-toggle{display:none}}

.panes{display:grid;grid-template-columns:1fr;gap:18px;margin-top:16px}
.panes.split{grid-template-columns:1fr 1fr}
@media (max-width:900px){.panes.split{grid-template-columns:1fr}}
.pane{min-width:0}
.note{font-size:13.5px;line-height:1.5;color:var(--muted);margin:0 0 12px;padding:10px 12px;
  border-left:2px solid var(--line);background:rgba(255,255,255,.02);border-radius:0 6px 6px 0;white-space:pre-wrap}
.pane-head{display:flex;align-items:center;gap:9px;margin-bottom:10px}
.pane-head select{font-family:Archivo,sans-serif;font-size:13px;font-weight:600;color:var(--ink);
  background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:5px 8px}

.readout{background:var(--panel);border:1px solid var(--rule);border-radius:3px;box-shadow:var(--shadow);overflow:hidden}
.bar{height:3px;background:var(--hair)}
.bar i{display:block;height:100%;background:var(--accent)}
.readout-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(96px,1fr))}
.cell{padding:13px 15px;border-right:1px solid var(--hair);border-top:1px solid var(--hair)}
.cell:first-child{border-top:0}
@media (min-width:760px){.cell{border-top:0}}
.cell:last-child{border-right:0}
.kpi-n{font-size:20px;font-weight:600;letter-spacing:-.02em;display:block;margin-bottom:2px}
.kpi-n small{font-size:12px;font-weight:400;color:var(--muted)}

.hero{background:var(--panel);border:1px solid var(--rule);border-left:3px solid var(--sev,var(--rule));
  border-radius:3px;padding:18px 20px;box-shadow:var(--shadow);margin-top:12px}
.hero.good{--sev:var(--good)}.hero.warn{--sev:var(--warn)}.hero.crit{--sev:var(--crit)}.hero.dim{--sev:var(--grey)}
.hero-top{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
.hero .v{font-size:36px;font-weight:700;letter-spacing:-.03em;line-height:1}
.chip{font-family:Archivo,sans-serif;font-size:10px;font-weight:700;letter-spacing:.08em;
  text-transform:uppercase;padding:3px 7px;border-radius:2px;white-space:nowrap}
.chip.good{background:var(--goodw);color:var(--good)}
.chip.warn{background:var(--warnw);color:var(--warn)}
.chip.crit{background:var(--critw);color:var(--crit)}
.chip.dim{background:var(--hair);color:var(--muted)}
.hero .best{margin-top:8px;font-size:12.5px;color:var(--muted)}

.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:10px}
@media (max-width:700px){.tiles{grid-template-columns:1fr 1fr}}
.tile{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:12px 14px;box-shadow:var(--shadow)}
.tile .v{font-size:21px;font-weight:700;letter-spacing:-.02em;margin:5px 0 0;display:block}
.good{color:var(--good)}.warn{color:var(--warn)}.crit{color:var(--crit)}.dim{color:var(--muted)}

.chart{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:14px 16px;
  box-shadow:var(--shadow);margin-top:10px;position:relative}
.chart h3{font-size:14px;font-weight:600;margin:0 0 8px;letter-spacing:-.01em;
  display:flex;justify-content:space-between;align-items:baseline;gap:10px}
.chart h3 .read{font-family:"JetBrains Mono",monospace;font-size:12px;font-weight:400;
  color:var(--muted);font-variant-numeric:tabular-nums;text-align:right}
.chart svg{display:block;width:100%;height:auto;touch-action:pan-y}
.empty{padding:18px;text-align:center;color:var(--muted);font-size:13.5px;background:var(--panel2);border-radius:3px}
.axis{stroke:var(--rule);stroke-width:1}
.zero{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3;opacity:.5}
.grid-l{stroke:var(--hair);stroke-width:1}
.lbl-s{font:600 10px "JetBrains Mono",monospace;fill:var(--muted)}
.scrub{stroke:var(--accent);stroke-width:1;opacity:.7}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:10px}
@media (max-width:700px){.grid2{grid-template-columns:1fr}}
h2{font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);margin:26px 0 2px}

details.gloss{margin-top:26px;background:var(--panel);border:1px solid var(--rule);border-radius:3px;box-shadow:var(--shadow)}
details.gloss summary{padding:12px 18px;cursor:pointer;font-family:Archivo,sans-serif;
  font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
details.gloss summary::-webkit-details-marker{color:var(--muted)}
details.gloss dl{margin:0;padding:0 18px 16px}
details.gloss dt{font-family:Archivo,sans-serif;font-weight:700;margin-top:14px;font-size:14px}
details.gloss dd{margin:3px 0 0;color:var(--soft);font-size:13.5px;max-width:78ch}
code{font-family:"JetBrains Mono",monospace;font-size:.87em;background:var(--accentw);color:var(--accent);padding:1px 4px;border-radius:2px}
footer{margin-top:28px;padding-top:12px;border-top:1px solid var(--rule);font-size:12px;color:var(--muted)}
</style></head><body>
<div class="wrap">

<header>
  <h1 id="title">cr-sim</h1>
  <button class="bell" id="bell" data-on="0" title="Notify when a run posts a new evaluation">Alerts off</button>
  <div class="stamp"><span class="dot" id="pulse"></span><span id="stamp"></span></div>
</header>

<div class="tabbar">
  <div class="tabs" id="tabs" role="tablist"></div>
  <button class="split-toggle" id="split" aria-pressed="false">Split</button>
</div>

<div class="panes" id="panes"></div>

<details class="gloss">
  <summary>What the numbers mean</summary>
  <dl>
    <dt>Lift vs control</dt>
    <dd>Return against a random agent over the same fixed battles, in standard deviations of the control's own spread. 0 is no better than random. Under about 0.25 is inside the noise &mdash; a +0.375 reading on this project measured &minus;0.033 over 300 battles.</dd>
    <dt>Beats its past self</dt>
    <dd>Self-play only. Wins and losses against the oldest version still in the opponent pool. Fixed reference, so more sensitive than lift &mdash; but movement, not skill.</dd>
    <dt>Explained variance</dt>
    <dd>How much of the outcome the critic predicts. 0 is no better than guessing the average, which makes PPO's advantages noise. Sat at 0.00 for four runs.</dd>
    <dt>Entropy</dt>
    <dd>How undecided the policy is. Falling means committing &mdash; good only alongside rising lift.</dd>
    <dt>Value loss and return spread</dt>
    <dd>Squared error, and the spread of what it fits. Meaningless apart; explained variance is the same comparison done properly.</dd>
    <dt>Rollout win rate</dt>
    <dd>Measured while exploring, about eighteen points optimistic here. Do not steer by it.</dd>
    <dt>Pass rate</dt>
    <dd>How often it declined to play. Never punished, so a run can quietly collapse into it.</dd>
  </dl>
</details>

<footer id="foot"></footer>
</div>

<script>
var DATA = __DATA__;

function num(x,d){return (x===null||x===undefined||(typeof x==='number'&&isNaN(x)))?'--':Number(x).toFixed(d===undefined?2:d);}
function pct(x){return (x===null||x===undefined)?'--':(x*100).toFixed(0)+'%';}
function esc(s){return String(s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
function last(p){return (p&&p.length)?p[p.length-1][1]:null;}
function commas(n){return Number(n||0).toLocaleString();}
function dur(s){if(!s&&s!==0)return '--';var h=Math.floor(s/3600),m=Math.round((s%3600)/60);return h?(h+'h '+m+'m'):(m+'m');}
/* Counts the "next eval" cells down once a second. Separate from rendering
   because the page deliberately re-renders only when the data changes, and a
   countdown that moves only then is wrong for the twenty minutes in between. */
function tickCountdowns(){
  var now=Date.now()/1000;
  Array.prototype.forEach.call(document.querySelectorAll('[data-role="next"]'),function(el){
    var at=parseFloat(el.dataset.at||'');
    if(!at||el.dataset.live!=='1'){el.textContent='--';return;}
    var left=at-now;
    if(left<=0){el.textContent='due';el.className='good';return;}
    el.className='';
    var m=Math.floor(left/60),sec=Math.floor(left%60);
    el.textContent=m>=60?((m/60).toFixed(1)+'h'):(m+'m '+(sec<10?'0':'')+sec+'s');
  });
}

function remember(k,v){try{localStorage.setItem(k,v);}catch(e){}}
function recall(k,f){try{var v=localStorage.getItem(k);return v===null?f:v;}catch(e){return f;}}

/* ---------------------------------------------------------------- charts */

var CHARTS = [];   // every drawn chart, so scrubbing can find them

function chart(id,title,series,opts){
  opts=opts||{};
  var all=[]; series.forEach(function(s){(s.points||[]).forEach(function(p){all.push(p);});});
  if(all.length<1){
    return '<div class="chart"><h3>'+title+'</h3><div class="empty">'
      +(opts.emptyText||'no evaluations yet')+'</div></div>';
  }
  if(all.length<2||series.every(function(s){return (s.points||[]).length<2;})){
    var only=series.filter(function(s){return (s.points||[]).length;}).map(function(s){
      var pt=s.points[s.points.length-1];
      return '<span style="color:'+s.color+'"><b>'+(opts.asPct?pct(pt[1]):num(pt[1],3))+'</b> '+s.name+'</span>';
    }).join('<span style="color:var(--muted)"> / </span>');
    return '<div class="chart"><h3>'+title+'</h3><div class="empty" style="font-size:14px">'+only
      +'<div style="margin-top:5px;font-size:12px;color:var(--muted)">one reading &mdash; a trend needs two</div></div></div>';
  }
  var w=640,h=170,padL=42,padR=Math.max(50,18+Math.max.apply(null,series.map(function(s){
    return (s.points||[]).length?s.name.length:0;}))*6.2),padT=10,padB=20;
  var xs=all.map(function(p){return p[0];}),ys=all.map(function(p){return p[1];});
  var x0=Math.min.apply(null,xs),x1=Math.max.apply(null,xs);
  var lo=Math.min.apply(null,ys),hi=Math.max.apply(null,ys);
  if(opts.zero){lo=Math.min(lo,0);hi=Math.max(hi,0);}
  if(hi-lo<1e-9){hi+=0.5;lo-=0.5;}
  var pd=(hi-lo)*0.12;hi+=pd;lo-=pd;
  var X=function(v){return padL+(x1===x0?0.5:(v-x0)/(x1-x0))*(w-padL-padR);};
  var Y=function(v){return padT+(1-(v-lo)/(hi-lo))*(h-padT-padB);};
  var body='';
  [0.33,0.66].forEach(function(f){var y=padT+f*(h-padT-padB);
    body+='<line class="grid-l" x1="'+padL+'" x2="'+(w-padR)+'" y1="'+y.toFixed(1)+'" y2="'+y.toFixed(1)+'"/>';});
  if(opts.zero) body+='<line class="zero" x1="'+padL+'" x2="'+(w-padR)+'" y1="'+Y(0).toFixed(1)+'" y2="'+Y(0).toFixed(1)+'"/>';
  series.forEach(function(s){
    var pts=s.points||[]; if(pts.length<2) return;
    if(opts.fill){
      var base=Y(Math.max(lo,0));
      var a=pts.map(function(p,i){return (i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1);}).join(' ')
        +' L'+X(pts[pts.length-1][0]).toFixed(1)+' '+base.toFixed(1)
        +' L'+X(pts[0][0]).toFixed(1)+' '+base.toFixed(1)+' Z';
      body+='<path d="'+a+'" fill="'+s.color+'" opacity=".08"/>';
    }
    body+='<path d="'+pts.map(function(p,i){return (i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1);}).join(' ')
      +'" fill="none" stroke="'+s.color+'" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>';
    var lp=pts[pts.length-1];
    body+='<circle cx="'+X(lp[0]).toFixed(1)+'" cy="'+Y(lp[1]).toFixed(1)+'" r="3.2" fill="'+s.color+'"/>';
    body+='<text class="lbl-s" x="'+(w-padR+6)+'" y="'+(Y(lp[1])+3.4).toFixed(1)+'" fill="'+s.color+'">'+s.name+'</text>';
  });
  body+='<line class="axis" x1="'+padL+'" x2="'+(w-padR)+'" y1="'+(h-padB)+'" y2="'+(h-padB)+'"/>';
  body+='<text class="lbl-s" x="3" y="'+(Y(hi)+8).toFixed(1)+'">'+hi.toFixed(2)+'</text>';
  body+='<text class="lbl-s" x="3" y="'+Y(lo).toFixed(1)+'">'+lo.toFixed(2)+'</text>';
  if(x1>x0){
    body+='<text class="lbl-s" x="'+padL+'" y="'+(h-5)+'">'+(x0/1000).toFixed(0)+'k</text>';
    body+='<text class="lbl-s" x="'+(w-padR)+'" y="'+(h-5)+'" text-anchor="end">'+(x1/1000).toFixed(0)+'k</text>';
  }else{
    body+='<text class="lbl-s" x="'+((padL+w-padR)/2)+'" y="'+(h-5)+'" text-anchor="middle">'+(x0/1000).toFixed(0)+'k</text>';
  }
  body+='<line class="scrub" id="'+id+'-line" x1="0" x2="0" y1="'+padT+'" y2="'+(h-padB)+'" style="display:none"/>';
  CHARTS.push({id:id,series:series,x0:x0,x1:x1,padL:padL,padR:padR,w:w,asPct:!!opts.asPct});
  return '<div class="chart"><h3>'+title+'<span class="read" id="'+id+'-read"></span></h3>'
    +'<svg id="'+id+'" viewBox="0 0 '+w+' '+h+'" role="img" aria-label="'+esc(title)+'">'+body+'</svg></div>';
}

/* Reading exact values by dragging along a chart. Endpoint labels tell you
   where a line finished; the interesting question is usually what it did in
   the middle, and squinting at a 170px-tall svg does not answer it. */
function wireScrub(){
  CHARTS.forEach(function(c){
    var svg=document.getElementById(c.id), read=document.getElementById(c.id+'-read'),
        line=document.getElementById(c.id+'-line');
    if(!svg||!read) return;
    function at(clientX){
      var box=svg.getBoundingClientRect();
      var vx=(clientX-box.left)/box.width*c.w;
      var t=Math.max(0,Math.min(1,(vx-c.padL)/(c.w-c.padL-c.padR)));
      var step=c.x0+t*(c.x1-c.x0);
      var parts=[];
      c.series.forEach(function(s){
        var pts=s.points||[]; if(!pts.length) return;
        var best=pts[0],bd=Infinity;
        pts.forEach(function(p){var d=Math.abs(p[0]-step); if(d<bd){bd=d;best=p;}});
        parts.push('<span style="color:'+s.color+'">'+(c.asPct?pct(best[1]):num(best[1],3))+'</span>');
        step=best[0];
      });
      read.innerHTML=commas(Math.round(step))+' &middot; '+parts.join(' / ');
      line.setAttribute('x1',c.padL+(step-c.x0)/((c.x1-c.x0)||1)*(c.w-c.padL-c.padR));
      line.setAttribute('x2',line.getAttribute('x1'));
      line.style.display='';
    }
    function clear(){read.innerHTML='';line.style.display='none';}
    svg.addEventListener('mousemove',function(e){at(e.clientX);});
    svg.addEventListener('mouseleave',clear);
    svg.addEventListener('touchstart',function(e){at(e.touches[0].clientX);},{passive:true});
    svg.addEventListener('touchmove',function(e){at(e.touches[0].clientX);},{passive:true});
    svg.addEventListener('touchend',clear);
  });
}

/* ------------------------------------------------------------------ panes */

function verdictFor(l){
  if(l===null||l===undefined) return ['dim','not measured','First evaluation lands at update 20.'];
  if(l>=0.5) return ['good','clearly better','Outside the control&rsquo;s noise.'];
  if(l>=0.25) return ['good','probably better','Outside the noise, but not by much.'];
  if(l>-0.25) return ['warn','inside the noise','A single positive reading here means nothing.'];
  return ['crit','worse than random','Committed to something actively bad.'];
}

function cell(v,k,sm){return '<div class="cell"><span class="kpi-n">'+v+(sm?' <small>'+sm+'</small>':'')
  +'</span><span class="lbl">'+k+'</span></div>';}

function paneMarkup(id){
  return '<div class="pane" data-pane="'+id+'">'
    +'<div class="pane-head"><span class="lbl">showing</span><select data-role="pick"></select></div>'
    +'<div data-role="note" class="note"></div>'
    +'<div class="readout"><div class="bar"><i data-role="bar" style="width:0%"></i></div>'
    +'<div class="readout-grid" data-role="general"></div></div>'
    +'<div data-role="hero"></div><div class="tiles" data-role="tiles"></div>'
    +'<h2>Is it learning</h2><div data-role="c1"></div>'
    +'<h2>Is the critic working</h2><div data-role="c2"></div></div>';
}

function fillPane(pane,name,slot){
  var run=DATA.runs[name]; if(!run) return;
  var S=run.series,s=run.summary,q=function(r){return pane.querySelector('[data-role="'+r+'"]');};
  var noteEl=q('note');
  if(noteEl){noteEl.innerHTML=run.note?esc(run.note):'';noteEl.style.display=run.note?'block':'none';}
  var done=s.total_steps?Math.min(1,(s.steps||0)/s.total_steps):0;
  q('bar').style.width=(done*100).toFixed(1)+'%';
  var remain=(s.total_steps&&s.steps_per_second)?(s.total_steps-s.steps)/s.steps_per_second:null;
  q('general').innerHTML=[
    cell(commas(s.steps),'steps',s.total_steps?('/ '+commas(s.total_steps)):''),
    cell(num(s.steps_per_second,1),'per sec'),
    cell(commas(s.episodes),'matches'),
    cell(dur(s.elapsed_seconds),'elapsed'),
    cell(remain===null?'--':dur(remain),'left'),
    cell(commas(s.evaluations),'evals'),
    cell('<span data-role="next">--</span>','next eval')
  ].join('');
  // Kept as a number the tick loop counts down, rather than re-rendered:
  // the page only re-renders when the data changes, which is every twenty
  // updates, and a countdown that only moves then is a clock that is wrong
  // for twenty minutes at a time.
  var nextEl=q('next');
  if(nextEl){
    nextEl.dataset.at=(s.next_eval_seconds===null||s.next_eval_seconds===undefined)
      ? '' : String((DATA.generated_at||0)+s.next_eval_seconds);
    nextEl.dataset.live=run.live?'1':'0';
  }

  var lift=s.latest_lift,r=verdictFor(lift);
  q('hero').innerHTML='<div class="hero '+r[0]+'"><div class="lbl">lift vs control</div>'
    +'<div class="hero-top"><span class="v '+r[0]+'">'
    +((lift===null||lift===undefined)?'&mdash;':((lift>0?'+':'')+num(lift)+' sd'))
    +'</span><span class="chip '+r[0]+'">'+r[1]+'</span></div>'
    +((s.best_lift===null||s.best_lift===undefined)?'':
      '<div class="best">best '+(s.best_lift>0?'+':'')+num(s.best_lift)+' at '+commas(s.best_at_steps)+'</div>')
    +'</div>';

  var ev=last(S.explained_variance),ent=last(S.entropy),vl=last(S.value_loss),
      rs=last(S.ret_std),np=last(S.noop),rw=last(S.rollout_win);
  var evCls=ev===null?'dim':(ev>=0.3?'good':ev>=0.1?'warn':'crit');
  function tile(k,v,cls){return '<div class="tile"><div class="lbl">'+k+'</div><span class="v '+(cls||'')+'">'+v+'</span></div>';}
  q('tiles').innerHTML=[
    tile('expl var',ev===null?'--':((ev>0?'+':'')+num(ev,3)),evCls),
    tile('win v control',(s.latest_win===null||s.latest_win===undefined)?'--':
      (pct(s.latest_win)+' <span class="dim" style="font-size:14px">v '+pct(s.control_win)+'</span>')),
    tile('past self',(s.ancestor_win===null||s.ancestor_win===undefined)?'--':
      (pct(s.ancestor_win)+' <span class="dim" style="font-size:14px">/ '+pct(s.ancestor_loss)+'</span>'),
      (s.ancestor_win===null||s.ancestor_win===undefined)?'dim':(s.ancestor_win>s.ancestor_loss?'good':'warn')),
    tile('entropy',num(ent,3)),
    tile('critic err',num(vl,3)+' <span class="dim" style="font-size:14px">/ '+num(rs,2)+'</span>'),
    tile('pass',np===null?'--':pct(np),(np!==null&&np>0.5)?'crit':'')
  ].join('');

  var A='#2E86AB',B='#8A6516',C='#A2352C',D='#26704F',G='#8695A4',p=slot;
  q('c1').innerHTML=chart('ch'+p+'a','Lift vs control',
      [{name:'lift',color:A,points:S.lift}],{zero:true,fill:true})
    +'<div class="grid2">'
    +chart('ch'+p+'b','Beating its past',
      [{name:'wins',color:D,points:S.ancestor_win},{name:'losses',color:C,points:S.ancestor_loss}],
      {asPct:true,emptyText:'not self-play'})
    +chart('ch'+p+'c','Win rate',
      [{name:'agent',color:A,points:S.win},{name:'random',color:G,points:S.control}],{asPct:true})
    +'</div>';
  q('c2').innerHTML=chart('ch'+p+'d','Explained variance',
      [{name:'expl var',color:D,points:S.explained_variance}],{zero:true,fill:true,emptyText:'too few updates'})
    +'<div class="grid2">'
    +chart('ch'+p+'e','Critic error and spread',
      [{name:'loss',color:C,points:S.value_loss},{name:'spread',color:G,points:S.ret_std}],{emptyText:'too few updates'})
    +chart('ch'+p+'f','Entropy and pass rate',
      [{name:'entropy',color:B,points:S.entropy},{name:'pass',color:G,points:S.noop}],{emptyText:'too few updates'})
    +'</div>';
}

/* ------------------------------------------------------------ notifications */

var seenEvals={};

function alertsOn(){return recall('crsim-alerts','0')==='1';}

function noticeNewEvaluations(){
  if(!alertsOn()||typeof Notification==='undefined'||Notification.permission!=='granted') return;
  DATA.order.forEach(function(n){
    var s=DATA.runs[n].summary,count=s.evaluations||0;
    if(seenEvals[n]===undefined){seenEvals[n]=count;return;}
    if(count>seenEvals[n]){
      seenEvals[n]=count;
      var lift=s.latest_lift;
      try{
        new Notification(n,{body:'lift '+((lift>0?'+':'')+num(lift))+' sd at '
          +commas(s.steps)+' steps',tag:n,renotify:false});
      }catch(e){}
    }else{seenEvals[n]=count;}
  });
}

/* ------------------------------------------------------------------- shell */

var split=false,picked=['',''];

function paintTabs(){
  document.getElementById('tabs').innerHTML=DATA.order.map(function(n){
    var r=DATA.runs[n],l=r.summary.latest_lift;
    var val=(l===null||l===undefined)?'':'<span class="val">'+(l>0?'+':'')+Number(l).toFixed(2)+'</span>';
    var on=picked.slice(0,split?2:1).indexOf(n)>=0;
    return '<button class="tab" role="tab" data-run="'+n+'" aria-selected="'+on+'">'
      +'<span class="pip'+(r.live?' on':'')+'"></span>'+n+val+'</button>';
  }).join('');
  Array.prototype.forEach.call(document.querySelectorAll('.tab'),function(t){
    t.addEventListener('click',function(){picked[0]=t.dataset.run;remember('crsim-pane0',picked[0]);draw();});
  });
}

function draw(){
  var panesEl=document.getElementById('panes');
  CHARTS=[];
  document.getElementById('title').textContent=split?'cr-sim':picked[0];
  panesEl.className='panes'+(split?' split':'');
  panesEl.innerHTML=paneMarkup(0)+(split?paneMarkup(1):'');
  for(var i=0;i<(split?2:1);i++){
    (function(i){
      var pane=panesEl.querySelector('[data-pane="'+i+'"]');
      var pick=pane.querySelector('[data-role="pick"]');
      pick.innerHTML=DATA.order.map(function(n){
        return '<option value="'+n+'"'+(n===picked[i]?' selected':'')+'>'+n+'</option>';}).join('');
      pick.addEventListener('change',function(){
        picked[i]=pick.value;remember('crsim-pane'+i,pick.value);draw();});
      pane.querySelector('.pane-head').style.display=split?'flex':'none';
      fillPane(pane,picked[i],i);
    })(i);
  }
  paintTabs();
  wireScrub();
  var s=DATA.runs[picked[0]];
  document.getElementById('foot').textContent=s
    ? (s.summary.updates+' updates, '+s.summary.evaluations+' evaluations')
    : '';
  var pulse=document.getElementById('pulse');
  pulse.className='dot'+(s&&s.live?'':' stale');
  document.getElementById('stamp').textContent=new Date().toLocaleTimeString();
}

function apply(next){
  DATA=next;
  DATA.order.forEach(function(n){if(picked.indexOf(n)<0&&!picked[0])picked[0]=n;});
  if(!DATA.runs[picked[0]])picked[0]=DATA.order[0];
  if(!DATA.runs[picked[1]])picked[1]=DATA.order[Math.min(1,DATA.order.length-1)];
  draw();
  noticeNewEvaluations();
}

/* Polled, and re-rendered only when the payload actually changed. A timed
   reload throws away scroll position, an open glossary and any chart being
   scrubbed, several times an hour, to redraw numbers that had not moved. */
function poll(){
  fetch('data.json',{cache:'no-store'}).then(function(r){return r.json();}).then(function(next){
    if(next.version!==DATA.version) apply(next);
    else document.getElementById('stamp').textContent=new Date().toLocaleTimeString();
  }).catch(function(){}).then(function(){setTimeout(poll,15000);});
}

(function start(){
  picked=[recall('crsim-pane0',DATA.order[0]),
          recall('crsim-pane1',DATA.order[Math.min(1,DATA.order.length-1)])]
         .map(function(n){return DATA.runs[n]?n:DATA.order[0];});
  split=recall('crsim-split','0')==='1'&&DATA.order.length>1;
  var toggle=document.getElementById('split');
  toggle.setAttribute('aria-pressed',String(split));
  if(DATA.order.length<2) toggle.style.display='none';
  toggle.addEventListener('click',function(){
    split=!split;remember('crsim-split',split?'1':'0');
    toggle.setAttribute('aria-pressed',String(split));draw();});

  var bell=document.getElementById('bell');
  function paintBell(){
    var on=alertsOn()&&typeof Notification!=='undefined'&&Notification.permission==='granted';
    bell.dataset.on=on?'1':'0';
    bell.textContent=on?'Alerts on':'Alerts off';
  }
  bell.addEventListener('click',function(){
    if(typeof Notification==='undefined'){bell.textContent='Unsupported';return;}
    if(alertsOn()){remember('crsim-alerts','0');paintBell();return;}
    Notification.requestPermission().then(function(p){
      remember('crsim-alerts',p==='granted'?'1':'0');paintBell();});
  });
  paintBell();

  DATA.order.forEach(function(n){seenEvals[n]=DATA.runs[n].summary.evaluations||0;});
  draw();
  tickCountdowns();
  setInterval(tickCountdowns,1000);
  if(location.protocol==='http:'||location.protocol==='https:') setTimeout(poll,15000);
})();
</script>
</body></html>
"""





#: A run whose metrics file has not been touched this recently is shown as
#: finished. Generous on purpose: an update at the slowest cadence measured
#: here takes a couple of minutes, and calling a live run dead is worse than
#: being slow to notice a finished one.
_LIVE_SECONDS = 300.0



def _local_addresses() -> list[str]:
    """Every address this machine can plausibly be reached on.

    Printed rather than guessed at, because the useful one is the wifi
    address and a machine typically has several -- loopback, a virtual
    adapter or two, and the real one.
    """
    import socket

    found: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       family=socket.AF_INET):
            address = info[4][0]
            if address not in found and not address.startswith("127."):
                found.append(address)
    except OSError:
        pass
    return found


def serve(page: Path, port: int) -> None:
    """Serve ``page`` on every interface until interrupted.

    The file is read per request rather than held in memory: the refresh loop
    rewrites it every few seconds, and a cached copy would show a phone the
    state at the moment the server started.
    """
    import http.server
    import socketserver

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - the stdlib spells it this way
            wanted = page
            kind = "text/html; charset=utf-8"
            if self.path.rstrip("/").endswith("data.json"):
                wanted = page.with_suffix(".json")
                kind = "application/json"
            elif self.path not in ("/", "/index.html", "/progress.html"):
                self.send_error(404)
                return
            try:
                body = wanted.read_bytes()
            except OSError:
                self.send_error(503, "not written yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            # No caching: the whole point is that it changes.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep the console for training output
            pass

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with Server(("0.0.0.0", port), Handler) as httpd:
        print(f"serving {page.name} on port {port}", flush=True)
        for address in _local_addresses():
            print(f"  http://{address}:{port}", flush=True)
        print("  (same wifi; Windows may ask to allow it through the firewall)",
              flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cr-sim-watch")
    parser.add_argument(
        "runs", nargs="*", type=Path, default=None,
        help="run directories to watch. Several are shown as tabs on one "
             "page, with a split view for comparing two at once -- the "
             "question is almost always whether this run is beating that "
             "one, and answering it across two files is how a difference "
             "gets missed. Defaults to every run under runs/.",
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the page (default: progress.html)")
    parser.add_argument("--open", action="store_true", help="open it in a browser")
    parser.add_argument("--once", action="store_true",
                        help="write once and exit rather than refreshing")
    parser.add_argument("--every", type=float, default=30.0)
    parser.add_argument(
        "--serve", type=int, default=0, metavar="PORT",
        help="also serve the page on this port, on every interface, so a "
             "phone on the same wifi can watch a run. 0 disables it.",
    )
    args = parser.parse_args(argv)

    def discover() -> list[Path]:
        """Which runs to show, re-asked on every refresh.

        Rescanned rather than listed once at startup: a run begun after the
        watcher was already going would otherwise never appear, and the page
        looks identical whether a run is missing or merely quiet. That is
        exactly how two diagnostic runs went unnoticed for several minutes.
        """
        if args.runs:
            return list(args.runs)
        root = ROOT / "runs"
        if not root.is_dir():
            return []
        found = [
            d for d in root.iterdir()
            if d.is_dir() and (d / "metrics.jsonl").is_file()
        ]
        # Ordered by when each run started, not by name. Alphabetical put
        # today's run between two from last week, and the question being
        # asked of this page is almost always "what changed since the last
        # one" -- which only reads properly in the order they happened.
        return sorted(found, key=_started_at)

    runs = discover()
    if not runs:
        print("no runs to watch", file=sys.stderr)
        return 1

    out = args.out or (runs[0] / "progress.html" if len(runs) == 1
                       else ROOT / "progress.html")
    out.parent.mkdir(parents=True, exist_ok=True)

    def once() -> int:
        now = time.time()
        collected = []
        notes: dict[str, str] = {}
        for run in discover():
            metrics = run / "metrics.jsonl"
            rows = read_metrics(metrics)
            if not rows:
                continue
            # Deduplicated by update: the files written before the
            # double-write fix are still on disk, and counting each update
            # twice is what once made a stalled run look busy.
            by_update = {r.get("updates", i): r for i, r in enumerate(rows)}
            rows = [by_update[k] for k in sorted(by_update)]
            live = metrics.is_file() and (now - metrics.stat().st_mtime) < _LIVE_SECONDS
            collected.append((run.name, rows, live))
            note = _note_of(run)
            if note:
                notes[run.name] = note
        if not collected:
            return 0
        html, body = render_multi(collected, notes=notes)
        out.write_text(html, encoding="utf-8")
        # Beside the page, because a served copy polls this rather than
        # reloading itself. Written second so a reader never sees data newer
        # than the page that shipped with it.
        out.with_suffix(".json").write_text(
            json.dumps(body), encoding="utf-8")
        return sum(len(rows) for _, rows, _ in collected)

    count = once()
    print(f"{len(discover())} run(s), {count} updates -> {out}", flush=True)
    if args.open:
        import webbrowser

        webbrowser.open(out.resolve().as_uri())
    if args.once:
        return 0

    if args.serve:
        # In a thread, because the refresh loop below is the thing that keeps
        # the page current and the server only hands out whatever it last
        # wrote.
        import threading

        threading.Thread(target=serve, args=(out, args.serve),
                         daemon=True).start()

    # Polled rather than watched. The page refreshes on a timer anyway, so
    # reacting within a second of a write buys nothing, and a filesystem
    # watcher would be more machinery than the interval deserves.
    try:
        while True:
            time.sleep(max(1.0, args.every))
            once()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
