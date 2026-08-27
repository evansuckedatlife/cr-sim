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




#: How a command line is reported on the page. Ordered, first match wins.
_JOB_KINDS = (
    ("train.run", "training"),
    ("make_demos", "recording demonstrations"),
    ("clone_policy", "cloning a policy"),
    ("measure_expert", "measuring the expert"),
    ("play.server", "play server"),
    ("train.watch", "this page"),
    ("soak", "soak test"),
)


def running_jobs() -> list[dict[str, Any]]:
    """Every cr-sim process alive right now, with what it is doing.

    Read from the operating system rather than from a registry of jobs that
    announced themselves. A registry only ever knows about the well-behaved
    ones, and the case worth catching is the opposite: three copies of the
    same script running at once because an interrupted command had already
    started before it was interrupted.

    Best-effort. If psutil is missing or the platform will not say, this
    returns nothing and the page simply does not show the strip -- an
    approximate listing is worth having, and a missing one must not take the
    rest of the page down with it.
    """
    try:
        import psutil
    except ImportError:
        return _jobs_via_shell()

    found: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent",
                                        "memory_info", "create_time"]):
        try:
            info = process.info
            if not info.get("name", "").lower().startswith("python"):
                continue
            line = " ".join(info.get("cmdline") or ())
            if "cr_sim" not in line and "scripts" not in line:
                continue
            kind = next((label for needle, label in _JOB_KINDS if needle in line), None)
            if kind is None:
                continue
            name = None
            if "--name" in line:
                parts = line.split("--name", 1)[1].split()
                name = parts[0] if parts else None
            elif "--shard" in line:
                parts = line.split("--shard", 1)[1].split()
                name = f"shard {parts[0]}" if parts else None
            found.append({
                "pid": info["pid"],
                "kind": kind,
                "name": name,
                "memory_mb": round((info.get("memory_info").rss if info.get("memory_info") else 0) / 1e6),
                "age_seconds": max(0.0, time.time() - (info.get("create_time") or time.time())),
            })
        except Exception:  # pragma: no cover - processes vanish mid-iteration
            continue
    return _group(found)


def _group(found: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per job, not one per process.

    Launching a module spawns a shell that then spawns the worker, so every
    job appears twice -- and a self-play run appears four times once its
    environment workers are counted. Listing all of them buries the thing this
    strip exists to show, which is a job running more than once by mistake.
    """
    groups: dict[tuple, dict[str, Any]] = {}
    for job in found:
        key = (job["kind"], job["name"])
        entry = groups.setdefault(key, {
            "kind": job["kind"], "name": job["name"], "pid": job["pid"],
            "memory_mb": 0, "age_seconds": job["age_seconds"], "processes": 0,
        })
        entry["processes"] += 1
        entry["memory_mb"] += job["memory_mb"]
        # The oldest process is the one that was actually launched; its
        # children are younger, and the launcher's age is the job's age.
        if job["age_seconds"] > entry["age_seconds"]:
            entry["age_seconds"] = job["age_seconds"]
            entry["pid"] = job["pid"]
    for entry in groups.values():
        # Launching a module costs two processes: a shell and the worker it
        # spawns. More than that means either genuine parallelism -- a
        # training run's environment workers -- or the same job started
        # several times over, which is what three simultaneous clones of one
        # script looked like when nothing on the page could show it.
        entry["suspicious"] = (
            entry["processes"] > 2 and entry["kind"] != "training")
    return sorted(groups.values(), key=lambda j: (j["kind"], str(j["name"])))


def _jobs_via_shell() -> list[dict[str, Any]]:
    """Fallback for a machine without psutil."""
    import subprocess

    try:
        out = subprocess.run(
            ["wmic", "process", "where", "name like '%python%'",
             "get", "ProcessId,CommandLine", "/format:csv"],
            capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    found: list[dict[str, Any]] = []
    for row in out.splitlines():
        kind = next((label for needle, label in _JOB_KINDS if needle in row), None)
        if kind is None:
            continue
        digits = [p for p in row.strip().split(",") if p.strip().isdigit()]
        found.append({"pid": int(digits[-1]) if digits else 0, "kind": kind,
                      "name": None, "memory_mb": 0, "age_seconds": 0.0})
    return sorted(found, key=lambda j: (j["kind"], j["pid"]))


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
    jobs: "Sequence[dict[str, Any]] | None" = None,
) -> str:
    """One page holding several runs, as ``(name, rows, live)`` triples.

    Runs are compared far more often than they are watched alone -- the
    question is almost always "is this one doing better than that one", and
    answering it by opening two files and scrolling both is how a difference
    gets missed. Tabs switch between them and a split view puts two
    side by side against shared axes.

    ``live`` marks a run whose metrics file was written recently, so a
    finished run is not presented as though it were still moving.
    """
    payload = json.dumps({
        "runs": {
            name: {
                "series": _series_of(rows),
                "summary": summarise(rows),
                "live": bool(live),
            }
            for name, rows, live in runs
        },
        "order": [name for name, _, _ in runs],
        "back": back,
        "jobs": jobs or [],
    })
    return _PAGE.replace("__DATA__", payload)


def render(rows: Sequence[dict[str, Any]], title: str,
           back: str | None = None) -> str:
    """The page for a single run. One file, no dependencies, no build step."""
    return render_multi([(title, rows, True)], back=back)


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="20">
<title>cr-sim training</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root{
  --ground:#EFF2F5; --panel:#FFFFFF; --panel2:#F6F8FA;
  --ink:#121A24; --soft:#3C4855; --muted:#626F7E;
  --rule:#D8DFE6; --hair:#E7ECF1;
  --accent:#14657F; --accentw:#E3EFF4;
  --good:#26704F; --goodw:#E2F0E9;
  --warn:#8A6516; --warnw:#F7EFDD;
  --crit:#A2352C; --critw:#F7E7E5;
  --grey:#8695A4;
  --shadow:0 1px 2px rgba(18,26,36,.06),0 8px 24px -12px rgba(18,26,36,.18);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0D1218; --panel:#141C25; --panel2:#1A232D;
  --ink:#E6ECF2; --soft:#C2CCD7; --muted:#8695A4;
  --rule:#26313D; --hair:#1E2833;
  --accent:#58B4D0; --accentw:#12303C;
  --good:#5FB68C; --goodw:#163529;
  --warn:#DCAB57; --warnw:#362B15;
  --crit:#E58177; --critw:#3A201D;
  --grey:#7C8B9A;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
}}
:root[data-theme="dark"]{
  --ground:#0D1218; --panel:#141C25; --panel2:#1A232D;
  --ink:#E6ECF2; --soft:#C2CCD7; --muted:#8695A4;
  --rule:#26313D; --hair:#1E2833;
  --accent:#58B4D0; --accentw:#12303C;
  --good:#5FB68C; --goodw:#163529;
  --warn:#DCAB57; --warnw:#362B15;
  --crit:#E58177; --critw:#3A201D;
  --grey:#7C8B9A;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
}
*{box-sizing:border-box}
body{
  background:var(--ground); color:var(--ink); margin:0; padding:0 20px 88px;
  font-family:"Source Sans 3",ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  font-size:16.5px; line-height:1.6; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1000px;margin:0 auto}
h1,h2,h3,.lbl,.kpi-n,.v{font-family:Archivo,ui-sans-serif,system-ui,sans-serif}
.mono,.v,.kpi-n,.num,td.n{font-family:"JetBrains Mono",ui-monospace,SFMono-Regular,Consolas,monospace;font-variant-numeric:tabular-nums}
.lbl{font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}

header{padding:52px 0 26px;border-bottom:1px solid var(--rule)}
h1{font-size:clamp(32px,5vw,46px);font-weight:700;letter-spacing:-.022em;line-height:1.05;margin:10px 0 0;text-wrap:balance}
.dek{margin:14px 0 0;max-width:64ch;font-size:17.5px;color:var(--soft)}
.meta{margin-top:20px;font-size:13px;color:var(--muted);display:flex;gap:8px 18px;flex-wrap:wrap;align-items:center}
.live{display:inline-flex;align-items:center;gap:7px;font-weight:600;color:var(--good)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--good);box-shadow:0 0 0 3px var(--goodw)}
@media (prefers-reduced-motion:no-preference){.dot{animation:p 2.4s ease-in-out infinite}}
@keyframes p{0%,100%{opacity:1}50%{opacity:.3}}

/* general stats strip */
.readout{margin-top:26px;background:var(--panel);border:1px solid var(--rule);border-radius:4px;box-shadow:var(--shadow);overflow:hidden}
.readout-head{display:flex;align-items:center;gap:10px;padding:11px 18px;border-bottom:1px solid var(--hair);background:var(--panel2)}
.bar{height:3px;background:var(--hair);position:relative}
.bar i{position:absolute;inset:0 auto 0 0;background:var(--accent);display:block}
.readout-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(124px,1fr))}
.cell{padding:16px 18px;border-right:1px solid var(--hair);border-top:1px solid var(--hair)}
.cell:first-child{border-top:0}
@media (min-width:760px){.cell{border-top:0}}
.cell:last-child{border-right:0}
.kpi-n{font-size:23px;font-weight:600;letter-spacing:-.02em;display:block;margin-bottom:3px}
.kpi-n small{font-size:13px;font-weight:400;color:var(--muted)}

h2{font-size:12px;font-weight:700;letter-spacing:.11em;text-transform:uppercase;color:var(--accent);margin:52px 0 5px}
.lede{margin:0 0 20px;max-width:70ch;color:var(--soft);font-size:15.5px}

/* headline verdict, severity-striped like the report */
.hero{background:var(--panel);border:1px solid var(--rule);border-left:3px solid var(--sev,var(--rule));border-radius:3px;padding:24px 26px;box-shadow:var(--shadow)}
.hero.good{--sev:var(--good)} .hero.warn{--sev:var(--warn)} .hero.crit{--sev:var(--crit)} .hero.dim{--sev:var(--grey)}
.hero-top{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
.hero .v{font-size:44px;font-weight:700;letter-spacing:-.03em;line-height:1}
.chip{font-family:Archivo,sans-serif;font-size:10.5px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;padding:3px 8px;border-radius:2px;white-space:nowrap}
.chip.good{background:var(--goodw);color:var(--good)}
.chip.warn{background:var(--warnw);color:var(--warn)}
.chip.crit{background:var(--critw);color:var(--crit)}
.chip.dim{background:var(--hair);color:var(--muted)}
.hero p{margin:12px 0 0;max-width:74ch;color:var(--soft);font-size:15px}
.hero .best{margin-top:11px;font-size:13px;color:var(--muted)}

.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(228px,1fr));gap:13px;margin-top:13px}
.tile{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:17px 19px;box-shadow:var(--shadow)}
.tile .v{font-size:26px;font-weight:700;letter-spacing:-.025em;margin:7px 0 0;display:block}
.tile .n{font-size:13.5px;color:var(--soft);margin-top:8px;line-height:1.5}
.tile .n b{color:var(--ink);font-weight:600}
.good{color:var(--good)}.warn{color:var(--warn)}.crit{color:var(--crit)}.dim{color:var(--muted)}

.chart{background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:19px 21px;box-shadow:var(--shadow);margin-top:13px}
.chart h3{font-size:16.5px;font-weight:600;margin:0 0 4px;letter-spacing:-.012em}
.chart p{margin:0 0 14px;font-size:14px;color:var(--soft);max-width:78ch}
.chart svg{display:block;width:100%;height:auto}
.empty{padding:24px;text-align:center;color:var(--muted);font-size:14px;background:var(--panel2);border-radius:3px}
.axis{stroke:var(--rule);stroke-width:1}
.zero{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3;opacity:.55}
.grid-l{stroke:var(--hair);stroke-width:1}
.lbl-s{font:600 10px "JetBrains Mono",ui-monospace,monospace;fill:var(--muted)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:13px}

dl.gloss{background:var(--panel);border:1px solid var(--rule);border-radius:3px;box-shadow:var(--shadow);margin:0;padding:4px 26px 22px}
dl.gloss dt{font-family:Archivo,sans-serif;font-weight:700;margin-top:20px;font-size:16px;letter-spacing:-.008em}
dl.gloss dd{margin:5px 0 0;color:var(--soft);font-size:14.5px;max-width:80ch}
dl.gloss dd.read{margin-top:7px;font-size:13.5px;color:var(--muted);padding-left:13px;border-left:2px solid var(--hair)}
code{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:.87em;background:var(--accentw);color:var(--accent);padding:1px 5px;border-radius:2px}
.jobs{margin-top:24px;background:var(--panel);border:1px solid var(--rule);
  border-left:3px solid var(--good);border-radius:3px;padding:14px 18px;box-shadow:var(--shadow)}
.jobs.idle{border-left-color:var(--muted)}
.jobs-head{display:flex;align-items:center;gap:9px;margin-bottom:9px}
.jobs-list{display:flex;flex-wrap:wrap;gap:7px}
.job{display:inline-flex;align-items:center;gap:7px;background:var(--panel2);
  border:1px solid var(--hair);border-radius:3px;padding:5px 10px;font-size:13px}
.job b{font-weight:600}
.job .meta{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:11.5px;color:var(--muted)}
.job.warn{border-color:var(--warn);background:var(--warnw)}
.tabbar{display:flex;align-items:flex-end;gap:12px;margin-top:30px;
  border-bottom:1px solid var(--rule);flex-wrap:wrap}
.tabs{display:flex;gap:2px;flex-wrap:wrap;flex:1}
.tab{appearance:none;border:1px solid transparent;border-bottom:0;background:none;
  font-family:Archivo,sans-serif;font-size:13.5px;font-weight:600;color:var(--muted);
  padding:9px 14px;border-radius:3px 3px 0 0;cursor:pointer;display:flex;align-items:center;gap:8px}
.tab:hover{color:var(--ink);background:var(--panel2)}
.tab[aria-selected="true"]{background:var(--panel);border-color:var(--rule);color:var(--ink);
  margin-bottom:-1px;padding-bottom:10px}
.tab .pip{width:7px;height:7px;border-radius:50%;background:var(--muted);flex:none}
.tab .pip.on{background:var(--good);box-shadow:0 0 0 3px var(--goodw)}
.tab .val{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:12px;font-variant-numeric:tabular-nums}
.split-toggle{appearance:none;background:var(--panel);border:1px solid var(--rule);border-radius:3px;
  font-family:Archivo,sans-serif;font-size:12px;font-weight:600;letter-spacing:.04em;color:var(--soft);
  padding:7px 13px;cursor:pointer;margin-bottom:8px}
.split-toggle:hover{color:var(--ink);border-color:var(--accent)}
.split-toggle[aria-pressed="true"]{background:var(--accentw);color:var(--accent);border-color:var(--accent)}
.panes{display:grid;grid-template-columns:1fr;gap:20px;margin-top:22px}
.panes.split{grid-template-columns:1fr 1fr}
@media (max-width:900px){.panes.split{grid-template-columns:1fr}}
.pane{min-width:0}
.pane-head{display:flex;align-items:center;gap:10px;margin-bottom:12px}
.pane-head select{font-family:Archivo,sans-serif;font-size:13.5px;font-weight:600;color:var(--ink);
  background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:6px 9px}
.panes.split .chart svg{max-height:170px}
.panes.split .readout-grid{grid-template-columns:repeat(auto-fit,minmax(96px,1fr))}
.panes.split .hero .v{font-size:34px}
footer{margin-top:56px;padding-top:18px;border-top:1px solid var(--rule);font-size:13px;color:var(--muted)}
</style></head><body>
<div class="wrap">

<header>
  <div class="lbl"><a id="back" href="#" style="color:var(--accent);text-decoration:none;display:none">&larr; all runs</a><span id="eyebrow">cr-sim &middot; reinforcement learning</span></div>
  <h1 id="title">cr-sim training</h1>
  <p class="dek">A PPO agent learning Clash Royale, rewarded by the change in what the board is already worth.</p>
  <div class="meta">
    <span class="live"><span class="dot"></span> live</span>
    <span>reloads every 20s</span>
    <span id="stamp"></span>
  </div>
</header>

<div id="jobs"></div>

<div class="tabbar">
  <div class="tabs" id="tabs" role="tablist"></div>
  <button class="split-toggle" id="split" aria-pressed="false">Split view</button>
</div>

<div class="panes" id="panes"></div>

<h2>What each number means</h2>
<p class="lede">Written out because most of these are easy to misread, and two of them were misread on this project already.</p>
<dl class="gloss">
  <dt>Lift vs control</dt>
  <dd>The only number that says whether the agent is any good. It plays a <b>uniform-random opponent</b> over 40 fixed battles, and a random agent plays the <em>same</em> 40. The gap is divided by how much the random agent&rsquo;s own score bounces around, so the unit is standard deviations of noise.</dd>
  <dd class="read">0 means no better than random, whatever the win rate says. Roughly 0.5 is a real effect. Below about 0.25 is inside the noise and should not be believed.</dd>

  <dt>Beats its past self</dt>
  <dd>Self-play only. Every so often the policy plays 30 battles against the
  <b>oldest version of itself</b> still kept in the opponent pool, and this is
  how it did.</dd>
  <dd class="read">The most sensitive measure here, because the reference is
  fixed, deterministic, and roughly the right difficulty. Lift against a random
  control has a spread wide enough that a +0.375 reading on this project
  measured &minus;0.033 when replayed over 300 battles. Read it for what it is
  though: beating your past self is evidence of movement, not of skill &mdash;
  two policies can trade wins while both stay hopeless, which is why the random
  control stays as an anchor.</dd>

  <dt>Win rate vs control win rate</dt>
  <dd>Share of those 40 battles each side won. Both look low because most 120-second matches end 0&ndash;0 with no tower falling, and a draw is not a win &mdash; so the control&rsquo;s own rate, not 50%, is the bar to clear.</dd>

  <dt>Explained variance</dt>
  <dd>How much of the match outcome the critic can actually predict, on a scale where 1 is perfect.</dd>
  <dd class="read">0 means no better than guessing the average every time. It sat at 0.06 on the previous run, which is the likeliest reason a policy committed hard to a strategy that never got stronger. Watching this climb is the point of the current reward design.</dd>

  <dt>Entropy</dt>
  <dd>How undecided the policy is. It starts near the maximum for 720 possible plays and falls as the agent develops preferences.</dd>
  <dd class="read">Falling entropy means commitment. That is only good news when lift rises with it &mdash; committing without improving is how a policy converges on one useless move it repeats forever.</dd>

  <dt>Value loss and return spread</dt>
  <dd>Value loss is the critic&rsquo;s squared error, and it is meaningless alone because it scales with the spread of what it is fitting. An error of 5 is dreadful against returns that vary by 0.5 and unremarkable against returns that vary by 3.</dd>
  <dd class="read">This exact confusion cost real time here: a value loss of 5 was called a broken critic when it was only a different reward scale. Always read it beside the spread.</dd>

  <dt>Rollout win rate</dt>
  <dd>The agent&rsquo;s win rate during training. The flattering number and the misleading one: measured while the policy is still exploring, on whatever battles it drew, and on this project it has run <b>about eighteen points optimistic</b> &mdash; reporting 55% for a policy that evaluated at 37%.</dd>
  <dd class="read">Shown for completeness. Steer by lift.</dd>

  <dt>Pass rate</dt>
  <dd>How often the agent chose to play nothing when it could have played something. Passing is the one action never punished, so a run that quietly collapses into always passing looks healthy on every other metric.</dd>

  <dt>Throughput</dt>
  <dd>Decisions per second and matches finished. Each decision covers 30 engine ticks &mdash; 1.5 seconds of match time &mdash; and each match is 120 seconds.</dd>
</dl>

<footer id="foot"></footer>
</div>

<script>
const DATA = __DATA__;

function num(x,d){return (x===null||x===undefined||isNaN(x))?'--':Number(x).toFixed(d===undefined?2:d);}
function pct(x){return (x===null||x===undefined)?'--':(x*100).toFixed(0)+'%';}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function last(p){return (p&&p.length)?p[p.length-1][1]:null;}
function commas(n){return Number(n||0).toLocaleString();}
function dur(sec){
  if(!sec&&sec!==0) return '--';
  var h=Math.floor(sec/3600), m=Math.round((sec%3600)/60);
  return h?(h+'h '+m+'m'):(m+'m');
}

function cell(v,k,sm){
  return '<div class="cell"><span class="kpi-n">'+v+(sm?' <small>'+sm+'</small>':'')+'</span><span class="lbl">'+k+'</span></div>';
}

function renderGeneral(s,S,pane){
  var doneFrac = s.total_steps ? Math.min(1, (s.steps||0)/s.total_steps) : 0;
  var bar = pane ? pane.querySelector('[data-role="bar"]') : null;
  if (bar) bar.style.width = (doneFrac*100).toFixed(1)+'%';
  var remain = (s.total_steps && s.steps_per_second)
      ? (s.total_steps-s.steps)/s.steps_per_second : null;
  return [
    cell(commas(s.steps), 'Decisions', s.total_steps?('of '+commas(s.total_steps)):''),
    cell(num(s.steps_per_second,1), 'Per second'),
    cell(commas(s.episodes), 'Matches played'),
    cell(commas(s.updates), 'Updates'),
    cell(dur(s.elapsed_seconds), 'Elapsed'),
    cell(remain===null?'--':dur(remain), 'Remaining', 'est'),
    cell(commas(s.evaluations), 'Evaluations')
  ].join('');
}

function verdictFor(lift){
  if(lift===null||lift===undefined) return ['dim','Not measured yet',
    'The first evaluation runs at update 10. Until one exists, nothing here says whether the agent is learning &mdash; only that it is running.'];
  if(lift>=0.5) return ['good','Clearly better than random',
    'Comfortably outside the control&rsquo;s own noise. This is a real effect.'];
  if(lift>=0.25) return ['good','Probably better than random',
    'Outside the noise, but not by much. Worth believing only once it holds across several evaluations.'];
  if(lift>-0.25) return ['warn','Indistinguishable from random',
    'Inside the control&rsquo;s own bounce. A single positive reading here means nothing &mdash; on the previous run, six evaluations averaged +0.04 while individual ones reached +0.23.'];
  return ['crit','Worse than random',
    'The policy has committed to something actively bad, which is what a critic that predicts nothing tends to produce.'];
}

function renderHero(s){
  var lift=s.latest_lift, r=verdictFor(lift), cls=r[0];
  var v=(lift===null||lift===undefined)?'&mdash;':((lift>0?'+':'')+num(lift)+' sd');
  var best=(s.best_lift===null||s.best_lift===undefined)?''
    :'<div class="best">best so far '+(s.best_lift>0?'+':'')+num(s.best_lift)+' sd at '+commas(s.best_at_steps)+' decisions</div>';
  return '<div class="hero '+cls+'"><div class="lbl">lift vs control &mdash; the honest score</div>'
    +'<div class="hero-top"><span class="v '+cls+'">'+v+'</span><span class="chip '+cls+'">'+r[1]+'</span></div>'
    +'<p>'+r[2]+'</p>'+best+'</div>';
}

function tile(k,v,note,cls){
  return '<div class="tile"><div class="lbl">'+k+'</div><span class="v '+(cls||'')+'">'+v+'</span>'
       + '<div class="n">'+(note||'')+'</div></div>';
}

function renderTiles(s,S){
  var ev=last(S.explained_variance), ent=last(S.entropy),
      vl=last(S.value_loss), rs=last(S.ret_std), np=last(S.noop), rw=last(S.rollout_win);
  var evCls = ev===null?'dim':(ev>=0.3?'good':ev>=0.1?'warn':'crit');
  return [
    tile('explained variance', ev===null?'--':((ev>0?'+':'')+num(ev,3)),
      'Share of the outcome the critic can predict. <b>0 = nothing.</b> The bottleneck on the last run.', evCls),
    tile('win rate vs control',
      (s.latest_win===null||s.latest_win===undefined)?'--':(pct(s.latest_win)+' <span class="dim" style="font-size:17px">v '+pct(s.control_win)+'</span>'),
      'Same 40 battles for both. Draws are not wins, so the control&rsquo;s rate is the bar &mdash; not 50%.'),
    tile('entropy', num(ent,3),
      'How undecided the policy is. Falling means committing &mdash; good only if lift rises too.'),
    tile('critic error', num(vl,3),
      'Squared error against returns that spread <b>'+num(rs,2)+'</b>. Meaningless without that second number.'),
    tile('pass rate', np===null?'--':pct(np),
      'How often it declined to play. Never punished, so a run can quietly collapse into it.',
      (np!==null&&np>0.5)?'crit':''),
    tile('rollout win rate', rw===null?'--':pct(rw),
      'Measured while exploring, and about eighteen points optimistic here. Do not steer by it.', 'dim'),
    tile('beats its past self',
      (s.ancestor_win===null||s.ancestor_win===undefined)?'--':
        (pct(s.ancestor_win)+' <span class="dim" style="font-size:17px">w / '+pct(s.ancestor_loss)+' l</span>'),
      (s.ancestor_win===null||s.ancestor_win===undefined)
        ? 'Self-play runs only. Nothing to climb against yet.'
        : ('Against generation <b>'+s.ancestor_age+'</b>, the oldest version kept. '
           +'Above 50% of decided battles means it is genuinely moving.'),
      (s.ancestor_win===null||s.ancestor_win===undefined) ? 'dim'
        : (s.ancestor_win > s.ancestor_loss ? 'good' : 'warn'))
  ].join('');
}

function chart(title,note,series,opts){
  opts=opts||{};
  var all=[]; series.forEach(function(s){(s.points||[]).forEach(function(p){all.push(p);});});
  if(all.length<1){
    return '<div class="chart"><h3>'+title+'</h3><p>'+note+'</p>'
      +'<div class="empty">'+(opts.emptyText||'no evaluations yet')+'</div></div>';
  }
  // A single reading is not nothing, and saying "no evaluations yet" over the
  // top of one is a lie the page told for its first hour. It cannot be drawn
  // as a line, so it is stated as a value instead.
  if(all.length<2 || series.every(function(s){return (s.points||[]).length<2;})){
    var only = series.filter(function(s){return (s.points||[]).length;}).map(function(s){
      var pt=s.points[s.points.length-1];
      return '<span style="color:'+s.color+'"><b>'+(opts.asPct?pct(pt[1]):num(pt[1],3))+'</b> '+s.name+'</span>';
    }).join('<span style="color:var(--muted)"> &nbsp;/&nbsp; </span>');
    return '<div class="chart"><h3>'+title+'</h3><p>'+note+'</p>'
      +'<div class="empty" style="font-size:15px">'+only
      +'<div style="margin-top:7px;font-size:13px;color:var(--muted)">one reading so far &mdash; a trend needs at least two</div>'
      +'</div></div>';
  }
  var longest=0; series.forEach(function(s){ if((s.points||[]).length) longest=Math.max(longest,s.name.length); });
  // Right padding sized to the longest series label, which was clipping the
  // legend to "rollout wc" at a fixed 64px.
  var w=640,h=200,padL=46,padR=Math.max(52, 20+longest*6.3),padT=14,padB=26;
  var xs=all.map(function(p){return p[0];}), ys=all.map(function(p){return p[1];});
  var x0=Math.min.apply(null,xs), x1=Math.max.apply(null,xs);
  var lo=Math.min.apply(null,ys), hi=Math.max.apply(null,ys);
  if(opts.zero){lo=Math.min(lo,0);hi=Math.max(hi,0);}
  if(hi-lo<1e-9){hi+=0.5;lo-=0.5;}
  var pd=(hi-lo)*0.12; hi+=pd; lo-=pd;
  var X=function(v){return padL+(x1===x0?0.5:(v-x0)/(x1-x0))*(w-padL-padR);};
  var Y=function(v){return padT+(1-(v-lo)/(hi-lo))*(h-padT-padB);};
  var body='';
  [0.25,0.5,0.75].forEach(function(f){
    var y=padT+f*(h-padT-padB);
    body+='<line class="grid-l" x1="'+padL+'" x2="'+(w-padR)+'" y1="'+y.toFixed(1)+'" y2="'+y.toFixed(1)+'"/>';
  });
  if(opts.zero) body+='<line class="zero" x1="'+padL+'" x2="'+(w-padR)+'" y1="'+Y(0).toFixed(1)+'" y2="'+Y(0).toFixed(1)+'"/>';
  series.forEach(function(s){
    var pts=s.points||[]; if(pts.length<2) return;
    if(opts.fill){
      var a=pts.map(function(p,i){return (i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1);}).join(' ')
        +' L'+X(pts[pts.length-1][0]).toFixed(1)+' '+Y(Math.max(lo,0)).toFixed(1)
        +' L'+X(pts[0][0]).toFixed(1)+' '+Y(Math.max(lo,0)).toFixed(1)+' Z';
      body+='<path d="'+a+'" fill="'+s.color+'" opacity=".08"/>';
    }
    var d=pts.map(function(p,i){return (i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1);}).join(' ');
    body+='<path d="'+d+'" fill="none" stroke="'+s.color+'" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>';
    var lp=pts[pts.length-1];
    body+='<circle cx="'+X(lp[0]).toFixed(1)+'" cy="'+Y(lp[1]).toFixed(1)+'" r="3.6" fill="'+s.color+'"/>';
    body+='<text class="lbl-s" x="'+(w-padR+7)+'" y="'+(Y(lp[1])+3.6).toFixed(1)+'" fill="'+s.color+'">'+s.name+'</text>';
  });
  body+='<line class="axis" x1="'+padL+'" x2="'+(w-padR)+'" y1="'+(h-padB)+'" y2="'+(h-padB)+'"/>';
  body+='<text class="lbl-s" x="4" y="'+(Y(hi)+9).toFixed(1)+'">'+hi.toFixed(2)+'</text>';
  body+='<text class="lbl-s" x="4" y="'+Y(lo).toFixed(1)+'">'+lo.toFixed(2)+'</text>';
  if(x1>x0){
    body+='<text class="lbl-s" x="'+padL+'" y="'+(h-7)+'">'+(x0/1000).toFixed(0)+'k</text>';
    body+='<text class="lbl-s" x="'+(w-padR)+'" y="'+(h-7)+'" text-anchor="end">'+(x1/1000).toFixed(0)+'k decisions</text>';
  } else {
    body+='<text class="lbl-s" x="'+((padL+w-padR)/2)+'" y="'+(h-7)+'" text-anchor="middle">'+(x0/1000).toFixed(0)+'k decisions</text>';
  }
  return '<div class="chart"><h3>'+title+'</h3><p>'+note+'</p>'
    +'<svg viewBox="0 0 '+w+' '+h+'" role="img" aria-label="'+esc(title)+'">'+body+'</svg></div>';
}

function paneMarkup(id){
  return '<div class="pane" data-pane="'+id+'">'
    + '<div class="pane-head"><span class="lbl">showing</span>'
    + '<select data-role="pick"></select></div>'
    + '<div class="readout"><div class="readout-head"><span class="lbl" style="color:var(--ink)" data-role="name"></span></div>'
    + '<div class="bar"><i data-role="bar" style="width:0%"></i></div>'
    + '<div class="readout-grid" data-role="general"></div></div>'
    + '<h2>Is it learning</h2><div data-role="hero"></div>'
    + '<div class="tiles" data-role="tiles"></div><div data-role="charts1"></div>'
    + '<h2>Is the critic working</h2><div data-role="charts2"></div></div>';
}

function fillPane(pane, name){
  var run = DATA.runs[name];
  if (!run) return;
  var S = run.series, s = run.summary;
  var q = function(role){ return pane.querySelector('[data-role="'+role+'"]'); };
  q('name').textContent = 'run "'+name+'"'+(run.live ? '' : ' (finished)');
  q('general').innerHTML = renderGeneral(s, S, pane);
  q('hero').innerHTML = renderHero(s);
  q('tiles').innerHTML = renderTiles(s, S);

  var A='#2E86AB', B='#8A6516', C='#A2352C', D='#26704F', G='#8695A4';
  q('charts1').innerHTML =
      chart('Beating its own past',
        'Wins and losses against the oldest version of itself still in the pool. A fixed, non-noisy reference at roughly the right difficulty -- unlike the random control, whose spread turned a +0.375 reading into -0.033 on retest. Empty for runs that were not self-play.',
        [{name:'wins',color:D,points:S.ancestor_win},{name:'losses',color:C,points:S.ancestor_loss}],
        {asPct:true, emptyText:'not a self-play run'})
    + chart('Lift vs control',
        'Each point is 40 fixed battles against a random agent, paired so both sides play the same ones. Zero is the line that matters.',
        [{name:'lift',color:A,points:S.lift}], {zero:true, fill:true})
    + '<div class="grid2">'
    + chart('Win rate, agent against the random control',
        'Draws are not wins, so the control&rsquo;s own rate is the bar.',
        [{name:'agent',color:A,points:S.win},{name:'random',color:G,points:S.control}], {asPct:true})
    + chart('Rollout win rate',
        'Measured while exploring, and about eighteen points optimistic on this project.',
        [{name:'rollout win',color:B,points:S.rollout_win}], {asPct:true, emptyText:'not enough updates yet'})
    + '</div>';
  q('charts2').innerHTML =
      chart('Explained variance',
        'Share of the outcome the critic can predict. 0 is no better than guessing the average.',
        [{name:'explained var',color:D,points:S.explained_variance}], {zero:true, fill:true, emptyText:'not enough updates yet'})
    + '<div class="grid2">'
    + chart('Critic error and what it is fitting',
        'Value loss scales with the spread of returns, so the two only mean anything together.',
        [{name:'value loss',color:C,points:S.value_loss},{name:'spread',color:G,points:S.ret_std}],
        {emptyText:'not enough updates yet'})
    + chart('Entropy and pass rate',
        'Falling entropy is commitment. A pass rate climbing toward 1 is collapse into never playing.',
        [{name:'entropy',color:B,points:S.entropy},{name:'pass',color:G,points:S.noop}],
        {emptyText:'not enough updates yet'})
    + '</div>';
}

function ago(seconds){
  if (seconds < 90) return Math.round(seconds)+'s';
  if (seconds < 5400) return Math.round(seconds/60)+'m';
  return (seconds/3600).toFixed(1)+'h';
}

function renderJobs(jobs){
  var el = document.getElementById('jobs');
  if (!jobs.length) {
    el.innerHTML = '<div class="jobs idle"><div class="jobs-head">'
      + '<span class="lbl">running now</span></div>'
      + '<div style="color:var(--muted);font-size:14px">Nothing running.</div></div>';
    return;
  }
  // Two copies of the same kind of job is nearly always a mistake: an
  // interrupted command had already started before it was interrupted, and
  // three clones of one script once shared this machine holding 1.3GB each
  // with nothing on this page to say so.
  // Two ways the same job runs twice, and both have to be caught: as two
  // entries when the runs are named differently, and as one entry holding too
  // many processes when they are not named at all -- which is what three
  // simultaneous clones of one script actually looked like.
  var counts = {};
  jobs.forEach(function(j){ counts[j.kind] = (counts[j.kind] || 0) + 1; });
  var chips = jobs.map(function(j){
    var duplicated = j.suspicious || (counts[j.kind] > 1 && j.kind !== 'training');
    return '<span class="job'+(duplicated ? ' warn' : '')+'">'
      + '<b>'+j.kind+'</b>'
      + (j.name ? '<span>'+j.name+'</span>' : '')
      + '<span class="meta">pid '+j.pid
      + (j.processes > 1 ? ' &middot; '+j.processes+' proc' : '')
      + (j.memory_mb ? ' &middot; '+j.memory_mb+'MB' : '')
      + (j.age_seconds ? ' &middot; '+ago(j.age_seconds) : '')
      + '</span></span>';
  }).join('');
  var duplicates = Object.keys(counts).filter(function(k){
    return counts[k] > 1 && k !== 'training'; });
  jobs.forEach(function(j){
    if (j.suspicious && duplicates.indexOf(j.kind) < 0) duplicates.push(j.kind);
  });
  var note = duplicates.length
    ? '<div style="margin-top:9px;font-size:13px;color:var(--warn)">'
      + duplicates.length + ' job kind' + (duplicates.length > 1 ? 's are' : ' is')
      + ' running more than once (' + duplicates.join(', ')
      + '). That is usually an interrupted command that had already started.</div>'
    : '';
  el.innerHTML = '<div class="jobs"><div class="jobs-head">'
    + '<span class="dot"></span><span class="lbl" style="color:var(--ink)">running now &mdash; '
    + jobs.length + ' process' + (jobs.length > 1 ? 'es' : '') + '</span></div>'
    + '<div class="jobs-list">' + chips + '</div>' + note + '</div>';
}

function remember(key, value){ try{ localStorage.setItem(key, value); }catch(e){} }
function recall(key, fallback){
  try{ var v = localStorage.getItem(key); return v === null ? fallback : v; }catch(e){ return fallback; }
}

(function start(){
  var order = DATA.order, panesEl = document.getElementById('panes');
  document.getElementById('stamp').textContent='updated '+new Date().toLocaleTimeString();
  renderJobs(DATA.jobs || []);
  if (DATA.back) {
    var b=document.getElementById('back');
    b.href=DATA.back; b.style.display='inline';
    document.getElementById('eyebrow').style.display='none';
  }

  var split = recall('crsim-split','0') === '1' && order.length > 1;
  var picked = [
    recall('crsim-pane0', order[0]),
    recall('crsim-pane1', order[Math.min(1, order.length-1)])
  ].map(function(n){ return DATA.runs[n] ? n : order[0]; });

  function paintTabs(){
    document.getElementById('tabs').innerHTML = order.map(function(n){
      var r = DATA.runs[n], lift = r.summary.latest_lift;
      var val = (lift === null || lift === undefined) ? '' :
        '<span class="val">'+(lift>0?'+':'')+Number(lift).toFixed(2)+'</span>';
      var on = picked.slice(0, split ? 2 : 1).indexOf(n) >= 0;
      return '<button class="tab" role="tab" data-run="'+n+'" aria-selected="'+on+'">'
        + '<span class="pip'+(r.live?' on':'')+'"></span>'+n+val+'</button>';
    }).join('');
    Array.prototype.forEach.call(document.querySelectorAll('.tab'), function(t){
      t.addEventListener('click', function(){
        picked[0] = t.dataset.run; remember('crsim-pane0', picked[0]); draw();
      });
    });
  }

  function draw(){
    document.getElementById('title').textContent = split ? 'cr-sim training' : picked[0];
    panesEl.className = 'panes' + (split ? ' split' : '');
    panesEl.innerHTML = paneMarkup(0) + (split ? paneMarkup(1) : '');
    for (var i = 0; i < (split ? 2 : 1); i++) {
      (function(i){
        var pane = panesEl.querySelector('[data-pane="'+i+'"]');
        var pick = pane.querySelector('[data-role="pick"]');
        pick.innerHTML = order.map(function(n){
          return '<option value="'+n+'"'+(n===picked[i]?' selected':'')+'>'+n+'</option>';
        }).join('');
        pick.addEventListener('change', function(){
          picked[i] = pick.value; remember('crsim-pane'+i, pick.value);
          fillPane(pane, picked[i]); paintTabs();
        });
        pane.querySelector('.pane-head').style.display = split ? 'flex' : 'none';
        fillPane(pane, picked[i]);
      })(i);
    }
    paintTabs();
  }

  var toggle = document.getElementById('split');
  toggle.setAttribute('aria-pressed', String(split));
  if (order.length < 2) toggle.style.display = 'none';
  toggle.addEventListener('click', function(){
    split = !split; remember('crsim-split', split ? '1' : '0');
    toggle.setAttribute('aria-pressed', String(split));
    draw();
  });

  draw();

  try{
    var y=localStorage.getItem('crsim-scroll');
    if(y) window.scrollTo(0, parseInt(y,10));
    window.addEventListener('beforeunload', function(){
      try{ localStorage.setItem('crsim-scroll', String(window.scrollY)); }catch(e){}
    });
  }catch(e){}
})();
</script>
</body></html>
"""





#: A run whose metrics file has not been touched this recently is shown as
#: finished. Generous on purpose: an update at the slowest cadence measured
#: here takes a couple of minutes, and calling a live run dead is worse than
#: being slow to notice a finished one.
_LIVE_SECONDS = 300.0


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
        if not collected:
            return 0
        out.write_text(render_multi(collected, jobs=running_jobs()),
                       encoding="utf-8")
        return sum(len(rows) for _, rows, _ in collected)

    count = once()
    print(f"{len(discover())} run(s), {count} updates -> {out}", flush=True)
    if args.open:
        import webbrowser

        webbrowser.open(out.resolve().as_uri())
    if args.once:
        return 0

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
