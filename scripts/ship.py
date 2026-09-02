"""Gate a checkpoint: measured against a baseline on one shared control, or not.

    python scripts/ship.py runs/my-run/best.pt --baseline runs/clone-v1-paired/cloned.pt

Exit 0 means SHIP, 2 means DON'T SHIP, 3 means the inputs were bad. The verdict
is a single, stated rule -- the candidate's greedy interval must not sit wholly
below the baseline's -- and both sides of it are printed, so the number that
decided it is on screen rather than in a log. Every reading is written to
--out and carries the control it faced, through the same evaluate_checkpoints
protocol every other verdict here uses; a gate that measured differently from
the thing it gates would be a second scale.

A ship verdict is not "better". Greedy here is deterministic and the two
intervals overlapping is the honest reading of most real comparisons; DON'T
SHIP is reserved for a demonstrated regression, which is the only thing a gate
can know.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATE = ROOT / "scripts" / "evaluate_checkpoints.py"


def _rows(candidate: Path, baseline: Path, episodes: int, tower_level: int,
          out: Path) -> list[dict]:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(EVALUATE), str(candidate), str(baseline),
           "--episodes", str(episodes), "--tower-level", str(tower_level),
           "--out", str(out)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-2000:])
        raise SystemExit(3)
    return json.loads(out.read_text(encoding="utf-8"))


def verdict(rows: list[dict], candidate: Path, baseline: Path) -> tuple[bool, str]:
    """SHIP unless the candidate's greedy interval is wholly below the baseline's."""
    def pick(path: Path, mode: str) -> dict:
        key = str(path)
        for r in rows:
            if r["mode"] == mode and str(r["checkpoint"]) == key:
                return r
        raise SystemExit(3)
    c, b = pick(candidate, "greedy"), pick(baseline, "greedy")
    regression = c["ci_high"] < b["ci_low"]
    why = (f"candidate greedy {c['lift']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] "
           f"vs baseline {b['lift']:+.3f} [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}] "
           f"against {c.get('eval_opponent', '?')}")
    return (not regression), why


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ship")
    p.add_argument("candidate", type=Path)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--tower-level", type=int, default=5)
    p.add_argument("--out", type=Path, default=None,
                   help="where the readings go; default runs/_ship/<stamp>.json")
    args = p.parse_args(argv)
    for path in (args.candidate, args.baseline):
        if not path.is_file():
            sys.stderr.write(f"no such checkpoint: {path}\n")
            return 3
    out = args.out or ROOT / "runs" / "_ship" / f"{time.strftime('%Y%m%d-%H%M%S')}_verdict.json"
    rows = _rows(args.candidate, args.baseline, args.episodes, args.tower_level, out)
    ship, why = verdict(rows, args.candidate, args.baseline)
    print(why)
    for r in rows:
        print(f"  {Path(r['checkpoint']).parent.name + '/' + Path(r['checkpoint']).name:40} "
              f"{r['mode']:8} win {r['win']:.0%} loss {r['loss']:.0%}  "
              f"{r['lift']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")
    print(f"readings: {out}")
    print("SHIP" if ship else "DON'T SHIP: candidate is a demonstrated regression")
    return 0 if ship else 2


if __name__ == "__main__":
    sys.exit(main())
