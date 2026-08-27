"""Interleaved A/B benchmark runner.

Runs bench.py against two trees alternately (A/B/A/B/...) in fresh subprocesses
and reports min-of-rounds per benchmark plus the B/A ratio. Interleaving is what
makes the comparison survive a machine that is running other jobs: both trees
see the same competing load, and the minimum picks the least-disturbed round.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(HERE, "bench.py")


def run(tree, only):
    cmd = [sys.executable, BENCH, "--tree", tree]
    if only:
        cmd += ["--only", only]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if out.returncode != 0:
        sys.stderr.write(out.stdout + out.stderr)
        raise SystemExit(f"bench failed for {tree}")
    return json.loads(out.stdout.strip().splitlines()[-1])["res"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline tree")
    ap.add_argument("--b", default=ROOT, help="candidate tree")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    acc = {"A": {}, "B": {}}
    for r in range(args.rounds):
        for label, tree in (("A", args.a), ("B", args.b)):
            res = run(tree, args.only)
            for k, v in res.items():
                cur = acc[label].get(k)
                if cur is None or v < cur:
                    acc[label][k] = v
        print(f"  round {r+1}/{args.rounds} done", file=sys.stderr)

    names = list(acc["A"])
    w = max(len(n) for n in names)
    print(f"{'bench'.ljust(w)}   {'A (base)':>12}   {'B (new)':>12}   {'speedup':>8}")
    for n in names:
        a, b = acc["A"][n], acc["B"][n]
        print(f"{n.ljust(w)}   {a*1e6:11.1f}u   {b*1e6:11.1f}u   {a/b:7.3f}x")


if __name__ == "__main__":
    main()
