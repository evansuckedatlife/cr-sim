"""Emit a per-tick state-hash trace for a scripted match, from a chosen tree.

Run against the pristine tree and against the optimised one; the two files must
be byte-identical. This is the correctness gate that every change here has to
pass before its speed is even interesting.
"""
from __future__ import annotations
import argparse, hashlib, os, sys, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--tree", default=ROOT)
ap.add_argument("--out", required=True)
ap.add_argument("--ticks", type=int, default=2400)
args = ap.parse_args()
sys.path.insert(0, args.tree)

spec = importlib.util.spec_from_file_location("bench", os.path.join(HERE, "bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

from cr_sim.engine.entity import Team
from cr_sim.engine.fixed import tiles
from cr_sim.engine.lookahead import committed_value, project
from cr_sim.replay import state_hash

lines = []
for seed in (1, 7, 99):
    for tps in (20, 60):
        w = bench.world()
        data, levels, registry = w
        from cr_sim.engine.battle import Battle, BattleConfig
        b = Battle(data, levels, registry, BattleConfig(
            seed=seed, ticks_per_second=tps,
            blue_deck=bench.DECK_A, red_deck=bench.DECK_B))
        plays = {}
        for tick, team, card, x, y in bench.SCRIPT:
            plays.setdefault(tick, []).append((team, card, x, y))
        for t in range(args.ticks):
            for team, card, x, y in plays.get(t % bench.PERIOD, ()):
                b.play_card(Team.BLUE if team == 0 else Team.RED, card, tiles(x), tiles(y))
            b.step()
            lines.append(f"{seed} {tps} {t} {state_hash(b.tick, b.entities)}")
            # Branch off mid-match and throw the branch away. Asking what
            # happens next must not change what happens next, so the rest of
            # this trace is itself the check on everything Battle.clone shares
            # with its branches.
            if t % 37 == 0:
                p = project(b, 60)
                lines.append(
                    f"{seed} {tps} P {t} {p.blue_crowns} {p.red_crowns} "
                    f"{p.blue_tower_hitpoints} {p.red_tower_hitpoints} {p.ticks} "
                    f"{p.decided} {committed_value(b, Team.BLUE, 60):.9f}"
                )
        # positions too, so a change that cancels out in the hash still shows
        for e in sorted(b.entities, key=lambda e: e.id):
            lines.append(f"{seed} {tps} E {e.id} {e.x} {e.y} {e.hitpoints} {e.target_id}")

blob = "\n".join(lines)
with open(args.out, "w") as fh:
    fh.write(blob)
print(hashlib.sha256(blob.encode()).hexdigest(), len(lines))
