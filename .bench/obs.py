"""Dump observations bit-exactly, from a chosen tree, so the two can be diffed."""
from __future__ import annotations
import argparse, hashlib, os, sys, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--tree", default=ROOT)
ap.add_argument("--out", required=True)
args = ap.parse_args()
sys.path.insert(0, args.tree)

spec = importlib.util.spec_from_file_location("bench", os.path.join(HERE, "bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

import numpy as np
from cr_sim.api.encoding import (
    build_encoding_config, encode_observation, legal_action_mask,
    total_tower_hitpoints,
)
from cr_sim.engine.entity import Team

w = bench.world()
registry = w[2]
blob = hashlib.sha256()
for upto in (60, 200, 400, 900, 1500, 2400):
    b = bench.busy_battle(w, upto)
    cfg = build_encoding_config(b.arena, bench.DECK_A, bench.DECK_B)
    for team in (Team.BLUE, Team.RED):
        obs = encode_observation(b, team, registry, cfg)
        blob.update(obs["grid"].tobytes())
        blob.update(obs["vector"].tobytes())
        blob.update(legal_action_mask(b, team, registry, cfg).tobytes())
        blob.update(repr(total_tower_hitpoints(b, team)).encode())

digest = blob.hexdigest()
with open(args.out, "w") as fh:
    fh.write(digest + "\n")
print(digest)
