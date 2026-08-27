import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util
spec = importlib.util.spec_from_file_location("bench", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

w = bench.world()
for upto in (200, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1400, 2000, 2160, 2200):
    b = bench.busy_battle(w, upto)
    print(upto, "alive", bench.alive(b), "entities", len(b.entities), "grave", len(b.graveyard))
