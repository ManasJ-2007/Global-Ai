"""Compute the lab's measured speed/compute numbers. Run from the project root.
  python lab_speed_numbers.py
Prints: (1) stage timeline from results/research_log.jsonl, (2) compute (cost units)
for replay-vs-live cascade sweeps from the stored run configs."""
import glob, json, math, os
from datetime import datetime

# (1) wall-clock from the shared research record
rows = []
if os.path.exists("results/research_log.jsonl"):
    for ln in open("results/research_log.jsonl", encoding="utf-8"):
        try: rows.append(json.loads(ln))
        except ValueError: pass
ts = [(datetime.fromisoformat(r["t"]), r.get("agent", "?"), r.get("decision", "")[:60]) for r in rows if r.get("t")]
if ts:
    ts.sort()
    print(f"log entries: {len(ts)}  span: {ts[-1][0]-ts[0][0]}  ({ts[0][0]} -> {ts[-1][0]})")
    for (a, ag, d), (b, _, _) in zip(ts, ts[1:] + [ts[-1]]):
        print(f"  {a:%H:%M:%S}  +{(b-a).total_seconds():6.0f}s  {ag:12s} {d}")

# (2) compute: replay vs running each cascade live (same formulas as analysis_tools._replay)
KS, KM = (0.25, 0.5, 0.75), (0.2, 0.4, 0.6)
for p in sorted(glob.glob("lab_data/runs/*.json")):
    r = json.load(open(p, encoding="utf-8"))
    if r.get("kind") != "instrumented" or not r.get("scores"): continue
    c = r["config"]["cost_per_evaluation"]; n = len(r["scores"]); seeds = len(next(iter(r["scores"].values())))
    inst = n * (c["small"] + c["medium"] + c["full"])
    live = sum(n*c["small"] + math.ceil(ks*n)*c["medium"] + max(1, math.ceil(km*math.ceil(ks*n)))*c["full"]
               for ks in KS for km in KM)
    full_only = n * c["full"]
    print(f"{r['run_id']}: per seed  instrumented={inst}  9 live cascades={live}  "
          f"-> replay saves {live/inst:.2f}x;  full-only={full_only}  (x{seeds} seeds)")
