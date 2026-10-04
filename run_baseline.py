"""Offline run of the full experiment, no LLM needed. Use it to sanity-check the engine
and to get baseline numbers before wiring up the agents:  python run_baseline.py"""
import json, sys, time
from lab import tools

def main(pool_size=40, seeds=(0, 1, 2)):
    rows = []
    for dist in tools.DISTRIBUTIONS:
        for seed in seeds:
            t0 = time.time()
            r = tools.compare_tests(dist, pool_size, seed)
            r["seed"] = seed
            r["wall_s"] = round(time.time() - t0, 1)
            rows.append(r)
            print(f"{dist:8s} seed={seed} tau_small={r['tau_small_vs_full']:.2f} "
                  f"tau_med={r['tau_medium_vs_full']:.2f} speedup={r['speedup_cost']:.2f}x "
                  f"regret_B={r['regret_B']:.4f} found_best={r['found_true_best']} ({r['wall_s']}s)")
            sys.stdout.flush()
    json.dump(rows, open("results/baseline_summary.json", "w"), indent=2)

if __name__ == "__main__":
    main()
