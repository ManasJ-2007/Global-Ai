"""Tools owned by the Experiment Agent (they RUN code; they never interpret it).

run_instrumented : "Test A, instrumented". Scores every candidate (registry
                   candidates + a built-in baseline pool) on one distribution,
                   over several seeds, at ALL THREE fidelities (small, medium,
                   full), in parallel. Writes a run record. Because every
                   candidate gets a full-scale label, tau can be measured and
                   ANY cascade can later be replayed for free from the stored
                   scores (see analysis_tools.analyze_runs).
run_heldout      : scores the top candidates plus first-fit and best-fit on
                   fresh instances the search never saw.
estimate_cost    : deterministic cost estimate in cost units (items processed),
                   so the planner can respect its budget before anything runs.

Parallelism, two levels:
  * inside a call: (candidate x seed) jobs fan out over a process pool
  * across calls : the director may start one experiment-agent session per
                   distribution at the same time; run files never collide.
Set EXP_WORKERS=1 to force serial execution (e.g. if multiprocessing misbehaves).
"""
import concurrent.futures as cf
import functools
import json
import math
import multiprocessing as mp
import os
import re
import time
import uuid
import zlib
from datetime import datetime, timezone

import numpy as np

from lab.algo_tools import _SAFE_BUILTINS, _static_check, _load as _load_registry
from lab.campaign import current as _campaign

RUN_DIR = os.environ.get("RUN_DIR", "lab_data/runs")
CAPACITY = 100
DISTRIBUTIONS = ("weibull", "uniform", "bimodal", "heavy_tailed", "discrete")
INST_SMALL, INST_MED, INST_FULL, FULL_ITEMS = 4, 4, 6, 1000
JOB_TIMEOUT_S = 60.0       # per (candidate, seed) job
POOL_TIMEOUT_S = 600.0     # whole pool; on timeout we fall back to serial
# Process startup (spawn re-imports numpy) costs about as much as a small run, so
# only use the pool for big runs. 150 is a guess from a 1-core test box: tune it.
PARALLEL_MIN_JOBS = int(os.environ.get("EXP_PARALLEL_MIN_JOBS", "150"))
HELDOUT_SEED_START = 10_000  # disjoint from search seeds (0..19)

# Baselines: first/best/worst fit plus variants whose ORDERING genuinely differs
# (monotone transforms of best fit would be identical decisions, so none here).
BASELINES = {
    "first_fit": "def priority(item, bins):\n    return -np.arange(len(bins), dtype=float)\n",
    "best_fit": "def priority(item, bins):\n    return -(bins - item)\n",
    "worst_fit": "def priority(item, bins):\n    return bins - item\n",
    "ffbf_0.5": "def priority(item, bins):\n    return -(bins - item) - 0.5 * np.arange(len(bins))\n",
    "ffbf_2": "def priority(item, bins):\n    return -(bins - item) - 2.0 * np.arange(len(bins))\n",
    "ffbf_8": "def priority(item, bins):\n    return -(bins - item) - 8.0 * np.arange(len(bins))\n",
    "ef_wf": "def priority(item, bins):\n    return np.where(bins == item, 1e6, bins - item)\n",
    "ef_ff": "def priority(item, bins):\n    return np.where(bins == item, 1e6, -np.arange(len(bins), dtype=float))\n",
    "tbf_10": "def priority(item, bins):\n    near = (bins - item) <= 10\n    return np.where(near, 1000.0 - (bins - item), -np.arange(len(bins), dtype=float))\n",
    "tbf_25": "def priority(item, bins):\n    near = (bins - item) <= 25\n    return np.where(near, 1000.0 - (bins - item), -np.arange(len(bins), dtype=float))\n",
    "mid_20": "def priority(item, bins):\n    return -np.abs((bins - item) - 20.0)\n",
    "mid_40": "def priority(item, bins):\n    return -np.abs((bins - item) - 40.0)\n",
}


# ----------------------------------------------------------------- helpers
def _i(x, default, lo, hi):
    try:
        v = int(float(x))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def _b(x):
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y")
    return bool(x)


def _workers(w):
    env = os.environ.get("EXP_WORKERS")
    if env:
        return _i(env, 1, 1, 64)
    w = _i(w, 0, 0, 64)
    return w if w > 0 else max(1, min(8, os.cpu_count() or 1))


def _costs(small_items):
    med = max(200, 2 * small_items)
    return {"small": INST_SMALL * small_items, "medium": INST_MED * med,
            "full": INST_FULL * FULL_ITEMS}, med


def boot_ci_raw(vals, n_boot=2000, seed=0):
    """95% bootstrap CI of the mean over seeds, UNROUNDED. None if fewer than 3 values.
    Use this for sign tests ("does the CI exclude 0?"): a bound like -0.00003 rounds to
    -0.0, and -0.0 < 0 is False, which silently turned small real effects into "inconclusive"."""
    v = np.asarray([x for x in vals if x == x], dtype=float)  # drop NaN
    if len(v) < 3:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def boot_ci(vals, n_boot=2000, seed=0):
    """Same interval rounded to 4 decimals, for display only (never test its sign)."""
    ci = boot_ci_raw(vals, n_boot, seed)
    return None if ci is None else [round(ci[0], 4), round(ci[1], 4)]


def _safe_id(run_id):
    s = str(run_id).strip()
    return s if re.fullmatch(r"[A-Za-z0-9_.\-]+", s) else None


def run_path(run_id):
    return os.path.join(RUN_DIR, f"{run_id}.json")


def load_run(run_id):
    rid = _safe_id(run_id)
    if not rid:
        return None
    try:
        with open(run_path(rid), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save_run(run):
    os.makedirs(RUN_DIR, exist_ok=True)
    tmp = run_path(run["run_id"]) + f".{uuid.uuid4().hex[:6]}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(run, f)
    os.replace(tmp, run_path(run["run_id"]))  # atomic: parallel sessions never collide


def _pool(include_baselines):
    pool = dict(BASELINES) if include_baselines else {}
    for cid, v in _load_registry().items():
        pool[cid if cid not in pool else f"cand_{cid}"] = v["code"]
    return pool


# ------------------------------------------------------------- simulation
def _sample(dist, n, rng):
    if dist == "weibull":
        x = np.ceil(rng.weibull(3.0, n) * 45)
    elif dist == "uniform":
        x = rng.integers(10, 71, n)
    elif dist == "bimodal":
        low = rng.random(n) < 0.6
        x = np.where(low, rng.normal(25, 6, n), rng.normal(65, 8, n))
    elif dist == "heavy_tailed":
        x = (rng.pareto(1.5, n) + 1) * 8
    else:  # discrete
        x = rng.choice([10, 20, 30, 40, 50], n)
    return np.clip(np.rint(x), 1, CAPACITY).astype(np.int64)


@functools.lru_cache(maxsize=256)
def _instances(dist, seed, fidelity, small_items, namespace):
    costs, med = _costs(small_items)
    n_inst, n_items = {"small": (INST_SMALL, small_items), "medium": (INST_MED, med),
                       "full": (INST_FULL, FULL_ITEMS)}[fidelity]
    key = zlib.crc32(f"{namespace}|{dist}|{fidelity}".encode())
    rng = np.random.default_rng([int(seed), key])
    return tuple(_sample(dist, n_items, rng) for _ in range(n_inst))


def _pack(fn, items, deadline):
    """Online packing. Validity holds by construction: an item is only ever put
    into a bin that has room, or into a new bin."""
    rem = np.empty(len(items), dtype=np.float64)
    nb = 0
    for t, it in enumerate(items):
        it = int(it)
        idx = np.flatnonzero(rem[:nb] >= it) if nb else np.empty(0, dtype=np.int64)
        if idx.size:
            scores = np.asarray(fn(it, rem[idx]), dtype=float)
            if scores.shape != idx.shape or not np.all(np.isfinite(scores)):
                raise ValueError("candidate returned bad scores")
            rem[idx[int(np.argmax(scores))]] -= it
        else:
            rem[nb] = CAPACITY - it
            nb += 1
        if t % 64 == 0 and time.perf_counter() > deadline:
            raise TimeoutError("candidate too slow")
    return nb


def _excess(fn, items, deadline):
    nb = _pack(fn, items, deadline)
    lb = max(1, math.ceil(int(items.sum()) / CAPACITY))
    return (nb - lb) / lb


def _job(args):
    """One (candidate, seed) job. Top-level so it can be pickled for workers."""
    cid, code, dist, seed, small_items, namespace, fids = args
    try:
        why = _static_check(code)
        if why:
            return {"id": cid, "seed": seed, "status": f"invalid: {why}"}
        ns = {"np": np, "math": math, "__builtins__": _SAFE_BUILTINS}
        exec(compile(code, "<candidate>", "exec"), ns)
        fn = ns["priority"]
        deadline = time.perf_counter() + JOB_TIMEOUT_S
        out = {"id": cid, "seed": seed, "status": "ok"}
        for f in fids:
            vals = [_excess(fn, inst, deadline) for inst in _instances(dist, seed, f, small_items, namespace)]
            out[f] = float(np.mean(vals))
        return out
    except Exception as e:
        return {"id": cid, "seed": seed, "status": f"invalid: {type(e).__name__}: {e}"[:140]}


def _run_jobs(jobs, workers):
    """Process pool with a serial fallback. Returns (results, mode)."""
    if workers > 1 and len(jobs) >= PARALLEL_MIN_JOBS:
        try:
            with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
                return list(ex.map(_job, jobs, chunksize=4, timeout=POOL_TIMEOUT_S)), f"process x{workers}"
        except Exception as e:  # pool unavailable or broken: do not fail the experiment
            return [_job(j) for j in jobs], f"serial (process pool failed: {type(e).__name__})"
    return [_job(j) for j in jobs], "serial"


# ------------------------------------------------------------------ tools
def estimate_cost(n_distributions=3, n_seeds=6, small_items=50, include_baselines=True) -> dict:
    """Estimate the cost (cost units = items processed) of instrumented runs
    BEFORE running them, for the planner's budget check. Also returns the
    cost of evaluating everything at full fidelity only, for comparison."""
    nd = _i(n_distributions, 3, 1, len(DISTRIBUTIONS))
    ns = _i(n_seeds, 6, 1, 20)
    si = _i(small_items, 50, 20, 400)
    n_cand = len(_pool(_b(include_baselines)))
    costs, med = _costs(si)
    per_cand_seed = sum(costs.values())
    return {"n_candidates": n_cand, "cost_per_evaluation": costs, "medium_items": med,
            "instrumented_cost_units": nd * ns * n_cand * per_cand_seed,
            "full_only_cost_units": nd * ns * n_cand * costs["full"],
            "note": "instrumented = all 3 fidelities per candidate (needed to measure tau)"}


def run_instrumented(distribution, n_seeds=6, small_items=50, seed_start=0,
                     include_baselines=True, max_cost_units=0, workers=0) -> dict:
    """Test A, instrumented, on ONE distribution. Returns a compact run summary
    with a run_id. Refuses to start if the estimate exceeds max_cost_units (>0)."""
    dist = str(distribution).strip().lower()
    if dist not in DISTRIBUTIONS:
        return {"error": f"distribution must be one of {list(DISTRIBUTIONS)}"}
    n_seeds = _i(n_seeds, 6, 1, 20)
    small_items = _i(small_items, 50, 20, 400)
    seed_start = _i(seed_start, 0, 0, 9_000)
    max_cost = _i(max_cost_units, 0, 0, 10**12)
    pool = _pool(_b(include_baselines))
    if not pool:
        return {"error": "no candidates: registry is empty and baselines are off"}
    costs, med = _costs(small_items)
    est = len(pool) * n_seeds * sum(costs.values())
    if max_cost and est > max_cost:
        return {"error": "over budget", "estimated_cost_units": est, "max_cost_units": max_cost}

    seeds = list(range(seed_start, seed_start + n_seeds))
    jobs = [(cid, code, dist, s, small_items, "search", ("small", "medium", "full"))
            for s in seeds for cid, code in pool.items()]
    w = _workers(workers)
    t0 = time.perf_counter()
    results, mode = _run_jobs(jobs, w)
    wall = time.perf_counter() - t0

    bad = {}
    for r in results:
        if r["status"] != "ok":
            bad.setdefault(r["id"], r["status"])
    scores = {cid: [[r["small"], r["medium"], r["full"]]
                    for s in seeds for r in results if r["id"] == cid and r["seed"] == s]
              for cid in pool if cid not in bad}
    run = {"run_id": f"{dist}-s{small_items}-{uuid.uuid4().hex[:6]}", "kind": "instrumented",
           "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "campaign_id": _campaign()["campaign_id"],   # only runs of the current campaign count as spent
           "distribution": dist,
           "config": {"seeds": seeds, "small_items": small_items, "medium_items": med,
                      "full_items": FULL_ITEMS, "instances": {"small": INST_SMALL, "medium": INST_MED, "full": INST_FULL},
                      "cost_per_evaluation": costs, "include_baselines": _b(include_baselines)},
           "candidates": list(scores), "invalid": bad, "scores": scores,
           "execution": {"mode": mode, "workers": w, "wall_seconds": round(wall, 2)}}
    save_run(run)
    return {"run_id": run["run_id"], "distribution": dist, "n_candidates": len(scores),
            "n_seeds": n_seeds, "small_items": small_items, "invalid": bad,
            "cost_units_spent": len(scores) * n_seeds * sum(costs.values()),
            "mode": mode, "wall_seconds": round(wall, 2)}


def run_heldout(run_id, top_k=3, n_seeds=10, workers=0) -> dict:
    """Score the top_k candidates of a run (ranked by full-fidelity search score)
    plus first_fit and best_fit on held-out instances the search never saw."""
    run = load_run(run_id)
    if not run or run.get("kind") != "instrumented":
        return {"error": f"unknown run_id {run_id}"}
    top_k = _i(top_k, 3, 1, 10)
    n_seeds = _i(n_seeds, 10, 3, 30)
    ids = list(run["scores"])
    ranked = sorted(ids, key=lambda c: float(np.mean([row[2] for row in run["scores"][c]])))
    pick = list(dict.fromkeys(ranked[:top_k] + ["first_fit", "best_fit"]))
    pool = _pool(True)
    pick = [c for c in pick if c in pool]
    seeds = list(range(HELDOUT_SEED_START, HELDOUT_SEED_START + n_seeds))
    jobs = [(c, pool[c], run["distribution"], s, run["config"]["small_items"], "heldout", ("full",))
            for s in seeds for c in pick]
    results, mode = _run_jobs(jobs, _workers(workers))
    by = {c: np.array([r["full"] for s in seeds for r in results
                       if r["id"] == c and r["seed"] == s and r["status"] == "ok"]) for c in pick}

    def verdict(c, ref):
        if ref not in by or len(by[c]) != len(by[ref]) or len(by[c]) < 3:
            return None
        d = by[c] - by[ref]
        raw = boot_ci_raw(d)                     # verdict from the unrounded bounds
        v = "no_clear_difference"
        if raw and raw[1] < 0:
            v = "better"
        elif raw and raw[0] > 0:
            v = "worse"
        return {"mean_diff": round(float(d.mean()), 4), "ci95": boot_ci(d), "verdict": v}

    rows = [{"id": c, "mean_excess": round(float(by[c].mean()), 4) if len(by[c]) else None,
             "vs_best_fit": verdict(c, "best_fit") if c != "best_fit" else None,
             "vs_first_fit": verdict(c, "first_fit") if c != "first_fit" else None} for c in pick]
    run["heldout"] = {"seeds": seeds, "rows": rows, "mode": mode}
    save_run(run)
    return {"run_id": run["run_id"], "heldout_seeds": f"{seeds[0]}..{seeds[-1]}", "rows": rows}