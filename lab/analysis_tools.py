"""Tool owned by the Analysis Agent: analyze_runs.

It executes NO candidate code. It reads run records written by the Experiment
Agent and computes everything deterministically, so the LLM only interprets:
  * Kendall tau-b (small vs full, medium vs full) per seed, mean + bootstrap CI
  * cascade REPLAY: because every candidate has scores at all three fidelities,
    any cascade (keep fractions) can be replayed with no new evaluation. A sweep
    over 9 cascades costs nothing.
  * top-1 / top-3 retention, regret, modelled cost speedup, and a random-subset
    baseline (a cascade must beat random pruning to be worth anything)
  * checks of the algorithm agent's machine-checkable hypotheses
  * a recommended policy per distribution, using PRE-REGISTERED thresholds
Speedups are modelled in cost units (items processed), not wall-clock time.
"""
import json
import math

import numpy as np

from lab.exp_tools import boot_ci, boot_ci_raw, load_run

TAU_SMALL_MIN = 0.70      # pre-registered with hypothesis H5
TAU_MEDIUM_MIN = 0.80
RETENTION_MIN = 0.90      # top-1 retention required of a recommended cascade
KEEP_SMALL = (0.25, 0.5, 0.75)
KEEP_MEDIUM = (0.2, 0.4, 0.6)   # fraction of the small-stage survivors
DEFAULT_CASCADE = (0.5, 0.4)


def kendall_tau(a, b):
    """Kendall tau-b (handles ties)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    iu = np.triu_indices(len(a), 1)
    x = np.sign(a[:, None] - a[None, :])[iu]
    y = np.sign(b[:, None] - b[None, :])[iu]
    d = math.sqrt(float((x != 0).sum()) * float((y != 0).sum()))
    return float((x * y).sum() / d) if d > 0 else float("nan")


def _r(x, k=3):
    return None if x is None or x != x else round(float(x), k)


def _replay(arr, ks, km, costs):
    n, n_seeds, _ = arr.shape
    k_s = max(1, math.ceil(ks * n))
    k_m = max(1, math.ceil(km * k_s))
    found, top3, regret = [], [], []
    for k in range(n_seeds):
        jit = np.random.default_rng(1000 + k).random(n) * 1e-12  # deterministic tie-break
        s, m, f = arr[:, k, 0], arr[:, k, 1], arr[:, k, 2]
        surv = np.lexsort((jit, s))[:k_s]
        fin = set(surv[np.lexsort((jit[surv], m[surv]))[:k_m]].tolist())
        reg = float(f[list(fin)].min() - f.min())
        regret.append(reg)
        found.append(reg <= 1e-12)
        top = np.argsort(f, kind="stable")[:3]
        top3.append(float(np.mean([t in fin for t in top])))
    cost_a = n * costs["full"]
    cost_b = n * costs["small"] + k_s * costs["medium"] + k_m * costs["full"]
    return {"keep_small": ks, "keep_medium": km, "n_full": k_m,
            "speedup": round(cost_a / cost_b, 2), "top1": round(float(np.mean(found)), 2),
            "top3": round(float(np.mean(top3)), 2), "regret": round(float(np.mean(regret)), 5),
            "random_top1": round(k_m / n, 2)}


def _analyze_run(run, t_small, t_med, ret_min):
    ids = list(run["scores"])
    arr = np.array([run["scores"][c] for c in ids], dtype=float)  # (cand, seed, fidelity)
    n, n_seeds, _ = arr.shape
    costs = run["config"]["cost_per_evaluation"]
    tau_s = [kendall_tau(arr[:, k, 0], arr[:, k, 2]) for k in range(n_seeds)]
    tau_m = [kendall_tau(arr[:, k, 1], arr[:, k, 2]) for k in range(n_seeds)]
    mean_full = arr[:, :, 2].mean(axis=1)
    best = int(np.argmin(mean_full))
    sweep = [_replay(arr, ks, km, costs) for ks in KEEP_SMALL for km in KEEP_MEDIUM]
    ok = [r for r in sweep if r["top1"] >= ret_min]
    rec = max(ok, key=lambda r: r["speedup"]) if ok else None
    ref = {c: _r(mean_full[ids.index(c)], 4) for c in ("first_fit", "best_fit") if c in ids}
    out = {"run_id": run["run_id"], "distribution": run["distribution"],
           "small_items": run["config"]["small_items"], "n_candidates": n, "n_seeds": n_seeds,
           "invalid_candidates": run.get("invalid", {}),
           "tau_small": {"mean": _r(np.nanmean(tau_s)), "ci95": boot_ci(tau_s),
                         "meets_min": bool(np.nanmean(tau_s) >= t_small)},
           "tau_medium": {"mean": _r(np.nanmean(tau_m)), "ci95": boot_ci(tau_m),
                          "meets_min": bool(np.nanmean(tau_m) >= t_med)},
           "best_full": {"id": ids[best], "mean_excess": _r(mean_full[best], 4)},
           "reference_full_excess": ref,
           "default_cascade": _replay(arr, *DEFAULT_CASCADE, costs),
           "sweep": sweep,
           "recommended_policy": ({"type": "cascade", **rec, "in_sample": True} if rec
                                  else {"type": "full_evaluation", "reason": f"no cascade reached top-1 retention {ret_min}"})}
    if run.get("heldout"):
        out["heldout"] = run["heldout"]["rows"]
    return out, arr, ids


def _check_hypotheses(hyps, per_run):
    checks = []
    for h in hyps:
        t = (h or {}).get("test") or {}
        cand, base, want = t.get("candidate_id"), t.get("baseline"), str(t.get("distribution", "any")).lower()
        better = str(t.get("direction", "better")).lower() != "worse"
        hit = False
        for run, arr, ids in per_run:
            if want not in ("any", run["distribution"]):
                continue
            hit = True
            if cand not in ids or base not in ids:
                checks.append({"hypothesis": h.get("id"), "run_id": run["run_id"],
                               "verdict": f"untestable: missing {cand if cand not in ids else base}"})
                continue
            d = arr[ids.index(cand), :, 2] - arr[ids.index(base), :, 2]  # lower excess is better
            ci = boot_ci(d)                      # rounded, for display only
            raw = boot_ci_raw(d)                 # unrounded, for the sign test
            v = "inconclusive"
            if not np.any(d):
                v = "no_difference (identical results)"
            elif raw:
                cand_better, cand_worse = raw[1] < 0, raw[0] > 0  # CI excludes 0
                if (better and cand_better) or (not better and cand_worse):
                    v = "supported"
                elif (better and cand_worse) or (not better and cand_better):
                    v = "refuted"
            checks.append({"hypothesis": h.get("id"), "run_id": run["run_id"],
                           "distribution": run["distribution"], "mean_diff_excess": _r(d.mean(), 4),
                           "ci95": ci, "verdict": v})
        if not hit:
            checks.append({"hypothesis": h.get("id"), "verdict": f"untestable: no run on {want}"})
    return checks


def analyze_runs(run_ids, hypotheses="", tau_small_min=TAU_SMALL_MIN,
                 tau_medium_min=TAU_MEDIUM_MIN, retention_min=RETENTION_MIN) -> dict:
    """Analyze instrumented runs. `run_ids`: list or JSON string or comma list.
    `hypotheses`: optional JSON list of {"id", "test": {"candidate_id",
    "baseline", "distribution", "direction"}} from the algorithm agent."""
    if isinstance(run_ids, str):
        try:
            run_ids = json.loads(run_ids)
        except ValueError:
            run_ids = [x.strip() for x in run_ids.split(",") if x.strip()]
    if isinstance(hypotheses, str):
        try:
            hypotheses = json.loads(hypotheses) if hypotheses.strip() else []
        except ValueError:
            hypotheses = []
    t_s, t_m, r_m = float(tau_small_min), float(tau_medium_min), float(retention_min)

    per_run, rows, missing = [], [], []
    for rid in run_ids:
        run = load_run(rid)
        if not run or run.get("kind") != "instrumented" or not run.get("scores"):
            missing.append(rid)
            continue
        out, arr, ids = _analyze_run(run, t_s, t_m, r_m)
        rows.append(out)
        per_run.append((run, arr, ids))
    if not rows:
        return {"error": "no usable runs", "missing_run_ids": missing}

    flags = []
    heldout_done = {o["distribution"] for o in rows if "heldout" in o}   # one held-out per distribution is enough
    for o in rows:
        tag = f"{o['distribution']} (small_items={o['small_items']}, {o['run_id']})"
        if not o["tau_small"]["meets_min"]:
            flags.append(f"{tag}: tau_small {o['tau_small']['mean']} below {t_s}")
        if not o["tau_medium"]["meets_min"]:
            flags.append(f"{tag}: tau_medium {o['tau_medium']['mean']} below {t_m}")
        if o["recommended_policy"]["type"] == "full_evaluation":
            flags.append(f"{tag}: no cascade met top-1 retention {r_m}; recommend full evaluation")
        if o["n_seeds"] < 5:
            flags.append(f"{tag}: only {o['n_seeds']} seeds, confidence intervals unreliable")
        if o["invalid_candidates"]:
            flags.append(f"{tag}: {len(o['invalid_candidates'])} invalid candidates excluded")
        if o["distribution"] not in heldout_done:
            flags.append(f"{tag}: no held-out run yet for {o['distribution']}")
    effect = {}
    for o in rows:
        effect.setdefault(o["distribution"], []).append(
            {"small_items": o["small_items"], "tau_small": o["tau_small"]["mean"], "run_id": o["run_id"]})
    effect = {d: sorted(v, key=lambda x: x["small_items"]) for d, v in effect.items() if len(v) > 1}

    return {"thresholds": {"tau_small_min": t_s, "tau_medium_min": t_m, "retention_min": r_m,
                           "preregistered": [t_s, t_m, r_m] == [TAU_SMALL_MIN, TAU_MEDIUM_MIN, RETENTION_MIN]},
            "runs": rows, "small_size_effect": effect,
            "hypothesis_checks": _check_hypotheses(hypotheses, per_run) if hypotheses else [],
            "flags": flags, "missing_run_ids": missing,
            "caveats": ["speedups are modelled in cost units (items processed), not wall-clock time",
                        "recommended cascades are chosen on the same seeds that scored them (in-sample); "
                        "validate on fresh seeds before trusting them"]}