"""Tool owned by the Critic Agent: critique_runs.

The critic's job is to find what would make the lab's conclusions wrong. This
tool executes NO candidate code. It re-derives everything from the Experiment
Agent's run records (independently of the Analysis Agent) and returns ranked,
machine-readable findings. Each finding may carry an `action`; the Planner's
plan_next tool consumes those actions through the persisted critique_id, so
nothing is retyped by an LLM in between.

Checks (all deterministic):
  duplicate_candidates       candidates with identical scores everywhere (a
                             hypothesis on them is vacuous)
  baseline_inflation         tau with baselines vs agent-generated candidates only
  screening diagnosis        WHY tau_small is low: noisy full ranking, noisy small
                             scores (pooling seeds fixes it) or biased small
                             instances (pooling does not)
  low_power                  too few seeds / wide tau confidence interval
  cascade_oos                cascade chosen on half the seeds, scored on the
                             other half (the replay in analysis is in-sample)
  confounded_size_comparison runs at different small sizes that do not share
                             seeds/candidates; otherwise a paired like-for-like
                             delta-tau with a bootstrap CI
  held-out integrity         seed disjointness, power, claims whose paired CI
                             includes 0
  invalid_candidates, untested_distribution, multiple_comparisons

Thresholds are the pre-registered ones from analysis_tools and are deliberately
NOT parameters: nobody can move them from here.
"""
import json
import math
import os
import uuid
from datetime import datetime, timezone

import numpy as np

from lab.analysis_tools import (KEEP_MEDIUM, KEEP_SMALL, RETENTION_MIN, TAU_MEDIUM_MIN,
                                TAU_SMALL_MIN, _replay, kendall_tau)
from lab.exp_tools import BASELINES, DISTRIBUTIONS, HELDOUT_SEED_START, _safe_id, boot_ci, boot_ci_raw, load_run

CRIT_DIR = os.environ.get("CRITIQUE_DIR", "lab_data/critiques")
SIZE_LADDER = (50, 150, 400)    # small_items values worth trying (run_instrumented caps at 400)
POWER_MIN_SEEDS = 8
MAX_SEEDS = 20                  # run_instrumented cap
HELDOUT_MAX_SEEDS = 30          # run_heldout cap
CI_WIDTH_MAX, CI_WIDTH_TARGET = 0.25, 0.15
INFLATION_GAP = 0.10
SEV_ORDER = {"high": 0, "medium": 1, "low": 2}


# ----------------------------------------------------------------- helpers
def _r(x, k=3):
    return None if x is None or x != x else round(float(x), k)


def _m(v):
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def _tau_by_seed(arr, f, idx=slice(None)):
    """Per-seed Kendall tau between fidelity f and full, optionally on a candidate subset."""
    return [kendall_tau(arr[idx, k, f], arr[idx, k, 2]) for k in range(arr.shape[1])]


def _ceiling(arr):
    """Noise ceiling: mean tau between full-scale rankings from different seeds. If two
    independent measurements of the SAME thing agree only this well, no cheaper proxy can
    be expected to beat it per seed."""
    ns = arr.shape[1]
    return _m([kendall_tau(arr[:, i, 2], arr[:, j, 2]) for i in range(ns) for j in range(i + 1, ns)])


def _dup_groups(ids, arr):
    groups = {}
    for i, c in enumerate(ids):
        groups.setdefault((np.round(arr[i], 12) + 0.0).tobytes(), []).append(c)  # +0.0 folds -0.0
    return [g for g in groups.values() if len(g) > 1]


def _cascade_cv(arr, costs, n_splits=12):
    """Choose the cascade on a random half of the seeds (same rule as analysis), score it on
    the other half. Cheap: replays only, no evaluation."""
    ns = arr.shape[1]
    if ns < 4:
        return None
    rng, h = np.random.default_rng(11), ns // 2
    got = []
    for _ in range(n_splits):
        p = rng.permutation(ns)
        a, b = np.sort(p[:h]), np.sort(p[h:2 * h])
        ok = [r for r in (_replay(arr[:, a, :], ks, km, costs) for ks in KEEP_SMALL for km in KEEP_MEDIUM)
              if r["top1"] >= RETENTION_MIN]
        if ok:
            best = max(ok, key=lambda r: r["speedup"])
            oos = _replay(arr[:, b, :], best["keep_small"], best["keep_medium"], costs)
            got.append((best["speedup"], oos["top1"], oos["random_top1"]))
    out = {"splits": n_splits, "half_seeds": h, "cascade_found_in_half": len(got)}
    if got:
        g = np.array(got)
        out.update(speedup_median=_r(np.median(g[:, 0]), 2), top1_oos_mean=_r(g[:, 1].mean(), 2),
                   meets_retention_frac=_r((g[:, 1] >= RETENTION_MIN).mean(), 2),
                   random_top1=_r(g[:, 2].mean(), 2))
    return out


def _diagnose(tau, pooled, ceil):
    if tau != tau:
        return "undetermined"
    if tau >= TAU_SMALL_MIN:
        return "meets_threshold"
    if ceil == ceil and ceil < TAU_SMALL_MIN:
        return "full_ranking_unstable"
    return "noise_limited" if pooled == pooled and pooled >= TAU_SMALL_MIN else "bias_limited"


def _next_size(s):
    return next((x for x in SIZE_LADDER if x > s), None)


def _seeds_needed(ns, width):
    if not width:
        return POWER_MIN_SEEDS
    return min(MAX_SEEDS, max(POWER_MIN_SEEDS, math.ceil(ns * (width / CI_WIDTH_TARGET) ** 2)))


def _log(decision, evidence):
    try:  # the tool writes its own audit entry, so no extra agent call is needed
        from lab.tools import log_decision
        log_decision("critic_agent", decision, json.dumps(evidence))
    except Exception:  # noqa: BLE001  logging must never break the critique
        pass


def load_critique(critique_id):
    cid = _safe_id(critique_id)
    if not cid:
        return None
    try:
        with open(os.path.join(CRIT_DIR, f"{cid}.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _save(c):
    os.makedirs(CRIT_DIR, exist_ok=True)
    path = os.path.join(CRIT_DIR, f"{c['critique_id']}.json")
    tmp = f"{path}.{uuid.uuid4().hex[:6]}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(c, f)
    os.replace(tmp, path)


# -------------------------------------------------------------- per-run review
def _review_run(run, F):
    ids = list(run["scores"])
    arr = np.array([run["scores"][c] for c in ids], dtype=float)  # (candidate, seed, fidelity)
    n, ns, _ = arr.shape
    cfg = run["config"]
    rid, dist, si, seeds = run["run_id"], run["distribution"], cfg["small_items"], cfg["seeds"]
    ts, tm = _tau_by_seed(arr, 0), _tau_by_seed(arr, 1)
    tau_s, tau_m = _m(ts), _m(tm)
    ci = boot_ci(ts)
    pooled = kendall_tau(arr[:, :, 0].mean(1), arr[:, :, 2].mean(1))
    ceil = _ceiling(arr)
    agent = np.array([c not in BASELINES for c in ids])
    n_agent = int(agent.sum())
    tau_agent = _m(_tau_by_seed(arr, 0, agent)) if n_agent >= 5 else None
    groups = _dup_groups(ids, arr)
    cv = _cascade_cv(arr, cfg["cost_per_evaluation"])
    diag = _diagnose(tau_s, pooled, ceil)
    base = {"distribution": dist, "seed_start": min(seeds)}

    # 1. duplicates
    if groups:
        replace = []
        for g in groups:
            agents = [c for c in g if c not in BASELINES]
            replace += agents if len(agents) < len(g) else agents[1:]
        ev = "; ".join("=".join(g) for g in groups[:4]) + " score identically at every seed and fidelity"
        F("high" if replace else "low", "duplicate_candidates", [rid], ev,
          {"type": "replace_candidates", "candidate_ids": replace, "why": ev} if replace else None)

    # 2. pool composition
    if tau_agent is not None and tau_s == tau_s and tau_s - tau_agent >= INFLATION_GAP:
        F("medium", "baseline_inflation", [rid],
          f"tau_small {_r(tau_s)} with baselines vs {_r(tau_agent)} on the {n_agent} agent candidates alone")
    elif 0 < n_agent < 5:
        F("low", "thin_agent_pool", [rid], f"only {n_agent} agent candidates; tau mostly reflects baselines")

    # 3. why tau_small is low
    if diag == "full_ranking_unstable":
        F("high", "unstable_full_ranking", [rid],
          f"tau_small {_r(tau_s)}; two single-seed full rankings agree only at tau {_r(ceil)} (< {TAU_SMALL_MIN}), "
          "so the target itself is noisy and a low tau_small cannot be blamed on instance size",
          {"type": "more_seeds", "n_seeds": min(MAX_SEEDS, max(2 * ns, POWER_MIN_SEEDS)), "small_items": si, **base}
          if ns < MAX_SEEDS else None)
    elif diag == "noise_limited":
        F("medium", "noisy_small_scores", [rid],
          f"tau_small {_r(tau_s)} per seed but {_r(pooled)} pooled over {ns} seeds (full ceiling {_r(ceil)}): "
          "small scores are right on average but noisy per seed; more small instances help, larger ones may not",
          {"type": "tool_change", "note": "score more small instances per candidate (INST_SMALL is a constant in exp_tools)"})
    elif diag == "bias_limited":
        nxt = _next_size(si)
        F("high", "biased_small_instances", [rid],
          f"tau_small {_r(tau_s)} per seed and {_r(pooled)} pooled (full ceiling {_r(ceil)}): small instances rank "
          "candidates differently from full scale, not merely noisily"
          + ("" if nxt else f"; no size above {si} left in the ladder"),
          {"type": "larger_small_items", "small_items": nxt, "n_seeds": max(ns, POWER_MIN_SEEDS), **base} if nxt else None)
    if tau_m == tau_m and tau_m < TAU_MEDIUM_MIN and ceil == ceil and ceil < TAU_MEDIUM_MIN:
        F("low", "threshold_above_noise_ceiling", [rid],
          f"tau_medium_min {TAU_MEDIUM_MIN} exceeds the full-ranking noise ceiling {_r(ceil)}: report as a limitation, do not move the pre-registered bar")

    # 4. power
    w = ci[1] - ci[0] if ci else None
    if ns < POWER_MIN_SEEDS or (w is not None and w > CI_WIDTH_MAX):
        need = _seeds_needed(ns, w)
        F("medium", "low_power", [rid], f"{ns} seeds; tau_small 95% CI {ci}",
          {"type": "more_seeds", "n_seeds": need, "small_items": si, **base} if need > ns else None)

    # 5. cascade out of sample
    if cv and "top1_oos_mean" in cv:
        rnd = cv["random_top1"]
        if cv["top1_oos_mean"] <= rnd + 0.10:
            F("high", "cascade_oos", [rid],
              f"cascade picked on half the seeds keeps the true best on the other half only {cv['top1_oos_mean']} of the time; random pruning of the same size gets {rnd}")
        elif cv["meets_retention_frac"] < 0.8:
            F("medium", "cascade_oos", [rid],
              f"a cascade meeting retention {RETENTION_MIN} on half the seeds meets it on the other half in only {cv['meets_retention_frac']} of {cv['cascade_found_in_half']} splits (random {rnd})")

    # 6. invalid candidates
    inv = run.get("invalid") or {}
    if inv:
        first = next(iter(inv.values()))
        F("medium" if len(inv) / (len(inv) + n) > 0.2 else "low", "invalid_candidates", [rid],
          f"{len(inv)} of {len(inv) + n} excluded (e.g. {first}); timeouts silently favour fast heuristics")

    # 7. held-out integrity and significance
    ho = run.get("heldout")
    if ho:
        hs = ho["seeds"]
        if not max(seeds) < HELDOUT_SEED_START <= min(hs):
            F("high", "heldout_overlap", [rid], f"search seeds up to {max(seeds)} vs held-out seeds from {min(hs)}")
        weak = [f"{row['id']} vs {ref}: diff {v['mean_diff']}, CI {v['ci95']}"
                for row in ho["rows"] for ref, key in (("best_fit", "vs_best_fit"), ("first_fit", "vs_first_fit"))
                for v in [row.get(key)] if v and v["mean_diff"] < 0 and v["verdict"] != "better"]
        if weak:
            F("medium", "heldout_not_significant", [rid],
              f"lower mean excess but the paired CI includes 0 ({len(hs)} seeds): " + "; ".join(weak[:4]),
              {"type": "heldout_more_seeds", "run_id": rid, "n_seeds": HELDOUT_MAX_SEEDS, "top_k": 3} if len(hs) < HELDOUT_MAX_SEEDS else None)

    summary = {"run_id": rid, "distribution": dist, "small_items": si, "n_seeds": ns, "n_candidates": n,
               "n_agent_candidates": n_agent, "has_heldout": bool(ho),
               "tau_small": {"mean": _r(tau_s), "ci95": ci, "agent_only": _r(tau_agent), "pooled_over_seeds": _r(pooled)},
               "tau_medium_mean": _r(tau_m), "full_ranking_ceiling": _r(ceil), "diagnosis": diag,
               "cascade_oos": cv, "duplicate_groups": groups}
    return summary, arr, ids, groups


# ------------------------------------------------------- cross-run reviews
def _size_pairs(entries, F):
    """Compare runs of one distribution at different small sizes, like for like."""
    by = {}
    for e in entries:
        by.setdefault(e[0]["distribution"], []).append(e)
    out = []
    for dist, es in by.items():
        es.sort(key=lambda e: e[0]["config"]["small_items"])
        for (ra, aa, ia, _), (rb, ab, ib, _) in zip(es, es[1:]):
            ca_, cb_ = ra["config"], rb["config"]
            if ca_["small_items"] == cb_["small_items"]:
                continue
            sa, sb = ca_["seeds"], cb_["seeds"]
            shared = [s for s in sa if s in set(sb)]
            common = [c for c in ia if c in set(ib)]
            row = {"distribution": dist, "runs": [ra["run_id"], rb["run_id"]],
                   "small_items": [ca_["small_items"], cb_["small_items"]],
                   "shared_seeds": len(shared), "common_candidates": len(common),
                   "n_candidates": [len(ia), len(ib)],
                   "baselines": [ca_.get("include_baselines"), cb_.get("include_baselines")]}
            if len(shared) >= 3 and len(common) >= 6:
                xa, xb = [ia.index(c) for c in common], [ib.index(c) for c in common]
                da = [kendall_tau(aa[xa, sa.index(s), 0], aa[xa, sa.index(s), 2]) for s in shared]
                db = [kendall_tau(ab[xb, sb.index(s), 0], ab[xb, sb.index(s), 2]) for s in shared]
                d = [y - x for x, y in zip(da, db)]
                dci = boot_ci(d)                 # rounded, for display
                raw = boot_ci_raw(d)             # unrounded, for the sign test
                v = "larger_better" if raw and raw[0] > 0 else "larger_worse" if raw and raw[1] < 0 else "no_reliable_difference"
                row.update(comparable=True, paired_delta_tau_small=_r(_m(d)), ci95=dci, verdict=v)
                if v == "no_reliable_difference":
                    F("low", "no_reliable_size_effect", row["runs"],
                      f"{dist}: paired delta tau_small {_r(_m(d))} (CI {dci}) from small_items {ca_['small_items']} to "
                      f"{cb_['small_items']} on {len(shared)} shared seeds and {len(common)} common candidates")
            else:
                row["comparable"] = False
                F("high", "confounded_size_comparison", row["runs"],
                  f"{dist}: runs at small_items {ca_['small_items']} and {cb_['small_items']} share {len(shared)} seeds and "
                  f"{len(common)} candidates (pools {len(ia)} vs {len(ib)}); a tau difference cannot be attributed to size",
                  {"type": "matched_size_rerun", "distribution": dist, "reference_run_id": ra["run_id"],
                   "small_items": cb_["small_items"], "seed_start": min(sa),
                   "n_seeds": min(MAX_SEEDS, max(len(sa), POWER_MIN_SEEDS))})
            out.append(row)
    return out


def _hypothesis_review(hyps, entries, expected, F):
    n_tests, vacuous, wanted = 0, [], set()
    for h in hyps:
        t = (h or {}).get("test") or {}
        cand, base, want = t.get("candidate_id"), t.get("baseline"), str(t.get("distribution", "any")).lower()
        if not (cand and base):
            F("low", "untestable_hypothesis", [], f"{(h or {}).get('id')} has no machine-checkable test")
            continue
        wanted.add(want)
        for run, _, _, groups in entries:
            if want in ("any", run["distribution"]):
                n_tests += 1
                if any(cand in g and base in g for g in groups):
                    vacuous.append(f"{h.get('id')} ({run['run_id']})")
    if vacuous:
        F("high", "vacuous_hypothesis", [], f"{', '.join(vacuous)}: candidate and baseline score identically, so the test says nothing")
    if n_tests >= 6:
        F("low", "multiple_comparisons", [],
          f"{n_tests} hypothesis tests at 95% intervals: expect about {0.05 * n_tests:.1f} false 'supported'/'refuted' verdicts; treat single verdicts as hypotheses")
    have = {e[0]["distribution"] for e in entries}
    for d in expected:
        if d not in have:
            used = d in wanted
            F("medium" if used else "low", "untested_distribution", [],
              f"no run on {d}" + ("; hypotheses that name it are untestable" if used else ""),
              {"type": "add_distribution", "distribution": d})


# ------------------------------------------------------------------- tool
def critique_runs(run_ids, hypotheses="", stage="interim", expected_distributions="") -> dict:
    """Find the weaknesses in the evidence behind the lab's conclusions.

    run_ids: list, JSON string or comma list of instrumented run ids (all distributions,
    including pivot runs). hypotheses: optional JSON list from the algorithm agent.
    stage: "interim" (before held-out) or "final" (missing held-out then counts as a finding).
    expected_distributions: comma list; defaults to all supported distributions.
    Returns a critique_id (persisted, consumed by the planner) plus ranked findings."""
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
    stage = "final" if str(stage).strip().lower() == "final" else "interim"
    expected = [d for d in dict.fromkeys(x.strip().lower() for x in str(expected_distributions).split(","))
                if d in DISTRIBUTIONS] or list(DISTRIBUTIONS)

    findings = []

    def F(sev, check, runs, evidence, action=None):
        findings.append({"severity": sev, "check": check, "run_ids": runs, "evidence": evidence, "action": action})

    entries, runs_out, missing = [], [], []
    for rid in dict.fromkeys(run_ids):
        run = load_run(rid)
        if not run or run.get("kind") != "instrumented" or not run.get("scores"):
            missing.append(rid)
            continue
        summary, arr, ids, groups = _review_run(run, F)
        entries.append((run, arr, ids, groups))
        runs_out.append(summary)
    if not entries:
        return {"error": "no usable runs", "missing_run_ids": missing}

    sizes = _size_pairs(entries, F)
    _hypothesis_review(hypotheses if isinstance(hypotheses, list) else [], entries, expected, F)
    if stage == "final":
        done = {s["distribution"] for s in runs_out if s["has_heldout"]}   # one held-out per distribution is enough
        for s in runs_out:
            if s["distribution"] not in done:
                F("medium", "no_heldout", [s["run_id"]], f"{s['distribution']}: no held-out result for this distribution")
                done.add(s["distribution"])                                 # report each distribution once

    findings.sort(key=lambda f: SEV_ORDER[f["severity"]])
    for i, f in enumerate(findings, 1):
        f["id"] = f"F{i}"
    counts = {s: sum(f["severity"] == s for f in findings) for s in SEV_ORDER}
    crit = {"critique_id": f"crit-{uuid.uuid4().hex[:6]}", "stage": stage,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_ids": [s["run_id"] for s in runs_out], "missing_run_ids": missing,
            "severity_counts": counts, "runs": runs_out, "size_comparisons": sizes, "findings": findings,
            "caveats": ["tau noise ceiling = mean tau between full-scale rankings of different single seeds; a proxy scored per seed should not be expected to beat it",
                        "cascade_oos splits the seeds of ONE run in half (fresh instances per seed, same candidates); it is a check on the in-sample replay, not a substitute for held-out seeds",
                        "pre-registered thresholds are fixed in analysis_tools and cannot be changed here"]}
    _save(crit)
    _log("critique", {"critique_id": crit["critique_id"], "run_ids": crit["run_ids"], "severity_counts": counts})
    return crit