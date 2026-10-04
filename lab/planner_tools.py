"""Tools owned by the Planner Agent. They run nothing and score nothing; they turn
budget, cost and the critic's findings into experiment SPECS the Experiment Agent
can execute as written.

plan_tests : first call. Prices Test A (instrumented) and Test B (live cascade),
             fits a plan inside the budget (keeping a pivot reserve and a
             held-out reserve) and returns ready-to-run specs, one per
             distribution.
plan_next  : after the critic. Reads the persisted critique by id, merges its
             actions into at most one spec per distribution, ranks them by how
             directly they answer the research question, and keeps what fits in
             the REMAINING budget (spend is read from the run records, so nothing
             depends on anyone's arithmetic).

Both write their own audit entry to the research log.

Budget accounting is a LOWER BOUND: evaluations of candidates that later failed
are not in the run records, and re-running a held-out check overwrites its record.
Cost unit = items processed.
"""
import glob
import json
import math
import os

from lab import campaign
from lab.analysis_tools import DEFAULT_CASCADE, RETENTION_MIN, TAU_MEDIUM_MIN, TAU_SMALL_MIN
from lab.critic_tools import MAX_SEEDS, POWER_MIN_SEEDS, load_critique
from lab.exp_tools import BASELINES, DISTRIBUTIONS, FULL_ITEMS, INST_FULL, RUN_DIR, _costs, _i, _pool

HELDOUT_ROWS, HELDOUT_SEEDS = 5, 10                     # top_k 3 + first_fit + best_fit, 10 seeds (run_heldout defaults)
HELDOUT_COST = HELDOUT_ROWS * HELDOUT_SEEDS * INST_FULL * FULL_ITEMS
MIN_SEEDS = 5                                           # below this the planner does not go (tau CIs are meaningless)
PRIORITY = {"matched_size_rerun": 1, "larger_small_items": 2, "more_seeds": 3, "add_distribution": 4}
LEARNS = {1: "paired change in tau_small across small-instance sizes on identical seeds and candidates (answers 'how large must cheap instances be?')",
          2: "whether larger cheap instances raise tau_small where small ones are biased",
          3: "tau_small confidence interval narrow enough to compare against the pre-registered threshold",
          4: "whether screening reliability holds on a distribution not yet tested"}


def _log(decision, evidence):
    try:
        from lab.tools import log_decision
        log_decision("planner", decision, json.dumps(evidence))
    except Exception:  # noqa: BLE001  logging must never break planning
        pass


def _spent():
    """Cost units spent so far IN THE CURRENT CAMPAIGN, read from the run records
    (instrumented + held-out). Records from earlier campaigns are on disk but are not
    charged to this budget; they are returned separately so nothing is hidden.
    Returns (total, by_run, ignored) with ignored = {"runs": n, "cost_units": c}."""
    camp = campaign.current()
    total, by_run, ignored = 0, {}, {"runs": 0, "cost_units": 0}
    for p in glob.glob(os.path.join(RUN_DIR, "*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                run = json.load(f)
            if run.get("kind") != "instrumented":
                continue
            cfg = run["config"]
            ins = len(run["candidates"]) * len(cfg["seeds"]) * sum(cfg["cost_per_evaluation"].values())
            h = run.get("heldout")
            ho = len(h["rows"]) * len(h["seeds"]) * cfg["cost_per_evaluation"]["full"] if h else 0
        except (OSError, ValueError, KeyError):
            continue
        if not campaign.in_campaign(run, camp):
            ignored["runs"] += 1
            ignored["cost_units"] += ins + ho
            continue
        by_run[run["run_id"]] = {"instrumented": ins, "heldout": ho}
        total += ins + ho
    return total, by_run, ignored


def _per_candidate_seed(small_items):
    return sum(_costs(small_items)[0].values())


def _est(n_cand, n_seeds, small_items):
    return n_cand * n_seeds * _per_candidate_seed(small_items)


def _budget(total):
    spent, by_run, ignored = _spent()
    camp = campaign.current()
    return {"total": total, "spent": spent, "remaining": total - spent, "spent_is_lower_bound": True,
            "campaign": {"id": camp["campaign_id"], "started": camp["started"]},
            "spent_by_run": {k: v["instrumented"] + v["heldout"] for k, v in by_run.items()},
            "ignored_runs_from_earlier_campaigns": ignored}, by_run


# ------------------------------------------------------------------ STEP 1
def plan_tests(budget_cost_units=4_000_000, distributions="weibull,uniform,bimodal", n_seeds=6,
               small_items=50, pivot_reserve=0.25) -> dict:
    """Price Test A (instrumented) vs Test B (live cascade) and return a plan, with specs,
    that fits the budget. `distributions` is in priority order: the last ones are dropped first."""
    total = _i(budget_cost_units, 4_000_000, 1, 10**12)
    ns = _i(n_seeds, 6, MIN_SEEDS, MAX_SEEDS)
    si = _i(small_items, 50, 20, 400)
    try:
        reserve = min(0.9, max(0.0, float(pivot_reserve)))
    except (TypeError, ValueError):
        reserve = 0.25
    dists = [d for d in dict.fromkeys(x.strip().lower() for x in str(distributions).split(",")) if d in DISTRIBUTIONS] \
        or ["weibull", "uniform", "bimodal"]
    nd = len(dists)
    budget, _ = _budget(total)
    usable = budget["remaining"] * (1 - reserve)
    n_cand = len(_pool(True))
    costs, _ = _costs(si)
    per = _per_candidate_seed(si) * n_cand
    full_only = nd * ns * n_cand * costs["full"]

    cost_a = nd * ns * per
    ks, km = DEFAULT_CASCADE
    k_s = math.ceil(ks * n_cand)
    k_m = math.ceil(km * k_s)
    cost_b = nd * ns * (n_cand * costs["small"] + k_s * costs["medium"] + k_m * costs["full"])
    options = [
        {"name": "A_instrumented", "estimated_cost_units": cost_a, "within_budget": cost_a + nd * HELDOUT_COST <= usable,
         "executable_now": True, "measures_tau": True, "full_labels_per_seed": n_cand, "tau_observations": nd * ns,
         "cascades_replayable_for_free": True, "cost_vs_full_only": round(cost_a / full_only, 2),
         "note": "every candidate scored at all three fidelities: pays a measurement overhead, gets tau and any cascade by replay"},
        {"name": "B_live_cascade", "estimated_cost_units": cost_b, "within_budget": cost_b + nd * HELDOUT_COST <= usable,
         "executable_now": False, "measures_tau": False, "full_labels_per_seed": k_m, "tau_observations": 0,
         "cascades_replayable_for_free": False, "cost_vs_full_only": round(cost_b / full_only, 2),
         "note": f"default cascade keep {ks}/{km}; pruned candidates never get a full-scale label, so tau cannot be measured; "
                 "no live-cascade tool exists, its cost is modelled by replaying A's stored scores"}]

    # fit Test A: keep as many distributions as possible, then as many seeds as possible (>= MIN_SEEDS)
    fit = None
    for k in range(nd, 0, -1):
        for s in range(ns, MIN_SEEDS - 1, -1):
            if k * s * per + k * HELDOUT_COST <= usable:
                fit = (k, s)
                break
        if fit:
            break
    underpowered = False
    if not fit:
        s3 = int((usable - HELDOUT_COST) // per)
        fit, underpowered = ((1, s3), True) if s3 >= 3 else ((0, 0), False)
    k, s = fit
    specs = [{"distribution": d, "n_seeds": s, "small_items": si, "seed_start": 0, "include_baselines": True,
              "max_cost_units": s * per} for d in dists[:k]]
    # largest pool for which ALL requested distributions fit at MIN_SEEDS (with held-out reserves)
    max_pool = max(0, int((usable - nd * HELDOUT_COST) // (nd * MIN_SEEDS * _per_candidate_seed(si))))
    n_registry = max(0, n_cand - len(BASELINES))
    notes = []
    if budget["ignored_runs_from_earlier_campaigns"]["runs"]:
        ig = budget["ignored_runs_from_earlier_campaigns"]
        notes.append(f"{ig['runs']} run records from earlier campaigns ({ig['cost_units']} cost units) are on disk and are NOT charged to this budget")
    if k == 0:
        if budget["spent"] > 0:
            notes.append(f"budget exhausted in this campaign: spent {budget['spent']} of {total} (see budget.spent_by_run); "
                         "ask the human for budget, or start a new campaign if those runs belong to an earlier loop")
        else:
            notes.append(f"nothing fits even though nothing is spent: the pool has {n_cand} candidates ({n_registry} from the registry) and "
                         f"at most {max_pool} candidates fit all {nd} distributions at {MIN_SEEDS} seeds; trim the registry "
                         "(python -m lab.campaign new --keep-last <n>) or ask the human for budget")
    elif underpowered:
        notes.append(f"even one distribution at {MIN_SEEDS} seeds does not fit: plan is underpowered ({s} seeds); tau intervals will be unreliable")
    if 0 < k < nd:
        notes.append(f"dropped for budget: {dists[k:]}")
        if n_cand > max_pool:
            notes.append(f"the pool is the binding constraint: {n_cand} candidates ({n_registry} from the registry) but only {max_pool} fit all "
                         f"{nd} distributions at {MIN_SEEDS} seeds. Trimming the registry to at most {max(0, max_pool - len(BASELINES))} candidates "
                         "(python -m lab.campaign new --keep-last <n>, before any experiment of this loop has run) would keep every distribution")
    out = {"budget": budget, "pivot_reserve_fraction": reserve, "heldout_reserve_per_distribution": HELDOUT_COST,
           "preregistered": {"tau_small_min": TAU_SMALL_MIN, "tau_medium_min": TAU_MEDIUM_MIN, "top1_retention_min": RETENTION_MIN},
           "pool": {"n_candidates": n_cand, "n_baselines": len(BASELINES), "n_registry": n_registry,
                    "max_pool_fitting_all_distributions": max_pool},
           "options": options, "default_choice": "A_instrumented" if options[0]["executable_now"] else None,
           "default_choice_rule": "A is the only executable test that measures tau, which the research question requires",
           "plan": {"distributions": dists[:k], "n_seeds": s, "small_items": si, "dropped_distributions": dists[k:],
                    "estimated_cost_units": k * s * per, "heldout_reserve": k * HELDOUT_COST,
                    "remaining_after_plan": int(budget["remaining"] - k * s * per)},
           "specs": specs, "notes": notes}
    _log("plan_tests", {"choice": out["default_choice"], "distributions": dists[:k], "n_seeds": s,
                        "estimated_cost_units": k * s * per, "remaining": budget["remaining"]})
    return out


# ------------------------------------------------------------------ STEP 2/3
def plan_next(critique_id, budget_cost_units=4_000_000) -> dict:
    """Turn a critique's findings into the next experiments, within the remaining budget.
    Returns specs (run them through the experiment agent as written), held-out specs,
    what was deferred and why, and feedback for the algorithm agent."""
    crit = load_critique(critique_id)
    if not crit:
        return {"error": f"unknown critique_id {critique_id}"}
    total = _i(budget_cost_units, 4_000_000, 1, 10**12)
    budget, _ = _budget(total)
    proposal_only = crit["stage"] == "final"
    runs = crit["runs"]
    base_si = min(r["small_items"] for r in runs)
    base_ns = max(r["n_seeds"] for r in runs)
    n_cand = len(_pool(True))

    need, held, tool_changes, driving, unaddressed, dup_from, dup_ids = {}, [], [], [], [], [], set()
    for f in crit["findings"]:
        a = f.get("action")
        driving.append({"id": f["id"], "severity": f["severity"], "check": f["check"], "evidence": f["evidence"][:200]})
        if not a:
            unaddressed.append(f"{f['id']} {f['check']}")
            continue
        t = a["type"]
        if t in PRIORITY:
            n = need.setdefault(a["distribution"], {
                "small_items": 0, "n_seeds": 0, "seed_start": None, "because": [], "pair_with": None, "prio": 9})
            n["small_items"] = max(n["small_items"], int(a.get("small_items") or 0))
            n["n_seeds"] = max(n["n_seeds"], int(a.get("n_seeds") or 0))
            if a.get("seed_start") is not None:
                n["seed_start"] = a["seed_start"] if n["seed_start"] is None else min(n["seed_start"], a["seed_start"])
            n["pair_with"] = n["pair_with"] or a.get("reference_run_id")
            n["prio"] = min(n["prio"], PRIORITY[t])
            n["because"].append(f["id"])
        elif t == "heldout_more_seeds":
            held.append({**a, "because": [f["id"]]})
        elif t == "replace_candidates":
            dup_ids.update(a["candidate_ids"])
            dup_from.append(f["id"])
        elif t == "tool_change":
            tool_changes.append(f"{f['id']}: {a['note']}")

    feedback = ([f"replace {sorted(dup_ids)}: each scores identically to best_fit, another baseline or another candidate "
                 f"in every seed (findings {', '.join(dup_from)}); propose different mechanisms"] if dup_ids else [])

    # one spec per distribution, sized to satisfy every action that touches it
    cands = []
    for d, n in need.items():
        si = min(400, n["small_items"] or base_si)
        ns = min(MAX_SEEDS, n["n_seeds"] or base_ns)
        ns = max(ns, MIN_SEEDS)
        start = 0 if n["seed_start"] is None else n["seed_start"]
        est = _est(n_cand, ns, si)
        cands.append({"prio": n["prio"], "est": est, "because": n["because"], "pair_with": n["pair_with"],
                      "spec": {"distribution": d, "n_seeds": ns, "small_items": si, "seed_start": start,
                               "include_baselines": True, "max_cost_units": est}})
    cands.sort(key=lambda c: (c["prio"], c["est"]))

    # held-out reserve: every distribution whose final run still has no held-out check
    pending = {r["distribution"] for r in runs} - {r["distribution"] for r in runs if r["has_heldout"]}
    selected, deferred, spent_plan = [], [], 0
    for c in cands:
        sp = c["spec"]
        want = sp["n_seeds"]
        dset = pending | {x["spec"]["distribution"] for x in selected} | {sp["distribution"]}
        reserve = 0 if proposal_only else len(dset) * HELDOUT_COST
        for s in range(want, MIN_SEEDS - 1, -1):          # shrink seeds before giving a spec up
            est = _est(n_cand, s, sp["small_items"])
            if spent_plan + est + reserve <= budget["remaining"]:
                sp["n_seeds"], sp["max_cost_units"], c["est"] = s, est, est
                c["reduced_from"] = want if s < want else None
                selected.append(c)
                spent_plan += est
                break
        else:
            est = _est(n_cand, MIN_SEEDS, sp["small_items"])
            deferred.append({"what": f"{sp['distribution']} rerun (small_items {sp['small_items']}, at least {MIN_SEEDS} seeds)",
                             "estimated_cost_units": est, "short_by": int(spent_plan + est + reserve - budget["remaining"]),
                             "because": c["because"]})
    left = budget["remaining"] - spent_plan
    heldout_specs = []
    for h in held:
        rows = h.get("top_k", 3) + 2
        cost = rows * h["n_seeds"] * INST_FULL * FULL_ITEMS
        if cost <= left:
            heldout_specs.append({"run_id": h["run_id"], "held_out": True, "top_k": h.get("top_k", 3), "n_seeds": h["n_seeds"]})
            left -= cost
        else:
            deferred.append({"what": f"held-out on {h['run_id']} with {h['n_seeds']} seeds", "estimated_cost_units": cost,
                             "short_by": int(cost - left), "because": h["because"]})

    out = {"critique_id": crit["critique_id"], "stage": crit["stage"], "proposal_only": proposal_only, "budget": budget,
           "specs": [c["spec"] for c in selected],
           "spec_notes": [{"distribution": c["spec"]["distribution"], "estimated_cost_units": c["est"], "because": c["because"],
                           "pair_with_run_id": c["pair_with"], "reduced_from_n_seeds": c["reduced_from"],
                           "learns": LEARNS[c["prio"]]} for c in selected],
           "heldout_specs": heldout_specs, "deferred": deferred,
           "remaining_after_plan": int(left),
           "heldout_reserve_kept": 0 if proposal_only else len(pending | {c["spec"]["distribution"] for c in selected}) * HELDOUT_COST,
           "algorithm_feedback": feedback, "needs_tool_change": tool_changes,
           "findings_without_action": unaddressed, "findings": driving,
           "caveat": "specs carry max_cost_units equal to their estimate: the experiment tool refuses anything dearer"}
    _log("plan_next", {"critique_id": crit["critique_id"], "specs": [c["spec"]["distribution"] for c in selected],
                       "deferred": len(deferred), "remaining_after_plan": out["remaining_after_plan"]})
    return out