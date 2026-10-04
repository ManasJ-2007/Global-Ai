"""Experiment engine + function tools for the bin-packing discovery lab.

Everything here is deterministic Python. The LLM agents decide WHAT to test;
this module produces the numbers. Every public function takes/returns plain
JSON-friendly types so Omnigent can expose it as a tool.

Heuristic interface (what agents write):

    def priority(item, bins):
        # item: int size; bins: np.ndarray of remaining capacities of the
        # open bins that can hold the item (in the order the bins were opened).
        # Return an np.ndarray of scores, same length. Highest score wins.
        return -(bins - item)          # best-fit
"""
from __future__ import annotations

import ast
import json
import math
import os
import time
from typing import Any

import numpy as np

CAPACITY = 100
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOG_PATH = os.path.join(RESULTS_DIR, "research_log.jsonl")

# Fidelity ladder: (items per instance, number of instances)
FIDELITY = {"small": (100, 3), "medium": (500, 3), "full": (2000, 3)}
DISTRIBUTIONS = ("weibull", "uniform", "bimodal")


# --------------------------------------------------------------------------
# Instances
# --------------------------------------------------------------------------
def make_instance(distribution: str, n_items: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if distribution == "weibull":
        x = rng.weibull(3.0, n_items) * 45.0
    elif distribution == "uniform":
        x = rng.uniform(20, 80, n_items)
    elif distribution == "bimodal":
        small = rng.uniform(5, 25, n_items)
        large = rng.uniform(45, 70, n_items)
        x = np.where(rng.random(n_items) < 0.6, small, large)
    else:
        raise ValueError(f"unknown distribution {distribution!r}; use {DISTRIBUTIONS}")
    return np.clip(np.rint(x), 1, CAPACITY).astype(int)


# --------------------------------------------------------------------------
# Safe-ish loading of agent-written heuristics (hackathon-grade, see README)
# --------------------------------------------------------------------------
_BANNED_NAMES = {
    "exec", "eval", "open", "compile", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "input", "breakpoint", "vars", "dir",
}
_BANNED_ATTRS = {
    "load", "save", "savez", "savetxt", "loadtxt", "fromfile", "tofile",
    "memmap", "genfromtxt", "system", "popen", "ctypeslib", "testing",
}


def _check_ast(code: str) -> None:
    tree = ast.parse(code)
    has_priority = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.While, ast.Global,
                             ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            raise ValueError(f"disallowed name: {node.id}")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr in _BANNED_ATTRS
        ):
            raise ValueError(f"disallowed attribute: {node.attr}")
        if isinstance(node, ast.FunctionDef) and node.name == "priority":
            has_priority = True
    if not has_priority:
        raise ValueError("code must define priority(item, bins)")


def load_priority(code: str):
    _check_ast(code)
    safe_builtins = {
        n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
        for n in ("len", "abs", "min", "max", "range", "sum", "float", "int",
                  "round", "enumerate", "zip", "bool", "pow")
    }
    ns: dict[str, Any] = {"np": np, "math": math, "__builtins__": safe_builtins}
    exec(compile(code, "<candidate>", "exec"), ns)  # noqa: S102
    return ns["priority"]


# --------------------------------------------------------------------------
# Online bin packing simulator
# --------------------------------------------------------------------------
def pack(priority, items: np.ndarray) -> np.ndarray:
    """Pack items online. Returns final loads of each bin."""
    rem = np.empty(len(items) + 1, dtype=float)
    n_open = 0
    for item in items:
        chosen = -1
        if n_open:
            view = rem[:n_open]
            fit_idx = np.nonzero(view >= item)[0]
            if len(fit_idx):
                scores = np.asarray(priority(int(item), view[fit_idx].copy()), dtype=float)
                if scores.shape != fit_idx.shape or not np.all(np.isfinite(scores)):
                    raise ValueError("priority must return finite scores, one per bin")
                chosen = int(fit_idx[int(np.argmax(scores))])
        if chosen < 0:
            rem[n_open] = CAPACITY
            chosen = n_open
            n_open += 1
        rem[chosen] -= item
    return CAPACITY - rem[:n_open]


def validate_packing(loads: np.ndarray, items: np.ndarray) -> None:
    if np.any(loads > CAPACITY + 1e-9):
        raise ValueError("a bin exceeds capacity")
    if abs(float(loads.sum()) - float(items.sum())) > 1e-6:
        raise ValueError("not every item was placed exactly once")


def excess_over_lower_bound(n_bins: int, items: np.ndarray) -> float:
    lb = math.ceil(items.sum() / CAPACITY)
    return (n_bins - lb) / lb


def _score(code: str, distribution: str, fidelity: str, seed: int = 0) -> dict:
    n_items, n_inst = FIDELITY[fidelity]
    priority = load_priority(code)
    ex, bins = [], []
    for k in range(n_inst):
        items = make_instance(distribution, n_items, seed * 1000 + k + 17)
        loads = pack(priority, items)
        validate_packing(loads, items)
        ex.append(excess_over_lower_bound(len(loads), items))
        bins.append(len(loads))
    return {
        "excess": float(np.mean(ex)),
        "bins": float(np.mean(bins)),
        "cost_units": n_items * n_inst,
    }


# --------------------------------------------------------------------------
# Candidate pool (seeds + parametric mutations; agents add their own code)
# --------------------------------------------------------------------------
SEEDS = {
    "first_fit": "def priority(item, bins):\n    return -np.arange(len(bins), dtype=float)\n",
    "best_fit": "def priority(item, bins):\n    return -(bins - item)\n",
    "worst_fit": "def priority(item, bins):\n    return (bins - item).astype(float)\n",
}
_TEMPLATE = (
    "def priority(item, bins):\n"
    "    slack = bins - item\n"
    "    return (-{a:.3f} * slack - {b:.3f} * np.arange(len(bins))\n"
    "            + {c:.3f} * (slack == 0) - {d:.3f} * np.abs(slack - {t}))\n"
)


def generate_candidate_pool(pool_size: int = 40, seed: int = 0) -> list[dict]:
    """Seed heuristics plus random parametric variants. Returns [{id, code}]."""
    rng = np.random.default_rng(seed)
    pool = [{"id": k, "code": v} for k, v in SEEDS.items()]
    i = 0
    while len(pool) < pool_size:
        code = _TEMPLATE.format(
            a=rng.uniform(0, 2), b=rng.uniform(0, 0.3),
            c=rng.uniform(0, 60), d=rng.uniform(0, 1), t=int(rng.integers(0, 40)),
        )
        pool.append({"id": f"cand_{i:03d}", "code": code})
        i += 1
    return pool


# --------------------------------------------------------------------------
# Tools exposed to agents
# --------------------------------------------------------------------------
def verify_candidate(code: str) -> dict:
    """Check that agent-written heuristic code is allowed, runs, and yields valid packings.

    Returns {"ok": bool, "reason": str}. Rejects fast-but-wrong candidates.
    """
    try:
        _score(code, "uniform", "small", seed=99)
        return {"ok": True, "reason": "valid packings on test instances"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def evaluate_candidate(code: str, distribution: str, fidelity: str = "full") -> dict:
    """Score one heuristic. fidelity is 'small', 'medium' or 'full'.

    Returns excess (fraction of bins above the lower bound; lower is better),
    mean bins, and cost_units (items processed; the compute-cost proxy).
    """
    v = verify_candidate(code)
    if not v["ok"]:
        return {"ok": False, "reason": v["reason"]}
    return {"ok": True, **_score(code, distribution, fidelity)}


def kendall_tau(a: list[float], b: list[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    if n < 3:
        return float("nan")
    s = 0
    for i in range(n - 1):
        s += int(np.sum(np.sign(a[i] - a[i + 1:]) * np.sign(b[i] - b[i + 1:])))
    return float(s / (n * (n - 1) / 2))


_TABLE_CACHE: dict[tuple, dict] = {}


def score_table(distribution: str, pool_size: int = 40, seed: int = 0) -> dict:
    """Score every pool candidate at every fidelity (cached). The ground truth."""
    key = (distribution, pool_size, seed)
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
    pool = generate_candidate_pool(pool_size, seed)
    table = {"ids": [c["id"] for c in pool], "codes": {c["id"]: c["code"] for c in pool}}
    for fid in FIDELITY:
        res = [_score(c["code"], distribution, fid, seed) for c in pool]
        table[fid] = [r["excess"] for r in res]
        table[f"{fid}_cost"] = res[0]["cost_units"]
    _TABLE_CACHE[key] = table
    return table


def screening_reliability(distribution: str, pool_size: int = 40, seed: int = 0) -> dict:
    """Kendall tau between cheap (small, medium) and full rankings for a distribution.

    Higher tau = cheap screening can be trusted more.
    """
    t = score_table(distribution, pool_size, seed)
    return {
        "distribution": distribution,
        "tau_small_vs_full": kendall_tau(t["small"], t["full"]),
        "tau_medium_vs_full": kendall_tau(t["medium"], t["full"]),
        "pool_size": pool_size,
    }


def run_experiment(test: str, distribution: str, pool_size: int = 40, seed: int = 0,
                   keep_small: float = 0.5, keep_medium: float = 0.25) -> dict:
    """Run Test A (all candidates at full fidelity) or Test B (small->medium->full cascade).

    Cost is counted in items processed. Returns best candidate, its full-fidelity
    excess, regret vs the true best in the pool, and cost_units.
    """
    t = score_table(distribution, pool_size, seed)
    ids, n = t["ids"], len(t["ids"])
    full, full_cost = t["full"], t["full_cost"]
    true_best = min(full)

    if test.upper() == "A":
        cost = n * full_cost
        best_i = int(np.argmin(full))
        survivors = {"full": n}
    elif test.upper() == "B":
        cost = n * t["small_cost"]
        k1 = max(1, int(math.ceil(n * keep_small)))
        s1 = list(np.argsort(t["small"])[:k1])
        cost += k1 * t["medium_cost"]
        k2 = max(1, int(math.ceil(n * keep_medium)))
        s2 = sorted(s1, key=lambda i: t["medium"][i])[:k2]
        cost += k2 * full_cost
        best_i = min(s2, key=lambda i: full[i])
        survivors = {"small": n, "medium": k1, "full": k2}
    else:
        raise ValueError("test must be 'A' or 'B'")

    return {
        "test": test.upper(),
        "distribution": distribution,
        "best_candidate": ids[int(best_i)],
        "best_full_excess": float(full[int(best_i)]),
        "true_best_full_excess": float(true_best),
        "regret": float(full[int(best_i)] - true_best),
        "cost_units": int(cost),
        "evaluated_per_stage": survivors,
    }


def compare_tests(distribution: str, pool_size: int = 40, seed: int = 0,
                  keep_small: float = 0.5, keep_medium: float = 0.25) -> dict:
    """Run A and B under matched conditions and report speedup, regret and tau."""
    a = run_experiment("A", distribution, pool_size, seed)
    b = run_experiment("B", distribution, pool_size, seed, keep_small, keep_medium)
    return {
        "distribution": distribution,
        "speedup_cost": a["cost_units"] / b["cost_units"],
        "regret_B": b["regret"],
        "found_true_best": b["regret"] <= 1e-12,
        "cost_A": a["cost_units"],
        "cost_B": b["cost_units"],
        **screening_reliability(distribution, pool_size, seed),
    }



def heldout_excess(code: str, distribution: str) -> dict:
    """Score a heuristic on held-out instances the search never saw (different seeds)."""
    ex = []
    for k in range(5):
        items = make_instance(distribution, FIDELITY["full"][0], 900_000 + k)
        loads = pack(load_priority(code), items)
        validate_packing(loads, items)
        ex.append(excess_over_lower_bound(len(loads), items))
    return {"heldout_excess": float(np.mean(ex))}


def sweep_cascade(distribution: str, pool_size: int = 40, seeds: str = "0,1,2,3,4") -> dict:
    """Grid-search the cascade's keep fractions across seeds.

    For each setting reports mean cost speedup vs Test A, mean regret, and the
    fraction of seeds in which the true best candidate survived screening.
    """
    seed_list = [int(x) for x in seeds.split(",")]
    grid = []
    for ks in (0.25, 0.4, 0.5, 0.75):
        for km in (0.1, 0.25, 0.5):
            if km > ks:
                continue
            sp, rg, hit = [], [], []
            for sd in seed_list:
                a = run_experiment("A", distribution, pool_size, sd)
                b = run_experiment("B", distribution, pool_size, sd, ks, km)
                sp.append(a["cost_units"] / b["cost_units"])
                rg.append(b["regret"])
                hit.append(b["regret"] <= 1e-12)
            grid.append({"keep_small": ks, "keep_medium": km,
                         "mean_speedup": float(np.mean(sp)),
                         "mean_regret": float(np.mean(rg)),
                         "p_found_best": float(np.mean(hit))})
    return {"distribution": distribution, "grid": grid}


def plan_next_step(distribution: str, pool_size: int = 40, seed: int = 0,
                   tau_trust: float = 0.3) -> dict:
    """Decision rule the planner uses after seeing screening reliability.

    - tau_small >= tau_trust: trust cheap screening, prune hard.
    - otherwise: do not trust small instances; skip the small stage (screen at
      medium only) and keep more survivors. Records the reason so the decision
      is reconstructable.
    """
    r = screening_reliability(distribution, pool_size, seed)
    if r["tau_small_vs_full"] >= tau_trust:
        plan = {"action": "aggressive_cascade", "keep_small": 0.4, "keep_medium": 0.15}
        why = "small-instance ranking is reliable enough to prune aggressively"
    else:
        plan = {"action": "conservative_cascade", "keep_small": 0.75, "keep_medium": 0.3}
        why = ("small-instance ranking is weak (tau below trust threshold); "
               "keep more survivors and reopen the screening assumption")
    return {"distribution": distribution, "tau_small": r["tau_small_vs_full"],
            "tau_medium": r["tau_medium_vs_full"], "tau_trust": tau_trust,
            "plan": plan, "reason": why}


def log_decision(agent: str, decision: str, evidence_json: str = "{}") -> dict:
    """Append one entry to the shared research record (results/research_log.jsonl)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        evidence = json.loads(evidence_json)
    except json.JSONDecodeError:
        evidence = {"raw": evidence_json}
    entry = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "agent": agent,
             "decision": decision, "evidence": evidence}
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"logged": True, "entries": sum(1 for _ in open(LOG_PATH, encoding="utf-8"))}


def read_log(last_n: int = 20) -> list[dict]:
    """Read the last N entries of the shared research record."""
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, encoding="utf-8") as f:
        lines = f.readlines()[-last_n:]
    return [json.loads(x) for x in lines]
