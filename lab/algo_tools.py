"""Tools owned by the Algorithm Agent.

register_candidates : gate + shared registry. A candidate is stored only if it
                      passes a static check (allowed syntax only) and a smoke
                      test (returns finite scores, one per bin). Rejections
                      carry a reason, so the agent can fix them.
list_candidates     : compact view of the registry (ids, ideas, sources; no code)
                      so the agent avoids duplicates and later agents can find
                      candidates by id.
get_candidate_code  : used by the Experiment side to load code by id.

The registry is the hand-off between agents: the Algorithm Agent returns ids,
not code, which keeps messages short and the audit trail exact.
Note: this is a fast first gate. The agent must still call verify_candidate
(full packing validity) before registering.
"""
import ast
import hashlib
import json
import math
import os
import threading

import numpy as np

REGISTRY_PATH = os.environ.get("CANDIDATE_REGISTRY", "lab_data/candidates.json")
ARXIV_CACHE = os.environ.get("ARXIV_CACHE", "lab_data/arxiv_cache.json")  # written by arxiv_tool
MAX_REGISTRY = 60       # hard cap on stored candidates
MAX_PER_CALL = 12       # hard cap per register call
_lock = threading.Lock()

_BANNED_NODES = (ast.Import, ast.ImportFrom, ast.While, ast.Global, ast.Nonlocal,
                 ast.Try, ast.With, ast.AsyncFunctionDef, ast.ClassDef, ast.Await)
_BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "getattr", "setattr",
                 "delattr", "globals", "locals", "vars", "input", "os", "sys", "subprocess"}
MAX_RANGE = 1000


def _capped_range(*args):
    """range() that refuses to build more than MAX_RANGE steps, so a candidate
    cannot hang a worker with range(10**9)."""
    r = range(*args)
    if len(r) > MAX_RANGE:
        raise ValueError(f"range too large (> {MAX_RANGE})")
    return r


_SAFE_BUILTINS = {"range": _capped_range, "len": len, "min": min, "max": max, "abs": abs,
                  "sum": sum, "float": float, "int": int, "round": round,
                  "enumerate": enumerate, "zip": zip}


def _load():
    try:
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save(reg):
    os.makedirs(os.path.dirname(REGISTRY_PATH) or ".", exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=1)


def _strip_version(arxiv_id):
    s = str(arxiv_id).strip()
    base, sep, ver = s.rpartition("v")
    return base if sep and ver.isdigit() and base else s


def _known_arxiv_ids():
    """ids the research agent actually retrieved (from arxiv_tool's cache)."""
    try:
        with open(ARXIV_CACHE, encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, ValueError):
        return set()
    return {_strip_version(p["arxiv_id"]) for papers in cache.values()
            for p in papers if isinstance(p, dict) and p.get("arxiv_id")}


def _static_check(code):
    """Return None if code is allowed, else a reason string."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e.msg} (line {e.lineno})"
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    others = [n for n in tree.body if not isinstance(n, ast.FunctionDef)
              and not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))]
    if len(funcs) != 1 or funcs[0].name != "priority" or others:
        return "code must contain exactly one function, def priority(item, bins)"
    if [a.arg for a in funcs[0].args.args] != ["item", "bins"]:
        return "signature must be priority(item, bins)"
    for node in ast.walk(tree):
        if isinstance(node, _BANNED_NODES):
            return f"not allowed: {type(node).__name__}"
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            return f"not allowed name: {node.id}"
        if (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool) and abs(node.value) > 1e7):
            return "numeric constant too large (> 1e7)"
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)
                and isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float))
                and node.right.value > 8):
            return "exponent too large (> 8)"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "dunder attribute access not allowed"
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "priority"):
            return "recursion not allowed"
    return None


def _smoke_test(code):
    """Run priority() on small synthetic inputs. Return None if OK, else reason."""
    ns = {"np": np, "math": math, "__builtins__": _SAFE_BUILTINS}
    try:
        exec(compile(code, "<candidate>", "exec"), ns)
        fn = ns["priority"]
        for item in (1, 10, 35, 50, 99):
            bins = np.array([item, min(100, item + 7), 100], dtype=float)
            out = np.asarray(fn(item, bins), dtype=float)
            if out.shape != bins.shape:
                return f"returned shape {out.shape}, expected {bins.shape}"
            if not np.all(np.isfinite(out)):
                return "returned NaN or inf"
        out = np.asarray(fn(50, np.array([60.0])), dtype=float)
        if out.shape != (1,):
            return "fails on a single open bin"
    except Exception as e:  # any runtime failure is a rejection
        return f"runtime error: {type(e).__name__}: {e}"
    return None


def register_candidates(candidates) -> dict:
    """Gate and store candidates. `candidates` is a list (or JSON string) of
    {"id","code","idea","hypothesis_id","source_arxiv_ids"}.
    Returns {"registered":[ids], "rejected":[{"id","reason"}], "registry_size":n}."""
    if isinstance(candidates, str):  # tolerate a JSON string from the runtime
        try:
            candidates = json.loads(candidates)
        except ValueError:
            return {"error": "candidates must be a JSON list", "registered": [], "rejected": []}
    if not isinstance(candidates, list):
        return {"error": "candidates must be a list", "registered": [], "rejected": []}

    registered, rejected = [], []
    with _lock:
        reg = _load()
        seen_hashes = {v["code_hash"]: k for k, v in reg.items()}
        known_ids = _known_arxiv_ids()
        for c in candidates[:MAX_PER_CALL]:
            cid = str(c.get("id", "")).strip()
            code = str(c.get("code", "")).strip()
            if not cid or not code:
                rejected.append({"id": cid or "?", "reason": "missing id or code"})
                continue
            if cid in reg:
                rejected.append({"id": cid, "reason": "id already registered"})
                continue
            if len(reg) >= MAX_REGISTRY:
                rejected.append({"id": cid, "reason": f"registry full ({MAX_REGISTRY})"})
                continue
            h = hashlib.sha1("".join(code.split()).encode()).hexdigest()[:12]
            if h in seen_hashes:
                rejected.append({"id": cid, "reason": f"duplicate of {seen_hashes[h]}"})
                continue
            srcs = c.get("source_arxiv_ids") or []
            if isinstance(srcs, str):
                srcs = [srcs]
            unknown = [x for x in srcs if _strip_version(x) not in known_ids]
            if unknown:
                rejected.append({"id": cid, "reason": f"cites ids the research agent never retrieved: {unknown}"})
                continue
            why = _static_check(code) or _smoke_test(code)
            if why:
                rejected.append({"id": cid, "reason": why})
                continue
            reg[cid] = {"code": code, "code_hash": h, "idea": c.get("idea", ""),
                        "hypothesis_id": c.get("hypothesis_id", ""),
                        "source_arxiv_ids": c.get("source_arxiv_ids", []),
                        "origin": "AGENT-GENERATED"}
            seen_hashes[h] = cid
            registered.append(cid)
        for c in candidates[MAX_PER_CALL:]:
            rejected.append({"id": str(c.get("id", "?")), "reason": f"over per-call cap ({MAX_PER_CALL})"})
        _save(reg)
    return {"registered": registered, "rejected": rejected, "registry_size": len(reg)}


def list_candidates() -> dict:
    """Compact registry view: id, idea, hypothesis_id, sources. No code."""
    reg = _load()
    return {"count": len(reg),
            "candidates": [{"id": k, "idea": v["idea"], "hypothesis_id": v["hypothesis_id"],
                            "source_arxiv_ids": v["source_arxiv_ids"]} for k, v in reg.items()]}


def get_candidate_code(candidate_id: str) -> dict:
    """Return the code for one registered candidate."""
    reg = _load()
    if candidate_id not in reg:
        return {"error": f"unknown candidate id {candidate_id}"}
    return {"id": candidate_id, "code": reg[candidate_id]["code"]}
