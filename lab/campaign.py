"""Campaign scoping for the lab's persistent state.

Why this exists
---------------
The lab keeps state on disk (lab_data/runs, lab_data/critiques, lab_data/candidates.json).
That state outlives a single loop. Two things then went wrong when a new loop started
on top of an old one:

  * the planner counted EVERY run record on disk as "spent", so a fresh 4,000,000-unit
    budget looked almost used up before any experiment of the new loop had run;
  * the candidate registry kept growing (C1..C24), so every instrumented run priced
    12 baselines + ALL registry candidates and Test A no longer fit the budget.

A *campaign* is one loop with one budget. lab_data/campaign.json names the current
campaign and when it started:

  * run records are stamped with the campaign id when they are written; only runs
    of the current campaign count as spent. Older runs are reported, not charged.
  * `python -m lab.campaign new` starts a fresh campaign: it archives old runs and
    critiques, and decides which registry candidates the new loop keeps.

If no campaign file exists (first use after upgrading), one is created automatically,
starting now. Existing run records are therefore NOT charged to the new campaign.

Usage (run from the project root, the directory that contains lab/):
    python -m lab.campaign status
    python -m lab.campaign new --keep-last 6          # keep the 6 newest candidates
    python -m lab.campaign new --keep-ids C19,C20     # keep named candidates
    python -m lab.campaign new --keep-all-candidates  # leave the registry alone
    python -m lab.campaign new                        # archive everything, empty registry
"""
import argparse
import glob
import json
import os
import shutil
import uuid
from datetime import datetime, timezone

CAMPAIGN_PATH = os.environ.get("CAMPAIGN_FILE", "lab_data/campaign.json")
ARCHIVE_DIR = os.environ.get("LAB_ARCHIVE_DIR", "lab_data/archive")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dirs():
    """Same locations and defaults as exp_tools / critic_tools / algo_tools (read at call
    time; this module must not import them, they import this one)."""
    return (os.environ.get("RUN_DIR", "lab_data/runs"),
            os.environ.get("CRITIQUE_DIR", "lab_data/critiques"),
            os.environ.get("CANDIDATE_REGISTRY", "lab_data/candidates.json"))


def _read():
    try:
        with open(CAMPAIGN_PATH, encoding="utf-8") as f:
            c = json.load(f)
        return c if isinstance(c, dict) and c.get("campaign_id") and c.get("started") else None
    except (OSError, ValueError):
        return None


def _write_new(camp):
    """Create the campaign file atomically. If another session created it first, keep theirs."""
    os.makedirs(os.path.dirname(CAMPAIGN_PATH) or ".", exist_ok=True)
    tmp = f"{CAMPAIGN_PATH}.{uuid.uuid4().hex[:6]}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(camp, f, indent=1)
    try:
        os.link(tmp, CAMPAIGN_PATH)           # atomic, fails if it already exists
    except FileExistsError:
        pass
    except OSError:                            # filesystem without hard links
        if not os.path.exists(CAMPAIGN_PATH):
            os.replace(tmp, CAMPAIGN_PATH)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def current():
    """The current campaign, created on first use (starting now)."""
    c = _read()
    if c:
        return c
    _write_new({"campaign_id": f"camp-{uuid.uuid4().hex[:6]}", "started": _now(),
                "note": "created automatically; run records from before this moment are not charged"})
    return _read() or {"campaign_id": "camp-unsaved", "started": _now(), "note": "could not write campaign file"}


def in_campaign(run, camp=None):
    """Does this run record belong to the current campaign?"""
    camp = camp or current()
    cid = run.get("campaign_id")
    if cid:
        return cid == camp["campaign_id"]
    return str(run.get("created", "")) >= camp["started"]     # legacy record: judge by time


def _registry_keep(reg, keep_ids, keep_last, keep_all):
    if keep_all:
        return dict(reg)
    if keep_ids:
        return {k: v for k, v in reg.items() if k in set(keep_ids)}
    if keep_last > 0:
        return dict(list(reg.items())[-keep_last:])           # dicts keep insertion order
    return {}


def start_new(keep_last=0, keep_ids=None, keep_all_candidates=False, note=""):
    """Archive the finished campaign's runs/critiques and start a fresh one.
    Nothing is deleted: everything moved or trimmed is kept under lab_data/archive/."""
    run_dir, crit_dir, reg_path = _dirs()
    old = _read()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    dest = os.path.join(ARCHIVE_DIR, f"{(old or {}).get('campaign_id', 'pre-campaign')}-{stamp}")
    moved = {"runs": 0, "critiques": 0}
    for key, d in (("runs", run_dir), ("critiques", crit_dir)):
        for p in glob.glob(os.path.join(d, "*.json")):
            os.makedirs(os.path.join(dest, key), exist_ok=True)
            shutil.move(p, os.path.join(dest, key, os.path.basename(p)))
            moved[key] += 1

    try:
        with open(reg_path, encoding="utf-8") as f:
            reg = json.load(f)
    except (OSError, ValueError):
        reg = {}
    kept = _registry_keep(reg, list(keep_ids or []), int(keep_last or 0), bool(keep_all_candidates))
    if reg and len(kept) != len(reg):
        os.makedirs(dest, exist_ok=True)
        shutil.copy2(reg_path, os.path.join(dest, "candidates.json"))
        os.makedirs(os.path.dirname(reg_path) or ".", exist_ok=True)
        with open(reg_path, "w", encoding="utf-8") as f:
            json.dump(kept, f, indent=1)

    camp = {"campaign_id": f"camp-{uuid.uuid4().hex[:6]}", "started": _now(), "note": note or "started by lab.campaign new"}
    os.makedirs(os.path.dirname(CAMPAIGN_PATH) or ".", exist_ok=True)
    tmp = f"{CAMPAIGN_PATH}.{uuid.uuid4().hex[:6]}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(camp, f, indent=1)
    os.replace(tmp, CAMPAIGN_PATH)
    return {"campaign": camp, "archived_to": dest if (moved["runs"] or moved["critiques"] or len(kept) != len(reg)) else None,
            "archived": moved, "registry_before": len(reg), "registry_kept": sorted(kept)}


def status():
    """What would the planner see right now?"""
    from lab.exp_tools import BASELINES, _pool
    from lab.planner_tools import HELDOUT_COST, MIN_SEEDS, _budget, _per_candidate_seed
    budget, _ = _budget(4_000_000)
    n_pool = len(_pool(True))
    per = _per_candidate_seed(50)
    usable = budget["remaining"] * 0.75
    fits = max(0, int((usable - 3 * HELDOUT_COST) // (3 * MIN_SEEDS * per)))
    return {"campaign": budget["campaign"], "spent": budget["spent"], "remaining": budget["remaining"],
            "ignored_runs_from_earlier_campaigns": budget["ignored_runs_from_earlier_campaigns"],
            "pool": {"n_candidates": n_pool, "n_baselines": len(BASELINES), "n_registry": n_pool - len(BASELINES)},
            "test_a_3_distributions_6_seeds": 3 * 6 * n_pool * per,
            "max_pool_for_3_distributions_at_5_seeds": fits,
            "max_registry_candidates_for_that": max(0, fits - len(BASELINES))}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lab.campaign", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="show the current campaign, spend and pool size")
    n = sub.add_parser("new", help="archive old runs/critiques and start a fresh campaign")
    g = n.add_mutually_exclusive_group()
    g.add_argument("--keep-last", type=int, default=0, help="keep the N most recently registered candidates")
    g.add_argument("--keep-ids", default="", help="comma list of candidate ids to keep")
    g.add_argument("--keep-all-candidates", action="store_true", help="do not touch the registry")
    n.add_argument("--note", default="")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        out = status()
    else:
        out = start_new(a.keep_last, [x.strip() for x in a.keep_ids.split(",") if x.strip()],
                        a.keep_all_candidates, a.note)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()