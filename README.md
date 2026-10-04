# Bin-Packing Discovery Lab (Hack-Nation x Databricks Omnigent, Challenge 03)

**Question:** When an agentic lab evolves online bin-packing heuristics, how
reliably do small-instance rankings predict full-scale rankings, and can that
reliability be used to prune candidates and cut full-scale evaluations?

## Layout

| File | Role |
|---|---|
| `lab.yaml` | Omnigent agent spec: lab director + 6 specialist sub-agents + policies |
| `lab/tools.py` | Deterministic experiment engine; exposed to agents as function tools |
| `run_baseline.py` | Offline run of the whole experiment, no LLM needed |
| `results/research_log.jsonl` | Shared research record (written by `log_decision`) |
| `results/baseline_summary.json` | Offline baseline numbers |

## Step 1: sanity-check the engine (no Omnigent, no API key)

```
pip install numpy
python run_baseline.py
```

Takes ~40 s. Prints, per distribution and seed: Kendall tau (small vs full,
medium vs full), cost speedup of the cascade over full evaluation, regret, and
whether the true best candidate survived screening.

## Step 2: run the lab in Omnigent

Run these from inside this folder so `lab.tools` is importable.

```
omnigent run lab.yaml -p "Run the full discovery loop on weibull, uniform and bimodal."
```

Then open the session in the web UI (http://localhost:6767) to watch sub-agents.

**If you see `ModuleNotFoundError: numpy`:** Omnigent was installed into its own
isolated environment. Reinstall with numpy included:

```
uv tool install --reinstall omnigent --with numpy
```

**If you see `No module named lab`:** set `PYTHONPATH` to this folder
(`$env:PYTHONPATH = (Get-Location)` in PowerShell).

**If a tool schema is rejected:** the spec allows an explicit `parameters:` JSON
schema on a function tool; add one for the failing tool.

**If a sub-agent errors on `executor`:** it is specified explicitly in each one;
compare with `examples/polly/config.yaml` in the Omnigent repo.

## How the experiment works

- Heuristic interface: `priority(item, bins)` returns scores for the open bins
  that fit; the highest score wins; if none fit, a new bin opens. Validity is
  checked on every packing (no bin over 100, every item placed once).
- Metric: excess bins over the lower bound `ceil(sum/100)`.
- Fidelity ladder: small (100 items x3), medium (500 x3), full (2000 x3).
- Cost: items processed (`cost_units`). Speedup = cost(A) / cost(B).
- Test A: every candidate at full fidelity. Test B: small -> medium -> full,
  keeping the top fractions at each stage.
- Held-out check: best candidate re-scored on instances never used in search.
- Pivot: `plan_next_step` reads tau. Below the trust threshold (default 0.3) the
  cascade becomes conservative and the director reopens the screening assumption.
  The threshold is a hypothesis-level setting, not a derived constant; report it
  as such.

## Honest limits (say these in the demo)

- Baseline numbers come from a pool of seed + parametric heuristics. That pool
  barely beats best-fit; the LLM agents' contribution is proposing better ones.
- Cascade runs replay precomputed score tables with cost accounting, rather than
  re-running every stage live.
- Sandbox is hackathon-grade: candidate code is AST-checked (no imports, no
  `while`, no file access). On native Windows, Omnigent gives no filesystem or
  network isolation, so do not run untrusted code. Use WSL if you can.
- A win means "improved over named baselines on stated distributions", not a new
  general algorithm. Validate on other problems before generalizing.
- Cite only sources the literature agent actually retrieved.
