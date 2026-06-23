"""Plot best-score-so-far vs budget spent for all MLRC Machine Unlearning runs."""

import json, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, "skeptic_gate", "results")
VANILLA_DIR = os.path.join(BASE, "results", "vanilla_autoresearch_run")
OUT = os.path.join(BASE, "mlrc_progress.png")

def load_run(results_jsonl, summary_json):
    summary = json.load(open(summary_json))
    steps = [json.loads(l) for l in open(results_jsonl)]
    budgets, bests = [], []
    for st in steps:
        kind = st.get("kind")
        if kind == "baseline":
            budgets.append(0)
            bests.append(st["score"])
        elif kind == "proposal":
            budgets.append(st.get("budget_spent_after", budgets[-1] + 1))
            bests.append(st.get("incumbent_best", bests[-1]))
    return summary, budgets, bests

# Collect runs
greedy_runs, causal_runs = [], []

for d in sorted(glob.glob(os.path.join(RESULTS_DIR, "mlrc_*"))):
    sf = os.path.join(d, "summary.json")
    rf = os.path.join(d, "results.jsonl")
    if not os.path.isfile(sf) or not os.path.isfile(rf):
        continue
    s = json.load(open(sf))
    if s["budget"] < 8:
        continue
    summary, budgets, bests = load_run(rf, sf)
    entry = {"name": os.path.basename(d), "seed": s["seed"], "budgets": budgets, "bests": bests}
    if s["arm"] == "greedy":
        greedy_runs.append(entry)
    else:
        causal_runs.append(entry)

# Laptop vanilla run
if os.path.isfile(os.path.join(VANILLA_DIR, "results.jsonl")):
    sf = os.path.join(VANILLA_DIR, "summary.json")
    rf = os.path.join(VANILLA_DIR, "results.jsonl")
    summary, budgets, bests = load_run(rf, sf)
    laptop_run = {"name": "vanilla_laptop", "seed": 0, "budgets": budgets, "bests": bests}

# ── Plot ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

greedy_colors = ["#e74c3c", "#c0392b", "#e67e22", "#d35400"]
causal_colors = ["#2980b9", "#3498db", "#1abc9c", "#16a085", "#2ecc71"]

# Left panel: all individual runs
ax = axes[0]
ax.set_title("Individual Runs", fontsize=14, fontweight="bold")

for i, run in enumerate(greedy_runs):
    label = f"greedy s{run['seed']}"
    ax.plot(run["budgets"], run["bests"], "-o", color=greedy_colors[i % len(greedy_colors)],
            markersize=5, linewidth=1.8, label=label, alpha=0.85)

for i, run in enumerate(causal_runs):
    label = f"causal s{run['seed']}"
    ax.plot(run["budgets"], run["bests"], "-s", color=causal_colors[i % len(causal_colors)],
            markersize=5, linewidth=1.8, label=label, alpha=0.85)

ax.axhline(0.054505, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="baseline (0.0545)")
ax.set_xlabel("Budget Spent (eval calls)", fontsize=12)
ax.set_ylabel("Best Score So Far", fontsize=12)
ax.legend(fontsize=8, loc="upper left", ncol=2)
ax.set_xlim(-0.3, 8.5)
ax.grid(True, alpha=0.3)

# Right panel: mean ± std envelope per arm
ax2 = axes[1]
ax2.set_title("Mean ± Std by Arm", fontsize=14, fontweight="bold")

def interpolate_to_grid(runs, grid):
    """Interpolate each run's best-so-far onto a common budget grid."""
    all_vals = []
    for run in runs:
        b, v = np.array(run["budgets"]), np.array(run["bests"])
        interped = np.interp(grid, b, v, left=v[0], right=v[-1])
        all_vals.append(interped)
    return np.array(all_vals)

grid = np.linspace(0, 8, 50)

if greedy_runs:
    g_vals = interpolate_to_grid(greedy_runs, grid)
    g_mean = g_vals.mean(axis=0)
    g_std = g_vals.std(axis=0)
    ax2.plot(grid, g_mean, "-", color="#e74c3c", linewidth=2.5, label=f"greedy (n={len(greedy_runs)})")
    ax2.fill_between(grid, g_mean - g_std, g_mean + g_std, color="#e74c3c", alpha=0.15)

if causal_runs:
    c_vals = interpolate_to_grid(causal_runs, grid)
    c_mean = c_vals.mean(axis=0)
    c_std = c_vals.std(axis=0)
    ax2.plot(grid, c_mean, "-", color="#2980b9", linewidth=2.5, label=f"causal (n={len(causal_runs)})")
    ax2.fill_between(grid, c_mean - c_std, c_mean + c_std, color="#2980b9", alpha=0.15)

ax2.axhline(0.054505, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="baseline")
ax2.set_xlabel("Budget Spent (eval calls)", fontsize=12)
ax2.legend(fontsize=10, loc="upper left")
ax2.set_xlim(-0.3, 8.5)
ax2.grid(True, alpha=0.3)

fig.suptitle("MLRC Machine Unlearning: Score Progression (Greedy vs Causal)",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved to {OUT}")
