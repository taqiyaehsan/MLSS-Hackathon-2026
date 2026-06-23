"""Plot progressive MLRC-Bench unlearning results: greedy vs causal score trajectories."""
import json
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results/mlrc_unlearning")
OUT = RESULTS

COMPLETE_RUNS = {
    "greedy": [
        "mlrc_greedy_s0_20260622T062831Z",
        "mlrc_greedy_s0_20260622T080539Z",
        "mlrc_greedy_s1_20260622T081217Z",
        "mlrc_greedy_s2_20260623T164710Z",
    ],
    "causal": [
        "mlrc_causal_s0_20260622T133732Z",
        "mlrc_causal_s2_20260622T144052Z",
        "mlrc_causal_s3_20260622T174319Z",
        "mlrc_causal_s4_20260622T174424Z",
        "mlrc_causal_s5_20260622T200351Z",
    ],
}

def parse_run(run_dir):
    """Parse a results.jsonl into a list of (step, incumbent_score) pairs."""
    path = RESULTS / run_dir / "results.jsonl"
    lines = [json.loads(l) for l in path.read_text().strip().splitlines() if l.strip()]
    trajectory = []
    incumbent = None
    for entry in lines:
        if entry["kind"] == "baseline":
            incumbent = entry["score"]
            trajectory.append((0, incumbent))
        elif entry["kind"] == "proposal":
            step = entry["step"]
            score = entry.get("mean_score", -10.0)
            if score == -10.0:
                score = None
            if entry.get("accepted", False) and score is not None:
                incumbent = score
            trajectory.append((step, incumbent))
    return trajectory

# --- Collect all trajectories ---
all_rows = []
trajectories = {"greedy": [], "causal": []}

for arm, runs in COMPLETE_RUNS.items():
    for run_id in runs:
        traj = parse_run(run_id)
        trajectories[arm].append((run_id, traj))
        summary = json.loads((RESULTS / run_id / "summary.json").read_text())
        for step, inc_score in traj:
            all_rows.append({
                "run_id": run_id,
                "arm": arm,
                "seed": summary["seed"],
                "step": step,
                "incumbent_score": inc_score,
                "best_score": summary["best_score"],
                "baseline_score": summary["baseline_score"],
            })

# --- Write CSV ---
csv_path = OUT / "progressive_results.csv"
with csv_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run_id", "arm", "seed", "step",
                                       "incumbent_score", "best_score",
                                       "baseline_score"])
    w.writeheader()
    w.writerows(all_rows)
print(f"CSV saved: {csv_path}")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

colors = {"greedy": "#e74c3c", "causal": "#2980b9"}
labels = {"greedy": "Greedy", "causal": "Causal (Skeptic)"}

# Left panel: individual runs
ax = axes[0]
for arm in ("greedy", "causal"):
    for i, (run_id, traj) in enumerate(trajectories[arm]):
        steps = [t[0] for t in traj]
        scores = [t[1] for t in traj]
        lbl = labels[arm] if i == 0 else None
        ax.plot(steps, scores, color=colors[arm], alpha=0.4, linewidth=1.2,
                marker="o", markersize=3, label=lbl)
ax.axhline(y=0.0545, color="gray", linestyle="--", alpha=0.5, label="Baseline")
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Incumbent Score", fontsize=12)
ax.set_title("Individual Runs", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Right panel: mean ± std envelope
ax = axes[1]
for arm in ("greedy", "causal"):
    # Pad all trajectories to same length (max steps)
    max_steps = max(len(traj) for _, traj in trajectories[arm])
    all_scores = []
    for _, traj in trajectories[arm]:
        scores = [t[1] for t in traj]
        # Pad with last value
        padded = scores + [scores[-1]] * (max_steps - len(scores))
        all_scores.append(padded)
    arr = np.array(all_scores)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    steps = list(range(max_steps))
    ax.plot(steps, mean, color=colors[arm], linewidth=2.5, marker="o",
            markersize=5, label=f"{labels[arm]} (n={len(all_scores)})")
    ax.fill_between(steps, mean - std, mean + std, color=colors[arm], alpha=0.15)

ax.axhline(y=0.0545, color="gray", linestyle="--", alpha=0.5, label="Baseline")
ax.set_xlabel("Step", fontsize=12)
ax.set_title("Mean ± Std", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

fig.suptitle("MLRC-Bench Machine Unlearning: Greedy vs Causal (Skeptic) Score Progression",
             fontsize=14, fontweight="bold")
plt.tight_layout()

img_path = OUT / "progressive_results.png"
fig.savefig(str(img_path), dpi=150, bbox_inches="tight")
print(f"Plot saved: {img_path}")
