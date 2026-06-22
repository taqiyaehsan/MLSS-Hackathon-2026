"""Figure for the REAL hyperparameter-search task (hpo_task.py).

Reads results/hpo_task/summary.json (no hand-edited numbers) and writes
results/figs/fig_hpo_real.png -- three panels:

  1. False accepts vs REAL eval noise   (greedy chases luck; causal holds)
  2. Shipped model quality vs noise      (greedy ships worse models under noise)
  3. Replication audit at the noisiest regime (kept vs survive)

If the summary carries per-seed std fields (multi-seed run), error bars are drawn;
a single-seed summary just plots the point estimates.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

# digits keeps the legacy results/hpo_task/ path; other datasets use results/hpo_<name>/.
def _subdir(dataset):
    return "hpo_task" if dataset == "digits" else f"hpo_{dataset}"

GREEDY_C, CAUSAL_C = "#c0392b", "#2c6fbf"  # red = chases noise, blue = skeptic


def _get(d, key):
    """Return (mean, std) whether the cell stores a scalar or mean/std pair."""
    if f"{key}_mean" in d:
        return d[f"{key}_mean"], d.get(f"{key}_std", 0.0)
    return d[key], 0.0


def fig_hpo_real(dataset="digits"):
    with open(os.path.join(RESULTS_DIR, _subdir(dataset), "summary.json")) as f:
        S = json.load(f)
    sweep = S["sweep"]
    sds = [r["eval_sd"] for r in sweep]
    labels = [r["regime"].split()[0] for r in sweep]

    def series(arm, key, sub=None):
        ms, ss = [], []
        for r in sweep:
            cell = r["arms"][arm]
            if sub:
                cell = cell[sub]
            m, s = _get(cell, key)
            ms.append(m); ss.append(s)
        return np.array(ms), np.array(ss)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    seeds = S.get("n_seeds", 1)
    desc = S.get("dataset_desc", S.get("dataset", dataset))
    fig.suptitle(f"Real task [{S.get('dataset', dataset)}]: MLP hyperparameter "
                 f"search on {desc} (same gates.py as the synthetic; eval really "
                 f"trains a model)"
                 + (f"  -  {seeds} outer seeds" if seeds > 1 else "  -  1 seed"),
                 fontsize=10)

    # -- Panel 1: false accepts vs noise --
    ax = axes[0]
    for arm, c in (("greedy", GREEDY_C), ("causal", CAUSAL_C)):
        m, s = series(arm, "n_vanished", sub="audit")
        ax.errorbar(sds, m, yerr=s, marker="o", color=c, capsize=3, label=arm)
    ax.set_xlabel("real per-eval noise (sd of val acc)")
    ax.set_ylabel("false accepts  (kept, true gain $\\leq$ 0)")
    ax.set_title("Causal stays near zero; greedy accrues false accepts under noise")
    ax.legend(); ax.grid(alpha=0.3)

    # -- Panel 2: shipped model quality vs noise --
    ax = axes[1]
    for arm, c in (("greedy", GREEDY_C), ("causal", CAUSAL_C)):
        m, s = series(arm, "final_true_test")
        ax.errorbar(sds, m, yerr=s, marker="s", color=c, capsize=3, label=arm)
    base_m, _ = _get(sweep[0], "baseline_test")
    ax.axhline(base_m, ls="--", color="gray", label="baseline")
    ax.set_xlabel("real per-eval noise (sd of val acc)")
    ax.set_ylabel("held-out TEST acc of shipped config")
    ax.set_title("Shipped-model accuracy: no clear gap (error bars overlap)")
    ax.legend(); ax.grid(alpha=0.3)

    # -- Panel 3: replication audit at noisiest regime --
    ax = axes[2]
    hi = sweep[-1]
    x = np.arange(2); w = 0.35
    for i, (arm, c) in enumerate((("greedy", GREEDY_C), ("causal", CAUSAL_C))):
        kept_m, kept_s = _get(hi["arms"][arm]["audit"], "n_kept")
        surv_m, surv_s = _get(hi["arms"][arm]["audit"], "n_survive")
        ax.bar(x[0] + i * w, kept_m, w, yerr=kept_s, color=c, alpha=0.5, capsize=3,
               label=f"{arm}: kept")
        ax.bar(x[1] + i * w, surv_m, w, yerr=surv_s, color=c, capsize=3,
               label=f"{arm}: survive")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(["accepted\n('wins')", "survive\nre-test"])
    ax.set_ylabel("count")
    ax.set_title(f"Replication audit @ {labels[-1]} noise")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    # digits keeps the legacy filename; other datasets get a per-dataset figure.
    name = "fig_hpo_real.png" if dataset == "digits" else f"fig_hpo_{dataset}.png"
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_hpo_real(sys.argv[1] if len(sys.argv) > 1 else "digits")
