"""Regenerate ALL figures from the saved results JSON (HANDOFF step 19).
No numbers are hand-edited; everything is read from results/*.json.

Figures written to results/figs/:
  fig_regime_heatmap.png   -- 2D: where causal beats greedy (both worlds)
  fig_crossover.png        -- 1D: TRUE perf vs noise, greedy vs causal
  fig_replication_audit.png-- greedy's kept "wins": survive vs vanish under re-test
  fig_false_accepts.png    -- false-acceptances vs noise, greedy vs causal
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def _load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def fig_regime_heatmap():
    grid = _load("regime_grid.json")
    sigmas = grid["meta"]["sigmas"]
    p_goods = grid["meta"]["p_goods"]
    worlds = grid["meta"]["worlds"]
    cells = grid["cells"]

    fig, axes = plt.subplots(1, len(worlds), figsize=(6.4 * len(worlds), 5.0),
                             squeeze=False)
    vmax = max(abs(c["causal_minus_greedy_T"]) for c in cells)
    for ax, world in zip(axes[0], worlds):
        Z = np.full((len(p_goods), len(sigmas)), np.nan)
        for c in cells:
            if c["world"] != world:
                continue
            i = p_goods.index(c["p_good"])
            j = sigmas.index(c["sigma"])
            Z[i, j] = c["causal_minus_greedy_T"]
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="RdBu",
                       vmin=-vmax, vmax=vmax)
        # contour at 0 = the regime boundary
        try:
            ax.contour(range(len(sigmas)), range(len(p_goods)), Z,
                       levels=[0.0], colors="black", linewidths=2)
        except Exception:
            pass
        ax.set_xticks(range(len(sigmas)))
        ax.set_xticklabels([str(s) for s in sigmas], rotation=45, ha="right")
        ax.set_yticks(range(len(p_goods)))
        ax.set_yticklabels([str(p) for p in p_goods])
        ax.set_xlabel("measurement noise  sigma")
        ax.set_ylabel("signal base-rate  p_good")
        ax.set_title(f"{world}: T(causal) - T(greedy)\nblue = causal wins")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Regime map: when does causal acceptance beat greedy? "
                 "(equal compute budget)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(FIG_DIR, "fig_regime_heatmap.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


def fig_crossover(p_good_pick=0.20):
    grid = _load("regime_grid.json")
    sigmas = grid["meta"]["sigmas"]
    worlds = grid["meta"]["worlds"]
    fig, axes = plt.subplots(1, len(worlds), figsize=(6.0 * len(worlds), 4.5),
                             squeeze=False)
    for ax, world in zip(axes[0], worlds):
        rows = [c for c in grid["cells"]
                if c["world"] == world and c["p_good"] == p_good_pick]
        rows.sort(key=lambda c: c["sigma"])
        xs = [c["sigma"] for c in rows]
        for arm, color in [("greedy", "tab:red"), ("causal", "tab:blue")]:
            ys = [c[f"{arm}_T_mean"] for c in rows]
            es = [c[f"{arm}_T_se"] for c in rows]
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=arm, color=color)
        ax.axhline(0.0, color="gray", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("measurement noise  sigma  (log)")
        ax.set_ylabel("final TRUE performance")
        ax.set_title(f"{world}  (p_good={p_good_pick})")
        ax.legend()
    fig.suptitle("Crossover: greedy wins at low noise, causal wins at high noise",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(FIG_DIR, "fig_crossover.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


def fig_replication_audit():
    a = _load("replication_audit.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # left: overall kept -> survive / vanish
    kept = a["kept"]; surv = a["survive_replication"]; vanish = a["vanish_replication"]
    ax1.bar(["kept by\ngreedy"], [kept], color="tab:red", alpha=0.7)
    ax1.bar(["survive\nre-test"], [surv], color="tab:green", alpha=0.8)
    ax1.bar(["vanish\nunder re-test"], [vanish], color="tab:gray", alpha=0.8)
    for i, v in enumerate([kept, surv, vanish]):
        ax1.text(i, v + 1, str(v), ha="center", fontweight="bold")
    ax1.set_ylabel("count of accepted changes")
    ax1.set_title(f"Greedy's 'wins' under replication\n"
                  f"{vanish}/{kept} ({100*vanish/kept:.0f}%) do NOT survive")

    # right: by kind (good vs null) survival
    kinds = list(a["by_kind"].keys())
    x = np.arange(len(kinds)); w = 0.38
    kept_k = [a["by_kind"][k]["kept"] for k in kinds]
    surv_k = [a["by_kind"][k]["surv_rep"] for k in kinds]
    ax2.bar(x - w/2, kept_k, w, label="kept", color="tab:red", alpha=0.7)
    ax2.bar(x + w/2, surv_k, w, label="survive re-test", color="tab:green", alpha=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(kinds)
    ax2.set_ylabel("count"); ax2.set_title("By ground-truth kind")
    ax2.legend()

    m = a["meta"]
    fig.suptitle(f"Replication audit (sigma={m['sigma']}, R={m['R']} re-runs, "
                 f"{m['n_outer']} greedy runs)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(FIG_DIR, "fig_replication_audit.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


def fig_false_accepts(p_good_pick=0.20, world="ceiling"):
    grid = _load("regime_grid.json")
    rows = [c for c in grid["cells"]
            if c["world"] == world and c["p_good"] == p_good_pick]
    rows.sort(key=lambda c: c["sigma"])
    xs = [c["sigma"] for c in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(xs, [c["greedy_false_mean"] for c in rows], "o-", color="tab:red", label="greedy")
    ax.plot(xs, [c["causal_false_mean"] for c in rows], "o-", color="tab:blue", label="causal")
    ax.set_xscale("log")
    ax.set_xlabel("measurement noise  sigma  (log)")
    ax.set_ylabel("false-acceptances per run (true effect <= 0)")
    ax.set_title(f"Causal gate suppresses false-acceptances at all noise levels\n"
                 f"({world} world, p_good={p_good_pick})")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig_false_accepts.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    fig_regime_heatmap()
    fig_crossover()
    fig_replication_audit()
    fig_false_accepts()
    print("All figures ->", FIG_DIR)
