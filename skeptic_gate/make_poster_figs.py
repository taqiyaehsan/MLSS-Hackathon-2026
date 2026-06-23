"""Generate publication-quality poster figures from the skeptic-gate results.

Reads the per-task result artifacts (method table, replay audit, regime sweep) and
emits four figures:

  1. fig_progress.png      baseline -> agent's best (held-out test), both tasks
  2. fig_pareto.png        accuracy vs FLOPs, Pareto frontier highlighted (per task)
  3. fig_regime_fp.png     false-positive rate vs eval noise, greedy vs causal (the headline)
  4. fig_regime_acc.png    final model accuracy vs eval noise, greedy vs causal

File discovery is by PATTERN (not hardcoded names), so it works whether the data
lives in the curated `results/skeptic_regime/<task>/` or the raw `study_<task>/`,
and regardless of the exact `methods_*` / `regime_*` suffix.

Usage:  python make_poster_figs.py            # auto-discovers data + writes figs/
        python make_poster_figs.py --data DIR --out DIR
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASKS = ["fashionmnist", "magic"]
TASK_LABEL = {"fashionmnist": "FashionMNIST", "magic": "MAGIC",
              "colored_mnist": "Colored MNIST"}

# consistent palette across every figure
GREEDY_C = "#c1352e"   # red  = the naive baseline
CAUSAL_C = "#2f6db5"   # blue = the skeptic
BASE_C = "#9aa0a6"     # grey = baseline / dominated
BEST_C = "#2f8f4e"     # green = agent's best
FRONTIER_C = "#2f6db5"
CRASH = -1e5           # methods at/below this are runtime crashes -> drop

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "legend.fontsize": 10, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})


# --------------------------------------------------------------------------- #
# data discovery + loading
# --------------------------------------------------------------------------- #
def _find_data_root(explicit: str | None) -> Path:
    here = Path(__file__).resolve().parent
    repo = here.parent
    candidates = [
        explicit,
        repo / "results" / "skeptic_regime",
        here / "results" / "skeptic_regime",
        here / "results",
        repo / "results",
    ]
    for c in candidates:
        if c and Path(c).is_dir():
            # accept a dir that contains at least one known task subdir
            if any((Path(c) / t).is_dir() for t in TASKS) or _looks_like_task_dir(Path(c)):
                return Path(c)
    raise SystemExit("could not locate a results dir; pass --data DIR")


def _looks_like_task_dir(d: Path) -> bool:
    return bool(list(d.glob("regime*eval*.json")) or list(d.glob("methods*.csv")))


def _task_dir(root: Path, task: str) -> Path | None:
    for cand in (root / task, root / f"study_{task}"):
        if cand.is_dir():
            return cand
    hits = [Path(p) for p in glob.glob(str(root / f"*{task}*")) if Path(p).is_dir()]
    return hits[0] if hits else None


def _pick(d: Path, *pattern_groups: list[str]) -> Path | None:
    """Return the first file matching the highest-priority pattern group."""
    for patterns in pattern_groups:
        for pat in patterns:
            hits = sorted(d.glob(pat))
            if hits:
                return hits[0]
    return None


def load_methods(d: Path) -> list[dict]:
    f = _pick(d, ["methods*llm*.csv"], ["methods*.csv"], ["*method*.csv"])
    if not f:
        return []
    rows = []
    with open(f) as fh:
        for r in csv.DictReader(fh):
            try:
                r["acc_val_mean"] = float(r.get("acc_val_mean", "nan"))
                r["stability_std"] = float(r.get("stability_std", "nan"))
                r["test"] = float(r.get("test", "nan"))
                r["gflops"] = float(r.get("gflops", r.get("flops", "nan")) or "nan")
                r["on_frontier"] = str(r.get("on_frontier", "0")).strip() in ("1", "True", "true")
            except ValueError:
                continue
            rows.append(r)
    return rows


def load_regime(d: Path) -> dict | None:
    f = _pick(d, ["regime*eval*.json"], ["regime*.json"])
    if not f:
        return None
    return json.loads(Path(f).read_text())


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_progress(data: dict, out: Path):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    xs, width = range(len(TASKS)), 0.36
    for i, task in enumerate(TASKS):
        rows = data[task]["methods"]
        if not rows:
            continue
        base = next((r for r in rows if r["intent"].strip().lower() == "baseline"), None)
        valid = [r for r in rows if r["test"] > CRASH]
        best = max(valid, key=lambda r: r["acc_val_mean"])  # pick by VAL (no test selection)
        b, t = base["test"], best["test"]
        ax.bar(i - width / 2, b, width, color=BASE_C, label="baseline" if i == 0 else None)
        ax.bar(i + width / 2, t, width, color=BEST_C, label="agent's best" if i == 0 else None)
        ax.text(i - width / 2, b + 0.008, f"{b:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, t + 0.008, f"{t:.3f}", ha="center", fontsize=9, weight="bold")
        ax.annotate(f"+{t - b:.3f}", (i, max(b, t) + 0.05), ha="center",
                    fontsize=10, color=BEST_C, weight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([TASK_LABEL[t] for t in TASKS])
    ax.set_ylabel("held-out test accuracy")
    ax.set_ylim(0.6, 1.0)
    ax.set_title("The agent makes real progress\n(baseline code → agent's edited model)")
    ax.legend(loc="lower right")
    fig.savefig(out / "fig_progress.png")
    plt.close(fig)


def _stability_sizes(rows):
    """Marker size encodes stability (the 3rd Pareto axis): larger = more stable
    (lower std over seeds). Makes a high-FLOPs/lower-acc frontier point legible."""
    stds = [r["stability_std"] for r in rows if r["stability_std"] == r["stability_std"]]
    lo, hi = (min(stds), max(stds)) if stds else (0.0, 1.0)
    rng = (hi - lo) or 1.0
    return {id(r): 70 + 230 * (hi - r["stability_std"]) / rng for r in rows}


def fig_pareto(data: dict, out: Path):
    fig, axes = plt.subplots(1, len(TASKS), figsize=(11, 4.6))
    for ax, task in zip(axes, TASKS):
        rows = [r for r in data[task]["methods"] if r["test"] > CRASH and r["gflops"] > 0]
        if not rows:
            continue
        size = _stability_sizes(rows)
        front = [r for r in rows if r["on_frontier"]]
        dom = [r for r in rows if not r["on_frontier"]]
        ax.scatter([r["gflops"] for r in dom], [r["acc_val_mean"] for r in dom],
                   s=[size[id(r)] for r in dom], c=BASE_C, alpha=0.85,
                   label="dominated", zorder=2)
        ax.scatter([r["gflops"] for r in front], [r["acc_val_mean"] for r in front],
                   s=[size[id(r)] for r in front], c=FRONTIER_C, edgecolor="k",
                   linewidth=0.7, label="Pareto-optimal", zorder=3)
        base = next((r for r in rows if r["intent"].strip().lower() == "baseline"), None)
        if base:
            ax.annotate("baseline", (base["gflops"], base["acc_val_mean"]),
                        textcoords="offset points", xytext=(8, -4), fontsize=9, color="#555")
        # headline annotations per task
        if task == "magic":
            best = max(rows, key=lambda r: r["acc_val_mean"])
            ax.annotate("simplest MLP:\nbest accuracy AND cheapest",
                        (best["gflops"], best["acc_val_mean"]),
                        textcoords="offset points", xytext=(18, -6), fontsize=9,
                        color=BEST_C, weight="bold",
                        arrowprops=dict(arrowstyle="->", color=BEST_C, lw=1))
        if task == "fashionmnist":
            top = max(rows, key=lambda r: r["acc_val_mean"])
            nonbase = [r for r in front if r["intent"].strip().lower() != "baseline"]
            cheap = min(nonbase, key=lambda r: r["gflops"]) if nonbase else None
            if cheap and cheap is not top:
                ratio = top["gflops"] / cheap["gflops"]
                dacc = top["acc_val_mean"] - cheap["acc_val_mean"]
                ax.annotate(f"most accurate:\n{ratio:,.0f}× the FLOPs of the\n"
                            f"cheap CNN for +{dacc:.3f} acc",
                            (top["gflops"], top["acc_val_mean"]),
                            textcoords="offset points", xytext=(-22, -56), fontsize=8.5,
                            color="#555", ha="center",
                            arrowprops=dict(arrowstyle="->", color="#888", lw=1))
        ax.set_xscale("log")
        ax.set_xlabel("compute  (GFLOPs, log scale)")
        ax.set_ylabel("validation accuracy")
        ax.set_title(TASK_LABEL[task])
        ax.legend(loc="lower right", title="marker size = stability\n(larger = more stable)",
                  title_fontsize=8)
    fig.suptitle("More compute ≠ better: Pareto-optimal on accuracy ↑ / stability ↓ / FLOPs ↓",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig_pareto.png")
    plt.close(fig)


def _regime_xy(reg: dict):
    pts = sorted(reg["points"], key=lambda p: p["noise_std"])
    x = [p["noise_std"] for p in pts]
    return pts, x


def fig_regime_fp(data: dict, out: Path):
    fig, axes = plt.subplots(1, len(TASKS), figsize=(11, 4.4), sharey=True)
    for ax, task in zip(axes, TASKS):
        reg = data[task]["regime"]
        if not reg:
            continue
        pts, x = _regime_xy(reg)
        ax.plot(x, [p["greedy"]["fp_rate"] for p in pts], "-o", color=GREEDY_C,
                label="greedy (accept on 1 score)", lw=2)
        ax.plot(x, [p["causal"]["fp_rate"] for p in pts], "-s", color=CAUSAL_C,
                label="skeptic (re-test)", lw=2)
        ax.set_xlabel("evaluation noise  (std of the noisy score)")
        ax.set_title(TASK_LABEL[task])
        ax.set_ylim(0, max(0.6, ax.get_ylim()[1]))
    axes[0].set_ylabel("false-positive rate\n(accepted a change that wasn't real)")
    axes[0].legend(loc="upper left")
    fig.suptitle("The headline: under noisy evaluation, greedy chases noise — the skeptic doesn't",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig_regime_fp.png")
    plt.close(fig)


def fig_regime_acc(data: dict, out: Path):
    fig, axes = plt.subplots(1, len(TASKS), figsize=(11, 4.4))
    for ax, task in zip(axes, TASKS):
        reg = data[task]["regime"]
        if not reg:
            continue
        pts, x = _regime_xy(reg)
        ax.plot(x, [p["greedy"]["mean_final_acc"] for p in pts], "-o", color=GREEDY_C,
                label="greedy", lw=2)
        ax.plot(x, [p["causal"]["mean_final_acc"] for p in pts], "-s", color=CAUSAL_C,
                label="skeptic", lw=2)
        ax.set_xlabel("evaluation noise  (std of the noisy score)")
        ax.set_ylabel("final model accuracy")
        ax.set_title(TASK_LABEL[task])
        ax.legend(loc="lower left")
    fig.suptitle("Cost of being fooled: on MAGIC the skeptic also keeps a better final model",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig_regime_acc.png")
    plt.close(fig)


def fig_spurious(cdata: dict, out: Path):
    """Colored MNIST's distinct story: a SPURIOUS color<->group correlation that
    REVERSES at test. Left panel = val-vs-test per method -- the linear baseline
    leans on color and COLLAPSES on the flipped test, while the agent's CNNs learn
    shape and stay robust (near the diagonal). Right panel = the seed-noise regime
    curve (the skeptic still cuts false positives). Drawn only if data is present."""
    rows = [r for r in cdata["methods"] if r["test"] > CRASH]
    reg = cdata["regime"]
    if not rows:
        print("  ! no colored_mnist methods; skipping fig_spurious")
        return
    base = next((r for r in rows if r["intent"].strip().lower() == "baseline"), None)
    cnns = [r for r in rows if r is not base]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))

    # -- Panel A: validation vs held-out test (robust = near the diagonal) --
    axA.plot([0, 1], [0, 1], "--", color="#bbb", lw=1, zorder=1,
             label="robust (test = val)")
    if cnns:
        axA.scatter([r["acc_val_mean"] for r in cnns], [r["test"] for r in cnns],
                    s=110, c=BEST_C, alpha=0.9, edgecolor="k", linewidth=0.5,
                    label="agent's CNNs", zorder=3)
        top = max(cnns, key=lambda r: r["acc_val_mean"])
        axA.annotate("agent's CNNs:\nlearn shape → robust",
                     (top["acc_val_mean"], top["test"]), textcoords="offset points",
                     xytext=(-10, -36), fontsize=9, color=BEST_C, weight="bold", ha="right")
    if base:
        axA.scatter([base["acc_val_mean"]], [base["test"]], s=180, c=GREEDY_C, marker="D",
                    edgecolor="k", linewidth=0.8, zorder=4, label="linear baseline")
        axA.annotate(f"leans on color →\ncollapses (val {base['acc_val_mean']:.2f} "
                     f"→ test {base['test']:.2f})", (base["acc_val_mean"], base["test"]),
                     textcoords="offset points", xytext=(14, 6), fontsize=9,
                     color=GREEDY_C, weight="bold", ha="left")
    axA.set_xlabel("validation accuracy  (what the gate optimizes)")
    axA.set_ylabel("held-out TEST accuracy\n(spurious correlation reversed)")
    axA.set_xlim(0.0, 1.0)
    axA.set_ylim(0.0, 1.0)
    axA.set_title("The baseline overfits to color (collapses);\nthe agent's CNN learns shape (robust)")
    axA.legend(loc="center left", fontsize=8.5)

    # -- Panel B: seed-noise regime curve (the skeptic still helps) --
    if reg:
        pts, x = _regime_xy(reg)
        axB.plot(x, [p["greedy"]["fp_rate"] for p in pts], "-o", color=GREEDY_C, lw=2,
                 label="greedy (accept on 1 score)")
        axB.plot(x, [p["causal"]["fp_rate"] for p in pts], "-s", color=CAUSAL_C, lw=2,
                 label="skeptic (re-test)")
        axB.set_xlabel("evaluation noise  (std of the noisy score)")
        axB.set_ylabel("false-positive rate")
        axB.set_ylim(0, max(0.3, axB.get_ylim()[1]))
        axB.set_title("Under eval noise the skeptic\nstill cuts false positives")
        axB.legend(loc="upper right", fontsize=9)
    else:
        axB.axis("off")
    fig.suptitle("Colored MNIST — a spurious cue the agent learns past: the baseline "
                 "collapses on the flipped test, the CNN doesn't", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig_spurious.png")
    plt.close(fig)


def fig_skeptic_value(regimes: dict, out: Path):
    """One-panel summary of what the SKEPTIC buys: the mean false-accept rate (greedy
    vs skeptic) across the eval-noise sweep, per task. This isolates the gate's
    contribution -- decision integrity -- which lives in the false-accept rate, not in
    final accuracy. Whiskers = range over noise levels. Drawn if any regime present."""
    import numpy as _np
    items = [(t, r) for t, r in regimes.items() if r]
    if not items:
        print("  ! no regime data; skipping fig_skeptic_value")
        return
    labels, g_mean, c_mean, g_err, c_err, factor = [], [], [], [], [], []
    for t, r in items:
        g = [p["greedy"]["fp_rate"] for p in r["points"]]
        c = [p["causal"]["fp_rate"] for p in r["points"]]
        gm, cm = float(_np.mean(g)), float(_np.mean(c))
        labels.append(TASK_LABEL.get(t, t))
        g_mean.append(gm); c_mean.append(cm)
        g_err.append([gm - min(g), max(g) - gm]); c_err.append([cm - min(c), max(c) - cm])
        factor.append(gm / max(cm, 1e-9))
    x = _np.arange(len(items)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.bar(x - w / 2, g_mean, w, yerr=_np.array(g_err).T, capsize=4, color=GREEDY_C,
           label="greedy (accept on 1 score)")
    ax.bar(x + w / 2, c_mean, w, yerr=_np.array(c_err).T, capsize=4, color=CAUSAL_C,
           label="skeptic (re-test over seeds)")
    for i in range(len(items)):
        top = max(g_mean[i] + g_err[i][1], c_mean[i] + c_err[i][1])
        ax.annotate(f"{factor[i]:.1f}× fewer", (x[i], top + 0.025), ha="center",
                    fontsize=10, weight="bold", color="#333")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("false-accept rate\n(kept a change that wasn't real)")
    ax.set_ylim(0, max(g_mean) + 0.2)
    ax.set_title("What the skeptic buys: fewer false ‘discoveries’ under noisy evaluation\n"
                 "(bars = mean over the eval-noise sweep; whiskers = range)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "fig_skeptic_value.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="dir containing per-task result subdirs")
    ap.add_argument("--out", default=None, help="output dir for figures")
    args = ap.parse_args()

    root = _find_data_root(args.data)
    out = Path(args.out) if args.out else (root / "figs")
    out.mkdir(parents=True, exist_ok=True)

    data = {}
    for task in TASKS:
        d = _task_dir(root, task)
        if not d:
            print(f"  ! no data dir found for {task}")
            data[task] = {"methods": [], "regime": None}
            continue
        data[task] = {"methods": load_methods(d), "regime": load_regime(d)}
        print(f"  {task:13s} <- {d}  ({len(data[task]['methods'])} methods, "
              f"regime={'yes' if data[task]['regime'] else 'no'})")

    fig_progress(data, out)
    fig_pareto(data, out)
    fig_regime_fp(data, out)
    fig_regime_acc(data, out)

    # Colored MNIST is a different KIND of result (a spurious-correlation trap, not a
    # progress/Pareto story), so it gets its own figure rather than a panel in the
    # two-task figures above. Drawn only if its data is present.
    cdir = _task_dir(root, "colored_mnist")
    cdata = None
    if cdir:
        cdata = {"methods": load_methods(cdir), "regime": load_regime(cdir)}
        print(f"  {'colored_mnist':13s} <- {cdir}  ({len(cdata['methods'])} methods, "
              f"regime={'yes' if cdata['regime'] else 'no'})")
        fig_spurious(cdata, out)
    else:
        print("  ! no data dir found for colored_mnist (skipping fig_spurious)")

    # what-the-skeptic-buys summary: false-accept rate (greedy vs skeptic) across tasks
    regimes = {t: data[t]["regime"] for t in TASKS}
    if cdata:
        regimes["colored_mnist"] = cdata["regime"]
    fig_skeptic_value(regimes, out)

    print(f"\nwrote figures to {out}")
    for p in sorted(out.glob("fig_*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
