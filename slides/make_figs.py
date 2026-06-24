"""Generate figures for the SAGE demo deck — DARK template palette.

Run from repo root with the project venv:
    .venv/bin/python slides/make_figs.py
Outputs -> slides/figs/*.png

Design rules (per the slide template):
  * dark background, light ink, purple / red / lavender accents
  * NO titles or subtitles on the figures (those go in slide text boxes)
  * keep only axis labels, ticks, legends, and essential data labels
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "skeptic_gate" / "results"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# ---- template palette ----
BG     = "#0f0f16"   # slide background
PANEL  = "#1b1b25"   # card panels
INK    = "#ecebf3"   # primary text
MUTE   = "#9a98ab"   # secondary text / ticks
PURPLE = "#a06bf0"   # skeptic / our method / good
RED    = "#f0593c"   # greedy / failure / alarm
LAV    = "#c3b0ea"   # lavender accent
GRAY   = "#7a7a8c"   # baseline / neutral
GRID   = "#262633"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTE, "ytick.color": MUTE,
    "axes.edgecolor": "#3a3a48", "axes.linewidth": 1.0,
    "font.size": 13, "legend.labelcolor": INK,
})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print("  wrote", name)


# =====================================================================
# Colored MNIST — spurious-cue data panel
# =====================================================================
def cmnist_data():
    from torchvision import datasets
    tr = datasets.MNIST(root=str(ROOT / "skeptic_gate" / "_data_mnist"),
                        train=True, download=False)
    data = tr.data.numpy().astype(np.float32) / 255.0; targ = tr.targets.numpy()
    def ex(d): return data[np.where(targ == d)[0][0]]
    def cm(digit, other, match):
        rgb = np.zeros((28, 28, 3), np.float32)
        rgb[..., 0] = digit
        rgb[..., 1] = digit if match else other
        return rgb
    cells = [(7, 3, True, "7"), (9, 1, True, "9"), (2, 8, False, "2"), (4, 6, False, "4")]
    fig, axes = plt.subplots(2, 4, figsize=(8.4, 4.4))
    fig.subplots_adjust(wspace=0.08, hspace=0.08)
    for col, (d, o, m, lab) in enumerate(cells):
        axes[0, col].imshow(cm(ex(d), ex(o), m))
        axes[1, col].imshow(cm(ex(d), ex(o), not m))
        axes[0, col].set_title(lab, fontsize=12, color=INK, pad=4)
        for r in (0, 1):
            axes[r, col].set_xticks([]); axes[r, col].set_yticks([])
            for sp in axes[r, col].spines.values():
                sp.set_visible(True); sp.set_color("#3a3a48"); sp.set_linewidth(0.6)
    axes[0, 0].set_ylabel("TRAIN / VAL", fontsize=12, color=INK)
    axes[1, 0].set_ylabel("TEST", fontsize=12, color=RED)
    save(fig, "cmnist_data.png")


def cmnist_fail():
    base = (0.607, 0.581)
    agent = [(0.876, 0.130), (0.872, 0.149), (0.864, 0.169),
             (0.853, 0.210), (0.851, 0.329), (0.859, 0.171)]
    fig, ax = plt.subplots(figsize=(7.6, 5.2)); style(ax); ax.grid(True, color=GRID, lw=0.9)
    ax.plot([0.5, 0.95], [0.5, 0.95], ls="--", color=MUTE, lw=1.3)
    ax.text(0.6, 0.625, "val = test", color=MUTE, fontsize=10, rotation=40)
    ax.scatter([p[0] for p in agent], [p[1] for p in agent], s=130, color=RED,
               zorder=4, edgecolor=BG, lw=1.0, label="agent's CNNs (accepted)")
    ax.scatter([base[0]], [base[1]], s=230, color=PURPLE, marker="*",
               zorder=5, edgecolor=BG, lw=1.0, label="baseline (reads shape)")
    ax.set_xlabel("validation accuracy"); ax.set_ylabel("held-out TEST accuracy")
    ax.set_xlim(0.55, 0.95); ax.set_ylim(0.05, 0.65)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "cmnist_fail.png")


# =====================================================================
# FashionMNIST samples
# =====================================================================
def fmnist_samples():
    from torchvision import datasets
    ds = datasets.FashionMNIST(root=str(ROOT / "skeptic_gate" / "_data_fmnist"),
                               train=True, download=False)
    names = ["T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Boot"]
    data = ds.data.numpy(); targ = ds.targets.numpy()
    fig, axes = plt.subplots(1, 10, figsize=(13.5, 1.9))
    fig.subplots_adjust(wspace=0.12)
    for d in range(10):
        idx = np.where(targ == d)[0][0]
        axes[d].imshow(data[idx], cmap="magma")
        axes[d].set_xticks([]); axes[d].set_yticks([])
        axes[d].set_title(names[d], fontsize=9.5, color=MUTE, pad=4)
        for sp in axes[d].spines.values():
            sp.set_visible(True); sp.set_color("#3a3a48"); sp.set_linewidth(0.6)
    save(fig, "fmnist_samples.png")


# =====================================================================
# Development progress bars
# =====================================================================
def progress():
    tasks = ["FashionMNIST", "MAGIC"]
    base = [0.749, 0.785]; agent = [0.887, 0.868]
    x = np.arange(len(tasks)); w = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 4.6)); style(ax)
    ax.bar(x - w/2, base, w, color=GRAY, label="baseline", zorder=3)
    ax.bar(x + w/2, agent, w, color=PURPLE, label="agent's best", zorder=3)
    for i in range(len(tasks)):
        ax.text(x[i] + w/2, agent[i] + 0.006, f"+{agent[i]-base[i]:.3f}",
                ha="center", color=PURPLE, fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(tasks, color=INK)
    ax.set_ylim(0.6, 0.95); ax.set_ylabel("held-out test accuracy")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "progress.png")


# =====================================================================
# Regime: false-discovery rate vs eval noise (FM + MAGIC)
# =====================================================================
def regime():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    for ax, task, label in zip(axes, ["fashionmnist", "magic"], ["FashionMNIST", "MAGIC"]):
        d = json.load(open(RES / f"study_{task}" / "regime_eval.json")); pts = d["points"]
        xs = np.array([p["noise_std"] for p in pts]); o = np.argsort(xs); xs = xs[o]
        g = np.array([p["greedy"]["fp_rate"] for p in pts])[o]
        c = np.array([p["causal"]["fp_rate"] for p in pts])[o]
        style(ax)
        ax.plot(xs, g, "-o", color=RED, lw=2.6, ms=6, label="greedy", zorder=4)
        ax.plot(xs, c, "-o", color=PURPLE, lw=2.6, ms=6, label="skeptic", zorder=4)
        ax.text(0.04, 0.92, label, transform=ax.transAxes, fontsize=12.5,
                color=INK, fontweight="bold")
        ax.set_xlabel("evaluation noise"); ax.set_ylim(0, 0.6)
    axes[0].set_ylabel("false-discovery rate")
    axes[0].legend(frameon=False, loc="center left")
    save(fig, "regime.png")


def skeptic_value():
    tasks = ["fashionmnist", "magic", "colored_mnist"]
    labels = ["FashionMNIST", "MAGIC", "Colored MNIST"]
    gmeans, cmeans = [], []
    for t in tasks:
        d = json.load(open(RES / f"study_{t}" / "regime_eval.json")); pts = d["points"]
        gmeans.append(np.mean([p["greedy"]["fp_rate"] for p in pts]))
        cmeans.append(np.mean([p["causal"]["fp_rate"] for p in pts]))
    x = np.arange(len(tasks)); w = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 4.6)); style(ax)
    ax.bar(x - w/2, gmeans, w, color=RED, label="greedy", zorder=3)
    ax.bar(x + w/2, cmeans, w, color=PURPLE, label="skeptic", zorder=3)
    for i in range(len(tasks)):
        ratio = gmeans[i] / max(cmeans[i], 1e-9)
        ax.text(x[i], max(gmeans[i], cmeans[i]) + 0.02, f"{ratio:.1f}× fewer",
                ha="center", color=INK, fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, color=INK)
    ax.set_ylim(0, max(gmeans) + 0.12); ax.set_ylabel("mean false-discovery rate")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "skeptic_value.png")


# =====================================================================
# MLRC Machine Unlearning — greedy vs skeptic score progression
# =====================================================================
MLRC_DIR = Path(__file__).resolve().parent / "mlrc_runs"

def _mu_runs(arm):
    """Read each run's (budget_spent, best-so-far) trajectory from results.jsonl."""
    import glob
    runs = []
    for f in sorted(glob.glob(str(MLRC_DIR / f"mlrc_{arm}_*.jsonl"))):
        pts = []
        for line in open(f):
            d = json.loads(line)
            if d.get("kind") == "baseline":
                pts.append((0.0, d["score"]))
            else:
                pts.append((d["budget_spent_after"], d["incumbent_best"]))
        if pts:
            runs.append(sorted(pts))
    return runs

def _stepfill(pts, grid):
    """Forward-fill best-so-far onto a common budget grid (x = eval calls spent)."""
    out = []
    for b in grid:
        v = pts[0][1]
        for pb, pv in pts:
            if pb <= b + 1e-9: v = pv
            else: break
        out.append(v)
    return np.array(out)

def mlrc_result():
    # REAL data from the team's MLRC runs; x = cumulative eval calls (budget spent),
    # so greedy and causal both span 0->8 at equal compute (not the proposal-index axis).
    grid = np.arange(0, 8.0001, 0.2); base = 0.0545
    greedy = np.array([_stepfill(r, grid) for r in _mu_runs("greedy")])
    causal = np.array([_stepfill(r, grid) for r in _mu_runs("causal")])
    gm, gs = greedy.mean(0), greedy.std(0)
    cm, cs = causal.mean(0), causal.std(0)
    fig, ax = plt.subplots(figsize=(9.2, 5.2)); style(ax)
    ax.axhline(base, color=MUTE, ls=(0, (5, 3)), lw=1.6)
    ax.text(0.1, base + 0.003, f"baseline {base:.3f}", color=MUTE, fontsize=10.5, va="bottom")
    ax.fill_between(grid, gm-gs, gm+gs, color=RED, alpha=0.12, lw=0)
    ax.fill_between(grid, cm-cs, cm+cs, color=PURPLE, alpha=0.12, lw=0)
    ax.plot(grid, gm, color=RED, lw=2.8, label=f"greedy (n={len(greedy)})", zorder=5)
    ax.plot(grid, cm, color=PURPLE, lw=2.8, label=f"skeptic (n={len(causal)})", zorder=5)
    ax.text(8.15, cm[-1] + 0.001, f"{cm[-1]:.3f}", color=PURPLE, fontsize=13, fontweight="bold", va="center")
    ax.text(8.15, gm[-1] - 0.001, f"{gm[-1]:.3f}", color=RED, fontsize=13, fontweight="bold", va="center")
    ax.set_xlabel("eval calls (budget spent)"); ax.set_ylabel("best unlearning score so far")
    ax.set_xlim(0, 9.6); ax.set_ylim(0.045, 0.16)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "mlrc_result.png")


# =====================================================================
# Optiver — failed experiment (no signal)
# =====================================================================
def optiver_phantom():
    """Optiver: greedy accepts phantom wins that vanish on re-test; skeptic banks ~none.
    Per-run means across the 3 noise regimes (from the Optiver hpo summary)."""
    # accepted / vanished per run, averaged over regimes
    greedy_acc = np.mean([2.2, 3.0, 1.0]); greedy_van = np.mean([1.0, 1.4, 1.0])
    causal_acc = np.mean([0.0, 1.0, 0.0]); causal_van = np.mean([0.0, 0.8, 0.0])
    arms = ["greedy", "skeptic"]
    total = [greedy_acc, causal_acc]
    vanished = [greedy_van, causal_van]
    survived = [t - v for t, v in zip(total, vanished)]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(7.0, 4.8)); style(ax)
    ax.bar(x, vanished, 0.5, color=RED, zorder=3, label="vanished on re-test")
    ax.bar(x, survived, 0.5, bottom=vanished, color=LAV, zorder=3, label="survived")
    for i, t in enumerate(total):
        ax.text(x[i], t + 0.05, f"{t:.1f}", ha="center", color=INK, fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(arms, color=INK, fontsize=13)
    ax.set_ylim(0, 2.6); ax.set_ylabel("'wins' accepted per run")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "optiver_phantom.png")


def optiver_fail():
    fig, ax = plt.subplots(figsize=(7.2, 4.8)); style(ax)
    labels = ["baseline", "agent's best\n(tuning)"]
    vals = [0.522, 0.5226]
    x = np.arange(len(labels))
    ax.bar(x, vals, 0.45, color=[GRAY, PURPLE], zorder=3)
    ax.axhline(0.50, color=MUTE, ls="--", lw=1.5)
    ax.text(1.45, 0.502, "chance = 0.50", color=MUTE, fontsize=11, va="bottom", ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", color=INK, fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, color=INK)
    ax.set_ylim(0.48, 0.56); ax.set_ylabel("held-out test accuracy")
    save(fig, "optiver_fail.png")


# =====================================================================
# Cost / compute frontier
# =====================================================================
def cost_compute():
    d = json.load(open(RES / "study_fashionmnist" / "llm.json"))
    ms = [m for m in d["methods"] if m.get("test", -1) > -1e5 and m.get("flops", 0) > 0]
    gf = np.array([m["flops"] / 1e9 for m in ms]); acc = np.array([m["test"] for m in ms])
    fig, ax = plt.subplots(figsize=(8.4, 4.8)); style(ax); ax.set_xscale("log")
    ax.scatter(gf, acc, s=110, color=PURPLE, zorder=4, edgecolor=BG, lw=1.0)
    ax.set_xlabel("compute per fit  (GFLOPs, log scale)")
    ax.set_ylabel("held-out test accuracy")
    save(fig, "cost_compute.png")


# =====================================================================
# Agent code-edit card  (before / after)
# =====================================================================
def code_edit():
    before = [
        "class MyMethod(BaseMethod):",
        "    def fit(self, X, y, seed):",
        "        torch.manual_seed(seed)",
        "        self.lin = nn.Linear(28*28, 10)",
        "        # train softmax on flat pixels",
        "    def predict(self, X):",
        "        return self.lin(X.flatten(1)).argmax(1)",
    ]
    after = [
        "class Net(nn.Module):       # agent wrote this",
        "    def __init__(self):",
        "        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)",
        "        self.bn1   = nn.BatchNorm2d(16)",
        "        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)",
        "        self.bn2   = nn.BatchNorm2d(32)",
        "        self.fc1   = nn.Linear(32*7*7, 128)",
        "        self.fc2   = nn.Linear(128, 10)",
        "",
        "  ...then over later proposals it added:",
        "  residual blocks · GroupNorm · MixUp ·",
        "  label smoothing · cosine-annealed LR",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.subplots_adjust(wspace=0.06)
    for ax, lines, head, hcol in [
        (axes[0], before, "BEFORE — linear baseline  (test 0.75)", MUTE),
        (axes[1], after,  "AFTER — agent edited the code  (test 0.89)", PURPLE)]:
        ax.set_facecolor(PANEL); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=PANEL, zorder=0))
        ax.text(0.05, 0.93, head, fontsize=12, color=hcol, fontweight="bold")
        for k, ln in enumerate(lines):
            hl = ln.strip().startswith(("residual", "label", "...")) or "agent wrote" in ln
            ax.text(0.05, 0.82 - k*0.066, ln, fontsize=9.5,
                    color=LAV if hl else INK, family="monospace",
                    style="italic" if hl else "normal", va="top")
    save(fig, "code_edit.png")


# =====================================================================
# Motivation — we over-trust confident agents
# =====================================================================
def motivation():
    fig, ax = plt.subplots(figsize=(10.5, 4.4)); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0.04, 0.08), 0.92, 0.84, color=PANEL, zorder=1))
    ax.add_patch(plt.Rectangle((0.04, 0.84), 0.92, 0.08, color="#12121a", zorder=2))
    ax.text(0.07, 0.88, "agent — autonomous mode", color=MUTE, fontsize=11,
            family="monospace", va="center", zorder=3)
    lines = [
        ("$ agent: cleaning up the project…",        INK),
        ('$ agent: ✓ removed 47 "unused" files',     PURPLE),
        ("$ agent: ✓ all checks pass — done!",       PURPLE),
        ("",                                         INK),
        ("you:  …that was my entire src/ folder",    RED),
    ]
    y = 0.74
    for txt, col in lines:
        ax.text(0.07, y, txt, color=col, fontsize=14, family="monospace", va="top", zorder=3)
        y -= 0.125
    save(fig, "motivation.png")


# =====================================================================
# Tables (dark) — score table, tasks attempted
# =====================================================================
def _table(name, cols, cw, rows, color_fn):
    xs = np.cumsum([0] + cw)
    fig, ax = plt.subplots(figsize=(13, 0.7 + 0.7*len(rows))); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ytop = 0.90
    for j, c in enumerate(cols):
        ax.text(xs[j] + 0.012, ytop + 0.06, c, ha="left", va="center",
                fontsize=12.5, color=LAV, fontweight="bold")
    ax.plot([0, 1], [ytop]*2, color=PURPLE, lw=2.2)
    rh = 0.84 / len(rows)
    for i, r in enumerate(rows):
        y = ytop - (i + 0.7) * rh
        for j, val in enumerate(r):
            ax.text(xs[j] + 0.012, y, val, ha="left", va="center", fontsize=12,
                    color=color_fn(i, j, val), fontweight="bold" if j == 0 else "normal")
        ax.plot([0, 1], [y - rh*0.45]*2, color=GRID, lw=0.8)
    save(fig, name)


def score_table():
    rows = [
        ["FashionMNIST", "vision",         "0.74 → 0.91", "0.75 → 0.89", "real gain"],
        ["MAGIC",        "astrophysics",   "0.80 → 0.87", "0.79 → 0.87", "real gain"],
        ["Colored MNIST","spurious-corr",  "0.61 → 0.88", "0.58 → 0.13", "val↑, test collapses"],
        ["Optiver",      "finance (tune)", "0.52 → 0.52", "0.52 → 0.52", "no signal"],
    ]
    cols = ["Task", "Domain", "Val  (base → agent)", "Test  (base → agent)", "Outcome"]
    def cfn(i, j, val):
        if j == 4 and val == "real gain": return PURPLE
        if j == 4: return RED
        return INK
    _table("score_table.png", cols, [0.18, 0.16, 0.21, 0.22, 0.23], rows, cfn)


def tasks_attempted():
    rows = [
        ["FashionMNIST",   "image classification",       "vision",          "writes code", "✓ real gain"],
        ["MAGIC Telescope","signal vs background",       "astrophysics",    "writes code", "✓ real gain"],
        ["CIFAR-10",       "machine unlearning",         "named benchmark", "writes code", "✓ skeptic wins"],
        ["Colored MNIST",  "classification (spurious)",  "vision",          "writes code", "✗ test collapses"],
        ["Optiver",        "price-direction prediction", "finance",         "tunes HP",    "✗ no signal"],
    ]
    cols = ["Dataset", "Task", "Domain", "Agent mode", "Result"]
    def cfn(i, j, val):
        if j == 3: return PURPLE if val.startswith("writes") else LAV
        if j == 4: return PURPLE if val.startswith("✓") else RED
        return INK
    _table("tasks_attempted.png", cols, [0.17, 0.26, 0.16, 0.15, 0.23], rows, cfn)


if __name__ == "__main__":
    print("generating figures ->", OUT)
    motivation()
    cmnist_data(); cmnist_fail()
    fmnist_samples()
    progress(); regime(); skeptic_value()
    mlrc_result(); optiver_fail(); optiver_phantom(); cost_compute()
    code_edit()
    score_table(); tasks_attempted()
    print("done.")
