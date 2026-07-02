"""Method registry for the multi-agent portfolio (agent = method).

Each TASK ships a fixed set of pre-written, vetted METHODS (e.g. cnn / mlp /
logreg). One LLM agent is locked to each method and only ever tunes that method's
bounded CONFIG -- it never writes model code. So:
  * the architectures are human-authored (no runtime-crash proposals; the static
    coherence check is COMPLETE -- a "broken" proposal is just an out-of-bounds
    config), and
  * compute stays bounded (we control how big the CNN is).

A Method bundles: the model builder, its tunable config space (+ baseline), a
validator (the coherence-gate building block), and a one-line brief for the LLM.

Training is shared and harness-owned (seeded, CPU, fixed protocol) so every method
is measured the same way; only the architecture + config differ. The evaluator
returns (accuracy, train_time) -- train_time feeds the Pareto COST axis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

import task_data

DEVICE = torch.device("cpu")
torch.set_num_threads(1)  # stationarity: no contention spikes between evals


# ---------------------------------------------------------------------------
# Model builders. Each takes (cfg, in_shape, n_classes) -> nn.Module.
# in_shape is the per-sample shape: (1,28,28) for images, (10,) for tabular.
# MLP/logreg flatten internally, so they work on any task; the CNN needs images.
# ---------------------------------------------------------------------------

def _flat_dim(in_shape) -> int:
    d = 1
    for s in in_shape:
        d *= int(s)
    return d


def build_logreg(cfg: dict, in_shape, n_classes: int) -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(_flat_dim(in_shape), n_classes)).to(DEVICE)


def build_mlp(cfg: dict, in_shape, n_classes: int) -> nn.Module:
    act = nn.ReLU() if cfg.get("activation", "relu") == "relu" else nn.Tanh()
    h = int(cfg["hidden"])
    return nn.Sequential(
        nn.Flatten(), nn.Linear(_flat_dim(in_shape), h), act,
        nn.Dropout(float(cfg["dropout"])), nn.Linear(h, n_classes),
    ).to(DEVICE)


def build_cnn(cfg: dict, in_shape, n_classes: int) -> nn.Module:
    """Small 2-conv CNN. Kept deliberately modest so the replication audit stays
    affordable (we own this code, unlike agent-written architectures)."""
    c_in = int(in_shape[0])
    ch = int(cfg["channels"])
    h, w = int(in_shape[1]), int(in_shape[2])
    flat = (ch * 2) * (h // 4) * (w // 4)   # two stride-2 pools
    return nn.Sequential(
        nn.Conv2d(c_in, ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(ch, ch * 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Dropout(float(cfg["dropout"])), nn.Linear(flat, n_classes),
    ).to(DEVICE)


# ---------------------------------------------------------------------------
# Method spec + registry
# ---------------------------------------------------------------------------

# Shared training knobs every method exposes (the harness runs the loop with them).
_SHARED_BOUNDS = {"lr": (1e-4, 1.0), "weight_decay": (0.0, 1e-1),
                  "epochs": (1, 30), "batch_size": (16, 256), "dropout": (0.0, 0.7)}
_INT_KEYS = ("epochs", "batch_size", "hidden", "channels")


@dataclass
class Method:
    name: str
    build: Callable
    bounds: dict                       # numeric key -> (lo, hi)
    baseline: dict                     # deliberately-mediocre starting config
    desc: str                          # one line for the LLM brief
    cats: dict = field(default_factory=dict)   # categorical key -> allowed tuple
    needs_image: bool = False          # CNN needs (C,H,W) input

    def validate(self, cfg: dict) -> tuple[bool, str]:
        """Cheap static coherence check -- trains nothing."""
        for k in self.baseline:
            if k not in cfg:
                return False, f"missing key {k}"
        for k, allowed in self.cats.items():
            if cfg[k] not in allowed:
                return False, f"{k}={cfg[k]!r} not in {allowed}"
        for k, (lo, hi) in self.bounds.items():
            v = cfg[k]
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return False, f"{k} not numeric"
            if not (lo <= v <= hi):
                return False, f"{k}={v} out of [{lo},{hi}]"
        for k in _INT_KEYS:
            if k in self.bounds and int(cfg[k]) != cfg[k]:
                return False, f"{k} must be integer"
        return True, ""


METHODS: dict[str, Method] = {
    "logreg": Method(
        "logreg", build_logreg,
        bounds={k: _SHARED_BOUNDS[k] for k in ("lr", "weight_decay", "epochs", "batch_size")},
        baseline={"lr": 0.01, "weight_decay": 0.0, "epochs": 6, "batch_size": 64},
        desc="multinomial logistic regression (a single linear layer on the flattened input)"),
    "mlp": Method(
        "mlp", build_mlp,
        bounds={**{k: _SHARED_BOUNDS[k] for k in _SHARED_BOUNDS}, "hidden": (8, 256)},
        baseline={"hidden": 32, "lr": 0.01, "weight_decay": 0.0, "epochs": 6,
                  "batch_size": 64, "dropout": 0.0, "activation": "relu"},
        desc="a one-hidden-layer MLP on the flattened input",
        cats={"activation": ("relu", "tanh")}),
    "cnn": Method(
        "cnn", build_cnn,
        bounds={**{k: _SHARED_BOUNDS[k] for k in ("lr", "weight_decay", "epochs",
                "batch_size", "dropout")}, "channels": (8, 64)},
        baseline={"channels": 16, "lr": 0.01, "weight_decay": 0.0, "epochs": 5,
                  "batch_size": 64, "dropout": 0.0},
        desc="a small 2-layer convolutional network (conv-pool-conv-pool-linear)",
        needs_image=True),
}


# ---------------------------------------------------------------------------
# Shared, harness-owned training + scoring (identical protocol for every method)
# ---------------------------------------------------------------------------

def train_score(method: Method, cfg: dict, data: dict, seed: int,
                split: str = "va", n_train: Optional[int] = None) -> tuple[float, float]:
    """Really train `method` with `cfg` from scratch and return (accuracy, train_s).
    Seed controls init + shuffle (+ the train subset, if any). CPU + single thread
    => stationary. train_s is the wall-clock cost that feeds the Pareto cost axis."""
    Xtr, ytr = data["X_tr"], data["y_tr"]
    Xev, yev = data[f"X_{split[:2]}"], data[f"y_{split[:2]}"]
    in_shape = tuple(Xtr.shape[1:])
    n_classes = int(ytr.max().item()) + 1
    n = Xtr.shape[0]
    if n_train is not None and n_train < n:
        sub = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 12345))[:n_train]
        Xtr, ytr, n = Xtr[sub], ytr[sub], n_train

    torch.manual_seed(seed)
    t0 = time.time()
    model = method.build(cfg, in_shape, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]),
                           weight_decay=float(cfg["weight_decay"]))
    loss_fn = nn.CrossEntropyLoss()
    bs = int(cfg["batch_size"])
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(int(cfg["epochs"])):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
    train_s = time.time() - t0
    model.eval()
    with torch.no_grad():
        acc = (model(Xev).argmax(1) == yev).float().mean().item()
    return acc, train_s


# ---------------------------------------------------------------------------
# Cost axis: FLOPs (hardware-independent, deterministic, stationary). We count
# multiply-accumulates (MACs) via forward hooks, then approximate one training
# run as fwd + ~2x bwd = 3x forward, over (samples x epochs). Wall-clock is also
# returned by train_score() as on-this-hardware context. FLOPs are exact and
# portable; wall-clock is the "felt" cost but reshuffles on GPU.
# ---------------------------------------------------------------------------

def forward_macs(model: nn.Module, in_shape) -> int:
    """Multiply-accumulates for ONE forward pass on a single sample (Linear+Conv)."""
    total = [0]

    def lin_hook(m, inp, out):
        total[0] += m.in_features * m.out_features

    def conv_hook(m, inp, out):
        oC, oH, oW = out.shape[1], out.shape[2], out.shape[3]
        total[0] += oC * oH * oW * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    handles = []
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(lin_hook))
        elif isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(conv_hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros((1,) + tuple(in_shape), device=DEVICE))
    for h in handles:
        h.remove()
    return total[0]


def train_macs(method: Method, cfg: dict, data: dict,
               n_train: Optional[int] = None) -> int:
    """Total MACs to TRAIN this config: forward MACs/sample x 3 (fwd+~2x bwd)
    x samples x epochs. Deterministic -- no training, just one sizing forward."""
    Xtr = data["X_tr"]
    in_shape = tuple(Xtr.shape[1:])
    n_classes = int(data["y_tr"].max().item()) + 1
    n = Xtr.shape[0] if (n_train is None or n_train >= Xtr.shape[0]) else n_train
    model = method.build(cfg, in_shape, n_classes)
    return forward_macs(model, in_shape) * 3 * n * int(cfg["epochs"])


def methods_for_task(task: str) -> list[str]:
    """Applicable methods per task: images get the CNN, tabular does not."""
    data = task_data.load_task(task)
    is_image = data["X_tr"].dim() == 4
    return ["logreg", "mlp"] + (["cnn"] if is_image else [])
