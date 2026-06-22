"""Data loading for the local MLRC-style tasks. Owned by the HARNESS, never by the
agent's method code. Each loader returns a fixed train/val/test split (test held
out from the whole loop) as CPU tensors.

Working sets are deliberately subsampled: the point is a real, cheap, STATIONARY
train-and-score loop, not a leaderboard model. A whole-dataset CNN would make the
replication audit (many seeds x many evals) explode for no gain to the gate story.

  fashionmnist -- raw images (N, 1, 28, 28) in [0,1], 10 classes  (CNN possible)
  magic        -- tabular (N, 10) z-scored, 2 classes (gamma signal vs hadron bg)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_FMNIST_ROOT = Path(__file__).resolve().parent / "_data_fmnist"

# Cache loaded splits so repeated harness calls in one process don't re-read disk.
_CACHE: dict[str, dict] = {}


def _stratified_split(X, y, n_total, seed=0, regression=False):
    """Subsample to a fixed n_total, then a fixed 60/20/20 split. Stratified for
    classification; plain (no stratify) for regression (continuous targets)."""
    from sklearn.model_selection import train_test_split
    y = np.asarray(y)
    strat = None if regression else y
    if n_total is not None and n_total < len(y):
        X, _, y, _ = train_test_split(X, y, train_size=n_total, random_state=seed,
                                      stratify=strat)
        strat = None if regression else y
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=strat)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=seed,
        stratify=None if regression else y_tmp)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def _pack(X_tr, y_tr, X_va, y_va, X_te, y_te, regression=False):
    f = lambda A: torch.from_numpy(np.asarray(A, dtype=np.float32))
    yt = (lambda A: torch.from_numpy(np.asarray(A, dtype=np.float32))) if regression \
        else (lambda A: torch.from_numpy(np.asarray(A, dtype=np.int64)))
    return {"X_tr": f(X_tr), "y_tr": yt(y_tr), "X_va": f(X_va), "y_va": yt(y_va),
            "X_te": f(X_te), "y_te": yt(y_te)}


def load_fashionmnist(n_total: int = 12_000) -> dict:
    """FashionMNIST as raw (N,1,28,28) images in [0,1] so the agent can write a CNN."""
    from torchvision import datasets
    tr = datasets.FashionMNIST(root=str(_FMNIST_ROOT), train=True, download=False)
    X = (tr.data.numpy().astype(np.float32) / 255.0)[:, None, :, :]   # (60000,1,28,28)
    y = tr.targets.numpy()
    parts = _stratified_split(X, y, n_total)
    return _pack(*parts)


def load_magic(n_total: int = 10_000) -> dict:
    """MAGIC Gamma Telescope: 10 z-scored features; gamma 'g'->0, hadron 'h'->1.
    Scaler is fit on TRAIN only (no leakage)."""
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    d = fetch_openml("MagicTelescope", version=1, as_frame=False)
    X = d.data.astype(np.float32)
    y = (np.asarray(d.target) == "h").astype(np.int64)
    X_tr, y_tr, X_va, y_va, X_te, y_te = _stratified_split(X, y, n_total)
    sc = StandardScaler().fit(X_tr)
    return _pack(sc.transform(X_tr), y_tr, sc.transform(X_va), y_va,
                 sc.transform(X_te), y_te)


def load_diabetes() -> dict:
    """sklearn diabetes: REGRESSION template (10 features -> continuous target).
    Bundled, no download. The 'example_regression' task; a shape teammates copy."""
    from sklearn.datasets import load_diabetes as _ld
    from sklearn.preprocessing import StandardScaler
    d = _ld()
    X = d.data.astype(np.float32)
    y = d.target.astype(np.float32)
    X_tr, y_tr, X_va, y_va, X_te, y_te = _stratified_split(X, y, None, regression=True)
    sc = StandardScaler().fit(X_tr)
    return _pack(sc.transform(X_tr), y_tr, sc.transform(X_va), y_va,
                 sc.transform(X_te), y_te, regression=True)


# Loader key MUST match the TaskSpec name in local_task.py (the harness loads data
# by task name). "example_regression" is the diabetes regression template.
LOADERS = {"fashionmnist": load_fashionmnist, "magic": load_magic,
           "example_regression": load_diabetes}


def load_task(name: str) -> dict:
    if name not in LOADERS:
        raise ValueError(f"unknown task {name!r}; choose from {list(LOADERS)}")
    if name not in _CACHE:
        _CACHE[name] = LOADERS[name]()
    return _CACHE[name]
