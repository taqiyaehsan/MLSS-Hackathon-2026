from pathlib import Path

p = Path("skeptic_gate/hpo_task.py")
s = p.read_text()

if "optiver_direction" in s:
    raise SystemExit(
        "optiver_direction already exists. Run `git checkout -- skeptic_gate/hpo_task.py` first."
    )

backup = p.with_suffix(".py.bak")
backup.write_text(s)
print(f"Backed up original file to {backup}")

loader = r'''
# ---------------------------------------------------------------------------
# Optiver Trading at the Close: binary direction classification.
#
# This HPO loader intentionally reads a small cached .npz file, not the raw
# Kaggle CSV. Build the cache once with:
#
#   python prepare_optiver_direction_hpo.py
#
# Label: 1 if the Optiver target is positive, else 0.
# ---------------------------------------------------------------------------
def _load_optiver_direction():
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "optiver" / "optiver_direction_hpo.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached Optiver tensor file: {path}\n"
            "Run from the repo root:\n"
            "  python prepare_optiver_direction_hpo.py "
            "--csv data/optiver/train.csv "
            "--out data/optiver/optiver_direction_hpo.npz"
        )

    z = np.load(path)

    print(
        f"[optiver_direction] loaded {path}; "
        f"train={z['X_tr'].shape}, val={z['X_va'].shape}, test={z['X_te'].shape}; "
        f"positive rate train/val/test="
        f"{z['y_tr'].mean():.3f}/{z['y_va'].mean():.3f}/{z['y_te'].mean():.3f}"
    )

    return {
        "X_tr": torch.from_numpy(z["X_tr"].astype(np.float32)),
        "y_tr": torch.from_numpy(z["y_tr"].astype(np.int64)),
        "X_va": torch.from_numpy(z["X_va"].astype(np.float32)),
        "y_va": torch.from_numpy(z["y_va"].astype(np.int64)),
        "X_te": torch.from_numpy(z["X_te"].astype(np.float32)),
        "y_te": torch.from_numpy(z["y_te"].astype(np.int64)),
    }

'''

entry = r'''
    "optiver_direction": DatasetSpec(
        "optiver_direction",
        _load_optiver_direction,
        30,
        2,
        "Optiver Trading at the Close: binary classification of whether the "
        "future closing-auction price movement target is positive, using "
        "market microstructure and auction-book features",
        [
            ("low (full data)", Fidelity("full", 1.0, {"train_subset": None})),
            ("med (1200 samples)", Fidelity("med", 1.0, {"train_subset": 1200})),
            ("high (300 samples)", Fidelity("high", 1.0, {"train_subset": 300})),
        ],
    ),
'''

marker_loader = "@dataclass\nclass DatasetSpec:"
if marker_loader not in s:
    raise RuntimeError("Could not find DatasetSpec marker.")

s = s.replace(marker_loader, loader + "\n" + marker_loader, 1)

marker_after = "# Active-dataset module state"
pos_after = s.find(marker_after)
if pos_after == -1:
    raise RuntimeError("Could not find Active-dataset module state marker.")

before = s[:pos_after]
after = s[pos_after:]

close_pos = before.rfind("}")
if close_pos == -1:
    raise RuntimeError("Could not find closing brace of DATASETS.")

s = before[:close_pos] + entry + "\n}" + before[close_pos + 1:] + after

s = s.replace(
    "python hpo_task.py [digits|fmnist|magic]",
    "python hpo_task.py [digits|fmnist|magic|optiver_direction]",
)

p.write_text(s)
print("Patched skeptic_gate/hpo_task.py with cached optiver_direction loader.")
