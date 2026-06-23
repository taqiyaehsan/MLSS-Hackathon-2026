from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_COLS = [
    "stock_id",
    "date_id",
    "seconds_in_bucket",
    "imbalance_size",
    "imbalance_buy_sell_flag",
    "reference_price",
    "matched_size",
    "far_price",
    "near_price",
    "bid_price",
    "bid_size",
    "ask_price",
    "ask_size",
    "wap",
    "target",
]

DTYPES = {
    "stock_id": "int16",
    "date_id": "int16",
    "seconds_in_bucket": "int16",
    "imbalance_size": "float32",
    "imbalance_buy_sell_flag": "int8",
    "reference_price": "float32",
    "matched_size": "float32",
    "far_price": "float32",
    "near_price": "float32",
    "bid_price": "float32",
    "bid_size": "float32",
    "ask_price": "float32",
    "ask_size": "float32",
    "wap": "float32",
    "target": "float32",
}

FEATURE_COLS = [
    "stock_id",
    "seconds_in_bucket",
    "imbalance_size",
    "imbalance_buy_sell_flag",
    "reference_price",
    "matched_size",
    "far_price",
    "near_price",
    "bid_price",
    "bid_size",
    "ask_price",
    "ask_size",
    "wap",
    "spread",
    "mid_price",
    "rel_spread",
    "depth_total",
    "depth_imbalance",
    "signed_imbalance",
    "imbalance_ratio",
    "signed_imbalance_ratio",
    "wap_minus_ref",
    "wap_minus_mid",
    "ref_minus_mid",
    "log_matched_size",
    "log_imbalance_size",
    "log_bid_size",
    "log_ask_size",
    "seconds_norm",
    "stock_norm",
]


def reservoir_update(current, new_rows, capacity, rng):
    if new_rows.empty:
        return current

    new_rows = new_rows.copy()
    new_rows["_u"] = rng.random(len(new_rows))

    if current is None or current.empty:
        combo = new_rows
    else:
        combo = pd.concat([current, new_rows], ignore_index=True)

    if len(combo) > capacity:
        combo = combo.nlargest(capacity, "_u")

    return combo.reset_index(drop=True)


def add_features(df):
    eps = 1e-6
    df = df.copy()

    df["spread"] = df["ask_price"] - df["bid_price"]
    df["mid_price"] = 0.5 * (df["ask_price"] + df["bid_price"])
    df["rel_spread"] = df["spread"] / (df["mid_price"].abs() + eps)

    df["depth_total"] = df["bid_size"] + df["ask_size"]
    df["depth_imbalance"] = (df["bid_size"] - df["ask_size"]) / (df["depth_total"] + eps)

    df["signed_imbalance"] = df["imbalance_size"] * df["imbalance_buy_sell_flag"]
    df["imbalance_ratio"] = df["imbalance_size"] / (
        df["matched_size"] + df["imbalance_size"] + eps
    )
    df["signed_imbalance_ratio"] = df["signed_imbalance"] / (
        df["matched_size"] + df["imbalance_size"] + eps
    )

    df["wap_minus_ref"] = df["wap"] - df["reference_price"]
    df["wap_minus_mid"] = df["wap"] - df["mid_price"]
    df["ref_minus_mid"] = df["reference_price"] - df["mid_price"]

    df["log_matched_size"] = np.log1p(df["matched_size"].clip(lower=0))
    df["log_imbalance_size"] = np.log1p(df["imbalance_size"].clip(lower=0))
    df["log_bid_size"] = np.log1p(df["bid_size"].clip(lower=0))
    df["log_ask_size"] = np.log1p(df["ask_size"].clip(lower=0))

    df["seconds_norm"] = df["seconds_in_bucket"] / 540.0
    df["stock_norm"] = df["stock_id"] / max(1.0, float(df["stock_id"].max()))

    df["label"] = (df["target"] > 0).astype(np.int64)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/optiver/train.csv")
    ap.add_argument("--out", default="data/optiver/optiver_direction_hpo.npz")
    ap.add_argument("--n-train", type=int, default=7200)
    ap.add_argument("--n-val", type=int, default=2400)
    ap.add_argument("--n-test", type=int, default=2400)
    ap.add_argument("--chunksize", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    print(f"[1/3] Scanning date_id from {csv_path} ...")
    dates = set()
    for ch in pd.read_csv(
        csv_path,
        usecols=["date_id"],
        dtype={"date_id": "int16"},
        chunksize=args.chunksize,
    ):
        dates.update(ch["date_id"].unique().tolist())

    dates = np.array(sorted(dates))
    n_dates = len(dates)

    train_dates = set(dates[: int(0.60 * n_dates)])
    val_dates = set(dates[int(0.60 * n_dates) : int(0.80 * n_dates)])
    test_dates = set(dates[int(0.80 * n_dates) :])

    print(
        f"date splits: train={len(train_dates)}, "
        f"val={len(val_dates)}, test={len(test_dates)}"
    )

    rng = np.random.default_rng(args.seed)
    samples = {"tr": None, "va": None, "te": None}
    caps = {"tr": args.n_train, "va": args.n_val, "te": args.n_test}

    print("[2/3] Chunked reservoir sampling ...")
    for k, ch in enumerate(
        pd.read_csv(
            csv_path,
            usecols=RAW_COLS,
            dtype=DTYPES,
            chunksize=args.chunksize,
        )
    ):
        ch = ch.dropna(subset=["target"])

        tr = ch[ch["date_id"].isin(train_dates)]
        va = ch[ch["date_id"].isin(val_dates)]
        te = ch[ch["date_id"].isin(test_dates)]

        samples["tr"] = reservoir_update(samples["tr"], tr, caps["tr"], rng)
        samples["va"] = reservoir_update(samples["va"], va, caps["va"], rng)
        samples["te"] = reservoir_update(samples["te"], te, caps["te"], rng)

        if k % 20 == 0:
            print(
                f"  chunk {k:4d}: "
                f"tr={0 if samples['tr'] is None else len(samples['tr'])}, "
                f"va={0 if samples['va'] is None else len(samples['va'])}, "
                f"te={0 if samples['te'] is None else len(samples['te'])}"
            )

    tr = samples["tr"].drop(columns=["_u"]).reset_index(drop=True)
    va = samples["va"].drop(columns=["_u"]).reset_index(drop=True)
    te = samples["te"].drop(columns=["_u"]).reset_index(drop=True)

    print("[3/3] Feature engineering sampled rows only ...")
    tr = add_features(tr)
    va = add_features(va)
    te = add_features(te)

    X_tr = tr[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    X_va = va[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    X_te = te[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)

    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)

    def impute(X):
        X = X.copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(med, np.where(bad)[1])
        return X

    X_tr = impute(X_tr)
    X_va = impute(X_va)
    X_te = impute(X_te)

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_va = scaler.transform(X_va).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    y_tr = tr["label"].to_numpy(np.int64)
    y_va = va["label"].to_numpy(np.int64)
    y_te = te["label"].to_numpy(np.int64)

    np.savez_compressed(
        out_path,
        X_tr=X_tr,
        y_tr=y_tr,
        X_va=X_va,
        y_va=y_va,
        X_te=X_te,
        y_te=y_te,
        feature_cols=np.array(FEATURE_COLS),
    )

    print(f"saved -> {out_path}")
    print(f"X_tr={X_tr.shape}, X_va={X_va.shape}, X_te={X_te.shape}")
    print(
        f"positive rates: "
        f"train={y_tr.mean():.3f}, val={y_va.mean():.3f}, test={y_te.mean():.3f}"
    )


if __name__ == "__main__":
    main()
