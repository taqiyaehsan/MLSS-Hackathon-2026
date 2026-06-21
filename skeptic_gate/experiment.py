"""Experiment drivers that PERSIST results to skeptic_gate/results/ so every
figure is regenerated from logged numbers (HANDOFF step 19, no hand-edited data).

Three experiments:
  1. regime_grid     -- arms x sigma x p_good x outer-seeds, both world variants.
                        Feeds the 2D regime heatmap + the 1D crossover curves.
  2. replication_audit -- run the greedy arm, take every change it ACCEPTED, re-run
                        each R times, report how many "improvements" survive
                        re-testing (HANDOFF step 14, the centerpiece).
  3. portfolio_selection -- the multi-method extension: one pipeline per sub-idea,
                        then SELECT the best. Naive best-of-N vs a causal selection
                        gate, swept over selection noise and over N (the max-of-N
                        / winner's-curse story).
"""

from __future__ import annotations

import json
import os
import numpy as np

from synthetic import SyntheticConfig, run_arm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ARMS = ["greedy", "causal"]
SIGMAS = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24]
P_GOODS = [0.05, 0.10, 0.20, 0.35, 0.50]
WORLDS = {"unbounded": None, "ceiling": 0.5}


# ---------------------------------------------------------------------------
# 1. Regime grid
# ---------------------------------------------------------------------------

def regime_grid(budget: float = 120.0, n_outer: int = 40) -> dict:
    cells = []
    for world_name, ceiling in WORLDS.items():
        for sigma in SIGMAS:
            for p_good in P_GOODS:
                cfg = SyntheticConfig(sigma=sigma, p_good=p_good, ceiling=ceiling)
                row = {"world": world_name, "sigma": sigma, "p_good": p_good}
                for arm in ARMS:
                    rs = [run_arm(arm, cfg, budget, s) for s in range(n_outer)]
                    T = np.array([r.true_performance for r in rs])
                    fa = np.array([r.n_false_accepts for r in rs])
                    acc = np.array([r.n_accepted for r in rs])
                    row[f"{arm}_T_mean"] = float(T.mean())
                    row[f"{arm}_T_se"] = float(T.std(ddof=1) / np.sqrt(n_outer))
                    row[f"{arm}_false_mean"] = float(fa.mean())
                    row[f"{arm}_accepted_mean"] = float(acc.mean())
                row["causal_minus_greedy_T"] = row["causal_T_mean"] - row["greedy_T_mean"]
                cells.append(row)
    out = {"meta": {"budget": budget, "n_outer": n_outer, "arms": ARMS,
                    "sigmas": SIGMAS, "p_goods": P_GOODS, "worlds": list(WORLDS)},
           "cells": cells}
    path = os.path.join(RESULTS_DIR, "regime_grid.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[regime_grid] wrote {len(cells)} cells -> {path}")
    return out


# ---------------------------------------------------------------------------
# 2. Replication audit
# ---------------------------------------------------------------------------

def replication_audit(sigma: float = 0.08, p_good: float = 0.25,
                      ceiling=0.5, budget: float = 120.0,
                      n_outer: int = 30, R: int = 20, z: float = 1.0) -> dict:
    """Run greedy; for every accepted change, re-evaluate it R times and test
    whether the improvement survives (mirrors re-running a kept experiment).

    'survives_replication': replicated paired effect clears the ~z-SE band.
    'survives_truth': the change's realized effect was actually > 0 (ground truth).
    These should closely agree, validating the replication test itself.
    """
    cfg = SyntheticConfig(sigma=sigma, p_good=p_good, ceiling=ceiling)
    rng = np.random.default_rng(12345)

    kept = surv_rep = surv_truth = 0
    by_kind: dict[str, dict] = {}
    per_seed = []

    for s in range(n_outer):
        _, world = run_arm("greedy", cfg, budget, s, return_world=True)
        recs = world.accepted_records
        k_seed = sr_seed = st_seed = 0
        for rec in recs:
            kept += 1
            k_seed += 1
            realized = rec["realized_delta"]
            # re-evaluate R times: paired (candidate - baseline) measurements.
            # each measurement carries fresh independent noise of scale sigma.
            noise = rng.normal(0.0, sigma, size=(R, 2))
            effects = realized + noise[:, 0] - noise[:, 1]
            mean_eff = effects.mean()
            se = effects.std(ddof=1) / np.sqrt(R)
            survives = (mean_eff - z * se) > 0
            truth = realized > 0
            surv_rep += int(survives); sr_seed += int(survives)
            surv_truth += int(truth); st_seed += int(truth)

            bk = by_kind.setdefault(rec["kind"], {"kept": 0, "surv_rep": 0, "surv_truth": 0})
            bk["kept"] += 1
            bk["surv_rep"] += int(survives)
            bk["surv_truth"] += int(truth)
        per_seed.append({"seed": s, "kept": k_seed, "surv_rep": sr_seed, "surv_truth": st_seed})

    out = {
        "meta": {"sigma": sigma, "p_good": p_good, "ceiling": ceiling,
                 "budget": budget, "n_outer": n_outer, "R": R, "z": z},
        "kept": kept,
        "survive_replication": surv_rep,
        "survive_truth": surv_truth,
        "vanish_replication": kept - surv_rep,
        "frac_vanish_replication": (kept - surv_rep) / kept if kept else None,
        "by_kind": by_kind,
        "per_seed": per_seed,
    }
    path = os.path.join(RESULTS_DIR, "replication_audit.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[replication_audit] greedy kept {kept}; survive re-test "
          f"{surv_rep} ({100*surv_rep/kept:.0f}%); "
          f"vanish {kept-surv_rep} ({100*(kept-surv_rep)/kept:.0f}%) -> {path}")
    return out


# ---------------------------------------------------------------------------
# 3. Portfolio selection (the multi-method extension)
# ---------------------------------------------------------------------------

SEL_SIGMAS = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16]
SEL_NS = [2, 3, 5, 8, 12]
RULES = ["single", "fixedK", "causal"]


def _run_selection_rules(T_row, sigma, sel_rng, n_methods, K, z):
    from portfolio import select_single, select_fixed_k, select_causal
    return {
        "single": select_single(T_row, sigma, sel_rng),
        "fixedK": select_fixed_k(T_row, sigma, sel_rng, K),
        # equal budget to fixedK (N*K), but spent adaptively on close contests
        "causal": select_causal(T_row, sigma, sel_rng, budget_cap=n_methods * K, z=z),
    }


def _aggregate(T, sigma, sel_rng, n_methods, K, z):
    """One pass over all portfolio rows at a given selection noise. Returns the
    mean P(pick the truly-best method), TRUE-performance regret, winner's-curse
    overstatement, and selection budget, per rule."""
    n_seeds = T.shape[0]
    agg = {r: {"correct": 0, "regret": 0.0, "overstate": 0.0, "units": 0.0} for r in RULES}
    for s in range(n_seeds):
        Trow = T[s]
        true_best = int(np.argmax(Trow))
        best_val = float(Trow[true_best])
        picks = _run_selection_rules(Trow, sigma, sel_rng, n_methods, K, z)
        for r, (w, units, obs) in picks.items():
            agg[r]["correct"] += int(w == true_best)
            agg[r]["regret"] += best_val - float(Trow[w])
            agg[r]["overstate"] += obs - float(Trow[w])
            agg[r]["units"] += units
    row = {"sigma": sigma, "n_methods": n_methods}
    for r in RULES:
        row[f"{r}_p_correct"] = agg[r]["correct"] / n_seeds
        row[f"{r}_regret"] = agg[r]["regret"] / n_seeds
        row[f"{r}_overstate"] = agg[r]["overstate"] / n_seeds
        row[f"{r}_units"] = agg[r]["units"] / n_seeds
    return row


def portfolio_selection(n_methods: int = 5, n_seeds: int = 300, K: int = 5,
                        z: float = 1.0, within_arm: str = "causal",
                        budget_per_method: float = 60.0,
                        fixed_sigma_for_Nsweep: float = 0.04) -> dict:
    from portfolio import build_quality_matrix

    # --- sigma sweep at fixed N (the selection regime curve) ---
    T, _ = build_quality_matrix(n_methods, n_seeds, within_arm=within_arm,
                                budget_per_method=budget_per_method)
    sel_rng = np.random.default_rng(777)
    sigma_rows = [_aggregate(T, sigma, sel_rng, n_methods, K, z) for sigma in SEL_SIGMAS]

    # --- N sweep at fixed selection noise (max-of-N inflation grows with N) ---
    n_rows = []
    for n in SEL_NS:
        Tn, _ = build_quality_matrix(n, n_seeds, within_arm=within_arm,
                                     budget_per_method=budget_per_method)
        n_rows.append(_aggregate(Tn, fixed_sigma_for_Nsweep, sel_rng, n, K, z))

    out = {
        "meta": {"n_methods": n_methods, "n_seeds": n_seeds, "K": K, "z": z,
                 "within_arm": within_arm, "budget_per_method": budget_per_method,
                 "sel_sigmas": SEL_SIGMAS, "sel_Ns": SEL_NS, "rules": RULES,
                 "fixed_sigma_for_Nsweep": fixed_sigma_for_Nsweep,
                 "note": "single budget=N; fixedK and causal budget=N*K (equal)."},
        "sigma_sweep": sigma_rows,
        "n_sweep": n_rows,
    }
    path = os.path.join(RESULTS_DIR, "portfolio_selection.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    # console headline at the highest-noise cell
    hi = sigma_rows[-1]
    print(f"[portfolio_selection] N={n_methods}, sel-sigma={hi['sigma']}: "
          f"P(pick best) naive={hi['single_p_correct']:.2f} vs "
          f"causal={hi['causal_p_correct']:.2f}; "
          f"regret naive={hi['single_regret']:.4f} vs causal={hi['causal_regret']:.4f}; "
          f"winner's-curse overstatement naive={hi['single_overstate']:.4f} "
          f"(equal budget to fixedK) -> {path}")
    return out


if __name__ == "__main__":
    print("Running regime grid (this is the headline data)...")
    regime_grid()
    print("\nRunning replication audit...")
    replication_audit()
    print("\nRunning portfolio selection (multi-method extension)...")
    portfolio_selection()
    print("\nDone. Now run: python plots.py")
