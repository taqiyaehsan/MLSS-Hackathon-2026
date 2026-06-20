"""Sweep noise (sigma) for BOTH world variants -- unbounded and ceiling
(diminishing returns) -- to locate the greedy vs causal crossover in each."""
import numpy as np
from synthetic import SyntheticConfig, run_arm

BUDGET = 120.0
N_OUTER = 60
SIGMAS = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24]


def mean_T(arm, cfg, budget, n_outer):
    rs = [run_arm(arm, cfg, budget, s) for s in range(n_outer)]
    Tf = np.array([r.true_performance for r in rs])
    fa = np.array([r.n_false_accepts for r in rs])
    return Tf.mean(), Tf.std(ddof=1) / np.sqrt(n_outer), fa.mean()


def sweep(ceiling, p_good=0.25):
    label = "UNBOUNDED" if ceiling is None else f"CEILING={ceiling}"
    print(f"\n===== {label}  (p_good={p_good}, budget={BUDGET}, outer={N_OUTER}) =====")
    print(f"{'sigma':>7} | {'T_greedy':>9}{'±':>7} | {'T_causal':>9}{'±':>7} | "
          f"{'winner':>8} | {'g.false':>8}{'c.false':>8}")
    first_causal = None
    for sigma in SIGMAS:
        cfg = SyntheticConfig(sigma=sigma, p_good=p_good, ceiling=ceiling)
        Tg, sg, gfa = mean_T("greedy", cfg, BUDGET, N_OUTER)
        Tc, sc, cfa = mean_T("causal", cfg, BUDGET, N_OUTER)
        winner = "causal" if Tc > Tg else "greedy"
        if winner == "causal" and first_causal is None:
            first_causal = sigma
        print(f"{sigma:>7.3f} | {Tg:>9.4f}{sg:>7.4f} | {Tc:>9.4f}{sc:>7.4f} | "
              f"{winner:>8} | {gfa:>8.1f}{cfa:>8.1f}")
    print(f"  -> causal first wins on TRUE perf at sigma >= "
          f"{first_causal if first_causal else 'never (in range)'}")


def main():
    sweep(ceiling=None)
    sweep(ceiling=0.5)


if __name__ == "__main__":
    main()
