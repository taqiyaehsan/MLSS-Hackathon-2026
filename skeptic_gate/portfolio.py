"""Portfolio of method-pipelines + the causal SELECTION gate.

This is the team's multi-method extension (discussed 2026-06-21). The background/
research files for a task list several bounded SUB-IDEAS ("methods"). Instead of
one agent loop, we instantiate ONE agent-loop pipeline per sub-idea, let each
improve its own method to a fixed budget, then SELECT the best method to ship.

The catch -- and why this is OUR project, not just a stronger baseline:
"pick the best of N by its final score" is a MAX-OF-N problem. With N methods each
measured under noise, the apparent winner is biased toward the LUCKY method, not
the truly best (the "winner's curse"). Multi-start does not escape false-
acceptance; it AMPLIFIES it. So the portfolio is the EXPLORATION layer and the
skeptic gate is the SELECTION layer:

  * naive selection  : 1 noisy final eval per method, argmax            (autoresearch-style)
  * fixed-K selection: K evals per method, argmax of the mean
  * causal selection : adaptively re-test the close contenders (a race), confirm
                       the leader's advantage clears a ~z-SE band, then ship it.

We REUSE synthetic.run_arm unchanged for the within-pipeline loop (its gates and
budget are exactly as in the single-method experiments). The only new machinery is
the cross-pipeline selection on top. Because the synthetic knows each method's
HIDDEN true performance T_m, we can score selection by TRUE-performance regret and
by how often the gate recovers the genuinely best method.

Design note (isolation): to study the SELECTION layer cleanly, the within-loop is
run at a fixed low noise so each method settles to a stable hidden quality T_m
(distinct per method via its latent ceiling). The selection NOISE is then swept
independently -- this is the dial that creates the max-of-N problem. The within-
loop regime (greedy vs causal vs noise) is already characterised by experiment.py;
here we hold it fixed and vary only what selection sees.
"""

from __future__ import annotations

import numpy as np

from synthetic import SyntheticConfig, run_arm


# Labels tie to the rainfall background.txt framework (WeatherFusionNet et al.);
# their TRUE qualities are drawn per portfolio run, because a-priori you do NOT
# know which sub-idea wins -- that uncertainty is the whole point of selection.
DEFAULT_METHODS = [
    "convlstm_temporal_head",
    "ensemble_output_heads",
    "loss_pos_weight_reweight",
    "dropout_branch_sweep",
    "filter_width_widen",
    "sat2rad_aux_loss",
    "phydnet_style_recurrence",
    "controller_weight_map",
    "skip_connection_depth",
    "augmentation_mix",
    "lr_schedule_warmup",
    "channel_attention",
]


# ---------------------------------------------------------------------------
# tiny stats (kept local so this module reads standalone)
# ---------------------------------------------------------------------------

def _mean(xs) -> float:
    return sum(xs) / len(xs)


def _se(xs) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return (v / n) ** 0.5


# ---------------------------------------------------------------------------
# Step 1: run one pipeline per method -> hidden true quality T_m
# ---------------------------------------------------------------------------

def build_quality_matrix(n_methods: int, n_seeds: int, *, within_arm: str = "causal",
                         within_sigma: float = 0.02, p_good: float = 0.25,
                         budget_per_method: float = 60.0,
                         q_mu: float = 0.40, q_spread: float = 0.10,
                         seed0: int = 0, quality_seed: int = 20260621):
    """For each portfolio run (row) and method, run the within-pipeline loop and
    record its final HIDDEN true performance T_m.

    Per the team's choice, all methods in a single portfolio row share the SAME
    within-loop seed (the candidate stream is identical) -- so the only structural
    difference between methods is their latent ceiling (their promise). This
    isolates "which sub-idea is genuinely better" from loop luck.
    Returns (T[n_seeds, n_methods], ceilings[n_seeds, n_methods]).
    """
    qrng = np.random.default_rng(quality_seed)
    ceilings = np.clip(qrng.normal(q_mu, q_spread, size=(n_seeds, n_methods)), 0.05, None)
    T = np.zeros((n_seeds, n_methods))
    for s in range(n_seeds):
        for m in range(n_methods):
            cfg = SyntheticConfig(sigma=within_sigma, p_good=p_good,
                                  ceiling=float(ceilings[s, m]))
            res = run_arm(within_arm, cfg, budget_per_method, outer_seed=seed0 + s)
            T[s, m] = res.true_performance
    return T, ceilings


# ---------------------------------------------------------------------------
# Step 2: selection rules over the noisy FINAL evals of the N methods
# ---------------------------------------------------------------------------

def select_single(T_row: np.ndarray, sigma: float, rng) -> tuple[int, int, float]:
    """One noisy final eval per method, pick the argmax. Budget = N.
    Returns (winner, units_spent, observed_score_the_pick_was_made_on)."""
    obs = T_row + rng.normal(0.0, sigma, size=len(T_row))
    w = int(np.argmax(obs))
    return w, len(T_row), float(obs[w])


def select_fixed_k(T_row: np.ndarray, sigma: float, rng, K: int) -> tuple[int, int, float]:
    """K evals per method, pick argmax of the means. Budget = N*K."""
    n = len(T_row)
    obs = T_row[None, :] + rng.normal(0.0, sigma, size=(K, n))
    means = obs.mean(axis=0)
    w = int(np.argmax(means))
    return w, n * K, float(means[w])


def select_causal(T_row: np.ndarray, sigma: float, rng, *, k0: int = 2, z: float = 1.0,
                  budget_cap: int | None = None) -> tuple[int, int, float]:
    """Adaptive racing selection gate. Start with k0 evals per method; repeatedly
    drop any method whose upper band (mean + z*SE) falls below the leader's lower
    band (mean - z*SE), and spend another eval on every method still in contention.
    Stop when one method remains or the selection budget cap is hit. This is the
    cross-pipeline analogue of CausalPolicy: confirm cheap, escalate when uncertain.
    """
    n = len(T_row)
    scores: list[list[float]] = [[] for _ in range(n)]
    units = 0
    cap = budget_cap if budget_cap is not None else 10 ** 9

    def draw(m: int) -> None:
        nonlocal units
        scores[m].append(float(T_row[m] + rng.normal(0.0, sigma)))
        units += 1

    for m in range(n):
        for _ in range(k0):
            draw(m)
    alive = set(range(n))
    while len(alive) > 1 and units < cap:
        means = {m: _mean(scores[m]) for m in alive}
        ses = {m: _se(scores[m]) for m in alive}
        leader = max(alive, key=lambda m: means[m])
        lb_leader = means[leader] - z * ses[leader]
        drop = {m for m in alive if m != leader and means[m] + z * ses[m] < lb_leader}
        alive -= drop
        if len(alive) <= 1:
            break
        for m in list(alive):
            if units >= cap:
                break
            draw(m)
    winner = max(range(n), key=lambda m: _mean(scores[m]))
    return winner, units, float(_mean(scores[winner]))
