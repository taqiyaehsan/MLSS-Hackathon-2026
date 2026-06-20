# skeptic_gate — the gate logic + synthetic control

A skeptic for an autonomous ML-research agent: a **coherence** gate (cull broken
changes before evaluating) and a **causal acceptance** gate (re-test a candidate
over seeds; accept only if the gain clears the noise band). This package develops
and validates the gate logic against a fully-controllable **synthetic control**,
then the SAME gate code (`gates.py`) drives the real MLRC task.

## Files
- `gates.py` — **task-agnostic** accept policies + generic loop:
  `GreedyPolicy` (the autoresearch baseline: single noisy number),
  `CausalPolicy` (adaptive sequential confirmation, ~1-SE band, k0=2..k_max=6),
  `CoherenceWrapper` (cheap pre-eval cull), `Budget` (equal-budget accounting that
  counts ALL spend incl. gate overhead), `run_loop`.
- `synthetic.py` — controllable toy world. Dials: `sigma` (measurement noise),
  `p_good` (signal base-rate), `p_broken`, effect sizes, and `ceiling`
  (None = unbounded; set = diminishing returns). `_realized()` applies the
  ceiling consistently in BOTH `evaluate()` and `on_accept()`.
- `experiment.py` — runs + PERSISTS results to `results/`:
  `regime_grid` (arms × sigma × p_good × seeds, both worlds) and
  `replication_audit` (greedy keeps N; how many survive re-testing).
- `plots.py` — regenerates ALL figures from `results/*.json` (no hand-edited numbers).
- `sanity.py` — quick 1-D sigma sweep for both worlds.

## Reproduce
```bash
source ../.venv/bin/activate
python experiment.py   # writes results/regime_grid.json, results/replication_audit.json
python plots.py        # writes results/figs/*.png
```

## Headline findings (synthetic)
- **Crossover is robust across both worlds:** greedy wins at low/moderate noise
  (throughput captures real gains); causal wins at high noise (greedy's accepted
  "wins" are luck and erode true performance). The realistic ceiling world moves
  the crossover earlier (sigma≈0.24 → ≈0.12) but the direction is unchanged.
- **Causal cuts false-acceptances ~5–10× in every regime** — reliability is
  regime-independent; only the net-performance tradeoff is regime-dependent.
- **Replication audit:** greedy kept 164 "improvements"; **60% (98) do not
  survive re-testing.** Truly-null changes almost all vanish; genuine gains
  survive far more often.

## Integrity notes
- Both `unbounded` and `ceiling` worlds are reported so the regime boundary is
  not an artifact of one modeling choice.
- The synthetic is NOT tuned to make the gate win; greedy in fact wins most of
  the grid. The contribution is the *characterization* (when skepticism pays)
  plus the *audit*, per the project's honest framing.
- `_realized()` guarantees the measured score and the realized gain never
  disagree (no sleight of hand in the diminishing-returns model).
