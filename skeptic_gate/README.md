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
  counts ALL spend incl. gate overhead), `run_loop`, and the **cost lever**
  `Fidelity` (a cost/accuracy operating point: a cheaper fidelity is charged
  `cost`<1 budget unit per eval and is noisier; presets `FULL`, `CHEAP`). Policies
  charge `eval_cost` so a fixed budget funds more cheap evals than full ones.
  Task adapters bind it: unlearning -> `MU_NUM_MODELS` env (fewer inner models);
  rainfall -> `max_epochs`/data fraction (when that adapter is built); synthetic ->
  a `sigma_mult` that inflates eval noise so the trade is testable offline.
- `synthetic.py` — controllable toy world. Dials: `sigma` (measurement noise),
  `p_good` (signal base-rate), `p_broken`, effect sizes, and `ceiling`
  (None = unbounded; set = diminishing returns). `_realized()` applies the
  ceiling consistently in BOTH `evaluate()` and `on_accept()`.
- `portfolio.py` — the **multi-method extension** + **causal selection gate**.
  Instantiate one pipeline per bounded sub-idea ("method"), improve each to budget
  (reusing `run_arm` unchanged), then SELECT the best. Selection rules: `single`
  (naive best-of-N, 1 eval/method), `fixed_k` (uniform re-test), `causal` (adaptive
  racing — re-test only the close contenders, confirm the leader clears a ~z-SE
  band). Picking the best of N noisy methods is a max-of-N problem; the gate is the
  fix.
- `experiment.py` — runs + PERSISTS results to `results/`:
  `regime_grid` (arms × sigma × p_good × seeds, both worlds),
  `replication_audit` (greedy keeps N; how many survive re-testing), and
  `portfolio_selection` (selection rules × selection-noise × N).
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
- **Portfolio selection (multi-method extension):** picking the best of N methods
  by a single noisy score is a max-of-N trap. Over N=5 methods and 300 runs:
  - **Winner's curse:** the naive pick's *reported* score overstates its true
    performance by up to **0.175** (several × the real gaps between methods, which
    are ~0.02–0.03); re-testing **halves** the inflation.
  - **Regret:** the causal selection gate roughly **halves** true-performance
    regret across the noisy regime (e.g. sigma=0.16: 0.0083 → 0.0048).
  - **Max-of-N tax:** P(pick the truly-best method) collapses for naive as N grows
    (0.56 → 0.14 from N=2→12); the gate holds it up (0.59 → 0.22).
  - **Equal budget:** `single` spends N evals; `fixed_k` and `causal` both spend
    N×K. At equal budget the adaptive `causal` rule edges out uniform `fixed_k` at
    high noise / large N — but the dominant win is "re-test at all," not the
    adaptive refinement (reported honestly).

## Integrity notes
- Both `unbounded` and `ceiling` worlds are reported so the regime boundary is
  not an artifact of one modeling choice.
- The synthetic is NOT tuned to make the gate win; greedy in fact wins most of
  the grid. The contribution is the *characterization* (when skepticism pays)
  plus the *audit*, per the project's honest framing.
- `_realized()` guarantees the measured score and the realized gain never
  disagree (no sleight of hand in the diminishing-returns model).
- The portfolio experiment **isolates the selection layer**: the within-pipeline
  loop runs at fixed low noise so each method settles to a stable hidden quality,
  and only the *selection* noise is swept (that is the dial that creates the
  max-of-N problem). The within-loop regime is already characterised separately by
  `regime_grid`. `causal` and `fixed_k` are compared at **equal budget** (N×K), so
  the gate's advantage is not bought with extra evals.
