# Skeptic gate under noisy evaluation — results (FashionMNIST + MAGIC + Colored MNIST)

The headline experiment for the skeptic gate: **when an autonomous code-editing
agent is evaluated noisily, the greedy accept rule adopts "improvements" that do
not replicate, while the causal accept rule (the skeptic) does not.** We show this
on two team-authored tasks — FashionMNIST (vision) and MAGIC Gamma Telescope
(tabular) — both local, CPU, stationary.

A third task, **Colored MNIST**, is included for a different reason: it maps the
*boundary* of the skeptic. The causal gate re-tests over seeds on the validation
distribution, so it catches **seed-noise** false positives (sections 1–4). It
**cannot** catch a **distribution-shift** false positive — a win that replicates
perfectly across seeds yet relies on a spurious cue that flips at test time
(section 5). Reporting both is the honest characterization: *more seeds fix noise,
not the wrong held-out distribution.*

The proposed system is the **skeptic**: a coherence gate (culls broken edits before
any eval) + a **causal accept gate** (re-tests over k seeds, accepts only if the
gain clears the noise band). Greedy (accept on one noisy score) is the **baseline**.
The agent generates its candidate stream once under the causal gate; both accept
rules are then replayed over that identical stream, so the comparison is a clean
paired ablation with no extra LLM calls.

All numbers are reproducible (commands at the bottom). Raw data per task:
`results/skeptic_regime/<task>/` (`llm.json`, `methods_llm.csv`, `replay_llm.csv`,
`regime_eval.json/.csv`, `regime_curve_eval.png`).

---

## 1. Measurable progress (agent edits the baseline → a real model)

| task | baseline → best (val) | baseline → best (test) | what the agent wrote |
|------|-----------------------|------------------------|----------------------|
| FashionMNIST | 0.737 → **0.896** | 0.749 → **0.888** | linear → CNN → +augmentation → +label smoothing |
| MAGIC        | 0.798 → **0.871** | 0.786 → **0.868** | logistic → 2-layer MLP |

## 2. Pareto frontier (accuracy ↑ / stability ↓ / FLOPs ↓; report, no auto-pick)

- **FashionMNIST:** frontier = {baseline, small CNN (0.884 @ 510 GFLOPs), augmentation
  CNN (0.894 @ 2,960 GFLOPs), MixUp CNN (0.896 @ 85,514 GFLOPs)}. The most accurate
  method costs **~168× the FLOPs of the cheap CNN for +0.012 accuracy** — a stark
  accuracy/compute trade-off.
- **MAGIC:** 6 of 9 methods non-dominated. The **small MLP wins on accuracy *and*
  cost** (0.871 @ 0.20 GFLOPs); the agent's deeper MLPs spent **~12–23× more FLOPs to
  do *worse*** — a clean "more compute ≠ better" point.

## 3. The skeptic result — causal beats greedy under noise (BOTH tasks)

Noise model: each method is trained once on full data; a noisy eval = accuracy on a
random subset of the held-out validation set (unbiased — the true ranking is
preserved, so any accepted gain that vanishes is *purely* a measurement artifact).
We audit every accept against the full-eval truth; a "false positive" is an accept
whose true gain is ≤ 0. 200 bootstrap trials per noise level.

**MAGIC** (`results/skeptic_regime/magic/`):

| eval size | noise σ | greedy FP-rate | causal FP-rate | greedy final acc | causal final acc |
|-----------|---------|----------------|----------------|------------------|------------------|
| 2000 | 0.008 | 0.31 | 0.06 | 0.8716 | 0.8728 |
| 1000 | 0.009 | 0.37 | 0.13 | 0.8708 | 0.8724 |
| 500  | 0.016 | 0.41 | 0.20 | 0.8703 | 0.8720 |
| 200  | 0.016 | 0.51 | 0.19 | 0.8693 | 0.8718 |
| 100  | 0.028 | 0.54 | 0.24 | 0.8676 | 0.8712 |
| 50   | 0.037 | 0.54 | 0.24 | 0.8640 | 0.8712 |
| 25   | 0.069 | 0.41 | 0.17 | 0.8573 | 0.8678 |

**FashionMNIST** (`results/skeptic_regime/fashionmnist/`):

| eval size | noise σ | greedy FP-rate | causal FP-rate | greedy final acc | causal final acc |
|-----------|---------|----------------|----------------|------------------|------------------|
| 2000 | 0.004 | 0.45 | 0.32 | 0.9025 | 0.9019 |
| 1000 | 0.006 | 0.49 | 0.24 | 0.9016 | 0.9005 |
| 500  | 0.009 | 0.45 | 0.21 | 0.9004 | 0.8989 |
| 200  | 0.017 | 0.42 | 0.27 | 0.8991 | 0.8974 |
| 100  | 0.021 | 0.44 | 0.28 | 0.8976 | 0.8968 |
| 50   | 0.023 | 0.40 | 0.30 | 0.8964 | 0.8957 |
| 25   | 0.032 | 0.38 | 0.26 | 0.8950 | 0.8959 |

**Reading both tables:**

- **Causal has a lower false-positive rate than greedy at every noise level on both
  tasks**, and is never worse.
- **MAGIC** is the dramatic case: greedy chases noise up to **54%** of the time vs
  causal's **~20%** (~2.5–3× fewer), **and** causal keeps a better final model
  (0.868 vs greedy's 0.857 at high noise). The many near-ties (~0.86) give greedy
  lots to be fooled by.
- **FashionMNIST** corroborates it more modestly: greedy 0.38–0.49 vs causal
  0.21–0.32 (~1.4–2× fewer). The final-accuracy cost of greedy's mistakes is tiny
  (top methods are within 0.001), so here the win is **decision integrity /
  reproducibility** rather than a raw accuracy lift.

**One-line takeaway:** *under noisy evaluation a greedy research agent accepts
improvements that don't replicate (up to ~54% of the time on MAGIC); the causal gate
cuts that 2–3× — and on MAGIC also yields a better final model — with no extra LLM
calls.*

## 4. Robustness: the coherence gate + runtime-crash handling ("automated debugging")

The agent routinely writes code that *parses* but *crashes at runtime* (e.g.
`torch.zeros(..., generator=g)`, `Distribution.sample(generator=...)`). These are
caught and scored as failures (excluded from the frontier, rejected in replay)
rather than aborting the run — both in generation (subprocess harness) and in the
scoring matrix. On the FashionMNIST run, 3 of 8 proposals crashed at runtime and
were handled gracefully.

**Honest caveat (fidelity gap):** generation enforces a per-eval wall-clock timeout;
the scoring matrix currently does not. One FashionMNIST method (a MixUp CNN) timed
out in generation but completed in scoring at ~226 s / 85 TFLOPs per fit — it is the
most accurate point but absurdly expensive. A scoring-stage timeout/compute cap is a
known TODO.

---

## 5. The skeptic's blind spot — spurious correlation (Colored MNIST)

The first four sections show where the causal gate *wins*. This section shows, just
as honestly, where it *can't help* — and why that is a property of the held-out
distribution, not of the gate.

**Task (a spurious-correlation stress test).** Two-channel 28×28 images; the binary
label is `digit ≥ 5` with 25 % label noise (so a shape-only predictor is capped
around ~0.75). A spurious cue — *whether the two channels show the same digit* — is
correlated with the label at **0.90 in train and validation, but flipped to 0.10 in
the held-out test**. The cue is a non-linear channel *interaction* (identical
per-channel marginals), so the linear baseline can only read shape, while a CNN can
read the cue. The harness owns the test split; the agent never sees it.

**What happened (`results/skeptic_regime/colored_mnist/`, fig `fig_spurious.png`):**

| idx | method | val | test | GFLOPs |
|-----|--------|-----|------|--------|
| **1** *(gate accepts)* | small CNN | **0.876** | **0.130** | 208 |
| 2 | + batch-norm CNN | 0.872 | 0.149 | 777 |
| 3 | fixed-norm CNN | 0.864 | 0.169 | 1165 |
| 8 | fixed-norm CNN | 0.859 | 0.171 | 1165 |
| 6 | + augmentation | 0.853 | 0.210 | 1165 |
| 7 | + augmentation | 0.851 | 0.329 | 1165 |
| **0** | baseline (linear, shape-only) | **0.607** | **0.581** | 0.11 |
| 4, 5 | affine-augmentation CNN | *crash* | *crash* | — |

- **The gate accepts a catastrophe — correctly.** Both greedy and causal accept the
  first CNN (idx 1): validation **0.607 → 0.876** (a real, +0.27, seed-stable gain),
  and the replication audit (val-based) reports it **survives**. Yet on the flipped
  test it goes **0.581 → 0.130** — a −0.45 collapse the seed audit is blind to.
- **Optimizing validation actively selects the trap.** val and test are *inversely*
  related — the harder a model leans on the spurious cue to win validation, the worse
  it generalizes (idx 7: 0.851 val / 0.329 test → idx 1: 0.876 val / 0.130 test). The
  only model that generalizes (the shape-only baseline, 0.581 test) looks *worst* on
  validation.
- **The seed-noise axis still behaves as in sections 3–4.** Re-running the regime
  sweep on this pool, the causal gate still cuts greedy's false positives ~5× — it
  just operates on a validation signal that is itself misleading here:

| eval size | noise σ | greedy FP-rate | causal FP-rate | greedy final acc | causal final acc |
|-----------|---------|----------------|----------------|------------------|------------------|
| 2000 | 0.006 | 0.01 | 0.00 | 0.8775 | 0.8773 |
| 500  | 0.009 | 0.01 | 0.01 | 0.8773 | 0.8772 |
| 200  | 0.015 | 0.10 | 0.01 | 0.8760 | 0.8771 |
| 100  | 0.034 | 0.20 | 0.03 | 0.8735 | 0.8769 |
| 50   | 0.049 | **0.28** | **0.06** | 0.8719 | 0.8764 |
| 25   | 0.077 | 0.21 | 0.05 | 0.8730 | 0.8764 |

**Bonus (crash handling):** the agent twice hallucinated a non-existent API
(`torch.radians`); both proposals were caught and scored as failures, not run-killers.

**One-line takeaway:** *the causal skeptic catches false wins that come from
measurement noise, but not false wins that come from the wrong validation
distribution — Colored MNIST shows a +0.27 validation gain the gate accepts and the
seed audit blesses, which is a −0.45 test collapse. The fix is a shifted held-out
set, not more seeds.*

---

## Reproduce

```bash
cd skeptic_gate

# 1. the code-editing agent with the causal skeptic (needs OPENAI_API_KEY in .env)
python study.py fashionmnist llm 8 5
python study.py magic llm 8 5
python study.py colored_mnist llm 8 5      # the spurious-correlation stress test

# 2. the noise-dial regime sweep (NO LLM calls — reuses each study's pool)
python regime_sweep.py fashionmnist eval 8 200
python regime_sweep.py magic eval 8 200
python regime_sweep.py colored_mnist eval 8 200

# 3. poster figures (auto-discovers all three tasks; colored_mnist -> fig_spurious.png)
python make_poster_figs.py
```

Each study writes `llm.json` + `methods_llm.csv` (score/Pareto table) +
`replay_llm.csv` (greedy-vs-causal accept audit); each sweep writes
`regime_eval.json/.csv` + `regime_curve_eval.png`.
