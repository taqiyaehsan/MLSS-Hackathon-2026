# Skeptic Gate for an Autonomous ML-Research Agent

A small, reproducible harness that runs a **vanilla "autoresearcher"** loop — an LLM proposes a code change, the change is evaluated, and a greedy rule keeps or discards it — on a real benchmark task (**MLRC-Bench Machine Unlearning**), plus a **suggested "skeptic" wrapper** that re-tests a candidate before believing its improvement.

The motivating question: *autonomous research agents accept changes based on a single noisy evaluation. How many of those "wins" are real, and how much compute should an agent spend confirming one before it trusts it?*

This repo contains:

1. **Full setup** for the autoresearcher + MLRC-Bench Machine Unlearning task, runnable on a laptop (CPU / Apple-MPS) or a CUDA box.
2. **Results** from an end-to-end run of the vanilla (greedy) autoresearcher with `gpt-4.1-mini` in the loop.
3. A **naive causal + coherence gate** implementation, offered as a *suggestion* for making the loop more skeptical.
4. An **observation** from repeated runs on one machine that is worth knowing before trusting any single score (see [`docs/OBSERVATIONS.md`](docs/OBSERVATIONS.md)).

---

## The idea in one picture

```
vanilla autoresearcher (greedy):
    propose edit ── evaluate once ── score > best ? keep : discard ── repeat

suggested skeptic wrapper:
    propose edit
      │
      ├─ [coherence gate]  parses / keeps the run() signature?  no → cull before wasting an eval
      │
      └─ [causal gate]     evaluate a few seeds; accept only if the gain clears the noise band
```

Only the accept rule changes between the two; the proposer, the budget, and the
evaluation are held identical, so the comparison is a clean paired ablation.

---

## Repository layout

```
skeptic-gate/
├── README.md                     ← this file
├── requirements.txt
├── setup/
│   ├── setup.sh                  ← one-shot environment setup
│   └── mlrc-local.patch          ← enabling patch for MLRC (MPS device + lazy Kaggle auth)
├── skeptic_gate/                 ← all the code
│   ├── gates.py                  ← accept policies: Greedy, Causal, Coherence (task-agnostic)
│   ├── mlrc_adapter.py           ← real MLRC world: LLM proposer + real eval + keep/discard
│   ├── run_mlrc.py               ← CLI: run any arm on the real task
│   ├── baseline_noise.py         ← characterize per-eval noise
│   ├── replication_audit_real.py ← re-run accepted changes N× to test if a "win" survives
│   ├── baseline_MyMethod.py      ← canonical baseline method (reset source of truth)
│   ├── synthetic.py / experiment.py / plots.py / sanity.py / tests.py
│   └── README.md                 ← notes on the controlled synthetic study
├── results/
│   ├── vanilla_autoresearch_run/ ← end-to-end greedy run (summary, per-step log, every proposal)
│   ├── baseline_observation/     ← repeated baseline evals from different time windows
│   └── synthetic_figs/           ← figures from the controlled synthetic study
└── docs/
    └── OBSERVATIONS.md           ← the non-stationary-evaluation finding
```

`MLRC-Bench/` is **not** committed — it is cloned and patched locally by `setup/setup.sh`.

---

## Setup

### Prerequisites

- Python 3.11+ (developed on 3.13)
- `git`
- An OpenAI API key (the in-loop proposer uses `gpt-4.1-mini`)
- ~2 GB disk for CIFAR-10 + model checkpoints
- A GPU is optional. Apple-Silicon (MPS) and CPU both work for the dev phase; CUDA is recommended for stable, comparable numbers (see the observation below).

### One-shot setup

From the repository root:

```bash
bash setup/setup.sh
```

This will:

1. Clone `MLRC-Bench` into the repo root.
2. Create a virtual environment in `.venv/` and install `requirements.txt`.
3. Install MLRC-Bench's `MLAgentBench` package in editable mode.
4. Apply `setup/mlrc-local.patch` (enables Apple-MPS device selection and lazy Kaggle auth so the **dev phase needs no Kaggle credentials**).
5. Download the Machine Unlearning data + checkpoints (CIFAR-10, pretrained + retrained ResNet-18 weights, forget index).

Then add an API key:

```bash
cp skeptic_gate/.env.example skeptic_gate/.env
# edit skeptic_gate/.env and set OPENAI_API_KEY=sk-...
```

`.env` is gitignored; never commit it.

### Manual setup (if the script is not preferred)

```bash
git clone https://github.com/yunx-z/MLRC-Bench.git
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e MLRC-Bench
git -C MLRC-Bench apply ../setup/mlrc-local.patch     # or: cd MLRC-Bench && git apply ../setup/mlrc-local.patch
python MLRC-Bench/MLAgentBench/benchmarks_base/machine_unlearning/scripts/prepare.py
```

> Note: `prepare.py` downloads CIFAR-10 from the Toronto mirror, which can be slow
> and occasionally stalls (torchvision has no download timeout). If it hangs,
> interrupt and re-run — it skips files that already exist.

### Verify the task runs

```bash
cd MLRC-Bench/MLAgentBench/benchmarks_base/machine_unlearning/env
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py -m my_method -p dev
```

A single eval takes a few minutes and prints a `Final Score` (higher is better; the unmodified baseline is in the ~0.05–0.12 range — see the observation about why that range is wide).

---

## Running the autoresearcher

All commands run from `skeptic_gate/` using the project venv.

### Vanilla (greedy) — this is plain autoresearcher behavior

```bash
cd skeptic_gate
../.venv/bin/python run_mlrc.py --arm greedy --budget 8
```

- `propose` → `gpt-4.1-mini` writes a new `MyMethod.py` given the task brief, the current best method, and a short history.
- `evaluate` → writes the file, runs the real MLRC eval, parses the score.
- accept rule → keep iff the single score beats the current best; otherwise revert.
- Budget unit = one eval. Every proposal (accepted, rejected, crashed) is snapshotted under `results/<run_id>/proposals/`, and every step is logged to `results/<run_id>/results.jsonl`.

The runner resets `MyMethod.py` from `baseline_MyMethod.py` at startup, so every run begins from the identical baseline.

Free dry run of the harness (no API, no eval):

```bash
../.venv/bin/python run_mlrc.py --arm greedy --mock-llm --mock-eval --budget 5
```

### Suggested skeptic arms

Swap the policy; everything else stays fixed:

```bash
../.venv/bin/python run_mlrc.py --arm causal      --budget 8   # re-test gate only
../.venv/bin/python run_mlrc.py --arm coh+greedy  --budget 8   # coherence cull only
../.venv/bin/python run_mlrc.py --arm coh+causal  --budget 8   # both gates
```

### Controlled synthetic study (no GPU, instant)

```bash
../.venv/bin/python experiment.py && ../.venv/bin/python plots.py && ../.venv/bin/python tests.py
```

See `skeptic_gate/README.md` for details. Figures land in `results/synthetic_figs/`.

---

## Results: end-to-end vanilla autoresearcher

A budget-8 greedy run with `gpt-4.1-mini` (`results/vanilla_autoresearch_run/`):

| step | proposal (intent, abridged)                              | score  | decision |
|------|----------------------------------------------------------|--------|----------|
| —    | baseline (1-epoch fine-tune on retain set)               | 0.1168 | incumbent |
| 1    | gradient ascent on forget + descent on retain            | 0.0032 | reject   |
| 2    | targeted ascent + distillation from original             | 0.0007 | reject   |
| 3    | ascent + descent (variant)                               | 0.0034 | reject   |
| 4    | ascent + descent + Fisher-style penalty                  | 0.1625 | **accept** |
| 5    | more aggressive ascent                                   | 0.0000 | reject   |
| 6    | more ascent steps                                        | 0.0000 | reject   |
| 7    | adaptive ascent step                                     | 0.0055 | reject   |

**Takeaways**

- The loop runs end-to-end with a real LLM in the loop: it proposes legitimate
  unlearning methods (gradient ascent/descent, distillation, Fisher penalties),
  evaluates them, and keeps/discards greedily.
- The agent is **honestly weak** on this task: six of seven proposals over-forget
  and collapse model utility (retain/test accuracy ratios fall toward zero), so the
  product score craters. This matches the MLRC-Bench finding that agents barely
  improve machine unlearning.
- One proposal (step 4) was accepted on a single eval — *but see the observation
  below before reading anything into that number.*

---

## Suggested skeptic wrapper (causal + coherence)

Offered as a starting point, not a finished result. Both gates live in
`skeptic_gate/gates.py` and plug into the same loop:

- **Coherence gate (pre-eval, cheap).** Before spending a multi-minute eval, check
  that the proposed file parses and keeps the required `run(self, net, retain_loader,
  forget_loader, val_loader)` signature. Broken proposals are culled without running.
  This is a cheap cull, not a logic checker — it will not catch a method that
  *runs* but over-forgets.

- **Causal gate (post-eval, the point).** Instead of trusting one score, evaluate a
  candidate over a few seeds and compare against the incumbent's own noise band;
  accept only if the gain clears the band, and spend more seeds only when a result
  is borderline (sequential confirmation, more compute-efficient than a fixed *K*).

The controlled synthetic study (`results/synthetic_figs/`) illustrates the intended
behavior under *clean* i.i.d. noise: the re-test gate cuts false-accepts several-fold
across regimes, while greedy keeps more throughput when noise is low. Whether this
transfers to the real task depends entirely on the evaluation behaving like clean
noise — which, on the test machine, it did not (next section).

---

## Observation worth knowing before trusting any score

Repeated evaluation of the **identical, unmodified baseline** on one machine
(Apple-MPS laptop) produced very different scores depending on *when* the eval ran:

| time window         | score(s)                   | eval wall-time |
|---------------------|----------------------------|----------------|
| A (8 evals)         | ~0.117 (tight)             | ~180 s         |
| B (1 eval)          | 0.054                      | ~180 s         |
| C (3 evals)         | ~0.001 (tight)             | ~290 s         |

Same code, ~100× spread in the score, tight *within* a window but shifting sharply
*across* windows — and the eval slowed down (180 s → 290 s) as the scores fell. The
likely cause is system-state drift (thermal throttling / device contention after
long back-to-back runs) changing how the per-eval training behaves; a single eval
re-trains ten unlearned models, and that training is where the nondeterminism enters.

**Why this matters:** this is *non-stationary* drift, not clean i.i.d. measurement
noise. It means a single score on this machine is not trustworthy, and a sequential
A-vs-B comparison can be confounded by *when* each arm ran rather than by the arm
itself. Treat local laptop numbers as qualitative; run quantitative comparisons on a
stationary environment (CUDA), and control determinism (seed torch, fix algorithms,
interleave/randomize arm order). Full detail and the raw per-eval data are in
[`docs/OBSERVATIONS.md`](docs/OBSERVATIONS.md) and `results/baseline_observation/`.

This observation is, in miniature, the reason a skeptic gate is worth exploring:
the vanilla loop accepted a "win" (step 4 above) from a single eval, on a task where
a single eval can swing 100× for no reason at all.

---

## Task reference: Machine Unlearning (dev phase)

- Goal: after "forgetting" a subset of training data, the model should behave like
  one retrained without it, while staying accurate on the rest.
- Edit target: `MLRC-Bench/.../machine_unlearning/env/methods/MyMethod.py`
  (`run(self, net, retain_loader, forget_loader, val_loader)`, modify `net` in place).
- Score = `forgetting_quality × (retain_acc_ratio) × (test_acc_ratio)`, higher is
  better. One eval averages `NUM_MODELS = 10` unlearning runs internally.
- Dev phase uses CIFAR-10 + ResNet-18 and needs no Kaggle credentials (with the patch).

## Credits / upstream

- Task: [MLRC-Bench](https://github.com/yunx-z/MLRC-Bench) (Machine Unlearning).
- Loop shape: Karpathy's [autoresearch](https://github.com/karpathy/autoresearch)
  (a template — its propose→evaluate→keep/discard structure is reimplemented here;
  the repo itself is not a dependency).
