# Skeptic Gate for an Autonomous ML-Research Agent

A small, reproducible harness that runs a **vanilla "autoresearcher"** loop — an LLM proposes a code change, the change is evaluated, and a greedy rule keeps or discards it — on a real benchmark task (**MLRC-Bench Machine Unlearning**), plus a **suggested "skeptic" wrapper** that re-tests a candidate before believing its improvement.

The motivating question: *autonomous research agents accept changes based on a single noisy evaluation. How many of those "wins" are real, and how much compute should an agent spend confirming one before it trusts it?*

This repo contains:

1. **Full setup** for the autoresearcher + MLRC-Bench Machine Unlearning task, runnable on a laptop (CPU / Apple-MPS) or a CUDA box.
2. **Results** from an end-to-end run of the vanilla (greedy) autoresearcher with `gpt-4.1-mini` in the loop.
3. A **naive causal + coherence gate** implementation, offered as a *suggestion* for making the loop more skeptical.
4. An **observation** from repeated runs on one machine that is worth knowing before trusting any single score (see [`docs/OBSERVATIONS.md`](docs/OBSERVATIONS.md)).

---

## Contents

- [The idea in one picture](#the-idea-in-one-picture)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Running on Google Cloud (GPU)](#running-on-google-cloud-gpu)
- [Running the autoresearcher](#running-the-autoresearcher)
- [Code-editing agent: plug-and-play tasks (the main pipeline)](#code-editing-agent-plug-and-play-tasks-the-main-pipeline)
- [Real local tasks: agentic hyperparameter search (FashionMNIST, MAGIC, digits)](#real-local-tasks-agentic-hyperparameter-search-fashionmnist-magic-digits)
- [Results: end-to-end vanilla autoresearcher](#results-end-to-end-vanilla-autoresearcher)
- [Suggested skeptic wrapper (causal + coherence)](#suggested-skeptic-wrapper-causal--coherence)
- [Observation worth knowing before trusting any score](#observation-worth-knowing-before-trusting-any-score)
- [Task reference: Machine Unlearning (dev phase)](#task-reference-machine-unlearning-dev-phase)
- [Credits / upstream](#credits--upstream)

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

### The full pipeline

```
PART A — the PROPOSED agent loop (one LLM agent per task)
         what we propose = the skeptic = COHERENCE gate (step 2) + CAUSAL accept gate (step 4)

repeat until the proposal budget runs out:

  1. Proposer        LLM agent rewrites the COMPLETE method file (baseline → CNN/MLP/...)
         │
         ▼
  2. Coherence gate  parses? keeps fit/predict signature?  ── no ──►  CULL (no eval spent)
         │ yes                                                        [SKEPTIC, part 1]
         ▼
  3. Evaluate        really train the method  →  noisy validation score
         │
         ▼
  4. CAUSAL gate     re-test over k seeds; accept ONLY if the gain        [SKEPTIC, part 2]
                     clears the noise band  (don't trust one noisy score)
         │
         ├── clears band ───►  ACCEPT  (new incumbent, steers the next proposal)
         └── within noise ──►  DISCARD (keep the incumbent)
         │
         ▼  record EVERY coherent method it wrote = the candidate stream
            (loop back to step 1 with the remaining budget)

PART B — evaluating the skeptic vs the GREEDY BASELINE (NO new agent calls)
         baseline = the "vanilla autoresearcher": accept any score > incumbent, no re-test

  5. Score matrix    re-score each method over S seeds (val) + one-touch TEST + FLOPs
         │
         ├─►  6a. Replay ablation   run the loop once to fix the candidate stream, then
         │            replay BOTH the causal gate (ours) and the greedy baseline over the
         │            IDENTICAL candidates + measurements (pure policy isolation), then a
         │            replication audit: which accepted gains vanish vs the full-seed truth
         │
         ├─►  6b. Pareto frontier   accuracy ↑ / stability ↓ / FLOPs ↓  (report, no auto-pick)
         │
         └─►  6c. Regime sweep      dial up evaluation noise; measure the causal-vs-greedy
                     false-positive rate  →  when does skepticism pay?
```

**What we propose is the skeptic: the coherence gate (step 2, culls broken edits
before any eval) + the causal accept gate (step 4, re-tests over k seeds and accepts
only if the gain clears the noise band).** Greedy — accepting on a single noisy
score — is the **baseline** (the vanilla autoresearcher), kept only so we can measure
what the skeptic buys.

To compare them *honestly*, the loop is run **once** to fix a candidate stream, and
**Part B replays both the causal gate (ours) and the greedy baseline over that
identical stream and its measurements** — so the accept rule is the *only* thing that
changes (a clean paired ablation, no extra LLM calls). Running two live loops instead
would diverge (different accepts → different proposals → confounded), which is why the
comparison is done by replay. The held-out test split and the many-seed replication
audit live **outside** the loop; the agent and the gates never see them, so reported
progress can't be selection-on-the-eval-set. The regime sweep (6c) is the headline
result — see [`docs/SKEPTIC_REGIME_RESULTS.md`](docs/SKEPTIC_REGIME_RESULTS.md).

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
│   │  # ── code-editing pipeline (the main pipeline) ──
│   ├── base_method.py            ← the fit(X,y,seed)/predict(X) interface the agent implements
│   ├── task_data.py              ← harness-owned loaders + fixed train/val/test split (test held out)
│   ├── run_method.py             ← trusted harness: train one method file, score it (noise dial)
│   ├── local_task.py             ← TaskSpec, coherence gate (AST), OpenAI proposer, the world
│   ├── tasks/<name>/             ← background.md + baseline_method.py (fashionmnist, magic, ...)
│   ├── study.py                  ← THE replay study: pool → score matrix → replay + audit + Pareto
│   ├── regime_sweep.py           ← noise-dial regime sweep: greedy vs causal false positives vs noise
│   │  # ── earlier paths (config-tuning + MLRC) ──
│   ├── hpo_task.py               ← real local tasks: agentic MLP hyperparameter search (digits/fmnist/magic)
│   ├── plots_hpo.py              ← figures for the local tasks
│   ├── mlrc_adapter.py           ← real MLRC world: LLM proposer + real eval + keep/discard
│   ├── run_mlrc.py               ← CLI: run any arm on the real task
│   ├── baseline_noise.py         ← characterize per-eval noise
│   ├── replication_audit_real.py ← re-run accepted changes N× to test if a "win" survives
│   ├── baseline_MyMethod.py      ← canonical baseline method (reset source of truth)
│   ├── synthetic.py / experiment.py / plots.py / sanity.py / tests.py
│   └── README.md                 ← notes on the controlled synthetic study
├── results/
│   ├── skeptic_regime/           ← THE skeptic result: regime curve + code-edit study + figure
│   ├── vanilla_autoresearch_run/ ← end-to-end greedy run (summary, per-step log, every proposal)
│   ├── baseline_observation/     ← repeated baseline evals from different time windows
│   └── synthetic_figs/           ← figures from the controlled synthetic study
└── docs/
    ├── SKEPTIC_REGIME_RESULTS.md ← skeptic gate under noisy eval: results + explanation
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

## Running on Google Cloud (GPU)

The whole project can run on one GCP GPU VM — the small tasks (synthetic,
FashionMNIST, MAGIC) on CPU, and Machine Unlearning on the GPU (where its eval is
**stationary**, unlike a laptop). Steps run from your **laptop** unless marked
*(on VM)*.

```bash
# 1. Install the gcloud CLI + authenticate (laptop)
brew install --cask google-cloud-sdk        # macOS
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. (likely needed) request GPU quota: IAM & Admin -> Quotas ->
#    "NVIDIA T4 GPUs" >= 1 in your zone. New projects default to 0; approval can
#    take minutes-to-hours, so do this FIRST.

# 3. Create the VM (Deep Learning image ships CUDA + driver)
gcloud compute instances create skeptic-gpu \
  --zone=us-central1-a --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 --maintenance-policy=TERMINATE \
  --image-family=common-cu121-debian-11 --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB --metadata=install-nvidia-driver=True

# 4. Connect (writes an SSH config entry usable by VS Code Remote-SSH too)
gcloud compute config-ssh
gcloud compute ssh skeptic-gpu --zone=us-central1-a

# 5. (on VM) clone + provision
git clone https://github.com/taqiyaehsan/MLSS-Hackathon-2026.git
cd MLSS-Hackathon-2026
bash setup/gcp_setup.sh        # CUDA env, deps, MLRC-Bench, Claude Code hints
```

After the env is set up (conda or venv), verify the code runs:

```bash
cd skeptic_gate && python smoke_test.py    # expect: SMOKE PASSED (5/5); CUDA available=True on the GPU
```

`smoke_test.py` needs no GPU, data, or API key — it checks imports, the synthetic
gate pipeline, and a real train-and-gate loop on bundled `digits`, and reports
whether CUDA is visible. Run it first to confirm the clone + environment are good.

Stop the VM when idle (`gcloud compute instances stop skeptic-gpu`) — a stopped
instance bills only for disk, not GPU. `setup/gcp_setup.sh` ends by printing how
to install **Claude Code** on the VM so the agent runs next to the GPU.

For a step-by-step walkthrough of running the **Machine Unlearning** task on the
GPU box — data prep, the cost lever, all greedy/skeptic arms, and the replication
audit — see [`docs/GCP_MACHINE_UNLEARNING.md`](docs/GCP_MACHINE_UNLEARNING.md).

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

## Code-editing agent: plug-and-play tasks (the main pipeline)

Each task is a tiny self-contained repo under `skeptic_gate/tasks/<name>/`: a
**problem statement** (`background.md`) + a **working but mediocre baseline**
(`baseline_method.py`, the "primary code") + a data loader (in `task_data.py`) +
a registered `TaskSpec`. **One LLM agent (`gpt-4.1-mini`) per task EDITS the
baseline code** to raise a held-out metric. The same task-agnostic `gates.py`
wraps the loop — a **coherence gate** culls edits that don't parse / keep the
interface (cheap), and the **causal gate** re-tests a gain before believing it.

```
the ONE code-editing agent, looping:
  read  background.md + current best MyMethod.py + history
  write a COMPLETE edited MyMethod.py        (e.g. logistic -> MLP, linear -> CNN)
        │
  coherence gate: parses / keeps fit·predict?   no -> cull (no eval spent)
        │ yes
  harness: train it, score it (held-out val)    ← run_method.py (CPU, seeded, timeout)
        │
  accept gate (greedy | causal): keep or revert
        │ repeat to budget
```

The harness owns data, the held-out **test** split, the seed, CPU + timeout, and
scoring — so numbers stay reproducible and stationary no matter what the agent
writes. The agent owns only the model + training inside `MyMethod`.

### Reference tasks (proven measurable progress)

| task | modality | metric | the agent's edit | held-out result |
|---|---|---|---|---|
| `fashionmnist` | vision | accuracy | linear → CNN | 0.75 → **0.87** |
| `magic` | tabular | accuracy | logistic → MLP | 0.786 → **0.844** |
| `example_regression` | tabular | R² | linear (template) | 0.30 baseline |

### Run

```bash
cd skeptic_gate
# test a baseline through the harness (no API)
../.venv/bin/python run_method.py --task magic \
    --method tasks/magic/baseline_method.py --metric accuracy

# the full study: agent writes methods, then the analysis
#   study.py <task> [llm] [N_PROPOSALS] [N_SEEDS]   (omit 'llm' for a FREE mock run)
../.venv/bin/python study.py fashionmnist llm 8 8
../.venv/bin/python study.py example_regression llm 8 8   # regression (R²)
```

### Two analysis outputs (`study.py`)

1. **Skeptic ablation (replay):** the agent's candidate stream is generated once
   and re-scored over seeds; greedy and causal are then replayed over the
   **identical** candidates + measurements — a clean paired ablation (live arms
   would diverge and confound it). A **replication audit** reports how many
   accepted "wins" **vanish** on honest re-test.
2. **Pareto frontier over the methods the agent wrote** (not a fixed menu): 3
   axes — accuracy (multi-seed mean ↑), stability (std ↓), cost (FLOPs ↓, with
   wall-clock as context). The frontier is **reported, not auto-picked** — you
   choose by your accuracy / cost / reliability priorities. The held-out test is
   touched once per point (report only, never to build the frontier).

### Add a new task (plug-and-play)

1. **`task_data.py`** — add `load_<name>()` returning the six CPU tensors
   `X_tr/y_tr/X_va/y_va/X_te/y_te` (features `float32`; labels `int64` for
   classification, `float32` for regression — pass `regression=True` to the
   split/pack helpers). Register it in `LOADERS` under `<name>`.
2. **`tasks/<name>/background.md`** — problem statement + approach space + the
   hard rules (the `fit`/`predict` contract).
3. **`tasks/<name>/baseline_method.py`** — a working, deliberately-mediocre
   `class MyMethod(BaseMethod)` with `fit(self, X, y, seed)` and `predict(self, X)`.
4. **`local_task.py`** — add `TaskSpec("<name>", time_limit=…, regimes=[…],
   metric="accuracy"|"r2")` to `TASKS`.

Then `../.venv/bin/python study.py <name> llm` runs the identical pipeline on your
task. (`base_method.py` is the interface; `run_method.py` the harness; `study.py`
the analysis — none of them change when you add a task.)

---

## Real local tasks: agentic hyperparameter search (FashionMNIST, MAGIC, digits)

> Secondary / alternative pipeline: instead of editing code, the agent tunes a
> **fixed MLP's hyperparameters**. Kept for the controlled noise-regime sweep.

This is the cheap, **stationary**, fully-local counterpart to the MLRC run. The
same skeptic gates drive an agent that tunes a **fixed MLP** on a real dataset —
no GPU, no multi-GB download, evals are ~0.1 s so the multi-seed replication audit
is affordable and the numbers are trustworthy (unlike the laptop MLRC eval, which
drifts). The agent does **not** write model code; it proposes **hyperparameter
configs** (`hidden, lr, dropout, weight_decay, epochs, batch_size, activation`),
which keeps "broken proposals" statically detectable (an out-of-bounds config),
so the coherence gate is complete by construction.

All commands run from `skeptic_gate/` using the project venv.

### Datasets shipped

| name      | what it is                                   | data source (auto, first run)            |
|-----------|----------------------------------------------|------------------------------------------|
| `digits`  | sklearn 8×8 handwritten digits, 10 classes   | bundled with scikit-learn (no download)  |
| `fmnist`  | FashionMNIST 28×28 → 784 pixels, 10 classes  | torchvision download, cached in `_data_fmnist/` (gitignored) |
| `magic`   | MAGIC Gamma Telescope, 10 features, binary   | OpenML fetch (cached by scikit-learn)    |

### Two proposers

- **Programmatic** (default): a bounded, reproducible hyperparameter mutator.
  Use it for the **quantitative proof** — it runs hundreds of cheap evals so the
  false-accept counts and regime curve are statistically meaningful.
- **LLM agent** (`gpt-4.1-mini`, needs `OPENAI_API_KEY` in `skeptic_gate/.env`):
  the genuinely **agentic** version — a real LLM tunes the MLP, the same gates
  decide keep/discard. Slower (one API call per proposal), so run fewer seeds.

### Commands

```bash
cd skeptic_gate

# --- programmatic regime sweep (greedy vs causal x 3 noise regimes x 5 seeds) ---
../.venv/bin/python hpo_task.py fmnist        # FashionMNIST
../.venv/bin/python hpo_task.py magic         # MAGIC
../.venv/bin/python hpo_task.py digits        # digits (also the no-arg default)

# --- AGENTIC sweep: the real LLM agent is the proposer ---
#   hpo_task.py <dataset> llm [SEEDS] [BUDGET]   (defaults: 3 seeds, budget 20)
../.venv/bin/python hpo_task.py fmnist llm 3 20
../.venv/bin/python hpo_task.py magic  llm 1 20   # 1 seed = quick/cheap look

# --- single live LLM-agent demo (prints every proposal + the gate's verdict) ---
../.venv/bin/python hpo_task.py llm fmnist

# --- figure from a finished run ---
../.venv/bin/python plots_hpo.py fmnist       # -> results/figs/fig_hpo_fmnist.png
```

Outputs land in `results/hpo_<dataset>/summary.json` (programmatic) or
`results/hpo_<dataset>_llm/summary.json` (agentic); `digits` uses the legacy
`results/hpo_task/` path. `results/` is gitignored.

### Reading the output table

```
noise regime        eval sd | greedy acc / false / test | causal acc / false / test
high (150 samples)   0.0606 |  6.8 /  2.0 /0.778±0.043  |  1.6 / 0.6 /0.807±0.011
```

- **acc** = number of edits the arm accepted. **false** = of those, how many had
  no real gain on an honest 30-seed re-test (the self-deception, counted).
  **test** = held-out test accuracy of the shipped config (the agent and the gate
  never see this split).
- The thesis: as eval noise rises, **greedy accepts more lucky wins that vanish**
  on re-test, and ships a worse model; the **causal** gate keeps false-accepts
  near zero and ships a stabler model — at the cost of spending more evals per
  decision.

### Adding a NEW dataset or downstream task

The pipeline is dataset-pluggable. To add one, register a `DatasetSpec` in the
`DATASETS` dict in `hpo_task.py`:

```python
DATASETS["mytask"] = DatasetSpec(
    name="mytask",
    loader=_load_mytask,     # () -> dict with CPU tensors:
                             #   X_tr,y_tr (train), X_va,y_va (gate's eval set),
                             #   X_te,y_te (HELD-OUT test, never seen by agent/gate)
    n_features=...,          # input width of the flat feature vector
    n_classes=...,           # number of classes
    desc="one line for the LLM brief (what the task is)",
    regimes=[                # noise dial: smaller train_subset => noisier eval
        ("low  (full data)", Fidelity("full", 1.0, {"train_subset": None})),
        ("med  (... )",      Fidelity("med",  1.0, {"train_subset": 200})),
        ("high (... )",      Fidelity("high", 1.0, {"train_subset": 80})),
    ],
)
```

Contract for the loader: return the six tensors above (features `float32`,
labels `int64`), scale/flatten features yourself (fit any scaler on **train
only**), and pick `train_subset` sizes so the three regimes show a clear
low→med→high spread in `eval sd` (run the sweep once and check the table). The
agent then tunes the same MLP on your task with no other code changes. For a
genuinely different model family, edit `_build_mlp` / the config space — but note
that widening the edit surface to free-form code reintroduces runtime-crash
proposals a static coherence gate can't catch.

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
