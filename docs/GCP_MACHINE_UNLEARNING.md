# Running the Machine Unlearning task on the skeptic-gate pipeline (GCP GPU)

This guide takes you from a freshly set-up GCP VM to a full **greedy-vs-skeptic
run on the real MLRC-Bench Machine Unlearning benchmark**, plus the replication
audit. Everything here runs on the VM (`te137` user, shared conda/venv).

> **Why the GPU box matters.** On a laptop (MPS) the Machine Unlearning eval is
> *non-stationary* — the same method scores differently run-to-run for reasons
> unrelated to the method, so the gate numbers can't be trusted. On the V100 the
> eval is **stationary**, so this is the box where the named-benchmark quantitative
> numbers are real. Don't quote MU numbers measured on a laptop.

---

## 0. What the pipeline actually does

One autonomous-research loop on the real task:

```
propose (OpenAI edits methods/MyMethod.py) -> evaluate (real MLRC dev eval) -> accept rule -> keep/discard
```

- The **eval** is `main.py -m my_method -p dev` inside the MLRC task `env/`. It
  fine-tunes/unlearns ResNet-18 on CIFAR-10 over `NUM_MODELS` inner models and a
  membership-inference attack, then writes `dev_results/my_method_results.npz`.
  We read `total_score` ("Final Score"); **higher is better**.
- The **accept rule** is the only thing that changes between arms. All arms share
  the exact policy objects in `gates.py`:
  - `greedy` — accept if the single observed score beats the incumbent.
  - `causal` — re-run the candidate over `k0..k_max` seeds; accept only if the
    mean gain clears the noise band (the skeptic).
  - `coh+greedy` / `coh+causal` — same, but a cheap static coherence check culls
    broken edits *before* spending an eval on them.

Driver: [`skeptic_gate/run_mlrc.py`](../skeptic_gate/run_mlrc.py).
Adapter (eval + OpenAI proposer): [`skeptic_gate/mlrc_adapter.py`](../skeptic_gate/mlrc_adapter.py).

---

## 1. One-time VM setup

You said MLRC is already set up — these steps just make sure the repo, env, and
paths line up with what the pipeline expects. Skip what you've already done.

```bash
# from your home on the VM
cd ~/MLSS-Hackathon-2026          # the repo clone (re-clone from GitHub if missing)
bash setup/gcp_setup.sh           # CUDA torch, deps, clones MLRC-Bench, applies the patch
```

**Critical path requirement:** the adapter resolves the task as
`<repo_root>/MLRC-Bench/MLAgentBench/benchmarks_base/machine_unlearning/env`
(see `mlrc_adapter.py` `REPO_ROOT`/`ENV_DIR`). `gcp_setup.sh` clones MLRC-Bench
*inside* the repo root, so this just works. If you set MLRC up somewhere else,
symlink or move it so it lives at `~/MLSS-Hackathon-2026/MLRC-Bench`, otherwise
`run_mlrc.py` won't find the eval.

Activate the env for everything below:

```bash
source ~/MLSS-Hackathon-2026/.venv/bin/activate    # or: conda activate <shared-env>
```

### Confirm the GPU is visible (so the eval is stationary)

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
# expect: cuda: True | Tesla V100-SXM2-16GB
```

Both `evaluation.py` and the baseline methods auto-select
`cuda -> mps -> cpu`, so on this box they pick CUDA with no extra flags.

---

## 2. Get the task data + weights

The dev eval needs CIFAR-10 plus two pretrained ResNet-18 checkpoints. The
prepare script downloads the weights (no Kaggle needed for the dev phase — the
patch makes Kaggle auth lazy):

```bash
python MLRC-Bench/MLAgentBench/benchmarks_base/machine_unlearning/scripts/prepare.py
```

After it finishes you should have, under `.../machine_unlearning/env/`:
`weights_resnet18_cifar10.pth`, `retrain_weights_resnet18_cifar10.pth`, and
`data/cifar-10-batches-py/`.

---

## 3. Enable the cost lever (one-line edit — REQUIRED for `--fidelity cheap`)

The `cheap` fidelity runs the eval over fewer inner models (3 instead of 10) so
the same budget funds ~3× more evals. The pipeline passes this via the
`MU_NUM_MODELS` env var — **but `evaluation.py` only honors it if line 34 reads
the env var.** That edit is in the original dev machine but is *not* in
`mlrc-local.patch`, so on a fresh clone the lever is silently ignored (you'd still
run 10 models and your "equal-budget" accounting would be wrong).

Apply this idempotent fix once:

```bash
EVAL=MLRC-Bench/MLAgentBench/benchmarks_base/machine_unlearning/env/evaluation.py
grep -q 'MU_NUM_MODELS' "$EVAL" || \
  sed -i 's/^NUM_MODELS = 10.*$/NUM_MODELS = int(os.environ.get("MU_NUM_MODELS", "10"))  # cost lever/' "$EVAL"
grep -n 'NUM_MODELS' "$EVAL"     # verify it now reads the env var
```

If you only ever run `--fidelity full` (10 models, the default), you can skip this
— but applying it is harmless and keeps the cost-lever results honest.

---

## 4. API key

```bash
cd ~/MLSS-Hackathon-2026/skeptic_gate
[ -f .env ] || cp .env.example .env
# edit .env  ->  OPENAI_API_KEY=sk-...     (.env is gitignored; never commit it)
```

Proposer defaults to `gpt-4.1-mini` (cheap). Override with `--model`.

---

## 5. Free smoke test (no GPU, no API)

Confirms the loop, logging, and file plumbing work before you spend evals or
tokens:

```bash
cd ~/MLSS-Hackathon-2026/skeptic_gate
python run_mlrc.py --mock-llm --mock-eval --budget 4
```

You should see a baseline measurement, a few proposal/accept/reject lines, and a
summary. No real eval runs.

---

## 6. One real iteration (sanity + first real number)

```bash
python run_mlrc.py --arm greedy --budget 1
```

This measures the **baseline** once (charged 1 unit) and stops. Note the
wall-clock printed for the baseline eval — that's your per-eval cost on this GPU.
Use it to size budgets below.

---

## 7. Full runs — greedy vs the skeptic

Budget unit = **one full eval** (one seed). Accounting:

- `greedy` / `coh+greedy`: ~1 eval per candidate → `--budget 8` ≈ baseline + ~7 candidates.
- `causal` / `coh+causal`: `k0..k_max` (2–6) evals per candidate → `--budget 8` ≈ baseline + ~2–3 candidates.

`--reset-from` defaults to `baseline_MyMethod.py`, so **every arm starts from the
identical incumbent** — that's what makes the comparison fair. Run the arms you
want to compare at the same budget:

```bash
python run_mlrc.py --arm greedy      --budget 8 --seed 0
python run_mlrc.py --arm causal      --budget 8 --seed 0
python run_mlrc.py --arm coh+greedy  --budget 8 --seed 0
python run_mlrc.py --arm coh+causal  --budget 8 --seed 0
```

Each run prints, per step, the status (`ACCEPT`/`REJECT`/`CULL`), the score, the
running best, and budget spent, e.g.:

```
[3] ***ACCEPT*** score=0.1842 best=0.1842 spent=4.00/8  :: add a short retain-set finetune after noise injection
```

### Cost-lever variant (after Section 3)

```bash
python run_mlrc.py --arm causal --budget 8 --fidelity cheap
```

`cheap` = 3 inner models, charged 0.3 units/eval (so budget 8 funds ~3× more
evals, but noisier). **Integrity note:** the 0.3 weight is an assumption — before
quoting any equal-budget claim, check it against the *measured* wall-clock ratio
of cheap-vs-full evals on this GPU and adjust if it's off.

---

## 8. Baseline-noise run + replication audit (the centerpiece)

This is the "single evals lie" result: take what greedy *accepted* and re-test it
to see how many wins survive. **Do not run this while a loop is running** — both
write `methods/MyMethod.py` and the eval npz.

First, characterize baseline noise (fresh independent re-evals of the baseline):

```bash
python baseline_noise.py --n 8 --run-id baseline_noise_n8
```

Then audit a greedy run (re-run each accepted change `--reps` times as fresh
evals and report which clear the baseline noise band):

```bash
python replication_audit_real.py --run <your_greedy_run_id> --reps 15
```

The `<your_greedy_run_id>` is the `run_id` printed at the start of the greedy run
(also the folder name under `skeptic_gate/results/`). A "win" that, on re-test,
falls back inside `baseline_mean + z·baseline_std` is flagged **VANISHES** — that's
a greedy false positive the skeptic arm would have rejected.

---

## 9. Where the outputs land & what to read

Every run writes to `skeptic_gate/results/<run_id>/`:

- `summary.json` — headline: `baseline_score`, `best_score`, `improvement`,
  `n_proposals`, `n_accepted`, `n_culled`, `n_crashed`, `eval_calls`, `budget_spent`.
- `results.jsonl` — one line per step (intent, status, all candidate scores,
  `delta_hat`/`se_hat`, wall time, snapshot path).
- `proposals/` — the actual `MyMethod.py` the agent wrote at each step.
- `meta.json` — the exact run config (arm, budget, model, fidelity, seeds).

The story to extract: **greedy** reports a bigger `improvement` but the audit shows
some of it VANISHES on re-test; **causal/coh+causal** report a smaller but
*replicable* improvement with fewer false accepts at equal budget.

---

## 10. Integrity rules (please keep these)

- **Equal budget across arms.** The baseline measurement and every gate re-run are
  charged to the same budget — don't give one arm more evals.
- **Same `--reset-from`** for every arm (the default), so they start identical.
- **GPU only for MU numbers.** Stationary on V100; never quote laptop/MPS MU scores.
- **Validate the cost-lever weight** (Section 7) before any equal-budget claim that
  mixes `cheap` and `full`.
- Report **both** arms (greedy and skeptic) — the honest framing is a
  characterization/audit ("single evals lie"), not "we beat greedy."

---

## 11. Pushing results back

Result artifacts under `skeptic_gate/results/` are gitignored in this repo. To
share a run, copy the small JSON/JSONL (not the weights) under the tracked
`results/` folder at the repo root, then commit:

```bash
mkdir -p ~/MLSS-Hackathon-2026/results/mu_<run_id>
cp skeptic_gate/results/<run_id>/{summary.json,results.jsonl,meta.json} \
   ~/MLSS-Hackathon-2026/results/mu_<run_id>/
cd ~/MLSS-Hackathon-2026 && git add results/mu_<run_id> && git commit -m "MU run <run_id>" && git push
```

(Pull the latest `main` first — the team merges straight to `main`, no PRs.)

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `run_mlrc.py` can't find the eval / `ENV_DIR` missing | MLRC-Bench must live at `<repo_root>/MLRC-Bench`; clone/symlink it there. |
| `cuda: False` | Re-run `setup/gcp_setup.sh` with the right `CUDA_TAG`; confirm the driver with `nvidia-smi`. |
| Eval errors about Kaggle | The patch makes Kaggle lazy for the dev phase; if it still complains, confirm `mlrc-local.patch` applied (`git -C MLRC-Bench apply --reverse --check setup/mlrc-local.patch`). |
| `--fidelity cheap` runs feel just as slow | You skipped Section 3 — `evaluation.py:34` isn't reading `MU_NUM_MODELS`. |
| Crash score `-10.0` shows up | The agent proposed a broken edit; greedy "wastes an eval then discards" — that's expected and is exactly what the coherence gate culls. |
| Missing weights / data | Re-run `prepare.py` (Section 2). |
