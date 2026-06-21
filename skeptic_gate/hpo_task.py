"""A REAL (not mocked) lightweight ML research task for the skeptic-gate pipeline.

Unlike synthetic.py -- whose evaluate() is a draw from a distribution (no model,
no data, no training) -- this is an ACTUAL train-and-score loop:

  * Real data: sklearn `digits` (1797 handwritten 8x8 images, 10 classes). It
    ships with sklearn, so there is NO download (the thing blocking Rainfall).
  * Real model: a small MLP, really trained from scratch on every eval.
  * Real, noisy metric: validation accuracy. It genuinely wobbles run-to-run
    (random weight init + minibatch shuffle) -- the noise the causal gate exists
    to detect -- and on CPU it is STATIONARY (no MPS thermal drift) and cheap
    (~0.2 s/eval), so seed-repeats and the replication audit are affordable.

The agent proposes HYPERPARAMETER edits (the bounded "config" edit surface we
planned for Rainfall). The SAME task-agnostic gates.py drives it: only the
AcceptPolicy differs across arms.

Why this task exists (see HANDOFF/PROGRESS):
  - The abstract synthetic carries the controlled regime curve but isn't a model.
  - The real MLRC unlearning eval on this laptop was NON-STATIONARY (thermal
    drift) -> no trustworthy numbers; Rainfall needs a multi-GB GCP download.
  - This is the missing middle: a genuine train/score loop that runs the real
    pipeline and produces real numbers TODAY. It is NOT an MLRC-Bench benchmark;
    it supplements, it does not replace, the required benchmark run.

Integrity knobs (same spirit as synthetic.py):
  - A config's TRUE performance is unknown a priori (real training); we estimate
    it by averaging accuracy over many seeds. That estimate is the ground truth
    for the replication audit -- exactly what greedy can't see in one run.
  - Final Layer-1 numbers are reported on a HELD-OUT TEST split the proposer and
    the gate never touch, so "progress" is not selection-on-the-eval-set.

Higher score is better (validation accuracy in [0, 1]).
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from gates import (Budget, Incumbent, GreedyPolicy, CausalPolicy, CoherenceWrapper,
                   run_loop, Fidelity, FULL)

# CPU only, on purpose: the task must be STATIONARY (the MPS unlearning eval was
# not). digits is tiny, so CPU is both fast and drift-free.
DEVICE = torch.device("cpu")
torch.set_num_threads(1)  # determinism + avoid contention spikes between evals

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Data: load once, fixed train / val / test split (test is held out from the
# whole loop -- the proposer and the gate never see it).
# ---------------------------------------------------------------------------

def _load_data():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32)
    # fixed split: 60% train, 20% val (the gate's eval set), 20% test (held out)
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=0.20, random_state=0, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=0, stratify=y_tmp)  # 0.25*0.8=0.2
    scaler = StandardScaler().fit(X_tr)
    to = lambda A: torch.from_numpy(scaler.transform(A).astype(np.float32))
    return {
        "X_tr": to(X_tr), "y_tr": torch.from_numpy(y_tr),
        "X_va": to(X_va), "y_va": torch.from_numpy(y_va),
        "X_te": to(X_te), "y_te": torch.from_numpy(y_te),
    }


_DATA = _load_data()
N_FEATURES, N_CLASSES = 64, 10


# ---------------------------------------------------------------------------
# Config edit surface + the real model
# ---------------------------------------------------------------------------

# Deliberately mediocre baseline so a real agent has genuine headroom to improve.
BASELINE_CONFIG = {
    "hidden": 16, "lr": 0.01, "dropout": 0.0, "weight_decay": 0.0,
    "epochs": 6, "batch_size": 64, "activation": "relu",
}

# Bounds the static coherence check enforces (anything outside -> "broken").
BOUNDS = {
    "hidden": (4, 256), "lr": (1e-4, 1.0), "dropout": (0.0, 0.9),
    "weight_decay": (0.0, 1e-1), "epochs": (1, 40), "batch_size": (8, 256),
}
ACTIVATIONS = ("relu", "tanh")


def validate_config(cfg: dict) -> tuple[bool, str]:
    """Cheap static check (Gate-1 building block). Does NOT train anything."""
    for k in BASELINE_CONFIG:
        if k not in cfg:
            return False, f"missing key {k}"
    if cfg["activation"] not in ACTIVATIONS:
        return False, f"bad activation {cfg['activation']!r}"
    for k, (lo, hi) in BOUNDS.items():
        v = cfg[k]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False, f"{k} not numeric"
        if not (lo <= v <= hi):
            return False, f"{k}={v} out of [{lo},{hi}]"
    for k in ("hidden", "epochs", "batch_size"):
        if int(cfg[k]) != cfg[k]:
            return False, f"{k} must be integer"
    return True, ""


def _build_mlp(cfg: dict) -> nn.Module:
    act = nn.ReLU() if cfg["activation"] == "relu" else nn.Tanh()
    return nn.Sequential(
        nn.Linear(N_FEATURES, int(cfg["hidden"])), act,
        nn.Dropout(float(cfg["dropout"])),
        nn.Linear(int(cfg["hidden"]), N_CLASSES),
    ).to(DEVICE)


def train_eval(cfg: dict, seed: int, split: str = "va",
               n_train: Optional[int] = None) -> float:
    """Really train an MLP with `cfg` from scratch (seed = init + shuffle) and
    return accuracy on `split` ('va' = the gate's eval set, 'te' = held-out test).
    This is the genuine, noisy, stationary measurement -- ~0.2 s on CPU.

    `n_train`: if set, each eval trains on a fresh seed-controlled random SUBSET
    of the training data. This is the real-world noise dial -- less data => higher
    run-to-run variance -- and it doubles as the cost lever (a smaller subset is a
    cheaper, noisier eval). None => use all training data (lowest noise)."""
    torch.manual_seed(seed)
    model = _build_mlp(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]),
                           weight_decay=float(cfg["weight_decay"]))
    loss_fn = nn.CrossEntropyLoss()
    Xtr, ytr = _DATA["X_tr"], _DATA["y_tr"]
    n = Xtr.shape[0]
    if n_train is not None and n_train < n:
        sub = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 12345))[:n_train]
        Xtr, ytr, n = Xtr[sub], ytr[sub], n_train
    bs = int(cfg["batch_size"])
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(int(cfg["epochs"])):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
    model.eval()
    Xev, yev = _DATA[f"X_{split}"], _DATA[f"y_{split}"]
    with torch.no_grad():
        acc = (model(Xev).argmax(1) == yev).float().mean().item()
    return acc


def true_score(cfg: dict, n_seeds: int = 30, split: str = "va",
               seed0: int = 90_000, n_train: Optional[int] = None) -> float:
    """Ground-truth performance estimate: mean accuracy over many fresh seeds at
    the SAME fidelity (`n_train`) the loop runs at -- a single noisy eval is a
    1-sample estimate of this, and the replication audit re-discovers it.
    Expensive on purpose; used offline only."""
    return float(np.mean([train_eval(cfg, seed0 + i, split, n_train) for i in range(n_seeds)]))


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    config: dict
    intent: str
    static_ok: bool = True
    static_reason: str = ""
    truth: Optional[dict] = None   # filled with the proposer's kind tag (for logs)


# ---------------------------------------------------------------------------
# Proposer: a bounded, incumbent-relative hyperparameter mutator.
# (Free + reproducible so the replication audit can run at scale. The real LLM
#  proposer of mlrc_adapter.py would slot in here unchanged: it returns a config
#  dict + intent.) Occasionally emits an out-of-bounds config so Gate 1 has real
#  broken proposals to cull -- the analog of the LLM proposing code that crashes.
# ---------------------------------------------------------------------------

class HyperProposer:
    def __init__(self, p_broken: float = 0.12):
        self.p_broken = p_broken

    def propose(self, incumbent_cfg: dict, rng: np.random.Generator,
                history=None) -> Candidate:
        if rng.random() < self.p_broken:
            return self._broken(incumbent_cfg, rng)
        cfg = dict(incumbent_cfg)
        # mutate 1-2 fields, multiplicatively for scales, by steps for ints
        fields = list(rng.choice(
            ["hidden", "lr", "dropout", "weight_decay", "epochs",
             "batch_size", "activation"],
            size=int(rng.integers(1, 3)), replace=False))
        notes = []
        for f in fields:
            if f == "hidden":
                cfg["hidden"] = int(np.clip(round(cfg["hidden"] * rng.choice([0.5, 1.5, 2.0])), *BOUNDS["hidden"]))
            elif f == "lr":
                cfg["lr"] = float(np.clip(cfg["lr"] * rng.choice([0.3, 0.5, 2.0, 3.0]), *BOUNDS["lr"]))
            elif f == "dropout":
                cfg["dropout"] = float(np.clip(cfg["dropout"] + rng.choice([-0.1, 0.1, 0.2]), *BOUNDS["dropout"]))
            elif f == "weight_decay":
                cfg["weight_decay"] = float(np.clip(cfg["weight_decay"] + rng.choice([1e-4, 1e-3, 1e-2]), *BOUNDS["weight_decay"]))
            elif f == "epochs":
                cfg["epochs"] = int(np.clip(cfg["epochs"] + rng.choice([-2, 2, 4, 6]), *BOUNDS["epochs"]))
            elif f == "batch_size":
                cfg["batch_size"] = int(np.clip(round(cfg["batch_size"] * rng.choice([0.5, 2.0])), *BOUNDS["batch_size"]))
            elif f == "activation":
                cfg["activation"] = "tanh" if cfg["activation"] == "relu" else "relu"
            notes.append(f"{f}->{cfg[f]}")
        ok, reason = validate_config(cfg)
        return Candidate(cfg, intent="set " + ", ".join(notes),
                         static_ok=ok, static_reason=reason,
                         truth={"kind": "proposal"})

    def _broken(self, incumbent_cfg: dict, rng: np.random.Generator) -> Candidate:
        cfg = dict(incumbent_cfg)
        choice = rng.choice(["hidden0", "lrhuge", "epochs0"])
        if choice == "hidden0":
            cfg["hidden"] = 0
        elif choice == "lrhuge":
            cfg["lr"] = 5.0
        else:
            cfg["epochs"] = 0
        ok, reason = validate_config(cfg)
        return Candidate(cfg, intent=f"[broken:{choice}]",
                         static_ok=ok, static_reason=reason,
                         truth={"kind": "broken"})


# ---------------------------------------------------------------------------
# LLM proposer: a REAL autonomous agent (OpenAI gpt-4.1-mini) proposing config
# edits. Same .propose(incumbent, rng, history) interface as HyperProposer, so it
# is a drop-in -- the gates and loop don't know or care which proposer is used.
# Mirrors mlrc_adapter.OpenAIProposer; here the edit surface is a config dict.
# ---------------------------------------------------------------------------

_LLM_BRIEF = textwrap.dedent("""\
    You are an autonomous ML researcher tuning a small MLP classifier on the
    sklearn `digits` dataset (8x8 handwritten digits, 10 classes). Each trial
    trains the MLP from scratch and reports validation accuracy (higher better).
    Propose ONE new hyperparameter config that you believe will raise val acc.

    Allowed keys and ranges (stay inside or the trial is discarded as invalid):
      hidden        int   [4, 256]
      lr            float [1e-4, 1.0]
      dropout       float [0.0, 0.9]
      weight_decay  float [0.0, 0.1]
      epochs        int   [1, 40]
      batch_size    int   [8, 256]
      activation    one of "relu", "tanh"

    Make a meaningful, non-trivial change relative to the current best config.
    Respond with JSON: {"intent": "<one short sentence>", "config": {all 7 keys}}.
""")


class LLMConfigProposer:
    def __init__(self, model: str = "gpt-4.1-mini", temperature: float = 0.8):
        import textwrap as _tw  # noqa: F401  (textwrap imported at module top)
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env")
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def propose(self, incumbent_cfg: dict, rng=None, history=None) -> Candidate:
        hist = history or []
        hist_txt = "\n".join(
            f"  - {'ACCEPTED' if h.get('accepted') else 'tried'} "
            f"val={h.get('score', float('nan')):.4f}: {h.get('config')}"
            for h in hist[-6:]) or "  (none yet)"
        user = (f"Current best config: {json.dumps(incumbent_cfg)}\n\n"
                f"History (most recent last):\n{hist_txt}\n\n"
                f"Propose ONE improved config as the JSON object described.")
        try:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": _LLM_BRIEF},
                          {"role": "user", "content": user}])
            obj = json.loads(resp.choices[0].message.content)
            raw = obj.get("config", {})
            intent = (obj.get("intent", "") or "").strip()[:200]
        except Exception as e:  # noqa: BLE001
            print(f"  [llm proposer] error: {type(e).__name__}: {e}")
            return Candidate(dict(incumbent_cfg), intent="[llm error -> noop]",
                             static_ok=False, static_reason="llm_error",
                             truth={"kind": "llm_error"})
        cfg = self._coerce(raw, incumbent_cfg)
        ok, reason = validate_config(cfg)
        return Candidate(cfg, intent=intent or "(no intent)",
                         static_ok=ok, static_reason=reason,
                         truth={"kind": "llm"})

    @staticmethod
    def _coerce(raw: dict, fallback: dict) -> dict:
        """Coerce types only (NOT ranges): an out-of-range value stays 'broken'
        so the coherence gate has real invalid proposals to cull -- honest."""
        cfg = dict(fallback)
        for k in ("hidden", "epochs", "batch_size"):
            try: cfg[k] = int(round(float(raw[k])))
            except Exception: pass
        for k in ("lr", "dropout", "weight_decay"):
            try: cfg[k] = float(raw[k])
            except Exception: pass
        if str(raw.get("activation")) in ACTIVATIONS:
            cfg["activation"] = str(raw["activation"])
        return cfg


# ---------------------------------------------------------------------------
# The world: owns the incumbent config, runs real evals, records accepts.
# Mirrors MLRCWorld / SyntheticWorld so gates.py drives it unchanged.
# ---------------------------------------------------------------------------

class HPOWorld:
    def __init__(self, proposer, prng: np.random.Generator,
                 fidelity: Fidelity = FULL):
        self.proposer = proposer
        self.prng = prng
        self.fidelity = fidelity
        self.best_config = dict(BASELINE_CONFIG)
        self.eval_calls = 0
        # replication-audit ledger: every accept with the incumbent it replaced
        self.accepted_records: list[dict] = []
        self.history: list[dict] = []   # accepted configs+scores, for the LLM proposer
        self.step_counter = 0

    def propose(self, _rng=None) -> Candidate:
        self.step_counter += 1
        return self.proposer.propose(self.best_config, self.prng, self.history)

    def evaluate(self, candidate: Candidate, seed: int) -> float:
        """One real, independent, noisy eval (train from scratch on val split).
        Noise level is set by the active fidelity's `train_subset` (less data =>
        noisier + cheaper eval)."""
        self.eval_calls += 1
        if not candidate.static_ok:
            return -1.0  # a broken config "crashes" -> very low score (like a real crash)
        n_train = self.fidelity.params.get("train_subset")
        return train_eval(candidate.config, seed, "va", n_train=n_train)

    def on_accept(self, candidate: Candidate, decision) -> None:
        prev = dict(self.best_config)
        self.best_config = dict(candidate.config)
        mean = float(np.mean(decision.candidate_scores)) if decision.candidate_scores else None
        self.accepted_records.append({
            "step": self.step_counter,
            "config_before": prev,
            "config_after": dict(candidate.config),
            "apparent_scores": list(decision.candidate_scores),
            "apparent_mean": mean,
            "intent": candidate.intent,
        })
        self.history.append({"config": dict(candidate.config), "score": mean,
                             "accepted": True})

    def is_broken(self, candidate: Candidate) -> bool:
        return not candidate.static_ok


# ---------------------------------------------------------------------------
# Run one arm of the real pipeline
# ---------------------------------------------------------------------------

def _build_policy(arm: str, world: HPOWorld, eval_cost: float):
    if arm == "greedy":
        return GreedyPolicy(eval_cost=eval_cost)
    if arm == "causal":
        return CausalPolicy(k0=2, k_max=6, z=1.0, eval_cost=eval_cost)
    if arm == "coh+greedy":
        return CoherenceWrapper(GreedyPolicy(eval_cost=eval_cost), world.is_broken)
    if arm == "coh+causal":
        return CoherenceWrapper(CausalPolicy(k0=2, k_max=6, z=1.0, eval_cost=eval_cost),
                                world.is_broken)
    raise ValueError(arm)


def run_arm(arm: str, budget_units: float, outer_seed: int,
            fidelity: Fidelity = FULL):
    # Same proposer-rng seed across arms -> each arm faces the same proposer
    # behaviour (proposals diverge only because incumbents diverge -- realistic,
    # since a real agent always builds on its current best).
    prng = np.random.default_rng(outer_seed * 1000 + 7)
    world = HPOWorld(HyperProposer(), prng, fidelity=fidelity)
    eval_cost = fidelity.cost
    policy = _build_policy(arm, world, eval_cost)
    budget = Budget(budget_units)

    # Baseline incumbent band: 2 real evals of the baseline config.
    base_scores = []
    for s in range(2):
        if budget.can_afford(eval_cost):
            base_scores.append(world.evaluate(
                Candidate(dict(BASELINE_CONFIG), "baseline"), 5_000 + s))
            budget.charge(eval_cost)
    if not base_scores:
        base_scores = [0.0]

    logs = run_loop(world.propose, world.evaluate, policy, budget,
                    world.on_accept, base_scores, rng=prng)

    n_accepted = len(world.accepted_records)
    n_culled = sum(1 for L in logs if L.culled)
    return {
        "arm": arm, "outer_seed": outer_seed, "budget_units": budget_units,
        "fidelity": fidelity.name,
        "n_steps": len(logs), "n_accepted": n_accepted, "n_culled": n_culled,
        "eval_calls": world.eval_calls, "budget_spent": budget.spent,
        "final_config": dict(world.best_config),
        "accepted_records": world.accepted_records,
        "base_scores": base_scores,
    }


# ---------------------------------------------------------------------------
# Replication audit (HANDOFF step 14) on REAL training noise.
# For every change an arm ACCEPTED, re-measure the true (many-seed) effect vs the
# incumbent it replaced. A "vanished" win = accepted but true val gain <= 0.
# ---------------------------------------------------------------------------

def replication_audit(accepted_records: list[dict], n_seeds: int = 30,
                      n_train: Optional[int] = None) -> dict:
    rows = []
    for rec in accepted_records:
        true_after = true_score(rec["config_after"], n_seeds, "va", n_train=n_train)
        true_before = true_score(rec["config_before"], n_seeds, "va", n_train=n_train)
        true_gain = true_after - true_before
        apparent_gain = (rec["apparent_mean"] - true_before) if rec["apparent_mean"] is not None else None
        rows.append({
            "step": rec["step"], "intent": rec["intent"],
            "apparent_mean": rec["apparent_mean"],
            "true_after": true_after, "true_before": true_before,
            "true_gain": true_gain, "apparent_gain": apparent_gain,
            "survives": bool(true_gain > 0.0),
        })
    n = len(rows)
    survive = sum(1 for r in rows if r["survives"])
    return {"n_kept": n, "n_survive": survive,
            "n_vanished": n - survive, "rows": rows}


# ---------------------------------------------------------------------------
# Main: run the real pipeline, report real numbers
# ---------------------------------------------------------------------------

# Noise regimes: less training data per eval => a noisier measurement. This is a
# REAL operating point of the evaluator, not a knob on a mocked distribution.
#
# IMPORTANT (fair experiment): cost is held at 1.0 across all three regimes so
# every arm gets the SAME number of proposals (~budget) at every noise level --
# the sweep then isolates NOISE alone. (Tying a cheaper cost to noisier evals
# would also hand the noisy regimes ~10x more proposals, confounding the trend.)
# The cost lever (cheaper == fewer budget units per eval) is a SEPARATE axis,
# demonstrated independently; here it is deliberately neutral.
REGIMES = [
    ("low  (full data)", Fidelity("full", 1.0, {"train_subset": None})),
    ("med  (200 samples)", Fidelity("med", 1.0, {"train_subset": 200})),
    ("high (80 samples)", Fidelity("high", 1.0, {"train_subset": 80})),
]


def run_llm_demo(budget: float = 12.0, regime_idx: int = 1, arm: str = "greedy",
                 seed: int = 0):
    """Live demo: a REAL OpenAI agent proposes configs; the same gate decides.
    Short on purpose (a dozen evals) -- it spends real API calls + real training.
    Prints every proposal and the gate's verdict, then the held-out test number."""
    label, fid = REGIMES[regime_idx]
    print("=" * 78)
    print(f"LLM-AGENT DEMO  (proposer=gpt-4.1-mini, arm={arm}, regime={label}, "
          f"budget={budget})")
    print("The agent proposes configs; the SAME gates.py decides keep/discard.")
    print("=" * 78)
    n_train = fid.params.get("train_subset")
    world = HPOWorld(LLMConfigProposer(), np.random.default_rng(seed), fidelity=fid)
    eval_cost = fid.cost
    policy = _build_policy(arm, world, eval_cost)
    budget_obj = Budget(budget)

    base_scores = []
    for s in range(2):
        base_scores.append(world.evaluate(Candidate(dict(BASELINE_CONFIG), "baseline"), 5000 + s))
        budget_obj.charge(eval_cost)
    incumbent = Incumbent(scores=list(base_scores))
    print(f"baseline val band: {[round(x, 4) for x in base_scores]}  (config {BASELINE_CONFIG})\n")

    seed_counter, step = 10_000, 0
    while budget_obj.remaining() > 1e-9:
        cand = world.propose()
        seed_counter += 100
        dec = policy.decide(cand, world.evaluate, incumbent, budget_obj, seed_counter)
        scores = [round(x, 4) for x in dec.candidate_scores]
        verdict = ("CULL (incoherent)" if dec.culled else
                   ("ACCEPT" if dec.accepted else "reject"))
        print(f"  step {step:2d}: \"{cand.intent}\"")
        print(f"           config={cand.config}")
        print(f"           -> evals={scores}  [{verdict}: {dec.reason}]")
        if dec.accepted:
            world.on_accept(cand, dec)
            if dec.candidate_scores:
                incumbent.scores = list(dec.candidate_scores)
        else:
            world.history.append({"config": cand.config, "accepted": False,
                                  "score": float(np.mean(dec.candidate_scores)) if dec.candidate_scores else None})
        step += 1
        if dec.units_spent == 0 and not dec.culled:
            break

    base_te = true_score(BASELINE_CONFIG, 30, "te")
    final_te = true_score(world.best_config, 30, "te")
    print(f"\n  final config: {world.best_config}")
    print(f"  held-out TEST acc: baseline {base_te:.4f} -> agent {final_te:.4f} "
          f"({final_te - base_te:+.4f})   [accepts={len(world.accepted_records)}, "
          f"evals={world.eval_calls}, budget_spent={budget_obj.spent:.1f}]")


def _ms(xs: list) -> tuple[float, float]:
    """mean and (sample) std, std=0 for a single value."""
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def main(seeds=(0, 1, 2, 3, 4)):
    t0 = time.time()
    BUDGET = 40.0
    AUDIT_SEEDS = 30

    print("=" * 78)
    print("REAL TASK: MLP hyperparameter search on sklearn `digits` (no download)")
    print(f"Same gates.py as the synthetic; eval REALLY trains a model. "
          f"{len(seeds)} outer seeds.")
    print("=" * 78)
    print(f"\nBaseline config: {BASELINE_CONFIG}")
    print(f"\n{'noise regime':20s} {'eval sd':>8s} | "
          f"{'greedy acc / false / test':>27s} | {'causal acc / false / test':>27s}")
    print("-" * 90)

    sweep = []
    for label, fid in REGIMES:
        n_train = fid.params.get("train_subset")
        base_samples = [train_eval(BASELINE_CONFIG, 70_000 + i, "va", n_train)
                        for i in range(AUDIT_SEEDS)]
        eval_sd = float(np.std(base_samples, ddof=1))
        base_te_m, base_te_s = _ms([true_score(BASELINE_CONFIG, AUDIT_SEEDS, "te")])

        row = {"regime": label, "eval_sd": eval_sd, "fidelity": fid.name,
               "train_subset": n_train,
               "baseline_test_mean": base_te_m, "baseline_test_std": base_te_s,
               "arms": {}}
        for arm in ("greedy", "causal"):
            accs, falses, kepts, survs, tests = [], [], [], [], []
            for sd in seeds:
                r = run_arm(arm, BUDGET, sd, fid)
                tests.append(true_score(r["final_config"], AUDIT_SEEDS, "te"))
                aud = replication_audit(r["accepted_records"], AUDIT_SEEDS, n_train=n_train)
                accs.append(r["n_accepted"]); falses.append(aud["n_vanished"])
                kepts.append(aud["n_kept"]); survs.append(aud["n_survive"])
            acc_m, acc_s = _ms(accs); f_m, f_s = _ms(falses)
            k_m, k_s = _ms(kepts); s_m, s_s = _ms(survs); t_m, t_s = _ms(tests)
            row["arms"][arm] = {
                "n_accepted_mean": acc_m, "n_accepted_std": acc_s,
                "final_true_test_mean": t_m, "final_true_test_std": t_s,
                "audit": {"n_vanished_mean": f_m, "n_vanished_std": f_s,
                          "n_kept_mean": k_m, "n_kept_std": k_s,
                          "n_survive_mean": s_m, "n_survive_std": s_s},
            }
        g, c = row["arms"]["greedy"], row["arms"]["causal"]
        sweep.append(row)
        print(f"{label:20s} {eval_sd:8.4f} | "
              f"{g['n_accepted_mean']:4.1f} /{g['audit']['n_vanished_mean']:4.1f} /"
              f"{g['final_true_test_mean']:.3f}+-{g['final_true_test_std']:.3f} | "
              f"{c['n_accepted_mean']:4.1f} /{c['audit']['n_vanished_mean']:4.1f} /"
              f"{c['final_true_test_mean']:.3f}+-{c['final_true_test_std']:.3f}")

    hi = sweep[-1]
    print("\n" + "-" * 78)
    print(f"REPLICATION AUDIT @ {hi['regime']} regime (mean over {len(seeds)} seeds):")
    ga, ca = hi["arms"]["greedy"]["audit"], hi["arms"]["causal"]["audit"]
    print(f"  greedy: kept {ga['n_kept_mean']:.1f} 'wins', "
          f"{ga['n_vanished_mean']:.1f} VANISH on re-test "
          f"({ga['n_survive_mean']:.1f} survive)")
    print(f"  causal: kept {ca['n_kept_mean']:.1f} 'wins', "
          f"{ca['n_vanished_mean']:.1f} vanish "
          f"({ca['n_survive_mean']:.1f} survive)")
    print("\n" + "=" * 78)
    print("READING: as real eval noise rises (less data), greedy accepts more")
    print("lucky wins that VANISH on re-test; the causal gate holds the line.")
    print("=" * 78)

    out = {"task": "digits_mlp_hpo", "budget_units": BUDGET, "n_seeds": len(seeds),
           "seeds": list(seeds), "audit_seeds": AUDIT_SEEDS,
           "baseline_config": BASELINE_CONFIG, "sweep": sweep,
           "wall_s": time.time() - t0}
    outdir = RESULTS_DIR / "hpo_task"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {outdir / 'summary.json'}   (wall {out['wall_s']:.1f}s)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "llm":
        run_llm_demo()
    else:
        main()
