"""Multi-agent method portfolio + 3-axis Pareto selection (the real-task version).

Agent = method (locked). For a task, one agent is created per applicable method
(see methods.METHODS); each agent tunes ONLY its method's bounded config through
the SAME gates.py (propose -> coherence + causal/greedy -> keep). That is LAYER 1.

LAYER 2 is the ship decision, and it is deliberately NOT an LLM judge and NOT a
single-accuracy argmax. We pool every coherent config any agent evaluated, then
honestly re-score each over S seeds and place it in a 3-axis space:

    accuracy   = mean held-out-VAL accuracy over S seeds   (higher better)
                 -- "confirmed", not a single lucky eval; the winner's-curse
                    regresses out because a lucky config has a lower mean.
    stability  = std of that accuracy over S seeds          (lower better)
    cost       = FLOPs (MACs) to train                       (lower better)
                 -- hardware-independent; wall-clock is reported alongside as
                    on-this-hardware context but is NOT a frontier axis.

We then report the PARETO FRONTIER (non-dominated points) -- no auto-pick. The
held-out TEST split is touched ONCE per frontier point for its final number; it is
never used to build the frontier (that would be selection-on-test). The naive pick
(max single-seed val accuracy) is flagged so you can see whether it is dominated.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import task_data
import methods as M
from gates import (Budget, GreedyPolicy, CausalPolicy, CoherenceWrapper, run_loop,
                   Fidelity, FULL)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class Candidate:
    config: dict
    intent: str
    static_ok: bool = True
    static_reason: str = ""
    truth: Optional[dict] = None


# ---------------------------------------------------------------------------
# Per-method proposers (programmatic = free/reproducible; LLM = the real agent)
# ---------------------------------------------------------------------------

class ProgrammaticMethodProposer:
    """Bounded random-walk mutator over ONE method's config space. Occasionally
    emits an out-of-bounds config so the coherence gate has real culls to make."""

    def __init__(self, method: M.Method, p_broken: float = 0.1):
        self.method = method
        self.p_broken = p_broken

    def propose(self, incumbent: dict, rng: np.random.Generator, history=None) -> Candidate:
        cfg = dict(incumbent)
        if rng.random() < self.p_broken:
            k = rng.choice(list(self.method.bounds))
            lo, hi = self.method.bounds[k]
            cfg[k] = hi * 10          # clearly out of range -> culled
            ok, reason = self.method.validate(cfg)
            return Candidate(cfg, f"[broken:{k}]", ok, reason, {"kind": "broken"})
        keys = list(self.method.bounds) + list(self.method.cats)
        for k in rng.choice(keys, size=int(rng.integers(1, 3)), replace=False):
            if k in self.method.cats:
                cfg[k] = str(rng.choice(self.method.cats[k]))
            else:
                lo, hi = self.method.bounds[k]
                if k in M._INT_KEYS:
                    cfg[k] = int(np.clip(round(cfg[k] * rng.choice([0.5, 1.5, 2.0])), lo, hi))
                else:
                    cfg[k] = float(np.clip(cfg[k] * rng.choice([0.3, 0.5, 2.0, 3.0]), lo, hi))
        ok, reason = self.method.validate(cfg)
        return Candidate(cfg, "mutate " + str(cfg), ok, reason, {"kind": "proposal"})


class LLMMethodProposer:
    """The real OpenAI agent, locked to ONE method, tuning its config."""

    def __init__(self, method: M.Method, model: str = "gpt-4.1-mini", temperature: float = 0.8):
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env")
        from openai import OpenAI
        self.client = OpenAI()
        self.method = method
        self.model = model
        self.temperature = temperature

    def _brief(self) -> str:
        rng = "\n".join(f"      {k:13s} float [{lo}, {hi}]" if k not in M._INT_KEYS
                        else f"      {k:13s} int   [{lo}, {hi}]"
                        for k, (lo, hi) in self.method.bounds.items())
        cats = "\n".join(f"      {k:13s} one of {list(v)}" for k, v in self.method.cats.items())
        return (f"You are tuning {self.method.desc}. Each trial trains it from "
                f"scratch and reports validation accuracy (higher better). Propose "
                f"ONE new config you believe will raise val acc.\n\n"
                f"Allowed keys and ranges (stay inside or the trial is discarded):\n"
                f"{rng}\n{cats}\n\n"
                f'Respond with JSON: {{"intent": "<one sentence>", "config": {{all keys}}}}.')

    def propose(self, incumbent: dict, rng=None, history=None) -> Candidate:
        hist = history or []
        htxt = "\n".join(f"  - val={h.get('score', float('nan')):.4f}: {h.get('config')}"
                         for h in hist[-6:]) or "  (none yet)"
        user = (f"Current best config: {json.dumps(incumbent)}\n\nHistory:\n{htxt}\n\n"
                f"Propose ONE improved config as the JSON object.")
        try:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": self._brief()},
                          {"role": "user", "content": user}])
            obj = json.loads(resp.choices[0].message.content)
            cfg = self._coerce(obj.get("config", {}), incumbent)
            intent = (obj.get("intent", "") or "").strip()[:200]
        except Exception as e:  # noqa: BLE001
            print(f"    [llm:{self.method.name}] error: {type(e).__name__}: {e}")
            return Candidate(dict(incumbent), "[llm error]", False, "llm_error", {"kind": "llm_error"})
        ok, reason = self.method.validate(cfg)
        return Candidate(cfg, intent or "(no intent)", ok, reason, {"kind": "llm"})

    def _coerce(self, raw: dict, fallback: dict) -> dict:
        cfg = dict(fallback)
        for k in self.method.bounds:
            if k in raw:
                try:
                    cfg[k] = int(round(float(raw[k]))) if k in M._INT_KEYS else float(raw[k])
                except Exception:
                    pass
        for k, allowed in self.method.cats.items():
            if str(raw.get(k)) in allowed:
                cfg[k] = str(raw[k])
        return cfg


# ---------------------------------------------------------------------------
# Per-method world (Layer 1): drives one agent's pipeline through gates.py
# ---------------------------------------------------------------------------

class MethodWorld:
    def __init__(self, method: M.Method, data: dict, proposer,
                 prng: np.random.Generator, fidelity: Fidelity = FULL):
        self.method = method
        self.data = data
        self.proposer = proposer
        self.prng = prng
        self.fidelity = fidelity
        self.n_train = fidelity.params.get("train_subset")
        self.best_config = dict(method.baseline)
        self.history: list[dict] = []
        self.pool: list[dict] = [dict(method.baseline)]   # every coherent config seen
        self.eval_calls = 0

    def propose(self, _rng=None) -> Candidate:
        cand = self.proposer.propose(self.best_config, self.prng, self.history)
        if cand.static_ok:
            self.pool.append(dict(cand.config))
        return cand

    def evaluate(self, candidate: Candidate, seed: int) -> float:
        self.eval_calls += 1
        if not candidate.static_ok:
            return -1.0
        acc, _ = M.train_score(self.method, candidate.config, self.data, seed, "va", self.n_train)
        return acc

    def on_accept(self, candidate: Candidate, decision) -> None:
        self.best_config = dict(candidate.config)
        mean = float(np.mean(decision.candidate_scores)) if decision.candidate_scores else None
        self.history.append({"config": dict(candidate.config), "score": mean, "accepted": True})

    def is_broken(self, candidate: Candidate) -> bool:
        return not candidate.static_ok


def _policy(arm: str, world: MethodWorld):
    if arm == "greedy":
        return GreedyPolicy()
    if arm == "causal":
        return CausalPolicy(k0=2, k_max=6, z=1.0)
    if arm == "coh+causal":
        return CoherenceWrapper(CausalPolicy(k0=2, k_max=6, z=1.0), world.is_broken)
    raise ValueError(arm)


def run_method_agent(method: M.Method, data: dict, arm: str, budget_units: float,
                     seed: int, use_llm: bool, model: str, fidelity: Fidelity) -> dict:
    """Layer 1: one agent tunes one method to a budget. Returns its pool + best."""
    prng = np.random.default_rng(seed * 1000 + 7)
    proposer = (LLMMethodProposer(method, model) if use_llm
                else ProgrammaticMethodProposer(method))
    world = MethodWorld(method, data, proposer, prng, fidelity)
    policy = _policy(arm, world)
    budget = Budget(budget_units)
    # baseline incumbent band: 2 real evals of the method's baseline config
    base = []
    for s in range(2):
        if budget.can_afford(1.0):
            base.append(world.evaluate(Candidate(dict(method.baseline), "baseline"), 5000 + s))
            budget.charge(1.0)
    run_loop(world.propose, world.evaluate, policy, budget, world.on_accept,
             base or [0.0], rng=prng)
    return {"method": method.name, "best_config": world.best_config,
            "pool": world.pool, "eval_calls": world.eval_calls}


# ---------------------------------------------------------------------------
# Layer 2: honest re-score of the pooled configs + 3-axis Pareto frontier
# ---------------------------------------------------------------------------

def _cfg_key(name: str, cfg: dict) -> str:
    return name + "|" + json.dumps(cfg, sort_keys=True)


def score_pool(task: str, data: dict, pool: list[tuple[str, dict]], n_seeds: int,
               seed0: int = 90_000) -> list[dict]:
    """For each (method, config): mean+std VAL acc over seeds (frontier axes),
    FLOPs cost, median wall-clock (context), and a one-touch TEST acc (report
    only). De-duplicated by (method, config)."""
    seen, rows = set(), []
    for name, cfg in pool:
        k = _cfg_key(name, cfg)
        if k in seen:
            continue
        seen.add(k)
        meth = M.METHODS[name]
        va = [M.train_score(meth, cfg, data, seed0 + i, "va") for i in range(n_seeds)]
        accs = [a for a, _ in va]
        walls = [t for _, t in va]
        test_acc, _ = M.train_score(meth, cfg, data, seed0 + 777, "te")  # one touch
        rows.append({
            "method": name, "config": cfg,
            "acc": float(np.mean(accs)),               # frontier axis 1 (max)
            "acc_single": float(accs[0]),              # one noisy eval = the naive pick's view
            "stability": float(np.std(accs, ddof=1)),  # frontier axis 2 (min)
            "cost_macs": int(M.train_macs(meth, cfg, data)),  # frontier axis 3 (min)
            "wall_ms": float(np.median(walls) * 1000), # context only
            "test_acc": float(test_acc),               # report only
        })
    return rows


def pareto_frontier(rows: list[dict]) -> list[dict]:
    """3-axis non-dominated set: acc (max), stability (min), cost_macs (min)."""
    def dominates(a, b):
        ge = (a["acc"] >= b["acc"] and a["stability"] <= b["stability"]
              and a["cost_macs"] <= b["cost_macs"])
        gt = (a["acc"] > b["acc"] or a["stability"] < b["stability"]
              or a["cost_macs"] < b["cost_macs"])
        return ge and gt
    front = []
    for r in rows:
        if not any(dominates(o, r) for o in rows if o is not r):
            front.append(r)
    return front


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_portfolio(task: str, *, arm: str = "causal", budget: float = 10.0,
                  seed: int = 0, use_llm: bool = False, model: str = "gpt-4.1-mini",
                  frontier_seeds: int = 10, fidelity: Fidelity = FULL) -> dict:
    t0 = time.time()
    data = task_data.load_task(task)
    method_names = M.methods_for_task(task)
    proposer_kind = f"LLM agent ({model})" if use_llm else "programmatic mutator"
    print("=" * 80)
    print(f"PORTFOLIO [{task}]  agents (1 per method) = {method_names}")
    print(f"Layer 1 proposer = {proposer_kind}, arm={arm}, budget={budget}/method")
    print("=" * 80)

    # -- Layer 1: one agent per method --
    pool: list[tuple[str, dict]] = []
    agents = []
    for name in method_names:
        print(f"  agent[{name}] tuning...")
        a = run_method_agent(M.METHODS[name], data, arm, budget, seed, use_llm, model, fidelity)
        agents.append(a)
        pool += [(name, cfg) for cfg in a["pool"]]
    print(f"  pooled {len(pool)} coherent (method,config) candidates "
          f"({len({_cfg_key(n, c) for n, c in pool})} unique)")

    # -- Layer 2: re-score + Pareto --
    print(f"  re-scoring over {frontier_seeds} seeds (val=frontier, test=report)...")
    rows = score_pool(task, data, pool, frontier_seeds)
    front = pareto_frontier(rows)
    # naive selector = ship whatever scored best on ONE noisy val eval (the
    # winner's-curse pick). It ignores stability and cost entirely.
    naive = max(rows, key=lambda r: r["acc_single"])
    front_keys = {_cfg_key(r["method"], r["config"]) for r in front}

    print("\n  === PARETO FRONTIER (ship-candidates; no auto-pick) ===")
    print(f"  {'method':7s} {'acc(val)':>9s} {'stab':>7s} {'cost(GMACs)':>12s} "
          f"{'wall(ms)':>9s} {'test':>6s}")
    for r in sorted(front, key=lambda r: -r["acc"]):
        print(f"  {r['method']:7s} {r['acc']:9.4f} {r['stability']:7.4f} "
              f"{r['cost_macs']/1e9:12.2f} {r['wall_ms']:9.0f} {r['test_acc']:6.3f}")
    dominated = [r for r in rows if _cfg_key(r["method"], r["config"]) not in front_keys]
    naive_on = _cfg_key(naive["method"], naive["config"]) in front_keys
    print(f"\n  {len(front)} on frontier, {len(dominated)} dominated (off-frontier).")
    print(f"  naive single-eval pick: {naive['method']} (val_single={naive['acc_single']:.4f}, "
          f"confirmed acc={naive['acc']:.4f}, stab={naive['stability']:.4f}, "
          f"test={naive['test_acc']:.3f})  -> {'on frontier' if naive_on else 'DOMINATED'}")

    out = {"task": task, "arm": arm, "proposer": proposer_kind, "budget": budget,
           "methods": method_names, "frontier_seeds": frontier_seeds,
           "n_candidates": len(rows), "frontier": front, "all_rows": rows,
           "wall_s": time.time() - t0}
    outdir = RESULTS_DIR / f"portfolio_{task}" / ("llm" if use_llm else "prog")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "frontier.json").write_text(json.dumps(out, indent=2))
    print(f"\n  saved -> {outdir / 'frontier.json'}  (wall {out['wall_s']:.1f}s)")
    return out


_USAGE = ("usage: python portfolio_real.py <task> [llm] [budget] [seeds]\n"
          "  tasks: fashionmnist | magic")

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(_USAGE)
    else:
        task = argv[0]
        use_llm = "llm" in argv[1:]
        nums = [a for a in argv[1:] if a != "llm"]
        bud = float(nums[0]) if len(nums) > 0 else 10.0
        fs = int(nums[1]) if len(nums) > 1 else 10
        run_portfolio(task, use_llm=use_llm, budget=bud, frontier_seeds=fs)
