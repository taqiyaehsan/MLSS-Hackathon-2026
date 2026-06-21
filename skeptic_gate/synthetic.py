"""The synthetic control: a fully-controllable toy version of the research loop.

Two dials (HANDOFF step 7):
  - sigma:   measurement noise on each eval (the wobble the causal gate detects)
  - p_good:  base-rate of genuinely-good candidate changes

A candidate has a HIDDEN true effect `delta` on the (hidden) true performance T.
Greedy never sees delta; it only sees noisy measurements T+delta+N(0,sigma).
Because we KNOW delta, we can score arms on TRUE performance and run the
replication audit exactly (which accepted changes were really null/harmful).

World model per candidate (drawn from the proposal RNG so the candidate stream
is IDENTICAL across arms at a given outer seed -- only the gate differs):

  with prob p_broken : broken (coherence-cullable; if evaluated, scores garbage)
  else with prob p_good : good,  delta = +HalfNormal(good_scale)
  else                  : null/harmful, delta ~ Normal(bad_mean<=0, bad_sd)

`evaluate(candidate, seed)` returns the noisy measured score of the model that
WOULD result from applying the candidate on top of the current incumbent's true
performance: (T_current + delta) + N(0, sigma). Broken candidates return a large
negative score (a "crash").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from gates import (Budget, Incumbent, GreedyPolicy, CausalPolicy,
                   CoherenceWrapper, run_loop, Fidelity, FULL)


@dataclass
class Candidate:
    delta: float           # hidden true effect on performance
    broken: bool
    truth: dict = field(default_factory=dict)


@dataclass
class SyntheticConfig:
    sigma: float = 0.03          # measurement noise
    p_good: float = 0.25         # base-rate of genuinely good changes
    p_broken: float = 0.15       # fraction of proposals that are broken code
    good_scale: float = 0.04     # scale of positive effects (HalfNormal)
    # Untested changes more often HURT than help; under noise you can't tell.
    bad_mean: float = -0.03      # mean effect of null/harmful changes (<=0)
    bad_sd: float = 0.03         # spread of null/harmful effects
    coherence_cost: float = 0.05 # budget units for one cheap coherence check
    crash_score: float = -10.0   # measured score of a broken candidate
    # Diminishing returns: if set, positive gains shrink as performance T
    # approaches the ceiling (improvements get harder near the optimum);
    # regressions still apply in full (you can always break a good model).
    # None => unbounded (gains accumulate linearly forever).
    ceiling: Optional[float] = None


class SyntheticWorld:
    """Holds the hidden true performance T and generates candidates / evals."""

    def __init__(self, cfg: SyntheticConfig, proposal_rng: np.random.Generator,
                 noise_rng: np.random.Generator, sigma_mult: float = 1.0):
        self.cfg = cfg
        self.prng = proposal_rng     # generates candidate stream (shared across arms)
        self.nrng = noise_rng        # generates eval noise (per arm)
        # cost lever: a cheaper fidelity inflates the per-eval noise by this factor
        # (fewer epochs / inner models => a noisier measurement of the same change).
        self.sigma_mult = sigma_mult
        self.T = 0.0                 # true performance, baseline = 0
        # bookkeeping for the audit
        self.accepted_truth: list[dict] = []
        # richer per-accept records for the replication audit
        self.accepted_records: list[dict] = []

    # -- proposal stream -----------------------------------------------------
    def propose(self, _rng) -> Candidate:
        cfg = self.cfg
        if self.prng.random() < cfg.p_broken:
            return Candidate(delta=0.0, broken=True,
                             truth={"kind": "broken", "delta": 0.0})
        if self.prng.random() < cfg.p_good:
            delta = abs(self.prng.normal(0.0, cfg.good_scale))
            return Candidate(delta=delta, broken=False,
                             truth={"kind": "good", "delta": delta})
        delta = self.prng.normal(cfg.bad_mean, cfg.bad_sd)
        return Candidate(delta=delta, broken=False,
                         truth={"kind": "null", "delta": delta})

    # -- realized effect (diminishing returns near the ceiling) --------------
    def _realized(self, delta: float) -> float:
        """The effect a change actually has if applied at the current T. With a
        ceiling, positive gains shrink with remaining headroom; negative effects
        (regressions) apply in full. Sign is preserved, so good/harmful labels
        are unchanged -- only magnitudes diminish. Used by BOTH evaluate() and
        on_accept() so the measured score and the realized gain never disagree."""
        if self.cfg.ceiling is None or delta <= 0:
            return delta
        headroom = max(0.0, (self.cfg.ceiling - self.T) / self.cfg.ceiling)
        return delta * headroom

    # -- noisy evaluation ----------------------------------------------------
    def evaluate(self, candidate: Candidate, seed: int) -> float:
        if candidate.broken:
            return self.cfg.crash_score
        true_val = self.T + self._realized(candidate.delta)
        return true_val + self.nrng.normal(0.0, self.cfg.sigma * self.sigma_mult)

    # -- commit on accept ----------------------------------------------------
    def on_accept(self, candidate: Candidate, decision) -> None:
        t_before = self.T
        realized = self._realized(candidate.delta)  # depends on current T; compute first
        self.T += realized                          # TRUE performance moves by realized effect
        if self.cfg.ceiling is not None:
            self.T = min(self.T, self.cfg.ceiling)
        self.accepted_truth.append(candidate.truth)
        self.accepted_records.append({
            "kind": candidate.truth["kind"],
            "nominal_delta": candidate.delta,
            "realized_delta": realized,   # the effect it ACTUALLY had at accept time
            "T_before": t_before,
        })

    def is_broken(self, candidate: Candidate) -> bool:
        return candidate.broken


# ---------------------------------------------------------------------------
# Running one arm
# ---------------------------------------------------------------------------

@dataclass
class ArmResult:
    arm: str
    sigma: float
    p_good: float
    outer_seed: int
    true_performance: float          # final hidden T (the thing that matters)
    n_steps: int
    n_accepted: int
    n_culled: int
    n_false_accepts: int             # accepted with true delta <= 0
    n_true_gain_accepts: int         # accepted with true delta > 0
    true_gain_kept: float            # sum of true delta over accepted GOOD changes
    false_gain_kept: float           # sum of true delta over accepted non-good (<=0; erodes T)
    budget_spent: float
    # replication-audit view: of accepted changes, how many were really null/harmful
    audit_kept: int = 0
    audit_survive: int = 0


def _build_policy(arm: str, world: SyntheticWorld, eval_cost: float = 1.0):
    if arm == "greedy":
        return GreedyPolicy(eval_cost=eval_cost)
    if arm == "causal":
        return CausalPolicy(k0=2, k_max=6, z=1.0, eval_cost=eval_cost)
    if arm == "coh+greedy":
        return CoherenceWrapper(GreedyPolicy(eval_cost=eval_cost), world.is_broken,
                                check_cost=world.cfg.coherence_cost)
    if arm == "coh+causal":
        return CoherenceWrapper(CausalPolicy(k0=2, k_max=6, z=1.0, eval_cost=eval_cost),
                                world.is_broken, check_cost=world.cfg.coherence_cost)
    raise ValueError(arm)


def run_arm(arm: str, cfg: SyntheticConfig, budget_units: float,
            outer_seed: int, return_world: bool = False, fidelity: Fidelity = FULL):
    # Candidate stream depends ONLY on outer_seed+cfg -> identical across arms.
    prng = np.random.default_rng(outer_seed * 1000 + 7)
    # Eval noise stream is arm-independent in seed so arms face comparable noise.
    nrng = np.random.default_rng(outer_seed * 1000 + 99)
    # cost lever: a cheaper fidelity charges less budget per eval but is noisier.
    sigma_mult = float(fidelity.params.get("sigma_mult", 1.0))
    eval_cost = fidelity.cost
    world = SyntheticWorld(cfg, prng, nrng, sigma_mult=sigma_mult)
    policy = _build_policy(arm, world, eval_cost=eval_cost)
    budget = Budget(budget_units)

    # Baseline incumbent: measure the starting point with 2 seeds so the causal
    # gate has a reference band. Charge these to the budget (counts as spend).
    base_scores = []
    for s in range(2):
        if budget.can_afford(eval_cost):
            base_scores.append(world.evaluate(
                Candidate(delta=0.0, broken=False), 5_000 + s))
            budget.charge(eval_cost)
    if not base_scores:
        base_scores = [0.0]

    logs = run_loop(world.propose, world.evaluate, policy, budget,
                    world.on_accept, base_scores, rng=prng)

    # ---- metrics from ground truth ----
    accepted = [t for t in world.accepted_truth]
    n_false = sum(1 for t in accepted if t["delta"] <= 0)
    n_true = sum(1 for t in accepted if t["delta"] > 0)
    true_gain = sum(t["delta"] for t in accepted if t["delta"] > 0)
    false_gain = sum(t["delta"] for t in accepted if t["delta"] <= 0)
    n_culled = sum(1 for L in logs if L.culled)

    result = ArmResult(
        arm=arm, sigma=cfg.sigma, p_good=cfg.p_good, outer_seed=outer_seed,
        true_performance=world.T, n_steps=len(logs),
        n_accepted=len(accepted), n_culled=n_culled,
        n_false_accepts=n_false, n_true_gain_accepts=n_true,
        true_gain_kept=true_gain, false_gain_kept=false_gain,
        budget_spent=budget.spent,
        audit_kept=len(accepted), audit_survive=n_true,
    )
    if return_world:
        return result, world
    return result
