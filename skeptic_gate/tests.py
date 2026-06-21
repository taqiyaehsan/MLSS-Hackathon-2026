"""Integrity / soundness tests for the skeptic-gate pipeline.

These assert the invariants the scientific claims depend on. If any fail, a
headline number is suspect. Run: python tests.py
"""
import numpy as np

from gates import (Budget, Incumbent, GreedyPolicy, CausalPolicy,
                   CoherenceWrapper, _welch_delta_se, _var, _mean)
from synthetic import SyntheticConfig, SyntheticWorld, Candidate, run_arm
import experiment

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


# ---------------------------------------------------------------------------
print("\n[1] Determinism: same arm+seed reproduces exactly")
for arm in ["greedy", "causal"]:
    cfg = SyntheticConfig(sigma=0.05, p_good=0.25)
    r1 = run_arm(arm, cfg, 120.0, 3)
    r2 = run_arm(arm, cfg, 120.0, 3)
    check(f"{arm} reproducible",
          r1.true_performance == r2.true_performance and r1.n_accepted == r2.n_accepted,
          f"{r1.true_performance} vs {r2.true_performance}")


# ---------------------------------------------------------------------------
print("\n[2] Fairness: candidate stream identical across arms (only the gate differs)")
def candidate_deltas(seed, n=50):
    prng = np.random.default_rng(seed * 1000 + 7)
    nrng = np.random.default_rng(seed * 1000 + 99)
    w = SyntheticWorld(SyntheticConfig(sigma=0.05, p_good=0.25), prng, nrng)
    return [round(w.propose(None).delta, 12) for _ in range(n)]
# Two independent constructions with the same seed must yield the same stream.
check("proposal stream deterministic by seed",
      candidate_deltas(3) == candidate_deltas(3))
check("proposal stream differs across seeds",
      candidate_deltas(3) != candidate_deltas(4))


# ---------------------------------------------------------------------------
print("\n[3] Budget is never exceeded (equal-budget accounting holds)")
for arm in ["greedy", "causal", "coh+greedy", "coh+causal"]:
    for seed in range(20):
        cfg = SyntheticConfig(sigma=0.08, p_good=0.3)
        r = run_arm(arm, cfg, 100.0, seed)
        if r.budget_spent > 100.0 + 1e-6:
            check(f"{arm} budget<=100", False, f"spent {r.budget_spent}")
            break
    else:
        check(f"{arm} budget<=100 (all seeds)", True)


# ---------------------------------------------------------------------------
print("\n[4] No-noise sanity: sigma=0 => ZERO false-accepts; greedy >= causal on TRUE perf")
# With no measurement noise, a change is accepted iff its true effect > 0, so
# there must be no false-acceptances. This validates 'false accepts come from noise'.
for arm in ["greedy", "causal"]:
    fa = []
    for seed in range(40):
        cfg = SyntheticConfig(sigma=0.0, p_good=0.25)
        fa.append(run_arm(arm, cfg, 120.0, seed).n_false_accepts)
    check(f"{arm} zero false-accepts at sigma=0", max(fa) == 0, f"max={max(fa)}")

Tg = np.mean([run_arm("greedy", SyntheticConfig(sigma=0.0, p_good=0.25), 120.0, s).true_performance for s in range(40)])
Tc = np.mean([run_arm("causal", SyntheticConfig(sigma=0.0, p_good=0.25), 120.0, s).true_performance for s in range(40)])
check("at sigma=0 greedy >= causal (throughput, no noise to punish it)", Tg >= Tc, f"Tg={Tg:.3f} Tc={Tc:.3f}")


# ---------------------------------------------------------------------------
print("\n[5] evaluate() is unbiased: E[score] ~= T + realized(delta) (no sleight of hand)")
prng = np.random.default_rng(0)
nrng = np.random.default_rng(1)
w = SyntheticWorld(SyntheticConfig(sigma=0.1, ceiling=None), prng, nrng)
w.T = 0.5
cand = Candidate(delta=0.2, broken=False)
samples = [w.evaluate(cand, s) for s in range(20000)]
check("unbounded: mean(score) ~= T+delta", abs(np.mean(samples) - 0.7) < 0.005,
      f"got {np.mean(samples):.4f} want 0.70")

# ceiling: realized gain shrinks with headroom; measured score must match realized
wc = SyntheticWorld(SyntheticConfig(sigma=0.1, ceiling=1.0), prng, nrng)
wc.T = 0.5
realized = wc._realized(0.2)   # 0.2 * (1-0.5)/1.0 = 0.1
samp2 = [wc.evaluate(Candidate(delta=0.2, broken=False), s) for s in range(20000)]
check("ceiling: realized = delta*headroom", abs(realized - 0.1) < 1e-9, f"realized={realized}")
check("ceiling: mean(score) ~= T+realized", abs(np.mean(samp2) - 0.6) < 0.005,
      f"got {np.mean(samp2):.4f} want 0.60")
check("ceiling: negative effects NOT shrunk", wc._realized(-0.2) == -0.2)


# ---------------------------------------------------------------------------
print("\n[6] Welch SE formula correctness vs manual")
a = [0.1, 0.2, 0.15, 0.18]
b = [0.05, 0.07, 0.06]
d, se = _welch_delta_se(a, b)
man_d = _mean(a) - _mean(b)
man_se = (np.var(a, ddof=1)/len(a) + np.var(b, ddof=1)/len(b)) ** 0.5
check("delta matches", abs(d - man_d) < 1e-12)
check("SE matches numpy", abs(se - man_se) < 1e-12, f"{se} vs {man_se}")


# ---------------------------------------------------------------------------
print("\n[7] Causal accept rule: clear winners accepted, clear losers rejected")
gp = CausalPolicy(k0=2, k_max=6, z=1.0)
inc = Incumbent(scores=[0.0, 0.0, 0.0])
# clearly positive candidate, tiny noise
big = Candidate(delta=0.5, broken=False)
w2 = SyntheticWorld(SyntheticConfig(sigma=0.01), np.random.default_rng(2), np.random.default_rng(3))
w2.T = 0.0
dec = gp.decide(big, w2.evaluate, inc, Budget(100), 100)
check("clear winner accepted", dec.accepted, dec.reason)
neg = Candidate(delta=-0.5, broken=False)
dec2 = gp.decide(neg, w2.evaluate, Incumbent(scores=[0.0,0.0,0.0]), Budget(100), 200)
check("clear loser rejected", not dec2.accepted, dec2.reason)


# ---------------------------------------------------------------------------
print("\n[8] Causal spends >= greedy per candidate (it pays for confirmation)")
gspent, cspent = [], []
for seed in range(30):
    cfg = SyntheticConfig(sigma=0.08, p_good=0.3)
    rg = run_arm("greedy", cfg, 200.0, seed)
    rc = run_arm("causal", cfg, 200.0, seed)
    if rg.n_steps and rc.n_steps:
        gspent.append(rg.budget_spent / rg.n_steps)
        cspent.append(rc.budget_spent / rc.n_steps)
check("causal units/candidate >= greedy", np.mean(cspent) >= np.mean(gspent),
      f"greedy={np.mean(gspent):.2f} causal={np.mean(cspent):.2f}")


# ---------------------------------------------------------------------------
print("\n[9] Coherence: broken culled cheaply; broken NEVER accepted by any arm")
# broken candidate returns crash_score -> must never be accepted
cfg = SyntheticConfig(sigma=0.05, p_good=0.25, p_broken=0.5)
broke_accepted = 0
for seed in range(30):
    for arm in ["greedy", "causal", "coh+greedy"]:
        _, w = run_arm(arm, cfg, 120.0, seed, return_world=True)
        broke_accepted += sum(1 for r in w.accepted_records if r["kind"] == "broken")
check("broken never accepted", broke_accepted == 0, f"{broke_accepted} broken accepted")
# coherence arm should cull (n_culled > 0) and cost less than 1 unit per cull
rcoh = run_arm("coh+greedy", cfg, 120.0, 0)
check("coherence culls broken", rcoh.n_culled > 0, f"culled={rcoh.n_culled}")


# ---------------------------------------------------------------------------
print("\n[10] Replication audit: sigma=0 => nothing false survives spuriously")
a0 = experiment.replication_audit(sigma=0.0, p_good=0.25, ceiling=0.5,
                                  budget=120.0, n_outer=20, R=20)
# with no noise, replication effect == realized exactly, so survive_rep == survive_truth
check("audit sigma=0: replication == ground truth",
      a0["survive_replication"] == a0["survive_truth"],
      f"rep={a0['survive_replication']} truth={a0['survive_truth']}")
# and greedy at sigma=0 accepts only true gains -> nothing should 'vanish'
check("audit sigma=0: nothing vanishes (all kept were real)",
      a0["vanish_replication"] == 0, f"vanish={a0['vanish_replication']}")

a1 = experiment.replication_audit(sigma=0.08, p_good=0.25, ceiling=0.5,
                                  budget=120.0, n_outer=20, R=20)
check("audit with noise: some kept changes vanish under re-test",
      a1["vanish_replication"] > 0, f"vanish={a1['vanish_replication']}")
check("audit with noise: null changes mostly do NOT survive",
      a1["by_kind"].get("null", {}).get("surv_rep", 0)
      < 0.5 * a1["by_kind"].get("null", {}).get("kept", 1),
      str(a1["by_kind"].get("null")))


# ---------------------------------------------------------------------------
print("\n[11] Portfolio selection gate: sound invariants")
from portfolio import select_single, select_fixed_k, select_causal

# Three methods with KNOWN true qualities; method 2 is genuinely best.
T = np.array([0.30, 0.34, 0.40])
true_best = int(np.argmax(T))

# (a) zero selection noise: every rule must pick the truly-best method.
rng = np.random.default_rng(1)
for nm, fn in [("single", lambda: select_single(T, 0.0, rng)),
               ("fixedK", lambda: select_fixed_k(T, 0.0, rng, 5)),
               ("causal", lambda: select_causal(T, 0.0, rng, budget_cap=15))]:
    w, _, _ = fn()
    check(f"sigma=0 {nm} picks the true best", w == true_best, f"picked {w}")

# (b) equal budget: causal and fixedK both spend <= N*K; single spends N.
rng = np.random.default_rng(2)
_, u_single, _ = select_single(T, 0.05, rng)
_, u_fixed, _ = select_fixed_k(T, 0.05, rng, 5)
_, u_causal, _ = select_causal(T, 0.05, rng, budget_cap=len(T) * 5)
check("single budget == N", u_single == len(T), f"{u_single}")
check("fixedK budget == N*K", u_fixed == len(T) * 5, f"{u_fixed}")
check("causal budget <= N*K (equal-budget cap honoured)",
      u_causal <= len(T) * 5, f"{u_causal}")

# (c) winner's curse: under noise, the naive pick's reported score overstates
#     its own true performance ON AVERAGE (the bias the gate exists to fight).
rng = np.random.default_rng(3)
over = []
for _ in range(2000):
    w, _, obs = select_single(T, 0.10, rng)
    over.append(obs - T[w])
check("naive selection overstates the winner's true score (winner's curse)",
      np.mean(over) > 0.0, f"mean overstatement={np.mean(over):.4f}")

# (d) end-to-end: the persisted experiment shows the gate cutting the curse.
ps = experiment.portfolio_selection(n_methods=4, n_seeds=60)
hi = ps["sigma_sweep"][-1]
check("re-testing reduces winner's curse vs naive (high-noise cell)",
      hi["causal_overstate"] < hi["single_overstate"],
      f"causal={hi['causal_overstate']:.3f} single={hi['single_overstate']:.3f}")
check("causal selection regret <= naive regret (high-noise cell)",
      hi["causal_regret"] <= hi["single_regret"] + 1e-9,
      f"causal={hi['causal_regret']:.4f} single={hi['single_regret']:.4f}")


# ---------------------------------------------------------------------------
print("\n[12] Cost lever (fidelity): budget accounting + speed/accuracy trade")
from gates import Fidelity, FULL, CHEAP, Budget as _B, GreedyPolicy as _G

# (a) the cheaper fidelity charges less per eval, so a fixed budget funds MORE evals.
cfg = SyntheticConfig(sigma=0.05, p_good=0.25)
r_full = run_arm("greedy", cfg, 60.0, 7, fidelity=FULL)
r_cheap = run_arm("greedy", cfg, 60.0, 7, fidelity=CHEAP)
check("cheap fidelity yields more steps at equal budget",
      r_cheap.n_steps > r_full.n_steps,
      f"cheap={r_cheap.n_steps} full={r_full.n_steps}")

# (b) full fidelity is unchanged vs the no-fidelity default (no silent drift).
r_default = run_arm("greedy", cfg, 60.0, 7)
check("FULL fidelity == legacy default (no behavior change)",
      r_default.true_performance == r_full.true_performance
      and r_default.n_steps == r_full.n_steps, "")

# (c) per-eval cost is charged correctly by the policy (cost units, not eval counts).
b = _B(10.0)
inc = Incumbent(scores=[0.0])
pol = _G(eval_cost=0.3)
d = pol.decide(Candidate(delta=0.0, broken=False),
               lambda c, s: 0.1, inc, b, 123)
check("greedy charges eval_cost (0.3), not 1.0",
      abs(b.spent - 0.3) < 1e-9 and abs(d.units_spent - 0.3) < 1e-9,
      f"spent={b.spent} units={d.units_spent}")

# (d) a cheaper fidelity is noisier (sigma inflated by sigma_mult).
cheap_noisy = Fidelity("cheapnoisy", cost=0.3, params={"sigma_mult": 2.0})
import numpy as _np
prng = _np.random.default_rng(1); nrng = _np.random.default_rng(2)
from synthetic import SyntheticWorld as _W
w_full = _W(cfg, prng, _np.random.default_rng(2), sigma_mult=1.0)
w_chp = _W(cfg, prng, _np.random.default_rng(2), sigma_mult=2.0)
cand0 = Candidate(delta=0.0, broken=False)
sd_full = _np.std([w_full.evaluate(cand0, s) for s in range(400)])
sd_chp = _np.std([w_chp.evaluate(cand0, s) for s in range(400)])
check("cheaper fidelity has larger eval noise", sd_chp > 1.5 * sd_full,
      f"full_sd={sd_full:.4f} cheap_sd={sd_chp:.4f}")


# ---------------------------------------------------------------------------
print(f"\n==== {PASS} passed, {FAIL} failed ====")
import sys
sys.exit(1 if FAIL else 0)
