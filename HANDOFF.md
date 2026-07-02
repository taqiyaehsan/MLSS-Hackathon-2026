# Build Handoff — A Skeptic Gate for an Autonomous ML Research Agent

**Event:** MLSS 2026 (Columbia) — "AI for Science" Hackathon (build, evaluate, and present autonomous ML research agents).
**Audience for this doc:** Claude Code, building with me in VSCode. Assume no prior context.

> **🧪 2026-06-22/23 (NEWEST-6) — THIRD TASK: Colored MNIST = a SPURIOUS-CORRELATION result (read `PROGRESS.md` "NEWEST-6" FIRST; FINAL = spec version, pushed GitHub `main` commit a345518).** Standard 10-class MNIST rendered as 3-channel RGB (digit in the red or green channel); color is SPURIOUSLY correlated with the digit's GROUP (0-4 vs 5-9) at p=0.90 train/val, REVERSED to 0.10 at test (color never a label, only baked into pixels). **Result:** the linear **baseline overfits to color and COLLAPSES** on the reversed test (val 0.827 → test **0.092**), while the agent's **CNNs learn shape and stay ROBUST** (val ~0.98 → test **0.94-0.97**); the gate accepts 3 improvements that ALL survive the audit AND generalize → the skeptic looks **good, not blind** here. Seed-noise sweep still holds (greedy FP 0.14-0.21 vs causal 0.01-0.07). **Also fixed a real infra bug:** `study.py score_matrix` had no per-eval timeout (long-standing TODO) → one pathological CNN hung a run ~3h; added a SIGALRM per-method cap (3× the per-eval limit) so slow/buggy methods are culled cleanly. `fig_spurious.png` reworked to "baseline collapses / agent robust" + `docs/SKEPTIC_REGIME_RESULTS.md` Section 5 rewritten as a spurious-correlation result. **NOTE:** an EARLIER same-session build (commit 8ee7fd1) used a non-linear "match-encoding" to make the trap fall on the STRONG model ("the skeptic's blind spot"); the user then supplied the precise spec above, which INVERTS the roles (weak baseline trapped, strong CNN robust) — so the spec version (a345518) supersedes it and the "blind spot" framing is dropped. **FOLLOW-ON (commit e151c0e):** added `fig_skeptic_value.png` ("what the skeptic buys" = greedy vs skeptic false-accept rate, FM 1.6× / MAGIC 2.5× / Colored MNIST 3.4× fewer) + a README **Results** section separating the two claims (agent measurable progress vs skeptic decision-integrity). **NAMING DECISION: call it the "skeptic gate", NOT "causal gate"** ("causal" overclaims) — applied to USER-FACING material only; the **code rename was OFFERED and the user chose to LEAVE the code as-is** (`CausalPolicy`/`--arm causal`/JSON `"causal"` unchanged; do NOT rename unless asked). Honesty framing: skeptic adds NO accuracy (except MAGIC +1.1pt) and SPENDS more compute — its payoff is integrity (2-5× fewer false wins).
>
> **⚠️ 2026-06-22 (LATEST) UPDATE — read `PROGRESS.md` "NEWEST-2 BOTH TASKS DONE" section FIRST.** Two big changes supersede this master plan: **(1) MLRC-Bench is DROPPED ENTIRELY and GCP is not needed** — the project is now a **plug-and-play code-EDITING agent pipeline** on **team-selected datasets**, all local/CPU/stationary. Each task = a tiny MLRC-style repo WE author (`background.md` + working mediocre `baseline_method.py` + data loader + harness); **ONE LLM agent per task EDITS the baseline code** to improve a held-out metric. **(2) Division of labour:** we own the **framework + reference tasks FashionMNIST (vision) & MAGIC (tabular) + the skeptic + cross-task analysis**; teammates plug in other datasets A–Z (financial regression, MNIST, text). The CORE code is now `base_method.py` / `task_data.py` / `run_method.py` / `local_task.py` / `tasks/` / `study.py` (replay study + Pareto) + `regime_sweep.py` (the headline noise-dial experiment), all reusing `gates.py` unchanged. The OLD `hpo_task.py` config-tuning approach (and the multi-agent "one agent per method" portfolio) are now SECONDARY/superseded. **The thesis, gate specs, and integrity rules below all still hold** — only the task format (code-editing instead of config-tuning) and venue (local instead of GCP) changed.
>
> **✅ HEADLINE EXPERIMENT COMPLETE (2026-06-22, GitHub commit ce5136e).** The causal gate now DRIVES generation (greedy = pure comparison baseline); the regime sweep shows **causal beats greedy at every eval-noise level on BOTH reference tasks**. MAGIC: progress 0.798→0.871 val / 0.786→0.868 test; greedy false-positive rate 0.30→0.54 vs causal 0.06→0.24 (~2.5-3× fewer) AND causal keeps a better final model (0.868 vs 0.857). FashionMNIST: progress 0.737→0.896 val / 0.749→0.888 test; greedy FP 0.38→0.49 vs causal 0.21→0.32 (~1.4-2×, win = decision integrity, tiny accuracy cost). Runtime-crash handling validated (3/8 FM proposals crashed gracefully = the "automated debugging" story). Results+CSVs+figures at `results/skeptic_regime/{fashionmnist,magic}/`; writeup `docs/SKEPTIC_REGIME_RESULTS.md`. NEXT = poster (showcase 6/24), lead with MAGIC. Known TODO: add a scoring-stage per-eval timeout (FM idx8 MixUp scored at 226s/85TFLOPs). See `PROGRESS.md` + [[project-technical-decisions]] for full detail.
>
> **🏷️ 2026-06-22 (NEWEST-5) — PROJECT NAMED "SAGE" (read `PROGRESS.md` "NEWEST-5" FIRST).** The project is **SAGE — Skeptical Autonomous aGent for Experimentation**. The hook: a *sage* is the opposite of a *novice*, so the name = the arc (greedy = the naive novice who trusts every lucky result; the causal skeptic = the sage who re-tests). Tagline: **"A good scientist tries to disprove their own results. So does SAGE."** Poster through-line = novice (greedy) → sage (causal). NOT yet applied to repo files (still `skeptic-gate`/`MLSS-Hackathon-2026`; README title unchanged) — optional TODO to thread it through. Also: README now has a worked input/output sample for teammates (GitHub commit 49234c0).
>
> **➕ 2026-06-22 (NEWEST-4) — read `PROGRESS.md` "NEWEST-4" + "NEWEST-3" FIRST.** Three things since the headline: **(1) Prompt-alignment confirmed** (re-read the official PDF): the project fits well and the ablation/error-analysis depth is a strength — but FRAMING RISKS for the poster = lead with measurable progress THEN the skeptic, NAME a final method per task (don't let "report frontier, no auto-pick" read as "never finished"), lean on Machine Unlearning for named-benchmark credibility, and cite all datasets (MLRC-Bench, MAGIC, FashionMNIST). **(2) Machine Unlearning on GCP is BACK ON as an additive quantitative track** — guide `docs/GCP_MACHINE_UNLEARNING.md`, cost-lever patch gap fixed; teammate runs it on the V100 (stationarity check FIRST, `--fidelity full`, replication audit = the strongest MU number); I own analysis + poster integration. **(3) Poster figures generated** — `skeptic_gate/make_poster_figs.py` → `results/skeptic_regime/figs/` (progress / Pareto / regime-FP / regime-acc). ⚠️ **INTEGRITY FIX:** the Pareto FLOPs were a TFLOPs-vs-GFLOPs unit mix-up — the FashionMNIST "most accurate costs ~28000× the cheap CNN" was WRONG; correct ratio is **~168×** for +0.012 acc. Doc fixed (commit c4e2a82). Do NOT reuse "28000×"/"167,000×".

### READ THESE FIRST, before any code
1. **The official hackathon prompt is in this repo** (filename like `HACKATHON_PROMPT.md`). It is the source of truth — if anything here conflicts with it, the prompt wins; flag the conflict to me.
2. **autoresearch** (base agent): `github.com/karpathy/autoresearch`. Read the loop; don't assume file/function names.
3. **MLRC-Bench** (task source): `github.com/yunx-z/MLRC-Bench`, Machine Unlearning task.

**Work phase by phase. Steps 4 and 9 are CHECKPOINTs — don't proceed past them until they pass.**

---

## 1. One-line pitch

**"Autonomous research agents keep improvements that are really just luck — we give the agent a skeptic that re-tests its own wins, and chart when that skepticism pays off versus when it just wastes compute."**

The bigger framing (use it in the intro and conclusion): as we hand science to autonomous agents, *who checks they aren't fooling themselves?* MLRC-Bench's own headline finding is that an LLM judge's sense of "novelty" doesn't predict real effectiveness. Our stance: let **causal evidence, not a lucky single run or an LLM's opinion**, decide what an autonomous scientist keeps. We're building the agent a conscience.

---

## 2. The reframe — read this carefully, it changed the project

We are **not** trying to prove "our gate beats the greedy agent." That claim is fragile: on a noisy task in a few hours, real improvements may be rare, and a confirmation gate spends compute that could have gone to exploration — so it can *lose* in a sparse-signal regime. Also, the AlphaLab result we cite (greedy underperforms) is about an **exploration** failure (path dependence / local optima); our gate fixes a **false-acceptance** failure. Don't overclaim that citation.

Instead, the contribution is a **characterization plus an audit**:
- **Characterization:** *when* does causal acceptance help vs. hurt? Show the regime boundary — where signal is sparse, confirmation wastes compute; where candidates are denser/noisier, confirmation pays. "Here's the signal-to-noise threshold above which an agent should stop trusting single runs" is more honest, more general, and bulletproof in Q&A because it doesn't depend on winning.
- **Replication audit (the headline evidence):** take the changes the *greedy* agent actually accepted and re-run each many times — report how many "improvements" vanish. Real data, devastating, honest.

This reframe converts the most likely outcome (a true-but-quiet result) into the actual finding.

---

## 3. The pieces (plain mental model)

- **MLRC-Bench = the exam.** Problem + starter code + baseline + scoring + a normalized metric (baseline = 0, top human = 100). We edit only `methods/MyMethod.py` and run `python main.py --method MyMethod --phase dev` to get a score.
- **The agent = the student.** autoresearch's loop: propose an edit to `MyMethod.py` → run eval → read score → keep/discard.
- **Our gates = a smarter "do I believe this helped?" rule.**
- **The synthetic control = a toy exam we fully control** (dials for noise and for how often a change is genuinely good), so we can draw the regime curve cleanly no matter what the messy real task does.

### Task selection — what we're optimizing for
Our project needs an unusual task profile (most teams only care about the first point; we need all three):
1. **Cheap per eval** — seed-repeats and the replication audit multiply evals, so a single eval must be minutes, not tens of minutes.
2. **Stochastic** — the score must wobble run-to-run, or the causal gate has nothing to detect. (A deterministic task makes the whole project pointless.)
3. **Some real signal** — agents can actually find above-baseline improvements, or the Layer-1 "measurable progress" number is hard to get.

These pull against each other, so there is **no single dominant task** — there's a trade. The numbers below are from the MLRC-Bench paper (Table 2 runtime/GPU caps; Table 3 best agent's relative-improvement-to-human).

| Task | Cap / GPU | Stochastic? | Agent progress (best in paper) | Verdict for us |
|---|---|---|---|---|
| **Machine Unlearning** | 0.5h / 16GB | **High** (scored vs many retrained checkpoints) | weak/variable (~+6, but Claude hit −94.7) | **Primary.** Cheap + very noisy + gameable → ideal for the audit & regime story. Risk: Layer-1 progress not guaranteed. |
| **Rainfall Prediction** | 0.5h test / 16–48GB; **long baseline training** (paper gave it 10h/100 steps) | Medium-high (training) | **Strong** (~+43 to +48) | **Progress backup.** Almost guarantees a positive Layer-1 number + generalization, but slow evals → fewer seeds. |
| **Backdoor Trigger Recovery** | 0.5h / 48GB | Medium (LLM sampling) | strong but variable (~+13 to +40) | Viable alt: decent signal, heavier 48GB memory. |
| **Cross-Domain Meta-Learning** | 3.5h / 16GB | High (episodic) | **negative** (hard for all agents) | Avoid: slow cap *and* hard to get progress — worst of both. |
| **LLM Merging** | 1h / 48GB | **None** (parameter averaging is deterministic) | ~+5 | **Do NOT use.** No wobble → the gate has nothing to detect. |
| **Next Product Recommendation** | 0.5h / 16GB | Low–medium | ~0 (agents barely moved it) | Avoid: weak signal. |
| **Temporal Action Localisation** | 0.5h / 16GB | Medium | ~0 (flat/negative) | Avoid: video/audio, heavy and fiddly. |

### The decision
The two finalists cover each other's weakness: **Machine Unlearning** maximizes *noise* (best for our headline: audit + regime curve) but risks *no Layer-1 progress*; **Rainfall** maximizes *progress* (satisfies the prompt) but risks *slow evals* (fewer seeds). Because our headline now needs noise + many cheap repeats, **Machine Unlearning is primary.** If compute allows, add **one** Rainfall run purely to bank a guaranteed positive Layer-1 number and show generalization — the only reason to add a second task.

**Decision rule (apply Friday after the Step-4 timing):**
- Machine Unlearning eval is a few minutes *and* a few iterations yield an above-baseline score → stay on it as primary.
- Machine Unlearning can't produce a positive Layer-1 number after a few iterations → keep it for the audit/noise story, but switch the *progress* demonstration to Rainfall (and lean harder on the synthetic for the regime curve, since real repeats will be scarce).
- Either eval is ~30 min → cut budget levels, fewer seeds, lean on the synthetic + replication audit.

The task is **not** sacred; the profile (fast + noisy + some signal) is what we're selecting for. Don't lock in before the timing numbers.

### Which to start development with
1. **Develop the gate logic against the SYNTHETIC control first** (step 7) — it's instant, free, and fully controllable, so you debug the coherence check, the adaptive causal gate, and the budget accounting without burning GPU or waiting on real evals. Get the gates *correct* here.
2. **Then validate on Machine Unlearning** as the first real task — cheapest real loop to iterate on.
3. **Rainfall only later**, and only as the progress/generalization run, because its slow evals will eat iteration time.

So: synthetic for building, Machine Unlearning for the primary real result, Rainfall as the optional progress backup.

### Two distinct LLM roles — don't conflate
1. **You (Claude Code)** = my dev assistant (adapter, gates, harness, plots). Strong model, my plan, separate from the experiment.
2. **The in-loop coding agent** = the model autoresearch calls to propose edits. Part of the experiment; held **identical and cheap** across all arms (ideally Gemini via Vertex on GCP credits).

---

## 4. UPDATED END-TO-END TO-DOs (with day markers)

> CHECKPOINTs at steps 4 and 9. The single biggest de-risk is building the **synthetic control (step 7) early** — it's your headline insurance if the real task has no signal.

**Setup (Friday, when registration opens)**
1. Register the team, join the Discussion channel, claim the GCP/Colab credits, and confirm what they cover (GPU only, or also Vertex/Gemini LLM calls?). Grab any sponsor API keys. Read the in-repo official prompt and reconcile any conflicts with this doc (prompt wins).
2. Spin up a GPU environment (a 16GB-class GPU is enough for Machine Unlearning).
3. Clone MLRC-Bench. Get the Machine Unlearning task's baseline running once: `python main.py --method <baseline> --phase dev`. Confirm it prints a score.
4. **CHECKPOINT — time one eval.** Run that baseline eval and clock the wall-time. A few minutes → full plan is on. ~30 min → cut budget levels and lean on the synthetic + replication audit. This one number decides how ambitious you can be.
5. Clone autoresearch, read its loop, and point its in-loop coding model at a cheap model (ideally Gemini via Vertex so it bills to credits).

**Wire the engine to the task**
6. Write the adapter so autoresearch's loop edits MLRC's `methods/MyMethod.py`, and its "run and score" step calls `python main.py --method MyMethod --phase dev` and parses the score. Confirm one full greedy iteration runs end-to-end: propose edit → eval → keep/discard.

**Build the synthetic control EARLY (your insurance — Saturday)**
7. Build a toy version of the loop with a *mocked* eval you control: two dials — metric noise (σ) and the base-rate of genuinely-good vs. null candidate changes. This lets you draw the regime curve (where causal acceptance beats greedy and where it doesn't) regardless of what the messy real task does. If the real-task run yields no signal, this is your headline figure. Saturday-afternoon scope.

**Harness + sanity (real task)**
8. Make the loop run to a fixed compute budget (wall-clock seconds on the fixed GPU), logging every proposal, decision, score, time, and reason.
9. **CHECKPOINT — base-vs-base.** Run the plain greedy agent twice at equal budget, different seeds. The two results should land close / overlap. If they don't, the harness is leaking compute somewhere — fix it now, not Wednesday.
10. Measure how often the greedy agent proposes broken or intent-mismatched code → decides whether the coherence gate is headline-worthy or just a cheap culler.
11. **Layer-1 progress check.** Confirm the base agent makes *some* genuine progress on the task (closes part of the baseline→human gap) within budget. The prompt requires "measurable improvement," so you need a positive Layer-1 number. If the agent finds nothing real, note it (it sharpens the rare-signal story) and rely more on the synthetic + replication audit for the headline.

**Build the gates** (develop and debug the gate logic against the SYNTHETIC from step 7 first — instant, free, controllable — then run the same code on the real task)
12. Gate 1 (coherence, before eval): check the edited method parses/imports, plus one cheap LLM "does this diff match its stated intent?" check. Broken → repair once or reject without running. Log every cull and the GPU saved.
13. Gate 2 (causal acceptance, after eval) — **adaptive confirmation, not a fixed K.** Run ~2 inner seeds; if the candidate is clearly better/worse than the current best's noise band, decide immediately; if it's borderline, spend more seeds; accept only if the gain clears the band. (Sequential "confirm cheap, escalate when uncertain" — maps to Foster's accept/challenge/reject interactive-proof framing, and is more compute-efficient than always running 3.) Use the effect estimate to steer the next proposal. Reference the autoresearch-speedrun fork's paired-seed funnel for mechanics.

**Headline evidence**
14. **Replication audit (the centerpiece).** Run the plain greedy agent once; take every change it *accepted*; re-run each one 15–20 times; report the fraction whose "improvement" vanishes under replication. "Greedy kept N improvements; under re-testing, M of them don't survive." Real data, honest, and the slide that makes the room go quiet.
15. Planted-null demo (keep as the clean illustration, secondary to #14): feed a change you know does nothing — greedy keeps it, your causal gate rejects it.

**Characterization experiment**
16. On the SYNTHETIC: arms = base (greedy), +causal (and +coherence, +both if cheap). Sweep noise (σ) × signal base-rate and draw the **regime curve** — where causal acceptance helps vs. hurts. This is your clean, controlled headline figure.
17. On the REAL MLRC task: run each arm over N≈5 outer seeds at a fixed budget (vary init / data-shuffle / agent-sampling across seeds). Record Layer-1 score (0–100), compute used, accepted/rejected/culled counts, and replication-survival. Lead with this as the required benchmark run; present the synthetic as the controlled supplement.
18. (If budget allows) repeat the MLRC run at 2–3 budget levels for a real-data gain-vs-compute Pareto; expect a crossover. If too slow, the synthetic regime curve carries the characterization.

**Report**
19. Regenerate all figures from logs with one script: the synthetic regime curve; the replication-audit bar ("N kept → M survive"); the per-arm score table with error bars (Layer 1); the compute summary; the planted-null result. No hand-edited numbers.
20. Frame the story: Layer 1 = "the agent makes measurable progress" (satisfies the prompt); headline = reliability + characterization ("when should an agent trust a single run?") + the replication finding. Pre-state the honest claim and don't move it.
21. Build the poster to the required sections; present Wednesday; awards at the Bloomberg event.

Rough calendar: **Fri** = 1–6. **Sat** = 7–11 (synthetic, harness, sanity, Layer-1 check). **Sun (hackathon day)** = 12–18 (gates, replication audit, experiments). **Mon/Tue** = 19–21, Tuesday as buffer. **Wed** present.

The three things that most often sink this: skipping step 4 (task too slow, discovered Sunday night), skipping step 9 (the comparison is secretly unfair), and **not building the synthetic control early (step 7)** — if the real task has no signal and you have no synthetic, you have no headline. Do all three early.

---

## 5. Gate specifications (detail for steps 12–13)

Base accept step (what we intercept):
```
proposal = agent.propose_edit(methods/MyMethod.py)
apply(proposal)
score = run_eval()                   # python main.py --method MyMethod --phase dev
if better(score, best_score):        # GREEDY: single-number comparison
    keep(); best_score = score
else:
    revert()
```

### Gate 1 — Coherence (pre-eval, cheap)
After the edit, before the eval: (1) static sanity — file parses/imports cleanly; (2) one cheap LLM consistency check — "does this diff plausibly implement its stated intent?" → keep / repair / reject. On failure, repair once or reject **without running the eval**. Log culls + GPU saved.

*Optional enhancement (counterexample-driven repair, from TraceFix):* instead of a single blind repair, on failure capture the **specific** failure as a structured counterexample — the actual error trace, the failing assertion, or the intent-mismatch diagnosis — and feed that exact counterexample back for a **bounded** repair loop (cap at 3–4 iterations, then give up and reject). This mirrors TraceFix's repair-until-verified loop. It is an upgrade to the repair step only; it does **not** change anything else, and it does **not** introduce TLA+/model-checking (that machinery does not apply to ML code). Skip if time-constrained.

### Gate 2 — Causal acceptance (post-eval, headline) — ADAPTIVE
```
run ~2 inner seeds for the candidate; have the best's cached seed scores
delta, sigma = paired_effect(candidate_scores, best_scores)
if clearly inside/outside the noise band:   decide now
else (borderline):                           run more seeds, re-test
accept only if gain clears noise band (pre-registered, ~1 SE rule)
on accept: cache the candidate's seed scores; use delta to steer next proposal
on reject: log an "honest discard"
```
- Sequential confirmation (confirm cheap, escalate when uncertain) — more compute-efficient than fixed K=3, which has almost no statistical power. Ties to Foster's interactive-proof accept/challenge/reject.

### Seeds — inner vs outer
- **Inner (adaptive, ~2+):** inside the loop, Gate 2's accept decision. Vary init, data shuffle, in-loop agent sampling.
- **Outer (N ≈ 5):** repeat the whole agent run per arm, for error bars.

### Equal-budget accounting (non-negotiable)
One unit = wall-clock seconds on the fixed GPU. Count ALL spend including gate overhead (Gate 1 LLM calls, Gate 2 reruns). Every arm gets the same total budget per outer seed. Stop at budget, not at a fixed proposal count.

---

## 6. What to report (lead with the headline, satisfy Layer 1)

- **Layer 1 (satisfy the prompt):** the agent makes measurable progress on Machine Unlearning — report on MLRC's 0–100 scale. Required by "measurable improvements." Necessary, not the headline.
- **Headline:** reliability + characterization. The replication audit ("N kept → M survive"), the synthetic regime curve (when causal acceptance helps vs. hurts), and the planted-null illustration. Frame gates as efficiency + integrity, not a leaderboard win.

**Pre-register** the acceptance threshold and the claim before looking at final numbers. **Fallback** if the real-task experiment is thin: the synthetic regime curve + the replication audit are a complete, defensible story on their own.

*Evaluation-design note (optional framing, from TraceFix):* this ablation mirrors the design of our prior TraceFix paper — a **paired ablation at fixed budget** measuring a **failure-rate reduction**, with the **largest separation under stress**. There, the failure was deadlock/livelock (cut 31.1%→14.1%) and the stress was fault injection; here, the failure is false-acceptance of noise and the stress is the high-noise regime / planted-null / gameable task. Use this as the template for the results section and, in the intro/conclusion, frame both gates as **runtime monitors that reject un-certified actions** (TraceFix's verification-first idea applied to an agent's self-deception instead of its coordination). This is framing only — do not add any TLA+/verification machinery.

---

## 7. Scope guardrails (enforce)
- One real task only (Machine Unlearning). A second task is future-work.
- The **synthetic stays a supplement**; lead with the MLRC run as the required benchmark, or you've drifted off the recommended benchmark.
- You **must** show a real Layer-1 improvement on the MLRC task so "measurable progress" is unambiguously met.
- No Lean compilation of ML code; coherence gate is lightweight consistency checking.
- No paper-writing / idea-generation pipeline. Don't rebuild MLRC-Bench's harness or the agent from scratch.
- Don't gold-plate gates; Gate 2 is the headline, Gate 1 is the warm-up.

---

## 8. Setup commands
```bash
# MLRC-Bench (task)
git clone https://github.com/yunx-z/MLRC-Bench.git
# follow README for Machine Unlearning env + data; then TIME one eval (Step 4):
#   python main.py --method <baseline> --phase dev

# autoresearch (engine)
git clone https://github.com/karpathy/autoresearch.git
# read the loop; set the in-loop coding model to a CHEAP model,
# ideally Gemini via Vertex AI so tokens bill to GCP credits.
```
Keep the in-loop model fixed and cheap across all arms. GPU via credits; Gate 2 reruns cost GPU, not tokens.

---

## 9. Confirm-on-Friday checklist
- Read the in-repo official prompt; reconcile conflicts (prompt wins).
- Credit scope: do credits cover (a) GPU and (b) Vertex/Gemini LLM calls?
- Sponsor API keys available? (ask in the Discussion channel)
- Step-4 number: real per-eval wall-clock for Machine Unlearning dev — decides seed/budget ambition.

---

## 10. Integrity rules (the project is *about* not fooling yourself)
- Eat the dog food: our own evaluation uses seed repeats and error bars; never report a single-seed number as a result.
- Pre-register the threshold and the claim; don't tune them to flatter the gates.
- Honest discards: log and report every rejected candidate, including ones greedy would have kept.
- No fabrication: report nulls and failures. A clean null + the replication audit is award-worthy.
- Hold the in-loop model and budget identical across arms; change only the gate.

---

## 11. References to cite / learn from
- **Karpathy autoresearch** — base agent.
- **autoresearch-speedrun fork** — paired-seed acceptance reference (study mechanics).
- **MLRC-Bench** — Zhang et al., 2025, arXiv:2504.09702 — task + 0/100 metric; LLM-judge novelty doesn't predict effectiveness; reports best-of-8-trials (an admission single runs are noisy — our gate is the principled in-loop version).
- **AlphaLab** — Hogan et al., 2026, arXiv:2604.08590 (Morgan Stanley) — greedy loop underperforms (motivates skepticism), adversarial Critic audits leakage (kin to coherence gate). NOTE: their failure is *exploration* (path dependence), ours is *false acceptance* — cite carefully, don't overclaim. Likely the Morgan Stanley lecture on the schedule.
- **Sakana AI Scientist** — the LLM-as-judge pipeline our objective approach contrasts with.
- **Dean Foster, MLSS 2026** (deanfoster.net/MLSS.pdf, 6/19) — CLOVER (consistency without compilation) → coherence gate; interactive-proof accept/challenge/reject → adaptive confirmation. Cite it.
- **Elias Bareinboim, MLSS 2026 causal foundations** — effect-estimation-under-noise framing behind the causal gate.
- **TraceFix** (Xia, Li, Ehsan, Ortiz — our own prior work, CAIS 2026; arXiv:2605.07935) — verification-first agent pipeline: counterexample-driven repair + a runtime monitor that rejects out-of-spec actions, evaluated by a fixed-budget paired ablation on a failure rate (DL/LL 31.1%→14.1%, largest under fault injection). We borrow (a) counterexample-driven bounded repair for Gate 1, (b) the "runtime monitor rejecting un-certified actions" framing for both gates, and (c) the paired-ablation eval template. We do **not** borrow the TLA+/coordination machinery — it doesn't apply to single-agent ML code.
- **MLAB** (MLAgentBench) — MLRC-Bench's shipped scaffold; fallback agent if the autoresearch↔MLRC adapter fights us (gate = robust top-snapshot selection there).

---

## 12. If a judge asks "what did you build?"
*"We give an autonomous ML-research agent a skeptic: a coherence check that culls broken code before it runs, and a causal acceptance gate that re-tests a candidate before believing its improvement. We audit the greedy agent and show how many of its 'wins' don't survive re-testing, and we chart the noise regime where this skepticism pays off versus where it just costs compute — so we know when an autonomous scientist should stop trusting a single run."*
