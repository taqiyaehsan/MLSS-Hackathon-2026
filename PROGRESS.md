# Running Handoff / Session State — Skeptic-Gate Project

> Living doc updated as work proceeds. Pairs with `HANDOFF.md` (the master plan) and
> `MLSS - Hackathon.pdf` (official prompt = source of truth). If starting a new chat,
> read this first, then `HANDOFF.md`.

**Last updated:** 2026-06-22 (late session). **BIG PIVOT: MLRC-Bench DROPPED ENTIRELY; GCP not needed.** The project is now a **plug-and-play code-EDITING agent pipeline** on team-selected datasets, all local/CPU/stationary. The `hpo_task.py` config-tuning approach below is now SECONDARY. Read the section directly below first.

## 2026-06-24 (NEWEST-8) — SAGE-2026 hardened + ALIGNED TO SLIDES + MU real-data figure + Optiver figure — **READ FIRST**

Fleshed out the public **SAGE-2026** repo, aligned it to the slides, fixed the MU figure with real data, and built an Optiver evidence figure.

- **SAGE-2026 (`github.com/taqiyaehsan/SAGE-2026`) expanded + ALIGNED TO SLIDES.** Added: `docs/CUSTOM_AGENT.md` (plug in your own LLM/agent — the `propose()` one-method contract; `OPENAI_BASE_URL` for local/other endpoints; custom proposer class). `results/` = **PROOF** (agent-written `best_method.py` extracted from each `llm.json` `code` field for FM/MAGIC; `accepted_method.py` for colored_mnist; MLRC best accepted proposal + `progressive_results.csv` + summaries + `progression.png`; optiver `summary.json` + `phantom_wins.png`; a `results/README.md` index). **CITATIONS** (MLRC-Bench + FashionMNIST/MAGIC/MNIST/CIFAR-10/IRM/Optiver) in README/RESULTS/MACHINE_UNLEARNING. Fixes: RESULTS.md `skeptic_gate`→`sage` + removed make_poster/fig_spurious refs; MLRC subsystem hardcoded `REPO_ROOT/"skeptic_gate"`→`"sage"` (run_mlrc/replication_audit/baseline_noise/run_method); MACHINE_UNLEARNING.md genericized (te137/GCP/other-repo refs removed). **⚠️ ALIGNED TO SLIDES — switched Colored MNIST to the MATCH build** (`_colorize_match`, 2-channel/binary, the **failure node**: agent CNN val 0.88→test 0.13; baseline shape-reader val 0.61/test 0.58 robust but looks worst on val) **and REMOVED `example_regression`** → repo now ships exactly the slide tasks (fashionmnist, magic, colored_mnist[match], + MLRC subsystem + optiver results). `download=True` kept; smoke-tested. Commits: fc21b37→a90a431→da120b7→b89bd8d→3ed88a9→bc64f2d.
- **MU figure FIXED (real data, both repos + slides).** Teammate's `results/mlrc_unlearning/progressive_results.png` (mlrc_bench) is on a **proposal-index** x-axis → causal looks **truncated at step ~5**. WHY (not a bug): budget = **eval calls**; causal spends ~2 evals/proposal (the seed re-test), so for budget 8 greedy fits 8 proposals but causal fits ~3-4 — both spent **8 eval calls** (`budget_spent=8.0`). Honest fix = x-axis = **cumulative eval calls** (`budget_spent_after` in `results.jsonl`). Rebuilt `slides/figs/mlrc_result.png` from the **REAL** logs (extracted to `slides/mlrc_runs/*.jsonl`; 4 greedy + 5 causal seeds): **greedy 0.092 / skeptic 0.104 / baseline 0.054** (matches the deck headline). The DECK figure was already on the eval-calls axis = correct; the teammate's is the misleading one (flagged to Musfiq). See [[project-technical-decisions]].
- **Optiver evidence figure `optiver_phantom.png` (= `phantom_wins.png` in SAGE-2026).** Stacked bars from the ziwei Optiver hpo summary: **greedy ~2.1** wins accepted/run (>half **vanish on re-test**; 100% at high noise) vs **skeptic ~0.3** — the negative-control "phantom wins" story.
- **SPEAKER_NOTES.md / SAGE_demo.md:** Optiver added to the failures slide (Colored MNIST = a failure the skeptic CAN'T catch; Optiver = one it catches PERFECTLY = negative control); References section + a References slide added. See [[project-demo-deck]].

## 2026-06-23 (NEWEST-7) — DEMO DECK + SPEAKER NOTES + PUBLIC REPO + branch review — **READ FIRST**

The hackathon deliverable (the demo) is essentially DONE. This session: built the presentation figures/deck/speaker-notes, reviewed teammate branches, and organized a public repo.

- **PUBLIC repo `github.com/taqiyaehsan/SAGE-2026`** (user created it; I organized it for public download/use). Local clone `/Users/taqiya/Documents/SAGE-2026/`. Layout: `sage/` = core code-edit pipeline (gates/study/regime_sweep/run_method/local_task/task_data/base_method + `tasks/{fashionmnist,magic,colored_mnist,example_regression}`) **+ the MLRC/unlearning subsystem** (mlrc_adapter/run_mlrc/mlrc_background_knowledge/baseline_*/replication_audit_real); `docs/` (ADD_A_TASK, RESULTS, MACHINE_UNLEARNING [genericized — removed te137/GCP/other-repo refs], OBSERVATIONS); `assets/` (5 figs); README + requirements + .gitignore + sage/.env.example. **Scope = Core+MLRC, NO LICENSE** (user's choice → all-rights-reserved). Flipped `task_data.py download=False→True` for fresh-clone usability; smoke-tested end-to-end (`cd sage && python study.py example_regression 3 2`); scanned for secrets (clean). Pushed `main` (no Claude trailer). Full detail in [[reference-paths]].
- **Slides + speaker notes (working dir `slides/`):** `make_figs.py` → 13 figures in the user's **DARK Google-Slides template** palette (BG #0f0f16, PURPLE #a06bf0=skeptic, RED #f0593c=greedy, LAV accent; **NO titles on figs** — titles live in slide text boxes); `SAGE_demo.md` (Marp content source); **`SPEAKER_NOTES.md`** = elaborate per-slide notes that justify every choice and SELL the **AI-for-Science** angle (agent = scientific method automated; falsification; reproducibility crisis; MAGIC + machine-unlearning = real science). Figures+make_figs also pushed to MLSS-Hackathon-2026 `slides/` (commit 5194e44). **FINAL Google deck = 9 slides:** Title · Motivation (terminal "agent deleted my src/" incident) · [agent-design placeholder / video slot] · Tasks attempted (Dataset|Task|…, MU dataset=CIFAR-10) · Development progress (score_table) · MLRC benchmark result · What does the skeptic do (regime) · Failed experiments+lessons (Colored MNIST) · Takeaways+QR. Results/failed slides phrased as QUESTIONS.
- **⚠️ MU DECISION (user reversed earlier "cautionary-only"):** MLRC unlearning is PRESENTED AS A WIN — `mlrc_result.png` greedy-vs-skeptic trajectory (skeptic 0.105 > greedy 0.092 > baseline 0.054), reconstructed in palette from teammate **Musfiq's `mlrc_progress.png`** multi-seed GPU means. The 120×-non-stationarity caveat is now SPEAKER-ONLY (claim the trend, not a decimal; corroborated by stationary FM/MAGIC). See [[project-demo-deck]] + [[feedback-integrity-rigor]].
- **⚠️ Colored MNIST for the deck = the MATCH build (8ee7fd1), NOT the spec build:** user wanted it as a demonstrated FAILURE NODE (agent's accepted CNN val 0.88→test 0.13). `cmnist_fail.png` = val-vs-test scatter (inverse trend; baseline-reads-shape is the only robust point). Numbers from committed history at 8ee7fd1.
- **Branch review + merge:** `mlrc_bench` (teammate Musfiq's MLRC enrichments — cross-run history, focus-best, parallel isolation, + later the progress plot) **MERGED to MLSS-Hackathon-2026 `main`** (clean, additive). `ziwei` (teammate ziwei-jiang) = the **Optiver "Trading at the Close"** task on the `hpo_task.py` config-tuning path — clean leak-free data prep but **at chance (0.522, no signal)** → the deck's second "failed experiment"; NOT merged (kept on branch). Both branches based on old 8ee7fd1.

## 2026-06-22/23 (NEWEST-6) — THIRD TASK: **Colored MNIST** (spurious correlation) — **READ FIRST**

> **⚑ FINAL STATE = the SPEC version (commit a345518). The "blind-spot/match-encoding" version below (commit 8ee7fd1) was an intermediate approach the user then REPLACED via a precise dataset spec.** Read this box first; the detail below is the journey.
>
> **The user supplied an exact spec** (3-channel RGB, red=ch0/green=ch1; **10-class digit label**; color spuriously correlated with the GROUP A=0-4/B=5-9 at p=0.90 train/val, REVERSED to 0.10 in test; color never stored as a label; uses MNIST train→train/val and MNIST test→reversed; NO label noise). I verified empirically (free probe) that this spec **INVERTS the roles** vs the match-encoding build: the **weak linear baseline overfits to color and COLLAPSES** (val 0.827 → test **0.092**), while the **agent's CNNs learn shape and stay ROBUST** (val ~0.98 → test **0.94-0.97**). So the gate ACCEPTS 3 genuine improvements that ALL survive the audit AND generalize on test — the skeptic looks **good (not blind)** here; the unique "blind-spot" claim is dropped. Seed-noise sweep still holds (greedy FP 0.14-0.21 vs causal 0.01-0.07 on the near-tied CNN pool). **Bonus: fixed a real infra bug** — `study.py score_matrix` had NO per-eval timeout (long-standing TODO), so one pathological CNN hung a run ~3h; added a SIGALRM per-method cap (3× the per-eval limit) → slow/buggy methods (MixUp/extra-conv, + `torch` API hallucinations) are culled cleanly. **Pushed `main` a345518** (code task_data.py/study.py/tasks + make_poster_figs.py fig_spurious reworked to "baseline collapses / agent robust" + docs Section 5 rewritten as a spurious-correlation result + refreshed results). Loader defaults: n_train_total=6000, n_test=2000, p=0.90; time_limit 120. Run unchanged: `study.py colored_mnist llm 8 5` → `regime_sweep.py colored_mnist eval 8 200` → `make_poster_figs.py`. `_data_mnist/` gitignored → fresh clone needs MNIST train+test downloaded.

> **FOLLOW-ON (2026-06-23, commit e151c0e):** added the **"what the skeptic buys" figure** (`make_poster_figs.py` → `fig_skeptic_value.png`: greedy vs skeptic mean false-accept rate across all 3 tasks — FM 0.43→0.27 =1.6×, MAGIC 0.44→0.17 =2.5×, Colored MNIST 0.16→0.05 =3.4×; whiskers = range over the noise sweep) and a **README Results section** that separates the two claims: (1) the AGENT's measurable progress on held-out test (FM 0.749→0.888, MAGIC 0.786→0.868, Colored MNIST 0.09→0.97), and (2) the SKEPTIC's decision-integrity contribution (2-5× fewer false accepts; +1.1pt final model on MAGIC only; tied elsewhere). **NAMING DECISION: "skeptic gate", NOT "causal gate"** ("causal" overclaims — no causal-inference machinery, just replication under re-sampling); applied to user-facing labels + README + a naming note. **Code rename (CausalPolicy→SkepticPolicy) OFFERED → user chose to LEAVE the code as-is** (do NOT rename unless asked; README note bridges it). Honesty framing the user landed on: skeptic adds NO accuracy (except MAGIC +1.1pt) and SPENDS more compute (re-tests) — its payoff is integrity (fewer false "discoveries"), and the regime curve says when that compute is worth it.

**[EARLIER, SUPERSEDED — the match-encoding "blind spot" approach, commit 8ee7fd1]** Added a third reference task that, by design, shows where the causal skeptic **cannot** help — the honest complement to the FashionMNIST/MAGIC headline. **Pushed to GitHub `main` (commit 8ee7fd1, no Claude co-author).**

- **Why it matters (the new result):** the causal gate re-tests over SEEDS on the **validation** distribution, so it catches **seed-noise** false positives (NEWEST-2/headline). It **cannot** catch a **distribution-shift** false positive — a win that replicates perfectly across every seed yet relies on a spurious cue that flips at test time. Colored MNIST demonstrates exactly that third kind of "vanishing win." One-line: **the skeptic fixes measurement noise, not the wrong held-out distribution; the fix is a shifted validation set, not more seeds.**
- **Task design (`task_data.load_colored_mnist`):** 2-channel 28×28 images; binary label = digit≥5 with 25% label noise (caps a SHAPE predictor ~0.75). Spurious cue = **whether the two channels show the SAME digit** (a NON-LINEAR channel interaction, identical marginals), correlated with the label **0.90 in train+val, flipped to 0.10 in the held-out test**. Because the cue is non-linear, the linear baseline can only read shape (~0.61 val / 0.58 test — robust), leaving validation HEADROOM a CNN fills by exploiting the cue.
- **Two design pivots this session (the journey — don't repeat the dead ends):**
  1. First tried the NAIVE encoding (color = which of 2 channels holds the digit). It showed the test collapse but the **causal gate accepted NOTHING** — color was LINEARLY trivial, so the linear baseline already SATURATED the 0.90 color ceiling → no headroom → no win to tempt the gate. Math: any linearly-extractable cue saturates a linear baseline. FIX = make the cue a non-linear channel interaction (the "match" encoding), found empirically via a free probe (`/tmp/probe_cmnist.py`): only "match" gave a >0.25 baseline→CNN gap. User REJECTED a grayscale/shape-only baseline (wants it to stay COLORED).
  2. CNNs were **timing out** (87s > old 60s `time_limit` on 7200 train, single-thread) → all batchnorm proposals crashed/starved the pool. FIX = `time_limit` 60→120 (local_task.py) + `n_total` default 12_000→5_000 (task_data.py; train 7200→3000, CNN fit 87s→~5s). User flagged "they're all showing the same results" (the −1e6 crash sentinel) which is what surfaced this.
- **RESULT (`results/study_colored_mnist/` working dir; `results/skeptic_regime/colored_mnist/` clone):** gate (greedy=causal) ACCEPTS the first CNN val **0.607 → 0.876** (+0.27, seed-stable; the val-based replication audit says it **SURVIVES**) but test **0.581 → 0.130** (−0.45 collapse the seed audit is blind to). **Inverse val↔test trend** (more spurious reliance = higher val, lower test; idx7 .851/.329 → idx1 .876/.130). The ONLY generalizing model (shape-only baseline, test .58) looks WORST on val → **optimizing the validation metric actively SELECTS the trap**. Bonus: the agent hallucinated `torch.radians` (doesn't exist; =deg2rad) → 2 crashes caught cleanly (crash-handling story). NOTE greedy vs causal do NOT diverge at full fidelity (CNNs cluster near the cue ceiling, only the 1st accepted) — colored_mnist's contribution is the TEST-COLLAPSE axis, NOT the seed-FP axis.
- **Noise sweep (the seed-FP axis still behaves):** `regime_sweep.py colored_mnist eval 8 200` → causal still cuts greedy's false positives ~5× (greedy 0.28 vs causal 0.06 at σ≈0.049) — it just operates on a val signal that is itself misleading here.
- **Poster figure + doc:** dedicated **`fig_spurious.png`** added to `make_poster_figs.py` (val-vs-test collapse scatter + seed-noise FP panel; colored_mnist deliberately KEPT OUT of the 2-task progress/Pareto figures since its "progress" on test is negative by design). **Section 5 "The skeptic's blind spot"** added to `docs/SKEPTIC_REGIME_RESULTS.md` (+ 3-task intro + reproduce). Existing FM/MAGIC figs reproduced byte-identical.
- **⚠️ Caveat:** `_data_mnist/` is gitignored (like `_data_fmnist/`) → a FRESH clone must download MNIST once (loader uses `download=False`, same limitation as FashionMNIST).
- **Run:** `python study.py colored_mnist llm 8 5` (LLM, ~16min CPU) → `python regime_sweep.py colored_mnist eval 8 200` (no API) → `python make_poster_figs.py`. Mock-verify free: `python study.py colored_mnist 4 3`.
- **Poster placement:** colored_mnist = the "blind spot" panel that bounds the claim (skeptic catches noise FPs, not distribution-shift FPs). Reinforces SAGE's integrity story (NEWEST-5): a sage knows the LIMITS of its own re-testing.

## 2026-06-22 (NEWEST-5) — PROJECT NAMED **SAGE** + README input/output sample — **READ FIRST**

- **NAME DECIDED (user): the project is "SAGE — Skeptical Autonomous aGent for Experimentation."** Chosen over "Popper" (user liked the Karl-Popper falsification tie but didn't want to name it after a person). Key insight that sold it: **a "sage" is the antithesis of a "novice," so the name IS the narrative arc** — the agent starts naive (greedy: believes every lucky result) and matures into a sage (causal: re-tests before believing). Tagline: **"A good scientist tries to disprove their own results. So does SAGE."** (keeps Popper's falsification idea, no person named).
  - **Poster narrative arc (the through-line, title→conclusion):** Problem = agents accept noisy wins like an over-eager NOVICE, a third-to-half don't replicate → Method = SAGE researches like a scientist (hypothesize → rough prototype → experiment → skeptically accept) → Result = **the novice (greedy) vs the sage (causal)** = the regime curve (sage fooled 2-3× less; on MAGIC also a better final model) → Close = the difference between an agent that *generates* results and one that *earns* them.
  - **Acronym alts if needed:** "Self-Auditing aGent for Experimentation" (leans on the replication audit) or "Skeptical Agent for Guided Experimentation" (softer, for *semi*-autonomous).
  - ⚠️ **NOT YET APPLIED to the repo** — the GitHub repo/dir is still `skeptic-gate` / `MLSS-Hackathon-2026` and the README title is unchanged (deliberately, to avoid breakage). OPEN TODO if wanted: thread "SAGE" through README title, figure titles in `make_poster_figs.py` (e.g. relabel regime panels "novice (greedy)" vs "sage (causal)"), and section headers.
- **README input/output sample ADDED (GitHub `main`, commit 49234c0):** a worked end-to-end `magic` example in the "Code-editing agent (main pipeline)" section so teammates can mirror the shape for THEIR tasks/datasets — the 3 inputs they author (`background.md` contract + mediocre `baseline_method.py` + the `LOADERS`/`TaskSpec` one-liners), the intermediate output (a real agent-written `MyMethod.py`), and the outputs (`run_method.py` JSON line, the `study.py` console with the real Pareto table, and a file-schema table with real sample rows of `methods_llm.csv`/`replay_llm.csv`/`regime_eval.json`). Long listings in `<details>`. Includes a "what good looks like for your task" note (greedy `n_vanished` ≥ causal; greedy `fp_rate` rises above causal under noise; if too low-noise, shrink the eval set).
- **NEXT (unchanged from NEWEST-4):** poster build (showcase 6/24) using SAGE framing + the 4 figures (lead with MAGIC regime curve); fold in the MU quantitative number when the teammate's V100 run lands (NEWEST-3); optionally apply the SAGE name across repo artifacts.

## 2026-06-22 (NEWEST-4) — PROMPT ALIGNMENT CONFIRMED + POSTER FIGURES + FLOPs UNIT FIX — **READ FIRST**

- **Prompt alignment checked (re-read the official `MLSS - Hackathon.pdf` directly):** the project fits the prompt well, and our ablation / error-analysis / failed-experiments / cost-compute depth is a STRENGTH (most teams are thin there). We can hit every recommended poster section. **Framing risks (not substance gaps) for the poster:** (1) LEAD with measurable progress, THEN the skeptic as the differentiator — don't let "explain progress" overshadow "agent made progress"; (2) NAME a final method per task (MAGIC → the cheap 2-layer MLP; FashionMNIST → the cheap CNN / aug-CNN) so "report frontier, no auto-pick" doesn't read as "never produced a final method" (prompt explicitly wants "produce a final method"); (3) lean on **Machine Unlearning** for named-benchmark credibility so FashionMNIST/MAGIC don't look "toy" (MAGIC = real astrophysics/AI-for-Science dataset); (4) CITE all tasks/datasets (MLRC-Bench, MAGIC Gamma Telescope, FashionMNIST) — the prompt requires it.
- **Poster figures generated:** `skeptic_gate/make_poster_figs.py` (auto-discovers per-task result files by PATTERN; greedy=red `#c1352e`, causal=blue `#2f6db5`) → `results/skeptic_regime/figs/`: `fig_progress.png` (baseline→best test: FM +0.138, MAGIC +0.082), `fig_pareto.png` (acc vs FLOPs, **marker size = stability** = the 3rd Pareto axis, frontier highlighted), `fig_regime_fp.png` (⭐ headline: false-positive rate vs noise, greedy vs causal), `fig_regime_acc.png` (final acc vs noise; MAGIC = skeptic keeps a better model). Run with the working-dir `.venv`. On GitHub `main` commit c4e2a82.
- **⚠️ INTEGRITY FIX (FLOPs units):** the Pareto FLOPs columns are **GFLOPs**; an earlier `SKEPTIC_REGIME_RESULTS.md` claim ("most accurate FM model costs ~28000× the cheap CNN") and my verbal "167,000×" were a **TFLOPs-vs-GFLOPs unit-mixing error**. CORRECT ratio = **~168×** (MixUp idx8 85,514 GFLOPs vs cheap CNN idx1 510 GFLOPs, for +0.012 acc); MAGIC deeper MLPs ~12–23× more FLOPs to do worse. Doc fixed + figure now COMPUTES the ratio (can't drift) in commit c4e2a82. **Do NOT reuse "28000×"/"167,000×" in any draft.**
- **NEXT:** poster build (showcase 6/24) — lead with MAGIC regime curve; use the 4 figures; fold in the MU quantitative number when the teammate's V100 run lands (see NEWEST-3); optionally a combined 2×2 panel + a score-table figure.

## 2026-06-22 (NEWEST-3) — MACHINE UNLEARNING ON GCP IS BACK ON (partial reversal of the "MLRC dropped" line above) — **READ FIRST**

**A teammate set up MLRC-Bench on the GCP V100, and the user decided (this session) to ADD a quantitative Machine-Unlearning number to the poster** — alongside (not replacing) the done FashionMNIST+MAGIC code-edit headline. MU = the named benchmark → credibility + the real-data "single evals lie" story. The code-edit pipeline (FashionMNIST/MAGIC) is STILL the project's core and the rigorous claim; MU corroborates.

- **Wrote `docs/GCP_MACHINE_UNLEARNING.md`** (in the GitHub clone, pushed `main`): full run guide — setup, data prep, cost-lever verify, stationarity check, all greedy/skeptic arms, baseline-noise + replication audit, a ⭐ "poster recipe", integrity rules, troubleshooting. Linked from README.
- **FIXED a real bug:** the `MU_NUM_MODELS` cost-lever edit (`evaluation.py:34`) was MISSING from `setup/mlrc-local.patch`, so on a fresh GCP clone `--fidelity cheap` was a SILENT no-op (still ran 10 models → equal-budget accounting wrong). Now folded into the patch; validated it applies cleanly to a pristine MLRC-Bench tree. (commit c706531; guide commit dea14a3.)
- **Integrity gates before any MU number is quotable (in the guide):** (1) CONFIRM stationarity on the V100 first — run `baseline_noise.py` in two separated time windows, means must agree within std (laptop MPS was non-stationary → invalid; V100 expected OK but must verify). (2) Use `--fidelity full` for the headline (sidesteps the unvalidated 0.3 cheap-cost weight). (3) The **replication audit** (`replication_audit_real.py`: how many of greedy's accepted wins vanish on re-test) is the STRONGEST MU result — it's within-arm, so immune to the live-arm divergence confound. (4) A live greedy-vs-causal head-to-head via `run_mlrc.py` diverges (different accepts→different proposals) → frame as corroborating, NOT the clean paired ablation (that's study.py's replay design, local tasks only).
- **NEXT (division of labor, decided this session):** the **TEAMMATE runs the V100 MU job** (guide is self-contained — stationarity check → greedy `--fidelity full` budget ~8 → baseline_noise → replication audit) and shares back `results.jsonl`/`summary.json` + the two `baseline_noise_{A,B}/summary.json`. **I own analysis + poster integration:** verify stationarity passed, sanity-check the audit (how many greedy wins vanish), and fold the MU number into the poster (6/24) as corroboration of the FashionMNIST+MAGIC headline. Do NOT quote any MU number if the stationarity windows disagree.

## 2026-06-22 (NEWEST-2) — BOTH TASKS DONE: causal>greedy on FashionMNIST + MAGIC; causal generation; all CSVs; pushed — **READ FIRST**

**Final headline experiment complete for BOTH reference tasks, on GitHub (commit ce5136e).**
- **Causal gate now DRIVES generation** (study.py `generate_pool` uses `CausalPolicy`; incumbent advances only on causally-verified gains) — greedy is now purely the replay baseline. Task-agnostic (both tasks). Per [[project-technical-decisions]].
- **CSV export added** to study.py (`methods_<tag>.csv`, `replay_<tag>.csv`) + regime_sweep.py (`regime_<mode>.csv`) — "save EVERYTHING."
- **Two bugs found+fixed this run:** (1) `score_matrix` had no runtime-crash guard → a coherence-passed but runtime-crashing method (`torch.zeros(generator=)`, `Distribution.sample(generator=)`) aborted the whole study; now try/except → CRASH_SCORE (the "automated debugging" story; 3/8 FM proposals crashed gracefully). (2) crashed methods (stub stability=0/flops=-1) wrongly sat on the Pareto frontier → now exclude CRASH_SCORE methods from `pareto()`. FM frontier recomputed from saved data (no re-run): [0,1,2,8].
- **RESULTS — causal beats greedy at EVERY noise level on BOTH tasks (run: study llm 8 5; regime eval 8 200):**
  - **MAGIC (the dramatic one):** progress val 0.798→0.871, test 0.786→0.868; Pareto 6/9 (small MLP wins acc AND cost; deeper MLPs 13-23× FLOPs for worse acc). Regime: greedy FP 0.30→0.54 vs causal 0.06→0.24 (~2.5-3×), AND causal keeps better final acc (0.868 vs greedy 0.857 at high noise) — wins on BOTH FP and accuracy (many near-ties ~0.86).
  - **FashionMNIST (corroborates, weaker):** progress val 0.737→0.896, test 0.749→0.888; Pareto {baseline, CNN 0.88@0.5GFLOPs, aug-CNN 0.894@3GFLOPs, MixUp-CNN 0.896@85TFLOPs}. Regime: greedy FP 0.38→0.49 vs causal 0.21→0.32 (~1.4-2×); final-acc cost tiny (top methods within 0.001) → win = DECISION INTEGRITY not raw accuracy.
- **KNOWN caveat (TODO):** `score_matrix` (and regime fit) have NO per-eval timeout (only generation does) → a slow method (FM idx8 MixUp) timed out in generation but scored in matrix at 226s/85TFLOPs per fit (made FM study take 53 min). Add a scoring-stage wall/compute cap.
- **GitHub commit ce5136e:** `results/skeptic_regime/{fashionmnist,magic}/` (llm.json + all CSVs + regime_curve_eval.png), `docs/SKEPTIC_REGIME_RESULTS.md` rewritten for both tasks, study.py (causal gen + CSV + crash guard + pareto fix), regime_sweep.py (CSV), README pipeline figure (causal=proposal, greedy=baseline). Earlier commits this session: 46b16ae, 4267aab, 4c62397, 7a06d1a, 67d225b.
- **NEXT:** poster (showcase 6/24) — lead with MAGIC regime curve; add scoring-stage timeout; optionally enrich pool / more near-ties.

## 2026-06-22 (NEWEST) — REGIME SWEEP RESULT: causal beats greedy under eval noise (FashionMNIST, FIRST run — SUPERSEDED by NEWEST-2 above)

**The headline skeptic result is DONE and on GitHub (commit 46b16ae).**

- **Real LLM code-edit run** (`study.py fashionmnist llm 5 8`): agent went linear→CNN; **val 0.741→0.914, TEST 0.749→0.899**; pool = baseline + 5 coherent CNNs (0 static culls; step-3 MixUp passed coherence but CRASHED at runtime → −1e6 → rejected). Pareto keeps **5/6** methods (acc↑/stab↓/FLOPs↓, ~4 orders of FLOPs; one method 0.905 at ~7× less compute than the best). At **full** eval, greedy==causal (0 false positives) — FashionMNIST is **low-noise**, so the skeptic correctly earns nothing there. Output: `results/study_fashionmnist/llm.json`.
- **NEW `regime_sweep.py`** (NO LLM calls — reuses the fixed pool): injects **EVALUATION noise = scoring on a random subset of the held-out val set** (unbiased; truth ranking preserved → vanished gains are PURELY noise, no regime-shift confound). Trains each method once, caches per-example correctness, resamples; 200 bootstrap trials/level. **Chose eval-subsample over train-subsample** (the run_method `frac` dial) precisely because train-subsample confounds variance with a small-data regime shift (idx0..5 truth could re-rank). `eval` mode = headline; `train` mode kept as secondary realistic variant. CLI: `python regime_sweep.py fashionmnist eval 8 200`.
- **RESULT (the win):** as eval noise rises, **greedy false-positive rate climbs to 0.15 (σ≈0.031, E=200 val imgs), causal stays ≤0.03** (~7× fewer), and **final accuracy is identical** (causal doesn't sacrifice real gains). FP peaks mid-noise then dips at extreme noise (trajectory degrades before the idx5-over-idx4 false accept can occur). The single false-positive opportunity = near-tie idx4(0.914, true best) vs idx5(0.901, worse, proposed after) → on a small noisy eval idx5 can outscore idx4; greedy adopts the worse method, causal re-tests and rejects. **Honest framing: the value is DECISION INTEGRITY/reproducibility (accuracy cost of the mistake is tiny here because the pool has only one near-tie); effect scales with noise/stakes/chain-length and a richer pool or MAGIC.** Figure: `results/study_fashionmnist/regime_curve_eval.png`; data `regime_eval.json`.
- **GitHub (commit 46b16ae, pushed to `main`):** `skeptic_gate/regime_sweep.py`, `docs/SKEPTIC_REGIME_RESULTS.md` (results+explanation), `results/skeptic_regime/` (json+figure+code-edit study), and **README full-pipeline figure UPDATED** (code-editing loop Part A + replay/Pareto/regime analysis Part B). Clone lives at `/Users/taqiya/Documents/skeptic-gate/`.
- **NEXT:** run the same sweep on **MAGIC** (likely noisier → bigger gap) and/or **enrich the pool** for a stronger effect; poster (showcase 6/24). User confirmed the explanation "lands."

## 2026-06-22 (LATEST) — CODE-EDITING PIPELINE PIVOT — **READ THIS FIRST**

### The shape of the project now
A team builds **autonomous code-editing research agents** on **team-selected datasets** (the official prompt explicitly allows this; MLRC-Bench is only "suggested"). Each task is a tiny **MLRC-style repo WE author**: a `background.md` (problem statement + approach space) + a **working but mediocre `baseline_method.py` (primary code)** + a data loader + the eval harness. **One LLM agent (gpt-4.1-mini) per task EDITS the baseline code** to improve a held-out metric. The **skeptic gate** (coherence + causal) wraps the loop; a **replication audit** + a **Pareto frontier** are the analysis outputs.

### Division of labour (IMPORTANT)
- **Us (this repo / me):** the **framework** + the **two reference tasks (FashionMNIST = vision, MAGIC = tabular)** + the **skeptic gate** + the **cross-task analysis** (replay ablation, replication audit, Pareto). This is the team's actual research contribution.
- **Teammates:** other datasets **A–Z** — they write `background.md` + `baseline_method.py` + a `task_data` loader, register a `TaskSpec`, and run the **identical pipeline** (plug-and-play). Confirmed teammate tasks: **financial regression**, **MNIST**, **text classification (20 Newsgroups)**. (So I do NOT build those.)

### The agentic workflow (exactly what runs — no confusion)
- **ONE code-editing LLM agent per task per run.** Loop: read `background.md` + current best `MyMethod.py` + history → **write a COMPLETE edited `MyMethod.py`** → **coherence gate** (parses? keeps fit/predict signature? → cull cheaply if not) → **harness trains+scores** (held-out val) → **accept gate** (greedy or causal) → repeat to budget.
- The "arms" are NOT extra agents — they are the SAME agent re-run under a different accept rule (**greedy** vs **causal**) to measure the skeptic. (NOT the old multi-agent "one agent per method" portfolio — that idea was SUPERSEDED.)
- **Measurable progress = held-out metric, baseline → agent's final.** PROVEN: **FashionMNIST 0.75→0.87** (agent wrote a CNN), **MAGIC 0.786→0.844** (agent wrote an MLP). digits 0.951→0.97 (older hpo_task config path).

### Two analysis outputs, both from the same agent runs (in `study.py`)
1. **Skeptic ablation — REPLAY design (the scientifically-legit comparison the user insisted on):** generate the agent's candidate stream ONCE; re-score every method over S seeds; then **replay greedy AND causal over the IDENTICAL candidates + IDENTICAL per-seed measurements** → pure policy isolation. Then the **replication audit**: of each arm's accepts, how many gains VANISH vs the full-seed truth. (Live closed-loop arms would diverge → confounded; replay is the clean paired ablation.)
2. **Pareto method-selection (folded INTO the code-edit pipeline, over the methods the AGENT WROTE — not a hardcoded menu):** 3 axes — **accuracy** (mean over seeds, ↑), **stability** (std over seeds, ↓), **cost = FLOPs** (`torch.utils.flop_counter.FlopCounterMode`, hardware-independent, ↓) with **wall-clock as context**. **Report the frontier; NO auto-pick** (user's call: not LLM-judge, not pure accuracy). Accuracy is multi-seed mean (lucky single evals regress out). Held-out TEST touched once per point (report only, never to build the frontier).

### Files (working dir `skeptic_gate/`) — what's CORE vs SECONDARY vs DEAD
**CORE — the code-editing pipeline (the project now):**
- `base_method.py` — the `fit(X,y,seed)` / `predict(X)` interface the agent implements (predict = class indices for classification, continuous for regression).
- `task_data.py` — harness-owned loaders + held-out test; `LOADERS = {fashionmnist, magic, example_regression(=diabetes)}`; regression-aware `_stratified_split`/`_pack`. ⚠️ This file got reverted once mid-session by a linter; regression edits were RE-APPLIED — current state HAS regression support (verify if unsure).
- `run_method.py` — trusted harness (subprocess, CPU, single-thread, timeout, seeded, train-resample noise dial). `--metric {accuracy,r2}`; `_score()` (both higher-better so gates unchanged); R² for regression.
- `local_task.py` — `TaskSpec(name, time_limit, regimes, metric)`; `static_check()` (coherence gate, AST-based, tolerant of type annotations); `OpenAIProposer` (LLM edits code); `LocalTaskWorld`; `run_arm()`; `TASKS = {fashionmnist, magic, example_regression}`; `CRASH_SCORE=-1e6`.
- `tasks/<name>/` — `background.md` + `baseline_method.py` for **fashionmnist** (linear→agent CNN), **magic** (logistic→agent MLP), **example_regression** (diabetes, linear, R² template for teammates).
- `study.py` — **THE replay study** (generate_pool → score_matrix[val/seeds + test + FLOPs] → replay greedy/causal → replication audit → 3-axis Pareto). VERIFIED FREE (mock) on classification + regression. CLI: `python study.py <task> [llm] [N_PROPOSALS] [N_SEEDS]`.
- `gates.py` — task-agnostic policies (Greedy, Causal, Coherence) + `run_loop` + `Fidelity` cost lever. THE shared core; reused unchanged by everything.

**SECONDARY — config-tuning pipeline (older; agent tunes a fixed-MLP config dict, no code editing):**
- `hpo_task.py` — dataset-pluggable (digits/fmnist/magic), programmatic OR LLM proposer, regime sweep + replication audit. Has the clean programmatic 5-seed FashionMNIST proof (greedy 2.0 false vs causal 0.6 @ high noise) and a 1-seed LLM run (shipped-gap visible, false-accepts underpowered at n=1). `plots_hpo.py` = figures.

**SUPERSEDED — to delete (replaced by `study.py`'s Pareto-over-agent-methods):** `methods.py`, `portfolio_real.py` (pre-written cnn/mlp/logreg registry + Pareto).

**LEGACY (synthetic + MLRC, MLRC now dropped from plan):** `synthetic.py`, `experiment.py`, `plots.py`, `tests.py`, `portfolio.py` (synthetic selection), `mlrc_adapter.py` (the reference pattern the code-edit pipeline mirrors), `run_mlrc.py`, `smoke_test.py`, etc.

### GitHub state (repo `taqiyaehsan/MLSS-Hackathon-2026`, push to `main`, NO Claude co-author trailer)
PUSHED this session: dataset-pluggable `hpo_task.py`, `plots_hpo.py`; **AND (now pushed, commit 36040f8) the whole code-edit pipeline** (`base_method.py`, `run_method.py`, `local_task.py`, `task_data.py` with regression, `tasks/{fashionmnist,magic,example_regression}/`, `study.py`) + README ("Code-editing agent (the main pipeline)" section + plug-and-play "add a task" guide + TOC). `methods.py`/`portfolio_real.py` were NEVER pushed (superseded — delete from working dir).

### Prompt-fit (re-read the PDF this session)
Goal = **measurable progress + clear explanation**; agent should **edit/create code**; engineering criteria (reproducible, bounded compute, robust error handling, auditable logs); poster needs score table, strongest result, failed experiments, cost/compute, reproducibility; **showcase 6/24**. The code-editing approach is the most prompt-aligned; the skeptic gate is the differentiator (ablation / lessons-learned). Foreground the measurable-progress numbers; the skeptic is the twist.

### ▶ RESUME HERE (exact next steps, in order)
1. **Run the real LLM study (the headline result):** `cd skeptic_gate && python study.py fashionmnist llm 8 8` (agent writes diverse methods incl. a CNN → real 3-axis Pareto + the legit greedy-vs-causal replay + audit). Then `python study.py magic llm 8 8`. (~15–25 API calls each; needs `OPENAI_API_KEY` in `skeptic_gate/.env`.)
2. **Delete superseded `methods.py` + `portfolio_real.py`** from the working dir (user agreed; replaced by `study.py`; never pushed).
3. ✅ DONE — code-edit pipeline + `study.py` + regression + plug-and-play README guide pushed to GitHub `main` (commit 36040f8).
4. **Poster build (showcase 6/24):** score table across tasks (baseline→agent), the replay ablation (greedy vs causal false-accepts), the Pareto frontier figure, cost/compute summary, the coherence-gate "automated debugging" story (agent's runtime-crash edits get culled), reproducibility notes.
5. Verify with the user before each big API run; they control launches and pause at checkpoints.

---

## 2026-06-22 (EARLIER, NOW SECONDARY) — hpo_task config-tuning state (kept for facts)

**Big pivot (group decision): RAINFALL DROPPED.** The official prompt explicitly allows a
"team-selected alternative" dataset (MLRC-Bench is only "suggested"). New portfolio:
- **Machine Unlearning** — the named MLRC benchmark (credibility + the qualitative "single evals lie" story).
- **FashionMNIST** (vision) + **MAGIC Gamma Telescope** (astrophysics / AI-for-Science) — clean QUANTITATIVE testbeds.
- **Synthetic** — the controlled regime curve (already done).
**Everything runs on GCP now** (V100 16GB) → Machine Unlearning is CUDA-stationary → all 3 quantitative.

**Built this session (works, verified):**
- `skeptic_gate/hpo_task.py` — REAL train-and-score loop: small MLP on bundled sklearn `digits` (no
  download), reuses `gates.py` UNCHANGED. Real, noisy, STATIONARY CPU eval (random init+shuffle). Two
  proposers: programmatic mutator + real OpenAI gpt-4.1-mini (LLM agent moved held-out TEST 0.951→0.97).
  `python hpo_task.py` = 5-seed regime sweep + replication audit → `results/hpo_task/summary.json`;
  `python hpo_task.py llm` = live LLM-agent demo. FINDINGS: causal pins false-accepts ~0.2 across noise;
  greedy accrues several; shipped-model test acc OVERLAPS → the win is RELIABILITY, not accuracy.
  INTEGRITY FIX: noise sweep holds cost=1.0 (decoupled from the cost lever) so arms get EQUAL proposals.
- `skeptic_gate/plots_hpo.py` → `results/figs/fig_hpo_real.png` (3-panel: false-accepts / shipped-acc / audit).
- `skeptic_gate/smoke_test.py` — no-GPU/data/key check (imports + synthetic + real train/gate loop + CUDA report); 5/5 locally.

**GCP:** VM `hackerthon-agent` (project `hackerthon-500119`, zone `us-central1-c`), Tesla V100-SXM2-16GB,
Ubuntu22.04 / CUDA12.9. SHARED home; team user = **te137** (gcloud logs me in as `taqiya` → act as te137 via
`sudo -u te137`). **Shared MINICONDA env** (a teammate is setting it up — use it, NOT venv/pip-torch). A
teammate will (re)clone the GitHub repo onto the VM. gcloud installed on laptop, authed te137@scarletmail.rutgers.edu.

**GitHub `main` updated + pushed** (no PR, NO Claude co-author trailer): latest code + `setup/gcp_setup.sh`
+ README "Running on Google Cloud" + `smoke_test.py`. Repo = `github.com/taqiyaehsan/MLSS-Hackathon-2026`.

**NEXT:** generalize `hpo_task.py` into a dataset-pluggable engine; add FashionMNIST + MAGIC adapters (MLP
on flattened pixels — keep cheap/stationary; a CNN blows the audit budget); run 5-seed sweeps + a combined
cross-dataset figure; optionally MU quantitative on the V100. Do this ON THE VM once the teammate's conda
env + clone are green (`python smoke_test.py` should pass 5/5 with CUDA available=True).

---

## 2026-06-21 PIVOT — Rainfall (weather_forcast) is the NEW PRIMARY task  [SUPERSEDED 6/22 — Rainfall DROPPED]
- **Decision (user):** drop Machine Unlearning as primary; **Rainfall Prediction = new primary**, run real on **GCP/CUDA** (laptop only for a light plumbing smoke). Rainfall almost guarantees a Layer-1 progress number, and on CUDA it's **noisy-but-stationary** — the clean real testbed the MPS unlearning runs never gave.
- **Why pivot is sound:** unlearning's non-stationarity was a *machine* problem (MPS thermal drift); CUDA fixes it. Rainfall training is unseeded (real run-to-run wobble) → the causal gate has signal.
- **Rainfall eval contract (read from repo):** `python main.py -m my_method -p dev` → `train.py --mode train` trains a 3D U-Net (up to 10 epochs, early-stop) → writes `output/model/my_method/dev_metrics.json`, score = `test_mcsi` (mean CSI). Cheap 3-step smoke = `train.py --mode debug` (bypasses main.py; `-p debug` via main.py does NOT shortcut).
- **Setup gotchas:** task pinned to OLD stack (py3.8 / torch1.12+cu116 / pytorch-lightning 1.7.7) — incompatible with our py3.13 venv → **Rainfall needs its own env**; cu116 wheels install on GCP Linux, NOT macOS. `accelerator="gpu"` + `strategy="ddp"` hardcoded in train.py:136/139 → needs a local device patch to run on laptop. Data = multi-GB (gdown zips + >1hr SFTP). Deps missing locally: pytorch_lightning, h5py, paramiko, gdown.
- **Decided design (user):** proposer **edit surface** = "let's talk" (leaning hybrid: config yaml + bounded MyMethod edits); cut `max_epochs`→~3–4 + pre-register reduced budget so seed-repeats/audit are affordable (the Rainfall analog of "lower NUM_MODELS").
- **TODO when resuming Rainfall build:** new `rainfall_adapter.py` mirroring `mlrc_adapter.py` (new TASK_BRIEF + `train.py` invocation + CSI parse), reuse `gates.py` unchanged; inject `background.txt`+`research_problem.txt` into the proposer (groupmates' idea). NOTE Rainfall's `background.txt` is ONE integrated framework (WeatherFusionNet+PhyDNet+sat2rad+ConvLSTM+ensemble), NOT a method menu like unlearning — diversity for the portfolio must come from bounded sub-ideas/seeds, not named methods.

## 2026-06-21 NEW FEATURE — COST LEVER (task-agnostic fidelity knob) — built + tested
- `gates.Fidelity` (name, cost, params) + presets FULL/CHEAP. Policies (`GreedyPolicy`, `CausalPolicy`) now charge `eval_cost` from the active fidelity instead of a hardcoded 1.0 → budget is cost-aware; a cheaper fidelity stretches the same budget into more evals (a noisier each). Defaults = FULL, so all prior synthetic results are byte-identical (39/39 tests pass).
- Bindings: synthetic `sigma_mult` (cheaper = noisier, testable offline); unlearning → `MU_NUM_MODELS` env (patched `evaluation.py:34` to read it, default 10); rainfall → `max_epochs`/data-fraction (documented, wired when that adapter is built). `run_mlrc.py --fidelity full|cheap`.
- Verified end-to-end on the real runner (mock path): budget 6 → FULL 5 proposals / CHEAP 19 proposals (~3.3× = 1/0.3 cost ratio). Equal-budget accounting holds.
- ⚠️ the 0.3 cost weight is PLANNED; validate against measured GPU wall-clock before equal-budget claims.

## 2026-06-21 NEW FEATURE — multi-method PORTFOLIO + causal SELECTION gate (proven on synthetic)
Team idea: instantiate one agent-loop pipeline per bounded sub-idea, improve each, then pick the best. KEY: "pick best of N by single score" is a **max-of-N / winner's-curse** problem → it's the skeptic gate applied at SELECTION time, not a separate thesis. Built + proven on the synthetic (cheap), real-task version deferred to the Rainfall build.
- **New files/fns:** `skeptic_gate/portfolio.py` (reuses `run_arm` unchanged; selection rules `single`/`fixed_k`/`causal`-adaptive-racing), `experiment.portfolio_selection()` → `results/portfolio_selection.json`, `plots.fig_portfolio_selection` + `fig_portfolio_scaling`.
- **Results (N=5, 300 runs):** winner's curse — naive pick's reported score overstates TRUE perf by up to **0.175**, re-testing halves it. Regret — causal gate **~halves** it across noise. Max-of-N tax — P(pick truly-best) collapses naive 0.56→0.14 (N=2→12), gate holds 0.59→0.22. `causal` vs `fixed_k` at EQUAL budget (N×K): causal edges ahead at high noise/N; dominant win is "re-test at all."
- **Reproduce:** `cd skeptic_gate && python experiment.py && python plots.py` (portfolio part ~3s).

---
### (prior state — Machine Unlearning, now demoted to secondary/qualitative)
**BLOCKER FOUND: local MPS eval is NON-STATIONARY (see below) → no quantitative real-task gate comparison from this laptop.** Synthetic carries the quantitative headline. All unlearning eval processes STOPPED.

## STEP 6 — DONE (built + verified end-to-end on real MLRC task)
Files (in `skeptic_gate/`):
- `mlrc_adapter.py` — real MLRC world: `OpenAIProposer` (gpt-4.1-mini, JSON mode, task brief + history + current best code), `MLRCWorld.evaluate()` (write file → `python main.py -m my_method -p dev` subprocess → score from npz `total_score`, deletes stale npz first, stdout cross-check, crash→-10), self-contained snapshot keep/discard (disk always = best between steps; evaluate() RESTORES best after each run so the causal gate can re-run a candidate k times cleanly), `static_check()` (Gate-1 building block), `is_broken()` predicate.
- `run_mlrc.py` — CLI driver. REUSES `gates.py` unchanged (Budget/Incumbent/`policy.decide`); only the policy object differs across arms (`--arm greedy|causal|coh+greedy|coh+causal`). Rich per-step JSONL + meta.json + summary.json. `--mock-llm`/`--mock-eval` for free harness tests. **Auto-resets `MyMethod.py` from `baseline_MyMethod.py` at startup so every arm starts from the IDENTICAL incumbent** (a prior accepting run otherwise leaves its method on disk — this bit us once).
- `baseline_MyMethod.py` — canonical baseline (git-HEAD 1-epoch fine-tune + our MPS DEVICE patch). The reset source of truth.
- `baseline_noise.py` — characterize per-eval noise via N fresh independent evals.
- `replication_audit_real.py` — re-run greedy's ACCEPTED snapshots N× vs the baseline noise band (BUILT, not yet meaningfully runnable until the eval is stationary).
- **`greedy` arm == vanilla autoresearch** (single noisy eval, accept iff score>best, else revert) = the control the gates intercept. Proposer + budget + eval held FIXED across arms; only the policy changes.

### Verified end-to-end (real task, 2026-06-20)
- Harness smoke (mock LLM + mock eval): greedy + causal + coh+causal all run end-to-end; causal spends 2–6 seeds adaptively; coherence culls broken proposals.
- **Real greedy run (`--arm greedy --budget 8`, `results/real_greedy_b8/`):** gpt-4.1-mini proposed 7 REAL methods (grad ascent/descent, Fisher penalties, distillation); 6 over-forgot (utility collapsed → score ~0) and were rejected; 1 "accept" at step 4 (score 0.1625 vs baseline-at-the-time 0.1168). **Autoresearcher-with-LLM runs end-to-end. ✅**
- The agent is honestly WEAK on this hard task (matches MLRC paper: agents barely move unlearning). No clean Layer-1 progress claim.

### ⚠️ CRITICAL FINDING — local MPS eval is NON-STATIONARY (overturns the earlier "CV 0.3%")
Identical baseline code (byte-for-byte) produced wildly different scores depending on WHEN it ran:
- `baseline_noise_n8` (×8) + greedy baseline: **~0.117** (~180s/eval)
- coh+causal baseline (1): **0.054** (~180s)
- `baseline_recheck` evals 1–3: **~0.001** (fq~0.004, **~290s**/eval)
- **Tight WITHIN a time window, huge shifts ACROSS windows** (0.001 ↔ 0.117 ≈ 100×). Eval wall-time drifted 180s→290s → almost certainly **thermal throttling / MPS contention** after hours of back-to-back training changing how the 10 unlearning models train. This is **system-state drift, NOT i.i.d. noise.**
- **Why one eval = training, not inference:** each eval re-runs the unlearning (gradient training) NUM_MODELS=10× then trains MI-attack classifiers; the training is where MPS nondeterminism enters.
- **Consequences:** (1) the earlier "baseline 0.117, CV 0.3%" was one non-representative cluster — DISCARD it. (2) The 0.054-vs-0.117 "level shift" is explained: it's the same non-stationarity. (3) The greedy step-4 "+0.046 win" is almost certainly drift, NOT real progress — greedy fooled by noise (ironically our thesis, live). (4) **A sequential greedy-vs-skeptic SCORE comparison on this laptop is INVALID** — drift confounds it (and even confounds a single run: incumbent measured early vs candidates later). (5) NUM_MODELS lowering is MOOT — the problem is stationarity, not noise magnitude.

### What the real task IS good for (given the above)
- ✅ End-to-end demonstration that the LLM-driven autoresearcher runs on a real task.
- ✅ A vivid QUALITATIVE point for the talk: "identical code scored 0.001–0.117 depending only on when we ran it → single real evals are untrustworthy" (real-data motivation for the skeptic).
- ❌ NOT a source of quantitative gate numbers on local MPS.

### NEXT — decision (open)
- (A) Keep real task = end-to-end demo + the qualitative "single evals lie" slide; **SYNTHETIC carries the quantitative headline** (regime curve + replication audit, i.i.d. controlled noise). This is the original plan; lowest risk.
- (B) Move quantitative real-task arms to the **GCP CUDA box** (stationary env; authoritative numbers) — more setup, needs the credits/GPU from HANDOFF step 1–2.
- If pursuing real-task numbers anywhere: control drift (seed torch + `torch.use_deterministic_algorithms`, interleave/randomize arm eval order, or re-measure incumbent adjacent to each candidate).

---


> NEW CHAT? Read this whole file, then HANDOFF.md. Memory index also has pointers (project-skeptic-gate, project-technical-decisions, reference-paths).

## IN-LOOP LLM (decided + verified 2026-06-20)
- Provider = **OpenAI GPT**. Key stored by user in `skeptic_gate/.env` as `OPENAI_API_KEY` (gitignored, chmod 600). Load via python-dotenv; **never echo the key**.
- Verified working: `gpt-4.1-mini` and `gpt-4.1-nano` both respond. **Use `gpt-4.1-mini`** as the proposer (good at code, cheap); `gpt-4.1-nano` = ultra-cheap fallback.
- Must be held FIXED across ALL arms (vanilla + gated). Loop is provider-agnostic so swappable later (e.g. Gemini on GCP credits).
- SDK installed: `openai` 2.43.0, `python-dotenv`.

## STEP 6 — HOW TO BUILD (the immediate next task)
Build a thin programmatic loop = "vanilla autoresearch" behavior on Machine Unlearning:
1. `propose(state)` → call OpenAI (`gpt-4.1-mini`) with the unlearning task context (from `scripts/research_problem.txt`) + current `MyMethod.py` → returns a new `MyMethod.py` body. Keep `run(net, retain_loader, forget_loader, val_loader)` signature; obey task rules (approximate unlearning, no full retrain).
2. `evaluate()` → write the file, run `PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py -m my_method -p dev` from `env/`, parse "Final Score" (HIGHER better).
3. Accept rule = **GreedyPolicy from `skeptic_gate/gates.py`** (single eval, accept iff score > best). This IS vanilla autoresearch. NO gates yet.
4. Keep/discard via git on a branch in MLRC-Bench repo (commit kept, revert discarded); log every step (proposal, score, decision, time) to a results file; keep results.tsv untracked.
5. Cap by a fixed budget (e.g. N iterations or wall-clock). Each eval ~3 min.
6. Harvest the list of ACCEPTED MyMethod.py edits → feeds the real replication audit (HANDOFF step 14).
Then Step 6b: lower `NUM_MODELS` in evaluation.py to raise per-run noise into the gate-relevant regime, and add the gated arms (reuse same gates.py).

## ✅ STEP 4 CHECKPOINT RESULT (passed)

## ✅ STEP 4 CHECKPOINT RESULT (passed)
- Baseline eval on **MPS**: **wall time 2:56 (~177s, <3 min)**, **Final Score 0.0540** (baseline band 0.053–0.055 ✅ — pipeline reproduces MLRC baseline).
- Forgetting Quality 0.0536; RAU/RAR 1.0034; TAU/TAR 1.0045.
- Smoke test confirmed device=mps, model params on mps:0. One unlearning run ≈15.5s.
- **Decision: full plan ON, Machine Unlearning stays primary.** At ~3 min/eval locally, seed-repeats + replication audit are feasible without the GCP GPU.
- Caveat: single run — run-to-run wobble to be measured at Step 9 (base-vs-base).

---

## Where we are in the HANDOFF plan
Working through HANDOFF §4 setup steps in order, toward **CHECKPOINT Step 4** (time one baseline eval).

| Step | State |
|---|---|
| 1 Register team / claim GCP credits / spin GPU | **USER ACTION, not done.** Register before 6/22 (prompt deadline). |
| 2 GPU env | **USER ACTION.** This Mac is CPU-only. |
| 3 Clone MLRC-Bench + inspect Machine Unlearning | ✅ done |
| 3b Install deps into local venv | ✅ done |
| 3c Verify imports + Kaggle token | ✅ done (lazy patch, see below) |
| 3d Download CIFAR-10 + weights (`prepare.py`) | 🔄 **in progress** (re-downloading via curl after a stall) |
| 4 **CHECKPOINT** time `python main.py -m my_method -p dev` | ⏳ next |
| 5 Clone autoresearch, point in-loop model at cheap LLM | ⏳ todo |

---

## Environment (local)
- Project root: `/Users/taqiya/Documents/MLSS-agents-for-science`
- venv: `.venv` (Python 3.13). Activate: `source /Users/taqiya/Documents/MLSS-agents-for-science/.venv/bin/activate`
- Installed: torch 2.12.1, torchvision 0.27.1, numpy, pandas, scikit-learn, tqdm, nbformat, requests, kaggle 2.2.2, + editable `MLAgentBench` package (`pip install -e .` in `MLRC-Bench/`).
- Platform: macOS, **no NVIDIA GPU** but **Apple Silicon MPS GPU available**. Patched the 3 `DEVICE=` lines (evaluation.py, MyMethod.py, BaseMethod.py) to be MPS-aware: `cuda` → else `mps` → else `cpu`. Run eval with `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported ops fall back to CPU. This is for local dev speed; the authoritative benchmark number still comes from a CUDA box (Kaggle test phase / GCP).
- macOS, NOT a git repo.

## Task: Machine Unlearning (primary)
- Path: `MLRC-Bench/MLAgentBench/benchmarks_base/machine_unlearning/`
- Edit target: `env/methods/MyMethod.py` (class `MyMethod`, registered key **`my_method`** in `env/methods/__init__.py`).
- Eval entry (run from `env/`): `python main.py -m my_method -p dev`  ← NOTE flags are `-m/-p`, key is `my_method`.
  - **HANDOFF says `--method MyMethod --phase dev` — WRONG.** Actual CLI is `-m my_method -p dev`. (HANDOFF warned not to assume names; confirmed.)
- Score written to `env/dev_results/my_method_results.npz` (`total_score`), also via `save_evals` to `output/`.

### Eval mechanics (confirmed by reading `env/evaluation.py`)
- Dev phase uses **CIFAR-10**; one eval loops `NUM_MODELS = 10` unlearning runs (deepcopy original → `method.run()` → measure).
- Baseline `MyMethod.run()` = 1 epoch SGD fine-tune on retain set (catastrophic-forgetting baseline).
- **Stochasticity confirmed** (good for project): `outputs_R = np.random.normal(...)` is UNSEEDED, no torch seeding in loop → score genuinely wobbles run-to-run. This is the noise the causal gate detects.
- Recorded GPU baseline (task README): time ≈ **505–542 s/eval**, score ≈ **0.053–0.055**.
- Final score = `forget_quality * (RAU/RAR) * (TAU/TAR)`.

## Decisions / deviations from HANDOFF (all flagged to user)
1. **Local venv** (not conda, not a GPU box yet) — user asked to "run things locally for now."
2. **Kaggle uses NEW token format** `KGAT_...` → saved at `~/.kaggle/access_token` (chmod 600), NOT `kaggle.json` (Kaggle no longer issues kaggle.json).
3. **Lazy Kaggle patch** in `env/evaluation.py`: replaced the import-time `kaggle.json` existence check + `api.authenticate()` with a lazy `_get_api()` (called only in test phase via `update_metadata`). Scoring-neutral; dev phase never uses Kaggle. This is the only edit to upstream code so far.

## Current blocker / in-flight
- `prepare.py`'s torchvision CIFAR download **stalled** (~133 MB, dead socket, no timeout in torchvision). Killed it.
- Re-downloading all 4 artifacts via `curl` with resume + `--speed-limit/--speed-time` guards (bg task). Targets:
  - `env/data/cifar-10-python.tar.gz` (~170 MB, md5 `c58f30108f718f92721af3b95e74349a`) from toronto mirror (slow ~115 KB/s)
  - `env/weights_resnet18_cifar10.pth`, `env/retrain_weights_resnet18_cifar10.pth` (from `storage.googleapis.com/unlearning-challenge/`)
  - `env/forget_idx.npy`
- After downloads: torchvision will extract the tarball on first eval (or run `prepare.py` again — it skips files that already exist; it also `rm`s the tarball at the end).

## How to resume (fresh chat)
1. `source .venv/bin/activate`
2. Confirm data present: `ls -lh MLRC-Bench/MLAgentBench/benchmarks_base/machine_unlearning/env/*.pth env/*.npy` and `env/data/cifar-10-batches-py/`.
3. If CIFAR not extracted: re-run `prepare.py` from `.../machine_unlearning/scripts/` (skips already-downloaded files).
4. **Step 4:** `cd .../machine_unlearning/env && time python main.py -m my_method -p dev` → record wall-time + score. (CPU: expect 45 min–2 hr; sanity only.)
5. Decision rule (HANDOFF §4): few-min eval on GPU → full plan; ~30 min → cut seeds/budget, lean on synthetic.

## Step 5 finding — autoresearch architecture (IMPORTANT reconciliation vs HANDOFF)
Cloned `autoresearch` at project root. Reality differs from HANDOFF's assumption:
- **No programmatic agent loop, no swappable LLM call.** The "agent" is an external coding CLI (Claude Code/Codex) you run *in the repo*; `program.md` is its NL instruction set. Repo files: `prepare.py`, `train.py` (the edited file = a GPT training script, NOT an agent loop), `program.md`. No model hook.
- The research loop is `program.md` steps: edit train.py → commit → `uv run train.py` (fixed 5-min budget) → grep `val_bpb` (lower=better) → log results.tsv → **"if improved keep (advance branch) else git reset (discard)"** → loop forever. Those keep/discard steps = the GREEDY accept we intercept, but they're a PROMPT, not code.
- ⇒ "Point in-loop model at Gemini" has no code hook. In-loop model = whichever coding CLI we run; hold it fixed across arms.
- ⇒ **DECISION (recommended): autoresearch is a conceptual TEMPLATE, not a dependency.** We write a thin Python control loop: LLM proposes edit to `MyMethod.py` → call MLRC eval as-is (`python main.py -m my_method -p dev`) → apply accept rule (greedy vs gates) → log proposal/score/time/reason → repeat to wall-clock budget. Honors "don't rebuild MLRC harness/agent": eval called unchanged, LLM API for proposals, we build only the experiment control loop (HANDOFF steps 6 & 8 require it anyway).

## Next steps
- **Step 7 (build NEXT per HANDOFF "synthetic first"):** synthetic control — mocked eval with dials for noise σ and signal base-rate; develop gate logic (coherence, adaptive causal, budget accounting) here first. Instant/free/controllable. This is the headline insurance + regime curve.
- Step 6: thin loop runner + adapter for real MLRC task (after gates work on synthetic).
- OPEN: in-loop LLM choice + credentials (Gemini via Vertex? Claude? other). Needed for Step 6, NOT for Step 7.

## Noise check (DONE) — real task is LOW-noise at full-eval granularity
- 4 full baseline evals: 0.0540, 0.0541, 0.0544, 0.0546. mean≈0.0543, std≈0.0003, **CV≈0.5%**.
- Cause: dev eval averages NUM_MODELS=10 unlearning runs internally → damps wobble.
- Implication: for the causal gate to have signal on the REAL task, lower `NUM_MODELS` (less internal averaging → realistic per-run noise) or compare at finer granularity. Synthetic carries the headline regime curve (we control noise there).

## Step 7 — SYNTHETIC CONTROL (built, working) — `skeptic_gate/`
Files (task-agnostic gates so SAME code drives real task later):
- `gates.py` — Budget, Incumbent, GreedyPolicy, CausalPolicy (adaptive seeds k0=2..k_max=6, ~1 SE band), CoherenceWrapper (pre-eval cull), generic `run_loop` with per-step logging.
- `synthetic.py` — SyntheticWorld with dials: sigma (noise), p_good (signal rate), p_broken, good_scale=0.04, bad_mean=-0.03 (untested changes usually hurt), `ceiling` (None=unbounded; set=diminishing returns). `_realized()` applies diminishing returns consistently in BOTH evaluate() and on_accept() (no sleight of hand). run_arm() returns ground-truth metrics incl. false-accepts.
- `sanity.py` — sigma sweep, both worlds.

### Integrity decision (user approved option a): report BOTH unbounded + ceiling worlds.
### Result (p_good=0.25, budget=120, 60 outer seeds):
- Unbounded: causal first wins TRUE-perf at sigma>=0.24. Ceiling=0.5: at sigma>=0.12.
- BOTH worlds: greedy wins low/moderate noise (throughput); causal wins high noise (greedy erodes T via lucky harmful accepts — T<0 at sigma=0.24). Direction robust; crossover location modeling-dependent.
- Causal cuts false-accepts ~5-10x in EVERY regime/world (e.g. 3.5->0.5). Reliability story is regime-independent.
- Quotable: causal pays when per-run effect < ~1/3 of noise; always buys reliability.

### Step 7 COMPLETE — full synthetic pipeline built + figures generated
- `experiment.py` persists to `results/regime_grid.json` (90 cells) + `results/replication_audit.json`.
- `plots.py` regenerates 4 figs from JSON -> `results/figs/`: regime_heatmap, crossover, replication_audit, false_accepts.
- `README.md` documents reproduce steps + findings + integrity notes.
- **Replication audit headline: greedy kept 164 "wins"; 60% (98) DON'T survive re-testing.** Null changes almost all vanish; genuine gains survive far more.
- Regime heatmap shows clear boundary (black contour); ceiling world boundary shifts left (causal wins more), direction robust.
- Reproduce: `cd skeptic_gate && python experiment.py && python plots.py`

### NEXT (Step 6): real MLRC loop runner + adapter
- Thin Python loop: LLM proposes edit to MyMethod.py -> `python main.py -m my_method -p dev` -> gates (SAME gates.py) -> log -> repeat to wall-clock budget.
- BLOCKER/decision: in-loop LLM choice + credentials (user said "decide later").
- Also: lower NUM_MODELS on real eval to raise per-run noise into the gate-relevant regime (currently CV~0.5%).
- Real replication audit (HANDOFF step 14) once loop runs: re-run greedy's accepted MyMethod.py edits 15-20x.
