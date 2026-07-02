# SAGE — Speaker Notes (final deck, 9 slides)

> Detailed, justify-everything notes keyed to the final presentation. Each block is
> written to be spoken. Bold = the line to land. The through-thread is **AI for
> Science**: if we want AI to *do* science autonomously, it has to do science
> *properly* — earn results, not just generate them.

---

## Slide 1 — Title: SAGE (Skeptical Autonomous aGent for Experimentation)

The name is the thesis, so let me start there. A *sage* is the opposite of a *novice*. This talk is the story of an agent that matures from a credulous novice — one that believes every lucky result — into a skeptical sage that re-tests before it believes.

Here's the setup. We're entering an era where we point a large language model at a research problem and let it run: write code, train models, read the result, improve. That's the dream of **AI for Science** — autonomous discovery at machine speed. But there's a catch that decides whether this dream is useful or dangerous, and it's not about how clever the model is. It's about whether the agent can tell a *real* finding from a *lucky* one.

**Our one-line claim: a good scientist tries to disprove their own results — and so should an autonomous agent. SAGE is that agent.** Everything that follows is evidence that building skepticism *into* the agent is what makes autonomous science trustworthy.

(Note the team + venue: this is framed for an ML-for-science audience — the datasets and benchmarks we chose are real science / real ML-research problems, not toys.)

---

## Slide 2 — Motivation

Start with the relatable version of the problem. We give agents more autonomy every week, and we trust their confidence. Here the agent reports — with green checkmarks — "removed 47 unused files, all checks pass, done!" And then: *that was my entire src folder.* The agent wasn't malicious. It was **confidently wrong**, and we trusted the confidence.

Now translate that to science, where the failure is quieter but worse. In autonomous research the agent doesn't delete your files — it hands you a **result that isn't real**. It tries a new method, the score jumps from 0.74 to 0.92, it says "great, keeping it," and builds the next ten experiments on top. But that 0.92 was a noisy evaluation or a lucky seed. **In a single evaluation, a lucky win and a real win are literally indistinguishable.**

Why this matters for AI for Science specifically: machine learning already *has* a reproducibility crisis — a large fraction of reported gains don't replicate. An autonomous agent that accepts the first good number doesn't escape that crisis; it **industrializes** it, accumulating fake progress and polluting its own research trajectory. **The bottleneck for autonomous discovery isn't generating hypotheses — LLMs are already excellent at that. It's knowing which results to believe.** That is exactly the job of the scientific method, and it's the job we automate.

---

## Slide 3 — Agent design / how SAGE works  *(your methodology slide; or play the pipeline video here)*

This is where we show SAGE *is* the scientific method, automated. A scientist runs a loop: form a hypothesis, check the experiment is even valid, run it, and — the step that makes it science — **reproduce it before believing it.** SAGE runs the same loop.

- **Hypothesize.** One LLM agent reads the task and the current best code and writes a *complete edited method*. Crucially, it edits **real code, not just hyperparameters** — on FashionMNIST it rewrote a linear classifier into a CNN on its own, then over later rounds added residual blocks, GroupNorm, MixUp, a cosine-annealed schedule. That is genuine method discovery — the actual substance of ML research.
- **Is the experiment valid?** The *coherence gate* culls broken or hallucinated code before it ever costs a training run.
- **Run it.** A trusted, seeded harness trains and scores on a **held-out** set.
- **Reproduce before believing** — the *skeptic gate*. A naïve agent accepts the moment a score goes up once. SAGE re-runs the candidate over multiple seeds and accepts **only if the improvement clears the run-to-run noise band.** This is the falsification step: try to make the win disappear; keep it only if it survives.

Justify the experimental design, because rigor is the point: "greedy" vs "skeptic" is the **same agent with one switch flipped**, and we compare them by *replaying both decision rules over the identical stream of candidates and the identical measurements.* That removes the confound where two live runs would diverge into different proposals. **We applied the scientific method to evaluating our own method** — controlled, paired, confound-free.

The AI-for-Science takeaway for this slide: **autonomous agents today generate; they rarely falsify. Falsification is the engine of science. SAGE puts it back in the loop.**

---

## Slide 4 — Tasks attempted

This slide is about **generality**, and generality is what separates a trick from a method. We did not cherry-pick one friendly dataset. We ran across four domains and two agent modes:

- **FashionMNIST** — image classification (vision).
- **MAGIC Gamma Telescope** — signal-vs-background, a **real astrophysics dataset**: separating gamma-ray showers from hadronic background. This is a genuine AI-for-science problem, not a toy.
- **CIFAR-10 / Machine Unlearning** — a **named ML-research benchmark** (MLRC-Bench): removing the influence of specific data from a trained model, which matters for privacy and safety.
- **Colored MNIST** — a spurious-correlation stress test we built deliberately.
- **Optiver** — financial price-direction prediction (tabular).

Two **agent modes** matter: SAGE both *writes code* and *tunes hyperparameters* — the two ways research actually gets done.

And notice we list our **failures right here, up front** (Colored MNIST, Optiver). That's deliberate. Science includes negative results, and showing them is how we map the method's operating range — and frankly, it's how the audience knows to trust the wins. **A research agent has to work across scientific domains; here's the evidence it does — and the honest edges where it doesn't.**

---

## Slide 5 — Development progress

Lead with the substance: **the agent made real, measurable, held-out progress.** FashionMNIST 0.75 → 0.89, MAGIC 0.79 → 0.87 — and it got there by *editing code*, not by us hand-tuning anything.

Now the part I most want you to notice — the **val-versus-test framing is the honesty check, and it's pure scientific method.** Where the progress is real, validation and test move together (FashionMNIST, MAGIC). Where it's a trap, they split: Colored MNIST climbs on validation but the held-out test *collapses*. Optiver doesn't move at all. **A single table tells you, at a glance, where the agent did real science and where it fooled itself** — which is exactly the discrimination an autonomous research system has to be able to make.

For an AI-for-science audience the message is: **progress has to be measured on held-out data and stress-tested for generalization — not claimed.** SAGE produces genuine progress *and* gives you the instrument to tell genuine from artifact.

---

## Slide 6 — MLRC Benchmark Result

This is the named-benchmark payoff. Machine Unlearning on CIFAR-10 from MLRC-Bench — a recognized, hard ML-research task. The plot is best-score-so-far versus evaluation budget; both arms start from the same baseline (the dashed line).

The result: **the skeptic reaches a higher *and* more reliable score than greedy — 0.105 versus 0.092 — and gets there faster and with a tighter band.** Justify *why*, because it's the deep point: by refusing lucky-but-worse candidates, the incumbent the skeptic carries forward is genuinely stronger and more stable. **Skepticism doesn't just avoid embarrassing mistakes; it produces a better scientific result.** That's the whole pitch in one figure — on a real benchmark, doing science skeptically *wins*.

*(Speaker-only honesty note, in case of Q&A: this benchmark's evaluation has known across-window variance — the same baseline can score differently at different times on shared hardware. These curves are multi-seed means from our GPU runs, and they're corroborated by the clean, stationary FashionMNIST/MAGIC result on the next slide. For a camera-ready number we'd confirm with an interleaved baseline/best rerun. Don't over-claim a single decimal; claim the trend, which is robust.)*

---

## Slide 7 — What does the skeptic do?

Slide 6 showed *that* the skeptic wins; this shows *why*, quantified — and it's the cleanest, most rigorous experiment in the talk. The y-axis is the **false-discovery rate**: how often the agent accepts a "win" that isn't real. As we turn up evaluation noise, **greedy's false-discovery rate climbs toward one-in-two; the skeptic stays two-to-three times lower** — on both FashionMNIST and the astrophysics task.

Justify the noise dial, because this is where reviewers probe: we add noise by **scoring on a random subset of the held-out set.** That's *unbiased* — the true ranking of methods is preserved by construction — so any gain that vanishes under noise is **purely a measurement artifact, not a real change in the task.** We specifically *rejected* the easier dial of shrinking the training set, because that confounds noise with an actual change in the problem. **That choice is itself a piece of scientific integrity, and it's why this result is trustworthy.**

And the honest characterization: on clean, low-noise evaluations the two arms are identical — **skepticism correctly earns nothing when there's no noise to be fooled by.** We are *not* claiming a universal win. We're mapping *when* skepticism pays: in proportion to how noisy and how high-stakes the decisions are. **Characterizing a method's operating envelope, rather than asserting it always wins, is what makes this science rather than salesmanship.**

---

## Slide 8 — Failed experiments + lessons learnt

This is the slide I'm proudest of, because it's where we show we know our own limits — and we built the trap *deliberately* to find them. Colored MNIST: the agent raises validation accuracy beautifully but the held-out **test collapses** (those red points — high validation, low test). It latched onto a spurious cue that's predictive in training and **flips at test time.** The brutal part: the only robust model — the one that reads shape — looks *worst* on validation. So **optimizing the validation metric actively selects the trap.**

Here's the precise, generalizable lesson, and it's a deep one for all of autonomous discovery: **the skeptic re-tests over *seeds*, on the *same* distribution. So it catches noise — a win that doesn't reproduce. It cannot catch a win that reproduces perfectly every single seed yet relies on a feature that won't hold at deployment.** In one line: **re-testing buys you reproducibility, not validity.** Those are different things, and conflating them is how rigorous-looking pipelines still ship wrong conclusions. The fix isn't more seeds — it's a *shifted* validation set.

**Now the opposite failure — Optiver.** Colored MNIST is the case the skeptic *can't* catch; Optiver is the case it gets exactly right. The task is predicting the **direction** of a stock's closing-auction price move — essentially a coin flip: our held-out accuracy sits at **0.52, no signal to find.** But watch what the two agents do with that *(the stacked bars — `optiver_phantom.png`)*. The **credulous** agent didn't sit still — it "found" **one to three improvements every run**, and **more than half evaporated on re-test** (100 % of them at high noise). The **skeptic** accepted **~none** — and lost nothing, because there *was* nothing. That's the **negative control**: on a task with no signal, the right answer is to find none, and only the skeptic does. And the one reason we can say "no signal" *honestly* instead of accidentally faking one is that the pipeline is **leak-free** — a proper temporal split, no lookahead. On financial data especially, the careful pipeline is what reveals there was never anything to learn.

Second lesson, more practical: a static code check isn't enough. Free-form LLM code parses fine and then **crashes at runtime** — it invents library functions that don't exist. Robust error handling and real automated debugging are part of the agent, not an afterthought.

For AI for Science: **the most useful thing an autonomous scientist can know is the boundary of what its own method can certify.** We can state ours precisely. That honesty is not a weakness of the result; it *is* the result.

---

## Slide 9 — Takeaways

Three lines. **One — the skeptic wins:** real, code-edited progress, a better result on the MLRC benchmark, and two-to-three times fewer false discoveries under noise. **Two — the honest limit:** it catches measurement noise, not distribution shift; reproducibility is not the same as validity. **Three — next:** a shifted validation set to catch distribution-shift failures, and an auto-calibrated noise band per task.

Then land the bigger claim, because this is the sentence to leave them with: **if we want AI to do science autonomously, the hard part was never generating ideas — it's earning the right to believe them.** SAGE is a step toward *trustworthy* autonomous science — an agent that runs the scientific method on its own work: hypothesize, experiment, and try to disprove before it accepts. **An agent that doesn't just generate results — it earns them.** Novice to sage.

And the repo / QR is up — everything here is reproducible: the tasks, the gates, the replay analysis, the figures.

---

## References slide (cite the tasks/datasets — the prompt requires it)

- **MLRC-Bench** (Machine-Unlearning benchmark) — github.com/yunx-z/MLRC-Bench
- **FashionMNIST** — Xiao, Rasul & Vollgraf, arXiv:1708.07747 (2017)
- **MAGIC Gamma Telescope** — Bock et al., UCI ML Repository (2007)
- **CIFAR-10** — Krizhevsky, *Learning Multiple Layers of Features from Tiny Images* (2009)
- **MNIST** — LeCun, Cortes & Burges
- **Colored MNIST / IRM** — Arjovsky, Bottou, Gulrajani & Lopez-Paz, arXiv:1907.02893 (2019)
- **Optiver — Trading at the Close** — Kaggle (2023)
