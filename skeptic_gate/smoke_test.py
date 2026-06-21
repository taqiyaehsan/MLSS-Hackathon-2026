"""Smoke test: confirm the repo's code actually runs after env setup.

Run from the skeptic_gate/ directory (in the project's conda/venv):

    python smoke_test.py

Self-contained: NO GPU, NO data download, NO API key needed. It uses sklearn's
bundled `digits` dataset, reports whether CUDA is visible (without requiring it),
and exercises imports + the synthetic gate pipeline + a real train-and-gate loop.
Exits 0 if everything works, non-zero on the first failure category.
"""

import sys
import traceback


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def t_imports():
    # every core module imports cleanly (catches missing deps / syntax errors)
    import numpy, sklearn, matplotlib  # noqa: F401  third-party
    import gates, synthetic, hpo_task, portfolio, experiment  # noqa: F401  ours
    import plots, plots_hpo, mlrc_adapter  # noqa: F401  ours


def t_torch():
    import torch
    cuda = torch.cuda.is_available()
    extra = f"  GPU={torch.cuda.get_device_name(0)}" if cuda else "  (CPU only)"
    print(f"        torch {torch.__version__}  CUDA available={cuda}{extra}")


def t_synthetic():
    # the pure-python gate pipeline (no torch) runs an arm to completion
    from synthetic import run_arm, SyntheticConfig
    r = run_arm("causal", SyntheticConfig(sigma=0.05), budget_units=30, outer_seed=0)
    assert r.n_steps > 0, "synthetic arm produced no steps"


def t_real_train():
    # a real MLP actually trains on bundled digits and scores sensibly
    import hpo_task
    acc = hpo_task.train_eval(hpo_task.BASELINE_CONFIG, seed=0)
    assert 0.0 <= acc <= 1.0, f"acc out of range: {acc}"
    assert acc > 0.5, f"baseline acc implausibly low ({acc:.3f}) — training may be broken"
    print(f"        digits baseline val acc = {acc:.3f}")


def t_real_loop():
    # the real gate loop (greedy, programmatic proposer — no API key) runs
    import hpo_task
    from gates import FULL
    r = hpo_task.run_arm("greedy", budget_units=6, outer_seed=0, fidelity=FULL)
    assert r["eval_calls"] > 0, "no evaluations happened"
    print(f"        greedy loop: {r['n_steps']} steps, "
          f"{r['n_accepted']} accepts, {r['eval_calls']} evals")


def main():
    print("=" * 64)
    print("SKEPTIC-GATE SMOKE TEST")
    print("=" * 64)
    checks = [
        ("core imports (third-party + all repo modules)", t_imports),
        ("torch import + CUDA report", t_torch),
        ("synthetic gate pipeline runs", t_synthetic),
        ("real MLP train+eval on digits", t_real_train),
        ("real gate loop (greedy) on digits", t_real_loop),
    ]
    results = [check(name, fn) for name, fn in checks]
    ok = all(results)
    print()
    print(f"==== SMOKE {'PASSED' if ok else 'FAILED'} "
          f"({sum(results)}/{len(results)} checks) ====")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
