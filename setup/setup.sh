#!/usr/bin/env bash
#
# One-shot setup for the autoresearcher + MLRC-Bench Machine Unlearning task.
# Run from the repository root:   bash setup/setup.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MLRC_URL="https://github.com/yunx-z/MLRC-Bench.git"
MU="MLRC-Bench/MLAgentBench/benchmarks_base/machine_unlearning"
PY="${PYTHON:-python3}"

echo "==> 1/5  Clone MLRC-Bench"
if [ ! -d "MLRC-Bench" ]; then
  git clone "$MLRC_URL"
else
  echo "    MLRC-Bench already present, skipping clone."
fi

echo "==> 2/5  Create venv + install requirements"
if [ ! -d ".venv" ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e MLRC-Bench

echo "==> 3/5  Apply local MLRC patch (MPS device + lazy Kaggle auth)"
if git -C MLRC-Bench apply --reverse --check "$REPO_ROOT/setup/mlrc-local.patch" 2>/dev/null; then
  echo "    Patch already applied, skipping."
else
  ( cd MLRC-Bench && git apply "$REPO_ROOT/setup/mlrc-local.patch" )
  echo "    Patch applied."
fi

echo "==> 4/5  Download Machine Unlearning data + checkpoints"
echo "    (CIFAR-10 mirror can be slow; if it stalls, Ctrl-C and re-run setup — it resumes.)"
"$PY" "$MU/scripts/prepare.py" || {
  echo "    prepare.py did not finish cleanly. Re-run 'bash setup/setup.sh' to resume the download."
}

echo "==> 5/5  API key"
if [ ! -f "skeptic_gate/.env" ]; then
  cp skeptic_gate/.env.example skeptic_gate/.env
  echo "    Created skeptic_gate/.env — edit it and set OPENAI_API_KEY."
else
  echo "    skeptic_gate/.env already exists."
fi

echo ""
echo "Done. Verify the task runs:"
echo "    cd $MU/env && PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py -m my_method -p dev"
echo "Then run the autoresearcher:"
echo "    cd skeptic_gate && ../.venv/bin/python run_mlrc.py --arm greedy --budget 8"
