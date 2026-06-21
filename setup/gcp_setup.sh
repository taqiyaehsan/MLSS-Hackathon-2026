#!/usr/bin/env bash
#
# Google Cloud GPU setup. Run ON the VM, from the repo root, after cloning:
#   bash setup/gcp_setup.sh
#
# Provisions a CUDA Python env, installs deps, sets up the MLRC-Bench Machine
# Unlearning task, and prints how to install Claude Code on the box. The small
# tasks (synthetic, FashionMNIST, MAGIC) run on CPU; Machine Unlearning uses the
# GPU and is STATIONARY on CUDA (unlike the laptop MPS eval).
#
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python3}"
CUDA_TAG="${CUDA_TAG:-cu121}"   # match the VM's CUDA toolkit; cu121 fits most recent drivers

echo "==> 0/6  System packages"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv python3-pip build-essential git
fi

echo "==> 1/6  Python venv"
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip -q

echo "==> 2/6  PyTorch (CUDA ${CUDA_TAG})"
pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "==> 3/6  Remaining Python deps (torch pins skipped; CUDA build kept)"
grep -viE '^(torch|torchvision)' requirements.txt > /tmp/reqs_notorch.txt
pip install -q -r /tmp/reqs_notorch.txt

echo "==> 4/6  Verify GPU"
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("   torch", torch.__version__, "| CUDA available:", ok,
      "|", torch.cuda.get_device_name(0) if ok else "NO GPU VISIBLE")
PY

echo "==> 5/6  MLRC-Bench Machine Unlearning (named benchmark)"
[ -d MLRC-Bench ] || git clone https://github.com/yunx-z/MLRC-Bench.git
pip install -q -e MLRC-Bench
# lazy-Kaggle + device patch (device auto-selects cuda on this box). Non-fatal:
if git -C MLRC-Bench apply --reverse --check setup/mlrc-local.patch 2>/dev/null; then
  echo "   patch already applied"
elif ( cd MLRC-Bench && git apply "$REPO_ROOT/setup/mlrc-local.patch" ) 2>/dev/null; then
  echo "   patch applied"
else
  echo "   NOTE: patch did not apply cleanly (upstream may have moved) — MU device"
  echo "         still defaults to cuda; only the lazy-Kaggle convenience is skipped."
fi
echo "   MU data (CIFAR-10 + weights) downloads on demand:"
echo "     python MLRC-Bench/MLAgentBench/benchmarks_base/machine_unlearning/scripts/prepare.py"

echo "==> 6/6  API key"
[ -f skeptic_gate/.env ] || cp skeptic_gate/.env.example skeptic_gate/.env
echo "   edit skeptic_gate/.env  ->  OPENAI_API_KEY=..."

cat <<'EOF'

------------------------------------------------------------------------------
Done. Sanity checks:
  source .venv/bin/activate
  cd skeptic_gate && python tests.py        # 39 invariant tests
  python hpo_task.py                         # real-task pipeline (CPU)

Install Claude Code on this VM (one time):
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y nodejs
  npm install -g @anthropic-ai/claude-code
  claude         # follow the login prompt (prints a URL; open it in your laptop
                 # browser, paste the code back) — or export ANTHROPIC_API_KEY=...
------------------------------------------------------------------------------
EOF
