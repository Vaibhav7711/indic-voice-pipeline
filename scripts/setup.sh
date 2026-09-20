#!/usr/bin/env bash
set -euo pipefail

echo "Installing dependencies (PyTorch is provided by Colab)..."
pip install --upgrade pip

# datasets is pinned below 4.0 on purpose.
#
# google/fleurs is still a script-backed dataset on the Hub. The datasets
# library tightened this over three releases:
#   < 2.20   loading scripts run by default
#   2.20-3.x scripts require trust_remote_code=True
#   >= 4.0   scripts removed entirely; trust_remote_code itself raises
#
# An unpinned install pulls 4.x and every FLEURS load fails with a RuntimeError
# that does not obviously point at the cause. datasets 4.x also changed the
# audio column from a {"array", "sampling_rate"} dict to a decoder object.
# benchmarks/fleurs.py degrades gracefully, but this pin is the supported path.
# Dependencies are declared once, in pyproject.toml.
echo "Installing project as editable package..."
pip install -e ".[demo,dev]"

python - <<'PY'
import torch
print(f"torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
import datasets
print(f"datasets {datasets.__version__}")
if int(datasets.__version__.split(".")[0]) >= 4:
    print("WARNING: datasets >= 4 cannot load google/fleurs (loading scripts removed).")
PY

echo "Setup complete."
