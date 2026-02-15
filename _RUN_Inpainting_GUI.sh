#!/usr/bin/env bash
set -euo pipefail

# ---- 1) Prefer CUDA 12.8 ----
export CUDA_HOME="/usr/local/cuda-12.8"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# ---- 2) Go to repo root ----
REPO_DIR="/mnt/home_peter_tuft/Movies/3D/StereoCrafter"
cd "$REPO_DIR"

# ---- 3) Ensure pyenv is available and select project env ----
if ! command -v pyenv >/dev/null 2>&1; then
  echo "ERROR: pyenv not found in PATH. Make sure your shell initializes pyenv."
  exit 1
fi

# Use the repo's local pyenv environment (creates .python-version effect)
pyenv local stereocrafter-312 >/dev/null

# Make sure shims are up-to-date and we use the right python
pyenv rehash
PYTHON="$(pyenv which python)"

echo "Using python: $PYTHON"
"$PYTHON" --version

# ---- 4) GPU selection (robust: use UUID) ----
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. NVIDIA driver not installed/working."
  exit 1
fi

GPU_COUNT="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"

if [[ "$GPU_COUNT" -gt 1 ]]; then
  echo "================================================="
  echo "Multiple GPUs detected:"
  # include UUID so we can select unambiguously
  nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader
  echo "================================================="
  read -r -p "Enter the ID of the GPU you want to use (e.g. 0 or 1): " GPU_ID

  # Resolve GPU_ID -> UUID
  GPU_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
    | awk -F',' -v id="$GPU_ID" '$1+0==id {gsub(/ /,"",$2); print $2; exit}')"

  if [[ -z "${GPU_UUID:-}" ]]; then
    echo "ERROR: Could not resolve UUID for GPU ID '$GPU_ID'."
    echo "Hint: valid IDs are shown in the table above."
    exit 1
  fi

  # IMPORTANT: set by UUID so CUDA/PyTorch can't mismatch indices
  export CUDA_VISIBLE_DEVICES="$GPU_UUID"
  echo "Script will now run using ONLY GPU $GPU_ID (UUID $GPU_UUID)."

elif [[ "$GPU_COUNT" -eq 1 ]]; then
  echo "Only 1 GPU detected. Running on default."
else
  echo "WARNING: No GPUs detected by nvidia-smi. Running anyway (CPU fallback may fail)."
fi

# ---- 4b) Sanity check what PyTorch sees (after CUDA_VISIBLE_DEVICES masking) ----
echo "================================================="
echo "PyTorch CUDA visibility check:"
"$PYTHON" - <<'PY'
import os
import torch
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch version        =", torch.__version__)
print("torch cuda version   =", torch.version.cuda)
print("cuda available       =", torch.cuda.is_available())
print("device_count         =", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  device {i}: {torch.cuda.get_device_name(i)}")
PY
echo "================================================="

# ---- 5) Run the script ----
set +e
"$PYTHON" inpainting_gui.py
EXIT_CODE=$?
set -e

if [[ "$EXIT_CODE" -eq 137 ]]; then
  echo "ERROR: inpainting_gui.py exited with code 137 (SIGKILL)."
  echo "Likely cause: host RAM exhaustion (OOM killer)."
  echo "Check your output folder for:"
  echo "  - inpainting_runtime.log"
  echo "  - inpaint_runtime_status_<video>.json"
fi

exit "$EXIT_CODE"
