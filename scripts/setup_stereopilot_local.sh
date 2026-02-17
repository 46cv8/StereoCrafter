#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEREOPILOT_REPO_DIR="${1:-$REPO_ROOT/weights/StereoPilot}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STEREOPILOT_REPO_URL="${STEREOPILOT_REPO_URL:-https://github.com/KlingTeam/StereoPilot.git}"
PREFETCH_MODELS="${PREFETCH_MODELS:-0}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$REPO_ROOT/weights/hf_cache}"
STEREOPILOT_MODEL_ID="${STEREOPILOT_MODEL_ID:-KlingTeam/StereoPilot}"
STEREOPILOT_MODEL_FILE="${STEREOPILOT_MODEL_FILE:-StereoPilot.safetensors}"
STEREOPILOT_BASE_MODEL_ID="${STEREOPILOT_BASE_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"

echo "[setup-stereopilot] StereoCrafter repo: $REPO_ROOT"
echo "[setup-stereopilot] StereoPilot repo dir: $STEREOPILOT_REPO_DIR"
echo "[setup-stereopilot] Python: $PYTHON_BIN"
echo "[setup-stereopilot] Prefetch models: $PREFETCH_MODELS"

if ! command -v git >/dev/null 2>&1; then
  echo "[setup-stereopilot] ERROR: git not found."
  exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[setup-stereopilot] ERROR: python binary not found: $PYTHON_BIN"
  exit 2
fi

mkdir -p "$(dirname "$STEREOPILOT_REPO_DIR")"

if [[ ! -d "$STEREOPILOT_REPO_DIR/.git" ]]; then
  echo "[setup-stereopilot] Cloning StereoPilot..."
  git clone "$STEREOPILOT_REPO_URL" "$STEREOPILOT_REPO_DIR"
fi

if [[ ! -f "$STEREOPILOT_REPO_DIR/sample.py" || ! -d "$STEREOPILOT_REPO_DIR/models" ]]; then
  echo "[setup-stereopilot] ERROR: StereoPilot repo appears incomplete: $STEREOPILOT_REPO_DIR"
  exit 2
fi

echo "[setup-stereopilot] Installing required python package(s)..."
"$PYTHON_BIN" -m pip install \
  "toml>=0.10.2" \
  "easydict>=1.13" \
  "ftfy>=6.3.1" \
  "safetensors>=0.5.3" \
  "torchvision>=0.24.1" \
  "decord>=0.6.0" \
  "imageio>=2.37.2" \
  "imageio-ffmpeg>=0.6.0" \
  "opencv-python>=4.11.0.86"

if [[ "$PREFETCH_MODELS" == "1" ]]; then
  echo "[setup-stereopilot] Prefetching Hugging Face model snapshots..."
  mkdir -p "$HF_CACHE_DIR"
  export STEREOPILOT_REPO_DIR HF_CACHE_DIR STEREOPILOT_MODEL_ID STEREOPILOT_MODEL_FILE STEREOPILOT_BASE_MODEL_ID
  "$PYTHON_BIN" - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

repo_dir = Path(os.environ["STEREOPILOT_REPO_DIR"]).expanduser().resolve()
cache_dir = Path(os.environ["HF_CACHE_DIR"]).expanduser().resolve()
model_id = os.environ["STEREOPILOT_MODEL_ID"]
model_file = os.environ["STEREOPILOT_MODEL_FILE"]
base_model_id = os.environ["STEREOPILOT_BASE_MODEL_ID"]

ckpt_dir = repo_dir / "ckpt"
ckpt_dir.mkdir(parents=True, exist_ok=True)

print(f"[setup-stereopilot] Downloading {model_id}/{model_file} -> {cache_dir}")
model_path = Path(hf_hub_download(repo_id=model_id, filename=model_file, cache_dir=str(cache_dir)))
target_model = ckpt_dir / "StereoPilot.safetensors"
if model_path.resolve() != target_model.resolve():
    shutil.copy2(model_path, target_model)
print(f"[setup-stereopilot] Checkpoint ready: {target_model}")

print(f"[setup-stereopilot] Downloading {base_model_id} snapshot -> {cache_dir}")
base_path = Path(snapshot_download(repo_id=base_model_id, cache_dir=str(cache_dir)))
target_base = ckpt_dir / "Wan2.1-T2V-1.3B"
target_base.mkdir(parents=True, exist_ok=True)
if base_path.resolve() != target_base.resolve():
    shutil.copytree(base_path, target_base, dirs_exist_ok=True)
print(f"[setup-stereopilot] Base model ready: {target_base}")
print("[setup-stereopilot] Model prefetch complete.")
PY
fi

echo "[setup-stereopilot] Done."
echo "[setup-stereopilot] Set GUI 'StereoPilot Repo Path' to: $STEREOPILOT_REPO_DIR"
echo "[setup-stereopilot] Optional cache path: $HF_CACHE_DIR"
