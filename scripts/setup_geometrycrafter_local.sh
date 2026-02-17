#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEOMETRY_REPO_DIR="${1:-$REPO_ROOT/weights/GeometryCrafter}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GEOMETRY_REPO_URL="${GEOMETRY_REPO_URL:-https://github.com/TencentARC/GeometryCrafter.git}"
PREFETCH_MODELS="${PREFETCH_MODELS:-0}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$REPO_ROOT/weights/hf_cache}"
GEOMETRY_MODEL_ID="${GEOMETRY_MODEL_ID:-TencentARC/GeometryCrafter}"
PRETRAIN_MODEL_ID="${PRETRAIN_MODEL_ID:-stabilityai/stable-video-diffusion-img2vid-xt}"

echo "[setup-geometry] StereoCrafter repo: $REPO_ROOT"
echo "[setup-geometry] Geometry repo dir: $GEOMETRY_REPO_DIR"
echo "[setup-geometry] Python: $PYTHON_BIN"
echo "[setup-geometry] Prefetch models: $PREFETCH_MODELS"

if ! command -v git >/dev/null 2>&1; then
  echo "[setup-geometry] ERROR: git not found."
  exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[setup-geometry] ERROR: python binary not found: $PYTHON_BIN"
  exit 2
fi

mkdir -p "$(dirname "$GEOMETRY_REPO_DIR")"

if [[ ! -d "$GEOMETRY_REPO_DIR/.git" ]]; then
  echo "[setup-geometry] Cloning GeometryCrafter..."
  git clone --recursive "$GEOMETRY_REPO_URL" "$GEOMETRY_REPO_DIR"
else
  echo "[setup-geometry] GeometryCrafter already present. Syncing submodules..."
  git -C "$GEOMETRY_REPO_DIR" submodule update --init --recursive
fi

echo "[setup-geometry] Installing required python package(s)..."
"$PYTHON_BIN" -m pip install \
  "kornia>=0.8.2" \
  "scipy>=1.10" \
  "trimesh>=4.0" \
  "pillow>=10.0" \
  "click>=8.0,<8.3.0"

if [[ "$PREFETCH_MODELS" == "1" ]]; then
  echo "[setup-geometry] Prefetching Hugging Face model snapshots..."
  mkdir -p "$HF_CACHE_DIR"
  export HF_CACHE_DIR GEOMETRY_MODEL_ID PRETRAIN_MODEL_ID
  "$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

cache_dir = os.path.expanduser(os.environ["HF_CACHE_DIR"])
geometry_model = os.environ["GEOMETRY_MODEL_ID"]
pretrain_model = os.environ["PRETRAIN_MODEL_ID"]

print(f"[setup-geometry] Downloading {geometry_model} -> {cache_dir}")
snapshot_download(repo_id=geometry_model, cache_dir=cache_dir)
print(f"[setup-geometry] Downloading {pretrain_model} -> {cache_dir}")
snapshot_download(repo_id=pretrain_model, cache_dir=cache_dir)
print("[setup-geometry] Model prefetch complete.")
PY
fi

echo "[setup-geometry] Done."
echo "[setup-geometry] Set GUI 'Geometry Repo Path' to: $GEOMETRY_REPO_DIR"
echo "[setup-geometry] Set GUI 'Geometry Cache Dir' to: $HF_CACHE_DIR"
