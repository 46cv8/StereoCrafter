#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=""
VENV_NAME=".venv-cloud"
PYTHON_BIN="python3"
INSTALL_DEPS="1"
PREWARM_MODELS="0"
LOCAL_FILES_ONLY="0"
DISABLE_XFORMERS="0"
CPU_OFFLOAD="model"
FLASH_ATTN_INSTALL="${FLASH_ATTN_INSTALL:-1}"
FLASH_ATTN_PIP_SPEC="${FLASH_ATTN_PIP_SPEC:-flash-attn}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"; shift 2 ;;
    --venv-name)
      VENV_NAME="$2"; shift 2 ;;
    --python-bin)
      PYTHON_BIN="$2"; shift 2 ;;
    --skip-install)
      INSTALL_DEPS="0"; shift ;;
    --prewarm-models)
      PREWARM_MODELS="1"; shift ;;
    --local-files-only)
      LOCAL_FILES_ONLY="1"; shift ;;
    --disable-xformers)
      DISABLE_XFORMERS="1"; shift ;;
    --cpu-offload)
      CPU_OFFLOAD="$2"; shift 2 ;;
    --skip-flash-attn)
      FLASH_ATTN_INSTALL="0"; shift ;;
    --flash-attn-spec)
      FLASH_ATTN_PIP_SPEC="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"
      exit 2 ;;
  esac
done

if [[ -z "$REPO_DIR" ]]; then
  echo "--repo-dir is required"
  exit 2
fi

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repo dir not found: $REPO_DIR"
  exit 2
fi

cd "$REPO_DIR"

echo "[bootstrap] Repo: $REPO_DIR"
echo "[bootstrap] Python: $PYTHON_BIN"
echo "[bootstrap] Venv: $VENV_NAME"
echo "[bootstrap] Optional flash-attn install: $FLASH_ATTN_INSTALL"
echo "[bootstrap] Optional flash-attn spec: $FLASH_ATTN_PIP_SPEC"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python binary not found: $PYTHON_BIN"
  exit 2
fi

if [[ ! -d "$VENV_NAME" ]]; then
  echo "[bootstrap] Creating virtual environment..."
  "$PYTHON_BIN" -m venv "$VENV_NAME"
fi

source "$VENV_NAME/bin/activate"

python -m pip install --upgrade pip wheel

if [[ "$INSTALL_DEPS" == "1" ]]; then
  echo "[bootstrap] Installing requirements.linux.txt..."
  python -m pip install -r requirements.linux.txt
fi

if [[ "$FLASH_ATTN_INSTALL" == "1" ]]; then
  echo "[bootstrap] Checking optional flash-attn..."
  FLASH_ATTN_PIP_SPEC="$FLASH_ATTN_PIP_SPEC" python - <<'PY'
import importlib
import os
import subprocess
import sys

def log(msg: str) -> None:
    print(f"[bootstrap] {msg}", flush=True)

try:
    importlib.import_module("flash_attn")
    log("flash-attn already installed.")
    raise SystemExit(0)
except Exception:
    pass

try:
    import torch
    if not torch.cuda.is_available():
        log("CUDA unavailable in this runtime; skipping optional flash-attn install.")
        raise SystemExit(0)
except Exception as exc:
    log(f"torch probe failed; skipping optional flash-attn install ({exc}).")
    raise SystemExit(0)

spec = os.environ.get("FLASH_ATTN_PIP_SPEC", "flash-attn").strip() or "flash-attn"
log(f"Installing optional flash-attn package: {spec}")
cmd = [sys.executable, "-m", "pip", "install", "--no-build-isolation", spec]
proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
if proc.returncode != 0:
    output = (proc.stdout or "").strip()
    tail = "\n".join(output.splitlines()[-20:]) if output else ""
    log("Optional flash-attn install failed; continuing without it.")
    if tail:
        print(tail, flush=True)
    raise SystemExit(0)

try:
    importlib.import_module("flash_attn")
    log("flash-attn installed successfully.")
except Exception as exc:
    log(f"flash-attn install completed but import failed ({exc}); continuing without it.")
PY
fi

if [[ "$PREWARM_MODELS" == "1" ]]; then
  echo "[bootstrap] Prewarming models..."
  PREWARM_ARGS=(
    --prewarm-only
    --output-dir "$REPO_DIR/cloud_jobs/prewarm"
    --cpu-offload "$CPU_OFFLOAD"
  )

  if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
    PREWARM_ARGS+=(--local-files-only)
  fi
  if [[ "$DISABLE_XFORMERS" == "1" ]]; then
    PREWARM_ARGS+=(--disable-xformers)
  fi

  python cloud/run_depth_job.py "${PREWARM_ARGS[@]}"
fi

echo "[bootstrap] Done."
