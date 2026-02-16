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
