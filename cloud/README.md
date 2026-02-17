# Cloud DepthCrafter (SSH-Only)

This workflow uses only `ssh` + `rsync` from your local machine.
No separate always-on API server is required.

## Files

- `cloud/cloudctl.py`: local controller (`bootstrap`, `run-job`, `run-batch`, `collect`)
- `cloud/bootstrap_remote.sh`: prepares remote venv + dependencies
- `cloud/run_depth_job.py`: remote headless job runner
- `cloud/release_vast_image.py`: local helper to build/push image and print/run `vastai create instance`
- `cloud/vast_worker_launch.py`: pick cheapest offer for a GPU profile, launch one instance, generate cloud config copy, and optionally run `cloudctl`
- `cloud/hf.env.example`: template for Hugging Face token env file
- `cloud/ghcr.env.example`: template for GHCR login env file
- `cloud/vast.env.example`: template for Vast API key env file

## Prerequisites

Local machine:

- `python3`
- `ssh`
- `rsync`

Remote machine (Vast instance):

- Linux with NVIDIA driver/CUDA runtime
- `python3` and `venv` support
- enough disk for model cache and outputs

## Refresh Cloud Image (when branch changes)

`cloud/Dockerfile.vastai` now does:

- `git clone` of your branch
- `git pull --ff-only` inside image build
- optional commit pin (`GIT_COMMIT`) and baked commit record at `/opt/stereocrafter_git_commit.txt`

Use this from repo root to rebuild and push with latest branch contents:

```bash
python cloud/release_vast_image.py \
  --image ghcr.io/46cv8/stereocrafter-cloud:latest \
  --git-repo-url https://github.com/46cv8/StereoCrafter.git \
  --env-file cloud/hf.env \
  --ghcr-env-file cloud/ghcr.env \
  --refresh-repo
```

`release_vast_image.py` now auto-detects your current local git branch when `--git-branch` is omitted.

Note: `--docker-pull-base` is intentionally omitted for speed. Add it only when you explicitly want to refresh the CUDA base image (it can invalidate more cache layers and trigger long rebuilds).

If you want to pin a specific commit:

```bash
python cloud/release_vast_image.py \
  --image ghcr.io/46cv8/stereocrafter-cloud:latest \
  --git-repo-url https://github.com/46cv8/StereoCrafter.git \
  --git-commit <commit_sha> \
  --env-file cloud/hf.env \
  --ghcr-env-file cloud/ghcr.env
```

## 0) One-command worker launch (recommended)

This flow avoids editing your default `config_depthcrafter.json`.
It will:

1. search offers for profile `5090_32gb`, `rtx_pro_6000_96gb`, or `nvidia_48gb_single`,
2. show candidate costs,
3. ask approval,
4. create one instance,
5. wait until ready,
6. write a generated config copy under `cloud/generated_configs/`,
7. print the exact `cloudctl` command to run.

Setup once:

```bash
cp cloud/vast.env.example cloud/vast.env
cp cloud/hf.env.example cloud/hf.env
cp cloud/ghcr.env.example cloud/ghcr.env
```

Fill:

- `cloud/vast.env` with `VAST_API_KEY=...`
- `cloud/hf.env` with your accepted-terms HF token.
- If your `--image` is private on GHCR: fill `cloud/ghcr.env` (`GHCR_USERNAME`, `GHCR_PAT` with at least `read:packages`).
- Optional blacklist file: `cloud/cloud_blacklist.json` (auto-created by GUI when you blacklist failed hosts).
- Optional provider history file: `cloud/cloud_provider_history.json` (auto-created by GUI to track prior provider usage).

Launch a 5090 worker using your current config as base:

```bash
python cloud/vast_worker_launch.py \
  --profile 5090_32gb \
  --image ghcr.io/46cv8/stereocrafter-cloud:latest \
  --base-config config_depthcrafter.json \
  --remote-root /opt/StereoCrafter \
  --remote-venv /opt/venv
```

Launch a 96GB RTX PRO 6000 worker:

```bash
python cloud/vast_worker_launch.py \
  --profile rtx_pro_6000_96gb \
  --image ghcr.io/46cv8/stereocrafter-cloud:latest \
  --base-config config_depthcrafter.json \
  --remote-root /opt/StereoCrafter \
  --remote-venv /opt/venv
```

Notes:

- `5090_32gb` profile defaults to `1664x896`.
- `rtx_pro_6000_96gb` profile defaults to `1920x1040`.
- `nvidia_48gb_single` targets any single-GPU CUDA-capable host with at least 48 GB VRAM and is intended for input-resolution cloud runs.
- default disk is `40GB` (override with `--disk`).
- use `--dry-run` to only rank/select offers without creating an instance.
- The script prompts `Type GO ...` after readiness; that starts remote processing immediately.
- Add `--run-now` to skip GO prompt.
- Add `--output-config /path/custom.json` to control generated config path.
- Add `--blacklist-file /path/to/cloud_blacklist.json` to override default blacklist location.
- Add `--git-sync-branch <branch>` to force which branch remote sync checks out before jobs. If omitted, your current local branch is auto-detected.

## 1) Bootstrap a fresh instance

Run from your local repo root:

```bash
python cloud/cloudctl.py bootstrap \
  --host <REMOTE_IP> \
  --user root \
  --port 22 \
  --identity ~/.ssh/id_rsa \
  --remote-root /workspace/StereoCrafter \
  --local-repo /home/peter/tuft/Movies/3D/StereoCrafter \
  --venv-name .venv-cloud \
  --python-bin python3 \
  --prewarm-models
```

Optional:

- add `--sync-weights` to copy local `weights/` instead of downloading.
- add `--local-files-only` if remote already has all model files and you want zero network model fetch.

## 2) Run one clip

```bash
python cloud/cloudctl.py run-job \
  --host <REMOTE_IP> \
  --user root \
  --identity ~/.ssh/id_rsa \
  --remote-root /workspace/StereoCrafter \
  --local-input "/path/to/Scene-0007.mkv" \
  --download-dir ./cloud_downloads \
  --target-width 1920 \
  --target-height 1036 \
  --window-size 75 \
  --overlap 25 \
  --inference-steps 25 \
  --guidance-scale 1.0 \
  --output-format main10_mp4 \
  --cpu-offload model
```

Notes:

- Height/width are automatically rounded to the nearest multiple of 8 on remote.
- `main10_mp4` writes HEVC Main10 output.

## 3) Run a batch

```bash
python cloud/cloudctl.py run-batch \
  --host <REMOTE_IP> \
  --user root \
  --identity ~/.ssh/id_rsa \
  --remote-root /workspace/StereoCrafter \
  --input-dir "/path/to/Clips_001" \
  --patterns "*.mkv,*.mp4" \
  --download-dir ./cloud_downloads \
  --target-width 1920 \
  --target-height 1036 \
  --window-size 75 \
  --overlap 25 \
  --inference-steps 25 \
  --output-format main10_mp4 \
  --continue-on-error
```

## 4) Collect output folder later

```bash
python cloud/cloudctl.py collect \
  --host <REMOTE_IP> \
  --user root \
  --identity ~/.ssh/id_rsa \
  --remote-root /workspace/StereoCrafter \
  --download-dir ./cloud_downloads
```

## Output layout

Remote:

- `cloud_jobs/incoming/` uploaded clips
- `cloud_jobs/output/<job_name>/` outputs + `job_status.json`

Local:

- `./cloud_downloads/<job_name>/` copied results

## Troubleshooting

- If model download fails on bootstrap, run again with `--prewarm-models` after connectivity is stable.
- If remote job OOMs, reduce `--window-size` (for example `75 -> 64 -> 56 -> 48`).
- If `--local-files-only` is set and models are missing remotely, job will fail immediately.
