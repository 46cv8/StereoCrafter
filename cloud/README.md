# Cloud DepthCrafter (SSH-Only)

This workflow uses only `ssh` + `rsync` from your local machine.
No separate always-on API server is required.

## Files

- `cloud/cloudctl.py`: local controller (`bootstrap`, `run-job`, `run-batch`, `collect`)
- `cloud/bootstrap_remote.sh`: prepares remote venv + dependencies
- `cloud/run_depth_job.py`: remote headless job runner
- `cloud/release_vast_image.py`: local helper to build/push image and print/run `vastai create instance`

## Prerequisites

Local machine:

- `python3`
- `ssh`
- `rsync`

Remote machine (Vast instance):

- Linux with NVIDIA driver/CUDA runtime
- `python3` and `venv` support
- enough disk for model cache and outputs

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
