# Vast.ai Docker Deployment Notes (DepthCrafter Cloud)

## Key feasibility answers

- Vast.ai instances pull container images from a registry (`--image` path).
- A Docker image that exists only in your local daemon is not directly deployable on Vast.ai.
- To use your custom image, build locally and push to a registry (`docker.io`, `ghcr.io`, etc.), then reference that registry tag in `vastai create instance --image ...`.

## Recommended base image choices

### Option A (recommended for your setup)

- `vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py312`

Why:

- CUDA 12.8.1 + cuDNN devel
- Python 3.12 already included
- Very compatible with your current `requirements.linux.txt` and torch cu128 wheel index

### Option B (leaner but more DIY)

- `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`

Why/Tradeoff:

- Smaller/fewer defaults, but you must install Python and other runtime pieces yourself.

## Dockerfile provided

- `cloud/Dockerfile.vastai`

It does the following:

1. Uses CUDA 12.8 base image.
2. Installs OS packages needed by your pipeline.
3. Creates virtual environment and installs pip deps from `requirements.linux.txt`.
4. Validates Python is `3.12.x` at build time.
5. Uses Hugging Face runtime auth (`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`) for gated model pulls.
6. Final step clones repo branch `004_depthcrafter_on_cloud` from your GitHub URL.

## Hugging Face credentials (gitignored file)

1. Copy template:

```bash
cp cloud/hf.env.example cloud/hf.env
```

2. Edit `cloud/hf.env` and set your accepted-terms token:

```bash
HF_TOKEN=hf_xxx...
```

3. `cloud/hf.env` is gitignored, so it stays local.

4. Convert env file into Vast `--env` format:

```bash
python cloud/envfile_to_vast_env.py --env-file cloud/hf.env
```

## Build and push

From repo root:

```bash
# Replace with your registry path.
export IMAGE=ghcr.io/<your_user>/stereocrafter-depthcrafter:004_depthcrafter_on_cloud

# Build. Replace GIT_REPO_URL with your repo URL.
docker build \
  -f cloud/Dockerfile.vastai \
  --build-arg GIT_REPO_URL=https://github.com/<your_user>/StereoCrafter.git \
  --build-arg GIT_BRANCH=004_depthcrafter_on_cloud \
  -t "$IMAGE" \
  .

# Push.
docker push "$IMAGE"
```

## One-command helper (build + push + print Vast command)

Use:

```bash
python cloud/release_vast_image.py \
  --image ghcr.io/<your_user>/stereocrafter-depthcrafter:004_depthcrafter_on_cloud \
  --git-repo-url https://github.com/<your_user>/StereoCrafter.git \
  --git-branch 004_depthcrafter_on_cloud \
  --env-file cloud/hf.env
```

This will:

1. Build using `cloud/Dockerfile.vastai`.
2. Push your image.
3. Print the exact `vastai create instance ...` command with `--env` already populated from `cloud/hf.env`.

To execute the create command directly, add:

```bash
--offer-id <OFFER_ID> --run-vast-create
```

## Vast.ai instance create example

```bash
HF_ENV_ARGS="$(python cloud/envfile_to_vast_env.py --env-file cloud/hf.env)"

vastai create instance <OFFER_ID> \
  --image ghcr.io/<your_user>/stereocrafter-depthcrafter:004_depthcrafter_on_cloud \
  --env "$HF_ENV_ARGS" \
  --disk 120 \
  --ssh --direct \
  --onstart-cmd "echo starting; nvidia-smi"
```

## Important sizing notes

- `--disk 40` is typically too small once image layers + HF cache + outputs accumulate.
- Start closer to `--disk 120` (or more) for reliability.

## Suggested improvements

1. Keep Hugging Face cache on a persistent volume/snapshot for faster repeat starts.
2. Keep `requirements.linux.txt` pinned (already done) and add immutable image tags (`:v1`, `:v2`) for reproducibility.
3. For faster startup, pre-pull image into your own template/snapshot instance if your workflow repeats often.
