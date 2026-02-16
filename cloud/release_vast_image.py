#!/usr/bin/env python3
"""Build/push StereoCrafter image and print or run Vast create command.

Typical flow:
1) Build Docker image using cloud/Dockerfile.vastai.
2) Push the image to your registry.
3) Read cloud/hf.env and generate --env payload for Vast.
4) Print the exact `vastai create instance ...` command.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from shutil import which
from typing import Dict, Iterable, List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from envfile_to_vast_env import build_vast_env_arg, parse_env_file  # noqa: E402


DEFAULT_ONSTART = "echo starting; nvidia-smi"


class ReleaseError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[release-vast] {msg}")


def shell_join(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def run(cmd: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess:
    log("$ " + shell_join(cmd))
    return subprocess.run(list(cmd), check=check)


def require_tool(name: str) -> None:
    if which(name) is None:
        raise ReleaseError(f"Required tool not found in PATH: {name}")


def validate_image_ref(image: str) -> None:
    if "/" not in image or ":" not in image:
        raise ReleaseError(
            "--image should include registry/repo:tag (example: ghcr.io/user/stereocrafter:cloud-v1)."
        )


def load_env_payload(env_file: Path, no_env_file: bool) -> str:
    if no_env_file:
        return ""

    env_vars: Dict[str, str] = parse_env_file(env_file)
    if not env_vars:
        raise ReleaseError(f"No env vars found in {env_file}")

    if "HF_TOKEN" not in env_vars and "HUGGING_FACE_HUB_TOKEN" not in env_vars:
        raise ReleaseError(
            f"{env_file} must include HF_TOKEN or HUGGING_FACE_HUB_TOKEN for gated model access."
        )

    return build_vast_env_arg(env_vars)


def build_docker_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        "docker",
        "build",
        "-f",
        str(args.dockerfile),
        "-t",
        args.image,
        "--build-arg",
        f"GIT_REPO_URL={args.git_repo_url}",
        "--build-arg",
        f"GIT_BRANCH={args.git_branch}",
    ]
    if args.base_image:
        cmd += ["--build-arg", f"BASE_IMAGE={args.base_image}"]
    if args.platform:
        cmd += ["--platform", args.platform]
    cmd.append(str(args.context))
    return cmd


def build_vast_cmd(args: argparse.Namespace, env_payload: str) -> List[str]:
    cmd = [
        args.vast_cli,
        "create",
        "instance",
        str(args.offer_id),
        "--image",
        args.image,
        "--disk",
        str(args.disk),
        "--ssh",
        "--direct",
        "--onstart-cmd",
        args.onstart_cmd,
    ]
    if env_payload:
        cmd += ["--env", env_payload]
    if args.vast_extra_args:
        cmd += shlex.split(args.vast_extra_args)
    return cmd


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build/push image and prepare Vast create command.")
    p.add_argument("--image", required=True, help="Registry image ref (registry/repo:tag).")
    p.add_argument("--git-repo-url", required=True, help="GitHub repo URL to clone in image build.")
    p.add_argument("--git-branch", default="004_depthcrafter_on_cloud")

    p.add_argument("--dockerfile", default=str(Path("cloud") / "Dockerfile.vastai"))
    p.add_argument("--context", default=".")
    p.add_argument("--platform", default="linux/amd64")
    p.add_argument("--base-image", default="")

    p.add_argument("--env-file", default=str(Path("cloud") / "hf.env"))
    p.add_argument("--no-env-file", action="store_true", help="Do not pass --env to vastai.")

    p.add_argument("--offer-id", default="<OFFER_ID>")
    p.add_argument("--disk", type=int, default=120)
    p.add_argument("--onstart-cmd", default=DEFAULT_ONSTART)
    p.add_argument("--vast-cli", default="vastai")
    p.add_argument("--vast-extra-args", default="", help="Extra args appended to vastai command.")

    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-push", action="store_true")
    p.add_argument(
        "--run-vast-create",
        action="store_true",
        help="Run vastai create instance after build/push. Default is print-only.",
    )
    return p


def main() -> int:
    args = parser().parse_args()

    validate_image_ref(args.image)
    env_file = Path(args.env_file).expanduser().resolve()
    dockerfile = Path(args.dockerfile).expanduser().resolve()
    context = Path(args.context).expanduser().resolve()

    args.dockerfile = dockerfile
    args.context = context

    if not args.skip_build or not args.skip_push:
        require_tool("docker")
    if args.run_vast_create:
        require_tool(args.vast_cli)

    if not dockerfile.exists():
        raise ReleaseError(f"Dockerfile not found: {dockerfile}")
    if not context.exists():
        raise ReleaseError(f"Build context not found: {context}")

    env_payload = load_env_payload(env_file, args.no_env_file)

    docker_build_cmd = build_docker_cmd(args)
    docker_push_cmd = ["docker", "push", args.image]
    vast_cmd = build_vast_cmd(args, env_payload)

    if not args.skip_build:
        run(docker_build_cmd)

    if not args.skip_push:
        run(docker_push_cmd)

    log("Vast create command:")
    print(shell_join(vast_cmd))

    if args.run_vast_create:
        if str(args.offer_id).strip() in {"", "<OFFER_ID>", "OFFER_ID"}:
            raise ReleaseError("--offer-id must be set to a concrete offer id when --run-vast-create is used.")
        run(vast_cmd)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseError as exc:
        log(f"ERROR: {exc}")
        raise SystemExit(2)
