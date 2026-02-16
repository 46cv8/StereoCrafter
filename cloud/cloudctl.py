#!/usr/bin/env python3
"""Local cloud controller for SSH-only DepthCrafter jobs.

Design goals:
- No always-on backend service.
- Start a cloud instance, run local commands, stop the instance.
- Push code/inputs, execute remote job, pull outputs.
"""

from __future__ import annotations

import argparse
import dataclasses
import posixpath
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence


DEFAULT_LOCAL_REPO = Path(__file__).resolve().parents[1]


@dataclasses.dataclass
class SSHConfig:
    host: str
    user: str
    port: int
    identity: str
    ssh_options: List[str]


class CloudCtlError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[cloudctl] {msg}")


def run_cmd(cmd: Sequence[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    log("$ " + " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(
        list(cmd),
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def ssh_base_args(cfg: SSHConfig) -> List[str]:
    cmd = ["ssh", "-p", str(cfg.port)]
    if cfg.identity:
        cmd += ["-i", cfg.identity]
    for opt in cfg.ssh_options:
        cmd += ["-o", opt]
    cmd += [f"{cfg.user}@{cfg.host}"]
    return cmd


def ssh_transport_string(cfg: SSHConfig) -> str:
    parts = ["ssh", "-p", str(cfg.port)]
    if cfg.identity:
        parts += ["-i", cfg.identity]
    for opt in cfg.ssh_options:
        parts += ["-o", opt]
    return " ".join(shlex.quote(p) for p in parts)


def ssh_run(cfg: SSHConfig, remote_script: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    wrapped = f"bash -lc {shlex.quote(remote_script)}"
    return run_cmd(ssh_base_args(cfg) + [wrapped], check=check, capture=capture)


def rsync_to_remote(
    cfg: SSHConfig,
    local_src: str,
    remote_dst: str,
    *,
    delete: bool = False,
    excludes: Iterable[str] = (),
) -> None:
    cmd = [
        "rsync",
        "-az",
        "--partial",
        "--info=progress2",
        "-e",
        ssh_transport_string(cfg),
    ]
    if delete:
        cmd.append("--delete")
    for pattern in excludes:
        cmd += ["--exclude", pattern]
    cmd += [local_src, f"{cfg.user}@{cfg.host}:{remote_dst}"]
    run_cmd(cmd)


def rsync_from_remote(
    cfg: SSHConfig,
    remote_src: str,
    local_dst: str,
    *,
    delete: bool = False,
) -> None:
    cmd = [
        "rsync",
        "-az",
        "--partial",
        "--info=progress2",
        "-e",
        ssh_transport_string(cfg),
    ]
    if delete:
        cmd.append("--delete")
    cmd += [f"{cfg.user}@{cfg.host}:{remote_src}", local_dst]
    run_cmd(cmd)


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or f"job_{int(time.time())}"


def require_tool(name: str) -> None:
    from shutil import which

    if which(name) is None:
        raise CloudCtlError(f"Required tool '{name}' not found in PATH.")


def remote_join(*parts: str) -> str:
    if not parts:
        return ""
    out = parts[0]
    for part in parts[1:]:
        out = posixpath.join(out, part)
    return out


def add_ssh_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", required=True, help="Remote host/IP.")
    parser.add_argument("--user", default="root", help="SSH username.")
    parser.add_argument("--port", type=int, default=22, help="SSH port.")
    parser.add_argument("--identity", default="", help="SSH private key path.")
    parser.add_argument(
        "--ssh-option",
        action="append",
        default=["StrictHostKeyChecking=accept-new"],
        help="Extra ssh -o option (can be repeated).",
    )
    parser.add_argument(
        "--remote-root",
        required=True,
        help="Repo path on remote host (e.g. /workspace/StereoCrafter).",
    )


def add_model_job_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--venv-name", default=".venv-cloud")
    parser.add_argument("--target-width", type=int, default=1920)
    parser.add_argument("--target-height", type=int, default=1040)
    parser.add_argument("--window-size", type=int, default=75)
    parser.add_argument("--overlap", type=int, default=25)
    parser.add_argument("--inference-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-fps", type=float, default=-1.0)
    parser.add_argument("--process-length", type=int, default=-1)
    parser.add_argument("--output-format", choices=["mp4", "main10_mp4"], default="main10_mp4")
    parser.add_argument("--cpu-offload", choices=["model", "sequential", "none"], default="model")
    parser.add_argument("--disable-xformers", action="store_true")
    parser.add_argument("--use-cudnn-benchmark", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--unet-path", default="tencent/DepthCrafter")
    parser.add_argument("--pretrain-path", default="stabilityai/stable-video-diffusion-img2vid-xt")


def cfg_from_args(args: argparse.Namespace) -> SSHConfig:
    return SSHConfig(
        host=args.host,
        user=args.user,
        port=args.port,
        identity=args.identity,
        ssh_options=list(args.ssh_option or []),
    )


def cmd_bootstrap(args: argparse.Namespace) -> int:
    require_tool("ssh")
    require_tool("rsync")

    cfg = cfg_from_args(args)
    local_repo = Path(args.local_repo).expanduser().resolve()
    if not local_repo.exists():
        raise CloudCtlError(f"Local repo path does not exist: {local_repo}")

    remote_root = args.remote_root.rstrip("/")

    log(f"Ensuring remote root exists: {remote_root}")
    ssh_run(cfg, f"mkdir -p {shlex.quote(remote_root)}")

    if not args.no_sync_code:
        excludes = [
            ".git/",
            "__pycache__/",
            "*.pyc",
            ".pytest_cache/",
            ".mypy_cache/",
            ".venv*/",
            "cloud_jobs/",
        ]
        if not args.sync_weights:
            excludes.append("weights/")

        log("Syncing repository to remote host...")
        rsync_to_remote(
            cfg,
            local_src=f"{local_repo}/",
            remote_dst=f"{remote_root}/",
            delete=False,
            excludes=excludes,
        )

    if args.sync_weights:
        local_weights = local_repo / "weights"
        if not local_weights.exists():
            raise CloudCtlError(f"--sync-weights requested but local weights folder not found: {local_weights}")
        log("Syncing local weights folder to remote (this may take a while)...")
        rsync_to_remote(
            cfg,
            local_src=f"{local_weights}/",
            remote_dst=f"{remote_root}/weights/",
            delete=False,
            excludes=(".git/",),
        )

    bootstrap_parts = [
        f"cd {shlex.quote(remote_root)}",
        "./cloud/bootstrap_remote.sh"
        f" --repo-dir {shlex.quote(remote_root)}"
        f" --venv-name {shlex.quote(args.venv_name)}"
        f" --python-bin {shlex.quote(args.python_bin)}",
    ]

    if args.skip_install:
        bootstrap_parts[-1] += " --skip-install"
    if args.prewarm_models:
        bootstrap_parts[-1] += " --prewarm-models"
    if args.local_files_only:
        bootstrap_parts[-1] += " --local-files-only"
    if args.disable_xformers:
        bootstrap_parts[-1] += " --disable-xformers"
    if args.cpu_offload:
        bootstrap_parts[-1] += f" --cpu-offload {shlex.quote(args.cpu_offload)}"

    log("Running remote bootstrap...")
    ssh_run(cfg, " && ".join(bootstrap_parts))
    log("Bootstrap complete.")
    return 0


def _build_remote_job_cmd(args: argparse.Namespace, remote_input_path: str, remote_job_output_dir: str, job_name: str) -> str:
    status_json = remote_join(remote_job_output_dir, "job_status.json")
    runner_args = [
        "python",
        "cloud/run_depth_job.py",
        "--input",
        remote_input_path,
        "--output-dir",
        remote_job_output_dir,
        "--status-json",
        status_json,
        "--job-name",
        job_name,
        "--target-width",
        str(args.target_width),
        "--target-height",
        str(args.target_height),
        "--window-size",
        str(args.window_size),
        "--overlap",
        str(args.overlap),
        "--inference-steps",
        str(args.inference_steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--seed",
        str(args.seed),
        "--target-fps",
        str(args.target_fps),
        "--process-length",
        str(args.process_length),
        "--output-format",
        args.output_format,
        "--cpu-offload",
        args.cpu_offload,
        "--unet-path",
        args.unet_path,
        "--pretrain-path",
        args.pretrain_path,
    ]

    if args.disable_xformers:
        runner_args.append("--disable-xformers")
    if args.use_cudnn_benchmark:
        runner_args.append("--use-cudnn-benchmark")
    if args.local_files_only:
        runner_args.append("--local-files-only")

    quoted_runner = " ".join(shlex.quote(x) for x in runner_args)
    remote_root = args.remote_root.rstrip("/")
    return (
        f"cd {shlex.quote(remote_root)}"
        f" && source {shlex.quote(remote_join(remote_root, args.venv_name, 'bin', 'activate'))}"
        f" && {quoted_runner}"
    )


def _run_one_job(
    cfg: SSHConfig,
    args: argparse.Namespace,
    local_input: Path,
    *,
    explicit_job_name: str = "",
    batch_index: int = 0,
    batch_total: int = 0,
) -> int:
    if not local_input.exists():
        raise CloudCtlError(f"Input clip not found: {local_input}")

    stem = sanitize_name(local_input.stem)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if explicit_job_name:
        job_name = sanitize_name(explicit_job_name)
    elif batch_total > 1:
        job_name = sanitize_name(f"{args.job_prefix}{batch_index:03d}_{stem}_{stamp}")
    else:
        job_name = sanitize_name(f"{args.job_prefix}{stem}_{stamp}")

    remote_root = args.remote_root.rstrip("/")
    remote_input_dir = remote_join(remote_root, args.remote_input_subdir)
    remote_output_dir = remote_join(remote_root, args.remote_output_subdir)
    remote_job_output_dir = remote_join(remote_output_dir, job_name)
    remote_input_filename = f"{job_name}{local_input.suffix.lower()}"
    remote_input_path = remote_join(remote_input_dir, remote_input_filename)

    log_prefix = f"[{batch_index}/{batch_total}] " if batch_total > 1 else ""
    log(f"{log_prefix}Preparing job {job_name}")

    ssh_run(
        cfg,
        " && ".join(
            [
                f"mkdir -p {shlex.quote(remote_input_dir)}",
                f"mkdir -p {shlex.quote(remote_output_dir)}",
                f"mkdir -p {shlex.quote(remote_job_output_dir)}",
            ]
        ),
    )

    log(f"{log_prefix}Uploading clip: {local_input}")
    rsync_to_remote(cfg, str(local_input), remote_input_path)

    remote_cmd = _build_remote_job_cmd(args, remote_input_path, remote_job_output_dir, job_name)
    log(f"{log_prefix}Running remote inference...")
    result = ssh_run(cfg, remote_cmd, check=False)

    local_job_dir = Path(args.download_dir).expanduser().resolve() / job_name
    if not args.skip_download:
        local_job_dir.mkdir(parents=True, exist_ok=True)
        log(f"{log_prefix}Downloading outputs to: {local_job_dir}")
        rsync_from_remote(cfg, f"{remote_job_output_dir}/", f"{local_job_dir}/")

    if not args.keep_remote_input:
        ssh_run(cfg, f"rm -f {shlex.quote(remote_input_path)}", check=False)

    if not args.keep_remote_output and not args.skip_download and result.returncode == 0:
        ssh_run(cfg, f"rm -rf {shlex.quote(remote_job_output_dir)}", check=False)

    if result.returncode != 0:
        raise CloudCtlError(f"Remote job failed (exit code {result.returncode}) for {local_input.name}")

    log(f"{log_prefix}Job complete: {job_name}")
    return 0


def cmd_run_job(args: argparse.Namespace) -> int:
    require_tool("ssh")
    require_tool("rsync")
    cfg = cfg_from_args(args)
    local_input = Path(args.local_input).expanduser().resolve()
    return _run_one_job(cfg, args, local_input, explicit_job_name=args.job_name)


def _discover_batch_inputs(input_dir: Path, patterns: List[str], recursive: bool) -> List[Path]:
    found: List[Path] = []
    for raw_pattern in patterns:
        pattern = raw_pattern.strip()
        if not pattern:
            continue
        iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
        found.extend(p for p in iterator if p.is_file())

    seen = set()
    ordered_unique = []
    for p in sorted(found):
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        ordered_unique.append(p)
    return ordered_unique


def cmd_run_batch(args: argparse.Namespace) -> int:
    require_tool("ssh")
    require_tool("rsync")

    cfg = cfg_from_args(args)
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise CloudCtlError(f"Input dir does not exist: {input_dir}")

    patterns = [p for p in args.patterns.split(",") if p.strip()]
    clips = _discover_batch_inputs(input_dir, patterns, bool(args.recursive))
    if args.max_jobs > 0:
        clips = clips[: args.max_jobs]

    if not clips:
        raise CloudCtlError("No clips matched the requested patterns.")

    log(f"Found {len(clips)} clip(s) for batch processing.")

    failed = []
    for idx, clip in enumerate(clips, start=1):
        try:
            _run_one_job(cfg, args, clip, batch_index=idx, batch_total=len(clips))
        except Exception as exc:  # pylint: disable=broad-except
            msg = f"Batch item failed for {clip.name}: {exc}"
            log(msg)
            failed.append(msg)
            if not args.continue_on_error:
                break

    if failed:
        for item in failed:
            log(f"ERROR: {item}")
        return 1

    log("Batch complete.")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    require_tool("ssh")
    require_tool("rsync")

    cfg = cfg_from_args(args)
    remote_root = args.remote_root.rstrip("/")
    remote_output_dir = remote_join(remote_root, args.remote_output_subdir)
    local_dir = Path(args.download_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    log(f"Collecting remote outputs from {remote_output_dir} -> {local_dir}")
    rsync_from_remote(cfg, f"{remote_output_dir}/", f"{local_dir}/", delete=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SSH-only cloud controller for DepthCrafter jobs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_boot = sub.add_parser("bootstrap", help="Sync repo and prepare remote Python env.")
    add_ssh_flags(p_boot)
    p_boot.add_argument("--local-repo", default=str(DEFAULT_LOCAL_REPO))
    p_boot.add_argument("--venv-name", default=".venv-cloud")
    p_boot.add_argument("--python-bin", default="python3")
    p_boot.add_argument("--no-sync-code", action="store_true")
    p_boot.add_argument("--sync-weights", action="store_true", help="Rsync local weights/ to remote.")
    p_boot.add_argument("--skip-install", action="store_true")
    p_boot.add_argument("--prewarm-models", action="store_true")
    p_boot.add_argument("--local-files-only", action="store_true")
    p_boot.add_argument("--disable-xformers", action="store_true")
    p_boot.add_argument("--cpu-offload", choices=["model", "sequential", "none"], default="model")
    p_boot.set_defaults(func=cmd_bootstrap)

    p_job = sub.add_parser("run-job", help="Upload one clip, run remote inference, download result.")
    add_ssh_flags(p_job)
    add_model_job_flags(p_job)
    p_job.add_argument("--local-input", required=True)
    p_job.add_argument("--job-name", default="")
    p_job.add_argument("--job-prefix", default="")
    p_job.add_argument("--remote-input-subdir", default="cloud_jobs/incoming")
    p_job.add_argument("--remote-output-subdir", default="cloud_jobs/output")
    p_job.add_argument("--download-dir", default="./cloud_downloads")
    p_job.add_argument("--skip-download", action="store_true")
    p_job.add_argument("--keep-remote-input", action="store_true")
    p_job.add_argument("--keep-remote-output", action="store_true")
    p_job.set_defaults(func=cmd_run_job)

    p_batch = sub.add_parser("run-batch", help="Upload and run a batch of clips sequentially.")
    add_ssh_flags(p_batch)
    add_model_job_flags(p_batch)
    p_batch.add_argument("--input-dir", required=True)
    p_batch.add_argument("--patterns", default="*.mkv,*.mp4,*.mov,*.avi")
    p_batch.add_argument("--recursive", action="store_true")
    p_batch.add_argument("--max-jobs", type=int, default=0)
    p_batch.add_argument("--continue-on-error", action="store_true")
    p_batch.add_argument("--job-prefix", default="")
    p_batch.add_argument("--remote-input-subdir", default="cloud_jobs/incoming")
    p_batch.add_argument("--remote-output-subdir", default="cloud_jobs/output")
    p_batch.add_argument("--download-dir", default="./cloud_downloads")
    p_batch.add_argument("--skip-download", action="store_true")
    p_batch.add_argument("--keep-remote-input", action="store_true")
    p_batch.add_argument("--keep-remote-output", action="store_true")
    p_batch.set_defaults(func=cmd_run_batch)

    p_collect = sub.add_parser("collect", help="Download remote output folder.")
    add_ssh_flags(p_collect)
    p_collect.add_argument("--remote-output-subdir", default="cloud_jobs/output")
    p_collect.add_argument("--download-dir", default="./cloud_downloads")
    p_collect.set_defaults(func=cmd_collect)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        return int(args.func(args))
    except CloudCtlError as exc:
        log(f"ERROR: {exc}")
        return 2
    except KeyboardInterrupt:
        log("Interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
