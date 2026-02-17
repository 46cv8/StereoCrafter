#!/usr/bin/env python3
"""Local cloud controller for SSH-only DepthCrafter jobs.

Design goals:
- No always-on backend service.
- Start a cloud instance, run local commands, stop the instance.
- Push code/inputs, execute remote job, pull outputs.
"""

from __future__ import annotations

import argparse
import codecs
import dataclasses
import json
import os
import posixpath
import re
import select
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterable, List, Sequence


DEFAULT_LOCAL_REPO = Path(__file__).resolve().parents[1]
if str(DEFAULT_LOCAL_REPO) not in sys.path:
    sys.path.insert(0, str(DEFAULT_LOCAL_REPO))

from dependency.clip_ordering import sort_paths_by_clip_id


@dataclasses.dataclass
class SSHConfig:
    host: str
    user: str
    port: int
    identity: str
    ssh_options: List[str]


@dataclasses.dataclass
class RemoteLogTailer:
    process: subprocess.Popen
    thread: threading.Thread
    stop_event: threading.Event


class CloudCtlError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[cloudctl] {msg}", flush=True)


def run_cmd(
    cmd: Sequence[str],
    check: bool = True,
    capture: bool = False,
    *,
    log_command: bool = True,
) -> subprocess.CompletedProcess:
    if log_command:
        log("$ " + " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(
        list(cmd),
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def run_cmd_stream(
    cmd: Sequence[str],
    check: bool = True,
    *,
    log_command: bool = True,
    line_prefix: str = "",
    heartbeat_sec: int = 30,
) -> subprocess.CompletedProcess:
    if log_command:
        log("$ " + " ".join(shlex.quote(c) for c in cmd))

    process = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    output_lines: List[str] = []
    start_ts = time.time()
    last_heartbeat = start_ts
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    line_buffer = ""
    latest_progress_text = ""
    progress_last_emit_ts = 0.0
    progress_emit_interval_sec = 1.0
    progress_updates_seen = 0
    progress_updates_emitted = 0

    def _emit_line(text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        output_lines.append(cleaned)
        log(f"{line_prefix}{cleaned}" if line_prefix else cleaned)

    def _emit_progress(text: str, force: bool = False) -> bool:
        nonlocal progress_last_emit_ts
        nonlocal progress_updates_emitted
        cleaned = text.strip()
        if not cleaned:
            return False
        now = time.time()
        if not force and (now - progress_last_emit_ts) < progress_emit_interval_sec:
            return False
        progress_last_emit_ts = now
        progress_updates_emitted += 1
        _emit_line(cleaned)
        return True

    def _consume_text(decoded_text: str) -> None:
        nonlocal line_buffer
        nonlocal latest_progress_text
        nonlocal progress_updates_seen
        for ch in decoded_text:
            if ch == "\r":
                candidate = line_buffer
                line_buffer = ""
                if candidate.strip():
                    latest_progress_text = candidate
                    progress_updates_seen += 1
                    _emit_progress(latest_progress_text, force=False)
                continue
            if ch == "\n":
                if line_buffer.strip():
                    _emit_line(line_buffer)
                elif latest_progress_text.strip():
                    _emit_progress(latest_progress_text, force=True)
                line_buffer = ""
                latest_progress_text = ""
                continue
            line_buffer += ch

    try:
        if process.stdout is not None:
            stdout_fd = process.stdout.fileno()
            while True:
                ready, _, _ = select.select([stdout_fd], [], [], 0.25)
                if ready:
                    chunk = os.read(stdout_fd, 4096)
                    if chunk:
                        _consume_text(decoder.decode(chunk))
                    else:
                        break

                now = time.time()
                if latest_progress_text.strip() and (now - progress_last_emit_ts) >= progress_emit_interval_sec:
                    _emit_progress(latest_progress_text, force=True)

                if process.poll() is None and (now - last_heartbeat) >= max(1, int(heartbeat_sec)):
                    elapsed = now - start_ts
                    log(
                        f"{line_prefix}(still running, elapsed {elapsed:.0f}s)"
                        if line_prefix
                        else f"(still running, elapsed {elapsed:.0f}s)"
                    )
                    last_heartbeat = now

                if process.poll() is not None:
                    remaining = process.stdout.read() or b""
                    if remaining:
                        _consume_text(decoder.decode(remaining))
                    break

            _consume_text(decoder.decode(b"", final=True))
            if line_buffer.strip():
                _emit_line(line_buffer)
            elif latest_progress_text.strip():
                _emit_progress(latest_progress_text, force=True)
        returncode = process.wait()
    finally:
        if process.stdout is not None:
            try:
                process.stdout.close()
            except Exception:
                pass

    suppressed = progress_updates_seen - progress_updates_emitted
    if suppressed > 0:
        log(
            f"{line_prefix}(suppressed {suppressed} carriage-return progress updates)"
            if line_prefix
            else f"(suppressed {suppressed} carriage-return progress updates)"
        )

    stdout_joined = "\n".join(output_lines)
    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, list(cmd), output=stdout_joined)
    return subprocess.CompletedProcess(list(cmd), returncode, stdout_joined)


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


def ssh_run(
    cfg: SSHConfig,
    remote_script: str,
    check: bool = True,
    capture: bool = False,
    *,
    log_command: bool = True,
) -> subprocess.CompletedProcess:
    wrapped = f"bash -lc {shlex.quote(remote_script)}"
    return run_cmd(ssh_base_args(cfg) + [wrapped], check=check, capture=capture, log_command=log_command)


def ssh_run_stream(
    cfg: SSHConfig,
    remote_script: str,
    check: bool = True,
    *,
    log_command: bool = True,
    line_prefix: str = "",
    heartbeat_sec: int = 30,
) -> subprocess.CompletedProcess:
    wrapped = f"bash -lc {shlex.quote(remote_script)}"
    return run_cmd_stream(
        ssh_base_args(cfg) + [wrapped],
        check=check,
        log_command=log_command,
        line_prefix=line_prefix,
        heartbeat_sec=heartbeat_sec,
    )


def _start_remote_log_tailer(
    cfg: SSHConfig,
    remote_log_path: str,
    *,
    line_prefix: str = "",
    log_command: bool = True,
) -> RemoteLogTailer:
    remote_script = f"touch {shlex.quote(remote_log_path)} && tail -n 0 -F {shlex.quote(remote_log_path)}"
    wrapped = f"bash -lc {shlex.quote(remote_script)}"
    cmd = ssh_base_args(cfg) + [wrapped]
    if log_command:
        log("$ " + " ".join(shlex.quote(c) for c in cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stop_event = threading.Event()

    def _reader() -> None:
        stream = proc.stdout
        if stream is None:
            return
        try:
            for raw in stream:
                if stop_event.is_set():
                    break
                text = raw.rstrip("\r\n")
                if text:
                    log(f"{line_prefix}{text}" if line_prefix else text)
        except Exception as exc:  # pylint: disable=broad-except
            if not stop_event.is_set():
                log(f"{line_prefix}[tail-warning] {exc}" if line_prefix else f"[tail-warning] {exc}")
        finally:
            try:
                stream.close()
            except Exception:
                pass

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return RemoteLogTailer(process=proc, thread=thread, stop_event=stop_event)


def _stop_remote_log_tailer(tailer: RemoteLogTailer | None) -> None:
    if tailer is None:
        return
    tailer.stop_event.set()
    proc = tailer.process
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except Exception:
                proc.kill()
                proc.wait(timeout=2.0)
    except Exception:
        pass
    try:
        tailer.thread.join(timeout=2.0)
    except Exception:
        pass


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


def wait_for_ssh_ready(cfg: SSHConfig, timeout_sec: int = 180, poll_sec: int = 4) -> None:
    timeout_sec = max(5, int(timeout_sec))
    poll_sec = max(1, int(poll_sec))
    deadline = time.time() + timeout_sec
    attempt = 0
    last_error = ""
    heartbeat_sec = 20
    next_heartbeat = time.time() + heartbeat_sec

    while time.time() < deadline:
        attempt += 1
        probe = ssh_run(
            cfg,
            "echo cloudctl_ssh_ready",
            check=False,
            capture=True,
            log_command=False,
        )
        if probe.returncode == 0:
            if attempt > 1:
                log(
                    f"SSH became ready after {attempt} probe(s) at "
                    f"{cfg.user}@{cfg.host}:{cfg.port}."
                )
            else:
                log(f"SSH ready at {cfg.user}@{cfg.host}:{cfg.port}.")
            return

        probe_output = (probe.stdout or "").strip()
        short_error = probe_output.splitlines()[-1] if probe_output else f"ssh exit {probe.returncode}"
        if short_error != last_error:
            log(
                f"Waiting for SSH readiness at {cfg.user}@{cfg.host}:{cfg.port} "
                f"(attempt {attempt}): {short_error}"
            )
            last_error = short_error
        now = time.time()
        if now >= next_heartbeat:
            elapsed = max(0, timeout_sec - int(max(0.0, deadline - now)))
            log(
                f"Still waiting for SSH readiness at {cfg.user}@{cfg.host}:{cfg.port} "
                f"(attempt {attempt}, elapsed {elapsed}s/{timeout_sec}s, last: {short_error})"
            )
            next_heartbeat = now + heartbeat_sec
        time.sleep(poll_sec)

    raise CloudCtlError(
        f"Timed out waiting for SSH readiness at {cfg.user}@{cfg.host}:{cfg.port} "
        f"after {timeout_sec}s. Last error: {last_error or 'unknown'}"
    )


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
    parser.add_argument(
        "--ssh-ready-timeout-sec",
        type=int,
        default=180,
        help="How long to wait for SSH to accept connections before failing.",
    )
    parser.add_argument(
        "--ssh-ready-poll-sec",
        type=int,
        default=4,
        help="Polling interval while waiting for SSH readiness.",
    )


def add_model_job_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--venv-name", default=".venv-cloud")
    parser.add_argument(
        "--model-backend",
        choices=["depthcrafter", "geometrycrafter_diff", "geometrycrafter_determ"],
        default="depthcrafter",
        help="Inference backend to run on remote.",
    )
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
    parser.add_argument("--geometry-model-path", default="TencentARC/GeometryCrafter")
    parser.add_argument("--geometry-repo-path", default="")
    parser.add_argument("--geometry-cache-dir", default="")
    parser.add_argument("--geometry-decode-chunk-size", type=int, default=8)
    parser.add_argument("--geometry-low-memory-usage", action="store_true")
    parser.add_argument(
        "--geometry-force-projection",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--geometry-force-fixed-focal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--geometry-use-extract-interp",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--auto-git-sync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Before inference, attempt fast-forward sync of remote repo from origin/<current-branch>.",
    )
    parser.add_argument(
        "--git-sync-branch",
        default="",
        help="Optional branch to force on remote before run (for example your current local branch).",
    )
    parser.add_argument("--unet-path", default="tencent/DepthCrafter")
    parser.add_argument("--pretrain-path", default="stabilityai/stable-video-diffusion-img2vid-xt")


def add_queue_name_flag(parser: argparse.ArgumentParser, *, default: str = "default") -> None:
    parser.add_argument("--queue-name", default=default, help="Queue worker name.")


def maybe_sync_remote_repo(cfg: SSHConfig, args: argparse.Namespace) -> None:
    if not bool(getattr(args, "auto_git_sync", True)):
        log("Remote git sync disabled (--no-auto-git-sync).")
        return

    remote_root = str(args.remote_root).rstrip("/")
    requested_branch = str(getattr(args, "git_sync_branch", "") or "").strip()
    if requested_branch and not re.fullmatch(r"[A-Za-z0-9._/-]+", requested_branch):
        raise CloudCtlError(
            f"Invalid --git-sync-branch '{requested_branch}'. "
            "Allowed characters: letters, digits, ., _, -, /"
        )

    if requested_branch:
        sync_script = (
            f"cd {shlex.quote(remote_root)}"
            " && if ! command -v git >/dev/null 2>&1; then echo 'skip: git not found'; exit 0; fi"
            " && if [ ! -d .git ]; then echo 'skip: no .git folder'; exit 0; fi"
            " && if ! git remote get-url origin >/dev/null 2>&1; then echo 'skip: no origin remote'; exit 0; fi"
            " && if ! git diff --quiet --ignore-submodules -- || ! git diff --cached --quiet --ignore-submodules --; then "
            "echo 'skip: dirty worktree'; exit 0; fi"
            f" && branch={shlex.quote(requested_branch)}"
            " && echo \"sync: target origin/$branch\""
            " && if ! git fetch origin \"$branch\" --prune; then echo \"skip: fetch failed for $branch\"; exit 0; fi"
            " && remote_sha=\"$(git rev-parse FETCH_HEAD 2>/dev/null || true)\""
            " && if [ -z \"$remote_sha\" ]; then echo \"skip: could not resolve FETCH_HEAD for $branch\"; exit 0; fi"
            " && current_branch=\"$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)\""
            " && local_sha_before=\"$(git rev-parse HEAD 2>/dev/null || true)\""
            " && if [ \"$current_branch\" != \"$branch\" ]; then "
            "if git show-ref --verify --quiet \"refs/heads/$branch\"; then git checkout \"$branch\"; "
            "else git checkout -B \"$branch\" \"$remote_sha\"; fi; fi"
            " && git reset --hard \"$remote_sha\""
            " && local_sha_after=\"$(git rev-parse HEAD 2>/dev/null || true)\""
            " && if [ \"$local_sha_before\" = \"$local_sha_after\" ]; then "
            "echo \"up-to-date: $local_sha_after\"; "
            "else echo \"updated: $local_sha_before -> $local_sha_after\"; fi"
        )
    else:
        sync_script = (
            f"cd {shlex.quote(remote_root)}"
            " && if ! command -v git >/dev/null 2>&1; then echo 'skip: git not found'; exit 0; fi"
            " && if [ ! -d .git ]; then echo 'skip: no .git folder'; exit 0; fi"
            " && branch=\"$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)\""
            " && if [ -z \"$branch\" ] || [ \"$branch\" = \"HEAD\" ]; then echo 'skip: detached/unknown branch'; exit 0; fi"
            " && if ! git remote get-url origin >/dev/null 2>&1; then echo 'skip: no origin remote'; exit 0; fi"
            " && if ! git diff --quiet --ignore-submodules -- || ! git diff --cached --quiet --ignore-submodules --; then "
            "echo 'skip: dirty worktree'; exit 0; fi"
            " && echo \"sync: checking origin/$branch\""
            " && if ! git fetch origin \"$branch\" --prune; then echo 'skip: fetch failed'; exit 0; fi"
            " && local_sha=\"$(git rev-parse HEAD 2>/dev/null || true)\""
            " && remote_sha=\"$(git rev-parse FETCH_HEAD 2>/dev/null || true)\""
            " && if [ -z \"$local_sha\" ] || [ -z \"$remote_sha\" ]; then echo 'skip: could not resolve commit shas'; exit 0; fi"
            " && if [ \"$local_sha\" = \"$remote_sha\" ]; then echo \"up-to-date: $local_sha\"; exit 0; fi"
            " && if git merge-base --is-ancestor \"$local_sha\" \"$remote_sha\"; then "
            "git reset --hard \"$remote_sha\" && echo \"updated: $local_sha -> $remote_sha\"; exit 0; fi"
            " && echo 'skip: local branch is ahead or diverged'; exit 0"
        )

    log("Remote git sync: checking for updates...")
    sync_result = ssh_run(
        cfg,
        sync_script,
        check=False,
        capture=True,
        log_command=False,
    )
    output = (sync_result.stdout or "").strip()
    output_lines: List[str] = []
    if output:
        for line in output.splitlines():
            line_clean = line.strip()
            if line_clean:
                output_lines.append(line_clean)
                log(f"[repo-sync] {line_clean}")
    has_skip_line = any(line.lower().startswith("skip:") for line in output_lines)
    has_fatal_line = any("fatal:" in line.lower() for line in output_lines)
    if requested_branch:
        if sync_result.returncode != 0 or has_skip_line or has_fatal_line:
            detail = "; ".join(output_lines[-4:]) if output_lines else f"exit={sync_result.returncode}"
            raise CloudCtlError(
                f"Remote git sync did not complete for requested branch '{requested_branch}'. "
                f"Details: {detail}. "
                "Push the branch/commits to origin and retry."
            )
    elif sync_result.returncode != 0:
        log(f"[repo-sync] warning: sync command exited with {sync_result.returncode}; continuing with current remote code.")


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
    wait_for_ssh_ready(
        cfg,
        timeout_sec=args.ssh_ready_timeout_sec,
        poll_sec=args.ssh_ready_poll_sec,
    )
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
        "-u",
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
        "--model-backend",
        args.model_backend,
        "--unet-path",
        args.unet_path,
        "--pretrain-path",
        args.pretrain_path,
        "--geometry-model-path",
        args.geometry_model_path,
        "--geometry-repo-path",
        args.geometry_repo_path,
        "--geometry-cache-dir",
        args.geometry_cache_dir,
        "--geometry-decode-chunk-size",
        str(max(1, int(args.geometry_decode_chunk_size))),
    ]

    if args.disable_xformers:
        runner_args.append("--disable-xformers")
    if args.use_cudnn_benchmark:
        runner_args.append("--use-cudnn-benchmark")
    if args.local_files_only:
        runner_args.append("--local-files-only")
    if bool(args.geometry_low_memory_usage):
        runner_args.append("--geometry-low-memory-usage")
    runner_args.append("--geometry-force-projection" if bool(args.geometry_force_projection) else "--no-geometry-force-projection")
    runner_args.append("--geometry-force-fixed-focal" if bool(args.geometry_force_fixed_focal) else "--no-geometry-force-fixed-focal")
    runner_args.append("--geometry-use-extract-interp" if bool(args.geometry_use_extract_interp) else "--no-geometry-use-extract-interp")

    quoted_runner = " ".join(shlex.quote(x) for x in runner_args)
    remote_root = args.remote_root.rstrip("/")
    return (
        f"cd {shlex.quote(remote_root)}"
        f" && source {shlex.quote(remote_join(remote_root, args.venv_name, 'bin', 'activate'))}"
        f" && {quoted_runner}"
    )


def _build_remote_batch_session_cmd(args: argparse.Namespace, remote_manifest_path: str) -> str:
    runner_args = [
        "python",
        "-u",
        "cloud/run_depth_batch_session.py",
        "--jobs-manifest",
        remote_manifest_path,
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
        "--model-backend",
        args.model_backend,
        "--unet-path",
        args.unet_path,
        "--pretrain-path",
        args.pretrain_path,
        "--geometry-model-path",
        args.geometry_model_path,
        "--geometry-repo-path",
        args.geometry_repo_path,
        "--geometry-cache-dir",
        args.geometry_cache_dir,
        "--geometry-decode-chunk-size",
        str(max(1, int(args.geometry_decode_chunk_size))),
    ]

    if args.disable_xformers:
        runner_args.append("--disable-xformers")
    if args.use_cudnn_benchmark:
        runner_args.append("--use-cudnn-benchmark")
    if args.local_files_only:
        runner_args.append("--local-files-only")
    if bool(args.geometry_low_memory_usage):
        runner_args.append("--geometry-low-memory-usage")
    if bool(getattr(args, "continue_on_error", False)):
        runner_args.append("--continue-on-error")
    runner_args.append("--geometry-force-projection" if bool(args.geometry_force_projection) else "--no-geometry-force-projection")
    runner_args.append("--geometry-force-fixed-focal" if bool(args.geometry_force_fixed_focal) else "--no-geometry-force-fixed-focal")
    runner_args.append("--geometry-use-extract-interp" if bool(args.geometry_use_extract_interp) else "--no-geometry-use-extract-interp")

    quoted_runner = " ".join(shlex.quote(x) for x in runner_args)
    remote_root = args.remote_root.rstrip("/")
    return (
        f"cd {shlex.quote(remote_root)}"
        f" && source {shlex.quote(remote_join(remote_root, args.venv_name, 'bin', 'activate'))}"
        f" && {quoted_runner}"
    )


def _normalize_queue_name(raw: str) -> str:
    value = sanitize_name(str(raw or "").strip())
    if not value:
        value = f"queue_{int(time.time())}"
    return value


def _remote_queue_paths(args: argparse.Namespace, queue_name: str) -> dict:
    remote_root = args.remote_root.rstrip("/")
    normalized = _normalize_queue_name(queue_name)
    queue_root = remote_join(remote_root, "cloud_jobs", "queues", normalized)
    return {
        "name": normalized,
        "root": queue_root,
        "pending": remote_join(queue_root, "pending"),
        "inprogress": remote_join(queue_root, "inprogress"),
        "done": remote_join(queue_root, "done"),
        "failed": remote_join(queue_root, "failed"),
        "control": remote_join(queue_root, "control"),
        "stop_file": remote_join(queue_root, "control", "stop"),
        "worker_pid": remote_join(queue_root, "worker.pid"),
        "worker_state": remote_join(queue_root, "worker_state.json"),
        "worker_log": remote_join(queue_root, "worker.log"),
        "remote_input_dir": remote_join(remote_root, args.remote_input_subdir),
        "remote_output_dir": remote_join(remote_root, args.remote_output_subdir),
    }


def _build_remote_queue_worker_cmd(args: argparse.Namespace, queue_root: str) -> str:
    runner_args = [
        "python",
        "-u",
        "cloud/run_depth_queue_worker.py",
        "--queue-dir",
        queue_root,
        "--poll-interval-sec",
        str(max(0.1, float(getattr(args, "queue_poll_sec", 1.0) or 1.0))),
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
        "--model-backend",
        args.model_backend,
        "--unet-path",
        args.unet_path,
        "--pretrain-path",
        args.pretrain_path,
        "--geometry-model-path",
        args.geometry_model_path,
        "--geometry-repo-path",
        args.geometry_repo_path,
        "--geometry-cache-dir",
        args.geometry_cache_dir,
        "--geometry-decode-chunk-size",
        str(max(1, int(args.geometry_decode_chunk_size))),
    ]

    if args.disable_xformers:
        runner_args.append("--disable-xformers")
    if args.use_cudnn_benchmark:
        runner_args.append("--use-cudnn-benchmark")
    if args.local_files_only:
        runner_args.append("--local-files-only")
    if bool(args.geometry_low_memory_usage):
        runner_args.append("--geometry-low-memory-usage")
    if bool(getattr(args, "continue_on_error", False)):
        runner_args.append("--continue-on-error")
    runner_args.append("--geometry-force-projection" if bool(args.geometry_force_projection) else "--no-geometry-force-projection")
    runner_args.append("--geometry-force-fixed-focal" if bool(args.geometry_force_fixed_focal) else "--no-geometry-force-fixed-focal")
    runner_args.append("--geometry-use-extract-interp" if bool(args.geometry_use_extract_interp) else "--no-geometry-use-extract-interp")

    quoted_runner = " ".join(shlex.quote(x) for x in runner_args)
    remote_root = args.remote_root.rstrip("/")
    return (
        f"cd {shlex.quote(remote_root)}"
        f" && source {shlex.quote(remote_join(remote_root, args.venv_name, 'bin', 'activate'))}"
        f" && {quoted_runner}"
    )


def _remote_worker_running(cfg: SSHConfig, queue_paths: dict) -> bool:
    pid_file = queue_paths["worker_pid"]
    proc = ssh_run(
        cfg,
        f"if [ -f {shlex.quote(pid_file)} ] && kill -0 \"$(cat {shlex.quote(pid_file)})\" 2>/dev/null; then echo RUNNING; else echo STOPPED; fi",
        check=False,
        capture=True,
        log_command=False,
    )
    text = (proc.stdout or "").strip().upper()
    return "RUNNING" in text


def _ensure_remote_queue_dirs(cfg: SSHConfig, queue_paths: dict) -> None:
    ssh_run(
        cfg,
        " && ".join(
            [
                f"mkdir -p {shlex.quote(queue_paths['root'])}",
                f"mkdir -p {shlex.quote(queue_paths['pending'])}",
                f"mkdir -p {shlex.quote(queue_paths['inprogress'])}",
                f"mkdir -p {shlex.quote(queue_paths['done'])}",
                f"mkdir -p {shlex.quote(queue_paths['failed'])}",
                f"mkdir -p {shlex.quote(queue_paths['control'])}",
                f"mkdir -p {shlex.quote(queue_paths['remote_input_dir'])}",
                f"mkdir -p {shlex.quote(queue_paths['remote_output_dir'])}",
            ]
        ),
    )


def _start_remote_queue_worker(cfg: SSHConfig, args: argparse.Namespace, queue_paths: dict) -> None:
    _ensure_remote_queue_dirs(cfg, queue_paths)
    if _remote_worker_running(cfg, queue_paths):
        log(f"Queue worker already running for '{queue_paths['name']}'.")
        return

    remote_cmd = _build_remote_queue_worker_cmd(args, queue_paths["root"])
    launch_script = " && ".join(
        [
            f"rm -f {shlex.quote(queue_paths['stop_file'])}",
            f"rm -f {shlex.quote(queue_paths['worker_state'])}",
            f"nohup bash -lc {shlex.quote(remote_cmd)} > {shlex.quote(queue_paths['worker_log'])} 2>&1 < /dev/null & echo $! > {shlex.quote(queue_paths['worker_pid'])}",
        ]
    )
    ssh_run(cfg, launch_script)

    poll_deadline = time.time() + 60.0
    next_wait_log = time.time() + 5.0
    while time.time() < poll_deadline:
        if _remote_worker_running(cfg, queue_paths):
            log(f"Queue worker started for '{queue_paths['name']}'.")
            return
        now = time.time()
        if now >= next_wait_log:
            elapsed = max(0.0, 60.0 - (poll_deadline - now))
            log(
                f"Waiting for queue worker startup '{queue_paths['name']}' "
                f"(elapsed {elapsed:.0f}s/60s)..."
            )
            next_wait_log = now + 5.0
        time.sleep(0.5)
    log_tail_proc = ssh_run(
        cfg,
        f"tail -n 80 {shlex.quote(queue_paths['worker_log'])}",
        check=False,
        capture=True,
        log_command=False,
    )
    log_tail_text = (log_tail_proc.stdout or "").strip()
    if log_tail_text:
        for line in log_tail_text.splitlines()[-80:]:
            text = line.strip()
            if text:
                log(f"[queue-start][remote] {text}")
    raise CloudCtlError(f"Queue worker failed to start for '{queue_paths['name']}'.")


def _stop_remote_queue_worker(
    cfg: SSHConfig,
    queue_paths: dict,
    *,
    wait_timeout_sec: float = 60.0,
) -> None:
    ssh_run(
        cfg,
        f"mkdir -p {shlex.quote(queue_paths['control'])} && touch {shlex.quote(queue_paths['stop_file'])}",
        check=False,
    )

    deadline = time.time() + max(0.0, float(wait_timeout_sec))
    while time.time() < deadline:
        if not _remote_worker_running(cfg, queue_paths):
            log(f"Queue worker stopped for '{queue_paths['name']}'.")
            return
        time.sleep(0.5)
    log(f"Queue worker stop timeout for '{queue_paths['name']}' (still running).")


def _enqueue_remote_queue_job(cfg: SSHConfig, queue_paths: dict, *, job_id: str, payload: dict) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=f"queue_job_{sanitize_name(job_id)}_",
        delete=False,
    ) as tmp:
        tmp.write(json.dumps(payload, indent=2, sort_keys=True))
        local_payload_path = Path(tmp.name)

    remote_payload_path = remote_join(queue_paths["pending"], f"{sanitize_name(job_id)}.json")
    try:
        rsync_to_remote(cfg, str(local_payload_path), remote_payload_path)
    finally:
        try:
            local_payload_path.unlink(missing_ok=True)
        except Exception:
            pass
    return remote_payload_path


def _wait_for_remote_queue_job_result(
    cfg: SSHConfig,
    queue_paths: dict,
    *,
    job_id: str,
    poll_sec: float,
) -> tuple[str, str]:
    safe_job_id = sanitize_name(job_id)
    done_path = remote_join(queue_paths["done"], f"{safe_job_id}.json")
    failed_path = remote_join(queue_paths["failed"], f"{safe_job_id}.json")
    poll = max(0.25, float(poll_sec))
    while True:
        probe = ssh_run(
            cfg,
            (
                f"if [ -f {shlex.quote(done_path)} ]; then echo DONE; "
                f"elif [ -f {shlex.quote(failed_path)} ]; then echo FAILED; "
                "else echo WAIT; fi"
            ),
            check=False,
            capture=True,
            log_command=False,
        )
        text = (probe.stdout or "").strip()
        if text == "DONE":
            return "done", done_path
        if text == "FAILED":
            return "failed", failed_path
        time.sleep(poll)


def _read_remote_json(cfg: SSHConfig, remote_path: str) -> dict | None:
    proc = ssh_run(
        cfg,
        f"cat {shlex.quote(remote_path)}",
        check=False,
        capture=True,
        log_command=False,
    )
    if proc.returncode != 0:
        return None
    text = (proc.stdout or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _download_job_outputs_and_log_stats(
    cfg: SSHConfig,
    args: argparse.Namespace,
    *,
    job_name: str,
    remote_job_output_dir: str,
    log_prefix: str,
) -> dict | None:
    download_root = Path(args.download_dir).expanduser().resolve()
    download_into_job_subdir = bool(getattr(args, "download_into_job_subdir", False))
    local_job_dir = download_root / job_name if download_into_job_subdir else download_root
    status_payload = None
    if not args.skip_download:
        local_job_dir.mkdir(parents=True, exist_ok=True)
        log(f"{log_prefix}Downloading outputs to: {local_job_dir}")
        rsync_from_remote(cfg, f"{remote_job_output_dir}/", f"{local_job_dir}/")
        status_path = local_job_dir / "job_status.json"
        status_alias_path = local_job_dir / f"{job_name}_job_status.json"
        if status_path.is_file() and status_alias_path != status_path:
            try:
                status_alias_path.write_text(status_path.read_text(encoding="utf-8"), encoding="utf-8")
                if not download_into_job_subdir:
                    status_path.unlink(missing_ok=True)
                    status_path = status_alias_path
            except Exception as status_copy_exc:  # pylint: disable=broad-except
                log(f"{log_prefix}Warning: could not materialize per-job status alias: {status_copy_exc}")
        if status_path.is_file():
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception as status_exc:  # pylint: disable=broad-except
                log(f"{log_prefix}Warning: could not parse job_status.json for VRAM summary: {status_exc}")
            else:
                if isinstance(payload, dict):
                    status_payload = payload
                    logged_gpu_stats = False
                    gpu_memory_smi = payload.get("gpu_memory_nvidia_smi", {})
                    if isinstance(gpu_memory_smi, dict) and int(gpu_memory_smi.get("device_count", 0) or 0) > 0:
                        peak_used = float(gpu_memory_smi.get("peak_used_mib_all_devices", 0.0) or 0.0)
                        peak_used_pct = float(gpu_memory_smi.get("peak_used_pct_total_all_devices", 0.0) or 0.0)
                        sample_count = int(gpu_memory_smi.get("sample_count", 0) or 0)
                        sample_interval = float(gpu_memory_smi.get("sample_interval_sec", 0.0) or 0.0)
                        device_count = int(gpu_memory_smi.get("device_count", 0) or 0)
                        log(
                            f"{log_prefix}GPU peak VRAM (nvidia-smi) | used={peak_used:.1f} MiB ({peak_used_pct:.2f}%), "
                            f"devices={device_count}, samples={sample_count}, dt={sample_interval:.2f}s"
                        )
                        logged_gpu_stats = True
                        per_device_smi = gpu_memory_smi.get("per_device", [])
                        if isinstance(per_device_smi, list):
                            for device_entry in per_device_smi:
                                if not isinstance(device_entry, dict):
                                    continue
                                log(
                                    f"{log_prefix}GPU{int(device_entry.get('index', 0) or 0)} peak (nvidia-smi) | "
                                    f"used={float(device_entry.get('peak_used_mib', 0.0) or 0.0):.1f} MiB, "
                                    f"total={float(device_entry.get('total_mib', 0.0) or 0.0):.1f} MiB, "
                                    f"util_peak={float(device_entry.get('peak_util_gpu_pct', 0.0) or 0.0):.1f}%, "
                                    f"name={str(device_entry.get('name', ''))}"
                                )

                    gpu_memory = payload.get("gpu_memory", {})
                    if isinstance(gpu_memory, dict):
                        peak_alloc = float(gpu_memory.get("peak_alloc_mib_all_devices", 0.0) or 0.0)
                        peak_reserved = float(gpu_memory.get("peak_reserved_mib_all_devices", 0.0) or 0.0)
                        peak_alloc_pct = float(gpu_memory.get("peak_alloc_pct_total_all_devices", 0.0) or 0.0)
                        peak_reserved_pct = float(gpu_memory.get("peak_reserved_pct_total_all_devices", 0.0) or 0.0)
                        device_count = int(gpu_memory.get("device_count", 0) or 0)
                        if device_count > 0:
                            log(
                                f"{log_prefix}GPU peak VRAM (torch) | alloc={peak_alloc:.1f} MiB ({peak_alloc_pct:.2f}%), "
                                f"reserved={peak_reserved:.1f} MiB ({peak_reserved_pct:.2f}%), devices={device_count}"
                            )
                            logged_gpu_stats = True
                            per_device = gpu_memory.get("per_device", [])
                            if isinstance(per_device, list):
                                for device_entry in per_device:
                                    if not isinstance(device_entry, dict):
                                        continue
                                    log(
                                        f"{log_prefix}GPU{int(device_entry.get('index', 0) or 0)} peak (torch) | "
                                        f"alloc={float(device_entry.get('peak_alloc_mib', 0.0) or 0.0):.1f} MiB, "
                                        f"reserved={float(device_entry.get('peak_reserved_mib', 0.0) or 0.0):.1f} MiB, "
                                        f"total={float(device_entry.get('total_mib', 0.0) or 0.0):.1f} MiB, "
                                        f"name={str(device_entry.get('name', ''))}"
                                    )
                    if not logged_gpu_stats:
                        log(f"{log_prefix}GPU peak VRAM stats were not present in job status JSON.")
    return status_payload


def _cleanup_remote_job_artifacts(
    cfg: SSHConfig,
    args: argparse.Namespace,
    *,
    remote_input_path: str,
    remote_job_output_dir: str,
) -> None:
    if not args.keep_remote_input:
        ssh_run(cfg, f"rm -f {shlex.quote(remote_input_path)}", check=False, log_command=False)
    if not args.keep_remote_output:
        ssh_run(cfg, f"rm -rf {shlex.quote(remote_job_output_dir)}", check=False, log_command=False)


def _run_one_job(
    cfg: SSHConfig,
    args: argparse.Namespace,
    local_input: Path,
    *,
    explicit_job_name: str = "",
    batch_index: int = 0,
    batch_total: int = 0,
    skip_upload_if_present: bool = False,
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

    remote_input_dir, remote_output_dir, remote_job_output_dir, remote_input_path = _remote_job_paths(
        args,
        job_name,
        local_input.suffix,
    )

    log_prefix = f"[{batch_index}/{batch_total}] " if batch_total > 1 else ""
    log(f"{log_prefix}Preparing job {job_name}")

    result = None
    caught_exception = None
    try:
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

        should_upload = True
        if skip_upload_if_present:
            check_proc = ssh_run(
                cfg,
                f"test -f {shlex.quote(remote_input_path)}",
                check=False,
                capture=False,
                log_command=False,
            )
            if check_proc.returncode == 0:
                should_upload = False
                log(f"{log_prefix}Remote clip already staged: {remote_input_path} (skip upload)")
        if should_upload:
            log(f"{log_prefix}Uploading clip: {local_input}")
            rsync_to_remote(cfg, str(local_input), remote_input_path)

        remote_cmd = _build_remote_job_cmd(args, remote_input_path, remote_job_output_dir, job_name)
        log(f"{log_prefix}Running remote inference...")
        result = ssh_run_stream(
            cfg,
            remote_cmd,
            check=False,
            line_prefix=f"{log_prefix}[remote] ",
            heartbeat_sec=30,
        )

        _download_job_outputs_and_log_stats(
            cfg,
            args,
            job_name=job_name,
            remote_job_output_dir=remote_job_output_dir,
            log_prefix=log_prefix,
        )
    except Exception as exc:  # pylint: disable=broad-except
        caught_exception = exc
    finally:
        _cleanup_remote_job_artifacts(
            cfg,
            args,
            remote_input_path=remote_input_path,
            remote_job_output_dir=remote_job_output_dir,
        )

    if caught_exception is not None:
        raise caught_exception

    if result is None:
        raise CloudCtlError(f"Remote job did not produce a result for {local_input.name}")
    if result.returncode != 0:
        raise CloudCtlError(f"Remote job failed (exit code {result.returncode}) for {local_input.name}")

    log(f"{log_prefix}Job complete: {job_name}")
    return 0


def cmd_run_job(args: argparse.Namespace) -> int:
    require_tool("ssh")
    require_tool("rsync")
    cfg = cfg_from_args(args)
    wait_for_ssh_ready(
        cfg,
        timeout_sec=args.ssh_ready_timeout_sec,
        poll_sec=args.ssh_ready_poll_sec,
    )
    maybe_sync_remote_repo(cfg, args)
    local_input = Path(args.local_input).expanduser().resolve()
    if bool(getattr(args, "queue_mode", True)):
        if not local_input.exists():
            raise CloudCtlError(f"Input clip not found: {local_input}")
        if not hasattr(args, "continue_on_error"):
            setattr(args, "continue_on_error", False)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        explicit_name = str(args.job_name or "").strip()
        if explicit_name:
            job_name = sanitize_name(explicit_name)
        else:
            job_name = sanitize_name(f"{args.job_prefix}{sanitize_name(local_input.stem)}_{stamp}")
        return _run_batch_with_worker_queue(cfg, args, [(1, local_input, job_name)])
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
    for p_str in sort_paths_by_clip_id(str(p) for p in found):
        p = Path(p_str)
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        ordered_unique.append(p)
    return ordered_unique


def _load_batch_manifest_inputs(manifest_path: Path) -> List[tuple[str, Path]]:
    """Read newline manifest with either '<path>' or '<job_name><TAB><path>' entries."""
    if not manifest_path.is_file():
        raise CloudCtlError(f"Manifest file not found: {manifest_path}")
    manifest_dir = manifest_path.parent
    jobs: List[tuple[str, Path]] = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            explicit_name, path_text = line.split("\t", 1)
            job_name = sanitize_name(explicit_name.strip())
        elif "|" in line:
            explicit_name, path_text = line.split("|", 1)
            job_name = sanitize_name(explicit_name.strip())
        else:
            job_name = ""
            path_text = line
        clip_path = Path(path_text.strip()).expanduser()
        if not clip_path.is_absolute():
            clip_path = (manifest_dir / clip_path)
        clip_path = clip_path.resolve()
        if not clip_path.is_file():
            raise CloudCtlError(f"Manifest clip path not found: {clip_path}")
        jobs.append((job_name, clip_path))

    # If no explicit names are provided, enforce deterministic clip-id ordering.
    if jobs and all(not name for name, _ in jobs):
        ordered_paths = [Path(p) for p in sort_paths_by_clip_id(path for _, path in jobs)]
        return [("", p) for p in ordered_paths]
    # Otherwise respect manifest order exactly.
    return jobs


def _build_batch_job_name(
    args: argparse.Namespace,
    clip: Path,
    *,
    batch_index: int,
    batch_total: int,
    batch_stamp: str,
    explicit_name: str = "",
) -> str:
    if explicit_name:
        return sanitize_name(explicit_name)
    stem = sanitize_name(clip.stem)
    if batch_total > 1:
        return sanitize_name(f"{args.job_prefix}{batch_index:03d}_{stem}_{batch_stamp}")
    return sanitize_name(f"{args.job_prefix}{stem}_{batch_stamp}")


def _remote_job_paths(args: argparse.Namespace, job_name: str, clip_suffix: str) -> tuple[str, str, str, str]:
    remote_root = args.remote_root.rstrip("/")
    remote_input_dir = remote_join(remote_root, args.remote_input_subdir)
    remote_output_dir = remote_join(remote_root, args.remote_output_subdir)
    remote_job_output_dir = remote_join(remote_output_dir, job_name)
    remote_input_filename = f"{job_name}{clip_suffix.lower()}"
    remote_input_path = remote_join(remote_input_dir, remote_input_filename)
    return remote_input_dir, remote_output_dir, remote_job_output_dir, remote_input_path


def _run_batch_with_persistent_sessions(
    cfg: SSHConfig,
    args: argparse.Namespace,
    plans: list[tuple[int, Path, str]],
    *,
    started_job_names: set[str],
    failed: list[str],
) -> None:
    session_size = max(2, int(getattr(args, "persistent_session_size", 8) or 8))
    if bool(getattr(args, "prefetch_upload_all", False)) or int(getattr(args, "prefetch_window", 0) or 0) > 0:
        log("Persistent sessions enabled: ignoring prefetch flags for this batch mode.")

    remote_root = args.remote_root.rstrip("/")
    remote_input_dir = remote_join(remote_root, args.remote_input_subdir)
    remote_output_dir = remote_join(remote_root, args.remote_output_subdir)
    remote_manifest_dir = remote_join(remote_output_dir, "_session_manifests")
    ssh_run(
        cfg,
        " && ".join(
            [
                f"mkdir -p {shlex.quote(remote_input_dir)}",
                f"mkdir -p {shlex.quote(remote_output_dir)}",
                f"mkdir -p {shlex.quote(remote_manifest_dir)}",
            ]
        ),
    )

    total = len(plans)
    for chunk_start in range(0, total, session_size):
        chunk = plans[chunk_start : chunk_start + session_size]
        if not chunk:
            continue
        first_idx = chunk[0][0]
        last_idx = chunk[-1][0]
        log(f"Persistent session chunk [{first_idx}-{last_idx}/{total}] starting.")

        runnable: list[tuple[int, Path, str]] = []
        for idx, clip, job_name in chunk:
            _, _, remote_job_output_dir, remote_input_path = _remote_job_paths(args, job_name, clip.suffix)
            try:
                ssh_run(
                    cfg,
                    f"mkdir -p {shlex.quote(remote_job_output_dir)}",
                    check=True,
                    log_command=False,
                )
                log(f"[{idx}/{total}] Uploading clip: {clip}")
                rsync_to_remote(cfg, str(clip), remote_input_path)
                started_job_names.add(job_name)
                runnable.append((idx, clip, job_name))
            except Exception as exc:  # pylint: disable=broad-except
                msg = f"Upload failed for {clip.name}: {exc}"
                log(msg)
                failed.append(msg)
                if not args.continue_on_error:
                    return

        if not runnable:
            continue

        manifest_rows = []
        for _, clip, job_name in runnable:
            _, _, remote_job_output_dir, remote_input_path = _remote_job_paths(args, job_name, clip.suffix)
            manifest_rows.append(
                {
                    "input": remote_input_path,
                    "output_dir": remote_job_output_dir,
                    "status_json": remote_join(remote_job_output_dir, "job_status.json"),
                    "job_name": job_name,
                }
            )

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix="cloud_batch_manifest_",
            delete=False,
        ) as manifest_tmp:
            manifest_tmp.write(json.dumps(manifest_rows, indent=2))
            local_manifest_path = Path(manifest_tmp.name)

        remote_manifest_path = remote_join(
            remote_manifest_dir,
            f"session_{first_idx:05d}_{last_idx:05d}_{int(time.time() * 1000)}.json",
        )

        result = None
        try:
            rsync_to_remote(cfg, str(local_manifest_path), remote_manifest_path)
            remote_cmd = _build_remote_batch_session_cmd(args, remote_manifest_path)
            result = ssh_run_stream(
                cfg,
                remote_cmd,
                check=False,
                line_prefix=f"[{first_idx}-{last_idx}/{total}] [remote] ",
                heartbeat_sec=30,
            )
        finally:
            try:
                local_manifest_path.unlink(missing_ok=True)
            except Exception:
                pass
            ssh_run(cfg, f"rm -f {shlex.quote(remote_manifest_path)}", check=False, log_command=False)

        chunk_failed = False
        for idx, clip, job_name in runnable:
            _, _, remote_job_output_dir, remote_input_path = _remote_job_paths(args, job_name, clip.suffix)
            status_payload = _download_job_outputs_and_log_stats(
                cfg,
                args,
                job_name=job_name,
                remote_job_output_dir=remote_job_output_dir,
                log_prefix=f"[{idx}/{total}] ",
            )
            _cleanup_remote_job_artifacts(
                cfg,
                args,
                remote_input_path=remote_input_path,
                remote_job_output_dir=remote_job_output_dir,
            )

            if isinstance(status_payload, dict):
                job_status = str(status_payload.get("status", "")).strip().lower()
                if job_status != "success":
                    msg = f"Batch item failed for {clip.name}: status={job_status or 'unknown'}"
                    log(msg)
                    failed.append(msg)
                    chunk_failed = True
            elif result is not None and result.returncode != 0:
                msg = f"Batch item failed for {clip.name}: missing status JSON and session exit code {result.returncode}"
                log(msg)
                failed.append(msg)
                chunk_failed = True

        if result is not None and result.returncode != 0:
            chunk_failed = True
            msg = f"Persistent session chunk [{first_idx}-{last_idx}/{total}] failed with exit code {result.returncode}."
            log(msg)
            failed.append(msg)

        if chunk_failed and not args.continue_on_error:
            return


def _run_batch_with_worker_queue(
    cfg: SSHConfig,
    args: argparse.Namespace,
    plans: list[tuple[int, Path, str]],
) -> int:
    queue_name = str(getattr(args, "queue_name", "") or "").strip()
    if not queue_name:
        queue_name = f"batch_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    queue_paths = _remote_queue_paths(args, queue_name)

    log(
        f"Queue mode enabled. queue={queue_paths['name']} "
        f"(poll={max(0.25, float(getattr(args, 'queue_poll_sec', 1.0) or 1.0)):.2f}s)"
    )
    if bool(getattr(args, "prefetch_upload_all", False)) or int(getattr(args, "prefetch_window", 0) or 0) > 0:
        log("Queue mode: ignoring prefetch flags.")
    if bool(getattr(args, "persistent_session", True)):
        log("Queue mode: persistent session chunking is bypassed.")

    _start_remote_queue_worker(cfg, args, queue_paths)

    failed: list[str] = []
    total = len(plans)
    enqueued: list[dict] = []
    log_tailer: RemoteLogTailer | None = None
    try:
        if bool(getattr(args, "queue_stream_logs", True)):
            log_tailer = _start_remote_log_tailer(
                cfg,
                queue_paths["worker_log"],
                line_prefix="[queue][remote] ",
                log_command=False,
            )

        for idx, clip, job_name in plans:
            _, _, remote_job_output_dir, remote_input_path = _remote_job_paths(args, job_name, clip.suffix)
            ssh_run(
                cfg,
                f"mkdir -p {shlex.quote(remote_job_output_dir)}",
                check=True,
                log_command=False,
            )
            log(f"[{idx}/{total}] Uploading clip: {clip}")
            rsync_to_remote(cfg, str(clip), remote_input_path)

            job_id = sanitize_name(f"{idx:05d}_{job_name}")
            queue_payload = {
                "job_id": job_id,
                "job_name": job_name,
                "input": remote_input_path,
                "output_dir": remote_job_output_dir,
                "status_json": remote_join(remote_job_output_dir, "job_status.json"),
            }
            _enqueue_remote_queue_job(cfg, queue_paths, job_id=job_id, payload=queue_payload)
            enqueued.append(
                {
                    "idx": idx,
                    "clip": clip,
                    "job_name": job_name,
                    "job_id": job_id,
                    "remote_input_path": remote_input_path,
                    "remote_job_output_dir": remote_job_output_dir,
                }
            )
            log(f"[{idx}/{total}] Enqueued job: {job_name} ({job_id})")

        poll_sec = max(0.25, float(getattr(args, "queue_poll_sec", 1.0) or 1.0))
        for item in enqueued:
            idx = int(item["idx"])
            clip = item["clip"]
            job_name = str(item["job_name"])
            job_id = str(item["job_id"])
            remote_input_path = str(item["remote_input_path"])
            remote_job_output_dir = str(item["remote_job_output_dir"])

            result_kind, remote_result_path = _wait_for_remote_queue_job_result(
                cfg,
                queue_paths,
                job_id=job_id,
                poll_sec=poll_sec,
            )
            result_payload = _read_remote_json(cfg, remote_result_path)
            if result_kind == "failed":
                status_message = ""
                if isinstance(result_payload, dict):
                    status_message = str(result_payload.get("status_message", "")).strip()
                msg = f"Batch item failed for {clip.name}: {status_message or 'queue worker reported failure'}"
                log(msg)
                failed.append(msg)
                if not bool(getattr(args, "continue_on_error", False)):
                    break

            status_payload = _download_job_outputs_and_log_stats(
                cfg,
                args,
                job_name=job_name,
                remote_job_output_dir=remote_job_output_dir,
                log_prefix=f"[{idx}/{total}] ",
            )

            if isinstance(status_payload, dict):
                status_value = str(status_payload.get("status", "")).strip().lower()
                if status_value != "success":
                    msg = f"Batch item failed for {clip.name}: status={status_value or 'unknown'}"
                    log(msg)
                    failed.append(msg)
                    if not bool(getattr(args, "continue_on_error", False)):
                        break

            _cleanup_remote_job_artifacts(
                cfg,
                args,
                remote_input_path=remote_input_path,
                remote_job_output_dir=remote_job_output_dir,
            )
            ssh_run(cfg, f"rm -f {shlex.quote(remote_result_path)}", check=False, log_command=False)

            log(f"[{idx}/{total}] Job complete: {job_name}")

    finally:
        if not bool(getattr(args, "keep_queue_worker", False)):
            _stop_remote_queue_worker(cfg, queue_paths, wait_timeout_sec=60.0)
        else:
            log(f"Leaving queue worker running for '{queue_paths['name']}'.")
        _stop_remote_log_tailer(log_tailer)

    if failed:
        for item in failed:
            log(f"ERROR: {item}")
        return 1

    log("Batch complete.")
    return 0


def cmd_run_batch(args: argparse.Namespace) -> int:
    require_tool("ssh")
    require_tool("rsync")

    cfg = cfg_from_args(args)
    wait_for_ssh_ready(
        cfg,
        timeout_sec=args.ssh_ready_timeout_sec,
        poll_sec=args.ssh_ready_poll_sec,
    )
    maybe_sync_remote_repo(cfg, args)
    clips_with_names: List[tuple[str, Path]]
    input_manifest_spec = str(getattr(args, "input_manifest", "") or "").strip()
    input_dir_spec = str(getattr(args, "input_dir", "") or "").strip()
    if not input_manifest_spec and not input_dir_spec:
        raise CloudCtlError("run-batch requires either --input-manifest or --input-dir.")

    if input_manifest_spec:
        manifest_path = Path(args.input_manifest).expanduser().resolve()
        clips_with_names = _load_batch_manifest_inputs(manifest_path)
    else:
        input_dir = Path(args.input_dir).expanduser().resolve()
        if not input_dir.exists():
            raise CloudCtlError(f"Input dir does not exist: {input_dir}")
        patterns = [p for p in args.patterns.split(",") if p.strip()]
        discovered = _discover_batch_inputs(input_dir, patterns, bool(args.recursive))
        clips_with_names = [("", clip) for clip in discovered]

    clips = [clip for _, clip in clips_with_names]
    if args.max_jobs > 0:
        clips_with_names = clips_with_names[: args.max_jobs]
        clips = [clip for _, clip in clips_with_names]

    if not clips:
        raise CloudCtlError("No clips matched the requested patterns.")

    log(f"Found {len(clips)} clip(s) for batch processing.")

    batch_stamp = time.strftime("%Y%m%d_%H%M%S")
    plans = []
    for idx, (explicit_name, clip) in enumerate(clips_with_names, start=1):
        job_name = _build_batch_job_name(
            args,
            clip,
            batch_index=idx,
            batch_total=len(clips_with_names),
            batch_stamp=batch_stamp,
            explicit_name=explicit_name,
        )
        plans.append((idx, clip, job_name))

    if bool(getattr(args, "queue_mode", True)):
        return _run_batch_with_worker_queue(cfg, args, plans)

    failed = []
    started_job_names = set()
    persistent_enabled = (
        bool(getattr(args, "persistent_session", True))
        and len(plans) > 1
        and int(getattr(args, "persistent_session_size", 8) or 8) > 1
    )
    if persistent_enabled:
        log(
            f"Persistent session mode enabled (chunk size={max(2, int(getattr(args, 'persistent_session_size', 8) or 8))}). "
            "Models will stay loaded across chunked jobs."
        )

    if (not persistent_enabled) and bool(getattr(args, "prefetch_upload_all", False)) and len(plans) > 1:
        remote_root = args.remote_root.rstrip("/")
        remote_input_dir = remote_join(remote_root, args.remote_input_subdir)
        remote_output_dir = remote_join(remote_root, args.remote_output_subdir)
        ssh_run(
            cfg,
            " && ".join(
                [
                    f"mkdir -p {shlex.quote(remote_input_dir)}",
                    f"mkdir -p {shlex.quote(remote_output_dir)}",
                ]
            ),
        )
        log("Prefetch mode: staging all clips to remote input before inference...")
        for idx, clip, job_name in plans:
            _, _, _, remote_input_path = _remote_job_paths(args, job_name, clip.suffix)
            log(f"[{idx}/{len(plans)}] Prefetch upload: {clip.name}")
            try:
                rsync_to_remote(cfg, str(clip), remote_input_path)
            except Exception as exc:  # pylint: disable=broad-except
                msg = f"Prefetch upload failed for {clip.name}: {exc}"
                log(msg)
                failed.append(msg)
                if not args.continue_on_error:
                    break
    prefetch_window = 0 if persistent_enabled else max(0, int(getattr(args, "prefetch_window", 0) or 0))
    pipeline_enabled = (
        prefetch_window > 0
        and len(plans) > 1
        and not bool(getattr(args, "prefetch_upload_all", False))
        and (not persistent_enabled)
    )

    if persistent_enabled:
        _run_batch_with_persistent_sessions(
            cfg,
            args,
            plans,
            started_job_names=started_job_names,
            failed=failed,
        )
    elif pipeline_enabled:
        remote_root = args.remote_root.rstrip("/")
        remote_input_dir = remote_join(remote_root, args.remote_input_subdir)
        remote_output_dir = remote_join(remote_root, args.remote_output_subdir)
        ssh_run(
            cfg,
            " && ".join(
                [
                    f"mkdir -p {shlex.quote(remote_input_dir)}",
                    f"mkdir -p {shlex.quote(remote_output_dir)}",
                ]
            ),
        )
        log(
            f"Pipeline mode: keeping up to {prefetch_window} additional clip(s) staged ahead of inference."
        )

        staged_jobs = set()
        staged_failures = {}
        next_upload_idx = 0
        started_count = 0
        stop_upload = False
        cond = threading.Condition()

        def _uploader() -> None:
            nonlocal next_upload_idx
            nonlocal stop_upload
            while True:
                with cond:
                    while (
                        not stop_upload
                        and next_upload_idx < len(plans)
                        and (next_upload_idx - started_count) >= (prefetch_window + 1)
                    ):
                        cond.wait(timeout=0.25)
                    if stop_upload or next_upload_idx >= len(plans):
                        return
                    idx, clip, job_name = plans[next_upload_idx]
                    next_upload_idx += 1

                try:
                    _, _, _, remote_input_path = _remote_job_paths(args, job_name, clip.suffix)
                    log(f"[{idx}/{len(plans)}] Prefetch upload: {clip.name}")
                    rsync_to_remote(cfg, str(clip), remote_input_path)
                    with cond:
                        staged_jobs.add(job_name)
                        cond.notify_all()
                except Exception as exc:  # pylint: disable=broad-except
                    err = f"Prefetch upload failed for {clip.name}: {exc}"
                    with cond:
                        staged_failures[job_name] = err
                        cond.notify_all()
                    if not args.continue_on_error:
                        with cond:
                            stop_upload = True
                            cond.notify_all()
                        return

        uploader_thread = threading.Thread(target=_uploader, daemon=True)
        uploader_thread.start()

        try:
            for idx, clip, job_name in plans:
                with cond:
                    while job_name not in staged_jobs and job_name not in staged_failures and not stop_upload:
                        cond.wait(timeout=0.25)
                    upload_error = staged_failures.get(job_name)
                    started_count += 1
                    cond.notify_all()

                if upload_error:
                    log(upload_error)
                    failed.append(upload_error)
                    if not args.continue_on_error:
                        break
                    continue

                try:
                    started_job_names.add(job_name)
                    _run_one_job(
                        cfg,
                        args,
                        clip,
                        explicit_job_name=job_name,
                        batch_index=idx,
                        batch_total=len(plans),
                        skip_upload_if_present=True,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    msg = f"Batch item failed for {clip.name}: {exc}"
                    log(msg)
                    failed.append(msg)
                    if not args.continue_on_error:
                        break
        finally:
            with cond:
                stop_upload = True
                cond.notify_all()
            uploader_thread.join(timeout=10.0)
    else:
        for idx, clip, job_name in plans:
            try:
                started_job_names.add(job_name)
                _run_one_job(
                    cfg,
                    args,
                    clip,
                    explicit_job_name=job_name,
                    batch_index=idx,
                    batch_total=len(plans),
                    skip_upload_if_present=bool(getattr(args, "prefetch_upload_all", False)),
                )
            except Exception as exc:  # pylint: disable=broad-except
                msg = f"Batch item failed for {clip.name}: {exc}"
                log(msg)
                failed.append(msg)
                if not args.continue_on_error:
                    break

    if (not args.keep_remote_input) or (not args.keep_remote_output):
        unstarted = [(idx, clip, job_name) for idx, clip, job_name in plans if job_name not in started_job_names]
        if unstarted:
            log(f"Remote cleanup: removing artifacts for {len(unstarted)} unstarted prefetched job(s).")
            for idx, clip, job_name in unstarted:
                _, _, remote_job_output_dir, remote_input_path = _remote_job_paths(args, job_name, clip.suffix)
                if not args.keep_remote_input:
                    ssh_run(cfg, f"rm -f {shlex.quote(remote_input_path)}", check=False, log_command=False)
                if not args.keep_remote_output:
                    ssh_run(cfg, f"rm -rf {shlex.quote(remote_job_output_dir)}", check=False, log_command=False)

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
    wait_for_ssh_ready(
        cfg,
        timeout_sec=args.ssh_ready_timeout_sec,
        poll_sec=args.ssh_ready_poll_sec,
    )
    remote_root = args.remote_root.rstrip("/")
    remote_output_dir = remote_join(remote_root, args.remote_output_subdir)
    local_dir = Path(args.download_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    log(f"Collecting remote outputs from {remote_output_dir} -> {local_dir}")
    rsync_from_remote(cfg, f"{remote_output_dir}/", f"{local_dir}/", delete=False)
    return 0


def cmd_queue_start(args: argparse.Namespace) -> int:
    require_tool("ssh")
    cfg = cfg_from_args(args)
    wait_for_ssh_ready(
        cfg,
        timeout_sec=args.ssh_ready_timeout_sec,
        poll_sec=args.ssh_ready_poll_sec,
    )
    maybe_sync_remote_repo(cfg, args)
    queue_paths = _remote_queue_paths(args, args.queue_name)
    _start_remote_queue_worker(cfg, args, queue_paths)
    log(f"Queue worker ready: name={queue_paths['name']} dir={queue_paths['root']}")
    return 0


def cmd_queue_status(args: argparse.Namespace) -> int:
    require_tool("ssh")
    cfg = cfg_from_args(args)
    wait_for_ssh_ready(
        cfg,
        timeout_sec=args.ssh_ready_timeout_sec,
        poll_sec=args.ssh_ready_poll_sec,
    )
    queue_paths = _remote_queue_paths(args, args.queue_name)
    running = _remote_worker_running(cfg, queue_paths)
    state_payload = _read_remote_json(cfg, queue_paths["worker_state"])
    if state_payload:
        log(
            f"Queue '{queue_paths['name']}' status: running={running} "
            f"processed={int(state_payload.get('processed_count', 0) or 0)} "
            f"success={int(state_payload.get('success_count', 0) or 0)} "
            f"failed={int(state_payload.get('failed_count', 0) or 0)} "
            f"current={state_payload.get('current_job_name', '') or '-'}"
        )
    else:
        log(f"Queue '{queue_paths['name']}' status: running={running} (no state JSON)")
    return 0


def cmd_queue_stop(args: argparse.Namespace) -> int:
    require_tool("ssh")
    cfg = cfg_from_args(args)
    wait_for_ssh_ready(
        cfg,
        timeout_sec=args.ssh_ready_timeout_sec,
        poll_sec=args.ssh_ready_poll_sec,
    )
    queue_paths = _remote_queue_paths(args, args.queue_name)
    _stop_remote_queue_worker(
        cfg,
        queue_paths,
        wait_timeout_sec=max(0.0, float(getattr(args, "queue_stop_timeout_sec", 60.0) or 60.0)),
    )
    return 0


def cmd_queue_enqueue(args: argparse.Namespace) -> int:
    require_tool("ssh")
    require_tool("rsync")
    cfg = cfg_from_args(args)
    wait_for_ssh_ready(
        cfg,
        timeout_sec=args.ssh_ready_timeout_sec,
        poll_sec=args.ssh_ready_poll_sec,
    )
    maybe_sync_remote_repo(cfg, args)

    queue_paths = _remote_queue_paths(args, args.queue_name)
    if bool(getattr(args, "queue_auto_start", True)):
        _start_remote_queue_worker(cfg, args, queue_paths)
    elif not _remote_worker_running(cfg, queue_paths):
        raise CloudCtlError(
            f"Queue worker '{queue_paths['name']}' is not running. Use queue-start or --queue-auto-start."
        )
    else:
        _ensure_remote_queue_dirs(cfg, queue_paths)

    local_input = Path(args.local_input).expanduser().resolve()
    if not local_input.exists():
        raise CloudCtlError(f"Input clip not found: {local_input}")

    job_name = sanitize_name(str(args.job_name or "").strip()) if str(args.job_name or "").strip() else sanitize_name(local_input.stem)
    _, _, remote_job_output_dir, remote_input_path = _remote_job_paths(args, job_name, local_input.suffix)
    ssh_run(
        cfg,
        f"mkdir -p {shlex.quote(remote_job_output_dir)}",
        check=True,
        log_command=False,
    )

    log(f"Uploading clip: {local_input}")
    rsync_to_remote(cfg, str(local_input), remote_input_path)

    enqueue_stamp = int(time.time() * 1000)
    job_id = sanitize_name(f"{enqueue_stamp}_{job_name}")
    queue_payload = {
        "job_id": job_id,
        "job_name": job_name,
        "input": remote_input_path,
        "output_dir": remote_job_output_dir,
        "status_json": remote_join(remote_job_output_dir, "job_status.json"),
    }
    _enqueue_remote_queue_job(cfg, queue_paths, job_id=job_id, payload=queue_payload)
    log(f"Queued job: {job_name} ({job_id}) on queue '{queue_paths['name']}'")

    if not bool(getattr(args, "wait", False)):
        return 0

    result_kind, remote_result_path = _wait_for_remote_queue_job_result(
        cfg,
        queue_paths,
        job_id=job_id,
        poll_sec=max(0.25, float(getattr(args, "queue_poll_sec", 1.0) or 1.0)),
    )
    if result_kind == "failed":
        payload = _read_remote_json(cfg, remote_result_path) or {}
        message = str(payload.get("status_message", "")).strip() or "queue worker reported failure"
        log(f"ERROR: queued job failed: {message}")
        if not args.keep_remote_input:
            ssh_run(cfg, f"rm -f {shlex.quote(remote_input_path)}", check=False, log_command=False)
        if not args.keep_remote_output:
            ssh_run(cfg, f"rm -rf {shlex.quote(remote_job_output_dir)}", check=False, log_command=False)
        ssh_run(cfg, f"rm -f {shlex.quote(remote_result_path)}", check=False, log_command=False)
        return 1

    _download_job_outputs_and_log_stats(
        cfg,
        args,
        job_name=job_name,
        remote_job_output_dir=remote_job_output_dir,
        log_prefix="",
    )
    _cleanup_remote_job_artifacts(
        cfg,
        args,
        remote_input_path=remote_input_path,
        remote_job_output_dir=remote_job_output_dir,
    )
    ssh_run(cfg, f"rm -f {shlex.quote(remote_result_path)}", check=False, log_command=False)
    log(f"Queued job complete: {job_name}")
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
    p_job.add_argument(
        "--queue-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use long-lived remote queue worker mode for run-job.",
    )
    p_job.add_argument("--queue-name", default="", help="Optional queue worker name. Auto-generated if omitted.")
    p_job.add_argument("--queue-poll-sec", type=float, default=1.0)
    p_job.add_argument(
        "--keep-queue-worker",
        action="store_true",
        help="Do not stop the remote queue worker when this job finishes.",
    )
    p_job.add_argument(
        "--queue-stream-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream remote queue worker log lines in real time.",
    )
    p_job.add_argument("--download-dir", default="./cloud_downloads")
    p_job.add_argument(
        "--download-into-job-subdir",
        action="store_true",
        help="Store downloads under <download-dir>/<job-name>/ instead of directly in <download-dir>.",
    )
    p_job.add_argument("--skip-download", action="store_true")
    p_job.add_argument("--keep-remote-input", action="store_true")
    p_job.add_argument("--keep-remote-output", action="store_true")
    p_job.set_defaults(func=cmd_run_job)

    p_batch = sub.add_parser("run-batch", help="Upload and run a batch of clips sequentially.")
    add_ssh_flags(p_batch)
    add_model_job_flags(p_batch)
    p_batch.add_argument("--input-dir", default="")
    p_batch.add_argument(
        "--input-manifest",
        default="",
        help="Optional text file of clips to process in order. Each line: <path> or <job_name><TAB><path>.",
    )
    p_batch.add_argument("--patterns", default="*.mkv,*.mp4,*.mov,*.avi")
    p_batch.add_argument("--recursive", action="store_true")
    p_batch.add_argument("--max-jobs", type=int, default=0)
    p_batch.add_argument("--continue-on-error", action="store_true")
    p_batch.add_argument(
        "--prefetch-window",
        type=int,
        default=0,
        help="Pipeline staging depth: number of additional clips to keep uploaded ahead of current inference.",
    )
    p_batch.add_argument(
        "--prefetch-upload-all",
        action="store_true",
        help="Stage all batch clips to remote first so inference can run back-to-back with less GPU idle.",
    )
    p_batch.add_argument(
        "--queue-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use long-lived remote queue worker mode for run-batch.",
    )
    p_batch.add_argument(
        "--queue-name",
        default="",
        help="Optional queue worker name. Auto-generated if omitted.",
    )
    p_batch.add_argument(
        "--queue-poll-sec",
        type=float,
        default=1.0,
        help="Polling interval when waiting for queued job completion.",
    )
    p_batch.add_argument(
        "--keep-queue-worker",
        action="store_true",
        help="Do not stop the remote queue worker when batch finishes.",
    )
    p_batch.add_argument(
        "--queue-stream-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream remote queue worker log lines in real time.",
    )
    p_batch.add_argument(
        "--persistent-session",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep one remote model process alive and run chunked jobs per session.",
    )
    p_batch.add_argument(
        "--persistent-session-size",
        type=int,
        default=8,
        help="Number of clips to process per persistent model session (>=2 enables session mode).",
    )
    p_batch.add_argument("--job-prefix", default="")
    p_batch.add_argument("--remote-input-subdir", default="cloud_jobs/incoming")
    p_batch.add_argument("--remote-output-subdir", default="cloud_jobs/output")
    p_batch.add_argument("--download-dir", default="./cloud_downloads")
    p_batch.add_argument(
        "--download-into-job-subdir",
        action="store_true",
        help="Store each job under <download-dir>/<job-name>/ instead of directly in <download-dir>.",
    )
    p_batch.add_argument("--skip-download", action="store_true")
    p_batch.add_argument("--keep-remote-input", action="store_true")
    p_batch.add_argument("--keep-remote-output", action="store_true")
    p_batch.set_defaults(func=cmd_run_batch)

    p_collect = sub.add_parser("collect", help="Download remote output folder.")
    add_ssh_flags(p_collect)
    p_collect.add_argument("--remote-output-subdir", default="cloud_jobs/output")
    p_collect.add_argument("--download-dir", default="./cloud_downloads")
    p_collect.set_defaults(func=cmd_collect)

    p_qstart = sub.add_parser("queue-start", help="Start (or reuse) a long-lived remote queue worker.")
    add_ssh_flags(p_qstart)
    add_model_job_flags(p_qstart)
    add_queue_name_flag(p_qstart, default="default")
    p_qstart.add_argument("--remote-input-subdir", default="cloud_jobs/incoming")
    p_qstart.add_argument("--remote-output-subdir", default="cloud_jobs/output")
    p_qstart.add_argument("--queue-poll-sec", type=float, default=1.0)
    p_qstart.set_defaults(func=cmd_queue_start)

    p_qstatus = sub.add_parser("queue-status", help="Show queue worker state.")
    add_ssh_flags(p_qstatus)
    add_queue_name_flag(p_qstatus, default="default")
    p_qstatus.add_argument("--remote-input-subdir", default="cloud_jobs/incoming")
    p_qstatus.add_argument("--remote-output-subdir", default="cloud_jobs/output")
    p_qstatus.set_defaults(func=cmd_queue_status)

    p_qstop = sub.add_parser("queue-stop", help="Request queue worker shutdown and wait.")
    add_ssh_flags(p_qstop)
    add_queue_name_flag(p_qstop, default="default")
    p_qstop.add_argument("--remote-input-subdir", default="cloud_jobs/incoming")
    p_qstop.add_argument("--remote-output-subdir", default="cloud_jobs/output")
    p_qstop.add_argument("--queue-stop-timeout-sec", type=float, default=60.0)
    p_qstop.set_defaults(func=cmd_queue_stop)

    p_qenqueue = sub.add_parser("queue-enqueue", help="Enqueue one clip onto a running queue worker.")
    add_ssh_flags(p_qenqueue)
    add_model_job_flags(p_qenqueue)
    add_queue_name_flag(p_qenqueue, default="default")
    p_qenqueue.add_argument("--local-input", required=True)
    p_qenqueue.add_argument("--job-name", default="")
    p_qenqueue.add_argument("--remote-input-subdir", default="cloud_jobs/incoming")
    p_qenqueue.add_argument("--remote-output-subdir", default="cloud_jobs/output")
    p_qenqueue.add_argument("--queue-auto-start", action=argparse.BooleanOptionalAction, default=True)
    p_qenqueue.add_argument("--queue-poll-sec", type=float, default=1.0)
    p_qenqueue.add_argument("--wait", action="store_true")
    p_qenqueue.add_argument("--download-dir", default="./cloud_downloads")
    p_qenqueue.add_argument(
        "--download-into-job-subdir",
        action="store_true",
        help="Store downloads under <download-dir>/<job-name>/ instead of directly in <download-dir>.",
    )
    p_qenqueue.add_argument("--skip-download", action="store_true")
    p_qenqueue.add_argument("--keep-remote-input", action="store_true")
    p_qenqueue.add_argument("--keep-remote-output", action="store_true")
    p_qenqueue.set_defaults(func=cmd_queue_enqueue)

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
