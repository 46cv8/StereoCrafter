#!/usr/bin/env python3
"""Launch one Vast.ai worker, prepare a cloud-safe config copy, and optionally run a job.

This script is designed for your DepthCrafter cloud workflow:
1) Find the cheapest acceptable offer for a GPU profile.
2) Show a readable cost/risk summary and ask for approval.
3) Create one Vast instance.
4) Wait until the instance is reachable via ssh-url.
5) Generate a non-destructive config copy with profile-tuned resolution/window.
6) Print (and optionally execute) the cloudctl command derived from that config.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from envfile_to_vast_env import build_vast_env_arg, parse_env_file  # noqa: E402


class VastWorkerLaunchError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[vast-worker] {msg}")


def shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def run_cmd(cmd: Sequence[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    log("$ " + shell_join(list(cmd)))
    return subprocess.run(
        list(cmd),
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def require_tool(name: str) -> None:
    if which(name) is None:
        raise VastWorkerLaunchError(f"Required tool not found in PATH: {name}")


def parse_json_like(raw_text: str) -> Any:
    text = (raw_text or "").strip()
    if not text:
        return None
    candidates = [text]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[-1] != text:
        candidates.append(lines[-1])
    for payload in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(payload)
            except Exception:
                continue
    return None


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def slug(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    out = out.strip("._-")
    return out or "job"


@dataclass(frozen=True)
class GPUProfile:
    key: str
    label: str
    offer_gpu_filter: str
    min_gpu_ram_gb: float
    target_width: int
    target_height: int
    window_size: int
    overlap: int


PROFILES: Dict[str, GPUProfile] = {
    "5090_32gb": GPUProfile(
        key="5090_32gb",
        label="RTX 5090 32GB",
        offer_gpu_filter="gpu_name=RTX_5090",
        min_gpu_ram_gb=30.0,
        target_width=1664,
        target_height=896,
        window_size=75,
        overlap=25,
    ),
    "rtx_pro_6000_96gb": GPUProfile(
        key="rtx_pro_6000_96gb",
        label="RTX PRO 6000 96GB",
        offer_gpu_filter="gpu_name in [RTX_PRO_6000_WS,RTX_PRO_6000_S]",
        min_gpu_ram_gb=92.0,
        target_width=1920,
        target_height=1040,
        window_size=75,
        overlap=25,
    ),
}


def resolve_api_key(args: argparse.Namespace) -> str:
    def _valid(secret: str) -> str:
        value = (secret or "").strip()
        if not value:
            return ""
        upper = value.upper()
        if "REPLACE_WITH" in upper or "YOUR_" in upper or value in {"<TOKEN>", "<API_KEY>", "CHANGEME"}:
            return ""
        return value

    direct = _valid(args.vast_api_key)
    if direct:
        return direct

    for env_key in ("VAST_API_KEY", "VASTAI_API_KEY"):
        val = _valid(os.environ.get(env_key, ""))
        if val:
            return val

    env_path = Path(args.vast_env_file).expanduser().resolve()
    if env_path.exists():
        env_data = parse_env_file(env_path)
        for key in ("VAST_API_KEY", "VASTAI_API_KEY", "VAST_KEY"):
            val = _valid(env_data.get(key, ""))
            if val:
                return val

    return ""


def resolve_hf_env_payload(args: argparse.Namespace) -> str:
    if args.no_hf_env:
        return ""

    env_path = Path(args.hf_env_file).expanduser().resolve()
    if not env_path.exists():
        log(f"HF env file not found ({env_path}); continuing without --env payload.")
        return ""

    env_vars = parse_env_file(env_path)
    if not env_vars:
        log(f"HF env file is empty ({env_path}); continuing without --env payload.")
        return ""

    if "HF_TOKEN" not in env_vars and "HUGGING_FACE_HUB_TOKEN" not in env_vars:
        log(f"HF env file has no HF token var ({env_path}); continuing anyway.")
    return build_vast_env_arg(env_vars)


def with_api_key(cmd: List[str], api_key: str) -> List[str]:
    if api_key:
        return cmd + ["--api-key", api_key]
    return cmd


def build_search_query(profile: GPUProfile, args: argparse.Namespace) -> str:
    parts = [
        profile.offer_gpu_filter,
        "rentable=true",
        "verified=true",
        "num_gpus=1",
        f"cuda_vers>={args.min_cuda}",
        f"reliability>={args.min_reliability}",
        f"disk_space>={args.disk}",
        f"direct_port_count>={args.min_direct_ports}",
        f"inet_down>={args.min_inet_down}",
        f"inet_up>={args.min_inet_up}",
    ]
    if args.max_dph > 0:
        parts.append(f"dph<={args.max_dph}")
    return " ".join(parts)


def normalized_gpu_ram_gb(offer: Dict[str, Any]) -> float:
    raw = as_float(offer.get("gpu_ram"), 0.0)
    if raw <= 0:
        return 0.0
    # Vast responses often expose gpu_ram in MB-like units (e.g. 32607).
    # Guard for either representation.
    return raw / 1024.0 if raw > 1000.0 else raw


def offer_hourly_cost(offer: Dict[str, Any]) -> float:
    for key in ("dph_total", "discounted_dph_total", "dph"):
        if key in offer:
            return as_float(offer.get(key), 0.0)
    return 0.0


def offer_cost_per_tb(offer: Dict[str, Any], direction: str) -> float:
    if direction == "up":
        if "internet_up_cost_per_tb" in offer:
            return as_float(offer.get("internet_up_cost_per_tb"), 0.0)
        return as_float(offer.get("inet_up_cost"), 0.0) * 1024.0
    if "internet_down_cost_per_tb" in offer:
        return as_float(offer.get("internet_down_cost_per_tb"), 0.0)
    return as_float(offer.get("inet_down_cost"), 0.0) * 1024.0


def annotate_offer(offer: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    hourly = offer_hourly_cost(offer)
    up_cost_tb = offer_cost_per_tb(offer, "up")
    down_cost_tb = offer_cost_per_tb(offer, "down")
    transfer_cost = (
        (args.expected_upload_gb / 1024.0) * up_cost_tb
        + (args.expected_download_gb / 1024.0) * down_cost_tb
    )
    runtime_cost = hourly * args.expected_runtime_hours
    total_est = runtime_cost + transfer_cost

    enriched = dict(offer)
    enriched["_hourly_cost"] = hourly
    enriched["_transfer_cost_est"] = transfer_cost
    enriched["_runtime_cost_est"] = runtime_cost
    enriched["_total_cost_est"] = total_est
    return enriched


def short_text(value: Any, width: int) -> str:
    text = str(value if value is not None else "")
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def print_offer_table(offers: List[Dict[str, Any]], show_top: int) -> None:
    header = (
        f"{'Idx':>3} {'OfferID':>9} {'$/h':>7} {'Est$':>8} {'Rel':>6} "
        f"{'GPU':<18} {'VRAM':>6} {'Down':>6} {'Up':>6} {'TB$up/down':>14} {'Location':<22}"
    )
    print(header)
    print("-" * len(header))
    for idx, offer in enumerate(offers[:show_top], start=1):
        offer_id = as_int(offer.get("id"), 0)
        hourly = as_float(offer.get("_hourly_cost"), 0.0)
        est_total = as_float(offer.get("_total_cost_est"), 0.0)
        reliability = as_float(offer.get("reliability"), 0.0)
        gpu_name = short_text(offer.get("gpu_name", ""), 18)
        gpu_ram_gb = normalized_gpu_ram_gb(offer)
        inet_down = as_float(offer.get("inet_down"), 0.0)
        inet_up = as_float(offer.get("inet_up"), 0.0)
        up_tb = offer_cost_per_tb(offer, "up")
        down_tb = offer_cost_per_tb(offer, "down")
        location = short_text(offer.get("geolocation", ""), 22)

        print(
            f"{idx:>3} {offer_id:>9} {hourly:>7.3f} {est_total:>8.3f} {reliability:>6.3f} "
            f"{gpu_name:<18} {gpu_ram_gb:>5.1f}G {inet_down:>6.0f} {inet_up:>6.0f} "
            f"{up_tb:>6.2f}/{down_tb:<6.2f} {location:<22}"
        )


def extract_instance_id(create_payload: Any, create_raw: str) -> int:
    if isinstance(create_payload, dict):
        for key in ("new_contract", "instance_id", "new_instance", "contract_id", "id"):
            if key in create_payload:
                val = as_int(create_payload.get(key), 0)
                if val > 0:
                    return val
    match = re.search(r"(?:new_contract|instance_id|contract_id)\D+(\d+)", create_raw)
    if match:
        return as_int(match.group(1), 0)
    return 0


def extract_instance_status(show_payload: Any) -> str:
    if not isinstance(show_payload, dict):
        return ""
    for key in (
        "actual_status",
        "status",
        "cur_state",
        "state",
        "status_msg",
        "intended_status",
    ):
        val = str(show_payload.get(key, "")).strip()
        if val:
            return val
    return ""


def parse_ssh_url(raw: str) -> Tuple[str, str, int]:
    payload = parse_json_like(raw)
    text = ""
    if isinstance(payload, dict):
        for key in ("ssh_url", "url", "value"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                text = val.strip()
                break
    elif isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, str):
            text = first.strip()
        elif isinstance(first, dict):
            for key in ("ssh_url", "url", "value"):
                val = first.get(key)
                if isinstance(val, str) and val.strip():
                    text = val.strip()
                    break

    if not text:
        text = (raw or "").strip()

    if not text:
        return "", "", 0

    patterns = [
        r"ssh://(?P<user>[^@]+)@(?P<host>[^:\s]+):(?P<port>\d+)",
        r"ssh\s+(?P<user>[^@\s]+)@(?P<host>[^\s]+)\s+-p\s+(?P<port>\d+)",
        r"-p\s+(?P<port>\d+)\s+(?P<user>[^@\s]+)@(?P<host>[^\s]+)",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if not match:
            continue
        user = match.group("user")
        host = match.group("host")
        port = as_int(match.group("port"), 0)
        if host and port > 0:
            return user, host, port
    return "", "", 0


def load_json_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise VastWorkerLaunchError(f"Config file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VastWorkerLaunchError(f"Failed to parse JSON config: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise VastWorkerLaunchError(f"Config root must be an object: {path}")
    return data


def update_config_for_profile(
    base_config: Dict[str, Any],
    profile: GPUProfile,
    instance_id: int,
    selected_offer: Dict[str, Any],
    host: str,
    port: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    cfg = dict(base_config)
    cfg["target_width"] = as_int(args.target_width_override, profile.target_width) if args.target_width_override > 0 else profile.target_width
    cfg["target_height"] = as_int(args.target_height_override, profile.target_height) if args.target_height_override > 0 else profile.target_height
    cfg["window_size"] = as_int(args.window_size_override, profile.window_size) if args.window_size_override > 0 else profile.window_size
    cfg["overlap"] = as_int(args.overlap_override, profile.overlap) if args.overlap_override >= 0 else profile.overlap

    if args.force_cpu_offload:
        cfg["cpu_offload"] = args.force_cpu_offload

    if args.disable_secondary_modes:
        cfg["enable_spatial_refine_mode_var"] = False
        cfg["enable_edge_guided_upscale_mode_var"] = False

    cloud_meta = {
        "enabled": True,
        "profile": profile.key,
        "profile_label": profile.label,
        "vast_instance_id": instance_id,
        "vast_offer_id": as_int(selected_offer.get("id"), 0),
        "vast_machine_id": as_int(selected_offer.get("machine_id"), 0),
        "vast_gpu_name": str(selected_offer.get("gpu_name", "")),
        "vast_gpu_ram_gb": round(normalized_gpu_ram_gb(selected_offer), 3),
        "vast_host": host,
        "vast_ssh_port": port,
        "vast_user": args.remote_user,
        "remote_root": args.remote_root,
        "remote_venv": args.remote_venv,
        "remote_image": args.image,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "cloud/vast_worker_launch.py",
    }
    cfg["cloud_remote"] = cloud_meta
    return cfg


def build_cloudctl_cmd(
    cfg: Dict[str, Any],
    host: str,
    port: int,
    args: argparse.Namespace,
) -> Tuple[List[str], str]:
    input_path = str(cfg.get("input_dir_or_file_var", "")).strip()
    if args.input_override:
        input_path = args.input_override

    if not input_path:
        raise VastWorkerLaunchError(
            "No input path found in config (input_dir_or_file_var). "
            "Set it in the base config or pass --input-override."
        )

    input_obj = Path(input_path).expanduser()
    is_file = input_obj.is_file()
    is_dir = input_obj.is_dir()
    if not is_file and not is_dir:
        suffix = input_obj.suffix.lower()
        if suffix in {".mkv", ".mp4", ".mov", ".avi", ".webm", ".m4v"}:
            is_file = True
        else:
            is_dir = True

    subcmd = "run-job" if is_file else "run-batch"
    cmd: List[str] = [
        sys.executable,
        str(REPO_ROOT / "cloud" / "cloudctl.py"),
        subcmd,
        "--host",
        host,
        "--user",
        args.remote_user,
        "--port",
        str(port),
        "--remote-root",
        args.remote_root,
        "--venv-name",
        args.remote_venv,
    ]

    if args.identity:
        cmd += ["--identity", args.identity]

    target_width = as_int(cfg.get("target_width"), 1920)
    target_height = as_int(cfg.get("target_height"), 1040)
    window_size = as_int(cfg.get("window_size"), 75)
    overlap = as_int(cfg.get("overlap"), 25)
    inference_steps = as_int(cfg.get("inference_steps"), 25)
    guidance_scale = as_float(cfg.get("guidance_scale"), 1.0)
    seed = as_int(cfg.get("seed"), 42)
    target_fps = as_float(cfg.get("target_fps"), -1.0)
    process_length = as_int(cfg.get("process_length"), -1)
    output_format = str(cfg.get("merge_output_format_var", "main10_mp4") or "main10_mp4")
    if output_format not in {"mp4", "main10_mp4"}:
        output_format = "main10_mp4"
    cpu_offload = str(cfg.get("cpu_offload", "model") or "model")
    if cpu_offload not in {"model", "sequential", "none"}:
        cpu_offload = "model"

    cmd += [
        "--target-width",
        str(target_width),
        "--target-height",
        str(target_height),
        "--window-size",
        str(window_size),
        "--overlap",
        str(overlap),
        "--inference-steps",
        str(inference_steps),
        "--guidance-scale",
        str(guidance_scale),
        "--seed",
        str(seed),
        "--target-fps",
        str(target_fps),
        "--process-length",
        str(process_length),
        "--output-format",
        output_format,
        "--cpu-offload",
        cpu_offload,
    ]

    if bool(cfg.get("disable_xformers_var", False)):
        cmd.append("--disable-xformers")
    if bool(cfg.get("use_cudnn_benchmark", False)):
        cmd.append("--use-cudnn-benchmark")
    if bool(cfg.get("use_local_models_only_var", False)):
        cmd.append("--local-files-only")

    download_dir = args.download_dir_override or str(cfg.get("output_dir", "")).strip() or "./cloud_downloads"
    cmd += ["--download-dir", download_dir]

    if subcmd == "run-job":
        cmd += ["--local-input", input_path]
    else:
        cmd += ["--input-dir", input_path, "--patterns", args.batch_patterns]

    return cmd, subcmd


def wait_for_instance_ready(instance_id: int, api_key: str, timeout_sec: int, poll_sec: int) -> Tuple[str, str, int]:
    deadline = time.time() + timeout_sec
    last_status = ""
    while time.time() < deadline:
        status_cmd = with_api_key(
            ["vastai", "show", "instance", str(instance_id), "--raw"],
            api_key,
        )
        status_proc = run_cmd(status_cmd, check=False, capture=True)
        status_raw = (status_proc.stdout or "").strip()
        status_payload = parse_json_like(status_raw)
        status_text = extract_instance_status(status_payload) or "unknown"

        ssh_cmd = with_api_key(
            ["vastai", "ssh-url", str(instance_id)],
            api_key,
        )
        ssh_proc = run_cmd(ssh_cmd, check=False, capture=True)
        ssh_raw = (ssh_proc.stdout or "").strip()
        ssh_user, ssh_host, ssh_port = parse_ssh_url(ssh_raw)

        if status_text != last_status:
            log(f"Instance {instance_id} status: {status_text}")
            last_status = status_text

        if ssh_host and ssh_port > 0:
            return ssh_user or "root", ssh_host, ssh_port

        time.sleep(max(1, poll_sec))

    raise VastWorkerLaunchError(
        f"Timed out waiting for instance {instance_id} to become ready after {timeout_sec}s."
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Launch one Vast worker (5090 or RTX PRO 6000), generate tuned cloud config, and optionally run cloudctl."
    )
    p.add_argument(
        "--profile",
        choices=sorted(PROFILES.keys()),
        required=True,
        help="GPU profile to target.",
    )
    p.add_argument("--image", required=True, help="Docker image for Vast create instance.")
    p.add_argument("--disk", type=int, default=40, help="Disk size in GB for Vast instance.")
    p.add_argument("--label", default="", help="Optional Vast instance label.")
    p.add_argument("--onstart-cmd", default="echo starting; nvidia-smi")
    p.add_argument("--vast-extra-args", default="", help="Extra args appended to vastai create instance.")
    p.add_argument("--cancel-unavail", action="store_true", help="Pass --cancel-unavail to vast create.")

    p.add_argument("--base-config", default=str(REPO_ROOT / "config_depthcrafter.json"))
    p.add_argument(
        "--output-config",
        default="",
        help="Output path for generated config copy. Default: cloud/generated_configs/<timestamp>.json",
    )
    p.add_argument("--target-width-override", type=int, default=0)
    p.add_argument("--target-height-override", type=int, default=0)
    p.add_argument("--window-size-override", type=int, default=0)
    p.add_argument("--overlap-override", type=int, default=-1)
    p.add_argument("--force-cpu-offload", choices=["model", "sequential", "none"], default="")
    p.add_argument(
        "--disable-secondary-modes",
        action="store_true",
        help="Force-disable spatial refine and edge-guided modes in generated config.",
    )

    p.add_argument("--remote-user", default="root")
    p.add_argument("--remote-root", default="/opt/StereoCrafter")
    p.add_argument("--remote-venv", default="/opt/venv")
    p.add_argument("--identity", default="", help="SSH private key path for cloudctl calls.")
    p.add_argument("--input-override", default="", help="Override config input_dir_or_file_var.")
    p.add_argument("--download-dir-override", default="", help="Override download/output dir for cloudctl.")
    p.add_argument("--batch-patterns", default="*.mkv,*.mp4,*.mov,*.avi")

    p.add_argument("--offer-limit", type=int, default=30)
    p.add_argument("--show-top", type=int, default=8)
    p.add_argument("--offer-type", choices=["on-demand", "reserved", "bid"], default="on-demand")
    p.add_argument("--search-order", default="dph_total")
    p.add_argument("--min-cuda", type=float, default=12.8)
    p.add_argument("--min-reliability", type=float, default=0.97)
    p.add_argument("--min-direct-ports", type=int, default=2)
    p.add_argument("--min-inet-down", type=float, default=200.0)
    p.add_argument("--min-inet-up", type=float, default=50.0)
    p.add_argument("--max-dph", type=float, default=0.0, help="Optional max hourly cap (0 disables cap).")
    p.add_argument("--expected-runtime-hours", type=float, default=1.0)
    p.add_argument("--expected-upload-gb", type=float, default=8.0)
    p.add_argument("--expected-download-gb", type=float, default=8.0)
    p.add_argument("--offer-id", type=int, default=0, help="Skip ranking and force this offer id.")

    p.add_argument("--vast-api-key", default="")
    p.add_argument("--vast-env-file", default=str(REPO_ROOT / "cloud" / "vast.env"))
    p.add_argument("--hf-env-file", default=str(REPO_ROOT / "cloud" / "hf.env"))
    p.add_argument("--no-hf-env", action="store_true")

    p.add_argument("--ready-timeout-sec", type=int, default=900)
    p.add_argument("--poll-sec", type=int, default=8)

    p.add_argument("--yes", action="store_true", help="Skip launch confirmation prompt.")
    p.add_argument("--dry-run", action="store_true", help="Show selected offer and exit before create instance.")
    p.add_argument("--run-now", action="store_true", help="Immediately run cloudctl command after ready.")
    p.add_argument("--no-go-prompt", action="store_true", help="Do not prompt for GO when not using --run-now.")
    return p


def main() -> int:
    args = parser().parse_args()
    require_tool("vastai")

    profile = PROFILES[args.profile]
    base_config_path = Path(args.base_config).expanduser().resolve()
    base_config = load_json_config(base_config_path)

    api_key = resolve_api_key(args)
    if not api_key:
        log("No explicit Vast API key found; relying on vastai local login state.")

    hf_env_payload = resolve_hf_env_payload(args)
    query = build_search_query(profile, args)

    search_cmd = [
        "vastai",
        "search",
        "offers",
        query,
        "--raw",
        "--limit",
        str(args.offer_limit),
        "--storage",
        str(args.disk),
        "--order",
        args.search_order,
        "--no-default",
    ]
    if args.offer_type != "on-demand":
        search_cmd += ["--type", args.offer_type]
    search_cmd = with_api_key(search_cmd, api_key)

    search_proc = run_cmd(search_cmd, capture=True)
    search_payload = parse_json_like(search_proc.stdout or "")
    if not isinstance(search_payload, list) or not search_payload:
        raise VastWorkerLaunchError(f"No offers matched query for profile '{profile.key}'. Query: {query}")

    offers = []
    for offer in search_payload:
        if not isinstance(offer, dict):
            continue
        if normalized_gpu_ram_gb(offer) + 1e-6 < profile.min_gpu_ram_gb:
            continue
        offers.append(annotate_offer(offer, args))
    if not offers:
        raise VastWorkerLaunchError(
            f"No offers matched query for profile '{profile.key}' after RAM guard ({profile.min_gpu_ram_gb:.1f}GB). "
            f"Query: {query}"
        )
    offers.sort(
        key=lambda offer: (
            as_float(offer.get("_total_cost_est"), 1e12),
            as_float(offer.get("_hourly_cost"), 1e12),
            -as_float(offer.get("reliability"), 0.0),
        )
    )

    log(
        f"Top offers for {profile.label} (sorted by expected total = "
        f"{args.expected_runtime_hours}h runtime + transfer {args.expected_upload_gb}/{args.expected_download_gb} GB):"
    )
    print_offer_table(offers, args.show_top)

    if args.offer_id > 0:
        selected_offer = next((o for o in offers if as_int(o.get("id"), 0) == args.offer_id), None)
        if selected_offer is None:
            raise VastWorkerLaunchError(f"--offer-id {args.offer_id} not found in the returned candidate list.")
    else:
        selected_offer = offers[0]

    selected_offer_id = as_int(selected_offer.get("id"), 0)
    selected_hourly = as_float(selected_offer.get("_hourly_cost"), 0.0)
    selected_total = as_float(selected_offer.get("_total_cost_est"), 0.0)
    selected_loc = str(selected_offer.get("geolocation", ""))
    selected_gpu = str(selected_offer.get("gpu_name", ""))

    log(
        f"Selected offer {selected_offer_id}: {selected_gpu} | {selected_loc} | "
        f"${selected_hourly:.3f}/hr | est total ${selected_total:.3f}"
    )

    if args.dry_run:
        log("Dry run requested; stopping before instance creation.")
        return 0

    if not args.yes:
        if not sys.stdin.isatty():
            raise VastWorkerLaunchError("Confirmation prompt required but stdin is not interactive. Re-run with --yes.")
        answer = input("Launch this instance? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            log("Cancelled before instance creation.")
            return 0

    create_cmd = [
        "vastai",
        "create",
        "instance",
        str(selected_offer_id),
        "--image",
        args.image,
        "--disk",
        str(args.disk),
        "--ssh",
        "--direct",
        "--onstart-cmd",
        args.onstart_cmd,
        "--raw",
    ]
    if args.label:
        create_cmd += ["--label", args.label]
    if args.cancel_unavail:
        create_cmd.append("--cancel-unavail")
    if hf_env_payload:
        create_cmd += ["--env", hf_env_payload]
    if args.vast_extra_args:
        create_cmd += shlex.split(args.vast_extra_args)
    create_cmd = with_api_key(create_cmd, api_key)

    create_proc = run_cmd(create_cmd, capture=True)
    create_raw = create_proc.stdout or ""
    create_payload = parse_json_like(create_raw)
    instance_id = extract_instance_id(create_payload, create_raw)
    if instance_id <= 0:
        raise VastWorkerLaunchError(
            "Could not determine instance id from vastai create response. "
            f"Raw response:\n{create_raw}"
        )

    log(f"Created instance id: {instance_id}. Waiting for readiness...")
    ssh_user, ssh_host, ssh_port = wait_for_instance_ready(
        instance_id=instance_id,
        api_key=api_key,
        timeout_sec=args.ready_timeout_sec,
        poll_sec=args.poll_sec,
    )

    log(f"READY: instance={instance_id} ssh={ssh_user}@{ssh_host}:{ssh_port}")

    generated_cfg = update_config_for_profile(
        base_config=base_config,
        profile=profile,
        instance_id=instance_id,
        selected_offer=selected_offer,
        host=ssh_host,
        port=ssh_port,
        args=args,
    )

    if args.output_config:
        output_cfg_path = Path(args.output_config).expanduser().resolve()
    else:
        out_dir = REPO_ROOT / "cloud" / "generated_configs"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_cfg_path = out_dir / f"config_depthcrafter.cloud.{slug(profile.key)}.{stamp}.json"
    output_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    output_cfg_path.write_text(json.dumps(generated_cfg, indent=2), encoding="utf-8")
    log(f"Wrote generated cloud config: {output_cfg_path}")

    cloudctl_cmd, mode = build_cloudctl_cmd(
        cfg=generated_cfg,
        host=ssh_host,
        port=ssh_port,
        args=args,
    )
    log(f"Prepared cloudctl command ({mode}):")
    print(shell_join(cloudctl_cmd))

    if args.run_now:
        log("Running remote job now (--run-now).")
        run_cmd(cloudctl_cmd, capture=False)
        return 0

    if not args.no_go_prompt and sys.stdin.isatty():
        answer = input("Type GO to start remote processing now, or press Enter to stop here: ").strip().upper()
        if answer == "GO":
            log("Launching cloudctl remote processing...")
            run_cmd(cloudctl_cmd, capture=False)
        else:
            log("Stopped after preparation. Instance is ready; run the printed cloudctl command when you are ready.")
    else:
        log("Preparation complete. Instance is ready; run the printed cloudctl command when you are ready.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VastWorkerLaunchError as exc:
        log(f"ERROR: {exc}")
        raise SystemExit(2)
