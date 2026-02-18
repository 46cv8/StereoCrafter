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
import posixpath
import re
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from envfile_to_vast_env import build_vast_env_arg, parse_env_file  # noqa: E402
import cloud_core  # noqa: E402

DEFAULT_VAST_API_BASE_URL = cloud_core.DEFAULT_VAST_API_BASE_URL


class VastWorkerLaunchError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[vast-worker] {msg}", flush=True)


def shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def remote_join(*parts: str) -> str:
    if not parts:
        return ""
    out = parts[0]
    for part in parts[1:]:
        out = posixpath.join(out, part)
    return out


def redact_sensitive_cmd(parts: Sequence[str]) -> List[str]:
    redacted: List[str] = []
    redact_next = False
    for part in parts:
        part_str = str(part)
        if redact_next:
            redacted.append("***")
            redact_next = False
            continue
        redacted.append(part_str)
        if part_str in {"--api-key", "--login", "--env", "--registry-login", "--vast-api-key"}:
            redact_next = True
    return redacted


def run_cmd(
    cmd: Sequence[str],
    *,
    check: bool = True,
    capture: bool = True,
    log_command: bool = True,
) -> subprocess.CompletedProcess:
    if log_command:
        log("$ " + shell_join(redact_sensitive_cmd(list(cmd))))
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


def detect_local_git_branch(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        return ""
    branch = (result.stdout or "").strip()
    if not branch or branch == "HEAD":
        return ""
    return branch


def parse_json_like(raw_text: str) -> Any:
    return cloud_core.parse_json_like(raw_text)


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
    key: GPUProfile(
        key=key,
        label=str(defaults.get("label", key)),
        offer_gpu_filter=str(defaults.get("offer_gpu_filter", "")),
        min_gpu_ram_gb=float(defaults.get("min_gpu_ram_gb", 0.0)),
        target_width=int(defaults.get("target_width", 1920)),
        target_height=int(defaults.get("target_height", 1040)),
        window_size=int(defaults.get("window_size", 75)),
        overlap=int(defaults.get("overlap", 25)),
    )
    for key, defaults in cloud_core.CLOUD_PROFILE_DEFAULTS.items()
}
GPU_RAM_TOLERANCE_GB = cloud_core.GPU_RAM_TOLERANCE_GB


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


def registry_from_image_ref(image_ref: str) -> str:
    image = (image_ref or "").strip()
    if "/" not in image:
        return ""
    return image.split("/", 1)[0].lower()


def resolve_auto_registry_login_arg(args: argparse.Namespace) -> str:
    if bool(getattr(args, "skip_image_login", False)):
        return ""

    explicit = (getattr(args, "registry_login", "") or "").strip()
    if explicit:
        return explicit

    registry = registry_from_image_ref(str(getattr(args, "image", "") or ""))
    if registry != "ghcr.io":
        return ""

    env_vars: Dict[str, str] = {}
    ghcr_env_path = Path(getattr(args, "ghcr_env_file", "") or "").expanduser().resolve()
    if ghcr_env_path.exists():
        try:
            env_vars = parse_env_file(ghcr_env_path)
        except Exception as exc:
            log(f"Warning: failed to parse GHCR env file {ghcr_env_path}: {exc}")

    def _pick(*keys: str) -> str:
        for key in keys:
            value = (env_vars.get(key, "") or os.environ.get(key, "")).strip()
            if value:
                return value
        return ""

    username = _pick("GHCR_USERNAME", "GHCR_USER", "GITHUB_USER")
    token = _pick("GHCR_PAT", "GITHUB_PAT", "GH_PAT")
    if not username:
        image_ref = str(getattr(args, "image", "") or "")
        parts = image_ref.split("/")
        if len(parts) >= 2:
            username = parts[1].strip()
    if not token:
        return ""
    if not username:
        return ""
    return f"-u {username} -p {token} {registry}"


def with_api_key(cmd: List[str], api_key: str) -> List[str]:
    if api_key:
        return cmd + ["--api-key", api_key]
    return cmd


def build_search_query(profile: GPUProfile, args: argparse.Namespace) -> str:
    require_verified = not bool(getattr(args, "allow_unverified", False))
    profile_defaults = cloud_core.get_cloud_profile_defaults(profile.key)
    return cloud_core.build_offer_search_query(
        profile_defaults,
        disk_gb=int(args.disk),
        require_verified_hosts=require_verified,
        max_dph=float(args.max_dph),
        min_cuda=float(args.min_cuda),
        min_reliability=float(args.min_reliability),
        min_direct_ports=int(args.min_direct_ports),
        min_inet_down=float(args.min_inet_down),
        min_inet_up=float(args.min_inet_up),
    )


def run_offer_search(query: str, args: argparse.Namespace, api_key: str) -> List[Dict[str, Any]]:
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
    if not isinstance(search_payload, list):
        return []
    return [offer for offer in search_payload if isinstance(offer, dict)]


def normalize_blacklist_payload(payload: Any) -> Dict[str, set]:
    return cloud_core.normalize_blacklist_data(payload if isinstance(payload, dict) else None)


def load_blacklist_file(path_value: str) -> Dict[str, set]:
    return cloud_core.load_blacklist_data(path_value)


def offer_is_blacklisted(offer: Dict[str, Any], blacklist: Dict[str, set]) -> bool:
    return cloud_core.offer_is_blacklisted(offer, blacklist)


def normalized_gpu_ram_gb(offer: Dict[str, Any]) -> float:
    return cloud_core.normalized_gpu_ram_gb(offer)


def offer_hourly_cost(offer: Dict[str, Any]) -> float:
    return cloud_core.offer_hourly_cost(offer)


def offer_cost_per_tb(offer: Dict[str, Any], direction: str) -> float:
    return cloud_core.offer_cost_per_tb(offer, direction)


def annotate_offer(offer: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cost_data = cloud_core.estimate_offer_total_cost(
        offer,
        expected_runtime_hours=float(args.expected_runtime_hours),
        expected_upload_gb=float(args.expected_upload_gb),
        expected_download_gb=float(args.expected_download_gb),
    )

    enriched = dict(offer)
    enriched["_hourly_cost"] = float(cost_data["hourly"])
    enriched["_transfer_cost_est"] = float(cost_data["transfer_cost"])
    enriched["_runtime_cost_est"] = float(cost_data["runtime_cost"])
    enriched["_total_cost_est"] = float(cost_data["total_cost"])
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
        "vast_host_id": as_int(selected_offer.get("host_id"), 0),
        "vast_geolocation": str(selected_offer.get("geolocation", "")),
        "vast_reliability": as_float(selected_offer.get("reliability"), 0.0),
        "vast_hourly_cost": as_float(selected_offer.get("_hourly_cost"), as_float(selected_offer.get("dph_total"), 0.0)),
        "vast_gpu_name": str(selected_offer.get("gpu_name", "")),
        "vast_gpu_ram_gb": round(normalized_gpu_ram_gb(selected_offer), 3),
        "vast_host": host,
        "vast_ssh_port": port,
        "vast_user": args.remote_user,
        "remote_root": args.remote_root,
        "remote_venv": args.remote_venv,
        "remote_image": args.image,
        "git_sync_branch": str(getattr(args, "git_sync_branch", "") or ""),
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

    git_sync_branch = str(getattr(args, "git_sync_branch", "") or "").strip()
    if not git_sync_branch:
        cloud_remote = cfg.get("cloud_remote", {})
        if isinstance(cloud_remote, dict):
            git_sync_branch = str(cloud_remote.get("git_sync_branch", "") or "").strip()
    if git_sync_branch:
        cmd += ["--git-sync-branch", git_sync_branch]

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
    model_backend = str(cfg.get("model_backend_var", "depthcrafter") or "depthcrafter").strip().lower()
    if model_backend not in {"depthcrafter", "geometrycrafter_diff", "geometrycrafter_determ", "stereopilot"}:
        model_backend = "depthcrafter"
    geometry_model_path = str(cfg.get("geometry_model_path_var", "TencentARC/GeometryCrafter") or "TencentARC/GeometryCrafter")
    geometry_repo_path = remote_join(args.remote_root.rstrip("/"), "weights", "GeometryCrafter")
    geometry_cache_dir = remote_join(args.remote_root.rstrip("/"), "weights", "hf_cache")
    geometry_decode_chunk_size = max(1, as_int(cfg.get("geometry_decode_chunk_size_var"), 8))
    geometry_low_memory_usage = bool(cfg.get("geometry_low_memory_usage_var", False))
    geometry_force_projection = bool(cfg.get("geometry_force_projection_var", True))
    geometry_force_fixed_focal = bool(cfg.get("geometry_force_fixed_focal_var", True))
    geometry_use_extract_interp = bool(cfg.get("geometry_use_extract_interp_var", False))
    stereopilot_model_path = str(cfg.get("stereopilot_model_path_var", "KlingTeam/StereoPilot") or "KlingTeam/StereoPilot")
    stereopilot_base_model_path = str(cfg.get("stereopilot_base_model_path_var", "Wan-AI/Wan2.1-T2V-1.3B") or "Wan-AI/Wan2.1-T2V-1.3B")
    stereopilot_repo_path = remote_join(args.remote_root.rstrip("/"), "weights", "StereoPilot")
    stereopilot_cache_dir = remote_join(args.remote_root.rstrip("/"), "weights", "hf_cache")
    stereopilot_prompt = str(cfg.get("stereopilot_prompt_var", "") or "")
    stereopilot_use_sidecar_prompt = bool(cfg.get("stereopilot_use_sidecar_prompt_var", True))
    stereopilot_output_mode = str(cfg.get("stereopilot_output_mode_var", "side_by_side") or "side_by_side").strip().lower()
    if stereopilot_output_mode not in {"opposite_eye", "side_by_side", "both"}:
        stereopilot_output_mode = "side_by_side"
    stereopilot_target_width = max(32, as_int(cfg.get("stereopilot_target_width_var"), 832))
    stereopilot_target_height = max(32, as_int(cfg.get("stereopilot_target_height_var"), 480))
    stereopilot_target_fps = max(1.0, as_float(cfg.get("stereopilot_target_fps_var"), 16.0))
    stereopilot_sampling_steps = max(1, as_int(cfg.get("stereopilot_sampling_steps_var"), 30))
    stereopilot_guide_scale = as_float(cfg.get("stereopilot_guide_scale_var"), 5.0)
    stereopilot_shift = as_float(cfg.get("stereopilot_shift_var"), 5.0)
    stereopilot_tail_pad_frames = max(0, as_int(cfg.get("stereopilot_tail_pad_frames_var"), 5))
    stereopilot_domain_label = 1 if as_int(cfg.get("stereopilot_domain_label_var"), 1) != 0 else 0
    stereopilot_dtype = str(cfg.get("stereopilot_dtype_var", "bfloat16") or "bfloat16").strip().lower()
    stereopilot_transformer_dtype = str(cfg.get("stereopilot_transformer_dtype_var", "float8") or "float8").strip().lower()

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
        "--model-backend",
        model_backend,
        "--geometry-model-path",
        geometry_model_path,
        "--geometry-repo-path",
        geometry_repo_path,
        "--geometry-cache-dir",
        geometry_cache_dir,
        "--geometry-decode-chunk-size",
        str(geometry_decode_chunk_size),
        "--stereopilot-model-path",
        stereopilot_model_path,
        "--stereopilot-base-model-path",
        stereopilot_base_model_path,
        "--stereopilot-repo-path",
        stereopilot_repo_path,
        "--stereopilot-cache-dir",
        stereopilot_cache_dir,
        "--stereopilot-output-mode",
        stereopilot_output_mode,
        "--stereopilot-target-width",
        str(stereopilot_target_width),
        "--stereopilot-target-height",
        str(stereopilot_target_height),
        "--stereopilot-target-fps",
        str(stereopilot_target_fps),
        "--stereopilot-sampling-steps",
        str(stereopilot_sampling_steps),
        "--stereopilot-guide-scale",
        str(stereopilot_guide_scale),
        "--stereopilot-shift",
        str(stereopilot_shift),
        "--stereopilot-tail-pad-frames",
        str(stereopilot_tail_pad_frames),
        "--stereopilot-domain-label",
        str(stereopilot_domain_label),
        "--stereopilot-dtype",
        stereopilot_dtype,
        "--stereopilot-transformer-dtype",
        stereopilot_transformer_dtype,
    ]

    if bool(cfg.get("disable_xformers_var", False)):
        cmd.append("--disable-xformers")
    if bool(cfg.get("use_cudnn_benchmark", False)):
        cmd.append("--use-cudnn-benchmark")
    if bool(cfg.get("use_local_models_only_var", False)):
        cmd.append("--local-files-only")
    if geometry_low_memory_usage:
        cmd.append("--geometry-low-memory-usage")
    cmd.append("--geometry-force-projection" if geometry_force_projection else "--no-geometry-force-projection")
    cmd.append("--geometry-force-fixed-focal" if geometry_force_fixed_focal else "--no-geometry-force-fixed-focal")
    cmd.append("--geometry-use-extract-interp" if geometry_use_extract_interp else "--no-geometry-use-extract-interp")
    cmd.append("--stereopilot-use-sidecar-prompt" if stereopilot_use_sidecar_prompt else "--no-stereopilot-use-sidecar-prompt")
    if stereopilot_prompt.strip():
        cmd += ["--stereopilot-prompt", stereopilot_prompt.strip()]

    download_dir = args.download_dir_override or str(cfg.get("output_dir", "")).strip() or "./cloud_downloads"
    cmd += ["--download-dir", download_dir]

    if subcmd == "run-job":
        cmd += ["--local-input", input_path]
    else:
        cmd += ["--input-dir", input_path, "--patterns", args.batch_patterns]

    return cmd, subcmd


def resolve_vast_api_base_url() -> str:
    for key in ("VAST_API_URL", "VAST_URL", "VAST_SERVER_URL"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value.rstrip("/")
    return DEFAULT_VAST_API_BASE_URL


def extract_instance_row_from_payload(payload: Any, instance_id: int) -> Optional[Dict[str, Any]]:
    return cloud_core.extract_instance_row_from_payload(payload, instance_id)


def extract_instance_status_from_row(row: Dict[str, Any]) -> str:
    return cloud_core.extract_instance_status_from_row(row)


def extract_ssh_from_instance_row(row: Dict[str, Any]) -> Tuple[str, int]:
    return cloud_core.extract_ssh_from_instance_row(row)


def fetch_instance_row_http(instance_id: int, api_key: str) -> Optional[Dict[str, Any]]:
    if not api_key:
        return None
    base_url = resolve_vast_api_base_url()
    query = urlencode({"owner": "me", "api_key": api_key})
    url = f"{base_url}/api/v0/instances?{query}"
    try:
        with urlopen(url, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return extract_instance_row_from_payload(payload, instance_id)
    except Exception as exc:
        log(f"HTTP poll warning for instance {instance_id}: {exc}")
        return None


def fetch_instance_row_cli(instance_id: int, api_key: str) -> Optional[Dict[str, Any]]:
    show_instances_cmd = with_api_key(
        ["vastai", "show", "instances", "--raw"],
        api_key,
    )
    show_instances_proc = run_cmd(show_instances_cmd, check=False, capture=True)
    payload = parse_json_like((show_instances_proc.stdout or "").strip())
    return extract_instance_row_from_payload(payload, instance_id)


def fetch_instance_logs_lines(instance_id: int, api_key: str) -> Tuple[int, List[str]]:
    logs_cmd = with_api_key(
        ["vastai", "logs", str(instance_id)],
        api_key,
    )
    logs_proc = run_cmd(logs_cmd, check=False, capture=True, log_command=False)
    raw_logs = (logs_proc.stdout or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw_logs.strip():
        return logs_proc.returncode, []
    lines = [line.rstrip() for line in raw_logs.split("\n")]
    # Keep non-empty lines only for cleaner console output.
    filtered = [line for line in lines if line.strip()]
    return logs_proc.returncode, filtered


def tcp_endpoint_ready(host: str, port: int, timeout_sec: float = 3.0) -> Tuple[bool, str]:
    host_str = str(host or "").strip()
    try:
        port_int = int(port)
    except Exception:
        port_int = 0
    if not host_str or port_int <= 0:
        return False, "missing host/port"
    try:
        with socket.create_connection((host_str, port_int), timeout=max(0.5, float(timeout_sec))):
            return True, ""
    except Exception as exc:
        return False, str(exc)


def ssh_auth_ready(
    host: str,
    port: int,
    user: str,
    identity_path: str,
    timeout_sec: float = 5.0,
) -> Tuple[bool, str]:
    host_str = str(host or "").strip()
    user_str = str(user or "").strip()
    identity_str = str(identity_path or "").strip()
    try:
        port_int = int(port)
    except Exception:
        port_int = 0
    if not host_str or port_int <= 0 or not user_str:
        return False, "missing ssh auth probe params"
    if not identity_str:
        return False, "missing identity key"

    cmd = [
        "ssh",
        "-p",
        str(port_int),
        "-i",
        identity_str,
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={max(1, int(timeout_sec))}",
        "-o",
        "StrictHostKeyChecking=accept-new",
        f"{user_str}@{host_str}",
        "echo vast_worker_auth_ready",
    ]
    proc = run_cmd(cmd, check=False, capture=True, log_command=False)
    output = (proc.stdout or "").strip()
    if proc.returncode == 0:
        return True, ""
    if output:
        return False, output.splitlines()[-1].strip()
    return False, f"ssh exit {proc.returncode}"


def wait_for_instance_ready(
    instance_id: int,
    api_key: str,
    timeout_sec: int,
    poll_sec: int,
    *,
    ssh_user: str = "root",
    identity_path: str = "",
) -> Tuple[str, str, int]:
    deadline = time.time() + timeout_sec
    last_status = ""
    last_probe_error = ""
    last_auth_error = ""
    last_log_lines: List[str] = []
    last_log_error = ""
    while time.time() < deadline:
        row = fetch_instance_row_http(instance_id, api_key)
        if row is None:
            row = fetch_instance_row_cli(instance_id, api_key)

        status_text = extract_instance_status_from_row(row) if isinstance(row, dict) else "unknown"
        ssh_host, ssh_port = extract_ssh_from_instance_row(row) if isinstance(row, dict) else ("", 0)

        if isinstance(row, dict) and (not ssh_host or ssh_port <= 0):
            ssh_cmd = with_api_key(
                ["vastai", "ssh-url", str(instance_id)],
                api_key,
            )
            ssh_proc = run_cmd(ssh_cmd, check=False, capture=True)
            ssh_raw = (ssh_proc.stdout or "").strip()
            _, ssh_host, ssh_port = parse_ssh_url(ssh_raw)

        if status_text != last_status:
            log(f"Instance {instance_id} status: {status_text}")
            last_status = status_text

        log_rc, log_lines = fetch_instance_logs_lines(instance_id, api_key)
        if log_rc == 0 and log_lines:
            # Print only unseen tail lines to avoid duplicating full log output each poll.
            new_lines: List[str] = []
            if last_log_lines and len(log_lines) >= len(last_log_lines) and log_lines[: len(last_log_lines)] == last_log_lines:
                new_lines = log_lines[len(last_log_lines):]
            elif last_log_lines:
                anchor = last_log_lines[-1]
                anchor_idx = -1
                for idx in range(len(log_lines) - 1, -1, -1):
                    if log_lines[idx] == anchor:
                        anchor_idx = idx
                        break
                if anchor_idx >= 0:
                    new_lines = log_lines[anchor_idx + 1:]
                else:
                    new_lines = log_lines
            else:
                new_lines = log_lines

            for line in new_lines:
                log(f"[instance-log] {line}")
            last_log_lines = log_lines
            last_log_error = ""
        elif log_rc != 0 and log_lines:
            log_error = log_lines[-1]
            if log_error != last_log_error:
                log(f"Instance {instance_id} log poll warning: {log_error}")
                last_log_error = log_error

        if ssh_host and ssh_port > 0:
            if identity_path:
                auth_ready, auth_error = ssh_auth_ready(
                    host=ssh_host,
                    port=ssh_port,
                    user=ssh_user,
                    identity_path=identity_path,
                    timeout_sec=5.0,
                )
                if auth_ready:
                    return ssh_user, ssh_host, ssh_port
                if auth_error and auth_error != last_auth_error:
                    log(
                        f"Instance {instance_id} has SSH endpoint {ssh_host}:{ssh_port} "
                        f"but auth is not ready yet: {auth_error}"
                    )
                    last_auth_error = auth_error
            else:
                is_open, probe_error = tcp_endpoint_ready(ssh_host, ssh_port, timeout_sec=3.0)
                if is_open:
                    return ssh_user, ssh_host, ssh_port
                if probe_error and probe_error != last_probe_error:
                    log(
                        f"Instance {instance_id} has SSH endpoint {ssh_host}:{ssh_port} "
                        f"but not accepting connections yet: {probe_error}"
                    )
                    last_probe_error = probe_error

        time.sleep(max(1, poll_sec))

    raise VastWorkerLaunchError(
        f"Timed out waiting for instance {instance_id} to become ready after {timeout_sec}s."
    )


def maybe_attach_identity_to_instance(instance_id: int, identity_path: str, api_key: str) -> None:
    identity_str = str(identity_path or "").strip()
    if not identity_str:
        return
    pub_path = Path(identity_str + ".pub")
    if not pub_path.is_file():
        log(f"Identity public key not found for auto-attach: {pub_path}. Skipping attach ssh.")
        return
    try:
        public_key_text = pub_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        log(f"Failed reading public key {pub_path}: {exc}. Skipping attach ssh.")
        return
    if not public_key_text:
        log(f"Public key file {pub_path} is empty. Skipping attach ssh.")
        return

    attach_cmd = with_api_key(
        ["vastai", "attach", "ssh", str(instance_id), public_key_text, "--raw"],
        api_key,
    )
    attach_proc = run_cmd(attach_cmd, check=False, capture=True, log_command=False)
    attach_output = (attach_proc.stdout or "").strip()
    if attach_proc.returncode == 0:
        log(f"Attached SSH key to instance {instance_id} using {pub_path}.")
        return
    if attach_output:
        short = attach_output.splitlines()[-1].strip()
    else:
        short = f"exit {attach_proc.returncode}"
    log(f"Warning: attach ssh for instance {instance_id} failed: {short}")


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
    p.add_argument(
        "--git-sync-branch",
        default="",
        help="Branch to force during remote git sync before running jobs. Default: current local branch.",
    )
    p.add_argument("--input-override", default="", help="Override config input_dir_or_file_var.")
    p.add_argument("--download-dir-override", default="", help="Override download/output dir for cloudctl.")
    p.add_argument("--batch-patterns", default="*.mkv,*.mp4,*.mov,*.avi")

    p.add_argument("--offer-limit", type=int, default=30)
    p.add_argument(
        "--blacklist-file",
        default=str(REPO_ROOT / "cloud" / "cloud_blacklist.json"),
        help="JSON file with blocked offer/machine/host ids.",
    )
    p.add_argument("--show-top", type=int, default=8)
    p.add_argument("--offer-type", choices=["on-demand", "reserved", "bid"], default="on-demand")
    p.add_argument("--search-order", default="dph_total")
    p.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Allow unverified hosts in Vast offer search (default requires verified=true).",
    )
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
    p.add_argument(
        "--ghcr-env-file",
        default=str(REPO_ROOT / "cloud" / "ghcr.env"),
        help="Env file used to auto-build GHCR --login args for private images.",
    )
    p.add_argument(
        "--registry-login",
        default="",
        help="Optional explicit value passed to vastai create instance --login.",
    )
    p.add_argument(
        "--skip-image-login",
        action="store_true",
        help="Skip automatic private-registry login injection.",
    )

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
    identity_path = str(Path(args.identity).expanduser().resolve()) if str(args.identity or "").strip() else ""
    if identity_path and not Path(identity_path).is_file():
        raise VastWorkerLaunchError(f"SSH identity file not found: {identity_path}")

    requested_sync_branch = str(getattr(args, "git_sync_branch", "") or "").strip()
    if not requested_sync_branch:
        requested_sync_branch = detect_local_git_branch(REPO_ROOT)
    args.git_sync_branch = requested_sync_branch
    if requested_sync_branch:
        log(f"Remote git sync branch: {requested_sync_branch}")
    else:
        log("Remote git sync branch: auto-detect failed; cloudctl will use remote current branch.")

    api_key = resolve_api_key(args)
    if not api_key:
        log("No explicit Vast API key found; relying on vastai local login state.")

    hf_env_payload = resolve_hf_env_payload(args)
    query = build_search_query(profile, args)

    search_payload = run_offer_search(query, args, api_key)
    if not search_payload:
        diagnostics = []
        if not bool(args.allow_unverified):
            args_unverified = argparse.Namespace(**vars(args))
            args_unverified.allow_unverified = True
            query_unverified = build_search_query(profile, args_unverified)
            payload_unverified = run_offer_search(query_unverified, args_unverified, api_key)
            if payload_unverified:
                diagnostics.append(
                    f"{len(payload_unverified)} offers appear when verified-host filter is disabled (--allow-unverified)"
                )
        if float(args.max_dph) > 0.0:
            args_uncapped = argparse.Namespace(**vars(args))
            args_uncapped.max_dph = 0.0
            query_uncapped = build_search_query(profile, args_uncapped)
            payload_uncapped = run_offer_search(query_uncapped, args_uncapped, api_key)
            if payload_uncapped:
                diagnostics.append(
                    f"{len(payload_uncapped)} offers appear when max-dph cap is removed (--max-dph 0)"
                )
        message = f"No offers matched query for profile '{profile.key}'. Query: {query}"
        if diagnostics:
            message += " Possible blockers: " + "; ".join(diagnostics)
        raise VastWorkerLaunchError(message)

    blacklist = load_blacklist_file(args.blacklist_file)
    blocked_offer_count = len(blacklist.get("blocked_offer_ids", set()))
    blocked_machine_count = len(blacklist.get("blocked_machine_ids", set()))
    blocked_host_count = len(blacklist.get("blocked_host_ids", set()))
    skipped_blacklist_count = 0
    offers = []
    skipped_vram_count = 0
    for offer in search_payload:
        if offer_is_blacklisted(offer, blacklist):
            skipped_blacklist_count += 1
            continue
        # Vast may report 48GB-class GPUs as ~47.99 GiB after unit normalization.
        if normalized_gpu_ram_gb(offer) + GPU_RAM_TOLERANCE_GB < profile.min_gpu_ram_gb:
            skipped_vram_count += 1
            continue
        offers.append(annotate_offer(offer, args))
    if not offers:
        detail = (
            f"raw={len(search_payload)}, blacklisted={skipped_blacklist_count}, "
            f"vram_filtered={skipped_vram_count}, min_vram={profile.min_gpu_ram_gb:.1f}GB"
        )
        raise VastWorkerLaunchError(
            f"No offers matched query for profile '{profile.key}' after filters ({detail}). Query: {query}"
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
    log(f"Offer filter: verified hosts {'required' if not args.allow_unverified else 'optional'}.")
    log(
        f"Blacklist filter: offers={blocked_offer_count}, machines={blocked_machine_count}, "
        f"hosts={blocked_host_count}, skipped_now={skipped_blacklist_count}."
    )
    print_offer_table(offers, args.show_top)

    selected_offer_from_ranked_list = True
    if args.offer_id > 0:
        selected_offer = next((o for o in offers if as_int(o.get("id"), 0) == args.offer_id), None)
        if selected_offer is None:
            # Offers are highly dynamic. A user-selected offer can legitimately fall out of
            # the latest ranked window between GUI preflight and launch.
            selected_offer_from_ranked_list = False
            selected_offer = {
                "id": int(args.offer_id),
            }
            log(
                f"Selected --offer-id {int(args.offer_id)} is not in the latest ranked list "
                f"(offer may have shifted outside --offer-limit or market changed). "
                "Attempting direct create-instance with that offer id."
            )
    else:
        selected_offer = offers[0]

    selected_offer_id = as_int(selected_offer.get("id"), 0)
    if selected_offer_id <= 0:
        raise VastWorkerLaunchError("Could not determine a valid offer id for instance creation.")

    if selected_offer_from_ranked_list:
        selected_hourly = as_float(selected_offer.get("_hourly_cost"), 0.0)
        selected_total = as_float(selected_offer.get("_total_cost_est"), 0.0)
        selected_loc = str(selected_offer.get("geolocation", ""))
        selected_gpu = str(selected_offer.get("gpu_name", ""))
        log(
            f"Selected offer {selected_offer_id}: {selected_gpu} | {selected_loc} | "
            f"${selected_hourly:.3f}/hr | est total ${selected_total:.3f}"
        )
    else:
        log(f"Selected offer {selected_offer_id}: direct launch mode (details unavailable in current ranked window).")

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
    registry = registry_from_image_ref(args.image)
    login_arg = resolve_auto_registry_login_arg(args)
    if login_arg:
        log(f"Using registry login for {registry or 'custom registry'} (credentials hidden).")
        create_cmd += ["--login", login_arg]
    elif registry == "ghcr.io":
        log(
            "No GHCR login credentials found. Assuming image is public; "
            "private GHCR images will fail to pull."
        )
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
    maybe_attach_identity_to_instance(instance_id=instance_id, identity_path=identity_path, api_key=api_key)
    ssh_user, ssh_host, ssh_port = wait_for_instance_ready(
        instance_id=instance_id,
        api_key=api_key,
        timeout_sec=args.ready_timeout_sec,
        poll_sec=args.poll_sec,
        ssh_user=str(args.remote_user or "root"),
        identity_path=identity_path,
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
