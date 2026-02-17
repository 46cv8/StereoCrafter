from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_VAST_API_BASE_URL = "https://console.vast.ai"
DEFAULT_CLOUD_PROFILE_KEY = "5090_32gb"
GPU_RAM_TOLERANCE_GB = 0.25

CLOUD_PROFILE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "5090_32gb": {
        "label": "RTX 5090 32GB",
        "offer_gpu_filter": "gpu_name=RTX_5090",
        "min_gpu_ram_gb": 30.0,
        "target_width": 1664,
        "target_height": 896,
        "window_size": 75,
        "overlap": 25,
        "use_source_resolution": False,
    },
    "rtx_pro_6000_96gb": {
        "label": "RTX PRO 6000 96GB",
        "offer_gpu_filter": "gpu_name in [RTX_PRO_6000_WS,RTX_PRO_6000_S]",
        "min_gpu_ram_gb": 92.0,
        "target_width": 1920,
        "target_height": 1040,
        "window_size": 75,
        "overlap": 25,
        "use_source_resolution": False,
    },
    "nvidia_48gb_single": {
        "label": "Any NVIDIA 48GB+ (Input Res)",
        # Empty filter means "all offers", then CUDA/VRAM guards narrow results.
        "offer_gpu_filter": "",
        "min_gpu_ram_gb": 48.0,
        "target_width": 1920,
        "target_height": 1040,
        "window_size": 75,
        "overlap": 25,
        "use_source_resolution": True,
    },
}


def get_cloud_profile_defaults(profile_key: Optional[str] = None) -> Dict[str, Any]:
    requested_key = str(profile_key or "").strip() or DEFAULT_CLOUD_PROFILE_KEY
    key = requested_key if requested_key in CLOUD_PROFILE_DEFAULTS else DEFAULT_CLOUD_PROFILE_KEY
    defaults = dict(CLOUD_PROFILE_DEFAULTS[key])
    defaults["key"] = key
    defaults.setdefault("use_source_resolution", False)
    return defaults


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


def normalize_blacklist_data(raw_data: Optional[Dict[str, Any]]) -> Dict[str, set]:
    normalized = {
        "blocked_offer_ids": set(),
        "blocked_machine_ids": set(),
        "blocked_host_ids": set(),
    }
    if not isinstance(raw_data, dict):
        return normalized

    key_aliases = {
        "blocked_offer_ids": ("blocked_offer_ids", "offer_ids"),
        "blocked_machine_ids": ("blocked_machine_ids", "machine_ids"),
        "blocked_host_ids": ("blocked_host_ids", "host_ids"),
    }
    for out_key, aliases in key_aliases.items():
        for key in aliases:
            values = raw_data.get(key)
            if isinstance(values, list):
                for value in values:
                    try:
                        value_int = int(value)
                    except Exception:
                        continue
                    if value_int > 0:
                        normalized[out_key].add(value_int)
    return normalized


def load_blacklist_data(path_value: str) -> Dict[str, set]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        return normalize_blacklist_data(None)
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw_data = json.load(fh)
    except Exception:
        return normalize_blacklist_data(None)
    return normalize_blacklist_data(raw_data)


def save_blacklist_data(path_value: str, blacklist_data: Dict[str, set], updated_by: str = "cloud/cloud_core.py") -> None:
    path = Path(path_value).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "blocked_offer_ids": sorted(int(x) for x in blacklist_data.get("blocked_offer_ids", set()) if int(x) > 0),
        "blocked_machine_ids": sorted(int(x) for x in blacklist_data.get("blocked_machine_ids", set()) if int(x) > 0),
        "blocked_host_ids": sorted(int(x) for x in blacklist_data.get("blocked_host_ids", set()) if int(x) > 0),
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated_by": updated_by,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2)


def normalize_provider_history_data(raw_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    def _normalize_count_map(value: Any) -> Dict[str, int]:
        normalized_map: Dict[str, int] = {}
        if not isinstance(value, dict):
            return normalized_map
        for key, count_value in value.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            try:
                count_int = int(count_value)
            except Exception:
                continue
            if count_int > 0:
                normalized_map[key_text] = count_int
        return normalized_map

    normalized = {
        "provider_counts": {},
        "offer_counts": {},
        "machine_counts": {},
        "host_counts": {},
        "recent_connections": [],
    }
    if not isinstance(raw_data, dict):
        return normalized

    normalized["provider_counts"] = _normalize_count_map(raw_data.get("provider_counts"))
    normalized["offer_counts"] = _normalize_count_map(raw_data.get("offer_counts"))
    normalized["machine_counts"] = _normalize_count_map(raw_data.get("machine_counts"))
    normalized["host_counts"] = _normalize_count_map(raw_data.get("host_counts"))

    recent = raw_data.get("recent_connections")
    if isinstance(recent, list):
        normalized["recent_connections"] = [row for row in recent[-200:] if isinstance(row, dict)]

    return normalized


def load_provider_history_data(path_value: str) -> Dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        return normalize_provider_history_data(None)
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw_data = json.load(fh)
    except Exception:
        return normalize_provider_history_data(None)
    return normalize_provider_history_data(raw_data)


def save_provider_history_data(path_value: str, history_data: Dict[str, Any], updated_by: str = "cloud/cloud_core.py") -> None:
    path = Path(path_value).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    def _sort_map(source: Dict[str, Any]) -> Dict[str, int]:
        sortable: List[Tuple[str, int]] = []
        for key, value in source.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            try:
                value_int = int(value)
            except Exception:
                continue
            if value_int > 0:
                sortable.append((key_text, value_int))
        sortable.sort(key=lambda item: item[0])
        return {key: value for key, value in sortable}

    serializable = {
        "provider_counts": _sort_map(history_data.get("provider_counts", {})),
        "offer_counts": _sort_map(history_data.get("offer_counts", {})),
        "machine_counts": _sort_map(history_data.get("machine_counts", {})),
        "host_counts": _sort_map(history_data.get("host_counts", {})),
        "recent_connections": list(history_data.get("recent_connections", []))[-200:],
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated_by": updated_by,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2)


def cloud_provider_key_from_ids(offer_id: int, machine_id: int, host_id: int, host: str = "") -> str:
    if host_id > 0:
        return f"host:{host_id}"
    if machine_id > 0:
        return f"machine:{machine_id}"
    if offer_id > 0:
        return f"offer:{offer_id}"
    host_text = str(host or "").strip().lower()
    if host_text:
        return f"ssh:{host_text}"
    return "unknown"


def cloud_provider_label_from_key(provider_key: str) -> str:
    key = str(provider_key or "").strip()
    if key.startswith("host:"):
        return f"Host {key.split(':', 1)[1]}"
    if key.startswith("machine:"):
        return f"Machine {key.split(':', 1)[1]}"
    if key.startswith("offer:"):
        return f"Offer {key.split(':', 1)[1]}"
    if key.startswith("ssh:"):
        return f"SSH {key.split(':', 1)[1]}"
    return "Unknown provider"


def cloud_provider_identity_for_offer(offer_entry: Dict[str, Any]) -> Dict[str, Any]:
    try:
        offer_id = int(offer_entry.get("id", 0) or 0)
    except Exception:
        offer_id = 0
    try:
        machine_id = int(offer_entry.get("machine_id", 0) or 0)
    except Exception:
        machine_id = 0
    try:
        host_id = int(offer_entry.get("host_id", 0) or 0)
    except Exception:
        host_id = 0

    provider_key = cloud_provider_key_from_ids(
        offer_id=offer_id,
        machine_id=machine_id,
        host_id=host_id,
    )
    return {
        "provider_key": provider_key,
        "provider_label": cloud_provider_label_from_key(provider_key),
        "offer_id": offer_id,
        "machine_id": machine_id,
        "host_id": host_id,
    }


def normalized_gpu_ram_gb(offer: Dict[str, Any]) -> float:
    try:
        raw = float(offer.get("gpu_ram", 0.0) or 0.0)
    except Exception:
        raw = 0.0
    if raw <= 0:
        return 0.0
    return raw / 1024.0 if raw > 1000.0 else raw


def offer_hourly_cost(offer: Dict[str, Any]) -> float:
    for key in ("dph_total", "discounted_dph_total", "dph"):
        if key in offer:
            try:
                return float(offer.get(key) or 0.0)
            except Exception:
                return 0.0
    return 0.0


def offer_cost_per_tb(offer: Dict[str, Any], direction: str) -> float:
    if direction == "up":
        if "internet_up_cost_per_tb" in offer:
            try:
                return float(offer.get("internet_up_cost_per_tb") or 0.0)
            except Exception:
                return 0.0
        try:
            return float(offer.get("inet_up_cost") or 0.0) * 1024.0
        except Exception:
            return 0.0
    if "internet_down_cost_per_tb" in offer:
        try:
            return float(offer.get("internet_down_cost_per_tb") or 0.0)
        except Exception:
            return 0.0
    try:
        return float(offer.get("inet_down_cost") or 0.0) * 1024.0
    except Exception:
        return 0.0


def estimate_offer_total_cost(
    offer: Dict[str, Any],
    expected_runtime_hours: float,
    expected_upload_gb: float,
    expected_download_gb: float,
) -> Dict[str, float]:
    runtime_hours = max(0.0, float(expected_runtime_hours))
    upload_gb = max(0.0, float(expected_upload_gb))
    download_gb = max(0.0, float(expected_download_gb))
    hourly = offer_hourly_cost(offer)
    up_tb = offer_cost_per_tb(offer, "up")
    down_tb = offer_cost_per_tb(offer, "down")
    transfer_cost = ((upload_gb / 1024.0) * up_tb) + ((download_gb / 1024.0) * down_tb)
    runtime_cost = hourly * runtime_hours
    total_cost = runtime_cost + transfer_cost
    return {
        "hourly": hourly,
        "up_tb": up_tb,
        "down_tb": down_tb,
        "runtime_cost": runtime_cost,
        "transfer_cost": transfer_cost,
        "total_cost": total_cost,
    }


def offer_is_blacklisted(offer_entry: Dict[str, Any], blacklist_data: Optional[Dict[str, set]]) -> bool:
    data = blacklist_data or normalize_blacklist_data(None)
    try:
        offer_id = int(offer_entry.get("id", 0) or 0)
    except Exception:
        offer_id = 0
    try:
        machine_id = int(offer_entry.get("machine_id", 0) or 0)
    except Exception:
        machine_id = 0
    try:
        host_id = int(offer_entry.get("host_id", 0) or 0)
    except Exception:
        host_id = 0
    return (
        (offer_id > 0 and offer_id in data.get("blocked_offer_ids", set()))
        or (machine_id > 0 and machine_id in data.get("blocked_machine_ids", set()))
        or (host_id > 0 and host_id in data.get("blocked_host_ids", set()))
    )


def build_offer_search_query(
    profile_defaults: Dict[str, Any],
    *,
    disk_gb: int,
    require_verified_hosts: bool,
    max_dph: float,
    min_cuda: float = 12.8,
    min_reliability: float = 0.97,
    min_direct_ports: int = 2,
    min_inet_down: float = 200.0,
    min_inet_up: float = 50.0,
) -> str:
    offer_gpu_filter = str(profile_defaults.get("offer_gpu_filter", "")).strip()
    min_gpu_ram_gb = max(0.0, float(profile_defaults.get("min_gpu_ram_gb", 0.0)))
    query_parts = [
        "rentable=true",
        "num_gpus=1",
        f"gpu_ram>={min_gpu_ram_gb:.1f}",
        f"cuda_vers>={float(min_cuda)}",
        f"reliability>={float(min_reliability)}",
        f"disk_space>={int(max(20, int(disk_gb)))}",
        f"direct_port_count>={int(max(1, int(min_direct_ports)))}",
        f"inet_down>={float(min_inet_down)}",
        f"inet_up>={float(min_inet_up)}",
    ]
    if offer_gpu_filter:
        query_parts.insert(0, offer_gpu_filter)
    if bool(require_verified_hosts):
        query_parts.append("verified=true")
    if float(max_dph) > 0.0:
        query_parts.append(f"dph<={float(max_dph)}")
    return " ".join(query_parts)


def rank_cloud_offers(
    parsed_payload: List[Dict[str, Any]],
    blacklist_data: Dict[str, set],
    *,
    min_gpu_ram_gb: float,
    expected_runtime_hours: float,
    expected_upload_gb: float,
    expected_download_gb: float,
    gpu_ram_tolerance_gb: float = GPU_RAM_TOLERANCE_GB,
) -> Tuple[List[Dict[str, Any]], int, int]:
    ranked_offers: List[Dict[str, Any]] = []
    skipped_blacklist_count = 0
    skipped_vram_count = 0

    for entry in parsed_payload:
        if offer_is_blacklisted(entry, blacklist_data):
            skipped_blacklist_count += 1
            continue
        vram_gb = normalized_gpu_ram_gb(entry)
        if vram_gb + float(gpu_ram_tolerance_gb) < float(min_gpu_ram_gb):
            skipped_vram_count += 1
            continue

        cost_data = estimate_offer_total_cost(
            entry,
            expected_runtime_hours=expected_runtime_hours,
            expected_upload_gb=expected_upload_gb,
            expected_download_gb=expected_download_gb,
        )
        enriched = dict(entry)
        enriched["_vram_gb"] = vram_gb
        enriched["_hourly"] = cost_data["hourly"]
        enriched["_up_tb"] = cost_data["up_tb"]
        enriched["_down_tb"] = cost_data["down_tb"]
        enriched["_runtime_cost"] = cost_data["runtime_cost"]
        enriched["_transfer_cost"] = cost_data["transfer_cost"]
        enriched["_total_cost"] = cost_data["total_cost"]
        ranked_offers.append(enriched)

    ranked_offers.sort(
        key=lambda offer: (
            float(offer.get("_total_cost", 1e12)),
            float(offer.get("_hourly", 1e12)),
            -float(offer.get("reliability", 0.0) or 0.0),
        )
    )
    return ranked_offers, skipped_blacklist_count, skipped_vram_count


def extract_instance_row_from_payload(payload: Any, instance_id: int) -> Optional[Dict[str, Any]]:
    target_id = int(instance_id)
    rows: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, dict)]
    elif isinstance(payload, dict):
        maybe_rows = payload.get("instances")
        if isinstance(maybe_rows, list):
            rows = [row for row in maybe_rows if isinstance(row, dict)]
        elif isinstance(maybe_rows, dict):
            rows = [maybe_rows]
        else:
            rows = [payload]

    for row in rows:
        try:
            row_id = int(row.get("id", -1))
        except Exception:
            continue
        if row_id == target_id:
            return row
    return None


def extract_instance_status_from_row(row: Dict[str, Any]) -> str:
    for key in ("actual_status", "status", "cur_state", "state", "status_msg", "intended_status"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def extract_ssh_from_instance_row(row: Dict[str, Any]) -> Tuple[str, int]:
    host = str(row.get("ssh_host", "") or "").strip()
    port_value = row.get("ssh_port")
    port = 0
    try:
        port = int(port_value)
    except Exception:
        port = 0

    ports_data = row.get("ports", {})
    used_22_map = False
    if isinstance(ports_data, dict):
        port_22_entries = ports_data.get("22/tcp")
        if isinstance(port_22_entries, list) and port_22_entries:
            first = port_22_entries[0]
            if isinstance(first, dict):
                host_port = first.get("HostPort")
                try:
                    mapped_port = int(host_port)
                except Exception:
                    mapped_port = 0
                if mapped_port > 0:
                    public_host = str(row.get("public_ipaddr", "") or "").strip()
                    if public_host:
                        host = public_host
                    port = mapped_port
                    used_22_map = True

    if not used_22_map and port > 0:
        runtype = str(row.get("image_runtype", "") or "").lower()
        if "jupyter" in runtype:
            port += 1

    if host and port > 0:
        return host, port
    return "", 0
