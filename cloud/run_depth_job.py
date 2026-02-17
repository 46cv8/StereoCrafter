#!/usr/bin/env python3
"""Headless DepthCrafter job runner for remote cloud instances.

This script runs a single depth-estimation job and writes a status JSON so local
controllers can reliably detect success/failure and output paths.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt="%H:%M:%S")


def _coerce_multiple_of_8(value: int, label: str) -> int:
    if value <= 0:
        raise ValueError(f"{label} must be > 0.")
    rounded = int(round(float(value) / 8.0) * 8)
    return max(8, rounded)


def _safe_json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _query_nvidia_smi_totals_mib() -> Dict[int, Dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        return {}
    if proc.returncode != 0:
        return {}

    totals: Dict[int, Dict[str, Any]] = {}
    for raw_line in (proc.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            total_mib = float(parts[2])
        except Exception:
            continue
        totals[idx] = {
            "name": parts[1],
            "total_mib": total_mib,
        }
    return totals


def _query_nvidia_smi_snapshot() -> List[Dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []

    rows: List[Dict[str, Any]] = []
    for raw_line in (proc.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", 4)]
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
            used_mib = float(parts[2])
            total_mib = float(parts[3])
            util_gpu = float(parts[4])
        except Exception:
            continue
        rows.append(
            {
                "index": idx,
                "name": parts[1],
                "used_mib": used_mib,
                "total_mib": total_mib,
                "util_gpu_pct": util_gpu,
            }
        )
    return rows


class _NvidiaSmiPeakTracker:
    def __init__(self, interval_sec: float = 0.5):
        self.interval_sec = max(0.2, float(interval_sec))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._sample_count = 0
        self._peak_used_sum_mib = 0.0
        self._totals_mib_sum = 0.0
        self._per_device: Dict[int, Dict[str, Any]] = {}

    def _sample_once(self) -> None:
        rows = _query_nvidia_smi_snapshot()
        if not rows:
            return
        used_sum = 0.0
        totals_sum = 0.0
        with self._lock:
            self._sample_count += 1
            for row in rows:
                idx = int(row["index"])
                used_mib = float(row["used_mib"])
                total_mib = float(row["total_mib"])
                util_gpu = float(row["util_gpu_pct"])
                used_sum += max(0.0, used_mib)
                totals_sum += max(0.0, total_mib)

                existing = self._per_device.get(idx)
                if existing is None:
                    self._per_device[idx] = {
                        "index": idx,
                        "name": str(row["name"]),
                        "total_mib": float(total_mib),
                        "peak_used_mib": float(used_mib),
                        "peak_util_gpu_pct": float(util_gpu),
                    }
                else:
                    existing["name"] = str(row["name"])
                    if total_mib > 0.0:
                        existing["total_mib"] = float(total_mib)
                    if used_mib > float(existing.get("peak_used_mib", 0.0) or 0.0):
                        existing["peak_used_mib"] = float(used_mib)
                    if util_gpu > float(existing.get("peak_util_gpu_pct", 0.0) or 0.0):
                        existing["peak_util_gpu_pct"] = float(util_gpu)
            if totals_sum > 0.0:
                self._totals_mib_sum = max(self._totals_mib_sum, totals_sum)
            self._peak_used_sum_mib = max(self._peak_used_sum_mib, used_sum)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._sample_once()
            self._stop_event.wait(self.interval_sec)

    def start(self) -> bool:
        # Validate nvidia-smi availability with a quick probe.
        probe_rows = _query_nvidia_smi_snapshot()
        if not probe_rows:
            return False
        self._sample_once()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop_and_summary(self) -> Dict[str, Any]:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._sample_once()
        with self._lock:
            per_device = []
            for idx in sorted(self._per_device.keys()):
                item = self._per_device[idx]
                total_mib = float(item.get("total_mib", 0.0) or 0.0)
                peak_used_mib = float(item.get("peak_used_mib", 0.0) or 0.0)
                used_pct = (peak_used_mib / total_mib * 100.0) if total_mib > 0.0 else 0.0
                per_device.append(
                    {
                        "index": idx,
                        "name": str(item.get("name", f"cuda:{idx}")),
                        "total_mib": round(total_mib, 3),
                        "peak_used_mib": round(peak_used_mib, 3),
                        "peak_used_pct_total": round(used_pct, 3),
                        "peak_util_gpu_pct": round(float(item.get("peak_util_gpu_pct", 0.0) or 0.0), 3),
                    }
                )
            totals_sum = float(self._totals_mib_sum)
            peak_used_sum = float(self._peak_used_sum_mib)
            peak_used_pct_total = (peak_used_sum / totals_sum * 100.0) if totals_sum > 0.0 else 0.0
            return {
                "device_count": int(len(per_device)),
                "sample_count": int(self._sample_count),
                "sample_interval_sec": round(float(self.interval_sec), 3),
                "per_device": per_device,
                "peak_used_mib_all_devices": round(peak_used_sum, 3),
                "total_mib_all_devices": round(totals_sum, 3),
                "peak_used_pct_total_all_devices": round(peak_used_pct_total, 3),
            }


def _compute_gpu_memory_stats(torch_module: Any, gpu_totals_mib: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    if torch_module is None:
        return {}
    if not getattr(torch_module, "cuda", None):
        return {}
    if not torch_module.cuda.is_available():
        return {}

    device_count = int(torch_module.cuda.device_count())
    per_device: List[Dict[str, Any]] = []
    peak_alloc_all = 0.0
    peak_reserved_all = 0.0
    total_all = 0.0

    for idx in range(device_count):
        try:
            name_default = str(torch_module.cuda.get_device_name(idx))
        except Exception:
            name_default = f"cuda:{idx}"
        totals_entry = gpu_totals_mib.get(idx, {})
        total_mib = float(totals_entry.get("total_mib", 0.0) or 0.0)
        name = str(totals_entry.get("name", name_default) or name_default)

        try:
            peak_alloc_mib = float(torch_module.cuda.max_memory_allocated(idx)) / (1024.0 ** 2)
        except Exception:
            peak_alloc_mib = 0.0
        try:
            peak_reserved_mib = float(torch_module.cuda.max_memory_reserved(idx)) / (1024.0 ** 2)
        except Exception:
            peak_reserved_mib = 0.0
        try:
            current_alloc_mib = float(torch_module.cuda.memory_allocated(idx)) / (1024.0 ** 2)
        except Exception:
            current_alloc_mib = 0.0
        try:
            current_reserved_mib = float(torch_module.cuda.memory_reserved(idx)) / (1024.0 ** 2)
        except Exception:
            current_reserved_mib = 0.0

        peak_alloc_all = max(peak_alloc_all, peak_alloc_mib)
        peak_reserved_all = max(peak_reserved_all, peak_reserved_mib)
        if total_mib > 0.0:
            total_all += total_mib

        alloc_pct = (peak_alloc_mib / total_mib * 100.0) if total_mib > 0.0 else 0.0
        reserved_pct = (peak_reserved_mib / total_mib * 100.0) if total_mib > 0.0 else 0.0
        per_device.append(
            {
                "index": idx,
                "name": name,
                "total_mib": round(total_mib, 3),
                "peak_alloc_mib": round(peak_alloc_mib, 3),
                "peak_reserved_mib": round(peak_reserved_mib, 3),
                "peak_alloc_pct_total": round(alloc_pct, 3),
                "peak_reserved_pct_total": round(reserved_pct, 3),
                "current_alloc_mib": round(current_alloc_mib, 3),
                "current_reserved_mib": round(current_reserved_mib, 3),
            }
        )

    return {
        "device_count": device_count,
        "per_device": per_device,
        "peak_alloc_mib_all_devices": round(peak_alloc_all, 3),
        "peak_reserved_mib_all_devices": round(peak_reserved_all, 3),
        "total_mib_all_devices": round(total_all, 3),
        "peak_alloc_pct_total_all_devices": round((peak_alloc_all / total_all * 100.0), 3) if total_all > 0.0 else 0.0,
        "peak_reserved_pct_total_all_devices": round((peak_reserved_all / total_all * 100.0), 3)
        if total_all > 0.0
        else 0.0,
    }


def _collect_stage_gpu_snapshot(torch_module: Any, gpu_totals_mib: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Collect a point-in-time GPU snapshot (nvidia-smi + torch CUDA memory)."""
    snapshot: Dict[str, Any] = {
        "nvidia_smi": {},
        "torch_cuda": {},
    }

    rows = _query_nvidia_smi_snapshot()
    if rows:
        per_device = []
        used_sum = 0.0
        total_sum = 0.0
        util_sum = 0.0
        for row in rows:
            idx = int(row.get("index", 0) or 0)
            used_mib = float(row.get("used_mib", 0.0) or 0.0)
            total_mib = float(row.get("total_mib", 0.0) or 0.0)
            util_gpu_pct = float(row.get("util_gpu_pct", 0.0) or 0.0)
            used_sum += used_mib
            total_sum += total_mib
            util_sum += util_gpu_pct
            per_device.append(
                {
                    "index": idx,
                    "name": str(row.get("name", gpu_totals_mib.get(idx, {}).get("name", f"cuda:{idx}"))),
                    "used_mib": round(used_mib, 3),
                    "total_mib": round(total_mib, 3),
                    "used_pct_total": round((used_mib / total_mib * 100.0), 3) if total_mib > 0.0 else 0.0,
                    "util_gpu_pct": round(util_gpu_pct, 3),
                }
            )
        snapshot["nvidia_smi"] = {
            "device_count": len(per_device),
            "per_device": per_device,
            "used_mib_all_devices": round(used_sum, 3),
            "total_mib_all_devices": round(total_sum, 3),
            "used_pct_total_all_devices": round((used_sum / total_sum * 100.0), 3) if total_sum > 0.0 else 0.0,
            "util_gpu_pct_mean": round(util_sum / len(per_device), 3) if per_device else 0.0,
        }

    if torch_module is not None and getattr(torch_module, "cuda", None) and torch_module.cuda.is_available():
        try:
            device_count = int(torch_module.cuda.device_count())
        except Exception:
            device_count = 0
        per_device_torch = []
        alloc_sum = 0.0
        reserved_sum = 0.0
        total_sum = 0.0
        for idx in range(device_count):
            totals_entry = gpu_totals_mib.get(idx, {})
            total_mib = float(totals_entry.get("total_mib", 0.0) or 0.0)
            try:
                alloc_mib = float(torch_module.cuda.memory_allocated(idx)) / (1024.0 ** 2)
            except Exception:
                alloc_mib = 0.0
            try:
                reserved_mib = float(torch_module.cuda.memory_reserved(idx)) / (1024.0 ** 2)
            except Exception:
                reserved_mib = 0.0
            try:
                peak_alloc_mib = float(torch_module.cuda.max_memory_allocated(idx)) / (1024.0 ** 2)
            except Exception:
                peak_alloc_mib = 0.0
            try:
                peak_reserved_mib = float(torch_module.cuda.max_memory_reserved(idx)) / (1024.0 ** 2)
            except Exception:
                peak_reserved_mib = 0.0
            alloc_sum += alloc_mib
            reserved_sum += reserved_mib
            total_sum += total_mib
            per_device_torch.append(
                {
                    "index": idx,
                    "name": str(totals_entry.get("name", f"cuda:{idx}")),
                    "alloc_mib": round(alloc_mib, 3),
                    "reserved_mib": round(reserved_mib, 3),
                    "peak_alloc_mib": round(peak_alloc_mib, 3),
                    "peak_reserved_mib": round(peak_reserved_mib, 3),
                    "total_mib": round(total_mib, 3),
                }
            )
        snapshot["torch_cuda"] = {
            "device_count": device_count,
            "per_device": per_device_torch,
            "alloc_mib_all_devices": round(alloc_sum, 3),
            "reserved_mib_all_devices": round(reserved_sum, 3),
            "total_mib_all_devices": round(total_sum, 3),
            "alloc_pct_total_all_devices": round((alloc_sum / total_sum * 100.0), 3) if total_sum > 0.0 else 0.0,
            "reserved_pct_total_all_devices": round((reserved_sum / total_sum * 100.0), 3) if total_sum > 0.0 else 0.0,
        }

    return snapshot


def _log_stage_gpu_snapshot(
    logger: logging.Logger,
    *,
    stage: str,
    torch_module: Any,
    gpu_totals_mib: Dict[int, Dict[str, Any]],
    stage_log_store: List[Dict[str, Any]],
    job_start_ts: float,
    payload: Dict[str, Any] | None = None,
) -> None:
    now_ts = time.time()
    snapshot = _collect_stage_gpu_snapshot(torch_module, gpu_totals_mib)
    entry: Dict[str, Any] = {
        "stage": stage,
        "time_unix": round(now_ts, 6),
        "elapsed_seconds": round(now_ts - job_start_ts, 3),
        "snapshot": snapshot,
    }
    if payload:
        entry["payload"] = payload
    stage_log_store.append(entry)

    nvsmi = snapshot.get("nvidia_smi", {}) if isinstance(snapshot, dict) else {}
    torch_cuda = snapshot.get("torch_cuda", {}) if isinstance(snapshot, dict) else {}
    nvsmi_used = float(nvsmi.get("used_mib_all_devices", 0.0) or 0.0)
    nvsmi_pct = float(nvsmi.get("used_pct_total_all_devices", 0.0) or 0.0)
    nvsmi_util = float(nvsmi.get("util_gpu_pct_mean", 0.0) or 0.0)
    torch_alloc = float(torch_cuda.get("alloc_mib_all_devices", 0.0) or 0.0)
    torch_reserved = float(torch_cuda.get("reserved_mib_all_devices", 0.0) or 0.0)
    torch_alloc_pct = float(torch_cuda.get("alloc_pct_total_all_devices", 0.0) or 0.0)
    torch_reserved_pct = float(torch_cuda.get("reserved_pct_total_all_devices", 0.0) or 0.0)

    logger.info(
        "GPU stage snapshot | stage=%s | t=%.2fs | nvidia_smi used=%.1f MiB (%.2f%%) util_mean=%.1f%% | torch alloc=%.1f MiB (%.2f%%) reserved=%.1f MiB (%.2f%%)",
        stage,
        now_ts - job_start_ts,
        nvsmi_used,
        nvsmi_pct,
        nvsmi_util,
        torch_alloc,
        torch_alloc_pct,
        torch_reserved,
        torch_reserved_pct,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one headless DepthCrafter job.")
    parser.add_argument("--input", required=False, help="Input video path on remote machine.")
    parser.add_argument("--output-dir", required=True, help="Output folder for depth results.")
    parser.add_argument("--status-json", default="", help="Path for status JSON output.")
    parser.add_argument("--job-name", default="", help="Optional output basename override.")

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

    parser.add_argument(
        "--prewarm-only",
        action="store_true",
        help="Only initialize models then exit. Useful for cache warmup.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def _resolve_job_name(args: argparse.Namespace) -> str:
    if args.job_name:
        return args.job_name
    if args.input:
        return Path(args.input).stem
    return f"prewarm_{int(time.time())}"


def _status_path(args: argparse.Namespace, output_dir: Path, job_name: str) -> Path:
    if args.status_json:
        return Path(args.status_json).expanduser().resolve()
    return output_dir / f"{job_name}_status.json"


def _validate_runtime_args(args: argparse.Namespace) -> None:
    if args.prewarm_only:
        return
    if not args.input:
        raise ValueError("--input is required unless --prewarm-only is set.")
    if args.window_size <= 0:
        raise ValueError("--window-size must be > 0.")
    if args.overlap < 0:
        raise ValueError("--overlap must be >= 0.")
    if args.overlap >= args.window_size:
        raise ValueError("--overlap must be < --window-size.")
    if args.inference_steps <= 0:
        raise ValueError("--inference-steps must be > 0.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    logger = logging.getLogger("cloud.run_depth_job")

    start_ts = time.time()
    job_name = _resolve_job_name(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_json_path = _status_path(args, output_dir, job_name)

    status: Dict[str, Any] = {
        "job_name": job_name,
        "status": "running",
        "start_time_unix": start_ts,
        "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_ts)),
        "input": str(Path(args.input).expanduser().resolve()) if args.input else "",
        "output_dir": str(output_dir),
        "params": {
            "target_width": args.target_width,
            "target_height": args.target_height,
            "window_size": args.window_size,
            "overlap": args.overlap,
            "inference_steps": args.inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "target_fps": args.target_fps,
            "process_length": args.process_length,
            "output_format": args.output_format,
            "cpu_offload": args.cpu_offload,
            "disable_xformers": bool(args.disable_xformers),
            "local_files_only": bool(args.local_files_only),
            "unet_path": args.unet_path,
            "pretrain_path": args.pretrain_path,
        },
    }
    stage_gpu_samples: List[Dict[str, Any]] = []
    status["stage_gpu_samples"] = stage_gpu_samples
    torch_module = None
    gpu_totals_mib: Dict[int, Dict[str, Any]] = {}
    nvidia_peak_tracker: _NvidiaSmiPeakTracker | None = None

    try:
        _validate_runtime_args(args)
        from depthcrafter.depthcrafter_logic import DepthCrafterDemo
        import torch as torch_module
        _log_stage_gpu_snapshot(
            logger,
            stage="runtime_imports_ready",
            torch_module=torch_module,
            gpu_totals_mib=gpu_totals_mib,
            stage_log_store=stage_gpu_samples,
            job_start_ts=start_ts,
        )

        target_w = _coerce_multiple_of_8(int(args.target_width), "target width")
        target_h = _coerce_multiple_of_8(int(args.target_height), "target height")

        if target_w != int(args.target_width) or target_h != int(args.target_height):
            logger.warning(
                "Adjusted target resolution from %sx%s to %sx%s to satisfy /8 model constraints.",
                args.target_width,
                args.target_height,
                target_w,
                target_h,
            )

        _log_stage_gpu_snapshot(
            logger,
            stage="model_init_start",
            torch_module=torch_module,
            gpu_totals_mib=gpu_totals_mib,
            stage_log_store=stage_gpu_samples,
            job_start_ts=start_ts,
            payload={"target_width": int(target_w), "target_height": int(target_h)},
        )
        logger.info("Initializing DepthCrafter model...")
        demo = DepthCrafterDemo(
            unet_path=args.unet_path,
            pre_train_path=args.pretrain_path,
            cpu_offload=args.cpu_offload,
            use_cudnn_benchmark=bool(args.use_cudnn_benchmark),
            local_files_only=bool(args.local_files_only),
            disable_xformers=bool(args.disable_xformers),
        )
        _log_stage_gpu_snapshot(
            logger,
            stage="model_init_end",
            torch_module=torch_module,
            gpu_totals_mib=gpu_totals_mib,
            stage_log_store=stage_gpu_samples,
            job_start_ts=start_ts,
        )

        if torch_module.cuda.is_available():
            gpu_totals_mib = _query_nvidia_smi_totals_mib()
            device_count = int(torch_module.cuda.device_count())
            for idx in range(device_count):
                try:
                    torch_module.cuda.reset_peak_memory_stats(idx)
                except Exception:
                    continue
            logger.info("GPU peak VRAM tracking enabled for %d CUDA device(s).", device_count)
        nvidia_peak_tracker = _NvidiaSmiPeakTracker(interval_sec=0.5)
        if nvidia_peak_tracker.start():
            logger.info("nvidia-smi peak VRAM sampling enabled (interval=%.2fs).", nvidia_peak_tracker.interval_sec)
        else:
            nvidia_peak_tracker = None
        _log_stage_gpu_snapshot(
            logger,
            stage="tracker_start_end",
            torch_module=torch_module,
            gpu_totals_mib=gpu_totals_mib,
            stage_log_store=stage_gpu_samples,
            job_start_ts=start_ts,
        )

        def _runtime_stage_callback(stage: str, payload: Dict[str, Any]) -> None:
            _log_stage_gpu_snapshot(
                logger,
                stage=stage,
                torch_module=torch_module,
                gpu_totals_mib=gpu_totals_mib,
                stage_log_store=stage_gpu_samples,
                job_start_ts=start_ts,
                payload=payload,
            )

        demo.runtime_stage_callback = _runtime_stage_callback

        if args.prewarm_only:
            status["status"] = "success"
            status["message"] = "Model prewarm completed successfully."
            status["end_time_unix"] = time.time()
            status["duration_seconds"] = round(status["end_time_unix"] - start_ts, 3)
            _safe_json_dump(status_json_path, status)
            logger.info("Prewarm complete. Status JSON: %s", status_json_path)
            return 0

        input_path = Path(args.input).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input clip not found: {input_path}")

        logger.info(
            "Running depth job '%s' | input=%s | target=%sx%s | window/overlap=%s/%s | steps=%s | output=%s",
            job_name,
            input_path,
            target_w,
            target_h,
            args.window_size,
            args.overlap,
            args.inference_steps,
            args.output_format,
        )
        _log_stage_gpu_snapshot(
            logger,
            stage="demo_run_start",
            torch_module=torch_module,
            gpu_totals_mib=gpu_totals_mib,
            stage_log_store=stage_gpu_samples,
            job_start_ts=start_ts,
            payload={"input": str(input_path)},
        )

        save_path, metadata = demo.run(
            video_path_or_frames_or_info=str(input_path),
            num_denoising_steps=int(args.inference_steps),
            guidance_scale=float(args.guidance_scale),
            base_output_folder=str(output_dir),
            gui_window_size=int(args.window_size),
            gui_overlap=int(args.overlap),
            process_length_for_read_full_video=int(args.process_length),
            target_height=int(target_h),
            target_width=int(target_w),
            seed=int(args.seed),
            original_video_basename_override=job_name,
            segment_job_info_param=None,
            keep_intermediate_npz_config=False,
            intermediate_segment_visual_format_config="none",
            save_final_json_for_this_job_config=True,
            full_video_output_format=str(args.output_format),
        )
        _log_stage_gpu_snapshot(
            logger,
            stage="demo_run_end",
            torch_module=torch_module,
            gpu_totals_mib=gpu_totals_mib,
            stage_log_store=stage_gpu_samples,
            job_start_ts=start_ts,
            payload={"save_path": str(save_path) if save_path else ""},
        )

        status["metadata"] = metadata
        status["save_path"] = str(save_path) if save_path else ""
        status["metadata_json"] = metadata.get("_individual_metadata_path") if isinstance(metadata, dict) else None

        if not save_path:
            run_state = metadata.get("status") if isinstance(metadata, dict) else "unknown"
            raise RuntimeError(f"DepthCrafter did not return output path (status={run_state}).")

        status["status"] = "success"
        status["message"] = "Depth job completed successfully."
        logger.info("Depth job completed. Output: %s", save_path)

    except Exception as exc:  # pylint: disable=broad-except
        status["status"] = "failed"
        status["message"] = str(exc)
        status["traceback"] = traceback.format_exc()
        logging.getLogger("cloud.run_depth_job").exception("Depth job failed: %s", exc)
    finally:
        _log_stage_gpu_snapshot(
            logger,
            stage="finalize_start",
            torch_module=torch_module,
            gpu_totals_mib=gpu_totals_mib,
            stage_log_store=stage_gpu_samples,
            job_start_ts=start_ts,
        )
        nvidia_smi_stats: Dict[str, Any] = {}
        if nvidia_peak_tracker is not None:
            try:
                nvidia_smi_stats = nvidia_peak_tracker.stop_and_summary()
            except Exception:
                nvidia_smi_stats = {}
        if nvidia_smi_stats and int(nvidia_smi_stats.get("device_count", 0) or 0) > 0:
            status["gpu_memory_nvidia_smi"] = nvidia_smi_stats
            logging.getLogger("cloud.run_depth_job").info(
                "GPU peak VRAM summary (nvidia-smi) | used=%.1f MiB (%.2f%% total), devices=%d, samples=%d",
                float(nvidia_smi_stats.get("peak_used_mib_all_devices", 0.0) or 0.0),
                float(nvidia_smi_stats.get("peak_used_pct_total_all_devices", 0.0) or 0.0),
                int(nvidia_smi_stats.get("device_count", 0) or 0),
                int(nvidia_smi_stats.get("sample_count", 0) or 0),
            )
            for device_entry in nvidia_smi_stats.get("per_device", []):
                if not isinstance(device_entry, dict):
                    continue
                logging.getLogger("cloud.run_depth_job").info(
                    "GPU%d peak (nvidia-smi) | used=%.1f MiB (%.2f%%), total=%.1f MiB, util_peak=%.1f%%, name=%s",
                    int(device_entry.get("index", 0) or 0),
                    float(device_entry.get("peak_used_mib", 0.0) or 0.0),
                    float(device_entry.get("peak_used_pct_total", 0.0) or 0.0),
                    float(device_entry.get("total_mib", 0.0) or 0.0),
                    float(device_entry.get("peak_util_gpu_pct", 0.0) or 0.0),
                    str(device_entry.get("name", "")),
                )

        gpu_memory_stats = _compute_gpu_memory_stats(torch_module, gpu_totals_mib)
        if gpu_memory_stats:
            status["gpu_memory"] = gpu_memory_stats
            logging.getLogger("cloud.run_depth_job").info(
                "GPU peak VRAM summary | alloc=%.1f MiB (%.2f%% total), reserved=%.1f MiB (%.2f%% total), devices=%d",
                float(gpu_memory_stats.get("peak_alloc_mib_all_devices", 0.0) or 0.0),
                float(gpu_memory_stats.get("peak_alloc_pct_total_all_devices", 0.0) or 0.0),
                float(gpu_memory_stats.get("peak_reserved_mib_all_devices", 0.0) or 0.0),
                float(gpu_memory_stats.get("peak_reserved_pct_total_all_devices", 0.0) or 0.0),
                int(gpu_memory_stats.get("device_count", 0) or 0),
            )
            for device_entry in gpu_memory_stats.get("per_device", []):
                if not isinstance(device_entry, dict):
                    continue
                logging.getLogger("cloud.run_depth_job").info(
                    "GPU%d peak | alloc=%.1f MiB (%.2f%%), reserved=%.1f MiB (%.2f%%), total=%.1f MiB, name=%s",
                    int(device_entry.get("index", 0) or 0),
                    float(device_entry.get("peak_alloc_mib", 0.0) or 0.0),
                    float(device_entry.get("peak_alloc_pct_total", 0.0) or 0.0),
                    float(device_entry.get("peak_reserved_mib", 0.0) or 0.0),
                    float(device_entry.get("peak_reserved_pct_total", 0.0) or 0.0),
                    float(device_entry.get("total_mib", 0.0) or 0.0),
                    str(device_entry.get("name", "")),
                )
        end_ts = time.time()
        status["end_time_unix"] = end_ts
        status["end_time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_ts))
        status["duration_seconds"] = round(end_ts - start_ts, 3)
        _safe_json_dump(status_json_path, status)
        logging.getLogger("cloud.run_depth_job").info("Wrote status JSON: %s", status_json_path)

    return 0 if status.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
