#!/usr/bin/env python3
"""Headless DepthCrafter job runner for remote cloud instances.

This script runs a single depth-estimation job and writes a status JSON so local
controllers can reliably detect success/failure and output paths.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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
_DEFAULT_GEOMETRY_REPO_URL = "https://github.com/TencentARC/GeometryCrafter.git"


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt="%H:%M:%S")


def _coerce_multiple_of_8(value: int, label: str) -> int:
    if value <= 0:
        raise ValueError(f"{label} must be > 0.")
    rounded = int(round(float(value) / 8.0) * 8)
    return max(8, rounded)


def _coerce_multiple(value: int, label: str, multiple: int) -> int:
    if multiple == 8:
        return _coerce_multiple_of_8(value, label)
    if value <= 0:
        raise ValueError(f"{label} must be > 0.")
    if multiple <= 0:
        raise ValueError("multiple must be > 0.")
    rounded = int(round(float(value) / float(multiple)) * float(multiple))
    return max(multiple, rounded)


def _safe_json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _run_cmd_checked(cmd: List[str], logger: logging.Logger, cwd: Path | None = None) -> None:
    pretty = " ".join(cmd)
    logger.info("[setup] $ %s", pretty)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        output = (proc.stdout or "").strip()
        raise RuntimeError(f"Command failed ({proc.returncode}): {pretty}\n{output}")


def _ensure_python_module(module_name: str, pip_spec: str, logger: logging.Logger) -> None:
    try:
        __import__(module_name)
        return
    except Exception:
        logger.warning("Missing module '%s'. Installing '%s'...", module_name, pip_spec)
    _run_cmd_checked([sys.executable, "-m", "pip", "install", pip_spec], logger=logger)
    try:
        __import__(module_name)
    except Exception as exc:
        raise RuntimeError(f"Module '{module_name}' is still unavailable after install.") from exc


def _is_geometry_repo(path: Path) -> bool:
    return (path / "geometrycrafter").is_dir() and (path / "third_party").is_dir()


def _resolve_geometry_repo_path(raw_value: str) -> Path:
    raw = str(raw_value or "").strip()
    if not raw:
        return (REPO_ROOT / "weights" / "GeometryCrafter").resolve()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _ensure_geometry_repo_available(raw_repo_path: str, logger: logging.Logger) -> Path:
    repo_path = _resolve_geometry_repo_path(raw_repo_path)
    if not repo_path.exists():
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        _run_cmd_checked(
            ["git", "clone", "--recursive", _DEFAULT_GEOMETRY_REPO_URL, str(repo_path)],
            logger=logger,
        )
    elif not _is_geometry_repo(repo_path):
        raise RuntimeError(
            f"Geometry repo path exists but is not a valid GeometryCrafter checkout: {repo_path}"
        )

    moge_marker = repo_path / "third_party" / "moge" / "moge" / "model"
    if not moge_marker.exists():
        _run_cmd_checked(
            ["git", "submodule", "update", "--init", "--recursive"],
            logger=logger,
            cwd=repo_path,
        )

    if not _is_geometry_repo(repo_path):
        raise RuntimeError(f"GeometryCrafter repository is incomplete at {repo_path}.")
    return repo_path


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


class _NvidiaSmiTimelineTracker:
    """Background nvidia-smi sampler for stage interval analytics."""

    def __init__(self, interval_sec: float = 0.5):
        self.interval_sec = max(0.2, float(interval_sec))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._samples: List[Dict[str, Any]] = []
        self._sample_count = 0

    def _sample_once(self) -> None:
        rows = _query_nvidia_smi_snapshot()
        if not rows:
            return
        used_sum = 0.0
        total_sum = 0.0
        util_sum = 0.0
        for row in rows:
            used_sum += max(0.0, float(row.get("used_mib", 0.0) or 0.0))
            total_sum += max(0.0, float(row.get("total_mib", 0.0) or 0.0))
            util_sum += max(0.0, float(row.get("util_gpu_pct", 0.0) or 0.0))
        device_count = len(rows)
        util_mean = (util_sum / device_count) if device_count > 0 else 0.0
        used_pct = (used_sum / total_sum * 100.0) if total_sum > 0.0 else 0.0
        entry = {
            "time_unix": float(time.time()),
            "used_mib_all_devices": float(used_sum),
            "total_mib_all_devices": float(total_sum),
            "used_pct_total_all_devices": float(used_pct),
            "util_gpu_pct_mean": float(util_mean),
            "device_count": int(device_count),
        }
        with self._lock:
            self._sample_count += 1
            self._samples.append(entry)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._sample_once()
            self._stop_event.wait(self.interval_sec)

    def start(self) -> bool:
        probe_rows = _query_nvidia_smi_snapshot()
        if not probe_rows:
            return False
        self._sample_once()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._sample_once()

    def sample_count(self) -> int:
        with self._lock:
            return int(self._sample_count)

    def get_samples_between(self, start_ts: float, end_ts: float) -> List[Dict[str, Any]]:
        if end_ts < start_ts:
            return []
        with self._lock:
            return [
                dict(sample)
                for sample in self._samples
                if float(sample.get("time_unix", 0.0) or 0.0) >= start_ts
                and float(sample.get("time_unix", 0.0) or 0.0) <= end_ts
            ]


def _build_stage_interval_stats(
    samples: List[Dict[str, Any]],
    *,
    start_ts: float,
    end_ts: float,
    bucket_sec: float = 5.0,
) -> Dict[str, Any]:
    duration_sec = max(0.0, float(end_ts - start_ts))
    if not samples:
        return {
            "duration_seconds": round(duration_sec, 3),
            "sample_count": 0,
            "bucket_seconds": float(bucket_sec),
            "summary": {},
            "buckets": [],
        }

    bucket_sec = max(0.5, float(bucket_sec))
    bucket_count = max(1, int(math.ceil(duration_sec / bucket_sec)))
    buckets: List[Dict[str, Any]] = [
        {
            "index": idx,
            "start_offset_sec": round(idx * bucket_sec, 3),
            "end_offset_sec": round(min(duration_sec, (idx + 1) * bucket_sec), 3),
            "sample_count": 0,
            "sum_used_mib": 0.0,
            "sum_used_pct": 0.0,
            "sum_util_pct": 0.0,
            "peak_used_mib": 0.0,
            "peak_used_pct": 0.0,
            "peak_util_pct": 0.0,
        }
        for idx in range(bucket_count)
    ]

    sum_used_mib = 0.0
    sum_used_pct = 0.0
    sum_util_pct = 0.0
    peak_used_mib = 0.0
    peak_used_pct = 0.0
    peak_util_pct = 0.0

    for sample in samples:
        sample_ts = float(sample.get("time_unix", start_ts) or start_ts)
        offset = min(max(0.0, sample_ts - start_ts), duration_sec)
        idx = int(offset // bucket_sec)
        if idx >= bucket_count:
            idx = bucket_count - 1

        used_mib = float(sample.get("used_mib_all_devices", 0.0) or 0.0)
        used_pct = float(sample.get("used_pct_total_all_devices", 0.0) or 0.0)
        util_pct = float(sample.get("util_gpu_pct_mean", 0.0) or 0.0)

        bucket = buckets[idx]
        bucket["sample_count"] = int(bucket["sample_count"]) + 1
        bucket["sum_used_mib"] += used_mib
        bucket["sum_used_pct"] += used_pct
        bucket["sum_util_pct"] += util_pct
        bucket["peak_used_mib"] = max(float(bucket["peak_used_mib"]), used_mib)
        bucket["peak_used_pct"] = max(float(bucket["peak_used_pct"]), used_pct)
        bucket["peak_util_pct"] = max(float(bucket["peak_util_pct"]), util_pct)

        sum_used_mib += used_mib
        sum_used_pct += used_pct
        sum_util_pct += util_pct
        peak_used_mib = max(peak_used_mib, used_mib)
        peak_used_pct = max(peak_used_pct, used_pct)
        peak_util_pct = max(peak_util_pct, util_pct)

    bucket_out: List[Dict[str, Any]] = []
    for bucket in buckets:
        count = int(bucket["sample_count"])
        if count <= 0:
            bucket_out.append(
                {
                    "index": int(bucket["index"]),
                    "start_offset_sec": float(bucket["start_offset_sec"]),
                    "end_offset_sec": float(bucket["end_offset_sec"]),
                    "sample_count": 0,
                    "avg_used_mib": 0.0,
                    "peak_used_mib": 0.0,
                    "avg_used_pct": 0.0,
                    "peak_used_pct": 0.0,
                    "avg_util_pct": 0.0,
                    "peak_util_pct": 0.0,
                }
            )
            continue
        bucket_out.append(
            {
                "index": int(bucket["index"]),
                "start_offset_sec": float(bucket["start_offset_sec"]),
                "end_offset_sec": float(bucket["end_offset_sec"]),
                "sample_count": count,
                "avg_used_mib": round(float(bucket["sum_used_mib"]) / count, 3),
                "peak_used_mib": round(float(bucket["peak_used_mib"]), 3),
                "avg_used_pct": round(float(bucket["sum_used_pct"]) / count, 3),
                "peak_used_pct": round(float(bucket["peak_used_pct"]), 3),
                "avg_util_pct": round(float(bucket["sum_util_pct"]) / count, 3),
                "peak_util_pct": round(float(bucket["peak_util_pct"]), 3),
            }
        )

    count_all = len(samples)
    return {
        "duration_seconds": round(duration_sec, 3),
        "sample_count": int(count_all),
        "bucket_seconds": float(bucket_sec),
        "summary": {
            "avg_used_mib": round(sum_used_mib / count_all, 3),
            "peak_used_mib": round(peak_used_mib, 3),
            "avg_used_pct": round(sum_used_pct / count_all, 3),
            "peak_used_pct": round(peak_used_pct, 3),
            "avg_util_pct": round(sum_util_pct / count_all, 3),
            "peak_util_pct": round(peak_util_pct, 3),
        },
        "buckets": bucket_out,
    }


def _log_stage_interval_stats(
    logger: logging.Logger,
    *,
    stage_name: str,
    start_ts: float,
    end_ts: float,
    timeline_tracker: _NvidiaSmiTimelineTracker | None,
    stage_interval_store: List[Dict[str, Any]],
    job_start_ts: float,
    payload: Dict[str, Any] | None = None,
    bucket_sec: float = 5.0,
) -> None:
    if timeline_tracker is None:
        entry = {
            "stage": stage_name,
            "start_elapsed_seconds": round(start_ts - job_start_ts, 3),
            "end_elapsed_seconds": round(end_ts - job_start_ts, 3),
            "interval_stats": {
                "duration_seconds": round(max(0.0, end_ts - start_ts), 3),
                "sample_count": 0,
                "bucket_seconds": float(bucket_sec),
                "summary": {},
                "buckets": [],
            },
        }
        if payload:
            entry["payload"] = payload
        stage_interval_store.append(entry)
        logger.info(
            "GPU stage interval | stage=%s | t=%.2fs..%.2fs | duration=%.2fs | no timeline samples",
            stage_name,
            start_ts - job_start_ts,
            end_ts - job_start_ts,
            max(0.0, end_ts - start_ts),
        )
        return

    samples = timeline_tracker.get_samples_between(start_ts, end_ts)
    stats = _build_stage_interval_stats(samples, start_ts=start_ts, end_ts=end_ts, bucket_sec=bucket_sec)
    entry = {
        "stage": stage_name,
        "start_elapsed_seconds": round(start_ts - job_start_ts, 3),
        "end_elapsed_seconds": round(end_ts - job_start_ts, 3),
        "interval_stats": stats,
    }
    if payload:
        entry["payload"] = payload
    stage_interval_store.append(entry)

    summary = stats.get("summary", {}) if isinstance(stats, dict) else {}
    logger.info(
        "GPU stage interval | stage=%s | t=%.2fs..%.2fs | duration=%.2fs | samples=%d | mem avg/peak=%.1f/%.1f MiB (avg/peak %%=%.2f/%.2f) | util avg/peak=%.1f/%.1f%%",
        stage_name,
        start_ts - job_start_ts,
        end_ts - job_start_ts,
        float(stats.get("duration_seconds", 0.0) or 0.0),
        int(stats.get("sample_count", 0) or 0),
        float(summary.get("avg_used_mib", 0.0) or 0.0),
        float(summary.get("peak_used_mib", 0.0) or 0.0),
        float(summary.get("avg_used_pct", 0.0) or 0.0),
        float(summary.get("peak_used_pct", 0.0) or 0.0),
        float(summary.get("avg_util_pct", 0.0) or 0.0),
        float(summary.get("peak_util_pct", 0.0) or 0.0),
    )

    for bucket in stats.get("buckets", []):
        if not isinstance(bucket, dict):
            continue
        logger.info(
            "GPU stage interval 5s | stage=%s | bucket=%d | dt=%.1f-%.1fs | samples=%d | mem avg/peak=%.1f/%.1f MiB (avg/peak %%=%.2f/%.2f) | util avg/peak=%.1f/%.1f%%",
            stage_name,
            int(bucket.get("index", 0) or 0),
            float(bucket.get("start_offset_sec", 0.0) or 0.0),
            float(bucket.get("end_offset_sec", 0.0) or 0.0),
            int(bucket.get("sample_count", 0) or 0),
            float(bucket.get("avg_used_mib", 0.0) or 0.0),
            float(bucket.get("peak_used_mib", 0.0) or 0.0),
            float(bucket.get("avg_used_pct", 0.0) or 0.0),
            float(bucket.get("peak_used_pct", 0.0) or 0.0),
            float(bucket.get("avg_util_pct", 0.0) or 0.0),
            float(bucket.get("peak_util_pct", 0.0) or 0.0),
        )

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

    parser.add_argument(
        "--model-backend",
        choices=["depthcrafter", "geometrycrafter_diff", "geometrycrafter_determ"],
        default="depthcrafter",
    )
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
            "model_backend": args.model_backend,
            "cpu_offload": args.cpu_offload,
            "disable_xformers": bool(args.disable_xformers),
            "local_files_only": bool(args.local_files_only),
            "unet_path": args.unet_path,
            "pretrain_path": args.pretrain_path,
            "geometry_model_path": args.geometry_model_path,
            "geometry_repo_path": args.geometry_repo_path,
            "geometry_cache_dir": args.geometry_cache_dir,
            "geometry_decode_chunk_size": args.geometry_decode_chunk_size,
            "geometry_low_memory_usage": bool(args.geometry_low_memory_usage),
            "geometry_force_projection": bool(args.geometry_force_projection),
            "geometry_force_fixed_focal": bool(args.geometry_force_fixed_focal),
            "geometry_use_extract_interp": bool(args.geometry_use_extract_interp),
        },
    }
    stage_gpu_samples: List[Dict[str, Any]] = []
    status["stage_gpu_samples"] = stage_gpu_samples
    stage_gpu_intervals: List[Dict[str, Any]] = []
    status["stage_gpu_intervals"] = stage_gpu_intervals
    torch_module = None
    gpu_totals_mib: Dict[int, Dict[str, Any]] = {}
    nvidia_peak_tracker: _NvidiaSmiPeakTracker | None = None
    nvidia_timeline_tracker: _NvidiaSmiTimelineTracker | None = None
    active_stage_windows: Dict[str, Dict[str, Any]] = {}

    try:
        _validate_runtime_args(args)
        model_backend = str(args.model_backend or "depthcrafter").strip().lower()
        if model_backend not in ("depthcrafter", "geometrycrafter_diff", "geometrycrafter_determ"):
            model_backend = "depthcrafter"
        if isinstance(status.get("params"), dict):
            status["params"]["model_backend"] = model_backend

        size_multiple = 64 if model_backend.startswith("geometrycrafter") else 8

        if model_backend == "depthcrafter":
            from depthcrafter.depthcrafter_logic import DepthCrafterDemo as SelectedDemo
        else:
            _ensure_python_module("kornia", "kornia>=0.8.2", logger)
            _ensure_python_module("scipy", "scipy>=1.10", logger)
            geometry_repo = _ensure_geometry_repo_available(args.geometry_repo_path, logger)
            if not str(args.geometry_repo_path or "").strip():
                args.geometry_repo_path = str(geometry_repo)
                if isinstance(status.get("params"), dict):
                    status["params"]["geometry_repo_path"] = str(geometry_repo)
            from depthcrafter.geometrycrafter_logic import GeometryCrafterDemo as SelectedDemo
        import torch as torch_module

        if torch_module.cuda.is_available():
            gpu_totals_mib = _query_nvidia_smi_totals_mib()

        nvidia_timeline_tracker = _NvidiaSmiTimelineTracker(interval_sec=0.5)
        if nvidia_timeline_tracker.start():
            logger.info(
                "nvidia-smi timeline sampling enabled (interval=%.2fs).",
                nvidia_timeline_tracker.interval_sec,
            )
        else:
            nvidia_timeline_tracker = None

        def _record_stage_event(stage: str, payload: Dict[str, Any] | None = None) -> None:
            _log_stage_gpu_snapshot(
                logger,
                stage=stage,
                torch_module=torch_module,
                gpu_totals_mib=gpu_totals_mib,
                stage_log_store=stage_gpu_samples,
                job_start_ts=start_ts,
                payload=payload,
            )

            now_ts = time.time()
            if stage.endswith("_start"):
                base_stage = stage[:-6]
                active_stage_windows[base_stage] = {
                    "start_ts": now_ts,
                    "payload_start": dict(payload or {}),
                }
                return
            if stage.endswith("_end"):
                base_stage = stage[:-4]
                started = active_stage_windows.pop(base_stage, None)
                if not started:
                    return
                merged_payload: Dict[str, Any] = {}
                payload_start = started.get("payload_start")
                if isinstance(payload_start, dict):
                    merged_payload.update(payload_start)
                if isinstance(payload, dict):
                    merged_payload.update(payload)
                _log_stage_interval_stats(
                    logger,
                    stage_name=base_stage,
                    start_ts=float(started.get("start_ts", now_ts) or now_ts),
                    end_ts=now_ts,
                    timeline_tracker=nvidia_timeline_tracker,
                    stage_interval_store=stage_gpu_intervals,
                    job_start_ts=start_ts,
                    payload=merged_payload if merged_payload else None,
                    bucket_sec=5.0,
                )

        _record_stage_event("runtime_imports_ready")

        target_w = _coerce_multiple(int(args.target_width), "target width", size_multiple)
        target_h = _coerce_multiple(int(args.target_height), "target height", size_multiple)

        if target_w != int(args.target_width) or target_h != int(args.target_height):
            logger.warning(
                "Adjusted target resolution from %sx%s to %sx%s to satisfy /%s model constraints.",
                args.target_width,
                args.target_height,
                target_w,
                target_h,
                size_multiple,
            )

        _record_stage_event(
            "model_init_start",
            payload={
                "target_width": int(target_w),
                "target_height": int(target_h),
                "model_backend": model_backend,
            },
        )
        logger.info("Initializing model backend: %s", model_backend)

        if model_backend == "depthcrafter":
            demo = SelectedDemo(
                unet_path=args.unet_path,
                pre_train_path=args.pretrain_path,
                cpu_offload=args.cpu_offload,
                use_cudnn_benchmark=bool(args.use_cudnn_benchmark),
                local_files_only=bool(args.local_files_only),
                disable_xformers=bool(args.disable_xformers),
            )
        else:
            demo = SelectedDemo(
                model_backend=model_backend,
                geometry_model_path=args.geometry_model_path,
                geometry_repo_path=args.geometry_repo_path,
                geometry_cache_dir=args.geometry_cache_dir,
                geometry_decode_chunk_size=max(1, int(args.geometry_decode_chunk_size)),
                geometry_low_memory_usage=bool(args.geometry_low_memory_usage),
                geometry_force_projection=bool(args.geometry_force_projection),
                geometry_force_fixed_focal=bool(args.geometry_force_fixed_focal),
                geometry_use_extract_interp=bool(args.geometry_use_extract_interp),
                pre_train_path=args.pretrain_path,
                cpu_offload=args.cpu_offload,
                use_cudnn_benchmark=bool(args.use_cudnn_benchmark),
                local_files_only=bool(args.local_files_only),
                disable_xformers=bool(args.disable_xformers),
            )
        _record_stage_event("model_init_end")

        if torch_module.cuda.is_available():
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
        _record_stage_event("tracker_start_end")

        def _runtime_stage_callback(stage: str, payload: Dict[str, Any]) -> None:
            _record_stage_event(stage, payload)

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
        _record_stage_event("demo_run_start", payload={"input": str(input_path)})

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
        _record_stage_event("demo_run_end", payload={"save_path": str(save_path) if save_path else ""})

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
        try:
            _record_stage_event("finalize_start")
        except Exception:
            pass

        # Close any stage spans that started but did not emit an *_end event.
        if active_stage_windows:
            close_ts = time.time()
            for unfinished_stage, started in list(active_stage_windows.items()):
                try:
                    _log_stage_interval_stats(
                        logger,
                        stage_name=f"{unfinished_stage}_incomplete",
                        start_ts=float(started.get("start_ts", close_ts) or close_ts),
                        end_ts=close_ts,
                        timeline_tracker=nvidia_timeline_tracker,
                        stage_interval_store=stage_gpu_intervals,
                        job_start_ts=start_ts,
                        payload={"unfinished": True},
                        bucket_sec=5.0,
                    )
                except Exception:
                    continue
            active_stage_windows.clear()

        if nvidia_timeline_tracker is not None:
            try:
                nvidia_timeline_tracker.stop()
                status["gpu_timeline_sample_count"] = int(nvidia_timeline_tracker.sample_count())
            except Exception:
                pass

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
