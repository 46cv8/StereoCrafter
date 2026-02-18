#!/usr/bin/env python3
"""Long-lived queue worker for remote DepthCrafter/GeometryCrafter/StereoPilot inference.

This worker keeps model weights loaded in memory and processes jobs from a
filesystem queue:
  <queue-dir>/pending/*.json
Result records are written to:
  <queue-dir>/done/*.json or <queue-dir>/failed/*.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import run_depth_batch_session as batch_session
import run_depth_job as job_runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a persistent queue worker for headless depth jobs."
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing queued jobs when one job fails.",
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

    parser.add_argument(
        "--model-backend",
        choices=["depthcrafter", "geometrycrafter_diff", "geometrycrafter_determ", "stereopilot"],
        default="depthcrafter",
    )
    parser.add_argument(
        "--ensure-python-deps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Install any missing runtime Python dependencies before inference starts.",
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
    parser.add_argument("--stereopilot-model-path", default="KlingTeam/StereoPilot")
    parser.add_argument("--stereopilot-base-model-path", default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--stereopilot-repo-path", default="")
    parser.add_argument("--stereopilot-cache-dir", default="")
    parser.add_argument("--stereopilot-prompt", default="")
    parser.add_argument(
        "--stereopilot-use-sidecar-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--stereopilot-output-mode",
        choices=["opposite_eye", "side_by_side", "both"],
        default="side_by_side",
    )
    parser.add_argument("--stereopilot-target-width", type=int, default=832)
    parser.add_argument("--stereopilot-target-height", type=int, default=480)
    parser.add_argument("--stereopilot-target-fps", type=float, default=16.0)
    parser.add_argument("--stereopilot-frame-count", type=int, default=81, help=argparse.SUPPRESS)
    parser.add_argument("--stereopilot-sampling-steps", type=int, default=30)
    parser.add_argument("--stereopilot-guide-scale", type=float, default=5.0)
    parser.add_argument("--stereopilot-shift", type=float, default=5.0)
    parser.add_argument("--stereopilot-tail-pad-frames", type=int, default=5)
    parser.add_argument("--stereopilot-domain-label", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--stereopilot-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--stereopilot-transformer-dtype",
        choices=["float8", "float16", "bfloat16", "float32"],
        default="float8",
    )

    parser.add_argument("--unet-path", default="tencent/DepthCrafter")
    parser.add_argument("--pretrain-path", default="stabilityai/stable-video-diffusion-img2vid-xt")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--queue-dir", required=True, help="Queue directory root on remote machine.")
    parser.add_argument("--poll-interval-sec", type=float, default=0.5, help="Queue polling interval.")
    parser.add_argument(
        "--stop-file-name",
        default="stop",
        help="Stop sentinel filename under <queue-dir>/control.",
    )
    return parser


def _safe_json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _claim_next_job(pending_dir: Path, inprogress_dir: Path) -> Optional[Tuple[str, Path]]:
    candidates = sorted(pending_dir.glob("*.json"))
    for src in candidates:
        job_id = src.stem
        dst = inprogress_dir / src.name
        try:
            os.replace(src, dst)
            return job_id, dst
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return None


def _job_result_payload(
    *,
    job_id: str,
    entry: Dict[str, Any],
    ok: bool,
    status_json_path: Path,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "job_id": job_id,
        "job_name": str(entry.get("job_name", "")).strip(),
        "input": str(entry.get("input", "")).strip(),
        "output_dir": str(entry.get("output_dir", "")).strip(),
        "status_json": str(status_json_path),
        "status": "success" if ok else "failed",
        "finished_time_unix": time.time(),
        "finished_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        status_payload = _load_json(status_json_path)
    except Exception:
        status_payload = {}
    if isinstance(status_payload, dict):
        payload["status_message"] = str(status_payload.get("message", "")).strip()
        if "save_path" in status_payload:
            payload["save_path"] = status_payload.get("save_path")
    return payload


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    job_runner._configure_logging(bool(args.verbose))
    logger = logging.getLogger("cloud.run_depth_queue_worker")

    queue_root = Path(args.queue_dir).expanduser().resolve()
    pending_dir = queue_root / "pending"
    inprogress_dir = queue_root / "inprogress"
    done_dir = queue_root / "done"
    failed_dir = queue_root / "failed"
    control_dir = queue_root / "control"
    stop_file = control_dir / str(args.stop_file_name or "stop")
    state_path = queue_root / "worker_state.json"
    pid_path = queue_root / "worker.pid"

    for d in (pending_dir, inprogress_dir, done_dir, failed_dir, control_dir):
        d.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    worker_started_ts = time.time()
    state: Dict[str, Any] = {
        "status": "starting",
        "pid": os.getpid(),
        "queue_dir": str(queue_root),
        "start_time_unix": worker_started_ts,
        "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(worker_started_ts)),
        "last_heartbeat_unix": worker_started_ts,
        "processed_count": 0,
        "success_count": 0,
        "failed_count": 0,
        "current_job_id": "",
        "current_job_name": "",
    }
    _safe_json_write(state_path, state)

    demo, torch_module, gpu_totals_mib, model_backend, target_w, target_h = batch_session._init_demo(args, logger)
    state["status"] = "running"
    state["model_backend"] = model_backend
    state["target_width"] = int(target_w)
    state["target_height"] = int(target_h)
    _safe_json_write(state_path, state)
    logger.info("Queue worker ready | queue=%s | backend=%s", queue_root, model_backend)

    poll_interval = max(0.1, float(args.poll_interval_sec))
    should_stop = False
    try:
        while True:
            now_ts = time.time()
            state["last_heartbeat_unix"] = now_ts
            _safe_json_write(state_path, state)

            if stop_file.exists():
                should_stop = True

            claimed = _claim_next_job(pending_dir, inprogress_dir)
            if claimed is None:
                if should_stop:
                    break
                time.sleep(poll_interval)
                continue

            job_id, claimed_path = claimed
            try:
                entry = _load_json(claimed_path)
            except Exception as exc:
                logger.error("Invalid queued job file %s: %s", claimed_path, exc)
                result_payload = {
                    "job_id": job_id,
                    "status": "failed",
                    "status_message": f"Invalid queued job JSON: {exc}",
                    "finished_time_unix": time.time(),
                    "finished_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                _safe_json_write(failed_dir / f"{job_id}.json", result_payload)
                claimed_path.unlink(missing_ok=True)
                state["processed_count"] = int(state.get("processed_count", 0) or 0) + 1
                state["failed_count"] = int(state.get("failed_count", 0) or 0) + 1
                continue

            job_name = str(entry.get("job_name", "")).strip() or job_id
            status_json_path = Path(str(entry.get("status_json", "")).strip() or (queue_root / f"{job_id}_status.json")).expanduser().resolve()
            state["current_job_id"] = job_id
            state["current_job_name"] = job_name
            _safe_json_write(state_path, state)
            logger.info("Dequeued job: %s (%s)", job_name, job_id)

            ok = batch_session._run_one_job(
                demo=demo,
                torch_module=torch_module,
                gpu_totals_mib=gpu_totals_mib,
                args=args,
                model_backend=model_backend,
                target_w=target_w,
                target_h=target_h,
                job_entry={
                    "input": str(entry.get("input", "")),
                    "output_dir": str(entry.get("output_dir", "")),
                    "status_json": str(status_json_path),
                    "job_name": job_name,
                },
                logger=logger,
            )

            result_payload = _job_result_payload(
                job_id=job_id,
                entry=entry if isinstance(entry, dict) else {},
                ok=ok,
                status_json_path=status_json_path,
            )
            if ok:
                _safe_json_write(done_dir / f"{job_id}.json", result_payload)
                state["success_count"] = int(state.get("success_count", 0) or 0) + 1
            else:
                _safe_json_write(failed_dir / f"{job_id}.json", result_payload)
                state["failed_count"] = int(state.get("failed_count", 0) or 0) + 1

            claimed_path.unlink(missing_ok=True)
            state["processed_count"] = int(state.get("processed_count", 0) or 0) + 1
            state["current_job_id"] = ""
            state["current_job_name"] = ""
    finally:
        state["status"] = "stopped"
        state["stop_time_unix"] = time.time()
        state["stop_time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state["current_job_id"] = ""
        state["current_job_name"] = ""
        _safe_json_write(state_path, state)
        logger.info("Queue worker stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
