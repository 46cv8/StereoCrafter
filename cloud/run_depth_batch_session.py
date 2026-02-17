#!/usr/bin/env python3
"""Persistent batch runner that keeps one inference backend loaded for many jobs.

This script is intended for SSH-driven cloud batches where model initialization
cost should be amortized across multiple clips. It loads the selected backend
once, then processes jobs listed in a JSON manifest.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import run_depth_job as job_runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run many headless DepthCrafter/GeometryCrafter jobs in one persistent process."
    )
    parser.add_argument("--jobs-manifest", required=True, help="JSON file containing a list of job entries.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining jobs when one job fails.",
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
        choices=["depthcrafter", "geometrycrafter_diff", "geometrycrafter_determ"],
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

    parser.add_argument("--unet-path", default="tencent/DepthCrafter")
    parser.add_argument("--pretrain-path", default="stabilityai/stable-video-diffusion-img2vid-xt")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _load_manifest(path: Path) -> List[Dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("jobs manifest must be a JSON list.")

    jobs: List[Dict[str, str]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"jobs manifest entry #{idx} is not a JSON object.")
        input_path = str(item.get("input", "")).strip()
        output_dir = str(item.get("output_dir", "")).strip()
        if not input_path:
            raise ValueError(f"jobs manifest entry #{idx} is missing 'input'.")
        if not output_dir:
            raise ValueError(f"jobs manifest entry #{idx} is missing 'output_dir'.")
        job_name = str(item.get("job_name", "")).strip() or Path(input_path).stem
        status_json = str(item.get("status_json", "")).strip()
        if not status_json:
            status_json = str(Path(output_dir) / "job_status.json")
        jobs.append(
            {
                "input": input_path,
                "output_dir": output_dir,
                "status_json": status_json,
                "job_name": job_name,
            }
        )
    return jobs


def _normalize_backend(name: str) -> str:
    backend = str(name or "depthcrafter").strip().lower()
    if backend not in ("depthcrafter", "geometrycrafter_diff", "geometrycrafter_determ"):
        return "depthcrafter"
    return backend


def _init_demo(
    args: argparse.Namespace,
    logger: logging.Logger,
) -> tuple[Any, Any, Dict[int, Dict[str, Any]], str, int, int]:
    model_backend = _normalize_backend(args.model_backend)
    if bool(args.ensure_python_deps):
        job_runner._ensure_runtime_python_deps(
            model_backend=model_backend,
            disable_xformers=bool(args.disable_xformers),
            logger=logger,
        )

    size_multiple = 64 if model_backend.startswith("geometrycrafter") else 8
    target_w = job_runner._coerce_multiple(int(args.target_width), "target width", size_multiple)
    target_h = job_runner._coerce_multiple(int(args.target_height), "target height", size_multiple)
    if target_w != int(args.target_width) or target_h != int(args.target_height):
        logger.warning(
            "Adjusted target resolution from %sx%s to %sx%s to satisfy /%s model constraints.",
            args.target_width,
            args.target_height,
            target_w,
            target_h,
            size_multiple,
        )

    if model_backend == "depthcrafter":
        from depthcrafter.depthcrafter_logic import DepthCrafterDemo as SelectedDemo
    else:
        geometry_repo = job_runner._ensure_geometry_repo_available(args.geometry_repo_path, logger)
        if not str(args.geometry_repo_path or "").strip():
            args.geometry_repo_path = str(geometry_repo)
        from depthcrafter.geometrycrafter_logic import GeometryCrafterDemo as SelectedDemo

    import torch as torch_module

    gpu_totals_mib: Dict[int, Dict[str, Any]] = {}
    if torch_module.cuda.is_available():
        gpu_totals_mib = job_runner._query_nvidia_smi_totals_mib()

    logger.info("Initializing persistent backend: %s", model_backend)
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
    logger.info("Persistent backend initialized.")
    return demo, torch_module, gpu_totals_mib, model_backend, target_w, target_h


def _build_job_status(
    *,
    args: argparse.Namespace,
    model_backend: str,
    job_name: str,
    input_path: Path,
    output_dir: Path,
    target_w: int,
    target_h: int,
    start_ts: float,
) -> Dict[str, Any]:
    return {
        "job_name": job_name,
        "status": "running",
        "start_time_unix": start_ts,
        "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_ts)),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "params": {
            "target_width": target_w,
            "target_height": target_h,
            "window_size": args.window_size,
            "overlap": args.overlap,
            "inference_steps": args.inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "target_fps": args.target_fps,
            "process_length": args.process_length,
            "output_format": args.output_format,
            "model_backend": model_backend,
            "ensure_python_deps": bool(args.ensure_python_deps),
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
            "persistent_batch_session": True,
        },
    }


def _run_one_job(
    *,
    demo: Any,
    torch_module: Any,
    gpu_totals_mib: Dict[int, Dict[str, Any]],
    args: argparse.Namespace,
    model_backend: str,
    target_w: int,
    target_h: int,
    job_entry: Dict[str, str],
    logger: logging.Logger,
) -> bool:
    input_path = Path(job_entry["input"]).expanduser().resolve()
    output_dir = Path(job_entry["output_dir"]).expanduser().resolve()
    status_json = Path(job_entry["status_json"]).expanduser().resolve()
    job_name = str(job_entry["job_name"]).strip() or input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    start_ts = time.time()
    status = _build_job_status(
        args=args,
        model_backend=model_backend,
        job_name=job_name,
        input_path=input_path,
        output_dir=output_dir,
        target_w=target_w,
        target_h=target_h,
        start_ts=start_ts,
    )
    stage_gpu_samples: List[Dict[str, Any]] = []
    status["stage_gpu_samples"] = stage_gpu_samples

    nvidia_peak_tracker: job_runner._NvidiaSmiPeakTracker | None = None

    def _record_stage(stage: str, payload: Dict[str, Any] | None = None) -> None:
        job_runner._log_stage_gpu_snapshot(
            logger,
            stage=stage,
            torch_module=torch_module,
            gpu_totals_mib=gpu_totals_mib,
            stage_log_store=stage_gpu_samples,
            job_start_ts=start_ts,
            payload=payload,
        )

    try:
        if not input_path.exists():
            raise FileNotFoundError(f"Input clip not found: {input_path}")
        if args.window_size <= 0:
            raise ValueError("--window-size must be > 0.")
        if args.overlap < 0:
            raise ValueError("--overlap must be >= 0.")
        if args.overlap >= args.window_size:
            raise ValueError("--overlap must be < --window-size.")
        if args.inference_steps <= 0:
            raise ValueError("--inference-steps must be > 0.")

        if torch_module.cuda.is_available():
            device_count = int(torch_module.cuda.device_count())
            for idx in range(device_count):
                try:
                    torch_module.cuda.reset_peak_memory_stats(idx)
                except Exception:
                    continue

        nvidia_peak_tracker = job_runner._NvidiaSmiPeakTracker(interval_sec=0.5)
        if nvidia_peak_tracker.start():
            logger.info("[%s] nvidia-smi peak VRAM tracking enabled.", job_name)
        else:
            nvidia_peak_tracker = None

        demo.runtime_stage_callback = _record_stage
        _record_stage("demo_run_start", payload={"input": str(input_path)})
        logger.info(
            "Running persistent job '%s' | input=%s | target=%sx%s | window/overlap=%s/%s | steps=%s | output=%s",
            job_name,
            input_path,
            target_w,
            target_h,
            args.window_size,
            args.overlap,
            args.inference_steps,
            args.output_format,
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
        _record_stage("demo_run_end", payload={"save_path": str(save_path) if save_path else ""})

        status["metadata"] = metadata
        status["save_path"] = str(save_path) if save_path else ""
        status["metadata_json"] = metadata.get("_individual_metadata_path") if isinstance(metadata, dict) else None
        if not save_path:
            run_state = metadata.get("status") if isinstance(metadata, dict) else "unknown"
            raise RuntimeError(f"Inference did not return output path (status={run_state}).")
        status["status"] = "success"
        status["message"] = "Depth job completed successfully."
        logger.info("[%s] Job completed: %s", job_name, save_path)
    except Exception as exc:  # pylint: disable=broad-except
        status["status"] = "failed"
        status["message"] = str(exc)
        status["traceback"] = traceback.format_exc()
        logger.exception("[%s] Job failed: %s", job_name, exc)
    finally:
        _record_stage("finalize_start")
        if nvidia_peak_tracker is not None:
            try:
                nvidia_smi_stats = nvidia_peak_tracker.stop_and_summary()
            except Exception:
                nvidia_smi_stats = {}
            if nvidia_smi_stats and int(nvidia_smi_stats.get("device_count", 0) or 0) > 0:
                status["gpu_memory_nvidia_smi"] = nvidia_smi_stats

        gpu_memory_stats = job_runner._compute_gpu_memory_stats(torch_module, gpu_totals_mib)
        if gpu_memory_stats:
            status["gpu_memory"] = gpu_memory_stats

        end_ts = time.time()
        status["end_time_unix"] = end_ts
        status["end_time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_ts))
        status["duration_seconds"] = round(end_ts - start_ts, 3)
        job_runner._safe_json_dump(status_json, status)
        logger.info("[%s] Wrote status JSON: %s", job_name, status_json)

    return status.get("status") == "success"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    job_runner._configure_logging(args.verbose)
    logger = logging.getLogger("cloud.run_depth_batch_session")

    manifest_path = Path(args.jobs_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    jobs = _load_manifest(manifest_path)
    if not jobs:
        logger.info("No jobs in manifest. Exiting.")
        return 0

    demo, torch_module, gpu_totals_mib, model_backend, target_w, target_h = _init_demo(args, logger)
    success_count = 0
    fail_count = 0

    for idx, job_entry in enumerate(jobs, start=1):
        logger.info(
            "Persistent batch item [%d/%d] starting: %s",
            idx,
            len(jobs),
            job_entry.get("job_name", ""),
        )
        ok = _run_one_job(
            demo=demo,
            torch_module=torch_module,
            gpu_totals_mib=gpu_totals_mib,
            args=args,
            model_backend=model_backend,
            target_w=target_w,
            target_h=target_h,
            job_entry=job_entry,
            logger=logger,
        )
        if ok:
            success_count += 1
            continue
        fail_count += 1
        if not args.continue_on_error:
            break

    logger.info(
        "Persistent batch session complete. success=%d failed=%d total=%d",
        success_count,
        fail_count,
        len(jobs),
    )
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
