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
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

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

    try:
        _validate_runtime_args(args)
        from depthcrafter.depthcrafter_logic import DepthCrafterDemo

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

        logger.info("Initializing DepthCrafter model...")
        demo = DepthCrafterDemo(
            unet_path=args.unet_path,
            pre_train_path=args.pretrain_path,
            cpu_offload=args.cpu_offload,
            use_cudnn_benchmark=bool(args.use_cudnn_benchmark),
            local_files_only=bool(args.local_files_only),
            disable_xformers=bool(args.disable_xformers),
        )

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
        end_ts = time.time()
        status["end_time_unix"] = end_ts
        status["end_time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_ts))
        status["duration_seconds"] = round(end_ts - start_ts, 3)
        _safe_json_dump(status_json_path, status)
        logging.getLogger("cloud.run_depth_job").info("Wrote status JSON: %s", status_json_path)

    return 0 if status.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
