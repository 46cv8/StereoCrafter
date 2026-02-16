import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from depthcrafter import merge_depth_segments
from depthcrafter.utils import read_video_frames, save_json_file, save_video

_logger = logging.getLogger(__name__)


def _parse_segment_npz_name(npz_filename: str, original_basename: str) -> Tuple[int, int]:
    pattern = rf"^{re.escape(original_basename)}_depth_(\d+)of(\d+)\.npz$"
    match = re.match(pattern, npz_filename)
    if not match:
        raise ValueError(f"Filename does not match expected segment NPZ pattern: {npz_filename}")
    idx_1b = int(match.group(1))
    total = int(match.group(2))
    if idx_1b <= 0 or total <= 0 or idx_1b > total:
        raise ValueError(f"Invalid segment index/total in filename: {npz_filename}")
    return idx_1b - 1, total


def _build_raw_reference_from_master(
    lowres_master_meta_path: str,
    merge_alignment_method: str,
) -> Tuple[np.ndarray, float, dict]:
    meta_data, n_overlap, sorted_jobs, base_dir = merge_depth_segments._load_and_validate_metadata(lowres_master_meta_path)

    if len(sorted_jobs) == 1:
        lowres_raw, lowres_fps = merge_depth_segments._load_single_segment_frames(sorted_jobs[0], base_dir)
    else:
        loaded_frames, job_meta_map, lowres_fps = merge_depth_segments._load_multiple_segments_data(sorted_jobs, base_dir)
        if len(loaded_frames) == 1:
            lowres_raw = loaded_frames[0]
        else:
            aligned = merge_depth_segments._align_segments_data(
                loaded_frames,
                job_meta_map,
                n_overlap,
                merge_alignment_method,
            )
            lowres_raw = merge_depth_segments._stitch_and_blend_segments_data(
                aligned,
                job_meta_map,
                n_overlap,
            )

    return lowres_raw.astype(np.float32), float(lowres_fps), meta_data


def _build_raw_reference_from_legacy_npz_folder(
    segment_folder: str,
    original_basename: str,
    overlap_frames: int,
    fallback_fps: float,
    merge_alignment_method: str,
) -> Tuple[np.ndarray, float, dict]:
    if not os.path.isdir(segment_folder):
        raise FileNotFoundError(f"Segment folder not found: {segment_folder}")

    npz_files = []
    for name in os.listdir(segment_folder):
        if not name.lower().endswith(".npz"):
            continue
        try:
            seg_id, total = _parse_segment_npz_name(name, original_basename)
            npz_files.append((seg_id, total, name))
        except ValueError:
            continue

    if not npz_files:
        raise FileNotFoundError(
            f"Legacy NPZ fallback for {original_basename}: no matching NPZ files in {segment_folder}"
        )

    npz_files.sort(key=lambda x: x[0])
    inferred_total = max(entry[1] for entry in npz_files)
    total_values = {entry[1] for entry in npz_files}
    if len(total_values) > 1:
        _logger.warning(
            f"Legacy NPZ fallback for {original_basename}: inconsistent total-segment indicators found: {sorted(total_values)}"
        )

    max_idx = max(entry[0] for entry in npz_files)
    if max_idx + 1 != len(npz_files):
        _logger.warning(
            f"Legacy NPZ fallback for {original_basename}: found {len(npz_files)} NPZ files with max index {max_idx}. "
            "Proceeding with available sorted files."
        )
    if len(npz_files) != inferred_total:
        _logger.warning(
            f"Legacy NPZ fallback for {original_basename}: found {len(npz_files)} segment NPZ files, "
            f"filename totals indicate {inferred_total}. Proceeding with available files only."
        )

    synthetic_jobs = []
    fps_for_jobs = float(fallback_fps) if fallback_fps > 0 else 23.976

    for seg_id, _, name in npz_files:
        synthetic_jobs.append(
            {
                "segment_id": int(seg_id),
                "output_segment_filename": name,
                "output_segment_format": "npz",
                "processed_at_fps": fps_for_jobs,
            }
        )

    if len(synthetic_jobs) == 1:
        lowres_raw, lowres_fps = merge_depth_segments._load_single_segment_frames(
            synthetic_jobs[0],
            segment_folder,
        )
    else:
        loaded_frames, job_meta_map, lowres_fps = merge_depth_segments._load_multiple_segments_data(
            synthetic_jobs,
            segment_folder,
        )
        if len(loaded_frames) == 1:
            lowres_raw = loaded_frames[0]
        else:
            aligned = merge_depth_segments._align_segments_data(
                loaded_frames,
                job_meta_map,
                int(overlap_frames),
                merge_alignment_method,
            )
            lowres_raw = merge_depth_segments._stitch_and_blend_segments_data(
                aligned,
                job_meta_map,
                int(overlap_frames),
            )

    meta_stub = {
        "source": "legacy_npz_fallback",
        "global_processing_settings": {
            "processed_as_segments": True,
            "segment_definition_output_overlap_frames": int(overlap_frames),
        },
        "legacy_segment_folder": os.path.abspath(segment_folder),
    }
    return lowres_raw.astype(np.float32), float(fps_for_jobs), meta_stub


def _build_reference_full(lowres_raw: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    return (
        F.interpolate(
            torch.from_numpy(lowres_raw).unsqueeze(1),
            size=(int(target_height), int(target_width)),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(1)
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def _frame_gray_from_rgb(frame_rgb: np.ndarray) -> np.ndarray:
    frame = frame_rgb.astype(np.float32, copy=False)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"Expected RGB frame [H,W,3], got {frame.shape}")
    # ITU-R BT.709 luma coefficients.
    gray = 0.2126 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 2]
    return gray.astype(np.float32)


def _edge_weight_from_gray(frame_gray: np.ndarray) -> np.ndarray:
    gy = np.abs(np.diff(frame_gray, axis=0, append=frame_gray[-1:, :]))
    gx = np.abs(np.diff(frame_gray, axis=1, append=frame_gray[:, -1:]))
    grad = np.sqrt(gx * gx + gy * gy, dtype=np.float32)
    p_low, p_high = np.percentile(grad, [70.0, 99.5]).astype(np.float32)
    denom = float(max(1e-6, p_high - p_low))
    edge = np.clip((grad - p_low) / denom, 0.0, 1.0).astype(np.float32)
    return edge


def _joint_bilateral_filter_3x3(
    depth_map: np.ndarray,
    guide_gray: np.ndarray,
    sigma_color: float,
    sigma_spatial: float,
    iterations: int,
) -> np.ndarray:
    if iterations <= 0:
        return depth_map.astype(np.float32, copy=False)

    sigma_color = max(1e-4, float(sigma_color))
    sigma_spatial = max(1e-4, float(sigma_spatial))
    color_denom = 2.0 * sigma_color * sigma_color

    h, w = depth_map.shape
    depth = torch.from_numpy(depth_map.astype(np.float32, copy=False)).view(1, 1, h, w)
    guide = torch.from_numpy(guide_gray.astype(np.float32, copy=False)).view(1, 1, h, w)

    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    spatial_w = {
        (dy, dx): float(np.exp(-((dx * dx + dy * dy) / (2.0 * sigma_spatial * sigma_spatial))))
        for (dy, dx) in offsets
    }

    with torch.no_grad():
        for _ in range(int(iterations)):
            depth_pad = F.pad(depth, (1, 1, 1, 1), mode="replicate")
            guide_pad = F.pad(guide, (1, 1, 1, 1), mode="replicate")
            num = torch.zeros_like(depth)
            den = torch.zeros_like(depth)

            for dy, dx in offsets:
                y0 = 1 + dy
                x0 = 1 + dx
                d_shift = depth_pad[:, :, y0:y0 + h, x0:x0 + w]
                g_shift = guide_pad[:, :, y0:y0 + h, x0:x0 + w]
                color_w = torch.exp(-((guide - g_shift) * (guide - g_shift)) / color_denom)
                w_all = spatial_w[(dy, dx)] * color_w
                num += w_all * d_shift
                den += w_all

            depth = num / torch.clamp(den, min=1e-6)

    return depth.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def _frame_affine_match_to_reference(
    pred_frame: np.ndarray,
    ref_frame: np.ndarray,
    gain_min: float = 0.85,
    gain_max: float = 1.15,
) -> np.ndarray:
    pred = pred_frame.astype(np.float32, copy=False)
    ref = ref_frame.astype(np.float32, copy=False)

    pred_lo, pred_hi = np.percentile(pred, [5.0, 95.0]).astype(np.float32)
    ref_lo, ref_hi = np.percentile(ref, [5.0, 95.0]).astype(np.float32)

    pred_span = float(max(1e-6, pred_hi - pred_lo))
    ref_span = float(max(1e-6, ref_hi - ref_lo))
    gain = float(np.clip(ref_span / pred_span, gain_min, gain_max))

    pred_med = float(np.median(pred))
    ref_med = float(np.median(ref))
    offset = float(ref_med - gain * pred_med)

    out = pred * gain + offset

    # Keep tails bounded near the reference tails to avoid clipping artifacts later.
    ref_q001, ref_q999 = np.percentile(ref, [0.1, 99.9]).astype(np.float32)
    span = float(max(1e-6, ref_q999 - ref_q001))
    out = np.clip(out, ref_q001 - 0.05 * span, ref_q999 + 0.05 * span)
    return out.astype(np.float32)


def build_edge_guided_depth_reference(
    lowres_raw: np.ndarray,
    source_frames_rgb: np.ndarray,
    target_height: int,
    target_width: int,
    edge_strength: float = 0.90,
    sigma_color: float = 0.04,
    sigma_spatial: float = 0.90,
    bilateral_iterations: int = 1,
    temporal_smooth: float = 0.03,
    edge_reinject_strength: float = 0.60,
    log_prefix: Optional[str] = None,
    log_frame_interval: int = 8,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if lowres_raw.ndim != 3:
        raise ValueError(f"Expected lowres_raw [T,H,W], got {lowres_raw.shape}")
    if source_frames_rgb.ndim != 4:
        raise ValueError(f"Expected source_frames_rgb [T,H,W,3], got {source_frames_rgb.shape}")

    frame_count = int(min(lowres_raw.shape[0], source_frames_rgb.shape[0]))
    if frame_count <= 0:
        raise ValueError("No overlapping frames available for edge-guided reference build.")

    target_h = int(target_height)
    target_w = int(target_width)
    if target_h <= 0 or target_w <= 0:
        raise ValueError("Target width/height must be > 0.")

    lowres = lowres_raw[:frame_count].astype(np.float32, copy=False)
    source = source_frames_rgb[:frame_count].astype(np.float32, copy=False)

    edge_strength = float(np.clip(edge_strength, 0.0, 1.0))
    temporal_smooth = float(np.clip(temporal_smooth, 0.0, 1.0))
    edge_reinject_strength = float(np.clip(edge_reinject_strength, 0.0, 1.0))
    sigma_color = float(max(1e-4, sigma_color))
    sigma_spatial = float(max(1e-4, sigma_spatial))
    bilateral_iterations = int(max(0, bilateral_iterations))

    ref_bilinear = _build_reference_full(lowres, target_h, target_w)
    ref_nearest = (
        F.interpolate(
            torch.from_numpy(lowres).unsqueeze(1),
            size=(target_h, target_w),
            mode="nearest",
        )
        .squeeze(1)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    output = np.empty_like(ref_bilinear, dtype=np.float32)
    prev_frame: Optional[np.ndarray] = None

    start = time.perf_counter()
    edge_means: List[float] = []
    blend_edge_means: List[float] = []

    for fi in range(frame_count):
        guide_gray = _frame_gray_from_rgb(source[fi])
        edge_w = _edge_weight_from_gray(guide_gray)
        edge_means.append(float(np.mean(edge_w)))

        edge_mix = edge_strength * edge_w
        frame = ref_bilinear[fi] * (1.0 - edge_mix) + ref_nearest[fi] * edge_mix

        if bilateral_iterations > 0:
            frame = _joint_bilateral_filter_3x3(
                depth_map=frame,
                guide_gray=guide_gray,
                sigma_color=sigma_color,
                sigma_spatial=sigma_spatial,
                iterations=bilateral_iterations,
            )

        # Re-inject nearest-depth structure near strong source edges for crisper alignment.
        edge_reinject = edge_reinject_strength * edge_mix
        frame = frame * (1.0 - edge_reinject) + ref_nearest[fi] * edge_reinject

        # Match each frame back to low-res master intensity profile.
        frame = _frame_affine_match_to_reference(frame, ref_bilinear[fi])

        if prev_frame is not None and temporal_smooth > 0.0:
            # Smooth mostly in non-edge regions; keep edge regions reactive.
            alpha = temporal_smooth * (1.0 - edge_w)
            frame = alpha * prev_frame + (1.0 - alpha) * frame

        output[fi] = frame.astype(np.float32, copy=False)
        prev_frame = output[fi]
        blend_edge_means.append(float(np.mean(edge_mix)))

        if log_prefix and (
            fi == 0
            or fi == frame_count - 1
            or ((fi + 1) % max(1, int(log_frame_interval)) == 0)
        ):
            _logger.info(
                f"{log_prefix}: edge-guided frame {fi + 1}/{frame_count} | "
                f"edge_mean={float(np.mean(edge_w)):.4f}, edge_mix_mean={float(np.mean(edge_mix)):.4f}"
            )

    # Final global affine match to low-res master reference.
    pred_vals = output[:, ::4, ::4].reshape(-1)
    ref_vals = ref_bilinear[:, ::4, ::4].reshape(-1)
    pred_lo, pred_hi = np.percentile(pred_vals, [5.0, 95.0]).astype(np.float32)
    ref_lo, ref_hi = np.percentile(ref_vals, [5.0, 95.0]).astype(np.float32)
    pred_span = float(max(1e-6, pred_hi - pred_lo))
    ref_span = float(max(1e-6, ref_hi - ref_lo))
    gain = float(np.clip(ref_span / pred_span, 0.9, 1.1))
    pred_med = float(np.median(pred_vals))
    ref_med = float(np.median(ref_vals))
    offset = float(ref_med - gain * pred_med)
    output *= gain
    output += offset

    ref_q001, ref_q999 = np.percentile(ref_vals, [0.1, 99.9]).astype(np.float32)
    ref_span_robust = float(max(1e-6, ref_q999 - ref_q001))
    np.clip(output, ref_q001 - 0.05 * ref_span_robust, ref_q999 + 0.05 * ref_span_robust, out=output)

    duration = time.perf_counter() - start
    diag = {
        "frames": float(frame_count),
        "target_width": float(target_w),
        "target_height": float(target_h),
        "edge_strength": float(edge_strength),
        "sigma_color": float(sigma_color),
        "sigma_spatial": float(sigma_spatial),
        "bilateral_iterations": float(bilateral_iterations),
        "temporal_smooth": float(temporal_smooth),
        "edge_reinject_strength": float(edge_reinject_strength),
        "edge_weight_mean": float(np.mean(edge_means)) if edge_means else 0.0,
        "edge_mix_mean": float(np.mean(blend_edge_means)) if blend_edge_means else 0.0,
        "global_gain": float(gain),
        "global_offset": float(offset),
        "duration_seconds": float(duration),
    }

    if log_prefix:
        _logger.info(
            f"{log_prefix}: edge-guided reference complete in {duration:.1f}s | "
            f"edge_mean={diag['edge_weight_mean']:.4f}, edge_mix_mean={diag['edge_mix_mean']:.4f}, "
            f"global_gain={diag['global_gain']:.4f}, global_offset={diag['global_offset']:.4f}, "
            f"reinject={diag['edge_reinject_strength']:.3f}"
        )

    return output.astype(np.float32), diag


def _estimate_video_stats(video_data: np.ndarray) -> Dict[str, float]:
    vals = video_data[:, ::4, ::4].reshape(-1).astype(np.float32)
    if vals.size <= 0:
        return {
            "robust_low": 0.0,
            "robust_high": 1.0,
            "core_mean": 0.0,
            "core_std": 0.0,
        }
    robust_low, robust_high = np.percentile(vals, [0.1, 99.9]).astype(np.float32)
    core_low, core_high = np.percentile(vals, [5.0, 95.0]).astype(np.float32)
    if core_high <= core_low + 1e-8:
        core_vals = vals
    else:
        m = (vals >= core_low) & (vals <= core_high)
        core_vals = vals[m] if np.any(m) else vals
    return {
        "robust_low": float(robust_low),
        "robust_high": float(robust_high),
        "core_mean": float(np.mean(core_vals)),
        "core_std": float(np.std(core_vals)),
    }


def run_edge_guided_hires_upscale(
    source_video_path: str,
    original_basename: str,
    output_dir: str,
    lowres_master_meta_path: str,
    target_height: int,
    target_width: int,
    temporal_overlap_frames: int,
    target_fps_setting: float = -1.0,
    output_suffix: str = "_edge_hires_depth",
    output_format: str = "main10_mp4",
    lowres_segment_folder: Optional[str] = None,
    allow_legacy_npz_fallback: bool = True,
    temporal_merge_alignment: str = "shift_scale",
    edge_strength: float = 0.90,
    sigma_color: float = 0.04,
    sigma_spatial: float = 0.90,
    bilateral_iterations: int = 1,
    temporal_smooth: float = 0.03,
    edge_reinject_strength: float = 0.60,
) -> Tuple[str, dict]:
    start_time = time.perf_counter()
    _logger.info(
        f"Starting Edge-Guided Hi-Res Upscale for {original_basename}: "
        f"target={int(target_width)}x{int(target_height)}"
    )

    if lowres_segment_folder is None and lowres_master_meta_path:
        lowres_segment_folder = os.path.dirname(lowres_master_meta_path)

    os.makedirs(output_dir, exist_ok=True)

    if lowres_master_meta_path and os.path.exists(lowres_master_meta_path):
        lowres_raw, lowres_fps, lowres_meta = _build_raw_reference_from_master(
            lowres_master_meta_path,
            merge_alignment_method=temporal_merge_alignment,
        )
    elif allow_legacy_npz_fallback and lowres_segment_folder and os.path.isdir(lowres_segment_folder):
        _logger.warning(
            f"Edge-guided mode for {original_basename}: missing _master_meta.json. "
            f"Attempting legacy NPZ fallback from {lowres_segment_folder}."
        )
        fallback_fps = target_fps_setting if target_fps_setting and target_fps_setting > 0 else 23.976
        lowres_raw, lowres_fps, lowres_meta = _build_raw_reference_from_legacy_npz_folder(
            segment_folder=lowres_segment_folder,
            original_basename=original_basename,
            overlap_frames=int(temporal_overlap_frames),
            fallback_fps=float(fallback_fps),
            merge_alignment_method=temporal_merge_alignment,
        )
        lowres_master_meta_path = ""
    else:
        raise FileNotFoundError(
            f"Missing low-res cache inputs for {original_basename}. "
            f"Master meta path: '{lowres_master_meta_path}', segment folder: '{lowres_segment_folder}'."
        )

    if not os.path.exists(source_video_path):
        raise FileNotFoundError(
            f"Edge-guided mode requires source video frames for guidance, but file was not found: {source_video_path}"
        )

    process_limit = int(lowres_raw.shape[0]) if lowres_raw.shape[0] > 0 else -1
    fps_for_hires_read = lowres_fps if lowres_fps > 0 else target_fps_setting

    _logger.info(
        f"{original_basename}: loading hi-res source frames for edge guidance "
        f"(target={int(target_width)}x{int(target_height)}, fps={float(fps_for_hires_read):.3f}, max_frames={process_limit})..."
    )
    hires_frames, hires_fps, _, _, proc_h, proc_w, _, _ = read_video_frames(
        source_video_path,
        process_length=process_limit,
        target_fps=fps_for_hires_read,
        target_height=target_height,
        target_width=target_width,
        start_frame_index=0,
        num_frames_to_load=-1,
        round_to_multiple=8,
        chunk_size_for_loading=8,
        progress_log_prefix=f"{original_basename} | edge-guided frame load",
    )
    if hires_frames is None or hires_frames.size == 0:
        raise ValueError("Hi-res source frame load returned no frames for edge-guided mode.")

    frame_count = min(int(lowres_raw.shape[0]), int(hires_frames.shape[0]))
    if frame_count <= 0:
        raise ValueError(
            f"No overlapping frames between low-res reference ({lowres_raw.shape[0]}) "
            f"and hi-res source ({hires_frames.shape[0]})."
        )

    lowres_raw = lowres_raw[:frame_count]
    hires_frames = hires_frames[:frame_count]
    if hires_fps <= 0:
        hires_fps = lowres_fps if lowres_fps > 0 else 23.976

    _logger.info(
        f"Edge-Guided Upscale: {original_basename} | frames={frame_count}, target={int(proc_w)}x{int(proc_h)}, fps={float(hires_fps):.3f}"
    )

    edge_guided_raw, edge_diag = build_edge_guided_depth_reference(
        lowres_raw=lowres_raw,
        source_frames_rgb=hires_frames,
        target_height=int(proc_h),
        target_width=int(proc_w),
        edge_strength=float(edge_strength),
        sigma_color=float(sigma_color),
        sigma_spatial=float(sigma_spatial),
        bilateral_iterations=int(bilateral_iterations),
        temporal_smooth=float(temporal_smooth),
        edge_reinject_strength=float(edge_reinject_strength),
        log_prefix=f"{original_basename} edge-upscale",
        log_frame_interval=8,
    )

    reference_full = _build_reference_full(lowres_raw, int(proc_h), int(proc_w))
    pred_stats = _estimate_video_stats(edge_guided_raw)
    ref_stats = _estimate_video_stats(reference_full)

    pred_core_std = float(pred_stats["core_std"])
    ref_core_std = float(ref_stats["core_std"])
    pred_core_mean = float(pred_stats["core_mean"])
    ref_core_mean = float(ref_stats["core_mean"])

    gain = 1.0
    offset = 0.0
    if pred_core_std > 1e-8 and ref_core_std > 1e-8:
        gain = float(np.clip(ref_core_std / pred_core_std, 0.9, 1.1))
        offset = float(ref_core_mean - gain * pred_core_mean)
        edge_guided_raw *= gain
        edge_guided_raw += offset

    ref_robust_low = float(ref_stats["robust_low"])
    ref_robust_high = float(ref_stats["robust_high"])
    robust_span = max(1e-6, ref_robust_high - ref_robust_low)
    norm_low = ref_robust_low - 0.01 * robust_span
    norm_high = ref_robust_high + 0.01 * robust_span
    norm_range = max(1e-6, norm_high - norm_low)

    edge_norm = edge_guided_raw.astype(np.float32, copy=False)
    edge_norm -= norm_low
    edge_norm /= norm_range
    np.clip(edge_norm, 0.0, 1.0, out=edge_norm)

    clip_low_fraction = float(np.mean(edge_norm <= 1e-6))
    clip_high_fraction = float(np.mean(edge_norm >= 1.0 - 1e-6))

    _logger.info(
        f"{original_basename}: edge-guided final normalize range={norm_low:.4f}..{norm_high:.4f} "
        f"(ref_robust={ref_robust_low:.4f}..{ref_robust_high:.4f}) | "
        f"clip_low={clip_low_fraction:.4%}, clip_high={clip_high_fraction:.4%}"
    )

    output_path = os.path.join(output_dir, f"{original_basename}{output_suffix}.mp4")
    save_video(edge_norm, output_path, fps=hires_fps, output_format=output_format)

    metadata = {
        "mode": "edge_guided_hires_upscale",
        "source_video_path": os.path.abspath(source_video_path),
        "lowres_master_meta_path": os.path.abspath(lowres_master_meta_path) if lowres_master_meta_path else None,
        "lowres_segment_folder": os.path.abspath(lowres_segment_folder) if lowres_segment_folder else None,
        "used_legacy_npz_fallback": bool(lowres_meta.get("source") == "legacy_npz_fallback") if isinstance(lowres_meta, dict) else False,
        "output_path": os.path.abspath(output_path),
        "output_format": output_format,
        "output_suffix": output_suffix,
        "target_height_requested": int(target_height),
        "target_width_requested": int(target_width),
        "processed_height": int(proc_h),
        "processed_width": int(proc_w),
        "frames_processed": int(frame_count),
        "fps_processed": float(hires_fps),
        "temporal_overlap_frames": int(temporal_overlap_frames),
        "temporal_merge_alignment": temporal_merge_alignment,
        "edge_strength": float(edge_strength),
        "sigma_color": float(sigma_color),
        "sigma_spatial": float(sigma_spatial),
        "bilateral_iterations": int(bilateral_iterations),
        "temporal_smooth": float(temporal_smooth),
        "edge_reinject_strength": float(edge_reinject_strength),
        "edge_guided_diag": edge_diag,
        "final_affine_gain": float(gain),
        "final_affine_offset": float(offset),
        "final_norm_low_raw": float(norm_low),
        "final_norm_high_raw": float(norm_high),
        "final_clip_low_fraction": float(clip_low_fraction),
        "final_clip_high_fraction": float(clip_high_fraction),
        "lowres_reference_fps": float(lowres_fps),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metadata_path = os.path.splitext(output_path)[0] + ".json"
    save_json_file(metadata, metadata_path, indent=4)

    duration = time.perf_counter() - start_time
    _logger.info(
        f"Edge-Guided Hi-Res Upscale complete for {original_basename} in {duration:.1f}s. Output: {output_path}"
    )

    summary = {
        "output_path": os.path.abspath(output_path),
        "metadata_path": os.path.abspath(metadata_path),
        "frames": int(frame_count),
        "processed_height": int(proc_h),
        "processed_width": int(proc_w),
        "fps": float(hires_fps),
        "duration_seconds": float(round(duration, 3)),
        "edge_weight_mean": float(edge_diag.get("edge_weight_mean", 0.0)),
        "edge_mix_mean": float(edge_diag.get("edge_mix_mean", 0.0)),
        "final_clip_low_fraction": float(clip_low_fraction),
        "final_clip_high_fraction": float(clip_high_fraction),
    }

    del lowres_raw, hires_frames, edge_guided_raw, reference_full, edge_norm
    return output_path, summary
