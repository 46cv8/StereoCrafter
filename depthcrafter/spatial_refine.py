import gc
import logging
import os
import re
import shutil
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from depthcrafter import merge_depth_segments
from depthcrafter.edge_guided_upscale import build_edge_guided_depth_reference
from depthcrafter.utils import (
    get_segment_output_folder_name,
    read_video_frames,
    save_json_file,
    save_video,
)

_logger = logging.getLogger(__name__)


def _round_up_to_multiple(value: int, multiple: int) -> int:
    multiple = max(1, int(multiple))
    value = int(value)
    return ((value + multiple - 1) // multiple) * multiple


def _get_spatial_model_multiple(demo) -> int:
    """
    Conservative spatial multiple for stable UNet skip-concat compatibility.
    For SVD-like models this is typically 64 (= vae_scale_factor * 2^(num_upsamplers)).
    """
    try:
        vae_scale = int(getattr(demo.pipe, "vae_scale_factor", 8))
    except Exception:
        vae_scale = 8

    num_upsamplers = 3
    try:
        up_block_types = getattr(demo.pipe.unet.config, "up_block_types", None)
        if up_block_types and len(up_block_types) > 0:
            num_upsamplers = max(0, len(up_block_types) - 1)
        else:
            up_blocks = getattr(demo.pipe.unet, "up_blocks", None)
            if up_blocks and len(up_blocks) > 0:
                num_upsamplers = max(0, len(up_blocks) - 1)
    except Exception:
        pass

    # Keep at least 8 to satisfy explicit pipeline checks.
    return max(8, int(vae_scale) * (2 ** int(num_upsamplers)))


def _define_temporal_segments_by_frame_count(
    total_frames: int,
    window_frames: int,
    overlap_frames: int,
) -> List[dict]:
    if total_frames <= 0:
        return []
    if window_frames <= 0:
        raise ValueError("Temporal window must be > 0.")
    if overlap_frames < 0 or overlap_frames >= window_frames:
        raise ValueError("Temporal overlap must be >= 0 and < temporal window.")

    segments: List[dict] = []
    advance = window_frames - overlap_frames
    if advance <= 0:
        raise ValueError("Invalid temporal settings: window-overlap must be > 0.")

    seg_id = 0
    start = 0
    while start < total_frames:
        count = min(window_frames, total_frames - start)
        if count <= 0:
            break
        segments.append(
            {
                "segment_id": seg_id,
                "start_frame_raw_index": int(start),
                "num_frames_to_load_raw": int(count),
            }
        )
        seg_id += 1
        if start + count >= total_frames:
            break
        start += advance

    total_segments = len(segments)
    for seg in segments:
        seg["total_segments"] = total_segments
    return segments


def _compute_axis_windows(
    length: int,
    tile_count: int,
    overlap_target_px: int,
    tile_multiple: int,
) -> List[Tuple[int, int]]:
    """
    Builds windows along one axis with tile extents snapped to tile_multiple.
    overlap_target_px is a target only; effective overlap is derived from full coverage constraints.
    """
    length = int(length)
    tile_count = int(tile_count)
    tile_multiple = max(1, int(tile_multiple))
    overlap_target_px = max(0, int(overlap_target_px))

    if length <= 0:
        return []
    if tile_count <= 1:
        if length % tile_multiple != 0:
            raise ValueError(
                f"Cannot run a single tile on axis length {length}: "
                f"length must be divisible by {tile_multiple}."
            )
        return [(0, length)]

    max_tile_size = (length // tile_multiple) * tile_multiple
    if max_tile_size < tile_multiple:
        raise ValueError(
            f"Axis length {length} is too small for tile_multiple={tile_multiple}."
        )

    target_tile_size = int(np.ceil((length + overlap_target_px * (tile_count - 1)) / float(tile_count)))
    tile_size = _round_up_to_multiple(target_tile_size, tile_multiple)
    tile_size = min(tile_size, max_tile_size)

    span = max(0, length - tile_size)
    starts: List[int] = []
    for idx in range(tile_count):
        if tile_count == 1:
            start = 0
        else:
            start = int(round((idx * span) / float(tile_count - 1)))
        if starts and start < starts[-1]:
            start = starts[-1]
        start = max(0, min(start, span))
        starts.append(start)
    starts[-1] = span
    if len(set(starts)) < tile_count:
        raise ValueError(
            f"Cannot place {tile_count} unique tiles over axis length {length} "
            f"with model multiple {tile_multiple}. Reduce tile count on this axis."
        )

    windows: List[Tuple[int, int]] = []
    for start in starts:
        end = start + tile_size
        if end > length:
            end = length
            start = max(0, end - tile_size)
        windows.append((int(start), int(end)))
    return windows


def _compute_spatial_tiles(
    width: int,
    height: int,
    tile_num_x: int,
    tile_num_y: int,
    overlap_x_px: int,
    overlap_y_px: int,
    tile_multiple: int,
) -> List[dict]:
    x_windows = _compute_axis_windows(width, tile_num_x, overlap_x_px, tile_multiple=tile_multiple)
    y_windows = _compute_axis_windows(height, tile_num_y, overlap_y_px, tile_multiple=tile_multiple)

    tiles = []
    tile_id = 0
    for row, (y0, y1) in enumerate(y_windows):
        for col, (x0, x1) in enumerate(x_windows):
            tiles.append(
                {
                    "tile_id": tile_id,
                    "row": row,
                    "col": col,
                    "x0": int(x0),
                    "y0": int(y0),
                    "x1": int(x1),
                    "y1": int(y1),
                }
            )
            tile_id += 1
    return tiles


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


def _parse_segment_npz_name(npz_filename: str, original_basename: str) -> Tuple[int, int]:
    """
    Parses '<basename>_depth_<idx>of<total>.npz' and returns zero-based (idx, total).
    Raises ValueError if pattern does not match.
    """
    pattern = rf"^{re.escape(original_basename)}_depth_(\d+)of(\d+)\.npz$"
    match = re.match(pattern, npz_filename)
    if not match:
        raise ValueError(f"Filename does not match expected segment NPZ pattern: {npz_filename}")
    idx_1b = int(match.group(1))
    total = int(match.group(2))
    if idx_1b <= 0 or total <= 0 or idx_1b > total:
        raise ValueError(f"Invalid segment index/total in filename: {npz_filename}")
    return idx_1b - 1, total


def _build_raw_reference_from_legacy_npz_folder(
    segment_folder: str,
    original_basename: str,
    overlap_frames: int,
    fallback_fps: float,
    merge_alignment_method: str,
) -> Tuple[np.ndarray, float, dict]:
    """
    Fallback loader for old caches when master_meta is missing.
    Uses NPZ filename ordering + configured overlap to stitch a usable low-res raw reference.
    """
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
            f"No parseable segment NPZ files found for '{original_basename}' in {segment_folder}"
        )

    npz_files.sort(key=lambda x: x[0])
    inferred_total = max(item[1] for item in npz_files)
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
    return lowres_raw.astype(np.float32), float(lowres_fps), meta_stub


def _compute_tile_feather_weights(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    full_width: int,
    full_height: int,
    overlap_x_px: int,
    overlap_y_px: int,
    left_overlap_px: Optional[int] = None,
    right_overlap_px: Optional[int] = None,
    top_overlap_px: Optional[int] = None,
    bottom_overlap_px: Optional[int] = None,
) -> np.ndarray:
    height = y1 - y0
    width = x1 - x0
    left_ov = int(overlap_x_px) if left_overlap_px is None else int(left_overlap_px)
    right_ov = int(overlap_x_px) if right_overlap_px is None else int(right_overlap_px)
    top_ov = int(overlap_y_px) if top_overlap_px is None else int(top_overlap_px)
    bottom_ov = int(overlap_y_px) if bottom_overlap_px is None else int(bottom_overlap_px)
    left_ov = max(0, left_ov)
    right_ov = max(0, right_ov)
    top_ov = max(0, top_ov)
    bottom_ov = max(0, bottom_ov)
    if left_ov <= 0 and right_ov <= 0 and top_ov <= 0 and bottom_ov <= 0:
        return np.ones((height, width), dtype=np.float32)

    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    left_f = float(max(1, left_ov))
    right_f = float(max(1, right_ov))
    top_f = float(max(1, top_ov))
    bottom_f = float(max(1, bottom_ov))

    wx = np.ones((width,), dtype=np.float32)
    wy = np.ones((height,), dtype=np.float32)

    if x0 > 0 and left_ov > 0:
        wx = np.minimum(wx, np.clip(x / left_f, 0.0, 1.0))
    if x1 < full_width and right_ov > 0:
        wx = np.minimum(wx, np.clip((width - 1 - x) / right_f, 0.0, 1.0))
    if y0 > 0 and top_ov > 0:
        wy = np.minimum(wy, np.clip(y / top_f, 0.0, 1.0))
    if y1 < full_height and bottom_ov > 0:
        wy = np.minimum(wy, np.clip((height - 1 - y) / bottom_f, 0.0, 1.0))

    weights = np.outer(wy, wx).astype(np.float32)
    return np.clip(weights, 1e-3, 1.0)


def _estimate_alignment_ranges(
    pred_video: np.ndarray,
    tgt_video: np.ndarray,
    sample_frame_cap: int = 8,
    sample_spatial_stride: int = 8,
) -> Tuple[float, float, float, float]:
    t = int(pred_video.shape[0])
    if t <= 0:
        raise ValueError("Cannot estimate alignment ranges for empty tile data.")

    n_sample_frames = min(max(1, sample_frame_cap), t)
    frame_indices = np.linspace(0, t - 1, num=n_sample_frames, dtype=int)

    pred_samples = []
    tgt_samples = []
    stride = max(1, int(sample_spatial_stride))
    for fi in frame_indices:
        pred_s = pred_video[fi, ::stride, ::stride]
        tgt_s = tgt_video[fi, ::stride, ::stride]
        finite = np.isfinite(pred_s) & np.isfinite(tgt_s)
        if not np.any(finite):
            continue
        pred_samples.append(pred_s[finite].reshape(-1))
        tgt_samples.append(tgt_s[finite].reshape(-1))

    if not pred_samples or not tgt_samples:
        raise ValueError("No finite sampled values for alignment range estimation.")

    pred_concat = np.concatenate(pred_samples, axis=0)
    tgt_concat = np.concatenate(tgt_samples, axis=0)
    pred_low, pred_high = np.percentile(pred_concat, [0.5, 99.5]).astype(np.float64)
    tgt_low, tgt_high = np.percentile(tgt_concat, [0.5, 99.5]).astype(np.float64)

    if not np.isfinite(pred_low) or not np.isfinite(pred_high) or abs(pred_high - pred_low) < 1e-8:
        pmin = float(np.nanmin(pred_concat))
        pmax = float(np.nanmax(pred_concat))
        if not np.isfinite(pmin) or not np.isfinite(pmax):
            pred_low, pred_high = -1.0, 1.0
        elif abs(pmax - pmin) < 1e-8:
            pred_low, pred_high = pmin - 1.0, pmax + 1.0
        else:
            pred_low, pred_high = pmin, pmax

    if not np.isfinite(tgt_low) or not np.isfinite(tgt_high) or abs(tgt_high - tgt_low) < 1e-8:
        tmin = float(np.nanmin(tgt_concat))
        tmax = float(np.nanmax(tgt_concat))
        if not np.isfinite(tmin) or not np.isfinite(tmax):
            tgt_low, tgt_high = -1.0, 1.0
        elif abs(tmax - tmin) < 1e-8:
            tgt_low, tgt_high = tmin - 1.0, tmax + 1.0
        else:
            tgt_low, tgt_high = tmin, tmax

    return float(pred_low), float(pred_high), float(tgt_low), float(tgt_high)


def _build_weighted_intensity_lut(
    pred_video: np.ndarray,
    tgt_video: np.ndarray,
    spatial_weights_2d: Optional[np.ndarray],
    num_bins: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    pred_low, pred_high, tgt_low, tgt_high = _estimate_alignment_ranges(pred_video, tgt_video)
    if pred_high <= pred_low:
        pred_high = pred_low + 1e-6
    if tgt_high <= tgt_low:
        tgt_high = tgt_low + 1e-6

    bins = max(256, int(num_bins))
    pred_edges = np.linspace(pred_low, pred_high, bins + 1, dtype=np.float64)
    tgt_edges = np.linspace(tgt_low, tgt_high, bins + 1, dtype=np.float64)
    pred_hist = np.zeros((bins,), dtype=np.float64)
    tgt_hist = np.zeros((bins,), dtype=np.float64)

    for frame_idx in range(int(pred_video.shape[0])):
        pred = pred_video[frame_idx]
        tgt = tgt_video[frame_idx]
        finite = np.isfinite(pred) & np.isfinite(tgt)
        if not np.any(finite):
            continue
        pred_vals = np.clip(pred[finite], pred_low, pred_high)
        tgt_vals = np.clip(tgt[finite], tgt_low, tgt_high)
        if spatial_weights_2d is not None:
            frame_weights = spatial_weights_2d[finite].astype(np.float64, copy=False)
        else:
            frame_weights = None
        pred_hist += np.histogram(pred_vals, bins=pred_edges, weights=frame_weights)[0]
        tgt_hist += np.histogram(tgt_vals, bins=tgt_edges, weights=frame_weights)[0]

    pred_mass = float(np.sum(pred_hist))
    tgt_mass = float(np.sum(tgt_hist))
    if pred_mass <= 0.0 or tgt_mass <= 0.0:
        # Fallback to simple linear mapping from estimated robust ranges.
        scale = (tgt_high - tgt_low) / max(pred_high - pred_low, 1e-6)
        shift = tgt_low - scale * pred_low
        lut_x = np.array([pred_low, pred_high], dtype=np.float32)
        lut_y = np.array([scale * pred_low + shift, scale * pred_high + shift], dtype=np.float32)
        diag = {
            "pred_low": float(pred_low),
            "pred_high": float(pred_high),
            "tgt_low": float(tgt_low),
            "tgt_high": float(tgt_high),
            "mode": "linear_fallback_no_hist",
        }
        return lut_x, lut_y, diag

    pred_cdf = np.cumsum(pred_hist)
    pred_cdf /= max(pred_cdf[-1], 1e-12)
    tgt_cdf = np.cumsum(tgt_hist)
    tgt_cdf /= max(tgt_cdf[-1], 1e-12)

    pred_centers = 0.5 * (pred_edges[:-1] + pred_edges[1:])
    tgt_centers = 0.5 * (tgt_edges[:-1] + tgt_edges[1:])

    # Deduplicate CDF entries for stable interpolation.
    uniq_mask = np.r_[True, np.diff(tgt_cdf) > 1e-12]
    tgt_cdf_unique = tgt_cdf[uniq_mask]
    tgt_centers_unique = tgt_centers[uniq_mask]
    if tgt_cdf_unique.size < 2:
        scale = (tgt_high - tgt_low) / max(pred_high - pred_low, 1e-6)
        shift = tgt_low - scale * pred_low
        lut_x = np.array([pred_low, pred_high], dtype=np.float32)
        lut_y = np.array([scale * pred_low + shift, scale * pred_high + shift], dtype=np.float32)
        diag = {
            "pred_low": float(pred_low),
            "pred_high": float(pred_high),
            "tgt_low": float(tgt_low),
            "tgt_high": float(tgt_high),
            "mode": "linear_fallback_low_cdf_variance",
        }
        return lut_x, lut_y, diag

    lut_y = np.interp(
        pred_cdf,
        tgt_cdf_unique,
        tgt_centers_unique,
        left=tgt_centers_unique[0],
        right=tgt_centers_unique[-1],
    )
    # Light smoothing and monotonic enforcement for a stable LUT.
    if lut_y.size >= 9:
        kernel = np.ones((9,), dtype=np.float64) / 9.0
        lut_y = np.convolve(lut_y, kernel, mode="same")
    lut_y = np.maximum.accumulate(lut_y)

    diag = {
        "pred_low": float(pred_low),
        "pred_high": float(pred_high),
        "tgt_low": float(tgt_low),
        "tgt_high": float(tgt_high),
        "mode": "weighted_histogram_lut",
    }
    return pred_centers.astype(np.float32), lut_y.astype(np.float32), diag


def _apply_lut_video(video_data: np.ndarray, lut_x: np.ndarray, lut_y: np.ndarray) -> np.ndarray:
    out = np.empty_like(video_data, dtype=np.float32)
    for frame_idx in range(int(video_data.shape[0])):
        frame = video_data[frame_idx].astype(np.float32, copy=False)
        mapped = np.interp(frame.reshape(-1), lut_x, lut_y, left=lut_y[0], right=lut_y[-1])
        out[frame_idx] = mapped.reshape(frame.shape).astype(np.float32)
    return out


def _axis_window_starts(length: int, window: int, stride: int) -> List[int]:
    length = int(length)
    window = max(1, int(window))
    stride = max(1, int(stride))
    if length <= window:
        return [0]

    starts = list(range(0, max(1, length - window + 1), stride))
    last_start = int(length - window)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _build_window_center_weights(height: int, width: int) -> np.ndarray:
    if height <= 1:
        wy = np.ones((1,), dtype=np.float32)
    else:
        wy = np.hanning(height).astype(np.float32)
        wy = np.clip(wy, 1e-3, None)
    if width <= 1:
        wx = np.ones((1,), dtype=np.float32)
    else:
        wx = np.hanning(width).astype(np.float32)
        wx = np.clip(wx, 1e-3, None)
    weights = np.outer(wy, wx).astype(np.float32)
    return np.clip(weights, 1e-3, 1.0)


def _robust_frame_normalize(frame_2d: np.ndarray) -> np.ndarray:
    frame = frame_2d.astype(np.float32, copy=False)
    finite = np.isfinite(frame)
    if not np.any(finite):
        return np.zeros_like(frame, dtype=np.float32)

    vals = frame[finite]
    low, high = np.percentile(vals, [1.0, 99.0]).astype(np.float32)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low + 1e-6:
        low = float(np.nanmin(vals))
        high = float(np.nanmax(vals))
    scale = float(high - low)
    if not np.isfinite(scale) or scale <= 1e-6:
        out = np.zeros_like(frame, dtype=np.float32)
        out[finite] = 0.5
        return out

    out = np.zeros_like(frame, dtype=np.float32)
    out[finite] = np.clip((frame[finite] - low) / scale, 0.0, 1.0)
    return out


def _smooth_frame(frame_2d: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    k = max(1, int(kernel_size))
    if k <= 1:
        return frame_2d.astype(np.float32, copy=False)
    if (k % 2) == 0:
        k += 1
    tensor = torch.from_numpy(frame_2d.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    smoothed = F.avg_pool2d(tensor, kernel_size=k, stride=1, padding=k // 2)
    return smoothed.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def _gradient_magnitude(frame_2d: np.ndarray) -> np.ndarray:
    frame = frame_2d.astype(np.float32, copy=False)
    gx = np.diff(frame, axis=1, append=frame[:, -1:])
    gy = np.diff(frame, axis=0, append=frame[-1:, :])
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 8 or b.size < 8 or a.size != b.size:
        return 0.0
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a_mean = float(np.mean(a))
    b_mean = float(np.mean(b))
    a_center = a - a_mean
    b_center = b - b_mean
    denom = float(np.sqrt(np.sum(a_center * a_center) * np.sum(b_center * b_center)))
    if not np.isfinite(denom) or denom <= 1e-12:
        return 0.0
    corr = float(np.sum(a_center * b_center) / denom)
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


def _window_feature_agreement_score(
    pred_depth_win: np.ndarray,
    tgt_depth_win: np.ndarray,
    pred_grad_win: np.ndarray,
    tgt_grad_win: np.ndarray,
) -> float:
    finite = (
        np.isfinite(pred_depth_win)
        & np.isfinite(tgt_depth_win)
        & np.isfinite(pred_grad_win)
        & np.isfinite(tgt_grad_win)
    )
    if np.count_nonzero(finite) < 64:
        return 0.0

    pred_d = pred_depth_win[finite]
    tgt_d = tgt_depth_win[finite]
    pred_g = pred_grad_win[finite]
    tgt_g = tgt_grad_win[finite]

    depth_corr = max(0.0, _pearson_corr(pred_d, tgt_d))
    grad_corr = max(0.0, _pearson_corr(pred_g, tgt_g))

    if pred_g.size < 16 or tgt_g.size < 16:
        edge_recall = 0.0
        edge_energy_ratio = 0.0
    else:
        pred_thr = float(np.percentile(pred_g, 85.0))
        tgt_thr = float(np.percentile(tgt_g, 85.0))
        pred_edge = pred_g >= pred_thr
        tgt_edge = tgt_g >= tgt_thr
        inter = float(np.count_nonzero(pred_edge & tgt_edge))
        tgt_count = float(np.count_nonzero(tgt_edge))
        edge_recall = inter / max(tgt_count, 1.0)
        if tgt_count > 0:
            tgt_energy = float(np.mean(tgt_g[tgt_edge]))
            pred_energy_on_tgt = float(np.mean(pred_g[tgt_edge]))
            edge_energy_ratio = pred_energy_on_tgt / max(tgt_energy, 1e-6)
        else:
            edge_energy_ratio = 0.0
        edge_energy_ratio = float(np.clip(edge_energy_ratio, 0.0, 1.0))

    score = (
        0.50 * edge_recall
        + 0.25 * edge_energy_ratio
        + 0.15 * grad_corr
        + 0.10 * depth_corr
    )
    return float(np.clip(score, 0.0, 1.0))


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
    return (x * x * (3.0 - 2.0 * x)).astype(np.float32)


def _compute_local_confidence_map(
    pred_video: np.ndarray,
    target_video: np.ndarray,
    window_size: int = 64,
    window_stride: int = 32,
    sample_frame_cap: int = 8,
    score_confidence_low: float = 0.45,
    score_confidence_high: float = 0.80,
    log_prefix: Optional[str] = None,
    log_sample_limit: int = 3,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if pred_video.shape != target_video.shape:
        raise ValueError(
            f"Local confidence map shape mismatch: pred={pred_video.shape}, target={target_video.shape}"
        )

    t, h, w = int(pred_video.shape[0]), int(pred_video.shape[1]), int(pred_video.shape[2])
    if t <= 0 or h <= 0 or w <= 0:
        raise ValueError("Cannot compute local confidence map for empty video data.")

    win_h = min(int(window_size), h)
    win_w = min(int(window_size), w)
    stride_h = min(max(1, int(window_stride)), win_h)
    stride_w = min(max(1, int(window_stride)), win_w)

    y_starts = _axis_window_starts(h, win_h, stride_h)
    x_starts = _axis_window_starts(w, win_w, stride_w)
    ny = len(y_starts)
    nx = len(x_starts)
    if ny <= 0 or nx <= 0:
        return np.ones((h, w), dtype=np.float32), {
            "windows_x": 0,
            "windows_y": 0,
            "sample_frames_used": 0,
            "confidence_mean": 1.0,
            "fallback_mean": 0.0,
        }

    n_sample_frames = min(max(1, int(sample_frame_cap)), t)
    frame_indices = np.linspace(0, t - 1, num=n_sample_frames, dtype=int)

    score_sums = np.zeros((ny, nx), dtype=np.float64)
    score_counts = np.zeros((ny, nx), dtype=np.float64)
    low = float(score_confidence_low)
    high = float(score_confidence_high)
    if high <= low:
        high = low + 1e-3

    local_start = time.perf_counter()
    if log_prefix:
        _logger.info(
            f"{log_prefix}: local reliability sampling start | "
            f"frames={len(frame_indices)}, window={int(win_w)}x{int(win_h)}, stride={int(stride_w)}x{int(stride_h)}, "
            f"window_grid={int(nx)}x{int(ny)}, conf_low/high={low:.3f}/{high:.3f}"
        )

    for sample_idx, fi in enumerate(frame_indices, start=1):
        pred_norm = _smooth_frame(_robust_frame_normalize(pred_video[fi]), kernel_size=5)
        tgt_norm = _smooth_frame(_robust_frame_normalize(target_video[fi]), kernel_size=5)
        pred_grad = _gradient_magnitude(pred_norm)
        tgt_grad = _gradient_magnitude(tgt_norm)
        frame_scores = np.zeros((ny, nx), dtype=np.float32)

        for yi, y0 in enumerate(y_starts):
            y1 = y0 + win_h
            for xi, x0 in enumerate(x_starts):
                x1 = x0 + win_w
                score = _window_feature_agreement_score(
                    pred_norm[y0:y1, x0:x1],
                    tgt_norm[y0:y1, x0:x1],
                    pred_grad[y0:y1, x0:x1],
                    tgt_grad[y0:y1, x0:x1],
                )
                frame_scores[yi, xi] = float(score)
                score_sums[yi, xi] += float(score)
                score_counts[yi, xi] += 1.0

        should_log_sample = bool(
            log_prefix
            and (
                sample_idx <= max(0, int(log_sample_limit))
                or sample_idx == len(frame_indices)
            )
        )
        if should_log_sample:
            frame_conf = _smoothstep01((frame_scores - low) / (high - low))
            _logger.info(
                f"{log_prefix}: sample frame {sample_idx}/{len(frame_indices)} (raw_idx={int(fi)}) | "
                f"score mean/min/max={float(np.mean(frame_scores)):.3f}/{float(np.min(frame_scores)):.3f}/{float(np.max(frame_scores)):.3f}, "
                f"conf mean/min/max={float(np.mean(frame_conf)):.3f}/{float(np.min(frame_conf)):.3f}/{float(np.max(frame_conf)):.3f}, "
                f"fallback_mean={float(np.mean(1.0 - frame_conf)):.3f}"
            )

    avg_scores = np.divide(
        score_sums,
        np.maximum(score_counts, 1.0),
        out=np.zeros_like(score_sums, dtype=np.float64),
    ).astype(np.float32)
    conf_windows = _smoothstep01((avg_scores - low) / (high - low))

    window_weights = _build_window_center_weights(win_h, win_w)
    conf_num = np.zeros((h, w), dtype=np.float32)
    conf_den = np.zeros((h, w), dtype=np.float32)

    for yi, y0 in enumerate(y_starts):
        y1 = y0 + win_h
        for xi, x0 in enumerate(x_starts):
            x1 = x0 + win_w
            conf_val = float(conf_windows[yi, xi])
            conf_num[y0:y1, x0:x1] += window_weights * conf_val
            conf_den[y0:y1, x0:x1] += window_weights

    confidence = np.divide(
        conf_num,
        np.maximum(conf_den, 1e-6),
        out=np.ones_like(conf_num, dtype=np.float32),
    )
    confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)
    fallback_alpha = 1.0 - confidence

    diag = {
        "window_size": int(window_size),
        "window_stride": int(window_stride),
        "windows_x": int(nx),
        "windows_y": int(ny),
        "sample_frames_used": int(len(frame_indices)),
        "score_mean": float(np.mean(avg_scores)),
        "score_min": float(np.min(avg_scores)),
        "score_max": float(np.max(avg_scores)),
        "confidence_mean": float(np.mean(confidence)),
        "confidence_min": float(np.min(confidence)),
        "confidence_max": float(np.max(confidence)),
        "fallback_mean": float(np.mean(fallback_alpha)),
        "fallback_gt_50pct_area": float(np.mean(fallback_alpha > 0.5)),
    }
    if log_prefix:
        _logger.info(
            f"{log_prefix}: local reliability sampling complete in {time.perf_counter() - local_start:.1f}s | "
            f"score mean/min/max={diag['score_mean']:.3f}/{diag['score_min']:.3f}/{diag['score_max']:.3f}, "
            f"conf mean/min/max={diag['confidence_mean']:.3f}/{diag['confidence_min']:.3f}/{diag['confidence_max']:.3f}, "
            f"fallback_mean={diag['fallback_mean']:.3f}"
        )
    return confidence, diag


def _stabilize_local_confidence_map(
    confidence_map: np.ndarray,
    feather_weights_2d: np.ndarray,
    smoothing_kernel_size: int = 7,
    edge_conf_floor: float = 0.35,
) -> np.ndarray:
    conf = np.clip(confidence_map, 0.0, 1.0).astype(np.float32, copy=False)
    smooth_conf = _smooth_frame(conf, kernel_size=max(3, int(smoothing_kernel_size)))
    conf = 0.50 * conf + 0.50 * np.clip(smooth_conf, 0.0, 1.0).astype(np.float32, copy=False)

    edge_floor = float(np.clip(edge_conf_floor, 0.0, 1.0))
    edge_gate = _smoothstep01(np.clip(feather_weights_2d.astype(np.float32, copy=False), 0.0, 1.0))
    conf *= (edge_floor + (1.0 - edge_floor) * edge_gate)
    return np.clip(conf, 0.0, 1.0).astype(np.float32)


def _estimate_sampled_video_stats(
    video_data: np.ndarray,
    sample_frame_cap: int = 12,
    sample_spatial_stride: int = 4,
    core_low_pct: float = 5.0,
    core_high_pct: float = 95.0,
    robust_low_pct: float = 0.10,
    robust_high_pct: float = 99.90,
) -> Dict[str, float]:
    if video_data.size <= 0:
        return {
            "count": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "core_mean": 0.0,
            "core_std": 0.0,
            "robust_low": 0.0,
            "robust_high": 1.0,
            "core_low": 0.0,
            "core_high": 1.0,
        }

    t = int(video_data.shape[0])
    n_frames = min(max(1, int(sample_frame_cap)), t)
    frame_indices = np.linspace(0, t - 1, num=n_frames, dtype=int)
    stride = max(1, int(sample_spatial_stride))
    sampled_values: List[np.ndarray] = []

    for fi in frame_indices:
        sample = video_data[fi, ::stride, ::stride].astype(np.float32, copy=False)
        finite = np.isfinite(sample)
        if np.any(finite):
            sampled_values.append(sample[finite].reshape(-1))

    if not sampled_values:
        return {
            "count": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "core_mean": 0.0,
            "core_std": 0.0,
            "robust_low": 0.0,
            "robust_high": 1.0,
            "core_low": 0.0,
            "core_high": 1.0,
        }

    vals = np.concatenate(sampled_values, axis=0).astype(np.float32, copy=False)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    vmed = float(np.median(vals))
    core_low = float(np.percentile(vals, float(core_low_pct)))
    core_high = float(np.percentile(vals, float(core_high_pct)))
    robust_low = float(np.percentile(vals, float(robust_low_pct)))
    robust_high = float(np.percentile(vals, float(robust_high_pct)))

    if core_high <= core_low + 1e-8:
        core_mean = float(np.mean(vals))
        core_std = float(np.std(vals))
    else:
        core_mask = (vals >= core_low) & (vals <= core_high)
        core_vals = vals[core_mask] if np.any(core_mask) else vals
        core_mean = float(np.mean(core_vals))
        core_std = float(np.std(core_vals))

    return {
        "count": float(vals.size),
        "min": vmin,
        "max": vmax,
        "median": vmed,
        "core_mean": core_mean,
        "core_std": core_std,
        "robust_low": robust_low,
        "robust_high": robust_high,
        "core_low": core_low,
        "core_high": core_high,
    }


def _harmonize_tile_to_existing_overlap(
    tile_video: np.ndarray,
    accum_video: np.ndarray,
    accum_weight_2d: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    sample_frame_cap: int = 6,
    min_overlap_pixels: int = 2048,
) -> Tuple[np.ndarray, Dict[str, float]]:
    overlap_weights = accum_weight_2d[y0:y1, x0:x1].astype(np.float32, copy=False)
    overlap_mask = overlap_weights > 1e-6
    overlap_pixels = int(np.count_nonzero(overlap_mask))

    diag: Dict[str, float] = {
        "applied": 0.0,
        "overlap_pixels": float(overlap_pixels),
        "sample_frames_used": 0.0,
        "gain": 1.0,
        "offset": 0.0,
        "offset_cap": 0.0,
        "rmse_before": 0.0,
        "rmse_after": 0.0,
        "rmse_improvement_ratio": 0.0,
    }

    if overlap_pixels < max(1, int(min_overlap_pixels)):
        return tile_video, diag

    t = int(tile_video.shape[0])
    sample_frames = min(max(1, int(sample_frame_cap)), t)
    frame_indices = np.linspace(0, t - 1, num=sample_frames, dtype=int)
    pred_samples: List[np.ndarray] = []
    tgt_samples: List[np.ndarray] = []

    for fi in frame_indices:
        existing_frame = accum_video[fi, y0:y1, x0:x1]
        existing_frame = np.divide(
            existing_frame,
            np.maximum(overlap_weights, 1e-6),
            out=np.zeros_like(existing_frame, dtype=np.float32),
        )
        pred_vals = tile_video[fi][overlap_mask].astype(np.float32, copy=False)
        tgt_vals = existing_frame[overlap_mask].astype(np.float32, copy=False)
        finite = np.isfinite(pred_vals) & np.isfinite(tgt_vals)
        if not np.any(finite):
            continue
        pred_samples.append(pred_vals[finite].reshape(-1))
        tgt_samples.append(tgt_vals[finite].reshape(-1))

    if not pred_samples or not tgt_samples:
        return tile_video, diag

    pred = np.concatenate(pred_samples, axis=0).astype(np.float32, copy=False)
    tgt = np.concatenate(tgt_samples, axis=0).astype(np.float32, copy=False)
    if pred.size < max(256, min_overlap_pixels // 4):
        return tile_video, diag

    pred_p10, pred_p50, pred_p90 = np.percentile(pred, [10.0, 50.0, 90.0]).astype(np.float64)
    tgt_p10, tgt_p50, tgt_p90 = np.percentile(tgt, [10.0, 50.0, 90.0]).astype(np.float64)

    pred_span = float(pred_p90 - pred_p10)
    tgt_span = float(tgt_p90 - tgt_p10)
    if pred_span <= 1e-8 or not np.isfinite(pred_span) or not np.isfinite(tgt_span):
        gain = 1.0
    else:
        gain = float(np.clip(tgt_span / pred_span, 0.90, 1.10))
    offset = float(tgt_p50 - gain * pred_p50)
    span_ref = float(max(pred_span, tgt_span, 1e-6))
    offset_cap = float(max(0.02, min(0.08, 0.35 * span_ref)))
    offset = float(np.clip(offset, -offset_cap, offset_cap))
    if not np.isfinite(gain):
        gain = 1.0
    if not np.isfinite(offset):
        offset = 0.0

    rmse_before = float(np.sqrt(np.mean((pred - tgt) ** 2)))
    rmse_after = float(np.sqrt(np.mean(((pred * gain + offset) - tgt) ** 2)))
    improve_ratio = float((rmse_before - rmse_after) / max(rmse_before, 1e-6))
    if rmse_before >= 0.01 and improve_ratio >= 0.03:
        tile_video *= gain
        tile_video += offset
        applied = 1.0
    else:
        gain = 1.0
        offset = 0.0
        offset_cap = 0.0
        rmse_after = rmse_before
        improve_ratio = 0.0
        applied = 0.0

    diag.update(
        {
            "applied": float(applied),
            "sample_frames_used": float(sample_frames),
            "gain": float(gain),
            "offset": float(offset),
            "offset_cap": float(offset_cap),
            "rmse_before": float(rmse_before),
            "rmse_after": float(rmse_after),
            "rmse_improvement_ratio": float(improve_ratio),
        }
    )
    return tile_video, diag


def _align_tile_frames_to_reference(
    tile_frames_raw: np.ndarray,
    reference_full_res_raw: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    spatial_weights_2d: Optional[np.ndarray] = None,
    tile_id_for_log: Optional[int] = None,
) -> np.ndarray:
    target_tile = reference_full_res_raw[:, y0:y1, x0:x1].astype(np.float32, copy=False)
    pred_tile = tile_frames_raw.astype(np.float32, copy=False)
    if pred_tile.shape != target_tile.shape:
        raise ValueError(
            f"Tile/reference shape mismatch: pred={pred_tile.shape}, target={target_tile.shape}"
        )

    lut_x, lut_y, diag = _build_weighted_intensity_lut(
        pred_tile,
        target_tile,
        spatial_weights_2d=spatial_weights_2d,
    )
    aligned = _apply_lut_video(pred_tile, lut_x, lut_y)
    tile_label = f"{tile_id_for_log}" if tile_id_for_log is not None else "?"
    _logger.debug(
        f"Tile {tile_label} LUT alignment: mode={diag.get('mode')} "
        f"pred_range={diag.get('pred_low'):.4f}..{diag.get('pred_high'):.4f} -> "
        f"target_range={diag.get('tgt_low'):.4f}..{diag.get('tgt_high'):.4f}"
    )
    return aligned


def _try_load_cached_tile_raw(
    tile_seg_folder: str,
    tile_basename: str,
    temporal_overlap_frames: int,
    process_fps: float,
    temporal_merge_alignment: str,
    expected_frames: int,
    expected_height: int,
    expected_width: int,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[str]]:
    if not os.path.isdir(tile_seg_folder):
        return None, None, "segment folder not found"

    try:
        fallback_fps = float(process_fps) if process_fps and process_fps > 0 else 23.976
        cached_raw, cached_fps, _ = _build_raw_reference_from_legacy_npz_folder(
            segment_folder=tile_seg_folder,
            original_basename=tile_basename,
            overlap_frames=int(temporal_overlap_frames),
            fallback_fps=float(fallback_fps),
            merge_alignment_method=temporal_merge_alignment,
        )
    except Exception as cache_err:
        return None, None, f"cache load error: {cache_err}"

    if cached_raw.ndim != 3:
        return None, None, f"unexpected cached tile rank {cached_raw.ndim} (expected 3)"
    if cached_raw.shape[1] != expected_height or cached_raw.shape[2] != expected_width:
        return (
            None,
            None,
            f"cached tile size {cached_raw.shape[2]}x{cached_raw.shape[1]} "
            f"does not match expected {expected_width}x{expected_height}",
        )
    if cached_raw.shape[0] < expected_frames:
        return (
            None,
            None,
            f"cached tile has {cached_raw.shape[0]} frames, expected at least {expected_frames}",
        )
    if cached_raw.shape[0] > expected_frames:
        _logger.info(
            f"Tile cache '{tile_basename}' has extra frames ({cached_raw.shape[0]} > {expected_frames}); "
            f"truncating to expected frame count."
        )

    return cached_raw[:expected_frames].astype(np.float32), float(cached_fps), None


def _infer_single_tile_raw(
    demo,
    source_video_path: str,
    tile_frames: np.ndarray,
    tile_basename: str,
    temp_output_root: str,
    process_fps: float,
    temporal_window_frames: int,
    temporal_overlap_frames: int,
    guidance_scale: float,
    inference_steps: int,
    seed: int,
    temporal_merge_alignment: str,
    reuse_existing_segments: bool = True,
) -> Tuple[np.ndarray, float, str, bool]:
    expected_frames = int(tile_frames.shape[0])
    expected_height = int(tile_frames.shape[1])
    expected_width = int(tile_frames.shape[2])
    tile_seg_folder = os.path.join(temp_output_root, get_segment_output_folder_name(tile_basename))

    cached_raw = None
    cached_fps = None
    if reuse_existing_segments:
        cached_raw, cached_fps, cache_reason = _try_load_cached_tile_raw(
            tile_seg_folder=tile_seg_folder,
            tile_basename=tile_basename,
            temporal_overlap_frames=int(temporal_overlap_frames),
            process_fps=float(process_fps),
            temporal_merge_alignment=temporal_merge_alignment,
            expected_frames=expected_frames,
            expected_height=expected_height,
            expected_width=expected_width,
        )
        if cached_raw is not None:
            _logger.info(f"Reusing cached tile data for {tile_basename} from: {tile_seg_folder}")
            fps_out = float(cached_fps) if cached_fps and cached_fps > 0 else float(process_fps)
            if fps_out <= 0:
                fps_out = 23.976
            return cached_raw, fps_out, tile_seg_folder, True
        if cache_reason:
            _logger.info(f"Tile {tile_basename}: cache reuse unavailable ({cache_reason}). Running inference.")

    segments = _define_temporal_segments_by_frame_count(
        total_frames=int(tile_frames.shape[0]),
        window_frames=int(temporal_window_frames),
        overlap_frames=int(temporal_overlap_frames),
    )
    if not segments:
        raise ValueError(f"No temporal segments generated for tile '{tile_basename}'.")
    _logger.info(
        f"Tile {tile_basename}: running inference over {len(segments)} temporal segment(s) "
        f"(window={int(temporal_window_frames)}, overlap={int(temporal_overlap_frames)})."
    )

    successful_jobs: List[dict] = []
    for seg in segments:
        start_idx = int(seg["start_frame_raw_index"])
        count = int(seg["num_frames_to_load_raw"])
        total_segments = int(seg["total_segments"])
        seg_id = int(seg["segment_id"])
        seg_end = start_idx + count - 1
        segment_frames = tile_frames[start_idx:start_idx + count]
        if segment_frames.size == 0:
            raise ValueError(f"Tile '{tile_basename}' segment {seg['segment_id']} has zero frames.")
        _logger.info(
            f"Tile {tile_basename}: temporal segment {seg_id + 1}/{total_segments} "
            f"(frames {start_idx}-{seg_end}, count={count}) starting inference..."
        )

        seg_job_info = {
            "segment_id": int(seg_id),
            "total_segments": int(total_segments),
            "start_frame_raw_index": int(start_idx),
            "num_frames_to_load_raw": int(count),
            "original_video_fps": float(process_fps),
            "gui_fps_setting_at_definition": float(process_fps),
            "original_basename": tile_basename,
            "source_type": "video_file",
            "video_path": source_video_path,
        }

        heartbeat_stop = threading.Event()
        seg_start_ts = time.perf_counter()

        def _segment_heartbeat() -> None:
            while not heartbeat_stop.wait(20.0):
                elapsed = time.perf_counter() - seg_start_ts
                _logger.info(
                    f"Tile {tile_basename}: temporal segment {seg_id + 1}/{total_segments} still running "
                    f"({elapsed:.1f}s elapsed)..."
                )

        heartbeat_thread = threading.Thread(target=_segment_heartbeat, daemon=True)
        heartbeat_thread.start()
        try:
            _, job_meta = demo.run(
                video_path_or_frames_or_info=segment_frames,
                num_denoising_steps=inference_steps,
                guidance_scale=guidance_scale,
                base_output_folder=temp_output_root,
                gui_window_size=temporal_window_frames,
                gui_overlap=temporal_overlap_frames,
                process_length_for_read_full_video=-1,
                target_height=int(segment_frames.shape[1]),
                target_width=int(segment_frames.shape[2]),
                seed=seed,
                original_video_basename_override=tile_basename,
                segment_job_info_param=seg_job_info,
                keep_intermediate_npz_config=False,
                intermediate_segment_visual_format_config="none",
                save_final_json_for_this_job_config=False,
            )
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=0.2)

        seg_elapsed = time.perf_counter() - seg_start_ts
        _logger.info(
            f"Tile {tile_basename}: temporal segment {seg_id + 1}/{total_segments} "
            f"finished in {seg_elapsed:.1f}s."
        )

        if not job_meta or job_meta.get("status") != "success" or not job_meta.get("output_segment_filename"):
            raise RuntimeError(
                f"Tile '{tile_basename}' segment {seg['segment_id']} failed with status: "
                f"{job_meta.get('status', 'unknown') if job_meta else 'missing_meta'}"
            )
        successful_jobs.append(job_meta)

    _logger.info(f"Tile {tile_basename}: loading/stitching temporal segment outputs...")
    sorted_jobs = sorted(successful_jobs, key=lambda x: int(x.get("segment_id", -1)))
    if len(sorted_jobs) == 1:
        tile_raw, tile_fps = merge_depth_segments._load_single_segment_frames(sorted_jobs[0], tile_seg_folder)
    else:
        loaded_frames, job_meta_map, tile_fps = merge_depth_segments._load_multiple_segments_data(sorted_jobs, tile_seg_folder)
        if len(loaded_frames) == 1:
            tile_raw = loaded_frames[0]
        else:
            aligned = merge_depth_segments._align_segments_data(
                loaded_frames,
                job_meta_map,
                temporal_overlap_frames,
                temporal_merge_alignment,
            )
            tile_raw = merge_depth_segments._stitch_and_blend_segments_data(
                aligned,
                job_meta_map,
                temporal_overlap_frames,
            )
    _logger.info(
        f"Tile {tile_basename}: temporal stitching complete | "
        f"frames={int(tile_raw.shape[0])}, size={int(tile_raw.shape[2])}x{int(tile_raw.shape[1])}, fps={float(tile_fps):.3f}"
    )

    return tile_raw.astype(np.float32), float(tile_fps), tile_seg_folder, False


def run_spatial_hires_refine(
    demo,
    source_video_path: str,
    original_basename: str,
    output_dir: str,
    lowres_master_meta_path: str,
    target_height: int,
    target_width: int,
    tile_num_x: int,
    tile_num_y: int,
    tile_overlap_x_px: int,
    tile_overlap_y_px: int,
    temporal_window_frames: int,
    temporal_overlap_frames: int,
    guidance_scale: float,
    inference_steps: int,
    seed: int,
    target_fps_setting: float = -1.0,
    lowres_anchor_weight: float = 0.15,
    output_suffix: str = "_hires_refined_depth",
    output_format: str = "main10_mp4",
    cleanup_temp: bool = True,
    temporal_merge_alignment: str = "shift_scale",
    lowres_segment_folder: Optional[str] = None,
    allow_legacy_npz_fallback: bool = True,
    local_reliability_window_size: int = 64,
    local_reliability_window_stride: int = 32,
    local_reliability_score_confidence_low: float = 0.45,
    local_reliability_score_confidence_high: float = 0.80,
    edge_guided_fallback_enabled: bool = False,
    edge_guided_fallback_mix: float = 0.75,
    edge_guided_strength: float = 0.90,
    edge_guided_sigma_color: float = 0.04,
    edge_guided_sigma_spatial: float = 0.90,
    edge_guided_bilateral_iterations: int = 1,
    edge_guided_temporal_smooth: float = 0.03,
    edge_guided_reinject_strength: float = 0.60,
) -> Tuple[str, dict]:
    start_time = time.perf_counter()
    _logger.info(
        f"Starting Hi-Res Spatial Refine for {original_basename}: "
        f"tile_grid={tile_num_x}x{tile_num_y}, overlap_xy={tile_overlap_x_px}/{tile_overlap_y_px}, "
        f"target={target_width}x{target_height}"
    )
    if tile_num_x < 1 or tile_num_y < 1:
        raise ValueError("Tile Number X/Y must both be >= 1.")
    if tile_overlap_x_px < 0 or tile_overlap_y_px < 0:
        raise ValueError("Tile Overlap X/Y must both be >= 0.")
    if temporal_window_frames <= 0:
        raise ValueError("Temporal window must be > 0.")
    if temporal_overlap_frames < 0 or temporal_overlap_frames >= temporal_window_frames:
        raise ValueError("Temporal overlap must be >=0 and < temporal window.")
    if local_reliability_window_size <= 0 or local_reliability_window_stride <= 0:
        raise ValueError("Local reliability window size/stride must be > 0.")
    if local_reliability_score_confidence_low < 0.0 or local_reliability_score_confidence_high > 1.0:
        raise ValueError("Local reliability confidence thresholds must be within [0, 1].")
    if local_reliability_score_confidence_high <= local_reliability_score_confidence_low:
        raise ValueError("Local reliability confidence high threshold must be greater than low threshold.")
    if edge_guided_fallback_mix < 0.0 or edge_guided_fallback_mix > 1.0:
        raise ValueError("Edge-guided fallback mix must be within [0, 1].")
    if edge_guided_strength < 0.0 or edge_guided_strength > 1.0:
        raise ValueError("Edge-guided strength must be within [0, 1].")
    if edge_guided_sigma_color <= 0.0 or edge_guided_sigma_spatial <= 0.0:
        raise ValueError("Edge-guided sigma color/spatial must both be > 0.")
    if edge_guided_bilateral_iterations < 0:
        raise ValueError("Edge-guided bilateral iterations must be >= 0.")
    if edge_guided_temporal_smooth < 0.0 or edge_guided_temporal_smooth > 1.0:
        raise ValueError("Edge-guided temporal smooth must be within [0, 1].")
    if edge_guided_reinject_strength < 0.0 or edge_guided_reinject_strength > 1.0:
        raise ValueError("Edge-guided reinject strength must be within [0, 1].")
    if lowres_segment_folder is None and lowres_master_meta_path:
        lowres_segment_folder = os.path.dirname(lowres_master_meta_path)

    os.makedirs(output_dir, exist_ok=True)
    temp_root = os.path.join(output_dir, f"{original_basename}_hires_refine_tmp")
    os.makedirs(temp_root, exist_ok=True)

    lowres_load_start = time.perf_counter()
    _logger.info(f"{original_basename}: loading low-res raw reference...")
    if lowres_master_meta_path and os.path.exists(lowres_master_meta_path):
        lowres_raw, lowres_fps, lowres_meta = _build_raw_reference_from_master(
            lowres_master_meta_path,
            merge_alignment_method=temporal_merge_alignment,
        )
    elif allow_legacy_npz_fallback and lowres_segment_folder and os.path.isdir(lowres_segment_folder):
        _logger.warning(
            f"Master metadata missing for {original_basename}. "
            f"Falling back to legacy NPZ cache parsing in {lowres_segment_folder}."
        )
        fallback_fps = target_fps_setting if target_fps_setting and target_fps_setting > 0 else 23.976
        lowres_raw, lowres_fps, lowres_meta = _build_raw_reference_from_legacy_npz_folder(
            segment_folder=lowres_segment_folder,
            original_basename=original_basename,
            overlap_frames=int(temporal_overlap_frames),
            fallback_fps=float(fallback_fps),
            merge_alignment_method=temporal_merge_alignment,
        )
        lowres_master_meta_path = ""  # legacy cache path used; no master metadata on disk.
    else:
        raise FileNotFoundError(
            f"Missing low-res cache inputs for {original_basename}. "
            f"Master meta path: '{lowres_master_meta_path}', "
            f"segment folder: '{lowres_segment_folder}'."
        )
    if lowres_raw.size == 0:
        raise ValueError("Low-res raw reference is empty.")
    _logger.info(
        f"{original_basename}: low-res reference ready | "
        f"frames={int(lowres_raw.shape[0])}, size={int(lowres_raw.shape[2])}x{int(lowres_raw.shape[1])}, "
        f"fps={float(lowres_fps):.3f} | took {time.perf_counter() - lowres_load_start:.1f}s"
    )

    fps_for_hires_read = lowres_fps if lowres_fps > 0 else target_fps_setting
    hires_frames: Optional[np.ndarray] = None
    hires_fps = float(fps_for_hires_read) if fps_for_hires_read and fps_for_hires_read > 0 else 23.976
    proc_h = max(8, int(round(float(target_height) / 8.0) * 8))
    proc_w = max(8, int(round(float(target_width) / 8.0) * 8))

    if os.path.exists(source_video_path):
        process_limit = int(lowres_raw.shape[0]) if lowres_raw.shape[0] > 0 else -1
        frame_load_chunk = 8
        _logger.info(
            f"{original_basename}: loading hi-res source frames "
            f"(target={int(target_width)}x{int(target_height)}, "
            f"fps={float(fps_for_hires_read):.3f}, max_frames={process_limit})..."
        )
        hires_load_start = time.perf_counter()
        hires_frames, hires_fps, _, _, proc_h, proc_w, _, _ = read_video_frames(
            source_video_path,
            process_length=process_limit,
            target_fps=fps_for_hires_read,
            target_height=target_height,
            target_width=target_width,
            start_frame_index=0,
            num_frames_to_load=-1,
            round_to_multiple=8,
            chunk_size_for_loading=frame_load_chunk,
            progress_log_prefix=f"{original_basename} | hi-res frame load",
        )
        if hires_frames is None or hires_frames.size == 0:
            raise ValueError("Hi-res source frame load returned no frames.")
        _logger.info(
            f"{original_basename}: hi-res frame load complete | "
            f"frames={int(hires_frames.shape[0])}, size={int(proc_w)}x{int(proc_h)}, "
            f"fps={float(hires_fps):.3f} | took {time.perf_counter() - hires_load_start:.1f}s"
        )

        frame_count = min(int(lowres_raw.shape[0]), int(hires_frames.shape[0]))
        if frame_count <= 0:
            raise ValueError(
                f"No overlapping frames between low-res reference ({lowres_raw.shape[0]}) "
                f"and hi-res source ({hires_frames.shape[0]})."
            )
        if lowres_raw.shape[0] != hires_frames.shape[0]:
            _logger.warning(
                f"Frame count mismatch for {original_basename}. "
                f"Low-res raw={lowres_raw.shape[0]}, hi-res read={hires_frames.shape[0]}. "
                f"Using first {frame_count} frames."
            )
        hires_frames = hires_frames[:frame_count]
        if hires_fps <= 0:
            hires_fps = lowres_fps if lowres_fps > 0 else 23.976
    else:
        _logger.warning(
            f"{original_basename}: source video not found at '{source_video_path}'. "
            f"Attempting cache-only tile merge from {temp_root}."
        )
        frame_count = int(lowres_raw.shape[0])
        if frame_count <= 0:
            raise ValueError(
                f"No frames available in low-res reference cache for cache-only merge: {original_basename}"
            )

    lowres_raw = lowres_raw[:frame_count]
    cache_only_mode = hires_frames is None
    if cache_only_mode:
        _logger.info(f"{original_basename}: running cache-only merge mode (no source-frame inference).")

    # Upsample low-res global reference to full-res once for alignment and final anchor blend.
    reference_full = (
        F.interpolate(
            torch.from_numpy(lowres_raw).unsqueeze(1),
            size=(proc_h, proc_w),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(1)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    reference_fallback_full = reference_full
    edge_guided_fallback_used = False
    edge_guided_fallback_diag: Dict[str, float] = {}
    edge_guided_fallback_mix_applied = 0.0
    edge_fallback_build_start = time.perf_counter()
    if bool(edge_guided_fallback_enabled):
        if hires_frames is None:
            _logger.warning(
                f"{original_basename}: edge-guided fallback was enabled but source frames are unavailable in cache-only mode. "
                "Falling back to bilinear low-res reference."
            )
        elif float(edge_guided_fallback_mix) <= 1e-6:
            _logger.info(
                f"{original_basename}: edge-guided fallback enabled but mix is ~0. Using bilinear low-res reference only."
            )
        else:
            _logger.info(f"{original_basename}: building edge-guided fallback reference...")
            edge_guided_full, edge_guided_fallback_diag = build_edge_guided_depth_reference(
                lowres_raw=lowres_raw,
                source_frames_rgb=hires_frames,
                target_height=int(proc_h),
                target_width=int(proc_w),
                edge_strength=float(edge_guided_strength),
                sigma_color=float(edge_guided_sigma_color),
                sigma_spatial=float(edge_guided_sigma_spatial),
                bilateral_iterations=int(edge_guided_bilateral_iterations),
                temporal_smooth=float(edge_guided_temporal_smooth),
                edge_reinject_strength=float(edge_guided_reinject_strength),
                log_prefix=f"{original_basename} edge-fallback",
                log_frame_interval=8,
            )
            edge_mix = float(np.clip(edge_guided_fallback_mix, 0.0, 1.0))
            edge_guided_fallback_mix_applied = float(edge_mix)
            if edge_mix >= 1.0 - 1e-6:
                reference_fallback_full = edge_guided_full
            else:
                blended_fallback = reference_full.copy()
                blended_fallback *= (1.0 - edge_mix)
                blended_fallback += edge_mix * edge_guided_full
                reference_fallback_full = blended_fallback
                del blended_fallback
            del edge_guided_full
            gc.collect()
            edge_guided_fallback_used = True
            _logger.info(
                f"{original_basename}: edge-guided fallback reference ready in "
                f"{time.perf_counter() - edge_fallback_build_start:.1f}s | mix={edge_guided_fallback_mix_applied:.3f}"
            )

    lowres_shape_for_meta = list(lowres_raw.shape)
    del lowres_raw
    gc.collect()

    model_spatial_multiple = _get_spatial_model_multiple(demo)
    tiles = _compute_spatial_tiles(
        proc_w,
        proc_h,
        int(tile_num_x),
        int(tile_num_y),
        int(tile_overlap_x_px),
        int(tile_overlap_y_px),
        tile_multiple=int(model_spatial_multiple),
    )
    _logger.info(
        f"Hi-Res Spatial Refine: {original_basename} | {len(tiles)} tiles "
        f"({tile_num_x}x{tile_num_y}), {frame_count} frames, target {proc_w}x{proc_h}"
    )
    _logger.info(
        f"Hi-Res local reliability settings: window={int(local_reliability_window_size)}, "
        f"stride={int(local_reliability_window_stride)}, "
        f"conf_low/high={float(local_reliability_score_confidence_low):.3f}/{float(local_reliability_score_confidence_high):.3f}"
    )
    _logger.info(
        f"Hi-Res edge-guided fallback settings: enabled={str(bool(edge_guided_fallback_enabled)).lower()}, "
        f"mix={float(edge_guided_fallback_mix):.3f}, strength={float(edge_guided_strength):.3f}, "
        f"sigma_color/spatial={float(edge_guided_sigma_color):.4f}/{float(edge_guided_sigma_spatial):.3f}, "
        f"iters={int(edge_guided_bilateral_iterations)}, temporal_smooth={float(edge_guided_temporal_smooth):.3f}, "
        f"reinject={float(edge_guided_reinject_strength):.3f}"
    )
    max_tile_w = 0
    max_tile_h = 0
    for tile in tiles:
        tile_w = int(tile["x1"] - tile["x0"])
        tile_h = int(tile["y1"] - tile["y0"])
        if tile_w % model_spatial_multiple != 0 or tile_h % model_spatial_multiple != 0:
            raise ValueError(
                f"Spatial preflight failed: tile size {tile_w}x{tile_h} is not divisible "
                f"by model multiple {model_spatial_multiple}. "
                f"Adjust Tile Grid and overlap X/Y."
            )
        max_tile_w = max(max_tile_w, tile_w)
        max_tile_h = max(max_tile_h, tile_h)

    x_windows = sorted(
        {(int(t["x0"]), int(t["x1"])) for t in tiles if int(t.get("row", -1)) == 0},
        key=lambda w: w[0],
    )
    y_windows = sorted(
        {(int(t["y0"]), int(t["y1"])) for t in tiles if int(t.get("col", -1)) == 0},
        key=lambda w: w[0],
    )
    effective_x_overlaps = [x_windows[i - 1][1] - x_windows[i][0] for i in range(1, len(x_windows))]
    effective_y_overlaps = [y_windows[i - 1][1] - y_windows[i][0] for i in range(1, len(y_windows))]
    x_edge_overlaps = [max(0, int(x_windows[i][1] - x_windows[i + 1][0])) for i in range(max(0, len(x_windows) - 1))]
    y_edge_overlaps = [max(0, int(y_windows[i][1] - y_windows[i + 1][0])) for i in range(max(0, len(y_windows) - 1))]
    _logger.info(
        f"Spatial preflight: model multiple {model_spatial_multiple}, largest tile {max_tile_w}x{max_tile_h}, "
        f"requested overlap X/Y={int(tile_overlap_x_px)}/{int(tile_overlap_y_px)} px, "
        f"effective overlap X/Y~"
        f"{(min(effective_x_overlaps), max(effective_x_overlaps)) if effective_x_overlaps else (0, 0)}/"
        f"{(min(effective_y_overlaps), max(effective_y_overlaps)) if effective_y_overlaps else (0, 0)} px."
    )

    accum = np.zeros((frame_count, proc_h, proc_w), dtype=np.float32)
    # Weights are spatially constant across time; keep one 2D accumulator to reduce RAM.
    weight_sum_2d = np.zeros((proc_h, proc_w), dtype=np.float32)
    tile_debug = []
    generated_tile_folders: List[str] = []
    local_conf_area_weight_sum = 0.0
    local_conf_area_pixels = 0.0
    local_fallback_area_weight_sum = 0.0
    seam_tiles_applied = 0
    seam_overlap_weight_sum = 0.0
    seam_rmse_before_weight_sum = 0.0
    seam_rmse_after_weight_sum = 0.0

    for tile in tiles:
        tile_id = tile["tile_id"]
        x0, y0, x1, y1 = tile["x0"], tile["y0"], tile["x1"], tile["y1"]
        tile_row = int(tile.get("row", -1))
        tile_col = int(tile.get("col", -1))
        tile_basename = f"{original_basename}_tile{tile_id:02d}"
        _logger.info(
            f"Spatial Tile {tile_id + 1}/{len(tiles)} for {original_basename}: "
            f"x={x0}:{x1}, y={y0}:{y1}"
        )

        tile_frames = None
        tile_h = int(y1 - y0)
        tile_w = int(x1 - x0)
        if tile_h % model_spatial_multiple != 0 or tile_w % model_spatial_multiple != 0:
            raise ValueError(
                f"Tile {tile_id} has non-compatible size {tile_w}x{tile_h}; "
                f"expected both dimensions divisible by {model_spatial_multiple}. "
                f"Try adjusting Tile Grid or overlap X/Y."
            )

        if hires_frames is None:
            tile_seg_folder = os.path.join(temp_root, get_segment_output_folder_name(tile_basename))
            tile_raw, tile_fps, cache_reason = _try_load_cached_tile_raw(
                tile_seg_folder=tile_seg_folder,
                tile_basename=tile_basename,
                temporal_overlap_frames=int(temporal_overlap_frames),
                process_fps=float(hires_fps),
                temporal_merge_alignment=temporal_merge_alignment,
                expected_frames=int(frame_count),
                expected_height=int(tile_h),
                expected_width=int(tile_w),
            )
            if tile_raw is None:
                raise FileNotFoundError(
                    f"Cache-only merge failed for tile {tile_id} ({tile_basename}): {cache_reason}. "
                    f"Source clip is unavailable and this tile cache cannot be reused."
                )
            tile_reused_cache = True
        else:
            tile_frames = hires_frames[:, y0:y1, x0:x1, :]
            tile_h = int(tile_frames.shape[1])
            tile_w = int(tile_frames.shape[2])
            tile_raw, tile_fps, tile_seg_folder, tile_reused_cache = _infer_single_tile_raw(
                demo=demo,
                source_video_path=source_video_path,
                tile_frames=tile_frames,
                tile_basename=tile_basename,
                temp_output_root=temp_root,
                process_fps=hires_fps,
                temporal_window_frames=temporal_window_frames,
                temporal_overlap_frames=temporal_overlap_frames,
                guidance_scale=guidance_scale,
                inference_steps=inference_steps,
                seed=seed,
                temporal_merge_alignment=temporal_merge_alignment,
                reuse_existing_segments=True,
            )
        generated_tile_folders.append(tile_seg_folder)

        if tile_raw.shape[0] < frame_count:
            raise ValueError(
                f"Tile {tile_id} produced fewer frames ({tile_raw.shape[0]}) than expected ({frame_count})."
            )
        tile_raw = tile_raw[:frame_count]
        if tile_raw.shape[1] != tile_h or tile_raw.shape[2] != tile_w:
            raise ValueError(
                f"Tile {tile_id} output size mismatch. "
                f"Output={tile_raw.shape[2]}x{tile_raw.shape[1]}, expected {tile_w}x{tile_h}."
            )

        target_tile = reference_full[:, y0:y1, x0:x1].astype(np.float32, copy=False)
        target_tile_fallback = reference_fallback_full[:, y0:y1, x0:x1].astype(np.float32, copy=False)
        left_eff_overlap = x_edge_overlaps[tile_col - 1] if 0 < tile_col <= len(x_edge_overlaps) else 0
        right_eff_overlap = x_edge_overlaps[tile_col] if 0 <= tile_col < len(x_edge_overlaps) else 0
        top_eff_overlap = y_edge_overlaps[tile_row - 1] if 0 < tile_row <= len(y_edge_overlaps) else 0
        bottom_eff_overlap = y_edge_overlaps[tile_row] if 0 <= tile_row < len(y_edge_overlaps) else 0
        weights = _compute_tile_feather_weights(
            x0,
            y0,
            x1,
            y1,
            proc_w,
            proc_h,
            int(tile_overlap_x_px),
            int(tile_overlap_y_px),
            left_overlap_px=int(left_eff_overlap),
            right_overlap_px=int(right_eff_overlap),
            top_overlap_px=int(top_eff_overlap),
            bottom_overlap_px=int(bottom_eff_overlap),
        )

        local_confidence, local_diag = _compute_local_confidence_map(
            pred_video=tile_raw,
            target_video=target_tile,
            window_size=int(local_reliability_window_size),
            window_stride=int(local_reliability_window_stride),
            sample_frame_cap=8,
            score_confidence_low=float(local_reliability_score_confidence_low),
            score_confidence_high=float(local_reliability_score_confidence_high),
            log_prefix=f"{original_basename} tile {int(tile_id)}",
            log_sample_limit=3,
        )
        raw_local_confidence = local_confidence.astype(np.float32, copy=False)
        local_confidence = _stabilize_local_confidence_map(
            raw_local_confidence,
            feather_weights_2d=weights,
            smoothing_kernel_size=7,
            edge_conf_floor=0.35,
        )
        local_diag["confidence_mean_raw"] = float(np.mean(raw_local_confidence))
        local_diag["confidence_min_raw"] = float(np.min(raw_local_confidence))
        local_diag["confidence_max_raw"] = float(np.max(raw_local_confidence))
        local_diag["confidence_mean"] = float(np.mean(local_confidence))
        local_diag["confidence_min"] = float(np.min(local_confidence))
        local_diag["confidence_max"] = float(np.max(local_confidence))
        local_diag["fallback_mean"] = float(np.mean(1.0 - local_confidence))
        local_diag["fallback_gt_50pct_area"] = float(np.mean((1.0 - local_confidence) > 0.5))
        conf_scale = np.divide(
            local_confidence,
            np.maximum(raw_local_confidence, 1e-6),
            out=np.ones_like(local_confidence, dtype=np.float32),
        )
        local_diag["boundary_conf_scale_mean"] = float(np.mean(conf_scale))
        local_diag["boundary_conf_scale_min"] = float(np.min(conf_scale))
        local_diag["boundary_conf_scale_max"] = float(np.max(conf_scale))
        lut_fit_weights = (weights * local_confidence).astype(np.float32, copy=False)

        if float(np.max(local_confidence)) <= 1e-4:
            aligned_tile = tile_raw.astype(np.float32, copy=True)
            _logger.info(
                f"Tile {tile_id}: confidence map indicates full fallback region; skipping LUT solve."
            )
        else:
            aligned_tile = _align_tile_frames_to_reference(
                tile_raw,
                reference_full,
                x0,
                y0,
                x1,
                y1,
                spatial_weights_2d=lut_fit_weights,
                tile_id_for_log=int(tile_id),
            )

        # First-pass reliability gating: blend corrected hi-res depth with fallback anchor map
        # (bilinear low-res reference, optionally edge-guided enhanced).
        conf3 = local_confidence[None, :, :]
        aligned_tile *= conf3
        aligned_tile += (1.0 - conf3) * target_tile_fallback

        aligned_tile, seam_diag = _harmonize_tile_to_existing_overlap(
            tile_video=aligned_tile,
            accum_video=accum,
            accum_weight_2d=weight_sum_2d,
            x0=int(x0),
            y0=int(y0),
            x1=int(x1),
            y1=int(y1),
            sample_frame_cap=6,
            min_overlap_pixels=2048,
        )
        seam_overlap_pixels = float(seam_diag.get("overlap_pixels", 0.0))
        seam_overlap_weight_sum += seam_overlap_pixels
        seam_rmse_before_weight_sum += seam_overlap_pixels * float(seam_diag.get("rmse_before", 0.0))
        seam_rmse_after_weight_sum += seam_overlap_pixels * float(seam_diag.get("rmse_after", 0.0))
        if float(seam_diag.get("applied", 0.0)) > 0.5:
            seam_tiles_applied += 1
            _logger.info(
                f"Tile {tile_id} seam harmonization | overlap_px={int(seam_overlap_pixels)}, "
                f"gain={float(seam_diag.get('gain', 1.0)):.4f}, offset={float(seam_diag.get('offset', 0.0)):.4f}, "
                f"rmse={float(seam_diag.get('rmse_before', 0.0)):.4f}->{float(seam_diag.get('rmse_after', 0.0)):.4f}, "
                f"improve={100.0 * float(seam_diag.get('rmse_improvement_ratio', 0.0)):.2f}%"
            )

        accum[:, y0:y1, x0:x1] += aligned_tile * weights[None, :, :]
        weight_sum_2d[y0:y1, x0:x1] += weights

        _logger.info(
            f"Tile {tile_id} local confidence stats | "
            f"score mean/min/max={local_diag.get('score_mean', 0.0):.3f}/"
            f"{local_diag.get('score_min', 0.0):.3f}/{local_diag.get('score_max', 0.0):.3f}, "
            f"conf mean/min/max={local_diag.get('confidence_mean', 1.0):.3f}/"
            f"{local_diag.get('confidence_min', 1.0):.3f}/{local_diag.get('confidence_max', 1.0):.3f}, "
            f"fallback_mean={local_diag.get('fallback_mean', 0.0):.3f}, "
            f"boundary_scale_mean={local_diag.get('boundary_conf_scale_mean', 1.0):.3f}"
        )
        tile_area = float((y1 - y0) * (x1 - x0))
        local_conf_area_weight_sum += tile_area * float(local_diag.get("confidence_mean", 1.0))
        local_fallback_area_weight_sum += tile_area * float(local_diag.get("fallback_mean", 0.0))
        local_conf_area_pixels += tile_area

        tile_debug.append(
            {
                "tile_id": int(tile_id),
                "x0": int(x0),
                "y0": int(y0),
                "x1": int(x1),
                "y1": int(y1),
                "tile_height": int(y1 - y0),
                "tile_width": int(x1 - x0),
                "tile_frames": int(aligned_tile.shape[0]),
                "tile_fps": float(tile_fps),
                "tile_segment_folder": os.path.abspath(tile_seg_folder),
                "reused_existing_cache": bool(tile_reused_cache),
                "local_confidence": local_diag,
                "effective_overlap_left_px": int(left_eff_overlap),
                "effective_overlap_right_px": int(right_eff_overlap),
                "effective_overlap_top_px": int(top_eff_overlap),
                "effective_overlap_bottom_px": int(bottom_eff_overlap),
                "seam_harmonization": seam_diag,
            }
        )

        if tile_frames is not None:
            del tile_frames
        del tile_raw, aligned_tile, weights, target_tile, target_tile_fallback, raw_local_confidence, local_confidence, lut_fit_weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # No longer needed once per-tile inference and stitching are complete.
    del hires_frames
    gc.collect()

    stitched_hires_raw = accum / np.maximum(weight_sum_2d[None, :, :], 1e-6)
    del accum, weight_sum_2d

    anchor = float(np.clip(lowres_anchor_weight, 0.0, 1.0))
    if anchor <= 0.0:
        final_hires_raw = stitched_hires_raw
    elif anchor >= 1.0:
        del stitched_hires_raw
        final_hires_raw = reference_full
    else:
        # Blend in-place to avoid one additional full-size [T,H,W] allocation.
        stitched_hires_raw *= (1.0 - anchor)
        stitched_hires_raw += anchor * reference_full
        final_hires_raw = stitched_hires_raw
    gc.collect()

    _logger.info(
        f"{original_basename}: applying final low-res master brightness alignment..."
    )
    final_master_lut_diag: Dict[str, float] = {"mode": "not_run"}
    try:
        final_lut_x, final_lut_y, final_master_lut_diag = _build_weighted_intensity_lut(
            final_hires_raw.astype(np.float32, copy=False),
            reference_full.astype(np.float32, copy=False),
            spatial_weights_2d=None,
            num_bins=4096,
        )
        final_hires_master_raw = _apply_lut_video(final_hires_raw, final_lut_x, final_lut_y)
        _logger.info(
            f"{original_basename}: final master LUT mode={final_master_lut_diag.get('mode')} | "
            f"pred_range={final_master_lut_diag.get('pred_low', 0.0):.4f}..{final_master_lut_diag.get('pred_high', 0.0):.4f} -> "
            f"target_range={final_master_lut_diag.get('tgt_low', 0.0):.4f}..{final_master_lut_diag.get('tgt_high', 0.0):.4f}"
        )
    except Exception as final_lut_err:
        _logger.warning(
            f"{original_basename}: final master LUT alignment failed ({final_lut_err}). "
            "Continuing without final LUT remap."
        )
        final_hires_master_raw = final_hires_raw.astype(np.float32, copy=False)
        final_master_lut_diag = {"mode": "bypass_error"}

    ref_min = float(np.nanmin(reference_full))
    ref_max = float(np.nanmax(reference_full))
    ref_stats = _estimate_sampled_video_stats(
        reference_full,
        sample_frame_cap=12,
        sample_spatial_stride=4,
        core_low_pct=5.0,
        core_high_pct=95.0,
        robust_low_pct=0.10,
        robust_high_pct=99.90,
    )
    pred_stats_pre = _estimate_sampled_video_stats(
        final_hires_master_raw,
        sample_frame_cap=12,
        sample_spatial_stride=4,
        core_low_pct=5.0,
        core_high_pct=95.0,
        robust_low_pct=0.10,
        robust_high_pct=99.90,
    )

    final_master_global_gain = 1.0
    final_master_global_offset = 0.0
    final_master_global_affine_applied = False
    ref_core_std = float(ref_stats.get("core_std", 0.0))
    pred_core_std = float(pred_stats_pre.get("core_std", 0.0))
    ref_core_mean = float(ref_stats.get("core_mean", 0.0))
    pred_core_mean = float(pred_stats_pre.get("core_mean", 0.0))
    if ref_core_std > 1e-8 and pred_core_std > 1e-8:
        gain_guess = ref_core_std / pred_core_std
        final_master_global_gain = float(np.clip(gain_guess, 0.85, 1.15))
        final_master_global_offset = float(ref_core_mean - final_master_global_gain * pred_core_mean)
        if np.isfinite(final_master_global_gain) and np.isfinite(final_master_global_offset):
            final_hires_master_raw *= final_master_global_gain
            final_hires_master_raw += final_master_global_offset
            final_master_global_affine_applied = True
    _logger.info(
        f"{original_basename}: final master affine match | "
        f"gain={final_master_global_gain:.4f}, offset={final_master_global_offset:.4f}, "
        f"ref_core_mean/std={ref_core_mean:.4f}/{ref_core_std:.4f}, "
        f"pred_core_mean/std={pred_core_mean:.4f}/{pred_core_std:.4f}, "
        f"applied={str(bool(final_master_global_affine_applied)).lower()}"
    )

    pred_stats_post = _estimate_sampled_video_stats(
        final_hires_master_raw,
        sample_frame_cap=12,
        sample_spatial_stride=4,
        core_low_pct=5.0,
        core_high_pct=95.0,
        robust_low_pct=0.10,
        robust_high_pct=99.90,
    )
    ref_robust_low = float(ref_stats.get("robust_low", ref_min))
    ref_robust_high = float(ref_stats.get("robust_high", ref_max))
    pred_robust_low = float(pred_stats_post.get("robust_low", ref_robust_low))
    pred_robust_high = float(pred_stats_post.get("robust_high", ref_robust_high))
    robust_span = max(ref_robust_high - ref_robust_low, 1e-6)
    robust_margin = 0.01 * robust_span

    # Keep global brightness anchored to low-res master and suppress extreme tail artifacts.
    outlier_low_thr = ref_robust_low - 0.05 * robust_span
    outlier_high_thr = ref_robust_high + 0.05 * robust_span
    outlier_mix_with_reference = 0.85
    outlier_low_count = 0
    outlier_high_count = 0
    total_values = 0
    for frame_idx in range(int(final_hires_master_raw.shape[0])):
        pred_frame = final_hires_master_raw[frame_idx]
        ref_frame = reference_full[frame_idx]
        low_mask = pred_frame < outlier_low_thr
        high_mask = pred_frame > outlier_high_thr
        low_count = int(np.count_nonzero(low_mask))
        high_count = int(np.count_nonzero(high_mask))
        outlier_low_count += low_count
        outlier_high_count += high_count
        total_values += int(pred_frame.size)
        if low_count > 0:
            pred_frame[low_mask] = (
                outlier_mix_with_reference * ref_frame[low_mask]
                + (1.0 - outlier_mix_with_reference) * pred_frame[low_mask]
            )
        if high_count > 0:
            pred_frame[high_mask] = (
                outlier_mix_with_reference * ref_frame[high_mask]
                + (1.0 - outlier_mix_with_reference) * pred_frame[high_mask]
            )

    outlier_low_fraction = float(outlier_low_count / max(1, total_values))
    outlier_high_fraction = float(outlier_high_count / max(1, total_values))
    if outlier_low_fraction > 0.0 or outlier_high_fraction > 0.0:
        _logger.info(
            f"{original_basename}: final master outlier repair | "
            f"low={outlier_low_fraction:.4%}, high={outlier_high_fraction:.4%}, "
            f"ref_mix={outlier_mix_with_reference:.2f}"
        )

    norm_low = ref_robust_low - robust_margin
    norm_high = ref_robust_high + robust_margin
    norm_range = float(norm_high - norm_low)

    if (not np.isfinite(norm_low)) or (not np.isfinite(norm_high)) or norm_range <= 1e-8:
        _logger.warning(
            f"{original_basename}: invalid low-res reference range for final normalization "
            f"(norm_low={norm_low}, norm_high={norm_high}). Using fallback constant output."
        )
        final_hires_norm = np.full_like(final_hires_master_raw, 0.5, dtype=np.float32)
        clip_low_fraction = 0.0
        clip_high_fraction = 0.0
    else:
        final_hires_norm = final_hires_master_raw.astype(np.float32, copy=False)
        final_hires_norm -= norm_low
        final_hires_norm /= norm_range
        np.clip(final_hires_norm, 0.0, 1.0, out=final_hires_norm)
        clip_low_fraction = float(np.mean(final_hires_norm <= 1e-6))
        clip_high_fraction = float(np.mean(final_hires_norm >= 1.0 - 1e-6))

    _logger.info(
        f"{original_basename}: final master normalization range={norm_low:.4f}..{norm_high:.4f} "
        f"(ref_robust={ref_robust_low:.4f}..{ref_robust_high:.4f}, "
        f"pred_robust={pred_robust_low:.4f}..{pred_robust_high:.4f}) | "
        f"clip_low={clip_low_fraction:.4%}, clip_high={clip_high_fraction:.4%}"
    )

    del final_hires_raw, final_hires_master_raw, reference_full, reference_fallback_full
    gc.collect()

    output_path = os.path.join(output_dir, f"{original_basename}{output_suffix}.mp4")
    save_video(final_hires_norm, output_path, fps=hires_fps, output_format=output_format)

    reused_cache_tiles = int(sum(1 for item in tile_debug if item.get("reused_existing_cache")))
    weighted_conf_mean = (
        float(local_conf_area_weight_sum / local_conf_area_pixels) if local_conf_area_pixels > 0 else 1.0
    )
    weighted_fallback_mean = (
        float(local_fallback_area_weight_sum / local_conf_area_pixels) if local_conf_area_pixels > 0 else 0.0
    )
    seam_rmse_before_mean = (
        float(seam_rmse_before_weight_sum / seam_overlap_weight_sum) if seam_overlap_weight_sum > 0 else 0.0
    )
    seam_rmse_after_mean = (
        float(seam_rmse_after_weight_sum / seam_overlap_weight_sum) if seam_overlap_weight_sum > 0 else 0.0
    )
    _logger.info(
        f"{original_basename}: tile cache reuse {reused_cache_tiles}/{len(tile_debug)} tiles."
    )
    _logger.info(
        f"{original_basename}: local reliability blend summary | "
        f"confidence_mean={weighted_conf_mean:.3f}, fallback_mean={weighted_fallback_mean:.3f}"
    )
    _logger.info(
        f"{original_basename}: seam harmonization summary | "
        f"tiles_adjusted={int(seam_tiles_applied)}/{len(tile_debug)}, "
        f"rmse={seam_rmse_before_mean:.4f}->{seam_rmse_after_mean:.4f}"
    )
    if bool(edge_guided_fallback_enabled):
        _logger.info(
            f"{original_basename}: edge-guided fallback summary | "
            f"used={str(bool(edge_guided_fallback_used)).lower()}, mix={edge_guided_fallback_mix_applied:.3f}, "
            f"edge_mean={float(edge_guided_fallback_diag.get('edge_weight_mean', 0.0)):.4f}, "
            f"edge_mix_mean={float(edge_guided_fallback_diag.get('edge_mix_mean', 0.0)):.4f}"
        )

    metadata = {
        "mode": "spatial_hires_refine_from_lowres_raw",
        "source_video_path": os.path.abspath(source_video_path),
        "lowres_master_meta_path": os.path.abspath(lowres_master_meta_path) if lowres_master_meta_path else None,
        "lowres_segment_folder": os.path.abspath(lowres_segment_folder) if lowres_segment_folder else None,
        "used_legacy_npz_fallback": bool(lowres_meta.get("source") == "legacy_npz_fallback") if isinstance(lowres_meta, dict) else False,
        "cache_only_mode_used": bool(cache_only_mode),
        "output_path": os.path.abspath(output_path),
        "output_format": output_format,
        "output_suffix": output_suffix,
        "target_height_requested": int(target_height),
        "target_width_requested": int(target_width),
        "processed_height": int(proc_h),
        "processed_width": int(proc_w),
        "frames_processed": int(frame_count),
        "fps_processed": float(hires_fps),
        "tile_num": int(tile_num_x),  # Backward-compatible key (legacy square-grid field).
        "tile_num_x": int(tile_num_x),
        "tile_num_y": int(tile_num_y),
        "tile_overlap_px": int(tile_overlap_x_px),  # Backward-compatible key (legacy single-overlap field).
        "tile_overlap_x_px": int(tile_overlap_x_px),
        "tile_overlap_y_px": int(tile_overlap_y_px),
        "model_spatial_multiple": int(model_spatial_multiple),
        "temporal_window_frames": int(temporal_window_frames),
        "temporal_overlap_frames": int(temporal_overlap_frames),
        "temporal_merge_alignment": temporal_merge_alignment,
        "anchor_weight": float(anchor),
        "guidance_scale": float(guidance_scale),
        "inference_steps": int(inference_steps),
        "seed": int(seed),
        "cleanup_temp_on_success": bool(cleanup_temp),
        "lowres_reference_shape": lowres_shape_for_meta,
        "tile_debug": tile_debug,
        "tile_cache_reused_count": int(reused_cache_tiles),
        "local_reliability_window_size": int(local_reliability_window_size),
        "local_reliability_window_stride": int(local_reliability_window_stride),
        "local_reliability_score_confidence_low": float(local_reliability_score_confidence_low),
        "local_reliability_score_confidence_high": float(local_reliability_score_confidence_high),
        "local_reliability_confidence_mean_weighted": float(weighted_conf_mean),
        "local_reliability_fallback_mean_weighted": float(weighted_fallback_mean),
        "edge_guided_fallback_enabled": bool(edge_guided_fallback_enabled),
        "edge_guided_fallback_used": bool(edge_guided_fallback_used),
        "edge_guided_fallback_mix": float(edge_guided_fallback_mix_applied),
        "edge_guided_strength": float(edge_guided_strength),
        "edge_guided_sigma_color": float(edge_guided_sigma_color),
        "edge_guided_sigma_spatial": float(edge_guided_sigma_spatial),
        "edge_guided_bilateral_iterations": int(edge_guided_bilateral_iterations),
        "edge_guided_temporal_smooth": float(edge_guided_temporal_smooth),
        "edge_guided_reinject_strength": float(edge_guided_reinject_strength),
        "edge_guided_fallback_diag": edge_guided_fallback_diag,
        "seam_harmonization_tiles_adjusted": int(seam_tiles_applied),
        "seam_harmonization_rmse_before_mean": float(seam_rmse_before_mean),
        "seam_harmonization_rmse_after_mean": float(seam_rmse_after_mean),
        "final_master_lut_mode": final_master_lut_diag.get("mode"),
        "final_master_lut_pred_low": float(final_master_lut_diag.get("pred_low")) if final_master_lut_diag.get("pred_low") is not None else None,
        "final_master_lut_pred_high": float(final_master_lut_diag.get("pred_high")) if final_master_lut_diag.get("pred_high") is not None else None,
        "final_master_lut_tgt_low": float(final_master_lut_diag.get("tgt_low")) if final_master_lut_diag.get("tgt_low") is not None else None,
        "final_master_lut_tgt_high": float(final_master_lut_diag.get("tgt_high")) if final_master_lut_diag.get("tgt_high") is not None else None,
        "final_master_reference_min_raw": float(ref_min),
        "final_master_reference_max_raw": float(ref_max),
        "final_master_reference_robust_low_raw": float(ref_robust_low),
        "final_master_reference_robust_high_raw": float(ref_robust_high),
        "final_master_pred_robust_low_raw": float(pred_robust_low),
        "final_master_pred_robust_high_raw": float(pred_robust_high),
        "final_master_global_affine_applied": bool(final_master_global_affine_applied),
        "final_master_global_gain": float(final_master_global_gain),
        "final_master_global_offset": float(final_master_global_offset),
        "final_master_outlier_low_fraction": float(outlier_low_fraction),
        "final_master_outlier_high_fraction": float(outlier_high_fraction),
        "final_master_norm_low_raw": float(norm_low),
        "final_master_norm_high_raw": float(norm_high),
        "final_master_clip_low_fraction": float(clip_low_fraction),
        "final_master_clip_high_fraction": float(clip_high_fraction),
        "lowres_reference_fps": float(lowres_fps),
        "lowres_meta_snapshot": lowres_meta.get("global_processing_settings", {}) if isinstance(lowres_meta, dict) else {},
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metadata_path = os.path.splitext(output_path)[0] + ".json"
    save_json_file(metadata, metadata_path, indent=4)

    if cleanup_temp:
        try:
            shutil.rmtree(temp_root)
            _logger.info(f"Removed temporary hi-res tile folder: {temp_root}")
        except Exception as cleanup_err:
            _logger.warning(f"Failed to remove temporary folder '{temp_root}': {cleanup_err}")

    duration = time.perf_counter() - start_time
    _logger.info(
        f"Hi-Res Spatial Refine complete for {original_basename} in {duration:.1f}s. "
        f"Output: {output_path}"
    )

    summary = {
        "output_path": os.path.abspath(output_path),
        "metadata_path": os.path.abspath(metadata_path),
        "frames": int(frame_count),
        "processed_height": int(proc_h),
        "processed_width": int(proc_w),
        "fps": float(hires_fps),
        "duration_seconds": float(round(duration, 3)),
        "temp_folder": os.path.abspath(temp_root),
        "cleanup_temp_on_success": bool(cleanup_temp),
        "cache_only_mode_used": bool(cache_only_mode),
        "generated_tile_segment_folders": [os.path.abspath(p) for p in generated_tile_folders],
        "tile_cache_reused_count": int(reused_cache_tiles),
        "local_reliability_confidence_mean_weighted": float(weighted_conf_mean),
        "local_reliability_fallback_mean_weighted": float(weighted_fallback_mean),
        "edge_guided_fallback_enabled": bool(edge_guided_fallback_enabled),
        "edge_guided_fallback_used": bool(edge_guided_fallback_used),
        "edge_guided_fallback_mix": float(edge_guided_fallback_mix_applied),
        "edge_guided_reinject_strength": float(edge_guided_reinject_strength),
        "seam_harmonization_tiles_adjusted": int(seam_tiles_applied),
        "seam_harmonization_rmse_before_mean": float(seam_rmse_before_mean),
        "seam_harmonization_rmse_after_mean": float(seam_rmse_after_mean),
        "final_master_global_affine_applied": bool(final_master_global_affine_applied),
        "final_master_global_gain": float(final_master_global_gain),
        "final_master_global_offset": float(final_master_global_offset),
        "final_master_outlier_low_fraction": float(outlier_low_fraction),
        "final_master_outlier_high_fraction": float(outlier_high_fraction),
        "final_master_norm_low_raw": float(norm_low),
        "final_master_norm_high_raw": float(norm_high),
        "final_master_clip_low_fraction": float(clip_low_fraction),
        "final_master_clip_high_fraction": float(clip_high_fraction),
    }
    return output_path, summary
