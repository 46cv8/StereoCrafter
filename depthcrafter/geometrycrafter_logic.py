import gc
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from depthcrafter.depthcrafter_logic import DepthCrafterDemo


_logger = logging.getLogger(__name__)

_SUPPORTED_BACKENDS = ("geometrycrafter_diff", "geometrycrafter_determ")
_DEFAULT_GEOMETRY_MODEL_PATH = "TencentARC/GeometryCrafter"
_DEFAULT_GEOMETRY_REPO_REL = os.path.join("weights", "GeometryCrafter")
_DEFAULT_PRETRAIN_PATH = "stabilityai/stable-video-diffusion-img2vid-xt"


def _coerce_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be > 0")
    rounded = int(round(float(value) / float(multiple)) * multiple)
    return max(multiple, rounded)


def _normalize_backend(name: str) -> str:
    normalized = str(name or "").strip().lower()
    if normalized in ("geometrycrafter", "geometry", "geom"):
        normalized = "geometrycrafter_diff"
    if normalized not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported GeometryCrafter backend '{name}'. "
            f"Supported backends: {', '.join(_SUPPORTED_BACKENDS)}"
        )
    return normalized


class GeometryCrafterDemo(DepthCrafterDemo):
    """DepthCrafter-compatible wrapper that runs GeometryCrafter inference."""

    def __init__(
        self,
        *,
        model_backend: str = "geometrycrafter_diff",
        geometry_model_path: str = _DEFAULT_GEOMETRY_MODEL_PATH,
        geometry_repo_path: str = "",
        geometry_cache_dir: str = "",
        geometry_decode_chunk_size: int = 8,
        geometry_low_memory_usage: bool = False,
        geometry_force_projection: bool = True,
        geometry_force_fixed_focal: bool = True,
        geometry_use_extract_interp: bool = False,
        pre_train_path: str = _DEFAULT_PRETRAIN_PATH,
        cpu_offload: Union[str, None] = "model",
        use_cudnn_benchmark: bool = False,
        local_files_only: bool = False,
        disable_xformers: bool = False,
        **_: object,
    ):
        self.runtime_stage_callback = None
        torch.backends.cudnn.benchmark = use_cudnn_benchmark

        self.model_backend = _normalize_backend(model_backend)
        self.geometry_model_path = str(geometry_model_path or _DEFAULT_GEOMETRY_MODEL_PATH).strip()
        if not self.geometry_model_path:
            self.geometry_model_path = _DEFAULT_GEOMETRY_MODEL_PATH
        self.pre_train_path = str(pre_train_path or _DEFAULT_PRETRAIN_PATH).strip()
        if not self.pre_train_path:
            self.pre_train_path = _DEFAULT_PRETRAIN_PATH

        self.geometry_repo_path = self._resolve_geometry_repo_path(geometry_repo_path)
        self.geometry_cache_dir = self._resolve_geometry_cache_dir(geometry_cache_dir)
        self.geometry_decode_chunk_size = max(1, int(geometry_decode_chunk_size))
        self.geometry_low_memory_usage = bool(geometry_low_memory_usage)
        self.geometry_force_projection = bool(geometry_force_projection)
        self.geometry_force_fixed_focal = bool(geometry_force_fixed_focal)
        self.geometry_use_extract_interp = bool(geometry_use_extract_interp)

        self.cpu_offload = str(cpu_offload or "none").strip().lower()
        self.local_files_only = bool(local_files_only)
        self.disable_xformers = bool(disable_xformers)

        self.pipe = None
        self.point_map_vae = None
        self.prior_model = None

        self._init_geometry_pipeline()

    def _resolve_geometry_repo_path(self, configured_path: str) -> str:
        repo_root = Path(__file__).resolve().parents[1]
        candidates = []

        if configured_path:
            candidates.append(Path(configured_path).expanduser())

        env_repo = os.environ.get("GEOMETRYCRAFTER_REPO", "").strip()
        if env_repo:
            candidates.append(Path(env_repo).expanduser())

        candidates.append(repo_root / _DEFAULT_GEOMETRY_REPO_REL)

        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except Exception:
                continue
            if not resolved.exists():
                continue
            if (resolved / "geometrycrafter").is_dir() and (resolved / "third_party").is_dir():
                return str(resolved)

        raise FileNotFoundError(
            "GeometryCrafter repository not found. "
            "Set 'Geometry Repo Path' in the GUI (or GEOMETRYCRAFTER_REPO env var) to a clone "
            "that contains 'geometrycrafter/' and 'third_party/'."
        )

    def _resolve_geometry_cache_dir(self, configured_cache: str) -> str:
        cache_dir = str(configured_cache or "").strip()
        if cache_dir:
            return str(Path(cache_dir).expanduser().resolve())
        env_cache = os.environ.get("GEOMETRYCRAFTER_CACHE_DIR", "").strip()
        if env_cache:
            return str(Path(env_cache).expanduser().resolve())
        return ""

    def _import_geometry_modules(self):
        repo_root = Path(self.geometry_repo_path).resolve()
        repo_path = str(repo_root)
        moge_root = str((repo_root / "third_party" / "moge").resolve())

        # GeometryCrafter ships MoGe/utils3d as a nested submodule tree.
        # We need both the repo root (for `geometrycrafter`, `third_party`) and
        # the MoGe root (for top-level `utils3d` imports used by MoGe).
        for import_root in (repo_path, moge_root):
            if import_root not in sys.path:
                sys.path.insert(0, import_root)

        try:
            from geometrycrafter import (  # pylint: disable=import-error
                GeometryCrafterDetermPipeline,
                GeometryCrafterDiffPipeline,
                PMapAutoencoderKLTemporalDecoder,
                UNetSpatioTemporalConditionModelVid2vid,
            )
            from third_party import MoGe  # pylint: disable=import-error
        except Exception as exc:
            missing_mod = ""
            try:
                if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", ""):
                    missing_mod = str(exc.name)
            except Exception:
                missing_mod = ""
            extra_hint = (
                f" Missing module: '{missing_mod}'. "
                if missing_mod
                else " "
            )
            raise RuntimeError(
                "Failed to import GeometryCrafter modules. "
                "Make sure the GeometryCrafter repo includes initialized submodules "
                "(especially third_party/moge) and required dependencies."
                + extra_hint
                + "Try running: "
                "PYTHON_BIN=<your_python> bash scripts/setup_geometrycrafter_local.sh"
            ) from exc

        return {
            "diff_pipeline": GeometryCrafterDiffPipeline,
            "determ_pipeline": GeometryCrafterDetermPipeline,
            "point_map_vae": PMapAutoencoderKLTemporalDecoder,
            "unet": UNetSpatioTemporalConditionModelVid2vid,
            "moge": MoGe,
        }

    def _init_geometry_pipeline(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("GeometryCrafter requires CUDA, but no CUDA device is available.")

        imports = self._import_geometry_modules()
        unet_subfolder = "unet_diff" if self.model_backend == "geometrycrafter_diff" else "unet_determ"

        common_kwargs = {
            "low_cpu_mem_usage": True,
            "local_files_only": self.local_files_only,
        }
        if self.geometry_cache_dir:
            common_kwargs["cache_dir"] = self.geometry_cache_dir

        _logger.info(
            "Initializing GeometryCrafter backend '%s' (repo=%s, cache=%s)...",
            self.model_backend,
            self.geometry_repo_path,
            self.geometry_cache_dir or "<default>",
        )

        unet = imports["unet"].from_pretrained(
            self.geometry_model_path,
            subfolder=unet_subfolder,
            torch_dtype=torch.float16,
            **common_kwargs,
        ).requires_grad_(False)

        point_map_vae = imports["point_map_vae"].from_pretrained(
            self.geometry_model_path,
            subfolder="point_map_vae",
            torch_dtype=torch.float32,
            **common_kwargs,
        ).requires_grad_(False)

        moge_kwargs = {}
        if self.geometry_cache_dir:
            moge_kwargs["cache_dir"] = self.geometry_cache_dir
        prior_model = imports["moge"](**moge_kwargs).requires_grad_(False)

        pipe_cls = (
            imports["diff_pipeline"]
            if self.model_backend == "geometrycrafter_diff"
            else imports["determ_pipeline"]
        )
        pipe = pipe_cls.from_pretrained(
            self.pre_train_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            **common_kwargs,
        )

        unet.to("cuda", dtype=torch.float16)
        point_map_vae.to("cuda", dtype=torch.float32)
        prior_model.to("cuda", dtype=torch.float32)

        if self.cpu_offload == "sequential":
            pipe.enable_sequential_cpu_offload()
            _logger.info("GeometryCrafter: CPU offload set to 'sequential' (pipeline).")
        elif self.cpu_offload == "model":
            pipe.enable_model_cpu_offload()
            _logger.info("GeometryCrafter: CPU offload set to 'model' (pipeline).")
        else:
            pipe.to("cuda")
            _logger.info("GeometryCrafter: CPU offload set to 'none' (pipeline on CUDA).")

        if self.disable_xformers:
            try:
                pipe.disable_xformers_memory_efficient_attention()
            except Exception:
                pass
            _logger.info("GeometryCrafter: xFormers disabled by GUI setting.")
        else:
            try:
                pipe.enable_xformers_memory_efficient_attention()
                _logger.info("GeometryCrafter: xFormers enabled.")
            except Exception as exc:
                _logger.warning("GeometryCrafter: could not enable xFormers (%s).", exc)

        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

        self.pipe = pipe
        self.point_map_vae = point_map_vae
        self.prior_model = prior_model
        _logger.info("GeometryCrafter backend initialized successfully.")

    def run(
        self,
        video_path_or_frames_or_info: Union[str, np.ndarray, dict],
        num_denoising_steps: int,
        guidance_scale: float,
        base_output_folder: str,
        gui_window_size: int,
        gui_overlap: int,
        process_length_for_read_full_video: int,
        target_height: int,
        target_width: int,
        seed: int,
        original_video_basename_override: Optional[str] = None,
        segment_job_info_param: Optional[dict] = None,
        keep_intermediate_npz_config: bool = False,
        intermediate_segment_visual_format_config: str = "none",
        save_final_json_for_this_job_config: bool = False,
        full_video_output_format: str = "mp4",
    ):
        adjusted_h = _coerce_multiple(int(target_height), 64)
        adjusted_w = _coerce_multiple(int(target_width), 64)
        if adjusted_h != int(target_height) or adjusted_w != int(target_width):
            _logger.warning(
                "GeometryCrafter requires /64 resolution. Adjusted target from %sx%s to %sx%s.",
                target_width,
                target_height,
                adjusted_w,
                adjusted_h,
            )

        return super().run(
            video_path_or_frames_or_info=video_path_or_frames_or_info,
            num_denoising_steps=num_denoising_steps,
            guidance_scale=guidance_scale,
            base_output_folder=base_output_folder,
            gui_window_size=gui_window_size,
            gui_overlap=gui_overlap,
            process_length_for_read_full_video=process_length_for_read_full_video,
            target_height=adjusted_h,
            target_width=adjusted_w,
            seed=seed,
            original_video_basename_override=original_video_basename_override,
            segment_job_info_param=segment_job_info_param,
            keep_intermediate_npz_config=keep_intermediate_npz_config,
            intermediate_segment_visual_format_config=intermediate_segment_visual_format_config,
            save_final_json_for_this_job_config=save_final_json_for_this_job_config,
            full_video_output_format=full_video_output_format,
        )

    def _perform_inference(
        self,
        actual_frames_to_process: np.ndarray,
        guidance_scale: float,
        num_denoising_steps: int,
        pipe_call_window_size: int,
        pipe_call_overlap: int,
        segment_job_info: Optional[dict],
        actual_processed_height: int,
        actual_processed_width: int,
    ) -> np.ndarray:
        current_pipe_window_for_call = int(pipe_call_window_size)
        current_pipe_overlap_for_call = int(pipe_call_overlap)
        if segment_job_info:
            current_pipe_window_for_call = int(actual_frames_to_process.shape[0])
            current_pipe_overlap_for_call = 0

        self._emit_runtime_stage(
            "logic_inference_start",
            frames=int(actual_frames_to_process.shape[0]),
            height=int(actual_processed_height),
            width=int(actual_processed_width),
            guidance_scale=float(guidance_scale),
            steps=int(num_denoising_steps),
            window_size=int(current_pipe_window_for_call),
            overlap=int(current_pipe_overlap_for_call),
            backend=self.model_backend,
        )

        _logger.debug(
            "GeometryCrafter inference start: frames=%s, target=%sx%s, steps=%s, window=%s, overlap=%s, backend=%s",
            int(actual_frames_to_process.shape[0]),
            int(actual_processed_width),
            int(actual_processed_height),
            int(num_denoising_steps),
            int(current_pipe_window_for_call),
            int(current_pipe_overlap_for_call),
            self.model_backend,
        )

        with torch.inference_mode():
            self._emit_runtime_stage(
                "pipe_call_start",
                backend=self.model_backend,
                decode_chunk_size=int(self.geometry_decode_chunk_size),
                low_memory_usage=bool(self.geometry_low_memory_usage),
            )

            rec_point_map, rec_valid_mask = self.pipe(
                actual_frames_to_process,
                self.point_map_vae,
                self.prior_model,
                height=int(actual_processed_height),
                width=int(actual_processed_width),
                num_inference_steps=int(num_denoising_steps),
                guidance_scale=float(guidance_scale),
                window_size=int(current_pipe_window_for_call),
                decode_chunk_size=int(self.geometry_decode_chunk_size),
                overlap=int(current_pipe_overlap_for_call),
                force_projection=bool(self.geometry_force_projection),
                force_fixed_focal=bool(self.geometry_force_fixed_focal),
                use_extract_interp=bool(self.geometry_use_extract_interp),
                low_memory_usage=bool(self.geometry_low_memory_usage),
            )

            self._emit_runtime_stage(
                "pipe_call_end",
                backend=self.model_backend,
            )

        if not torch.is_tensor(rec_point_map):
            rec_point_map = torch.as_tensor(rec_point_map)
        if not torch.is_tensor(rec_valid_mask):
            rec_valid_mask = torch.as_tensor(rec_valid_mask)

        rec_point_map = rec_point_map.float()
        rec_valid_mask = rec_valid_mask > 0

        depth = rec_point_map[..., 2]
        disparity = torch.zeros_like(depth, dtype=torch.float32)
        valid_pixels = rec_valid_mask & torch.isfinite(depth) & (depth > 1e-6)
        if bool(valid_pixels.any()):
            disparity[valid_pixels] = torch.reciprocal(torch.clamp(depth[valid_pixels], min=1e-6))
        disparity = torch.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)

        res = disparity.detach().cpu().numpy().astype(np.float32)
        if res.ndim != 3:
            raise RuntimeError(f"GeometryCrafter produced unexpected disparity shape: {res.shape}")

        self._emit_runtime_stage(
            "logic_inference_end",
            result_frames=int(res.shape[0]),
            result_height=int(res.shape[1]),
            result_width=int(res.shape[2]),
            backend=self.model_backend,
        )
        _logger.debug("GeometryCrafter inference completed. Result shape: %s", res.shape)

        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return res
