import gc
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import imageio
import numpy as np
import torch
from decord import VideoReader, cpu

from depthcrafter.utils import get_sidecar_json_filename, read_video_frames, save_json_file


_logger = logging.getLogger(__name__)

_SUPPORTED_BACKEND = ("stereopilot",)
_DEFAULT_STEREOPILOT_REPO_REL = os.path.join("weights", "StereoPilot")
_DEFAULT_STEREOPILOT_MODEL_ID = "KlingTeam/StereoPilot"
_DEFAULT_STEREOPILOT_BASE_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
_DEFAULT_PROMPT = "A realistic monocular video scene with natural depth and motion."


class StereoPilotDemo:
    """DepthCrafter-compatible wrapper that runs StereoPilot stereo inference."""

    def __init__(
        self,
        *,
        model_backend: str = "stereopilot",
        stereopilot_model_path: str = _DEFAULT_STEREOPILOT_MODEL_ID,
        stereopilot_base_model_path: str = _DEFAULT_STEREOPILOT_BASE_MODEL_ID,
        stereopilot_repo_path: str = "",
        stereopilot_cache_dir: str = "",
        stereopilot_prompt_default: str = _DEFAULT_PROMPT,
        stereopilot_use_sidecar_prompt: bool = True,
        stereopilot_output_mode: str = "side_by_side",
        stereopilot_target_width: int = 832,
        stereopilot_target_height: int = 480,
        stereopilot_target_fps: float = 16.0,
        stereopilot_sampling_steps: int = 30,
        stereopilot_guide_scale: float = 5.0,
        stereopilot_shift: float = 5.0,
        stereopilot_domain_label: int = 1,
        stereopilot_tail_pad_frames: int = 5,
        stereopilot_dtype: str = "bfloat16",
        stereopilot_transformer_dtype: str = "float8",
        **_: object,
    ):
        self.runtime_stage_callback = None

        backend = str(model_backend or "stereopilot").strip().lower()
        if backend not in _SUPPORTED_BACKEND:
            raise ValueError(
                f"Unsupported StereoPilot backend '{model_backend}'. "
                f"Supported backends: {', '.join(_SUPPORTED_BACKEND)}"
            )

        self.model_backend = backend
        self.stereopilot_model_path = str(stereopilot_model_path or _DEFAULT_STEREOPILOT_MODEL_ID).strip()
        self.stereopilot_base_model_path = str(stereopilot_base_model_path or _DEFAULT_STEREOPILOT_BASE_MODEL_ID).strip()
        self.stereopilot_repo_path = self._resolve_stereopilot_repo_path(stereopilot_repo_path)
        self.stereopilot_cache_dir = self._resolve_cache_dir(stereopilot_cache_dir)

        self.stereopilot_prompt_default = str(stereopilot_prompt_default or _DEFAULT_PROMPT).strip() or _DEFAULT_PROMPT
        self.stereopilot_use_sidecar_prompt = bool(stereopilot_use_sidecar_prompt)

        mode = str(stereopilot_output_mode or "side_by_side").strip().lower()
        if mode not in {"opposite_eye", "side_by_side", "both"}:
            mode = "side_by_side"
        self.stereopilot_output_mode = mode

        self.stereopilot_target_width = self._coerce_multiple(
            int(stereopilot_target_width),
            label="StereoPilot target width",
            multiple=8,
            minimum=32,
        )
        self.stereopilot_target_height = self._coerce_multiple(
            int(stereopilot_target_height),
            label="StereoPilot target height",
            multiple=8,
            minimum=32,
        )
        self.stereopilot_target_fps = max(1.0, float(stereopilot_target_fps))
        self.stereopilot_sampling_steps = max(1, int(stereopilot_sampling_steps))
        self.stereopilot_guide_scale = float(stereopilot_guide_scale)
        self.stereopilot_shift = float(stereopilot_shift)
        self.stereopilot_domain_label = 1 if int(stereopilot_domain_label) != 0 else 0
        self.stereopilot_tail_pad_frames = max(0, int(stereopilot_tail_pad_frames))

        self.stereopilot_dtype = str(stereopilot_dtype or "bfloat16").strip().lower()
        self.stereopilot_transformer_dtype = str(stereopilot_transformer_dtype or "float8").strip().lower()

        if not torch.cuda.is_available():
            raise RuntimeError("StereoPilot requires CUDA, but no CUDA device is available.")

        self.device = "cuda:0"
        self._pipe = None
        self._cache_video_fn = None
        self._dtype_map = None

        self._transformer_ckpt_path = self._resolve_transformer_ckpt_path(self.stereopilot_model_path)
        self._base_ckpt_dir = self._resolve_base_ckpt_dir(self.stereopilot_base_model_path)

        self._init_stereopilot_pipeline()

    @staticmethod
    def _coerce_multiple(value: int, *, label: str, multiple: int, minimum: int) -> int:
        rounded = int(round(float(value) / float(multiple)) * float(multiple))
        rounded = max(int(minimum), int(rounded))
        if rounded != int(value):
            _logger.warning("%s adjusted from %s to %s (multiple of %s).", label, value, rounded, multiple)
        return rounded

    def _resolve_stereopilot_repo_path(self, configured_path: str) -> str:
        repo_root = Path(__file__).resolve().parents[1]
        candidates = []

        if configured_path:
            candidates.append(Path(configured_path).expanduser())

        env_repo = os.environ.get("STEREOPILOT_REPO", "").strip()
        if env_repo:
            candidates.append(Path(env_repo).expanduser())

        candidates.append(repo_root / _DEFAULT_STEREOPILOT_REPO_REL)

        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except Exception:
                continue
            if not resolved.exists():
                continue
            if (resolved / "sample.py").is_file() and (resolved / "models").is_dir():
                return str(resolved)

        raise FileNotFoundError(
            "StereoPilot repository not found. "
            "Set 'StereoPilot Repo Path' in the GUI (or STEREOPILOT_REPO env var) "
            "to a clone that contains sample.py and models/. "
            "You can install it with: PYTHON_BIN=<your_python> bash scripts/setup_stereopilot_local.sh"
        )

    def _resolve_cache_dir(self, configured_cache: str) -> str:
        cache_dir = str(configured_cache or "").strip()
        if cache_dir:
            return str(Path(cache_dir).expanduser().resolve())
        env_cache = os.environ.get("STEREOPILOT_CACHE_DIR", "").strip()
        if env_cache:
            return str(Path(env_cache).expanduser().resolve())
        return ""

    def _resolve_transformer_ckpt_path(self, configured_model_path: str) -> str:
        raw = str(configured_model_path or "").strip()
        if not raw:
            raw = _DEFAULT_STEREOPILOT_MODEL_ID

        candidate = Path(raw).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        if not candidate.is_absolute():
            joined = (Path(self.stereopilot_repo_path) / candidate).resolve()
            if joined.is_file():
                return str(joined)

        default_local = (Path(self.stereopilot_repo_path) / "ckpt" / "StereoPilot.safetensors").resolve()
        if default_local.is_file():
            return str(default_local)

        raise FileNotFoundError(
            "StereoPilot transformer checkpoint not found. "
            f"Expected local file at '{default_local}' (or set 'StereoPilot Model Path'). "
            "Use setup script: PYTHON_BIN=<your_python> bash scripts/setup_stereopilot_local.sh"
        )

    def _resolve_base_ckpt_dir(self, configured_base_path: str) -> str:
        raw = str(configured_base_path or "").strip()
        if not raw:
            raw = _DEFAULT_STEREOPILOT_BASE_MODEL_ID

        candidate = Path(raw).expanduser()
        if candidate.is_dir():
            return str(candidate.resolve())
        if not candidate.is_absolute():
            joined = (Path(self.stereopilot_repo_path) / candidate).resolve()
            if joined.is_dir():
                return str(joined)

        default_local = (Path(self.stereopilot_repo_path) / "ckpt" / "Wan2.1-T2V-1.3B").resolve()
        if default_local.is_dir():
            return str(default_local)

        raise FileNotFoundError(
            "StereoPilot base model directory not found. "
            f"Expected local directory at '{default_local}' (or set 'StereoPilot Base Model Path'). "
            "Use setup script: PYTHON_BIN=<your_python> bash scripts/setup_stereopilot_local.sh"
        )

    def _import_stereopilot_modules(self):
        repo_root = Path(self.stereopilot_repo_path).resolve()
        repo_path = str(repo_root)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        try:
            from models import StereoPilot as stereopilot_mod  # pylint: disable=import-error
            from utils.common import DTYPE_MAP, cache_video  # pylint: disable=import-error
        except Exception as exc:
            raise RuntimeError(
                "Failed to import StereoPilot modules. "
                "Ensure repo/dependencies are installed. "
                "Try: PYTHON_BIN=<your_python> bash scripts/setup_stereopilot_local.sh"
            ) from exc

        return {
            "stereopilot_mod": stereopilot_mod,
            "dtype_map": DTYPE_MAP,
            "cache_video": cache_video,
        }

    def _patch_wan_attention_fallback_if_needed(self) -> None:
        """
        StereoPilot/Wan imports `flash_attention` directly and may assert when flash-attn
        is missing. Patch those callsites to Wan's generic attention() fallback.
        """
        try:
            from wan.modules import attention as wan_attention_mod  # pylint: disable=import-error
            from wan.modules import model as wan_model_mod  # pylint: disable=import-error
            import wan.modules as wan_modules_pkg  # pylint: disable=import-error
            try:
                from wan.modules import clip as wan_clip_mod  # pylint: disable=import-error
            except Exception:
                wan_clip_mod = None
        except Exception:
            return

        flash2 = bool(getattr(wan_attention_mod, "FLASH_ATTN_2_AVAILABLE", False))
        flash3 = bool(getattr(wan_attention_mod, "FLASH_ATTN_3_AVAILABLE", False))
        flash_available = flash2 or flash3
        if flash_available:
            _logger.info("StereoPilot: flash-attn available (fa2=%s, fa3=%s).", flash2, flash3)
            return

        if bool(getattr(wan_model_mod, "_stereocrafter_safe_flash_patched", False)):
            return

        attention_fn = getattr(wan_attention_mod, "attention", None)
        if attention_fn is None:
            return

        def _safe_flash_attention(
            q,
            k,
            v,
            q_lens=None,
            k_lens=None,
            dropout_p=0.0,
            softmax_scale=None,
            q_scale=None,
            causal=False,
            window_size=(-1, -1),
            deterministic=False,
            dtype=torch.bfloat16,
            version=None,
            fa_version=None,
        ):
            selected_fa_version = fa_version if fa_version is not None else version
            return attention_fn(
                q=q,
                k=k,
                v=v,
                q_lens=q_lens,
                k_lens=k_lens,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                q_scale=q_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                dtype=dtype,
                fa_version=selected_fa_version,
            )

        wan_model_mod.flash_attention = _safe_flash_attention
        wan_modules_pkg.flash_attention = _safe_flash_attention
        if wan_clip_mod is not None and hasattr(wan_clip_mod, "flash_attention"):
            wan_clip_mod.flash_attention = _safe_flash_attention
        setattr(wan_model_mod, "_stereocrafter_safe_flash_patched", True)
        _logger.warning(
            "StereoPilot: flash-attn not available; using SDPA fallback. "
            "This is slower but avoids assertion failures."
        )

    def _init_stereopilot_pipeline(self) -> None:
        imports = self._import_stereopilot_modules()
        self._patch_wan_attention_fallback_if_needed()
        dtype_map = imports["dtype_map"]

        if self.stereopilot_dtype not in dtype_map:
            raise ValueError(
                f"Unsupported StereoPilot dtype '{self.stereopilot_dtype}'. "
                f"Available: {', '.join(sorted(dtype_map.keys()))}"
            )
        if self.stereopilot_transformer_dtype not in dtype_map:
            raise ValueError(
                f"Unsupported StereoPilot transformer dtype '{self.stereopilot_transformer_dtype}'. "
                f"Available: {', '.join(sorted(dtype_map.keys()))}"
            )

        model_cfg = {
            "type": "stereopilot",
            "ckpt_path": self._base_ckpt_dir,
            "transformer_path": self._transformer_ckpt_path,
            "pretrained_path": self._transformer_ckpt_path,
            "dtype": dtype_map[self.stereopilot_dtype],
            "transformer_dtype": dtype_map[self.stereopilot_transformer_dtype],
        }
        config = {"model": model_cfg}

        _logger.info(
            "Initializing StereoPilot backend (repo=%s, transformer=%s, base=%s)",
            self.stereopilot_repo_path,
            self._transformer_ckpt_path,
            self._base_ckpt_dir,
        )

        pipe = imports["stereopilot_mod"].StereoPilotPipeline(config)
        pipe.load_diffusion_model()
        pipe.register_custom_op()

        pipe.transformer.eval()
        torch.set_grad_enabled(False)

        pipe.transformer.to(self.device)
        pipe.vae.model.to(self.device)
        pipe.vae.mean = pipe.vae.mean.to(self.device)
        pipe.vae.std = pipe.vae.std.to(self.device)
        pipe.text_encoder.model.to(self.device)

        self._pipe = pipe
        self._cache_video_fn = imports["cache_video"]
        self._dtype_map = dtype_map

        _logger.info("StereoPilot backend initialized successfully.")

    def _emit_runtime_stage(self, stage: str, **payload):
        callback = getattr(self, "runtime_stage_callback", None)
        if callback is None:
            return
        try:
            callback(stage, payload)
        except Exception:
            pass

    def _resolve_input_video(self, video_path_or_frames_or_info: Union[str, np.ndarray, dict], segment_job_info_param: Optional[dict]) -> Tuple[str, str]:
        if segment_job_info_param is not None:
            raise RuntimeError(
                "StereoPilot backend does not support segmented processing. "
                "Disable 'Process as Segments' for StereoPilot runs."
            )

        if isinstance(video_path_or_frames_or_info, np.ndarray):
            raise RuntimeError("StereoPilot backend does not support raw frame-array input in this GUI path.")

        if isinstance(video_path_or_frames_or_info, dict):
            source_type = str(video_path_or_frames_or_info.get("source_type", "video_file") or "video_file").strip()
            if source_type not in {"video_file", "single_video_file"}:
                raise RuntimeError("StereoPilot backend currently supports video files only.")
            video_path = str(video_path_or_frames_or_info.get("video_path", "")).strip()
            original_basename = str(video_path_or_frames_or_info.get("original_basename", "")).strip()
            if not original_basename and video_path:
                original_basename = Path(video_path).stem
        elif isinstance(video_path_or_frames_or_info, str):
            video_path = str(video_path_or_frames_or_info).strip()
            original_basename = Path(video_path).stem
        else:
            raise ValueError(
                "StereoPilotDemo.run: video_path_or_frames_or_info must be str or dict. "
                f"Got {type(video_path_or_frames_or_info).__name__}."
            )

        if not video_path:
            raise ValueError("StereoPilot input video path is empty.")

        video_path_resolved = str(Path(video_path).expanduser().resolve())
        if not Path(video_path_resolved).is_file():
            raise FileNotFoundError(f"StereoPilot input video not found: {video_path_resolved}")

        return video_path_resolved, (original_basename or Path(video_path_resolved).stem)

    def _load_frames_for_processing(
        self,
        input_video_path: str,
        process_length_for_read: int,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        (
            frames_f32,
            actual_output_fps,
            source_height,
            source_width,
            processed_height,
            processed_width,
            video_stream_info,
            _,
        ) = read_video_frames(
            input_video_path,
            process_length=int(process_length_for_read),
            target_fps=-1.0,
            target_height=int(self.stereopilot_target_height),
            target_width=int(self.stereopilot_target_width),
            round_to_multiple=8,
        )
        if frames_f32 is None or frames_f32.size == 0:
            raise RuntimeError(f"StereoPilot input has no decodable frames: {input_video_path}")

        prepared = np.clip(np.round(frames_f32[..., :3] * 255.0), 0.0, 255.0).astype(np.uint8)
        total_frames_stream = 0
        if isinstance(video_stream_info, dict):
            try:
                total_frames_stream = int(video_stream_info.get("nb_frames", 0) or 0)
            except Exception:
                total_frames_stream = 0
        info = {
            "source_total_frames": int(total_frames_stream) if total_frames_stream > 0 else int(prepared.shape[0]),
            "source_height": int(source_height),
            "source_width": int(source_width),
            "processed_frames": int(prepared.shape[0]),
            "processed_height": int(processed_height),
            "processed_width": int(processed_width),
            "processed_fps": float(actual_output_fps) if float(actual_output_fps) > 0 else float(self.stereopilot_target_fps),
            "preserve_source_fps": True,
        }
        return prepared, info

    @staticmethod
    def _plan_temporal_windows(total_frames: int, window_size: int, overlap: int) -> List[Tuple[int, int]]:
        total = max(0, int(total_frames))
        if total <= 0:
            return []

        window = max(1, min(int(window_size), total))
        overlap_clamped = max(0, min(int(overlap), window - 1))
        step = max(1, window - overlap_clamped)
        windows: List[Tuple[int, int]] = []
        start = 0
        while start < total:
            end = min(start + window, total)
            windows.append((int(start), int(end)))
            if end >= total:
                break
            start = int(start + step)
        return windows

    @staticmethod
    def _ensure_frame_count(frames_rgb: np.ndarray, target_count: int) -> np.ndarray:
        target = max(1, int(target_count))
        if frames_rgb.ndim != 4 or frames_rgb.shape[-1] < 3:
            raise RuntimeError("StereoPilot output frame tensor has invalid shape.")
        count = int(frames_rgb.shape[0])
        if count == target:
            return np.asarray(frames_rgb[..., :3], dtype=np.uint8)
        if count <= 0:
            raise RuntimeError("StereoPilot produced zero output frames for a window.")
        if count > target:
            return np.asarray(frames_rgb[:target, ..., :3], dtype=np.uint8)
        pad_count = target - count
        pad = np.repeat(frames_rgb[count - 1 : count, ..., :3], pad_count, axis=0)
        return np.concatenate([frames_rgb[..., :3], pad], axis=0).astype(np.uint8)

    @staticmethod
    def _decode_output_tensor_to_frames(output_video: torch.Tensor) -> np.ndarray:
        if not torch.is_tensor(output_video):
            raise RuntimeError("StereoPilot model returned non-tensor output.")

        tensor = output_video.detach().to(torch.float32).cpu()
        if tensor.ndim == 5:
            if int(tensor.shape[0]) <= 0:
                raise RuntimeError("StereoPilot output tensor batch is empty.")
            tensor = tensor[0]
        if tensor.ndim != 4:
            raise RuntimeError(f"Unexpected StereoPilot output tensor shape: {tuple(tensor.shape)}")

        if int(tensor.shape[0]) in (1, 3, 4):
            tensor = tensor[:3].permute(1, 2, 3, 0).contiguous()
        elif int(tensor.shape[1]) in (1, 3, 4):
            tensor = tensor[:, :3, :, :].permute(0, 2, 3, 1).contiguous()
        else:
            raise RuntimeError(f"Unable to infer StereoPilot output layout: {tuple(tensor.shape)}")

        min_val = float(tensor.min().item())
        max_val = float(tensor.max().item())
        if min_val >= -1e-4 and max_val <= 1.0001:
            tensor = tensor.clamp(0.0, 1.0).mul(255.0)
        else:
            tensor = tensor.clamp(-1.0, 1.0).add(1.0).mul(127.5)

        return tensor.round().clamp(0.0, 255.0).to(torch.uint8).numpy()

    def _encode_condition_latents(self, prepared_frames: np.ndarray) -> torch.Tensor:
        if prepared_frames.ndim != 4 or prepared_frames.shape[-1] < 3:
            raise RuntimeError("StereoPilot prepared frame tensor has invalid shape.")
        tensor = torch.from_numpy(prepared_frames[..., :3]).to(torch.float32) / 255.0
        tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        tensor = tensor * 2.0 - 1.0
        latents = self._pipe.vae.model.encode(tensor.to(self.device), self._pipe.vae.scale).squeeze()
        if not torch.is_tensor(latents):
            raise RuntimeError("StereoPilot VAE encode did not return a tensor.")
        return latents

    def _write_video(self, frames_rgb: np.ndarray, output_path: str, fps: float) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        writer = imageio.get_writer(output_path, fps=float(fps), codec="libx264", quality=8)
        try:
            for frame in frames_rgb:
                writer.append_data(frame)
        finally:
            writer.close()

    def _write_sbs_video(self, left_frames: np.ndarray, right_frames: np.ndarray, output_path: str, fps: float) -> int:
        count = min(int(left_frames.shape[0]), int(right_frames.shape[0]))
        if count <= 0:
            raise RuntimeError("StereoPilot SBS write failed: no frames available.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        writer = imageio.get_writer(output_path, fps=float(fps), codec="libx264", quality=8)
        try:
            for idx in range(count):
                left = np.asarray(left_frames[idx, ..., :3], dtype=np.uint8)
                right = np.asarray(right_frames[idx, ..., :3], dtype=np.uint8)
                if left.shape[0] != right.shape[0] or left.shape[1] != right.shape[1]:
                    right = cv2.resize(right, (int(left.shape[1]), int(left.shape[0])), interpolation=cv2.INTER_LINEAR)
                writer.append_data(np.concatenate([left, right], axis=1))
        finally:
            writer.close()
        return count

    def _load_video_frames(self, video_path: str) -> np.ndarray:
        vr = VideoReader(video_path, ctx=cpu(0))
        frame_count = int(len(vr))
        if frame_count <= 0:
            raise RuntimeError(f"StereoPilot output has no decodable frames: {video_path}")
        frames = vr.get_batch(list(range(frame_count))).asnumpy()
        return np.asarray(frames[..., :3], dtype=np.uint8)

    def _resolve_prompt(self, input_video_path: str) -> Tuple[str, str]:
        if self.stereopilot_use_sidecar_prompt:
            prompt_path = Path(input_video_path).with_suffix(".txt")
            if prompt_path.is_file():
                try:
                    text = prompt_path.read_text(encoding="utf-8").strip()
                except Exception:
                    text = ""
                if text:
                    return text.splitlines()[0].strip(), str(prompt_path)

        return self.stereopilot_prompt_default, "default"

    def _render_sbs(self, left_frames: np.ndarray, right_frames: np.ndarray) -> np.ndarray:
        count = min(int(left_frames.shape[0]), int(right_frames.shape[0]))
        if count <= 0:
            raise RuntimeError("StereoPilot SBS render failed: no frames available.")

        left = left_frames[:count]
        right = right_frames[:count]

        if left.shape[1] != right.shape[1] or left.shape[2] != right.shape[2]:
            resized_right = [
                cv2.resize(frame, (int(left.shape[2]), int(left.shape[1])), interpolation=cv2.INTER_LINEAR)
                for frame in right
            ]
            right = np.asarray(resized_right, dtype=np.uint8)

        return np.concatenate([left, right], axis=2)

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
        del num_denoising_steps
        del guidance_scale
        del target_height
        del target_width
        del keep_intermediate_npz_config
        del intermediate_segment_visual_format_config
        del full_video_output_format

        input_video_path, inferred_basename = self._resolve_input_video(
            video_path_or_frames_or_info,
            segment_job_info_param,
        )
        output_basename = original_video_basename_override or inferred_basename

        output_dir = Path(base_output_folder).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        right_filename = f"{output_basename}_stereopilot_right.mp4"
        sbs_filename = f"{output_basename}_stereopilot_sbs.mp4"

        right_output_path = output_dir / right_filename
        sbs_output_path = output_dir / sbs_filename

        temp_dir = Path(tempfile.mkdtemp(prefix=f"stereopilot_{output_basename}_", dir=str(output_dir)))

        prompt_text = ""
        prompt_source = "default"
        primary_output_path = right_output_path
        frame_count_written = 0
        wrote_right = False
        wrote_sbs = False

        try:
            self._emit_runtime_stage("demo_run_start", input=input_video_path, backend=self.model_backend)

            self._emit_runtime_stage(
                "logic_load_frames_start",
                input=input_video_path,
                target_width=int(self.stereopilot_target_width),
                target_height=int(self.stereopilot_target_height),
                target_fps_requested=float(self.stereopilot_target_fps),
                preserve_source_fps=True,
                gui_window_size=int(gui_window_size),
                gui_overlap=int(gui_overlap),
            )
            t_load_start = time.perf_counter()
            self._emit_runtime_stage("pipe_decode_sample_start")
            prepared_frames, load_info = self._load_frames_for_processing(
                input_video_path=input_video_path,
                process_length_for_read=int(process_length_for_read_full_video),
            )
            self._emit_runtime_stage("pipe_decode_sample_end", **load_info)
            prompt_text, prompt_source = self._resolve_prompt(input_video_path)
            output_fps = float(load_info.get("processed_fps", self.stereopilot_target_fps))
            if output_fps <= 0:
                output_fps = float(self.stereopilot_target_fps)
            if abs(float(output_fps) - float(self.stereopilot_target_fps)) > 1e-3:
                _logger.info(
                    "StereoPilot FPS behavior: preserving source FPS %.3f (requested %.3f).",
                    float(output_fps),
                    float(self.stereopilot_target_fps),
                )

            total_frames = int(prepared_frames.shape[0])
            if total_frames <= 0:
                raise RuntimeError("StereoPilot preprocessing produced no frames.")

            gui_window_requested = int(gui_window_size) if int(gui_window_size) > 0 else 81
            effective_window = max(1, min(total_frames, gui_window_requested))
            effective_overlap = max(0, min(int(gui_overlap), max(0, effective_window - 1)))
            if int(gui_overlap) >= effective_window:
                _logger.warning(
                    "StereoPilot overlap %s is >= window size %s; clamped to %s.",
                    int(gui_overlap),
                    int(effective_window),
                    int(effective_overlap),
                )

            self._emit_runtime_stage(
                "pipe_window_plan_start",
                total_frames=int(total_frames),
                window_size=int(effective_window),
                overlap=int(effective_overlap),
            )
            windows = self._plan_temporal_windows(total_frames, effective_window, effective_overlap)
            self._emit_runtime_stage(
                "pipe_window_plan_end",
                window_count=int(len(windows)),
                window_size=int(effective_window),
                overlap=int(effective_overlap),
            )
            if not windows:
                raise RuntimeError("StereoPilot failed to create temporal windows.")

            self._emit_runtime_stage(
                "logic_load_frames_end",
                frames=int(total_frames),
                processed_height=int(prepared_frames.shape[1]),
                processed_width=int(prepared_frames.shape[2]),
                source_total_frames=load_info.get("source_total_frames", 0),
                source_height=load_info.get("source_height", 0),
                source_width=load_info.get("source_width", 0),
                output_fps=float(output_fps),
                window_count=int(len(windows)),
                window_size=int(effective_window),
                overlap=int(effective_overlap),
                tail_pad_frames=int(self.stereopilot_tail_pad_frames),
                elapsed_sec=round(time.perf_counter() - t_load_start, 3),
            )
            _logger.info(
                "StereoPilot load/prep complete | src=%sx%s %sfr -> proc=%sx%s %sfr @ %.2ffps | windows=%s size=%s overlap=%s tail_pad=%s | prompt_source=%s",
                load_info.get("source_width", 0),
                load_info.get("source_height", 0),
                load_info.get("source_total_frames", 0),
                int(prepared_frames.shape[2]),
                int(prepared_frames.shape[1]),
                int(total_frames),
                float(output_fps),
                len(windows),
                int(effective_window),
                int(effective_overlap),
                int(self.stereopilot_tail_pad_frames),
                prompt_source,
            )

            self._emit_runtime_stage(
                "logic_inference_start",
                sampling_steps=int(self.stereopilot_sampling_steps),
                guide_scale=float(self.stereopilot_guide_scale),
                shift=float(self.stereopilot_shift),
                domain_label=int(self.stereopilot_domain_label),
                total_frames=int(total_frames),
                window_count=int(len(windows)),
                window_size=int(effective_window),
                overlap=int(effective_overlap),
                tail_pad_frames=int(self.stereopilot_tail_pad_frames),
            )
            result_h = int(prepared_frames.shape[1])
            result_w = int(prepared_frames.shape[2])
            accum_frames = np.zeros((total_frames, result_h, result_w, 3), dtype=np.float32)
            accum_weights = np.zeros((total_frames, 1, 1, 1), dtype=np.float32)

            infer_start_ts = time.perf_counter()
            for window_idx, (start_idx, end_idx) in enumerate(windows, start=1):
                window_len = int(end_idx - start_idx)
                if window_len <= 0:
                    continue

                window_frames = np.asarray(prepared_frames[start_idx:end_idx, ..., :3], dtype=np.uint8)
                window_tail_pad = int(self.stereopilot_tail_pad_frames)
                model_frame_count = int(window_len + window_tail_pad)
                if window_tail_pad > 0:
                    pad_block = np.repeat(window_frames[-1:, ...], window_tail_pad, axis=0)
                    window_frames_for_model = np.concatenate([window_frames, pad_block], axis=0)
                else:
                    window_frames_for_model = window_frames
                self._emit_runtime_stage(
                    "pipe_window_inference_start",
                    window_index=int(window_idx),
                    window_count=int(len(windows)),
                    start_frame=int(start_idx),
                    end_frame=int(end_idx),
                    window_frames=int(window_len),
                    model_frames=int(model_frame_count),
                    tail_pad_frames=int(window_tail_pad),
                )
                _logger.info(
                    "StereoPilot window %s/%s | frames %s:%s (%s) | model_frames=%s tail_pad=%s",
                    int(window_idx),
                    int(len(windows)),
                    int(start_idx),
                    int(end_idx),
                    int(window_len),
                    int(model_frame_count),
                    int(window_tail_pad),
                )

                self._emit_runtime_stage(
                    "pipe_encode_condition_start",
                    window_index=int(window_idx),
                    window_count=int(len(windows)),
                )
                t_encode_start = time.perf_counter()
                latents_video_condition = self._encode_condition_latents(window_frames_for_model)
                self._emit_runtime_stage(
                    "pipe_encode_condition_end",
                    window_index=int(window_idx),
                    window_count=int(len(windows)),
                    latent_shape=str(tuple(latents_video_condition.shape)),
                    elapsed_sec=round(time.perf_counter() - t_encode_start, 3),
                )

                self._emit_runtime_stage(
                    "pipe_sample_start",
                    window_index=int(window_idx),
                    window_count=int(len(windows)),
                    frame_num=int(model_frame_count),
                )
                t_sample_start = time.perf_counter()
                output_video = self._pipe.sample(
                    prompt=prompt_text,
                    video_condition=latents_video_condition,
                    size=(int(result_w), int(result_h)),
                    frame_num=int(model_frame_count),
                    shift=float(self.stereopilot_shift),
                    sample_solver="unipc",
                    sampling_steps=int(self.stereopilot_sampling_steps),
                    guide_scale=float(self.stereopilot_guide_scale),
                    n_prompt="",
                    seed=int(seed),
                    domain_label=int(self.stereopilot_domain_label),
                )
                self._emit_runtime_stage(
                    "pipe_sample_end",
                    window_index=int(window_idx),
                    window_count=int(len(windows)),
                    elapsed_sec=round(time.perf_counter() - t_sample_start, 3),
                )

                if isinstance(output_video, (list, tuple)):
                    if not output_video:
                        raise RuntimeError(f"StereoPilot model returned an empty output list for window {window_idx}.")
                    output_video = output_video[0]

                self._emit_runtime_stage(
                    "pipe_decode_output_start",
                    window_index=int(window_idx),
                    window_count=int(len(windows)),
                )
                window_right_frames = self._decode_output_tensor_to_frames(output_video)
                window_right_frames = self._ensure_frame_count(window_right_frames, model_frame_count)
                if window_tail_pad > 0:
                    window_right_frames = window_right_frames[:window_len]
                if (
                    int(window_right_frames.shape[1]) != result_h
                    or int(window_right_frames.shape[2]) != result_w
                ):
                    window_right_frames = np.asarray(
                        [
                            cv2.resize(
                                frame,
                                (result_w, result_h),
                                interpolation=cv2.INTER_LINEAR,
                            )
                            for frame in window_right_frames
                        ],
                        dtype=np.uint8,
                    )
                self._emit_runtime_stage(
                    "pipe_decode_output_end",
                    window_index=int(window_idx),
                    window_count=int(len(windows)),
                    decoded_frames=int(window_right_frames.shape[0]),
                    model_frames=int(model_frame_count),
                    tail_pad_frames=int(window_tail_pad),
                )

                local_weights = np.ones((window_len,), dtype=np.float32)
                if effective_overlap > 0:
                    if start_idx > 0:
                        fade = min(effective_overlap, window_len)
                        local_weights[:fade] *= np.linspace(
                            1.0 / float(fade + 1),
                            1.0,
                            num=fade,
                            dtype=np.float32,
                        )
                    if end_idx < total_frames:
                        fade = min(effective_overlap, window_len)
                        local_weights[-fade:] *= np.linspace(
                            1.0,
                            1.0 / float(fade + 1),
                            num=fade,
                            dtype=np.float32,
                        )
                local_weights = np.clip(local_weights, 1e-3, None)
                local_weights_4d = local_weights[:, None, None, None]

                self._emit_runtime_stage(
                    "pipe_window_blend_start",
                    window_index=int(window_idx),
                    window_count=int(len(windows)),
                )
                accum_frames[start_idx:end_idx] += window_right_frames.astype(np.float32) * local_weights_4d
                accum_weights[start_idx:end_idx] += local_weights_4d
                self._emit_runtime_stage(
                    "pipe_window_blend_end",
                    window_index=int(window_idx),
                    window_count=int(len(windows)),
                    weight_min=float(local_weights.min()) if local_weights.size else 0.0,
                    weight_max=float(local_weights.max()) if local_weights.size else 0.0,
                )

                elapsed_total = time.perf_counter() - infer_start_ts
                _logger.info(
                    "StereoPilot window %s/%s done | elapsed %.2fs total",
                    int(window_idx),
                    int(len(windows)),
                    float(elapsed_total),
                )
                self._emit_runtime_stage(
                    "pipe_window_inference_end",
                    window_index=int(window_idx),
                    window_count=int(len(windows)),
                    start_frame=int(start_idx),
                    end_frame=int(end_idx),
                    model_frames=int(model_frame_count),
                    tail_pad_frames=int(window_tail_pad),
                )

            right_frames = np.clip(
                np.round(accum_frames / np.maximum(accum_weights, 1e-6)),
                0.0,
                255.0,
            ).astype(np.uint8)
            frame_count_written = int(right_frames.shape[0])

            self._emit_runtime_stage("logic_save_full_video_start")
            if self.stereopilot_output_mode in {"opposite_eye", "both"}:
                self._emit_runtime_stage("pipe_write_right_video_start")
                t_write_right_start = time.perf_counter()
                self._write_video(right_frames, str(right_output_path), output_fps)
                self._emit_runtime_stage(
                    "pipe_write_right_video_end",
                    elapsed_sec=round(time.perf_counter() - t_write_right_start, 3),
                    output_path=str(right_output_path),
                )
                if not right_output_path.is_file():
                    raise RuntimeError("StereoPilot failed to write opposite-eye output video.")
                wrote_right = True
                primary_output_path = right_output_path
            else:
                self._emit_runtime_stage("pipe_write_right_video_skipped", output_mode=self.stereopilot_output_mode)

            if self.stereopilot_output_mode in {"side_by_side", "both"}:
                self._emit_runtime_stage("pipe_render_sbs_start")
                sbs_count = self._write_sbs_video(prepared_frames, right_frames, str(sbs_output_path), output_fps)
                self._emit_runtime_stage(
                    "pipe_render_sbs_end",
                    frames=int(sbs_count),
                    output_path=str(sbs_output_path),
                )
                frame_count_written = int(sbs_count)
                wrote_sbs = True
                primary_output_path = sbs_output_path

            self._emit_runtime_stage("logic_save_full_video_end", save_path=str(primary_output_path))
            self._emit_runtime_stage(
                "logic_inference_end",
                save_path=str(primary_output_path),
                frames_in_output=int(frame_count_written),
            )
            self._emit_runtime_stage("demo_run_end", save_path=str(primary_output_path))

            metadata = {
                "original_video_basename": output_basename,
                "status": "success",
                "model_backend": self.model_backend,
                "processed_height": int(prepared_frames.shape[1]),
                "processed_width": int(prepared_frames.shape[2]),
                "frames_in_output_video": int(frame_count_written),
                "source_total_frames": int(load_info.get("source_total_frames", total_frames)),
                "processed_input_frames": int(total_frames),
                "target_fps_setting": float(output_fps),
                "output_video_filename": str(primary_output_path.name),
                "output_video_format": "mp4",
                "stereopilot_output_mode": self.stereopilot_output_mode,
                "stereopilot_opposite_eye_path": str(right_output_path) if wrote_right else "",
                "stereopilot_opposite_eye_written": bool(wrote_right),
                "stereopilot_window_count": int(len(windows)),
                "stereopilot_window_size_used": int(effective_window),
                "stereopilot_overlap_used": int(effective_overlap),
                "stereopilot_tail_pad_frames": int(self.stereopilot_tail_pad_frames),
                "stereopilot_gui_window_size": int(gui_window_size),
                "stereopilot_gui_overlap": int(gui_overlap),
                "prompt_source": prompt_source,
                "prompt_text": prompt_text,
                "sampling_steps": int(self.stereopilot_sampling_steps),
                "guide_scale": float(self.stereopilot_guide_scale),
                "shift": float(self.stereopilot_shift),
                "domain_label": int(self.stereopilot_domain_label),
                "_individual_metadata_path": None,
            }
            if wrote_sbs and sbs_output_path.is_file():
                metadata["stereopilot_sbs_path"] = str(sbs_output_path)

            if save_final_json_for_this_job_config:
                sidecar_path = get_sidecar_json_filename(str(primary_output_path))
                sidecar_payload = {
                    "source_video": input_video_path,
                    "output_video": str(primary_output_path),
                    "backend": self.model_backend,
                    "status": metadata["status"],
                    "processed_width": metadata["processed_width"],
                    "processed_height": metadata["processed_height"],
                    "frames_in_output_video": metadata["frames_in_output_video"],
                    "source_total_frames": metadata["source_total_frames"],
                    "processed_input_frames": metadata["processed_input_frames"],
                    "target_fps_setting": metadata["target_fps_setting"],
                    "stereopilot_output_mode": metadata["stereopilot_output_mode"],
                    "stereopilot_opposite_eye_path": metadata["stereopilot_opposite_eye_path"],
                    "stereopilot_opposite_eye_written": metadata["stereopilot_opposite_eye_written"],
                    "stereopilot_sbs_path": metadata.get("stereopilot_sbs_path"),
                    "stereopilot_window_count": metadata["stereopilot_window_count"],
                    "stereopilot_window_size_used": metadata["stereopilot_window_size_used"],
                    "stereopilot_overlap_used": metadata["stereopilot_overlap_used"],
                    "stereopilot_tail_pad_frames": metadata["stereopilot_tail_pad_frames"],
                    "prompt_source": metadata["prompt_source"],
                    "prompt_text": metadata["prompt_text"],
                    "sampling_steps": metadata["sampling_steps"],
                    "guide_scale": metadata["guide_scale"],
                    "shift": metadata["shift"],
                    "domain_label": metadata["domain_label"],
                }
                if save_json_file(sidecar_payload, sidecar_path):
                    metadata["_individual_metadata_path"] = sidecar_path

            return str(primary_output_path), metadata

        finally:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
