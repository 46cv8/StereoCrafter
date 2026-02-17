import threading
import codecs
import gc
import os
import sys
import glob
import shutil
import json
import ast
import re
import shlex
import select
import socket
import subprocess
import tkinter as tk
from tkinter import Toplevel, Label
from tkinter import filedialog, messagebox, ttk
import queue # Still needed for progress updates
import time
import numpy as np
import torch
import logging # Import standard logging
import random
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

# Configure a logger for this module
_logger = logging.getLogger(__name__)

# Import backend logic classes
from depthcrafter.depthcrafter_logic import DepthCrafterDemo
from depthcrafter.geometrycrafter_logic import GeometryCrafterDemo
from depthcrafter.stereopilot_logic import StereoPilotDemo

from depthcrafter.utils import (
    format_duration,
    get_segment_output_folder_name,
    get_segment_npz_output_filename,
    get_full_video_output_filename,
    get_sidecar_json_filename,
    get_image_sequence_metadata,
    get_single_image_metadata,
    define_video_segments,
    load_json_file,
    save_json_file,
    save_depth_visual_as_mp4_util,
    save_depth_visual_as_png_sequence_util,
    save_depth_visual_as_exr_sequence_util,
    save_depth_visual_as_single_exr_util,
    get_video_stream_info,
)
from dependency.clip_ordering import clip_sort_key, sort_paths_by_clip_id

try:
    from depthcrafter import merge_depth_segments
except ImportError as e:
    _logger.warning(f"Could not import 'merge_depth_segments'. Merging functionality will not be available. Error: {e}")
    merge_depth_segments = None

try:
    from depthcrafter.spatial_refine import run_spatial_hires_refine
except ImportError as e:
    _logger.warning(f"Could not import 'spatial_refine'. Hi-Res Spatial Refine mode will be unavailable. Error: {e}")
    run_spatial_hires_refine = None

try:
    from depthcrafter.edge_guided_upscale import run_edge_guided_hires_upscale
except ImportError as e:
    _logger.warning(f"Could not import 'edge_guided_upscale'. Edge-Guided Hi-Res mode will be unavailable. Error: {e}")
    run_edge_guided_hires_upscale = None

try:
    import OpenEXR
    import Imath
    OPENEXR_AVAILABLE_GUI = True
except ImportError:
    OPENEXR_AVAILABLE_GUI = False
    _logger.warning("OpenEXR or Imath module not found. EXR options might be limited.")


from typing import Optional, Tuple, List, Dict, Any

try:
    from ttkthemes import ThemedTk
    THEMEDTK_AVAILABLE = True
except ImportError:
    THEMEDTK_AVAILABLE = False
    _logger.warning("ttkthemes not found. Dark mode functionality will be disabled.")
# Optional helper from cloud tooling for env-file parsing.
try:
    from cloud.envfile_to_vast_env import parse_env_file as _parse_cloud_env_file
except Exception:
    _parse_cloud_env_file = None
try:
    from cloud import cloud_core
except Exception:
    cloud_core = None
# --- Imports End ---

GUI_VERSION = "25-11-01.0"
_HELP_TEXTS = {}

DARK_MODE_COLORS = {
    "bg": "#2b2b2b",
    "fg": "white",
    "entry_bg": "#3c3c3c",
    "tooltip_bg": "#4a4a4a",
    "tooltip_fg": "white",
    "theme_name": "black", # A common ttkthemes dark theme
}
LIGHT_MODE_COLORS = {
    "bg": "#d9d9d9",
    "fg": "black",
    "entry_bg": "#ffffff",
    "tooltip_bg": "#ffffe0",
    "tooltip_fg": "black",
    "theme_name": "default", # A solid default theme
}
def _create_hover_tooltip(widget, help_key):
    """Creates a mouse-over tooltip for the given widget using text from _HELP_TEXTS."""
    if help_key in _HELP_TEXTS:
        Tooltip(widget, _HELP_TEXTS[help_key])
    else:
        _logger.warning(f"No help text found for key '{help_key}' to create tooltip for widget {widget}.")

class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
        self.widget.bind("<ButtonPress>", self.hide_tooltip) # Hide on click

    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text:
            return
        # Adjust position slightly for better visibility
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        # Find the main root window (it holds the app_instance attribute)
        root_window = self.widget._root() # A common tkinter internal way to get the root Tk object
        
        # Access the main DepthCrafterGUI instance via the root widget
        gui_instance = getattr(root_window, 'app_instance', None) 
        
        if gui_instance:
            bg_color = gui_instance.current_theme_colors["tooltip_bg"]
            fg_color = gui_instance.current_theme_colors["tooltip_fg"]
        else:
            # Fallback colors if instance couldn't be found
            bg_color = "#ffffe0"
            fg_color = "black"

        self.tooltip_window = Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True) # Remove window decorations
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = Label(self.tooltip_window, text=self.text, background="#ffffe0", relief="solid", borderwidth=1, justify="left", wraplength=250)
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None
        
class DepthCrafterGUI:
    CONFIG_FILENAME = "config_depthcrafter.json"
    HELP_CONTENT_FILENAME = os.path.join("depthcrafter", "help_content.json")
    MOVE_ORIGINAL_TO_FINISHED_FOLDER_ON_COMPLETION = True
    SETTINGS_FILETYPES = [("JSON files", "*.json"), ("All files", "*.*")]
    LAST_SETTINGS_DIR_CONFIG_KEY = "last_settings_dir"
    VIDEO_EXTENSIONS = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm", "*.flv", "*.gif"]
    IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.exr"]
    DEFAULT_VAST_API_BASE_URL = cloud_core.DEFAULT_VAST_API_BASE_URL if cloud_core is not None else "https://console.vast.ai"
    CLOUD_BLACKLIST_PATH = os.path.join("cloud", "cloud_blacklist.json")
    CLOUD_PROVIDER_HISTORY_PATH = os.path.join("cloud", "cloud_provider_history.json")
    CLOUD_GPU_RAM_TOLERANCE_GB = cloud_core.GPU_RAM_TOLERANCE_GB if cloud_core is not None else 0.25
    CLOUD_PROFILE_DEFAULTS = (
        cloud_core.CLOUD_PROFILE_DEFAULTS
        if cloud_core is not None
        else {
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
                "offer_gpu_filter": "",
                "min_gpu_ram_gb": 48.0,
                "target_width": 1920,
                "target_height": 1040,
                "window_size": 75,
                "overlap": 25,
                "use_source_resolution": True,
            },
        }
    )

    def __init__(self, root):
        self.root = root
        self.root.title(f"DepthCrafter GUI Seg {GUI_VERSION}")
        self.dark_mode_var = tk.BooleanVar(value=False)
        self.current_theme_colors = LIGHT_MODE_COLORS # Initialize theme colors dictionary
        self.input_dir_or_file_var = tk.StringVar(value=os.path.normpath("./input_clips"))
        self.output_dir = tk.StringVar(value=os.path.normpath("./output_depthmaps"))
        self.guidance_scale = tk.DoubleVar(value=1.0)
        self.inference_steps = tk.IntVar(value=5)
        self.seed = tk.IntVar(value=42)
        self.cpu_offload = tk.StringVar(value="model")
        self.model_backend_var = tk.StringVar(value="depthcrafter")
        self.geometry_model_path_var = tk.StringVar(value="TencentARC/GeometryCrafter")
        self.geometry_repo_path_var = tk.StringVar(value=os.path.normpath("./weights/GeometryCrafter"))
        self.geometry_cache_dir_var = tk.StringVar(value=os.path.normpath("./weights/hf_cache"))
        self.geometry_decode_chunk_size_var = tk.IntVar(value=8)
        self.geometry_low_memory_usage_var = tk.BooleanVar(value=False)
        self.geometry_force_projection_var = tk.BooleanVar(value=True)
        self.geometry_force_fixed_focal_var = tk.BooleanVar(value=True)
        self.geometry_use_extract_interp_var = tk.BooleanVar(value=False)
        self.geometry_local_status_var = tk.StringVar(value="Geometry local prerequisites: not checked.")
        self.stereopilot_model_path_var = tk.StringVar(value="KlingTeam/StereoPilot")
        self.stereopilot_base_model_path_var = tk.StringVar(value="Wan-AI/Wan2.1-T2V-1.3B")
        self.stereopilot_repo_path_var = tk.StringVar(value=os.path.normpath("./weights/StereoPilot"))
        self.stereopilot_cache_dir_var = tk.StringVar(value=os.path.normpath("./weights/hf_cache"))
        self.stereopilot_prompt_var = tk.StringVar(value="A realistic monocular video scene with natural depth and motion.")
        self.stereopilot_use_sidecar_prompt_var = tk.BooleanVar(value=True)
        self.stereopilot_output_mode_var = tk.StringVar(value="side_by_side")
        self.stereopilot_target_width_var = tk.IntVar(value=832)
        self.stereopilot_target_height_var = tk.IntVar(value=480)
        self.stereopilot_target_fps_var = tk.DoubleVar(value=16.0)
        self.stereopilot_window_size_var = tk.IntVar(value=81)
        self.stereopilot_overlap_var = tk.IntVar(value=25)
        self.stereopilot_sampling_steps_var = tk.IntVar(value=30)
        self.stereopilot_guide_scale_var = tk.DoubleVar(value=5.0)
        self.stereopilot_shift_var = tk.DoubleVar(value=5.0)
        self.stereopilot_domain_label_var = tk.IntVar(value=1)
        self.stereopilot_dtype_var = tk.StringVar(value="bfloat16")
        self.stereopilot_transformer_dtype_var = tk.StringVar(value="float8")
        self.use_cudnn_benchmark = tk.BooleanVar(value=False)
        self.process_length = tk.IntVar(value=-1)
        self.target_fps = tk.DoubleVar(value=-1.0)
        self.window_size = tk.IntVar(value=110)
        self.overlap = tk.IntVar(value=25)
        self.process_as_segments_var = tk.BooleanVar(value=False)
        # Isolated special mode: only process clips/segments missing NPZ raw outputs.
        self.npz_backfill_missing_only_var = tk.BooleanVar(value=False)
        # Special secondary mode: panelized hi-res refinement anchored to low-res global depth.
        self.enable_spatial_refine_mode_var = tk.BooleanVar(value=False)
        # Special secondary mode: edge-guided hi-res upscaling from low-res raw depth + source RGB edges.
        self.enable_edge_guided_upscale_mode_var = tk.BooleanVar(value=False)
        # Special mode: dispatch DepthCrafter jobs to a Vast.ai cloud worker.
        self.enable_cloud_dispatch_mode_var = tk.BooleanVar(value=False)
        self.spatial_refine_options_expanded_var = tk.BooleanVar(value=False)
        self.cloud_options_expanded_var = tk.BooleanVar(value=False)
        self.spatial_refine_tile_num_var = tk.IntVar(value=2)
        self.spatial_refine_tile_num_y_var = tk.IntVar(value=2)
        self.spatial_refine_tile_overlap_var = tk.IntVar(value=128)  # Legacy single-overlap setting.
        self.spatial_refine_tile_overlap_x_var = tk.IntVar(value=128)
        self.spatial_refine_tile_overlap_y_var = tk.IntVar(value=128)
        self.spatial_refine_target_width_var = tk.IntVar(value=1920)
        self.spatial_refine_target_height_var = tk.IntVar(value=1080)
        self.spatial_refine_anchor_weight_var = tk.DoubleVar(value=0.15)
        self.spatial_refine_local_window_size_var = tk.IntVar(value=64)
        self.spatial_refine_local_window_stride_var = tk.IntVar(value=32)
        self.spatial_refine_local_confidence_low_var = tk.DoubleVar(value=0.45)
        self.spatial_refine_local_confidence_high_var = tk.DoubleVar(value=0.80)
        self.spatial_refine_use_edge_fallback_var = tk.BooleanVar(value=False)
        self.spatial_refine_edge_fallback_mix_var = tk.DoubleVar(value=0.75)
        self.edge_guided_strength_var = tk.DoubleVar(value=0.90)
        self.edge_guided_sigma_color_var = tk.DoubleVar(value=0.04)
        self.edge_guided_sigma_spatial_var = tk.DoubleVar(value=0.90)
        self.edge_guided_iterations_var = tk.IntVar(value=1)
        self.edge_guided_temporal_smooth_var = tk.DoubleVar(value=0.03)
        self.edge_guided_reinject_strength_var = tk.DoubleVar(value=0.60)
        self.edge_guided_output_suffix_var = tk.StringVar(value="_edge_hires_depth")
        self.spatial_refine_cleanup_temp_var = tk.BooleanVar(value=True)
        self.spatial_refine_output_suffix_var = tk.StringVar(value="_hires_refined_depth")
        self.cloud_profile_var = tk.StringVar(value="5090_32gb")
        # 0 => use selected cloud profile default resolution.
        self.cloud_target_width_override_var = tk.IntVar(value=0)
        self.cloud_target_height_override_var = tk.IntVar(value=0)
        # 0 / -1 => inherit from main dialog values.
        self.cloud_window_size_override_var = tk.IntVar(value=0)
        self.cloud_overlap_override_var = tk.IntVar(value=-1)
        self.cloud_image_var = tk.StringVar(value="ghcr.io/46cv8/stereocrafter-cloud:latest")
        self.cloud_disk_gb_var = tk.IntVar(value=40)
        self.cloud_reuse_existing_instance_var = tk.BooleanVar(value=True)
        self.cloud_last_instance_id_var = tk.IntVar(value=0)
        self.cloud_last_instance_profile_var = tk.StringVar(value="")
        self.cloud_last_host_var = tk.StringVar(value="")
        self.cloud_last_port_var = tk.IntVar(value=22)
        self.cloud_last_offer_id_var = tk.IntVar(value=0)
        self.cloud_last_machine_id_var = tk.IntVar(value=0)
        self.cloud_last_host_id_var = tk.IntVar(value=0)
        self.cloud_remote_user_var = tk.StringVar(value="root")
        self.cloud_remote_root_var = tk.StringVar(value="/opt/StereoCrafter")
        self.cloud_remote_venv_var = tk.StringVar(value="/opt/venv")
        self.cloud_identity_file_var = tk.StringVar(value="~/.ssh/id_vastai_ed25519")
        self.cloud_vast_env_file_var = tk.StringVar(value="cloud/vast.env")
        self.cloud_hf_env_file_var = tk.StringVar(value="cloud/hf.env")
        self.cloud_no_hf_env_var = tk.BooleanVar(value=False)
        self.cloud_use_private_registry_login_var = tk.BooleanVar(value=False)
        self.cloud_offer_limit_var = tk.IntVar(value=30)
        self.cloud_require_verified_hosts_var = tk.BooleanVar(value=True)
        self.cloud_max_dph_var = tk.DoubleVar(value=0.0)
        self.cloud_expected_runtime_hours_var = tk.DoubleVar(value=1.0)
        self.cloud_expected_upload_gb_var = tk.DoubleVar(value=8.0)
        self.cloud_expected_download_gb_var = tk.DoubleVar(value=8.0)
        self.cloud_auto_destroy_instance_var = tk.BooleanVar(value=True)
        self.cloud_secondary_summary_var = tk.StringVar(
            value="  ↳ Launches a Vast worker, uploads clip(s), runs remote depth, downloads outputs."
        )
        self.cloud_profile_default_summary_var = tk.StringVar(value="")
        self.cloud_effective_processing_summary_var = tk.StringVar(value="")
        self.cloud_inherited_processing_summary_var = tk.StringVar(value="")
        self.cloud_blacklist_summary_var = tk.StringVar(value="Blacklist: offers=0, machines=0, hosts=0")
        self._legacy_cloud_image_migrations = {
            "ghcr.io/46cv8/stereocrafter-depthcrafter:004_depthcrafter_on_cloud": "ghcr.io/46cv8/stereocrafter-cloud:latest",
            "ghcr.io/46cv8/stereocrafter-depthcrafter:latest": "ghcr.io/46cv8/stereocrafter-cloud:latest",
        }
        self.save_final_output_json_var = tk.BooleanVar(value=False)
        self.merge_output_format_var = tk.StringVar(value="mp4")
        self.merge_alignment_method_var = tk.StringVar(value="Shift & Scale")
        self.merge_dither_var = tk.BooleanVar(value=False)
        self.merge_dither_strength_var = tk.DoubleVar(value=0.5)
        self.merge_gamma_correct_var = tk.BooleanVar(value=False)
        self.merge_gamma_value_var = tk.DoubleVar(value=1.5)
        self.merge_percentile_norm_var = tk.BooleanVar(value=False)
        self.merge_norm_low_perc_var = tk.DoubleVar(value=0.1)
        self.merge_norm_high_perc_var = tk.DoubleVar(value=99.9)
        self.keep_intermediate_npz_var = tk.BooleanVar(value=False)
        self.min_frames_to_keep_npz_var = tk.IntVar(value=0)
        self.keep_intermediate_segment_visual_format_var = tk.StringVar(value="mp4")
        self.merge_output_suffix_var = tk.StringVar(value="_depth") # New Variable
        # self.merge_script_gui_silence_level_var = tk.StringVar(value="Normal (Info)") # Removed GUI verbosity control
        self.current_input_mode = "batch_folder" # "batch_folder", "single_video_file", "single_image_file", "image_sequence_folder"
        self.single_file_mode_active = False # True if a single file/sequence folder is explicitly loaded
        self.effective_move_original_on_completion = self.MOVE_ORIGINAL_TO_FINISHED_FOLDER_ON_COMPLETION
        self.use_local_models_only_var = tk.BooleanVar(value=False)
        self.status_message_var = tk.StringVar(value="Ready")
        self.current_filename_var = tk.StringVar(value="N/A")
        self.current_resolution_var = tk.StringVar(value="N/A")
        self.current_frames_var = tk.StringVar(value="N/A")
        self.target_height = tk.IntVar(value=384) # Initial default height
        self.target_width = tk.IntVar(value=640)  # Initial default width
        self.debug_logging_enabled = tk.BooleanVar(value=False) # Default to OFF (INFO level)
        self.enable_dual_output_robust_norm = tk.BooleanVar(value=False) # Default to ON for testing
        self.robust_norm_low_percentile = tk.DoubleVar(value=0.0)      # Example default
        self.robust_norm_high_percentile = tk.DoubleVar(value=75.5)     # Example default
        self.robust_norm_output_min = tk.DoubleVar(value=0.0)
        self.robust_norm_output_max = tk.DoubleVar(value=1.0)
        self.robust_output_suffix = tk.StringVar(value="_clipped_depth")
        self.is_depth_far_black = tk.BooleanVar(value=True)        
        self.disable_xformers_var = tk.BooleanVar(value=True)

        self.all_tk_vars = {
            "input_dir_or_file_var": self.input_dir_or_file_var,
            "output_dir": self.output_dir,
            "guidance_scale": self.guidance_scale,
            "inference_steps": self.inference_steps,
            "seed": self.seed,
            "cpu_offload": self.cpu_offload,
            "model_backend_var": self.model_backend_var,
            "geometry_model_path_var": self.geometry_model_path_var,
            "geometry_repo_path_var": self.geometry_repo_path_var,
            "geometry_cache_dir_var": self.geometry_cache_dir_var,
            "geometry_decode_chunk_size_var": self.geometry_decode_chunk_size_var,
            "geometry_low_memory_usage_var": self.geometry_low_memory_usage_var,
            "geometry_force_projection_var": self.geometry_force_projection_var,
            "geometry_force_fixed_focal_var": self.geometry_force_fixed_focal_var,
            "geometry_use_extract_interp_var": self.geometry_use_extract_interp_var,
            "stereopilot_model_path_var": self.stereopilot_model_path_var,
            "stereopilot_base_model_path_var": self.stereopilot_base_model_path_var,
            "stereopilot_repo_path_var": self.stereopilot_repo_path_var,
            "stereopilot_cache_dir_var": self.stereopilot_cache_dir_var,
            "stereopilot_prompt_var": self.stereopilot_prompt_var,
            "stereopilot_use_sidecar_prompt_var": self.stereopilot_use_sidecar_prompt_var,
            "stereopilot_output_mode_var": self.stereopilot_output_mode_var,
            "stereopilot_target_width_var": self.stereopilot_target_width_var,
            "stereopilot_target_height_var": self.stereopilot_target_height_var,
            "stereopilot_target_fps_var": self.stereopilot_target_fps_var,
            "stereopilot_window_size_var": self.stereopilot_window_size_var,
            "stereopilot_overlap_var": self.stereopilot_overlap_var,
            "stereopilot_sampling_steps_var": self.stereopilot_sampling_steps_var,
            "stereopilot_guide_scale_var": self.stereopilot_guide_scale_var,
            "stereopilot_shift_var": self.stereopilot_shift_var,
            "stereopilot_domain_label_var": self.stereopilot_domain_label_var,
            "stereopilot_dtype_var": self.stereopilot_dtype_var,
            "stereopilot_transformer_dtype_var": self.stereopilot_transformer_dtype_var,
            "use_cudnn_benchmark": self.use_cudnn_benchmark,
            "process_length": self.process_length,
            "target_fps": self.target_fps,
            "window_size": self.window_size,
            "overlap": self.overlap,
            "process_as_segments_var": self.process_as_segments_var,
            "npz_backfill_missing_only_var": self.npz_backfill_missing_only_var,
            "enable_spatial_refine_mode_var": self.enable_spatial_refine_mode_var,
            "enable_edge_guided_upscale_mode_var": self.enable_edge_guided_upscale_mode_var,
            "enable_cloud_dispatch_mode_var": self.enable_cloud_dispatch_mode_var,
            "spatial_refine_options_expanded_var": self.spatial_refine_options_expanded_var,
            "cloud_options_expanded_var": self.cloud_options_expanded_var,
            "spatial_refine_tile_num_var": self.spatial_refine_tile_num_var,
            "spatial_refine_tile_num_y_var": self.spatial_refine_tile_num_y_var,
            "spatial_refine_tile_overlap_var": self.spatial_refine_tile_overlap_var,
            "spatial_refine_tile_overlap_x_var": self.spatial_refine_tile_overlap_x_var,
            "spatial_refine_tile_overlap_y_var": self.spatial_refine_tile_overlap_y_var,
            "spatial_refine_target_width_var": self.spatial_refine_target_width_var,
            "spatial_refine_target_height_var": self.spatial_refine_target_height_var,
            "spatial_refine_anchor_weight_var": self.spatial_refine_anchor_weight_var,
            "spatial_refine_local_window_size_var": self.spatial_refine_local_window_size_var,
            "spatial_refine_local_window_stride_var": self.spatial_refine_local_window_stride_var,
            "spatial_refine_local_confidence_low_var": self.spatial_refine_local_confidence_low_var,
            "spatial_refine_local_confidence_high_var": self.spatial_refine_local_confidence_high_var,
            "spatial_refine_use_edge_fallback_var": self.spatial_refine_use_edge_fallback_var,
            "spatial_refine_edge_fallback_mix_var": self.spatial_refine_edge_fallback_mix_var,
            "edge_guided_strength_var": self.edge_guided_strength_var,
            "edge_guided_sigma_color_var": self.edge_guided_sigma_color_var,
            "edge_guided_sigma_spatial_var": self.edge_guided_sigma_spatial_var,
            "edge_guided_iterations_var": self.edge_guided_iterations_var,
            "edge_guided_temporal_smooth_var": self.edge_guided_temporal_smooth_var,
            "edge_guided_reinject_strength_var": self.edge_guided_reinject_strength_var,
            "edge_guided_output_suffix_var": self.edge_guided_output_suffix_var,
            "spatial_refine_cleanup_temp_var": self.spatial_refine_cleanup_temp_var,
            "spatial_refine_output_suffix_var": self.spatial_refine_output_suffix_var,
            "cloud_profile_var": self.cloud_profile_var,
            "cloud_target_width_override_var": self.cloud_target_width_override_var,
            "cloud_target_height_override_var": self.cloud_target_height_override_var,
            "cloud_window_size_override_var": self.cloud_window_size_override_var,
            "cloud_overlap_override_var": self.cloud_overlap_override_var,
            "cloud_image_var": self.cloud_image_var,
            "cloud_disk_gb_var": self.cloud_disk_gb_var,
            "cloud_reuse_existing_instance_var": self.cloud_reuse_existing_instance_var,
            "cloud_last_instance_id_var": self.cloud_last_instance_id_var,
            "cloud_last_instance_profile_var": self.cloud_last_instance_profile_var,
            "cloud_last_host_var": self.cloud_last_host_var,
            "cloud_last_port_var": self.cloud_last_port_var,
            "cloud_last_offer_id_var": self.cloud_last_offer_id_var,
            "cloud_last_machine_id_var": self.cloud_last_machine_id_var,
            "cloud_last_host_id_var": self.cloud_last_host_id_var,
            "cloud_remote_user_var": self.cloud_remote_user_var,
            "cloud_remote_root_var": self.cloud_remote_root_var,
            "cloud_remote_venv_var": self.cloud_remote_venv_var,
            "cloud_identity_file_var": self.cloud_identity_file_var,
            "cloud_vast_env_file_var": self.cloud_vast_env_file_var,
            "cloud_hf_env_file_var": self.cloud_hf_env_file_var,
            "cloud_no_hf_env_var": self.cloud_no_hf_env_var,
            "cloud_use_private_registry_login_var": self.cloud_use_private_registry_login_var,
            "cloud_offer_limit_var": self.cloud_offer_limit_var,
            "cloud_require_verified_hosts_var": self.cloud_require_verified_hosts_var,
            "cloud_max_dph_var": self.cloud_max_dph_var,
            "cloud_expected_runtime_hours_var": self.cloud_expected_runtime_hours_var,
            "cloud_expected_upload_gb_var": self.cloud_expected_upload_gb_var,
            "cloud_expected_download_gb_var": self.cloud_expected_download_gb_var,
            "cloud_auto_destroy_instance_var": self.cloud_auto_destroy_instance_var,
            "save_final_output_json_var": self.save_final_output_json_var,
            "merge_output_format_var": self.merge_output_format_var,
            "merge_alignment_method_var": self.merge_alignment_method_var,
            "merge_dither_var": self.merge_dither_var,
            "merge_dither_strength_var": self.merge_dither_strength_var,
            "merge_gamma_correct_var": self.merge_gamma_correct_var,
            "merge_gamma_value_var": self.merge_gamma_value_var,
            "merge_percentile_norm_var": self.merge_percentile_norm_var,
            "merge_norm_low_perc_var": self.merge_norm_low_perc_var,
            "merge_norm_high_perc_var": self.merge_norm_high_perc_var,
            "keep_intermediate_npz_var": self.keep_intermediate_npz_var,
            "min_frames_to_keep_npz_var": self.min_frames_to_keep_npz_var,
            "keep_intermediate_segment_visual_format_var": self.keep_intermediate_segment_visual_format_var,
            "merge_output_suffix_var": self.merge_output_suffix_var,
            "use_local_models_only_var": self.use_local_models_only_var,
            "target_height": self.target_height,
            "target_width": self.target_width,
            "enable_dual_output_robust_norm": self.enable_dual_output_robust_norm,
            "robust_norm_low_percentile": self.robust_norm_low_percentile,
            "robust_norm_high_percentile": self.robust_norm_high_percentile,
            "robust_norm_output_min": self.robust_norm_output_min,
            "robust_norm_output_max": self.robust_norm_output_max,
            "robust_output_suffix": self.robust_output_suffix,
            "is_depth_far_black": self.is_depth_far_black,
            "dark_mode_var": self.dark_mode_var,
            "disable_xformers_var": self.disable_xformers_var,
        }
        self.initial_default_settings = self._collect_all_settings()
        self._help_data = None
        self.spatial_refine_settings_dialog = None
        self.spatial_refine_settings_widgets = []
        self.geometry_settings_dialog = None
        self.geometry_settings_widgets = []
        self.cloud_settings_dialog = None
        self.cloud_settings_widgets = []
        self.cloud_processing_overrides_expanded = False
        self.cloud_processing_overrides_toggle_btn = None
        self.cloud_processing_overrides_frame = None
        self.geometry_settings_toggle_btn = None
        self.active_external_process = None

        self.last_settings_dir = os.getcwd()
        self.message_queue = queue.Queue() # Still needed for progress updates

        # Removed message_catalog setup
        # set_gui_logger_callback(self._queue_message_for_gui_log)
        # set_gui_verbosity(self._get_mapped_gui_verbosity_level()) # Removed
        # mc_configure_timestamps(console=True, gui=False) # Removed

        self.load_config() 
        self.stop_event = threading.Event()
        self.processing_thread = None
        self.secondary_output_widgets_references = []
        self._load_help_content()
        
        self.style = ttk.Style(self.root)
        self._apply_theme(is_startup=True)
        
        # Set initial logging level based on the default value of debug_logging_enabled
        if self.debug_logging_enabled.get():
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
        _logger.info(f"Initial logging level set to {'DEBUG' if self.debug_logging_enabled.get() else 'INFO'}.")
        # --------------------------------------

        self._create_menubar()
        self.create_widgets() 
        self._bind_geometry_status_traces()
        self._refresh_geometry_local_status()
        self._apply_backend_specific_controls()
        self._bind_cloud_processing_summary_traces()
        self._refresh_cloud_processing_summary()
        self.root.app_instance = self 
        # --------------------------------------
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.toggle_merge_related_options_active_state()
        self.toggle_secondary_output_options_active_state()
                
        _logger.debug("DepthCrafter GUI initialized successfully.")

    def _apply_all_settings(self, settings_data: dict):
        normalized_settings = dict(settings_data)
        # Backward compatibility: old settings had one tile grid field; mirror it into Y when missing.
        if (
            "spatial_refine_tile_num_var" in normalized_settings
            and "spatial_refine_tile_num_y_var" not in normalized_settings
        ):
            normalized_settings["spatial_refine_tile_num_y_var"] = normalized_settings["spatial_refine_tile_num_var"]
        if (
            "spatial_refine_tile_overlap_var" in normalized_settings
            and "spatial_refine_tile_overlap_x_var" not in normalized_settings
        ):
            normalized_settings["spatial_refine_tile_overlap_x_var"] = normalized_settings["spatial_refine_tile_overlap_var"]
        if (
            "spatial_refine_tile_overlap_var" in normalized_settings
            and "spatial_refine_tile_overlap_y_var" not in normalized_settings
        ):
            normalized_settings["spatial_refine_tile_overlap_y_var"] = normalized_settings["spatial_refine_tile_overlap_var"]
        if (
            "stereopilot_frame_count_var" in normalized_settings
            and "stereopilot_window_size_var" not in normalized_settings
        ):
            normalized_settings["stereopilot_window_size_var"] = normalized_settings["stereopilot_frame_count_var"]
        normalized_settings.pop("stereopilot_frame_count_var", None)

        for key, value_from_json in normalized_settings.items():
            if key == "target_fps": # Specific debug
                _logger.debug(f"_apply_all_settings: Loading target_fps from JSON. Value: {value_from_json}, Type: {type(value_from_json)}")
            if key in self.all_tk_vars:
                try:
                    self.all_tk_vars[key].set(value_from_json)
                    # After setting, get it back to see what DoubleVar stored
                    if key == "target_fps":
                        val_in_doublevar = self.all_tk_vars[key].get()
                        _logger.debug(f"_apply_all_settings: target_fps in DoubleVar after set: {val_in_doublevar}, Type: {type(val_in_doublevar)}")
                except tk.TclError:
                     _logger.warning(f"Warning: Could not set value for setting '{key}' to '{value_from_json}'. Skipping.")
            else:
                _logger.warning(f"Warning: Unknown setting '{key}' found in settings file. Ignoring.")
        if hasattr(self, 'process_as_segments_var'):
            self.toggle_merge_related_options_active_state()
        self._apply_spatial_refine_options_visibility()
        self._refresh_geometry_local_status()
        self._apply_backend_specific_controls()
        # Removed update GUI verbosity

    def _apply_theme(self, is_startup: bool = False):
        """Applies the selected theme (dark or light) to the GUI."""
        
        if not THEMEDTK_AVAILABLE:
            # ...
            return
        
        # --- Core Theme Application (Must happen before detailed styling) ---
        if self.dark_mode_var.get():
            colors = DARK_MODE_COLORS
        else:
            colors = LIGHT_MODE_COLORS
        
        theme_name = colors["theme_name"]
        self.current_theme_colors = colors
        
        if THEMEDTK_AVAILABLE:
             # Apply the theme first
             self.root.set_theme(theme_name) 

        # --- Detailed TEntry/TCombobox Styling (Apply to CURRENT Theme) ---
        # NOTE: We use style.map() for backgrounds to override theme defaults
        entry_bg = colors["entry_bg"]
        entry_fg = colors["fg"]
        
        # 1. TEntry Styling
        self.style.configure("TEntry", foreground=entry_fg, insertcolor=entry_fg)
        # Use map to force the fieldbackground for the default state (empty tuple)
        self.style.map('TEntry', 
                       fieldbackground=[('', entry_bg)], # '' is the default state
                       foreground=[('', entry_fg)])
        
        # 2. TCombobox Styling
        self.style.configure("TCombobox", foreground=entry_fg) 
        self.style.map('TCombobox', 
                       fieldbackground=[('readonly', entry_bg)], 
                       foreground=[('readonly', entry_fg)])


        # --- Manual Coloring for raw TK Menu ---
        root_bg_color = colors["bg"]
        root_fg_color = colors["fg"]
        menu_active_bg = "#555555" if self.dark_mode_var.get() else "#dddddd"
        menu_active_fg = "white" if self.dark_mode_var.get() else "black"

        self.root.config(bg=root_bg_color)
        
        # NOTE: Since the widgets are now ttk, this is mainly for the root frame and menu
        if hasattr(self, 'menubar'): 
            # Menubar and Menus are raw tk.Menu and need manual color
            self.menubar.config(bg=root_bg_color, fg=root_fg_color, activebackground=menu_active_bg, activeforeground=menu_active_fg)
            if hasattr(self, 'file_menu'): self.file_menu.config(bg=root_bg_color, fg=root_fg_color)
            if hasattr(self, 'help_menu'): self.help_menu.config(bg=root_bg_color, fg=root_fg_color)
           
        self.root.update_idletasks()
    
    def _cleanup_segment_folder(self, segment_subfolder_path, original_basename, master_meta):
        del_folder = False
        if not self.keep_intermediate_npz_var.get():
            _logger.debug(f"Deleting intermediate segment subfolder for {original_basename} (Keep NPZ unchecked)...")
            del_folder = True
        else:
            min_frames = self.min_frames_to_keep_npz_var.get()
            if min_frames > 0:
                orig_frames = master_meta.get("original_video_details", {}).get("raw_frame_count", 0)
                if orig_frames < min_frames:
                    _logger.info(f"  Video frames ({orig_frames}) < threshold ({min_frames}). Deleting segment folder for {original_basename} despite 'Keep NPZ'.")
                    del_folder = True
                else:
                    _logger.info(f"  Video frames ({orig_frames}) >= threshold ({min_frames}). Segment folder for {original_basename} will be kept.")
            else:
                _logger.debug(f"Keeping intermediate NPZ files for {original_basename} (Keep NPZ checked, no positive frame threshold).")
        if del_folder:
            if os.path.exists(segment_subfolder_path):
                try: 
                    shutil.rmtree(segment_subfolder_path)
                    _logger.debug(f"Successfully deleted segment subfolder for {original_basename}.")
                except Exception as e:
                    _logger.error(f"  Error deleting segment subfolder {segment_subfolder_path}: {e}")
            else:
                _logger.warning(f"  Segment subfolder not found for deletion: {segment_subfolder_path}")
        else:
            _logger.debug(f"Keeping intermediate NPZ files and _master_meta.json in {segment_subfolder_path}")

    def _collect_all_settings(self) -> dict:
        settings_data = {}
        for key, tk_var in self.all_tk_vars.items():
            try:
                value = tk_var.get()
                settings_data[key] = value
                if key == "target_fps": # Specific debug for target_fps
                    _logger.debug(f"_collect_all_settings: target_fps raw value: {value}, type: {type(value)}")
            except tk.TclError:
                _logger.warning(f"Warning: Could not get value for setting '{key}'. Skipping.")
        return settings_data

    def _create_menubar(self):
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Load Settings...", command=self._load_all_settings)
        self.file_menu.add_command(label="Save Settings As...", command=self._save_all_settings_as)
        self.file_menu.add_command(label="Reset Settings to Default", command=self._reset_settings_to_defaults)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Restore Finished Input Files...", command=lambda: self._restore_input_files(folder_type="finished"))
        self.file_menu.add_command(label="Restore Failed Input Files...", command=lambda: self._restore_input_files(folder_type="failed"))
        self.file_menu.add_separator()
        self.file_menu.add_checkbutton(label="Use Local Models Only", variable=self.use_local_models_only_var, onvalue=True, offvalue=False)
        self.file_menu.add_checkbutton(label="Disable xFormers (VRAM Save)", variable=self.disable_xformers_var, onvalue=True, offvalue=False)
        if THEMEDTK_AVAILABLE:
            self.file_menu.add_checkbutton(label="Dark Mode", variable=self.dark_mode_var, command=self._apply_theme)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.on_close)

        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_checkbutton(label="Enable Debug Logging", variable=self.debug_logging_enabled, command=self._toggle_debug_logging)
        self.help_menu.add_separator() # Optional separator for clarity
        self.help_menu.add_command(label="GUI Overview", command=lambda: self._show_help_for("general_gui_overview"))
        # -----------------------------------------

    def _determine_input_mode_from_path(self, path_str: str) -> Tuple[str, bool]:
        """
        Analyzes a path string and determines the input mode and if it's a single source.
        Returns: (input_mode_str, is_single_source_bool)
        """
        if not path_str or not os.path.exists(path_str):
            _logger.warning(f"GUI Input: Path '{path_str}' is invalid or does not exist. Cannot determine input mode accurately.")
            return "batch_folder", False

        is_single_source = False
        mode = "batch_folder"

        if os.path.isfile(path_str):
            is_single_source = True
            ext = os.path.splitext(path_str)[1].lower()
            is_video = any(ext in vid_ext.replace("*", "") for vid_ext in self.VIDEO_EXTENSIONS)
            is_image = any(ext in img_ext.replace("*", "") for img_ext in self.IMAGE_EXTENSIONS)

            if is_video:
                mode = "single_video_file"
            elif is_image:
                mode = "single_image_file"
            else:
                _logger.warning(f"GUI Input: Typed path '{path_str}' is a file of unknown type. Treating as non-single source (batch fallback).")
                mode = "batch_folder"
                is_single_source = False
        elif os.path.isdir(path_str):
            if self._is_image_sequence_folder(path_str):
                mode = "image_sequence_folder"
                is_single_source = True
            else:
                mode = "batch_folder"
                is_single_source = False
        else:
            _logger.warning(f"GUI Input: Path '{path_str}' exists but is not a regular file or directory.")
            mode = "batch_folder"
            is_single_source = False
            
        _logger.debug(f"GUI Input: Determined mode for path '{path_str}' as '{mode}', is_single_source: {is_single_source}.")
        return mode, is_single_source

    def _determine_video_paths_and_processing_mode(self, original_basename, master_meta_for_this_vid):
        main_output_dir_for_video = self.output_dir.get()
        was_processed_as_segments = master_meta_for_this_vid["global_processing_settings"]["processed_as_segments"]
        segment_subfolder_path = None
        if was_processed_as_segments:
            segment_subfolder_name = get_segment_output_folder_name(original_basename)
            segment_subfolder_path = os.path.join(main_output_dir_for_video, segment_subfolder_name)
        return main_output_dir_for_video, segment_subfolder_path, was_processed_as_segments

    def _execute_re_merge_wrapper(self, remerge_args_dict):
        try: self._execute_re_merge(remerge_args_dict)
        finally: self.message_queue.put(("set_ui_state", False))

    def _execute_re_merge(self, remerge_args_dict):
        self.stop_event.clear(); self.progress["value"] = 0; self.progress["maximum"] = 1
        start_time = time.perf_counter()
        primary_output_path = "N/A (Merge Failed)" # Initialize to ensure it's always defined
        try:
            if merge_depth_segments:
                primary_output_path = merge_depth_segments.merge_depth_segments(**remerge_args_dict)
                if primary_output_path:
                    _logger.info(f"Re-Merge completed. Primary output saved to: {primary_output_path}")
                else:
                    _logger.warning("Re-Merge completed, but no primary output path was returned.")
            else: 
                _logger.warning("Segment merging for N/A for re-merge action skipped: merge_depth_segments module not available.")
        except Exception as e:
            _logger.exception(f"ERROR during re-merge execution: {e}")
            self.status_message_var.set(f"Re-Merge Error: {e.__class__.__name__}") # Update GUI status
        finally:
            duration = format_duration(time.perf_counter() - start_time)
            _logger.info(f"--- Re-Merge for: {os.path.basename(remerge_args_dict['master_meta_path'])} finished in {duration}. ---")
            # If a primary output path was generated, show it in status for better feedback
            if primary_output_path and primary_output_path != "N/A (Merge Failed)":
                self.status_message_var.set(f"Re-Merge Finished. Output: {os.path.basename(primary_output_path)}")
            else:
                self.status_message_var.set("Re-Merge Finished (No primary output).")
            self.message_queue.put(("progress", 1))

    def _execute_generate_segment_visuals_wrapper(self, gen_visual_args_dict):
        try: self._execute_generate_segment_visuals(gen_visual_args_dict)
        finally: self.message_queue.put(("set_ui_state", False))

    def _execute_generate_segment_visuals(self, gen_visual_args_dict):
        self.stop_event.clear(); self.progress["value"] = 0
        master_path = gen_visual_args_dict["master_meta_path"]
        vis_fmt = gen_visual_args_dict["visual_format_to_generate"]
        start_time = time.perf_counter()
        
        meta_data = load_json_file(master_path) 
        if not meta_data: return 
        
        jobs = [j for j in meta_data.get("jobs_info", []) if j.get("status") == "success" and j.get("output_segment_filename")]
        if not jobs: 
            _logger.warning(f"No successful segments with output filenames found in {os.path.basename(master_path)} for visual generation.")
            return
        self.progress["maximum"] = len(jobs)
        seg_folder_path = os.path.dirname(master_path)
        updated_visual_paths = {}

        for i, job_meta in enumerate(jobs):
            if self.stop_event.is_set(): 
                _logger.info("Segment visual generation cancelled during processing.")
                break
            seg_id, npz_name = job_meta.get("segment_id"), job_meta.get("output_segment_filename")
            npz_path = os.path.join(seg_folder_path, npz_name)
            _logger.debug(f"  Visual Gen - Processing segment {seg_id + 1 if seg_id is not None else '?'}/{len(jobs)}: {npz_name} for {vis_fmt}") 
            
            if not os.path.exists(npz_path): 
                _logger.error(f"File not found: {npz_path}")
                continue
            try:
                with np.load(npz_path) as data:
                    if 'frames' not in data.files: 
                        _logger.error(f"Key 'frames' not found in NPZ: {npz_name}")
                        continue
                    raw_frames = data['frames']
                if raw_frames.size == 0: 
                    _logger.warning(f"    Visual Gen - WARNING: Segment {npz_name} is empty. Skipping.")
                    continue
                
                norm_frames = (raw_frames - raw_frames.min()) / (raw_frames.max() - raw_frames.min()) if raw_frames.max() != raw_frames.min() else np.zeros_like(raw_frames)
                norm_frames = np.clip(norm_frames, 0, 1)
                base_name_no_ext = os.path.splitext(npz_name)[0]
                save_path, save_err = None, None
                fps = float(job_meta.get("processed_at_fps", meta_data.get("original_video_details", {}).get("original_fps", 30.0)))
                if fps <= 0: fps = 30.0

                if vis_fmt == "mp4" or vis_fmt == "main10_mp4":
                    save_path, save_err = save_depth_visual_as_mp4_util(
                        norm_frames, 
                        os.path.join(seg_folder_path, f"{base_name_no_ext}_visual.mp4"),
                        fps,
                        output_format=vis_fmt
                    )
                elif vis_fmt == "png_sequence":
                    save_path, save_err = save_depth_visual_as_png_sequence_util(norm_frames, seg_folder_path, base_name_no_ext)
                elif vis_fmt == "exr_sequence":
                     if OPENEXR_AVAILABLE_GUI: save_path, save_err = save_depth_visual_as_exr_sequence_util(norm_frames, seg_folder_path, base_name_no_ext)
                     else: save_err = "OpenEXR module not available in GUI environment."
                elif vis_fmt == "exr":
                    if OPENEXR_AVAILABLE_GUI:
                        first_frame = norm_frames[0] if len(norm_frames) > 0 else None
                        if first_frame is None: save_err = "No frame data for single EXR."
                        else: save_path, save_err = save_depth_visual_as_single_exr_util(first_frame, seg_folder_path, base_name_no_ext)
                    else: save_err = "OpenEXR module not available in GUI environment."

                if save_path:
                    _logger.debug(f"    Visual Gen - Successfully saved visual: {save_path}") 
                    if seg_id is not None: updated_visual_paths[seg_id] = {"path": os.path.abspath(save_path), "format": vis_fmt}
                if save_err: 
                    _logger.error(f"    Visual Gen - ERROR saving visual for {npz_name}: {save_err}, format requested: {vis_fmt}") 
            except Exception as e:
                _logger.exception(f"    Visual Gen - ERROR processing segment {npz_name}: {e}") 
            self.message_queue.put(("progress", i + 1))
        
        if updated_visual_paths:
            _logger.info("Visual Gen - Updating master metadata with new visual paths...")
            meta_content_update = load_json_file(master_path)
            if meta_content_update:
                updated_count = 0
                for job_entry in meta_content_update.get("jobs_info", []):
                    s_id = job_entry.get("segment_id")
                    if s_id in updated_visual_paths:
                        job_entry["intermediate_visual_path"] = updated_visual_paths[s_id]["path"]
                        job_entry["intermediate_visual_format_saved"] = updated_visual_paths[s_id]["format"]
                        updated_count +=1
                if updated_count > 0:
                    if save_json_file(meta_content_update, master_path, indent=4):
                         _logger.info(f"Visual Gen - Master metadata updated for {updated_count} segments.")
                else: _logger.info("Visual Gen - No segments in master metadata needed visual path updates.")
        
        duration = format_duration(time.perf_counter() - start_time)
        _logger.info(f"--- Segment Visual Generation for: {os.path.basename(master_path)} (Format: {vis_fmt}) finished in {duration}. ---")
        self.message_queue.put(("progress", len(jobs)))

    def _finalize_video_processing(self, current_video_path, original_basename, master_meta_for_this_vid):
        if master_meta_for_this_vid["completed_failed_jobs"] == 0:
            master_meta_for_this_vid["overall_status"] = "all_success"
        elif master_meta_for_this_vid["completed_successful_jobs"] > 0:
            master_meta_for_this_vid["overall_status"] = "partial_success"
        else:
            master_meta_for_this_vid["overall_status"] = "all_failed"

        _logger.debug(f"Finished processing for {original_basename}. Overall Status: {master_meta_for_this_vid['overall_status']}.")
        
        main_output_dir, segment_subfolder_path, was_segments = self._determine_video_paths_and_processing_mode(original_basename, master_meta_for_this_vid)
        master_meta_filepath, meta_saved = None, False
        # --- FIX: Initialize merge_success and final_merged_path BEFORE conditional assignment ---
        merge_success, final_merged_path = False, "N/A (Merge not applicable or failed)"
        
        try:
            master_meta_filepath, meta_saved = self._save_master_metadata_and_cleanup_segment_json(master_meta_for_this_vid, original_basename, main_output_dir, was_segments, segment_subfolder_path)
            
            if was_segments and meta_saved and master_meta_for_this_vid["overall_status"] in ["all_success", "partial_success"]:
                try: # Nested try-except to catch errors specifically from merging
                    merge_success, final_merged_path = self._handle_segment_merging(master_meta_filepath, original_basename, main_output_dir, master_meta_for_this_vid)
                except Exception as e_merge:
                    _logger.error(f"Error during segment merging for {original_basename}: {e_merge}", exc_info=True)
                    self.status_message_var.set(f"Merge Failed: {e_merge.__class__.__name__}")
                    merge_success, final_merged_path = False, f"N/A (Merge failed due to {e_merge.__class__.__name__})"
            elif was_segments:
                # If segments were processed but not merged (e.g., all_failed status, or no successful segments)
                _logger.debug(f"Skipping merge for {original_basename} (status: {master_meta_for_this_vid['overall_status']}, meta_saved: {meta_saved}). Segments remain in {segment_subfolder_path or 'N/A'}")
                # No change to merge_success/final_merged_path as they were initialized to False/N/A
            
            if self.save_final_output_json_var.get():
                self._save_final_output_sidecar_json(original_basename, final_merged_path, master_meta_filepath, master_meta_for_this_vid, was_segments, merge_success)
            
            if was_segments and segment_subfolder_path:
                self._cleanup_segment_folder(segment_subfolder_path, original_basename, master_meta_for_this_vid)
        except Exception as e:
            _logger.exception(f"Error during finalization for {original_basename}: {e}")
            self.status_message_var.set(f"Finalization Error: {e.__class__.__name__} for {original_basename}")

        final_status = master_meta_for_this_vid.get("overall_status", "all_failed")

        if self.effective_move_original_on_completion:
            target_subfolder_name = ""
            if final_status == "all_success":
                target_subfolder_name = "finished"
            elif final_status in ["partial_success", "all_failed"]:
                target_subfolder_name = "failed"
            else:
                _logger.warning(f"Move Original: Could not determine 'finished' or 'failed' status for '{original_basename}' (status: '{final_status}'). Original file will not be moved.")

            if target_subfolder_name:
                self._move_original_source(current_video_path, original_basename, target_subfolder_name)
        else:
            _logger.info(f"Skipped moving original source '{original_basename}' (single file/sequence mode).")

    def _get_segments_to_resume_or_overwrite(self, vid_path, original_basename, 
                                             segment_subfolder_path, all_potential_segments_from_define,
                                             base_job_info_for_video_ref: dict):
        master_meta_path = os.path.join(segment_subfolder_path, f"{original_basename}_master_meta.json")
        base_job_info_for_video_ref["pre_existing_successful_jobs"] = []

        if os.path.exists(master_meta_path):
            msg_dialog = (f"Master metadata found for '{original_basename}'. This video was previously processed/finalized.\n"
                          f"Path: {master_meta_path}\n\n"
                          f"Do you want to:\n"
                          f"- 'Yes': Re-process only FAILED segments and update master metadata?\n"
                          f"         (Existing successful segments will be preserved in the new master metadata).\n"
                          f"- 'No': Delete ALL existing segments and master metadata and start fresh?\n"
                          f"- 'Cancel': Skip this video entirely?")
            choice = messagebox.askyesnocancel("Resume or Overwrite Finalized Segments?", msg_dialog, parent=self.root)

            if choice is True:
                _logger.info(f"Attempting to re-process failed segments for {original_basename} based on existing master metadata.")
                master_data = load_json_file(master_meta_path)
                if not master_data or "jobs_info" not in master_data:
                    _logger.warning(f"Could not load master metadata or 'jobs_info' missing for {original_basename}. Defaulting to overwrite.")
                    choice = False # Fallthrough
                else:
                    failed_segment_jobs_to_run = []
                    successful_jobs_from_old_master = []
                    potential_segments_dict = {seg_job['segment_id']: seg_job for seg_job in all_potential_segments_from_define}

                    for job_in_meta in master_data.get("jobs_info", []):
                        seg_id = job_in_meta.get("segment_id")
                        if job_in_meta.get("status") == "success":
                            successful_jobs_from_old_master.append(job_in_meta)
                        elif seg_id is not None and seg_id in potential_segments_dict:
                            failed_segment_jobs_to_run.append(potential_segments_dict[seg_id])
                            _logger.debug(f"  Queueing segment ID {seg_id} (status: {job_in_meta.get('status', 'unknown')}) for {original_basename} for re-processing.")
                        else:
                            _logger.warning(f"  Warning: Segment (ID: {seg_id}, Status: {job_in_meta.get('status')}) from master_meta for {original_basename} not re-queueable. It will be ignored.")
                    
                    if not failed_segment_jobs_to_run:
                        _logger.info(f"No re-processable failed segments found in master_meta for {original_basename}. All existing successful segments will be preserved if merging.")
                        base_job_info_for_video_ref["pre_existing_successful_jobs"] = successful_jobs_from_old_master
                        return [], "skipped_no_failed_segments_in_master_for_reprocessing"
                    
                    try:
                        backup_master_meta_path = master_meta_path + f".backup_{time.strftime('%Y%m%d%H%M%S')}"
                        shutil.move(master_meta_path, backup_master_meta_path)
                        _logger.debug(f"Backed up existing file {os.path.basename(master_meta_path)} to: {os.path.basename(backup_master_meta_path)}")
                    except Exception as e:
                        _logger.warning(f"  Warning: Could not back up existing master metadata: {e}. It might be overwritten.")

                    base_job_info_for_video_ref["pre_existing_successful_jobs"] = successful_jobs_from_old_master
                    return failed_segment_jobs_to_run, "reprocessing_failed_from_master"
            
            if choice is False: 
                _logger.info(f"User chose/defaulted to delete existing segment folder and start fresh for {original_basename}: {segment_subfolder_path}")
                try:
                    if os.path.exists(segment_subfolder_path): shutil.rmtree(segment_subfolder_path)
                    _logger.debug(f"  Successfully deleted: {segment_subfolder_path}")
                except Exception as e:
                    _logger.error(f"  Error deleting {segment_subfolder_path}: {e}. Processing may fail or overwrite.")
                return all_potential_segments_from_define, "overwriting_finalized"
            
            else: # Cancel
                _logger.info(f"Skipping {original_basename} (user chose to cancel on finalized segments).")
                return [], "skipped_finalized"

        elif os.path.exists(segment_subfolder_path):
            msg_dialog_incomplete = (f"Incomplete segment data found for '{original_basename}' (no master metadata file).\n"
                                     f"Path: {segment_subfolder_path}\n\n"
                                     f"Do you want to:\n"
                                     f"- 'Yes': Resume by processing only missing/failed segments?\n"
                                     f"         (Existing successful segments will be preserved).\n"
                                     f"- 'No': Delete existing incomplete segments and start fresh?\n"
                                     f"- 'Cancel': Skip this video entirely?")
            choice_incomplete = messagebox.askyesnocancel("Resume Incomplete Segments?", msg_dialog_incomplete, parent=self.root)

            if choice_incomplete is True:
                _logger.debug(f"Attempting to resume incomplete segments for {original_basename}.")
                segments_to_run = []
                num_already_complete = 0
                completed_segment_metadata_from_json = []

                for potential_segment_job in all_potential_segments_from_define:
                    seg_id = potential_segment_job["segment_id"]
                    total_segs = potential_segment_job["total_segments"]
                    expected_npz_filename = get_segment_npz_output_filename(original_basename, seg_id, total_segs)
                    expected_json_filename = get_sidecar_json_filename(expected_npz_filename)
                    npz_path = os.path.join(segment_subfolder_path, expected_npz_filename)
                    json_path = os.path.join(segment_subfolder_path, expected_json_filename)

                    is_complete_and_successful = False
                    if os.path.exists(npz_path) and os.path.exists(json_path):
                        segment_meta = load_json_file(json_path)
                        if segment_meta and segment_meta.get("status") == "success":
                            is_complete_and_successful = True
                            num_already_complete += 1
                            completed_segment_metadata_from_json.append(segment_meta)
                        else:
                            status_msg = segment_meta.get('status', 'unknown') if segment_meta else 'JSON missing/corrupt'
                            _logger.info(f"  Segment {seg_id+1}/{total_segs} for {original_basename} found but not successful (status: {status_msg}). Will re-process.")
                    else:
                        _logger.debug(f"  Segment {seg_id+1}/{total_segs} for {original_basename} (NPZ: {expected_npz_filename}) not found or JSON missing. Will process.")

                    if not is_complete_and_successful:
                        segments_to_run.append(potential_segment_job)
                
                if num_already_complete > 0:
                    _logger.info(f"Found {num_already_complete} successfully completed segments for {original_basename} that will be skipped during processing.")
                
                base_job_info_for_video_ref["pre_existing_successful_jobs"] = completed_segment_metadata_from_json

                if not segments_to_run and num_already_complete == len(all_potential_segments_from_define):
                    _logger.warning(f"  All segments for {original_basename} appear complete from individual files, but master_meta was missing. Consider re-merging. Skipping processing.")
                    return [], "skipped_all_segments_found_complete_no_master"
                elif not segments_to_run and num_already_complete < len(all_potential_segments_from_define):
                     _logger.warning(f"  No segments to run for {original_basename}, but not all were found complete. Total defined: {len(all_potential_segments_from_define)}, Found complete: {num_already_complete}")
                     return [], "skipped_no_segments_to_run_incomplete"
                return segments_to_run, "resuming_incomplete"

            elif choice_incomplete is False:
                _logger.info(f"User chose to delete existing incomplete segment folder and start fresh for {original_basename}: {segment_subfolder_path}")
                try:
                    if os.path.exists(segment_subfolder_path): shutil.rmtree(segment_subfolder_path)
                    _logger.debug(f"  Successfully deleted: {segment_subfolder_path}")
                except Exception as e:
                    _logger.error(f"  Error deleting {segment_subfolder_path}: {e}. Processing may fail or overwrite.")
                return all_potential_segments_from_define, "overwriting_incomplete"
            
            else: # Cancel
                _logger.info(f"Skipping {original_basename} (user chose to cancel on incomplete segments).")
                return [], "skipped_incomplete"
                
        else: # Segment folder does not exist
            return all_potential_segments_from_define, "fresh_processing"

    def _get_missing_segments_npz_only(
        self,
        original_basename: str,
        segment_subfolder_path: str,
        all_potential_segments_from_define: list,
    ) -> list:
        """
        Isolated special mode:
        Return only segment jobs whose expected NPZ file is missing.
        """
        missing_segments = []
        for potential_segment_job in all_potential_segments_from_define:
            seg_id = int(potential_segment_job["segment_id"])
            total_segs = int(potential_segment_job["total_segments"])
            expected_npz_filename = get_segment_npz_output_filename(original_basename, seg_id, total_segs)
            expected_npz_path = os.path.join(segment_subfolder_path, expected_npz_filename)
            if not os.path.exists(expected_npz_path):
                missing_segments.append(potential_segment_job)

        missing_count = len(missing_segments)
        total_count = len(all_potential_segments_from_define)
        if missing_count > 0:
            _logger.info(
                f"NPZ Backfill Mode: {original_basename} has {missing_count}/{total_count} missing NPZ segments. "
                "Only missing segments will be processed."
            )
        else:
            _logger.info(f"NPZ Backfill Mode: {original_basename} has no missing NPZ segments. Skipping.")

        return missing_segments

    def _handle_segment_merging(self, master_meta_filepath, original_basename, main_output_dir, master_meta) -> Tuple[bool, str]:
        """
        Handles the merging of segments, potentially generating a second robustly normalized output.
        Returns a tuple: (bool indicating merge success, str path of the primary merged output).
        """
        if not merge_depth_segments:
            _logger.warning(f"Segment merging for {original_basename} skipped: merge_depth_segments module not available.")
            return False, "N/A (Merge module not available - module missing)"
        
        out_fmt = self.merge_output_format_var.get()
        output_suffix = self.merge_output_suffix_var.get()
        merged_base_name = f"{original_basename}{output_suffix}"

        align_method = "linear_blend" if self.merge_alignment_method_var.get() == "Linear Blend" else "shift_scale"
        
        enable_dual_output = self.enable_dual_output_robust_norm.get() 
        robust_low_perc = self.robust_norm_low_percentile.get()
        robust_high_perc = self.robust_norm_high_percentile.get()
        robust_output_min = self.robust_norm_output_min.get()
        robust_output_max = self.robust_norm_output_max.get()
        robust_output_suffix_val = self.robust_output_suffix.get()
        is_depth_far_black_val = self.is_depth_far_black.get()

        try:
            primary_output_path = merge_depth_segments.merge_depth_segments(
                master_meta_path=master_meta_filepath, 
                output_path_arg=main_output_dir,
                do_dithering=self.merge_dither_var.get(), 
                dither_strength_factor=self.merge_dither_strength_var.get(),
                apply_gamma_correction=self.merge_gamma_correct_var.get(), 
                gamma_value=self.merge_gamma_value_var.get(),
                use_percentile_norm=self.merge_percentile_norm_var.get(), 
                norm_low_percentile=self.merge_norm_low_perc_var.get(),
                norm_high_percentile=self.merge_norm_high_perc_var.get(), 
                output_format=out_fmt,
                merge_alignment_method=align_method, 
                output_filename_override_base=merged_base_name,
                enable_dual_output_robust_norm=enable_dual_output,
                robust_norm_low_percentile=robust_low_perc,
                robust_norm_high_percentile=robust_high_perc,
                robust_norm_output_min=robust_output_min,
                robust_norm_output_max=robust_output_max,
                robust_output_suffix=robust_output_suffix_val,
                is_depth_far_black=is_depth_far_black_val
            )
            
            # If primary_output_path is None, the merge failed or didn't produce a path
            if primary_output_path is None:
                _logger.error(f"merge_depth_segments returned None for {original_basename}. Merge considered failed.")
                return False, f"N/A (Merge module returned no path)"
            else:
                _logger.debug(f"Primary merge for {original_basename} successful. Output: {primary_output_path}")
                return True, primary_output_path # Successful merge
                
        except Exception as e: 
            _logger.exception(f"Exception during merge_depth_segments call for {original_basename}: {e}")
            self.status_message_var.set(f"Merge Error: {e.__class__.__name__} for {original_basename}")
            return False, f"N/A (Merge failed due to {e.__class__.__name__})"
        
    def _initialize_master_metadata_entry(self, original_basename, job_info_for_original_details, total_expected_jobs_for_this_video):
        entry = {
            "original_video_basename": original_basename,
            "original_video_details": {
                "raw_frame_count": job_info_for_original_details.get("original_video_raw_frame_count", 0),
                "original_fps": job_info_for_original_details.get("original_video_fps", 30.0)
            },
            "global_processing_settings": {
                "guidance_scale": self.guidance_scale.get(),
                "inference_steps": self.inference_steps.get(),
                "target_height_setting": self.target_height.get(),
                "target_width_setting": self.target_width.get(),
                "seed_setting": self.seed.get(),
                "target_fps_setting": self.target_fps.get(),
                "process_max_frames_setting": self.process_length.get(),
                "gui_window_size_setting": self.window_size.get(),
                "gui_overlap_setting": self.overlap.get(),
                "processed_as_segments": self.process_as_segments_var.get(),
                "npz_backfill_missing_only_mode": self.npz_backfill_missing_only_var.get(),
                "model_backend": self.model_backend_var.get(),
                "geometry_model_path": self.geometry_model_path_var.get(),
                "geometry_repo_path": self.geometry_repo_path_var.get(),
                "geometry_cache_dir": self.geometry_cache_dir_var.get(),
                "geometry_decode_chunk_size": self.geometry_decode_chunk_size_var.get(),
                "geometry_low_memory_usage": self.geometry_low_memory_usage_var.get(),
                "geometry_force_projection": self.geometry_force_projection_var.get(),
                "geometry_force_fixed_focal": self.geometry_force_fixed_focal_var.get(),
                "geometry_use_extract_interp": self.geometry_use_extract_interp_var.get(),
                "stereopilot_model_path": self.stereopilot_model_path_var.get(),
                "stereopilot_base_model_path": self.stereopilot_base_model_path_var.get(),
                "stereopilot_repo_path": self.stereopilot_repo_path_var.get(),
                "stereopilot_cache_dir": self.stereopilot_cache_dir_var.get(),
                "stereopilot_prompt": self.stereopilot_prompt_var.get(),
                "stereopilot_use_sidecar_prompt": self.stereopilot_use_sidecar_prompt_var.get(),
                "stereopilot_output_mode": self.stereopilot_output_mode_var.get(),
                "stereopilot_target_width": self.stereopilot_target_width_var.get(),
                "stereopilot_target_height": self.stereopilot_target_height_var.get(),
                "stereopilot_target_fps": self.stereopilot_target_fps_var.get(),
                "stereopilot_window_size": self.stereopilot_window_size_var.get(),
                "stereopilot_overlap": self.stereopilot_overlap_var.get(),
                "stereopilot_sampling_steps": self.stereopilot_sampling_steps_var.get(),
                "stereopilot_guide_scale": self.stereopilot_guide_scale_var.get(),
                "stereopilot_shift": self.stereopilot_shift_var.get(),
                "stereopilot_domain_label": self.stereopilot_domain_label_var.get(),
                "stereopilot_dtype": self.stereopilot_dtype_var.get(),
                "stereopilot_transformer_dtype": self.stereopilot_transformer_dtype_var.get(),
            },
            "jobs_info": [], "overall_status": "pending",
            "total_expected_jobs": total_expected_jobs_for_this_video,
            "completed_successful_jobs": 0, "completed_failed_jobs": 0,
        }
        if self.process_as_segments_var.get():
            entry["global_processing_settings"]["segment_definition_output_window_frames"] = job_info_for_original_details.get("gui_desired_output_window_frames", self.window_size.get())
            entry["global_processing_settings"]["segment_definition_output_overlap_frames"] = job_info_for_original_details.get("gui_desired_output_overlap_frames", self.overlap.get())
        return entry

    def _is_image_sequence_folder(self, folder_path: str) -> bool:
        """Rudimentary check if a folder looks like an image sequence."""
        if not os.path.isdir(folder_path): return False
        
        image_files_count = 0
        video_files_count = 0
        sub_dirs_count = 0

        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                sub_dirs_count += 1
                continue
            
            ext = os.path.splitext(item)[1].lower()
            if any(ext in img_ext.replace("*", "") for img_ext in self.IMAGE_EXTENSIONS):
                image_files_count +=1
            elif any(ext in vid_ext.replace("*", "") for vid_ext in self.VIDEO_EXTENSIONS):
                video_files_count +=1
        
        return image_files_count > 5 and video_files_count == 0 and sub_dirs_count == 0

    def _load_help_content(self):
        if self._help_data is None: # Only load once
            raw_data = load_json_file(DepthCrafterGUI.HELP_CONTENT_FILENAME)
            if raw_data:
                self._help_data = raw_data # Keep raw data for future potential uses
                # Populate the module-level _HELP_TEXTS dictionary for tooltips
                global _HELP_TEXTS
                _HELP_TEXTS.clear() # Clear existing in case of reload (e.g., in future)
                for key, content in raw_data.items():
                    if "text" in content: # Ensure 'text' key exists
                        _HELP_TEXTS[key] = content["text"]
            else:
                self._help_data = {} # Indicate loading failed
                _logger.warning(f"Warning: Could not load help content from {DepthCrafterGUI.HELP_CONTENT_FILENAME}. Tooltips will be limited/show 'not found'.")
        return self._help_data

    def _load_all_settings(self):
        filepath = filedialog.askopenfilename(title="Load Settings File", filetypes=self.SETTINGS_FILETYPES, initialdir=self.last_settings_dir)
        if not filepath:
            _logger.info("Load settings cancelled by user.")
            return
        self.last_settings_dir = os.path.dirname(filepath)
        settings_data = load_json_file(filepath)
        if settings_data:
            self._apply_all_settings(settings_data)
            _logger.debug(f"Successfully loaded settings from: {filepath}")
        else:
            messagebox.showerror("Load Error", f"Could not load settings from:\n{filepath}\nSee console log for details.")

    def _move_original_source(self, current_video_path: str, original_basename: str, target_subfolder: str):
        _logger.debug(f"Moving original source '{original_basename}' to '{target_subfolder}' folder.")
        try:
            path_from_gui_input_field = self.input_dir_or_file_var.get()

            actual_input_root_for_target_folder: str
            if os.path.isdir(path_from_gui_input_field):
                actual_input_root_for_target_folder = path_from_gui_input_field
            elif os.path.isfile(path_from_gui_input_field):
                actual_input_root_for_target_folder = os.path.dirname(path_from_gui_input_field)
            else:
                _logger.warning(f"Move Original: The GUI input path '{path_from_gui_input_field}' is invalid for determining the target folder root. Using dirname of processed item as fallback.")
                actual_input_root_for_target_folder = os.path.dirname(current_video_path)
                if not os.path.isdir(actual_input_root_for_target_folder):
                    _logger.error(f"Move Original: Cannot determine a valid root directory for target folder based on input path '{current_video_path}'.")
                    _logger.error(f"ERROR moving original '{original_basename}': Cannot determine valid root for target folder.")
                    return

            destination_dir = os.path.join(actual_input_root_for_target_folder, target_subfolder)
            os.makedirs(destination_dir, exist_ok=True)
            
            dest_filename = os.path.basename(current_video_path)
            dest_path = os.path.join(destination_dir, dest_filename)

            if os.path.exists(current_video_path):
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(dest_filename) 
                    new_dest_name = f"{base}{time.strftime('_%Y%m%d%H%M%S')}{ext}"
                    dest_path = os.path.join(destination_dir, new_dest_name)
                    _logger.info(f"Move Original: Destination already exists. Renaming '{dest_filename}' to '{new_dest_name}'.")
                
                shutil.move(current_video_path, dest_path)
                _logger.debug(f"Successfully moved original source '{dest_filename}' to '{target_subfolder}' folder.")
            else:
                _logger.warning(f"Move Original: Source path to move does not exist: {current_video_path}")
        except Exception as e:
            _logger.exception(f"ERROR moving original '{original_basename}': {e}")

    def _process_spatial_refine_source(self, demo, source_spec: dict, effective_seed_for_run: int) -> Tuple[bool, str]:
        """
        Runs the secondary hi-res spatial refine mode for a single source.
        Returns (success, output_path_or_reason).
        """
        if run_spatial_hires_refine is None:
            return False, "spatial_refine module unavailable"

        current_video_path = source_spec["path"]
        original_basename = source_spec["basename"]
        source_mode = source_spec["type"]

        if source_mode not in ["video_file", "single_video_file"]:
            msg = f"unsupported source mode '{source_mode}' (video files only)"
            _logger.warning(f"Skipping hi-res spatial refine for {original_basename}: {msg}")
            return False, msg

        lowres_segment_folder = os.path.join(
            self.output_dir.get(),
            get_segment_output_folder_name(original_basename),
        )
        lowres_master_meta_path = os.path.join(
            lowres_segment_folder,
            f"{original_basename}_master_meta.json",
        )
        if not os.path.isdir(lowres_segment_folder):
            msg = f"missing low-res segment cache folder: {lowres_segment_folder}"
            _logger.warning(f"Skipping hi-res spatial refine for {original_basename}: {msg}")
            return False, msg
        if not os.path.exists(lowres_master_meta_path):
            _logger.warning(
                f"Spatial refine for {original_basename}: missing _master_meta.json. "
                f"Attempting legacy NPZ fallback from {lowres_segment_folder}."
            )

        target_w = int(self.spatial_refine_target_width_var.get())
        target_h = int(self.spatial_refine_target_height_var.get())
        tile_num_x = int(self.spatial_refine_tile_num_var.get())
        tile_num_y = int(self.spatial_refine_tile_num_y_var.get())
        tile_overlap_x = int(self.spatial_refine_tile_overlap_x_var.get())
        tile_overlap_y = int(self.spatial_refine_tile_overlap_y_var.get())
        anchor_weight = float(self.spatial_refine_anchor_weight_var.get())
        local_window_size = int(self.spatial_refine_local_window_size_var.get())
        local_window_stride = int(self.spatial_refine_local_window_stride_var.get())
        local_conf_low = float(self.spatial_refine_local_confidence_low_var.get())
        local_conf_high = float(self.spatial_refine_local_confidence_high_var.get())
        edge_fallback_enabled = bool(self.spatial_refine_use_edge_fallback_var.get())
        edge_fallback_mix = float(self.spatial_refine_edge_fallback_mix_var.get())
        edge_strength = float(self.edge_guided_strength_var.get())
        edge_sigma_color = float(self.edge_guided_sigma_color_var.get())
        edge_sigma_spatial = float(self.edge_guided_sigma_spatial_var.get())
        edge_iters = int(self.edge_guided_iterations_var.get())
        edge_temporal_smooth = float(self.edge_guided_temporal_smooth_var.get())
        edge_reinject_strength = float(self.edge_guided_reinject_strength_var.get())
        output_suffix = str(self.spatial_refine_output_suffix_var.get()).strip() or "_hires_refined_depth"

        output_path, summary = run_spatial_hires_refine(
            demo=demo,
            source_video_path=current_video_path,
            original_basename=original_basename,
            output_dir=self.output_dir.get(),
            lowres_master_meta_path=lowres_master_meta_path,
            lowres_segment_folder=lowres_segment_folder,
            target_height=target_h,
            target_width=target_w,
            tile_num_x=tile_num_x,
            tile_num_y=tile_num_y,
            tile_overlap_x_px=tile_overlap_x,
            tile_overlap_y_px=tile_overlap_y,
            temporal_window_frames=int(self.window_size.get()),
            temporal_overlap_frames=int(self.overlap.get()),
            guidance_scale=float(self.guidance_scale.get()),
            inference_steps=int(self.inference_steps.get()),
            seed=int(effective_seed_for_run),
            target_fps_setting=float(self.target_fps.get()),
            lowres_anchor_weight=anchor_weight,
            output_suffix=output_suffix,
            output_format="main10_mp4",
            cleanup_temp=bool(self.spatial_refine_cleanup_temp_var.get()),
            temporal_merge_alignment="shift_scale",
            allow_legacy_npz_fallback=True,
            local_reliability_window_size=local_window_size,
            local_reliability_window_stride=local_window_stride,
            local_reliability_score_confidence_low=local_conf_low,
            local_reliability_score_confidence_high=local_conf_high,
            edge_guided_fallback_enabled=edge_fallback_enabled,
            edge_guided_fallback_mix=edge_fallback_mix,
            edge_guided_strength=edge_strength,
            edge_guided_sigma_color=edge_sigma_color,
            edge_guided_sigma_spatial=edge_sigma_spatial,
            edge_guided_bilateral_iterations=edge_iters,
            edge_guided_temporal_smooth=edge_temporal_smooth,
            edge_guided_reinject_strength=edge_reinject_strength,
        )

        self.current_resolution_var.set(f"{summary['processed_width']}x{summary['processed_height']}")
        self.current_frames_var.set(str(summary["frames"]))
        return True, output_path

    def _process_edge_guided_upscale_source(self, source_spec: dict) -> Tuple[bool, str]:
        """
        Runs the standalone edge-guided hi-res upscale mode for a single source.
        Returns (success, output_path_or_reason).
        """
        if run_edge_guided_hires_upscale is None:
            return False, "edge_guided_upscale module unavailable"

        current_video_path = source_spec["path"]
        original_basename = source_spec["basename"]
        source_mode = source_spec["type"]

        if source_mode not in ["video_file", "single_video_file"]:
            msg = f"unsupported source mode '{source_mode}' (video files only)"
            _logger.warning(f"Skipping edge-guided upscale for {original_basename}: {msg}")
            return False, msg

        lowres_segment_folder = os.path.join(
            self.output_dir.get(),
            get_segment_output_folder_name(original_basename),
        )
        lowres_master_meta_path = os.path.join(
            lowres_segment_folder,
            f"{original_basename}_master_meta.json",
        )
        if not os.path.isdir(lowres_segment_folder):
            msg = f"missing low-res segment cache folder: {lowres_segment_folder}"
            _logger.warning(f"Skipping edge-guided upscale for {original_basename}: {msg}")
            return False, msg
        if not os.path.exists(lowres_master_meta_path):
            _logger.warning(
                f"Edge-guided mode for {original_basename}: missing _master_meta.json. "
                f"Attempting legacy NPZ fallback from {lowres_segment_folder}."
            )

        target_w = int(self.spatial_refine_target_width_var.get())
        target_h = int(self.spatial_refine_target_height_var.get())
        output_suffix = str(self.edge_guided_output_suffix_var.get()).strip() or "_edge_hires_depth"

        output_path, summary = run_edge_guided_hires_upscale(
            source_video_path=current_video_path,
            original_basename=original_basename,
            output_dir=self.output_dir.get(),
            lowres_master_meta_path=lowres_master_meta_path,
            lowres_segment_folder=lowres_segment_folder,
            target_height=target_h,
            target_width=target_w,
            temporal_overlap_frames=int(self.overlap.get()),
            target_fps_setting=float(self.target_fps.get()),
            output_suffix=output_suffix,
            output_format="main10_mp4",
            allow_legacy_npz_fallback=True,
            temporal_merge_alignment="shift_scale",
            edge_strength=float(self.edge_guided_strength_var.get()),
            sigma_color=float(self.edge_guided_sigma_color_var.get()),
            sigma_spatial=float(self.edge_guided_sigma_spatial_var.get()),
            bilateral_iterations=int(self.edge_guided_iterations_var.get()),
            temporal_smooth=float(self.edge_guided_temporal_smooth_var.get()),
            edge_reinject_strength=float(self.edge_guided_reinject_strength_var.get()),
        )

        self.current_resolution_var.set(f"{summary['processed_width']}x{summary['processed_height']}")
        self.current_frames_var.set(str(summary["frames"]))
        return True, output_path

    def _process_single_job(self, demo, job_info, master_meta_for_this_vid):
        
        returned_job_specific_metadata = {}
        job_successful = False
        is_segment_job = job_info.get("is_segment", False)
        original_basename = job_info["original_basename"]
        
        snapshotted_settings = master_meta_for_this_vid["global_processing_settings"]
        guidance_scale_for_job = snapshotted_settings["guidance_scale"]
        inference_steps_for_job = snapshotted_settings["inference_steps"]
        seed_for_job = snapshotted_settings["seed_setting"]
        process_length_for_run_param = snapshotted_settings["process_max_frames_setting"] if not is_segment_job else -1
        selected_backend = str(snapshotted_settings.get("model_backend", "depthcrafter") or "depthcrafter").strip().lower()
        if selected_backend == "stereopilot":
            window_size_for_pipe_call = max(
                1,
                int(
                    snapshotted_settings.get(
                        "stereopilot_window_size",
                        snapshotted_settings.get(
                            "stereopilot_frame_count",
                            self.stereopilot_window_size_var.get(),
                        ),
                    )
                ),
            )
            overlap_for_pipe_call = max(
                0,
                int(
                    snapshotted_settings.get(
                        "stereopilot_overlap",
                        self.stereopilot_overlap_var.get(),
                    )
                ),
            )
        else:
            window_size_for_pipe_call = snapshotted_settings["gui_window_size_setting"]
            overlap_for_pipe_call = snapshotted_settings["gui_overlap_setting"]

        try:
            keep_npz_for_this_job_run = False
            if is_segment_job:
                if self.keep_intermediate_npz_var.get():
                    min_frames_thresh = self.min_frames_to_keep_npz_var.get()
                    orig_vid_frame_count = job_info.get("original_video_raw_frame_count", 0)
                    if min_frames_thresh <= 0 or orig_vid_frame_count >= min_frames_thresh:
                        keep_npz_for_this_job_run = True
            
            saved_data_filepath, returned_job_specific_metadata = demo.run(
                video_path_or_frames_or_info=job_info,
                num_denoising_steps=inference_steps_for_job, 
                guidance_scale=guidance_scale_for_job,
                base_output_folder=self.output_dir.get(), 
                gui_window_size=window_size_for_pipe_call,
                gui_overlap=overlap_for_pipe_call, 
                process_length_for_read_full_video=process_length_for_run_param, 
                target_height=self.target_height.get(),
                target_width=self.target_width.get(),
                seed=seed_for_job, 
                original_video_basename_override=original_basename,
                segment_job_info_param=job_info if is_segment_job else None,
                keep_intermediate_npz_config=keep_npz_for_this_job_run,
                intermediate_segment_visual_format_config=self.keep_intermediate_segment_visual_format_var.get(),
                save_final_json_for_this_job_config=self.save_final_output_json_var.get()
            )
            if not returned_job_specific_metadata:
                returned_job_specific_metadata = {"status": "failure_no_metadata_from_run"}
                _logger.warning(f"Warning: No job-specific metadata returned from run for {original_basename}.")
            
            if saved_data_filepath and returned_job_specific_metadata.get("status") == "success":
                job_successful = True
            else:
                log_msg_prefix_local = f"Segment {job_info.get('segment_id', -1)+1}/{job_info.get('total_segments', 0)}" if is_segment_job else "Full video"
                _logger.info(f"  Job for {original_basename} ({log_msg_prefix_local}) status: {returned_job_specific_metadata.get('status', 'unknown_status')}")
        
        except Exception as e:
            if not returned_job_specific_metadata: returned_job_specific_metadata = {}
            returned_job_specific_metadata["status"] = "exception_in_gui_process_single_job"
            returned_job_specific_metadata["error_message"] = str(e)
            log_msg_prefix_local = f"Segment {job_info.get('segment_id', -1)+1}/{job_info.get('total_segments', 0)}" if is_segment_job else "Full video"
            _logger.exception(f"  Exception during job for {original_basename} ({log_msg_prefix_local}): {e}")
            self.status_message_var.set(f"Error: {e.__class__.__name__} during {original_basename}")
        return job_successful, returned_job_specific_metadata

    def _reset_gpu_peak_tracking_for_clip(self, clip_label: str) -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            device_count = int(torch.cuda.device_count())
        except Exception:
            return False
        if device_count <= 0:
            return False

        reset_count = 0
        for idx in range(device_count):
            try:
                torch.cuda.reset_peak_memory_stats(idx)
                reset_count += 1
            except Exception:
                continue

        if reset_count <= 0:
            return False

        _logger.info(f"{clip_label}: GPU peak VRAM tracking reset for {reset_count} CUDA device(s).")
        return True

    def _log_gpu_peak_tracking_summary_for_clip(self, clip_label: str):
        if not torch.cuda.is_available():
            return
        try:
            device_count = int(torch.cuda.device_count())
        except Exception:
            return
        if device_count <= 0:
            return

        max_peak_alloc_mib = 0.0
        max_peak_reserved_mib = 0.0
        for idx in range(device_count):
            try:
                peak_alloc_mib = float(torch.cuda.max_memory_allocated(idx)) / (1024.0 ** 2)
            except Exception:
                peak_alloc_mib = 0.0
            try:
                peak_reserved_mib = float(torch.cuda.max_memory_reserved(idx)) / (1024.0 ** 2)
            except Exception:
                peak_reserved_mib = 0.0

            max_peak_alloc_mib = max(max_peak_alloc_mib, peak_alloc_mib)
            max_peak_reserved_mib = max(max_peak_reserved_mib, peak_reserved_mib)

        _logger.info(
            f"{clip_label}: GPU peak VRAM summary | alloc_max={max_peak_alloc_mib:.1f} MiB, "
            f"reserved_max={max_peak_reserved_mib:.1f} MiB, devices={device_count}"
        )

        for idx in range(device_count):
            try:
                device_name = str(torch.cuda.get_device_name(idx))
            except Exception:
                device_name = f"cuda:{idx}"
            try:
                total_mib = float(torch.cuda.get_device_properties(idx).total_memory) / (1024.0 ** 2)
            except Exception:
                total_mib = 0.0
            try:
                peak_alloc_mib = float(torch.cuda.max_memory_allocated(idx)) / (1024.0 ** 2)
            except Exception:
                peak_alloc_mib = 0.0
            try:
                peak_reserved_mib = float(torch.cuda.max_memory_reserved(idx)) / (1024.0 ** 2)
            except Exception:
                peak_reserved_mib = 0.0

            alloc_pct = (peak_alloc_mib / total_mib * 100.0) if total_mib > 0.0 else 0.0
            reserved_pct = (peak_reserved_mib / total_mib * 100.0) if total_mib > 0.0 else 0.0
            _logger.info(
                f"{clip_label}: GPU{idx} peak | alloc={peak_alloc_mib:.1f} MiB ({alloc_pct:.2f}%), "
                f"reserved={peak_reserved_mib:.1f} MiB ({reserved_pct:.2f}%), total={total_mib:.1f} MiB, "
                f"name={device_name}"
            )

    def _recolor_tk_widgets(self, parent, bg_color, fg_color, entry_bg):
        """Recursively recolors raw tk widgets within a parent container."""
        for widget in parent.winfo_children():
            widget_type = widget.winfo_class()
            try:
                # Basic widgets that support bg/fg config
                if widget_type in ('Label', 'Checkbutton'):
                    widget.config(bg=bg_color, fg=fg_color)
                elif widget_type == 'Entry':
                    widget.config(bg=entry_bg, fg=fg_color, insertbackground=fg_color)
                # Buttons usually look better controlled by the theme/style
                # elif widget_type == 'Button':
                #     widget.config(bg=bg_color, fg=fg_color) 
                # Containers
                elif widget_type in ('Frame', 'Toplevel', 'Menubutton'):
                    widget.config(bg=bg_color)
                # LabelFrame title
                elif widget_type == 'Labelframe':
                    widget.config(bg=bg_color, fg=fg_color)
            except tk.TclError:
                # Some widgets (like a tk.Text in a Log window, if you had one)
                # or ttk widgets passed to this function will raise an error. Ignore.
                pass 
            
            # Recurse into children
            self._recolor_tk_widgets(widget, bg_color, fg_color, entry_bg)

    def _reset_settings_to_defaults(self):
        if messagebox.askyesno("Reset Settings", "Are you sure you want to reset all settings to their default values?"):
            self._apply_all_settings(self.initial_default_settings)
            _logger.info("All settings have been reset to their initial defaults.")
            self.status_message_var.set("Settings reset to defaults.")

    def _restore_input_files(self, folder_type: str): # Added folder_type argument
        """Moves original input files from a specified 'finished' or 'failed' subfolder back to their input directory."""
        display_folder_name = folder_type.capitalize() # "Finished" or "Failed"

        if not messagebox.askyesno(f"Restore {display_folder_name} Input Files", 
                                   f"Are you sure you want to move original input files from the '{display_folder_name}' subfolder "
                                   f"back to their original input directory?"):
            _logger.info(f"Restore {display_folder_name} operation cancelled by user confirmation.")
            return

        source_input_path = self.input_dir_or_file_var.get()

        if not os.path.isdir(source_input_path):
            messagebox.showerror("Restore Error", f"Restore '{display_folder_name}' input files operation is only applicable when 'Input Folder/File' is set to a directory (batch mode).")
            _logger.warning(f"Restore {display_folder_name} operation skipped: Input Folder/File is not a directory: {source_input_path}")
            self.status_message_var.set(f"Restore {display_folder_name} failed: Input is not a directory.")
            return

        restored_count = 0
        errors_count = 0
        
        # Only process the specified folder_type
        finished_source_folder = os.path.join(source_input_path, folder_type) # Use folder_type directly
        
        if os.path.isdir(finished_source_folder):
            _logger.info(f"==> Restoring input files from: {finished_source_folder}")
            for filename in os.listdir(finished_source_folder):
                src_path = os.path.join(finished_source_folder, filename)
                dest_path = os.path.join(source_input_path, filename) 
                
                if os.path.isfile(src_path):
                    try:
                        if os.path.exists(dest_path):
                            base, ext = os.path.splitext(filename)
                            new_filename = f"{base}_restored_{time.strftime('%Y%m%d%H%M%S')}{ext}"
                            dest_path = os.path.join(source_input_path, new_filename)
                            _logger.warning(f"Input file '{filename}' already exists in '{source_input_path}'. Restoring as '{new_filename}'.")
                        
                        shutil.move(src_path, dest_path)
                        restored_count += 1
                        _logger.debug(f"Moved input file '{filename}' to '{source_input_path}'")
                    except Exception as e:
                        errors_count += 1
                        _logger.error(f"Error moving input file '{filename}' from '{finished_source_folder}': {e}", exc_info=True)
            
            # Clean up empty folder after restoring
            try:
                if not os.listdir(finished_source_folder):
                    os.rmdir(finished_source_folder)
                    _logger.info(f"Removed empty folder: {finished_source_folder}")
            except OSError as e:
                _logger.warning(f"Could not remove empty folder '{finished_source_folder}': {e}")
        else:
            _logger.info(f"==> Input '{display_folder_name}' folder not found: {finished_source_folder}")


        # Final status update
        if restored_count > 0 or errors_count > 0:
            self.status_message_var.set(f"Restore {display_folder_name} complete: {restored_count} input files moved, {errors_count} errors.")
            messagebox.showinfo("Restoration Complete", 
                                f"{display_folder_name} input files restoration attempted.\n"
                                f"Successfully restored: {restored_count} file(s)\n"
                                f"Skipped (due to error/conflict): {errors_count} file(s)")
        else:
            self.status_message_var.set(f"No {display_folder_name.lower()} input files found to restore.")
            messagebox.showinfo("Restoration Complete", f"No input files found in the '{display_folder_name}' folder to restore.")

    def _save_all_settings_as(self):
        filepath = filedialog.asksaveasfilename(title="Save Settings As", filetypes=self.SETTINGS_FILETYPES, defaultextension=".json", initialdir=self.last_settings_dir)
        if not filepath:
            _logger.info("Save settings cancelled by user.")
            return
        self.last_settings_dir = os.path.dirname(filepath)
        current_settings = self._collect_all_settings()
        if save_json_file(current_settings, filepath, indent=4):
            _logger.info(f"Successfully saved settings to: {filepath}")
            messagebox.showinfo("Save Successful", f"Settings saved to:\n{filepath}")
        else:
            messagebox.showerror("Save Error", f"Could not save settings to:\n{filepath}\nSee console log for details.")

    def _save_final_output_sidecar_json(self, original_basename, final_merged_path, master_meta_filepath, master_meta, was_segments, merge_successful):
        json_path, json_content = None, {}
        output_suffix_val = self.merge_output_suffix_var.get()

        if was_segments:
            if merge_successful and final_merged_path and not final_merged_path.startswith("N/A"):
                out_fmt_selected = self.merge_output_format_var.get()
                
                json_content = {
                    "source_video_basename": original_basename, "processing_mode": "segmented_then_merged",
                    "final_output_path": os.path.abspath(final_merged_path), 
                    "final_output_format_selected": out_fmt_selected,
                    "master_metadata_path_source": os.path.abspath(master_meta_filepath) if master_meta_filepath else None,
                    "global_processing_settings_summary": master_meta.get("global_processing_settings"),
                    "merge_settings_summary": {
                        "output_format_selected": out_fmt_selected, 
                        "output_suffix": output_suffix_val,
                        "alignment_method": self.merge_alignment_method_var.get(),
                        "dithering": self.merge_dither_var.get(), "dither_strength": self.merge_dither_strength_var.get(),
                        "gamma_correction": self.merge_gamma_correct_var.get(),
                        "gamma_value_if_applied": self.merge_gamma_value_var.get() if self.merge_gamma_correct_var.get() else 1.0,
                        "percentile_norm": self.merge_percentile_norm_var.get(),
                        "norm_low_perc": self.merge_norm_low_perc_var.get(), "norm_high_perc": self.merge_norm_high_perc_var.get(),
                    }, "generation_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
                if os.path.isdir(final_merged_path):
                    json_path = os.path.join(os.path.dirname(final_merged_path.rstrip(os.sep)), f"{os.path.basename(final_merged_path.rstrip(os.sep))}.json")
                elif os.path.isfile(final_merged_path):
                    json_path = get_sidecar_json_filename(final_merged_path)
                else: _logger.warning(f"    Cannot determine final JSON path for merged {original_basename} (output path: {final_merged_path}).") 
            else: _logger.info(f"  Skipping final JSON for merged {original_basename} (merge not successful/path invalid).") 
        else:
            if master_meta and master_meta.get("jobs_info"):
                job_info = master_meta["jobs_info"][0]
                relative_output_filename = job_info.get("output_video_filename") 
                if relative_output_filename:
                    out_path = os.path.join(self.output_dir.get(), relative_output_filename)
                    out_fmt_from_ext = os.path.splitext(relative_output_filename)[1].lstrip('.') 

                    json_content = {
                        "source_video_basename": original_basename, "processing_mode": "full_video",
                        "final_output_path": os.path.abspath(out_path), 
                        "final_output_format": out_fmt_from_ext,
                        "global_processing_settings": master_meta.get("global_processing_settings"),
                        "job_specific_details": job_info,
                        "generation_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    }
                    if os.path.isdir(out_path): 
                         json_path = os.path.join(os.path.dirname(out_path.rstrip(os.sep)), f"{os.path.basename(out_path.rstrip(os.sep))}.json")
                    elif os.path.isfile(out_path):
                        json_path = get_sidecar_json_filename(out_path)
                    else: _logger.warning(f"    Cannot determine final JSON path for full video {original_basename} (output path: {out_path}).") 
                else: _logger.warning(f"  Skipping final JSON for full video {original_basename} (output path/format missing from job_info).") 
            else: _logger.warning(f"  Skipping final JSON for full video {original_basename} (master_meta or job_info missing).") 

        if json_path and json_content:
            _logger.debug(f"    Attempting to save final output JSON to: {json_path}") 
            if save_json_file(json_content, json_path): 
                _logger.debug(f"  Successfully saved sidecar JSON for final output: {json_path}") 
        elif self.save_final_output_json_var.get():
            mode = "merged" if was_segments else "full video"
            _logger.warning(f"  Final output JSON for {mode} '{original_basename}' not created (conditions not met, or save failed).")

    def _save_master_metadata_and_cleanup_segment_json(self, master_meta_to_save, original_basename, main_output_dir, was_segments, segment_subfolder_path):
        master_meta_filepath, meta_saved = None, False
        if was_segments:
            if not segment_subfolder_path:
                segment_subfolder_path = os.path.join(main_output_dir, get_segment_output_folder_name(original_basename))
            os.makedirs(segment_subfolder_path, exist_ok=True)
            master_meta_filepath = os.path.join(segment_subfolder_path, f"{original_basename}_master_meta.json")
        else:
            master_meta_filepath = os.path.join(main_output_dir, f"{original_basename}_master_meta.json")

        should_save_master_meta_here = was_segments
        if should_save_master_meta_here:
            if save_json_file(master_meta_to_save, master_meta_filepath):
                _logger.debug(f"Saved master metadata for {original_basename} to {master_meta_filepath}")
                meta_saved = True
            if was_segments and meta_saved and segment_subfolder_path:
                _logger.debug(f"  Attempting to delete individual segment JSONs for {original_basename} (master created).")
                deleted_count = 0
                for job_data in master_meta_to_save.get("jobs_info", []):
                    npz_file = job_data.get("output_segment_filename")
                    if npz_file:
                        json_to_del = os.path.join(segment_subfolder_path, get_sidecar_json_filename(npz_file))
                        if os.path.exists(json_to_del):
                            try: os.remove(json_to_del); deleted_count += 1
                            except Exception as e: _logger.error(f"ERROR deleting individual segment JSON {json_to_del}: {e}")
                _logger.debug(f"    Deleted {deleted_count} individual segment JSONs.")
            elif was_segments and not meta_saved:
                _logger.warning(f"  Skipping deletion of individual segment JSONs for {original_basename} (master_meta.json not saved).")
        elif not was_segments:
            _logger.debug(f"Skipping save of '{os.path.basename(master_meta_filepath)}' by _save_master_metadata for full video mode for {original_basename}.")
            meta_saved = False
        return master_meta_filepath, meta_saved

    def _set_ui_processing_state(self, is_processing: bool):
        new_state = tk.DISABLED if is_processing else tk.NORMAL
        cancel_state = tk.NORMAL if is_processing else tk.DISABLED
        unique_widgets = list(set(self.widgets_to_disable_during_processing))

        for widget in unique_widgets:
            if widget == self.cancel_button: continue
            if hasattr(widget, 'configure'):
                try:
                    if isinstance(widget, ttk.Combobox): widget.configure(state='disabled' if is_processing else 'readonly')
                    else: widget.configure(state=new_state)
                except tk.TclError: pass 
                
        if hasattr(self, 'help_menu') and self.help_menu:
            try:
                self.help_menu.entryconfig("Enable Debug Logging", state=new_state)
            except tk.TclError: pass
        
        if hasattr(self, 'cancel_button') and self.cancel_button:
             try: self.cancel_button.configure(state=cancel_state)
             except tk.TclError: pass

        if hasattr(self, 'file_menu'):
            try:
                self.file_menu.entryconfig("Use Local Models Only", state=new_state)
                for item_label in ["Load Settings...", "Save Settings As...", "Reset Settings to Default"]:
                    self.file_menu.entryconfig(item_label, state=new_state)
            except tk.TclError: pass

        # --- Phase 2: If processing has *finished*, re-evaluate conditional states ---
        # This prevents conditional toggles from overriding the DISABLED state prematurely.
        if not is_processing:
            self.toggle_merge_related_options_active_state()
            self.toggle_secondary_output_options_active_state()
            self._apply_backend_specific_controls()

    def _show_help_for(self, help_key: str):
        """Displays help content for a given key in a Tkinter Toplevel window."""
        # _help_data should already be loaded by __init__
        content = self._help_data.get(help_key)
        
        if not content:
            messagebox.showinfo("Help Not Found", f"No help information available for '{help_key}'.\nEnsure '{DepthCrafterGUI.HELP_CONTENT_FILENAME}' is present and contains this key.")
            _logger.warning(f"No help content found for key: '{help_key}' in {DepthCrafterGUI.HELP_CONTENT_FILENAME}.")
            return

        help_title = content.get("title", "Help Information")
        help_text_str = content.get("text", "No details available.")
        
        # Now, create the Toplevel window as it was before
        help_window = tk.Toplevel(self.root)
        help_window.title(help_title)
        help_window.minsize(400, 200)
        help_window.transient(self.root)
        help_window.grab_set()
        
        text_frame = ttk.Frame(help_window, padding="10")
        text_frame.pack(expand=True, fill="both")
        
        help_text_widget = tk.Text(text_frame, wrap=tk.WORD, relief="flat", borderwidth=0, padx=5, pady=5, font=("Segoe UI", 9))
        help_text_widget.insert(tk.END, help_text_str)
        help_text_widget.config(state=tk.DISABLED)
        
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=help_text_widget.yview)
        help_text_widget['yscrollcommand'] = scrollbar.set
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        help_text_widget.pack(side=tk.LEFT, expand=True, fill="both")
        
        button_frame = ttk.Frame(help_window, padding=(0, 5, 0, 10))
        button_frame.pack(fill=tk.X)
        ok_button = ttk.Button(button_frame, text="OK", command=help_window.destroy, width=10)
        ok_button.pack()
        
        self.root.update_idletasks()
        help_window.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (help_window.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (help_window.winfo_height() // 2)
        help_window.geometry(f"+{x}+{y}")
        ok_button.focus_set()
        help_window.wait_window()
        _logger.debug(f"Displayed help overview for '{help_key}'.")
        
    def _start_processing_wrapper(self, source_specs_to_process, effective_seed_for_run):
        try: 
            self.start_processing(source_specs_to_process, effective_seed_for_run)
        finally: 
            self._set_ui_processing_state(False)

    def _toggle_debug_logging(self):
        if self.debug_logging_enabled.get():
            logging.getLogger().setLevel(logging.DEBUG) # Set root logger to DEBUG
            _logger.info("Debug logging ENABLED.")
        else:
            logging.getLogger().setLevel(logging.INFO)  # Set root logger back to INFO
            _logger.info("Debug logging DISABLED (set to INFO level).")

    def _update_gui_info_on_job_start(self, job_info_to_run, original_basename, log_msg_prefix):
        """Updates GUI processing info labels with target/expected values before a job starts."""
        _logger.debug(f"DEBUG GUI UPDATE (Initial): Starting update for {original_basename}")
        
        self.current_filename_var.set(f"{original_basename} ({log_msg_prefix})") 
        
        # Initial Resolution (using target_height/width setting and original dimensions as a hint)
        target_h_setting = self.target_height.get()
        target_w_setting = self.target_width.get()
        is_segment_job = job_info_to_run.get("is_segment", False)

        initial_display_res = "N/A"
        if target_h_setting > 0 and target_w_setting > 0:
            initial_display_res = f"{target_w_setting}x{target_h_setting}"
        else:
            # Fallback to detected original dimensions if target H/W are not set
            original_h = job_info_to_run.get("original_height", "N/A")
            original_w = job_info_to_run.get("original_width", "N/A")
            if original_h != "N/A" and original_w != "N/A":
                initial_display_res = f"{original_w}x{original_h} (Original/Fallback)"
        self.current_resolution_var.set(initial_display_res)

        # Initial Frames (using gui settings for segment/process_length)
        total_frames_orig_vid = job_info_to_run.get('original_video_raw_frame_count', 'N/A')
        
        initial_display_frames_str = "N/A"
        if is_segment_job:
            num_frames_to_load_raw = job_info_to_run.get("num_frames_to_load_raw", "N/A")
            if num_frames_to_load_raw != "N/A":
                initial_display_frames_str = f"{num_frames_to_load_raw}"
            
            if total_frames_orig_vid != "N/A" and str(num_frames_to_load_raw) != str(total_frames_orig_vid):
                initial_display_frames_str += f" of {total_frames_orig_vid}"
        else: # Full video
            process_length_setting = self.process_length.get()
            if process_length_setting != -1:
                initial_display_frames_str = f"{process_length_setting}"
            else:
                initial_display_frames_str = f"{total_frames_orig_vid}"
        self.current_frames_var.set(initial_display_frames_str)

        self.root.update_idletasks() # Force GUI update for initial display
        
    def _update_gui_info_on_job_finish(self, job_info_to_run, current_job_specific_metadata):
        """Updates GUI processing info labels with actual/processed values after a job finishes."""
        # _logger.debug(f"DEBUG GUI UPDATE (Final): Starting update for {job_info_to_run['original_basename']}")

        # RESOLUTION UPDATE (from actual processed values)
        processed_h = current_job_specific_metadata.get("processed_height", "N/A")
        processed_w = current_job_specific_metadata.get("processed_width", "N/A")
        
        final_display_res = "N/A"
        if processed_h != "N/A" and processed_w != "N/A":
            final_display_res = f"{processed_w}x{processed_h}" # This is the actual processed resolution
        else:
            final_display_res = self.current_resolution_var.get() + " (Failed to confirm)" # Append if couldn't get actual
        self.current_resolution_var.set(final_display_res)
        
        # FRAMES UPDATE (from actual processed values)
        processed_frames_for_job = current_job_specific_metadata.get("frames_in_output_video", "N/A")
        total_frames_orig_vid = job_info_to_run.get('original_video_raw_frame_count', 'N/A')
        is_segment_job = job_info_to_run.get("is_segment", False)
        
        final_display_frames_str = "N/A"

        if processed_frames_for_job != "N/A":
            if is_segment_job:
                final_display_frames_str = f"{processed_frames_for_job}"
                if total_frames_orig_vid != "N/A" and str(processed_frames_for_job) != str(total_frames_orig_vid):
                    final_display_frames_str += f" of {total_frames_orig_vid} total"
            else: # Full video processing
                final_display_frames_str = f"{processed_frames_for_job}"
                if total_frames_orig_vid != "N/A" and str(processed_frames_for_job) != str(total_frames_orig_vid):
                    final_display_frames_str += f" (of {total_frames_orig_vid} total)"
        else:
            final_display_frames_str = self.current_frames_var.get() + " (Failed to confirm)" # Append if couldn't get actual
        
        self.current_frames_var.set(final_display_frames_str)
        self.root.update_idletasks() # Force GUI update for final display
    
    def add_param(self, parent, label, var, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry = ttk.Entry(parent, textvariable=var, width=20)
        entry.grid(row=row, column=1, padx=5, pady=2, sticky="w")
        return entry

    def browse_input_folder(self):
        folder = filedialog.askdirectory(initialdir=self.input_dir_or_file_var.get())
        if folder:
            self.input_dir_or_file_var.set(os.path.normpath(folder))
            self.single_file_mode_active = False
            if self._is_image_sequence_folder(folder):
                self.current_input_mode = "image_sequence_folder"
                _logger.info(f"GUI: Input mode set to Image Sequence Folder: {folder}")
            else:
                self.current_input_mode = "batch_folder"
                _logger.info(f"GUI: Input mode set to Batch Folder: {folder}")

    def browse_output(self):
        folder = filedialog.askdirectory(initialdir=self.output_dir.get())
        if folder: self.output_dir.set(os.path.normpath(folder))

    def browse_single_input_file(self):
        filetypes = [("All Supported", "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.gif *.png *.jpg *.jpeg *.bmp *.tiff *.exr"),
                     ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.gif"),
                     ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.exr")]
        
        initial_dir_guess = self.input_dir_or_file_var.get()
        if os.path.isfile(initial_dir_guess): initial_dir_guess = os.path.dirname(initial_dir_guess)
        if not os.path.isdir(initial_dir_guess): initial_dir_guess = os.path.expanduser("~")


        filepath = filedialog.askopenfilename(initialdir=initial_dir_guess, filetypes=filetypes)
        if filepath:
            self.input_dir_or_file_var.set(os.path.normpath(filepath))
            self.single_file_mode_active = True
            ext = os.path.splitext(filepath)[1].lower()
            is_video = any(ext in vid_ext.replace("*", "") for vid_ext in self.VIDEO_EXTENSIONS)
            is_image = any(ext in img_ext.replace("*", "") for img_ext in self.IMAGE_EXTENSIONS)

            if is_video:
                self.current_input_mode = "single_video_file"
                _logger.info(f"GUI: Input mode set to Single Video File: {filepath}")
            elif is_image:
                self.current_input_mode = "single_image_file"
                _logger.info(f"GUI: Input mode set to Single Image File: {filepath}")
            else:
                _logger.warning(f"GUI: Could not determine type of single file: {filepath}. Assuming video.")
                self.current_input_mode = "single_video_file" 
                messagebox.showwarning("Unknown File Type", f"Could not determine if '{os.path.basename(filepath)}' is a video or image. Assuming video.")

    def create_widgets(self):
        self.widgets_to_disable_during_processing = []
        
        # --- Input Source Frame ---
        dir_frame = ttk.LabelFrame(self.root, text="Input Source")
        dir_frame.pack(fill="x", padx=10, pady=5, expand=False)        
        
        ttk.Label(dir_frame, text="Input Folder/File:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.entry_input_dir_or_file = ttk.Entry(dir_frame, textvariable=self.input_dir_or_file_var, width=50)
        self.entry_input_dir_or_file.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        _create_hover_tooltip(self.entry_input_dir_or_file, "input_dir_or_file") # Tooltip for entry
        
        browse_buttons_frame = ttk.Frame(dir_frame)
        browse_buttons_frame.grid(row=0, column=2, padx=5, pady=0, sticky="w")
        
        self.browse_input_folder_btn = ttk.Button(browse_buttons_frame, text="Browse Folder", command=self.browse_input_folder)
        self.browse_input_folder_btn.pack(side=tk.LEFT, padx=(0,2))
        _create_hover_tooltip(self.browse_input_folder_btn, "browse_input_folder") # Tooltip for button
        
        self.browse_single_file_btn = ttk.Button(browse_buttons_frame, text="Load Single File", command=self.browse_single_input_file)
        self.browse_single_file_btn.pack(side=tk.LEFT, padx=(2,0))
        _create_hover_tooltip(self.browse_single_file_btn, "browse_single_file") # Tooltip for button
        
        dir_frame.columnconfigure(1, weight=1)
        self.widgets_to_disable_during_processing.extend([
            self.entry_input_dir_or_file, 
            self.browse_input_folder_btn, 
            self.browse_single_file_btn
        ])

        ttk.Label(dir_frame, text="Output Folder:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.entry_output_dir = ttk.Entry(dir_frame, textvariable=self.output_dir, width=50)
        self.entry_output_dir.grid(row=1, column=1, padx=5, pady=2)
        _create_hover_tooltip(self.entry_output_dir, "output_dir") # Tooltip for entry
        
        self.browse_output_btn = ttk.Button(dir_frame, text="Browse", command=self.browse_output)
        self.browse_output_btn.grid(row=1, column=2, padx=5, pady=2)
        _create_hover_tooltip(self.browse_output_btn, "browse_output") # Tooltip for button
        
        self.widgets_to_disable_during_processing.extend([self.entry_output_dir, self.browse_output_btn])

        # --- Settings Container Frame (New) ---
        # This frame will hold the Main Params, Frame & Segment Control, Merged Output, and Secondary Output frames.
        settings_container_frame = ttk.Frame(self.root)
        settings_container_frame.pack(fill="x", padx=10, pady=0, expand=False)
        settings_container_frame.columnconfigure(0, weight=1)
        settings_container_frame.columnconfigure(1, weight=1)

        # --- Main Parameters Frame ---
        main_params_frame = ttk.LabelFrame(settings_container_frame, text="Main Parameters")
        main_params_frame.grid(row=0, column=0, padx=(0,5), pady=5, sticky="nsew") # Placed in new container
        row_idx = 0
        
        # Guidance Scale
        ttk.Label(main_params_frame, text="Guidance Scale:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry_guidance_scale = ttk.Entry(main_params_frame, textvariable=self.guidance_scale, width=18)
        entry_guidance_scale.grid(row=row_idx, column=1, padx=(5,0), pady=2, sticky="w")
        _create_hover_tooltip(entry_guidance_scale, "guidance_scale")
        self.widgets_to_disable_during_processing.append(entry_guidance_scale); row_idx += 1
        
        # Inference Steps
        ttk.Label(main_params_frame, text="Inference Steps:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry_inference_steps = ttk.Entry(main_params_frame, textvariable=self.inference_steps, width=18)
        entry_inference_steps.grid(row=row_idx, column=1, padx=(5,0), pady=2, sticky="w")
        _create_hover_tooltip(entry_inference_steps, "inference_steps")
        self.widgets_to_disable_during_processing.append(entry_inference_steps); row_idx += 1

        # Target Width
        ttk.Label(main_params_frame, text="Target Width:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry_target_width = ttk.Entry(main_params_frame, textvariable=self.target_width, width=18)
        entry_target_width.grid(row=row_idx, column=1, padx=(5,0), pady=2, sticky="w")
        _create_hover_tooltip(entry_target_width, "target_width")
        self.widgets_to_disable_during_processing.append(entry_target_width); row_idx += 1

        # Target Height
        ttk.Label(main_params_frame, text="Target Height:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry_target_height = ttk.Entry(main_params_frame, textvariable=self.target_height, width=18)
        entry_target_height.grid(row=row_idx, column=1, padx=(5,0), pady=2, sticky="w")
        _create_hover_tooltip(entry_target_height, "target_height")
        self.widgets_to_disable_during_processing.append(entry_target_height); row_idx += 1

        # Seed
        ttk.Label(main_params_frame, text="Seed:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry_seed = ttk.Entry(main_params_frame, textvariable=self.seed, width=18)
        entry_seed.grid(row=row_idx, column=1, padx=(5,0), pady=2, sticky="w")
        _create_hover_tooltip(entry_seed, "seed")
        self.widgets_to_disable_during_processing.append(entry_seed); row_idx += 1
        
        # CPU Offload Mode
        ttk.Label(main_params_frame, text="CPU Offload Mode:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        self.combo_cpu_offload = ttk.Combobox(main_params_frame, textvariable=self.cpu_offload, values=["model", "sequential", "none"], width=17, state="readonly")
        self.combo_cpu_offload.grid(row=row_idx, column=1, padx=5, pady=2, sticky="w")
        _create_hover_tooltip(self.combo_cpu_offload, "cpu_offload")
        self.widgets_to_disable_during_processing.append(self.combo_cpu_offload); row_idx += 1

        # Model backend selector
        ttk.Label(main_params_frame, text="Model Backend:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        self.combo_model_backend = ttk.Combobox(
            main_params_frame,
            textvariable=self.model_backend_var,
            values=["depthcrafter", "geometrycrafter_diff", "geometrycrafter_determ", "stereopilot"],
            width=20,
            state="readonly",
        )
        self.combo_model_backend.grid(row=row_idx, column=1, padx=5, pady=2, sticky="w")
        _create_hover_tooltip(self.combo_model_backend, "model_backend")
        self.widgets_to_disable_during_processing.append(self.combo_model_backend); row_idx += 1

        geometry_settings_btn = ttk.Button(
            main_params_frame,
            text="  ↳ Configure Backend...",
            command=self.toggle_geometry_settings_visibility,
            width=31,
        )
        geometry_settings_btn.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=(2, 2))
        self.geometry_settings_toggle_btn = geometry_settings_btn
        self.widgets_to_disable_during_processing.append(geometry_settings_btn)
        row_idx += 1

        self.geometry_local_status_label = ttk.Label(
            main_params_frame,
            textvariable=self.geometry_local_status_var,
            wraplength=360,
            justify="left",
        )
        self.geometry_local_status_label.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 4))
        self.widgets_to_disable_during_processing.append(self.geometry_local_status_label)
        row_idx += 1

        # --- Frame & Segment Control Frame ---
        fs_frame = ttk.LabelFrame(settings_container_frame, text="Frame & Segment Control")
        fs_frame.grid(row=0, column=1, padx=(5,0), pady=5, sticky="nsew") # Placed in new container

        row_idx = 0 
        
        # Window Size
        ttk.Label(fs_frame, text="Window Size:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry_window_size = ttk.Entry(fs_frame, textvariable=self.window_size, width=18)
        entry_window_size.grid(row=row_idx, column=1, padx=(5,0), pady=2, sticky="w")
        _create_hover_tooltip(entry_window_size, "window_size")
        self.widgets_to_disable_during_processing.append(entry_window_size); row_idx += 1
        
        # Overlap
        ttk.Label(fs_frame, text="Overlap:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry_overlap = ttk.Entry(fs_frame, textvariable=self.overlap, width=18)
        entry_overlap.grid(row=row_idx, column=1, padx=(5,0), pady=2, sticky="w")
        _create_hover_tooltip(entry_overlap, "overlap")
        self.widgets_to_disable_during_processing.append(entry_overlap); row_idx += 1
        
        # Target FPS
        ttk.Label(fs_frame, text="Target FPS (-1 Original):").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry_target_fps = ttk.Entry(fs_frame, textvariable=self.target_fps, width=18)
        entry_target_fps.grid(row=row_idx, column=1, padx=(5,0), pady=2, sticky="w")
        _create_hover_tooltip(entry_target_fps, "target_fps")
        self.widgets_to_disable_during_processing.append(entry_target_fps); row_idx += 1
        
        # Process Max Frames
        ttk.Label(fs_frame, text="Process Max Frames (-1 All):").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry_process_length = ttk.Entry(fs_frame, textvariable=self.process_length, width=18)
        entry_process_length.grid(row=row_idx, column=1, padx=(5,0), pady=2, sticky="w")
        _create_hover_tooltip(entry_process_length, "process_length")
        self.widgets_to_disable_during_processing.append(entry_process_length); row_idx += 1
        
        # Save Sidecar JSON for Final Output
        self.save_final_json_cb = ttk.Checkbutton(fs_frame, text="Save Sidecar JSON for Final Output", variable=self.save_final_output_json_var)
        self.save_final_json_cb.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.save_final_json_cb, "save_final_json")
        self.widgets_to_disable_during_processing.append(self.save_final_json_cb); row_idx +=1

        # Process as Segments
        self.process_as_segments_cb = ttk.Checkbutton(fs_frame, text="Process as Segments (Low VRAM Mode)", variable=self.process_as_segments_var, command=self.toggle_merge_related_options_active_state)
        self.process_as_segments_cb.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.process_as_segments_cb, "process_as_segments")
        self.widgets_to_disable_during_processing.append(self.process_as_segments_cb); row_idx += 1

        # Special mode: backfill only clips/segments missing NPZ raw outputs.
        self.npz_backfill_missing_only_cb = ttk.Checkbutton(
            fs_frame,
            text="Special: Backfill Missing NPZ Only",
            variable=self.npz_backfill_missing_only_var
        )
        self.npz_backfill_missing_only_cb.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.npz_backfill_missing_only_cb, "npz_backfill_missing_only")
        self.widgets_to_disable_during_processing.append(self.npz_backfill_missing_only_cb); row_idx += 1

        # --- Merged Output Options Frame ---
        merge_opts_frame = ttk.LabelFrame(settings_container_frame, text="Merged Output Options (if segments processed)")
        merge_opts_frame.grid(row=1, column=0, padx=(0,5), pady=5, sticky="nsew") # Placed in new container
        merge_opts_frame.columnconfigure(0, minsize=120) # Ensure column 0 for labels is wide enough
        self.merge_related_widgets_references = []
        self.keep_npz_dependent_widgets = []
        row_idx = 0

        # Keep intermediate NPZ files
        self.keep_npz_cb = ttk.Checkbutton(merge_opts_frame, text="Keep intermediate NPZ", variable=self.keep_intermediate_npz_var, command=self.toggle_keep_npz_dependent_options_state)
        self.keep_npz_cb.grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.keep_npz_cb, "keep_npz")
        self.merge_related_widgets_references.append(self.keep_npz_cb)
        self.widgets_to_disable_during_processing.append(self.keep_npz_cb); row_idx += 1

        # Min Orig. Vid Frames to Keep NPZ
        self.lbl_min_frames_npz = ttk.Label(merge_opts_frame, text="  ↳ Min thesh. to Keep NPZ:")
        self.lbl_min_frames_npz.grid(row=row_idx, column=0, sticky="e", padx=(20,2), pady=2)
        self.entry_min_frames_npz = ttk.Entry(merge_opts_frame, textvariable=self.min_frames_to_keep_npz_var, width=7)
        self.entry_min_frames_npz.grid(row=row_idx, column=1, padx=(0,2), pady=2, sticky="w")
        _create_hover_tooltip(self.entry_min_frames_npz, "min_frames_npz")
        self.keep_npz_dependent_widgets.extend([self.lbl_min_frames_npz, self.entry_min_frames_npz])
        self.widgets_to_disable_during_processing.extend([self.lbl_min_frames_npz, self.entry_min_frames_npz]); row_idx += 1

        # Segment Visual Format
        self.lbl_intermediate_fmt = ttk.Label(merge_opts_frame, text="  ↳ Segment Format:")
        self.lbl_intermediate_fmt.grid(row=row_idx, column=0, sticky="e", padx=(20,2), pady=2)
        combo_intermediate_fmt_values = ["png_sequence", "mp4", "main10_mp4", "none"]
        if OPENEXR_AVAILABLE_GUI: combo_intermediate_fmt_values.extend(["exr_sequence", "exr"])
        self.combo_intermediate_fmt = ttk.Combobox(merge_opts_frame, textvariable=self.keep_intermediate_segment_visual_format_var, values=combo_intermediate_fmt_values, width=17, state="readonly")
        self.combo_intermediate_fmt.grid(row=row_idx, column=1, padx=(0,2), pady=2, sticky="w")
        _create_hover_tooltip(self.combo_intermediate_fmt, "segment_visual_format")
        self.keep_npz_dependent_widgets.extend([self.lbl_intermediate_fmt, self.combo_intermediate_fmt])
        self.widgets_to_disable_during_processing.extend([self.lbl_intermediate_fmt, self.combo_intermediate_fmt]); row_idx += 1
        self.toggle_keep_npz_dependent_options_state()

        # Dithering (MP4)
        self.merge_dither_cb = ttk.Checkbutton(merge_opts_frame, text="Dithering", variable=self.merge_dither_var, command=self.toggle_dither_options_active_state)
        self.merge_dither_cb.grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.merge_dither_cb, "merge_dither") # Tooltip on checkbox
        
        dither_details_frame = ttk.Frame(merge_opts_frame)
        dither_details_frame.grid(row=row_idx, column=1, sticky="w", padx=(0,0))
        self.lbl_dither_str = ttk.Label(dither_details_frame, text="Strength:")
        self.lbl_dither_str.pack(side=tk.LEFT, padx=(0, 2))
        self.entry_dither_str = ttk.Entry(dither_details_frame, textvariable=self.merge_dither_strength_var, width=7)
        self.entry_dither_str.pack(side=tk.LEFT, padx=(0, 0))
        _create_hover_tooltip(self.entry_dither_str, "merge_dither_strength") # Tooltip on entry
        
        self.merge_related_widgets_references.append((self.merge_dither_cb, dither_details_frame))
        self.widgets_to_disable_during_processing.extend([self.merge_dither_cb, self.lbl_dither_str, self.entry_dither_str]); row_idx += 1
        self.toggle_dither_options_active_state() # Call after creation

        # Gamma Correct (MP4)
        self.merge_gamma_cb = ttk.Checkbutton(merge_opts_frame, text="Gamma Adjust", variable=self.merge_gamma_correct_var, command=self.toggle_gamma_options_active_state)
        self.merge_gamma_cb.grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.merge_gamma_cb, "merge_gamma") # Tooltip on checkbox
        
        gamma_details_frame = ttk.Frame(merge_opts_frame)
        gamma_details_frame.grid(row=row_idx, column=1, sticky="w", padx=(0,0))
        self.lbl_gamma_val = ttk.Label(gamma_details_frame, text="Value:")
        self.lbl_gamma_val.pack(side=tk.LEFT, padx=(0, 2))
        self.entry_gamma_val = ttk.Entry(gamma_details_frame, textvariable=self.merge_gamma_value_var, width=7)
        self.entry_gamma_val.pack(side=tk.LEFT, padx=(0, 0))
        _create_hover_tooltip(self.entry_gamma_val, "merge_gamma_value") # Tooltip on entry
        
        self.merge_related_widgets_references.append((self.merge_gamma_cb, gamma_details_frame))
        self.widgets_to_disable_during_processing.extend([self.merge_gamma_cb, self.lbl_gamma_val, self.entry_gamma_val]); row_idx += 1
        self.toggle_gamma_options_active_state() # Call after creation

        # Percentile Normalization
        self.merge_perc_norm_cb = ttk.Checkbutton(merge_opts_frame, text="Normalization", variable=self.merge_percentile_norm_var, command=self.toggle_percentile_norm_options_active_state)
        self.merge_perc_norm_cb.grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.merge_perc_norm_cb, "merge_percentile_norm") # Tooltip on checkbox
        
        low_high_frame = ttk.Frame(merge_opts_frame)
        low_high_frame.grid(row=row_idx, column=1, sticky="w", padx=(0,0))
        self.lbl_low_perc = ttk.Label(low_high_frame, text="Low:")
        self.lbl_low_perc.pack(side=tk.LEFT, padx=(0,2))
        self.entry_low_perc = ttk.Entry(low_high_frame, textvariable=self.merge_norm_low_perc_var, width=7)
        self.entry_low_perc.pack(side=tk.LEFT, padx=(0,10))
        self.lbl_high_perc = ttk.Label(low_high_frame, text="High:")
        self.lbl_high_perc.pack(side=tk.LEFT, padx=(0,2))
        self.entry_high_perc = ttk.Entry(low_high_frame, textvariable=self.merge_norm_high_perc_var, width=7)
        self.entry_high_perc.pack(side=tk.LEFT, padx=(0,0))
        ttk.Label(merge_opts_frame, text="  ↳").grid(row=row_idx, column=0, sticky="e", padx=(10,2)) # Aligns with the checkbox
        _create_hover_tooltip(self.entry_low_perc, "merge_norm_low_perc") # Tooltip on entry
        _create_hover_tooltip(self.entry_high_perc, "merge_norm_high_perc") # Tooltip on entry
        
        self.merge_related_widgets_references.append(self.merge_perc_norm_cb)
        self.widgets_to_disable_during_processing.extend([self.lbl_low_perc, self.entry_low_perc, self.lbl_high_perc, self.entry_high_perc]); row_idx += 1
        self.toggle_percentile_norm_options_active_state()

        # Alignment Method
        lbl_merge_alignment = ttk.Label(merge_opts_frame, text="Alignment Method:")
        lbl_merge_alignment.grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        self.combo_merge_alignment = ttk.Combobox(merge_opts_frame, textvariable=self.merge_alignment_method_var, values=["Shift & Scale", "Linear Blend"], width=17, state="readonly")
        self.combo_merge_alignment.grid(row=row_idx, column=1, padx=(0,2), pady=2, sticky="w")
        _create_hover_tooltip(self.combo_merge_alignment, "merge_alignment_method")
        self.merge_related_widgets_references.append((lbl_merge_alignment, self.combo_merge_alignment))
        self.widgets_to_disable_during_processing.extend([lbl_merge_alignment, self.combo_merge_alignment]); row_idx += 1
        
        # Output Format
        lbl_merge_fmt = ttk.Label(merge_opts_frame, text="Output Format:")
        lbl_merge_fmt.grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        merge_fmt_values = ["mp4", "main10_mp4", "png_sequence"] + (["exr_sequence", "exr"] if OPENEXR_AVAILABLE_GUI else [])
        self.combo_merge_fmt = ttk.Combobox(merge_opts_frame, textvariable=self.merge_output_format_var, values=merge_fmt_values, width=17, state="readonly")
        self.combo_merge_fmt.grid(row=row_idx, column=1, padx=(0,2), pady=2, sticky="w")
        _create_hover_tooltip(self.combo_merge_fmt, "merge_output_format")
        self.merge_related_widgets_references.append((lbl_merge_fmt, self.combo_merge_fmt))
        self.widgets_to_disable_during_processing.extend([lbl_merge_fmt, self.combo_merge_fmt]); row_idx += 1

        # Output Suffix
        lbl_merge_suffix = ttk.Label(merge_opts_frame, text="Output Suffix:")
        lbl_merge_suffix.grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        self.entry_merge_suffix = ttk.Entry(merge_opts_frame, textvariable=self.merge_output_suffix_var, width=18)
        self.entry_merge_suffix.grid(row=row_idx, column=1, padx=(0,2), pady=2, sticky="w")
        _create_hover_tooltip(self.entry_merge_suffix, "merge_output_suffix")
        self.merge_related_widgets_references.append((lbl_merge_suffix, self.entry_merge_suffix))
        self.widgets_to_disable_during_processing.extend([lbl_merge_suffix, self.entry_merge_suffix]); row_idx += 1

        # --- NEW: Secondary Output Frame ---
        secondary_output_frame = ttk.LabelFrame(settings_container_frame, text="Secondary Output")
        secondary_output_frame.grid(row=1, column=1, padx=(5,0), pady=5, sticky="nsew") # Placed in new container
        secondary_output_frame.columnconfigure(0, minsize=140) # Adjust as needed
        
        row_idx = 0
        # Enable Secondary Output Checkbox
        self.enable_secondary_output_cb = ttk.Checkbutton(secondary_output_frame, text="Enable Secondary Output", variable=self.enable_dual_output_robust_norm, command=self.toggle_secondary_output_options_active_state)
        self.enable_secondary_output_cb.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.enable_secondary_output_cb, "enable_secondary_output") # Add help_content.json entry
        self.widgets_to_disable_during_processing.append(self.enable_secondary_output_cb); row_idx += 1
        
        # Depth Range (0-1) Low / High
        ttk.Label(secondary_output_frame, text="Depth Output Range (0-1):").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        depth_range_frame = ttk.Frame(secondary_output_frame)
        depth_range_frame.grid(row=row_idx, column=1, sticky="w", padx=0, pady=0)
        
        lbl_out_min = ttk.Label(depth_range_frame, text="Low:")
        lbl_out_min.pack(side=tk.LEFT, padx=(0,2))
        entry_out_min = ttk.Entry(depth_range_frame, textvariable=self.robust_norm_output_min, width=7)
        entry_out_min.pack(side=tk.LEFT, padx=(0,10))
        _create_hover_tooltip(entry_out_min, "robust_norm_output_min") # Add help_content.json entry
        
        lbl_out_max = ttk.Label(depth_range_frame, text="High:")
        lbl_out_max.pack(side=tk.LEFT, padx=(0,2))
        entry_out_max = ttk.Entry(depth_range_frame, textvariable=self.robust_norm_output_max, width=7)
        entry_out_max.pack(side=tk.LEFT, padx=(0,0))
        _create_hover_tooltip(entry_out_max, "robust_norm_output_max") # Add help_content.json entry
        
        self.secondary_output_widgets_references.extend([lbl_out_min, entry_out_min, lbl_out_max, entry_out_max])
        self.widgets_to_disable_during_processing.extend([lbl_out_min, entry_out_min, lbl_out_max, entry_out_max]); row_idx += 1

        # Normalize % Low / High
        ttk.Label(secondary_output_frame, text="Clipped Output % Range:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        norm_perc_frame = ttk.Frame(secondary_output_frame)
        norm_perc_frame.grid(row=row_idx, column=1, sticky="w", padx=0, pady=0)
        
        lbl_norm_low = ttk.Label(norm_perc_frame, text="Low:")
        lbl_norm_low.pack(side=tk.LEFT, padx=(0,2))
        entry_norm_low = ttk.Entry(norm_perc_frame, textvariable=self.robust_norm_low_percentile, width=7)
        entry_norm_low.pack(side=tk.LEFT, padx=(0,10))
        _create_hover_tooltip(entry_norm_low, "robust_norm_low_percentile") # Add help_content.json entry
        
        lbl_norm_high = ttk.Label(norm_perc_frame, text="High:")
        lbl_norm_high.pack(side=tk.LEFT, padx=(0,2))
        entry_norm_high = ttk.Entry(norm_perc_frame, textvariable=self.robust_norm_high_percentile, width=7)
        entry_norm_high.pack(side=tk.LEFT, padx=(0,0))
        _create_hover_tooltip(entry_norm_high, "robust_norm_high_percentile") # Add help_content.json entry
        
        self.secondary_output_widgets_references.extend([lbl_norm_low, entry_norm_low, lbl_norm_high, entry_norm_high])
        self.widgets_to_disable_during_processing.extend([lbl_norm_low, entry_norm_low, lbl_norm_high, entry_norm_high]); row_idx += 1

        # Output Suffix
        lbl_robust_suffix = ttk.Label(secondary_output_frame, text="Output Suffix:")
        lbl_robust_suffix.grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry_robust_suffix = ttk.Entry(secondary_output_frame, textvariable=self.robust_output_suffix, width=18)
        entry_robust_suffix.grid(row=row_idx, column=1, padx=(0,2), pady=2, sticky="w")
        _create_hover_tooltip(entry_robust_suffix, "robust_output_suffix") # Add help_content.json entry
        self.secondary_output_widgets_references.extend([lbl_robust_suffix, entry_robust_suffix])
        self.widgets_to_disable_during_processing.extend([lbl_robust_suffix, entry_robust_suffix]); row_idx += 1

        ttk.Separator(secondary_output_frame, orient="horizontal").grid(
            row=row_idx, column=0, columnspan=2, sticky="ew", padx=5, pady=(6, 4)
        )
        row_idx += 1

        # Special mode: panelized hi-res refine using low-res raw global depth as anchor.
        self.spatial_refine_mode_cb = ttk.Checkbutton(
            secondary_output_frame,
            text="Special: Spatial Hi-Res Refine (Secondary Run)",
            variable=self.enable_spatial_refine_mode_var
        )
        self.spatial_refine_mode_cb.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.spatial_refine_mode_cb, "spatial_refine_mode")
        self.widgets_to_disable_during_processing.append(self.spatial_refine_mode_cb); row_idx += 1

        # Special mode: direct edge-guided upscaling from low-res depth cache.
        self.edge_guided_mode_cb = ttk.Checkbutton(
            secondary_output_frame,
            text="Special: Edge-Guided Hi-Res Upscale (Secondary Run)",
            variable=self.enable_edge_guided_upscale_mode_var
        )
        self.edge_guided_mode_cb.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.edge_guided_mode_cb, "edge_guided_upscale_mode")
        self.widgets_to_disable_during_processing.append(self.edge_guided_mode_cb); row_idx += 1

        self.spatial_refine_toggle_btn = ttk.Button(
            secondary_output_frame,
            text="  ↳ Configure Hi-Res / Edge Settings...",
            command=self.toggle_spatial_refine_options_visibility,
        )
        self.spatial_refine_toggle_btn.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=(15, 5), pady=2)
        _create_hover_tooltip(self.spatial_refine_toggle_btn, "hires_edge_open_settings")
        self.widgets_to_disable_during_processing.append(self.spatial_refine_toggle_btn); row_idx += 1

        self.spatial_refine_summary_label = ttk.Label(
            secondary_output_frame,
            text="  ↳ Tile/overlap, edge-guided fallback, standalone edge upscale, and output suffixes are in popup settings.",
            anchor="w",
        )
        self.spatial_refine_summary_label.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=(15, 5), pady=(0, 2))
        self.widgets_to_disable_during_processing.append(self.spatial_refine_summary_label); row_idx += 1

        ttk.Separator(secondary_output_frame, orient="horizontal").grid(
            row=row_idx, column=0, columnspan=2, sticky="ew", padx=5, pady=(6, 4)
        )
        row_idx += 1

        self.cloud_dispatch_mode_cb = ttk.Checkbutton(
            secondary_output_frame,
            text="Special: Cloud Dispatch Mode (Vast.ai, Sequential)",
            variable=self.enable_cloud_dispatch_mode_var
        )
        self.cloud_dispatch_mode_cb.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        self.widgets_to_disable_during_processing.append(self.cloud_dispatch_mode_cb); row_idx += 1

        self.cloud_settings_toggle_btn = ttk.Button(
            secondary_output_frame,
            text="  ↳ Configure Cloud Dispatch...",
            command=self.toggle_cloud_settings_visibility,
        )
        self.cloud_settings_toggle_btn.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=(15, 5), pady=2)
        self.widgets_to_disable_during_processing.append(self.cloud_settings_toggle_btn); row_idx += 1

        self.cloud_summary_label = ttk.Label(
            secondary_output_frame,
            textvariable=self.cloud_secondary_summary_var,
            anchor="w",
        )
        self.cloud_summary_label.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=(15, 5), pady=(0, 2))
        self.widgets_to_disable_during_processing.append(self.cloud_summary_label); row_idx += 1

        self._apply_spatial_refine_options_visibility()
        self._apply_cloud_options_visibility()

        # --- Progress Bar and Status ---
        progress_bar_frame = ttk.Frame(self.root)
        progress_bar_frame.pack(pady=(10, 0), padx=10, fill="x", expand=False)
        
        self.progress = ttk.Progressbar(progress_bar_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(fill=tk.X, expand=True, padx=0, pady=0)

        # Status Label (NEW)
        self.style.configure("Status.TLabel", anchor="center") 
        self.status_label = ttk.Label(progress_bar_frame, text="Ready")
        self.status_label.pack(padx=0, pady=2)

        # --- Control Buttons ---
        ctrl_frame = ttk.Frame(self.root)
        ctrl_frame.pack(pady=(5, 10), padx=10, fill="x", expand=False)

        # --- Container frame for buttons to center them ---
        button_container_frame = ttk.Frame(ctrl_frame)
        button_container_frame.pack(anchor="center") # Centers the button_container_frame within ctrl_frame

        # --- Current Processing Information Frame ---
        processing_info_frame = ttk.LabelFrame(self.root, text="Current Processing Information")
        processing_info_frame.pack(fill="x", padx=10, pady=5, expand=False)
        
        # Grid layout for labels inside this frame
        processing_info_frame.columnconfigure(0, weight=0) # Labels
        processing_info_frame.columnconfigure(1, weight=1) # Values

        row_idx = 0
        # Filename
        ttk.Label(processing_info_frame, text="Filename:").grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
        lbl_filename = ttk.Label(processing_info_frame, textvariable=self.current_filename_var, anchor=tk.W)
        lbl_filename.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=2)
        _create_hover_tooltip(lbl_filename, "current_filename") # Add tooltip
        row_idx += 1

        # Resolution
        ttk.Label(processing_info_frame, text="Resolution:").grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
        lbl_resolution = ttk.Label(processing_info_frame, textvariable=self.current_resolution_var, anchor=tk.W)
        lbl_resolution.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=2)
        _create_hover_tooltip(lbl_resolution, "current_resolution") # Add tooltip
        row_idx += 1

        # Frames
        ttk.Label(processing_info_frame, text="Frames:").grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
        lbl_frames = ttk.Label(processing_info_frame, textvariable=self.current_frames_var, anchor=tk.W)
        lbl_frames.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=2)
        _create_hover_tooltip(lbl_frames, "current_frames") # Add tooltip
        row_idx += 1

        start_frame = ttk.Frame(button_container_frame); start_frame.pack(side=tk.LEFT, padx=(0,2))
        self.start_button = ttk.Button(start_frame, text="Start", command=self.start_thread, width=10)
        self.start_button.pack(side=tk.LEFT)
        _create_hover_tooltip(self.start_button, "start_button")        

        cancel_frame = ttk.Frame(button_container_frame); cancel_frame.pack(side=tk.LEFT, padx=(2,2))
        self.cancel_button = ttk.Button(cancel_frame, text="Cancel", command=self.stop_processing, width=10, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT)
        _create_hover_tooltip(self.cancel_button, "cancel_button")

        remerge_frame = ttk.Frame(button_container_frame); remerge_frame.pack(side=tk.LEFT, padx=(2,2))
        self.remerge_button = ttk.Button(remerge_frame, text="Re-Merge Segments", command=self.re_merge_from_gui, width=18)
        self.remerge_button.pack(side=tk.LEFT)
        _create_hover_tooltip(self.remerge_button, "remerge_button")

        genvis_frame = ttk.Frame(button_container_frame); genvis_frame.pack(side=tk.LEFT, padx=(2,2))
        self.generate_visuals_button = ttk.Button(genvis_frame, text="Generate Seg Visuals", command=self.generate_segment_visuals_from_gui, width=20)
        self.generate_visuals_button.pack(side=tk.LEFT)
        _create_hover_tooltip(self.generate_visuals_button, "generate_visuals_button")

        self.widgets_to_disable_during_processing.extend([
            self.start_button, self.remerge_button,
            self.generate_visuals_button
        ])

        # self.toggle_merge_related_options_active_state()

    def generate_segment_visuals_from_gui(self):
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Busy", "Another process is running. Please wait."); return
        meta_file = filedialog.askopenfilename(title="Select Master Metadata JSON for Segment Visual Generation", filetypes=[("JSON files", "*.json"), ("All files", "*.*")], initialdir=self.output_dir.get())
        if not meta_file: 
            _logger.info("Segment visual generation cancelled: No master metadata file selected.")
            return
        vis_fmt = self.keep_intermediate_segment_visual_format_var.get()
        if vis_fmt == "none": 
            messagebox.showinfo("Info", "Segment Visual Format is 'none'. Select a valid format."); return
        if not messagebox.askyesno("Generate/Overwrite Visuals?", f"Generate '{vis_fmt}' visuals for segments in '{os.path.basename(meta_file)}'?\nThis may overwrite existing visuals."):
            _logger.info("Segment visual generation cancelled by user.")
            return
        args = {"master_meta_path": meta_file, "visual_format_to_generate": vis_fmt}
        _logger.info(f"--- Starting Segment Visual Generation for: {os.path.basename(meta_file)} (Format: {vis_fmt}) ---")
        self._set_ui_processing_state(True)
        self.processing_thread = threading.Thread(target=self._execute_generate_segment_visuals_wrapper, args=(args,), daemon=True); self.processing_thread.start()
        self.root.after(100, self.process_queue)

    def load_config(self):
        if os.path.exists(self.CONFIG_FILENAME):
            try:
                with open(self.CONFIG_FILENAME, "r") as f: config = json.load(f)
                if (
                    "spatial_refine_tile_num_var" in config
                    and "spatial_refine_tile_num_y_var" not in config
                ):
                    config["spatial_refine_tile_num_y_var"] = config["spatial_refine_tile_num_var"]
                if (
                    "spatial_refine_tile_overlap_var" in config
                    and "spatial_refine_tile_overlap_x_var" not in config
                ):
                    config["spatial_refine_tile_overlap_x_var"] = config["spatial_refine_tile_overlap_var"]
                if (
                    "spatial_refine_tile_overlap_var" in config
                    and "spatial_refine_tile_overlap_y_var" not in config
                ):
                    config["spatial_refine_tile_overlap_y_var"] = config["spatial_refine_tile_overlap_var"]
                if (
                    "stereopilot_frame_count_var" in config
                    and "stereopilot_window_size_var" not in config
                ):
                    config["stereopilot_window_size_var"] = config["stereopilot_frame_count_var"]
                config.pop("stereopilot_frame_count_var", None)
                legacy_cloud_image = str(config.get("cloud_image_var", "")).strip()
                if legacy_cloud_image in self._legacy_cloud_image_migrations:
                    migrated_image = self._legacy_cloud_image_migrations[legacy_cloud_image]
                    config["cloud_image_var"] = migrated_image
                    _logger.info(
                        "GUI: Migrated legacy cloud image '%s' -> '%s'.",
                        legacy_cloud_image,
                        migrated_image,
                    )
                loaded_settings_for_tkvars = {k: v for k, v in config.items() if k in self.all_tk_vars}
                for key, value in loaded_settings_for_tkvars.items():
                    if key in self.all_tk_vars:
                        try: self.all_tk_vars[key].set(value)
                        except tk.TclError: 
                            _logger.warning(f"Warning (GUI load_config): Could not set var {key} during early config load.")
                
                self.last_settings_dir = config.get(self.LAST_SETTINGS_DIR_CONFIG_KEY, os.getcwd())
                
                self.current_input_mode = config.get("current_input_mode", "batch_folder")
                self.single_file_mode_active = config.get("single_file_mode_active", False)
                
                _logger.info(f"GUI: Configuration loaded from '{self.CONFIG_FILENAME}'.")
            except Exception as e:
                _logger.warning(f"Warning (GUI load_config): Could not load config '{self.CONFIG_FILENAME}': {e}")
                self.last_settings_dir = os.getcwd()
                self.current_input_mode = "batch_folder"
                self.single_file_mode_active = False
        else: 
            self.last_settings_dir = os.getcwd()
            self.current_input_mode = "batch_folder"
            self.single_file_mode_active = False
            _logger.info(f"GUI: Configuration file '{self.CONFIG_FILENAME}' not found. Using default settings.")

    def on_close(self):
        self._close_spatial_refine_settings_dialog()
        self._close_geometry_settings_dialog()
        self._close_cloud_settings_dialog()
        self.save_config()
        if self.processing_thread and self.processing_thread.is_alive():
            _logger.info("Stopping processing before exit...")
            self.stop_event.set()
            if self.active_external_process is not None and self.active_external_process.poll() is None:
                try:
                    self.active_external_process.terminate()
                except Exception:
                    pass
            self.processing_thread.join(timeout=10)
            if self.processing_thread.is_alive(): 
                _logger.warning("Processing thread did not terminate gracefully. Forcing exit.")
        
        self.root.destroy()

    def process_queue(self):
        # The message queue is still used for progress bar updates
        while not self.message_queue.empty():
            try:
                msg_type, content = self.message_queue.get_nowait()
                if msg_type == "progress":
                    self.progress["value"] = content
                elif msg_type == "status":
                    self.status_message_var.set(content)
                elif msg_type == "set_ui_state":
                    self._set_ui_processing_state(content)
            except queue.Empty:
                break
            except Exception as e:
                _logger.exception(f"Error processing GUI queue: {e}")
        
        self.root.after(100, self.process_queue)

    def re_merge_from_gui(self):
        if not merge_depth_segments: 
            messagebox.showerror("Error", "Merge module not available."); return
        meta_file = filedialog.askopenfilename(title="Select Master Metadata JSON for Re-Merging", filetypes=[("JSON files", "*.json"), ("All files", "*.*")], initialdir=self.output_dir.get())
        if not meta_file: return
        
        _logger.debug(f"DEBUG (re_merge_from_gui): enable_dual_output_robust_norm.get() is {self.enable_dual_output_robust_norm.get()}")
        
        base_name_from_meta = os.path.splitext(os.path.basename(meta_file))[0].replace("_master_meta", "")
        output_suffix = self.merge_output_suffix_var.get()
        remerge_base_name = f"{base_name_from_meta}{output_suffix}"

        out_fmt = self.merge_output_format_var.get()
        
        def_ext_fmt = out_fmt
        if out_fmt == "main10_mp4":
            def_ext_fmt = "mp4"
        elif out_fmt in ["png_sequence", "exr_sequence"]:
            def_ext_fmt = ""
        elif out_fmt == "exr":
            def_ext_fmt = "exr"

        def_ext = f".{def_ext_fmt}" if def_ext_fmt else ""

        ftypes_map = {
            "mp4": [("MP4 (H.264 8-bit)", "*.mp4")],
            "main10_mp4": [("MP4 (HEVC 10-bit)", "*.mp4")],
            "png_sequence": [("PNG Seq (Select Folder)", "")],
            "exr_sequence": [("EXR Seq (Select Folder)", "")],
            "exr": [("EXR File", "*.exr")]
        }
        curr_ftypes = ftypes_map.get(out_fmt, []) + [("All files", "*.*")]
        out_path = None

        if "sequence" in out_fmt:
            parent_dir = filedialog.askdirectory(title=f"Select Parent Dir for Re-Merged {out_fmt.upper()} Sequence...", initialdir=self.output_dir.get())
            if parent_dir: out_path = parent_dir
        else:
            initial_filename_for_dialog_actual = f"{remerge_base_name}{def_ext}"

            out_path = filedialog.asksaveasfilename(
                title=f"Save Re-Merged {out_fmt.upper()} As...", 
                initialdir=self.output_dir.get(), 
                initialfile=f"{remerge_base_name}{def_ext}",
                defaultextension=def_ext, 
                filetypes=curr_ftypes
            )

        if not out_path: 
            _logger.info("Re-merge cancelled: No output path selected.")
            return
            
        align_method = "linear_blend" if self.merge_alignment_method_var.get() == "Linear Blend" else "shift_scale"
        
        args = {"master_meta_path": meta_file, "output_path_arg": out_path,
                "do_dithering": self.merge_dither_var.get(), "dither_strength_factor": self.merge_dither_strength_var.get(),
                "apply_gamma_correction": self.merge_gamma_correct_var.get(), "gamma_value": self.merge_gamma_value_var.get(),
                "use_percentile_norm": self.merge_percentile_norm_var.get(), "norm_low_percentile": self.merge_norm_low_perc_var.get(),
                "norm_high_percentile": self.merge_norm_high_perc_var.get(), "output_format": out_fmt,
                "merge_alignment_method": align_method,
                "output_filename_override_base": remerge_base_name,
                "enable_dual_output_robust_norm": self.enable_dual_output_robust_norm.get(),
                "robust_norm_low_percentile": self.robust_norm_low_percentile.get(),
                "robust_norm_high_percentile": self.robust_norm_high_percentile.get(),
                "robust_norm_output_min": self.robust_norm_output_min.get(),
                "robust_norm_output_max": self.robust_norm_output_max.get(),
                "robust_output_suffix": self.robust_output_suffix.get(),
                "is_depth_far_black": self.is_depth_far_black.get()
                }

        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Busy", "Another process is running. Please wait."); return
        
        _logger.info(f"--- Starting Re-Merge for: {os.path.basename(meta_file)} ---")
        self._set_ui_processing_state(True)
        self.processing_thread = threading.Thread(target=self._execute_re_merge_wrapper, args=(args,), daemon=True); self.processing_thread.start()
        self.root.after(100, self.process_queue)

    def start_thread(self):
        if self.processing_thread and self.processing_thread.is_alive():
            _logger.warning("Processing is already running.")
            return

        input_path_str = self.input_dir_or_file_var.get()
        if not input_path_str or not os.path.exists(input_path_str):
            _logger.error(f"GUI: Input path field is empty or path does not exist: {input_path_str}")
            messagebox.showerror("Error", f"Input path does not exist: {input_path_str}")
            return

        if self.npz_backfill_missing_only_var.get() and not self.process_as_segments_var.get():
            messagebox.showerror(
                "Invalid Setting",
                "Special mode 'Backfill Missing NPZ Only' requires 'Process as Segments' to be enabled."
            )
            _logger.error("Start blocked: NPZ backfill mode requires segment processing.")
            return

        npz_backfill_mode = bool(self.npz_backfill_missing_only_var.get())
        spatial_refine_mode = bool(self.enable_spatial_refine_mode_var.get())
        edge_guided_mode = bool(self.enable_edge_guided_upscale_mode_var.get())
        cloud_dispatch_mode = bool(self.enable_cloud_dispatch_mode_var.get())
        selected_special_modes = int(npz_backfill_mode) + int(spatial_refine_mode) + int(edge_guided_mode)
        if selected_special_modes > 1:
            messagebox.showerror(
                "Invalid Setting",
                "Select only one special mode at a time: NPZ backfill OR Spatial Hi-Res Refine OR Edge-Guided Hi-Res Upscale."
            )
            _logger.error("Start blocked: conflicting special modes selected.")
            return
        if cloud_dispatch_mode and selected_special_modes > 0:
            messagebox.showerror(
                "Invalid Setting",
                "Cloud Dispatch mode cannot be combined with NPZ backfill, Spatial Hi-Res Refine, or Edge-Guided Hi-Res mode."
            )
            _logger.error("Start blocked: cloud mode selected with local special modes.")
            return
        if spatial_refine_mode and run_spatial_hires_refine is None:
            messagebox.showerror(
                "Unavailable",
                "Spatial Hi-Res Refine module could not be imported. Check console for details."
            )
            _logger.error("Start blocked: spatial_refine module unavailable.")
            return

        if edge_guided_mode and run_edge_guided_hires_upscale is None:
            messagebox.showerror(
                "Unavailable",
                "Edge-Guided Hi-Res Upscale module could not be imported. Check console for details."
            )
            _logger.error("Start blocked: edge_guided_upscale module unavailable.")
            return

        if spatial_refine_mode or edge_guided_mode:
            try:
                hires_w = int(self.spatial_refine_target_width_var.get())
                hires_h = int(self.spatial_refine_target_height_var.get())
            except (ValueError, tk.TclError):
                messagebox.showerror("Error", "Hi-res target width/height contain invalid values.")
                _logger.error("Start blocked: invalid hi-res target dimensions.")
                return
            if hires_w <= 0 or hires_h <= 0:
                messagebox.showerror("Error", "Hi-Res Width/Height must be >0.")
                _logger.error("Start blocked: invalid hi-res target dimensions range.")
                return

        if spatial_refine_mode:
            try:
                tile_num_x = int(self.spatial_refine_tile_num_var.get())
                tile_num_y = int(self.spatial_refine_tile_num_y_var.get())
                tile_overlap_x = int(self.spatial_refine_tile_overlap_x_var.get())
                tile_overlap_y = int(self.spatial_refine_tile_overlap_y_var.get())
                anchor_weight = float(self.spatial_refine_anchor_weight_var.get())
                local_window_size = int(self.spatial_refine_local_window_size_var.get())
                local_window_stride = int(self.spatial_refine_local_window_stride_var.get())
                local_conf_low = float(self.spatial_refine_local_confidence_low_var.get())
                local_conf_high = float(self.spatial_refine_local_confidence_high_var.get())
                edge_fallback_mix = float(self.spatial_refine_edge_fallback_mix_var.get())
            except (ValueError, tk.TclError):
                messagebox.showerror("Error", "Spatial refine settings contain invalid values.")
                _logger.error("Start blocked: invalid spatial refine parameter types.")
                return

            if (
                tile_num_x < 1
                or tile_num_y < 1
                or tile_overlap_x < 0
                or tile_overlap_y < 0
                or local_window_size <= 0
                or local_window_stride <= 0
                or not (0.0 <= anchor_weight <= 1.0)
                or not (0.0 <= local_conf_low <= 1.0)
                or not (0.0 <= local_conf_high <= 1.0)
                or local_conf_high <= local_conf_low
                or not (0.0 <= edge_fallback_mix <= 1.0)
            ):
                messagebox.showerror(
                    "Error",
                    "Spatial refine values must satisfy: Tile Grid X/Y >=1, Overlap X/Y >=0, "
                    "Hi-Res Width/Height >0, Anchor Blend 0.0-1.0, "
                    "Local Window Size/Stride >0, Confidence Low/High in [0,1], High > Low, and Edge Fallback Mix in [0,1]."
                )
                _logger.error("Start blocked: invalid spatial refine parameter ranges.")
                return
            if (tile_overlap_x % 64) != 0 or (tile_overlap_y % 64) != 0:
                messagebox.showerror(
                    "Error",
                    "Spatial refine Overlap X and Overlap Y must be multiples of 64."
                )
                _logger.error("Start blocked: overlap X/Y not aligned to 64px increments.")
                return

        if spatial_refine_mode and not bool(self.spatial_refine_use_edge_fallback_var.get()):
            use_edge_param_validation = False
        else:
            use_edge_param_validation = bool(spatial_refine_mode and bool(self.spatial_refine_use_edge_fallback_var.get())) or bool(edge_guided_mode)

        if use_edge_param_validation:
            try:
                edge_strength = float(self.edge_guided_strength_var.get())
                edge_sigma_color = float(self.edge_guided_sigma_color_var.get())
                edge_sigma_spatial = float(self.edge_guided_sigma_spatial_var.get())
                edge_iters = int(self.edge_guided_iterations_var.get())
                edge_temporal = float(self.edge_guided_temporal_smooth_var.get())
                edge_reinject = float(self.edge_guided_reinject_strength_var.get())
            except (ValueError, tk.TclError):
                messagebox.showerror("Error", "Edge-guided settings contain invalid values.")
                _logger.error("Start blocked: invalid edge-guided parameter types.")
                return

            if (
                not (0.0 <= edge_strength <= 1.0)
                or edge_sigma_color <= 0.0
                or edge_sigma_spatial <= 0.0
                or edge_iters < 0
                or edge_iters > 8
                or not (0.0 <= edge_temporal <= 1.0)
                or not (0.0 <= edge_reinject <= 1.0)
            ):
                messagebox.showerror(
                    "Error",
                    "Edge-guided values must satisfy: Strength in [0,1], Sigma Color/Spatial >0, "
                    "Bilateral Iterations in [0,8], Temporal Smooth in [0,1], and Reinject Strength in [0,1]."
                )
                _logger.error("Start blocked: invalid edge-guided parameter ranges.")
                return
        
        # --- ADD THESE LINES HERE ---
        _logger.info("Scanning input folder: Please wait...")
        self.status_message_var.set("Scanning input folder...")
        self.root.update_idletasks() # Force GUI update to show "Scanning..." immediately
        # ----------------------------

        determined_mode, determined_single_source = self._determine_input_mode_from_path(input_path_str)
        
        self.current_input_mode = determined_mode
        self.single_file_mode_active = determined_single_source

        if not os.path.exists(input_path_str):
            _logger.error(f"GUI: Input path is invalid or does not exist: {input_path_str}")
            messagebox.showerror("Error", f"Input path does not exist: {input_path_str}")
            return

        sources_to_process_specs = []

        if self.single_file_mode_active:
            self.effective_move_original_on_completion = False
            basename = ""
            if self.current_input_mode == "image_sequence_folder":
                basename = os.path.basename(input_path_str)
            else:
                basename = os.path.splitext(os.path.basename(input_path_str))[0]
            
            sources_to_process_specs.append({
                "path": input_path_str,
                "type": self.current_input_mode, 
                "basename": basename
            })
        else:
            self.effective_move_original_on_completion = self.MOVE_ORIGINAL_TO_FINISHED_FOLDER_ON_COMPLETION
            if self.current_input_mode == "batch_folder":
                try:
                    for item_name in sort_paths_by_clip_id(os.listdir(input_path_str)):
                        item_full_path = os.path.join(input_path_str, item_name)
                        if os.path.isfile(item_full_path):
                            ext = os.path.splitext(item_name)[1].lower()
                            if any(ext in vid_ext.replace("*", "") for vid_ext in self.VIDEO_EXTENSIONS):
                                basename = os.path.splitext(item_name)[0]
                                sources_to_process_specs.append({
                                    "path": item_full_path,
                                    "type": "video_file",
                                    "basename": basename
                                })
                        elif os.path.isdir(item_full_path):
                            if self._is_image_sequence_folder(item_full_path):
                                basename = item_name 
                                sources_to_process_specs.append({
                                    "path": item_full_path,
                                    "type": "image_sequence_folder",
                                    "basename": basename
                                })
                except NotADirectoryError:
                    _logger.error(f"GUI Input: Path '{input_path_str}' is not a directory, but batch processing mode was attempted.")
                    messagebox.showerror("Error", f"Input path is not a directory for batch processing: {input_path_str}")
                    return
                except OSError as e:
                    _logger.error(f"GUI Input: OS error when trying to list directory '{input_path_str}'. Error: {e}")
                    messagebox.showerror("Error", f"Could not read directory contents for '{input_path_str}':\n{e}")
                    return
            else:
                _logger.critical(f"GUI Start Thread: Unexpected mode '{self.current_input_mode}' for path '{input_path_str}' after explicit determination. This indicates a logic error.")
                messagebox.showerror("Internal Error", f"Unexpected input mode '{self.current_input_mode}' for path '{input_path_str}'. Please report this.")
                return

        # Ensure deterministic ordering by parsed clip id (with lexical fallback).
        sources_to_process_specs.sort(key=lambda spec: clip_sort_key(spec.get("basename", spec.get("path", ""))))


        if not sources_to_process_specs:
            _logger.warning(f"GUI: No valid video files or image sequences found in '{input_path_str}' for mode '{self.current_input_mode}'.")
            return

        if cloud_dispatch_mode:
            non_video_sources = [
                spec for spec in sources_to_process_specs
                if spec.get("type") not in ("video_file", "single_video_file")
            ]
            if non_video_sources:
                messagebox.showerror(
                    "Unsupported Input",
                    "Cloud Dispatch currently supports video clips only. "
                    "Image sequence and single-image inputs are not supported in cloud mode yet."
                )
                _logger.error("Start blocked: cloud mode received non-video input source(s).")
                return

        selected_backend = str(self.model_backend_var.get() or "depthcrafter").strip().lower()
        if selected_backend == "stereopilot":
            if self.process_as_segments_var.get():
                messagebox.showerror(
                    "Unsupported Option",
                    "StereoPilot backend currently supports full-video processing only. "
                    "Disable 'Process as Segments'."
                )
                _logger.error("Start blocked: StereoPilot does not support segmented mode.")
                return
            if self.npz_backfill_missing_only_var.get():
                messagebox.showerror(
                    "Unsupported Option",
                    "StereoPilot backend does not support NPZ backfill mode."
                )
                _logger.error("Start blocked: StereoPilot does not support NPZ backfill mode.")
                return
            if self.enable_spatial_refine_mode_var.get() or self.enable_edge_guided_upscale_mode_var.get():
                messagebox.showerror(
                    "Unsupported Option",
                    "StereoPilot backend does not support spatial/edge hi-res special modes."
                )
                _logger.error("Start blocked: StereoPilot with spatial/edge special mode is unsupported.")
                return
            non_video_sources = [
                spec for spec in sources_to_process_specs
                if spec.get("type") not in ("video_file", "single_video_file")
            ]
            if non_video_sources:
                messagebox.showerror(
                    "Unsupported Input",
                    "StereoPilot backend currently supports video inputs only."
                )
                _logger.error("Start blocked: StereoPilot received non-video input source(s).")
                return
        
        # --- NEW SEED GENERATION GUARD ---
        gui_seed_setting = self.seed.get()
        effective_seed_for_run = gui_seed_setting
        if effective_seed_for_run < 0:
            effective_seed_for_run = random.randint(0, 2**32 - 1)
            _logger.debug(f"GUI: Seed was set to {gui_seed_setting} (negative). Generating a new random seed for this run: {effective_seed_for_run}")
        else:
            _logger.debug(f"GUI: Using user-specified seed: {effective_seed_for_run}")
        
        # --- PHASE 1: FAST SCAN FOR TOTAL SOURCES (REPLACING HEAVY METADATA/SEGMENT DEFINITION) ---
        final_jobs_to_process_sources = sources_to_process_specs # The list of file/folder specs

        if final_jobs_to_process_sources:
            # --- ADDED THESE LINES TO RESET PREVIOUS JOB INFO ---
            self.current_filename_var.set("N/A")
            self.current_resolution_var.set("N/A")
            self.current_frames_var.set("N/A")
            # --------------------------------------------------
            self.status_message_var.set(f"Starting processing {len(final_jobs_to_process_sources)} files/sequences...")
            self.progress["value"] = 0 # Initialize progress bar value
            self.progress["maximum"] = len(final_jobs_to_process_sources) # Set progress bar max to total files/sources
            self._set_ui_processing_state(True)
            
            # Pass the list of source specs. The ffprobe/segment definition will happen inside start_processing.
            self.processing_thread = threading.Thread(target=self._start_processing_wrapper, 
                                                      args=(final_jobs_to_process_sources, effective_seed_for_run), 
                                                      daemon=True)
            self.processing_thread.start()
            self.root.after(100, self.process_queue) # Start queue processing for progress updates
        else:
            _logger.info("No videos/segments to process after considering existing data and user choices (or all skipped).")

    def start_processing(self, source_specs_to_process, effective_seed_for_run):
        self.stop_event.clear()
        npz_backfill_mode_active = self.npz_backfill_missing_only_var.get()
        spatial_refine_mode_active = self.enable_spatial_refine_mode_var.get()
        edge_guided_mode_active = self.enable_edge_guided_upscale_mode_var.get()
        cloud_dispatch_mode_active = self.enable_cloud_dispatch_mode_var.get()
        
        # Progress max is already set to len(source_specs_to_process) in start_thread
        _logger.debug(f"Starting lazy batch processing for {len(source_specs_to_process)} sources...")
        if npz_backfill_mode_active:
            _logger.info("Special Mode Active: Backfill Missing NPZ Only.")
        if spatial_refine_mode_active:
            _logger.info("Special Mode Active: Spatial Hi-Res Refine (Secondary Run).")
        if edge_guided_mode_active:
            _logger.info("Special Mode Active: Edge-Guided Hi-Res Upscale (Secondary Run).")
        if cloud_dispatch_mode_active:
            _logger.info("Special Mode Active: Cloud Dispatch (Vast.ai Sequential Jobs).")
        self.status_message_var.set("Starting processing...")

        if cloud_dispatch_mode_active:
            try:
                self._run_cloud_dispatch_mode(
                    source_specs_to_process=source_specs_to_process,
                    effective_seed_for_run=effective_seed_for_run,
                )
            except Exception as cloud_exc:
                _logger.exception(f"Cloud dispatch exception: {cloud_exc}")
                self.status_message_var.set(f"Cloud Error: {cloud_exc.__class__.__name__}")
            return

        # Initialize a dict to store master metadata for each video/sequence path
        all_videos_master_metadata = {}
        base_job_info_map = {} # Map to store base_job_info for each video path

        try:
            demo = None
            # 1. Initialize DepthCrafterDemo only for modes that run model inference.
            if edge_guided_mode_active:
                _logger.info("Edge-guided special mode does not require DepthCrafter model initialization.")
            else:
                if not self.use_local_models_only_var.get():
                    _logger.info("Attempting to check model at Hugging Face Hub against local.")
                else:
                    _logger.info("Attempting to load local model.")

                disable_xformers_for_run = self.disable_xformers_var.get()
                selected_backend = str(self.model_backend_var.get()).strip().lower()
                if selected_backend in ("geometrycrafter_diff", "geometrycrafter_determ"):
                    demo = GeometryCrafterDemo(
                        model_backend=selected_backend,
                        geometry_model_path=self.geometry_model_path_var.get().strip() or "TencentARC/GeometryCrafter",
                        geometry_repo_path=self.geometry_repo_path_var.get().strip(),
                        geometry_cache_dir=self.geometry_cache_dir_var.get().strip(),
                        geometry_decode_chunk_size=max(1, int(self.geometry_decode_chunk_size_var.get())),
                        geometry_low_memory_usage=bool(self.geometry_low_memory_usage_var.get()),
                        geometry_force_projection=bool(self.geometry_force_projection_var.get()),
                        geometry_force_fixed_focal=bool(self.geometry_force_fixed_focal_var.get()),
                        geometry_use_extract_interp=bool(self.geometry_use_extract_interp_var.get()),
                        pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
                        cpu_offload=self.cpu_offload.get(),
                        use_cudnn_benchmark=self.use_cudnn_benchmark.get(),
                        local_files_only=self.use_local_models_only_var.get(),
                        disable_xformers=disable_xformers_for_run,
                    )
                    _logger.info(
                        "GeometryCrafter backend initialized (%s). Starting source processing loop.",
                        selected_backend,
                    )
                elif selected_backend == "stereopilot":
                    demo = StereoPilotDemo(
                        model_backend=selected_backend,
                        stereopilot_model_path=self.stereopilot_model_path_var.get().strip() or "KlingTeam/StereoPilot",
                        stereopilot_base_model_path=self.stereopilot_base_model_path_var.get().strip() or "Wan-AI/Wan2.1-T2V-1.3B",
                        stereopilot_repo_path=self.stereopilot_repo_path_var.get().strip(),
                        stereopilot_cache_dir=self.stereopilot_cache_dir_var.get().strip(),
                        stereopilot_prompt_default=self.stereopilot_prompt_var.get().strip(),
                        stereopilot_use_sidecar_prompt=bool(self.stereopilot_use_sidecar_prompt_var.get()),
                        stereopilot_output_mode=self.stereopilot_output_mode_var.get().strip() or "side_by_side",
                        stereopilot_target_width=max(32, int(self.stereopilot_target_width_var.get())),
                        stereopilot_target_height=max(32, int(self.stereopilot_target_height_var.get())),
                        stereopilot_target_fps=max(1.0, float(self.stereopilot_target_fps_var.get())),
                        stereopilot_sampling_steps=max(1, int(self.stereopilot_sampling_steps_var.get())),
                        stereopilot_guide_scale=float(self.stereopilot_guide_scale_var.get()),
                        stereopilot_shift=float(self.stereopilot_shift_var.get()),
                        stereopilot_domain_label=1 if int(self.stereopilot_domain_label_var.get()) != 0 else 0,
                        stereopilot_dtype=self.stereopilot_dtype_var.get().strip() or "bfloat16",
                        stereopilot_transformer_dtype=self.stereopilot_transformer_dtype_var.get().strip() or "float8",
                    )
                    _logger.info("StereoPilot backend initialized. Starting source processing loop.")
                else:
                    demo = DepthCrafterDemo(
                        unet_path="tencent/DepthCrafter",
                        pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
                        cpu_offload=self.cpu_offload.get(),
                        use_cudnn_benchmark=self.use_cudnn_benchmark.get(),
                        local_files_only=self.use_local_models_only_var.get(),
                        disable_xformers=disable_xformers_for_run,
                    )
                    _logger.info("DepthCrafter model initialized. Starting source processing loop.")
        except Exception as e:
            _logger.exception(f"CRITICAL: Failed to initialize inference backend: {e}")
            self.status_message_var.set(f"Error: Model initialization failed. See console.")
            self.current_filename_var.set("N/A")
            self.current_resolution_var.set("N/A")
            self.current_frames_var.set("N/A")
            return 
        
        total_sources_processed = 0
        
        # 2. Main Loop: Process one source (file/folder) at a time
        for source_idx, source_spec in enumerate(source_specs_to_process):
            if self.stop_event.is_set():
                _logger.info("Processing cancelled by user.")
                self.status_message_var.set("Cancelled.")
                break

            current_video_path = source_spec["path"] 
            original_basename = source_spec["basename"]
            current_gui_mode = source_spec["type"] 
            clip_gpu_tracking_enabled = self._reset_gpu_peak_tracking_for_clip(original_basename)
            
            gui_fps_setting = self.target_fps.get()
            gui_len_setting = self.process_length.get()
            gui_win_setting = self.window_size.get()
            gui_ov_setting = self.overlap.get()

            log_msg_base = f"Source {source_idx+1}/{len(source_specs_to_process)}: {original_basename}"

            if spatial_refine_mode_active:
                self.current_filename_var.set(f"{original_basename} (Hi-Res Spatial Refine)")
                self.status_message_var.set(f"Hi-Res Refine {source_idx+1} of {len(source_specs_to_process)}")
                self.root.update_idletasks()
                try:
                    _logger.info(f"Launching Hi-Res Spatial Refine for source '{original_basename}'...")
                    success_refine, output_or_reason = self._process_spatial_refine_source(
                        demo=demo,
                        source_spec=source_spec,
                        effective_seed_for_run=effective_seed_for_run,
                    )
                    if success_refine:
                        _logger.info(f"Hi-Res Spatial Refine complete for {original_basename}: {output_or_reason}")
                    else:
                        _logger.warning(f"Hi-Res Spatial Refine skipped/failed for {original_basename}: {output_or_reason}")
                except Exception as e_refine:
                    _logger.exception(f"Hi-Res Spatial Refine exception for {original_basename}: {e_refine}")
                    self.status_message_var.set(f"Hi-Res Refine Error: {e_refine.__class__.__name__} for {original_basename}")

                if clip_gpu_tracking_enabled:
                    self._log_gpu_peak_tracking_summary_for_clip(original_basename)
                total_sources_processed += 1
                self.message_queue.put(("progress", total_sources_processed))
                continue

            if edge_guided_mode_active:
                self.current_filename_var.set(f"{original_basename} (Edge-Guided Hi-Res Upscale)")
                self.status_message_var.set(f"Edge Upscale {source_idx+1} of {len(source_specs_to_process)}")
                self.root.update_idletasks()
                try:
                    _logger.info(f"Launching Edge-Guided Hi-Res Upscale for source '{original_basename}'...")
                    success_edge, output_or_reason = self._process_edge_guided_upscale_source(
                        source_spec=source_spec,
                    )
                    if success_edge:
                        _logger.info(f"Edge-Guided Hi-Res Upscale complete for {original_basename}: {output_or_reason}")
                    else:
                        _logger.warning(f"Edge-Guided Hi-Res Upscale skipped/failed for {original_basename}: {output_or_reason}")
                except Exception as e_edge:
                    _logger.exception(f"Edge-Guided Hi-Res Upscale exception for {original_basename}: {e_edge}")
                    self.status_message_var.set(f"Edge Upscale Error: {e_edge.__class__.__name__} for {original_basename}")

                if clip_gpu_tracking_enabled:
                    self._log_gpu_peak_tracking_summary_for_clip(original_basename)
                total_sources_processed += 1
                self.message_queue.put(("progress", total_sources_processed))
                continue

            _logger.debug(f"--- Defining Jobs for {log_msg_base}...")
            self.status_message_var.set(f"Defining jobs for {source_idx+1} of {len(source_specs_to_process)}")
            self.root.update_idletasks()
            
            # Determine source type for define_video_segments
            source_type_for_define = ""
            if current_gui_mode == "single_video_file" or current_gui_mode == "video_file":
                source_type_for_define = "video_file"
            elif current_gui_mode == "image_sequence_folder":
                source_type_for_define = "image_sequence_folder"
            elif current_gui_mode == "single_image_file":
                source_type_for_define = "single_image_file"
            else:
                _logger.error(f"Lazy Job Definition: Unknown source_spec type '{current_gui_mode}' for basename '{original_basename}'. Skipping.")
                total_sources_processed += 1
                self.message_queue.put(("progress", total_sources_processed))
                continue

            # A. FFPROBE/METADATA EXTRACTION (Heavy lifting happens here - Phase 2 start)
            try:
                all_potential_segments_for_video, base_job_info_initial = define_video_segments(
                    video_path_or_folder=current_video_path,
                    original_basename=original_basename,
                    gui_target_fps_setting=gui_fps_setting,
                    gui_process_length_overall=gui_len_setting,
                    gui_segment_output_window_frames=gui_win_setting,
                    gui_segment_output_overlap_frames=gui_ov_setting,
                    source_type=source_type_for_define,
                    gui_target_height_setting=self.target_height.get(),
                    gui_target_width_setting=self.target_width.get(),
                )            
            except Exception as e_metadata:
                _logger.error(f"Skipping {original_basename}: File not found or metadata extraction failed. Error: {e_metadata.__class__.__name__}: {e_metadata}")
                
                # Update status message if this is the only file, but continue the loop otherwise
                self.status_message_var.set(f"Error: Missing file or metadata fail for {original_basename}. Skipping.")
                
                total_sources_processed += 1
                self.message_queue.put(("progress", total_sources_processed))
                continue 

            if not base_job_info_initial:
                _logger.info(f"Skipping {original_basename}: Issues in metadata extraction/segment definition.")
                total_sources_processed += 1
                self.message_queue.put(("progress", total_sources_processed))
                continue
            
            # Store base info (includes raw frame count, fps, etc. gathered by define_video_segments)
            base_job_info_map[current_video_path] = base_job_info_initial.copy()
            
            jobs_to_process_for_this_source = []
            is_segment_processing = self.process_as_segments_var.get()
            
            # B. Decide on the actual job list (segments or full video)
            if is_segment_processing:
                if not all_potential_segments_for_video:
                    reason_skip = "Too short or invalid overlap/settings" if base_job_info_initial.get("original_video_raw_frame_count", 0) > 0 else "Source issue or zero frames/duration"
                    _logger.info(f"Skipping {original_basename}: No segments defined by settings (Reason: {reason_skip}).")
                    total_sources_processed += 1
                    self.message_queue.put(("progress", total_sources_processed))
                    continue
                    
                segment_subfolder_name = get_segment_output_folder_name(original_basename)
                segment_subfolder_path = os.path.join(self.output_dir.get(), segment_subfolder_name)
                current_video_base_info_ref = base_job_info_map[current_video_path]

                if npz_backfill_mode_active:
                    jobs_to_process_for_this_source = self._get_missing_segments_npz_only(
                        original_basename=original_basename,
                        segment_subfolder_path=segment_subfolder_path,
                        all_potential_segments_from_define=all_potential_segments_for_video,
                    )
                    if not jobs_to_process_for_this_source:
                        total_sources_processed += 1
                        self.message_queue.put(("progress", total_sources_processed))
                        continue
                else:
                    # This call handles the resume/overwrite logic
                    jobs_to_process_for_this_source, action_taken = self._get_segments_to_resume_or_overwrite(
                        current_video_path, original_basename, segment_subfolder_path, 
                        all_potential_segments_for_video, current_video_base_info_ref
                    )
                    _logger.debug(f"For source '{original_basename}': Action '{action_taken}', {len(jobs_to_process_for_this_source)} segments will be processed.")
                    
                    if not jobs_to_process_for_this_source and not current_video_base_info_ref.get("pre_existing_successful_jobs"):
                         _logger.info(f"Skipping {original_basename}: Job definition/resume resulted in no segments to process.")
                         total_sources_processed += 1
                         self.message_queue.put(("progress", total_sources_processed))
                         continue
                     
            else: # Full video processing mode
                full_out_check_path = os.path.join(self.output_dir.get(), get_full_video_output_filename(original_basename, "mp4"))
                proceed_full = True
                if os.path.exists(full_out_check_path):
                    if not messagebox.askyesno("Overwrite?", f"An output file for '{original_basename}' might exist (e.g., MP4):\n{full_out_check_path}\n\nOverwrite if it exists?"):
                        _logger.info(f"Skipping {original_basename} (full video processing, user chose not to overwrite).")
                        proceed_full = False
                
                if proceed_full:
                    full_source_job = {
                        **base_job_info_initial,
                        "is_segment": False,
                        "gui_desired_output_window_frames": gui_win_setting, 
                        "gui_desired_output_overlap_frames": gui_ov_setting 
                    }
                    jobs_to_process_for_this_source.append(full_source_job)
                else:
                    total_sources_processed += 1
                    self.message_queue.put(("progress", total_sources_processed))
                    continue # Skip to next source
            
            # C. Initialize Master Metadata for this video/sequence
            if is_segment_processing:
                if npz_backfill_mode_active:
                    total_expected_jobs_overall = len(jobs_to_process_for_this_source)
                else:
                    total_expected_jobs_overall = len(all_potential_segments_for_video)
            else:
                total_expected_jobs_overall = 1
            all_videos_master_metadata[current_video_path] = self._initialize_master_metadata_entry(
                original_basename, 
                base_job_info_initial,
                total_expected_jobs_overall
            )
            master_meta_for_this_vid = all_videos_master_metadata[current_video_path]
            
            # --- UPDATE THE SNAPSHOTTED SEED HERE ---
            master_meta_for_this_vid["global_processing_settings"]["seed_setting"] = effective_seed_for_run
            
            # Add pre-existing successful segments (only relevant for resume in segment mode)
            pre_existing_successful_segment_metadatas = base_job_info_map[current_video_path].get("pre_existing_successful_jobs", [])
            if pre_existing_successful_segment_metadatas:
                 _logger.debug(f"Loading {len(pre_existing_successful_segment_metadatas)} pre-existing successful segment metadata entries for {original_basename} into current run's master data.")
                 master_meta_for_this_vid["jobs_info"].extend(pre_existing_successful_segment_metadatas)
                 master_meta_for_this_vid["completed_successful_jobs"] += len(pre_existing_successful_segment_metadatas)

            # D. Process the actual jobs (segments or full video)
            total_jobs_for_source = len(jobs_to_process_for_this_source)
            
            for job_idx, job_info_to_run in enumerate(jobs_to_process_for_this_source):
                if self.stop_event.is_set(): break
                
                is_segment_job = job_info_to_run.get("is_segment", False)
                log_msg_prefix = f"Segment {job_info_to_run.get('segment_id', -1)+1}/{job_info_to_run.get('total_segments', 0)} ({job_idx+1}/{total_jobs_for_source})" if is_segment_job else "Full video (1/1)"
                
                # --- START PART A: INITIAL GUI UPDATE (TARGET/EXPECTED VALUES) ---
                self._update_gui_info_on_job_start(job_info_to_run, original_basename, log_msg_prefix)
                # --- END PART A: INITIAL GUI UPDATE ---

                _logger.info(f"Processing {original_basename} - {log_msg_prefix}")
                self.status_message_var.set(f"Processing {source_idx + 1} of {len(source_specs_to_process)} ({log_msg_prefix})")

                job_successful, current_job_specific_metadata = self._process_single_job(demo, job_info_to_run, master_meta_for_this_vid)
                
                if current_job_specific_metadata is None:
                    _logger.error(f"Error: _process_single_job for {original_basename} returned None metadata. Initializing to empty dict.")
                    current_job_specific_metadata = {}

                # --- START PART B: FINAL GUI UPDATE (ACTUAL/PROCESSED VALUES) ---
                self._update_gui_info_on_job_finish(job_info_to_run, current_job_specific_metadata)
                # --- END PART B: FINAL GUI UPDATE ---

                if is_segment_job and "segment_id" not in current_job_specific_metadata:
                    current_job_specific_metadata["segment_id"] = job_info_to_run.get("segment_id", -1)
                
                if "_individual_metadata_path" in current_job_specific_metadata:
                    del current_job_specific_metadata["_individual_metadata_path"]
                
                master_meta_for_this_vid["jobs_info"].append(current_job_specific_metadata)
                
                if job_successful:
                    master_meta_for_this_vid["completed_successful_jobs"] += 1
                else:
                    master_meta_for_this_vid["completed_failed_jobs"] += 1
            
            # E. Finalize the source (merge/cleanup/move original) if all its jobs are accounted for
            total_accounted_for_vid = master_meta_for_this_vid["completed_successful_jobs"] + master_meta_for_this_vid["completed_failed_jobs"]
            
            if total_accounted_for_vid >= master_meta_for_this_vid["total_expected_jobs"]:
                if npz_backfill_mode_active:
                    _logger.info(
                        f"NPZ Backfill Mode: Completed processing for {original_basename}. "
                        "Skipping merge/finalization and source-file move."
                    )
                # Finalize only if not cancelled *within* the segment loop
                elif not self.stop_event.is_set():
                    self._finalize_video_processing(current_video_path, original_basename, master_meta_for_this_vid)
                else:
                    _logger.info(f"Skipping finalization of {original_basename} due to user cancellation.")

            if clip_gpu_tracking_enabled:
                self._log_gpu_peak_tracking_summary_for_clip(original_basename)

            # F. Update Main Progress Bar (1 unit per source file/folder)
            total_sources_processed += 1
            self.message_queue.put(("progress", total_sources_processed))

        if not self.stop_event.is_set():
            _logger.info("All processing sources complete!")
            if npz_backfill_mode_active:
                self.status_message_var.set("NPZ backfill complete.")
            elif spatial_refine_mode_active:
                self.status_message_var.set("Hi-res spatial refine complete.")
            elif edge_guided_mode_active:
                self.status_message_var.set("Edge-guided hi-res upscale complete.")
            else:
                self.status_message_var.set("Processing Finished.")
        else:
            self.status_message_var.set("Processing Cancelled.")
        
        # G. Cleanup
        if 'demo' in locals() and demo is not None:
            try:
                if hasattr(demo, 'pipe') and demo.pipe is not None:
                    if hasattr(demo.pipe, 'vae') and demo.pipe.vae is not None: del demo.pipe.vae
                    if hasattr(demo.pipe, 'unet') and demo.pipe.unet is not None: del demo.pipe.unet
                    del demo.pipe
                del demo
                _logger.debug("DepthCrafter model components released.")
            except Exception as e_cleanup:
                _logger.warning(f"Error during DepthCrafter model cleanup: {e_cleanup}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            _logger.info("CUDA cache cleared.")

    def stop_processing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            _logger.info("Cancel request received. Processing will stop after current item.")
            self.stop_event.set()
            if self.active_external_process is not None and self.active_external_process.poll() is None:
                _logger.info("Terminating active cloud subprocess...")
                try:
                    self.active_external_process.terminate()
                except Exception as term_exc:
                    _logger.warning(f"Could not terminate active cloud subprocess cleanly: {term_exc}")
        else: 
            _logger.info("No processing is currently active to cancel.")

    def save_config(self):
        config = self._collect_all_settings()
        config[self.LAST_SETTINGS_DIR_CONFIG_KEY] = self.last_settings_dir
        
        config["current_input_mode"] = self.current_input_mode 
        config["single_file_mode_active"] = self.single_file_mode_active
        
        try:
            with open(self.CONFIG_FILENAME, "w") as f: json.dump(config, f, indent=4)
        except Exception as e: 
            _logger.warning(f"Warning (GUI save_config): Could not save config: {e}")

    def _close_spatial_refine_settings_dialog(self):
        tracked_widgets = list(self.spatial_refine_settings_widgets)
        if self.spatial_refine_settings_dialog is not None:
            try:
                self.spatial_refine_settings_dialog.destroy()
            except tk.TclError:
                pass
        if tracked_widgets:
            self.widgets_to_disable_during_processing = [
                w for w in self.widgets_to_disable_during_processing if w not in tracked_widgets
            ]
        self.spatial_refine_settings_dialog = None
        self.spatial_refine_settings_widgets = []
        self.spatial_refine_options_expanded_var.set(False)
        self._apply_spatial_refine_options_visibility()

    def _register_spatial_refine_dialog_widget(self, widget):
        self.spatial_refine_settings_widgets.append(widget)
        self.widgets_to_disable_during_processing.append(widget)
        return widget

    def _open_spatial_refine_settings_dialog(self):
        if self.spatial_refine_settings_dialog is not None:
            try:
                if self.spatial_refine_settings_dialog.winfo_exists():
                    self.spatial_refine_settings_dialog.lift()
                    self.spatial_refine_settings_dialog.focus_force()
                    self.spatial_refine_options_expanded_var.set(True)
                    self._apply_spatial_refine_options_visibility()
                    return
            except tk.TclError:
                self._close_spatial_refine_settings_dialog()

        dialog = tk.Toplevel(self.root)
        dialog.title("Hi-Res Refine / Edge Upscale Settings")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.protocol("WM_DELETE_WINDOW", self._close_spatial_refine_settings_dialog)

        self.spatial_refine_settings_dialog = dialog
        self.spatial_refine_settings_widgets = [dialog]
        self.spatial_refine_options_expanded_var.set(True)

        outer = ttk.Frame(dialog, padding=12)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(outer, text="Tile Grid X:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_tile_x = self._register_spatial_refine_dialog_widget(
            ttk.Entry(outer, textvariable=self.spatial_refine_tile_num_var, width=18)
        )
        entry_tile_x.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(entry_tile_x, "spatial_refine_tile_num_x")
        row += 1

        ttk.Label(outer, text="Tile Grid Y:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_tile_y = self._register_spatial_refine_dialog_widget(
            ttk.Entry(outer, textvariable=self.spatial_refine_tile_num_y_var, width=18)
        )
        entry_tile_y.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(entry_tile_y, "spatial_refine_tile_num_y")
        row += 1

        ttk.Label(outer, text="Overlap X (px):").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        spin_overlap_x = self._register_spatial_refine_dialog_widget(
            ttk.Spinbox(
                outer,
                from_=0,
                to=8192,
                increment=64,
                textvariable=self.spatial_refine_tile_overlap_x_var,
                width=16,
            )
        )
        spin_overlap_x.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(spin_overlap_x, "spatial_refine_tile_overlap_x")
        row += 1

        ttk.Label(outer, text="Overlap Y (px):").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        spin_overlap_y = self._register_spatial_refine_dialog_widget(
            ttk.Spinbox(
                outer,
                from_=0,
                to=8192,
                increment=64,
                textvariable=self.spatial_refine_tile_overlap_y_var,
                width=16,
            )
        )
        spin_overlap_y.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(spin_overlap_y, "spatial_refine_tile_overlap_y")
        row += 1

        ttk.Label(outer, text="Hi-Res Target W:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_target_w = self._register_spatial_refine_dialog_widget(
            ttk.Entry(outer, textvariable=self.spatial_refine_target_width_var, width=18)
        )
        entry_target_w.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(entry_target_w, "spatial_refine_target_width")
        row += 1

        ttk.Label(outer, text="Hi-Res Target H:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_target_h = self._register_spatial_refine_dialog_widget(
            ttk.Entry(outer, textvariable=self.spatial_refine_target_height_var, width=18)
        )
        entry_target_h.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(entry_target_h, "spatial_refine_target_height")
        row += 1

        ttk.Label(outer, text="Anchor Blend (0-1):").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_anchor = self._register_spatial_refine_dialog_widget(
            ttk.Entry(outer, textvariable=self.spatial_refine_anchor_weight_var, width=18)
        )
        entry_anchor.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(entry_anchor, "spatial_refine_anchor_weight")
        row += 1

        ttk.Separator(outer, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=(6, 4))
        row += 1

        ttk.Label(outer, text="Local Window Size:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        spin_local_window = self._register_spatial_refine_dialog_widget(
            ttk.Spinbox(
                outer,
                from_=16,
                to=512,
                increment=16,
                textvariable=self.spatial_refine_local_window_size_var,
                width=16,
            )
        )
        spin_local_window.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(spin_local_window, "spatial_refine_local_window_size")
        row += 1

        ttk.Label(outer, text="Local Window Stride:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        spin_local_stride = self._register_spatial_refine_dialog_widget(
            ttk.Spinbox(
                outer,
                from_=8,
                to=512,
                increment=8,
                textvariable=self.spatial_refine_local_window_stride_var,
                width=16,
            )
        )
        spin_local_stride.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(spin_local_stride, "spatial_refine_local_window_stride")
        row += 1

        ttk.Label(outer, text="Confidence Low:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_conf_low = self._register_spatial_refine_dialog_widget(
            ttk.Entry(outer, textvariable=self.spatial_refine_local_confidence_low_var, width=18)
        )
        entry_conf_low.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(entry_conf_low, "spatial_refine_local_confidence_low")
        row += 1

        ttk.Label(outer, text="Confidence High:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_conf_high = self._register_spatial_refine_dialog_widget(
            ttk.Entry(outer, textvariable=self.spatial_refine_local_confidence_high_var, width=18)
        )
        entry_conf_high.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(entry_conf_high, "spatial_refine_local_confidence_high")
        row += 1

        ttk.Separator(outer, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=(6, 4))
        row += 1

        edge_fallback_cb = self._register_spatial_refine_dialog_widget(
            ttk.Checkbutton(
                outer,
                text="Use Edge-Guided Fallback in Spatial Refine",
                variable=self.spatial_refine_use_edge_fallback_var,
            )
        )
        edge_fallback_cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(edge_fallback_cb, "spatial_refine_use_edge_fallback")
        row += 1

        ttk.Label(outer, text="Edge Fallback Mix (0-1):").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_edge_mix = self._register_spatial_refine_dialog_widget(
            ttk.Entry(outer, textvariable=self.spatial_refine_edge_fallback_mix_var, width=18)
        )
        entry_edge_mix.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(entry_edge_mix, "spatial_refine_edge_fallback_mix")
        row += 1

        ttk.Label(outer, text="Edge Strength:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_edge_strength = self._register_spatial_refine_dialog_widget(
            ttk.Entry(outer, textvariable=self.edge_guided_strength_var, width=18)
        )
        entry_edge_strength.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(entry_edge_strength, "edge_guided_strength")
        row += 1

        ttk.Label(outer, text="Edge Sigma Color:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_sigma_color = self._register_spatial_refine_dialog_widget(
            ttk.Entry(outer, textvariable=self.edge_guided_sigma_color_var, width=18)
        )
        entry_sigma_color.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(entry_sigma_color, "edge_guided_sigma_color")
        row += 1

        ttk.Label(outer, text="Edge Sigma Spatial:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_sigma_spatial = self._register_spatial_refine_dialog_widget(
            ttk.Entry(outer, textvariable=self.edge_guided_sigma_spatial_var, width=18)
        )
        entry_sigma_spatial.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(entry_sigma_spatial, "edge_guided_sigma_spatial")
        row += 1

        ttk.Label(outer, text="Edge Bilateral Iters:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        spin_edge_iters = self._register_spatial_refine_dialog_widget(
            ttk.Spinbox(
                outer,
                from_=0,
                to=8,
                increment=1,
                textvariable=self.edge_guided_iterations_var,
                width=16,
            )
        )
        spin_edge_iters.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(spin_edge_iters, "edge_guided_iterations")
        row += 1

        ttk.Label(outer, text="Edge Temporal Smooth:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_edge_temporal = self._register_spatial_refine_dialog_widget(
            ttk.Entry(outer, textvariable=self.edge_guided_temporal_smooth_var, width=18)
        )
        entry_edge_temporal.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(entry_edge_temporal, "edge_guided_temporal_smooth")
        row += 1

        ttk.Label(outer, text="Edge Reinject Strength:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_edge_reinject = self._register_spatial_refine_dialog_widget(
            ttk.Entry(outer, textvariable=self.edge_guided_reinject_strength_var, width=18)
        )
        entry_edge_reinject.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(entry_edge_reinject, "edge_guided_reinject_strength")
        row += 1

        ttk.Separator(outer, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=(6, 4))
        row += 1

        ttk.Label(outer, text="Hi-Res Output Suffix:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_suffix = self._register_spatial_refine_dialog_widget(
            ttk.Entry(outer, textvariable=self.spatial_refine_output_suffix_var, width=18)
        )
        entry_suffix.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(entry_suffix, "spatial_refine_output_suffix")
        row += 1

        ttk.Label(outer, text="Edge Mode Output Suffix:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_edge_suffix = self._register_spatial_refine_dialog_widget(
            ttk.Entry(outer, textvariable=self.edge_guided_output_suffix_var, width=18)
        )
        entry_edge_suffix.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(entry_edge_suffix, "edge_guided_output_suffix")
        row += 1

        cleanup_cb = self._register_spatial_refine_dialog_widget(
            ttk.Checkbutton(
                outer,
                text="Cleanup Hi-Res Temp Tile Data",
                variable=self.spatial_refine_cleanup_temp_var,
            )
        )
        cleanup_cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(cleanup_cb, "spatial_refine_cleanup_temp")
        row += 1

        close_btn = self._register_spatial_refine_dialog_widget(
            ttk.Button(outer, text="Close", command=self._close_spatial_refine_settings_dialog, width=12)
        )
        close_btn.grid(row=row, column=0, columnspan=2, sticky="e", padx=5, pady=(8, 2))

        self._apply_spatial_refine_options_visibility()

    def _close_geometry_settings_dialog(self):
        tracked_widgets = list(self.geometry_settings_widgets)
        if self.geometry_settings_dialog is not None:
            try:
                self.geometry_settings_dialog.destroy()
            except tk.TclError:
                pass
        if tracked_widgets:
            tracked_ids = {id(widget) for widget in tracked_widgets}
            self.widgets_to_disable_during_processing = [
                w for w in self.widgets_to_disable_during_processing if id(w) not in tracked_ids
            ]
        self.geometry_settings_dialog = None
        self.geometry_settings_widgets = []
        self._apply_geometry_options_visibility()

    def _register_geometry_dialog_widget(self, widget):
        self.geometry_settings_widgets.append(widget)
        self.widgets_to_disable_during_processing.append(widget)
        return widget

    def _open_geometry_settings_dialog(self):
        if self.geometry_settings_dialog is not None:
            try:
                if self.geometry_settings_dialog.winfo_exists():
                    self.geometry_settings_dialog.lift()
                    self.geometry_settings_dialog.focus_force()
                    self._apply_geometry_options_visibility()
                    return
            except tk.TclError:
                self._close_geometry_settings_dialog()

        dialog = tk.Toplevel(self.root)
        dialog.title("Backend Settings")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.protocol("WM_DELETE_WINDOW", self._close_geometry_settings_dialog)

        self.geometry_settings_dialog = dialog
        self.geometry_settings_widgets = [dialog]

        outer = ttk.Frame(dialog, padding=12)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(1, weight=1)

        row = 0
        info_label = self._register_geometry_dialog_widget(
            ttk.Label(
                outer,
                text=(
                    "These settings are used for GeometryCrafter and StereoPilot backends."
                ),
                justify="left",
                wraplength=520,
            )
        )
        info_label.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=(0, 8))
        row += 1

        ttk.Label(outer, text="Geometry Model Path:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_geometry_model_path = self._register_geometry_dialog_widget(
            ttk.Entry(outer, textvariable=self.geometry_model_path_var, width=48)
        )
        entry_geometry_model_path.grid(row=row, column=1, columnspan=2, sticky="ew", padx=(5, 0), pady=2)
        _create_hover_tooltip(entry_geometry_model_path, "geometry_model_path")
        row += 1

        ttk.Label(outer, text="Geometry Repo Path:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        repo_frame = ttk.Frame(outer)
        repo_frame.grid(row=row, column=1, columnspan=2, sticky="ew", padx=(5, 0), pady=2)
        repo_frame.columnconfigure(0, weight=1)
        entry_geometry_repo_path = self._register_geometry_dialog_widget(
            ttk.Entry(repo_frame, textvariable=self.geometry_repo_path_var, width=40)
        )
        entry_geometry_repo_path.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        _create_hover_tooltip(entry_geometry_repo_path, "geometry_repo_path")
        btn_geometry_repo = self._register_geometry_dialog_widget(
            ttk.Button(
                repo_frame,
                text="Browse...",
                command=lambda: self._browse_directory_into_var(
                    self.geometry_repo_path_var, "Select Geometry Repo Folder"
                ),
                width=10,
            )
        )
        btn_geometry_repo.grid(row=0, column=1, sticky="w")
        row += 1

        ttk.Label(outer, text="Geometry Cache Dir:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        cache_frame = ttk.Frame(outer)
        cache_frame.grid(row=row, column=1, columnspan=2, sticky="ew", padx=(5, 0), pady=2)
        cache_frame.columnconfigure(0, weight=1)
        entry_geometry_cache_dir = self._register_geometry_dialog_widget(
            ttk.Entry(cache_frame, textvariable=self.geometry_cache_dir_var, width=40)
        )
        entry_geometry_cache_dir.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        _create_hover_tooltip(entry_geometry_cache_dir, "geometry_cache_dir")
        btn_geometry_cache = self._register_geometry_dialog_widget(
            ttk.Button(
                cache_frame,
                text="Browse...",
                command=lambda: self._browse_directory_into_var(
                    self.geometry_cache_dir_var, "Select Geometry Cache Folder"
                ),
                width=10,
            )
        )
        btn_geometry_cache.grid(row=0, column=1, sticky="w")
        row += 1

        ttk.Label(outer, text="Geometry Decode Chunk:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_geometry_decode_chunk = self._register_geometry_dialog_widget(
            ttk.Spinbox(
                outer,
                from_=1,
                to=128,
                increment=1,
                textvariable=self.geometry_decode_chunk_size_var,
                width=12,
            )
        )
        entry_geometry_decode_chunk.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        _create_hover_tooltip(entry_geometry_decode_chunk, "geometry_decode_chunk_size")
        row += 1

        self.geometry_low_mem_cb = self._register_geometry_dialog_widget(
            ttk.Checkbutton(
                outer,
                text="Geometry Low-Memory Mode",
                variable=self.geometry_low_memory_usage_var,
            )
        )
        self.geometry_low_mem_cb.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.geometry_low_mem_cb, "geometry_low_memory_usage")
        row += 1

        self.geometry_force_projection_cb = self._register_geometry_dialog_widget(
            ttk.Checkbutton(
                outer,
                text="Geometry Force Projection",
                variable=self.geometry_force_projection_var,
            )
        )
        self.geometry_force_projection_cb.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.geometry_force_projection_cb, "geometry_force_projection")
        row += 1

        self.geometry_force_fixed_focal_cb = self._register_geometry_dialog_widget(
            ttk.Checkbutton(
                outer,
                text="Geometry Force Fixed Focal",
                variable=self.geometry_force_fixed_focal_var,
            )
        )
        self.geometry_force_fixed_focal_cb.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.geometry_force_fixed_focal_cb, "geometry_force_fixed_focal")
        row += 1

        self.geometry_extract_interp_cb = self._register_geometry_dialog_widget(
            ttk.Checkbutton(
                outer,
                text="Geometry Use Extract Interp",
                variable=self.geometry_use_extract_interp_var,
            )
        )
        self.geometry_extract_interp_cb.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.geometry_extract_interp_cb, "geometry_use_extract_interp")
        row += 1

        separator = self._register_geometry_dialog_widget(ttk.Separator(outer, orient="horizontal"))
        separator.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=(8, 6))
        row += 1

        stereopilot_info_label = self._register_geometry_dialog_widget(
            ttk.Label(
                outer,
                text="StereoPilot backend settings (novel-view/stereo video generation).",
                justify="left",
                wraplength=520,
            )
        )
        stereopilot_info_label.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=(0, 4))
        row += 1

        ttk.Label(outer, text="StereoPilot Model Path:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_stereopilot_model_path = self._register_geometry_dialog_widget(
            ttk.Entry(outer, textvariable=self.stereopilot_model_path_var, width=48)
        )
        entry_stereopilot_model_path.grid(row=row, column=1, columnspan=2, sticky="ew", padx=(5, 0), pady=2)
        _create_hover_tooltip(entry_stereopilot_model_path, "stereopilot_model_path")
        row += 1

        ttk.Label(outer, text="StereoPilot Base Model Path:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_stereopilot_base_model_path = self._register_geometry_dialog_widget(
            ttk.Entry(outer, textvariable=self.stereopilot_base_model_path_var, width=48)
        )
        entry_stereopilot_base_model_path.grid(row=row, column=1, columnspan=2, sticky="ew", padx=(5, 0), pady=2)
        _create_hover_tooltip(entry_stereopilot_base_model_path, "stereopilot_base_model_path")
        row += 1

        ttk.Label(outer, text="StereoPilot Repo Path:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        stereopilot_repo_frame = ttk.Frame(outer)
        stereopilot_repo_frame.grid(row=row, column=1, columnspan=2, sticky="ew", padx=(5, 0), pady=2)
        stereopilot_repo_frame.columnconfigure(0, weight=1)
        entry_stereopilot_repo_path = self._register_geometry_dialog_widget(
            ttk.Entry(stereopilot_repo_frame, textvariable=self.stereopilot_repo_path_var, width=40)
        )
        entry_stereopilot_repo_path.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        _create_hover_tooltip(entry_stereopilot_repo_path, "stereopilot_repo_path")
        btn_stereopilot_repo = self._register_geometry_dialog_widget(
            ttk.Button(
                stereopilot_repo_frame,
                text="Browse...",
                command=lambda: self._browse_directory_into_var(
                    self.stereopilot_repo_path_var, "Select StereoPilot Repo Folder"
                ),
                width=10,
            )
        )
        btn_stereopilot_repo.grid(row=0, column=1, sticky="w")
        row += 1

        ttk.Label(outer, text="StereoPilot Cache Dir:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        stereopilot_cache_frame = ttk.Frame(outer)
        stereopilot_cache_frame.grid(row=row, column=1, columnspan=2, sticky="ew", padx=(5, 0), pady=2)
        stereopilot_cache_frame.columnconfigure(0, weight=1)
        entry_stereopilot_cache_dir = self._register_geometry_dialog_widget(
            ttk.Entry(stereopilot_cache_frame, textvariable=self.stereopilot_cache_dir_var, width=40)
        )
        entry_stereopilot_cache_dir.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        _create_hover_tooltip(entry_stereopilot_cache_dir, "stereopilot_cache_dir")
        btn_stereopilot_cache = self._register_geometry_dialog_widget(
            ttk.Button(
                stereopilot_cache_frame,
                text="Browse...",
                command=lambda: self._browse_directory_into_var(
                    self.stereopilot_cache_dir_var, "Select StereoPilot Cache Folder"
                ),
                width=10,
            )
        )
        btn_stereopilot_cache.grid(row=0, column=1, sticky="w")
        row += 1

        ttk.Label(outer, text="Stereo Prompt (fallback):").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_stereopilot_prompt = self._register_geometry_dialog_widget(
            ttk.Entry(outer, textvariable=self.stereopilot_prompt_var, width=48)
        )
        entry_stereopilot_prompt.grid(row=row, column=1, columnspan=2, sticky="ew", padx=(5, 0), pady=2)
        _create_hover_tooltip(entry_stereopilot_prompt, "stereopilot_prompt")
        row += 1

        stereo_options_frame = self._register_geometry_dialog_widget(ttk.Frame(outer))
        stereo_options_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=2)
        for idx in range(4):
            stereo_options_frame.columnconfigure(idx, weight=1)
        stereo_sidecar_cb = self._register_geometry_dialog_widget(
            ttk.Checkbutton(
                stereo_options_frame,
                text="Use sidecar .txt prompt when available",
                variable=self.stereopilot_use_sidecar_prompt_var,
            )
        )
        stereo_sidecar_cb.grid(row=0, column=0, columnspan=2, sticky="w")
        _create_hover_tooltip(stereo_sidecar_cb, "stereopilot_use_sidecar_prompt")
        ttk.Label(stereo_options_frame, text="Output Mode:").grid(row=0, column=2, sticky="e", padx=(6, 2))
        stereo_output_combo = self._register_geometry_dialog_widget(
            ttk.Combobox(
                stereo_options_frame,
                textvariable=self.stereopilot_output_mode_var,
                values=["opposite_eye", "side_by_side", "both"],
                width=16,
                state="readonly",
            )
        )
        stereo_output_combo.grid(row=0, column=3, sticky="w")
        _create_hover_tooltip(stereo_output_combo, "stereopilot_output_mode")
        row += 1

        stereo_numeric_frame = self._register_geometry_dialog_widget(ttk.Frame(outer))
        stereo_numeric_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=(2, 4))
        for idx in range(6):
            stereo_numeric_frame.columnconfigure(idx, weight=0)

        ttk.Label(stereo_numeric_frame, text="W").grid(row=0, column=0, sticky="e", padx=(0, 2))
        entry_sp_w = self._register_geometry_dialog_widget(ttk.Entry(stereo_numeric_frame, textvariable=self.stereopilot_target_width_var, width=7))
        entry_sp_w.grid(row=0, column=1, sticky="w", padx=(0, 6))
        ttk.Label(stereo_numeric_frame, text="H").grid(row=0, column=2, sticky="e", padx=(0, 2))
        entry_sp_h = self._register_geometry_dialog_widget(ttk.Entry(stereo_numeric_frame, textvariable=self.stereopilot_target_height_var, width=7))
        entry_sp_h.grid(row=0, column=3, sticky="w", padx=(0, 6))
        ttk.Label(stereo_numeric_frame, text="FPS").grid(row=0, column=4, sticky="e", padx=(0, 2))
        entry_sp_fps = self._register_geometry_dialog_widget(ttk.Entry(stereo_numeric_frame, textvariable=self.stereopilot_target_fps_var, width=7))
        entry_sp_fps.grid(row=0, column=5, sticky="w")
        _create_hover_tooltip(entry_sp_w, "stereopilot_target_width")
        _create_hover_tooltip(entry_sp_h, "stereopilot_target_height")
        _create_hover_tooltip(entry_sp_fps, "stereopilot_target_fps")
        row += 1

        stereo_window_frame = self._register_geometry_dialog_widget(ttk.Frame(outer))
        stereo_window_frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=(0, 4))
        ttk.Label(stereo_window_frame, text="Window").grid(row=0, column=0, sticky="e", padx=(0, 2))
        entry_sp_window = self._register_geometry_dialog_widget(
            ttk.Entry(stereo_window_frame, textvariable=self.stereopilot_window_size_var, width=7)
        )
        entry_sp_window.grid(row=0, column=1, sticky="w", padx=(0, 8))
        ttk.Label(stereo_window_frame, text="Overlap").grid(row=0, column=2, sticky="e", padx=(0, 2))
        entry_sp_overlap = self._register_geometry_dialog_widget(
            ttk.Entry(stereo_window_frame, textvariable=self.stereopilot_overlap_var, width=7)
        )
        entry_sp_overlap.grid(row=0, column=3, sticky="w")
        _create_hover_tooltip(entry_sp_window, "stereopilot_window_size")
        _create_hover_tooltip(entry_sp_overlap, "stereopilot_overlap")
        row += 1

        stereo_numeric_frame_2 = self._register_geometry_dialog_widget(ttk.Frame(outer))
        stereo_numeric_frame_2.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=(0, 4))
        ttk.Label(stereo_numeric_frame_2, text="Sampling Steps").grid(row=0, column=0, sticky="e", padx=(0, 2))
        entry_sp_steps = self._register_geometry_dialog_widget(ttk.Entry(stereo_numeric_frame_2, textvariable=self.stereopilot_sampling_steps_var, width=7))
        entry_sp_steps.grid(row=0, column=1, sticky="w", padx=(0, 8))
        ttk.Label(stereo_numeric_frame_2, text="Guide Scale").grid(row=0, column=2, sticky="e", padx=(0, 2))
        entry_sp_guide = self._register_geometry_dialog_widget(ttk.Entry(stereo_numeric_frame_2, textvariable=self.stereopilot_guide_scale_var, width=7))
        entry_sp_guide.grid(row=0, column=3, sticky="w", padx=(0, 8))
        ttk.Label(stereo_numeric_frame_2, text="Shift").grid(row=0, column=4, sticky="e", padx=(0, 2))
        entry_sp_shift = self._register_geometry_dialog_widget(ttk.Entry(stereo_numeric_frame_2, textvariable=self.stereopilot_shift_var, width=7))
        entry_sp_shift.grid(row=0, column=5, sticky="w", padx=(0, 8))
        ttk.Label(stereo_numeric_frame_2, text="Domain").grid(row=0, column=6, sticky="e", padx=(0, 2))
        entry_sp_domain = self._register_geometry_dialog_widget(ttk.Entry(stereo_numeric_frame_2, textvariable=self.stereopilot_domain_label_var, width=5))
        entry_sp_domain.grid(row=0, column=7, sticky="w")
        _create_hover_tooltip(entry_sp_steps, "stereopilot_sampling_steps")
        _create_hover_tooltip(entry_sp_guide, "stereopilot_guide_scale")
        _create_hover_tooltip(entry_sp_shift, "stereopilot_shift")
        _create_hover_tooltip(entry_sp_domain, "stereopilot_domain_label")
        row += 1

        stereo_dtype_frame = self._register_geometry_dialog_widget(ttk.Frame(outer))
        stereo_dtype_frame.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=(0, 4))
        ttk.Label(stereo_dtype_frame, text="Model Dtype:").grid(row=0, column=0, sticky="e", padx=(0, 2))
        combo_sp_dtype = self._register_geometry_dialog_widget(
            ttk.Combobox(
                stereo_dtype_frame,
                textvariable=self.stereopilot_dtype_var,
                values=["float16", "bfloat16", "float32"],
                width=10,
                state="readonly",
            )
        )
        combo_sp_dtype.grid(row=0, column=1, sticky="w", padx=(0, 8))
        ttk.Label(stereo_dtype_frame, text="Transformer Dtype:").grid(row=0, column=2, sticky="e", padx=(0, 2))
        combo_sp_transformer_dtype = self._register_geometry_dialog_widget(
            ttk.Combobox(
                stereo_dtype_frame,
                textvariable=self.stereopilot_transformer_dtype_var,
                values=["float8", "float16", "bfloat16", "float32"],
                width=10,
                state="readonly",
            )
        )
        combo_sp_transformer_dtype.grid(row=0, column=3, sticky="w")
        _create_hover_tooltip(combo_sp_dtype, "stereopilot_dtype")
        _create_hover_tooltip(combo_sp_transformer_dtype, "stereopilot_transformer_dtype")
        row += 1

        status_label = self._register_geometry_dialog_widget(
            ttk.Label(
                outer,
                textvariable=self.geometry_local_status_var,
                wraplength=520,
                justify="left",
            )
        )
        status_label.grid(row=row, column=0, columnspan=3, sticky="w", padx=5, pady=(4, 4))
        row += 1

        close_btn = self._register_geometry_dialog_widget(
            ttk.Button(outer, text="Close", command=self._close_geometry_settings_dialog, width=12)
        )
        close_btn.grid(row=row, column=0, columnspan=3, sticky="e", padx=5, pady=(8, 2))

        self._refresh_geometry_local_status()
        self._apply_geometry_options_visibility()

    def _apply_geometry_options_visibility(self):
        if not hasattr(self, 'geometry_settings_toggle_btn') or self.geometry_settings_toggle_btn is None:
            return
        is_open = (
            self.geometry_settings_dialog is not None
            and bool(self.geometry_settings_dialog.winfo_exists())
        )
        button_text = "  ↳ Backend Settings Open" if is_open else "  ↳ Configure Backend..."
        try:
            self.geometry_settings_toggle_btn.configure(text=button_text)
        except tk.TclError:
            pass

    def toggle_geometry_settings_visibility(self):
        self._open_geometry_settings_dialog()

    def _browse_file_into_var(self, tk_var, title, filetypes):
        initial_guess = tk_var.get().strip()
        if initial_guess:
            initial_guess = os.path.expanduser(initial_guess)
            if os.path.isfile(initial_guess):
                initial_dir = os.path.dirname(initial_guess)
            elif os.path.isdir(initial_guess):
                initial_dir = initial_guess
            else:
                initial_dir = os.path.expanduser("~")
        else:
            initial_dir = os.path.expanduser("~")
        selected = filedialog.askopenfilename(
            title=title,
            initialdir=initial_dir,
            filetypes=filetypes,
        )
        if selected:
            tk_var.set(os.path.normpath(selected))

    def _browse_directory_into_var(self, tk_var, title):
        initial_guess = tk_var.get().strip()
        if initial_guess:
            initial_guess = os.path.expanduser(initial_guess)
            if os.path.isdir(initial_guess):
                initial_dir = initial_guess
            else:
                initial_dir = os.path.dirname(initial_guess)
        else:
            initial_dir = os.path.expanduser("~")
        if not initial_dir:
            initial_dir = os.path.expanduser("~")
        selected = filedialog.askdirectory(
            title=title,
            initialdir=initial_dir,
        )
        if selected:
            tk_var.set(os.path.normpath(selected))

    def _safe_int_from_tk_var(self, tk_var, fallback: int) -> int:
        try:
            return int(tk_var.get())
        except Exception:
            return int(fallback)

    def _safe_float_from_tk_var(self, tk_var, fallback: float) -> float:
        try:
            return float(tk_var.get())
        except Exception:
            return float(fallback)

    def _is_stereopilot_backend_selected(self) -> bool:
        backend = str(self.model_backend_var.get() or "depthcrafter").strip().lower()
        return backend == "stereopilot"

    def _apply_backend_specific_controls(self):
        if not hasattr(self, "process_as_segments_cb") or self.process_as_segments_cb is None:
            return

        stereopilot_active = self._is_stereopilot_backend_selected()
        if stereopilot_active and bool(self.process_as_segments_var.get()):
            self.process_as_segments_var.set(False)

        desired_state = tk.NORMAL
        if stereopilot_active:
            desired_state = tk.DISABLED
        else:
            try:
                if (
                    hasattr(self, "start_button")
                    and self.start_button is not None
                    and hasattr(self, "cancel_button")
                    and self.cancel_button is not None
                    and self.start_button.cget("state") == tk.DISABLED
                    and self.cancel_button.cget("state") == tk.NORMAL
                ):
                    desired_state = tk.DISABLED
            except tk.TclError:
                pass

        try:
            self.process_as_segments_cb.configure(state=desired_state)
        except tk.TclError:
            pass
        self.toggle_merge_related_options_active_state()

    def _on_geometry_status_setting_changed(self, *_):
        self._refresh_geometry_local_status()
        self._apply_geometry_options_visibility()
        self._apply_backend_specific_controls()

    def _bind_geometry_status_traces(self):
        tracked_vars = [
            self.model_backend_var,
            self.geometry_model_path_var,
            self.geometry_repo_path_var,
            self.geometry_cache_dir_var,
            self.stereopilot_model_path_var,
            self.stereopilot_base_model_path_var,
            self.stereopilot_repo_path_var,
            self.stereopilot_cache_dir_var,
            self.enable_cloud_dispatch_mode_var,
        ]
        for tk_var in tracked_vars:
            try:
                tk_var.trace_add("write", self._on_geometry_status_setting_changed)
            except Exception:
                continue

    def _refresh_geometry_local_status(self):
        if not hasattr(self, "geometry_local_status_var"):
            return

        backend = str(self.model_backend_var.get() or "depthcrafter").strip().lower()
        if backend not in ("geometrycrafter_diff", "geometrycrafter_determ", "stereopilot"):
            self.geometry_local_status_var.set("DepthCrafter backend active. No extra local prerequisites.")
            return

        if bool(self.enable_cloud_dispatch_mode_var.get()):
            if backend.startswith("geometrycrafter"):
                self.geometry_local_status_var.set(
                    "Geometry backend selected in Cloud mode: local Geometry repo is optional. "
                    "Remote worker auto-provisions repo/submodules and model weights."
                )
            else:
                self.geometry_local_status_var.set(
                    "StereoPilot backend selected in Cloud mode: local StereoPilot repo is optional. "
                    "Remote worker auto-provisions repo and model weights."
                )
            return

        if backend == "stereopilot":
            repo_root = os.path.dirname(os.path.abspath(__file__))
            repo_path_raw = str(self.stereopilot_repo_path_var.get() or "").strip()
            if repo_path_raw:
                repo_path = os.path.expanduser(repo_path_raw)
                if not os.path.isabs(repo_path):
                    repo_path = os.path.normpath(os.path.join(repo_root, repo_path))
            else:
                repo_path = os.path.normpath(os.path.join(repo_root, "weights", "StereoPilot"))

            model_raw = str(self.stereopilot_model_path_var.get() or "KlingTeam/StereoPilot").strip() or "KlingTeam/StereoPilot"
            base_raw = str(self.stereopilot_base_model_path_var.get() or "Wan-AI/Wan2.1-T2V-1.3B").strip() or "Wan-AI/Wan2.1-T2V-1.3B"
            cache_raw = str(self.stereopilot_cache_dir_var.get() or "").strip()

            missing_items = []
            if not os.path.isdir(repo_path):
                missing_items.append("repo folder")
            else:
                if not os.path.isfile(os.path.join(repo_path, "sample.py")):
                    missing_items.append("sample.py")
                if not os.path.isdir(os.path.join(repo_path, "models")):
                    missing_items.append("models/")
                if not os.path.isdir(os.path.join(repo_path, "utils")):
                    missing_items.append("utils/")

            transformer_local = os.path.join(repo_path, "ckpt", "StereoPilot.safetensors")
            base_local = os.path.join(repo_path, "ckpt", "Wan2.1-T2V-1.3B")
            model_path_expanded = os.path.expanduser(model_raw)
            base_path_expanded = os.path.expanduser(base_raw)
            model_status_note = ""
            base_status_note = ""

            if os.path.isfile(model_path_expanded):
                model_status_note = f"transformer: local file ({model_path_expanded})"
            elif os.path.isfile(transformer_local):
                model_status_note = f"transformer: local cache ({transformer_local})"
            else:
                if "/" in model_raw:
                    model_status_note = "transformer: missing local checkpoint (will need HF download/setup script)"
                else:
                    model_status_note = f"transformer: missing ({model_raw})"
                missing_items.append("StereoPilot.safetensors")

            if os.path.isdir(base_path_expanded):
                base_status_note = f"base model: local dir ({base_path_expanded})"
            elif os.path.isdir(base_local):
                base_status_note = f"base model: local cache ({base_local})"
            else:
                if "/" in base_raw:
                    base_status_note = "base model: missing local checkpoint directory (will need HF download/setup script)"
                else:
                    base_status_note = f"base model: missing ({base_raw})"
                missing_items.append("Wan2.1-T2V-1.3B")

            try:
                import toml  # noqa: F401
            except Exception:
                missing_items.append("python package: toml")
            try:
                import easydict  # noqa: F401
            except Exception:
                missing_items.append("python package: easydict")
            try:
                import ftfy  # noqa: F401
            except Exception:
                missing_items.append("python package: ftfy")
            try:
                import safetensors  # noqa: F401
            except Exception:
                missing_items.append("python package: safetensors")

            cache_note = f"cache dir: {cache_raw}" if cache_raw else "cache dir: default"
            setup_hint = "setup: PYTHON_BIN=<your_python> bash scripts/setup_stereopilot_local.sh"

            if missing_items:
                self.geometry_local_status_var.set(
                    "StereoPilot local status: missing "
                    + ", ".join(missing_items)
                    + f". Repo path checked: {repo_path} | {model_status_note} | {base_status_note} | {cache_note} | {setup_hint}"
                )
            else:
                self.geometry_local_status_var.set(
                    f"StereoPilot local status: ready | repo: {repo_path} | {model_status_note} | {base_status_note} | {cache_note}"
                )
            return

        repo_root = os.path.dirname(os.path.abspath(__file__))
        repo_path_raw = str(self.geometry_repo_path_var.get() or "").strip()
        if repo_path_raw:
            repo_path = os.path.expanduser(repo_path_raw)
            if not os.path.isabs(repo_path):
                repo_path = os.path.normpath(os.path.join(repo_root, repo_path))
        else:
            repo_path = os.path.normpath(os.path.join(repo_root, "weights", "GeometryCrafter"))

        geometry_model_raw = str(self.geometry_model_path_var.get() or "TencentARC/GeometryCrafter").strip()
        if not geometry_model_raw:
            geometry_model_raw = "TencentARC/GeometryCrafter"
        geometry_cache_raw = str(self.geometry_cache_dir_var.get() or "").strip()

        model_status_note = "model cache: unknown"
        local_model_path = os.path.expanduser(geometry_model_raw)
        if not os.path.isabs(local_model_path):
            local_model_path = os.path.normpath(os.path.join(repo_root, local_model_path))
        if os.path.isdir(local_model_path):
            model_status_note = f"model source: local path ({local_model_path})"
        elif "/" in geometry_model_raw:
            cache_roots = []
            if geometry_cache_raw:
                cache_roots.append(os.path.expanduser(geometry_cache_raw))
            hf_cache_env = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
            if hf_cache_env:
                cache_roots.append(os.path.expanduser(hf_cache_env))
            hf_home = os.environ.get("HF_HOME", "").strip()
            if hf_home:
                cache_roots.append(os.path.join(os.path.expanduser(hf_home), "hub"))
            cache_roots.append(os.path.expanduser("~/.cache/huggingface/hub"))

            cache_hit_path = ""
            cache_key = f"models--{geometry_model_raw.replace('/', '--')}"
            for root in cache_roots:
                try:
                    model_cache_dir = os.path.join(root, cache_key)
                    snapshots_dir = os.path.join(model_cache_dir, "snapshots")
                    if os.path.isdir(snapshots_dir):
                        with os.scandir(snapshots_dir) as it:
                            for entry in it:
                                if entry.is_dir():
                                    cache_hit_path = snapshots_dir
                                    break
                    if cache_hit_path:
                        break
                except Exception:
                    continue
            if cache_hit_path:
                model_status_note = f"model cache: found ({cache_hit_path})"
            else:
                model_status_note = (
                    "model cache: missing (will auto-download from Hugging Face on first local run)"
                )
        else:
            model_status_note = (
                f"model source '{geometry_model_raw}' not found as local path and not a Hugging Face model id"
            )

        missing_items = []
        if not os.path.isdir(repo_path):
            missing_items.append("repo folder")
        else:
            if not os.path.isdir(os.path.join(repo_path, "geometrycrafter")):
                missing_items.append("geometrycrafter/")
            if not os.path.isdir(os.path.join(repo_path, "third_party")):
                missing_items.append("third_party/")
            if not os.path.isdir(os.path.join(repo_path, "third_party", "moge", "moge", "model")):
                missing_items.append("third_party/moge submodule")
            if not os.path.isdir(os.path.join(repo_path, "third_party", "moge", "utils3d")):
                missing_items.append("third_party/moge/utils3d")

        try:
            import kornia  # noqa: F401
        except Exception:
            missing_items.append("python package: kornia")
        try:
            import scipy  # noqa: F401
        except Exception:
            missing_items.append("python package: scipy")

        if missing_items:
            self.geometry_local_status_var.set(
                "Geometry local status: missing "
                + ", ".join(missing_items)
                + f". Repo path checked: {repo_path} | {model_status_note}"
            )
        else:
            self.geometry_local_status_var.set(
                f"Geometry local status: ready ({backend}) | repo: {repo_path} | {model_status_note}"
            )

    def _get_cloud_profile_defaults(self, profile_key: Optional[str] = None) -> Dict[str, object]:
        key = (profile_key or self.cloud_profile_var.get().strip() or "5090_32gb").strip()
        if cloud_core is not None:
            return cloud_core.get_cloud_profile_defaults(key)
        defaults = self.CLOUD_PROFILE_DEFAULTS.get(key)
        if defaults is None:
            key = "5090_32gb"
            defaults = self.CLOUD_PROFILE_DEFAULTS[key]
        out = {"key": key, **defaults}
        out.setdefault("use_source_resolution", False)
        return out

    def _get_effective_cloud_processing_settings(self) -> Dict[str, object]:
        profile_defaults = self._get_cloud_profile_defaults()
        use_source_resolution = bool(profile_defaults.get("use_source_resolution", False))
        width_override = max(0, self._safe_int_from_tk_var(self.cloud_target_width_override_var, 0))
        height_override = max(0, self._safe_int_from_tk_var(self.cloud_target_height_override_var, 0))
        window_override = self._safe_int_from_tk_var(self.cloud_window_size_override_var, 0)
        overlap_override = self._safe_int_from_tk_var(self.cloud_overlap_override_var, -1)
        selected_backend = str(self.model_backend_var.get() or "depthcrafter").strip().lower()

        if selected_backend == "stereopilot":
            base_window = max(1, self._safe_int_from_tk_var(self.stereopilot_window_size_var, 81))
            base_overlap = max(0, self._safe_int_from_tk_var(self.stereopilot_overlap_var, 25))
            base_window_label = "stereopilot window"
            base_overlap_label = "stereopilot overlap"
        else:
            base_window = max(1, self._safe_int_from_tk_var(self.window_size, int(profile_defaults["window_size"])))
            base_overlap = max(0, self._safe_int_from_tk_var(self.overlap, int(profile_defaults["overlap"])))
            base_window_label = "main window"
            base_overlap_label = "main overlap"

        effective_width = width_override if width_override > 0 else int(profile_defaults["target_width"])
        effective_height = height_override if height_override > 0 else int(profile_defaults["target_height"])
        effective_window = window_override if window_override > 0 else base_window
        effective_overlap = overlap_override if overlap_override >= 0 else base_overlap

        if width_override > 0 or height_override > 0:
            target_source = "cloud override"
        elif use_source_resolution:
            target_source = "source clip resolution (auto per file)"
        else:
            target_source = f"profile default ({profile_defaults['key']})"
        window_source = "cloud override" if window_override > 0 else base_window_label
        overlap_source = "cloud override" if overlap_override >= 0 else base_overlap_label

        return {
            "profile_key": str(profile_defaults["key"]),
            "profile_label": str(profile_defaults["label"]),
            "profile_target_width": int(profile_defaults["target_width"]),
            "profile_target_height": int(profile_defaults["target_height"]),
            "profile_window_size": int(profile_defaults["window_size"]),
            "profile_overlap": int(profile_defaults["overlap"]),
            "use_source_resolution": bool(use_source_resolution),
            "target_width_override": width_override,
            "target_height_override": height_override,
            "window_size_override": window_override,
            "overlap_override": overlap_override,
            "target_width": int(effective_width),
            "target_height": int(effective_height),
            "window_size": int(effective_window),
            "overlap": int(effective_overlap),
            "target_source": target_source,
            "window_source": window_source,
            "overlap_source": overlap_source,
            "base_window": int(base_window),
            "base_overlap": int(base_overlap),
            "selected_backend": selected_backend,
        }

    def _refresh_cloud_processing_summary(self):
        settings = self._get_effective_cloud_processing_settings()
        require_verified_hosts = bool(self.cloud_require_verified_hosts_var.get())
        if bool(settings.get("use_source_resolution", False)) and int(settings.get("target_width_override", 0)) <= 0 and int(settings.get("target_height_override", 0)) <= 0:
            self.cloud_profile_default_summary_var.set(
                (
                    f"{settings['profile_label']} defaults: source clip resolution (auto), "
                    f"window={settings['profile_window_size']}, overlap={settings['profile_overlap']}"
                )
            )
            self.cloud_effective_processing_summary_var.set(
                (
                    f"Effective cloud run: source clip resolution (auto per file), "
                    f"window={settings['window_size']} ({settings['window_source']}), "
                    f"overlap={settings['overlap']} ({settings['overlap_source']})."
                )
            )
        else:
            self.cloud_profile_default_summary_var.set(
                (
                    f"{settings['profile_label']} defaults: "
                    f"{settings['profile_target_width']}x{settings['profile_target_height']}, "
                    f"window={settings['profile_window_size']}, overlap={settings['profile_overlap']}"
                )
            )
            self.cloud_effective_processing_summary_var.set(
                (
                    f"Effective cloud run: {settings['target_width']}x{settings['target_height']} "
                    f"({settings['target_source']}), window={settings['window_size']} "
                    f"({settings['window_source']}), overlap={settings['overlap']} "
                    f"({settings['overlap_source']})."
                )
            )
        self.cloud_inherited_processing_summary_var.set(
            (
                f"Current backend base values: window={settings['base_window']}, overlap={settings['base_overlap']}. "
                "Advanced cloud overrides replace these only when set."
            )
        )
        if bool(settings.get("use_source_resolution", False)) and int(settings.get("target_width_override", 0)) <= 0 and int(settings.get("target_height_override", 0)) <= 0:
            target_hint = "source clip resolution (auto)"
        else:
            target_hint = f"{settings['target_width']}x{settings['target_height']}"
        self.cloud_secondary_summary_var.set(
            (
                "  ↳ Launches a Vast worker, uploads clip(s), runs remote depth, downloads outputs. "
                f"Cloud target: {target_hint}. "
                f"Verified hosts: {'required' if require_verified_hosts else 'optional'}."
            )
        )
        self._refresh_cloud_blacklist_summary()

    def _resolve_cloud_blacklist_path(self) -> str:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(repo_root, self.CLOUD_BLACKLIST_PATH))

    def _resolve_cloud_provider_history_path(self) -> str:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(repo_root, self.CLOUD_PROVIDER_HISTORY_PATH))

    def _normalize_cloud_blacklist_data(self, raw_data: Optional[Dict[str, Any]]) -> Dict[str, set]:
        if cloud_core is not None:
            return cloud_core.normalize_blacklist_data(raw_data)
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

    def _load_cloud_blacklist_data(self) -> Dict[str, set]:
        blacklist_path = self._resolve_cloud_blacklist_path()
        if cloud_core is not None:
            return cloud_core.load_blacklist_data(blacklist_path)
        if not os.path.isfile(blacklist_path):
            return self._normalize_cloud_blacklist_data(None)
        try:
            with open(blacklist_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            return self._normalize_cloud_blacklist_data(raw_data)
        except Exception as exc:
            _logger.warning(f"[CLOUD] Failed to parse blacklist file '{blacklist_path}': {exc}")
            return self._normalize_cloud_blacklist_data(None)

    def _save_cloud_blacklist_data(self, blacklist_data: Dict[str, set]):
        blacklist_path = self._resolve_cloud_blacklist_path()
        if cloud_core is not None:
            cloud_core.save_blacklist_data(
                blacklist_path,
                blacklist_data,
                updated_by="depthcrafter_gui_seg.py",
            )
            return
        os.makedirs(os.path.dirname(blacklist_path), exist_ok=True)
        serializable = {
            "blocked_offer_ids": sorted(int(x) for x in blacklist_data.get("blocked_offer_ids", set()) if int(x) > 0),
            "blocked_machine_ids": sorted(int(x) for x in blacklist_data.get("blocked_machine_ids", set()) if int(x) > 0),
            "blocked_host_ids": sorted(int(x) for x in blacklist_data.get("blocked_host_ids", set()) if int(x) > 0),
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updated_by": "depthcrafter_gui_seg.py",
        }
        with open(blacklist_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def _refresh_cloud_blacklist_summary(self):
        data = self._load_cloud_blacklist_data()
        offer_count = len(data.get("blocked_offer_ids", set()))
        machine_count = len(data.get("blocked_machine_ids", set()))
        host_count = len(data.get("blocked_host_ids", set()))
        self.cloud_blacklist_summary_var.set(
            f"Blacklist: offers={offer_count}, machines={machine_count}, hosts={host_count}"
        )

    def _normalize_cloud_provider_history_data(self, raw_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if cloud_core is not None:
            return cloud_core.normalize_provider_history_data(raw_data)

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

    def _load_cloud_provider_history_data(self) -> Dict[str, Any]:
        history_path = self._resolve_cloud_provider_history_path()
        if cloud_core is not None:
            return cloud_core.load_provider_history_data(history_path)
        if not os.path.isfile(history_path):
            return self._normalize_cloud_provider_history_data(None)
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            return self._normalize_cloud_provider_history_data(raw_data)
        except Exception as exc:
            _logger.warning(f"[CLOUD] Failed to parse provider history '{history_path}': {exc}")
            return self._normalize_cloud_provider_history_data(None)

    def _save_cloud_provider_history_data(self, history_data: Dict[str, Any]):
        history_path = self._resolve_cloud_provider_history_path()
        if cloud_core is not None:
            cloud_core.save_provider_history_data(
                history_path,
                history_data,
                updated_by="depthcrafter_gui_seg.py",
            )
            return
        os.makedirs(os.path.dirname(history_path), exist_ok=True)

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
            "updated_by": "depthcrafter_gui_seg.py",
        }
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def _cloud_provider_key_from_ids(
        self,
        offer_id: int,
        machine_id: int,
        host_id: int,
        host: str = "",
    ) -> str:
        if cloud_core is not None:
            return cloud_core.cloud_provider_key_from_ids(
                offer_id=offer_id,
                machine_id=machine_id,
                host_id=host_id,
                host=host,
            )
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

    def _cloud_provider_label_from_key(self, provider_key: str) -> str:
        if cloud_core is not None:
            return cloud_core.cloud_provider_label_from_key(provider_key)
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

    def _cloud_provider_identity_for_offer(self, offer_entry: Dict[str, Any]) -> Dict[str, Any]:
        if cloud_core is not None:
            return cloud_core.cloud_provider_identity_for_offer(offer_entry)
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
        provider_key = self._cloud_provider_key_from_ids(
            offer_id=offer_id,
            machine_id=machine_id,
            host_id=host_id,
        )
        return {
            "provider_key": provider_key,
            "provider_label": self._cloud_provider_label_from_key(provider_key),
            "offer_id": offer_id,
            "machine_id": machine_id,
            "host_id": host_id,
        }

    def _record_cloud_connection_history(
        self,
        connection_info: Dict[str, object],
        profile_key: str,
        connection_origin: str,
    ):
        try:
            offer_id = int(connection_info.get("offer_id", 0) or 0)
            machine_id = int(connection_info.get("machine_id", 0) or 0)
            host_id = int(connection_info.get("host_id", 0) or 0)
            host = str(connection_info.get("host", "") or "").strip()
            port = int(connection_info.get("port", 0) or 0)
        except Exception as exc:
            _logger.warning(f"[CLOUD] Could not parse connection info for history: {exc}")
            return

        provider_key = self._cloud_provider_key_from_ids(
            offer_id=offer_id,
            machine_id=machine_id,
            host_id=host_id,
            host=host,
        )

        history_data = self._load_cloud_provider_history_data()
        provider_counts = dict(history_data.get("provider_counts", {}))
        offer_counts = dict(history_data.get("offer_counts", {}))
        machine_counts = dict(history_data.get("machine_counts", {}))
        host_counts = dict(history_data.get("host_counts", {}))

        provider_counts[provider_key] = int(provider_counts.get(provider_key, 0)) + 1
        if offer_id > 0:
            offer_key = str(offer_id)
            offer_counts[offer_key] = int(offer_counts.get(offer_key, 0)) + 1
        if machine_id > 0:
            machine_key = str(machine_id)
            machine_counts[machine_key] = int(machine_counts.get(machine_key, 0)) + 1
        if host_id > 0:
            host_key = str(host_id)
            host_counts[host_key] = int(host_counts.get(host_key, 0)) + 1

        record = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "profile": str(profile_key or ""),
            "origin": str(connection_origin or ""),
            "instance_id": int(connection_info.get("instance_id", 0) or 0),
            "offer_id": offer_id,
            "machine_id": machine_id,
            "host_id": host_id,
            "provider_key": provider_key,
            "provider_label": self._cloud_provider_label_from_key(provider_key),
            "ssh_host": host,
            "ssh_port": port,
            "gpu_name": str(connection_info.get("gpu_name", "") or ""),
            "geolocation": str(connection_info.get("geolocation", "") or ""),
        }
        recent_connections = list(history_data.get("recent_connections", []))
        recent_connections.append(record)
        history_data["recent_connections"] = recent_connections[-200:]
        history_data["provider_counts"] = provider_counts
        history_data["offer_counts"] = offer_counts
        history_data["machine_counts"] = machine_counts
        history_data["host_counts"] = host_counts

        self._save_cloud_provider_history_data(history_data)
        _logger.info(
            "[CLOUD] History updated | provider=%s count=%s offer=%s offer_count=%s",
            provider_key,
            provider_counts.get(provider_key, 0),
            offer_id,
            offer_counts.get(str(offer_id), 0) if offer_id > 0 else 0,
        )

    def _is_cloud_offer_blacklisted(
        self,
        offer_entry: Dict[str, Any],
        blacklist_data: Optional[Dict[str, set]] = None,
    ) -> bool:
        if cloud_core is not None:
            data = blacklist_data if blacklist_data is not None else self._load_cloud_blacklist_data()
            return cloud_core.offer_is_blacklisted(offer_entry, data)
        data = blacklist_data if blacklist_data is not None else self._load_cloud_blacklist_data()
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

    def _blacklist_cached_cloud_target_from_ui(self):
        try:
            instance_id = int(self.cloud_last_instance_id_var.get())
        except Exception:
            instance_id = 0
        if instance_id <= 0:
            messagebox.showerror("Cloud", "Set a valid cached instance ID first.")
            return

        try:
            cached_offer_id = int(self.cloud_last_offer_id_var.get())
        except Exception:
            cached_offer_id = 0
        try:
            cached_machine_id = int(self.cloud_last_machine_id_var.get())
        except Exception:
            cached_machine_id = 0
        try:
            cached_host_id = int(self.cloud_last_host_id_var.get())
        except Exception:
            cached_host_id = 0

        row = self._get_cloud_instance_row(instance_id)
        row_machine_id = 0
        row_host_id = 0
        if isinstance(row, dict):
            try:
                row_machine_id = int(row.get("machine_id", 0) or 0)
            except Exception:
                row_machine_id = 0
            try:
                row_host_id = int(row.get("host_id", 0) or 0)
            except Exception:
                row_host_id = 0

        offer_id = cached_offer_id
        machine_id = row_machine_id if row_machine_id > 0 else cached_machine_id
        host_id = row_host_id if row_host_id > 0 else cached_host_id
        if offer_id <= 0 and machine_id <= 0 and host_id <= 0:
            messagebox.showerror(
                "Cloud",
                "No offer/machine/host identifiers are available to blacklist for the cached instance."
            )
            return

        confirm_lines = [
            f"Blacklist cached cloud target from future offer selection?",
            "",
            f"Instance ID: {instance_id}",
            f"Offer ID: {offer_id if offer_id > 0 else 'n/a'}",
            f"Machine ID: {machine_id if machine_id > 0 else 'n/a'}",
            f"Host ID: {host_id if host_id > 0 else 'n/a'}",
            "",
            f"Blacklist file: {self._resolve_cloud_blacklist_path()}",
        ]
        if not messagebox.askyesno("Cloud Blacklist", "\n".join(confirm_lines)):
            return

        data = self._load_cloud_blacklist_data()
        if offer_id > 0:
            data["blocked_offer_ids"].add(offer_id)
        if machine_id > 0:
            data["blocked_machine_ids"].add(machine_id)
        if host_id > 0:
            data["blocked_host_ids"].add(host_id)
        self._save_cloud_blacklist_data(data)
        self._refresh_cloud_blacklist_summary()
        self.message_queue.put(("status", f"Cloud blacklist updated from instance {instance_id}."))

    def _on_cloud_processing_setting_changed(self, *_):
        self._refresh_cloud_processing_summary()

    def _bind_cloud_processing_summary_traces(self):
        tracked_vars = [
            self.model_backend_var,
            self.cloud_profile_var,
            self.cloud_target_width_override_var,
            self.cloud_target_height_override_var,
            self.cloud_window_size_override_var,
            self.cloud_overlap_override_var,
            self.cloud_require_verified_hosts_var,
            self.window_size,
            self.overlap,
            self.stereopilot_window_size_var,
            self.stereopilot_overlap_var,
        ]
        for tk_var in tracked_vars:
            try:
                tk_var.trace_add("write", self._on_cloud_processing_setting_changed)
            except Exception:
                continue

    def _apply_cloud_processing_overrides_visibility(self):
        frame = self.cloud_processing_overrides_frame
        btn = self.cloud_processing_overrides_toggle_btn
        if frame is None or btn is None:
            return
        if self.cloud_processing_overrides_expanded:
            frame.grid()
            btn.configure(text="  ↳ Hide Advanced Cloud Overrides")
        else:
            frame.grid_remove()
            btn.configure(text="  ↳ Show Advanced Cloud Overrides")

    def _toggle_cloud_processing_overrides_visibility(self):
        self.cloud_processing_overrides_expanded = not bool(self.cloud_processing_overrides_expanded)
        self._apply_cloud_processing_overrides_visibility()

    def _close_cloud_settings_dialog(self):
        tracked_widgets = list(self.cloud_settings_widgets)
        if self.cloud_settings_dialog is not None:
            try:
                self.cloud_settings_dialog.destroy()
            except tk.TclError:
                pass
        if tracked_widgets:
            self.widgets_to_disable_during_processing = [
                w for w in self.widgets_to_disable_during_processing if w not in tracked_widgets
            ]
        self.cloud_settings_dialog = None
        self.cloud_settings_widgets = []
        self.cloud_processing_overrides_toggle_btn = None
        self.cloud_processing_overrides_frame = None
        self.cloud_options_expanded_var.set(False)
        self._apply_cloud_options_visibility()

    def _register_cloud_dialog_widget(self, widget):
        self.cloud_settings_widgets.append(widget)
        self.widgets_to_disable_during_processing.append(widget)
        return widget

    def _open_cloud_settings_dialog(self):
        if self.cloud_settings_dialog is not None:
            try:
                if self.cloud_settings_dialog.winfo_exists():
                    self.cloud_settings_dialog.lift()
                    self.cloud_settings_dialog.focus_force()
                    self.cloud_options_expanded_var.set(True)
                    self._apply_cloud_options_visibility()
                    return
            except tk.TclError:
                self._close_cloud_settings_dialog()

        dialog = tk.Toplevel(self.root)
        dialog.title("Cloud Dispatch Settings")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.protocol("WM_DELETE_WINDOW", self._close_cloud_settings_dialog)

        self.cloud_settings_dialog = dialog
        self.cloud_settings_widgets = [dialog]
        self.cloud_options_expanded_var.set(True)

        outer = ttk.Frame(dialog, padding=12)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(outer, text="GPU Profile:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        combo_profile = self._register_cloud_dialog_widget(
            ttk.Combobox(
                outer,
                textvariable=self.cloud_profile_var,
                values=["5090_32gb", "rtx_pro_6000_96gb", "nvidia_48gb_single"],
                width=30,
                state="readonly",
            )
        )
        combo_profile.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        combo_profile.bind("<<ComboboxSelected>>", lambda _evt: self._refresh_cloud_processing_summary())
        row += 1

        ttk.Label(outer, text="Profile Defaults:").grid(row=row, column=0, sticky="ne", padx=5, pady=2)
        lbl_profile_defaults = self._register_cloud_dialog_widget(
            ttk.Label(
                outer,
                textvariable=self.cloud_profile_default_summary_var,
                anchor="w",
                justify="left",
                wraplength=460,
            )
        )
        lbl_profile_defaults.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        row += 1

        ttk.Label(outer, text="Cloud Resolution Override:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        resolution_override_frame = ttk.Frame(outer)
        resolution_override_frame.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        spin_cloud_width = self._register_cloud_dialog_widget(
            ttk.Spinbox(
                resolution_override_frame,
                from_=0,
                to=8192,
                increment=8,
                textvariable=self.cloud_target_width_override_var,
                width=8,
            )
        )
        spin_cloud_width.pack(side=tk.LEFT)
        ttk.Label(resolution_override_frame, text="x").pack(side=tk.LEFT, padx=(4, 4))
        spin_cloud_height = self._register_cloud_dialog_widget(
            ttk.Spinbox(
                resolution_override_frame,
                from_=0,
                to=8192,
                increment=8,
                textvariable=self.cloud_target_height_override_var,
                width=8,
            )
        )
        spin_cloud_height.pack(side=tk.LEFT)
        ttk.Label(resolution_override_frame, text="(0 = profile default)").pack(side=tk.LEFT, padx=(8, 0))
        row += 1

        ttk.Label(outer, text="Effective Cloud Params:").grid(row=row, column=0, sticky="ne", padx=5, pady=2)
        lbl_effective = self._register_cloud_dialog_widget(
            ttk.Label(
                outer,
                textvariable=self.cloud_effective_processing_summary_var,
                anchor="w",
                justify="left",
                wraplength=460,
            )
        )
        lbl_effective.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        row += 1

        self.cloud_processing_overrides_toggle_btn = self._register_cloud_dialog_widget(
            ttk.Button(
                outer,
                text="  ↳ Show Advanced Cloud Overrides",
                command=self._toggle_cloud_processing_overrides_visibility,
            )
        )
        self.cloud_processing_overrides_toggle_btn.grid(row=row, column=0, columnspan=2, sticky="w", padx=(10, 5), pady=2)
        row += 1

        self.cloud_processing_overrides_frame = self._register_cloud_dialog_widget(ttk.Frame(outer))
        self.cloud_processing_overrides_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=(20, 0), pady=(0, 4))
        self.cloud_processing_overrides_frame.columnconfigure(1, weight=1)

        adv_row = 0
        ttk.Label(self.cloud_processing_overrides_frame, text="Window Override:").grid(
            row=adv_row, column=0, sticky="e", padx=5, pady=2
        )
        spin_window_override = self._register_cloud_dialog_widget(
            ttk.Spinbox(
                self.cloud_processing_overrides_frame,
                from_=0,
                to=512,
                increment=1,
                textvariable=self.cloud_window_size_override_var,
                width=10,
            )
        )
        spin_window_override.grid(row=adv_row, column=1, sticky="w", padx=(5, 0), pady=2)
        ttk.Label(self.cloud_processing_overrides_frame, text="(0 = inherit main dialog)").grid(
            row=adv_row, column=2, sticky="w", padx=(8, 0), pady=2
        )
        adv_row += 1

        ttk.Label(self.cloud_processing_overrides_frame, text="Overlap Override:").grid(
            row=adv_row, column=0, sticky="e", padx=5, pady=2
        )
        spin_overlap_override = self._register_cloud_dialog_widget(
            ttk.Spinbox(
                self.cloud_processing_overrides_frame,
                from_=-1,
                to=512,
                increment=1,
                textvariable=self.cloud_overlap_override_var,
                width=10,
            )
        )
        spin_overlap_override.grid(row=adv_row, column=1, sticky="w", padx=(5, 0), pady=2)
        ttk.Label(self.cloud_processing_overrides_frame, text="(-1 = inherit main dialog)").grid(
            row=adv_row, column=2, sticky="w", padx=(8, 0), pady=2
        )
        adv_row += 1

        lbl_inherit = self._register_cloud_dialog_widget(
            ttk.Label(
                self.cloud_processing_overrides_frame,
                textvariable=self.cloud_inherited_processing_summary_var,
                anchor="w",
                justify="left",
                wraplength=430,
            )
        )
        lbl_inherit.grid(row=adv_row, column=0, columnspan=3, sticky="w", padx=5, pady=(2, 4))
        row += 1

        ttk.Label(outer, text="Image:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_image = self._register_cloud_dialog_widget(
            ttk.Entry(outer, textvariable=self.cloud_image_var, width=44)
        )
        entry_image.grid(row=row, column=1, sticky="ew", padx=(5, 0), pady=2)
        row += 1

        ttk.Label(outer, text="Disk (GB):").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        spin_disk = self._register_cloud_dialog_widget(
            ttk.Spinbox(outer, from_=20, to=4096, increment=5, textvariable=self.cloud_disk_gb_var, width=12)
        )
        spin_disk.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        row += 1

        reuse_instance_cb = self._register_cloud_dialog_widget(
            ttk.Checkbutton(
                outer,
                text="Reuse existing instance when profile matches",
                variable=self.cloud_reuse_existing_instance_var,
            )
        )
        reuse_instance_cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        row += 1

        ttk.Label(outer, text="Cached Instance ID:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        cached_instance_frame = ttk.Frame(outer)
        cached_instance_frame.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        entry_cached_instance = self._register_cloud_dialog_widget(
            ttk.Entry(cached_instance_frame, textvariable=self.cloud_last_instance_id_var, width=10)
        )
        entry_cached_instance.pack(side=tk.LEFT)
        lbl_cached_profile = self._register_cloud_dialog_widget(
            ttk.Label(cached_instance_frame, textvariable=self.cloud_last_instance_profile_var)
        )
        lbl_cached_profile.pack(side=tk.LEFT, padx=(8, 0))
        row += 1

        ttk.Label(outer, text="Cached Host:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        cached_host_frame = ttk.Frame(outer)
        cached_host_frame.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        entry_cached_host = self._register_cloud_dialog_widget(
            ttk.Entry(cached_host_frame, textvariable=self.cloud_last_host_var, width=26)
        )
        entry_cached_host.pack(side=tk.LEFT, padx=(0, 4))
        entry_cached_port = self._register_cloud_dialog_widget(
            ttk.Entry(cached_host_frame, textvariable=self.cloud_last_port_var, width=7)
        )
        entry_cached_port.pack(side=tk.LEFT)
        row += 1

        ttk.Label(outer, text="Cached Offer/Machine/Host:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        cached_ids_frame = ttk.Frame(outer)
        cached_ids_frame.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        ttk.Label(cached_ids_frame, text="Offer").pack(side=tk.LEFT)
        entry_cached_offer = self._register_cloud_dialog_widget(
            ttk.Entry(cached_ids_frame, textvariable=self.cloud_last_offer_id_var, width=8)
        )
        entry_cached_offer.pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(cached_ids_frame, text="Machine").pack(side=tk.LEFT)
        entry_cached_machine = self._register_cloud_dialog_widget(
            ttk.Entry(cached_ids_frame, textvariable=self.cloud_last_machine_id_var, width=8)
        )
        entry_cached_machine.pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(cached_ids_frame, text="Host").pack(side=tk.LEFT)
        entry_cached_host_id = self._register_cloud_dialog_widget(
            ttk.Entry(cached_ids_frame, textvariable=self.cloud_last_host_id_var, width=8)
        )
        entry_cached_host_id.pack(side=tk.LEFT, padx=(4, 0))
        row += 1

        admin_btn_frame = ttk.Frame(outer)
        admin_btn_frame.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 2))
        btn_force_start = self._register_cloud_dialog_widget(
            ttk.Button(
                admin_btn_frame,
                text="Force Start",
                command=self._force_start_cached_cloud_instance_from_ui,
                width=12,
            )
        )
        btn_force_start.pack(side=tk.LEFT, padx=(0, 4))
        btn_force_stop = self._register_cloud_dialog_widget(
            ttk.Button(
                admin_btn_frame,
                text="Force Stop",
                command=self._force_stop_cached_cloud_instance_from_ui,
                width=12,
            )
        )
        btn_force_stop.pack(side=tk.LEFT, padx=(0, 4))
        btn_clear_cache = self._register_cloud_dialog_widget(
            ttk.Button(
                admin_btn_frame,
                text="Clear Cache",
                command=self._clear_cached_cloud_instance_from_ui,
                width=12,
            )
        )
        btn_clear_cache.pack(side=tk.LEFT)
        row += 1

        blacklist_btn_frame = ttk.Frame(outer)
        blacklist_btn_frame.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 2))
        btn_blacklist_cached = self._register_cloud_dialog_widget(
            ttk.Button(
                blacklist_btn_frame,
                text="Blacklist Cached Host",
                command=self._blacklist_cached_cloud_target_from_ui,
                width=20,
            )
        )
        btn_blacklist_cached.pack(side=tk.LEFT, padx=(0, 8))
        lbl_blacklist_summary = self._register_cloud_dialog_widget(
            ttk.Label(blacklist_btn_frame, textvariable=self.cloud_blacklist_summary_var)
        )
        lbl_blacklist_summary.pack(side=tk.LEFT)
        row += 1

        ttk.Label(outer, text="SSH Identity Key:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        identity_frame = ttk.Frame(outer)
        identity_frame.grid(row=row, column=1, sticky="ew", padx=(5, 0), pady=2)
        identity_frame.columnconfigure(0, weight=1)
        entry_identity = self._register_cloud_dialog_widget(
            ttk.Entry(identity_frame, textvariable=self.cloud_identity_file_var, width=34)
        )
        entry_identity.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        btn_identity = self._register_cloud_dialog_widget(
            ttk.Button(
                identity_frame,
                text="Browse...",
                command=lambda: self._browse_file_into_var(
                    self.cloud_identity_file_var,
                    "Select SSH Private Key",
                    [("Private keys", "*"), ("All files", "*.*")],
                ),
                width=10,
            )
        )
        btn_identity.grid(row=0, column=1, sticky="w")
        row += 1

        ttk.Label(outer, text="Vast API Env File:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        vast_env_frame = ttk.Frame(outer)
        vast_env_frame.grid(row=row, column=1, sticky="ew", padx=(5, 0), pady=2)
        vast_env_frame.columnconfigure(0, weight=1)
        entry_vast_env = self._register_cloud_dialog_widget(
            ttk.Entry(vast_env_frame, textvariable=self.cloud_vast_env_file_var, width=34)
        )
        entry_vast_env.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        btn_vast_env = self._register_cloud_dialog_widget(
            ttk.Button(
                vast_env_frame,
                text="Browse...",
                command=lambda: self._browse_file_into_var(
                    self.cloud_vast_env_file_var,
                    "Select Vast API Env File",
                    [("Env files", "*.env"), ("All files", "*.*")],
                ),
                width=10,
            )
        )
        btn_vast_env.grid(row=0, column=1, sticky="w")
        row += 1

        ttk.Label(outer, text="HF Env File:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        hf_env_frame = ttk.Frame(outer)
        hf_env_frame.grid(row=row, column=1, sticky="ew", padx=(5, 0), pady=2)
        hf_env_frame.columnconfigure(0, weight=1)
        entry_hf_env = self._register_cloud_dialog_widget(
            ttk.Entry(hf_env_frame, textvariable=self.cloud_hf_env_file_var, width=34)
        )
        entry_hf_env.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        btn_hf_env = self._register_cloud_dialog_widget(
            ttk.Button(
                hf_env_frame,
                text="Browse...",
                command=lambda: self._browse_file_into_var(
                    self.cloud_hf_env_file_var,
                    "Select Hugging Face Env File",
                    [("Env files", "*.env"), ("All files", "*.*")],
                ),
                width=10,
            )
        )
        btn_hf_env.grid(row=0, column=1, sticky="w")
        row += 1

        no_hf_cb = self._register_cloud_dialog_widget(
            ttk.Checkbutton(
                outer,
                text="Skip HF env payload (--no-hf-env)",
                variable=self.cloud_no_hf_env_var,
            )
        )
        no_hf_cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        row += 1

        private_registry_login_cb = self._register_cloud_dialog_widget(
            ttk.Checkbutton(
                outer,
                text="Use private registry login for image pull (--login)",
                variable=self.cloud_use_private_registry_login_var,
            )
        )
        private_registry_login_cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        row += 1

        ttk.Label(outer, text="Remote User:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_remote_user = self._register_cloud_dialog_widget(
            ttk.Entry(outer, textvariable=self.cloud_remote_user_var, width=18)
        )
        entry_remote_user.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        row += 1

        ttk.Label(outer, text="Remote Root:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_remote_root = self._register_cloud_dialog_widget(
            ttk.Entry(outer, textvariable=self.cloud_remote_root_var, width=44)
        )
        entry_remote_root.grid(row=row, column=1, sticky="ew", padx=(5, 0), pady=2)
        row += 1

        ttk.Label(outer, text="Remote Venv:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_remote_venv = self._register_cloud_dialog_widget(
            ttk.Entry(outer, textvariable=self.cloud_remote_venv_var, width=44)
        )
        entry_remote_venv.grid(row=row, column=1, sticky="ew", padx=(5, 0), pady=2)
        row += 1

        ttk.Label(outer, text="Offer Limit:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        spin_offer_limit = self._register_cloud_dialog_widget(
            ttk.Spinbox(outer, from_=1, to=200, increment=1, textvariable=self.cloud_offer_limit_var, width=12)
        )
        spin_offer_limit.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        row += 1

        require_verified_cb = self._register_cloud_dialog_widget(
            ttk.Checkbutton(
                outer,
                text="Require verified hosts in Vast search",
                variable=self.cloud_require_verified_hosts_var,
            )
        )
        require_verified_cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        row += 1

        ttk.Label(outer, text="Max $/hr (0=off):").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_max_dph = self._register_cloud_dialog_widget(
            ttk.Entry(outer, textvariable=self.cloud_max_dph_var, width=12)
        )
        entry_max_dph.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        row += 1

        ttk.Label(outer, text="Expected Runtime (h):").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_expected_runtime = self._register_cloud_dialog_widget(
            ttk.Entry(outer, textvariable=self.cloud_expected_runtime_hours_var, width=12)
        )
        entry_expected_runtime.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        row += 1

        ttk.Label(outer, text="Expected Upload (GB):").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_expected_up = self._register_cloud_dialog_widget(
            ttk.Entry(outer, textvariable=self.cloud_expected_upload_gb_var, width=12)
        )
        entry_expected_up.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        row += 1

        ttk.Label(outer, text="Expected Download (GB):").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry_expected_down = self._register_cloud_dialog_widget(
            ttk.Entry(outer, textvariable=self.cloud_expected_download_gb_var, width=12)
        )
        entry_expected_down.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)
        row += 1

        auto_destroy_cb = self._register_cloud_dialog_widget(
            ttk.Checkbutton(
                outer,
                text="Auto destroy cloud instance after GUI run",
                variable=self.cloud_auto_destroy_instance_var,
            )
        )
        auto_destroy_cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        row += 1

        close_btn = self._register_cloud_dialog_widget(
            ttk.Button(outer, text="Close", command=self._close_cloud_settings_dialog, width=12)
        )
        close_btn.grid(row=row, column=0, columnspan=2, sticky="e", padx=5, pady=(8, 2))

        self._refresh_cloud_processing_summary()
        self._apply_cloud_processing_overrides_visibility()
        self._apply_cloud_options_visibility()

    def _apply_cloud_options_visibility(self):
        if not hasattr(self, 'cloud_settings_toggle_btn'):
            return
        is_open = (
            self.cloud_settings_dialog is not None
            and bool(self.cloud_settings_dialog.winfo_exists())
        )
        self.cloud_options_expanded_var.set(bool(is_open))
        button_text = "  ↳ Cloud Settings Open" if is_open else "  ↳ Configure Cloud Dispatch..."
        try:
            self.cloud_settings_toggle_btn.configure(text=button_text)
        except tk.TclError:
            pass

    def toggle_cloud_settings_visibility(self):
        self._open_cloud_settings_dialog()

    def _parse_cloud_env_file_data(self, path_value: str) -> Dict[str, str]:
        parsed: Dict[str, str] = {}
        candidate = path_value.strip()
        if not candidate:
            return parsed
        env_path = os.path.expanduser(candidate)
        if not os.path.isfile(env_path):
            return parsed

        try:
            if _parse_cloud_env_file is not None:
                parsed = _parse_cloud_env_file(Path(env_path))
            else:
                with open(env_path, "r", encoding="utf-8") as env_file:
                    for raw in env_file.readlines():
                        line = raw.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("export "):
                            line = line[len("export "):].strip()
                        if "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip("'").strip('"')
                        if key:
                            parsed[key] = value
        except Exception as parse_exc:
            _logger.warning(f"Cloud env parsing warning for '{env_path}': {parse_exc}")
        return parsed

    def _resolve_cloud_api_key(self) -> str:
        env_data = self._parse_cloud_env_file_data(self.cloud_vast_env_file_var.get())
        for key in ("VAST_API_KEY", "VASTAI_API_KEY", "VAST_KEY"):
            value = env_data.get(key, "").strip()
            if value and "REPLACE_WITH" not in value.upper():
                return value
        return ""

    def _resolve_cloud_api_base_url(self) -> str:
        env_data = self._parse_cloud_env_file_data(self.cloud_vast_env_file_var.get())
        for key in ("VAST_API_URL", "VAST_URL", "VAST_SERVER_URL"):
            value = env_data.get(key, "").strip()
            if value:
                return value.rstrip("/")
        return self.DEFAULT_VAST_API_BASE_URL

    def _extract_instance_row_from_payload(self, payload, instance_id: int) -> Optional[Dict[str, Any]]:
        if cloud_core is not None:
            return cloud_core.extract_instance_row_from_payload(payload, instance_id)
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

    def _extract_status_from_instance_row(self, row: Dict[str, Any]) -> str:
        if cloud_core is not None:
            return cloud_core.extract_instance_status_from_row(row)
        for key in ("actual_status", "status", "cur_state", "state", "status_msg", "intended_status"):
            value = str(row.get(key, "")).strip()
            if value:
                return value
        return ""

    def _extract_ssh_from_instance_row(self, row: Dict[str, Any]) -> Tuple[str, int]:
        if cloud_core is not None:
            return cloud_core.extract_ssh_from_instance_row(row)
        host = str(row.get("ssh_host", "") or "").strip()
        port_value = row.get("ssh_port")
        port = 0
        try:
            port = int(port_value)
        except Exception:
            port = 0

        # Prefer explicit exposed 22/tcp mapping when available.
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

        # Match Vast CLI behavior for jupyter runtype when using ssh_host/ssh_port path.
        if not used_22_map and port > 0:
            runtype = str(row.get("image_runtype", "") or "").lower()
            if "jupyter" in runtype:
                port += 1

        if host and port > 0:
            return host, port
        return "", 0

    def _fetch_cloud_instance_row_http(self, instance_id: int) -> Optional[Dict[str, Any]]:
        api_key = self._resolve_cloud_api_key()
        if not api_key:
            return None

        base_url = self._resolve_cloud_api_base_url()
        query = urlencode({"owner": "me", "api_key": api_key})
        url = f"{base_url}/api/v0/instances?{query}"
        try:
            with urlopen(url, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return self._extract_instance_row_from_payload(payload, instance_id)
        except Exception as http_exc:
            _logger.debug(f"[CLOUD] HTTP instance poll failed for {instance_id}: {http_exc}")
            return None

    def _fetch_cloud_instance_row_cli(self, instance_id: int) -> Optional[Dict[str, Any]]:
        show_cmd = self._build_vast_cli_cmd(["show", "instances", "--raw"])
        rc, lines = self._run_external_command_with_logging(
            show_cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if rc != 0:
            return None
        payload = self._parse_json_like_payload("\n".join(lines).strip())
        return self._extract_instance_row_from_payload(payload, instance_id)

    def _get_cloud_instance_row(self, instance_id: int) -> Optional[Dict[str, Any]]:
        row = self._fetch_cloud_instance_row_http(instance_id)
        if row is not None:
            return row
        return self._fetch_cloud_instance_row_cli(instance_id)

    def _build_vast_cli_cmd(self, args: List[str]) -> List[str]:
        cmd = ["vastai"] + list(args)
        api_key = self._resolve_cloud_api_key()
        if api_key:
            cmd.extend(["--api-key", api_key])
        return cmd

    def _parse_json_like_payload(self, raw_text: str):
        if cloud_core is not None:
            return cloud_core.parse_json_like(raw_text)
        text = (raw_text or "").strip()
        if not text:
            return None
        candidates = [text]
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines and lines[-1] not in candidates:
            candidates.append(lines[-1])
        for payload in candidates:
            for parser in (json.loads, ast.literal_eval):
                try:
                    return parser(payload)
                except Exception:
                    continue
        return None

    def _parse_ssh_url_text(self, raw_text: str) -> Tuple[str, str, int]:
        payload = self._parse_json_like_payload(raw_text)
        text = ""
        if isinstance(payload, dict):
            for key in ("ssh_url", "url", "value"):
                maybe = payload.get(key)
                if isinstance(maybe, str) and maybe.strip():
                    text = maybe.strip()
                    break
        elif isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, str):
                text = first.strip()
            elif isinstance(first, dict):
                for key in ("ssh_url", "url", "value"):
                    maybe = first.get(key)
                    if isinstance(maybe, str) and maybe.strip():
                        text = maybe.strip()
                        break
        if not text:
            text = (raw_text or "").strip()
        if not text:
            return "", "", 0

        patterns = [
            r"ssh://(?P<user>[^@]+)@(?P<host>[^:\s]+):(?P<port>\d+)",
            r"ssh\s+(?P<user>[^@\s]+)@(?P<host>[^\s]+)\s+-p\s+(?P<port>\d+)",
            r"-p\s+(?P<port>\d+)\s+(?P<user>[^@\s]+)@(?P<host>[^\s]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            user = match.group("user")
            host = match.group("host")
            try:
                port = int(match.group("port"))
            except Exception:
                port = 0
            if user and host and port > 0:
                return user, host, port
        return "", "", 0

    def _normalized_gpu_ram_gb(self, offer: Dict[str, Any]) -> float:
        if cloud_core is not None:
            return cloud_core.normalized_gpu_ram_gb(offer)
        try:
            raw = float(offer.get("gpu_ram", 0.0) or 0.0)
        except Exception:
            raw = 0.0
        if raw <= 0:
            return 0.0
        return raw / 1024.0 if raw > 1000.0 else raw

    def _offer_hourly_cost(self, offer: Dict[str, Any]) -> float:
        if cloud_core is not None:
            return cloud_core.offer_hourly_cost(offer)
        for key in ("dph_total", "discounted_dph_total", "dph"):
            if key in offer:
                try:
                    return float(offer.get(key) or 0.0)
                except Exception:
                    return 0.0
        return 0.0

    def _offer_cost_per_tb(self, offer: Dict[str, Any], direction: str) -> float:
        if cloud_core is not None:
            return cloud_core.offer_cost_per_tb(offer, direction)
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

    def _estimate_offer_total_cost(self, offer: Dict[str, Any]) -> Dict[str, float]:
        expected_runtime_hours = max(0.0, self._safe_float_from_tk_var(self.cloud_expected_runtime_hours_var, 1.0))
        expected_upload_gb = max(0.0, self._safe_float_from_tk_var(self.cloud_expected_upload_gb_var, 0.0))
        expected_download_gb = max(0.0, self._safe_float_from_tk_var(self.cloud_expected_download_gb_var, 0.0))
        if cloud_core is not None:
            cost_data = cloud_core.estimate_offer_total_cost(
                offer,
                expected_runtime_hours=expected_runtime_hours,
                expected_upload_gb=expected_upload_gb,
                expected_download_gb=expected_download_gb,
            )
            return {
                **cost_data,
                "expected_runtime_hours": expected_runtime_hours,
                "expected_upload_gb": expected_upload_gb,
                "expected_download_gb": expected_download_gb,
            }
        hourly = self._offer_hourly_cost(offer)
        up_tb = self._offer_cost_per_tb(offer, "up")
        down_tb = self._offer_cost_per_tb(offer, "down")
        transfer_cost = ((expected_upload_gb / 1024.0) * up_tb) + ((expected_download_gb / 1024.0) * down_tb)
        runtime_cost = hourly * expected_runtime_hours
        total_cost = runtime_cost + transfer_cost
        return {
            "hourly": hourly,
            "up_tb": up_tb,
            "down_tb": down_tb,
            "runtime_cost": runtime_cost,
            "transfer_cost": transfer_cost,
            "total_cost": total_cost,
            "expected_runtime_hours": expected_runtime_hours,
            "expected_upload_gb": expected_upload_gb,
            "expected_download_gb": expected_download_gb,
        }

    def _build_cloud_offer_search_query(
        self,
        profile_key: str,
        require_verified_override: Optional[bool] = None,
        max_dph_override: Optional[float] = None,
    ) -> Tuple[str, Dict[str, object]]:
        profile_defaults = self._get_cloud_profile_defaults(profile_key)
        if require_verified_override is None:
            require_verified_hosts = bool(self.cloud_require_verified_hosts_var.get())
        else:
            require_verified_hosts = bool(require_verified_override)
        min_reliability = 0.97
        min_cuda = 12.8
        min_direct_ports = 2
        min_inet_down = 200.0
        min_inet_up = 50.0
        disk_gb = max(20, self._safe_int_from_tk_var(self.cloud_disk_gb_var, 40))
        if max_dph_override is None:
            max_dph = max(0.0, self._safe_float_from_tk_var(self.cloud_max_dph_var, 0.0))
        else:
            try:
                max_dph = max(0.0, float(max_dph_override))
            except Exception:
                max_dph = 0.0
        if cloud_core is not None:
            query = cloud_core.build_offer_search_query(
                profile_defaults,
                disk_gb=disk_gb,
                require_verified_hosts=require_verified_hosts,
                max_dph=max_dph,
                min_cuda=min_cuda,
                min_reliability=min_reliability,
                min_direct_ports=min_direct_ports,
                min_inet_down=min_inet_down,
                min_inet_up=min_inet_up,
            )
            return query, profile_defaults

        offer_gpu_filter = str(profile_defaults.get("offer_gpu_filter", "")).strip()
        min_gpu_ram_gb = max(0.0, float(profile_defaults.get("min_gpu_ram_gb", 0.0)))
        query_parts = [
            "rentable=true",
            "num_gpus=1",
            f"gpu_ram>={min_gpu_ram_gb:.1f}",
            f"cuda_vers>={min_cuda}",
            f"reliability>={min_reliability}",
            f"disk_space>={disk_gb}",
            f"direct_port_count>={min_direct_ports}",
            f"inet_down>={min_inet_down}",
            f"inet_up>={min_inet_up}",
        ]
        if offer_gpu_filter:
            query_parts.insert(0, offer_gpu_filter)
        if require_verified_hosts:
            query_parts.append("verified=true")
        if max_dph > 0.0:
            query_parts.append(f"dph<={max_dph}")
        return " ".join(query_parts), profile_defaults

    def _run_cloud_offer_search_query(self, query: str, offer_limit: int, disk_gb: int) -> List[Dict[str, Any]]:
        search_cmd = self._build_vast_cli_cmd([
            "search",
            "offers",
            query,
            "--raw",
            "--limit",
            str(max(1, int(offer_limit))),
            "--storage",
            str(max(20, int(disk_gb))),
            "--order",
            "dph_total",
            "--no-default",
        ])
        rc, lines = self._run_external_command_with_logging(
            search_cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if rc != 0:
            raise RuntimeError(f"vastai search offers failed with exit code {rc}.")
        raw_payload = "\n".join(lines).strip()
        parsed_payload = self._parse_json_like_payload(raw_payload)
        if not isinstance(parsed_payload, list):
            return []
        return [entry for entry in parsed_payload if isinstance(entry, dict)]

    def _fetch_ranked_cloud_offers_for_confirmation(self, profile_key: str) -> List[Dict[str, Any]]:
        query, profile_defaults = self._build_cloud_offer_search_query(profile_key)
        offer_limit = max(1, self._safe_int_from_tk_var(self.cloud_offer_limit_var, 30))
        disk_gb = max(20, self._safe_int_from_tk_var(self.cloud_disk_gb_var, 40))
        require_verified_hosts = bool(self.cloud_require_verified_hosts_var.get())
        max_dph = max(0.0, self._safe_float_from_tk_var(self.cloud_max_dph_var, 0.0))
        parsed_payload = self._run_cloud_offer_search_query(query, offer_limit=offer_limit, disk_gb=disk_gb)
        if not parsed_payload:
            diagnostics: List[str] = []
            if require_verified_hosts:
                unverified_query, _ = self._build_cloud_offer_search_query(
                    profile_key,
                    require_verified_override=False,
                    max_dph_override=max_dph,
                )
                unverified_payload = self._run_cloud_offer_search_query(
                    unverified_query,
                    offer_limit=offer_limit,
                    disk_gb=disk_gb,
                )
                if unverified_payload:
                    diagnostics.append(
                        f"{len(unverified_payload)} offers appear when verified-host filter is disabled"
                    )
            if max_dph > 0.0:
                uncapped_query, _ = self._build_cloud_offer_search_query(
                    profile_key,
                    require_verified_override=require_verified_hosts,
                    max_dph_override=0.0,
                )
                uncapped_payload = self._run_cloud_offer_search_query(
                    uncapped_query,
                    offer_limit=offer_limit,
                    disk_gb=disk_gb,
                )
                if uncapped_payload:
                    diagnostics.append(
                        f"{len(uncapped_payload)} offers appear when max $/h cap is removed"
                    )
            message = "No cloud offers returned for current profile/filter settings."
            if diagnostics:
                message += " Possible blockers: " + "; ".join(diagnostics) + "."
            raise RuntimeError(message)

        blacklist_data = self._load_cloud_blacklist_data()
        min_gpu_ram_gb = float(profile_defaults.get("min_gpu_ram_gb", 0.0))
        expected_runtime_hours = max(0.0, self._safe_float_from_tk_var(self.cloud_expected_runtime_hours_var, 1.0))
        expected_upload_gb = max(0.0, self._safe_float_from_tk_var(self.cloud_expected_upload_gb_var, 0.0))
        expected_download_gb = max(0.0, self._safe_float_from_tk_var(self.cloud_expected_download_gb_var, 0.0))
        if cloud_core is not None:
            ranked_offers, skipped_blacklist_count, skipped_vram_count = cloud_core.rank_cloud_offers(
                parsed_payload,
                blacklist_data,
                min_gpu_ram_gb=min_gpu_ram_gb,
                expected_runtime_hours=expected_runtime_hours,
                expected_upload_gb=expected_upload_gb,
                expected_download_gb=expected_download_gb,
                gpu_ram_tolerance_gb=self.CLOUD_GPU_RAM_TOLERANCE_GB,
            )
        else:
            ranked_offers = []
            skipped_blacklist_count = 0
            skipped_vram_count = 0
            for entry in parsed_payload:
                if self._is_cloud_offer_blacklisted(entry, blacklist_data):
                    skipped_blacklist_count += 1
                    continue
                vram_gb = self._normalized_gpu_ram_gb(entry)
                if vram_gb + self.CLOUD_GPU_RAM_TOLERANCE_GB < min_gpu_ram_gb:
                    skipped_vram_count += 1
                    continue
                cost_data = self._estimate_offer_total_cost(entry)
                enriched = dict(entry)
                enriched["_vram_gb"] = vram_gb
                enriched["_hourly"] = cost_data["hourly"]
                enriched["_up_tb"] = cost_data["up_tb"]
                enriched["_down_tb"] = cost_data["down_tb"]
                enriched["_runtime_cost"] = cost_data["runtime_cost"]
                enriched["_transfer_cost"] = cost_data["transfer_cost"]
                enriched["_total_cost"] = cost_data["total_cost"]
                ranked_offers.append(enriched)

        if not ranked_offers:
            detail = (
                f"raw={len(parsed_payload)}, blacklisted={skipped_blacklist_count}, "
                f"vram_filtered={skipped_vram_count}, min_vram={min_gpu_ram_gb:.1f}GB"
            )
            raise RuntimeError(
                f"No cloud offers passed filters for profile '{profile_key}' "
                f"({detail})."
            )

        if cloud_core is None:
            ranked_offers.sort(
                key=lambda offer: (
                    float(offer.get("_total_cost", 1e12)),
                    float(offer.get("_hourly", 1e12)),
                    -float(offer.get("reliability", 0.0) or 0.0),
                )
            )
        return ranked_offers

    def _format_cloud_offer_selection_entry(
        self,
        rank_idx: int,
        offer: Dict[str, Any],
        history_data: Dict[str, Any],
    ) -> str:
        identity = self._cloud_provider_identity_for_offer(offer)
        provider_key = identity["provider_key"]
        provider_label = identity["provider_label"]
        offer_id = int(identity["offer_id"])
        machine_id = int(identity["machine_id"])
        host_id = int(identity["host_id"])

        provider_count = int(history_data.get("provider_counts", {}).get(provider_key, 0) or 0)
        offer_count = int(history_data.get("offer_counts", {}).get(str(offer_id), 0) or 0) if offer_id > 0 else 0

        gpu_name = str(offer.get("gpu_name", "Unknown"))
        location = str(offer.get("geolocation", "Unknown"))
        reliability = float(offer.get("reliability", 0.0) or 0.0)
        vram_gb = float(offer.get("_vram_gb", 0.0) or 0.0)
        inet_down = float(offer.get("inet_down", 0.0) or 0.0)
        inet_up = float(offer.get("inet_up", 0.0) or 0.0)
        hourly = float(offer.get("_hourly", 0.0) or 0.0)
        up_tb = float(offer.get("_up_tb", 0.0) or 0.0)
        down_tb = float(offer.get("_down_tb", 0.0) or 0.0)
        runtime_cost = float(offer.get("_runtime_cost", 0.0) or 0.0)
        transfer_cost = float(offer.get("_transfer_cost", 0.0) or 0.0)
        total_cost = float(offer.get("_total_cost", 0.0) or 0.0)

        return (
            f"#{rank_idx}  Offer ID {offer_id} | Projected total ${total_cost:.4f} "
            f"(runtime ${runtime_cost:.4f} + transfer ${transfer_cost:.4f})\n"
            f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM) | Location: {location} | Reliability: {reliability:.3f}\n"
            f"Network down/up: {inet_down:.0f}/{inet_up:.0f} Mbps | $/hour ${hourly:.4f} | "
            f"up TB ${up_tb:.4f} | down TB ${down_tb:.4f}\n"
            f"Provider: {provider_label} | Provider connections: {provider_count} | "
            f"Offer connections: {offer_count} | machine={machine_id if machine_id > 0 else 'n/a'} | "
            f"host={host_id if host_id > 0 else 'n/a'}"
        )

    def _show_cloud_offer_selection_dialog(
        self,
        profile_key: str,
        source_count: int,
        ranked_offers: List[Dict[str, Any]],
        reason: str = "",
    ) -> Optional[int]:
        if not ranked_offers:
            return None

        top_offers = ranked_offers[:10]
        top_count = len(top_offers)
        settings = self._get_effective_cloud_processing_settings()
        profile_defaults = self._get_cloud_profile_defaults(profile_key)
        blacklist_data = self._load_cloud_blacklist_data()
        blocked_offer_count = len(blacklist_data.get("blocked_offer_ids", set()))
        blocked_machine_count = len(blacklist_data.get("blocked_machine_ids", set()))
        blocked_host_count = len(blacklist_data.get("blocked_host_ids", set()))
        history_data = self._load_cloud_provider_history_data()
        history_path = self._resolve_cloud_provider_history_path()

        expected_runtime_hours = max(0.0, self._safe_float_from_tk_var(self.cloud_expected_runtime_hours_var, 1.0))
        expected_upload_gb = max(0.0, self._safe_float_from_tk_var(self.cloud_expected_upload_gb_var, 0.0))
        expected_download_gb = max(0.0, self._safe_float_from_tk_var(self.cloud_expected_download_gb_var, 0.0))

        header_lines = [
            f"Choose cloud offer for profile '{profile_defaults.get('label', profile_key)}' and {source_count} clip(s).",
            f"Effective cloud params: {settings['target_width']}x{settings['target_height']}, "
            f"window={settings['window_size']}, overlap={settings['overlap']}.",
            f"Showing top {top_count} offers sorted by projected total cost (blacklisted entries excluded).",
            f"Cost model: runtime={expected_runtime_hours:.3f}h, upload={expected_upload_gb:.3f}GB, "
            f"download={expected_download_gb:.3f}GB.",
            f"Require verified hosts: {'Yes' if bool(self.cloud_require_verified_hosts_var.get()) else 'No'} | "
            f"Blacklist active: offers={blocked_offer_count}, machines={blocked_machine_count}, hosts={blocked_host_count}.",
            f"Provider history file: {history_path}",
        ]
        if reason:
            header_lines.extend(["", f"Launch note: {reason}"])

        first_offer_id = int(top_offers[0].get("id", 0) or 0)
        selected_offer_var = tk.IntVar(value=first_offer_id if first_offer_id > 0 else 0)
        selection_result: Dict[str, Optional[int]] = {"offer_id": None}

        dialog = tk.Toplevel(self.root)
        dialog.title("Cloud Offer Selection")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.geometry("1080x780")
        dialog.minsize(860, 560)

        outer = ttk.Frame(dialog, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        header_label = ttk.Label(
            outer,
            text="\n".join(header_lines),
            justify=tk.LEFT,
            anchor="w",
            wraplength=1040,
        )
        header_label.pack(fill=tk.X, anchor="w", pady=(0, 8))

        list_container = ttk.Frame(outer)
        list_container.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(list_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        scrollable_frame = ttk.Frame(canvas)
        canvas_window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        offer_labels: List[ttk.Label] = []

        def _set_selection(offer_id_value: int):
            selected_offer_var.set(int(offer_id_value))

        for idx, offer in enumerate(top_offers, start=1):
            offer_id = int(offer.get("id", 0) or 0)
            row_frame = ttk.Frame(scrollable_frame, padding=(8, 8), relief=tk.GROOVE, borderwidth=1)
            row_frame.pack(fill=tk.X, expand=True, pady=(0, 6))
            row_frame.columnconfigure(1, weight=1)

            offer_radio = ttk.Radiobutton(row_frame, variable=selected_offer_var, value=offer_id)
            offer_radio.grid(row=0, column=0, sticky="n", padx=(0, 8))
            offer_text = self._format_cloud_offer_selection_entry(
                rank_idx=idx,
                offer=offer,
                history_data=history_data,
            )
            offer_label = ttk.Label(row_frame, text=offer_text, justify=tk.LEFT, anchor="w", wraplength=980)
            offer_label.grid(row=0, column=1, sticky="w")
            offer_labels.append(offer_label)

            row_frame.bind("<Button-1>", lambda _event, oid=offer_id: _set_selection(oid))
            offer_label.bind("<Button-1>", lambda _event, oid=offer_id: _set_selection(oid))

        def _on_scrollable_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfigure(canvas_window_id, width=event.width)
            wraplength = max(500, int(event.width) - 90)
            for label in offer_labels:
                label.configure(wraplength=wraplength)
            header_label.configure(wraplength=max(500, int(event.width) - 20))

        scrollable_frame.bind("<Configure>", _on_scrollable_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        button_frame = ttk.Frame(outer)
        button_frame.pack(fill=tk.X, pady=(8, 0))

        def _cancel():
            selection_result["offer_id"] = None
            dialog.destroy()

        def _confirm():
            try:
                selected_offer_id = int(selected_offer_var.get())
            except Exception:
                selected_offer_id = 0
            if selected_offer_id <= 0:
                messagebox.showerror("Cloud Dispatch", "Select an offer before launching.", parent=dialog)
                return
            selection_result["offer_id"] = selected_offer_id
            dialog.destroy()

        ttk.Button(button_frame, text="Cancel", command=_cancel, width=14).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(button_frame, text="Launch Selected Offer", command=_confirm, width=24).pack(side=tk.RIGHT)

        dialog.protocol("WM_DELETE_WINDOW", _cancel)
        try:
            dialog.focus_set()
        except tk.TclError:
            pass
        try:
            if dialog.winfo_exists():
                dialog.wait_visibility()
        except tk.TclError:
            # Window can be closed before map/visibility event settles.
            pass
        try:
            if dialog.winfo_exists():
                self.root.wait_window(dialog)
        except tk.TclError:
            pass
        return selection_result["offer_id"]

    def _confirm_new_cloud_launch(
        self,
        profile_key: str,
        source_count: int,
        reason: str = "",
    ) -> Optional[int]:
        try:
            ranked_offers = self._fetch_ranked_cloud_offers_for_confirmation(profile_key)
            return self._show_cloud_offer_selection_dialog(
                profile_key=profile_key,
                source_count=source_count,
                ranked_offers=ranked_offers,
                reason=reason,
            )
        except Exception as preflight_exc:
            _logger.warning(f"[CLOUD] Offer pricing preflight failed: {preflight_exc}")
            fallback_lines = [
                f"Launch cloud worker profile '{profile_key}' and process {source_count} clip(s)?",
            ]
            if reason:
                fallback_lines.extend(["", reason])
            fallback_lines.extend([
                "",
                "Could not fetch cloud offer ranking from Vast right now.",
                f"Reason: {preflight_exc}",
                "",
                "Continue and let launcher pick the cheapest available offer?",
            ])
            should_continue = messagebox.askyesno("Cloud Dispatch", "\n".join(fallback_lines))
            return 0 if should_continue else None

    def _get_cloud_instance_status(self, instance_id: int) -> str:
        row = self._get_cloud_instance_row(instance_id)
        if not isinstance(row, dict):
            return ""
        return self._extract_status_from_instance_row(row)

    def _is_cloud_tcp_endpoint_ready(self, host: str, port: int, timeout_sec: float = 3.0) -> Tuple[bool, str]:
        host_str = str(host or "").strip()
        try:
            port_int = int(port)
        except Exception:
            port_int = 0
        if not host_str or port_int <= 0:
            return False, "missing host/port"
        try:
            with socket.create_connection((host_str, port_int), timeout=max(0.5, float(timeout_sec))):
                return True, ""
        except Exception as tcp_exc:
            return False, str(tcp_exc)

    def _resolve_cloud_identity_path(self) -> str:
        candidate = self.cloud_identity_file_var.get().strip()
        if not candidate:
            return ""
        return os.path.expanduser(candidate)

    def _is_cloud_ssh_auth_ready(
        self,
        host: str,
        port: int,
        user: str,
        identity_path: str,
        timeout_sec: float = 5.0,
    ) -> Tuple[bool, str]:
        host_str = str(host or "").strip()
        user_str = str(user or "").strip()
        identity_str = str(identity_path or "").strip()
        try:
            port_int = int(port)
        except Exception:
            port_int = 0
        if not host_str or port_int <= 0 or not user_str:
            return False, "missing host/port/user"
        if not identity_str:
            return False, "missing identity path"
        if not os.path.isfile(identity_str):
            return False, f"identity file not found: {identity_str}"

        cmd = [
            "ssh",
            "-p",
            str(port_int),
            "-i",
            identity_str,
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={max(1, int(timeout_sec))}",
            "-o",
            "StrictHostKeyChecking=accept-new",
            f"{user_str}@{host_str}",
            "echo gui_cloud_auth_ready",
        ]
        try:
            result = subprocess.run(
                cmd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except Exception as auth_exc:
            return False, str(auth_exc)

        output = (result.stdout or "").strip()
        if result.returncode == 0:
            return True, ""
        if output:
            return False, output.splitlines()[-1].strip()
        return False, f"ssh exit {result.returncode}"

    def _attach_cloud_identity_to_instance(self, instance_id: int, identity_pub_path: str) -> bool:
        pub_path = str(identity_pub_path or "").strip()
        if instance_id <= 0:
            return False
        if not pub_path or not os.path.isfile(pub_path):
            _logger.warning(f"[CLOUD] SSH public key not found for attach: {pub_path}")
            return False

        try:
            with open(pub_path, "r", encoding="utf-8") as f:
                public_key_text = f.read().strip()
        except Exception as read_exc:
            _logger.warning(f"[CLOUD] Failed reading SSH public key '{pub_path}': {read_exc}")
            return False
        if not public_key_text:
            _logger.warning(f"[CLOUD] SSH public key file is empty: {pub_path}")
            return False

        attach_cmd = self._build_vast_cli_cmd(["attach", "ssh", str(instance_id), public_key_text, "--raw"])
        attach_rc, attach_lines = self._run_external_command_with_logging(
            attach_cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if attach_rc == 0:
            _logger.info(f"[CLOUD] Attached SSH key '{pub_path}' to instance {instance_id}.")
            return True

        attach_tail = (attach_lines[-1].strip() if attach_lines else f"exit code {attach_rc}")
        _logger.warning(
            f"[CLOUD] Failed to attach SSH key '{pub_path}' to instance {instance_id}: {attach_tail}"
        )
        return False

    def _wait_for_cloud_instance_ready(self, instance_id: int, timeout_sec: int = 900, poll_sec: int = 8) -> Dict[str, object]:
        deadline = time.time() + max(15, int(timeout_sec))
        last_status = ""
        last_probe_error = ""
        last_auth_error = ""
        attach_attempted = False
        identity_path = self._resolve_cloud_identity_path()
        identity_pub_path = f"{identity_path}.pub" if identity_path else ""
        remote_user = self.cloud_remote_user_var.get().strip() or "root"
        while time.time() < deadline:
            if self.stop_event.is_set():
                raise RuntimeError("Cancelled while waiting for cloud instance readiness.")

            row = self._get_cloud_instance_row(instance_id)
            status_text = self._extract_status_from_instance_row(row) if isinstance(row, dict) else ""
            if status_text and status_text != last_status:
                _logger.info(f"[CLOUD] Instance {instance_id} status: {status_text}")
                self.message_queue.put(("status", f"Cloud instance {instance_id}: {status_text}"))
                last_status = status_text

            ssh_host, ssh_port = self._extract_ssh_from_instance_row(row) if isinstance(row, dict) else ("", 0)
            if isinstance(row, dict) and (not ssh_host or ssh_port <= 0):
                # Final fallback for older payload shapes where ssh_host/ssh_port are omitted.
                ssh_cmd = self._build_vast_cli_cmd(["ssh-url", str(instance_id)])
                ssh_rc, ssh_lines = self._run_external_command_with_logging(
                    ssh_cmd,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                )
                if ssh_rc == 0:
                    _, ssh_host, ssh_port = self._parse_ssh_url_text("\n".join(ssh_lines))

            if ssh_host and ssh_port > 0:
                auth_ready = False
                auth_error = ""
                if identity_path:
                    auth_ready, auth_error = self._is_cloud_ssh_auth_ready(
                        host=ssh_host,
                        port=ssh_port,
                        user=remote_user,
                        identity_path=identity_path,
                        timeout_sec=5.0,
                    )
                else:
                    # Backward-compat fallback when no identity file is configured.
                    tcp_ready, probe_error = self._is_cloud_tcp_endpoint_ready(ssh_host, ssh_port, timeout_sec=3.0)
                    auth_ready = bool(tcp_ready)
                    auth_error = probe_error if not tcp_ready else ""

                if auth_ready:
                    row_offer_id = 0
                    row_machine_id = 0
                    row_host_id = 0
                    if isinstance(row, dict):
                        try:
                            row_offer_id = int(row.get("offer_id", 0) or 0)
                        except Exception:
                            row_offer_id = 0
                        try:
                            row_machine_id = int(row.get("machine_id", 0) or 0)
                        except Exception:
                            row_machine_id = 0
                        try:
                            row_host_id = int(row.get("host_id", 0) or 0)
                        except Exception:
                            row_host_id = 0
                    return {
                        "instance_id": int(instance_id),
                        "host": ssh_host,
                        "port": int(ssh_port),
                        "offer_id": row_offer_id,
                        "machine_id": row_machine_id,
                        "host_id": row_host_id,
                        "user": self.cloud_remote_user_var.get().strip() or "root",
                        "remote_root": self.cloud_remote_root_var.get().strip() or "/opt/StereoCrafter",
                        "remote_venv": self.cloud_remote_venv_var.get().strip() or "/opt/venv",
                    }
                if identity_path and not attach_attempted and identity_pub_path and os.path.isfile(identity_pub_path):
                    attach_attempted = True
                    self.message_queue.put(("status", f"Cloud instance {instance_id}: attaching SSH key..."))
                    self._attach_cloud_identity_to_instance(instance_id, identity_pub_path)

                if auth_error:
                    if identity_path:
                        if auth_error != last_auth_error:
                            _logger.info(
                                f"[CLOUD] Instance {instance_id} has SSH endpoint {ssh_host}:{ssh_port} "
                                f"but auth is not ready yet: {auth_error}"
                            )
                            self.message_queue.put(("status", f"Cloud instance {instance_id}: waiting for SSH auth..."))
                            last_auth_error = auth_error
                    elif auth_error != last_probe_error:
                        _logger.info(
                            f"[CLOUD] Instance {instance_id} has SSH endpoint {ssh_host}:{ssh_port} "
                            f"but it is not accepting connections yet: {auth_error}"
                        )
                        self.message_queue.put(("status", f"Cloud instance {instance_id}: waiting for SSH service..."))
                        last_probe_error = auth_error
            time.sleep(max(1, int(poll_sec)))

        raise RuntimeError(f"Timed out waiting for cloud instance {instance_id} readiness.")

    def _cache_cloud_instance_connection(self, connection_info: Dict[str, object], profile: str):
        previous_instance_id = int(self.cloud_last_instance_id_var.get() or 0)
        instance_id = int(connection_info.get("instance_id", 0) or 0)
        host = str(connection_info.get("host", "") or "")
        port = int(connection_info.get("port", 22) or 22)
        offer_id = int(connection_info.get("offer_id", 0) or 0)
        machine_id = int(connection_info.get("machine_id", 0) or 0)
        host_id = int(connection_info.get("host_id", 0) or 0)
        if instance_id > 0:
            self.cloud_last_instance_id_var.set(instance_id)
            self.cloud_last_instance_profile_var.set(profile or "")
            self.cloud_last_host_var.set(host)
            self.cloud_last_port_var.set(port)
            if offer_id > 0:
                self.cloud_last_offer_id_var.set(offer_id)
            elif previous_instance_id != instance_id:
                self.cloud_last_offer_id_var.set(0)
            if machine_id > 0:
                self.cloud_last_machine_id_var.set(machine_id)
            elif previous_instance_id != instance_id:
                self.cloud_last_machine_id_var.set(0)
            if host_id > 0:
                self.cloud_last_host_id_var.set(host_id)
            elif previous_instance_id != instance_id:
                self.cloud_last_host_id_var.set(0)

    def _start_cloud_instance_and_wait(self, instance_id: int) -> Dict[str, object]:
        start_cmd = self._build_vast_cli_cmd(["start", "instance", str(instance_id), "--raw"])
        start_rc, _ = self._run_external_command_with_logging(start_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        if start_rc != 0:
            raise RuntimeError(f"Failed to start cloud instance {instance_id}.")
        return self._wait_for_cloud_instance_ready(instance_id)

    def _stop_cloud_instance(self, instance_id: int):
        stop_cmd = self._build_vast_cli_cmd(["stop", "instance", str(instance_id), "--raw"])
        stop_rc, _ = self._run_external_command_with_logging(stop_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        if stop_rc != 0:
            raise RuntimeError(f"Failed to stop cloud instance {instance_id}.")

    def _run_cloud_admin_action_async(self, action_name: str, action_fn):
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Busy", "Processing is currently active. Wait for it to finish before cloud admin actions.")
            return
        self.stop_event.clear()

        def _runner():
            try:
                self.message_queue.put(("status", f"Cloud {action_name}..."))
                action_fn()
                self.message_queue.put(("status", f"Cloud {action_name} complete."))
            except Exception as admin_exc:
                _logger.exception(f"Cloud {action_name} failed: {admin_exc}")
                self.message_queue.put(("status", f"Cloud {action_name} failed: {admin_exc.__class__.__name__}"))

        threading.Thread(target=_runner, daemon=True).start()

    def _force_start_cached_cloud_instance_from_ui(self):
        try:
            instance_id = int(self.cloud_last_instance_id_var.get())
        except Exception:
            instance_id = 0
        if instance_id <= 0:
            messagebox.showerror("Cloud", "Set a valid cached instance ID first.")
            return

        def _start():
            conn = self._start_cloud_instance_and_wait(instance_id)
            profile = self.cloud_last_instance_profile_var.get().strip() or self.cloud_profile_var.get().strip()
            self._cache_cloud_instance_connection(conn, profile)

        self._run_cloud_admin_action_async(f"start instance {instance_id}", _start)

    def _force_stop_cached_cloud_instance_from_ui(self):
        try:
            instance_id = int(self.cloud_last_instance_id_var.get())
        except Exception:
            instance_id = 0
        if instance_id <= 0:
            messagebox.showerror("Cloud", "Set a valid cached instance ID first.")
            return

        def _stop():
            self._stop_cloud_instance(instance_id)

        self._run_cloud_admin_action_async(f"stop instance {instance_id}", _stop)

    def _clear_cached_cloud_instance_from_ui(self):
        self.cloud_last_instance_id_var.set(0)
        self.cloud_last_instance_profile_var.set("")
        self.cloud_last_host_var.set("")
        self.cloud_last_port_var.set(22)
        self.cloud_last_offer_id_var.set(0)
        self.cloud_last_machine_id_var.set(0)
        self.cloud_last_host_id_var.set(0)
        self.message_queue.put(("status", "Cloud cached instance cleared."))

    def _run_external_command_with_logging(self, cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, List[str]]:
        redacted_cmd_parts: List[str] = []
        redact_next = False
        for part in cmd:
            part_str = str(part)
            if redact_next:
                redacted_cmd_parts.append("***")
                redact_next = False
                continue
            redacted_cmd_parts.append(part_str)
            if part_str in {"--api-key", "--login", "--env", "--registry-login", "--vast-api-key"}:
                redact_next = True
        log_cmd = " ".join(shlex.quote(part) for part in redacted_cmd_parts)
        _logger.info(f"[CLOUD] $ {log_cmd}")
        output_lines: List[str] = []
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        self.active_external_process = process
        terminate_requested = False
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        line_buffer = ""
        latest_progress_text = ""
        progress_last_emit_ts = 0.0
        progress_emit_interval_sec = 1.0
        progress_updates_seen = 0
        progress_updates_emitted = 0
        last_progress_emitted_text = ""

        def _looks_like_progress_update(text: str) -> bool:
            cleaned = text.strip()
            if not cleaned:
                return False
            lower = cleaned.lower()
            if "to-chk=" in lower or "xfr#" in lower:
                return True
            if "it/s" in lower or "s/it" in lower:
                return True
            if "%" in cleaned and "|" in cleaned:
                return True
            if "%" in cleaned and "/" in cleaned and ("eta" in lower or "<" in cleaned):
                return True
            return False

        def _emit_line(text: str) -> None:
            cleaned = text.strip()
            if not cleaned:
                return
            output_lines.append(cleaned)
            _logger.info(cleaned)

        def _emit_progress(text: str, force: bool = False) -> bool:
            nonlocal progress_last_emit_ts
            nonlocal progress_updates_emitted
            nonlocal last_progress_emitted_text
            cleaned = text.strip()
            if not cleaned:
                return False
            if cleaned == last_progress_emitted_text:
                return False
            now = time.time()
            if not force and (now - progress_last_emit_ts) < progress_emit_interval_sec:
                return False
            progress_last_emit_ts = now
            progress_updates_emitted += 1
            last_progress_emitted_text = cleaned
            _emit_line(cleaned)
            return True

        def _consume_text(decoded_text: str) -> None:
            nonlocal line_buffer
            nonlocal latest_progress_text
            nonlocal progress_updates_seen
            for ch in decoded_text:
                if ch == "\r":
                    candidate = line_buffer
                    line_buffer = ""
                    if candidate.strip():
                        if _looks_like_progress_update(candidate):
                            latest_progress_text = candidate
                            progress_updates_seen += 1
                            _emit_progress(latest_progress_text, force=False)
                        else:
                            latest_progress_text = ""
                            _emit_line(candidate)
                    continue
                if ch == "\n":
                    if line_buffer.strip():
                        _emit_line(line_buffer)
                    elif latest_progress_text.strip():
                        _emit_progress(latest_progress_text, force=True)
                    line_buffer = ""
                    latest_progress_text = ""
                    continue
                line_buffer += ch

        try:
            if process.stdout is not None:
                stdout_fd = process.stdout.fileno()
                while True:
                    if self.stop_event.is_set() and process.poll() is None and not terminate_requested:
                        _logger.info("Cancellation requested; terminating active cloud command...")
                        terminate_requested = True
                        try:
                            process.terminate()
                        except Exception:
                            pass

                    ready, _, _ = select.select([stdout_fd], [], [], 0.25)
                    if ready:
                        chunk = os.read(stdout_fd, 4096)
                        if chunk:
                            _consume_text(decoder.decode(chunk))
                        else:
                            break

                    now = time.time()
                    if latest_progress_text.strip() and (now - progress_last_emit_ts) >= progress_emit_interval_sec:
                        _emit_progress(latest_progress_text, force=True)

                    if process.poll() is not None:
                        remaining = process.stdout.read() or b""
                        if remaining:
                            _consume_text(decoder.decode(remaining))
                        break

                _consume_text(decoder.decode(b"", final=True))
                if line_buffer.strip():
                    _emit_line(line_buffer)
                elif latest_progress_text.strip():
                    _emit_progress(latest_progress_text, force=True)
            returncode = process.wait()
            suppressed = progress_updates_seen - progress_updates_emitted
            if suppressed > 0:
                _logger.info(f"[CLOUD] (suppressed {suppressed} carriage-return progress updates)")
            return returncode, output_lines
        finally:
            self.active_external_process = None
            if process.stdout is not None:
                try:
                    process.stdout.close()
                except Exception:
                    pass

    def _build_cloud_launcher_command(
        self,
        repo_root: str,
        base_cfg_path: str,
        generated_cfg_path: str,
        selected_offer_id: int = 0,
    ) -> List[str]:
        cloud_settings = self._get_effective_cloud_processing_settings()
        cloud_image = self.cloud_image_var.get().strip()
        if not cloud_image:
            raise RuntimeError("Cloud image is empty. Set it in Cloud Dispatch settings.")

        cloud_target_width_override = int(cloud_settings["target_width_override"])
        cloud_target_height_override = int(cloud_settings["target_height_override"])
        cloud_window_override = int(cloud_settings["window_size_override"])
        cloud_overlap_override = int(cloud_settings["overlap_override"])
        # vast_worker_launch expects explicit values; for window/overlap we inherit main dialog when sentinel is used.
        launcher_window_override = (
            cloud_window_override if cloud_window_override > 0 else int(cloud_settings["base_window"])
        )
        launcher_overlap_override = (
            cloud_overlap_override if cloud_overlap_override >= 0 else int(cloud_settings["base_overlap"])
        )

        cmd = [
            sys.executable,
            os.path.join(repo_root, "cloud", "vast_worker_launch.py"),
            "--profile",
            self.cloud_profile_var.get().strip() or "5090_32gb",
            "--image",
            cloud_image,
            "--disk",
            str(int(self.cloud_disk_gb_var.get())),
            "--base-config",
            base_cfg_path,
            "--output-config",
            generated_cfg_path,
            "--remote-user",
            self.cloud_remote_user_var.get().strip() or "root",
            "--remote-root",
            self.cloud_remote_root_var.get().strip() or "/opt/StereoCrafter",
            "--remote-venv",
            self.cloud_remote_venv_var.get().strip() or "/opt/venv",
            "--offer-limit",
            str(int(self.cloud_offer_limit_var.get())),
            "--blacklist-file",
            self._resolve_cloud_blacklist_path(),
            "--expected-runtime-hours",
            str(float(self.cloud_expected_runtime_hours_var.get())),
            "--expected-upload-gb",
            str(float(self.cloud_expected_upload_gb_var.get())),
            "--expected-download-gb",
            str(float(self.cloud_expected_download_gb_var.get())),
            "--target-width-override",
            str(cloud_target_width_override),
            "--target-height-override",
            str(cloud_target_height_override),
            "--window-size-override",
            str(launcher_window_override),
            "--overlap-override",
            str(launcher_overlap_override),
            "--force-cpu-offload",
            self.cpu_offload.get(),
            "--yes",
            "--no-go-prompt",
        ]

        if not bool(self.cloud_require_verified_hosts_var.get()):
            cmd.append("--allow-unverified")

        max_dph_value = float(self.cloud_max_dph_var.get())
        if max_dph_value > 0.0:
            cmd.extend(["--max-dph", str(max_dph_value)])

        if int(selected_offer_id) > 0:
            cmd.extend(["--offer-id", str(int(selected_offer_id))])

        identity_file = self.cloud_identity_file_var.get().strip()
        if identity_file:
            cmd.extend(["--identity", os.path.expanduser(identity_file)])

        vast_env_file = self.cloud_vast_env_file_var.get().strip()
        if vast_env_file:
            cmd.extend(["--vast-env-file", os.path.expanduser(vast_env_file)])

        if bool(self.cloud_no_hf_env_var.get()):
            cmd.append("--no-hf-env")
        else:
            hf_env_file = self.cloud_hf_env_file_var.get().strip()
            if hf_env_file:
                cmd.extend(["--hf-env-file", os.path.expanduser(hf_env_file)])

        # Default-safe behavior: do not send registry login creds unless explicitly enabled.
        if not bool(self.cloud_use_private_registry_login_var.get()):
            cmd.append("--skip-image-login")

        git_sync_branch = self._detect_local_git_branch(repo_root)
        if git_sync_branch:
            cmd.extend(["--git-sync-branch", git_sync_branch])

        return cmd

    def _detect_local_git_branch(self, repo_root: str) -> str:
        try:
            probe = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception:
            return ""
        branch = (probe.stdout or "").strip()
        if not branch or branch == "HEAD":
            return ""
        return branch

    def _launch_cloud_worker_for_gui_run(
        self,
        source_specs_to_process: List[Dict],
        effective_seed_for_run: int,
        selected_offer_id: int = 0,
    ) -> Dict[str, object]:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        cloud_tmp_root = os.path.join(self.output_dir.get(), ".cloud_gui_tmp")
        os.makedirs(cloud_tmp_root, exist_ok=True)
        run_stamp = time.strftime("%Y%m%d_%H%M%S")

        base_cfg_path = os.path.join(cloud_tmp_root, f"cloud_gui_base_{run_stamp}.json")
        generated_cfg_path = os.path.join(cloud_tmp_root, f"cloud_gui_generated_{run_stamp}.json")

        base_settings = self._collect_all_settings()
        base_settings["seed"] = effective_seed_for_run
        base_settings["output_dir"] = self.output_dir.get()
        if source_specs_to_process:
            base_settings["input_dir_or_file_var"] = source_specs_to_process[0]["path"]

        with open(base_cfg_path, "w", encoding="utf-8") as base_cfg_file:
            json.dump(base_settings, base_cfg_file, indent=2)

        launcher_cmd = self._build_cloud_launcher_command(
            repo_root=repo_root,
            base_cfg_path=base_cfg_path,
            generated_cfg_path=generated_cfg_path,
            selected_offer_id=selected_offer_id,
        )

        self.status_message_var.set("Cloud: selecting offer and launching worker...")
        self.root.update_idletasks()
        rc, _ = self._run_external_command_with_logging(launcher_cmd, cwd=repo_root)
        if rc != 0:
            raise RuntimeError(f"Cloud worker launch failed with exit code {rc}.")
        if self.stop_event.is_set():
            raise RuntimeError("Cancelled while launching cloud worker.")

        generated_cfg = load_json_file(generated_cfg_path)
        if not isinstance(generated_cfg, dict):
            raise RuntimeError("Cloud launch did not produce a readable generated config.")
        cloud_remote = generated_cfg.get("cloud_remote", {})
        if not isinstance(cloud_remote, dict):
            raise RuntimeError("Generated cloud config missing 'cloud_remote' details.")

        host = str(cloud_remote.get("vast_host", "")).strip()
        if not host:
            raise RuntimeError("Cloud launch completed but host was not found in generated config.")

        connection = {
            "instance_id": int(cloud_remote.get("vast_instance_id", 0) or 0),
            "host": host,
            "port": int(cloud_remote.get("vast_ssh_port", 22) or 22),
            "offer_id": int(cloud_remote.get("vast_offer_id", 0) or 0),
            "machine_id": int(cloud_remote.get("vast_machine_id", 0) or 0),
            "host_id": int(cloud_remote.get("vast_host_id", 0) or 0),
            "gpu_name": str(cloud_remote.get("vast_gpu_name", "") or ""),
            "geolocation": str(cloud_remote.get("vast_geolocation", "") or ""),
            "hourly_cost": float(cloud_remote.get("vast_hourly_cost", 0.0) or 0.0),
            "reliability": float(cloud_remote.get("vast_reliability", 0.0) or 0.0),
            "user": str(cloud_remote.get("vast_user", "root") or "root"),
            "remote_root": str(cloud_remote.get("remote_root", self.cloud_remote_root_var.get()) or self.cloud_remote_root_var.get()),
            "remote_venv": str(cloud_remote.get("remote_venv", self.cloud_remote_venv_var.get()) or self.cloud_remote_venv_var.get()),
            "generated_cfg_path": generated_cfg_path,
            "base_cfg_path": base_cfg_path,
            "target_width": int(generated_cfg.get("target_width", self._safe_int_from_tk_var(self.target_width, 1920))),
            "target_height": int(generated_cfg.get("target_height", self._safe_int_from_tk_var(self.target_height, 1040))),
            "window_size": int(generated_cfg.get("window_size", self._safe_int_from_tk_var(self.window_size, 75))),
            "overlap": int(generated_cfg.get("overlap", self._safe_int_from_tk_var(self.overlap, 25))),
            "cpu_offload": str(generated_cfg.get("cpu_offload", self.cpu_offload.get()) or self.cpu_offload.get()),
            "git_sync_branch": str(cloud_remote.get("git_sync_branch", self._detect_local_git_branch(repo_root)) or ""),
        }
        self._cache_cloud_instance_connection(connection, self.cloud_profile_var.get().strip())
        return connection

    def _detect_source_video_resolution_for_cloud(self, source_path: str) -> Tuple[int, int]:
        path = str(source_path or "").strip()
        if not path:
            return 0, 0
        if not os.path.isfile(path):
            return 0, 0
        try:
            stream_info_result = get_video_stream_info(path)
        except Exception as meta_exc:
            _logger.warning(f"[CLOUD] Could not probe source resolution for '{os.path.basename(path)}': {meta_exc}")
            return 0, 0

        stream_info = None
        if isinstance(stream_info_result, tuple) and len(stream_info_result) >= 1:
            if isinstance(stream_info_result[0], dict):
                stream_info = stream_info_result[0]
        elif isinstance(stream_info_result, dict):
            stream_info = stream_info_result

        if not isinstance(stream_info, dict):
            return 0, 0

        try:
            width = int(stream_info.get("width", 0) or 0)
            height = int(stream_info.get("height", 0) or 0)
        except Exception:
            return 0, 0
        if width <= 0 or height <= 0:
            return 0, 0
        return width, height

    def _build_cloudctl_run_job_command(
        self,
        connection_info: Dict[str, object],
        source_spec: Dict,
        effective_seed_for_run: int,
        source_idx: int,
        total_sources: int,
    ) -> Tuple[List[str], str]:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        cloud_settings = self._get_effective_cloud_processing_settings()
        cloud_target_width = int(connection_info.get("target_width", int(cloud_settings["target_width"])))
        cloud_target_height = int(connection_info.get("target_height", int(cloud_settings["target_height"])))
        cloud_window_size = int(connection_info.get("window_size", int(cloud_settings["window_size"])))
        cloud_overlap = int(connection_info.get("overlap", int(cloud_settings["overlap"])))
        cloud_cpu_offload = str(connection_info.get("cpu_offload", self.cpu_offload.get()) or self.cpu_offload.get())
        cloud_git_sync_branch = str(
            connection_info.get("git_sync_branch", self._detect_local_git_branch(repo_root))
            or self._detect_local_git_branch(repo_root)
        )
        model_backend = str(self.model_backend_var.get() or "depthcrafter").strip().lower()
        if model_backend not in ("depthcrafter", "geometrycrafter_diff", "geometrycrafter_determ", "stereopilot"):
            model_backend = "depthcrafter"
        if model_backend == "stereopilot":
            cloud_window_size = max(1, int(self.stereopilot_window_size_var.get()))
            cloud_overlap = max(0, int(self.stereopilot_overlap_var.get()))

        geometry_model_path = str(self.geometry_model_path_var.get() or "TencentARC/GeometryCrafter").strip()
        remote_root = str(connection_info.get("remote_root", "") or "").strip()
        geometry_repo_path = os.path.join(remote_root, "weights", "GeometryCrafter") if remote_root else ""
        geometry_cache_dir = os.path.join(remote_root, "weights", "hf_cache") if remote_root else ""
        geometry_decode_chunk = max(1, int(self.geometry_decode_chunk_size_var.get()))
        geometry_low_memory = bool(self.geometry_low_memory_usage_var.get())
        geometry_force_projection = bool(self.geometry_force_projection_var.get())
        geometry_force_fixed_focal = bool(self.geometry_force_fixed_focal_var.get())
        geometry_use_extract_interp = bool(self.geometry_use_extract_interp_var.get())
        stereopilot_model_path = str(self.stereopilot_model_path_var.get() or "KlingTeam/StereoPilot").strip()
        stereopilot_base_model_path = str(self.stereopilot_base_model_path_var.get() or "Wan-AI/Wan2.1-T2V-1.3B").strip()
        stereopilot_repo_path = os.path.join(remote_root, "weights", "StereoPilot") if remote_root else ""
        stereopilot_cache_dir = os.path.join(remote_root, "weights", "hf_cache") if remote_root else ""
        stereopilot_prompt = str(self.stereopilot_prompt_var.get() or "").strip()
        stereopilot_output_mode = str(self.stereopilot_output_mode_var.get() or "side_by_side").strip().lower()
        if stereopilot_output_mode not in {"opposite_eye", "side_by_side", "both"}:
            stereopilot_output_mode = "side_by_side"
        stereopilot_target_width = max(32, int(self.stereopilot_target_width_var.get()))
        stereopilot_target_height = max(32, int(self.stereopilot_target_height_var.get()))
        stereopilot_target_fps = max(1.0, float(self.stereopilot_target_fps_var.get()))
        stereopilot_sampling_steps = max(1, int(self.stereopilot_sampling_steps_var.get()))
        stereopilot_guide_scale = float(self.stereopilot_guide_scale_var.get())
        stereopilot_shift = float(self.stereopilot_shift_var.get())
        stereopilot_domain_label = 1 if int(self.stereopilot_domain_label_var.get()) != 0 else 0
        stereopilot_dtype = str(self.stereopilot_dtype_var.get() or "bfloat16").strip().lower()
        stereopilot_transformer_dtype = str(self.stereopilot_transformer_dtype_var.get() or "float8").strip().lower()
        original_basename = source_spec["basename"]

        if (
            bool(cloud_settings.get("use_source_resolution", False))
            and int(cloud_settings.get("target_width_override", 0)) <= 0
            and int(cloud_settings.get("target_height_override", 0)) <= 0
        ):
            src_w, src_h = self._detect_source_video_resolution_for_cloud(source_spec.get("path", ""))
            if src_w > 0 and src_h > 0:
                cloud_target_width = int(src_w)
                cloud_target_height = int(src_h)
                _logger.info(
                    f"[CLOUD] {original_basename}: using source resolution {cloud_target_width}x{cloud_target_height} "
                    f"for profile '{cloud_settings.get('profile_key', '')}'."
                )
            else:
                _logger.warning(
                    f"[CLOUD] {original_basename}: source resolution probe failed; falling back to "
                    f"{cloud_target_width}x{cloud_target_height}."
                )

        # Keep cloud output naming identical to local naming to avoid downstream pipeline mismatches.
        job_name = original_basename
        cloud_output_format = str(self.merge_output_format_var.get())
        if cloud_output_format not in ("mp4", "main10_mp4"):
            cloud_output_format = "main10_mp4"

        cmd = [
            sys.executable,
            "-u",
            os.path.join(repo_root, "cloud", "cloudctl.py"),
            "run-job",
            "--host",
            str(connection_info["host"]),
            "--user",
            str(connection_info["user"]),
            "--port",
            str(int(connection_info["port"])),
            "--remote-root",
            str(connection_info["remote_root"]),
            "--venv-name",
            str(connection_info["remote_venv"]),
            "--local-input",
            source_spec["path"],
            "--job-name",
            job_name,
            "--download-dir",
            self.output_dir.get(),
            "--target-width",
            str(cloud_target_width),
            "--target-height",
            str(cloud_target_height),
            "--window-size",
            str(cloud_window_size),
            "--overlap",
            str(cloud_overlap),
            "--inference-steps",
            str(int(self.inference_steps.get())),
            "--guidance-scale",
            str(float(self.guidance_scale.get())),
            "--seed",
            str(int(effective_seed_for_run)),
            "--target-fps",
            str(float(self.target_fps.get())),
            "--process-length",
            str(int(self.process_length.get())),
            "--output-format",
            cloud_output_format,
            "--cpu-offload",
            cloud_cpu_offload,
            "--model-backend",
            model_backend,
            "--geometry-model-path",
            geometry_model_path,
            "--geometry-repo-path",
            geometry_repo_path,
            "--geometry-cache-dir",
            geometry_cache_dir,
            "--geometry-decode-chunk-size",
            str(geometry_decode_chunk),
            "--stereopilot-model-path",
            stereopilot_model_path,
            "--stereopilot-base-model-path",
            stereopilot_base_model_path,
            "--stereopilot-repo-path",
            stereopilot_repo_path,
            "--stereopilot-cache-dir",
            stereopilot_cache_dir,
            "--stereopilot-output-mode",
            stereopilot_output_mode,
            "--stereopilot-target-width",
            str(stereopilot_target_width),
            "--stereopilot-target-height",
            str(stereopilot_target_height),
            "--stereopilot-target-fps",
            str(stereopilot_target_fps),
            "--stereopilot-sampling-steps",
            str(stereopilot_sampling_steps),
            "--stereopilot-guide-scale",
            str(stereopilot_guide_scale),
            "--stereopilot-shift",
            str(stereopilot_shift),
            "--stereopilot-domain-label",
            str(stereopilot_domain_label),
            "--stereopilot-dtype",
            stereopilot_dtype,
            "--stereopilot-transformer-dtype",
            stereopilot_transformer_dtype,
        ]

        identity_file = self.cloud_identity_file_var.get().strip()
        if identity_file:
            cmd.extend(["--identity", os.path.expanduser(identity_file)])

        if cloud_git_sync_branch:
            cmd.extend(["--git-sync-branch", cloud_git_sync_branch])

        if bool(self.disable_xformers_var.get()):
            cmd.append("--disable-xformers")
        if bool(self.use_cudnn_benchmark.get()):
            cmd.append("--use-cudnn-benchmark")
        if bool(self.use_local_models_only_var.get()):
            cmd.append("--local-files-only")
        if geometry_low_memory:
            cmd.append("--geometry-low-memory-usage")
        cmd.append("--geometry-force-projection" if geometry_force_projection else "--no-geometry-force-projection")
        cmd.append("--geometry-force-fixed-focal" if geometry_force_fixed_focal else "--no-geometry-force-fixed-focal")
        cmd.append("--geometry-use-extract-interp" if geometry_use_extract_interp else "--no-geometry-use-extract-interp")
        cmd.append("--stereopilot-use-sidecar-prompt" if bool(self.stereopilot_use_sidecar_prompt_var.get()) else "--no-stereopilot-use-sidecar-prompt")
        if stereopilot_prompt:
            cmd.extend(["--stereopilot-prompt", stereopilot_prompt])

        return cmd, job_name

    def _write_cloud_batch_manifest(self, source_specs: List[Dict]) -> str:
        output_root = self.output_dir.get().strip() or "."
        cloud_tmp_dir = os.path.join(output_root, ".cloud_gui_tmp")
        os.makedirs(cloud_tmp_dir, exist_ok=True)
        manifest_path = os.path.join(
            cloud_tmp_dir,
            f"cloud_batch_manifest_{time.strftime('%Y%m%d_%H%M%S')}.txt",
        )
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            for spec in source_specs:
                clip_path = str(spec.get("path", "")).strip()
                clip_name = str(spec.get("basename", "")).strip()
                if not clip_path:
                    continue
                if not clip_name:
                    clip_name = os.path.splitext(os.path.basename(clip_path))[0]
                # Format: <job_name><TAB><path>
                manifest_file.write(f"{clip_name}\t{clip_path}\n")
        return manifest_path

    def _build_cloudctl_run_batch_command(
        self,
        connection_info: Dict[str, object],
        source_specs: List[Dict],
        effective_seed_for_run: int,
    ) -> Tuple[List[str], str]:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        cloud_settings = self._get_effective_cloud_processing_settings()
        cloud_target_width = int(connection_info.get("target_width", int(cloud_settings["target_width"])))
        cloud_target_height = int(connection_info.get("target_height", int(cloud_settings["target_height"])))
        cloud_window_size = int(connection_info.get("window_size", int(cloud_settings["window_size"])))
        cloud_overlap = int(connection_info.get("overlap", int(cloud_settings["overlap"])))
        cloud_cpu_offload = str(connection_info.get("cpu_offload", self.cpu_offload.get()) or self.cpu_offload.get())
        cloud_git_sync_branch = str(
            connection_info.get("git_sync_branch", self._detect_local_git_branch(repo_root))
            or self._detect_local_git_branch(repo_root)
        )
        model_backend = str(self.model_backend_var.get() or "depthcrafter").strip().lower()
        if model_backend not in ("depthcrafter", "geometrycrafter_diff", "geometrycrafter_determ", "stereopilot"):
            model_backend = "depthcrafter"
        if model_backend == "stereopilot":
            cloud_window_size = max(1, int(self.stereopilot_window_size_var.get()))
            cloud_overlap = max(0, int(self.stereopilot_overlap_var.get()))

        geometry_model_path = str(self.geometry_model_path_var.get() or "TencentARC/GeometryCrafter").strip()
        remote_root = str(connection_info.get("remote_root", "") or "").strip()
        geometry_repo_path = os.path.join(remote_root, "weights", "GeometryCrafter") if remote_root else ""
        geometry_cache_dir = os.path.join(remote_root, "weights", "hf_cache") if remote_root else ""
        geometry_decode_chunk = max(1, int(self.geometry_decode_chunk_size_var.get()))
        geometry_low_memory = bool(self.geometry_low_memory_usage_var.get())
        geometry_force_projection = bool(self.geometry_force_projection_var.get())
        geometry_force_fixed_focal = bool(self.geometry_force_fixed_focal_var.get())
        geometry_use_extract_interp = bool(self.geometry_use_extract_interp_var.get())
        stereopilot_model_path = str(self.stereopilot_model_path_var.get() or "KlingTeam/StereoPilot").strip()
        stereopilot_base_model_path = str(self.stereopilot_base_model_path_var.get() or "Wan-AI/Wan2.1-T2V-1.3B").strip()
        stereopilot_repo_path = os.path.join(remote_root, "weights", "StereoPilot") if remote_root else ""
        stereopilot_cache_dir = os.path.join(remote_root, "weights", "hf_cache") if remote_root else ""
        stereopilot_prompt = str(self.stereopilot_prompt_var.get() or "").strip()
        stereopilot_output_mode = str(self.stereopilot_output_mode_var.get() or "side_by_side").strip().lower()
        if stereopilot_output_mode not in {"opposite_eye", "side_by_side", "both"}:
            stereopilot_output_mode = "side_by_side"
        stereopilot_target_width = max(32, int(self.stereopilot_target_width_var.get()))
        stereopilot_target_height = max(32, int(self.stereopilot_target_height_var.get()))
        stereopilot_target_fps = max(1.0, float(self.stereopilot_target_fps_var.get()))
        stereopilot_sampling_steps = max(1, int(self.stereopilot_sampling_steps_var.get()))
        stereopilot_guide_scale = float(self.stereopilot_guide_scale_var.get())
        stereopilot_shift = float(self.stereopilot_shift_var.get())
        stereopilot_domain_label = 1 if int(self.stereopilot_domain_label_var.get()) != 0 else 0
        stereopilot_dtype = str(self.stereopilot_dtype_var.get() or "bfloat16").strip().lower()
        stereopilot_transformer_dtype = str(self.stereopilot_transformer_dtype_var.get() or "float8").strip().lower()
        if (
            bool(cloud_settings.get("use_source_resolution", False))
            and int(cloud_settings.get("target_width_override", 0)) <= 0
            and int(cloud_settings.get("target_height_override", 0)) <= 0
            and source_specs
        ):
            src_w, src_h = self._detect_source_video_resolution_for_cloud(source_specs[0].get("path", ""))
            if src_w > 0 and src_h > 0:
                cloud_target_width = int(src_w)
                cloud_target_height = int(src_h)
                _logger.info(
                    f"[CLOUD] Batch mode using source resolution from first clip: {cloud_target_width}x{cloud_target_height}."
                )
            else:
                _logger.warning(
                    "[CLOUD] Could not determine source resolution for batch run; using configured cloud target %sx%s.",
                    cloud_target_width,
                    cloud_target_height,
                )

        manifest_path = self._write_cloud_batch_manifest(source_specs)
        cloud_output_format = str(self.merge_output_format_var.get())
        if cloud_output_format not in ("mp4", "main10_mp4"):
            cloud_output_format = "main10_mp4"

        cmd = [
            sys.executable,
            "-u",
            os.path.join(repo_root, "cloud", "cloudctl.py"),
            "run-batch",
            "--host",
            str(connection_info["host"]),
            "--user",
            str(connection_info["user"]),
            "--port",
            str(int(connection_info["port"])),
            "--remote-root",
            str(connection_info["remote_root"]),
            "--venv-name",
            str(connection_info["remote_venv"]),
            "--input-manifest",
            manifest_path,
            "--download-dir",
            self.output_dir.get(),
            "--target-width",
            str(cloud_target_width),
            "--target-height",
            str(cloud_target_height),
            "--window-size",
            str(cloud_window_size),
            "--overlap",
            str(cloud_overlap),
            "--inference-steps",
            str(int(self.inference_steps.get())),
            "--guidance-scale",
            str(float(self.guidance_scale.get())),
            "--seed",
            str(int(effective_seed_for_run)),
            "--target-fps",
            str(float(self.target_fps.get())),
            "--process-length",
            str(int(self.process_length.get())),
            "--output-format",
            cloud_output_format,
            "--cpu-offload",
            cloud_cpu_offload,
            "--model-backend",
            model_backend,
            "--geometry-model-path",
            geometry_model_path,
            "--geometry-repo-path",
            geometry_repo_path,
            "--geometry-cache-dir",
            geometry_cache_dir,
            "--geometry-decode-chunk-size",
            str(geometry_decode_chunk),
            "--stereopilot-model-path",
            stereopilot_model_path,
            "--stereopilot-base-model-path",
            stereopilot_base_model_path,
            "--stereopilot-repo-path",
            stereopilot_repo_path,
            "--stereopilot-cache-dir",
            stereopilot_cache_dir,
            "--stereopilot-output-mode",
            stereopilot_output_mode,
            "--stereopilot-target-width",
            str(stereopilot_target_width),
            "--stereopilot-target-height",
            str(stereopilot_target_height),
            "--stereopilot-target-fps",
            str(stereopilot_target_fps),
            "--stereopilot-sampling-steps",
            str(stereopilot_sampling_steps),
            "--stereopilot-guide-scale",
            str(stereopilot_guide_scale),
            "--stereopilot-shift",
            str(stereopilot_shift),
            "--stereopilot-domain-label",
            str(stereopilot_domain_label),
            "--stereopilot-dtype",
            stereopilot_dtype,
            "--stereopilot-transformer-dtype",
            stereopilot_transformer_dtype,
            "--prefetch-window",
            "3",
        ]

        identity_file = self.cloud_identity_file_var.get().strip()
        if identity_file:
            cmd.extend(["--identity", os.path.expanduser(identity_file)])

        if cloud_git_sync_branch:
            cmd.extend(["--git-sync-branch", cloud_git_sync_branch])

        if bool(self.disable_xformers_var.get()):
            cmd.append("--disable-xformers")
        if bool(self.use_cudnn_benchmark.get()):
            cmd.append("--use-cudnn-benchmark")
        if bool(self.use_local_models_only_var.get()):
            cmd.append("--local-files-only")
        if geometry_low_memory:
            cmd.append("--geometry-low-memory-usage")
        cmd.append("--geometry-force-projection" if geometry_force_projection else "--no-geometry-force-projection")
        cmd.append("--geometry-force-fixed-focal" if geometry_force_fixed_focal else "--no-geometry-force-fixed-focal")
        cmd.append("--geometry-use-extract-interp" if geometry_use_extract_interp else "--no-geometry-use-extract-interp")
        cmd.append("--stereopilot-use-sidecar-prompt" if bool(self.stereopilot_use_sidecar_prompt_var.get()) else "--no-stereopilot-use-sidecar-prompt")
        if stereopilot_prompt:
            cmd.extend(["--stereopilot-prompt", stereopilot_prompt])

        return cmd, manifest_path

    def _extract_cloud_batch_outcomes(self, output_lines: List[str]) -> Tuple[set, set]:
        completed_jobs = set()
        failed_filenames = set()
        complete_pattern = re.compile(r"Job complete:\s*([A-Za-z0-9._-]+)")
        failed_pattern = re.compile(r"Batch item failed for\s+(.+?):")
        for line in output_lines:
            text = str(line or "").strip()
            if not text:
                continue
            complete_match = complete_pattern.search(text)
            if complete_match:
                completed_jobs.add(complete_match.group(1).strip())
            failed_match = failed_pattern.search(text)
            if failed_match:
                failed_filenames.add(failed_match.group(1).strip())
        return completed_jobs, failed_filenames

    def _update_gui_info_from_cloud_status(self, job_info_stub: Dict, status_json_path: str):
        status_data = load_json_file(status_json_path)
        if not isinstance(status_data, dict):
            return
        metadata = status_data.get("metadata", {})
        if not isinstance(metadata, dict):
            return
        self._update_gui_info_on_job_finish(job_info_stub, metadata)

    def _resolve_cloud_status_json_path(self, job_name: str) -> str:
        output_root = self.output_dir.get()
        candidates = [
            os.path.join(output_root, f"{job_name}_job_status.json"),
            os.path.join(output_root, "job_status.json"),
            os.path.join(output_root, job_name, "job_status.json"),  # Legacy layout fallback.
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return candidates[0]

    def _destroy_cloud_instance(self, instance_id: int):
        if instance_id <= 0:
            return
        destroy_cmd = self._build_vast_cli_cmd(["destroy", "instance", str(instance_id), "--raw"])
        try:
            rc, _ = self._run_external_command_with_logging(destroy_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
            if rc != 0:
                _logger.warning(f"Cloud instance destroy returned exit code {rc} for instance {instance_id}.")
        except Exception as destroy_exc:
            _logger.warning(f"Cloud instance destroy failed for {instance_id}: {destroy_exc}")

    def _get_reusable_cloud_connection(self, instance_id: int) -> Dict[str, object]:
        status = (self._get_cloud_instance_status(instance_id) or "").strip().lower()
        if status:
            _logger.info(f"[CLOUD] Cached instance {instance_id} reported status: {status}")
        if any(token in status for token in ("stopped", "offline", "exited")):
            _logger.info(f"[CLOUD] Starting cached instance {instance_id}...")
            return self._start_cloud_instance_and_wait(instance_id)
        if any(token in status for token in ("destroy", "failed", "error")):
            raise RuntimeError(f"Cached instance {instance_id} is not reusable (status: {status}).")
        return self._wait_for_cloud_instance_ready(instance_id)

    def _run_cloud_dispatch_mode(self, source_specs_to_process: List[Dict], effective_seed_for_run: int):
        if not source_specs_to_process:
            return
        unsupported_sources = [
            spec for spec in source_specs_to_process
            if spec.get("type") not in ("video_file", "single_video_file")
        ]
        if unsupported_sources:
            raise RuntimeError("Cloud dispatch mode currently supports video files only (no image sequences/images).")
        selected_profile = self.cloud_profile_var.get().strip() or "5090_32gb"
        cloud_settings = self._get_effective_cloud_processing_settings()
        _logger.info(
            "[CLOUD] Effective run settings | profile=%s target=%sx%s window=%s overlap=%s",
            selected_profile,
            cloud_settings["target_width"],
            cloud_settings["target_height"],
            cloud_settings["window_size"],
            cloud_settings["overlap"],
        )
        reuse_enabled = bool(self.cloud_reuse_existing_instance_var.get())
        cached_profile = self.cloud_last_instance_profile_var.get().strip()
        try:
            cached_instance_id = int(self.cloud_last_instance_id_var.get())
        except Exception:
            cached_instance_id = 0
        connection_info: Optional[Dict[str, object]] = None
        should_prompt_new_launch = False
        new_launch_reason = ""
        selected_offer_id_for_launch = 0

        if reuse_enabled and cached_instance_id > 0 and cached_profile == selected_profile:
            cached_host = self.cloud_last_host_var.get().strip()
            cached_port = self._safe_int_from_tk_var(self.cloud_last_port_var, 22)
            cached_host_text = f"{cached_host}:{cached_port}" if cached_host else "unknown"
            reuse_choice = messagebox.askyesnocancel(
                "Cloud Dispatch",
                f"Reuse cached instance {cached_instance_id} for profile '{selected_profile}'?\n"
                f"Cached host: {cached_host_text}\n\n"
                "Yes = reuse/start cached instance\n"
                "No = launch new cheapest worker\n"
                "Cancel = abort"
            )
            if reuse_choice is None:
                _logger.info("Cloud dispatch cancelled by user.")
                self.status_message_var.set("Cloud dispatch cancelled.")
                return
            if reuse_choice:
                try:
                    self.status_message_var.set(f"Cloud: checking cached instance {cached_instance_id}...")
                    self.root.update_idletasks()
                    connection_info = self._get_reusable_cloud_connection(cached_instance_id)
                    self._cache_cloud_instance_connection(connection_info, selected_profile)
                except Exception as reuse_exc:
                    _logger.warning(f"Cached instance reuse failed: {reuse_exc}")
                    fallback_launch = messagebox.askyesno(
                        "Cloud Dispatch",
                        f"Could not reuse cached instance {cached_instance_id}.\n\n"
                        f"Reason: {reuse_exc}\n\nLaunch a new worker instead?"
                    )
                    if not fallback_launch:
                        self.status_message_var.set("Cloud dispatch cancelled.")
                        return
                    should_prompt_new_launch = True
                    new_launch_reason = f"Reuse failed for cached instance {cached_instance_id}: {reuse_exc}"
            else:
                should_prompt_new_launch = True
                new_launch_reason = (
                    f"Cached instance {cached_instance_id} was not selected for reuse."
                )
            # If user pressed "No", fall through to new launch.
        else:
            should_prompt_new_launch = True
            if not reuse_enabled:
                new_launch_reason = "Reuse is disabled in Cloud settings."
            elif cached_instance_id <= 0:
                new_launch_reason = "No cached instance is set for reuse."
            elif cached_profile != selected_profile:
                new_launch_reason = (
                    f"Cached instance profile '{cached_profile or 'n/a'}' does not match selected profile '{selected_profile}'."
                )

        if connection_info is None and should_prompt_new_launch:
            selected_offer_id_or_none = self._confirm_new_cloud_launch(
                profile_key=selected_profile,
                source_count=len(source_specs_to_process),
                reason=new_launch_reason,
            )
            if selected_offer_id_or_none is None:
                _logger.info("Cloud dispatch cancelled by user before worker launch.")
                self.status_message_var.set("Cloud dispatch cancelled.")
                return
            selected_offer_id_for_launch = int(selected_offer_id_or_none)

        if connection_info is None:
            connection_info = self._launch_cloud_worker_for_gui_run(
                source_specs_to_process=source_specs_to_process,
                effective_seed_for_run=effective_seed_for_run,
                selected_offer_id=selected_offer_id_for_launch,
            )
            connection_origin = "new_launch"
        else:
            connection_origin = "reused_instance"
        self._record_cloud_connection_history(
            connection_info=connection_info,
            profile_key=selected_profile,
            connection_origin=connection_origin,
        )
        instance_id = int(connection_info.get("instance_id", 0) or 0)
        total_sources_processed = 0
        had_cloud_errors = False
        use_cloud_batch_queue = len(source_specs_to_process) > 1

        try:
            if use_cloud_batch_queue:
                self.status_message_var.set(
                    f"Cloud batch dispatch: pipelined queue for {len(source_specs_to_process)} clips..."
                )
                self.root.update_idletasks()
                batch_cmd, manifest_path = self._build_cloudctl_run_batch_command(
                    connection_info=connection_info,
                    source_specs=source_specs_to_process,
                    effective_seed_for_run=effective_seed_for_run,
                )
                _logger.info(f"[CLOUD] Using queued batch mode with manifest: {manifest_path}")
                batch_rc, batch_output_lines = self._run_external_command_with_logging(
                    batch_cmd,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                )
                completed_jobs, failed_filenames = self._extract_cloud_batch_outcomes(batch_output_lines)

                for source_spec in source_specs_to_process:
                    original_basename = source_spec["basename"]
                    current_video_path = source_spec["path"]
                    file_name = os.path.basename(current_video_path)
                    outcome = "pending"
                    if original_basename in completed_jobs:
                        outcome = "finished"
                    elif file_name in failed_filenames:
                        outcome = "failed"
                    elif batch_rc != 0:
                        # Batch aborted before this clip was attempted.
                        outcome = "pending"

                    if outcome == "finished":
                        job_stub = {
                            "is_segment": False,
                            "original_video_raw_frame_count": "N/A",
                        }
                        self._update_gui_info_on_job_start(job_stub, original_basename, "Cloud")
                        status_json_path = self._resolve_cloud_status_json_path(original_basename)
                        self._update_gui_info_from_cloud_status(job_stub, status_json_path)
                        if self.effective_move_original_on_completion and os.path.isfile(current_video_path):
                            self._move_original_source(current_video_path, original_basename, "finished")
                        total_sources_processed += 1
                        self.message_queue.put(("progress", total_sources_processed))
                    elif outcome == "failed":
                        if self.effective_move_original_on_completion and os.path.isfile(current_video_path):
                            self._move_original_source(current_video_path, original_basename, "failed")
                        total_sources_processed += 1
                        self.message_queue.put(("progress", total_sources_processed))

                if batch_rc != 0:
                    _logger.error(f"Cloud batch job failed with exit code {batch_rc}.")
                    had_cloud_errors = True
            else:
                for source_idx, source_spec in enumerate(source_specs_to_process):
                    if self.stop_event.is_set():
                        _logger.info("Cloud dispatch cancelled by user.")
                        break

                    current_video_path = source_spec["path"]
                    original_basename = source_spec["basename"]
                    self.current_filename_var.set(f"{original_basename} (Cloud)")
                    self.current_resolution_var.set("N/A")
                    self.current_frames_var.set("N/A")
                    self.status_message_var.set(
                        f"Cloud processing {source_idx + 1} of {len(source_specs_to_process)}: {original_basename}"
                    )
                    self.root.update_idletasks()

                    job_stub = {
                        "is_segment": False,
                        "original_video_raw_frame_count": "N/A",
                    }
                    self._update_gui_info_on_job_start(job_stub, original_basename, "Cloud")

                    cloud_cmd, job_name = self._build_cloudctl_run_job_command(
                        connection_info=connection_info,
                        source_spec=source_spec,
                        effective_seed_for_run=effective_seed_for_run,
                        source_idx=source_idx,
                        total_sources=len(source_specs_to_process),
                    )

                    rc, _ = self._run_external_command_with_logging(
                        cloud_cmd,
                        cwd=os.path.dirname(os.path.abspath(__file__)),
                    )
                    if rc != 0:
                        _logger.error(f"Cloud job failed for {original_basename} with exit code {rc}.")
                        had_cloud_errors = True
                        if self.effective_move_original_on_completion and os.path.isfile(current_video_path):
                            self._move_original_source(current_video_path, original_basename, "failed")
                    else:
                        status_json_path = self._resolve_cloud_status_json_path(job_name)
                        self._update_gui_info_from_cloud_status(job_stub, status_json_path)
                        if self.effective_move_original_on_completion and os.path.isfile(current_video_path):
                            self._move_original_source(current_video_path, original_basename, "finished")

                    total_sources_processed += 1
                    self.message_queue.put(("progress", total_sources_processed))

            if self.stop_event.is_set():
                self.status_message_var.set("Cloud processing cancelled.")
            else:
                if had_cloud_errors:
                    self.status_message_var.set("Cloud processing finished with errors.")
                else:
                    self.status_message_var.set("Cloud processing finished.")
        finally:
            if bool(self.cloud_auto_destroy_instance_var.get()) and instance_id > 0:
                self.status_message_var.set(f"Destroying cloud instance {instance_id}...")
                self.root.update_idletasks()
                self._destroy_cloud_instance(instance_id)
                self.cloud_last_instance_id_var.set(0)
                self.cloud_last_instance_profile_var.set("")
                self.cloud_last_host_var.set("")
                self.cloud_last_port_var.set(22)
                self.cloud_last_offer_id_var.set(0)
                self.cloud_last_machine_id_var.set(0)
                self.cloud_last_host_id_var.set(0)

    def _apply_spatial_refine_options_visibility(self):
        if not hasattr(self, 'spatial_refine_toggle_btn'):
            return

        is_open = (
            self.spatial_refine_settings_dialog is not None
            and bool(self.spatial_refine_settings_dialog.winfo_exists())
        )
        self.spatial_refine_options_expanded_var.set(bool(is_open))
        button_text = "  ↳ Hi-Res / Edge Settings Open" if is_open else "  ↳ Configure Hi-Res / Edge Settings..."
        try:
            self.spatial_refine_toggle_btn.configure(text=button_text)
        except tk.TclError:
            pass

    def toggle_spatial_refine_options_visibility(self):
        self._open_spatial_refine_settings_dialog()

    def toggle_dither_options_active_state(self, *args):
        if not (hasattr(self, 'process_as_segments_var') and hasattr(self, 'merge_dither_var')): return
        active = self.process_as_segments_var.get() and self.merge_dither_var.get()
        state = tk.NORMAL if active else tk.DISABLED
        for attr_name in ['lbl_dither_str', 'entry_dither_str']:
            if hasattr(self, attr_name):
                widget = getattr(self, attr_name)
                if widget and hasattr(widget, 'configure'):
                    try: widget.configure(state=state)
                    except tk.TclError: pass

    def toggle_gamma_options_active_state(self, *args):
        if not (hasattr(self, 'process_as_segments_var') and hasattr(self, 'merge_gamma_correct_var')): return
        active = self.process_as_segments_var.get() and self.merge_gamma_correct_var.get()
        state = tk.NORMAL if active else tk.DISABLED
        for attr_name in ['lbl_gamma_val', 'entry_gamma_val']:
            if hasattr(self, attr_name):
                widget = getattr(self, attr_name)
                if widget and hasattr(widget, 'configure'):
                    try: widget.configure(state=state)
                    except tk.TclError: pass

    def toggle_keep_npz_dependent_options_state(self, *args):
        if not (hasattr(self, 'process_as_segments_var') and hasattr(self, 'keep_intermediate_npz_var') and hasattr(self, 'keep_npz_dependent_widgets')):
            return
        active = self.process_as_segments_var.get() and self.keep_intermediate_npz_var.get()
        state = tk.NORMAL if active else tk.DISABLED
        for widget in self.keep_npz_dependent_widgets:
            if hasattr(widget, 'configure'):
                try:
                    if isinstance(widget, ttk.Combobox): widget.configure(state='readonly' if active else 'disabled')
                    else: widget.configure(state=state)
                except tk.TclError: pass

    def toggle_merge_related_options_active_state(self, *args):
        if not hasattr(self, 'process_as_segments_var'): return
        active = self.process_as_segments_var.get()
        current_processing_state = tk.DISABLED
        if hasattr(self, 'start_button') and self.start_button and hasattr(self, 'cancel_button') and self.cancel_button:
            try:
                if self.start_button.cget('state') == tk.DISABLED and self.cancel_button.cget('state') == tk.NORMAL:
                    current_processing_state = tk.DISABLED
                else: current_processing_state = tk.NORMAL
            except tk.TclError: pass
        effective_state_for_merge_options = tk.DISABLED
        if current_processing_state == tk.NORMAL and active: effective_state_for_merge_options = tk.NORMAL
        if hasattr(self, 'merge_related_widgets_references'):
            for widget_tuple_or_item in self.merge_related_widgets_references:
                items_to_configure = widget_tuple_or_item if isinstance(widget_tuple_or_item, tuple) else (widget_tuple_or_item,)
                for widget_item in items_to_configure:
                    if hasattr(widget_item, 'configure'):
                        try:
                            if isinstance(widget_item, ttk.Combobox): widget_item.configure(state='readonly' if effective_state_for_merge_options == tk.NORMAL else 'disabled')
                            else: widget_item.configure(state=effective_state_for_merge_options)
                        except tk.TclError: pass
        if not active:
            if current_processing_state == tk.NORMAL:
                for var_attr_name in ['keep_intermediate_npz_var', 'merge_dither_var', 'merge_gamma_correct_var', 'merge_percentile_norm_var']:
                    if hasattr(self, var_attr_name):
                        var_to_set = getattr(self, var_attr_name)
                        if var_to_set: var_to_set.set(False)
        self.toggle_keep_npz_dependent_options_state()
        self.toggle_dither_options_active_state()
        self.toggle_gamma_options_active_state()
        self.toggle_percentile_norm_options_active_state()

    def toggle_percentile_norm_options_active_state(self, *args):
        if not (hasattr(self, 'process_as_segments_var') and hasattr(self, 'merge_percentile_norm_var')): return
        active = self.process_as_segments_var.get() and self.merge_percentile_norm_var.get()
        state = tk.NORMAL if active else tk.DISABLED
        for attr_name in ['lbl_low_perc', 'entry_low_perc', 'lbl_high_perc', 'entry_high_perc']:
            if hasattr(self, attr_name):
                widget = getattr(self, attr_name)
                if widget and hasattr(widget, 'configure'):
                    try: widget.configure(state=state)
                    except tk.TclError: pass

    def toggle_secondary_output_options_active_state(self, *args):
        if not hasattr(self, 'enable_dual_output_robust_norm') or not hasattr(self, 'secondary_output_widgets_references'):
            return

        active = self.enable_dual_output_robust_norm.get()
        state = tk.NORMAL if active else tk.DISABLED

        for widget_item in self.secondary_output_widgets_references:
            if isinstance(widget_item, tuple): # Handle cases where we might store (label, entry_frame)
                for item in widget_item:
                    if hasattr(item, 'configure'):
                        try:
                            if isinstance(item, ttk.Combobox): item.configure(state='readonly' if active else 'disabled')
                            else: item.configure(state=state)
                        except tk.TclError: pass
            elif hasattr(widget_item, 'configure'):
                try:
                    if isinstance(widget_item, ttk.Combobox): widget_item.configure(state='readonly' if active else 'disabled')
                    else: widget_item.configure(state=state)
                except tk.TclError: pass

if __name__ == "__main__":
    # Configure basic logging for console output
    logging.basicConfig(level=logging.DEBUG, # Default to INFO level
                        format='%(asctime)s - %(message)s',
                        datefmt='%H:%M:%S')

    
    if THEMEDTK_AVAILABLE:
        root = ThemedTk(theme="default") # Use ThemedTk for theme support
    else:
        root = tk.Tk()
    app = DepthCrafterGUI(root)
    root.mainloop()
