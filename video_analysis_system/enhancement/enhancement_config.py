"""
enhancement/enhancement_config.py — Configuration dataclasses for the Enhancement Engine.

Design notes
────────────
All Enhancement Engine behaviour is driven by EnhancementConfig.
No magic globals; always pass the config explicitly.

Enhancement scope options
─────────────────────────
FULL_FRAME          — Enhance the entire frame (applied after FrameSource)
ROI_ONLY            — Enhance only the pixels inside each active ROI crop
DISPLAY_ONLY        — Enhanced frame used for display; analysis keeps raw/processed
ANALYSIS_ONLY       — Enhanced frame used for feature extraction and AI inference
                      only; raw frame shown to the user
DUAL               — Both display and analysis use the enhanced frame

Pipeline insertion options
──────────────────────────
PRE_PREPROCESS     — Before Preprocessor (Option A in spec)
POST_PREPROCESS    — After Preprocessor but before ROI extraction (Option B)
ROI_AFTER_EXTRACT  — After ROIManager.extract(), per-ROI (Option B variant)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class EnhancementType(str, Enum):
    """Individual enhancement operations."""
    SUPER_RESOLUTION = "super_resolution"
    SHARPENING       = "sharpening"
    DENOISING        = "denoising"
    DEBLURRING       = "deblurring"
    COMBINED         = "combined"       # chain as configured


class EnhancementMode(str, Enum):
    """Which image the enhancement result replaces or supplements."""
    DISPLAY_ONLY   = "display_only"     # show enhanced; analyse raw
    ANALYSIS_ONLY  = "analysis_only"    # analyse enhanced; show raw
    DUAL           = "dual"             # both display AND analysis enhanced
    PREVIEW        = "preview"          # single-frame on-demand preview only


class EnhancementScope(str, Enum):
    """Spatial extent of enhancement."""
    FULL_FRAME     = "full_frame"
    ROI_ONLY       = "roi_only"
    SELECTED_ROI   = "selected_roi"     # only the currently selected ROI


class PipelinePosition(str, Enum):
    """Where enhancement is injected into the analysis pipeline."""
    PRE_PREPROCESS    = "pre_preprocess"
    POST_PREPROCESS   = "post_preprocess"
    ROI_AFTER_EXTRACT = "roi_after_extract"


@dataclass
class EnhancementConfig:
    """
    Master configuration for the Image Enhancement Engine.

    Defaults to a lightweight sharpening-only mode that is safe to run
    on CPU without any optional dependencies.
    """

    # ── Global enable ─────────────────────────────────────────────────────
    enabled: bool = False                   # master on/off switch

    # ── What to apply ────────────────────────────────────────────────────
    enable_super_resolution: bool  = False
    enable_sharpening:        bool = True
    enable_denoising:         bool = False
    enable_deblurring:        bool = False

    # Enhancement pipeline order (applied left → right)
    pipeline_order: List[EnhancementType] = field(default_factory=lambda: [
        EnhancementType.DENOISING,
        EnhancementType.DEBLURRING,
        EnhancementType.SUPER_RESOLUTION,
        EnhancementType.SHARPENING,
    ])

    # ── Where / how ──────────────────────────────────────────────────────
    scope:             EnhancementScope    = EnhancementScope.FULL_FRAME
    mode:              EnhancementMode     = EnhancementMode.DISPLAY_ONLY
    pipeline_position: PipelinePosition    = PipelinePosition.POST_PREPROCESS

    # Super resolution parameters
    scale_factor: int = 2                  # 2× or 4×
    target_size:  Optional[tuple] = None   # (w, h) override; None = use scale_factor

    # ── Hardware ──────────────────────────────────────────────────────────
    use_gpu: bool = False                  # True → CUDA / MPS if available

    # ── Runtime behaviour ─────────────────────────────────────────────────
    batch_mode:      bool = False          # process multiple frames in one call
    cache_results:   bool = True           # cache enhanced crops per roi_id
    preview_enabled: bool = True           # allow preview mode
    max_cache_size:  int  = 64             # LRU cache size (frames)

    # ── Active model names (look up in EnhancementModelManager) ──────────
    sr_model_name:        Optional[str] = None   # super resolution model
    sharpen_model_name:   Optional[str] = None   # sharpening model (None = CV2 fallback)
    denoise_model_name:   Optional[str] = None   # denoising model  (None = CV2 fallback)
    deblur_model_name:    Optional[str] = None   # deblurring model (None = CV2 fallback)

    def active_types(self) -> List[EnhancementType]:
        """Return only the enhancement types that are currently enabled."""
        types = []
        order_lookup = {t: i for i, t in enumerate(self.pipeline_order)}
        candidates = []
        if self.enable_denoising:        candidates.append(EnhancementType.DENOISING)
        if self.enable_deblurring:       candidates.append(EnhancementType.DEBLURRING)
        if self.enable_super_resolution: candidates.append(EnhancementType.SUPER_RESOLUTION)
        if self.enable_sharpening:       candidates.append(EnhancementType.SHARPENING)
        # Respect pipeline order
        candidates.sort(key=lambda t: order_lookup.get(t, 99))
        return candidates
