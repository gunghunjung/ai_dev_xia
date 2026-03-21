"""
enhancement/enhancement_model_manager.py — Model registry and loader for enhancement models.

Manages EnhancementModelMeta records, optional ONNX/Torch loading,
and tracks the single active model per EnhancementType.
Built-in OpenCV fallback models are registered on init.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from enhancement.enhancement_config import EnhancementConfig, EnhancementType

logger = logging.getLogger(__name__)


@dataclass
class EnhancementModelMeta:
    """Metadata record for a single enhancement model."""

    model_name: str
    version: str = "1.0"
    framework: str = "opencv"           # "opencv", "onnx", "torch"
    task_type: EnhancementType = EnhancementType.SHARPENING
    input_size: Optional[Tuple[int, int]] = None   # (w, h) or None = any size
    scale_factor: int = 1
    latency_ms_estimate: float = 0.0
    quality_note: str = ""
    file_path: Optional[str] = None
    is_active: bool = False
    is_deployed: bool = False


class EnhancementModelManager:
    """
    Registry for enhancement models.

    Lifecycle:
      1. register()  — add an EnhancementModelMeta record
      2. load()      — for onnx/torch models, load the file into _sessions
      3. set_active() — nominate one model per task type
      4. get_active() — retrieve the active model meta for a task type
    """

    def __init__(self, cfg: EnhancementConfig) -> None:
        self._cfg = cfg
        self._registry: Dict[str, EnhancementModelMeta] = {}
        self._active: Dict[EnhancementType, str] = {}   # task_type -> model_name
        self._sessions: Dict[str, object] = {}           # model_name -> loaded object

        self._register_builtin_fallbacks()

    # ------------------------------------------------------------------
    # Built-in OpenCV fallbacks
    # ------------------------------------------------------------------

    def _register_builtin_fallbacks(self) -> None:
        """Register lightweight OpenCV-based fallback models."""
        builtins = [
            EnhancementModelMeta(
                model_name="cv2_sharpen",
                version="1.0",
                framework="opencv",
                task_type=EnhancementType.SHARPENING,
                quality_note="OpenCV unsharp mask fallback",
                is_deployed=True,
            ),
            EnhancementModelMeta(
                model_name="cv2_denoise",
                version="1.0",
                framework="opencv",
                task_type=EnhancementType.DENOISING,
                quality_note="cv2.fastNlMeansDenoisingColored fallback",
                is_deployed=True,
            ),
            EnhancementModelMeta(
                model_name="cv2_deblur",
                version="1.0",
                framework="opencv",
                task_type=EnhancementType.DEBLURRING,
                quality_note="OpenCV simple deblur approximation fallback",
                is_deployed=True,
            ),
        ]
        for meta in builtins:
            self.register(meta)
            # Auto-activate builtins as default for each task
            self._active[meta.task_type] = meta.model_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, meta: EnhancementModelMeta) -> None:
        """Add or overwrite a model record in the registry."""
        self._registry[meta.model_name] = meta
        logger.debug("Registered model '%s' (%s/%s)", meta.model_name, meta.framework, meta.task_type)

    def load(self, model_name: str) -> bool:
        """
        Load the model file for onnx or torch frameworks.
        Returns True on success, False on failure or if not needed.
        """
        meta = self._registry.get(model_name)
        if meta is None:
            logger.warning("load(): model '%s' not found in registry", model_name)
            return False

        if meta.framework == "opencv":
            meta.is_deployed = True
            return True

        if meta.file_path is None:
            logger.warning("load(): model '%s' has no file_path set", model_name)
            return False

        if meta.framework == "onnx":
            session = self._load_onnx(meta)
            if session is not None:
                self._sessions[model_name] = session
                meta.is_deployed = True
                logger.info("Loaded ONNX model '%s'", model_name)
                return True
            return False

        if meta.framework == "torch":
            model_obj = self._load_torch(meta)
            if model_obj is not None:
                self._sessions[model_name] = model_obj
                meta.is_deployed = True
                logger.info("Loaded Torch model '%s'", model_name)
                return True
            return False

        logger.warning("load(): unknown framework '%s' for model '%s'", meta.framework, model_name)
        return False

    def get(self, model_name: str) -> Optional[EnhancementModelMeta]:
        """Return the meta record for a given model name, or None."""
        return self._registry.get(model_name)

    def list_by_task(self, task_type: EnhancementType) -> List[EnhancementModelMeta]:
        """Return all registered models that match the given task type."""
        return [m for m in self._registry.values() if m.task_type == task_type]

    def set_active(self, task_type: EnhancementType, model_name: str) -> None:
        """Designate a model as the active one for a task type."""
        if model_name not in self._registry:
            raise KeyError(f"Model '{model_name}' is not registered")
        meta = self._registry[model_name]
        if meta.task_type != task_type:
            raise ValueError(
                f"Model '{model_name}' is for task '{meta.task_type}', not '{task_type}'"
            )
        # Deactivate previous active model for this task
        prev = self._active.get(task_type)
        if prev and prev in self._registry:
            self._registry[prev].is_active = False

        self._active[task_type] = model_name
        meta.is_active = True
        logger.debug("Active model for %s set to '%s'", task_type, model_name)

    def get_active(self, task_type: EnhancementType) -> Optional[EnhancementModelMeta]:
        """Return the active model meta for a task type, or None."""
        name = self._active.get(task_type)
        if name is None:
            return None
        return self._registry.get(name)

    def get_session(self, model_name: str) -> Optional[object]:
        """Return the loaded model object (ONNX session or Torch module), or None."""
        return self._sessions.get(model_name)

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_onnx(self, meta: EnhancementModelMeta) -> object:
        """Load an ONNX model file and return an InferenceSession, or None."""
        try:
            import onnxruntime as ort  # type: ignore
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self._cfg.use_gpu \
                        else ["CPUExecutionProvider"]
            session = ort.InferenceSession(meta.file_path, providers=providers)
            return session
        except ImportError:
            logger.warning("onnxruntime not installed; cannot load '%s'", meta.model_name)
            return None
        except Exception as exc:
            logger.error("Failed to load ONNX model '%s': %s", meta.model_name, exc)
            return None

    def _load_torch(self, meta: EnhancementModelMeta) -> object:
        """Load a Torch model file and return a nn.Module, or None."""
        try:
            import torch  # type: ignore
            device = "cuda" if (self._cfg.use_gpu and torch.cuda.is_available()) else "cpu"
            model = torch.load(meta.file_path, map_location=device)
            model.eval()
            return model
        except ImportError:
            logger.warning("torch not installed; cannot load '%s'", meta.model_name)
            return None
        except Exception as exc:
            logger.error("Failed to load Torch model '%s': %s", meta.model_name, exc)
            return None
