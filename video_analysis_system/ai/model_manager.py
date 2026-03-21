"""
ai/model_manager.py — Model loading, registry, and backend selection.

Supports:
  - PlaceholderModel (always available — for development/testing)
  - ONNXModel        (requires onnxruntime)
  - PyTorchModel     (requires torch)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from config import AIConfig
from core.interfaces import IModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PlaceholderModel — deterministic fake model for dev / CI
# ---------------------------------------------------------------------------

class PlaceholderModel(IModel):
    """
    Returns random-but-reproducible predictions.
    Used when no real model file is provided.
    """

    def __init__(self, num_classes: int = 2, seed: int = 42):
        self._num_classes = num_classes
        self._rng = np.random.default_rng(seed)
        self._loaded = False

    def load(self, path: str) -> None:
        logger.info("PlaceholderModel: ignoring path '%s', using random outputs", path)
        self._loaded = True

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Return softmax-like float32 array of shape (num_classes,)."""
        raw = self._rng.random(self._num_classes).astype(np.float32)
        return raw / raw.sum()

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------------------
# ONNXModel — wraps onnxruntime.InferenceSession
# ---------------------------------------------------------------------------

class ONNXModel(IModel):
    """Inference via ONNX Runtime."""

    def __init__(self, input_size: tuple[int, int] = (224, 224)):
        self._input_size = input_size
        self._session = None
        self._input_name: str = ""

    def load(self, path: str) -> None:
        try:
            import onnxruntime as ort  # type: ignore
            self._session = ort.InferenceSession(
                path, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            logger.info("ONNX model loaded from %s", path)
        except ImportError:
            raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime")
        except Exception as exc:
            raise RuntimeError(f"Failed to load ONNX model: {exc}") from exc

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("Model not loaded")
        import cv2
        if input_data.ndim == 3:
            img = cv2.resize(input_data, self._input_size)
            blob = img.astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1)[np.newaxis]  # NCHW
        else:
            blob = input_data
        outputs = self._session.run(None, {self._input_name: blob})
        return np.array(outputs[0]).flatten()

    @property
    def is_loaded(self) -> bool:
        return self._session is not None


# ---------------------------------------------------------------------------
# PyTorchModel — wraps a torch.nn.Module
# ---------------------------------------------------------------------------

class PyTorchModel(IModel):
    """Inference via PyTorch. Expects a .pt or .pth file saved with torch.save."""

    def __init__(self, input_size: tuple[int, int] = (224, 224), device: str = "cpu"):
        self._input_size = input_size
        self._device_name = device
        self._model = None
        self._device = None

    def load(self, path: str) -> None:
        try:
            import torch  # type: ignore
            self._device = torch.device(self._device_name)
            self._model = torch.load(path, map_location=self._device)
            self._model.eval()
            logger.info("PyTorch model loaded from %s on %s", path, self._device_name)
        except ImportError:
            raise RuntimeError("torch not installed. Run: pip install torch")
        except Exception as exc:
            raise RuntimeError(f"Failed to load PyTorch model: {exc}") from exc

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        import torch, cv2  # type: ignore
        img = cv2.resize(input_data, self._input_size)
        blob = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(blob.transpose(2, 0, 1)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            out = self._model(tensor)
        return out.cpu().numpy().flatten()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------

class ModelManager:
    """
    Registry that loads and exposes one or more AI models.
    Selects the correct backend from AIConfig.
    """

    def __init__(self, cfg: AIConfig):
        self._cfg = cfg
        self._models: Dict[str, IModel] = {}
        self._default_key = "default"

    def load_default(self) -> None:
        """Load the model configured in AIConfig as the 'default' model."""
        model = self._build_model(self._cfg.model_type)
        model.load(self._cfg.model_path or "")
        self._models[self._default_key] = model
        logger.info("Default model ready (type=%s)", self._cfg.model_type)

    def register(self, name: str, model: IModel) -> None:
        self._models[name] = model

    def get(self, name: str = "default") -> IModel:
        if name not in self._models:
            raise KeyError(f"Model '{name}' not registered")
        return self._models[name]

    @property
    def default(self) -> IModel:
        return self.get(self._default_key)

    def list_models(self) -> List[str]:
        return list(self._models.keys())

    def _build_model(self, model_type: str) -> IModel:
        if model_type == "onnx":
            return ONNXModel(self._cfg.input_size)
        elif model_type == "pytorch":
            return PyTorchModel(self._cfg.input_size, self._cfg.device)
        elif model_type == "placeholder":
            num_classes = len(self._cfg.class_names)
            return PlaceholderModel(num_classes=num_classes)
        else:
            logger.warning("Unknown model_type '%s', falling back to placeholder", model_type)
            return PlaceholderModel()
