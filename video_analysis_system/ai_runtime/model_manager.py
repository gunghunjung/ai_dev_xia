"""
ai_runtime/model_manager.py — 런타임 모델 관리 (ModelMetadata 통합)

ai/model_manager.py 의 기본 구현을 래핑하여:
  - ModelMetadata 기반 모델 레지스트리
  - 헬스체크 / 상태 조회
  - 핫-스왑(실행 중 모델 교체) 지원

ai/ 패키지의 저수준 클래스(PlaceholderModel, ONNXModel, PyTorchModel)를
그대로 재사용한다.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from ai.model_manager import ModelManager as _LowLevelManager
from ai.model_manager import ONNXModel, PlaceholderModel, PyTorchModel
from config import AIConfig
from core.data_models import ModelMetadata
from core.interfaces import IModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RuntimeModelManager
# ---------------------------------------------------------------------------

class RuntimeModelManager:
    """
    실시간 추론에 사용하는 모델을 관리하는 고수준 레지스트리.

    ModelMetadata 와 실제 IModel 인스턴스를 함께 추적하고,
    현재 운영(production) 모델을 빠르게 조회하거나 교체할 수 있다.

    기존 코드 호환성을 위해 내부적으로 _LowLevelManager 를 보유하며
    .default 프로퍼티를 그대로 노출한다.
    """

    def __init__(self, cfg: AIConfig):
        self._cfg = cfg
        self._inner = _LowLevelManager(cfg)
        # name → (IModel, ModelMetadata)
        self._registry: Dict[str, tuple[IModel, ModelMetadata]] = {}
        self._active_name: str = "default"
        self._last_load_time: float = 0.0

    # ── 초기화 ───────────────────────────────────────────────────────────

    def load_default(self) -> None:
        """설정 파일 기준으로 기본 모델을 로드한다."""
        self._inner.load_default()
        model = self._inner.default
        meta = ModelMetadata(
            model_name=self._cfg.model_type,
            version="1.0.0",
            framework=self._cfg.model_type,
            file_path=self._cfg.model_path or "",
            status="production",
            class_map={i: n for i, n in enumerate(self._cfg.class_names)},
        )
        self._registry["default"] = (model, meta)
        self._active_name = "default"
        self._last_load_time = time.time()
        logger.info("RuntimeModelManager: default model ready (%s)", self._cfg.model_type)

    # ── 레지스트리 ─────────────────────────────────────────────────────────

    def register(self, name: str, model: IModel,
                 metadata: Optional[ModelMetadata] = None) -> None:
        """외부 모델을 이름으로 등록한다."""
        meta = metadata or ModelMetadata(model_name=name)
        self._registry[name] = (model, meta)
        self._inner.register(name, model)
        logger.info("RuntimeModelManager: registered '%s'", name)

    def get_model(self, name: str = "default") -> IModel:
        if name in self._registry:
            return self._registry[name][0]
        return self._inner.get(name)

    def get_metadata(self, name: str = "default") -> Optional[ModelMetadata]:
        entry = self._registry.get(name)
        return entry[1] if entry else None

    def list_models(self) -> List[str]:
        return list(self._registry.keys())

    # ── 활성 모델 ──────────────────────────────────────────────────────────

    def set_active(self, name: str) -> None:
        """현재 추론에 사용할 모델을 교체한다 (핫-스왑)."""
        if name not in self._registry:
            raise KeyError(f"모델 '{name}' 이 등록되지 않았습니다.")
        self._active_name = name
        logger.info("RuntimeModelManager: active model → '%s'", name)

    @property
    def active_name(self) -> str:
        return self._active_name

    @property
    def default(self) -> IModel:
        """기존 코드 호환: active 모델의 IModel 을 반환한다."""
        return self.get_model(self._active_name)

    # ── 헬스체크 ──────────────────────────────────────────────────────────

    def health_check(self) -> dict:
        """모든 등록된 모델의 로드 상태를 반환한다."""
        status = {}
        for name, (model, meta) in self._registry.items():
            status[name] = {
                "loaded":     model.is_loaded,
                "framework":  meta.framework,
                "status":     meta.status,
                "is_active":  name == self._active_name,
            }
        return status

    @property
    def last_load_time(self) -> float:
        return self._last_load_time
