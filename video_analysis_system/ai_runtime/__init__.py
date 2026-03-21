"""
ai_runtime — 실시간 추론 런타임 패키지

ModelMetadata 통합, 핫-스왑, 통계 수집 기능이 추가된
고수준 AI 런타임 레이어.

주요 클래스:
  RuntimeModelManager   — 모델 레지스트리 + 핫-스왑
  RuntimeInferenceEngine — 추론 지연 시간 통계 포함
  RuntimeDecisionEngine  — 규칙 발동 카운터 포함
"""

from ai_runtime.model_manager import RuntimeModelManager
from ai_runtime.inference_engine import RuntimeInferenceEngine
from ai_runtime.decision_engine import RuntimeDecisionEngine

__all__ = [
    "RuntimeModelManager",
    "RuntimeInferenceEngine",
    "RuntimeDecisionEngine",
]
