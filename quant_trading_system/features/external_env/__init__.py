# features/external_env/__init__.py — 외부환경 분석 모듈 공개 API
"""
외부환경 분석 파이프라인:
  뉴스 → StructuredEvent → NLP → 피처 벡터 → 가중치 자기교정

사용 예시:
    from features.external_env import (
        StructuredEvent, EventCategory, ImpactDirection, EventDuration,
        NewsEventCategorizer, NLPAnalyzer,
        EventAccumulator, ExternalEnvFeatureEngineer,
        CategoryWeightTuner, VerificationRecord,
    )
"""
from .event_structure import (
    EventCategory, CATEGORY_WEIGHTS,
    ImpactDirection, EventDuration,
    StructuredEvent,
)
from .categorizer import NewsEventCategorizer
from .nlp_analyzer import NLPAnalyzer, get_analyzer
from .accumulator import EventAccumulator, ACCUMULATOR_FEATURE_DIM
from .feature_engineer import (
    ExternalEnvFeatureEngineer,
    EXTERNAL_FEATURE_DIM, EXTENDED_FEATURE_DIM,
)
__all__ = [
    "EventCategory", "CATEGORY_WEIGHTS",
    "ImpactDirection", "EventDuration",
    "StructuredEvent",
    "NewsEventCategorizer",
    "NLPAnalyzer", "get_analyzer",
    "EventAccumulator", "ACCUMULATOR_FEATURE_DIM",
    "ExternalEnvFeatureEngineer",
    "EXTERNAL_FEATURE_DIM", "EXTENDED_FEATURE_DIM",
]
