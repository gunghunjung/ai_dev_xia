"""
ai_runtime/decision_engine.py — 런타임 의사결정 엔진 (재내보내기 + 통계 확장)

ai/decision_engine.py 의 DecisionEngine 을 그대로 재사용하며
실행 시간 통계와 규칙 히트 카운터를 추가한다.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Dict

# ai/ 패키지 구현을 직접 재사용
from ai.decision_engine import (  # noqa: F401  (외부에서 import 가능하도록)
    AIScoreRule,
    DecisionEngine as _BaseDecisionEngine,
    DriftRule,
    LowConfidenceWarningRule,
    OscillationRule,
    StuckRule,
    SuddenChangeRule,
    _derive_roi_state,
)
from config import DecisionConfig
from core.data_models import FrameContext
from core.interfaces import IDecisionRule


class RuntimeDecisionEngine(_BaseDecisionEngine):
    """
    _BaseDecisionEngine 에 런타임 통계를 추가한 확장 클래스.

    추가 기능:
      - 규칙별 발동 횟수 카운터
      - 프레임 처리 지연 시간 통계
      - get_stats() 로 현황 조회
    """

    def __init__(self, cfg: DecisionConfig):
        super().__init__(cfg)
        self._rule_hits: Counter = Counter()
        self._total_decide_ms: float = 0.0
        self._decide_count: int = 0

    def decide(self, ctx: FrameContext) -> None:
        t0 = time.perf_counter()
        super().decide(ctx)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self._decide_count += 1
        self._total_decide_ms += elapsed_ms

        for r in ctx.triggered_rules:
            self._rule_hits[r.rule_name] += 1

    def reset(self) -> None:
        super().reset()
        self._rule_hits.clear()
        self._total_decide_ms = 0.0
        self._decide_count = 0

    def get_stats(self) -> Dict:
        avg_ms = (self._total_decide_ms / self._decide_count
                  if self._decide_count else 0.0)
        return {
            "frames_processed": self._decide_count,
            "avg_decide_ms":    round(avg_ms, 3),
            "rule_hits":        dict(self._rule_hits),
            "current_state":    self._current_state.value,
            "consecutive_abn":  self._consecutive_abnormal,
            "consecutive_nor":  self._consecutive_normal,
        }
