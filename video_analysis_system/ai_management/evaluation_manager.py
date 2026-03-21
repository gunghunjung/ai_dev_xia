"""
ai_management/evaluation_manager.py — 모델 평가 및 보고서 생성 모듈.

EvaluationManager 는 모델 성능을 평가하고 혼동 행렬, 클래스별 지표,
실패 사례를 분석한다. 평가 결과는 JSON 보고서로 저장할 수 있다.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EvalResult — 평가 결과 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """단일 모델 평가 결과를 담는 데이터 클래스."""

    eval_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """평가 고유 식별자."""

    model_name: str = ""
    """평가된 모델 이름."""

    dataset_name: str = ""
    """평가에 사용된 데이터셋 이름."""

    evaluated_at: float = field(default_factory=time.time)
    """평가 시각 (Unix timestamp)."""

    # 전체 지표
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc: float = 0.0

    # 상세 결과
    confusion_matrix: Optional[np.ndarray] = None
    """혼동 행렬 (shape: [num_classes, num_classes])."""

    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    """클래스별 precision / recall / f1 / support."""

    failure_cases: List[Dict[str, Any]] = field(default_factory=list)
    """오분류된 샘플 목록."""

    def to_dict(self) -> dict:
        """JSON 직렬화용 딕셔너리 반환."""
        d = asdict(self)
        # numpy 배열 → 리스트 변환
        if self.confusion_matrix is not None:
            d["confusion_matrix"] = self.confusion_matrix.tolist()
        return d


# ---------------------------------------------------------------------------
# EvaluationManager
# ---------------------------------------------------------------------------

class EvaluationManager:
    """
    모델 성능을 평가하고 분석 보고서를 생성하는 클래스.

    실제 추론 엔진이 없을 경우에는 랜덤 메트릭을 반환하는
    플레이스홀더 모드로 동작한다.
    """

    def __init__(self) -> None:
        """EvaluationManager 초기화."""
        logger.debug("EvaluationManager 초기화 완료.")

    # ------------------------------------------------------------------
    # 모델 평가
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        model_path: str,
        dataset_path: str,
        task_type: str = "classification",
    ) -> dict:
        """
        모델을 데이터셋에 대해 평가하고 성능 지표 딕셔너리를 반환한다.

        현재 구현은 Placeholder 로 랜덤 메트릭을 생성한다.
        실제 구현 시 이 메서드를 교체하면 된다.

        Parameters
        ----------
        model_path:
            평가할 모델 파일 경로.
        dataset_path:
            평가 데이터셋 디렉터리 경로.
        task_type:
            태스크 종류 (classification / detection / segmentation).

        Returns
        -------
        dict
            accuracy, precision, recall, f1, auc 등 성능 지표.
        """
        logger.info(
            "모델 평가 시작 (placeholder): model=%s, dataset=%s, task=%s",
            os.path.basename(model_path),
            os.path.basename(dataset_path),
            task_type,
        )
        # Placeholder: 랜덤 메트릭 생성
        acc = random.uniform(0.75, 0.98)
        prec = random.uniform(0.70, 0.97)
        rec = random.uniform(0.70, 0.97)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        auc = random.uniform(0.80, 0.99)

        metrics = {
            "model_path": model_path,
            "dataset_path": dataset_path,
            "task_type": task_type,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "auc": round(auc, 4),
            "evaluated_at": time.time(),
        }
        logger.info("평가 완료: accuracy=%.4f, f1=%.4f", acc, f1)
        return metrics

    # ------------------------------------------------------------------
    # 혼동 행렬
    # ------------------------------------------------------------------

    def generate_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str],
    ) -> np.ndarray:
        """
        실제 레이블과 예측 레이블로부터 혼동 행렬을 계산한다.

        Parameters
        ----------
        y_true:
            정답 레이블 인덱스 목록.
        y_pred:
            예측 레이블 인덱스 목록.
        class_names:
            클래스 이름 목록. 인덱스 순서와 일치해야 한다.

        Returns
        -------
        np.ndarray
            shape (num_classes, num_classes) 의 혼동 행렬.
            cm[i][j] = 실제 클래스 i 를 클래스 j 로 예측한 횟수.
        """
        n = len(class_names)
        cm = np.zeros((n, n), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < n and 0 <= p < n:
                cm[t][p] += 1
        logger.debug("혼동 행렬 생성 완료: shape=%s", cm.shape)
        return cm

    # ------------------------------------------------------------------
    # 클래스별 지표
    # ------------------------------------------------------------------

    def compute_per_class_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        클래스별 precision, recall, f1, support 를 계산한다.

        Parameters
        ----------
        y_true:
            정답 레이블 인덱스 목록.
        y_pred:
            예측 레이블 인덱스 목록.
        class_names:
            클래스 이름 목록.

        Returns
        -------
        Dict[str, Dict[str, float]]
            ``{class_name: {precision, recall, f1, support}}`` 형태.
        """
        n = len(class_names)
        result: Dict[str, Dict[str, float]] = {}

        for cls_idx, cls_name in enumerate(class_names):
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls_idx and p == cls_idx)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls_idx and p == cls_idx)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls_idx and p != cls_idx)
            support = sum(1 for t in y_true if t == cls_idx)

            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)

            result[cls_name] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": float(support),
            }

        logger.debug("클래스별 지표 계산 완료: classes=%d", n)
        return result

    # ------------------------------------------------------------------
    # 실패 사례 탐색
    # ------------------------------------------------------------------

    def find_failure_cases(
        self,
        y_true: List[int],
        y_pred: List[int],
        sample_paths: List[str],
    ) -> List[Dict[str, Any]]:
        """
        오분류된 샘플 목록을 반환한다.

        Parameters
        ----------
        y_true:
            정답 레이블 인덱스 목록.
        y_pred:
            예측 레이블 인덱스 목록.
        sample_paths:
            각 샘플의 파일 경로 목록. y_true 와 같은 순서여야 한다.

        Returns
        -------
        List[dict]
            오분류 샘플별 ``{index, path, true_label, pred_label}`` 딕셔너리 목록.
        """
        failures: List[Dict[str, Any]] = []
        for i, (t, p) in enumerate(zip(y_true, y_pred)):
            if t != p:
                path = sample_paths[i] if i < len(sample_paths) else ""
                failures.append({
                    "index": i,
                    "path": path,
                    "true_label": t,
                    "pred_label": p,
                })

        logger.info(
            "실패 사례 탐색 완료: total=%d, failures=%d",
            len(y_true),
            len(failures),
        )
        return failures

    # ------------------------------------------------------------------
    # 보고서 생성
    # ------------------------------------------------------------------

    def generate_report(
        self,
        eval_result: dict,
        output_path: str,
    ) -> str:
        """
        평가 결과를 JSON 파일로 저장하고 저장 경로를 반환한다.

        Parameters
        ----------
        eval_result:
            ``evaluate_model`` 또는 ``EvalResult.to_dict()`` 의 결과.
        output_path:
            저장할 JSON 파일 경로.

        Returns
        -------
        str
            저장된 파일의 절대 경로.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # numpy 타입 → Python 기본 타입 변환
        def _convert(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        # 얕은 변환
        serializable = {}
        for k, v in eval_result.items():
            serializable[k] = _convert(v)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=2, ensure_ascii=False, default=str)

        abs_path = os.path.abspath(output_path)
        logger.info("평가 보고서 저장: path=%s", abs_path)
        return abs_path
