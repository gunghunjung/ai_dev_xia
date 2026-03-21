"""
ai_management/prediction_analysis_manager.py — 예측 결과 분석 모듈.

PredictionAnalysisManager 는 배치 추론 실행, 레이블과의 비교,
어려운 사례(hard example) 탐색, FP/FN 분류, 신뢰도 분포 분석,
오류 보고서 생성을 담당한다.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 이미지 파일 확장자
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class PredictionAnalysisManager:
    """
    모델 예측 결과를 분석하고 오류 패턴을 추출하는 클래스.

    - 배치 추론 (placeholder)
    - 레이블과 예측 비교
    - Hard example, FP, FN 탐색
    - 신뢰도 분포 통계
    - JSON 오류 보고서 생성
    """

    def __init__(self) -> None:
        """PredictionAnalysisManager 초기화."""
        logger.debug("PredictionAnalysisManager 초기화 완료.")

    # ------------------------------------------------------------------
    # 배치 추론
    # ------------------------------------------------------------------

    def run_batch_inference(
        self,
        model_path: str,
        sample_dir: str,
        output_dir: str,
    ) -> dict:
        """
        sample_dir 의 이미지에 대해 배치 추론을 실행하고 결과를 저장한다.

        현재 구현은 Placeholder 로 랜덤 예측을 생성한다.
        실제 구현 시 이 메서드를 onnxruntime / torch 추론으로 교체한다.

        Parameters
        ----------
        model_path:
            모델 파일 경로 (현재는 사용하지 않음).
        sample_dir:
            추론할 이미지 디렉터리.
        output_dir:
            결과 JSON 을 저장할 디렉터리.

        Returns
        -------
        dict
            ``total``, ``results`` (List[dict]), ``output_path`` 를 포함하는 딕셔너리.
        """
        # 이미지 파일 수집
        image_paths = self._collect_images(sample_dir)
        if not image_paths:
            logger.warning("이미지 파일을 찾을 수 없음: dir=%s", sample_dir)
            return {"total": 0, "results": [], "output_path": ""}

        results: List[dict] = []
        for img_path in image_paths:
            pred_class = random.randint(0, 1)
            confidence = round(random.uniform(0.5, 1.0), 4)
            results.append({
                "path": img_path,
                "pred_class": pred_class,
                "confidence": confidence,
                "scores": [round(1 - confidence, 4), confidence] if pred_class == 1
                          else [confidence, round(1 - confidence, 4)],
            })

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "batch_inference_results.json")
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)

        logger.info(
            "배치 추론 완료 (placeholder): total=%d, output=%s",
            len(results),
            output_path,
        )
        return {
            "total": len(results),
            "results": results,
            "output_path": os.path.abspath(output_path),
        }

    # ------------------------------------------------------------------
    # 레이블 비교
    # ------------------------------------------------------------------

    def compare_with_labels(
        self,
        predictions: List[dict],
        labels: List[dict],
    ) -> dict:
        """
        예측 결과와 정답 레이블을 비교하여 전체 정확도와 불일치 목록을 반환한다.

        Parameters
        ----------
        predictions:
            ``run_batch_inference`` 반환값의 ``results`` 리스트.
            각 항목에 ``path``, ``pred_class`` 가 있어야 한다.
        labels:
            각 항목에 ``path``, ``true_class`` 가 있는 딕셔너리 목록.

        Returns
        -------
        dict
            ``total``, ``correct``, ``accuracy``, ``mismatches`` 를 포함하는 딕셔너리.
        """
        # path → true_class 매핑
        label_map: Dict[str, int] = {
            item["path"]: item["true_class"] for item in labels
        }

        total = 0
        correct = 0
        mismatches: List[dict] = []

        for pred in predictions:
            path = pred.get("path", "")
            if path not in label_map:
                continue
            true_cls = label_map[path]
            pred_cls = pred.get("pred_class", -1)
            total += 1
            if pred_cls == true_cls:
                correct += 1
            else:
                mismatches.append({
                    "path": path,
                    "true_class": true_cls,
                    "pred_class": pred_cls,
                    "confidence": pred.get("confidence", 0.0),
                })

        accuracy = correct / total if total > 0 else 0.0
        logger.info(
            "레이블 비교: total=%d, correct=%d, accuracy=%.4f",
            total, correct, accuracy,
        )
        return {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "mismatches": mismatches,
        }

    # ------------------------------------------------------------------
    # Hard example 탐색
    # ------------------------------------------------------------------

    def find_hard_examples(
        self,
        results: List[dict],
        threshold: float = 0.7,
    ) -> List[dict]:
        """
        신뢰도가 threshold 미만인 예측을 hard example 로 반환한다.

        Parameters
        ----------
        results:
            배치 추론 결과 목록.
        threshold:
            신뢰도 임계값. 이 값 미만인 샘플이 선택된다.

        Returns
        -------
        List[dict]
            신뢰도 오름차순으로 정렬된 hard example 목록.
        """
        hard = [r for r in results if r.get("confidence", 1.0) < threshold]
        hard.sort(key=lambda r: r.get("confidence", 1.0))
        logger.info(
            "Hard example 탐색: total=%d, hard=%d, threshold=%.2f",
            len(results), len(hard), threshold,
        )
        return hard

    # ------------------------------------------------------------------
    # FP / FN 탐색
    # ------------------------------------------------------------------

    def find_false_positives(self, results: List[dict]) -> List[dict]:
        """
        False Positive (실제 0, 예측 1) 샘플을 반환한다.

        Parameters
        ----------
        results:
            ``path``, ``true_class``, ``pred_class`` 가 포함된 결과 목록.

        Returns
        -------
        List[dict]
            FP 샘플 목록.
        """
        fp = [
            r for r in results
            if r.get("true_class") == 0 and r.get("pred_class") == 1
        ]
        logger.info("FP 탐색: total=%d, fp=%d", len(results), len(fp))
        return fp

    def find_false_negatives(self, results: List[dict]) -> List[dict]:
        """
        False Negative (실제 1, 예측 0) 샘플을 반환한다.

        Parameters
        ----------
        results:
            ``path``, ``true_class``, ``pred_class`` 가 포함된 결과 목록.

        Returns
        -------
        List[dict]
            FN 샘플 목록.
        """
        fn = [
            r for r in results
            if r.get("true_class") == 1 and r.get("pred_class") == 0
        ]
        logger.info("FN 탐색: total=%d, fn=%d", len(results), len(fn))
        return fn

    # ------------------------------------------------------------------
    # 신뢰도 분포
    # ------------------------------------------------------------------

    def get_confidence_distribution(self, results: List[dict]) -> dict:
        """
        예측 신뢰도의 통계 분포를 반환한다.

        Parameters
        ----------
        results:
            각 항목에 ``confidence`` 가 있는 결과 목록.

        Returns
        -------
        dict
            ``count``, ``mean``, ``std``, ``min``, ``max``,
            ``histogram`` (10개 구간) 를 포함하는 딕셔너리.
        """
        confidences = [r.get("confidence", 0.0) for r in results]
        if not confidences:
            return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "histogram": []}

        n = len(confidences)
        mean = sum(confidences) / n
        variance = sum((c - mean) ** 2 for c in confidences) / n
        std = variance ** 0.5
        c_min = min(confidences)
        c_max = max(confidences)

        # 10개 구간 히스토그램 (0.0 ~ 1.0)
        bins = 10
        histogram = [0] * bins
        for c in confidences:
            idx = min(int(c * bins), bins - 1)
            histogram[idx] += 1

        return {
            "count": n,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(c_min, 4),
            "max": round(c_max, 4),
            "histogram": histogram,
        }

    # ------------------------------------------------------------------
    # 오류 보고서 생성
    # ------------------------------------------------------------------

    def generate_error_report(
        self,
        results: List[dict],
        output_path: str,
    ) -> str:
        """
        예측 오류 분석 보고서를 JSON 파일로 저장한다.

        Parameters
        ----------
        results:
            ``true_class``, ``pred_class``, ``confidence``, ``path`` 를 포함한 결과 목록.
        output_path:
            저장할 JSON 파일 경로.

        Returns
        -------
        str
            저장된 파일의 절대 경로.
        """
        fp_list = self.find_false_positives(results)
        fn_list = self.find_false_negatives(results)
        hard_list = self.find_hard_examples(results)
        conf_dist = self.get_confidence_distribution(results)

        total = len(results)
        fp_count = len(fp_list)
        fn_count = len(fn_list)
        hard_count = len(hard_list)

        report: Dict[str, Any] = {
            "summary": {
                "total_samples": total,
                "false_positives": fp_count,
                "false_negatives": fn_count,
                "hard_examples": hard_count,
                "fp_rate": round(fp_count / total, 4) if total > 0 else 0.0,
                "fn_rate": round(fn_count / total, 4) if total > 0 else 0.0,
            },
            "confidence_distribution": conf_dist,
            "false_positives": fp_list[:50],   # 상위 50개만 포함
            "false_negatives": fn_list[:50],
            "hard_examples": hard_list[:50],
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)

        abs_path = os.path.abspath(output_path)
        logger.info(
            "오류 보고서 저장: path=%s, fp=%d, fn=%d, hard=%d",
            abs_path, fp_count, fn_count, hard_count,
        )
        return abs_path

    # ------------------------------------------------------------------
    # 내부 유틸리티
    # ------------------------------------------------------------------

    def _collect_images(self, directory: str) -> List[str]:
        """
        directory 아래 이미지 파일 목록을 재귀적으로 수집한다.

        Parameters
        ----------
        directory:
            탐색할 루트 디렉터리.

        Returns
        -------
        List[str]
            절대 경로 이미지 파일 목록.
        """
        if not os.path.isdir(directory):
            return []

        paths: List[str] = []
        for root, _dirs, files in os.walk(directory):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in _IMAGE_EXTS:
                    paths.append(os.path.join(root, fname))
        return sorted(paths)
