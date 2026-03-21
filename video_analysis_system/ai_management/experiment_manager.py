"""
ai_management/experiment_manager.py — 학습 실험 추적 및 비교 모듈.

ExperimentManager 는 학습 실험의 생성·진행·완료·실패 상태를 관리하고
에포크별 지표를 기록한다. JSON 파일로 모든 실험을 영구 저장할 수 있다.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from core.data_models import ExperimentRecord, TrainingJobConfig

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    학습 실험의 전체 생애주기를 추적하는 클래스.

    - 실험 생성·완료·실패 상태 전환
    - 에포크별 지표 누적
    - 최적 실험 검색 및 다중 실험 비교
    - JSON 직렬화/역직렬화
    """

    def __init__(self) -> None:
        """ExperimentManager 초기화."""
        self.experiments: Dict[str, ExperimentRecord] = {}
        logger.debug("ExperimentManager 초기화 완료.")

    # ------------------------------------------------------------------
    # 실험 생성
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        job_config: TrainingJobConfig,
        notes: str = "",
    ) -> str:
        """
        새 실험 레코드를 생성하고 exp_id 를 반환한다.

        Parameters
        ----------
        name:
            실험 이름.
        job_config:
            학습 설정 데이터 클래스.
        notes:
            선택적 메모.

        Returns
        -------
        str
            생성된 실험 ID (UUID4).
        """
        exp_id = str(uuid.uuid4())
        record = ExperimentRecord(
            experiment_id=exp_id,
            name=name,
            job_config=job_config,
            status="pending",
            notes=notes,
        )
        self.experiments[exp_id] = record
        logger.info("실험 생성: exp_id=%s, name=%s", exp_id, name)
        return exp_id

    # ------------------------------------------------------------------
    # 지표 갱신
    # ------------------------------------------------------------------

    def update_metrics(
        self,
        exp_id: str,
        metrics: Dict[str, float],
        epoch: int,
    ) -> None:
        """
        특정 에포크의 지표를 실험 레코드에 추가한다.

        Parameters
        ----------
        exp_id:
            갱신할 실험 ID.
        metrics:
            ``{metric_name: value}`` 딕셔너리.
        epoch:
            에포크 번호.

        Raises
        ------
        KeyError
            exp_id 가 존재하지 않는 경우.
        """
        record = self._get_record(exp_id)
        if record.status == "pending":
            record.status = "running"
            record.start_time = time.time()

        entry: Dict[str, Any] = {"epoch": epoch, **metrics}
        # metrics_history 는 ExperimentRecord 에 없으므로 notes 를 통해 관리하거나
        # 메모리 전용 확장 속성으로 보관한다.
        # 여기서는 _epoch_log 내부 dict 로 관리한다.
        if not hasattr(record, "_epoch_log"):
            object.__setattr__(record, "_epoch_log", [])
        record._epoch_log.append(entry)  # type: ignore[attr-defined]

        # best_epoch 갱신 (val_accuracy 또는 accuracy 기준)
        val_acc = metrics.get("val_accuracy", metrics.get("accuracy", -1.0))
        best_acc = record.metrics.get("best_val_accuracy", -1.0)
        if val_acc > best_acc:
            record.metrics["best_val_accuracy"] = val_acc
            record.best_epoch = epoch

        # 최신 지표를 record.metrics 에 덮어씀
        record.metrics.update({k: v for k, v in metrics.items()})
        logger.debug("지표 갱신: exp_id=%s, epoch=%d, metrics=%s", exp_id, epoch, metrics)

    # ------------------------------------------------------------------
    # 완료 / 실패 처리
    # ------------------------------------------------------------------

    def complete_experiment(
        self,
        exp_id: str,
        model_path: str,
        metrics: Dict[str, float],
    ) -> None:
        """
        실험을 completed 상태로 전환한다.

        Parameters
        ----------
        exp_id:
            완료할 실험 ID.
        model_path:
            저장된 최종 모델 파일 경로.
        metrics:
            최종 평가 지표.
        """
        record = self._get_record(exp_id)
        record.status = "completed"
        record.model_path = model_path
        record.end_time = time.time()
        record.metrics.update(metrics)
        logger.info(
            "실험 완료: exp_id=%s, model_path=%s, metrics=%s",
            exp_id, model_path, metrics,
        )

    def fail_experiment(self, exp_id: str, reason: str) -> None:
        """
        실험을 failed 상태로 전환한다.

        Parameters
        ----------
        exp_id:
            실패 처리할 실험 ID.
        reason:
            실패 원인 메시지.
        """
        record = self._get_record(exp_id)
        record.status = "failed"
        record.end_time = time.time()
        record.notes = f"[실패] {reason}\n{record.notes}"
        logger.warning("실험 실패: exp_id=%s, reason=%s", exp_id, reason)

    # ------------------------------------------------------------------
    # 목록 조회
    # ------------------------------------------------------------------

    def list_experiments(
        self,
        status_filter: Optional[str] = None,
    ) -> List[ExperimentRecord]:
        """
        등록된 실험 레코드 목록을 반환한다.

        Parameters
        ----------
        status_filter:
            None 이면 전체, 문자열이면 해당 status 만 필터링.
            예: ``"completed"``, ``"running"``, ``"failed"``.

        Returns
        -------
        List[ExperimentRecord]
            필터링된 실험 레코드 목록.
        """
        records = list(self.experiments.values())
        if status_filter is not None:
            records = [r for r in records if r.status == status_filter]
        return records

    # ------------------------------------------------------------------
    # 최적 실험 검색
    # ------------------------------------------------------------------

    def get_best_experiment(
        self,
        metric: str = "accuracy",
        higher_better: bool = True,
    ) -> Optional[ExperimentRecord]:
        """
        completed 실험 중 지정 지표가 최적인 실험을 반환한다.

        Parameters
        ----------
        metric:
            비교할 지표 키.
        higher_better:
            True 이면 최대값, False 이면 최솟값을 찾는다.

        Returns
        -------
        Optional[ExperimentRecord]
            최적 실험 레코드. completed 실험이 없으면 None.
        """
        completed = [r for r in self.experiments.values() if r.status == "completed"]
        if not completed:
            return None

        def _key(r: ExperimentRecord) -> float:
            val = r.metrics.get(metric, float("-inf") if higher_better else float("inf"))
            return val if higher_better else -val

        best = max(completed, key=_key)
        logger.info(
            "최적 실험: exp_id=%s, metric=%s, value=%s",
            best.experiment_id,
            metric,
            best.metrics.get(metric),
        )
        return best

    # ------------------------------------------------------------------
    # 다중 실험 비교
    # ------------------------------------------------------------------

    def compare_experiments(
        self,
        exp_ids: List[str],
    ) -> List[dict]:
        """
        여러 실험의 지표를 나란히 비교하는 딕셔너리 목록을 반환한다.

        Parameters
        ----------
        exp_ids:
            비교할 실험 ID 목록.

        Returns
        -------
        List[dict]
            각 실험의 요약 딕셔너리 목록
            (exp_id, name, status, metrics, best_epoch, duration_sec).
        """
        result: List[dict] = []
        for exp_id in exp_ids:
            record = self.experiments.get(exp_id)
            if record is None:
                logger.warning("비교 요청: 실험을 찾을 수 없음 exp_id=%s", exp_id)
                continue
            result.append({
                "exp_id": record.experiment_id,
                "name": record.name,
                "status": record.status,
                "metrics": dict(record.metrics),
                "best_epoch": record.best_epoch,
                "duration_sec": round(record.duration_sec, 2),
                "model_path": record.model_path,
            })
        return result

    # ------------------------------------------------------------------
    # 직렬화 / 역직렬화
    # ------------------------------------------------------------------

    def save_all(self, path: str) -> None:
        """
        모든 실험 레코드를 JSON 파일로 저장한다.

        Parameters
        ----------
        path:
            저장할 JSON 파일 경로.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        payload: Dict[str, Any] = {}
        for exp_id, record in self.experiments.items():
            config_dict: Optional[dict] = None
            if record.job_config is not None:
                try:
                    from dataclasses import asdict
                    config_dict = asdict(record.job_config)
                except Exception:  # noqa: BLE001
                    config_dict = str(record.job_config)  # type: ignore[assignment]

            payload[exp_id] = {
                "experiment_id": record.experiment_id,
                "name": record.name,
                "status": record.status,
                "start_time": record.start_time,
                "end_time": record.end_time,
                "metrics": record.metrics,
                "best_epoch": record.best_epoch,
                "model_path": record.model_path,
                "log_path": record.log_path,
                "notes": record.notes,
                "job_config": config_dict,
            }

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info("실험 저장: path=%s, count=%d", path, len(self.experiments))

    def load_all(self, path: str) -> None:
        """
        JSON 파일에서 실험 레코드를 복원한다. 기존 레코드를 덮어쓴다.

        Parameters
        ----------
        path:
            읽어들일 JSON 파일 경로.

        Raises
        ------
        FileNotFoundError
            파일이 존재하지 않는 경우.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"실험 파일을 찾을 수 없음: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            payload: dict = json.load(fh)

        self.experiments = {}
        for exp_id, d in payload.items():
            # job_config 는 단순 dict 로만 저장 (TrainingJobConfig 재구성 생략)
            record = ExperimentRecord(
                experiment_id=d.get("experiment_id", exp_id),
                name=d.get("name", ""),
                status=d.get("status", "pending"),
                start_time=d.get("start_time", 0.0),
                end_time=d.get("end_time", 0.0),
                metrics=d.get("metrics", {}),
                best_epoch=d.get("best_epoch", -1),
                model_path=d.get("model_path", ""),
                log_path=d.get("log_path", ""),
                notes=d.get("notes", ""),
            )
            self.experiments[exp_id] = record

        logger.info("실험 로드: path=%s, count=%d", path, len(self.experiments))

    # ------------------------------------------------------------------
    # 내부 유틸리티
    # ------------------------------------------------------------------

    def _get_record(self, exp_id: str) -> ExperimentRecord:
        """exp_id 로 ExperimentRecord 를 가져온다. 없으면 KeyError."""
        record = self.experiments.get(exp_id)
        if record is None:
            raise KeyError(f"실험을 찾을 수 없음: exp_id={exp_id}")
        return record
