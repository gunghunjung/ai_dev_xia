"""
ai_management/training_manager.py — 모델 학습 작업 관리 모듈.

TrainingManager 는 학습 작업(job)을 생성·실행·중단하고 체크포인트를 저장한다.
PlaceholderTrainer 는 실제 딥러닝 라이브러리 없이도 학습 루프를 시뮬레이션하여
UI 연동을 테스트할 수 있도록 한다.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from core.data_models import TrainingJobConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 타입 별칭
# ---------------------------------------------------------------------------

EpochCallback = Callable[[int, Dict[str, float]], None]
"""에포크 완료 시 호출되는 콜백. (epoch_num, metrics_dict) → None."""


# ---------------------------------------------------------------------------
# 내부 작업 레코드
# ---------------------------------------------------------------------------

@dataclass
class _JobRecord:
    """학습 작업 하나의 런타임 상태를 담는 내부 레코드."""

    job_id: str
    config: TrainingJobConfig
    status: str = "created"        # created | running | completed | failed | stopped
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    current_epoch: int = 0
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint_paths: List[str] = field(default_factory=list)
    error_message: str = ""
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None

    def to_dict(self) -> dict:
        """직렬화 가능한 딕셔너리 반환 (threading 객체 제외)."""
        return {
            "job_id": self.job_id,
            "config": asdict(self.config) if self.config else {},
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "current_epoch": self.current_epoch,
            "metrics_history": self.metrics_history,
            "checkpoint_paths": self.checkpoint_paths,
            "error_message": self.error_message,
        }


# ---------------------------------------------------------------------------
# PlaceholderTrainer — fake 학습 루프
# ---------------------------------------------------------------------------

class PlaceholderTrainer:
    """
    실제 학습 라이브러리 없이 학습 루프를 시뮬레이션하는 플레이스홀더 트레이너.

    에포크마다 loss 는 지수 감쇠, accuracy 는 선형 증가 + 노이즈로 계산된다.
    stop_event 가 세트되면 즉시 루프를 종료한다.
    """

    def __init__(
        self,
        job_record: _JobRecord,
        callback: Optional[EpochCallback] = None,
    ) -> None:
        """
        Parameters
        ----------
        job_record:
            학습할 작업 레코드. 직접 상태를 갱신한다.
        callback:
            에포크 완료 시 호출될 함수. ``(epoch, metrics)`` 형태.
        """
        self._record = job_record
        self._callback = callback

    def run(self) -> None:
        """백그라운드 스레드에서 실행되는 메인 학습 루프."""
        record = self._record
        config = record.config
        epochs: int = getattr(config, "epochs", 30)
        record.status = "running"
        record.started_at = time.time()
        logger.info("학습 시작: job_id=%s, epochs=%d", record.job_id, epochs)

        try:
            for epoch in range(1, epochs + 1):
                if record.stop_event.is_set():
                    logger.info("학습 중단 요청 수신: job_id=%s, epoch=%d", record.job_id, epoch)
                    record.status = "stopped"
                    break

                # --- fake metric 계산 ---
                progress = epoch / epochs
                loss = max(0.01, 2.0 * (0.85 ** epoch) + random.uniform(-0.05, 0.05))
                acc = min(0.99, 0.5 + progress * 0.4 + random.uniform(-0.02, 0.02))
                val_loss = loss * random.uniform(1.0, 1.3)
                val_acc = max(0.0, acc - random.uniform(0.0, 0.05))

                metrics: Dict[str, float] = {
                    "epoch": float(epoch),
                    "loss": round(loss, 4),
                    "accuracy": round(acc, 4),
                    "val_loss": round(val_loss, 4),
                    "val_accuracy": round(val_acc, 4),
                }
                record.current_epoch = epoch
                record.metrics_history.append(metrics)

                logger.debug(
                    "epoch=%d/%d  loss=%.4f  acc=%.4f  val_loss=%.4f  val_acc=%.4f",
                    epoch, epochs, loss, acc, val_loss, val_acc,
                )

                if self._callback is not None:
                    try:
                        self._callback(epoch, metrics)
                    except Exception as cb_err:  # noqa: BLE001
                        logger.warning("콜백 오류 (무시됨): %s", cb_err)

                # 에포크당 약 0.1초 지연 (시뮬레이션)
                time.sleep(0.1)

            else:
                # 정상 완료
                record.status = "completed"
                logger.info("학습 완료: job_id=%s", record.job_id)

        except Exception as exc:  # noqa: BLE001
            record.status = "failed"
            record.error_message = str(exc)
            logger.exception("학습 중 예외 발생: job_id=%s", record.job_id)
        finally:
            record.finished_at = time.time()


# ---------------------------------------------------------------------------
# TrainingManager
# ---------------------------------------------------------------------------

class TrainingManager:
    """
    학습 작업의 전체 생애주기를 관리하는 클래스.

    - 작업 생성·시작·중단
    - 에포크별 콜백 지원
    - 체크포인트(JSON 메타) 저장
    - 작업 목록 조회 및 상태 확인
    """

    def __init__(self, checkpoint_base_dir: str = "checkpoints") -> None:
        """
        Parameters
        ----------
        checkpoint_base_dir:
            체크포인트 파일을 저장할 기본 디렉터리.
        """
        self.jobs: Dict[str, _JobRecord] = {}
        self._checkpoint_base_dir = checkpoint_base_dir
        logger.debug("TrainingManager 초기화 완료.")

    # ------------------------------------------------------------------
    # 작업 생성
    # ------------------------------------------------------------------

    def create_job(self, config: TrainingJobConfig) -> str:
        """
        새 학습 작업을 생성하고 job_id 를 반환한다.

        Parameters
        ----------
        config:
            학습 설정 데이터 클래스.

        Returns
        -------
        str
            UUID4 기반 job_id.
        """
        job_id = str(uuid.uuid4())
        record = _JobRecord(job_id=job_id, config=config)
        self.jobs[job_id] = record
        logger.info(
            "학습 작업 생성: job_id=%s, job_name=%s",
            job_id,
            getattr(config, "job_name", ""),
        )
        return job_id

    # ------------------------------------------------------------------
    # 작업 시작
    # ------------------------------------------------------------------

    def start_job(
        self,
        job_id: str,
        callback: Optional[EpochCallback] = None,
    ) -> None:
        """
        학습 작업을 백그라운드 스레드로 실행한다.

        Parameters
        ----------
        job_id:
            시작할 작업 ID.
        callback:
            에포크 완료 시 호출될 콜백 함수.

        Raises
        ------
        KeyError
            job_id 가 존재하지 않는 경우.
        RuntimeError
            이미 실행 중인 작업을 다시 시작하려는 경우.
        """
        record = self._get_record(job_id)
        if record.status == "running":
            raise RuntimeError(f"이미 실행 중인 작업입니다: job_id={job_id}")

        # 재시작 시 stop_event 초기화
        record.stop_event.clear()
        record.status = "created"

        trainer = PlaceholderTrainer(job_record=record, callback=callback)
        thread = threading.Thread(
            target=trainer.run,
            name=f"trainer-{job_id[:8]}",
            daemon=True,
        )
        record.thread = thread
        thread.start()
        logger.info("학습 스레드 시작: job_id=%s", job_id)

    # ------------------------------------------------------------------
    # 작업 중단
    # ------------------------------------------------------------------

    def stop_job(self, job_id: str) -> None:
        """
        실행 중인 학습 작업에 중단 신호를 보낸다.

        중단은 현재 에포크가 끝난 후 적용된다.

        Parameters
        ----------
        job_id:
            중단할 작업 ID.
        """
        record = self._get_record(job_id)
        record.stop_event.set()
        logger.info("학습 중단 요청: job_id=%s", job_id)

    # ------------------------------------------------------------------
    # 상태 조회
    # ------------------------------------------------------------------

    def get_job_status(self, job_id: str) -> dict:
        """
        특정 작업의 현재 상태 딕셔너리를 반환한다.

        Parameters
        ----------
        job_id:
            조회할 작업 ID.

        Returns
        -------
        dict
            작업 메타정보 및 현재 메트릭 이력.
        """
        record = self._get_record(job_id)
        return record.to_dict()

    def list_jobs(self) -> List[dict]:
        """
        등록된 모든 학습 작업의 요약 목록을 반환한다.

        Returns
        -------
        List[dict]
            각 작업의 상태 딕셔너리 목록.
        """
        return [r.to_dict() for r in self.jobs.values()]

    # ------------------------------------------------------------------
    # 체크포인트
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        job_id: str,
        epoch: int,
        metrics: Dict[str, float],
    ) -> str:
        """
        특정 에포크의 체크포인트 메타정보를 JSON 파일로 저장한다.

        실제 모델 가중치 저장은 지원하지 않으며, 학습 진행 기록용이다.

        Parameters
        ----------
        job_id:
            체크포인트를 저장할 작업 ID.
        epoch:
            에포크 번호.
        metrics:
            저장할 성능 지표 딕셔너리.

        Returns
        -------
        str
            저장된 체크포인트 파일의 절대 경로.
        """
        record = self._get_record(job_id)
        ckpt_dir = os.path.join(self._checkpoint_base_dir, job_id)
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.json")
        payload = {
            "job_id": job_id,
            "epoch": epoch,
            "metrics": metrics,
            "saved_at": time.time(),
        }
        with open(ckpt_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        record.checkpoint_paths.append(os.path.abspath(ckpt_path))
        logger.info("체크포인트 저장: job_id=%s, epoch=%d, path=%s", job_id, epoch, ckpt_path)
        return os.path.abspath(ckpt_path)

    # ------------------------------------------------------------------
    # 재개 (placeholder)
    # ------------------------------------------------------------------

    def resume_job(self, job_id: str) -> None:
        """
        마지막 체크포인트에서 학습을 재개한다. (현재는 placeholder)

        Parameters
        ----------
        job_id:
            재개할 작업 ID.
        """
        record = self._get_record(job_id)
        if not record.checkpoint_paths:
            logger.warning("재개할 체크포인트가 없습니다: job_id=%s", job_id)
            return

        last_ckpt = record.checkpoint_paths[-1]
        logger.info("학습 재개 (placeholder): job_id=%s, last_checkpoint=%s", job_id, last_ckpt)
        # 실제 구현에서는 모델 가중치를 로드한 후 start_job 을 호출한다.
        self.start_job(job_id)

    # ------------------------------------------------------------------
    # 내부 유틸리티
    # ------------------------------------------------------------------

    def _get_record(self, job_id: str) -> _JobRecord:
        """job_id 로 _JobRecord 를 가져온다. 없으면 KeyError."""
        record = self.jobs.get(job_id)
        if record is None:
            raise KeyError(f"학습 작업을 찾을 수 없음: job_id={job_id}")
        return record
