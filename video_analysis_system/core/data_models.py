"""
core/data_models.py — 시스템 전체 공유 데이터 모델
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from core.states import AbnormalityType, ROIState, SystemState


# ---------------------------------------------------------------------------
# DetectionResult — AI 모델 단일 추론 결과
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """AI 모델 추론 결과를 담는 데이터 클래스.

    단일 프레임 또는 ROI에 대한 분류/탐지 결과를 표현한다.
    source_model_name, source_model_version 필드로 어느 모델이
    생성한 결과인지 추적할 수 있다.
    """

    class_id: int = -1
    label: str = "unknown"
    confidence: float = 0.0
    scores: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    source_model_name: str = ""
    source_model_version: str = ""

    @property
    def is_abnormal(self) -> bool:
        """class_id가 1이면 이상 상태로 판단한다 (0=정상, 1=이상)."""
        return self.class_id == 1


# ---------------------------------------------------------------------------
# RuleResult — 규칙 기반 판단 결과
# ---------------------------------------------------------------------------

@dataclass
class RuleResult:
    """규칙 기반 판단 결과를 담는 데이터 클래스.

    개별 규칙의 트리거 여부, 심각도(0.0~1.0), 이상 유형,
    사람이 읽을 수 있는 메시지를 포함한다.
    """

    rule_name: str = ""
    triggered: bool = False
    severity: float = 0.0
    abnormality_type: Optional[AbnormalityType] = None
    message: str = ""


# ---------------------------------------------------------------------------
# TrackingInfo — 객체 추적 상태 정보
# ---------------------------------------------------------------------------

@dataclass
class TrackingInfo:
    """객체 추적 상태 정보를 담는 데이터 클래스.

    트래커 종류, 활성 여부, 소실 여부, 신뢰도,
    마지막 업데이트 프레임 번호를 보관한다.
    """

    tracker_type: str = "none"
    is_active: bool = False
    is_lost: bool = False
    confidence: float = 1.0
    last_update_frame: int = -1


# ---------------------------------------------------------------------------
# TemporalSummary — ROI 시계열 통계 요약
# ---------------------------------------------------------------------------

@dataclass
class TemporalSummary:
    """ROI의 시간적 특성 요약 데이터 클래스.

    슬라이딩 윈도우 구간 동안의 강도 이력, 분산, 델타를 저장하고
    최근 평균/표준편차, 추세 기울기, 진동 진폭, 급격한 변화 여부 등
    파생된 시계열 통계를 함께 보관한다.
    """

    roi_id: str = ""
    window_size: int = 0
    mean_intensity_history: List[float] = field(default_factory=list)
    variance_history: List[float] = field(default_factory=list)
    delta_history: List[float] = field(default_factory=list)
    recent_mean: float = 0.0
    recent_std: float = 0.0
    trend_slope: float = 0.0
    oscillation_amplitude: float = 0.0
    max_sudden_change: float = 0.0
    is_stuck: bool = False
    is_drifting: bool = False
    is_oscillating: bool = False
    has_sudden_change: bool = False


# ---------------------------------------------------------------------------
# ROIData — 단일 관심 영역(ROI) 완전 정보
# ---------------------------------------------------------------------------

@dataclass
class ROIData:
    """관심 영역(ROI) 데이터를 담는 데이터 클래스.

    원본 프레임 좌표(original_rect)와 화면 표시 좌표(display_rect)를
    분리하여 관리하며, 크롭 이미지, 피처, 추적 정보, 탐지 결과,
    시간적 요약, 규칙 결과, UI 상태 등을 통합한다.
    """

    roi_id: str = ""
    name: str = ""
    original_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    display_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    normalized_rect: Optional[Tuple[float, float, float, float]] = None
    cropped: Optional[np.ndarray] = None
    cropped_prev: Optional[np.ndarray] = None
    features: Dict[str, float] = field(default_factory=dict)
    tracking_info: TrackingInfo = field(default_factory=TrackingInfo)
    detection: Optional[DetectionResult] = None
    temporal_summary: Optional[TemporalSummary] = None
    rule_results: List[RuleResult] = field(default_factory=list)
    roi_state: ROIState = ROIState.UNKNOWN
    abnormality_reason: str = ""
    visible: bool = True
    editable: bool = True
    selected: bool = False
    color_hint: str = ""

    @property
    def x(self) -> int:
        """원본 좌표 기준 x (좌상단)."""
        return self.original_rect[0]

    @property
    def y(self) -> int:
        """원본 좌표 기준 y (좌상단)."""
        return self.original_rect[1]

    @property
    def w(self) -> int:
        """원본 좌표 기준 너비."""
        return self.original_rect[2]

    @property
    def h(self) -> int:
        """원본 좌표 기준 높이."""
        return self.original_rect[3]

    @property
    def center(self) -> Tuple[int, int]:
        """원본 좌표 기준 중심점 (cx, cy)."""
        return (
            self.original_rect[0] + self.original_rect[2] // 2,
            self.original_rect[1] + self.original_rect[3] // 2,
        )

    @property
    def display_center(self) -> Tuple[int, int]:
        """화면 표시 좌표 기준 중심점 (cx, cy)."""
        return (
            self.display_rect[0] + self.display_rect[2] // 2,
            self.display_rect[1] + self.display_rect[3] // 2,
        )


# ---------------------------------------------------------------------------
# FrameContext — 파이프라인 전체를 통과하는 중심 운반 객체
# ---------------------------------------------------------------------------

@dataclass
class FrameContext:
    """단일 프레임의 처리 컨텍스트 전체를 담는 데이터 클래스.

    파이프라인 전체를 통과하는 중심 운반 객체이다.
    원본/처리 프레임, 좌표 변환 정보(transform_info), ROI 목록,
    프레임 피처, 탐지 결과, 시스템 상태, 이벤트 정보,
    파이프라인 성능 지표, 경고, 어노테이션을 통합 관리한다.
    """

    # ── 식별 정보 ────────────────────────────────────────────────────────
    frame_index: int = 0
    timestamp: float = field(default_factory=time.time)

    # ── 프레임 데이터 ─────────────────────────────────────────────────────
    raw_frame: Optional[np.ndarray] = None
    processed_frame: Optional[np.ndarray] = None
    transform_info: Optional[Any] = None

    # ── ROI 목록 ──────────────────────────────────────────────────────────
    rois: List[ROIData] = field(default_factory=list)

    # ── 프레임 수준 피처 ──────────────────────────────────────────────────
    frame_features: Dict[str, float] = field(default_factory=dict)

    # ── AI 추론 결과 ──────────────────────────────────────────────────────
    frame_detection: Optional[DetectionResult] = None
    inference_time_ms: float = 0.0

    # ── 의사결정 엔진 출력 ────────────────────────────────────────────────
    system_state: SystemState = SystemState.INITIALIZING
    state_confidence: float = 0.0
    triggered_rules: List[RuleResult] = field(default_factory=list)
    consecutive_abnormal: int = 0
    consecutive_normal: int = 0

    # ── 이벤트 플래그 ─────────────────────────────────────────────────────
    is_event: bool = False
    event_type: str = ""
    event_severity: float = 0.0

    # ── 디버그 / 프로파일링 ───────────────────────────────────────────────
    debug_info: Dict[str, Any] = field(default_factory=dict)
    pipeline_times_ms: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    annotations: Dict[str, Any] = field(default_factory=dict)

    @property
    def frame_shape(self) -> Optional[Tuple[int, ...]]:
        """raw_frame의 shape를 반환한다. 프레임이 없으면 None."""
        if self.raw_frame is not None:
            return self.raw_frame.shape
        return None

    def get_roi(self, roi_id: str) -> Optional[ROIData]:
        """roi_id에 해당하는 ROIData를 반환한다. 없으면 None."""
        for roi in self.rois:
            if roi.roi_id == roi_id:
                return roi
        return None

    def add_warning(self, msg: str) -> None:
        """경고 메시지를 warnings 목록에 추가한다."""
        self.warnings.append(msg)

    def mark_event(self, event_type: str, severity: float = 1.0) -> None:
        """현재 프레임을 이벤트로 표시하고 유형과 심각도를 기록한다."""
        self.is_event = True
        self.event_type = event_type
        self.event_severity = severity


# ---------------------------------------------------------------------------
# TrainingJobConfig — 학습 작업 설정
# ---------------------------------------------------------------------------

@dataclass
class TrainingJobConfig:
    """학습 작업 설정을 담는 데이터 클래스.

    데이터셋 경로, 모델 패밀리, 하이퍼파라미터(배치 크기, 에폭,
    학습률, 스케줄러), 증강 설정, 검증 분할 비율,
    출력 형식(onnx/torchscript/saved_model) 등을 보관한다.
    """

    job_name: str = "job_001"
    dataset_path: str = ""
    label_path: str = ""
    task_type: str = "classification"
    model_family: str = "resnet18"
    input_shape: Tuple[int, int, int] = (3, 224, 224)
    batch_size: int = 32
    epochs: int = 50
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    scheduler: str = "cosine"
    augmentation_config: Dict[str, Any] = field(default_factory=dict)
    validation_split: float = 0.2
    export_format: str = "onnx"
    output_dir: str = "models"
    notes: str = ""
    tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ModelMetadata — 모델 레지스트리 메타데이터
# ---------------------------------------------------------------------------

@dataclass
class ModelMetadata:
    """배포된 모델의 메타데이터를 담는 데이터 클래스.

    모델 이름, 버전, 입력 스펙(shape/dtype/mean/std), 클래스 맵,
    학습 날짜, 성능 지표, 파일 경로, 상태
    (draft/validated/production/archived), 태그, 노트를 관리한다.
    """

    model_name: str = ""
    version: str = "1.0.0"
    task_type: str = "classification"
    framework: str = "onnx"
    input_spec: Dict[str, Any] = field(default_factory=dict)
    class_map: Dict[int, str] = field(default_factory=dict)
    training_date: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    file_path: str = ""
    status: str = "draft"
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def is_production(self) -> bool:
        """status가 'production'이면 True를 반환한다."""
        return self.status == "production"

    def to_dict(self) -> Dict[str, Any]:
        """모델 메타데이터를 직렬화 가능한 딕셔너리로 변환한다."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "task_type": self.task_type,
            "framework": self.framework,
            "input_spec": self.input_spec,
            "class_map": {str(k): v for k, v in self.class_map.items()},
            "training_date": self.training_date,
            "metrics": self.metrics,
            "file_path": self.file_path,
            "status": self.status,
            "tags": self.tags,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# ExperimentRecord — 학습 실험 기록
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRecord:
    """학습 실험 기록을 담는 데이터 클래스.

    실험 ID, 이름, 작업 설정(TrainingJobConfig), 시작/종료 시각,
    상태(pending/running/completed/failed), 성능 지표,
    최적 에폭, 모델/로그 경로를 보관한다.
    """

    experiment_id: str = ""
    name: str = ""
    job_config: Optional[TrainingJobConfig] = None
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = "pending"
    metrics: Dict[str, float] = field(default_factory=dict)
    best_epoch: int = -1
    model_path: str = ""
    log_path: str = ""
    notes: str = ""

    @property
    def duration_sec(self) -> float:
        """실험 소요 시간을 초 단위로 반환한다."""
        return self.end_time - self.start_time


# ---------------------------------------------------------------------------
# EventRecord — 이벤트 로그 저장 레코드
# ---------------------------------------------------------------------------

@dataclass
class EventRecord:
    """시스템 이벤트 기록을 담는 데이터 클래스.

    탐지된 이상 이벤트를 영속적으로 저장하기 위한 구조이며
    이벤트 ID, 프레임 인덱스, 타임스탬프, 이벤트 유형, 심각도,
    시스템 상태, ROI ID, 이상 유형, 메시지, 스냅샷 경로를 포함한다.
    """

    event_id: str = ""
    frame_index: int = 0
    timestamp: float = 0.0
    event_type: str = ""
    severity: float = 0.0
    system_state: str = ""
    roi_id: str = ""
    abnormality_type: str = ""
    message: str = ""
    snapshot_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """이벤트 레코드를 직렬화 가능한 딕셔너리로 변환한다."""
        return {
            "event_id": self.event_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "severity": self.severity,
            "system_state": self.system_state,
            "roi_id": self.roi_id,
            "abnormality_type": self.abnormality_type,
            "message": self.message,
            "snapshot_path": self.snapshot_path,
        }
