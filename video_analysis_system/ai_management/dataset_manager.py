"""
ai_management/dataset_manager.py — 데이터셋 등록 및 관리 모듈.

DatasetManager 는 학습/평가에 사용되는 데이터셋을 레지스트리로 관리한다.
경로 기반으로 샘플 수를 인덱싱하고, JSON 직렬화를 통해 상태를 영구 저장한다.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DatasetInfo — 데이터셋 항목 메타정보
# ---------------------------------------------------------------------------

@dataclass
class DatasetInfo:
    """등록된 데이터셋 하나의 메타정보를 담는 데이터 클래스."""

    id: str
    """고유 식별자 (UUID4 기반)."""

    name: str
    """사용자 지정 이름."""

    path: str
    """데이터셋 루트 디렉터리 절대 경로."""

    task_type: str
    """태스크 종류. 예: classification, detection, segmentation."""

    sample_count: int = 0
    """인덱싱된 샘플 파일 수."""

    description: str = ""
    """선택적 설명 문자열."""

    created_at: float = field(default_factory=time.time)
    """등록 시각 (Unix timestamp)."""

    def to_dict(self) -> dict:
        """JSON 직렬화용 딕셔너리 반환."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DatasetInfo:
        """딕셔너리에서 DatasetInfo 복원."""
        return cls(
            id=d["id"],
            name=d["name"],
            path=d["path"],
            task_type=d["task_type"],
            sample_count=d.get("sample_count", 0),
            description=d.get("description", ""),
            created_at=d.get("created_at", time.time()),
        )


# ---------------------------------------------------------------------------
# DatasetManager
# ---------------------------------------------------------------------------

class DatasetManager:
    """
    데이터셋 레지스트리를 관리하는 클래스.

    - 데이터셋 등록/조회/삭제
    - 경로 아래 파일을 글로브하여 샘플 수 인덱싱
    - 레지스트리를 JSON 파일로 저장/복원
    """

    # 샘플로 인정하는 확장자 목록
    _SAMPLE_EXTENSIONS: tuple[str, ...] = (
        "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff",
        "*.npy", "*.npz",
    )

    def __init__(self) -> None:
        """DatasetManager 초기화."""
        self.datasets: Dict[str, DatasetInfo] = {}
        logger.debug("DatasetManager 초기화 완료.")

    # ------------------------------------------------------------------
    # 등록 / 조회 / 삭제
    # ------------------------------------------------------------------

    def register_dataset(
        self,
        name: str,
        path: str,
        task_type: str,
        description: str = "",
    ) -> str:
        """
        새 데이터셋을 레지스트리에 등록한다.

        Parameters
        ----------
        name:
            데이터셋 이름 (중복 허용).
        path:
            데이터셋 루트 디렉터리 경로.
        task_type:
            태스크 종류 (classification / detection / segmentation 등).
        description:
            선택적 설명.

        Returns
        -------
        str
            생성된 dataset_id (UUID4).
        """
        dataset_id = str(uuid.uuid4())
        info = DatasetInfo(
            id=dataset_id,
            name=name,
            path=os.path.abspath(path),
            task_type=task_type,
            description=description,
        )
        self.datasets[dataset_id] = info
        logger.info("데이터셋 등록: id=%s, name=%s, path=%s", dataset_id, name, path)
        return dataset_id

    def list_datasets(self) -> List[dict]:
        """
        등록된 모든 데이터셋의 정보를 딕셔너리 리스트로 반환한다.

        Returns
        -------
        List[dict]
            각 데이터셋의 메타정보 딕셔너리 목록.
        """
        return [info.to_dict() for info in self.datasets.values()]

    def get_dataset(self, dataset_id: str) -> Optional[DatasetInfo]:
        """
        dataset_id 로 DatasetInfo 를 조회한다.

        Parameters
        ----------
        dataset_id:
            조회할 데이터셋 ID.

        Returns
        -------
        Optional[DatasetInfo]
            존재하면 DatasetInfo, 없으면 None.
        """
        info = self.datasets.get(dataset_id)
        if info is None:
            logger.warning("데이터셋을 찾을 수 없음: id=%s", dataset_id)
        return info

    def remove_dataset(self, dataset_id: str) -> None:
        """
        레지스트리에서 데이터셋을 제거한다. 실제 파일은 삭제하지 않는다.

        Parameters
        ----------
        dataset_id:
            제거할 데이터셋 ID.

        Raises
        ------
        KeyError
            해당 ID 가 존재하지 않는 경우.
        """
        if dataset_id not in self.datasets:
            raise KeyError(f"데이터셋을 찾을 수 없음: {dataset_id}")
        name = self.datasets[dataset_id].name
        del self.datasets[dataset_id]
        logger.info("데이터셋 제거: id=%s, name=%s", dataset_id, name)

    # ------------------------------------------------------------------
    # 샘플 인덱싱
    # ------------------------------------------------------------------

    def index_samples(self, dataset_id: str) -> int:
        """
        데이터셋 경로 아래 이미지/배열 파일을 재귀적으로 글로브하여 샘플 수를 갱신한다.

        Parameters
        ----------
        dataset_id:
            인덱싱할 데이터셋 ID.

        Returns
        -------
        int
            발견된 샘플 파일 수.

        Raises
        ------
        KeyError
            해당 ID 가 존재하지 않는 경우.
        ValueError
            경로가 존재하지 않는 경우.
        """
        info = self.datasets.get(dataset_id)
        if info is None:
            raise KeyError(f"데이터셋을 찾을 수 없음: {dataset_id}")

        if not os.path.isdir(info.path):
            raise ValueError(f"데이터셋 경로가 유효하지 않음: {info.path}")

        found: set[str] = set()
        for pattern in self._SAMPLE_EXTENSIONS:
            # 재귀 글로브
            matches = glob.glob(os.path.join(info.path, "**", pattern), recursive=True)
            found.update(matches)

        count = len(found)
        info.sample_count = count
        logger.info(
            "샘플 인덱싱 완료: id=%s, count=%d, path=%s",
            dataset_id,
            count,
            info.path,
        )
        return count

    def get_sample_count(self, dataset_id: str) -> int:
        """
        마지막으로 인덱싱된 샘플 수를 반환한다. 인덱싱이 아직 안 된 경우 0.

        Parameters
        ----------
        dataset_id:
            조회할 데이터셋 ID.

        Returns
        -------
        int
            샘플 수.
        """
        info = self.datasets.get(dataset_id)
        if info is None:
            logger.warning("데이터셋을 찾을 수 없음: id=%s", dataset_id)
            return 0
        return info.sample_count

    # ------------------------------------------------------------------
    # 직렬화 / 역직렬화
    # ------------------------------------------------------------------

    def save_registry(self, path: str) -> None:
        """
        현재 레지스트리를 JSON 파일로 저장한다.

        Parameters
        ----------
        path:
            저장할 JSON 파일 경로.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {did: info.to_dict() for did, info in self.datasets.items()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info("데이터셋 레지스트리 저장: path=%s, count=%d", path, len(self.datasets))

    def load_registry(self, path: str) -> None:
        """
        JSON 파일에서 레지스트리를 복원한다. 기존 레지스트리를 덮어쓴다.

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
            raise FileNotFoundError(f"레지스트리 파일을 찾을 수 없음: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            payload: dict = json.load(fh)

        self.datasets = {
            did: DatasetInfo.from_dict(d) for did, d in payload.items()
        }
        logger.info("데이터셋 레지스트리 로드: path=%s, count=%d", path, len(self.datasets))
