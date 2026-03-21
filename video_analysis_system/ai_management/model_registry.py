"""
ai_management/model_registry.py — 모델 레지스트리 관리 모듈.

ModelRegistry 는 학습·배포된 모델을 등록·조회·검색하고
production 지정 및 아카이브 처리를 담당한다.
레지스트리는 JSON 파일로 내보내고 불러올 수 있다.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Dict, List, Optional

from core.data_models import ModelMetadata

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    배포 가능한 모델들의 메타정보를 중앙에서 관리하는 클래스.

    - 모델 등록·조회·삭제
    - production 지정 (기존 production → validated 로 다운그레이드)
    - 아카이브 처리
    - 이름/태그/태스크 타입 기반 검색
    - JSON 직렬화/역직렬화
    """

    def __init__(self) -> None:
        """ModelRegistry 초기화."""
        self.models: Dict[str, ModelMetadata] = {}
        logger.debug("ModelRegistry 초기화 완료.")

    # ------------------------------------------------------------------
    # 등록
    # ------------------------------------------------------------------

    def register(self, meta: ModelMetadata) -> str:
        """
        모델 메타정보를 레지스트리에 등록한다.

        model_id 가 비어 있으면 자동으로 UUID4 를 부여한다.

        Parameters
        ----------
        meta:
            등록할 ModelMetadata 인스턴스.

        Returns
        -------
        str
            등록된 model_id.
        """
        if not meta.model_name:
            raise ValueError("model_name 은 비어 있을 수 없습니다.")

        # model_id 자동 생성
        model_id = str(uuid.uuid4())
        # ModelMetadata 는 dataclass 이므로 새 인스턴스로 교체
        registered = ModelMetadata(
            model_name=meta.model_name,
            version=meta.version,
            task_type=meta.task_type,
            framework=meta.framework,
            input_spec=dict(meta.input_spec),
            class_map=dict(meta.class_map),
            training_date=meta.training_date,
            metrics=dict(meta.metrics),
            file_path=meta.file_path,
            status=meta.status,
            tags=list(meta.tags),
            notes=meta.notes,
        )
        # model_id 를 notes 보조 필드로 기록하는 대신 레지스트리 키로만 관리
        self.models[model_id] = registered
        logger.info(
            "모델 등록: model_id=%s, name=%s, version=%s, status=%s",
            model_id,
            meta.model_name,
            meta.version,
            meta.status,
        )
        return model_id

    # ------------------------------------------------------------------
    # 조회
    # ------------------------------------------------------------------

    def get(self, model_id: str) -> Optional[ModelMetadata]:
        """
        model_id 로 ModelMetadata 를 반환한다.

        Parameters
        ----------
        model_id:
            조회할 모델 ID.

        Returns
        -------
        Optional[ModelMetadata]
            존재하면 ModelMetadata, 없으면 None.
        """
        meta = self.models.get(model_id)
        if meta is None:
            logger.warning("모델을 찾을 수 없음: model_id=%s", model_id)
        return meta

    def list_models(
        self,
        task_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[ModelMetadata]:
        """
        등록된 모델 목록을 반환한다.

        Parameters
        ----------
        task_type:
            None 이면 전체, 문자열이면 해당 task_type 만 필터링.
        status:
            None 이면 전체, 문자열이면 해당 status 만 필터링.
            예: ``"production"``, ``"validated"``, ``"archived"``.

        Returns
        -------
        List[ModelMetadata]
            필터링된 ModelMetadata 목록.
        """
        result = list(self.models.values())
        if task_type is not None:
            result = [m for m in result if m.task_type == task_type]
        if status is not None:
            result = [m for m in result if m.status == status]
        return result

    # ------------------------------------------------------------------
    # 상태 변경
    # ------------------------------------------------------------------

    def set_production(self, model_id: str) -> None:
        """
        특정 모델을 production 상태로 지정한다.

        기존 production 모델은 모두 validated 로 다운그레이드된다.

        Parameters
        ----------
        model_id:
            production 으로 지정할 모델 ID.

        Raises
        ------
        KeyError
            model_id 가 존재하지 않는 경우.
        """
        target = self._get_meta(model_id)

        # 동일 task_type 의 기존 production → validated
        for mid, meta in self.models.items():
            if mid != model_id and meta.status == "production" and meta.task_type == target.task_type:
                meta.status = "validated"
                logger.info("기존 production → validated: model_id=%s", mid)

        target.status = "production"
        logger.info("production 지정: model_id=%s, name=%s", model_id, target.model_name)

    def archive(self, model_id: str) -> None:
        """
        모델을 archived 상태로 변경한다.

        Parameters
        ----------
        model_id:
            아카이브할 모델 ID.

        Raises
        ------
        KeyError
            model_id 가 존재하지 않는 경우.
        """
        meta = self._get_meta(model_id)
        prev_status = meta.status
        meta.status = "archived"
        logger.info(
            "모델 아카이브: model_id=%s, name=%s, prev_status=%s",
            model_id,
            meta.model_name,
            prev_status,
        )

    def delete(self, model_id: str) -> None:
        """
        레지스트리에서 모델을 삭제한다. 실제 파일은 삭제하지 않는다.

        Parameters
        ----------
        model_id:
            삭제할 모델 ID.

        Raises
        ------
        KeyError
            model_id 가 존재하지 않는 경우.
        """
        if model_id not in self.models:
            raise KeyError(f"모델을 찾을 수 없음: model_id={model_id}")
        name = self.models[model_id].model_name
        del self.models[model_id]
        logger.info("모델 삭제: model_id=%s, name=%s", model_id, name)

    # ------------------------------------------------------------------
    # 검색
    # ------------------------------------------------------------------

    def search(self, query: str) -> List[ModelMetadata]:
        """
        이름, 태그, 태스크 타입을 기준으로 모델을 검색한다.

        대소문자를 구분하지 않는 부분 일치 검색이다.

        Parameters
        ----------
        query:
            검색어.

        Returns
        -------
        List[ModelMetadata]
            검색 결과 ModelMetadata 목록.
        """
        q = query.lower()
        result: List[ModelMetadata] = []
        for meta in self.models.values():
            if (
                q in meta.model_name.lower()
                or q in meta.task_type.lower()
                or any(q in tag.lower() for tag in meta.tags)
                or q in meta.notes.lower()
            ):
                result.append(meta)
        logger.debug("모델 검색: query=%r, hits=%d", query, len(result))
        return result

    # ------------------------------------------------------------------
    # 직렬화 / 역직렬화
    # ------------------------------------------------------------------

    def export_registry(self, path: str) -> None:
        """
        레지스트리를 JSON 파일로 내보낸다.

        Parameters
        ----------
        path:
            저장할 JSON 파일 경로.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload: dict = {}
        for model_id, meta in self.models.items():
            d = meta.to_dict()
            d["_model_id"] = model_id          # 레지스트리 키 보존
            payload[model_id] = d

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info("모델 레지스트리 내보내기: path=%s, count=%d", path, len(self.models))

    def import_registry(self, path: str) -> None:
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

        self.models = {}
        for model_id, d in payload.items():
            # class_map 키가 문자열로 저장될 수 있으므로 int 로 변환
            raw_class_map = d.get("class_map", {})
            class_map = {int(k): v for k, v in raw_class_map.items()}
            meta = ModelMetadata(
                model_name=d.get("model_name", ""),
                version=d.get("version", "1.0.0"),
                task_type=d.get("task_type", "classification"),
                framework=d.get("framework", "onnx"),
                input_spec=d.get("input_spec", {}),
                class_map=class_map,
                training_date=d.get("training_date", ""),
                metrics=d.get("metrics", {}),
                file_path=d.get("file_path", ""),
                status=d.get("status", "draft"),
                tags=d.get("tags", []),
                notes=d.get("notes", ""),
            )
            self.models[model_id] = meta

        logger.info("모델 레지스트리 불러오기: path=%s, count=%d", path, len(self.models))

    # ------------------------------------------------------------------
    # 내부 유틸리티
    # ------------------------------------------------------------------

    def _get_meta(self, model_id: str) -> ModelMetadata:
        """model_id 로 ModelMetadata 를 가져온다. 없으면 KeyError."""
        meta = self.models.get(model_id)
        if meta is None:
            raise KeyError(f"모델을 찾을 수 없음: model_id={model_id}")
        return meta
