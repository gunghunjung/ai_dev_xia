"""
ai_management/deployment_preparation_manager.py — 모델 배포 준비 모듈.

DeploymentPreparationManager 는 학습된 모델을 ONNX / TorchScript 로 내보내고,
런타임 메타데이터를 생성하며, 사전 배포 검증 체크리스트를 제공한다.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple

from core.data_models import ModelMetadata

logger = logging.getLogger(__name__)


class DeploymentPreparationManager:
    """
    모델 배포 준비를 담당하는 클래스.

    - PyTorch 모델 → ONNX 내보내기
    - PyTorch 모델 → TorchScript 내보내기
    - ONNX 모델 실행 검증 (onnxruntime)
    - 런타임 메타데이터 JSON 생성
    - 배포 전 스펙 체크리스트
    """

    def __init__(self) -> None:
        """DeploymentPreparationManager 초기화."""
        logger.debug("DeploymentPreparationManager 초기화 완료.")

    # ------------------------------------------------------------------
    # ONNX 내보내기
    # ------------------------------------------------------------------

    def export_onnx(
        self,
        model_path: str,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> str:
        """
        PyTorch 체크포인트(.pt / .pth)를 ONNX 파일로 내보낸다.

        Parameters
        ----------
        model_path:
            PyTorch 모델 파일 경로 (.pt / .pth).
        output_path:
            저장할 ONNX 파일 경로.
        input_shape:
            더미 입력 텐서 shape. 기본값 (1, 3, 224, 224).

        Returns
        -------
        str
            저장된 ONNX 파일의 절대 경로.

        Raises
        ------
        ValueError
            torch 가 설치되어 있지 않은 경우.
        FileNotFoundError
            model_path 가 존재하지 않는 경우.
        """
        try:
            import torch  # type: ignore
        except ImportError:
            raise ValueError(
                "PyTorch 가 필요합니다. `pip install torch` 로 설치하세요. "
                "ONNX 내보내기는 torch.onnx.export 를 사용합니다."
            )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없음: {model_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        logger.info(
            "ONNX 내보내기 시작: model=%s, output=%s, input_shape=%s",
            os.path.basename(model_path),
            output_path,
            input_shape,
        )

        model = torch.load(model_path, map_location="cpu")
        model.eval()

        dummy_input = torch.randn(*input_shape)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            verbose=False,
        )

        abs_path = os.path.abspath(output_path)
        logger.info("ONNX 내보내기 완료: path=%s", abs_path)
        return abs_path

    # ------------------------------------------------------------------
    # TorchScript 내보내기
    # ------------------------------------------------------------------

    def export_torchscript(
        self,
        model_path: str,
        output_path: str,
    ) -> str:
        """
        PyTorch 체크포인트를 TorchScript(.pt) 로 내보낸다.

        Parameters
        ----------
        model_path:
            PyTorch 모델 파일 경로.
        output_path:
            저장할 TorchScript 파일 경로.

        Returns
        -------
        str
            저장된 TorchScript 파일의 절대 경로.

        Raises
        ------
        ValueError
            torch 가 설치되어 있지 않은 경우.
        FileNotFoundError
            model_path 가 존재하지 않는 경우.
        """
        try:
            import torch  # type: ignore
        except ImportError:
            raise ValueError(
                "PyTorch 가 필요합니다. `pip install torch` 로 설치하세요."
            )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없음: {model_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        logger.info(
            "TorchScript 내보내기 시작: model=%s, output=%s",
            os.path.basename(model_path),
            output_path,
        )

        model = torch.load(model_path, map_location="cpu")
        model.eval()

        scripted = torch.jit.script(model)
        scripted.save(output_path)

        abs_path = os.path.abspath(output_path)
        logger.info("TorchScript 내보내기 완료: path=%s", abs_path)
        return abs_path

    # ------------------------------------------------------------------
    # ONNX 검증
    # ------------------------------------------------------------------

    def validate_onnx(
        self,
        onnx_path: str,
        test_input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> dict:
        """
        onnxruntime 으로 ONNX 모델을 실행하여 기본 검증을 수행한다.

        Parameters
        ----------
        onnx_path:
            검증할 ONNX 파일 경로.
        test_input_shape:
            테스트용 랜덤 입력 텐서 shape.

        Returns
        -------
        dict
            ``valid``, ``output_shapes``, ``latency_ms``, ``message`` 를 포함하는 딕셔너리.

        Raises
        ------
        ImportError
            onnxruntime 이 설치되어 있지 않은 경우.
        FileNotFoundError
            onnx_path 가 존재하지 않는 경우.
        """
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            raise ImportError(
                "onnxruntime 이 필요합니다. `pip install onnxruntime` 으로 설치하세요."
            )

        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX 파일을 찾을 수 없음: {onnx_path}")

        try:
            import numpy as np

            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name

            dummy = np.random.rand(*test_input_shape).astype(np.float32)

            start = time.perf_counter()
            outputs = session.run(None, {input_name: dummy})
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            output_shapes = [list(o.shape) for o in outputs]

            result: dict = {
                "valid": True,
                "output_shapes": output_shapes,
                "latency_ms": round(elapsed_ms, 2),
                "message": "ONNX 모델 검증 성공.",
            }
            logger.info(
                "ONNX 검증 성공: path=%s, latency=%.2fms, output_shapes=%s",
                onnx_path, elapsed_ms, output_shapes,
            )
        except Exception as exc:  # noqa: BLE001
            result = {
                "valid": False,
                "output_shapes": [],
                "latency_ms": 0.0,
                "message": f"ONNX 검증 실패: {exc}",
            }
            logger.error("ONNX 검증 실패: path=%s, error=%s", onnx_path, exc)

        return result

    # ------------------------------------------------------------------
    # 런타임 메타데이터
    # ------------------------------------------------------------------

    def prepare_runtime_metadata(
        self,
        model_path: str,
        meta: ModelMetadata,
        output_dir: str,
    ) -> str:
        """
        런타임에서 사용하는 모델 메타데이터 JSON 파일을 생성한다.

        Parameters
        ----------
        model_path:
            배포할 모델 파일 경로 (ONNX / TorchScript 등).
        meta:
            ModelMetadata 인스턴스.
        output_dir:
            메타데이터 파일을 저장할 디렉터리.

        Returns
        -------
        str
            저장된 JSON 파일의 절대 경로.
        """
        os.makedirs(output_dir, exist_ok=True)

        runtime_meta: Dict[str, Any] = {
            "model_name": meta.model_name,
            "version": meta.version,
            "task_type": meta.task_type,
            "framework": meta.framework,
            "model_path": os.path.abspath(model_path),
            "input_spec": meta.input_spec,
            "class_map": {str(k): v for k, v in meta.class_map.items()},
            "num_classes": len(meta.class_map),
            "training_date": meta.training_date,
            "metrics": meta.metrics,
            "status": meta.status,
            "tags": meta.tags,
            "notes": meta.notes,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        safe_name = meta.model_name.replace(" ", "_").replace("/", "_")
        filename = f"{safe_name}_v{meta.version}_runtime_meta.json"
        output_path = os.path.join(output_dir, filename)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(runtime_meta, fh, indent=2, ensure_ascii=False)

        abs_path = os.path.abspath(output_path)
        logger.info("런타임 메타데이터 생성: path=%s", abs_path)
        return abs_path

    # ------------------------------------------------------------------
    # 배포 전 스펙 체크
    # ------------------------------------------------------------------

    def check_model_spec(self, meta: ModelMetadata) -> List[str]:
        """
        모델 배포 전 필수/권고 스펙을 검사하고 경고 메시지 목록을 반환한다.

        경고가 없으면 빈 리스트를 반환한다.

        Parameters
        ----------
        meta:
            검사할 ModelMetadata 인스턴스.

        Returns
        -------
        List[str]
            경고 메시지 목록. 비어 있으면 모든 체크 통과.
        """
        warnings: List[str] = []

        # 필수 필드
        if not meta.model_name:
            warnings.append("[ERROR] model_name 이 비어 있습니다.")
        if not meta.file_path:
            warnings.append("[ERROR] file_path 가 설정되지 않았습니다.")
        elif not os.path.isfile(meta.file_path):
            warnings.append(f"[ERROR] 모델 파일이 존재하지 않습니다: {meta.file_path}")

        # 권고 사항
        if not meta.class_map:
            warnings.append("[WARN] class_map 이 비어 있습니다. 클래스 이름을 등록하세요.")
        if not meta.input_spec:
            warnings.append("[WARN] input_spec 이 비어 있습니다. 입력 shape/dtype 을 명시하세요.")
        if not meta.training_date:
            warnings.append("[WARN] training_date 가 설정되지 않았습니다.")
        if meta.status not in ("validated", "production"):
            warnings.append(
                f"[WARN] 모델 status 가 '{meta.status}' 입니다. "
                "배포 전 'validated' 또는 'production' 으로 변경하세요."
            )
        accuracy = meta.metrics.get("accuracy", None)
        if accuracy is not None and accuracy < 0.7:
            warnings.append(
                f"[WARN] accuracy ({accuracy:.3f}) 가 0.7 미만입니다. "
                "충분한 성능인지 확인하세요."
            )
        if not meta.version or meta.version == "1.0.0":
            warnings.append("[INFO] version 이 기본값(1.0.0) 입니다. 버전을 명시적으로 관리하세요.")
        if not meta.tags:
            warnings.append("[INFO] tags 가 비어 있습니다. 태그를 추가하면 검색이 쉬워집니다.")

        if warnings:
            logger.warning(
                "모델 스펙 체크 경고 %d건: model=%s", len(warnings), meta.model_name
            )
        else:
            logger.info("모델 스펙 체크 통과: model=%s", meta.model_name)

        return warnings
