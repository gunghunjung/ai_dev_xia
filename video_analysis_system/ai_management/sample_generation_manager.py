"""
ai_management/sample_generation_manager.py — 샘플 생성 및 내보내기 모듈.

OpenCV VideoCapture 를 사용해 동영상에서 ROI 크롭 이미지 또는
시계열 시퀀스(numpy 배열)를 추출하여 학습 샘플로 저장한다.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SampleGenerationManager:
    """
    동영상 파일에서 학습용 샘플을 생성·저장하는 클래스.

    주요 기능:
    - ROI 크롭 이미지를 지정된 FPS 로 추출
    - 시계열 윈도우 시퀀스를 numpy 배열로 저장
    - 다양한 포맷으로 샘플 내보내기
    - 생성 통계 조회
    """

    def __init__(self) -> None:
        """SampleGenerationManager 초기화."""
        logger.debug("SampleGenerationManager 초기화 완료.")

    # ------------------------------------------------------------------
    # ROI 크롭 이미지 추출
    # ------------------------------------------------------------------

    def generate_roi_crops(
        self,
        video_path: str,
        roi_rect: Tuple[int, int, int, int],
        output_dir: str,
        label: str,
        fps: float = 1.0,
    ) -> int:
        """
        동영상에서 ROI 영역을 크롭한 이미지 샘플을 저장한다.

        Parameters
        ----------
        video_path:
            입력 동영상 파일 경로.
        roi_rect:
            크롭 영역 (x, y, w, h) — 픽셀 좌표.
        output_dir:
            크롭 이미지를 저장할 디렉터리. 없으면 자동 생성.
        label:
            샘플 레이블 (하위 폴더명으로 사용).
        fps:
            초당 추출 프레임 수. 기본값 1.0 (1초에 1프레임).

        Returns
        -------
        int
            실제로 저장된 샘플 이미지 수.

        Raises
        ------
        ImportError
            opencv-python 이 설치되어 있지 않은 경우.
        FileNotFoundError
            video_path 가 존재하지 않는 경우.
        ValueError
            roi_rect 가 유효하지 않거나 fps <= 0 인 경우.
        """
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "opencv-python 이 필요합니다. `pip install opencv-python` 으로 설치하세요."
            ) from exc

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"동영상 파일을 찾을 수 없음: {video_path}")

        x, y, w, h = roi_rect
        if w <= 0 or h <= 0:
            raise ValueError(f"ROI 크기가 유효하지 않음: w={w}, h={h}")
        if fps <= 0:
            raise ValueError(f"fps 는 양수여야 합니다: fps={fps}")

        # 레이블 하위 디렉터리 생성
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"동영상을 열 수 없음: {video_path}")

        video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        # 추출 간격 (프레임 단위)
        frame_interval = max(1, int(round(video_fps / fps)))

        saved = 0
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    fh, fw = frame.shape[:2]
                    # 경계 클램핑
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(fw, x + w)
                    y2 = min(fh, y + h)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        logger.warning(
                            "프레임 %d: ROI 크롭 결과가 비어있음 — 건너뜀", frame_idx
                        )
                        frame_idx += 1
                        continue

                    filename = f"{label}_{frame_idx:06d}.jpg"
                    save_path = os.path.join(label_dir, filename)
                    cv2.imwrite(save_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved += 1

                frame_idx += 1
        finally:
            cap.release()

        logger.info(
            "ROI 크롭 완료: video=%s, label=%s, saved=%d, output=%s",
            os.path.basename(video_path),
            label,
            saved,
            label_dir,
        )
        return saved

    # ------------------------------------------------------------------
    # 시계열 시퀀스 추출
    # ------------------------------------------------------------------

    def generate_temporal_sequence(
        self,
        video_path: str,
        roi_rect: Tuple[int, int, int, int],
        output_dir: str,
        window_size: int = 16,
        stride: int = 8,
    ) -> int:
        """
        동영상 ROI 에서 슬라이딩 윈도우 방식으로 시계열 시퀀스를 추출해 npy 로 저장한다.

        각 시퀀스는 shape (window_size, H, W, C) 의 numpy uint8 배열이다.

        Parameters
        ----------
        video_path:
            입력 동영상 파일 경로.
        roi_rect:
            크롭 영역 (x, y, w, h).
        output_dir:
            npy 파일을 저장할 디렉터리.
        window_size:
            시퀀스 길이 (프레임 수). 기본값 16.
        stride:
            슬라이딩 스트라이드 (프레임 수). 기본값 8.

        Returns
        -------
        int
            저장된 시퀀스 파일 수.

        Raises
        ------
        ImportError
            opencv-python 이 설치되어 있지 않은 경우.
        FileNotFoundError
            video_path 가 존재하지 않는 경우.
        """
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "opencv-python 이 필요합니다. `pip install opencv-python` 으로 설치하세요."
            ) from exc

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"동영상 파일을 찾을 수 없음: {video_path}")

        os.makedirs(output_dir, exist_ok=True)

        x, y, w, h = roi_rect
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"동영상을 열 수 없음: {video_path}")

        # 모든 크롭 프레임을 메모리에 수집
        crops: list[np.ndarray] = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                fh, fw = frame.shape[:2]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(fw, x + w), min(fh, y + h)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)
        finally:
            cap.release()

        if len(crops) < window_size:
            logger.warning(
                "동영상 프레임 수(%d) 가 window_size(%d) 보다 작음 — 시퀀스 생성 불가",
                len(crops),
                window_size,
            )
            return 0

        # 첫 번째 크롭의 크기를 기준으로 나머지를 리사이즈
        ref_h, ref_w = crops[0].shape[:2]
        normalized: list[np.ndarray] = []
        for c in crops:
            if c.shape[:2] != (ref_h, ref_w):
                import cv2 as _cv2  # noqa: F811
                c = _cv2.resize(c, (ref_w, ref_h))
            normalized.append(c)

        saved = 0
        start = 0
        while start + window_size <= len(normalized):
            seq = np.stack(normalized[start : start + window_size], axis=0)  # (T, H, W, C)
            filename = f"seq_{start:06d}_{start + window_size - 1:06d}.npy"
            np.save(os.path.join(output_dir, filename), seq)
            saved += 1
            start += stride

        logger.info(
            "시계열 시퀀스 생성 완료: video=%s, sequences=%d, output=%s",
            os.path.basename(video_path),
            saved,
            output_dir,
        )
        return saved

    # ------------------------------------------------------------------
    # 샘플 포맷 변환
    # ------------------------------------------------------------------

    def export_samples(
        self,
        input_dir: str,
        output_dir: str,
        format: str = "jpg",  # noqa: A002
    ) -> int:
        """
        input_dir 의 이미지 파일을 지정된 포맷으로 output_dir 에 재저장한다.

        Parameters
        ----------
        input_dir:
            원본 이미지 디렉터리 (재귀 탐색).
        output_dir:
            변환된 이미지를 저장할 디렉터리.
        format:
            출력 포맷 확장자. 예: ``"jpg"``, ``"png"``, ``"bmp"``.

        Returns
        -------
        int
            변환·저장된 파일 수.

        Raises
        ------
        ImportError
            opencv-python 이 설치되어 있지 않은 경우.
        """
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "opencv-python 이 필요합니다. `pip install opencv-python` 으로 설치하세요."
            ) from exc

        os.makedirs(output_dir, exist_ok=True)
        fmt = format.lower().lstrip(".")

        pattern_list = [
            "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff",
        ]
        src_files: list[str] = []
        for pat in pattern_list:
            src_files.extend(glob.glob(os.path.join(input_dir, "**", pat), recursive=True))

        converted = 0
        for src_path in src_files:
            img = cv2.imread(src_path)
            if img is None:
                logger.warning("이미지 로드 실패, 건너뜀: %s", src_path)
                continue

            # 상대 경로 유지
            rel = os.path.relpath(src_path, input_dir)
            base = os.path.splitext(rel)[0]
            dst_path = os.path.join(output_dir, f"{base}.{fmt}")
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            cv2.imwrite(dst_path, img)
            converted += 1

        logger.info(
            "샘플 내보내기 완료: input=%s, output=%s, format=%s, count=%d",
            input_dir,
            output_dir,
            fmt,
            converted,
        )
        return converted

    # ------------------------------------------------------------------
    # 통계 조회
    # ------------------------------------------------------------------

    def get_generation_stats(self, output_dir: str) -> Dict[str, object]:
        """
        output_dir 내 생성된 샘플 파일 통계를 반환한다.

        Parameters
        ----------
        output_dir:
            통계를 조회할 디렉터리.

        Returns
        -------
        dict
            ``total_files``, ``total_bytes``, ``by_extension``, ``by_label`` 를 포함하는 딕셔너리.
        """
        if not os.path.isdir(output_dir):
            return {
                "total_files": 0,
                "total_bytes": 0,
                "by_extension": {},
                "by_label": {},
            }

        total_files = 0
        total_bytes = 0
        by_extension: Dict[str, int] = {}
        by_label: Dict[str, int] = {}

        for root, _dirs, files in os.walk(output_dir):
            label = os.path.relpath(root, output_dir)
            for fname in files:
                fpath = os.path.join(root, fname)
                ext = os.path.splitext(fname)[1].lower()
                try:
                    size = os.path.getsize(fpath)
                except OSError:
                    size = 0
                total_files += 1
                total_bytes += size
                by_extension[ext] = by_extension.get(ext, 0) + 1
                by_label[label] = by_label.get(label, 0) + 1

        stats: Dict[str, object] = {
            "total_files": total_files,
            "total_bytes": total_bytes,
            "by_extension": by_extension,
            "by_label": by_label,
        }
        logger.debug("생성 통계: %s", stats)
        return stats
