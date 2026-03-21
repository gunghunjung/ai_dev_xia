# features/cv_features.py — ROI → 이미지 변환 (GAF / 캔들스틱)
from __future__ import annotations
import logging
from typing import Tuple
import numpy as np

logger = logging.getLogger("quant.cv")


class CVFeatureExtractor:
    """
    ROI 세그먼트를 이미지로 변환
    지원 방식:
    1. GAF(Gramian Angular Field): 시계열 → 2D 상관 행렬 이미지
       - 각도로 변환 후 내적(GASF) 또는 차(GADF)
    2. 캔들스틱: OHLC → 미니 캔들 이미지
    3. Recurrence Plot: 거리 행렬 기반
    """

    def __init__(self, image_size: int = 64, method: str = "gasf"):
        """
        Args:
            image_size: 출력 이미지 크기 (image_size × image_size)
            method: "gasf" / "gadf" / "combined" / "candle"
        """
        self.image_size = image_size
        self.method = method

    def transform(self, segments: np.ndarray) -> np.ndarray:
        """
        Args:
            segments: (N, T, 5) OHLCV 정규화 세그먼트
        Returns:
            images: (N, C, image_size, image_size) float32
        """
        if len(segments) == 0:
            return np.zeros((0, self._channels(), self.image_size, self.image_size),
                            dtype=np.float32)

        images = []
        for seg in segments:
            img = self._segment_to_image(seg)
            images.append(img)

        return np.stack(images).astype(np.float32)

    def _channels(self) -> int:
        if self.method == "combined":
            return 2  # GASF + GADF
        elif self.method == "candle":
            return 3  # RGB
        else:
            return 1  # GASF 또는 GADF 단일

    def _segment_to_image(self, seg: np.ndarray) -> np.ndarray:
        """단일 세그먼트 → 이미지 (C, H, W)"""
        close = seg[:, 3]  # Close 컬럼
        T = len(close)

        if self.method == "gasf":
            img = self._gasf(close, self.image_size)
            return img[np.newaxis]  # (1, H, W)
        elif self.method == "gadf":
            img = self._gadf(close, self.image_size)
            return img[np.newaxis]
        elif self.method == "combined":
            gasf = self._gasf(close, self.image_size)
            gadf = self._gadf(close, self.image_size)
            return np.stack([gasf, gadf])  # (2, H, W)
        elif self.method == "candle":
            return self._candle_image(seg, self.image_size)  # (3, H, W)
        else:
            img = self._gasf(close, self.image_size)
            return img[np.newaxis]

    # ──────────────────────────────────────────────────
    # GAF (Gramian Angular Field)
    # ──────────────────────────────────────────────────

    def _normalize_minmax(self, x: np.ndarray) -> np.ndarray:
        """[-1, 1] 정규화"""
        xmin, xmax = x.min(), x.max()
        if xmax - xmin < 1e-10:
            return np.zeros_like(x)
        return 2 * (x - xmin) / (xmax - xmin) - 1

    def _to_angular(self, x: np.ndarray) -> np.ndarray:
        """정규화 시계열 → 각도 (arccos)"""
        x = np.clip(self._normalize_minmax(x), -1, 1)
        return np.arccos(x)

    def _resize_1d_to_n(self, x: np.ndarray, n: int) -> np.ndarray:
        """1D 시계열을 n 포인트로 리샘플링"""
        from scipy.interpolate import interp1d
        try:
            t_old = np.linspace(0, 1, len(x))
            t_new = np.linspace(0, 1, n)
            f = interp1d(t_old, x, kind="linear")
            return f(t_new)
        except Exception:
            if len(x) >= n:
                indices = np.linspace(0, len(x)-1, n, dtype=int)
                return x[indices]
            else:
                return np.resize(x, n)

    def _gasf(self, x: np.ndarray, size: int) -> np.ndarray:
        """Gramian Angular Summation Field"""
        x_r = self._resize_1d_to_n(x, size)
        phi = self._to_angular(x_r)
        # GASF: cos(phi_i + phi_j)
        phi_i = phi[:, np.newaxis]
        phi_j = phi[np.newaxis, :]
        gasf = np.cos(phi_i + phi_j)
        return gasf.astype(np.float32)

    def _gadf(self, x: np.ndarray, size: int) -> np.ndarray:
        """Gramian Angular Difference Field"""
        x_r = self._resize_1d_to_n(x, size)
        phi = self._to_angular(x_r)
        # GADF: sin(phi_i - phi_j)
        phi_i = phi[:, np.newaxis]
        phi_j = phi[np.newaxis, :]
        gadf = np.sin(phi_i - phi_j)
        return gadf.astype(np.float32)

    # ──────────────────────────────────────────────────
    # 캔들스틱 이미지
    # ──────────────────────────────────────────────────

    def _candle_image(self, seg: np.ndarray, size: int) -> np.ndarray:
        """
        OHLC → 3채널 캔들스틱 이미지
        Ch0: 종가 라인 (회색조)
        Ch1: 상승 캔들 마스크 (양봉)
        Ch2: 하락 캔들 마스크 (음봉)
        """
        T = len(seg)
        img = np.zeros((3, size, size), dtype=np.float32)

        if T < 2:
            return img

        open_ = seg[:, 0]
        high = seg[:, 1]
        low = seg[:, 2]
        close = seg[:, 3]

        # 전체 가격 범위로 정규화
        all_prices = np.concatenate([open_, high, low, close])
        pmin, pmax = all_prices.min(), all_prices.max()
        price_range = pmax - pmin + 1e-10

        def price_to_row(p):
            r = int((pmax - p) / price_range * (size - 1))
            return np.clip(r, 0, size - 1)

        # 각 캔들 그리기
        for i in range(T):
            col_start = int(i * size / T)
            col_end = max(col_start + 1, int((i+1) * size / T))
            col_center = (col_start + col_end) // 2

            # 고가-저가 선 (그림자)
            r_high = price_to_row(high[i])
            r_low = price_to_row(low[i])
            for r in range(r_high, r_low+1):
                img[0, r, col_center] = 0.5  # 그림자

            # 본체
            r_open = price_to_row(open_[i])
            r_close = price_to_row(close[i])
            r_top = min(r_open, r_close)
            r_bot = max(r_open, r_close)
            is_bull = close[i] >= open_[i]

            for r in range(r_top, r_bot+1):
                for c in range(col_start, col_end):
                    img[0, r, c] = 1.0 if is_bull else 0.3
                    if is_bull:
                        img[1, r, c] = 1.0  # 양봉
                    else:
                        img[2, r, c] = 1.0  # 음봉

        return img
