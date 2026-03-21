# features/roi.py — ROI(관심 구간) 감지 및 추출
# 핵심 원칙: 오직 과거 데이터만 사용 (미래 참조 절대 금지)
from __future__ import annotations
import logging
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("quant.roi")


class ROIDetector:
    """
    금융 시계열 ROI(Region of Interest) 감지기

    감지 조건 (OR 결합):
    1. 변동성 급증: 롤링 변동성 z-score > vol_z_threshold
    2. 가격 돌파: 가격 z-score > breakout_threshold (절댓값)
    3. 거래량 급증: 거래량 z-score > volume_spike_threshold

    각 ROI는 감지 시점 기준 fixed-length 세그먼트로 추출됨.
    레이블: lookahead일 후 수익률 (연속), 또는 방향 (이진)
    """

    def __init__(
        self,
        segment_length: int = 30,
        lookahead: int = 5,
        vol_z_threshold: float = 1.5,
        breakout_threshold: float = 2.0,
        volume_spike_threshold: float = 2.0,
        min_roi_spacing: int = 5,
        rolling_window: int = 20,
    ):
        self.segment_length = segment_length
        self.lookahead = lookahead
        self.vol_z_threshold = vol_z_threshold
        self.breakout_threshold = breakout_threshold
        self.volume_spike_threshold = volume_spike_threshold
        self.min_roi_spacing = min_roi_spacing
        self.rolling_window = rolling_window

    # ──────────────────────────────────────────────────
    # 메인 API
    # ──────────────────────────────────────────────────

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        전체 OHLCV 데이터프레임에서 ROI 감지
        Returns: ROI 인덱스 목록을 담은 DataFrame
            columns: ['date', 'idx', 'trigger', 'label_return', 'label_dir']
        """
        if len(df) < self.rolling_window + self.segment_length + self.lookahead:
            return pd.DataFrame()

        scores = self._compute_scores(df)
        roi_indices = self._find_roi_indices(scores, df)
        return self._build_roi_table(df, roi_indices, scores)

    def extract_segments(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
        """
        ROI 세그먼트 배열 추출
        Returns:
            segments: (N, segment_length, 5) — OHLCV 정규화 세그먼트
            labels:   (N,) — 레이블 수익률
            dates:    ROI 감지 날짜 목록
        """
        roi_table = self.detect(df)
        if roi_table.empty:
            return np.zeros((0, self.segment_length, 5)), np.zeros(0), []

        close = df["Close"].values
        open_ = df["Open"].values
        high = df["High"].values
        low = df["Low"].values
        volume = df["Volume"].values if "Volume" in df.columns else np.ones_like(close)

        segments, labels, dates = [], [], []
        for row in roi_table.itertuples():
            idx = row.idx
            start = idx - self.segment_length + 1
            if start < 0:
                continue

            # 미래 참조 없음: [start, idx] 구간만 사용
            seg = np.stack([
                open_[start:idx+1],
                high[start:idx+1],
                low[start:idx+1],
                close[start:idx+1],
                volume[start:idx+1],
            ], axis=1)  # (segment_length, 5)

            seg = self._normalize_segment(seg)
            segments.append(seg)
            labels.append(row.label_return)
            dates.append(row.date)

        if not segments:
            return np.zeros((0, self.segment_length, 5)), np.zeros(0), []

        return np.stack(segments), np.array(labels), dates

    # ──────────────────────────────────────────────────
    # 내부 메서드
    # ──────────────────────────────────────────────────

    def _compute_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """각 날짜의 ROI 트리거 점수 계산 (과거 데이터만 사용)"""
        close = df["Close"]
        volume = df["Volume"] if "Volume" in df.columns else pd.Series(
            np.ones(len(df)), index=df.index)

        # 1) 롤링 수익률 기반 변동성 z-score
        ret = close.pct_change()
        roll_vol = ret.rolling(self.rolling_window).std()
        vol_mean = roll_vol.rolling(self.rolling_window * 3).mean()
        vol_std = roll_vol.rolling(self.rolling_window * 3).std()
        vol_z = (roll_vol - vol_mean) / (vol_std + 1e-10)

        # 2) 가격 z-score (롤링 평균 대비)
        price_mean = close.rolling(self.rolling_window).mean()
        price_std = close.rolling(self.rolling_window).std()
        price_z = (close - price_mean) / (price_std + 1e-10)

        # 3) 거래량 z-score
        vol_log = np.log1p(volume)
        vol_roll_mean = vol_log.rolling(self.rolling_window).mean()
        vol_roll_std = vol_log.rolling(self.rolling_window).std()
        volume_z = (vol_log - vol_roll_mean) / (vol_roll_std + 1e-10)

        return pd.DataFrame({
            "vol_z": vol_z,
            "price_z": price_z.abs(),
            "volume_z": volume_z,
        }, index=df.index)

    def _find_roi_indices(self, scores: pd.DataFrame,
                          df: pd.DataFrame) -> List[int]:
        """트리거 조건 만족하는 인덱스 목록 (중복 제거)"""
        vol_trigger = scores["vol_z"] > self.vol_z_threshold
        price_trigger = scores["price_z"] > self.breakout_threshold
        volume_trigger = scores["volume_z"] > self.volume_spike_threshold

        is_roi = (vol_trigger | price_trigger | volume_trigger).values

        # 최솟값 간격 강제 (중복 방지)
        roi_indices = []
        last_idx = -self.min_roi_spacing - 1
        n = len(df)
        min_start = self.rolling_window + self.segment_length
        max_end = n - self.lookahead - 1

        for i in range(min_start, max_end):
            if is_roi[i] and (i - last_idx) >= self.min_roi_spacing:
                roi_indices.append(i)
                last_idx = i

        return roi_indices

    def _build_roi_table(self, df: pd.DataFrame,
                         roi_indices: List[int],
                         scores: pd.DataFrame) -> pd.DataFrame:
        """ROI 테이블 구성 (레이블 포함)"""
        if not roi_indices:
            return pd.DataFrame()

        close = df["Close"].values
        dates = df.index
        records = []

        for idx in roi_indices:
            # 레이블: lookahead일 후 수익률 (log return)
            future_price = close[idx + self.lookahead]
            current_price = close[idx]
            label_ret = np.log(future_price / current_price + 1e-10)
            label_dir = 1 if label_ret > 0 else 0

            # 가장 강한 트리거 이름
            row_scores = scores.iloc[idx]
            triggers = []
            if row_scores["vol_z"] > self.vol_z_threshold:
                triggers.append("변동성급증")
            if row_scores["price_z"] > self.breakout_threshold:
                triggers.append("가격돌파")
            if row_scores["volume_z"] > self.volume_spike_threshold:
                triggers.append("거래량급증")

            records.append({
                "date": dates[idx],
                "idx": idx,
                "trigger": "+".join(triggers) if triggers else "복합",
                "label_return": label_ret,
                "label_dir": label_dir,
                "vol_z": row_scores["vol_z"],
                "price_z": row_scores["price_z"],
                "volume_z": row_scores["volume_z"],
            })

        return pd.DataFrame(records)

    def _normalize_segment(self, seg: np.ndarray) -> np.ndarray:
        """
        세그먼트 정규화
        - OHLC: 첫 날 종가 대비 상대값 (수익률 스케일)
        - Volume: log 변환 후 z-score
        """
        result = seg.copy().astype(float)
        base_price = seg[0, 3] + 1e-10  # 첫 날 Close

        # OHLC 정규화 (대비 비율)
        result[:, :4] = (seg[:, :4] - base_price) / base_price

        # Volume 정규화
        vol = seg[:, 4]
        log_vol = np.log1p(vol)
        mu, sigma = log_vol.mean(), log_vol.std() + 1e-10
        result[:, 4] = (log_vol - mu) / sigma

        return result
