# features/ts_features.py — ROI 세그먼트 시계열 피처 추출
# 핵심: 모든 피처는 세그먼트 내부 과거 데이터만 사용
from __future__ import annotations
import logging
import numpy as np

logger = logging.getLogger("quant.ts")


class TSFeatureExtractor:
    """
    ROI 세그먼트에서 정량적 시계열 피처 추출

    피처 카테고리:
    1. 가격 피처: 수익률, 모멘텀, 추세
    2. 변동성 피처: 실현 변동성, ATR, 범위
    3. 거래량 피처: 거래량 비율, OBV 변화
    4. 롤링 통계: 평균, 표준편차, 왜도, 첨도
    5. 상대 강도: RSI-like
    """

    def __init__(self, n_features: int = 32):
        self.n_features = n_features

    def transform(self, segments: np.ndarray) -> np.ndarray:
        """
        Args:
            segments: (N, T, 5) OHLCV 세그먼트
        Returns:
            features: (N, n_features) float32
        """
        if len(segments) == 0:
            return np.zeros((0, self.n_features), dtype=np.float32)

        feats = [self._extract_one(seg) for seg in segments]
        result = np.stack(feats).astype(np.float32)

        # NaN/Inf 처리 — 발생 시 로그 경고
        nan_count = np.isnan(result).sum()
        inf_count = np.isinf(result).sum()
        if nan_count > 0 or inf_count > 0:
            logger.debug(
                f"TS피처 이상값 감지: NaN={nan_count}, Inf={inf_count} "
                f"(총 {result.size}개 중) → 0으로 대체"
            )
        result = np.nan_to_num(result, nan=0.0, posinf=3.0, neginf=-3.0)
        return result

    def _extract_one(self, seg: np.ndarray) -> np.ndarray:
        """단일 세그먼트 피처 벡터 추출"""
        open_ = seg[:, 0]
        high = seg[:, 1]
        low = seg[:, 2]
        close = seg[:, 3]
        volume = seg[:, 4]
        T = len(close)

        feats = []

        # ── 1. 수익률/모멘텀 피처 ──────────────────────
        ret = np.diff(close) / (np.abs(close[:-1]) + 1e-10)

        feats.append(close[-1] - close[0])            # 총 수익률 (정규화 스케일)
        feats.append(ret.mean())                       # 평균 일간 수익률
        feats.append(ret.std() if len(ret) > 0 else 0.0)  # 수익률 표준편차
        feats.append(ret[-1] if len(ret) > 0 else 0.0)  # 마지막 일간 수익률

        # 모멘텀 (반쪽 기간 비교)
        half = T // 2
        if half > 0:
            mom1 = close[-1] - close[-half]
            mom2 = close[-half] - close[0]
            feats.append(mom1)
            feats.append(mom2)
            feats.append(mom1 - mom2)  # 모멘텀 가속도
        else:
            feats.extend([0.0, 0.0, 0.0])

        # ── 2. 변동성 피처 ────────────────────────────
        feats.append(ret.std() * np.sqrt(252) if len(ret) > 0 else 0.0)  # 연간화 변동성
        feats.append(np.max(np.abs(ret)) if len(ret) > 0 else 0.0)     # 최대 절대 수익률

        # ATR (Average True Range)
        hl = high - low
        hc = np.abs(high[1:] - close[:-1])
        lc = np.abs(low[1:] - close[:-1])
        if len(hc) > 0:
            tr = np.maximum(hl[1:], np.maximum(hc, lc))
            feats.append(tr.mean())
            feats.append(tr[-1] if len(tr) > 0 else 0)
        else:
            feats.extend([hl.mean(), hl[-1]])

        # 가격 범위
        price_range = (high.max() - low.min()) / (np.abs(close[0]) + 1e-10)
        feats.append(price_range)

        # ── 3. 거래량 피처 ────────────────────────────
        if volume.std() > 1e-10:
            feats.append(volume[-1] - volume.mean())    # 최근 거래량 vs 평균
            feats.append(volume[-3:].mean() - volume[:3].mean()
                         if T >= 6 else 0.0)            # 후반 vs 초반 거래량
        else:
            feats.extend([0.0, 0.0])

        # OBV (On-Balance Volume) 변화율
        obv = self._obv(close, volume)
        obv_norm = (obv - obv.mean()) / (obv.std() + 1e-10)
        feats.append(obv_norm[-1] if len(obv_norm) > 0 else 0)

        # ── 4. 롤링 통계 ──────────────────────────────
        # 왜도 (분포 비대칭)
        if ret.std() > 1e-10:
            skew = np.mean(((ret - ret.mean()) / ret.std()) ** 3)
        else:
            skew = 0.0
        feats.append(skew)

        # 첨도 (꼬리 두께)
        if ret.std() > 1e-10:
            kurt = np.mean(((ret - ret.mean()) / ret.std()) ** 4) - 3
        else:
            kurt = 0.0
        feats.append(kurt)

        # ── 5. RSI-like 상대 강도 ─────────────────────
        rsi_val = self._rsi(close, period=min(14, T-1))
        feats.append(rsi_val)

        # ── 6. 캔들 패턴 피처 ─────────────────────────
        # 상승/하락 캔들 비율
        bull_ratio = np.sum(close > open_) / max(T, 1)
        feats.append(bull_ratio)

        # 도지 비율 (몸통 < 0.1 × 범위)
        body = np.abs(close - open_)
        range_ = high - low + 1e-10
        doji_ratio = np.mean(body / range_ < 0.1)
        feats.append(doji_ratio)

        # 마지막 캔들 특성
        last_body = (close[-1] - open_[-1])
        last_range = (high[-1] - low[-1] + 1e-10)
        feats.append(last_body / last_range)

        # ── 7. 추세 피처 ──────────────────────────────
        # 선형 회귀 기울기 (Close)
        t = np.arange(T, dtype=float)
        if T > 1:
            slope = np.polyfit(t, close, 1)[0]
        else:
            slope = 0.0
        feats.append(slope)

        # R-squared (추세 선명도)
        if T > 2:
            close_fit = np.polyval(np.polyfit(t, close, 1), t)
            ss_res = np.sum((close - close_fit) ** 2)
            ss_tot = np.sum((close - close.mean()) ** 2) + 1e-10
            r2 = max(0, 1 - ss_res / ss_tot)
        else:
            r2 = 0.0
        feats.append(r2)

        # 세그먼트 내 최고점/최저점 위치 (0=처음, 1=끝)
        feats.append(np.argmax(high) / max(T-1, 1))
        feats.append(np.argmin(low) / max(T-1, 1))

        # ── 8. 추가 시장 구조 피처 (zero-padding 제거 → 의미 있는 7개) ─────
        # ① 가격 채널 내 위치: (Close - Low_min) / (High_max - Low_min)
        #    0 = 구간 최저, 1 = 구간 최고 → 현재 상대적 위치
        price_pos = ((close[-1] - low.min())
                     / (high.max() - low.min() + 1e-10))
        feats.append(price_pos)

        # ② 연속 상승/하락 캔들 수 (마지막 시점 기준, 부호 포함)
        #    +k = 마지막 k개 연속 상승, -k = 마지막 k개 연속 하락
        streak = 0
        if len(close) >= 2:
            direction = 1 if close[-1] > close[-2] else -1
            for j in range(len(close) - 1, 0, -1):
                curr_dir = 1 if close[j] > close[j-1] else -1
                if curr_dir == direction:
                    streak += direction
                else:
                    break
        feats.append(float(streak) / max(T, 1))   # T로 정규화

        # ③ 거래량 가중 평균 가격 대비 현재 Close 위치
        #    VWAP-like: Σ(close×volume) / Σ(volume)
        vol_abs = np.abs(volume)
        vwap = (np.sum(close * vol_abs) / (np.sum(vol_abs) + 1e-10))
        feats.append((close[-1] - vwap) / (vwap + 1e-10))

        # ④ 일중 변동성 비율: ATR / (고가 - 저가) 평균 비율
        #    값이 클수록 갭(gap) 이 많음을 의미
        hl_avg = (high - low).mean() + 1e-10
        if len(hc) > 0:
            tr_mean_val = np.maximum(hl[1:], np.maximum(hc, lc)).mean()
        else:
            tr_mean_val = hl_avg
        feats.append(tr_mean_val / hl_avg)

        # ⑤ 단기 모멘텀 가속도: 최근 1/4 구간 수익률 - 직전 1/4 구간 수익률
        q = max(1, T // 4)
        mom_recent = close[-1] - close[-q-1] if len(close) > q else 0.0
        mom_prev   = close[-q-1] - close[-2*q-1] if len(close) > 2*q else 0.0
        feats.append(mom_recent - mom_prev)

        # ⑥ 변동성 레짐: 후반부 변동성 / 전반부 변동성
        #    > 1 이면 최근 변동성이 증가 (불안정), < 1 이면 안정화
        half = max(1, T // 2)
        vol_recent = np.std(close[-half:]) + 1e-10
        vol_early  = np.std(close[:half])  + 1e-10
        feats.append(vol_recent / vol_early)

        # ⑦ 음영비율(Shadow ratio): 몸통 대비 위/아래 꼬리 총합
        #    크면 불확실성 높음 (양방향 탐색 활발)
        body_total   = np.abs(close - open_).mean() + 1e-10
        upper_shadow = (high - np.maximum(close, open_)).mean()
        lower_shadow = (np.minimum(close, open_) - low).mean()
        feats.append((upper_shadow + lower_shadow) / body_total)

        # ── 피처 벡터 정렬 ────────────────────────────
        feat_arr = np.array(feats, dtype=np.float64)

        # 목표 차원에 맞게 패딩/자르기
        if len(feat_arr) < self.n_features:
            feat_arr = np.pad(feat_arr, (0, self.n_features - len(feat_arr)))
        else:
            feat_arr = feat_arr[:self.n_features]

        return feat_arr

    # ──────────────────────────────────────────────────
    # 보조 함수
    # ──────────────────────────────────────────────────

    @staticmethod
    def _obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """On-Balance Volume — 벡터화 구현 (Python loop 제거)"""
        if len(close) < 2:
            return np.zeros_like(close)
        direction = np.sign(np.diff(close))          # +1, 0, -1
        signed_vol = direction * volume[1:]          # 방향 반영 거래량
        obv = np.empty_like(close)
        obv[0] = 0.0
        obv[1:] = np.cumsum(signed_vol)
        return obv

    @staticmethod
    def _rsi(close: np.ndarray, period: int = 14) -> float:
        """RSI 마지막 값"""
        if len(close) < period + 1:
            return 50.0
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = gain[-period:].mean()
        avg_loss = loss[-period:].mean()
        if avg_loss < 1e-10:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)
