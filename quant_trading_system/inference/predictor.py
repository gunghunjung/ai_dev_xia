# inference/predictor.py — 프로덕션 등급 미래 예측 엔진
"""
======================================================================
InferencePredictor — 학습된 HybridModel을 이용한 미래 주가 방향 예측

핵심 원칙 (CRITICAL PRINCIPLES):
  1. 데이터 누수 ZERO — 예측 시점 t에서 t 이후 데이터 절대 사용 금지
  2. 학습-추론 파이프라인 일관성 — 정규화, 피처, 입력 스키마 완전 동일
  3. 체크포인트 config 우선 — 저장된 모델 아키텍처 그대로 복원
  4. 현실적 출력 — NaN/inf 검증, 비현실적 값 클리핑, 확률 기반 신호

파이프라인:
  ① load_model()         — 체크포인트에서 아키텍처+가중치 복원
  ② prepare_features()   — 최근 segment_length개 봉 → 학습과 동일한 정규화
  ③ predict_future()     — 모델 추론 → (μ, σ) → 확률·방향·신뢰도·행동
  ④ build_portfolio()    — 전 종목 순위화 → 포트폴리오 비중 산출
======================================================================
"""
from __future__ import annotations

import datetime
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import AppSettings
from data import DataLoader
from features import CVFeatureExtractor, TSFeatureExtractor
from models.hybrid import HybridModel
from models.store import ModelStore
from models.trainer import ModelTrainer

try:
    from scipy.stats import norm as _scipy_norm
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

logger = logging.getLogger("quant.inference")

# ──────────────────────────────────────────────────────────────────────────────
# 보조 함수
# ──────────────────────────────────────────────────────────────────────────────

def _prob_up_gaussian(mu: float, sigma: float) -> float:
    """
    P(X > 0) under X ~ N(mu, sigma²).
    sigma는 반드시 양수여야 함.
    """
    sigma = max(sigma, 1e-8)
    z = mu / sigma
    if _HAS_SCIPY:
        return float(_scipy_norm.cdf(z))
    # 간단한 erf 근사 (scipy 없을 때 fallback)
    import math
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normalize_segment_roi(seg: np.ndarray) -> np.ndarray:
    """
    ROIDetector._normalize_segment() 와 IDENTICAL 한 정규화.

    - OHLC: (price - first_close) / first_close  (상대 수익률 스케일)
    - Volume: log1p → z-score

    Args:
        seg: (T, 5) raw OHLCV float array

    Returns:
        (T, 5) normalized float32 array
    """
    result    = seg.copy().astype(float)
    base_close = seg[0, 3] + 1e-10        # 첫 날 Close

    # OHLC: 첫 날 종가 대비 상대값
    result[:, :4] = (seg[:, :4] - base_close) / base_close

    # Volume: log1p + z-score
    vol     = np.maximum(seg[:, 4], 0.0)
    log_vol = np.log1p(vol)
    mu_v    = log_vol.mean()
    std_v   = log_vol.std() + 1e-10
    result[:, 4] = (log_vol - mu_v) / std_v

    return result.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 메인 클래스
# ──────────────────────────────────────────────────────────────────────────────

class InferencePredictor:
    """
    프로덕션 등급 미래 예측 엔진.

    Examples
    --------
    >>> predictor = InferencePredictor(settings, model_dir, cache_dir)
    >>> predictor.load_all_models(symbols)
    >>> results = predictor.predict_all(symbols, data_dict)
    >>> portfolio = predictor.build_portfolio_from_predictions(results)
    """

    # ── 분류 임계값 (조정 가능) ────────────────────────────────────────────
    _PROB_BUY_THRESH    = 0.60   # 상승 확률 ≥ 60% → UP
    _PROB_SELL_THRESH   = 0.40   # 상승 확률 ≤ 40% → DOWN
    _MU_MIN_SIGNAL      = 0.003  # 예상 수익률 절댓값 ≥ 0.3% 이상이어야 방향성 인정
    _SNR_HIGH           = 1.0    # |μ/σ| ≥ 1.0 → HIGH confidence
    _SNR_MEDIUM         = 0.50   # |μ/σ| ≥ 0.5 → MEDIUM confidence
    _MU_CLIP            = 2.0    # 200% 초과 수익률은 비현실적 → 클리핑

    def __init__(
        self,
        settings:  AppSettings,
        model_dir: str,
        cache_dir: str,
    ):
        self.settings  = settings
        self.model_dir = model_dir
        self.cache_dir = cache_dir

        self._store  = ModelStore(model_dir)
        self._loader = DataLoader(cache_dir, settings.data.cache_ttl_hours)

        # symbol → { model, trainer, device, seg_len, img_size, lookahead, config }
        self._loaded: Dict[str, Dict[str, Any]] = {}

        # 예측 이력 로그 (최근 500건)
        self._history: List[Dict[str, Any]] = []

    # ══════════════════════════════════════════════════════════════════════════
    # 1. 모델 로딩
    # ══════════════════════════════════════════════════════════════════════════

    def load_model(self, symbol: str, device: str = "cpu") -> bool:
        """
        종목별 학습된 모델을 체크포인트에서 로드합니다.

        체크포인트에 저장된 config(아키텍처)를 최우선으로 사용하여
        학습 당시와 동일한 모델 구조를 복원합니다.

        Returns
        -------
        bool : 성공 여부
        """
        if not self._store.has_model(symbol):
            logger.warning(f"[{symbol}] 저장된 모델 없음")
            return False

        try:
            import torch

            latest_path = os.path.join(self._store._sym_dir(symbol), "latest.pt")
            raw = torch.load(latest_path, map_location="cpu", weights_only=False)

            cfg      = raw.get("config",         {})
            feat_cfg = raw.get("feature_config", {})  # 학습 시 피처 설정 (있으면)

            # ── 아키텍처 복원 ─────────────────────────────────────────────
            model = HybridModel(
                img_in_channels    = cfg.get("img_in_channels",    1),
                cnn_channels       = cfg.get("cnn_channels",
                                              self.settings.model.cnn_channels),
                cnn_out_dim        = cfg.get("cnn_out_dim",
                                              self.settings.model.cnn_out_dim),
                ts_input_dim       = cfg.get("ts_input_dim",       32),
                d_model            = cfg.get("d_model",
                                              self.settings.model.d_model),
                nhead              = cfg.get("nhead",
                                              self.settings.model.nhead),
                num_encoder_layers = cfg.get("num_encoder_layers",
                                              self.settings.model.num_encoder_layers),
                dim_feedforward    = cfg.get("dim_feedforward",
                                              self.settings.model.dim_feedforward),
                dropout            = cfg.get("dropout",
                                              self.settings.model.dropout),
            )

            ok = self._store.load(model, symbol, device=device)
            if not ok:
                logger.error(f"[{symbol}] state_dict 로드 실패")
                return False

            model.eval()

            # ── 피처 설정 (체크포인트 우선, fallback → settings) ──────────
            seg_len  = (feat_cfg.get("segment_length")
                        or cfg.get("segment_length")
                        or self.settings.roi.segment_length)
            img_size = (feat_cfg.get("image_size")
                        or cfg.get("image_size")
                        or self.settings.model.image_size)
            lookahead = (feat_cfg.get("lookahead")
                         or cfg.get("lookahead")
                         or self.settings.roi.lookahead)

            self._loaded[symbol] = {
                "model":    model,
                "trainer":  ModelTrainer(model, device=device),
                "device":   device,
                "config":   cfg,
                "seg_len":  int(seg_len),
                "img_size": int(img_size),
                "lookahead":int(lookahead),
            }

            logger.info(
                f"[{symbol}] 모델 로드 완료 "
                f"(d_model={cfg.get('d_model','?')}, "
                f"seg_len={seg_len}, lookahead={lookahead})"
            )
            return True

        except Exception as exc:
            logger.error(f"[{symbol}] 모델 로드 실패: {exc}", exc_info=True)
            return False

    def load_all_models(
        self,
        symbols: List[str],
        device:  str = "cpu",
        progress_cb=None,
    ) -> Dict[str, bool]:
        """모든 종목 모델을 일괄 로드합니다. {symbol: success} 반환."""
        results = {}
        for i, sym in enumerate(symbols):
            if progress_cb:
                progress_cb(i / len(symbols), f"[{sym}] 모델 로드 중...")
            if self._store.has_model(sym):
                results[sym] = self.load_model(sym, device=device)
            else:
                results[sym] = False
        if progress_cb:
            progress_cb(1.0, "로드 완료")
        return results

    # ══════════════════════════════════════════════════════════════════════════
    # 2. 피처 준비 (학습과 동일한 파이프라인)
    # ══════════════════════════════════════════════════════════════════════════

    def prepare_features_for_inference(
        self,
        df:          pd.DataFrame,
        symbol:      str,
        as_of_date:  Optional[pd.Timestamp] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        최신 데이터에서 추론용 피처를 추출합니다.

        학습 파이프라인과 100% 동일한 순서:
          1. 시간 경계 적용 (as_of_date 이후 데이터 제거)
          2. 마지막 segment_length 개 봉 추출
          3. ROIDetector._normalize_segment() 와 동일한 정규화
          4. CVFeatureExtractor.transform() → GAF 이미지
          5. TSFeatureExtractor.transform() → 정량 피처 벡터

        Parameters
        ----------
        df           : OHLCV DataFrame (Open/High/Low/Close/Volume 컬럼 필요)
        symbol       : 종목 코드
        as_of_date   : 이 날짜 이후 데이터를 사용하지 않음 (데이터 누수 방지)

        Returns
        -------
        images   : (1, C, H, W) float32
        ts_feats : (1, 32) float32

        Raises
        ------
        ValueError : 데이터 부족, 컬럼 누락, NaN 잔존 등
        """
        model_info = self._loaded.get(symbol, {})
        seg_len    = model_info.get("seg_len",  self.settings.roi.segment_length)
        img_size   = model_info.get("img_size", self.settings.model.image_size)

        # ① 시간 경계 (미래 데이터 차단)
        if as_of_date is not None:
            df = df[df.index <= as_of_date]

        # ② 데이터 충분성 검증
        if len(df) < seg_len:
            raise ValueError(
                f"[{symbol}] 데이터 부족: {len(df)}행 "
                f"(최소 {seg_len}행 필요)\n"
                "데이터 탭에서 기간을 늘려 다시 다운로드하세요."
            )

        # ③ 컬럼 검증
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"[{symbol}] 누락 컬럼: {missing}")

        # ④ 최근 seg_len 봉 추출
        recent = df.tail(seg_len)[required_cols].copy()

        # ⑤ NaN 처리 (forward fill → backward fill)
        recent = recent.ffill().bfill()
        raw_seg = recent.values.astype(np.float64)  # (T, 5)

        if np.isnan(raw_seg).any():
            raise ValueError(
                f"[{symbol}] 데이터에 NaN이 포함되어 있습니다. "
                "데이터를 다시 다운로드해 주세요."
            )

        # ⑥ ROIDetector 와 동일한 정규화 (EXACT MATCH)
        norm_seg = _normalize_segment_roi(raw_seg)   # (T, 5)

        # ⑦ 배치 차원 추가: (1, T, 5)
        seg_batch = norm_seg[np.newaxis, :, :]

        # ⑧ CV 피처 (GAF 이미지)
        cv_ext = CVFeatureExtractor(image_size=img_size)
        images = cv_ext.transform(seg_batch)             # (1, C, H, W)

        # ⑨ TS 피처 (정량 시계열 피처)
        ts_ext   = TSFeatureExtractor(n_features=32)
        ts_feats = ts_ext.transform(seg_batch)           # (1, 32)

        # ⑩ 최종 NaN/Inf 검증
        if np.isnan(images).any() or np.isinf(images).any():
            raise ValueError(f"[{symbol}] CV 피처에 NaN/Inf 발생")
        if np.isnan(ts_feats).any() or np.isinf(ts_feats).any():
            raise ValueError(f"[{symbol}] TS 피처에 NaN/Inf 발생")

        # ⑪ 모델 입력 차원 일치 검증
        expected_ts = model_info.get("config", {}).get("ts_input_dim", 32)
        if ts_feats.shape[1] != expected_ts:
            raise ValueError(
                f"[{symbol}] TS 피처 차원 불일치: "
                f"현재 {ts_feats.shape[1]} ≠ 학습 시 {expected_ts}\n"
                "모델을 재학습하거나 설정을 확인하세요."
            )

        return images.astype(np.float32), ts_feats.astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. 단일 종목 예측
    # ══════════════════════════════════════════════════════════════════════════

    def predict_future(
        self,
        symbol:       str,
        df:           pd.DataFrame,
        as_of_date:   Optional[pd.Timestamp] = None,
        horizon_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        단일 종목에 대한 미래 수익률·방향·신뢰도를 예측합니다.

        Parameters
        ----------
        symbol       : 종목 코드
        df           : OHLCV DataFrame (최신 데이터 포함)
        as_of_date   : 예측 기준 날짜 (None → df 최신 날짜)
        horizon_days : 예측 기간 (일). None → 체크포인트 lookahead 사용

        Returns
        -------
        Dict containing:
          symbol, current_price, predicted_return, uncertainty,
          prob_up, prob_down, snr, direction, confidence, action,
          horizon_days, explanation, timestamp, as_of_date, error
        """
        if symbol not in self._loaded:
            return self._error_result(symbol, "모델이 로드되지 않았습니다")

        model_info    = self._loaded[symbol]
        lookahead     = horizon_days or model_info.get("lookahead", 5)
        current_price = float(df["Close"].iloc[-1]) if len(df) > 0 else 0.0

        try:
            # ── 피처 추출 (누수 없음) ─────────────────────────────────────
            images, ts_feats = self.prepare_features_for_inference(
                df, symbol, as_of_date=as_of_date
            )

            # ── 모델 추론 ─────────────────────────────────────────────────
            trainer         = model_info["trainer"]
            mu_arr, sig_arr = trainer.predict(images, ts_feats)

            mu    = float(mu_arr[0])
            sigma = max(float(sig_arr[0]), 1e-8)

            # ── 유효성 검사 ───────────────────────────────────────────────
            if np.isnan(mu) or np.isnan(sigma):
                return self._error_result(
                    symbol, "모델 예측값이 NaN입니다. 모델을 재학습하세요."
                )
            if abs(mu) > self._MU_CLIP:
                logger.warning(f"[{symbol}] 비현실적 mu={mu:.3f} → 클리핑")
                mu = float(np.clip(mu, -self._MU_CLIP, self._MU_CLIP))

            # ── 통계적 파생 지표 ──────────────────────────────────────────
            prob_up   = _prob_up_gaussian(mu, sigma)
            prob_down = 1.0 - prob_up
            snr       = abs(mu) / sigma          # 신호 대 잡음비 (신뢰도)

            # ── 방향 분류 ─────────────────────────────────────────────────
            # 확률 AND 최소 수익률 임계값을 동시에 만족해야 방향성 인정
            if prob_up >= self._PROB_BUY_THRESH and mu >= self._MU_MIN_SIGNAL:
                direction = "UP"
            elif prob_down >= (1 - self._PROB_SELL_THRESH) and mu <= -self._MU_MIN_SIGNAL:
                direction = "DOWN"
            else:
                direction = "NEUTRAL"

            # ── 신뢰도 등급 ───────────────────────────────────────────────
            if snr >= self._SNR_HIGH and abs(prob_up - 0.5) >= 0.15:
                confidence = "HIGH"
            elif snr >= self._SNR_MEDIUM and abs(prob_up - 0.5) >= 0.08:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            # ── 행동 추천 (백테스트 신호 로직과 동일) ────────────────────
            action = self.generate_signal_from_prediction({
                "direction": direction, "confidence": confidence,
                "prob_up": prob_up, "mu": mu,
            })

            # ── 한국어 설명 생성 ──────────────────────────────────────────
            explanation = self._make_explanation(
                symbol, direction, prob_up, mu, sigma, lookahead, confidence
            )

            result: Dict[str, Any] = {
                "symbol":           symbol,
                "current_price":    current_price,
                "predicted_return": mu,
                "uncertainty":      sigma,
                "prob_up":          prob_up,
                "prob_down":        prob_down,
                "snr":              snr,
                "direction":        direction,
                "confidence":       confidence,
                "action":           action,
                "horizon_days":     lookahead,
                "explanation":      explanation,
                "timestamp":        datetime.datetime.now(),
                "as_of_date":       df.index[-1] if len(df) > 0 else None,
                "error":            None,
            }

            # 이력 기록
            self._append_history(result)

            logger.info(
                f"[{symbol}] 예측 완료 | "
                f"direction={direction} | "
                f"μ={mu:+.4f} | σ={sigma:.4f} | "
                f"P(↑)={prob_up:.1%} | "
                f"confidence={confidence} | action={action}"
            )
            return result

        except Exception as exc:
            logger.error(f"[{symbol}] 예측 실패: {exc}", exc_info=True)
            return self._error_result(symbol, str(exc))

    # ══════════════════════════════════════════════════════════════════════════
    # 4. 전 종목 일괄 예측
    # ══════════════════════════════════════════════════════════════════════════

    def predict_all(
        self,
        symbols:      List[str],
        data_dict:    Dict[str, pd.DataFrame],
        device:       str = "cpu",
        horizon_days: Optional[int] = None,
        high_conf_only: bool = False,
        progress_cb=None,
    ) -> List[Dict[str, Any]]:
        """
        모든 종목에 대해 미래 예측을 실행합니다.

        Parameters
        ----------
        symbols        : 예측 대상 종목 목록
        data_dict      : {symbol: OHLCV DataFrame}
        device         : "cpu" 또는 "cuda"
        horizon_days   : 예측 기간 (None → 종목별 체크포인트 설정 사용)
        high_conf_only : True이면 HIGH 신뢰도 종목만 반환
        progress_cb    : (pct: float, msg: str) → None 콜백

        Returns
        -------
        List[Dict] — 예상 수익률 내림차순 정렬. 오류 종목은 맨 뒤.
        """
        results = []
        total   = max(len(symbols), 1)

        for i, sym in enumerate(symbols):
            if progress_cb:
                progress_cb(i / total, f"[{sym}] 예측 중... ({i+1}/{total})")

            # 미로드 모델 자동 로드
            if sym not in self._loaded:
                ok = self.load_model(sym, device=device)
                if not ok:
                    results.append(self._error_result(sym, "학습된 모델 없음"))
                    continue

            df = data_dict.get(sym)
            if df is None or len(df) < 10:
                results.append(self._error_result(
                    sym, "데이터 없음 — 데이터 탭에서 다운로드 필요"
                ))
                continue

            pred = self.predict_future(sym, df, horizon_days=horizon_days)
            results.append(pred)

        if progress_cb:
            progress_cb(1.0, f"예측 완료 ({len(results)}개 종목)")

        # 분류: 유효/오류
        valid   = [r for r in results if r["error"] is None]
        invalid = [r for r in results if r["error"] is not None]

        # 유효 결과: 예상 수익률 내림차순
        valid.sort(key=lambda x: x["predicted_return"], reverse=True)

        # 고신뢰도 필터
        if high_conf_only:
            valid = [r for r in valid if r["confidence"] == "HIGH"] + \
                    [r for r in valid if r["confidence"] != "HIGH"]

        return valid + invalid

    # ══════════════════════════════════════════════════════════════════════════
    # 5. 신호 생성 (백테스트 로직과 일치)
    # ══════════════════════════════════════════════════════════════════════════

    def generate_signal_from_prediction(self, pred: Dict[str, Any]) -> str:
        """
        예측 딕셔너리에서 BUY / HOLD / SELL / WATCH 신호를 생성합니다.

        백테스트 SignalGenerator 로직과 동일하게 동작합니다.

        Rules
        -----
        BUY   : UP + (HIGH or MEDIUM) confidence
        SELL  : DOWN + (HIGH or MEDIUM) confidence
        WATCH : UP + LOW confidence (관망하되 눈여겨 볼 것)
        HOLD  : NEUTRAL or insufficient signal
        """
        direction  = pred.get("direction",  "NEUTRAL")
        confidence = pred.get("confidence", "LOW")

        if direction == "UP" and confidence in ("HIGH", "MEDIUM"):
            return "BUY"
        elif direction == "DOWN" and confidence in ("HIGH", "MEDIUM"):
            return "SELL"
        elif direction == "UP" and confidence == "LOW":
            return "WATCH"
        else:
            return "HOLD"

    # ══════════════════════════════════════════════════════════════════════════
    # 6. 포트폴리오 구성
    # ══════════════════════════════════════════════════════════════════════════

    def build_portfolio_from_predictions(
        self,
        predictions:    List[Dict[str, Any]],
        method:         str = "prob_weight",   # "equal" | "return" | "prob_weight"
        top_n:          int = 5,
        min_confidence: str = "LOW",           # "HIGH" | "MEDIUM" | "LOW"
        data_dict:      Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """
        예측 결과로부터 포트폴리오 비중을 산출합니다.

        Parameters
        ----------
        predictions    : predict_all() 의 반환값
        method         : 비중 산출 방법 ("equal" / "return" / "prob_weight")
        top_n          : 상위 N 개 종목 선택
        min_confidence : 최소 신뢰도 필터 ("HIGH" / "MEDIUM" / "LOW")
        data_dict      : 변동성 조정용 OHLCV 딕셔너리 (optional)

        Returns
        -------
        {
          "weights":         {symbol: float},
          "ranked":          [sorted prediction dicts],
          "summary":         str,
          "portfolio_score": float,   # 가중 평균 상승 확률
          "n_selected":      int,
        }
        """
        # ① 유효한 예측만 사용
        valid = [p for p in predictions if p.get("error") is None]

        # ② 신뢰도 필터
        _conf_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        min_rank   = _conf_rank.get(min_confidence, 1)
        filtered   = [p for p in valid
                      if _conf_rank.get(p.get("confidence", "LOW"), 1) >= min_rank]
        if not filtered:
            filtered = valid   # fallback

        # ③ 매수 후보: BUY or WATCH 신호만
        buy_cands = [p for p in filtered
                     if p.get("action") in ("BUY", "WATCH")]
        buy_cands.sort(key=lambda x: x.get("predicted_return", 0), reverse=True)
        top = buy_cands[:top_n]

        if not top:
            return {
                "weights":         {},
                "ranked":          filtered,
                "summary":         (
                    "현재 매수 신호가 있는 종목이 없습니다.\n"
                    "시장 상황을 관망하거나 신뢰도 필터를 낮춰보세요."
                ),
                "portfolio_score": 0.0,
                "n_selected":      0,
            }

        # ④ 비중 산출
        raw_w: Dict[str, float] = {}
        for p in top:
            sym = p["symbol"]
            if method == "equal":
                raw_w[sym] = 1.0
            elif method == "return":
                raw_w[sym] = max(p.get("predicted_return", 0.0), 1e-6)
            else:  # prob_weight (default)
                raw_w[sym] = max(p.get("prob_up", 0.5) - 0.5, 1e-6) * 2.0

        total = sum(raw_w.values())
        weights = {s: w / total for s, w in raw_w.items()} if total > 0 else {
            s: 1.0 / len(top) for s in raw_w
        }

        # ⑤ 최대 비중 제한
        weights = self._cap_weights(weights, self.settings.portfolio.max_weight)

        # ⑥ 변동성 조정 (data_dict 제공 시)
        if data_dict:
            weights = self._vol_adjust_weights(weights, data_dict)

        # ⑦ 포트폴리오 기대 점수 (가중 평균 상승 확률)
        port_score = sum(
            weights.get(p["symbol"], 0.0) * p.get("prob_up", 0.5)
            for p in top
        )

        summary = self._portfolio_summary(top, weights, port_score)

        return {
            "weights":         weights,
            "ranked":          buy_cands,
            "summary":         summary,
            "portfolio_score": port_score,
            "n_selected":      len(top),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # 7. 내부 헬퍼
    # ══════════════════════════════════════════════════════════════════════════

    # ── 비중 처리 ─────────────────────────────────────────────────────────────

    def _cap_weights(self, weights: Dict[str, float], max_w: float) -> Dict[str, float]:
        """최대 비중 초과분을 나머지 종목에 비례 재배분합니다."""
        for _ in range(30):
            capped = {s: min(w, max_w) for s, w in weights.items()}
            total  = sum(capped.values())
            if abs(total - 1.0) < 1e-6:
                return capped
            overflow  = 1.0 - total
            uncapped  = {s: w for s, w in capped.items() if w < max_w - 1e-8}
            if not uncapped:
                break
            extra = overflow / len(uncapped)
            weights = {s: w + (extra if s in uncapped else 0.0)
                       for s, w in capped.items()}
        total = sum(weights.values())
        return {s: w / total for s, w in weights.items()} if total > 1e-9 else weights

    def _vol_adjust_weights(
        self,
        weights:   Dict[str, float],
        data_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """변동성이 높은 종목의 비중을 축소합니다 (목표 변동성 기반)."""
        target_vol = getattr(
            getattr(self.settings, "portfolio", None), "target_volatility", 0.15
        )
        adj: Dict[str, float] = {}
        for sym, w in weights.items():
            df = data_dict.get(sym)
            if df is not None and len(df) >= 20:
                rets = df["Close"].pct_change().dropna().tail(20)
                ann_vol = float(rets.std() * np.sqrt(252)) if len(rets) > 1 else 0.20
            else:
                ann_vol = 0.20
            scale      = min(target_vol / max(ann_vol, 1e-6), 1.5)
            adj[sym]   = w * scale

        total = sum(adj.values())
        return {s: v / total for s, v in adj.items()} if total > 1e-9 else weights

    # ── 설명 생성 ─────────────────────────────────────────────────────────────

    def _make_explanation(
        self,
        symbol:     str,
        direction:  str,
        prob_up:    float,
        mu:         float,
        sigma:      float,
        lookahead:  int,
        confidence: str,
    ) -> str:
        """초보자 친화적 한국어 예측 설명을 생성합니다."""
        week_str = (f"약 {lookahead // 5}주" if lookahead >= 5
                    else f"{lookahead}일")
        horizon  = f"향후 {lookahead}거래일({week_str})"
        ret_pct  = mu * 100.0

        if direction == "UP":
            line1 = f"📈 상승 가능성이 높습니다  (상승 확률: {prob_up:.0%})"
            line2 = f"{horizon} 동안 약 {ret_pct:+.1f}% 수익이 예상됩니다."
        elif direction == "DOWN":
            line1 = f"📉 하락 가능성이 높습니다  (하락 확률: {1-prob_up:.0%})"
            line2 = f"{horizon} 동안 약 {ret_pct:+.1f}% 하락이 예상됩니다."
        else:
            line1 = f"↔️ 방향이 불분명합니다  (상승 확률: {prob_up:.0%})"
            line2 = f"{horizon} 동안 뚜렷한 방향성이 확인되지 않습니다."

        conf_map = {
            "HIGH":   "신뢰도: 높음 ✅  (AI의 예측 확신이 강합니다)",
            "MEDIUM": "신뢰도: 보통 ⚡  (어느 정도 확신이 있습니다)",
            "LOW":    "신뢰도: 낮음 ⚠️  (불확실성이 높습니다. 신중히 참고하세요)",
        }
        line3 = conf_map.get(confidence, "")
        line4 = f"불확실성(σ): {sigma:.4f}  |  신호 강도(μ/σ): {abs(mu)/sigma:.2f}"

        return "\n".join([line1, line2, line3, line4])

    def _portfolio_summary(
        self,
        top:        List[Dict[str, Any]],
        weights:    Dict[str, float],
        port_score: float,
    ) -> str:
        """포트폴리오 제안 요약 텍스트를 생성합니다."""
        lines = [f"📊 포트폴리오 제안  ({len(top)}개 종목 선정)"]
        lines.append("─" * 52)
        for p in top:
            sym  = p["symbol"]
            w    = weights.get(sym, 0.0)
            conf = p.get("confidence", "LOW")
            conf_icon = {"HIGH": "✅", "MEDIUM": "⚡", "LOW": "⚠️"}.get(conf, "")
            lines.append(
                f"  {sym:<14}  {w:.1%} 비중  "
                f"예상 수익: {p['predicted_return']*100:+.1f}%  "
                f"상승확률: {p['prob_up']:.0%}  "
                f"{conf_icon} {conf}"
            )
        lines.append("─" * 52)
        lines.append(f"포트폴리오 기대 상승 확률: {port_score:.1%}")
        if port_score >= 0.65:
            lines.append("종합 판단: 매수 비중 확대 고려 ✅")
        elif port_score >= 0.55:
            lines.append("종합 판단: 소규모 매수 고려 ⚡")
        else:
            lines.append("종합 판단: 관망 또는 현금 유지 권장 ⚠️")
        lines.append("\n※ AI 예측은 참고용이며 실제 투자 결과를 보장하지 않습니다.")
        return "\n".join(lines)

    # ── 공통 헬퍼 ────────────────────────────────────────────────────────────

    def _error_result(self, symbol: str, msg: str) -> Dict[str, Any]:
        return {
            "symbol":           symbol,
            "current_price":    0.0,
            "predicted_return": 0.0,
            "uncertainty":      0.0,
            "prob_up":          0.5,
            "prob_down":        0.5,
            "snr":              0.0,
            "direction":        "NEUTRAL",
            "confidence":       "LOW",
            "action":           "HOLD",
            "horizon_days":     self.settings.roi.lookahead,
            "explanation":      f"⚠️ {msg}",
            "timestamp":        datetime.datetime.now(),
            "as_of_date":       None,
            "error":            msg,
        }

    def _append_history(self, result: Dict[str, Any]):
        self._history.append({
            "timestamp": result["timestamp"],
            "symbol":    result["symbol"],
            "mu":        result["predicted_return"],
            "sigma":     result["uncertainty"],
            "prob_up":   result["prob_up"],
            "direction": result["direction"],
            "action":    result["action"],
            "horizon":   result["horizon_days"],
        })
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

    # ── 공개 유틸리티 ────────────────────────────────────────────────────────

    def has_model(self, symbol: str) -> bool:
        """ModelStore에 저장된 모델이 있는지 확인합니다."""
        return self._store.has_model(symbol)

    def is_loaded(self, symbol: str) -> bool:
        """현재 세션에 모델이 로드되어 있는지 확인합니다."""
        return symbol in self._loaded

    def get_history(self) -> List[Dict[str, Any]]:
        """최근 예측 이력을 반환합니다."""
        return list(self._history)

    def symbols_with_models(self) -> List[str]:
        """학습된 모델이 있는 종목 목록을 반환합니다."""
        return [s for s in self.settings.data.symbols if self._store.has_model(s)]
