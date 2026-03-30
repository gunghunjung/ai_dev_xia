"""
history/verifier.py
===================
Truth-verification engine.

Compares stored PredictionRecord objects against realised market data and
writes VerificationRecord + updates PredictionRecord.verified fields via the
PredictionLogger.

Design principles
-----------------
* Zero look-ahead: uses only data available *after* the prediction horizon.
* Idempotent: re-running on the same record is a no-op (already verified).
* Batch-capable: verifies all mature (horizon elapsed) predictions in one call.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import pandas as pd

from history.schema import PredictionRecord, VerificationRecord
from history.logger import PredictionLogger

logger = logging.getLogger("quant.history.verifier")


def _trading_days_elapsed(from_date: str, to_date: str,
                           price_index: pd.DatetimeIndex) -> int:
    """
    from_date 와 to_date 사이에 price_index 기준 거래일이 몇 개 있는지 반환.
    """
    try:
        t0 = pd.Timestamp(from_date)
        t1 = pd.Timestamp(to_date)
        mask = (price_index >= t0) & (price_index <= t1)
        return int(mask.sum())
    except Exception:
        return 0


def _find_horizon_price(df: pd.DataFrame,
                        prediction_date: str,
                        horizon_days: int) -> float | None:
    """
    prediction_date 이후 horizon_days 번째 거래일 종가를 반환.
    해당 날짜가 없으면 가장 가까운 이후 날짜를 사용.

    Returns:
        float | None  — None if data is insufficient.
    """
    try:
        t0 = pd.Timestamp(prediction_date).normalize()
        future = df[df.index.normalize() > t0]
        if len(future) < horizon_days:
            return None
        return float(future["Close"].iloc[horizon_days - 1])
    except Exception:
        return None


class TruthVerifier:
    """
    PredictionLogger 에 저장된 예측을 실제 시장 데이터와 비교·검증.

    Usage
    -----
    from data import DataLoader
    from history.logger import PredictionLogger
    from history.verifier import TruthVerifier

    pl  = PredictionLogger("outputs/history")
    dl  = DataLoader("cache")
    tv  = TruthVerifier(pl, dl)
    results = tv.verify_all(settings)
    """

    def __init__(self, prediction_logger: PredictionLogger,
                 data_loader: Any) -> None:
        """
        Parameters
        ----------
        prediction_logger : PredictionLogger
        data_loader       : DataLoader  (or any object with .load(sym, period, interval))
        """
        self._logger = prediction_logger
        self._loader = data_loader
        self._price_cache: dict[str, pd.DataFrame] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def verify_all(self, period: str = "1y",
                   interval: str = "1d") -> list[VerificationRecord]:
        """
        검증 안 된 *만료된* 예측을 모두 찾아 검증하고 결과를 반환한다.

        'Matured'(만료) 기준: prediction_date + horizon_days < today
        """
        unverified = self._logger.load_unverified()
        today = datetime.now(tz=timezone.utc).date()

        results: list[VerificationRecord] = []
        for rec in unverified:
            pred_date = pd.Timestamp(rec.timestamp).date()
            maturity  = pred_date + timedelta(days=int(rec.horizon_days * 1.5))
            if maturity > today:
                continue   # 아직 만료 안 됨

            vr = self._verify_one(rec, period, interval)
            if vr is not None:
                results.append(vr)

        logger.info("Verified %d predictions", len(results))
        return results

    def verify_symbol(self, symbol: str,
                      period: str = "1y",
                      interval: str = "1d") -> list[VerificationRecord]:
        """특정 종목의 미검증 예측만 검증."""
        recs = [r for r in self._logger.load_unverified()
                if r.symbol == symbol]
        results = []
        for rec in recs:
            vr = self._verify_one(rec, period, interval)
            if vr is not None:
                results.append(vr)
        return results

    def summary(self) -> dict[str, Any]:
        """전체 예측 정확도 요약."""
        return self._logger.summary_stats()

    # ──────────────────────────────────────────────────────────────────────────
    # 단일 검증
    # ──────────────────────────────────────────────────────────────────────────

    def _verify_one(self, rec: PredictionRecord,
                    period: str, interval: str) -> VerificationRecord | None:
        """
        단일 PredictionRecord 를 검증하고 VerificationRecord 를 반환.
        logger에도 자동으로 업데이트를 기록한다.
        """
        df = self._get_price_data(rec.symbol, period, interval)
        if df is None or df.empty:
            logger.warning("No price data for %s — skipping", rec.symbol)
            return None

        price_at_horizon = _find_horizon_price(df, rec.timestamp[:10],
                                               rec.horizon_days)
        if price_at_horizon is None:
            logger.debug("Horizon data not yet available for %s (%s)",
                         rec.symbol, rec.timestamp[:10])
            return None

        p0   = rec.price_at_prediction
        if p0 == 0:
            logger.warning("Zero price_at_prediction for %s — skipping", rec.id)
            return None

        actual_return = (price_at_horizon - p0) / p0 * 100   # percent

        # Directional hit?
        if rec.predicted_direction == "UP":
            hit = actual_return > 0
        elif rec.predicted_direction == "DOWN":
            hit = actual_return < 0
        else:
            hit = abs(actual_return) < 1.0   # NEUTRAL: small move = correct

        abs_error = abs(actual_return - rec.predicted_return_pct)

        # Hypothetical P&L if action had been followed
        if rec.action == "BUY":
            profit_if_followed = actual_return
        elif rec.action == "SELL":
            profit_if_followed = -actual_return
        else:
            profit_if_followed = 0.0

        vr = VerificationRecord(
            prediction_id       = rec.id,
            symbol              = rec.symbol,
            verification_date   = datetime.now(tz=timezone.utc).isoformat(),
            price_at_prediction = p0,
            price_at_horizon    = price_at_horizon,
            actual_return_pct   = round(actual_return, 4),
            predicted_return_pct= rec.predicted_return_pct,
            hit                 = hit,
            abs_error_pct       = round(abs_error, 4),
            profit_if_followed_pct = round(profit_if_followed, 4),
        )

        # Update the prediction record in the logger
        self._logger.update_verification(rec.id, {
            "verified":           True,
            "actual_return_pct":  round(actual_return, 4),
            "hit":                hit,
            "verification_date":  vr.verification_date,
        })

        logger.info(
            "Verified %s (%s): actual=%.2f%% predicted=%.2f%% hit=%s",
            rec.symbol, rec.timestamp[:10],
            actual_return, rec.predicted_return_pct, hit,
        )
        return vr

    # ──────────────────────────────────────────────────────────────────────────
    # Data helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_price_data(self, symbol: str, period: str,
                        interval: str) -> pd.DataFrame | None:
        if symbol in self._price_cache:
            return self._price_cache[symbol]
        try:
            df = self._loader.load(symbol, period, interval)
            if df is not None and not df.empty:
                self._price_cache[symbol] = df
            return df
        except Exception as e:
            logger.error("Failed to load price data for %s: %s", symbol, e)
            return None

    def clear_cache(self) -> None:
        """가격 캐시를 비운다 (새 데이터 강제 조회 시 호출)."""
        self._price_cache.clear()
