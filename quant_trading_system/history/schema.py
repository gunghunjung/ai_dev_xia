"""
history/schema.py
=================
Dataclass schemas for prediction records, verification records, and backtest sessions.
Each class provides to_dict() / from_dict() for JSON serialisation.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


def _new_uuid() -> str:
    """Return a new UUID4 string."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# PredictionRecord
# ---------------------------------------------------------------------------

@dataclass
class PredictionRecord:
    """
    A single model prediction recorded at inference time.

    Attributes
    ----------
    id : str
        Unique identifier (UUID4).
    timestamp : str
        ISO-8601 UTC timestamp at which the prediction was made.
    symbol : str
        Ticker symbol (e.g. "005930" for Samsung).
    name : str
        Human-readable name of the security.
    model_version : str
        Version tag of the model that produced the prediction.
    predicted_direction : str
        One of "UP", "DOWN", "NEUTRAL".
    prob_up : float
        Model's estimated probability of an upward move [0, 1].
    predicted_return_pct : float
        Expected return in percent over *horizon_days* (e.g. 2.5 means +2.5 %).
    confidence : str
        Derived confidence tier: "HIGH", "MEDIUM", or "LOW".
    action : str
        Suggested action: "BUY", "SELL", "HOLD", or "WATCH".
    price_at_prediction : float
        Security price when the prediction was recorded.
    horizon_days : int
        Number of trading days over which the prediction is valid.
    segment_length : int
        Input sequence length used by the model.
    lookahead : int
        Lookahead steps the model was trained with.
    d_model : int
        Transformer / model hidden dimension.
    notes : str
        Free-form notes (default empty string).
    verified : bool
        Whether the prediction outcome has been verified.
    actual_return_pct : Optional[float]
        Realised return in percent (populated after verification).
    hit : Optional[bool]
        True if the directional prediction was correct.
    verification_date : Optional[str]
        ISO-8601 date on which verification was performed.
    """

    # --- identity / meta ---
    id: str = field(default_factory=_new_uuid)
    timestamp: str = field(default_factory=_now_iso)

    # --- security ---
    symbol: str = ""
    name: str = ""
    model_version: str = ""

    # --- prediction payload (primary / representative horizon) ---
    predicted_direction: str = "NEUTRAL"   # "UP" | "DOWN" | "NEUTRAL"
    prob_up: float = 0.5
    predicted_return_pct: float = 0.0
    confidence: str = "LOW"                # "HIGH" | "MEDIUM" | "LOW"
    action: str = "HOLD"                   # "BUY" | "SELL" | "HOLD" | "WATCH"
    price_at_prediction: float = 0.0

    # --- multi-horizon unified prediction (1d / 3d / 5d / 20d) ---
    # Each entry: {"direction": str, "prob_up": float, "return_pct": float, "confidence": str}
    horizons: dict = field(default_factory=dict)

    # --- model hyper-params captured at inference time ---
    horizon_days: int = 5
    segment_length: int = 60
    lookahead: int = 5
    d_model: int = 64

    # --- external environment snapshot ---
    macro_sentiment: float = 0.0        # -1(악재) ~ +1(호재)
    macro_event_tags: str = ""          # "FOMC,RATE,CPI" 등 쉼표 구분

    # --- optional annotation ---
    notes: str = ""

    # --- verification state (populated later) ---
    verified: bool = False
    actual_return_pct: Optional[float] = None
    hit: Optional[bool] = None
    verification_date: Optional[str] = None

    # --- per-horizon verification (populated later) ---
    # { "1": {"actual_return_pct": float, "hit": bool}, "3": {...}, ... }
    horizon_verifications: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON encoding."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "name": self.name,
            "model_version": self.model_version,
            "predicted_direction": self.predicted_direction,
            "prob_up": self.prob_up,
            "predicted_return_pct": self.predicted_return_pct,
            "confidence": self.confidence,
            "action": self.action,
            "price_at_prediction": self.price_at_prediction,
            "horizons": self.horizons,
            "horizon_days": self.horizon_days,
            "segment_length": self.segment_length,
            "lookahead": self.lookahead,
            "d_model": self.d_model,
            "macro_sentiment": self.macro_sentiment,
            "macro_event_tags": self.macro_event_tags,
            "notes": self.notes,
            "verified": self.verified,
            "actual_return_pct": self.actual_return_pct,
            "hit": self.hit,
            "verification_date": self.verification_date,
            "horizon_verifications": self.horizon_verifications,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PredictionRecord:
        """Deserialise from a plain dict (e.g. parsed from JSON)."""
        return cls(
            id=data.get("id", _new_uuid()),
            timestamp=data.get("timestamp", _now_iso()),
            symbol=data.get("symbol", ""),
            name=data.get("name", ""),
            model_version=data.get("model_version", ""),
            predicted_direction=data.get("predicted_direction", "NEUTRAL"),
            prob_up=float(data.get("prob_up", 0.5)),
            predicted_return_pct=float(data.get("predicted_return_pct", 0.0)),
            confidence=data.get("confidence", "LOW"),
            action=data.get("action", "HOLD"),
            price_at_prediction=float(data.get("price_at_prediction", 0.0)),
            horizons=data.get("horizons", {}),
            horizon_days=int(data.get("horizon_days", 5)),
            segment_length=int(data.get("segment_length", 60)),
            lookahead=int(data.get("lookahead", 5)),
            d_model=int(data.get("d_model", 64)),
            macro_sentiment=float(data.get("macro_sentiment", 0.0)),
            macro_event_tags=data.get("macro_event_tags", ""),
            notes=data.get("notes", ""),
            verified=bool(data.get("verified", False)),
            actual_return_pct=(
                float(data["actual_return_pct"])
                if data.get("actual_return_pct") is not None
                else None
            ),
            hit=(
                bool(data["hit"])
                if data.get("hit") is not None
                else None
            ),
            verification_date=data.get("verification_date"),
            horizon_verifications=data.get("horizon_verifications", {}),
        )


# ---------------------------------------------------------------------------
# VerificationRecord
# ---------------------------------------------------------------------------

@dataclass
class VerificationRecord:
    """
    The outcome of verifying a single prediction against realised market data.

    Attributes
    ----------
    prediction_id : str
        Foreign-key reference to ``PredictionRecord.id``.
    symbol : str
        Ticker symbol.
    verification_date : str
        ISO-8601 date on which verification was run.
    price_at_prediction : float
        Security price when the prediction was originally recorded.
    price_at_horizon : float
        Security price at (prediction_date + horizon_days).
    actual_return_pct : float
        Realised return in percent.
    predicted_return_pct : float
        The model's predicted return in percent (copied from PredictionRecord).
    hit : bool
        True if the directional call was correct.
    abs_error_pct : float
        ``|actual_return_pct - predicted_return_pct|``.
    profit_if_followed_pct : float
        Hypothetical P&L if the suggested *action* was followed.
        Positive when action="BUY" and price went up, or action="SELL"
        and price went down; negative otherwise.
    """

    prediction_id: str = ""
    symbol: str = ""
    verification_date: str = field(default_factory=_now_iso)
    price_at_prediction: float = 0.0
    price_at_horizon: float = 0.0
    actual_return_pct: float = 0.0
    predicted_return_pct: float = 0.0
    hit: bool = False
    abs_error_pct: float = 0.0
    profit_if_followed_pct: float = 0.0

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON encoding."""
        return {
            "prediction_id": self.prediction_id,
            "symbol": self.symbol,
            "verification_date": self.verification_date,
            "price_at_prediction": self.price_at_prediction,
            "price_at_horizon": self.price_at_horizon,
            "actual_return_pct": self.actual_return_pct,
            "predicted_return_pct": self.predicted_return_pct,
            "hit": self.hit,
            "abs_error_pct": self.abs_error_pct,
            "profit_if_followed_pct": self.profit_if_followed_pct,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerificationRecord:
        """Deserialise from a plain dict."""
        return cls(
            prediction_id=data.get("prediction_id", ""),
            symbol=data.get("symbol", ""),
            verification_date=data.get("verification_date", _now_iso()),
            price_at_prediction=float(data.get("price_at_prediction", 0.0)),
            price_at_horizon=float(data.get("price_at_horizon", 0.0)),
            actual_return_pct=float(data.get("actual_return_pct", 0.0)),
            predicted_return_pct=float(data.get("predicted_return_pct", 0.0)),
            hit=bool(data.get("hit", False)),
            abs_error_pct=float(data.get("abs_error_pct", 0.0)),
            profit_if_followed_pct=float(data.get("profit_if_followed_pct", 0.0)),
        )


# ---------------------------------------------------------------------------
# BacktestSession
# ---------------------------------------------------------------------------

@dataclass
class BacktestSession:
    """
    A complete walk-forward backtest session record.

    Attributes
    ----------
    session_id : str
        Unique identifier (UUID4).
    timestamp : str
        ISO-8601 UTC timestamp when the session was run.
    strategy_config : dict
        Arbitrary strategy / model configuration snapshot.
    capital : float
        Initial capital in base currency units.
    transaction_cost : float
        Round-trip transaction cost fraction (e.g. 0.001 = 0.1 %).
    slippage : float
        One-way slippage fraction (e.g. 0.0005 = 0.05 %).
    execution_delay : int
        Number of bars delay between signal and execution.
    train_days : int
        Number of calendar days in each training window.
    test_days : int
        Number of calendar days in each test window.
    step_days : int
        Step size in calendar days between successive windows.
    symbols : list[str]
        Universe of tickers included in this session.
    total_return_pct : float
        Cumulative return over the full backtest period (percent).
    cagr : float
        Compound annual growth rate (percent).
    max_drawdown : float
        Maximum peak-to-trough drawdown (percent, negative value).
    sharpe : float
        Annualised Sharpe ratio.
    sortino : float
        Annualised Sortino ratio.
    win_rate : float
        Fraction of trades that were profitable [0, 1].
    n_windows : int
        Total number of walk-forward windows evaluated.
    n_trades : int
        Total number of trades executed across all windows.
    equity_curve : list[float]
        Portfolio value at each time step.
    equity_dates : list[str]
        ISO date strings corresponding to ``equity_curve`` entries.
    wf_results : list[dict]
        Per-window walk-forward metrics.
    trades : list[dict]
        Individual trade records (symbol, entry_date, exit_date, pnl_pct, ...).
    notes : str
        Free-form notes.
    """

    # --- identity ---
    session_id: str = field(default_factory=_new_uuid)
    timestamp: str = field(default_factory=_now_iso)

    # --- strategy / simulation config ---
    strategy_config: dict = field(default_factory=dict)
    capital: float = 100_000.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    execution_delay: int = 1

    # --- walk-forward parameters ---
    train_days: int = 252
    test_days: int = 63
    step_days: int = 21

    # --- universe ---
    symbols: list[str] = field(default_factory=list)

    # --- aggregate performance metrics ---
    total_return_pct: float = 0.0
    cagr: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    win_rate: float = 0.0
    n_windows: int = 0
    n_trades: int = 0

    # --- time-series results ---
    equity_curve: list[float] = field(default_factory=list)
    equity_dates: list[str] = field(default_factory=list)
    wf_results: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)

    notes: str = ""

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON encoding."""
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "strategy_config": self.strategy_config,
            "capital": self.capital,
            "transaction_cost": self.transaction_cost,
            "slippage": self.slippage,
            "execution_delay": self.execution_delay,
            "train_days": self.train_days,
            "test_days": self.test_days,
            "step_days": self.step_days,
            "symbols": self.symbols,
            "total_return_pct": self.total_return_pct,
            "cagr": self.cagr,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "win_rate": self.win_rate,
            "n_windows": self.n_windows,
            "n_trades": self.n_trades,
            "equity_curve": self.equity_curve,
            "equity_dates": self.equity_dates,
            "wf_results": self.wf_results,
            "trades": self.trades,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BacktestSession:
        """Deserialise from a plain dict."""
        return cls(
            session_id=data.get("session_id", _new_uuid()),
            timestamp=data.get("timestamp", _now_iso()),
            strategy_config=data.get("strategy_config", {}),
            capital=float(data.get("capital", 100_000.0)),
            transaction_cost=float(data.get("transaction_cost", 0.001)),
            slippage=float(data.get("slippage", 0.0005)),
            execution_delay=int(data.get("execution_delay", 1)),
            train_days=int(data.get("train_days", 252)),
            test_days=int(data.get("test_days", 63)),
            step_days=int(data.get("step_days", 21)),
            symbols=list(data.get("symbols", [])),
            total_return_pct=float(data.get("total_return_pct", 0.0)),
            cagr=float(data.get("cagr", 0.0)),
            max_drawdown=float(data.get("max_drawdown", 0.0)),
            sharpe=float(data.get("sharpe", 0.0)),
            sortino=float(data.get("sortino", 0.0)),
            win_rate=float(data.get("win_rate", 0.0)),
            n_windows=int(data.get("n_windows", 0)),
            n_trades=int(data.get("n_trades", 0)),
            equity_curve=list(data.get("equity_curve", [])),
            equity_dates=list(data.get("equity_dates", [])),
            wf_results=list(data.get("wf_results", [])),
            trades=list(data.get("trades", [])),
            notes=data.get("notes", ""),
        )
