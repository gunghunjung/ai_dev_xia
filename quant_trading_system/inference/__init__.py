# inference/__init__.py — 추론(예측) 패키지
"""
Production-grade inference pipeline for the quant trading system.

Usage:
    from inference import InferencePredictor

    predictor = InferencePredictor(settings, model_dir, cache_dir)
    predictor.load_all_models(symbols, device="cuda")
    results   = predictor.predict_all(symbols, data_dict)
    portfolio = predictor.build_portfolio_from_predictions(results)
"""

from .predictor import InferencePredictor

__all__ = ["InferencePredictor"]
