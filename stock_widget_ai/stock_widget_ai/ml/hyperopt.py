"""
Hyperopt — Optuna 기반 하이퍼파라미터 탐색
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Any, Optional
from ..logger_config import get_logger

log = get_logger("ml.hyperopt")


class HyperOptimizer:
    def __init__(
        self,
        n_trials: int = 20,
        timeout: Optional[float] = None,
        on_trial_end: Optional[Callable[[int, float, Dict], None]] = None,
    ) -> None:
        self.n_trials   = n_trials
        self.timeout    = timeout
        self.on_trial_end = on_trial_end
        self.best_params: Dict[str, Any] = {}
        self.best_value: float = float("inf")

    def search(
        self,
        objective_fn: Callable[[Dict], float],
        search_space: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        objective_fn : params → val_loss (float, lower=better)
        Returns best params dict
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            log.warning("optuna 미설치 — 기본 파라미터 반환")
            return self._default_params()

        study = optuna.create_study(direction="minimize")

        def _optuna_obj(trial):
            params = self._sample(trial, search_space)
            val = objective_fn(params)
            if self.on_trial_end:
                self.on_trial_end(trial.number, val, params)
            return val

        study.optimize(_optuna_obj, n_trials=self.n_trials, timeout=self.timeout)
        self.best_params = study.best_params
        self.best_value  = study.best_value
        log.info(f"Hyperopt 완료: best_val={self.best_value:.6f}, params={self.best_params}")
        return self.best_params

    @staticmethod
    def _sample(trial, search_space: Optional[Dict]) -> Dict[str, Any]:
        import optuna
        if search_space:
            params = {}
            for k, v in search_space.items():
                if v["type"] == "float":
                    params[k] = trial.suggest_float(k, v["low"], v["high"], log=v.get("log", False))
                elif v["type"] == "int":
                    params[k] = trial.suggest_int(k, v["low"], v["high"])
                elif v["type"] == "categorical":
                    params[k] = trial.suggest_categorical(k, v["choices"])
            return params
        # default search space
        return {
            "lr":           trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "hidden_dim":   trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "num_layers":   trial.suggest_int("num_layers", 1, 4),
            "dropout":      trial.suggest_float("dropout", 0.1, 0.5),
            "seq_len":      trial.suggest_int("seq_len", 20, 120),
            "batch_size":   trial.suggest_categorical("batch_size", [32, 64, 128]),
            "optimizer":    trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        }

    @staticmethod
    def _default_params() -> Dict[str, Any]:
        return {
            "lr": 5e-4, "hidden_dim": 128, "num_layers": 2,
            "dropout": 0.2, "seq_len": 60, "batch_size": 64,
            "optimizer": "adamw",
        }
