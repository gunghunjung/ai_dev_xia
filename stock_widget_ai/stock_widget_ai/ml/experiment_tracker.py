"""
ExperimentTracker — 실험 결과를 timestamp 폴더로 저장
"""
from __future__ import annotations
import json
import os
import csv
from datetime import datetime
from typing import Any, Dict, Optional
from ..logger_config import get_logger

log = get_logger("ml.experiment")


class ExperimentTracker:
    def __init__(self, base_dir: str = "outputs/experiments") -> None:
        self._base = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self._exp_dir: Optional[str] = None
        self._record: Dict[str, Any] = {}

    def start(self, symbol: str, model_name: str, params: Dict) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{ts}_{symbol}_{model_name}"
        self._exp_dir = os.path.join(self._base, name)
        os.makedirs(self._exp_dir, exist_ok=True)
        self._record = {
            "timestamp": ts,
            "symbol": symbol,
            "model": model_name,
            "params": params,
        }
        log.info(f"실험 시작: {self._exp_dir}")
        return self._exp_dir

    def log_metrics(self, phase: str, metrics: Dict[str, float]) -> None:
        self._record.setdefault("metrics", {})[phase] = metrics

    def log_training_history(self, history: Dict) -> None:
        self._record["training_history"] = history
        if self._exp_dir:
            p = os.path.join(self._exp_dir, "training_history.json")
            with open(p, "w") as f:
                json.dump(history, f, indent=2)

    def save_predictions(self, df, filename: str = "predictions.csv") -> str:
        if not self._exp_dir:
            return ""
        path = os.path.join(self._exp_dir, filename)
        df.to_csv(path, index=True)
        self._record["prediction_path"] = path
        return path

    def save_backtest(self, result: Dict, filename: str = "backtest.json") -> str:
        if not self._exp_dir:
            return ""
        path = os.path.join(self._exp_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        self._record["backtest_path"] = path
        return path

    def finish(self, model_path: str = "") -> str:
        self._record["model_path"] = model_path
        if not self._exp_dir:
            return ""
        summary_path = os.path.join(self._exp_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self._record, f, indent=2, ensure_ascii=False, default=str)
        # 전역 CSV에도 한 줄 추가
        self._append_global_csv()
        log.info(f"실험 완료: {summary_path}")
        return summary_path

    def _append_global_csv(self) -> None:
        csv_path = os.path.join(self._base, "experiments.csv")
        flat = {
            "timestamp": self._record.get("timestamp"),
            "symbol":    self._record.get("symbol"),
            "model":     self._record.get("model"),
        }
        m = self._record.get("metrics", {})
        for phase, vals in m.items():
            for k, v in vals.items():
                flat[f"{phase}_{k}"] = v
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(flat.keys()))
            if write_header:
                w.writeheader()
            w.writerow(flat)

    @property
    def exp_dir(self) -> Optional[str]:
        return self._exp_dir
