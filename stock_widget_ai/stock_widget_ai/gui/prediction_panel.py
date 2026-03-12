"""
Prediction Panel — 예측 실행 + 결과 테이블 + export
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QGroupBox, QFormLayout,
)
from PyQt6.QtCore import pyqtSignal
import numpy as np
from typing import Optional, Dict


class PredictionPanel(QWidget):
    predict_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._result: Optional[Dict] = None
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        btn_row = QHBoxLayout()
        self._pred_btn   = QPushButton("🔮  예측 실행")
        self._export_btn = QPushButton("💾  CSV 저장")
        self._pred_btn.clicked.connect(self.predict_requested.emit)
        self._export_btn.clicked.connect(self._export_csv)
        self._export_btn.setEnabled(False)
        btn_row.addWidget(self._pred_btn)
        btn_row.addWidget(self._export_btn)
        layout.addLayout(btn_row)

        # 요약 박스
        grp = QGroupBox("예측 요약")
        form = QFormLayout()
        self._lbl_next   = QLabel("—")
        self._lbl_trend  = QLabel("—")
        self._lbl_conf   = QLabel("—")
        self._lbl_lower  = QLabel("—")
        self._lbl_upper  = QLabel("—")
        form.addRow("다음 예측값 (수익률)", self._lbl_next)
        form.addRow("방향",                self._lbl_trend)
        form.addRow("신뢰도 (std)",         self._lbl_conf)
        form.addRow("하한 (95% CI)",        self._lbl_lower)
        form.addRow("상한 (95% CI)",        self._lbl_upper)
        grp.setLayout(form)
        layout.addWidget(grp)

        # 결과 테이블
        layout.addWidget(QLabel("전체 예측 결과"))
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["Date", "Actual", "Pred", "Lower", "Upper"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table)

    def show_result(
        self,
        dates,
        actual: np.ndarray,
        pred: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> None:
        self._result = {"dates": dates, "actual": actual, "pred": pred,
                        "lower": lower, "upper": upper}
        # 요약 (마지막 예측)
        lp = pred[-1]
        ll = lower[-1]
        lu = upper[-1]
        self._lbl_next.setText(f"{lp:+.4f}")
        self._lbl_trend.setText("📈 상승 예측" if lp > 0 else "📉 하락 예측")
        std = (lu - ll) / (2 * 1.96) if lu != ll else 0
        self._lbl_conf.setText(f"{std:.4f}")
        self._lbl_lower.setText(f"{ll:+.4f}")
        self._lbl_upper.setText(f"{lu:+.4f}")

        # 테이블
        n = len(pred)
        self._table.setRowCount(n)
        for i in range(n):
            d = str(dates[i]) if i < len(dates) else ""
            a = f"{actual[i]:.4f}" if i < len(actual) else "—"
            self._table.setItem(i, 0, QTableWidgetItem(d))
            self._table.setItem(i, 1, QTableWidgetItem(a))
            self._table.setItem(i, 2, QTableWidgetItem(f"{pred[i]:.4f}"))
            self._table.setItem(i, 3, QTableWidgetItem(f"{lower[i]:.4f}"))
            self._table.setItem(i, 4, QTableWidgetItem(f"{upper[i]:.4f}"))
        self._export_btn.setEnabled(True)

    def _export_csv(self) -> None:
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "예측 결과 저장", "prediction.csv", "CSV (*.csv)")
        if not path:
            return
        import csv
        r = self._result
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["date", "actual", "pred", "lower", "upper"])
            for i in range(len(r["pred"])):
                w.writerow([
                    r["dates"][i] if i < len(r["dates"]) else "",
                    r["actual"][i] if i < len(r["actual"]) else "",
                    r["pred"][i], r["lower"][i], r["upper"][i],
                ])
