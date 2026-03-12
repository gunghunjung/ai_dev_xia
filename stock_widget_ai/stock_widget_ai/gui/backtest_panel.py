"""
Backtest Panel — 설정(左) / 투자시뮬레이션·성과지표·거래내역(右)
"""
from __future__ import annotations
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QDoubleSpinBox, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QSplitter,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor
from typing import Optional, Dict
from ..state_schema import AppState


class BacktestPanel(QWidget):
    backtest_requested = pyqtSignal(dict)

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state       = state
        self._result:     Optional[Dict] = None
        self._future:     Optional[Dict] = None
        self._pred_dates  = None   # 전체 예측 날짜 (DatetimeIndex)
        self._test_dates  = None   # 테스트 기간 날짜 (DatetimeIndex)
        self._build()

    # ── UI 구성 ─────────────────────────────────────────────────
    def _build(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # ── LEFT: 요약 + 버튼 ─────────────────────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        info_grp = QGroupBox("현재 설정 요약")
        info_form = QFormLayout(info_grp)
        info_form.setVerticalSpacing(4)

        self._period_lbl  = QLabel("—")
        self._cash_lbl    = QLabel("—")
        self._strat_lbl   = QLabel("—")
        for l in [self._period_lbl, self._cash_lbl, self._strat_lbl]:
            l.setStyleSheet("color: #8b949e; font-size: 11px;")
            l.setWordWrap(True)
        info_form.addRow("기간",   self._period_lbl)
        info_form.addRow("자본",   self._cash_lbl)
        info_form.addRow("전략",   self._strat_lbl)
        left_layout.addWidget(info_grp)

        hint = QLabel("💡 실행 버튼을 누르면 상세 설정\n   창이 열립니다.")
        hint.setStyleSheet("color: #484f58; font-size: 11px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(hint)

        self._run_btn    = QPushButton("📊  백테스트 실행")
        self._export_btn = QPushButton("💾  결과 저장")
        self._run_btn.clicked.connect(self._on_run)
        self._export_btn.clicked.connect(self._export)
        self._export_btn.setEnabled(False)
        left_layout.addWidget(self._run_btn)
        left_layout.addWidget(self._export_btn)
        left_layout.addStretch()

        # ── RIGHT: 결과 ────────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # 투자 시뮬레이션 (예측 기반)
        self._sim_grp = QGroupBox("💰  지금 투자하면? (예측 기반 시뮬레이션)")
        sim_layout = QFormLayout()
        sim_layout.setVerticalSpacing(4)

        self._sim_current_lbl = QLabel("—  (예측을 먼저 실행하세요)")
        self._sim_target_lbl  = QLabel("—")
        self._sim_pred_lbl    = QLabel("—")
        self._sim_ci_lbl      = QLabel("—")
        self._sim_profit_lbl  = QLabel("—")
        for l in [self._sim_current_lbl, self._sim_target_lbl,
                  self._sim_ci_lbl]:
            l.setStyleSheet("color: #c9d1d9;")

        sim_layout.addRow("현재가",          self._sim_current_lbl)
        sim_layout.addRow("예측 목표일",      self._sim_target_lbl)
        sim_layout.addRow("예측가 (수익률)",  self._sim_pred_lbl)
        sim_layout.addRow("95% CI",          self._sim_ci_lbl)
        sim_layout.addRow("예상 손익",        self._sim_profit_lbl)
        self._sim_grp.setLayout(sim_layout)
        right_layout.addWidget(self._sim_grp)

        # 성과 지표
        right_layout.addWidget(QLabel("📈  성과 지표  (백테스트 실행 후 표시)"))
        self._perf_table = QTableWidget(0, 2)
        self._perf_table.setHorizontalHeaderLabels(["지표", "값"])
        self._perf_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._perf_table.setMaximumHeight(190)
        right_layout.addWidget(self._perf_table)

        # 거래 내역
        right_layout.addWidget(QLabel("📋  거래 내역"))
        self._trade_table = QTableWidget(0, 6)
        self._trade_table.setHorizontalHeaderLabels(
            ["진입일", "청산일", "진입가", "청산가", "수량", "PnL"])
        self._trade_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right_layout.addWidget(self._trade_table, stretch=1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([180, 440])
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter)

    # ── 날짜 정보 주입 (predict 완료 후 호출) ────────────────────
    def set_prediction_dates(self, pred_dates, test_dates=None) -> None:
        """예측 결과의 날짜 범위를 설정 — 다이얼로그 날짜 범위 초기화에 사용"""
        self._pred_dates = pred_dates
        self._test_dates = test_dates

        # 요약 레이블 업데이트
        if pred_dates is not None and len(pred_dates) > 0:
            p0 = str(pred_dates[0])[:10]
            p1 = str(pred_dates[-1])[:10]
            if test_dates is not None and len(test_dates) > 0:
                t0 = str(test_dates[0])[:10]
                t1 = str(test_dates[-1])[:10]
                self._period_lbl.setText(
                    f"전체: {p0} ~ {p1}\n"
                    f"테스트: {t0} ~ {t1}"
                )
            else:
                self._period_lbl.setText(f"{p0} ~ {p1}")

    # ── 투자 시뮬레이션 업데이트 (predict 완료 후 호출) ──────────
    def set_future_forecast(self, future: Optional[Dict]) -> None:
        self._future = future
        if not future:
            self._sim_current_lbl.setText("—  (예측을 먼저 실행하세요)")
            self._sim_target_lbl.setText("—")
            self._sim_pred_lbl.setText("—")
            self._sim_ci_lbl.setText("—")
            self._sim_profit_lbl.setText("—")
            return

        last_close  = future.get("last_close", 0)
        pred_price  = future.get("pred_price", 0)
        lower_price = future.get("lower_price", 0)
        upper_price = future.get("upper_price", 0)
        pred_return = future.get("pred_return", 0)
        last_date   = str(future.get("last_date", ""))[:10]
        target_date = str(future.get("target_date", ""))[:10]
        horizon     = future.get("horizon", 0)
        initial     = self._state.backtest.initial_cash

        arrow  = "📈" if pred_return > 0 else "📉"
        color  = "#3fb950" if pred_return > 0 else "#f85149"
        profit = initial * pred_return
        denom  = max(last_close, 1)
        profit_l = initial * (lower_price / denom - 1)
        profit_u = initial * (upper_price / denom - 1)

        self._sim_current_lbl.setText(f"{last_close:,.0f}원  (기준일: {last_date})")
        self._sim_target_lbl.setText(f"{target_date}  ({horizon}거래일 후)")
        self._sim_pred_lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
        self._sim_pred_lbl.setText(f"{arrow}  {pred_price:,.0f}원  ({pred_return:+.2%})")
        self._sim_ci_lbl.setText(f"{lower_price:,.0f}원 ~ {upper_price:,.0f}원")
        self._sim_profit_lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
        self._sim_profit_lbl.setText(
            f"초기자본 {initial:,.0f}원 기준  →  {profit:+,.0f}원\n"
            f"(CI: {profit_l:+,.0f} ~ {profit_u:+,.0f}원)"
        )
        # 자본 요약 레이블도 갱신
        self._cash_lbl.setText(f"{initial:,.0f}원")

    # ── 실행 버튼 → 다이얼로그 열기 ─────────────────────────────
    def _on_run(self) -> None:
        from .backtest_config_dialog import BacktestConfigDialog
        dlg = BacktestConfigDialog(
            state=self._state,
            pred_dates=self._pred_dates,
            test_dates=self._test_dates,
            parent=self,
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        params = dlg.get_params()
        if not params:
            return

        # 요약 레이블 업데이트
        self._cash_lbl.setText(f"{params['initial_cash']:,.0f}원")
        self._strat_lbl.setText(
            f"매수≥{params['buy_threshold']:+.4f}  "
            f"매도≤{params['sell_threshold']:+.4f}  "
            f"포지션={params['position_size']:.0%}"
        )
        period_type = "테스트 기간" if params.get("test_only") else "전체 기간"
        self._period_lbl.setText(
            f"{params['start_date']} ~ {params['end_date']}\n({period_type})"
        )
        self.backtest_requested.emit(params)

    # ── 결과 표시 ────────────────────────────────────────────────
    def show_result(self, result: Dict) -> None:
        self._result = result
        perf = result.get("performance", {})

        _pct = lambda v: f"{v:.2%}"
        _f2  = lambda v: f"{v:.2f}"
        _int = lambda v: str(int(v))
        labels = {
            "total_return": ("총 수익률",     _pct),
            "cagr":         ("CAGR (연율)",   _pct),
            "sharpe":       ("Sharpe Ratio",  _f2),
            "sortino":      ("Sortino Ratio", _f2),
            "max_drawdown": ("최대 낙폭(MDD)", _pct),
            "win_rate":     ("승률",           _pct),
            "n_trades":     ("거래 횟수",      _int),
        }
        mode = result.get("mode", "")
        rows = [(lbl, fmt(perf.get(k, 0))) for k, (lbl, fmt) in labels.items()]
        if mode:
            rows.insert(0, ("기간", mode))
        # 백테스트 날짜 범위 표시
        dates = result.get("dates", [])
        if dates:
            rows.insert(0, ("실제 기간",
                            f"{str(dates[0])[:10]}  ~  {str(dates[-1])[:10]}"))

        self._perf_table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            item_k = QTableWidgetItem(k)
            item_v = QTableWidgetItem(v)
            if k in ("총 수익률", "CAGR (연율)"):
                try:
                    raw = perf.get("total_return" if k == "총 수익률" else "cagr", 0)
                    clr = QColor("#3fb950") if float(raw) >= 0 else QColor("#f85149")
                    item_v.setForeground(clr)
                except Exception:
                    pass
            self._perf_table.setItem(i, 0, item_k)
            self._perf_table.setItem(i, 1, item_v)

        trades = result.get("trades", [])
        self._trade_table.setRowCount(len(trades))
        for i, t in enumerate(trades):
            pnl = t.get("pnl", 0)
            for j, val in enumerate([
                t.get("entry", ""),
                t.get("exit", ""),
                f"{t.get('entry_price', 0):.0f}",
                f"{t.get('exit_price', 0):.0f}",
                f"{t.get('shares', 0):.2f}",
                f"{pnl:+,.0f}",
            ]):
                item = QTableWidgetItem(str(val))
                if j == 5:
                    item.setForeground(QColor("#3fb950") if pnl >= 0 else QColor("#f85149"))
                self._trade_table.setItem(i, j, item)

        self._export_btn.setEnabled(True)
        if self._future:
            self.set_future_forecast(self._future)

    def _export(self) -> None:
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "백테스트 저장", "backtest.json", "JSON (*.json)")
        if not path:
            return
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._result, f, indent=2, ensure_ascii=False, default=str)
