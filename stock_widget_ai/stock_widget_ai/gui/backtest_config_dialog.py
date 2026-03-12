"""
BacktestConfigDialog — 백테스트 실행 전 상세 설정 다이얼로그
"""
from __future__ import annotations
import pandas as pd
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QDateEdit, QDoubleSpinBox,
    QPushButton, QCheckBox, QFrame,
)
from PyQt6.QtCore import QDate, Qt
from typing import Optional, Dict
from ..state_schema import AppState


class BacktestConfigDialog(QDialog):
    """
    백테스트 실행 설정 다이얼로그.

    Parameters
    ----------
    state      : AppState — 현재 앱 상태 (기본값 읽기용)
    pred_dates : 전체 예측 날짜 범위 (DatetimeIndex)
    test_dates : 테스트 기간 날짜 범위 (DatetimeIndex, None 허용)
    """

    def __init__(
        self,
        state: AppState,
        pred_dates=None,
        test_dates=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("📊  백테스트 설정")
        self.setMinimumWidth(420)
        self.setModal(True)

        self._state      = state
        self._pred_dates = pred_dates   # 전체 예측 기간
        self._test_dates = test_dates   # 테스트 기간 (없으면 pred_dates와 동일)
        self._result: Optional[Dict] = None

        self._build()

    # ──────────────────────────────────────────────────────────────
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setSpacing(8)

        st = self._state.backtest

        # ── 가용 기간 정보 표시 ──────────────────────────────────
        info_box = QGroupBox("예측 데이터 가용 기간")
        info_layout = QFormLayout(info_box)
        info_layout.setVerticalSpacing(4)

        full_start = full_end = test_start = test_end = "—"
        if self._pred_dates is not None and len(self._pred_dates) > 0:
            full_start = str(self._pred_dates[0])[:10]
            full_end   = str(self._pred_dates[-1])[:10]
        if self._test_dates is not None and len(self._test_dates) > 0:
            test_start = str(self._test_dates[0])[:10]
            test_end   = str(self._test_dates[-1])[:10]

        lbl_full = QLabel(f"{full_start}  ~  {full_end}"
                          f"  ({len(self._pred_dates) if self._pred_dates is not None else 0}일)")
        lbl_full.setStyleSheet("color: #8b949e;")
        lbl_test = QLabel(f"{test_start}  ~  {test_end}"
                          f"  ({len(self._test_dates) if self._test_dates is not None else 0}일)  ← 학습에 미사용")
        lbl_test.setStyleSheet("color: #3fb950;")
        info_layout.addRow("전체 예측:",  lbl_full)
        info_layout.addRow("테스트 기간:", lbl_test)
        outer.addWidget(info_box)

        # ── 투자 기간 설정 ──────────────────────────────────────
        period_box = QGroupBox("투자 기간 설정")
        period_layout = QFormLayout(period_box)
        period_layout.setVerticalSpacing(6)

        # 날짜 범위 min/max 결정
        qmin = QDate(2000, 1, 1)
        qmax = QDate.currentDate()
        if self._pred_dates is not None and len(self._pred_dates) > 0:
            d0 = pd.Timestamp(self._pred_dates[0])
            d1 = pd.Timestamp(self._pred_dates[-1])
            qmin = QDate(d0.year, d0.month, d0.day)
            qmax = QDate(d1.year, d1.month, d1.day)

        # 기본값: 테스트 기간 시작 → 끝
        default_start = qmin
        default_end   = qmax
        if self._test_dates is not None and len(self._test_dates) > 0:
            ts = pd.Timestamp(self._test_dates[0])
            te = pd.Timestamp(self._test_dates[-1])
            default_start = QDate(ts.year, ts.month, ts.day)
            default_end   = QDate(te.year, te.month, te.day)

        self._start_de = QDateEdit(default_start)
        self._start_de.setMinimumDate(qmin)
        self._start_de.setMaximumDate(qmax)
        self._start_de.setDisplayFormat("yyyy-MM-dd")
        self._start_de.setCalendarPopup(True)

        self._end_de = QDateEdit(default_end)
        self._end_de.setMinimumDate(qmin)
        self._end_de.setMaximumDate(qmax)
        self._end_de.setDisplayFormat("yyyy-MM-dd")
        self._end_de.setCalendarPopup(True)

        self._test_only_chk = QCheckBox("테스트 기간 내에서만 선택 가능하게 제한")
        self._test_only_chk.setChecked(True)
        self._test_only_chk.setToolTip(
            "체크: 학습에 사용하지 않은 테스트 기간만 선택 (권장 — out-of-sample)\n"
            "해제: 전체 예측 기간에서 자유롭게 선택 (in-sample 포함, 결과 과장 가능)"
        )
        self._test_only_chk.toggled.connect(self._on_test_only_toggled)

        period_layout.addRow("시작일", self._start_de)
        period_layout.addRow("종료일", self._end_de)
        period_layout.addRow("",       self._test_only_chk)

        # 기간 안내 레이블
        self._period_info_lbl = QLabel()
        self._period_info_lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
        period_layout.addRow("", self._period_info_lbl)

        self._start_de.dateChanged.connect(self._update_period_info)
        self._end_de.dateChanged.connect(self._update_period_info)
        self._update_period_info()
        outer.addWidget(period_box)

        # ── 자본 설정 ────────────────────────────────────────────
        cap_box = QGroupBox("자본 및 비용 설정")
        cap_form = QFormLayout(cap_box)
        cap_form.setVerticalSpacing(6)

        def _dspin(val, lo, hi, step, dec):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi); sp.setDecimals(dec)
            sp.setSingleStep(step); sp.setValue(val)
            return sp

        self._cash_sp  = _dspin(st.initial_cash,    1000, 1e9, 1000000, 0)
        self._fee_sp   = _dspin(st.fee,              0,   0.01,  0.0001, 5)
        self._slip_sp  = _dspin(st.slippage,         0,   0.01,  0.0001, 5)
        cap_form.addRow("초기 자본 (원)", self._cash_sp)
        cap_form.addRow("수수료",         self._fee_sp)
        cap_form.addRow("슬리피지",       self._slip_sp)
        outer.addWidget(cap_box)

        # ── 전략 설정 ────────────────────────────────────────────
        strat_box = QGroupBox("거래 전략 설정")
        strat_form = QFormLayout(strat_box)
        strat_form.setVerticalSpacing(6)

        self._buy_sp  = _dspin(st.buy_threshold,  -0.1, 0.1, 0.001, 4)
        self._sell_sp = _dspin(st.sell_threshold, -0.1, 0.1, 0.001, 4)
        self._pos_sp  = _dspin(st.position_size,    0,  1.0,   0.1,  2)

        buy_hint  = QLabel("예측 수익률이 이 값 이상이면 매수")
        sell_hint = QLabel("예측 수익률이 이 값 이하이면 매도")
        for l in [buy_hint, sell_hint]:
            l.setStyleSheet("color:#8b949e; font-size:10px;")

        strat_form.addRow("매수 threshold", self._buy_sp)
        strat_form.addRow("",               buy_hint)
        strat_form.addRow("매도 threshold", self._sell_sp)
        strat_form.addRow("",               sell_hint)
        strat_form.addRow("포지션 크기",    self._pos_sp)
        outer.addWidget(strat_box)

        # ── 버튼 ────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #30363d;")
        outer.addWidget(sep)

        btn_row = QHBoxLayout()
        self._ok_btn  = QPushButton("📊  백테스트 실행")
        self._ok_btn.setDefault(True)
        cancel_btn = QPushButton("취소")
        self._ok_btn.clicked.connect(self._on_ok)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._ok_btn)
        btn_row.addWidget(cancel_btn)
        outer.addLayout(btn_row)

    def _on_test_only_toggled(self, checked: bool) -> None:
        if self._test_dates is not None and len(self._test_dates) > 0:
            ts = pd.Timestamp(self._test_dates[0])
            te = pd.Timestamp(self._test_dates[-1])
            qts = QDate(ts.year, ts.month, ts.day)
            qte = QDate(te.year, te.month, te.day)
            if checked:
                self._start_de.setMinimumDate(qts)
                self._start_de.setMaximumDate(qte)
                self._end_de.setMinimumDate(qts)
                self._end_de.setMaximumDate(qte)
                # 날짜를 테스트 기간으로 리셋
                if self._start_de.date() < qts or self._start_de.date() > qte:
                    self._start_de.setDate(qts)
                if self._end_de.date() < qts or self._end_de.date() > qte:
                    self._end_de.setDate(qte)
            else:
                if self._pred_dates is not None and len(self._pred_dates) > 0:
                    p0 = pd.Timestamp(self._pred_dates[0])
                    p1 = pd.Timestamp(self._pred_dates[-1])
                    self._start_de.setMinimumDate(QDate(p0.year, p0.month, p0.day))
                    self._start_de.setMaximumDate(QDate(p1.year, p1.month, p1.day))
                    self._end_de.setMinimumDate(QDate(p0.year, p0.month, p0.day))
                    self._end_de.setMaximumDate(QDate(p1.year, p1.month, p1.day))
        self._update_period_info()

    def _update_period_info(self) -> None:
        s = self._start_de.date()
        e = self._end_de.date()
        days = s.daysTo(e) + 1
        if days < 1:
            self._period_info_lbl.setText("⚠️  종료일이 시작일보다 앞에 있습니다")
            if hasattr(self, "_ok_btn"):
                self._ok_btn.setEnabled(False)
        else:
            self._period_info_lbl.setText(f"선택 기간: 약 {days}일 (캘린더 기준)")
            if hasattr(self, "_ok_btn"):
                self._ok_btn.setEnabled(True)

    def _on_ok(self) -> None:
        s = self._start_de.date()
        e = self._end_de.date()
        if s > e:
            return  # 버튼이 이미 비활성화되어 있지만 방어
        self._result = {
            "start_date":    f"{s.year():04d}-{s.month():02d}-{s.day():02d}",
            "end_date":      f"{e.year():04d}-{e.month():02d}-{e.day():02d}",
            "initial_cash":  self._cash_sp.value(),
            "fee":           self._fee_sp.value(),
            "slippage":      self._slip_sp.value(),
            "buy_threshold": self._buy_sp.value(),
            "sell_threshold":self._sell_sp.value(),
            "position_size": self._pos_sp.value(),
            "test_only":     self._test_only_chk.isChecked(),
        }
        self.accept()

    def get_params(self) -> Optional[Dict]:
        return self._result
