"""
DataPanel + FeaturePanel — 데이터 로드 & 피처 설정 패널
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QPushButton,
    QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QSpinBox, QHeaderView, QProgressBar, QDialog,
)
from PyQt6.QtCore import pyqtSignal, Qt
from typing import Callable, Optional
from ..state_schema import AppState


# ── 공통 스타일 헬퍼 ───────────────────────────────────────────────
def _lbl(text: str, bold: bool = False) -> QLabel:
    l = QLabel(text)
    if bold:
        l.setStyleSheet("font-weight: bold;")
    return l


# ════════════════════════════════════════════════════════════════
class DataPanel(QWidget):
    load_requested = pyqtSignal(str, str, str, str, bool)  # symbol, bm, period, interval, force

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        grp = QGroupBox("데이터 설정")
        form = QFormLayout()

        # ── 종목 코드 입력 + 검색 버튼 ───────────────────────────
        self._sym_edit = QLineEdit(self._state.data.symbol)
        self._sym_edit.setPlaceholderText("예) AAPL  /  005930.KS  /  ^GSPC")

        self._sym_search_btn = QPushButton("🔍")
        self._sym_search_btn.setFixedWidth(32)
        self._sym_search_btn.setFixedHeight(26)
        self._sym_search_btn.setToolTip("종목 검색 (이름으로 티커 찾기)")
        self._sym_search_btn.clicked.connect(self._open_symbol_search)

        sym_row = QHBoxLayout()
        sym_row.setSpacing(4)
        sym_row.addWidget(self._sym_edit)
        sym_row.addWidget(self._sym_search_btn)
        sym_widget = QWidget()
        sym_widget.setLayout(sym_row)

        # ── 벤치마크 입력 + 검색 버튼 ───────────────────────────
        self._bm_edit = QLineEdit(self._state.data.benchmark_symbol)
        self._bm_edit.setPlaceholderText("예) ^GSPC  /  ^KS11  /  ^IXIC")

        self._bm_search_btn = QPushButton("🔍")
        self._bm_search_btn.setFixedWidth(32)
        self._bm_search_btn.setFixedHeight(26)
        self._bm_search_btn.setToolTip("벤치마크 검색")
        self._bm_search_btn.clicked.connect(self._open_bm_search)

        bm_row = QHBoxLayout()
        bm_row.setSpacing(4)
        bm_row.addWidget(self._bm_edit)
        bm_row.addWidget(self._bm_search_btn)
        bm_widget = QWidget()
        bm_widget.setLayout(bm_row)

        self._period_cb = QComboBox()
        self._period_cb.addItems(["1y", "2y", "3y", "5y", "10y", "max"])
        self._period_cb.setCurrentText(self._state.data.period)
        self._intv_cb = QComboBox()
        self._intv_cb.addItems(["1d", "1wk", "1mo"])
        self._intv_cb.setCurrentText(self._state.data.interval)

        form.addRow("종목 코드", sym_widget)
        form.addRow("벤치마크",  bm_widget)
        form.addRow("기간",      self._period_cb)
        form.addRow("interval",  self._intv_cb)
        grp.setLayout(form)
        layout.addWidget(grp)

        btn_row = QHBoxLayout()
        self._load_btn   = QPushButton("📥  데이터 로드")
        self._reload_btn = QPushButton("🔄  강제 재다운로드")
        self._load_btn.clicked.connect(lambda: self._on_load(False))
        self._reload_btn.clicked.connect(lambda: self._on_load(True))
        btn_row.addWidget(self._load_btn)
        btn_row.addWidget(self._reload_btn)
        layout.addLayout(btn_row)

        # 데이터 미리보기 테이블
        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(["Date", "Open", "High", "Low", "Close", "Volume"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(_lbl("데이터 미리보기", bold=True))
        layout.addWidget(self._table)
        layout.addStretch()

    # ── 종목 검색 다이얼로그 ──────────────────────────────────────
    def _open_symbol_search(self) -> None:
        self._open_search_for(self._sym_edit)

    def _open_bm_search(self) -> None:
        self._open_search_for(self._bm_edit)

    def _open_search_for(self, target_edit: QLineEdit) -> None:
        from .stock_search_dialog import StockSearchDialog
        dlg = StockSearchDialog(self)
        # 현재 입력값을 초기 검색어로
        current = target_edit.text().strip()
        if current:
            dlg._input.setText(current)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.selected_ticker:
            target_edit.setText(dlg.selected_ticker)

    def _on_load(self, force: bool) -> None:
        sym = self._sym_edit.text().strip().upper()
        bm  = self._bm_edit.text().strip()
        per = self._period_cb.currentText()
        iv  = self._intv_cb.currentText()
        # 상태 업데이트
        self._state.data.symbol           = sym
        self._state.data.benchmark_symbol = bm
        self._state.data.period           = per
        self._state.data.interval         = iv
        if sym not in self._state.data.recent_symbols:
            self._state.data.recent_symbols.insert(0, sym)
            self._state.data.recent_symbols = self._state.data.recent_symbols[:10]
        self.load_requested.emit(sym, bm, per, iv, force)

    def fill_table(self, df) -> None:
        import pandas as pd
        sample = df.tail(50)
        self._table.setRowCount(len(sample))
        for row_i, (idx, row) in enumerate(sample.iterrows()):
            self._table.setItem(row_i, 0, QTableWidgetItem(str(idx.date() if hasattr(idx, "date") else idx)))
            for col_i, col in enumerate(["open", "high", "low", "close", "volume"]):
                val = row.get(col, "")
                self._table.setItem(row_i, col_i + 1,
                    QTableWidgetItem(f"{val:,.2f}" if isinstance(val, float) else str(val)))

    def apply_state(self, state: AppState) -> None:
        self._state = state
        self._sym_edit.setText(state.data.symbol)
        self._bm_edit.setText(state.data.benchmark_symbol)
        self._period_cb.setCurrentText(state.data.period)
        self._intv_cb.setCurrentText(state.data.interval)


# ════════════════════════════════════════════════════════════════
class FeaturePanel(QWidget):
    build_requested = pyqtSignal()   # 피처 생성 버튼 → MainWindow로 전달

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        grp = QGroupBox("피처 & 라벨 설정")
        form = QFormLayout()

        # Feature groups
        self._chk_price  = QCheckBox("가격/수익률")
        self._chk_ta     = QCheckBox("기술지표 (RSI/MACD/BB ...)")
        self._chk_market = QCheckBox("시장 비교 (벤치마크)")
        for chk, g in [(self._chk_price, "price"), (self._chk_ta, "ta"), (self._chk_market, "market")]:
            chk.setChecked(g in self._state.model.feature_groups)

        form.addRow("피처 그룹", self._chk_price)
        form.addRow("",         self._chk_ta)
        form.addRow("",         self._chk_market)

        # Target
        self._target_cb = QComboBox()
        self._target_cb.addItems(["return", "close", "direction"])
        self._target_cb.setCurrentText(self._state.model.target_type)
        form.addRow("예측 타깃", self._target_cb)

        # Sequence length
        self._seq_spin = QSpinBox()
        self._seq_spin.setRange(10, 250)
        self._seq_spin.setValue(self._state.model.sequence_length)
        form.addRow("시퀀스 길이", self._seq_spin)

        # Horizon
        self._hor_spin = QSpinBox()
        self._hor_spin.setRange(1, 30)
        self._hor_spin.setValue(self._state.model.prediction_horizon)
        form.addRow("예측 Horizon (일)", self._hor_spin)

        grp.setLayout(form)
        layout.addWidget(grp)

        # 피처 생성 버튼
        self._build_btn = QPushButton("🔧  피처 생성")
        self._build_btn.setToolTip(
            "설정된 그룹·타깃·시퀀스 기준으로 기술지표를 계산합니다.\n"
            "학습 시작 시 자동으로도 실행됩니다."
        )
        self._build_btn.clicked.connect(self._on_build)
        layout.addWidget(self._build_btn)

        # 결과 레이블 (생성된 피처 수 / 샘플 수 표시)
        self._info_lbl = QLabel("—")
        self._info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_lbl.setStyleSheet("color:#8b949e; font-size:11px;")
        layout.addWidget(self._info_lbl)

        layout.addStretch()

    def _on_build(self) -> None:
        self.sync_to_state(self._state)
        self._build_btn.setEnabled(False)
        self._build_btn.setText("🔧  생성 중...")
        self.build_requested.emit()

    def on_build_done(self, n_features: int, n_samples: int) -> None:
        self._build_btn.setEnabled(True)
        self._build_btn.setText("🔧  피처 생성")
        self._info_lbl.setText(f"✅  피처 {n_features}개  /  샘플 {n_samples}개")

    def on_build_error(self) -> None:
        self._build_btn.setEnabled(True)
        self._build_btn.setText("🔧  피처 생성")
        self._info_lbl.setText("❌  생성 실패 — 데이터를 먼저 로드하세요")

    def get_feature_groups(self):
        groups = []
        if self._chk_price.isChecked():  groups.append("price")
        if self._chk_ta.isChecked():     groups.append("ta")
        if self._chk_market.isChecked(): groups.append("market")
        return groups or ["price"]

    def sync_to_state(self, state: AppState) -> None:
        state.model.feature_groups     = self.get_feature_groups()
        state.model.target_type        = self._target_cb.currentText()
        state.model.sequence_length    = self._seq_spin.value()
        state.model.prediction_horizon = self._hor_spin.value()
