"""
ExplainPanel — AI 예측 설명 패널 (GUI)
──────────────────────────────────────────
탭 구성:
  1. 종합 판단 탭  : 예측 요약 + 신호 + 경고
  2. 피처 중요도   : SHAP/중요도 바 차트 (matplotlib)
  3. 다중 시계     : horizon별 예측 표
  4. 보고서        : 전체 텍스트 보고서
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QTextEdit, QTableWidget, QTableWidgetItem,
    QGroupBox, QPushButton, QHeaderView, QProgressBar,
    QSplitter, QScrollArea, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from ..logger_config import get_logger

log = get_logger("gui.explain_panel")


class _SignalWidget(QWidget):
    """종합 신호 표시 위젯"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._lbl_signal = QLabel("—")
        f = QFont()
        f.setPointSize(22)
        f.setBold(True)
        self._lbl_signal.setFont(f)
        self._lbl_signal.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._lbl_prob = QLabel("↑ — / ↓ —")
        self._lbl_prob.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._bar_up   = QProgressBar()
        self._bar_up.setRange(0, 100)
        self._bar_up.setFormat("상승 %p%")
        self._bar_up.setStyleSheet("QProgressBar::chunk{background:#2ea043;}")

        self._bar_dn   = QProgressBar()
        self._bar_dn.setRange(0, 100)
        self._bar_dn.setFormat("하락 %p%")
        self._bar_dn.setStyleSheet("QProgressBar::chunk{background:#da3633;}")

        for w in [self._lbl_signal, self._lbl_prob, self._bar_up, self._bar_dn]:
            layout.addWidget(w)

    def update(
        self, signal: str, prob_up: float, prob_down: float, point_ret: float
    ) -> None:
        self._lbl_signal.setText(signal)
        self._lbl_prob.setText(
            f"↑ {prob_up:.1%}  /  ↓ {prob_down:.1%}  |  예측수익률 {point_ret:+.2%}"
        )
        self._bar_up.setValue(int(prob_up * 100))
        self._bar_dn.setValue(int(prob_down * 100))

        # 배경색 변경
        color = self._signal_color(signal)
        self._lbl_signal.setStyleSheet(f"color: {color}; font-weight: bold;")

    @staticmethod
    def _signal_color(signal: str) -> str:
        if "매우 강세" in signal: return "#2ea043"
        if "강세"   in signal:   return "#56d364"
        if "중립"   in signal:   return "#8b949e"
        if "약세"   in signal:   return "#f85149"
        return "#da3633"


# ════════════════════════════════════════════════════════════════════════
class ExplainPanel(QWidget):
    """
    예측 설명 패널.
    ForecastEngine.forecast() 결과를 받아 시각화.
    """

    export_requested = pyqtSignal(str)   # 보고서 텍스트

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build()
        self._apply_style()

    # ── 빌드 ──────────────────────────────────────────────────────────────
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # 탭
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_summary_tab(),    "📊 종합 판단")
        self._tabs.addTab(self._build_importance_tab(), "🔬 피처 중요도")
        self._tabs.addTab(self._build_horizon_tab(),    "📅 다중 시계")
        self._tabs.addTab(self._build_report_tab(),     "📄 보고서")
        root.addWidget(self._tabs)

    def _build_summary_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        # 신호 박스
        signal_grp = QGroupBox("종합 신호")
        sl = QVBoxLayout(signal_grp)
        self._signal_widget = _SignalWidget()
        sl.addWidget(self._signal_widget)
        layout.addWidget(signal_grp)

        # 신뢰구간 표
        ci_grp = QGroupBox("예측 신뢰구간")
        cl = QVBoxLayout(ci_grp)
        self._ci_table = QTableWidget(5, 2)
        self._ci_table.setHorizontalHeaderLabels(["항목", "값"])
        self._ci_table.verticalHeader().setVisible(False)
        self._ci_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._ci_table.setMaximumHeight(160)
        cl.addWidget(self._ci_table)
        layout.addWidget(ci_grp)

        # 경고
        warn_grp = QGroupBox("⚠ 리스크 경고")
        wl = QVBoxLayout(warn_grp)
        self._warn_text = QTextEdit()
        self._warn_text.setReadOnly(True)
        self._warn_text.setMaximumHeight(120)
        wl.addWidget(self._warn_text)
        layout.addWidget(warn_grp)

        # 국면
        regime_grp = QGroupBox("시장 국면")
        rl = QVBoxLayout(regime_grp)
        self._lbl_regime = QLabel("—")
        self._lbl_regime.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_regime.setFont(QFont("", 12))
        rl.addWidget(self._lbl_regime)
        layout.addWidget(regime_grp)

        layout.addStretch()
        return w

    def _build_importance_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        # 피처 중요도 테이블
        self._imp_table = QTableWidget(0, 4)
        self._imp_table.setHorizontalHeaderLabels(["순위", "피처", "중요도", "기여 방향"])
        self._imp_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._imp_table.verticalHeader().setVisible(False)
        self._imp_table.setAlternatingRowColors(True)
        layout.addWidget(self._imp_table)

        # SHAP 설명 텍스트
        shap_grp = QGroupBox("SHAP 기반 설명")
        sl = QVBoxLayout(shap_grp)
        self._shap_text = QTextEdit()
        self._shap_text.setReadOnly(True)
        self._shap_text.setMaximumHeight(150)
        sl.addWidget(self._shap_text)
        layout.addWidget(shap_grp)

        return w

    def _build_horizon_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self._horizon_table = QTableWidget(0, 5)
        self._horizon_table.setHorizontalHeaderLabels(
            ["시계", "예측 수익률", "90% 하단", "90% 상단", "상승 확률"]
        )
        self._horizon_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._horizon_table.verticalHeader().setVisible(False)
        self._horizon_table.setAlternatingRowColors(True)
        layout.addWidget(self._horizon_table)

        note = QLabel(
            "ℹ 다중 시계 예측은 단일 시점 예측의 단순 외삽 근사입니다.\n"
            "장기 예측일수록 불확실성이 크게 증가합니다."
        )
        note.setStyleSheet("color: #8b949e; font-size: 11px;")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()
        return w

    def _build_report_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self._report_text = QTextEdit()
        self._report_text.setReadOnly(True)
        self._report_text.setFont(QFont("Courier New", 9))
        layout.addWidget(self._report_text)

        btn = QPushButton("📋  보고서 복사")
        btn.clicked.connect(self._copy_report)
        layout.addWidget(btn)
        return w

    # ── 데이터 업데이트 ───────────────────────────────────────────────────
    def update_forecast(
        self,
        fc:       Dict[str, Any],
        report:   str = "",
        shap_info: Optional[Dict[str, Any]] = None,
        importances: Optional[List[tuple]] = None,
    ) -> None:
        """ForecastEngine 결과로 전체 패널 갱신"""
        try:
            self._update_summary(fc)
            self._update_importance(shap_info, importances)
            self._update_horizon(fc)
            self._update_report(report, fc)
        except Exception as e:
            log.exception(f"ExplainPanel 업데이트 오류: {e}")

    def _update_summary(self, fc: Dict[str, Any]) -> None:
        signal     = fc.get("signal", "—")
        prob_up    = fc.get("prob_up",   0.5)
        prob_down  = fc.get("prob_down", 0.5)
        point_ret  = fc.get("point_return", 0.0)
        ci90_l     = fc.get("ci_90_lower", float("nan"))
        ci90_u     = fc.get("ci_90_upper", float("nan"))
        ci50_l     = fc.get("ci_50_lower", float("nan"))
        ci50_u     = fc.get("ci_50_upper", float("nan"))
        conf_l     = fc.get("conformal_lower", float("nan"))
        conf_u     = fc.get("conformal_upper", float("nan"))
        target_p   = fc.get("target_prob",    float("nan"))
        stop_p     = fc.get("stop_loss_prob", float("nan"))
        regime     = fc.get("regime_label", "—")

        self._signal_widget.update(signal, prob_up, prob_down, point_ret)

        # CI 테이블
        rows = [
            ("90% 신뢰구간",    f"[{ci90_l:+.2%}, {ci90_u:+.2%}]"),
            ("50% 신뢰구간",    f"[{ci50_l:+.2%}, {ci50_u:+.2%}]"),
            ("Conformal 구간",  f"[{conf_l:+.2%}, {conf_u:+.2%}]"),
            (f"목표가 도달 확률", f"{target_p:.1%}" if not self._nan(target_p) else "—"),
            (f"손절 이탈 확률",  f"{stop_p:.1%}"  if not self._nan(stop_p) else "—"),
        ]
        self._ci_table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self._ci_table.setItem(i, 0, QTableWidgetItem(k))
            self._ci_table.setItem(i, 1, QTableWidgetItem(v))

        # 경고
        warnings = fc.get("warnings", [])
        self._warn_text.setPlainText("\n".join(warnings) if warnings else "특이 경고 없음")

        # 국면
        self._lbl_regime.setText(regime)

    def _update_importance(
        self,
        shap_info:   Optional[Dict[str, Any]],
        importances: Optional[List[tuple]],
    ) -> None:
        items: List[tuple] = importances or []

        # SHAP shap_info 통합
        if shap_info and "top_positive" in shap_info:
            pos = shap_info.get("top_positive", [])
            neg = shap_info.get("top_negative", [])
            items = (
                [(n, v, "📈 상승")  for n, v in pos[:5]] +
                [(n, abs(v), "📉 하락") for n, v in neg[:5]]
            )
            self._shap_text.setPlainText(
                shap_info.get("explanation_text", "")
            )
        elif not items:
            self._shap_text.setPlainText("SHAP 분석 없음")

        self._imp_table.setRowCount(len(items))
        for row, item in enumerate(items):
            name = item[0] if len(item) > 0 else "—"
            val  = item[1] if len(item) > 1 else 0.0
            direction = item[2] if len(item) > 2 else "—"
            self._imp_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self._imp_table.setItem(row, 1, QTableWidgetItem(name))
            self._imp_table.setItem(row, 2, QTableWidgetItem(f"{val:.4f}"))
            self._imp_table.setItem(row, 3, QTableWidgetItem(direction))

    def _update_horizon(self, fc: Dict[str, Any]) -> None:
        mh = fc.get("multi_horizon", {})
        rows = sorted(mh.items())
        self._horizon_table.setRowCount(len(rows))
        for i, (h, info) in enumerate(rows):
            pt   = info.get("point",    float("nan"))
            lo   = info.get("lower_90", float("nan"))
            hi   = info.get("upper_90", float("nan"))
            pu   = info.get("prob_up",  0.5)
            self._horizon_table.setItem(i, 0, QTableWidgetItem(f"{h}일"))
            self._horizon_table.setItem(i, 1, QTableWidgetItem(f"{pt:+.2%}"))
            self._horizon_table.setItem(i, 2, QTableWidgetItem(f"{lo:+.2%}"))
            self._horizon_table.setItem(i, 3, QTableWidgetItem(f"{hi:+.2%}"))
            self._horizon_table.setItem(i, 4, QTableWidgetItem(f"{pu:.1%}"))

    def _update_report(self, report: str, fc: Dict[str, Any]) -> None:
        if report:
            self._report_text.setPlainText(report)
        else:
            disclaimer = fc.get("disclaimer", "")
            self._report_text.setPlainText(
                f"보고서 생성 중...\n\n{disclaimer}"
            )

    def _copy_report(self) -> None:
        from PyQt6.QtWidgets import QApplication
        text = self._report_text.toPlainText()
        QApplication.clipboard().setText(text)
        self.export_requested.emit(text)

    def clear(self) -> None:
        self._signal_widget.update("—", 0.5, 0.5, 0.0)
        self._warn_text.clear()
        self._report_text.clear()
        self._imp_table.setRowCount(0)
        self._horizon_table.setRowCount(0)
        self._shap_text.clear()
        self._lbl_regime.setText("—")

    # ── 스타일 ────────────────────────────────────────────────────────────
    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #30363d; }
            QTabBar::tab {
                background: #21262d; color: #8b949e;
                padding: 6px 14px; border: 1px solid #30363d;
            }
            QTabBar::tab:selected { background: #161b22; color: #c9d1d9; }
            QGroupBox {
                border: 1px solid #30363d; border-radius: 4px;
                margin-top: 6px; color: #8b949e; font-size: 11px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
            QTableWidget {
                background: #161b22; alternate-background-color: #1c2128;
                color: #c9d1d9; gridline-color: #30363d;
            }
            QTextEdit {
                background: #161b22; color: #c9d1d9;
                border: 1px solid #30363d;
            }
            QProgressBar {
                background: #21262d; border: 1px solid #30363d;
                border-radius: 3px; text-align: center; color: #c9d1d9;
            }
        """)

    @staticmethod
    def _nan(v: Any) -> bool:
        try:
            import math
            return math.isnan(float(v))
        except Exception:
            return True
