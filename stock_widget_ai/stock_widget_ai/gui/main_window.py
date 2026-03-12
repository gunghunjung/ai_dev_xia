"""
MainWindow — PyQt6 메인 창
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QTabWidget,
    QVBoxLayout, QHBoxLayout, QStatusBar, QMessageBox, QSizeGrip, QPushButton,
    QScrollArea, QProgressBar, QLabel,
)
from PyQt6.QtCore import Qt, QThread, QTimer
from PyQt6.QtGui import QCloseEvent, QResizeEvent

from ..app_controller import AppController
from ..config_manager import ConfigManager
from ..logger_config import get_logger
from .panels import DataPanel, FeaturePanel
from .training_panel import TrainingPanel
from .prediction_panel import PredictionPanel
from .backtest_panel import BacktestPanel
from .settings_panel import SettingsPanel
from .chart_widget import ChartWidget
from .worker_threads import WorkerThread
from .help_dialog import HelpDialog

log = get_logger("gui.main")

_STYLE_DARK = """
QMainWindow, QWidget { background: #0d1117; color: #c9d1d9; }
QGroupBox { border: 1px solid #30363d; border-radius: 4px; margin-top: 8px; padding-top: 8px; }
QGroupBox::title { color: #8b949e; subcontrol-origin: margin; left: 8px; }
QPushButton { background: #21262d; border: 1px solid #30363d; border-radius: 4px;
              padding: 4px 10px; color: #c9d1d9; }
QPushButton:hover  { background: #30363d; }
QPushButton:pressed { background: #161b22; }
QPushButton:disabled { color: #484f58; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
  background: #161b22; border: 1px solid #30363d; border-radius: 3px;
  padding: 2px 6px; color: #c9d1d9; }
QTabBar::tab { background: #161b22; color: #8b949e; padding: 6px 14px;
               border: 1px solid #30363d; border-bottom: none; }
QTabBar::tab:selected { background: #21262d; color: #c9d1d9; }
QTableWidget { background: #0d1117; gridline-color: #21262d; color: #c9d1d9; }
QHeaderView::section { background: #161b22; color: #8b949e; border: 1px solid #30363d; }
QScrollBar:vertical { background: #0d1117; width: 8px; }
QScrollBar::handle:vertical { background: #30363d; border-radius: 4px; }
QTextEdit { background: #161b22; color: #c9d1d9; border: 1px solid #30363d; }
QProgressBar { background: #161b22; border: 1px solid #30363d; border-radius: 3px; height: 6px; }
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                        stop:0 #238636, stop:0.5 #388bfd, stop:1 #238636); }
QCheckBox { color: #c9d1d9; }
QSplitter::handle { background: #21262d; }
QSplitter::handle:hover { background: #388bfd; }
QStatusBar { background: #161b22; color: #8b949e; }
QStatusBar QLabel { color: #f0883e; font-weight: bold; }
QSizeGrip { background: transparent; width: 16px; height: 16px; }
"""


class MainWindow(QMainWindow):
    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__()
        self._cfg  = cfg
        self._ctrl = AppController(cfg)
        self._worker: WorkerThread | None = None
        self._setup_ui()
        self.setStyleSheet(_STYLE_DARK if cfg.state.chart.dark_mode else "")
        # apply_to_window은 show() 이후에 호출해야 정확히 적용됨
        # → main.py에서 show() 직후에 호출. 여기선 기본 크기만 설정
        self.resize(cfg.state.window.width, cfg.state.window.height)
        log.info("MainWindow 초기화 완료")

    # ════════════════════════════════════════════════════════════
    #  UI 구성
    # ════════════════════════════════════════════════════════════
    @staticmethod
    def _scroll_wrap(widget: QWidget) -> QScrollArea:
        """패널을 스크롤 가능하게 감쌈 — 창 크기와 무관하게 내용이 잘리지 않음"""
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sa.setWidget(widget)
        # 스크롤 영역 자체에 최소 너비를 주지 않음 → 탭 위젯이 결정
        sa.setContentsMargins(0, 0, 0, 0)
        return sa

    def _setup_ui(self) -> None:
        self.setWindowTitle("Stock Widget AI — 주가 예측 플랫폼")
        self.setMinimumSize(900, 550)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── 왼쪽: 설정 탭 (스크롤 포함) ──────────────────────────
        self._left_tabs = QTabWidget()
        self._left_tabs.setMinimumWidth(300)
        self._left_tabs.setMaximumWidth(520)   # 넓은 화면에서 조금 더 여유 있게

        state = self._cfg.state
        self._data_panel     = DataPanel(state)
        self._feat_panel     = FeaturePanel(state)
        self._train_panel    = TrainingPanel(state)
        self._settings_panel = SettingsPanel(state)

        # 각 패널을 QScrollArea로 감싸서 창 높이가 작아져도 스크롤 가능
        self._left_tabs.addTab(self._scroll_wrap(self._data_panel),     "📊 데이터")
        self._left_tabs.addTab(self._scroll_wrap(self._feat_panel),     "🔧 피처")
        self._left_tabs.addTab(self._scroll_wrap(self._train_panel),    "🧠 학습")
        self._left_tabs.addTab(self._scroll_wrap(self._settings_panel), "⚙️ 설정")

        # ── 오른쪽: 차트 + 결과 탭 ───────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        self._chart = ChartWidget(dark_mode=state.chart.dark_mode)
        # chart가 확장 가능하도록 SizePolicy 설정
        from PyQt6.QtWidgets import QSizePolicy
        self._chart.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        right_layout.addWidget(self._chart, stretch=2)   # 차트 비중 축소 (3→2)

        self._right_tabs = QTabWidget()
        self._right_tabs.setMinimumHeight(240)   # 최소 높이 확보
        self._pred_panel = PredictionPanel()
        self._bt_panel   = BacktestPanel(state)
        self._right_tabs.addTab(self._scroll_wrap(self._pred_panel), "🔮 예측")
        self._right_tabs.addTab(self._scroll_wrap(self._bt_panel),   "📈 백테스트")
        right_layout.addWidget(self._right_tabs, stretch=2)

        self._splitter.addWidget(self._left_tabs)
        self._splitter.addWidget(right_widget)
        # 좌측: 고정 폭(stretch=0), 우측: 남은 공간 모두 차지(stretch=1)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        # 초기 분할: 좌 380px, 우 나머지
        self._splitter.setSizes([380, 1020])
        main_layout.addWidget(self._splitter)

        # ── 차트 상태 캐시 (탭 전환 시 각자 차트 복원) ──────────
        self._chart_cache: dict = {}   # {0: ("prediction", data...), 1: ("equity", data...)}
        self._right_tabs.currentChanged.connect(self._on_right_tab_changed)

        # ── 상태바 + 도움말 버튼 + 우하단 대각선 리사이즈 그립 ──────
        self._status = QStatusBar()
        self._status.setSizeGripEnabled(True)   # ◢ 우하단 대각선 리사이즈

        # ── 작업 진행 표시 위젯 ────────────────────────────────────
        # 왼쪽: 스피너 아이콘 + 작업명 레이블
        self._busy_lbl = QLabel()
        self._busy_lbl.setVisible(False)
        self._status.addWidget(self._busy_lbl)   # 왼쪽 영역

        # 우측 상단에 고정: 인디케이터 바 (range 0,0 = 무한 애니메이션)
        self._busy_bar = QProgressBar()
        self._busy_bar.setRange(0, 0)            # indeterminate 모드
        self._busy_bar.setFixedWidth(160)
        self._busy_bar.setFixedHeight(6)
        self._busy_bar.setTextVisible(False)
        self._busy_bar.setVisible(False)
        self._status.addPermanentWidget(self._busy_bar)

        # 스피너 텍스트 회전용 타이머
        self._spinner_chars = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        self._spinner_idx   = 0
        self._spinner_label = ""
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(80)
        self._spinner_timer.timeout.connect(self._tick_spinner)

        # 상태바 우측에 도움말 버튼 고정
        help_btn = QPushButton("📖 도움말")
        help_btn.setFixedWidth(80)
        help_btn.setToolTip("AI 예측 메커니즘 및 각 컨트롤 설명 (F1)")
        help_btn.clicked.connect(self._on_help)
        self._status.addPermanentWidget(help_btn)
        self.setStatusBar(self._status)
        self._status.showMessage("준비  |  F1: 도움말")

        # ── 시그널 연결 ───────────────────────────────────────────
        self._data_panel.load_requested.connect(self._on_load_data)
        self._feat_panel.build_requested.connect(self._on_build_features)
        self._train_panel.train_requested.connect(self._on_train)
        self._train_panel.stop_requested.connect(self._on_stop)
        self._pred_panel.predict_requested.connect(self._on_predict)
        self._bt_panel.backtest_requested.connect(self._on_backtest)
        self._settings_panel.reset_requested.connect(self._on_reset_settings)

    # ════════════════════════════════════════════════════════════
    #  바쁨 표시 헬퍼
    # ════════════════════════════════════════════════════════════
    def _set_busy(self, busy: bool, label: str = "") -> None:
        """작업 진행 중 / 완료 상태 전환 — progress bar + 스피너 + 버튼 잠금"""
        self._busy_bar.setVisible(busy)
        self._busy_lbl.setVisible(busy)
        if busy:
            self._spinner_label = label
            self._spinner_idx   = 0
            self._tick_spinner()
            self._spinner_timer.start()
            # 학습 중에는 왼쪽 탭 유지 (로그 확인 필요)
            # 데이터·피처·예측·백테스트는 좌측 입력 잠금
            if label != "학습":
                self._left_tabs.setEnabled(False)
        else:
            self._spinner_timer.stop()
            self._busy_lbl.setText("")
            self._left_tabs.setEnabled(True)

    def _tick_spinner(self) -> None:
        ch = self._spinner_chars[self._spinner_idx % len(self._spinner_chars)]
        self._busy_lbl.setText(f"{ch}  {self._spinner_label} 중…")
        self._spinner_idx += 1

    # ════════════════════════════════════════════════════════════
    #  슬롯
    # ════════════════════════════════════════════════════════════
    def _on_load_data(self, sym, bm, per, iv, force) -> None:
        self._set_busy(True, f"데이터 로드 ({sym})")
        self._status.showMessage(f"데이터 로드 중: {sym} …")

        def _load():
            return self._ctrl.load_data(sym, bm, per, iv, force)

        self._run_worker(_load, self._on_load_done, "데이터 로드")

    def _on_build_features(self) -> None:
        """피처 패널의 '피처 생성' 버튼 → 독립 실행"""
        self._set_busy(True, "피처 생성")
        self._status.showMessage("피처 생성 중 …")
        st = self._cfg.state.model

        def _build():
            return self._ctrl.build_features(
                feature_groups=st.feature_groups,
                target_type=st.target_type,
                horizon=st.prediction_horizon,
            )

        def _done(result):
            self._set_busy(False)
            if result is None:
                self._feat_panel.on_build_error()
                self._status.showMessage("피처 생성 실패")
                return
            X, y = result
            self._feat_panel.on_build_done(X.shape[1], len(X))
            self._status.showMessage(
                f"피처 생성 완료 — {X.shape[1]}개 피처 / {len(X)}개 샘플"
            )

        self._run_worker(_build, _done, "피처 생성")

    def _on_load_done(self, df) -> None:
        self._set_busy(False)
        if df is None or df.empty:
            self._show_error("데이터를 가져오지 못했습니다")
            return
        self._data_panel.fill_table(df)
        self._feat_panel.sync_to_state(self._cfg.state)
        self._status.showMessage(f"로드 완료: {len(df)}행")
        # 차트에 가격 표시
        self._chart.plot_price_and_prediction(
            df["close"], df.index[:0], np.array([]),
            title=f"{self._cfg.state.data.symbol} — 종가"
        )

    def _on_train(self, params: dict) -> None:
        # 피처 먼저 생성
        try:
            self._feat_panel.sync_to_state(self._cfg.state)
            st = self._cfg.state.model
            self._ctrl.build_features(
                feature_groups=st.feature_groups,
                target_type=st.target_type,
                horizon=st.prediction_horizon,
            )
        except Exception as e:
            self._show_error(str(e))
            self._train_panel.on_train_done()
            return

        if self._worker and self._worker.isRunning():
            self._status.showMessage("이미 '학습' 작업 진행 중")
            return

        self._set_busy(True, "학습")
        self._status.showMessage("학습 중 …")

        # WorkerThread의 log_msg/progress 시그널을 통해 GUI 메서드를 메인 스레드에서
        # 호출하도록 연결 — 워커 스레드에서 GUI 메서드를 직접 호출하면
        # "recursive repaint" 경고가 발생하므로 emit_log/emit_progress를 콜백으로 전달
        worker = WorkerThread(lambda: self._ctrl.train(
            params,
            on_log=worker.emit_log,
            on_progress=worker.emit_progress,
        ))
        worker.log_msg.connect(self._train_panel.append_log)
        worker.progress.connect(self._train_panel.set_progress)
        worker.finished.connect(self._on_train_done)
        worker.error.connect(lambda e: self._on_worker_error(e, "학습"))
        self._worker = worker
        self._worker.start()

    def _on_train_done(self, history) -> None:
        self._set_busy(False)
        self._train_panel.on_train_done()
        if history:
            n_ep = len(history.get("train_loss", []))
            self._status.showMessage(f"학습 완료 ({n_ep}epoch)")
        else:
            self._status.showMessage("학습 완료")

    def _on_stop(self) -> None:
        self._ctrl.stop_training()
        if self._worker:
            self._worker.request_stop()
        self._set_busy(False)
        self._status.showMessage("중단됨")

    def _on_predict(self) -> None:
        self._set_busy(True, "AI 예측")
        self._status.showMessage("예측 중 …")

        def _pred():
            return self._ctrl.predict()

        self._run_worker(_pred, self._on_predict_done, "예측")

    def _on_predict_done(self, result) -> None:
        self._set_busy(False)
        if result is None:
            self._show_error("예측 실패")
            return
        df  = self._ctrl.df
        st  = self._cfg.state
        idx = result["dates"]

        # ── 콘솔 로그 출력 (디버깅용) ────────────────────────────
        m = result.get("metrics", {})
        log.info("=" * 60)
        log.info(f"[PREDICT] 심볼={st.data.symbol}  타깃={st.model.target_type}")
        log.info(f"[PREDICT] 총 예측 샘플: {len(result['pred'])}")
        log.info(f"[PREDICT] 메트릭: { {k: round(float(v),6) for k,v in m.items()} }")
        # 마지막 10개 예측값 출력
        n_show = min(10, len(result["pred"]))
        log.info(f"[PREDICT] 최근 {n_show}개 예측 결과 (actual / pred / lower / upper):")
        for i in range(-n_show, 0):
            dt  = str(idx[i])[:10]
            act = float(result["actual"][i])
            prd = float(result["pred"][i])
            lo  = float(result["lower"][i])
            hi  = float(result["upper"][i])
            log.info(f"  {dt}  actual={act:+.6f}  pred={prd:+.6f}"
                     f"  CI=[{lo:+.6f}, {hi:+.6f}]")
        log.info("=" * 60)

        # ── 차트 ─────────────────────────────────────────────────
        close_ser = df["close"] if df is not None else pd.Series(dtype=float)
        future    = result.get("future")   # 미래 1스텝 예측 dict

        # 예측 차트 데이터를 캐시에 저장 (탭 전환 시 복원용)
        self._chart_cache[0] = {
            "type": "prediction",
            "price": close_ser,
            "result": result,
            "target_type": st.model.target_type,
            "title": f"{st.data.symbol} — Prediction ({st.model.target_type})",
            "future": future,
        }
        self._chart.plot_prediction_result(
            close_ser,
            result,
            target_type=st.model.target_type,
            title=f"{st.data.symbol} — Prediction ({st.model.target_type})",
            future=future,
        )

        # ── 예측 패널 ────────────────────────────────────────────
        self._pred_panel.show_result(
            list(idx), result["actual"],
            result["pred"], result["lower"], result["upper"],
        )

        # ── 백테스트 패널에 날짜 범위 + 투자 시뮬레이션 업데이트 ──
        ctrl = self._ctrl
        pred_dates = result["dates"]   # 전체 예측 날짜 DatetimeIndex
        # 테스트 기간 날짜 계산 (n_train 오프셋 기반)
        test_dates = None
        seq_len  = st.model.sequence_length
        n_train  = getattr(ctrl, "_n_train", 0)
        t_offset = n_train - seq_len
        if 0 < t_offset < len(pred_dates):
            test_dates = pred_dates[t_offset:]
        self._bt_panel.set_prediction_dates(pred_dates, test_dates)
        self._bt_panel.set_future_forecast(future)

        self._right_tabs.setCurrentIndex(0)   # 🔮 예측 탭으로 자동 전환

        # 상태바: 메트릭 + 미래 예측 요약
        status_msg = "예측 완료 | " + " | ".join(
            f"{k}={v:.4f}" for k, v in list(m.items())[:3]
        )
        if future:
            fp = future["pred_price"]
            fr = future["pred_return"]
            fd = str(future["target_date"])[:10]
            arrow = "📈" if fr > 0 else "📉"
            status_msg += f"  ‖  {arrow} {fd} 예측={fp:,.0f}원 ({fr:+.2%})"
        self._status.showMessage(status_msg)

    def _on_backtest(self, params: dict) -> None:
        self._set_busy(True, "백테스트")
        self._status.showMessage("백테스트 실행 중 …")

        def _bt():
            return self._ctrl.backtest(params)

        self._run_worker(_bt, self._on_bt_done, "백테스트")

    def _on_bt_done(self, result) -> None:
        self._set_busy(False)
        if result is None:
            self._show_error("백테스트 실패")
            return
        self._bt_panel.show_result(result)

        # 백테스트 차트 데이터를 캐시에 저장 (탭 전환 시 복원용)
        eq_title = f"{self._cfg.state.data.symbol} — Equity Curve"
        self._chart_cache[1] = {
            "type": "equity",
            "equity_curve": result["equity_curve"],
            "dates": result["dates"],
            "title": eq_title,
        }
        self._chart.plot_equity_curve(
            result["equity_curve"], result["dates"], title=eq_title
        )
        self._right_tabs.setCurrentIndex(1)   # 📈 백테스트 탭으로 자동 전환
        perf = result.get("performance", {})
        self._status.showMessage(
            f"백테스트 완료 | "
            f"수익률={perf.get('total_return',0):.1%}  "
            f"CAGR={perf.get('cagr',0):.1%}  "
            f"Sharpe={perf.get('sharpe',0):.2f}  "
            f"MDD={perf.get('max_drawdown',0):.1%}  "
            f"승률={perf.get('win_rate',0):.0%}  "
            f"거래={perf.get('n_trades',0):.0f}회"
        )

    def _on_right_tab_changed(self, idx: int) -> None:
        """오른쪽 탭(예측/백테스트)이 전환될 때 해당 탭의 차트를 복원"""
        cache = self._chart_cache.get(idx)
        if not cache:
            return
        if cache["type"] == "prediction":
            self._chart.plot_prediction_result(
                cache["price"],
                cache["result"],
                target_type=cache["target_type"],
                title=cache["title"],
                future=cache["future"],
            )
        elif cache["type"] == "equity":
            self._chart.plot_equity_curve(
                cache["equity_curve"],
                cache["dates"],
                title=cache["title"],
            )

    def _on_help(self) -> None:
        # 모달리스: 이미 열려 있으면 앞으로만 가져옴, 없으면 새로 생성
        if not hasattr(self, "_help_dlg") or self._help_dlg is None:
            from PyQt6.QtCore import Qt
            self._help_dlg = HelpDialog(self)
            self._help_dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            self._help_dlg.destroyed.connect(lambda: setattr(self, "_help_dlg", None))
        self._help_dlg.show()
        self._help_dlg.raise_()
        self._help_dlg.activateWindow()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """창 크기가 변할 때 좌측 패널 폭을 보존 — 전체화면에서도 차트가 공간을 다 차지하도록"""
        super().resizeEvent(event)
        # 현재 좌측 폭을 유지하고 나머지를 우측에 부여
        sizes = self._splitter.sizes()
        if sizes and len(sizes) == 2:
            left_w = max(300, min(520, sizes[0]))   # 좌: min/max 범위 내 고정
            total  = self._splitter.width()
            right_w = max(400, total - left_w)
            self._splitter.setSizes([left_w, right_w])

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_F1:
            self._on_help()
        else:
            super().keyPressEvent(event)

    def _on_reset_settings(self) -> None:
        from ..state_schema import AppState
        self._cfg._state = AppState()
        self._cfg.safe_save()
        self._status.showMessage("설정이 초기화되었습니다. 재시작 시 적용됩니다.")

    # ════════════════════════════════════════════════════════════
    #  Worker 헬퍼
    # ════════════════════════════════════════════════════════════
    def _run_worker(self, fn, on_done, label: str) -> None:
        if self._worker and self._worker.isRunning():
            self._status.showMessage(f"이미 '{label}' 작업 진행 중")
            return
        self._worker = WorkerThread(fn)
        self._worker.finished.connect(on_done)
        self._worker.error.connect(lambda e: self._on_worker_error(e, label))
        self._worker.start()

    def _on_worker_error(self, msg: str, label: str) -> None:
        self._set_busy(False)
        log.error(f"[{label}] {msg}")
        self._show_error(f"[{label}] 오류:\n{msg}")
        self._train_panel.on_train_done()
        self._status.showMessage(f"오류 발생: {label}")

    def _show_error(self, msg: str) -> None:
        QMessageBox.critical(self, "오류", msg)

    # ════════════════════════════════════════════════════════════
    #  종료 → 설정 저장
    # ════════════════════════════════════════════════════════════
    def closeEvent(self, event: QCloseEvent) -> None:
        # 백그라운드 작업 중단
        if self._worker and self._worker.isRunning():
            self._ctrl.stop_training()
            self._worker.request_stop()
            self._worker.wait(2000)   # 최대 2초 대기

        # 패널 상태 → state 동기화
        self._settings_panel.sync_to_state(self._cfg.state)

        # ★ 창 geometry 저장 (saveGeometry 방식)
        self._cfg.update_from_window(self)
        self._cfg.safe_save()
        log.info(
            f"설정 저장 완료 — "
            f"{self._cfg.state.window.width}×{self._cfg.state.window.height} "
            f"@ ({self._cfg.state.window.x},{self._cfg.state.window.y})"
        )
        super().closeEvent(event)
