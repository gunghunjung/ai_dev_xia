from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QSlider, QSizePolicy, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def apply_dark_axes(ax):
    """matplotlib 축에 다크 테마 적용"""
    ax.set_facecolor('#141420')
    ax.tick_params(colors='#888aaa')
    ax.spines['bottom'].set_color('#2a2a45')
    ax.spines['left'].set_color('#2a2a45')
    ax.spines['top'].set_color('#2a2a45')
    ax.spines['right'].set_color('#2a2a45')
    ax.xaxis.label.set_color('#888aaa')
    ax.yaxis.label.set_color('#888aaa')
    ax.title.set_color('#e0e0ff')


def make_dark_figure(nrows=1, ncols=1):
    """다크 테마 Figure 생성"""
    fig = Figure(facecolor='#0a0a0f')
    axes = []
    for i in range(nrows * ncols):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        apply_dark_axes(ax)
        axes.append(ax)
    if nrows * ncols == 1:
        return fig, axes[0]
    return fig, axes


class SliderWidget(QWidget):
    """레이블 + 슬라이더 + 값 표시 복합 위젯"""

    def __init__(self, label, min_val, max_val, default, decimals=1, parent=None):
        super().__init__(parent)
        self.decimals = decimals
        self.min_val = min_val
        self.max_val = max_val
        self._scale = 10 ** decimals

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(2)

        top_row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #888aaa; font-size: 12px;")
        self.val_label = QLabel(f"{default:.{decimals}f}")
        self.val_label.setStyleSheet("color: #e0e0ff; font-size: 12px; font-weight: bold;")
        top_row.addWidget(lbl)
        top_row.addStretch()
        top_row.addWidget(self.val_label)
        layout.addLayout(top_row)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(min_val * self._scale))
        self.slider.setMaximum(int(max_val * self._scale))
        self.slider.setValue(int(default * self._scale))
        self.slider.valueChanged.connect(self._on_change)
        layout.addWidget(self.slider)

        self._callbacks = []

    def _on_change(self, int_val):
        val = int_val / self._scale
        self.val_label.setText(f"{val:.{self.decimals}f}")
        for cb in self._callbacks:
            cb(val)

    def value(self):
        return self.slider.value() / self._scale

    def connect(self, callback):
        self._callbacks.append(callback)


class ModulePage(QWidget):
    """모든 모듈 페이지의 기본 클래스"""

    TIMER_INTERVAL = 50  # ms

    def __init__(self, title: str, color: str = '#6c63ff', parent=None):
        super().__init__(parent)
        self.title = title
        self.color = color
        self._animating = False
        self._anim_frame = 0
        self._timer = QTimer(self)
        self._timer.setInterval(self.TIMER_INTERVAL)
        self._timer.timeout.connect(self._on_timer)

        self._build_ui()
        self.setup_module()
        self.update_plot()

    # ------------------------------------------------------------------
    # UI 구성
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # 좌측: matplotlib 캔버스
        canvas_frame = QFrame()
        canvas_frame.setStyleSheet(
            "QFrame { background: #0e0e1a; border: 1px solid #2a2a45; border-radius: 12px; }"
        )
        canvas_layout = QVBoxLayout(canvas_frame)
        canvas_layout.setContentsMargins(8, 8, 8, 8)

        self.figure, self.ax = self._create_figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background: transparent;")
        canvas_layout.addWidget(self.canvas)

        root.addWidget(canvas_frame, stretch=3)

        # 우측: 파라미터 + 정보
        right_panel = QWidget()
        right_panel.setFixedWidth(280)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # 모듈 제목
        title_lbl = QLabel(self.title)
        title_lbl.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title_lbl.setStyleSheet(f"color: {self.color}; margin-bottom: 8px;")
        right_layout.addWidget(title_lbl)

        # 스크롤 영역 (슬라이더 & 컨트롤)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        self.ctrl_layout = QVBoxLayout(scroll_content)
        self.ctrl_layout.setContentsMargins(0, 0, 8, 0)
        self.ctrl_layout.setSpacing(6)
        scroll.setWidget(scroll_content)
        right_layout.addWidget(scroll, stretch=1)

        # 정보 표시 패널
        info_frame = QFrame()
        info_frame.setStyleSheet(
            "QFrame { background: #1e1e2e; border: 1px solid #2a2a45; border-radius: 8px; padding: 4px; }"
        )
        self.info_layout = QVBoxLayout(info_frame)
        self.info_layout.setContentsMargins(10, 8, 10, 8)
        self.info_layout.setSpacing(4)
        right_layout.addWidget(info_frame)

        # 하단 버튼 (재생/정지)
        btn_row = QHBoxLayout()
        self.play_btn = QPushButton("▶  재생")
        self.play_btn.setStyleSheet(
            f"QPushButton {{ background: {self.color}; border: none; color: #fff; "
            f"padding: 8px 16px; border-radius: 8px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: #8880ff; }}"
        )
        self.play_btn.clicked.connect(self._toggle_animation)
        self.stop_btn = QPushButton("⏹  초기화")
        self.stop_btn.clicked.connect(self._reset_animation)
        btn_row.addWidget(self.play_btn)
        btn_row.addWidget(self.stop_btn)
        right_layout.addLayout(btn_row)

        root.addWidget(right_panel, stretch=0)

    def _create_figure(self):
        """기본 Figure 생성. 서브클래스에서 오버라이드 가능."""
        fig = Figure(facecolor='#0a0a0f')
        ax = fig.add_subplot(111)
        apply_dark_axes(ax)
        return fig, ax

    # ------------------------------------------------------------------
    # 헬퍼
    # ------------------------------------------------------------------
    def create_slider(self, label, min_val, max_val, default, decimals=1):
        """슬라이더 위젯 생성 후 ctrl_layout에 추가"""
        sw = SliderWidget(label, min_val, max_val, default, decimals)
        self.ctrl_layout.addWidget(sw)
        return sw

    def add_separator(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #2a2a45;")
        self.ctrl_layout.addWidget(sep)

    def set_info(self, lines: list):
        """우측 정보 패널 갱신. lines: list of (label, value) tuples or plain strings.
        위젯을 재사용해 매 프레임 생성/삭제 비용을 없앤다."""
        # 캐시된 행 수와 다르면 전부 재생성
        if not hasattr(self, '_info_cache') or len(self._info_cache) != len(lines):
            while self.info_layout.count():
                item = self.info_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self._info_cache = []
            for line in lines:
                if isinstance(line, tuple):
                    row = QHBoxLayout()
                    lbl = QLabel(line[0])
                    lbl.setStyleSheet("color: #888aaa; font-size: 12px;")
                    val = QLabel(str(line[1]))
                    val.setStyleSheet("color: #e0e0ff; font-size: 12px; font-weight: bold;")
                    val.setAlignment(Qt.AlignmentFlag.AlignRight)
                    row.addWidget(lbl)
                    row.addStretch()
                    row.addWidget(val)
                    container = QWidget()
                    container.setStyleSheet("background: transparent;")
                    container.setLayout(row)
                    self.info_layout.addWidget(container)
                    self._info_cache.append(('tuple', lbl, val))
                else:
                    lbl = QLabel(str(line))
                    lbl.setStyleSheet("color: #aaaacc; font-size: 11px;")
                    lbl.setWordWrap(True)
                    self.info_layout.addWidget(lbl)
                    self._info_cache.append(('str', lbl, None))
        else:
            # 텍스트만 업데이트 (위젯 재사용)
            for (kind, lbl, val), line in zip(self._info_cache, lines):
                if kind == 'tuple' and isinstance(line, tuple):
                    val.setText(str(line[1]))
                elif kind == 'str' and isinstance(line, str):
                    lbl.setText(line)

    def add_ctrl_label(self, text: str):
        """컨트롤 패널에 섹션 레이블 추가"""
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #6c63ff; font-size: 12px; font-weight: bold; margin-top: 6px;")
        self.ctrl_layout.addWidget(lbl)

    def add_ctrl_stretch(self):
        self.ctrl_layout.addStretch()

    # ------------------------------------------------------------------
    # 애니메이션
    # ------------------------------------------------------------------
    def _toggle_animation(self):
        if self._animating:
            self._timer.stop()
            self._animating = False
            self.play_btn.setText("▶  재생")
        else:
            self._timer.start()
            self._animating = True
            self.play_btn.setText("⏸  일시정지")

    def _reset_animation(self):
        self._timer.stop()
        self._animating = False
        self.play_btn.setText("▶  재생")
        self._anim_frame = 0
        self.on_reset()
        self.update_plot()

    def _on_timer(self):
        self._anim_frame += 1
        self.on_animate(self._anim_frame)

    # ------------------------------------------------------------------
    # 서브클래스에서 구현
    # ------------------------------------------------------------------
    def setup_module(self):
        """슬라이더, 컨트롤 등 초기화. 서브클래스에서 구현."""
        pass

    def update_plot(self):
        """플롯 갱신. 서브클래스에서 구현."""
        pass

    def on_animate(self, frame: int):
        """매 타이머 틱마다 호출. 애니메이션 있는 모듈에서 오버라이드."""
        pass

    def on_reset(self):
        """초기화 시 호출."""
        pass
