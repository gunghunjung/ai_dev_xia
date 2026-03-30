"""
Dark Theme — 팟플레이어/VLC 수준의 전문적인 다크 테마
"""

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}

/* ── 비디오 영역 ── */
#video_frame {
    background-color: #000000;
    border: none;
}

/* ── 커스텀 타이틀바 ── */
#title_bar {
    background-color: rgba(22, 33, 62, 0.85);
    border-bottom: 1px solid #0f3460;
}

/* ── 컨트롤 바 ── */
#controls_bar {
    background-color: #16213e;
    border-top: 1px solid #0f3460;
    padding: 6px 10px;
}

/* ── 타임라인 슬라이더 ── */
QSlider#timeline {
    height: 18px;
}
QSlider#timeline::groove:horizontal {
    height: 5px;
    background: #2d2d4e;
    border-radius: 3px;
}
QSlider#timeline::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #e94560, stop:1 #0f3460);
    border-radius: 3px;
}
QSlider#timeline::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: #e94560;
    border: 2px solid #ffffff;
}
QSlider#timeline::handle:horizontal:hover {
    background: #ff6b8a;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}

/* ── 볼륨 슬라이더 ── */
QSlider#volume {
    height: 16px;
    max-width: 90px;
    min-width: 90px;
}
QSlider#volume::groove:horizontal {
    height: 4px;
    background: #2d2d4e;
    border-radius: 2px;
}
QSlider#volume::sub-page:horizontal {
    background: #53c0f0;
    border-radius: 2px;
}
QSlider#volume::handle:horizontal {
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
    background: #53c0f0;
}

/* ── 버튼 공통 ── */
QPushButton {
    background: transparent;
    border: none;
    color: #e0e0e0;
    padding: 5px 8px;
    border-radius: 4px;
    font-size: 15px;
}
QPushButton:hover {
    background-color: #0f3460;
    color: #ffffff;
}
QPushButton:pressed {
    background-color: #e94560;
}
QPushButton#btn_play {
    font-size: 20px;
    padding: 4px 12px;
}

/* ── 레이블 ── */
QLabel#time_label {
    color: #a0a0c0;
    font-size: 12px;
    font-family: 'Consolas', monospace;
    min-width: 110px;
}
QLabel#title_label {
    color: #c0c0e0;
    font-size: 12px;
}

/* ── 메뉴바 ── */
QMenuBar {
    background-color: #16213e;
    color: #c0c0e0;
    padding: 2px;
    border-bottom: 1px solid #0f3460;
}
QMenuBar::item:selected {
    background-color: #0f3460;
    border-radius: 4px;
}
QMenu {
    background-color: #1a1a2e;
    border: 1px solid #0f3460;
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px;
    border-radius: 3px;
}
QMenu::item:selected {
    background-color: #0f3460;
}
QMenu::separator {
    height: 1px;
    background: #0f3460;
    margin: 4px 0;
}

/* ── 상태바 ── */
QStatusBar {
    background-color: #0f3460;
    color: #a0a0c0;
    font-size: 11px;
    padding: 2px 8px;
}

/* ── 볼륨 아이콘 버튼 ── */
QPushButton#btn_mute {
    font-size: 16px;
    padding: 2px 4px;
    min-width: 26px;
}

/* ── 스크롤바 ── */
QScrollBar:vertical {
    background: #1a1a2e;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #0f3460;
    border-radius: 4px;
    min-height: 20px;
}
"""


def get_stylesheet() -> str:
    return DARK_STYLE
