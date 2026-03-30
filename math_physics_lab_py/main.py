import sys
import os

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# matplotlib 한글 폰트 설정 (앱 시작 전 설정)
import matplotlib
import matplotlib.pyplot as plt

def _setup_matplotlib_font():
    """Windows 한글 폰트 설정"""
    import matplotlib.font_manager as fm
    # Windows 기본 한글 폰트 우선 시도
    candidates = ['Malgun Gothic', '맑은 고딕', 'NanumGothic', 'NanumBarunGothic', 'AppleGothic']
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = None
    for c in candidates:
        if c in available:
            chosen = c
            break
    if chosen:
        matplotlib.rc('font', family=chosen)
    matplotlib.rcParams['axes.unicode_minus'] = False

_setup_matplotlib_font()

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from ui.main_window import MainWindow, DARK_QSS


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("수학·물리 학습")
    app.setStyleSheet(DARK_QSS)

    # 기본 폰트 설정
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
