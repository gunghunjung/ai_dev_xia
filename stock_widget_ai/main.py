#!/usr/bin/env python3
"""
Stock Widget AI — AI 기반 주가 예측 데스크톱 플랫폼
실행: python main.py
"""
import os
import sys

# 프로젝트 루트를 sys.path에 추가
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from stock_widget_ai.logger_config import setup_logging
from stock_widget_ai.config_manager import ConfigManager
from stock_widget_ai.utils import ensure_dirs


def main() -> None:
    # 기본 디렉토리 생성
    ensure_dirs("logs", "data", "models",
                "outputs", "outputs/predictions",
                "outputs/backtests", "outputs/experiments")

    # 설정 로드 (없으면 자동 생성)
    cfg = ConfigManager()

    # 로깅 초기화
    setup_logging(cfg.state.log_level)

    # PyQt6 앱 시작
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from stock_widget_ai.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Stock Widget AI")
    app.setOrganizationName("StockWidgetAI")

    win = MainWindow(cfg)
    win.show()

    # ★ show() 이후에 geometry 복원해야 윈도우 매니저가 무시하지 않음
    cfg.apply_to_window(win)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
