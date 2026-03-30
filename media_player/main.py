"""
PyPlayer — 진입점

[Fix 1] 스플래시 스크린: QApplication 생성 직후 즉시 표시
         → VLC/Qt 무거운 초기화 중에도 창이 바로 뜸
"""

import sys
import os


def _make_splash(app):
    """VLC 로딩 전에 즉시 표시할 스플래시 스크린."""
    from PyQt6.QtWidgets import QSplashScreen
    from PyQt6.QtGui import QPixmap, QColor, QPainter, QFont
    from PyQt6.QtCore import Qt, QRect

    W, H = 420, 200
    pm = QPixmap(W, H)
    pm.fill(QColor("#1a1a2e"))

    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    # 제목
    p.setPen(QColor("#e0e0e0"))
    p.setFont(QFont("Malgun Gothic", 20, QFont.Weight.Bold))
    p.drawText(QRect(0, 40, W, 60), Qt.AlignmentFlag.AlignCenter, "경훈이 플레이어")

    # 버전
    p.setPen(QColor("#53c0f0"))
    p.setFont(QFont("Segoe UI", 11))
    p.drawText(QRect(0, 100, W, 30), Qt.AlignmentFlag.AlignCenter, "v3.0")

    # 로딩 메시지
    p.setPen(QColor("#506070"))
    p.setFont(QFont("Segoe UI", 10))
    p.drawText(QRect(0, 155, W, 30), Qt.AlignmentFlag.AlignCenter, "미디어 엔진 초기화 중...")

    p.end()

    splash = QSplashScreen(pm, Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()   # 스플래시를 즉시 화면에 렌더링
    return splash


def main() -> None:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QIcon

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("경훈이 플레이어")
    app.setApplicationVersion("3.0")
    app.setOrganizationName("경훈이 플레이어")

    icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # ── 스플래시 즉시 표시 ────────────────────────────────────────────────
    splash = _make_splash(app)

    # ── 무거운 초기화 (VLC 플러그인 로딩 등) ─────────────────────────────
    try:
        from gui.main_window import MainWindow
        window = MainWindow()
    except (SystemExit, RuntimeError) as e:
        splash.close()
        QMessageBox.critical(
            None, "경훈이 플레이어 — 시작 오류",
            f"<b>미디어 엔진을 초기화할 수 없습니다.</b><br><br>"
            f"{str(e).replace(chr(10), '<br>')}<br><br>"
            "<a href='https://www.videolan.org/vlc/'>VLC 다운로드</a>"
        )
        sys.exit(1)

    # ── 스플래시 종료 + 메인 창 표시 ──────────────────────────────────────
    # 초기 크기 448×252 (키-1 프리셋), 화면 중앙
    from PyQt6.QtWidgets import QApplication as _App
    W, H = 120, 68
    screen = _App.primaryScreen().geometry()
    window.resize(W, H)
    window.move((screen.width() - W) // 2, (screen.height() - H) // 2)

    splash.finish(window)   # 메인 창이 뜨는 순간 스플래시 사라짐
    window.show()

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        window._load_file(sys.argv[1])

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
