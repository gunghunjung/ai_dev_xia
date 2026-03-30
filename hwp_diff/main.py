"""Entry point for the HWP Document Diff Tool."""
import sys
import os

# Ensure the project root is on the Python path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.utils.logger import setup_logger


def main():
    setup_logger()

    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from app.ui.main_window import MainWindow

    # Enable HiDPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("문서 변경점 비교기")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("HwpDiff")

    # Apply base stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background: #F0F0F0;
        }
        QGroupBox {
            border: 1px solid #CCCCCC;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 6px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            color: #1F4E79;
        }
        QStatusBar {
            background: #E8EEF5;
            color: #333333;
            font-size: 11px;
        }
        QToolBar {
            background: #E8EEF5;
            border-bottom: 1px solid #CCCCCC;
            spacing: 4px;
            padding: 2px;
        }
        QMenuBar {
            background: #1F4E79;
            color: white;
        }
        QMenuBar::item:selected {
            background: #2E6096;
        }
        QMenu {
            background: #FFFFFF;
            border: 1px solid #CCCCCC;
        }
        QMenu::item:selected {
            background: #BDD7EE;
        }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
