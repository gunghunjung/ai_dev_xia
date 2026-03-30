from __future__ import annotations

import sys
from pathlib import Path

# Add client directory to path so relative imports work
_CLIENT_DIR = Path(__file__).resolve().parent
if str(_CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CLIENT_DIR))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont

from config import ClientConfig, load_config
from gui.theme import get_stylesheet
from gui.main_window import MainWindow
from gui.connect_dialog import ConnectDialog


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("경훈이 런처")
    app.setOrganizationName("Launcher")
    app.setStyle("Fusion")
    app.setStyleSheet(get_stylesheet())

    # Default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Load saved config
    config = load_config()

    # Create main window
    window = MainWindow(config)
    window.show()

    # Auto-connect if previous session had a server configured
    if config.server_url and config.token and config.last_connected:
        try:
            window.connect_to_server(config.server_url, config.token)
        except Exception as e:
            print(f"[main] Auto-connect failed: {e}")
            # Fall through to show connect dialog
            _show_connect_dialog(window, config)
    else:
        # Show connect dialog on first run
        _show_connect_dialog(window, config)

    sys.exit(app.exec())


def _show_connect_dialog(window: MainWindow, config: ClientConfig) -> None:
    """Show the connect dialog and connect if user confirms."""
    dlg = ConnectDialog(
        window,
        initial_url=config.server_url,
        initial_token=config.token,
    )
    if dlg.exec():
        result = dlg.get_result()
        if result:
            url, token = result
            window.connect_to_server(url, token)


if __name__ == "__main__":
    main()
