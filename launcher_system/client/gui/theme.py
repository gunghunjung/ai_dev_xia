from __future__ import annotations


def get_stylesheet() -> str:
    """Return the dark Netflix-inspired QSS stylesheet string."""
    return """
/* ── Global ─────────────────────────────────────────────────────── */
QWidget {
    background-color: #141414;
    color: #ffffff;
    font-family: "Segoe UI", "Noto Sans KR", "Malgun Gothic", sans-serif;
    font-size: 13px;
}

QMainWindow {
    background-color: #141414;
}

/* ── Scroll Areas ────────────────────────────────────────────────── */
QScrollArea {
    border: none;
    background-color: #141414;
}

QScrollBar:vertical {
    background-color: #1f1f1f;
    width: 8px;
    margin: 0;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background-color: #555555;
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #888888;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
    background: none;
}

QScrollBar:horizontal {
    background-color: #1f1f1f;
    height: 8px;
    margin: 0;
    border-radius: 4px;
}

QScrollBar::handle:horizontal {
    background-color: #555555;
    border-radius: 4px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #888888;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0;
    background: none;
}

/* ── App Cards ───────────────────────────────────────────────────── */
QFrame#AppCard {
    background-color: #1f1f1f;
    border-radius: 8px;
    border: 1px solid #2a2a2a;
}

QFrame#AppCard:hover {
    background-color: #2a2a2a;
    border: 1px solid #3a3a3a;
}

/* ── Buttons ─────────────────────────────────────────────────────── */
QPushButton {
    background-color: #333333;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #444444;
}

QPushButton:pressed {
    background-color: #555555;
}

QPushButton:disabled {
    background-color: #2a2a2a;
    color: #666666;
}

QPushButton#PrimaryButton {
    background-color: #e50914;
    color: #ffffff;
    font-weight: 700;
    padding: 8px 16px;
    border-radius: 4px;
}

QPushButton#PrimaryButton:hover {
    background-color: #f40612;
}

QPushButton#PrimaryButton:pressed {
    background-color: #c10812;
}

QPushButton#SecondaryButton {
    background-color: #444444;
    color: #ffffff;
    font-weight: 500;
    padding: 8px 16px;
    border-radius: 4px;
}

QPushButton#SecondaryButton:hover {
    background-color: #555555;
}

QPushButton#DangerButton {
    background-color: transparent;
    color: #b3b3b3;
    border: 1px solid #555555;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
}

QPushButton#DangerButton:hover {
    background-color: #3a1a1a;
    color: #ff6666;
    border-color: #ff4444;
}

QPushButton#IconButton {
    background-color: transparent;
    border: none;
    padding: 4px;
    border-radius: 4px;
    color: #b3b3b3;
    font-size: 16px;
}

QPushButton#IconButton:hover {
    background-color: #333333;
    color: #ffffff;
}

/* ── Top Bar ─────────────────────────────────────────────────────── */
QFrame#TopBar {
    background-color: #0a0a0a;
    border-bottom: 1px solid #2a2a2a;
}

QLabel#LogoLabel {
    color: #e50914;
    font-size: 22px;
    font-weight: 900;
    letter-spacing: 1px;
}

QLabel#StatusLabel {
    color: #b3b3b3;
    font-size: 12px;
}

/* ── Labels ──────────────────────────────────────────────────────── */
QLabel#AppName {
    color: #ffffff;
    font-size: 14px;
    font-weight: 700;
}

QLabel#AppVersion {
    color: #b3b3b3;
    font-size: 11px;
}

QLabel#AppDescription {
    color: #b3b3b3;
    font-size: 12px;
}

QLabel#BadgeInstalled {
    background-color: #1a6b3a;
    color: #4cff8a;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 700;
}

QLabel#BadgeNotInstalled {
    background-color: #333333;
    color: #888888;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 700;
}

QLabel#BadgeUpdate {
    background-color: #7a3c00;
    color: #ffaa44;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 700;
}

/* ── Progress Bar ────────────────────────────────────────────────── */
QProgressBar {
    background-color: #333333;
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background-color: #e50914;
    border-radius: 3px;
}

/* ── Line Edits ──────────────────────────────────────────────────── */
QLineEdit {
    background-color: #2a2a2a;
    color: #ffffff;
    border: 1px solid #444444;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 13px;
    selection-background-color: #e50914;
}

QLineEdit:focus {
    border-color: #e50914;
    background-color: #303030;
}

QLineEdit:disabled {
    background-color: #1f1f1f;
    color: #666666;
}

/* ── Dialogs ─────────────────────────────────────────────────────── */
QDialog {
    background-color: #1f1f1f;
    border: 1px solid #333333;
    border-radius: 8px;
}

/* ── Status Bar ──────────────────────────────────────────────────── */
QStatusBar {
    background-color: #0a0a0a;
    color: #b3b3b3;
    border-top: 1px solid #2a2a2a;
    font-size: 11px;
}

/* ── Tool Tips ───────────────────────────────────────────────────── */
QToolTip {
    background-color: #2a2a2a;
    color: #ffffff;
    border: 1px solid #444444;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
}

/* ── Message Box ─────────────────────────────────────────────────── */
QMessageBox {
    background-color: #1f1f1f;
}

QMessageBox QLabel {
    color: #ffffff;
}

/* ── Download Panel ──────────────────────────────────────────────── */
QFrame#DownloadPanel {
    background-color: #0f0f0f;
    border-top: 1px solid #2a2a2a;
}

QLabel#DownloadPanelTitle {
    color: #e50914;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* ── Connect Dialog ──────────────────────────────────────────────── */
QLabel#ConnectTitle {
    color: #ffffff;
    font-size: 18px;
    font-weight: 700;
}

QLabel#ConnectSubtitle {
    color: #b3b3b3;
    font-size: 12px;
}

QLabel#FieldLabel {
    color: #b3b3b3;
    font-size: 12px;
    font-weight: 600;
}

QLabel#ConnectStatus {
    font-size: 12px;
    padding: 4px;
}
"""
