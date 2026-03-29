from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QRect, pyqtSignal
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QLinearGradient,
    QPainter, QPainterPath, QPixmap,
)
from PyQt6.QtWidgets import QFrame, QWidget


def _make_placeholder(name: str, w: int, h: int) -> QPixmap:
    """앱 이름 첫 글자를 중앙에 그린 다크 플레이스홀더 이미지."""
    pm = QPixmap(w, h)
    pm.fill(QColor("#1f1f1f"))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    # 그라데이션 배경
    grad = QLinearGradient(0, 0, 0, h)
    grad.setColorAt(0.0, QColor("#2a2a3a"))
    grad.setColorAt(1.0, QColor("#111118"))
    p.fillRect(0, 0, w, h, QBrush(grad))

    # 첫 글자
    letter = (name[:1] if name else "?").upper()
    font = QFont("Malgun Gothic", h // 3, QFont.Weight.Bold)
    p.setFont(font)
    p.setPen(QColor("#444466"))
    p.drawText(QRect(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, letter)
    p.end()
    return pm


class AppCard(QFrame):
    """
    Netflix 포스터 스타일 앱 카드.

    - title.jpg 가 카드 전체를 채움 (없으면 플레이스홀더)
    - 하단 그라데이션 위에 앱 이름 · 버전 표시
    - 마우스 오버 → 반투명 어두운 오버레이 + 중앙 액션 버튼("▶ 실행" 등)
    - 다운로드 중 → 하단 프로그레스바
    """

    download_clicked   = pyqtSignal(str)   # app_id
    launch_clicked     = pyqtSignal(str)
    update_clicked     = pyqtSignal(str)
    uninstall_clicked  = pyqtSignal(str)

    CARD_W = 200
    CARD_H = 300
    RADIUS = 10

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(self.CARD_W, self.CARD_H)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._app_id: str = ""
        self._name: str = ""
        self._version: str = "1.0.0"
        self._installed: bool = False
        self._update_available: bool = False
        self._hovered: bool = False
        self._downloading: int = -1          # -1 = not downloading, 0-100 = %
        self._title_pixmap: Optional[QPixmap] = None
        self._placeholder: QPixmap = _make_placeholder("?", self.CARD_W, self.CARD_H)

    # ── Public API ────────────────────────────────────────────────────────

    def set_app_data(
        self,
        app_meta: dict,
        installed: bool,
        installed_version: Optional[str],
        update_available: bool,
    ) -> None:
        self._app_id = app_meta.get("id", "")
        self._name = app_meta.get("name", self._app_id)
        self._version = app_meta.get("version", "1.0.0")
        self._installed = installed
        self._update_available = update_available
        self._placeholder = _make_placeholder(self._name, self.CARD_W, self.CARD_H)
        self._title_pixmap = None
        self.update()

    def set_title_image(self, pixmap: QPixmap) -> None:
        """title.jpg 로드 완료 후 호출 — 카드 전체 이미지로 표시."""
        if pixmap.isNull():
            return
        # 카드 크기를 꽉 채우도록 크롭 스케일
        scaled = pixmap.scaled(
            self.CARD_W, self.CARD_H,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._title_pixmap = scaled
        self.update()

    # set_icon은 하위 호환을 위해 유지 (title_image 없을 때 아이콘 사용)
    def set_icon(self, pixmap: QPixmap) -> None:
        if self._title_pixmap is None:
            self.set_title_image(pixmap)

    def set_downloading(self, progress: int) -> None:
        """0-100: 진행 중, -1: 완료/숨김."""
        self._downloading = progress
        self.update()

    def app_id(self) -> str:
        return self._app_id

    # ── 렌더링 ────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        W, H = self.CARD_W, self.CARD_H

        # ── 1. 둥근 클리핑 ─────────────────────────────────────────
        clip = QPainterPath()
        clip.addRoundedRect(0, 0, W, H, self.RADIUS, self.RADIUS)
        p.setClipPath(clip)

        # ── 2. 배경 이미지 (중앙 크롭) ─────────────────────────────
        pm = self._title_pixmap or self._placeholder
        ox = (pm.width()  - W) // 2
        oy = (pm.height() - H) // 2
        p.drawPixmap(0, 0, pm, ox, oy, W, H)

        # ── 3. 하단 그라데이션 (텍스트 가독성) ────────────────────
        grad = QLinearGradient(0, H - 90, 0, H)
        grad.setColorAt(0.0, QColor(0, 0, 0, 0))
        grad.setColorAt(1.0, QColor(0, 0, 0, 210))
        p.fillRect(0, H - 90, W, 90, QBrush(grad))

        # ── 4. 앱 이름 ─────────────────────────────────────────────
        p.setPen(QColor("#ffffff"))
        p.setFont(QFont("Malgun Gothic", 10, QFont.Weight.Bold))
        p.drawText(
            QRect(10, H - 46, W - 20, 22),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._name,
        )

        # ── 5. 버전 ────────────────────────────────────────────────
        p.setPen(QColor("#aaaaaa"))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(
            QRect(10, H - 26, W - 20, 18),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"v{self._version}",
        )

        # ── 6. 호버 오버레이 ───────────────────────────────────────
        if self._hovered and self._downloading < 0:
            p.fillRect(0, 0, W, H, QColor(0, 0, 0, 145))

            # 액션 레이블 (아이콘 + 텍스트)
            action = self._action_label()
            p.setPen(QColor("#ffffff"))
            p.setFont(QFont("Malgun Gothic", 14, QFont.Weight.Bold))
            p.drawText(
                QRect(0, 0, W, H),
                Qt.AlignmentFlag.AlignCenter,
                action,
            )

            # 업데이트 배지
            if self._update_available:
                p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                p.setPen(QColor("#ffffff"))
                p.setBrush(QColor("#e50914"))
                p.drawRoundedRect(QRect(W - 68, 10, 58, 20), 4, 4)
                p.drawText(
                    QRect(W - 68, 10, 58, 20),
                    Qt.AlignmentFlag.AlignCenter,
                    "업데이트",
                )

        # ── 7. 다운로드 프로그레스 ─────────────────────────────────
        if self._downloading >= 0:
            # 하단 어두운 바
            p.fillRect(0, H - 36, W, 36, QColor(0, 0, 0, 190))

            # 텍스트
            p.setPen(QColor("#ffffff"))
            p.setFont(QFont("Segoe UI", 8))
            p.drawText(
                QRect(0, H - 34, W, 18),
                Qt.AlignmentFlag.AlignCenter,
                f"다운로드 중... {self._downloading}%",
            )

            # 프로그레스바 트랙
            bar_y = H - 10
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(60, 60, 60))
            p.drawRoundedRect(QRect(8, bar_y, W - 16, 6), 3, 3)

            # 프로그레스바 채움
            fill_w = max(6, int((W - 16) * self._downloading / 100))
            p.setBrush(QColor("#e50914"))
            p.drawRoundedRect(QRect(8, bar_y, fill_w, 6), 3, 3)

        p.end()

    # ── 마우스 이벤트 ──────────────────────────────────────────────────────

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._downloading < 0:
            self._trigger_action()
        super().mousePressEvent(event)

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────

    def _action_label(self) -> str:
        if self._update_available:
            return "↑  업데이트"
        if self._installed:
            return "▶  실행"
        return "⬇  다운로드"

    def _trigger_action(self) -> None:
        if not self._app_id:
            return
        if self._update_available:
            self.update_clicked.emit(self._app_id)
        elif self._installed:
            self.launch_clicked.emit(self._app_id)
        else:
            self.download_clicked.emit(self._app_id)
