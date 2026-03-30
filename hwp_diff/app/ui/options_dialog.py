"""Options dialog for comparison settings."""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox,
    QSlider, QLabel, QLineEdit, QPushButton, QDialogButtonBox,
    QFormLayout, QSpinBox, QDoubleSpinBox,
)
from PySide6.QtCore import Qt


class OptionsDialog(QDialog):
    """
    Dialog for configuring comparison options.

    Options:
    - 공백 무시
    - 대소문자 무시
    - 줄바꿈 무시
    - 서식 변경 포함
    - 이동 탐지
    - 유사도 임계값 (0.3 - 0.9)
    - 표 비교 민감도
    - 중요 키워드
    """

    def __init__(self, current_options: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("비교 옵션")
        self.setMinimumWidth(420)
        self.setModal(True)
        self._options = dict(current_options)
        self._init_ui()
        self._load_options()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # --- Text normalization group ---
        norm_group = QGroupBox("텍스트 정규화")
        norm_layout = QVBoxLayout(norm_group)

        self.cb_ignore_whitespace = QCheckBox("공백 무시 (연속 공백을 단일 공백으로)")
        self.cb_ignore_case = QCheckBox("대소문자 무시")
        self.cb_ignore_newline = QCheckBox("줄바꿈 무시")

        norm_layout.addWidget(self.cb_ignore_whitespace)
        norm_layout.addWidget(self.cb_ignore_case)
        norm_layout.addWidget(self.cb_ignore_newline)
        layout.addWidget(norm_group)

        # --- Detection options group ---
        detect_group = QGroupBox("탐지 옵션")
        detect_layout = QVBoxLayout(detect_group)

        self.cb_include_format = QCheckBox("서식 변경 포함 (서식만 변경된 경우도 표시)")
        self.cb_detect_moves = QCheckBox("이동 탐지 (위치가 이동된 문단 감지)")

        detect_layout.addWidget(self.cb_include_format)
        detect_layout.addWidget(self.cb_detect_moves)
        layout.addWidget(detect_group)

        # --- Similarity threshold ---
        sim_group = QGroupBox("유사도 설정")
        sim_layout = QFormLayout(sim_group)

        self.lbl_threshold = QLabel("0.60")
        self.lbl_threshold.setMinimumWidth(40)

        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setMinimum(30)
        self.slider_threshold.setMaximum(90)
        self.slider_threshold.setValue(60)
        self.slider_threshold.setTickInterval(10)
        self.slider_threshold.setTickPosition(QSlider.TicksBelow)
        self.slider_threshold.valueChanged.connect(self._on_threshold_changed)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("낮음 (0.3)"))
        threshold_row.addWidget(self.slider_threshold)
        threshold_row.addWidget(QLabel("높음 (0.9)"))
        threshold_row.addWidget(self.lbl_threshold)

        sim_layout.addRow("유사도 임계값:", threshold_row)

        self.lbl_table_sens = QLabel("0.50")
        self.slider_table_sens = QSlider(Qt.Horizontal)
        self.slider_table_sens.setMinimum(10)
        self.slider_table_sens.setMaximum(90)
        self.slider_table_sens.setValue(50)
        self.slider_table_sens.setTickInterval(10)
        self.slider_table_sens.setTickPosition(QSlider.TicksBelow)
        self.slider_table_sens.valueChanged.connect(self._on_table_sens_changed)

        table_row = QHBoxLayout()
        table_row.addWidget(QLabel("낮음"))
        table_row.addWidget(self.slider_table_sens)
        table_row.addWidget(QLabel("높음"))
        table_row.addWidget(self.lbl_table_sens)

        sim_layout.addRow("표 비교 민감도:", table_row)
        layout.addWidget(sim_group)

        # --- Keywords ---
        kw_group = QGroupBox("중요 키워드")
        kw_layout = QVBoxLayout(kw_group)
        kw_hint = QLabel("쉼표로 구분하여 입력. 해당 단어가 포함된 변경사항은 중요도 '높음'으로 처리됩니다.")
        kw_hint.setWordWrap(True)
        kw_hint.setStyleSheet("color: #666; font-size: 10px;")
        self.edit_keywords = QLineEdit()
        self.edit_keywords.setPlaceholderText("예: 허용오차, 합격기준, 최대값, 온도")
        kw_layout.addWidget(kw_hint)
        kw_layout.addWidget(self.edit_keywords)
        layout.addWidget(kw_group)

        # --- Buttons ---
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        btn_box.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._restore_defaults)
        layout.addWidget(btn_box)

    def _on_threshold_changed(self, value: int) -> None:
        self.lbl_threshold.setText(f"{value / 100:.2f}")

    def _on_table_sens_changed(self, value: int) -> None:
        self.lbl_table_sens.setText(f"{value / 100:.2f}")

    def _load_options(self) -> None:
        """Load current options into UI controls."""
        self.cb_ignore_whitespace.setChecked(
            self._options.get("ignore_whitespace", False)
        )
        self.cb_ignore_case.setChecked(
            self._options.get("ignore_case", False)
        )
        self.cb_ignore_newline.setChecked(
            self._options.get("ignore_newline", False)
        )
        self.cb_include_format.setChecked(
            self._options.get("include_format_changes", True)
        )
        self.cb_detect_moves.setChecked(
            self._options.get("detect_moves", True)
        )

        threshold = self._options.get("similarity_threshold", 0.6)
        self.slider_threshold.setValue(int(threshold * 100))

        table_sens = self._options.get("table_sensitivity", 0.5)
        self.slider_table_sens.setValue(int(table_sens * 100))

        keywords = self._options.get("important_keywords", [])
        if isinstance(keywords, list):
            self.edit_keywords.setText(", ".join(keywords))
        else:
            self.edit_keywords.setText(str(keywords))

    def _restore_defaults(self) -> None:
        """Restore default option values."""
        self._options = {
            "ignore_whitespace": False,
            "ignore_case": False,
            "ignore_newline": False,
            "include_format_changes": True,
            "detect_moves": True,
            "similarity_threshold": 0.6,
            "table_sensitivity": 0.5,
            "important_keywords": [],
        }
        self._load_options()

    def _on_accept(self) -> None:
        """Save options and close dialog."""
        self._options = self.get_options()
        self.accept()

    def get_options(self) -> dict:
        """Return the current options as a dict."""
        kw_text = self.edit_keywords.text().strip()
        keywords = [
            kw.strip() for kw in kw_text.split(",") if kw.strip()
        ] if kw_text else []

        return {
            "ignore_whitespace": self.cb_ignore_whitespace.isChecked(),
            "ignore_case": self.cb_ignore_case.isChecked(),
            "ignore_newline": self.cb_ignore_newline.isChecked(),
            "include_format_changes": self.cb_include_format.isChecked(),
            "detect_moves": self.cb_detect_moves.isChecked(),
            "similarity_threshold": self.slider_threshold.value() / 100,
            "table_sensitivity": self.slider_table_sens.value() / 100,
            "important_keywords": keywords,
        }
