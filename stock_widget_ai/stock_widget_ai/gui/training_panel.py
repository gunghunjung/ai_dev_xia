"""
Training Panel — 모델 선택, 학습 설정, 학습 실행/중단
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QPushButton, QProgressBar, QTextEdit,
    QSpinBox, QDoubleSpinBox, QHBoxLayout, QLabel, QCheckBox,
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from ..state_schema import AppState

MODEL_CHOICES = ["lstm", "gru", "temporal_cnn", "transformer",
                 "tft_like", "nbeats_like", "cnn_lstm", "ensemble"]


class TrainingPanel(QWidget):
    train_requested = pyqtSignal(dict)
    stop_requested  = pyqtSignal()

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # ── 모델 설정 ──────────────────────────────────────────────
        grp = QGroupBox("모델 및 학습 설정")
        form = QFormLayout()

        self._model_cb = QComboBox()
        self._model_cb.addItems(MODEL_CHOICES)
        self._model_cb.setCurrentText(self._state.model.selected_model)
        form.addRow("모델", self._model_cb)

        self._epochs_sp = QSpinBox()
        self._epochs_sp.setRange(1, 500)
        self._epochs_sp.setValue(self._state.model.epochs)
        form.addRow("Epoch", self._epochs_sp)

        self._bs_sp = QSpinBox()
        self._bs_sp.setRange(8, 512)
        self._bs_sp.setValue(self._state.model.batch_size)
        form.addRow("Batch Size", self._bs_sp)

        self._lr_sp = QDoubleSpinBox()
        self._lr_sp.setRange(1e-6, 1.0)
        self._lr_sp.setDecimals(6)
        self._lr_sp.setSingleStep(1e-4)
        self._lr_sp.setValue(self._state.model.learning_rate)
        form.addRow("Learning Rate", self._lr_sp)

        self._hidden_sp = QSpinBox()
        self._hidden_sp.setRange(16, 512)
        self._hidden_sp.setValue(self._state.model.hidden_dim)
        form.addRow("Hidden Dim", self._hidden_sp)

        self._layers_sp = QSpinBox()
        self._layers_sp.setRange(1, 8)
        self._layers_sp.setValue(self._state.model.num_layers)
        form.addRow("Num Layers", self._layers_sp)

        self._drop_sp = QDoubleSpinBox()
        self._drop_sp.setRange(0.0, 0.8)
        self._drop_sp.setDecimals(2)
        self._drop_sp.setSingleStep(0.05)
        self._drop_sp.setValue(self._state.model.dropout)
        form.addRow("Dropout", self._drop_sp)

        self._loss_cb = QComboBox()
        self._loss_cb.addItems(["huber", "mse", "mae", "direction"])
        self._loss_cb.setCurrentText(self._state.model.loss_type)
        form.addRow("Loss", self._loss_cb)

        self._optim_cb = QComboBox()
        self._optim_cb.addItems(["adamw", "adam", "rmsprop"])
        self._optim_cb.setCurrentText(self._state.model.optimizer)
        form.addRow("Optimizer", self._optim_cb)

        self._sched_cb = QComboBox()
        self._sched_cb.addItems(["onecycle", "cosine", "plateau", "none"])
        self._sched_cb.setCurrentText(self._state.model.scheduler)
        form.addRow("Scheduler", self._sched_cb)

        self._device_cb = QComboBox()
        self._device_cb.addItems(["auto", "cpu", "cuda"])
        self._device_cb.setCurrentText(self._state.model.device)
        form.addRow("Device", self._device_cb)

        self._aug_chk = QCheckBox("Data Augmentation")
        self._aug_chk.setChecked(self._state.model.use_augmentation)
        form.addRow("", self._aug_chk)

        grp.setLayout(form)
        layout.addWidget(grp)

        # ── 버튼 ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._train_btn = QPushButton("🚀  학습 시작")
        self._stop_btn  = QPushButton("⏹  중단")
        self._stop_btn.setEnabled(False)
        self._train_btn.clicked.connect(self._on_train)
        self._stop_btn.clicked.connect(self.stop_requested.emit)
        btn_row.addWidget(self._train_btn)
        btn_row.addWidget(self._stop_btn)
        layout.addLayout(btn_row)

        # ── 진행 ───────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)

        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setMinimumHeight(120)   # 최소 높이만 지정 — 스크롤로 더 볼 수 있음
        layout.addWidget(QLabel("학습 로그"))
        layout.addWidget(self._log_box)
        layout.addStretch()

    def _on_train(self) -> None:
        params = self._collect_params()
        self._state.model.selected_model = params["model"]
        self._train_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self.train_requested.emit(params)

    def _collect_params(self) -> dict:
        return {
            "model":        self._model_cb.currentText(),
            "epochs":       self._epochs_sp.value(),
            "batch_size":   self._bs_sp.value(),
            "lr":           self._lr_sp.value(),
            "hidden_dim":   self._hidden_sp.value(),
            "num_layers":   self._layers_sp.value(),
            "dropout":      self._drop_sp.value(),
            "loss":         self._loss_cb.currentText(),
            "optimizer":    self._optim_cb.currentText(),
            "scheduler":    self._sched_cb.currentText(),
            "device":       self._device_cb.currentText(),
            "augmentation": self._aug_chk.isChecked(),
        }

    def on_train_done(self) -> None:
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def append_log(self, msg: str) -> None:
        self._log_box.append(msg)
        # setValue를 이벤트 루프 다음 tick으로 지연 → paintEvent 중 recursive repaint 방지
        QTimer.singleShot(0, lambda: self._log_box.verticalScrollBar().setValue(
            self._log_box.verticalScrollBar().maximum()
        ))

    def set_progress(self, pct: int) -> None:
        self._progress.setValue(pct)
