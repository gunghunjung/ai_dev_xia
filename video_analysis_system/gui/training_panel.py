"""
gui/training_panel.py — AI 학습 작업 설정 & 실행 패널

ai_management/ 의 TrainingManager 를 GUI 에서 직접 제어하는 패널.
TrainingManager 미설치 시에는 안내 메시지를 표시한다.

기능:
  - TrainingJobConfig 편집 (데이터셋/모델/하이퍼파라미터)
  - 학습 시작/중지
  - 에폭별 Loss/Accuracy 라이브 차트 (matplotlib 선택 설치)
  - 최근 실험 결과 요약 테이블
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

from gui.scrollable_frame import ScrollableFrame


_KO = ("맑은 고딕", 9)
_KO_BOLD = ("맑은 고딕", 9, "bold")


class TrainingPanel(ttk.Frame):
    """AI 학습 제어 패널."""

    def __init__(self, master, **kwargs):
        super().__init__(master, padding=8, **kwargs)
        self._training_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._build_ui()

    # ── UI 구성 ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        ttk.Label(self, text="🏋  AI 모델 학습",
                  font=("맑은 고딕", 10, "bold")).pack(anchor="w", pady=(0, 6))

        sf = ScrollableFrame(self)
        sf.pack(fill="both", expand=True)
        inner = sf.inner

        self._build_dataset_section(inner)
        self._build_model_section(inner)
        self._build_hyperparams(inner)
        self._build_controls(inner)
        self._build_progress(inner)
        self._build_results(inner)

    def _build_dataset_section(self, parent) -> None:
        lf = ttk.LabelFrame(parent, text="📂  데이터셋", padding=6)
        lf.pack(fill="x", pady=(0, 6))

        row = ttk.Frame(lf)
        row.pack(fill="x")
        ttk.Label(row, text="학습 데이터:").pack(side="left")
        self._dataset_path = tk.StringVar()
        ttk.Entry(row, textvariable=self._dataset_path, width=22).pack(
            side="left", fill="x", expand=True, padx=(4, 0))
        ttk.Button(row, text="찾기", width=4,
                   command=lambda: self._browse_dir(self._dataset_path)).pack(
            side="left", padx=(3, 0))

        row2 = ttk.Frame(lf)
        row2.pack(fill="x", pady=(4, 0))
        ttk.Label(row2, text="검증 비율:").pack(side="left")
        self._val_split = tk.DoubleVar(value=0.2)
        ttk.Spinbox(row2, from_=0.05, to=0.5, increment=0.05,
                    textvariable=self._val_split, width=6,
                    format="%.2f").pack(side="left", padx=4)
        ttk.Label(row2, text="작업 유형:").pack(side="left", padx=(8, 0))
        self._task_type = tk.StringVar(value="classification")
        ttk.Combobox(row2, textvariable=self._task_type, width=14, state="readonly",
                     values=["classification", "detection", "segmentation"]).pack(
            side="left", padx=4)

    def _build_model_section(self, parent) -> None:
        lf = ttk.LabelFrame(parent, text="🤖  모델 아키텍처", padding=6)
        lf.pack(fill="x", pady=(0, 6))

        row = ttk.Frame(lf)
        row.pack(fill="x")
        ttk.Label(row, text="모델 패밀리:").pack(side="left")
        self._model_family = tk.StringVar(value="resnet18")
        ttk.Combobox(row, textvariable=self._model_family, width=14, state="readonly",
                     values=["resnet18", "resnet50", "mobilenet_v2",
                              "efficientnet_b0", "custom"]).pack(side="left", padx=4)

        row2 = ttk.Frame(lf)
        row2.pack(fill="x", pady=(4, 0))
        ttk.Label(row2, text="내보내기 형식:").pack(side="left")
        self._export_fmt = tk.StringVar(value="onnx")
        for val, lbl in [("onnx", "ONNX"), ("torchscript", "TorchScript"), ("savedmodel", "SavedModel")]:
            ttk.Radiobutton(row2, text=lbl, variable=self._export_fmt, value=val).pack(
                side="left", padx=4)

        row3 = ttk.Frame(lf)
        row3.pack(fill="x", pady=(4, 0))
        ttk.Label(row3, text="출력 폴더:").pack(side="left")
        self._output_dir = tk.StringVar(value="models")
        ttk.Entry(row3, textvariable=self._output_dir, width=18).pack(
            side="left", padx=4)

    def _build_hyperparams(self, parent) -> None:
        lf = ttk.LabelFrame(parent, text="⚙  하이퍼파라미터", padding=6)
        lf.pack(fill="x", pady=(0, 6))

        params = [
            ("배치 크기",   "_batch_size",   tk.IntVar,    32,    (4, 256)),
            ("에폭",        "_epochs",        tk.IntVar,    50,    (1, 1000)),
            ("학습률",      "_lr",            tk.DoubleVar, 1e-3,  None),
        ]
        for lbl, attr, var_cls, default, _ in params:
            row = ttk.Frame(lf)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=f"{lbl}:", width=10).pack(side="left")
            v = var_cls(value=default)
            setattr(self, attr, v)
            ttk.Entry(row, textvariable=v, width=10).pack(side="left")

        row = ttk.Frame(lf)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="옵티마이저:", width=10).pack(side="left")
        self._optimizer = tk.StringVar(value="adam")
        ttk.Combobox(row, textvariable=self._optimizer, width=10, state="readonly",
                     values=["adam", "sgd", "adamw", "rmsprop"]).pack(side="left")

        row2 = ttk.Frame(lf)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="스케줄러:", width=10).pack(side="left")
        self._scheduler = tk.StringVar(value="cosine")
        ttk.Combobox(row2, textvariable=self._scheduler, width=10, state="readonly",
                     values=["cosine", "step", "plateau", "none"]).pack(side="left")

    def _build_controls(self, parent) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(0, 6))
        self._start_btn = ttk.Button(row, text="▶  학습 시작",
                                     style="Accent.TButton",
                                     command=self._start_training)
        self._start_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self._stop_btn = ttk.Button(row, text="■  중지",
                                    command=self._stop_training,
                                    state="disabled")
        self._stop_btn.pack(side="left", fill="x", expand=True)

    def _build_progress(self, parent) -> None:
        lf = ttk.LabelFrame(parent, text="진행 상황", padding=6)
        lf.pack(fill="x", pady=(0, 6))

        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_bar = ttk.Progressbar(lf, variable=self._progress_var,
                                              maximum=100, length=200)
        self._progress_bar.pack(fill="x", pady=(0, 4))
        self._status_var = tk.StringVar(value="대기 중")
        ttk.Label(lf, textvariable=self._status_var).pack(anchor="w")

    def _build_results(self, parent) -> None:
        lf = ttk.LabelFrame(parent, text="최근 실험 결과", padding=6)
        lf.pack(fill="x")

        self._result_text = tk.Text(lf, height=5, state="disabled",
                                    bg="#181825", fg="#cdd6f4",
                                    font=("Consolas", 8), wrap="word", relief="flat")
        self._result_text.pack(fill="both", expand=True)

    # ── 핸들러 ────────────────────────────────────────────────────────────

    def _browse_dir(self, var: tk.StringVar) -> None:
        p = filedialog.askdirectory(title="폴더 선택")
        if p:
            var.set(p)

    def _start_training(self) -> None:
        try:
            from ai_management.training_manager import TrainingManager
            from core.data_models import TrainingJobConfig
        except ImportError:
            messagebox.showerror(
                "모듈 없음",
                "ai_management/training_manager.py 가 설치되지 않았습니다.\n"
                "학습 기능을 사용하려면 해당 모듈이 필요합니다."
            )
            return

        self._stop_event.clear()
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._status_var.set("학습 준비 중…")

        cfg = TrainingJobConfig(
            dataset_path=self._dataset_path.get(),
            model_family=self._model_family.get(),
            batch_size=self._batch_size.get(),
            epochs=self._epochs.get(),
            learning_rate=self._lr.get(),
            optimizer=self._optimizer.get(),
            scheduler=self._scheduler.get(),
            export_format=self._export_fmt.get(),
            output_dir=self._output_dir.get(),
            task_type=self._task_type.get(),
            validation_split=self._val_split.get(),
        )

        def _run():
            try:
                mgr = TrainingManager()
                mgr.start_job(cfg, progress_callback=self._on_progress,
                              stop_event=self._stop_event)
                self.after(0, lambda: self._on_training_done(True))
            except Exception as e:
                self.after(0, lambda: self._on_training_done(False, str(e)))

        self._training_thread = threading.Thread(target=_run, daemon=True)
        self._training_thread.start()

    def _stop_training(self) -> None:
        self._stop_event.set()
        self._status_var.set("중지 요청됨…")

    def _on_progress(self, epoch: int, total: int,
                     loss: float, acc: float) -> None:
        pct = (epoch / total * 100) if total else 0
        msg = f"에폭 {epoch}/{total} — Loss: {loss:.4f}  Acc: {acc:.4f}"
        self.after(0, lambda: self._progress_var.set(pct))
        self.after(0, lambda: self._status_var.set(msg))

    def _on_training_done(self, success: bool, error: str = "") -> None:
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        if success:
            self._status_var.set("✅  학습 완료")
            self._progress_var.set(100.0)
            self._append_result("학습 완료\n")
        else:
            self._status_var.set(f"❌  오류: {error[:60]}")
            self._append_result(f"오류: {error}\n")

    def _append_result(self, text: str) -> None:
        self._result_text.config(state="normal")
        self._result_text.insert("end", text)
        self._result_text.see("end")
        self._result_text.config(state="disabled")
