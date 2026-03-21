"""
gui/settings_panel.py — 시스템 설정 패널 (독립형)

panels.py 의 ControlPanel 을 대체하거나 보완하는 독립형 설정 패널.
소스 설정 + 재생 제어 + AI 모델 설정 + 전처리 옵션을 하나의 탭에 통합.

외부 인터페이스:
  panel.get_settings() → dict       — 현재 설정값 조회
  panel.set_settings(d: dict)       — 설정값 일괄 지정
  on_start(settings: dict) 콜백     — 시작 버튼 클릭 시 호출
  on_stop() 콜백                    — 정지 버튼 클릭 시 호출
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk
from typing import Callable, Dict, Optional


_KO = ("맑은 고딕", 9)


class SettingsPanel(ttk.Frame):
    """소스·재생·AI·전처리 설정을 통합한 패널."""

    def __init__(
        self,
        master,
        on_start: Optional[Callable] = None,
        on_stop: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(master, padding=8, **kwargs)
        self._on_start = on_start
        self._on_stop  = on_stop
        self._running  = False

        self._build_source()
        self._build_playback()
        self._build_ai()
        self._build_preprocess()
        self._build_start_stop()

    # ── 소스 설정 ─────────────────────────────────────────────────────────

    def _build_source(self) -> None:
        lf = ttk.LabelFrame(self, text="📁  비디오 소스", padding=6)
        lf.pack(fill="x", pady=(0, 6))

        self._src_type = tk.StringVar(value="file")
        row = ttk.Frame(lf)
        row.pack(fill="x")
        for val, lbl in [("file", "파일"), ("camera", "카메라"), ("images", "이미지 폴더")]:
            ttk.Radiobutton(row, text=lbl, variable=self._src_type,
                            value=val, command=self._on_src_change).pack(
                side="left", padx=4)

        path_row = ttk.Frame(lf)
        path_row.pack(fill="x", pady=(6, 0))
        self._path_var = tk.StringVar()
        self._path_entry = ttk.Entry(path_row, textvariable=self._path_var, width=26)
        self._path_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(path_row, text="찾기…", width=5,
                   command=self._browse).pack(side="left", padx=(3, 0))

        cam_row = ttk.Frame(lf)
        cam_row.pack(fill="x", pady=(5, 0))
        ttk.Label(cam_row, text="카메라 번호:").pack(side="left")
        self._cam_idx = tk.IntVar(value=0)
        ttk.Spinbox(cam_row, from_=0, to=8, textvariable=self._cam_idx,
                    width=5).pack(side="left", padx=4)
        self._loop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(cam_row, text="반복 재생", variable=self._loop_var).pack(
            side="left", padx=(8, 0))

        self._on_src_change()

    def _on_src_change(self) -> None:
        is_cam = self._src_type.get() == "camera"
        self._path_entry.config(state="disabled" if is_cam else "normal")

    def _browse(self) -> None:
        t = self._src_type.get()
        if t == "file":
            p = filedialog.askopenfilename(
                title="비디오 파일 선택",
                filetypes=[("비디오", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("전체", "*.*")],
            )
        else:
            p = filedialog.askdirectory(title="이미지 폴더 선택")
        if p:
            self._path_var.set(p)

    # ── 재생 제어 ─────────────────────────────────────────────────────────

    def _build_playback(self) -> None:
        lf = ttk.LabelFrame(self, text="▶  재생 제어", padding=6)
        lf.pack(fill="x", pady=(0, 6))

        row = ttk.Frame(lf)
        row.pack(fill="x")
        ttk.Label(row, text="목표 FPS:").pack(side="left")
        self._fps_var = tk.DoubleVar(value=30.0)
        ttk.Spinbox(row, from_=1, to=120, increment=5, textvariable=self._fps_var,
                    width=6, format="%.0f").pack(side="left", padx=4)

        row2 = ttk.Frame(lf)
        row2.pack(fill="x", pady=(4, 0))
        ttk.Label(row2, text="최대 프레임:").pack(side="left")
        self._max_frames_var = tk.StringVar(value="")
        ttk.Entry(row2, textvariable=self._max_frames_var, width=10).pack(
            side="left", padx=4)
        ttk.Label(row2, text="(빈칸 = 무제한)").pack(side="left")

    # ── AI 모델 설정 ──────────────────────────────────────────────────────

    def _build_ai(self) -> None:
        lf = ttk.LabelFrame(self, text="🤖  AI 모델", padding=6)
        lf.pack(fill="x", pady=(0, 6))

        row = ttk.Frame(lf)
        row.pack(fill="x")
        ttk.Label(row, text="모델 유형:").pack(side="left")
        self._model_type = tk.StringVar(value="placeholder")
        for val, lbl in [("placeholder", "테스트"), ("onnx", "ONNX"), ("pytorch", "PyTorch")]:
            ttk.Radiobutton(row, text=lbl, variable=self._model_type, value=val).pack(
                side="left", padx=4)

        path_row = ttk.Frame(lf)
        path_row.pack(fill="x", pady=(5, 0))
        ttk.Label(path_row, text="모델 파일:").pack(side="left")
        self._model_path = tk.StringVar()
        ttk.Entry(path_row, textvariable=self._model_path, width=22).pack(
            side="left", fill="x", expand=True, padx=(4, 0))
        ttk.Button(path_row, text="찾기…", width=5, command=self._browse_model).pack(
            side="left", padx=(3, 0))

    def _browse_model(self) -> None:
        p = filedialog.askopenfilename(
            title="모델 파일 선택",
            filetypes=[("모델 파일", "*.onnx *.pt *.pth *.pkl"), ("전체", "*.*")],
        )
        if p:
            self._model_path.set(p)

    # ── 전처리 옵션 ───────────────────────────────────────────────────────

    def _build_preprocess(self) -> None:
        lf = ttk.LabelFrame(self, text="⚙  전처리 옵션", padding=6)
        lf.pack(fill="x", pady=(0, 6))

        self._grayscale_var = tk.BooleanVar(value=False)
        self._denoise_var   = tk.BooleanVar(value=False)
        self._equalize_var  = tk.BooleanVar(value=False)

        row = ttk.Frame(lf)
        row.pack(fill="x")
        ttk.Checkbutton(row, text="흑백 변환",    variable=self._grayscale_var).pack(side="left", padx=4)
        ttk.Checkbutton(row, text="노이즈 제거",  variable=self._denoise_var  ).pack(side="left", padx=4)
        ttk.Checkbutton(row, text="히스토그램 평활화", variable=self._equalize_var ).pack(side="left", padx=4)

    # ── 시작/정지 버튼 ────────────────────────────────────────────────────

    def _build_start_stop(self) -> None:
        row = ttk.Frame(self)
        row.pack(fill="x", pady=(6, 0))
        self._start_btn = ttk.Button(row, text="▶  분석 시작",
                                     style="Accent.TButton",
                                     command=self._start)
        self._start_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self._stop_btn = ttk.Button(row, text="■  정지", command=self._stop)
        self._stop_btn.pack(side="left", fill="x", expand=True)
        self._stop_btn.config(state="disabled")

    def _start(self) -> None:
        if self._on_start:
            self._on_start(self.get_settings())
        self._running = True
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")

    def _stop(self) -> None:
        if self._on_stop:
            self._on_stop()
        self.set_stopped()

    def set_stopped(self) -> None:
        self._running = False
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")

    # ── 설정 조회/설정 ────────────────────────────────────────────────────

    def get_settings(self) -> Dict:
        max_frames_str = self._max_frames_var.get().strip()
        try:
            max_frames = int(max_frames_str) if max_frames_str else None
        except ValueError:
            max_frames = None

        return {
            "source_type":  self._src_type.get(),
            "source_path":  self._path_var.get().strip(),
            "camera_index": self._cam_idx.get(),
            "loop":         self._loop_var.get(),
            "fps":          self._fps_var.get(),
            "max_frames":   max_frames,
            "model_type":   self._model_type.get(),
            "model_path":   self._model_path.get().strip() or None,
            "grayscale":    self._grayscale_var.get(),
            "denoise":      self._denoise_var.get(),
            "equalize":     self._equalize_var.get(),
        }

    def set_settings(self, d: Dict) -> None:
        if "source_type"  in d: self._src_type.set(d["source_type"])
        if "source_path"  in d: self._path_var.set(d["source_path"])
        if "camera_index" in d: self._cam_idx.set(d["camera_index"])
        if "loop"         in d: self._loop_var.set(d["loop"])
        if "fps"          in d: self._fps_var.set(d["fps"])
        if "model_type"   in d: self._model_type.set(d["model_type"])
        if "model_path"   in d: self._model_path.set(d["model_path"] or "")
        if "grayscale"    in d: self._grayscale_var.set(d["grayscale"])
        if "denoise"      in d: self._denoise_var.set(d["denoise"])
        if "equalize"     in d: self._equalize_var.set(d["equalize"])
        self._on_src_change()
