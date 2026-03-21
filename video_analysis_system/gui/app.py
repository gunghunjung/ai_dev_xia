"""
gui/app.py — 메인 Tkinter 애플리케이션 윈도우 (한글 UI)

레이아웃:

  ┌──────────── 메뉴 바 ────────────────────────────────────┐
  │                         │                              │
  │   VideoCanvas           │   탭 노트북                   │
  │   (좌측, ~70%)           │  ┌────┬──────┬──────┬─────┐  │
  │                         │  │설정│ROI관리│ 상태 │이벤트│  │
  │                         │  └────┴──────┴──────┴─────┘  │
  ├─────────────────────────┴──────────────────────────────┤
  │  상태 바  (프레임 / FPS / 시스템상태 / 이벤트 건수)        │
  └────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

from config import (
    AIConfig, DecisionConfig, LoggingConfig, PreprocessConfig,
    ROIConfig, SystemConfig, TemporalConfig, VideoConfig, VisualizationConfig,
)
from gui.panels import ControlPanel, EventPanel, ROIPanel, StatusPanel
from gui.video_canvas import VideoCanvas
from gui.workers import EngineWorker, FrameResult


# ---------------------------------------------------------------------------
# 다크 테마 (Catppuccin Mocha 계열)
# ---------------------------------------------------------------------------

def _apply_dark_theme(root: tk.Tk) -> None:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    BG   = "#1e1e2e"
    BG2  = "#181825"
    FG   = "#cdd6f4"
    SEL  = "#313244"
    ACC  = "#cba6f7"
    BRD  = "#45475a"

    style.configure(".",
                    background=BG, foreground=FG,
                    fieldbackground=BG2, troughcolor=BG2,
                    selectbackground=SEL, selectforeground=FG,
                    bordercolor=BRD, lightcolor=BRD, darkcolor=BRD,
                    relief="flat",
                    font=("맑은 고딕", 9))

    style.configure("TFrame",       background=BG)
    style.configure("TLabel",       background=BG, foreground=FG)
    style.configure("TLabelFrame",  background=BG, foreground=ACC)
    style.configure("TLabelframe",  background=BG, foreground=ACC)
    style.configure("TLabelframe.Label",
                    background=BG, foreground=ACC,
                    font=("맑은 고딕", 9, "bold"))
    style.configure("TNotebook",    background=BG2, borderwidth=0)
    style.configure("TNotebook.Tab",
                    background=BG2, foreground=FG,
                    padding=[10, 4],
                    font=("맑은 고딕", 9))
    style.map("TNotebook.Tab",
              background=[("selected", BG)],
              foreground=[("selected", ACC)])
    style.configure("TButton",      background=SEL, foreground=FG, padding=[6, 3])
    style.map("TButton",
              background=[("active", "#45475a")],
              foreground=[("active", FG)])
    style.configure("Accent.TButton",
                    background=ACC, foreground=BG2,
                    font=("맑은 고딕", 9, "bold"))
    style.map("Accent.TButton",
              background=[("active", "#d0bfff")])
    style.configure("Help.TButton",
                    background="#313244", foreground="#89dceb",
                    font=("맑은 고딕", 9, "bold"), padding=[6, 3])
    style.map("Help.TButton",
              background=[("active", "#45475a")])
    style.configure("TEntry",       fieldbackground=BG2, foreground=FG, insertcolor=FG)
    style.configure("TSpinbox",     fieldbackground=BG2, foreground=FG, arrowcolor=FG)
    style.configure("TCheckbutton", background=BG, foreground=FG)
    style.configure("TRadiobutton", background=BG, foreground=FG)
    style.configure("Treeview",
                    background=BG2, foreground=FG,
                    fieldbackground=BG2, rowheight=20)
    style.configure("Treeview.Heading",
                    background=SEL, foreground=ACC,
                    font=("맑은 고딕", 8, "bold"))
    style.map("Treeview", background=[("selected", SEL)])
    style.configure("Vertical.TScrollbar",
                    background=BG2, troughcolor=BG2, arrowcolor=FG)
    style.configure("TSeparator", background=BRD)

    root.configure(bg=BG)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class App(tk.Tk):
    """루트 애플리케이션 윈도우."""

    CANVAS_W = 820
    CANVAS_H = 560
    PANEL_W  = 350
    POLL_MS  = 30       # GUI 폴링 주기 (ms)

    def __init__(self):
        super().__init__()
        self.title("비디오 분석 시스템")
        self.resizable(True, True)
        _apply_dark_theme(self)

        self._worker: Optional[EngineWorker] = None
        self._source_w: int = self.CANVAS_W
        self._source_h: int = self.CANVAS_H
        self._poll_id = None

        self._build_menu()
        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.update_idletasks()
        self.minsize(920, 640)

    # ── 메뉴 ──────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        menubar = tk.Menu(self, bg="#181825", fg="#cdd6f4",
                          activebackground="#313244", activeforeground="#cdd6f4",
                          borderwidth=0)

        # 파일 메뉴
        file_menu = tk.Menu(menubar, tearoff=False,
                            bg="#181825", fg="#cdd6f4",
                            activebackground="#313244")
        file_menu.add_command(label="비디오 파일 열기…",  command=self._menu_open_file)
        file_menu.add_command(label="카메라 연결…",       command=self._menu_open_camera)
        file_menu.add_separator()
        file_menu.add_command(label="로그 폴더 열기…",    command=self._menu_open_log_folder)
        file_menu.add_command(label="로그 내보내기 정보",  command=self._menu_export_info)
        file_menu.add_separator()
        file_menu.add_command(label="종료",               command=self._on_close)
        menubar.add_cascade(label="파일", menu=file_menu)

        # 보기 메뉴
        view_menu = tk.Menu(menubar, tearoff=False,
                            bg="#181825", fg="#cdd6f4",
                            activebackground="#313244")
        view_menu.add_command(label="이벤트 목록 초기화",  command=self._menu_clear_events)
        view_menu.add_separator()
        view_menu.add_command(label="상태 탭으로 이동",    command=lambda: self._notebook.select(2))
        view_menu.add_command(label="이벤트 탭으로 이동",  command=lambda: self._notebook.select(3))
        menubar.add_cascade(label="보기", menu=view_menu)

        # 도움말 메뉴
        help_menu = tk.Menu(menubar, tearoff=False,
                            bg="#181825", fg="#cdd6f4",
                            activebackground="#313244")
        help_menu.add_command(label="사용법 도움말…",      command=self._show_help)
        help_menu.add_separator()
        help_menu.add_command(label="프로그램 정보",        command=self._menu_about)
        menubar.add_cascade(label="도움말", menu=help_menu)

        self.config(menu=menubar)

    # ── 레이아웃 ──────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        # 상단 툴바 (사용법 버튼 포함)
        toolbar = tk.Frame(self, bg="#181825", pady=4)
        toolbar.pack(fill="x", padx=0)
        tk.Label(toolbar, text="  🎬 비디오 분석 시스템",
                 bg="#181825", fg="#cba6f7",
                 font=("맑은 고딕", 11, "bold")).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="❓ 사용법 도움말",
                   style="Help.TButton",
                   command=self._show_help).pack(side="right", padx=(0, 10))
        ttk.Separator(self, orient="horizontal").pack(fill="x")

        # 메인 컨텐츠 영역
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=6, pady=(4, 0))

        # 좌측 — 비디오 캔버스
        left = ttk.Frame(main_frame)
        left.pack(side="left", fill="both", expand=True)

        self._canvas = VideoCanvas(left, width=self.CANVAS_W, height=self.CANVAS_H)
        self._canvas.pack(fill="both", expand=True)

        # 우측 — 탭 패널
        right = ttk.Frame(main_frame, width=self.PANEL_W)
        right.pack(side="left", fill="y", padx=(6, 0))
        right.pack_propagate(False)

        self._notebook = ttk.Notebook(right)
        self._notebook.pack(fill="both", expand=True)

        # 탭 1 — 설정 (컨트롤)
        self._ctrl_panel = ControlPanel(
            self._notebook, on_start=self._on_start, on_stop=self._on_stop
        )
        self._notebook.add(self._ctrl_panel, text="⚙  설정")

        # 탭 2 — ROI 관리
        self._roi_panel = ROIPanel(
            self._notebook, on_rois_changed=self._on_rois_changed
        )
        self._roi_panel.draw_callback = self._toggle_draw_mode
        self._notebook.add(self._roi_panel, text="□  ROI 관리")

        # 탭 3 — 상태
        self._status_panel = StatusPanel(self._notebook)
        self._notebook.add(self._status_panel, text="◉  상태")

        # 탭 4 — 이벤트
        self._event_panel = EventPanel(self._notebook)
        self._notebook.add(self._event_panel, text="⚠  이벤트")

        # 상태 바
        self._build_status_bar()

        # 캔버스 ROI 그리기 연결 (초기에는 비활성)
        self._canvas.enable_roi_drawing(callback=self._on_canvas_roi_drawn)
        self._canvas.disable_roi_drawing()

    def _build_status_bar(self) -> None:
        bar = tk.Frame(self, bg="#181825")
        bar.pack(fill="x", side="bottom", padx=0, pady=0)
        ttk.Separator(bar, orient="horizontal").pack(fill="x")
        inner = tk.Frame(bar, bg="#181825", pady=3)
        inner.pack(fill="x")

        self._sb_frame  = tk.StringVar(value="프레임: —")
        self._sb_state  = tk.StringVar(value="상태: 대기")
        self._sb_fps    = tk.StringVar(value="FPS: —")
        self._sb_events = tk.StringVar(value="이벤트: 0건")

        for v, w in [(self._sb_frame, 14), (self._sb_state, 18),
                     (self._sb_fps, 12), (self._sb_events, 14)]:
            tk.Label(inner, textvariable=v, bg="#181825", fg="#6c7086",
                     font=("맑은 고딕", 8), width=w, anchor="w").pack(side="left", padx=(8, 0))

        # 우측: 버전 정보
        tk.Label(inner, text="Video Analysis System v1.0",
                 bg="#181825", fg="#313244",
                 font=("맑은 고딕", 7)).pack(side="right", padx=10)

    # ── 시작 / 정지 ───────────────────────────────────────────────────────

    def _on_start(self, overrides: dict) -> None:
        # 소스 경로 기본 검증
        src_type = overrides.get("source_type", "file")
        src_path = overrides.get("source_path", "").strip()
        if src_type in ("file", "images") and not src_path:
            messagebox.showwarning("소스 없음",
                                   "비디오 파일 또는 이미지 폴더 경로를 입력해 주세요.\n"
                                   "[찾기…] 버튼으로 탐색할 수 있습니다.")
            self._ctrl_panel.set_stopped()
            return

        if self._worker and self._worker.is_running:
            self._on_stop()

        cfg = self._build_config(overrides)
        self._worker = EngineWorker(cfg)
        self._worker.start()
        self._start_polling()
        self._set_status("분석 중…")
        # 분석 시작 시 상태 탭으로 자동 이동
        self._notebook.select(2)

    def _on_stop(self) -> None:
        if self._worker:
            self._worker.stop()
        self._stop_polling()
        self._canvas.clear_frame()
        self._ctrl_panel.set_stopped()
        self._set_status("정지됨.")

    def _build_config(self, ov: dict) -> SystemConfig:
        rois = [
            ROIConfig(
                roi_id=r["roi_id"],
                bbox=tuple(r["bbox"]),    # type: ignore[arg-type]
                label=r.get("label", ""),
            )
            for r in self._roi_panel.get_roi_configs()
        ]

        cfg = SystemConfig()
        cfg.video = VideoConfig(
            source_type=ov.get("source_type", "file"),
            source_path=ov.get("source_path", ""),
            camera_index=ov.get("camera_index", 0),
            target_fps=ov.get("fps", 30.0),
            loop=ov.get("loop", False),
        )
        cfg.rois       = rois
        cfg.preprocess = PreprocessConfig()
        cfg.temporal   = TemporalConfig(window_size=60, min_frames_for_decision=10)
        cfg.ai         = AIConfig(
            model_type=ov.get("model_type", "placeholder"),
            model_path=ov.get("model_path"),
            class_names=["normal", "abnormal"],
        )
        cfg.decision   = DecisionConfig()
        cfg.logging    = LoggingConfig(log_dir="logs")
        cfg.visualization = VisualizationConfig()
        cfg.max_frames = ov.get("max_frames")
        return cfg

    # ── GUI 폴링 ──────────────────────────────────────────────────────────

    def _start_polling(self) -> None:
        self._poll()

    def _stop_polling(self) -> None:
        if self._poll_id:
            self.after_cancel(self._poll_id)
            self._poll_id = None

    def _poll(self) -> None:
        if self._worker is None:
            return

        drained = 0
        while drained < 4:
            try:
                result: FrameResult = self._worker.result_queue.get_nowait()
            except Exception:
                break
            drained += 1
            self._handle_result(result)

        if not self._worker.is_running:
            if self._worker.last_error:
                messagebox.showerror("엔진 오류",
                                     f"분석 중 오류가 발생했습니다:\n\n{self._worker.last_error}\n\n"
                                     "로그 파일을 확인하거나 소스 설정을 점검해 주세요.")
            self._ctrl_panel.set_stopped()
            self._set_status("완료.")
            return

        self._poll_id = self.after(self.POLL_MS, self._poll)

    def _handle_result(self, result: FrameResult) -> None:
        if result.warning:
            self._set_status(f"⚠  {result.warning}")
            return

        # 비디오 캔버스 갱신 (CoordinateTransform 은 <Configure> 이벤트로 자동 갱신)
        if result.display_frame is not None:
            self._canvas.update_frame(result.display_frame)

        # 상태 패널 갱신
        self._status_panel.update_status(result)

        # 이벤트 처리
        if result.new_event:
            self._event_panel.add_event(result.new_event)
            cnt = len(self._event_panel._events)
            self._sb_events.set(f"이벤트: {cnt}건")
            # 첫 이벤트 발생 시 이벤트 탭으로 전환
            if cnt == 1:
                self._notebook.select(3)

        # 상태 바 갱신
        self._sb_frame.set(f"프레임: {result.frame_index:,}")
        self._sb_state.set(f"상태: {result.system_state.value}")
        self._sb_fps.set(f"FPS: {result.fps_actual:.1f}")

    # ── ROI 관리 ──────────────────────────────────────────────────────────

    def _toggle_draw_mode(self, enabled: bool) -> None:
        if enabled:
            self._canvas.enable_roi_drawing(callback=self._on_canvas_roi_drawn)
        else:
            self._canvas.disable_roi_drawing()

    def _on_canvas_roi_drawn(self, roi_id: str,
                              original_rect: tuple) -> None:
        """캔버스에서 ROI 드래그가 완료됐을 때 호출됩니다.

        VideoCanvas 가 이미 원본 프레임 좌표로 역변환해서 전달하므로
        추가 변환 없이 바로 저장합니다.
        """
        self._roi_panel.add_roi(roi_id, original_rect)
        self._canvas.add_roi_overlay(roi_id, original_rect)

    def _on_rois_changed(self, roi_list: list) -> None:
        """ROI 목록이 변경되면 캔버스 오버레이를 동기화합니다."""
        self._canvas.clear_roi_overlays()
        for r in roi_list:
            self._canvas.add_roi_overlay(r["roi_id"], tuple(r["bbox"]))

    # ── 도움말 ────────────────────────────────────────────────────────────

    def _show_help(self) -> None:
        from gui.help_dialog import HelpDialog
        HelpDialog(self)

    # ── 메뉴 액션 ─────────────────────────────────────────────────────────

    def _menu_open_file(self) -> None:
        from tkinter import filedialog
        p = filedialog.askopenfilename(
            title="비디오 파일 선택",
            filetypes=[("비디오 파일", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("전체", "*.*")]
        )
        if p:
            self._ctrl_panel._src_type.set("file")
            self._ctrl_panel._path_var.set(p)
            self._ctrl_panel._on_src_change()
            self._notebook.select(0)

    def _menu_open_camera(self) -> None:
        self._ctrl_panel._src_type.set("camera")
        self._ctrl_panel._on_src_change()
        self._notebook.select(0)

    def _menu_open_log_folder(self) -> None:
        import os, subprocess
        log_path = os.path.abspath("logs")
        os.makedirs(log_path, exist_ok=True)
        try:
            subprocess.Popen(f'explorer "{log_path}"')
        except Exception:
            messagebox.showinfo("로그 폴더", f"로그 저장 위치:\n{log_path}")

    def _menu_export_info(self) -> None:
        messagebox.showinfo(
            "로그 내보내기",
            "분석이 종료(■ 정지)되면 자동으로 로그가 저장됩니다.\n\n"
            "저장 위치:  logs/\n"
            "  • frame_log_YYYYMMDD_HHmmss.csv\n"
            "  • frame_log_YYYYMMDD_HHmmss.json\n"
            "  • events/events.json\n"
            "  • events/evt_*_snapshot.jpg\n"
            "  • events/evt_*_clip.avi"
        )

    def _menu_clear_events(self) -> None:
        self._event_panel._clear()
        self._sb_events.set("이벤트: 0건")

    def _menu_about(self) -> None:
        messagebox.showinfo(
            "프로그램 정보",
            "비디오 분석 시스템  v1.0\n\n"
            "모듈형 영상 이상 감지 파이프라인\n\n"
            "처리 순서:\n"
            "  FrameSource → Preprocessor → ROIManager\n"
            "  → FeatureExtractor → TemporalBuffer\n"
            "  → InferenceEngine → DecisionEngine\n"
            "  → Visualization → Logger\n\n"
            "사용 기술: Python · OpenCV · Tkinter · NumPy\n"
            "AI 백엔드: Placeholder / ONNX / PyTorch\n\n"
            "자세한 사용법은 [도움말] → [사용법 도움말]을 참조하세요."
        )

    # ── 윈도우 종료 ───────────────────────────────────────────────────────

    def _on_close(self) -> None:
        if self._worker and self._worker.is_running:
            if not messagebox.askyesno("종료 확인", "분석이 실행 중입니다.\n종료하시겠습니까?\n\n(로그는 자동 저장됩니다.)"):
                return
        self._on_stop()
        if self._worker:
            self._worker.join(timeout=2.0)
        self.destroy()

    def _set_status(self, msg: str) -> None:
        self._sb_state.set(msg)
