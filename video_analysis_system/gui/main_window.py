"""
gui/main_window.py — Main window with full viewport control, keyboard shortcuts,
and ROI fullscreen toggle.

Layout
──────
┌──────────── 툴바 ──────────────────────────────────────────────┐
├──── 좌측 (비디오 캔버스) ──────┬─ 우측 (탭 노트북) ──────────────┤
│                                │ ⚙설정│□ROI│◉상태│⚠이벤트      │
│   VideoCanvas                  │ 🤖AI관리                       │
│   • 마우스 휠    → 줌           │                                │
│   • 좌클릭 드래그 → 팬           │                                │
│   • PIP 미니맵  (우상단 오버레이) │                                │
├────────────────────────────────┴────────────────────────────────┤
│  상태바: 프레임 / FPS / 상태 / 이벤트 / 소스해상도 / 줌 / 모드       │
└──────────────────────────────────────────────────────────────────┘

Keyboard shortcuts
──────────────────
  +  / =          Zoom in
  -               Zoom out
  0               Zoom reset (100 %)
  F               ROI fullscreen toggle (selected ROI)
  Esc             Exit fullscreen / reset viewport
  R               Enable ROI-draw mode
  P               Disable ROI-draw mode (pan mode)
  ← ↑ → ↓        Pan the viewport (5 display-pixel steps)
"""

from __future__ import annotations

import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from config import (
    AIConfig, DecisionConfig, LoggingConfig, PreprocessConfig,
    ROIConfig, SystemConfig, TemporalConfig, VideoConfig, VisualizationConfig,
)
from gui.panels import ControlPanel, EventPanel, ROIPanel, StatusPanel
from gui.video_canvas import VideoCanvas
from gui.workers import EngineWorker, FrameResult


# ---------------------------------------------------------------------------
# Dark theme (Catppuccin Mocha palette)
# ---------------------------------------------------------------------------

def _apply_dark_theme(root: tk.Tk) -> None:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    BG  = "#1e1e2e";  BG2 = "#181825";  FG  = "#cdd6f4"
    SEL = "#313244";  ACC = "#cba6f7";  BRD = "#45475a"

    style.configure(".", background=BG, foreground=FG,
                    fieldbackground=BG2, troughcolor=BG2,
                    selectbackground=SEL, selectforeground=FG,
                    bordercolor=BRD, lightcolor=BRD, darkcolor=BRD,
                    relief="flat", font=("맑은 고딕", 9))
    style.configure("TFrame",      background=BG)
    style.configure("TLabel",      background=BG, foreground=FG)
    style.configure("TLabelFrame", background=BG, foreground=ACC)
    style.configure("TLabelframe", background=BG, foreground=ACC)
    style.configure("TLabelframe.Label", background=BG, foreground=ACC,
                    font=("맑은 고딕", 9, "bold"))
    style.configure("TNotebook",   background=BG2, borderwidth=0)
    style.configure("TNotebook.Tab", background=BG2, foreground=FG,
                    padding=[9, 4], font=("맑은 고딕", 9))
    style.map("TNotebook.Tab",
              background=[("selected", BG)],
              foreground=[("selected", ACC)])
    style.configure("TButton",     background=SEL, foreground=FG, padding=[6, 3])
    style.map("TButton",
              background=[("active", BRD)],
              foreground=[("active", FG)])
    style.configure("Accent.TButton", background=ACC, foreground=BG2,
                    font=("맑은 고딕", 9, "bold"))
    style.map("Accent.TButton", background=[("active", "#d0bfff")])
    style.configure("Help.TButton", background=SEL, foreground="#89dceb",
                    font=("맑은 고딕", 9, "bold"), padding=[6, 3])
    style.map("Help.TButton", background=[("active", BRD)])
    style.configure("Warn.TButton", background=BRD, foreground="#f38ba8",
                    font=("맑은 고딕", 9, "bold"), padding=[6, 3])
    style.configure("TEntry",     fieldbackground=BG2, foreground=FG, insertcolor=FG)
    style.configure("TSpinbox",   fieldbackground=BG2, foreground=FG, arrowcolor=FG)
    style.configure("TCheckbutton", background=BG, foreground=FG)
    style.configure("TRadiobutton", background=BG, foreground=FG)
    style.configure("Treeview",   background=BG2, foreground=FG,
                    fieldbackground=BG2, rowheight=20)
    style.configure("Treeview.Heading", background=SEL, foreground=ACC,
                    font=("맑은 고딕", 8, "bold"))
    style.map("Treeview", background=[("selected", SEL)])
    style.configure("Vertical.TScrollbar",
                    background=BG2, troughcolor=BG2, arrowcolor=FG)
    style.configure("TSeparator", background=BRD)
    root.configure(bg=BG)


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

_PAN_STEP = 30   # arrow-key pan step in display pixels


class MainWindow(tk.Tk):
    """
    Main application window with professional viewport control.

    Key capabilities (beyond the original App):
      • ViewportManager inside VideoCanvas for zoom + pan + ROI fullscreen
      • Keyboard shortcuts for all viewport operations
      • PIP minimap drawn directly on the canvas
      • Status bar shows zoom level and current interaction mode
      • Non-functional UI elements removed (no stale Refresh buttons)
      • AI 관리 탭 (lazy-loaded)
    """

    CANVAS_W = 820
    CANVAS_H = 560
    PANEL_W  = 370
    POLL_MS  = 30

    def __init__(self) -> None:
        super().__init__()
        self.title("비디오 분석 시스템 — AI 통합 에디션 v3")
        self.resizable(True, True)
        _apply_dark_theme(self)

        self._worker:       Optional[EngineWorker] = None
        self._poll_id:      Optional[str]          = None
        self._event_count:  int  = 0

        self._build_menu()
        self._build_layout()
        self._bind_keyboard()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.update_idletasks()
        self.minsize(980, 660)

    # ─────────────────────────────────────────────────────────────────────
    # Menu
    # ─────────────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        _c = dict(bg="#181825", fg="#cdd6f4",
                  activebackground="#313244", activeforeground="#cdd6f4")
        mb = tk.Menu(self, **_c, borderwidth=0)

        # 파일
        fm = tk.Menu(mb, tearoff=False, **_c)
        fm.add_command(label="비디오 파일 열기…",  command=self._menu_open_file)
        fm.add_command(label="카메라 연결…",        command=self._menu_open_camera)
        fm.add_separator()
        fm.add_command(label="로그 폴더 열기",      command=self._menu_log_folder)
        fm.add_separator()
        fm.add_command(label="종료",                command=self._on_close)
        mb.add_cascade(label="파일", menu=fm)

        # 보기
        vm = tk.Menu(mb, tearoff=False, **_c)
        vm.add_command(label="줌 확대  (+)",        command=self._zoom_in,    accelerator="+")
        vm.add_command(label="줌 축소  (−)",        command=self._zoom_out,   accelerator="-")
        vm.add_command(label="줌 초기화  (0)",      command=self._zoom_reset, accelerator="0")
        vm.add_separator()
        vm.add_command(label="ROI 확대 보기  (F)",  command=self._toggle_roi_fullscreen,
                       accelerator="F")
        vm.add_command(label="화면 초기화  (Esc)",  command=self._escape_mode, accelerator="Esc")
        vm.add_separator()
        vm.add_command(label="ROI 그리기 모드  (R)", command=self._set_draw_mode,   accelerator="R")
        vm.add_command(label="팬 모드  (P)",         command=self._set_pan_mode,    accelerator="P")
        vm.add_separator()
        vm.add_command(label="이벤트 목록 초기화",    command=self._menu_clear_events)
        mb.add_cascade(label="보기", menu=vm)

        # 도움말
        hm = tk.Menu(mb, tearoff=False, **_c)
        hm.add_command(label="사용법 도움말…",      command=self._show_help)
        hm.add_command(label="단축키 목록",          command=self._show_shortcuts)
        hm.add_command(label="프로그램 정보",        command=self._menu_about)
        mb.add_cascade(label="도움말", menu=hm)

        self.config(menu=mb)

    # ─────────────────────────────────────────────────────────────────────
    # Layout
    # ─────────────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        # ── Toolbar ───────────────────────────────────────────────────────
        toolbar = tk.Frame(self, bg="#181825", pady=4)
        toolbar.pack(fill="x")

        tk.Label(toolbar, text="  🎬 비디오 분석 시스템",
                 bg="#181825", fg="#cba6f7",
                 font=("맑은 고딕", 11, "bold")).pack(side="left", padx=(8, 0))

        # Zoom buttons
        zm = tk.Frame(toolbar, bg="#181825")
        zm.pack(side="left", padx=(20, 0))
        for txt, cmd, tip in [
            ("🔍+",   self._zoom_in,    "줌 확대 (+)"),
            ("🔍−",   self._zoom_out,   "줌 축소 (-)"),
            ("⟲ 1:1", self._zoom_reset, "줌 초기화 (0)"),
        ]:
            btn = tk.Button(
                zm, text=txt, bg="#313244", fg="#cdd6f4",
                font=("맑은 고딕", 9), relief="flat", padx=6, pady=2,
                command=cmd,
            )
            btn.pack(side="left", padx=2)

        # ROI-draw / Pan mode buttons
        mode_sep = tk.Frame(toolbar, bg="#45475a", width=1)
        mode_sep.pack(side="left", fill="y", padx=(12, 4), pady=2)
        for txt, cmd, tip in [
            ("✏ ROI 그리기", self._set_draw_mode, "ROI 그리기 모드 (R)"),
            ("✋ 팬",         self._set_pan_mode,  "팬 모드 (P)"),
            ("⬛ ROI 확대",   self._toggle_roi_fullscreen, "선택한 ROI 전체 화면 (F)"),
        ]:
            btn = tk.Button(
                toolbar, text=txt, bg="#313244", fg="#cdd6f4",
                font=("맑은 고딕", 9), relief="flat", padx=6, pady=2,
                command=cmd,
            )
            btn.pack(side="left", padx=2)

        # Help button (right-aligned)
        ttk.Button(
            toolbar, text="❓ 사용법", style="Help.TButton",
            command=self._show_help,
        ).pack(side="right", padx=(0, 10))

        ttk.Separator(self, orient="horizontal").pack(fill="x")

        # ── Main content ──────────────────────────────────────────────────
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=6, pady=(4, 0))

        # Left — video canvas
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        self._canvas = VideoCanvas(left, width=self.CANVAS_W, height=self.CANVAS_H)
        self._canvas.pack(fill="both", expand=True)

        # Right — tab notebook
        right = ttk.Frame(main, width=self.PANEL_W)
        right.pack(side="left", fill="y", padx=(6, 0))
        right.pack_propagate(False)

        self._notebook = ttk.Notebook(right)
        self._notebook.pack(fill="both", expand=True)

        self._ctrl_panel = ControlPanel(
            self._notebook, on_start=self._on_start, on_stop=self._on_stop)
        self._notebook.add(self._ctrl_panel, text="⚙  설정")

        self._roi_panel = ROIPanel(
            self._notebook, on_rois_changed=self._on_rois_changed)
        self._roi_panel.draw_callback = self._toggle_draw_mode
        self._notebook.add(self._roi_panel, text="□  ROI")

        self._status_panel = StatusPanel(self._notebook)
        self._notebook.add(self._status_panel, text="◉  상태")

        self._event_panel = EventPanel(self._notebook)
        self._notebook.add(self._event_panel, text="⚠  이벤트")

        # AI management tab (lazy load)
        self._ai_panel_frame = ttk.Frame(self._notebook)
        self._notebook.add(self._ai_panel_frame, text="🤖  AI 관리")
        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # ── Status bar ────────────────────────────────────────────────────
        self._build_status_bar()

        # Canvas virtual event listeners
        self._canvas.bind("<<ZoomChanged>>",    self._on_zoom_changed)
        self._canvas.bind("<<ViewportChanged>>", self._on_viewport_changed)

        # ROI draw mode off by default
        self._canvas.enable_roi_drawing(callback=self._on_canvas_roi_drawn)
        self._canvas.disable_roi_drawing()

    def _build_status_bar(self) -> None:
        bar = tk.Frame(self, bg="#181825")
        bar.pack(fill="x", side="bottom")
        ttk.Separator(bar, orient="horizontal").pack(fill="x")
        inner = tk.Frame(bar, bg="#181825", pady=3)
        inner.pack(fill="x")

        self._sb_frame  = tk.StringVar(value="프레임: —")
        self._sb_state  = tk.StringVar(value="상태: 대기")
        self._sb_fps    = tk.StringVar(value="FPS: —")
        self._sb_events = tk.StringVar(value="이벤트: 0건")
        self._sb_src    = tk.StringVar(value="소스: —")
        self._sb_zoom   = tk.StringVar(value="줌: 100%")
        self._sb_mode   = tk.StringVar(value="모드: 팬")

        for v, w_ in [
            (self._sb_frame,  12), (self._sb_state,  16),
            (self._sb_fps,    10), (self._sb_events,  10),
            (self._sb_src,    14), (self._sb_zoom,     9),
            (self._sb_mode,   10),
        ]:
            tk.Label(inner, textvariable=v, bg="#181825", fg="#6c7086",
                     font=("맑은 고딕", 8), width=w_, anchor="w").pack(
                side="left", padx=(8, 0))

        tk.Label(inner, text="Video Analysis System v3.0",
                 bg="#181825", fg="#313244",
                 font=("맑은 고딕", 7)).pack(side="right", padx=10)

    # ─────────────────────────────────────────────────────────────────────
    # Keyboard bindings
    # ─────────────────────────────────────────────────────────────────────

    def _bind_keyboard(self) -> None:
        """Register all keyboard shortcuts on the root window."""
        # Zoom
        self.bind_all("<plus>",     lambda _e: self._zoom_in())
        self.bind_all("<equal>",    lambda _e: self._zoom_in())    # = without Shift
        self.bind_all("<minus>",    lambda _e: self._zoom_out())
        self.bind_all("<Key-0>",    lambda _e: self._zoom_reset())

        # Viewport escape
        self.bind_all("<Escape>",   lambda _e: self._escape_mode())

        # ROI fullscreen toggle
        self.bind_all("<Key-f>",    lambda _e: self._toggle_roi_fullscreen())
        self.bind_all("<Key-F>",    lambda _e: self._toggle_roi_fullscreen())

        # Mode switch
        self.bind_all("<Key-r>",    lambda _e: self._set_draw_mode())
        self.bind_all("<Key-R>",    lambda _e: self._set_draw_mode())
        self.bind_all("<Key-p>",    lambda _e: self._set_pan_mode())
        self.bind_all("<Key-P>",    lambda _e: self._set_pan_mode())

        # Arrow-key pan
        self.bind_all("<Left>",     lambda _e: self._canvas.pan(-_PAN_STEP, 0))
        self.bind_all("<Right>",    lambda _e: self._canvas.pan( _PAN_STEP, 0))
        self.bind_all("<Up>",       lambda _e: self._canvas.pan(0, -_PAN_STEP))
        self.bind_all("<Down>",     lambda _e: self._canvas.pan(0,  _PAN_STEP))

    # ─────────────────────────────────────────────────────────────────────
    # Viewport commands
    # ─────────────────────────────────────────────────────────────────────

    def _zoom_in(self) -> None:
        self._canvas.zoom_in()
        self._update_viewport_status()

    def _zoom_out(self) -> None:
        self._canvas.zoom_out()
        self._update_viewport_status()

    def _zoom_reset(self) -> None:
        self._canvas.zoom_reset()
        self._update_viewport_status()

    def _escape_mode(self) -> None:
        """ESC: exit ROI fullscreen, then reset zoom."""
        if self._canvas.viewport.is_roi_fullscreen:
            self._canvas.exit_focus()
        else:
            self._canvas.zoom_reset()
        self._canvas.disable_roi_drawing()
        self._update_viewport_status()

    def _toggle_roi_fullscreen(self) -> None:
        """
        Toggle ROI fullscreen for the currently selected ROI.
        If no ROI is selected, picks the first available one.
        """
        roi_id = self._canvas.selected_roi_id
        if roi_id is None:
            # Fall back to first available ROI
            ids = self._canvas.list_roi_ids()
            if ids:
                roi_id = ids[0]
        if roi_id:
            self._canvas.toggle_focus(roi_id)
            self._update_viewport_status()

    def _on_zoom_changed(self, _event: tk.Event) -> None:
        self._update_viewport_status()

    def _on_viewport_changed(self, _event: tk.Event) -> None:
        self._update_viewport_status()

    def _update_viewport_status(self) -> None:
        zoom_pct = self._canvas.zoom_factor * 100
        if self._canvas.viewport.is_roi_fullscreen:
            self._sb_zoom.set(f"줌: {zoom_pct:.0f}% [ROI]")
        else:
            self._sb_zoom.set(f"줌: {zoom_pct:.0f}%")

    # ─────────────────────────────────────────────────────────────────────
    # Interaction mode: ROI draw vs pan
    # ─────────────────────────────────────────────────────────────────────

    def _set_draw_mode(self) -> None:
        self._canvas.enable_roi_drawing(callback=self._on_canvas_roi_drawn)
        self._sb_mode.set("모드: ROI그리기")

    def _set_pan_mode(self) -> None:
        self._canvas.disable_roi_drawing()
        self._sb_mode.set("모드: 팬")

    def _toggle_draw_mode(self, enabled: bool) -> None:
        """Called by ROIPanel draw button."""
        if enabled:
            self._set_draw_mode()
        else:
            self._set_pan_mode()

    # ─────────────────────────────────────────────────────────────────────
    # Tab change
    # ─────────────────────────────────────────────────────────────────────

    def _on_tab_changed(self, _event: tk.Event) -> None:
        idx = self._notebook.index("current")
        if idx == 4 and not hasattr(self, "_ai_panel_loaded"):
            self._load_ai_panel()

    def _load_ai_panel(self) -> None:
        self._ai_panel_loaded = True
        try:
            from gui.ai_management_panel import AIManagementPanel
            panel = AIManagementPanel(self._ai_panel_frame)
            panel.pack(fill="both", expand=True)
        except ImportError:
            ttk.Label(
                self._ai_panel_frame,
                text=(
                    "🤖  AI 관리 패널\n\n"
                    "ai_management/ 모듈이 준비되지 않았습니다.\n\n"
                    "필요 모듈:\n"
                    "  • ai_management/dataset_manager.py\n"
                    "  • ai_management/training_manager.py\n"
                    "  • ai_management/model_registry.py\n"
                    "  • (기타 5개 모듈)"
                ),
                justify="left",
                wraplength=320,
            ).pack(padx=20, pady=30)

    # ─────────────────────────────────────────────────────────────────────
    # Engine start / stop
    # ─────────────────────────────────────────────────────────────────────

    def _on_start(self, overrides: dict) -> None:
        src_type = overrides.get("source_type", "file")
        src_path = overrides.get("source_path", "").strip()
        if src_type in ("file", "images") and not src_path:
            messagebox.showwarning(
                "소스 없음",
                "비디오 파일 또는 이미지 폴더 경로를 입력해 주세요.",
            )
            self._ctrl_panel.set_stopped()
            return

        if self._worker and self._worker.is_running:
            self._on_stop()

        cfg = self._build_config(overrides)
        self._worker = EngineWorker(cfg)
        self._worker.start()
        self._start_polling()
        self._sb_state.set("분석 중…")
        self._notebook.select(2)    # 상태 탭

    def _on_stop(self) -> None:
        if self._worker:
            self._worker.stop()
        self._stop_polling()
        self._canvas.clear_frame()
        self._canvas.zoom_reset()
        self._ctrl_panel.set_stopped()
        self._sb_state.set("정지됨.")
        self._sb_src.set("소스: —")
        self._update_viewport_status()

    def _build_config(self, ov: dict) -> SystemConfig:
        rois = [
            ROIConfig(roi_id=r["roi_id"], bbox=tuple(r["bbox"]),  # type: ignore[arg-type]
                      label=r.get("label", ""))
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
        cfg.ai = AIConfig(
            model_type=ov.get("model_type", "placeholder"),
            model_path=ov.get("model_path"),
            class_names=["normal", "abnormal"],
        )
        cfg.decision      = DecisionConfig()
        cfg.logging       = LoggingConfig(log_dir="logs")
        cfg.visualization = VisualizationConfig()
        cfg.max_frames    = ov.get("max_frames")
        return cfg

    # ─────────────────────────────────────────────────────────────────────
    # GUI polling
    # ─────────────────────────────────────────────────────────────────────

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
                messagebox.showerror(
                    "엔진 오류",
                    f"분석 중 오류가 발생했습니다:\n\n{self._worker.last_error}",
                )
            self._ctrl_panel.set_stopped()
            self._sb_state.set("완료.")
            return

        self._poll_id = self.after(self.POLL_MS, self._poll)

    def _handle_result(self, result: FrameResult) -> None:
        if result.warning:
            self._sb_state.set(f"⚠  {result.warning}")
            return

        if result.display_frame is not None:
            h, w = result.display_frame.shape[:2]
            # VideoCanvas handles viewport cropping internally
            self._canvas.update_frame(result.display_frame, src_w=w, src_h=h)
            self._sb_src.set(f"소스: {w}×{h}")

        # Push all annotation data to VideoCanvas as fixed-position Tkinter
        # items — these are NEVER affected by zoom / pan.
        self._canvas.update_annotations(
            state         = result.system_state,
            confidence    = result.state_confidence,
            frame_index   = result.frame_index,
            timestamp     = result.timestamp,
            fps           = result.fps_actual,
            inference_ms  = result.inference_ms,
            is_event      = result.is_event,
            event_type    = result.event_type,
            triggered_rules = result.triggered_rules,
        )

        self._status_panel.update_status(result)

        if result.new_event:
            self._event_panel.add_event(result.new_event)
            self._event_count += 1
            self._sb_events.set(f"이벤트: {self._event_count}건")
            if self._event_count == 1:
                self._notebook.select(3)

        self._sb_frame.set(f"프레임: {result.frame_index:,}")
        self._sb_state.set(f"상태: {result.system_state.value}")
        self._sb_fps.set(f"FPS: {result.fps_actual:.1f}")
        self._update_viewport_status()

    # ─────────────────────────────────────────────────────────────────────
    # ROI management
    # ─────────────────────────────────────────────────────────────────────

    def _on_canvas_roi_drawn(self, roi_id: str, original_rect: tuple) -> None:
        """VideoCanvas calls this after converting mouse coords to source-pixel."""
        self._roi_panel.add_roi(roi_id, original_rect)
        self._canvas.add_roi_overlay(roi_id, original_rect)
        # Auto-select the newly drawn ROI
        self._canvas.selected_roi_id = roi_id

    def _on_rois_changed(self, roi_list: list) -> None:
        self._canvas.clear_roi_overlays()
        for r in roi_list:
            self._canvas.add_roi_overlay(r["roi_id"], tuple(r["bbox"]))

    # ─────────────────────────────────────────────────────────────────────
    # Help & dialogs
    # ─────────────────────────────────────────────────────────────────────

    def _show_help(self) -> None:
        from gui.help_dialog import HelpDialog
        HelpDialog(self)

    def _show_shortcuts(self) -> None:
        """Pop a compact keyboard shortcut reference dialog."""
        shortcuts = (
            "┌─────────────────────────────────────────┐\n"
            "│         키보드 단축키 목록                │\n"
            "├──────────────┬──────────────────────────┤\n"
            "│  +  /  =     │ 줌 확대                  │\n"
            "│  -           │ 줌 축소                  │\n"
            "│  0           │ 줌 초기화 (100 %)        │\n"
            "│  마우스 휠    │ 줌 인/아웃               │\n"
            "├──────────────┼──────────────────────────┤\n"
            "│  좌클릭 드래그 │ 팬 (팬 모드일 때)        │\n"
            "│  ← ↑ → ↓     │ 화살표 팬                │\n"
            "│  PIP 클릭    │ 해당 위치로 뷰포트 이동   │\n"
            "├──────────────┼──────────────────────────┤\n"
            "│  F           │ 선택한 ROI 전체 화면 토글  │\n"
            "│  Esc         │ 전체 화면 해제 / 줌 초기화 │\n"
            "├──────────────┼──────────────────────────┤\n"
            "│  R           │ ROI 그리기 모드           │\n"
            "│  P           │ 팬 모드                  │\n"
            "└──────────────┴──────────────────────────┘"
        )
        messagebox.showinfo("단축키 목록", shortcuts)

    # ─────────────────────────────────────────────────────────────────────
    # Menu actions
    # ─────────────────────────────────────────────────────────────────────

    def _menu_open_file(self) -> None:
        p = filedialog.askopenfilename(
            title="비디오 파일 선택",
            filetypes=[
                ("비디오 파일", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("전체", "*.*"),
            ],
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

    def _menu_log_folder(self) -> None:
        log_path = os.path.abspath("logs")
        os.makedirs(log_path, exist_ok=True)
        try:
            subprocess.Popen(f'explorer "{log_path}"')
        except Exception:
            messagebox.showinfo("로그 폴더", f"로그 저장 위치:\n{log_path}")

    def _menu_clear_events(self) -> None:
        self._event_panel._clear()
        self._event_count = 0
        self._sb_events.set("이벤트: 0건")

    def _menu_about(self) -> None:
        messagebox.showinfo(
            "프로그램 정보",
            "비디오 분석 시스템 v3.0  (AI + 뷰포트 통합 에디션)\n\n"
            "모듈형 영상 이상 감지 파이프라인\n\n"
            "신규 기능 (v3):\n"
            "  • ViewportManager — 중심 기준 줌 & 팬\n"
            "  • PIP 미니맵 (우상단 오버레이)\n"
            "  • ROI 전체 화면 (F 키)\n"
            "  • 키보드 단축키 체계\n\n"
            "처리 단계:\n"
            "  FrameSource → Preprocessor → ROIManager\n"
            "  → FeatureExtractor → TemporalBuffer\n"
            "  → InferenceEngine → DecisionEngine\n"
            "  → Visualization → Logger\n\n"
            "기술 스택: Python · OpenCV · Tkinter · NumPy\n"
            "AI 백엔드: Placeholder / ONNX / PyTorch",
        )

    # ─────────────────────────────────────────────────────────────────────
    # Close
    # ─────────────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        if self._worker and self._worker.is_running:
            if not messagebox.askyesno(
                "종료 확인",
                "분석이 실행 중입니다.\n종료하시겠습니까?\n(로그는 자동 저장됩니다.)",
            ):
                return
        self._on_stop()
        if self._worker:
            self._worker.join(timeout=2.0)
        self.destroy()
