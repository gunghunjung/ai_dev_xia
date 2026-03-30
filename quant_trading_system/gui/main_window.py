# gui/main_window.py — 메인 윈도우 (한국어 Tkinter UI)
from __future__ import annotations
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import logging

# 프로젝트 루트 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import load_settings, save_settings
from utils.gpu_monitor import GPUMonitor
from utils.logger import setup_logger
from .data_panel import DataPanel
from .training_panel import TrainingPanel
from .backtest_panel import BacktestPanel
from .prediction_panel import PredictionPanel
from .portfolio_panel import PortfolioPanel
from .gpu_panel import GPUPanel
from .settings_panel import SettingsPanel
from .help_dialog import HelpDialog
from .tooltip import add_tooltip

logger = logging.getLogger("quant.gui")


class MainWindow:
    """
    메인 윈도우 — 탭 기반 Tkinter UI

    탭 구성:
    1. 데이터       — 종목 선택, 데이터 다운로드
    2. 학습         — 모델 학습/중단/저장
    3. 백테스트     — Walk-Forward 백테스트
    4. 포트폴리오   — 실시간 포트폴리오 모니터링
    5. GPU 모니터   — GPU/CPU 사용량 실시간 표시
    6. 설정         — 전체 설정 조정
    """

    # 탭 순서 → 워크플로우 단계 매핑
    _TAB_STEPS = [
        ("📊 데이터",       "1단계",  "분석할 종목을 추가하고 데이터를 다운로드하세요"),
        ("🧠 학습",         "2단계",  "AI 모델을 학습시키세요"),
        ("📈 백테스트",     "3단계",  "전략의 과거 성과를 검증하세요"),
        ("🔮 미래 예측",    "4단계",  "학습된 모델로 향후 주가 방향을 예측하세요"),
        ("💼 포트폴리오",   "5단계",  "실시간 포트폴리오 현황을 확인하세요"),
        ("🖥️ GPU 모니터",  "정보",   "GPU 사용 현황"),
        ("⚙️ 설정",         "설정",   "세부 파라미터를 조정하세요"),
    ]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.settings = load_settings()
        setup_logger("quant", self.settings.log_level,
                     os.path.join(BASE_DIR, self.settings.output_dir))

        self._configure_root()

        # 초보자/전문가 모드 상태
        self._beginner_mode = tk.BooleanVar(value=True)

        # GPU 모니터 — _build_ui() 전에 생성 (패널에서 콜백 등록하므로)
        self.gpu_monitor = GPUMonitor(update_interval=1.5)

        self._build_ui()

        # 콜백 등록 후 폴링 시작
        self.gpu_monitor.start()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        logger.info("메인 윈도우 시작")

    # ──────────────────────────────────────────────────
    # 초기화
    # ──────────────────────────────────────────────────

    def _configure_root(self):
        self.root.title("AI 주식 분석 시스템 — 퀀트 트레이딩")
        w = self.settings.window_width
        h = self.settings.window_height
        self.root.geometry(f"{w}x{h}")
        self.root.minsize(1100, 720)
        self.root.state("zoomed")   # 전체화면(최대화)으로 시작

        # 스타일
        style = ttk.Style()
        available = style.theme_names()
        for theme in ("clam", "alt", "default"):
            if theme in available:
                style.theme_use(theme)
                break

        self._apply_style(style)

    def _apply_style(self, style: ttk.Style):
        """커스텀 스타일 적용"""
        bg        = "#1e1e2e"
        fg        = "#cdd6f4"
        accent    = "#89b4fa"
        panel_bg  = "#181825"
        select_bg = "#313244"
        dim_fg    = "#9399b2"   # 보조 텍스트 (이전 #6c7086 보다 밝게)
        thumb_bg  = "#585b70"   # 스크롤바 썸 — 배경과 확실히 구분

        style.configure(".", background=bg, foreground=fg, font=("맑은 고딕", 10))

        # ── 노트북 탭 ────────────────────────────────────────────────
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab",
                        background="#2a2a3e", foreground=dim_fg,
                        padding=[12, 6], font=("맑은 고딕", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", select_bg), ("active", "#313244")],
                  foreground=[("selected", accent),    ("active", fg)])

        # ── 기본 위젯 ────────────────────────────────────────────────
        style.configure("TFrame",  background=bg)
        style.configure("TLabel",  background=bg, foreground=fg)
        style.configure("TSeparator", background=thumb_bg)

        # ── 버튼 ─────────────────────────────────────────────────────
        style.configure("TButton",
                        background="#45475a", foreground=fg,
                        font=("맑은 고딕", 10), padding=[8, 4],
                        relief="flat", borderwidth=0)
        style.map("TButton",
                  background=[("active",   "#585b70"),
                               ("pressed",  "#74c7ec"),
                               ("disabled", "#313244")],
                  foreground=[("active",   "#ffffff"),
                               ("disabled", dim_fg)])
        style.configure("Accent.TButton",
                        background=accent, foreground="#1e1e2e",
                        font=("맑은 고딕", 10, "bold"), padding=[10, 5])
        style.map("Accent.TButton",
                  background=[("active",  "#74c7ec"),
                               ("pressed", "#89dceb")])
        style.configure("Danger.TButton",
                        background="#f38ba8", foreground="#1e1e2e",
                        font=("맑은 고딕", 10, "bold"), padding=[8, 4])
        style.map("Danger.TButton",
                  background=[("active", "#ff6b8a")])

        # ── 입력 필드 ─────────────────────────────────────────────────
        style.configure("TEntry",
                        fieldbackground=panel_bg, foreground=fg,
                        insertcolor=fg,
                        selectbackground=accent,
                        selectforeground="#1e1e2e",
                        bordercolor=thumb_bg,
                        lightcolor=thumb_bg, darkcolor=thumb_bg)

        # ── 콤보박스 ──────────────────────────────────────────────────
        style.configure("TCombobox",
                        fieldbackground=panel_bg, foreground=fg,
                        background="#45475a",      # 드롭다운 화살표 버튼 배경
                        arrowcolor=fg,             # 화살표 색
                        selectbackground=accent,
                        selectforeground="#1e1e2e",
                        bordercolor=thumb_bg)
        style.map("TCombobox",
                  fieldbackground=[("readonly", panel_bg),
                                   ("disabled", "#2a2a3e")],
                  foreground=[("readonly", fg),
                               ("disabled", dim_fg)],
                  selectbackground=[("readonly", select_bg)],
                  selectforeground=[("readonly", fg)],
                  background=[("active", "#585b70")])

        # ── 스크롤바 ─────────────────────────────────────────────────
        #  background = 썸(움직이는 부분), troughcolor = 레일 배경
        style.configure("TScrollbar",
                        background=thumb_bg,
                        troughcolor=panel_bg,
                        arrowcolor=fg,
                        bordercolor=panel_bg,
                        lightcolor=thumb_bg,
                        darkcolor=thumb_bg,
                        relief="flat")
        style.map("TScrollbar",
                  background=[("active",  "#7f849c"),
                               ("pressed", accent)])

        # ── 체크버튼 ─────────────────────────────────────────────────
        style.configure("TCheckbutton",
                        background=bg, foreground=fg,
                        indicatorcolor=panel_bg,
                        indicatorbackground=panel_bg,
                        focuscolor="")
        style.map("TCheckbutton",
                  foreground=[("disabled", dim_fg)],
                  indicatorcolor=[("selected",  accent),
                                  ("!selected", "#45475a")])

        # ── 라디오버튼 ───────────────────────────────────────────────
        style.configure("TRadiobutton",
                        background=bg, foreground=fg, focuscolor="")
        style.map("TRadiobutton",
                  foreground=[("disabled", dim_fg)])

        # ── 프레임 테두리 ─────────────────────────────────────────────
        style.configure("TLabelframe",
                        background=bg, foreground=fg,
                        bordercolor=thumb_bg, relief="solid")
        style.configure("TLabelframe.Label",
                        background=bg, foreground=accent,
                        font=("맑은 고딕", 10, "bold"))

        # ── 진행바 ────────────────────────────────────────────────────
        style.configure("TProgressbar",
                        troughcolor="#2a2a3e", background=accent,
                        bordercolor=panel_bg)

        # ── 트리뷰 ────────────────────────────────────────────────────
        style.configure("Treeview",
                        background=panel_bg, foreground=fg,
                        fieldbackground=panel_bg, rowheight=24)
        style.configure("Treeview.Heading",
                        background="#2a2a3e", foreground=accent,
                        font=("맑은 고딕", 9, "bold"),
                        relief="flat")
        style.map("Treeview",
                  background=[("selected", accent)],
                  foreground=[("selected", "#1e1e2e")])

        self.root.configure(bg=bg)
        self._bg = bg
        self._fg = fg
        self._accent = accent

    def _build_ui(self):
        """UI 구성"""
        # 상단 헤더
        self._build_header()

        # 워크플로우 가이드 배너
        self._build_workflow_guide()

        # 메인 탭
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=4, pady=(0, 0))
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # 탭 패널 생성
        self.data_panel       = DataPanel(self.notebook, self.settings, self._on_settings_change)
        self.train_panel      = TrainingPanel(self.notebook, self.settings, self._on_settings_change)
        self.backtest_panel   = BacktestPanel(self.notebook, self.settings)
        self.prediction_panel = PredictionPanel(self.notebook, self.settings)
        self.portfolio_panel  = PortfolioPanel(self.notebook, self.settings)
        self.gpu_panel        = GPUPanel(self.notebook)
        self.settings_panel   = SettingsPanel(self.notebook, self.settings, self._on_settings_change)

        self.notebook.add(self.data_panel.frame,       text="  📊 데이터  ")
        self.notebook.add(self.train_panel.frame,      text="  🧠 학습  ")
        self.notebook.add(self.backtest_panel.frame,   text="  📈 백테스트  ")
        self.notebook.add(self.prediction_panel.frame, text="  🔮 미래 예측  ")
        self.notebook.add(self.portfolio_panel.frame,  text="  💼 포트폴리오  ")
        self.notebook.add(self.gpu_panel.frame,        text="  🖥️ GPU  ")
        self.notebook.add(self.settings_panel.frame,   text="  ⚙️ 설정  ")

        # GPU 모니터 연동
        self.gpu_monitor.register_callback(self.gpu_panel.update_stats)

        # 상태바
        self._build_statusbar()

    def _build_header(self):
        """상단 헤더 바"""
        hdr = tk.Frame(self.root, bg="#11111b", height=54)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        # 타이틀
        tk.Label(
            hdr,
            text="  🤖 AI 주식 분석 시스템",
            font=("맑은 고딕", 15, "bold"),
            bg="#11111b", fg="#cba6f7",
        ).pack(side="left", padx=10)

        tk.Label(
            hdr,
            text="AI 기반 주가 예측 · 백테스트 · 포트폴리오 관리",
            font=("맑은 고딕", 9),
            bg="#11111b", fg="#9399b2",
        ).pack(side="left", padx=(0, 20))

        # 오른쪽: 버튼들
        btn_fr = tk.Frame(hdr, bg="#11111b")
        btn_fr.pack(side="right", padx=8, pady=6)

        help_btn = ttk.Button(btn_fr, text="❓ 도움말", command=self._show_help)
        help_btn.pack(side="right", padx=(4, 0))
        add_tooltip(help_btn, "사용 방법 및 각 기능 설명을 확인합니다")

        save_btn = ttk.Button(btn_fr, text="💾 설정 저장", command=self._save_settings)
        save_btn.pack(side="right", padx=4)
        add_tooltip(save_btn, "현재 설정을 파일에 저장합니다\n(프로그램 재시작 후에도 유지됨)")

        # 초보자/전문가 모드 토글
        mode_fr = tk.Frame(hdr, bg="#11111b")
        mode_fr.pack(side="right", padx=(0, 12), pady=6)

        tk.Label(mode_fr, text="모드:", bg="#11111b", fg="#9399b2",
                 font=("맑은 고딕", 9)).pack(side="left", padx=(0, 4))

        beginner_rb = tk.Radiobutton(
            mode_fr, text="초보자", variable=self._beginner_mode, value=True,
            bg="#11111b", fg="#a6e3a1", selectcolor="#11111b",
            activebackground="#11111b", activeforeground="#a6e3a1",
            font=("맑은 고딕", 9, "bold"),
            command=self._on_mode_change,
        )
        beginner_rb.pack(side="left")
        add_tooltip(beginner_rb, "도움말과 초보자 추천 설정을 강조합니다")

        expert_rb = tk.Radiobutton(
            mode_fr, text="전문가", variable=self._beginner_mode, value=False,
            bg="#11111b", fg="#89b4fa", selectcolor="#11111b",
            activebackground="#11111b", activeforeground="#89b4fa",
            font=("맑은 고딕", 9, "bold"),
            command=self._on_mode_change,
        )
        expert_rb.pack(side="left", padx=(4, 0))
        add_tooltip(expert_rb, "도움말을 숨기고 전문가 옵션을 표시합니다")

    def _build_workflow_guide(self):
        """
        워크플로우 안내 배너 — 현재 탭 위에 표시.
        초보자가 어떤 순서로 진행해야 하는지 한눈에 보여줍니다.
        """
        self._guide_frame = tk.Frame(self.root, bg="#181825", height=34)
        self._guide_frame.pack(fill="x", padx=4, pady=(2, 0))
        self._guide_frame.pack_propagate(False)

        self._guide_step_var  = tk.StringVar(value="")
        self._guide_desc_var  = tk.StringVar(value="")

        # 단계 표시 (왼쪽)
        self._step_label = tk.Label(
            self._guide_frame,
            textvariable=self._guide_step_var,
            bg="#181825", fg="#89b4fa",
            font=("맑은 고딕", 9, "bold"),
            width=8, anchor="w",
        )
        self._step_label.pack(side="left", padx=(10, 0))

        # 설명 (가운데)
        self._desc_label = tk.Label(
            self._guide_frame,
            textvariable=self._guide_desc_var,
            bg="#181825", fg="#cdd6f4",
            font=("맑은 고딕", 9),
            anchor="w",
        )
        self._desc_label.pack(side="left", padx=8, fill="x", expand=True)

        # 탭 이동 빠른 버튼 (오른쪽)
        nav_fr = tk.Frame(self._guide_frame, bg="#181825")
        nav_fr.pack(side="right", padx=6)

        steps = [
            ("① 데이터", 0), ("② 학습", 1), ("③ 백테스트", 2), ("④ 예측", 3),
        ]
        for text, idx in steps:
            btn = tk.Button(
                nav_fr, text=text,
                bg="#2a2a3e", fg="#9399b2",
                relief="flat", bd=0,
                font=("맑은 고딕", 8),
                padx=6, pady=1,
                cursor="hand2",
                command=lambda i=idx: self.notebook.select(i),
            )
            btn.pack(side="left", padx=2)

        # 초기 업데이트
        self._update_workflow_guide(0)

    def _update_workflow_guide(self, tab_index: int):
        """현재 탭에 맞게 워크플로우 안내 업데이트"""
        if tab_index < len(self._TAB_STEPS):
            _name, step, desc = self._TAB_STEPS[tab_index]
            self._guide_step_var.set(f"📍 {step}")
            self._guide_desc_var.set(desc)
        else:
            self._guide_step_var.set("")
            self._guide_desc_var.set("")

    def _build_statusbar(self):
        """하단 상태바"""
        self.status_var = tk.StringVar(value="준비")
        bar = tk.Frame(self.root, bg="#11111b", height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self.status_label = tk.Label(
            bar, textvariable=self.status_var,
            font=("맑은 고딕", 9),
            bg="#11111b", fg="#a6e3a1",
            anchor="w",
        )
        self.status_label.pack(side="left", padx=10, fill="x", expand=True)

        # GPU 빠른 상태 표시
        self.gpu_status_var = tk.StringVar(value="GPU: —")
        tk.Label(
            bar, textvariable=self.gpu_status_var,
            font=("맑은 고딕", 9),
            bg="#11111b", fg="#89b4fa",
        ).pack(side="right", padx=10)

        self.gpu_monitor.register_callback(self._update_gpu_status)

    # ──────────────────────────────────────────────────
    # 콜백
    # ──────────────────────────────────────────────────

    def _on_tab_changed(self, _event=None):
        """탭 전환 시 워크플로우 안내 업데이트"""
        try:
            idx = self.notebook.index(self.notebook.select())
            self._update_workflow_guide(idx)
        except Exception:
            pass

    def _on_mode_change(self):
        """초보자/전문가 모드 전환"""
        is_beginner = self._beginner_mode.get()
        mode_str = "초보자" if is_beginner else "전문가"
        self.status_var.set(f"모드 변경: {mode_str} 모드")
        # 설정 패널에 모드 변경 알림 (구현된 경우)
        try:
            self.settings_panel.set_mode(is_beginner)
        except Exception:
            pass

    def _on_settings_change(self, new_settings=None):
        """설정 변경 이벤트 — 종목 목록 변경 시 학습/예측 탭에도 반영"""
        if new_settings:
            self.settings = new_settings
        self.status_var.set("설정 변경됨 (미저장)")
        # 학습 탭 종목 목록 동기화
        try:
            self.train_panel.refresh_symbols()
        except Exception:
            pass
        # 예측 탭 종목 목록 동기화
        try:
            self.prediction_panel.refresh_symbols()
        except Exception:
            pass
        # 포트폴리오 탭 종목 목록 동기화
        try:
            self.portfolio_panel.refresh_symbols()
        except Exception:
            pass
        # 백테스트 탭 종목 목록 동기화
        try:
            self.backtest_panel.refresh_symbols()
        except Exception:
            pass

    def _save_settings(self):
        """설정 저장"""
        try:
            save_settings(self.settings,
                          os.path.join(BASE_DIR, "settings.json"))
            self.status_var.set("✅ 설정 저장 완료")
        except Exception as e:
            messagebox.showerror("오류", f"설정 저장 실패:\n{e}")

    def _show_help(self):
        """도움말 다이얼로그"""
        HelpDialog(self.root)

    def _update_gpu_status(self, stats):
        """하단바 GPU 상태 업데이트"""
        if not stats:
            return
        s = stats[0]
        text = (f"GPU: {s['util_gpu']:.0f}% | "
                f"VRAM: {s['mem_used_mb']:.0f}/{s['mem_total_mb']:.0f}MB | "
                f"{s['temperature']:.0f}°C")
        try:
            self.gpu_status_var.set(text)
        except Exception:
            pass

    def _on_close(self):
        """종료 처리"""
        self.gpu_monitor.stop()
        # 실시간 시세 스레드 정리
        if hasattr(self, "data_panel") and hasattr(self.data_panel, "_stop_realtime"):
            self.data_panel._stop_realtime()
        self._save_settings()
        self.root.destroy()

    def set_status(self, msg: str):
        """상태바 메시지 업데이트 (스레드 안전)"""
        self.root.after(0, lambda: self.status_var.set(msg))
