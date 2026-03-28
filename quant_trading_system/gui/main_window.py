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
from .program_info_dialog import open_program_info
from .tooltip import add_tooltip
from .chart_panel import ChartPanel
from .history_panel import HistoryPanel
from .failure_panel import FailurePanel
from .market_env_panel import MarketEnvPanel
from .requirements_panel import RequirementsPanel
from .layout_manager import LayoutManager
# BNF 탐지기 — 안전한 선택적 임포트 (의존성 없어도 앱 실행)
try:
    from .bnf_panel import BNFPanel
    _BNF_PANEL_AVAILABLE = True
except Exception as _bnf_import_err:
    logger.warning(f"BNF 패널 로드 실패 (무시): {_bnf_import_err}")
    _BNF_PANEL_AVAILABLE = False
from .pipeline_manager import PipelineManager, STAGE_DEFS, STATUS_STYLE, STATUS_ICON

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
        ("📉 차트",         "참고",   "전문가급 HTS 차트로 종목을 분석하세요  (캔들·지표·크로스헤어)"),
        ("📋 예측이력",     "이력",   "과거 예측 기록을 조회하고 실제 결과와 비교하세요"),
        ("🔬 실패분석",     "교정",   "예측 실패 패턴을 해부하고 자기교정 방향을 제시합니다"),
        ("🌐 외부환경",     "환경",   "뉴스·이벤트·시장 분위기를 실시간으로 파악하고 예측에 반영"),
        ("🖥️ GPU 모니터",  "정보",   "GPU / CUDA 사용 현황"),
        ("⚙️ 설정",         "설정",   "세부 파라미터를 조정하세요"),
        ("📋 요구사항",     "관리",   "AI 시스템 개발 방향 및 요구사항을 관리합니다"),
    ]

    # 탭 인덱스 상수
    _IDX_CHART   = 5
    _IDX_HISTORY = 6
    _IDX_FAILURE = 7
    _IDX_MACRO   = 8

    def __init__(self, root: tk.Tk):
        self.root = root
        self.settings = load_settings()
        setup_logger("quant", self.settings.log_level,
                     os.path.join(BASE_DIR, self.settings.output_dir))

        self._configure_root()

        # 초보자/전문가 모드 상태
        self._beginner_mode = tk.BooleanVar(value=True)

        # 윈도우 크기 자동 저장 (디바운스)
        self._geo_save_job: str | None = None
        self.root.bind("<Configure>", self._on_root_configure)

        # 데이터 신선도 검사 타이머 핸들 (중복 방지용)
        self._freshness_job: str | None = None

        # GPU 모니터 — _build_ui() 전에 생성 (패널에서 콜백 등록하므로)
        self.gpu_monitor = GPUMonitor(update_interval=1.5)

        # 파이프라인 매니저 — _build_ui() 전에 생성 (패널 생성 시 콜백으로 전달)
        _pipe_file = os.path.join(BASE_DIR, "outputs", "pipeline.json")
        self._pipeline = PipelineManager(
            _pipe_file,
            on_update=lambda: self.root.after(0, self._on_pipeline_update),
        )

        self._build_ui()

        # 콜백 등록 후 폴링 시작
        self.gpu_monitor.start()

        # 레이아웃 매니저: PanedWindow sash 위치 자동 저장/복원
        _layout_file = os.path.join(BASE_DIR, "outputs", "layout.json")
        self._layout_mgr = LayoutManager(_layout_file)
        # UI 완전 렌더링 후 attach (400ms 지연)
        self.root.after(400, lambda: self._layout_mgr.attach(self.root))

        # 시작 직후 파이프라인 + 알림 바 초기 렌더
        self.root.after(600, self._on_pipeline_update)
        # 시작 직후 데이터 신선도 검사 (1초 후 실행 — UI 완전 로드 후)
        self.root.after(1000, self._check_data_freshness)
        # 시작 시 VERSION.txt 갱신 (요구사항 현황 동기화)
        self.root.after(2000, self._sync_version_file)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        logger.info("메인 윈도우 시작")

    # ──────────────────────────────────────────────────
    # 초기화
    # ──────────────────────────────────────────────────

    def _configure_root(self):
        self.root.title("AI 주식 분석 시스템 — 퀀트 트레이딩")
        self.root.geometry(f"{self.settings.window_width}x{self.settings.window_height}")
        self.root.minsize(1100, 720)
        self.root.state("zoomed")   # 항상 최대화로 시작

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
                        padding=[7, 3], font=("맑은 고딕", 9, "bold"))
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

        # 데이터 신선도 알람 바 (헤더 바로 아래)
        self._build_alert_bar()

        # 워크플로우 가이드 배너
        self._build_workflow_guide()

        # 메인 탭
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=4, pady=(0, 0))
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # 탭 패널 생성
        self.data_panel       = DataPanel(self.notebook, self.settings,
                                          self._on_settings_change,
                                          on_data_downloaded=self._on_data_downloaded)
        self.train_panel      = TrainingPanel(
            self.notebook, self.settings, self._on_settings_change,
            on_complete=lambda: self._pipeline.mark_done("train"),
        )
        self.backtest_panel   = BacktestPanel(
            self.notebook, self.settings,
            on_complete=lambda: self._pipeline.mark_done("backtest"),
        )
        self.prediction_panel = PredictionPanel(
            self.notebook, self.settings,
            on_complete=lambda: self._pipeline.mark_done("predict"),
        )
        self.portfolio_panel  = PortfolioPanel(self.notebook, self.settings)
        self.chart_panel      = ChartPanel(self.notebook, settings=self.settings)
        # 앱 시작 시 포트폴리오 종목을 즉시 드롭다운에 채움 (종목명 포함)
        if self.settings.data.symbols:
            self.chart_panel.set_symbols(self.settings.data.symbols)
        self.history_panel    = HistoryPanel(self.notebook, self.settings)
        self.failure_panel    = FailurePanel(self.notebook, self.settings)
        self.market_env_panel = MarketEnvPanel(
            self.notebook, self.settings,
            on_mood_update=lambda n_bull, n_bear, top:
                self.root.after(0, lambda: self._update_mood_bar(n_bull, n_bear, top)),
        )
        self.gpu_panel        = GPUPanel(self.notebook)
        self.settings_panel   = SettingsPanel(self.notebook, self.settings, self._on_settings_change)
        _out_dir = os.path.join(BASE_DIR, self.settings.output_dir)
        self.requirements_panel = RequirementsPanel(self.notebook, _out_dir)

        # BNF 탐지기 패널 (선택적)
        self.bnf_panel = None
        if _BNF_PANEL_AVAILABLE:
            try:
                self.bnf_panel = BNFPanel(self.notebook, self.settings)
            except Exception as _e:
                logger.warning(f"BNF 패널 초기화 실패 (무시): {_e}")

        self.notebook.add(self.data_panel.frame,          text="  📊 데이터  ")
        self.notebook.add(self.train_panel.frame,          text="  🧠 학습  ")
        self.notebook.add(self.backtest_panel.frame,       text="  📈 백테스트  ")
        self.notebook.add(self.prediction_panel.frame,     text="  🔮 미래 예측  ")
        self.notebook.add(self.portfolio_panel.frame,      text="  💼 포트폴리오  ")
        self.notebook.add(self.chart_panel.frame,          text="  📉 차트  ")
        self.notebook.add(self.history_panel.frame,        text="  📋 예측이력  ")
        self.notebook.add(self.failure_panel.frame,        text="  🔬 실패분석  ")
        self.notebook.add(self.market_env_panel.frame,     text="  🌐 외부환경  ")
        if self.bnf_panel is not None:
            self.notebook.add(self.bnf_panel.frame,        text="  🎯 BNF탐지  ")
        self.notebook.add(self.gpu_panel.frame,            text="  🖥️ GPU  ")
        self.notebook.add(self.settings_panel.frame,       text="  ⚙️ 설정  ")
        self.notebook.add(self.requirements_panel.frame,   text="  📋 요구사항  ")

        # GPU 모니터 연동
        self.gpu_monitor.register_callback(self.gpu_panel.update_stats)

        # 차트 탭 전환 시 데이터 자동 로드
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # 상태바
        self._build_statusbar()

    def _build_header(self):
        """상단 헤더 바"""
        hdr = tk.Frame(self.root, bg="#11111b", height=44)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        # 타이틀
        tk.Label(
            hdr,
            text="  🤖 AI 주식 분석 시스템",
            font=("맑은 고딕", 12, "bold"),
            bg="#11111b", fg="#cba6f7",
        ).pack(side="left", padx=10)

        tk.Label(
            hdr,
            text="AI 기반 주가 예측 · 백테스트 · 포트폴리오 관리",
            font=("맑은 고딕", 8),
            bg="#11111b", fg="#9399b2",
        ).pack(side="left", padx=(0, 12))

        # 오른쪽: 버튼들
        btn_fr = tk.Frame(hdr, bg="#11111b")
        btn_fr.pack(side="right", padx=8, pady=6)

        help_btn = ttk.Button(btn_fr, text="❓ 도움말", command=self._show_help)
        help_btn.pack(side="right", padx=(4, 0))
        add_tooltip(help_btn, "사용 방법 및 각 기능 설명을 확인합니다")

        info_btn = ttk.Button(btn_fr, text="📄 프로그램 정보",
                              command=lambda: open_program_info(self.root))
        info_btn.pack(side="right", padx=4)
        add_tooltip(info_btn,
                    "프로그램 구조 · 요청사항 현황 · 구현상태 · 파일구조 · 데이터플로우\n"
                    "언제든 전체 개발 현황을 확인합니다")

        req_btn = ttk.Button(btn_fr, text="📋 요구사항",
                             command=self._open_requirements)
        req_btn.pack(side="right", padx=4)
        add_tooltip(req_btn, "개발 요구사항 및 로드맵을 관리합니다")

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

    def _build_alert_bar(self):
        """
        통합 상태 바 (단일 행, 좌/우 절반):
          [왼쪽 50%: 알림 — 파이프라인 상태 + 데이터 신선도]
          [오른쪽 50%: 뉴스 감성 — 호재/악재 건수 + 헤드라인]
        배경색으로 상태 표현:  녹색=정상 / 주황=갱신필요 / 빨강=데이터만료
        """
        combined = tk.Frame(self.root, height=26)
        combined.pack(fill="x")
        combined.pack_propagate(False)

        # ── 왼쪽: 알림 영역 ───────────────────────────────────────────────
        self._alert_half = tk.Frame(combined, bg="#0a2a12")
        self._alert_half.pack(side="left", fill="both", expand=True)

        self._alert_lbl = tk.Label(
            self._alert_half,
            text="  🔄  파이프라인 확인 중...",
            bg="#0a2a12", fg="#a6e3a1",
            font=("맑은 고딕", 8, "bold"), anchor="w",
        )
        self._alert_lbl.pack(fill="both", expand=True, padx=(8, 4))

        # 구분선
        tk.Frame(combined, bg="#313244", width=1).pack(side="left", fill="y")

        # ── 오른쪽: 뉴스 감성 영역 ───────────────────────────────────────
        self._mood_half = tk.Frame(combined, bg="#12122a")
        self._mood_half.pack(side="left", fill="both", expand=True)

        self._mood_bull_lbl = tk.Label(
            self._mood_half, text="🟢 호재 0건",
            bg="#12122a", fg="#89dceb",
            font=("맑은 고딕", 8, "bold"), anchor="w",
        )
        self._mood_bull_lbl.pack(side="left", padx=(8, 4))

        self._mood_bear_lbl = tk.Label(
            self._mood_half, text="🔴 악재 0건",
            bg="#12122a", fg="#f38ba8",
            font=("맑은 고딕", 8, "bold"), anchor="w",
        )
        self._mood_bear_lbl.pack(side="left", padx=(0, 4))

        tk.Label(self._mood_half, text="│",
                 bg="#12122a", fg="#313244",
                 font=("맑은 고딕", 8)).pack(side="left")

        self._mood_news_lbl = tk.Label(
            self._mood_half, text="  뉴스 로드 중...",
            bg="#12122a", fg="#6c7086",
            font=("맑은 고딕", 8), anchor="w",
        )
        self._mood_news_lbl.pack(side="left", padx=4, fill="x", expand=True)

        # 상태 추적
        self._alert_visible = False
        self._alert_msg_data = ""   # 데이터 신선도 메시지 (빈 문자열 = 정상)

    # ──────────────────────────────────────────────────────────────────
    # 알림 절반 업데이트 — 데이터 신선도 + 파이프라인 통합
    # ──────────────────────────────────────────────────────────────────

    def _refresh_alert_half(self):
        """알림 영역(왼쪽 절반) 재렌더링 — 항상 이 메서드 하나로 처리"""
        if not hasattr(self, "_alert_half"):
            return
        stale = (self._pipeline.get_stale_stages()
                 if hasattr(self, "_pipeline") else [])
        data_stale = bool(self._alert_msg_data)

        if data_stale:
            # 최고 우선순위: 데이터 만료 (빨강)
            bg, fg = "#3a0808", "#ff6b6b"
            text = f"  ⚠  데이터 만료 — {self._alert_msg_data}"
        elif stale:
            # 파이프라인 갱신 필요 (주황)
            bg, fg = "#2a1500", "#f9e2af"
            labels = "  ·  ".join(stale)
            text = f"  ⚠  갱신 필요:  {labels}"
        else:
            # 전체 정상 (녹색)
            bg, fg = "#0a2a12", "#a6e3a1"
            text = "  ✅  파이프라인 정상"

        self._alert_half.configure(bg=bg)
        self._alert_lbl.configure(text=text, bg=bg, fg=fg)

    def _show_alert(self, msg: str):
        """데이터 신선도 경고 설정 → 알림 절반 갱신."""
        self._alert_msg_data = msg
        self._alert_visible  = True
        self._refresh_alert_half()

    def _hide_alert(self):
        """데이터 신선도 정상 → 알림 절반 갱신."""
        self._alert_msg_data = ""
        self._alert_visible  = False
        self._refresh_alert_half()

    def _update_mood_bar(self, n_bull: int, n_bear: int, top_title: str):
        """뉴스 감성 영역(오른쪽 절반) 업데이트."""
        if not hasattr(self, "_mood_half"):
            return

        self._mood_bull_lbl.configure(text=f"🟢 호재 {n_bull}건")
        self._mood_bear_lbl.configure(text=f"🔴 악재 {n_bear}건")

        net = n_bull - n_bear
        if net > 2:
            bg, news_fg = "#0d1f2d", "#89dceb"
        elif net < -2:
            bg, news_fg = "#2d0d0d", "#f38ba8"
        else:
            bg, news_fg = "#12122a", "#9399b2"

        self._mood_half.configure(bg=bg)
        for w in self._mood_half.winfo_children():
            try:
                w.configure(bg=bg)
            except Exception:
                pass

        short = (top_title[:55] + "…") if len(top_title) > 58 else top_title
        self._mood_news_lbl.configure(
            text=f"  📰 {short}" if short else "  뉴스 로드 완료",
            fg=news_fg,
        )

    def _check_data_freshness(self):
        """데이터 파일 신선도 검사 — 오래된 종목 감지 후 알람 (중복 방지)"""
        # 이미 실행 예약된 주기적 타이머가 있으면 취소 후 재예약
        if hasattr(self, "_freshness_job") and self._freshness_job:
            try:
                self.root.after_cancel(self._freshness_job)
            except Exception:
                pass
            self._freshness_job = None

        def _do():
            try:
                import datetime
                import numpy as np
                from data import DataLoader
                cache_dir = os.path.join(BASE_DIR, self.settings.data.cache_dir)
                loader    = DataLoader(cache_dir, self.settings.data.cache_ttl_hours)

                # 현재 settings에서 period/interval 읽기 (default fallback)
                period   = getattr(self.settings.data, "period",   "5y")
                interval = getattr(self.settings.data, "interval", "1d")

                stale_syms  = []
                missing_syms = []

                for sym in self.settings.data.symbols:
                    try:
                        df = loader.load_cached_only(sym, period, interval)
                        if df is None or df.empty:
                            missing_syms.append(sym)
                            continue
                        last_date = df.index[-1]
                        if hasattr(last_date, "date"):
                            last_date = last_date.date()
                        today = datetime.date.today()
                        # 영업일(Business Day) 기준으로 stale 판정
                        # 주말/공휴일 고려: 금요일 데이터 → 월요일까지는 stale 아님
                        bd_delta = np.busday_count(last_date, today)
                        if bd_delta >= 2:
                            stale_syms.append((sym, last_date))
                    except Exception:
                        missing_syms.append(sym)

                msg_parts = []
                if missing_syms:
                    msg_parts.append(f"데이터 없음: {', '.join(missing_syms[:3])}"
                                     + (" 외" if len(missing_syms) > 3 else ""))
                if stale_syms:
                    oldest = min(stale_syms, key=lambda x: x[1])
                    msg_parts.append(f"최신 업데이트 필요!! {len(stale_syms)}개 종목 "
                                     f"(가장 오래된: {oldest[0]} {oldest[1]})")

                if msg_parts:
                    full_msg = "  |  ".join(msg_parts)
                    self.root.after(0, lambda m=full_msg: self._show_alert(
                        m + "  →  [데이터 탭]에서 [💾 데이터 다운로드] 클릭"))
                else:
                    self.root.after(0, self._hide_alert)

            except Exception as e:
                logger.debug(f"데이터 신선도 검사 실패: {e}")

        threading.Thread(target=_do, daemon=True, name="freshness-check").start()
        # 1시간 후 다시 체크 (job ID 저장 → 수동 호출 시 취소·재예약 가능)
        self._freshness_job = self.root.after(3_600_000, self._check_data_freshness)

    def _build_workflow_guide(self):
        """
        파이프라인 상태 바:
          [📍 단계  설명]   [📊데이터✅ → 🧠학습⚠ → 📈백테스트○ → 🔮예측○]
        """
        self._guide_frame = tk.Frame(self.root, bg="#181825", height=30)
        self._guide_frame.pack(fill="x", padx=4, pady=(1, 0))
        self._guide_frame.pack_propagate(False)

        self._guide_step_var = tk.StringVar(value="")
        self._guide_desc_var = tk.StringVar(value="")

        # 왼쪽: 단계 + 설명
        tk.Label(self._guide_frame, textvariable=self._guide_step_var,
                 bg="#181825", fg="#89b4fa",
                 font=("맑은 고딕", 8, "bold"), width=8, anchor="w",
                 ).pack(side="left", padx=(8, 0))
        tk.Label(self._guide_frame, textvariable=self._guide_desc_var,
                 bg="#181825", fg="#9399b2",
                 font=("맑은 고딕", 8), anchor="w",
                 ).pack(side="left", padx=4, fill="x", expand=True)

        # 오른쪽: 파이프라인 단계 칩
        pipe_fr = tk.Frame(self._guide_frame, bg="#181825")
        pipe_fr.pack(side="right", padx=6)

        self._pipe_chips: dict[str, tk.Button] = {}
        for i, (key, label, tab_idx, icon) in enumerate(STAGE_DEFS):
            if i > 0:
                tk.Label(pipe_fr, text="→", bg="#181825", fg="#45475a",
                         font=("맑은 고딕", 8)).pack(side="left")
            chip = tk.Button(
                pipe_fr,
                text=f"{icon} {label}  ○",
                bg="#2a2a3e", fg="#6c7086",
                relief="flat", bd=0, padx=6, pady=1,
                font=("맑은 고딕", 8, "bold"), cursor="hand2",
                command=lambda i=tab_idx: self.notebook.select(i),
            )
            chip.pack(side="left", padx=1)
            self._pipe_chips[key] = chip

        self._update_workflow_guide(0)

    def _update_workflow_guide(self, tab_index: int):
        """현재 탭에 맞게 안내 + 파이프라인 칩 업데이트"""
        if tab_index < len(self._TAB_STEPS):
            _name, step, desc = self._TAB_STEPS[tab_index]
            self._guide_step_var.set(f"📍 {step}")
            self._guide_desc_var.set(desc)
        else:
            self._guide_step_var.set("")
            self._guide_desc_var.set("")
        self._refresh_pipeline_chips()

    def _on_pipeline_update(self):
        """파이프라인 상태 변경 시 — 칩 + 알림 바 동시 갱신"""
        self._refresh_pipeline_chips()
        self._refresh_alert_half()

    def _refresh_pipeline_chips(self):
        """파이프라인 칩 색상·아이콘 갱신"""
        if not hasattr(self, "_pipe_chips") or not hasattr(self, "_pipeline"):
            return
        for key, label, _, icon in STAGE_DEFS:
            chip = self._pipe_chips.get(key)
            if chip is None:
                continue
            status = self._pipeline.get_status(key)
            bg, fg = STATUS_STYLE.get(status, ("#2a2a3e", "#6c7086"))
            sicon  = STATUS_ICON.get(status, "○")
            ts     = self._pipeline.get_ts(key)
            tip    = f"{icon} {label}\n상태: {status}\n최근: {ts}"
            chip.configure(text=f"{icon} {label}  {sicon}", bg=bg, fg=fg)
            try:
                from .tooltip import add_tooltip
                add_tooltip(chip, tip)
            except Exception:
                pass

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
        """탭 전환 시 워크플로우 안내 업데이트 및 각 패널 자동 갱신"""
        try:
            idx = self.notebook.index(self.notebook.select())
            self._update_workflow_guide(idx)

            # 뱃지 제거 — 해당 탭 방문 시 🔔 제거
            _badge_clear = {
                1: "  🧠 학습  ",
                2: "  📈 백테스트  ",
                3: "  🔮 미래 예측  ",
            }
            if idx in _badge_clear:
                self._clear_tab_badge(idx, _badge_clear[idx])

            if idx == self._IDX_CHART:
                try:
                    self.chart_panel.set_symbols(self.settings.data.symbols)
                except Exception:
                    pass
                if self.settings.data.symbols:
                    self._load_chart_for_first_symbol()

            elif idx == self._IDX_HISTORY:
                try:
                    self.history_panel.refresh()
                except Exception:
                    pass

            elif idx == self._IDX_FAILURE:
                try:
                    self.failure_panel.refresh()
                except Exception:
                    pass

            elif idx == self._IDX_MACRO:
                # 외부환경 탭 진입 시 즉시 갱신
                try:
                    self.market_env_panel.refresh()
                except Exception:
                    pass

        except Exception:
            pass

    def _on_root_configure(self, event=None):
        """윈도우 크기/위치 변경 시 설정에 자동 저장 (디바운스 500ms)"""
        if event and event.widget is not self.root:
            return
        if self._geo_save_job:
            try:
                self.root.after_cancel(self._geo_save_job)
            except Exception:
                pass
        self._geo_save_job = self.root.after(500, self._save_geometry)

    def _save_geometry(self):
        """현재 윈도우 geometry를 settings에 저장"""
        try:
            geo = self.root.geometry()   # "WxH+X+Y"
            import re
            m = re.match(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", geo)
            if m:
                w, h, x, y = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
                if w >= 800 and h >= 500:   # 최소 유효 크기
                    self.settings.window_width  = w
                    self.settings.window_height = h
                    # window_x/y는 AppSettings에 동적 속성으로 저장
                    self.settings.window_x = x
                    self.settings.window_y = y
                    # 즉시 파일에 반영
                    self._flush_settings()
        except Exception as e:
            logger.debug(f"geometry 저장 실패: {e}")

    def _flush_settings(self):
        """설정을 조용히 파일에 저장 (상태바 메시지 없음)"""
        try:
            from config import save_settings
            from dataclasses import asdict
            import json
            path = os.path.join(BASE_DIR, "settings.json")
            data = asdict(self.settings)
            # window_x/y 는 dataclass 필드가 아니므로 수동으로 추가
            data["window_x"] = getattr(self.settings, "window_x", 0)
            data["window_y"] = getattr(self.settings, "window_y", 0)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _load_chart_for_first_symbol(self) -> None:
        """차트 탭에서 첫 번째 등록 종목의 최근 데이터를 비동기로 로드."""
        sym = self.settings.data.symbols[0]

        def _do():
            try:
                import os
                from data import DataLoader
                cache_dir = os.path.join(BASE_DIR, self.settings.data.cache_dir)
                loader    = DataLoader(cache_dir, self.settings.data.cache_ttl_hours)
                df = loader.load(sym, self.settings.data.period,
                                 self.settings.data.interval)
                if df is not None and not df.empty:
                    from data.korean_stocks import get_name
                    name = get_name(sym)
                    disp = f"{name} ({sym.split('.')[0]})" if name and name != sym else sym
                    self.root.after(0, lambda: self.chart_panel.load_symbol(disp, df))
            except Exception:
                pass

        threading.Thread(target=_do, daemon=True).start()

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

        # 데이터 다운로드 완료 시에도 이 콜백이 호출됨
        # → 기존 1시간 예약 타이머를 취소하고 즉시(1초 후) 재검사
        if self._freshness_job:
            try:
                self.root.after_cancel(self._freshness_job)
            except Exception:
                pass
            self._freshness_job = None
        self.root.after(1000, self._check_data_freshness)
        # 각 탭 종목 목록 동기화
        for panel_name, method in [
            ("train_panel",      "refresh_symbols"),
            ("prediction_panel", "refresh_symbols"),
            ("portfolio_panel",  "refresh_symbols"),
            ("backtest_panel",   "refresh_symbols"),
            ("bnf_panel",        "refresh_symbols"),   # BNF 탐지기 동기화
        ]:
            panel = getattr(self, panel_name, None)
            if panel:
                try:
                    getattr(panel, method)()
                except Exception as e:
                    logger.debug(f"{panel_name}.{method}() 오류: {e}")

        try:
            self.chart_panel.set_symbols(self.settings.data.symbols)
        except Exception as e:
            logger.debug(f"chart_panel.set_symbols() 오류: {e}")

    def _on_data_downloaded(self):
        """
        데이터 탭에서 다운로드 완료 시 호출.
        학습 / 백테스트 / 예측 탭에 🔔 뱃지 + 패널 내 알림 배너 표시.
        파이프라인 상태: data=fresh, 하위 단계 stale 처리.
        """
        self._pipeline.mark_done("data")
        # 탭 뱃지 추가 (탭 텍스트 앞에 🔔)
        _badge_targets = {
            1: ("  🧠 학습  ",    "  🔔 🧠 학습  "),
            2: ("  📈 백테스트  ", "  🔔 📈 백테스트  "),
            3: ("  🔮 미래 예측  ","  🔔 🔮 미래 예측  "),
        }
        for idx, (_, badged) in _badge_targets.items():
            try:
                self.notebook.tab(idx, text=badged)
            except Exception:
                pass

        # 각 패널 내부 노란 알림 배너 표시
        try:
            self.train_panel.notify_data_updated()
        except Exception:
            pass
        try:
            self.prediction_panel.notify_data_updated()
        except Exception:
            pass

        # 하단 상태바 메시지
        self.status_var.set("🔔 데이터 업데이트 완료 — 학습 / 예측 재실행을 권장합니다")

    def _clear_tab_badge(self, tab_idx: int, normal_text: str):
        """탭 방문 시 🔔 뱃지 제거"""
        try:
            current = self.notebook.tab(tab_idx, "text")
            if "🔔" in current:
                self.notebook.tab(tab_idx, text=normal_text)
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

    def _open_requirements(self):
        """요구사항 관리 탭으로 즉시 이동."""
        try:
            for i in range(self.notebook.index("end")):
                if "요구사항" in self.notebook.tab(i, "text"):
                    self.notebook.select(i)
                    return
        except Exception:
            pass

    def _sync_version_file(self):
        """시작 시 requirements store를 통해 VERSION.txt 최신화."""
        try:
            if hasattr(self, "requirements_panel"):
                self.requirements_panel._store._write_version_file()
                logger.info("VERSION.txt 갱신 완료")
        except Exception as e:
            logger.debug(f"VERSION.txt 갱신 실패: {e}")

    def _update_gpu_status(self, stats):
        """하단바 GPU 상태 업데이트 (CUDA 메모리 포함)"""
        if not stats:
            return
        s = stats[0]
        cuda_alloc = s.get("cuda_allocated_mb", 0.0)
        total_mb   = s.get("mem_total_mb", 0.0)
        text = (
            f"GPU {s['util_gpu']:.0f}% | "
            f"VRAM {s['mem_used_mb']:.0f}/{total_mb:.0f}MB | "
            f"CUDA {cuda_alloc:.0f}MB | "
            f"{s['temperature']:.0f}°C"
        )
        try:
            self.gpu_status_var.set(text)
        except Exception:
            pass

    def _on_close(self):
        """종료 처리"""
        self.gpu_monitor.stop()
        self._save_settings()
        # 레이아웃 최종 저장 (종료 시 sash 위치 보장)
        if hasattr(self, "_layout_mgr"):
            self._layout_mgr.save_now()
        self.root.destroy()

    def set_status(self, msg: str):
        """상태바 메시지 업데이트 (스레드 안전)"""
        self.root.after(0, lambda: self.status_var.set(msg))
